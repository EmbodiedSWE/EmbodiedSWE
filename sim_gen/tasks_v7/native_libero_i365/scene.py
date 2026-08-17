"""CrankFerryScene — deliver the red cube into a sealed under-floor vault by
CRANKING a Scotch-yoke ferry (sim_gen task `native_libero_i365`).

Derived from libero/native_libero (LIBERO pick-and-place: grasp the goal object,
carry it through free space, set it on/in an open goal region), but STRATEGICALLY
different: here the goal region is a sealed VAULT under the floor of a sealed,
roofed TUNNEL — no free-space carry can reach it. The only entry is a hole in
the tunnel floor, and the only thing that can move cargo along the tunnel is a
three-sided FERRY CAGE driven by a rotary CRANK through a Scotch-yoke pin/slot
(cage_x = 0.240 + 0.145*cos(theta)). The robot never carries the cargo to the
goal: it (1) cranks the cage to the loading window's dead center, (2) pushes the
cube sideways off an apron through the window into the cage, and (3) cranks
onward so the cage plows the cube down the lane until it tips through the floor
hole into the vault. Plan-level contrast with the seed: rotary-to-linear motion
conversion, an alignment-gated loading step, an ORDER requirement (align before
load — a cube pushed through the window early lands loose in the lane and a
returning cage just plows it into the near-end pocket, stranded), and a terminal
gravity discharge. The seed's strategy (pick the cube up and put it at the goal)
is impossible by construction: the vault is sealed (roof slit 32 mm < cube), and
smoke proves a cube dropped right above the vault rests on the roof.

At theta=+/-180 the yoke is at DEAD CENTER: the cage self-locks against forces
along its rail (loading cannot shove it away), and from there EITHER crank
direction moves the cage toward the hole — the mechanism is friction-held and
holds whatever state it is left in; the outcome persists hands-off.

Assets are fully procedural (compound-spawner pattern; children of one body
never self-collide): structure (KINEMATIC: tunnel walls/floor/roof with window,
slit, floor hole, sealed vault, apron table, gantry), ferry cage (DYNAMIC:
3-sided open-bottom box + mast through the roof slit + overhead yoke channel),
crank disc (DYNAMIC: disc + drive pin + handle peg + visual axle), and two free
cubes (red cargo, blue decoy). Joints are authored per-env at bind time
(structure=body0 kinematic, follower=body1; collision filtering applies to that
pair only, so pin<->channel and cage<->cargo contacts — the transmission — still
collide). Sliding surfaces carry a bound polished material (mu ~0.06, min
combine) — PhysX's default ~0.5 friction would fight the yoke.

Per-episode randomization (readback-verified by smoke): crank angle theta0
(sign and magnitude; sets where the cage starts), and the xy poses + yaws of
both cubes on the apron.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.10  align   — cage ever within 12 mm of the window dead center (latched)
  0.20  loaded  — cargo ever aboard the cage (latched)
  0.25  ferry   — running-max aboard progress toward the floor hole (latched)
  0.30  dropped — cargo ever below floor level inside the vault (latched)
capped at 0.85; exactly 1.0 iff success(): cargo at rest on the vault floor,
everything settled and finite. Null policy ~0 (nothing moves; the sampled start
angle band keeps the cage away from the dead center).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawners -----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _cyl(stage, path: str, *, r: float, h: float, center, color, collide: Callable):
    """Z-axis cylinder child at local `center`."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(r))
    cyl.CreateHeightAttr(float(h))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_structure(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the static structure at `prim_path`: KINEMATIC compound. Local
    frame: z=0 ground, tunnel bore along x (interior y in [-bore_hy, bore_hy],
    floor top z=floor_z1, roof underside z=roof_z0). Pieces: sealed tunnel with
    a loading WINDOW in the +y wall, a longitudinal roof SLIT for the cage mast,
    a floor HOLE over the sealed under-floor VAULT, the apron loading table on
    the +y side, and the crank gantry (pedestal + overhead arm) on the -y side."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    x0, x1 = c.bore_x0, c.bore_x1
    hy, wt = c.bore_hy, c.wall_t
    zw = c.roof_z1  # walls run ground -> roof top
    wall = c.wall_color
    # end walls + side walls (the -y wall is solid; the +y wall is split by the window)
    _span(stage, f"{prim_path}/end_near", x=(x0 - wt, x0), y=(-hy - wt, hy + wt),
          z=(0.0, zw), color=wall, collide=collide)
    _span(stage, f"{prim_path}/end_far", x=(x1, x1 + wt), y=(-hy - wt, hy + wt),
          z=(0.0, zw), color=wall, collide=collide)
    _span(stage, f"{prim_path}/wall_yn", x=(x0 - wt, x1 + wt), y=(-hy - wt, -hy),
          z=(0.0, zw), color=wall, collide=collide)
    _span(stage, f"{prim_path}/wall_yp_near", x=(x0 - wt, c.win_x0), y=(hy, hy + wt),
          z=(0.0, zw), color=wall, collide=collide)
    _span(stage, f"{prim_path}/wall_yp_far", x=(c.win_x1, x1 + wt), y=(hy, hy + wt),
          z=(0.0, zw), color=wall, collide=collide)
    _span(stage, f"{prim_path}/win_sill", x=(c.win_x0, c.win_x1), y=(hy, hy + wt),
          z=(0.0, c.win_z0), color=wall, collide=collide)
    _span(stage, f"{prim_path}/win_header", x=(c.win_x0, c.win_x1), y=(hy, hy + wt),
          z=(c.win_z1, zw), color=wall, collide=collide)
    # floor slab with the vault hole
    _span(stage, f"{prim_path}/floor_main", x=(x0, c.hole_x0), y=(-hy, hy),
          z=(c.floor_z0, c.floor_z1), color=c.floor_color, collide=collide)
    _span(stage, f"{prim_path}/floor_far", x=(c.hole_x1, x1), y=(-hy, hy),
          z=(c.floor_z0, c.floor_z1), color=c.floor_color, collide=collide)
    # roof with the longitudinal mast slit
    _span(stage, f"{prim_path}/roof_yp", x=(x0, x1), y=(c.slit_hy, hy),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    _span(stage, f"{prim_path}/roof_yn", x=(x0, x1), y=(-hy, -c.slit_hy),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    _span(stage, f"{prim_path}/roof_cap_near", x=(x0, c.slit_x0), y=(-c.slit_hy, c.slit_hy),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    _span(stage, f"{prim_path}/roof_cap_far", x=(c.slit_x1, x1), y=(-c.slit_hy, c.slit_hy),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    # sealed under-floor vault (side walls above already run to the ground)
    _span(stage, f"{prim_path}/vault_near", x=(c.hole_x0 - wt, c.hole_x0), y=(-hy, hy),
          z=(0.0, c.floor_z0), color=c.vault_color, collide=collide)
    _span(stage, f"{prim_path}/vault_far", x=(c.hole_x1, c.hole_x1 + wt), y=(-hy, hy),
          z=(0.0, c.floor_z0), color=c.vault_color, collide=collide)
    # apron loading table — its top steps DOWN onto the sill and then the lane
    # floor (monotonic descent: a slid cube never catches a flush proud edge)
    _span(stage, f"{prim_path}/apron", x=(c.apron_x0, c.apron_x1), y=(hy + wt, c.apron_y1),
          z=(0.0, c.apron_z1), color=c.apron_color, collide=collide)
    # crank gantry: pedestal on the -y side + overhead arm above the handle circle
    _span(stage, f"{prim_path}/pedestal", x=(c.ped_x0, c.ped_x1), y=(c.ped_y0, c.ped_y1),
          z=(0.0, c.arm_z0), color=c.gantry_color, collide=collide)
    _span(stage, f"{prim_path}/arm", x=(c.ped_x0, c.ped_x1), y=(c.ped_y0, c.arm_y1),
          z=(c.arm_z0, c.arm_z1), color=c.gantry_color, collide=collide)
    body = _mk_material(prim_path, "body", c.body_mu_s, c.body_mu_d, "average")
    bind_physics_material(prim_path, body)
    return root


def _spawn_cage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the ferry cage with root origin at ground level on the rail axis
    (the prismatic joint coordinate IS cage_x - crank_x). Three walls (open
    toward the +y window, open-bottomed — cargo rides on the tunnel floor),
    a top strip carrying the mast up through the roof slit, and the overhead
    cap with two rails forming the transverse Scotch-yoke channel."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    hi, ho = c.half_in, c.half_out
    _span(stage, f"{prim_path}/wall_rear", x=(-ho, -hi), y=(-c.wall_hy, c.wall_hy),
          z=(c.wall_z0, c.wall_z1), color=c.cage_color, collide=collide)
    _span(stage, f"{prim_path}/wall_front", x=(hi, ho), y=(-c.wall_hy, c.wall_hy),
          z=(c.wall_z0, c.wall_z1), color=c.cage_color, collide=collide)
    _span(stage, f"{prim_path}/wall_yn", x=(-ho, ho), y=(-c.wall_hy, -c.wall_hy + 0.006),
          z=(c.wall_z0, c.wall_z1), color=c.cage_color, collide=collide)
    _span(stage, f"{prim_path}/top_strip", x=(-ho, ho), y=(-0.015, 0.015),
          z=(c.wall_z1, c.wall_z1 + 0.006), color=c.cage_color, collide=collide)
    _span(stage, f"{prim_path}/mast", x=(-c.mast_hx, c.mast_hx), y=(-c.mast_hx, c.mast_hx),
          z=(c.wall_z1 + 0.006, c.cap_z0), color=c.mast_color, collide=collide)
    _span(stage, f"{prim_path}/cap", x=(-0.030, 0.030), y=(-c.rail_hy - 0.010, c.rail_hy + 0.010),
          z=(c.cap_z0, c.cap_z1), color=c.mast_color, collide=collide)
    _span(stage, f"{prim_path}/rail_p", x=(c.chan_hx, c.chan_hx + 0.012),
          y=(-c.rail_hy, c.rail_hy), z=(c.cap_z1, c.rail_z1),
          color=c.rail_color, collide=collide)
    _span(stage, f"{prim_path}/rail_n", x=(-c.chan_hx - 0.012, -c.chan_hx),
          y=(-c.rail_hy, c.rail_hy), z=(c.cap_z1, c.rail_z1),
          color=c.rail_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawn_disc(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the crank disc with root origin at the crank axis (revolute anchor
    = root; MassAPI mass keeps the CoM on the axle — no gravity torque). Disc +
    downward drive PIN at radius crank_r (rides in the cage channel) + upward
    HANDLE peg at the same azimuth (what the robot pushes) + a visual axle stub
    reaching up toward the gantry arm."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _cyl(stage, f"{prim_path}/disc", r=c.disc_r, h=c.disc_t, center=(0.0, 0.0, 0.0),
         color=c.disc_color, collide=collide)
    _cyl(stage, f"{prim_path}/pin", r=c.pin_r, h=c.pin_h,
         center=(c.crank_r, 0.0, c.pin_dz), color=c.pin_color, collide=collide)
    _cyl(stage, f"{prim_path}/handle", r=c.handle_r, h=c.handle_h,
         center=(c.crank_r, 0.0, c.handle_dz), color=c.handle_color, collide=collide)
    _cyl(stage, f"{prim_path}/axle", r=0.012, h=0.080, center=(0.0, 0.0, 0.050),
         color=c.pin_color, collide=collide)
    _dyn_body(root, c.mass, 0.05, c.ang_damp)
    slick = _mk_material(prim_path, "slick", c.slide_mu_s, c.slide_mu_d, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "structure" not in _SPAWNER_CACHE:

        @configclass
        class StructureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_structure)
            bore_x0: float = -0.040
            bore_x1: float = 0.442
            bore_hy: float = 0.050
            wall_t: float = 0.008
            floor_z0: float = 0.115
            floor_z1: float = 0.130
            roof_z0: float = 0.220
            roof_z1: float = 0.232
            win_x0: float = 0.062
            win_x1: float = 0.127
            win_z0: float = 0.130
            win_z1: float = 0.200
            hole_x0: float = 0.345
            hole_x1: float = 0.425
            slit_x0: float = 0.075
            slit_x1: float = 0.405
            slit_hy: float = 0.016
            apron_x0: float = -0.010
            apron_x1: float = 0.230
            apron_y1: float = 0.230
            apron_z1: float = 0.133
            ped_x0: float = 0.215
            ped_x1: float = 0.265
            ped_y0: float = -0.335
            ped_y1: float = -0.295
            arm_y1: float = 0.045
            arm_z0: float = 0.470
            arm_z1: float = 0.500
            wall_color: tuple = (0.55, 0.57, 0.62)
            floor_color: tuple = (0.42, 0.44, 0.50)
            roof_color: tuple = (0.60, 0.62, 0.68)
            vault_color: tuple = (0.30, 0.32, 0.38)
            apron_color: tuple = (0.72, 0.60, 0.42)
            gantry_color: tuple = (0.35, 0.37, 0.42)
            contact_offset: float = 0.0015
            body_mu_s: float = 0.35
            body_mu_d: float = 0.30

        @configclass
        class CageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cage)
            half_in: float = 0.045
            half_out: float = 0.051
            wall_hy: float = 0.045
            wall_z0: float = 0.132
            wall_z1: float = 0.190
            mast_hx: float = 0.010
            cap_z0: float = 0.300
            cap_z1: float = 0.306
            rail_z1: float = 0.346
            chan_hx: float = 0.011
            rail_hy: float = 0.160
            mass: float = 0.50
            lin_damp: float = 1.5
            ang_damp: float = 2.0
            cage_color: tuple = (0.86, 0.68, 0.12)
            mast_color: tuple = (0.75, 0.58, 0.10)
            rail_color: tuple = (0.55, 0.42, 0.08)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        @configclass
        class DiscSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_disc)
            disc_r: float = 0.170
            disc_t: float = 0.020
            crank_r: float = 0.145
            pin_r: float = 0.007
            pin_h: float = 0.054
            pin_dz: float = -0.035
            handle_r: float = 0.009
            handle_h: float = 0.078
            handle_dz: float = 0.049
            mass: float = 0.60
            ang_damp: float = 1.5
            disc_color: tuple = (0.25, 0.27, 0.32)
            pin_color: tuple = (0.70, 0.72, 0.78)
            handle_color: tuple = (0.80, 0.15, 0.10)
            contact_offset: float = 0.0015
            slide_mu_s: float = 0.06
            slide_mu_d: float = 0.05

        _SPAWNER_CACHE["structure"] = StructureSpawnerCfg
        _SPAWNER_CACHE["cage"] = CageSpawnerCfg
        _SPAWNER_CACHE["disc"] = DiscSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CrankFerrySceneCfg(BaseCfg):
    """Config for `CrankFerryScene`. The mechanism contract is asserted in
    `__post_init__`: the window really sits inside the aligned cage's interior,
    both cubes really pass the window but not the roof slit, the plowed cargo
    really reaches past the hole edge, mis-ordered cargo really fits the
    near-end pocket (no jam), and every sweep really clears the static geometry."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    vault_gx0: float = tunable(0.337)     # success box (cargo center, env frame)
    vault_gx1: float = tunable(0.433)
    vault_gy: float = tunable(0.055)
    vault_gz: float = tunable(0.070)      # cargo center at/below this = on the vault floor
    settle_lin: float = tunable(0.05)     # max |lin vel| of movers when judging (m/s)
    settle_ang: float = tunable(0.40)     # max |ang vel| of the disc when judging (rad/s)
    align_tol: float = tunable(0.012)     # cage-at-window latch tolerance (m)
    aboard_x: float = tunable(0.055)      # |cargo_x - cage_x| for "aboard"
    aboard_y: float = tunable(0.052)      # |cargo_y| for "aboard" (inside the bore)
    aboard_z0: float = tunable(0.120)     # aboard z band (riding the tunnel floor)
    aboard_z1: float = tunable(0.212)
    ferry_z0: float = tunable(0.100)      # ferry credit keeps latching through the tip
    ferry_end: float = tunable(0.350)     # aboard progress normalizer (the hole edge)
    drop_z: float = tunable(0.090)        # cargo center below this = fallen through

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    th0_lo: float = tunable(55.0)         # |theta0| band, degrees (sign random)
    th0_hi: float = tunable(125.0)
    cargo_x: tuple = tunable((0.030, 0.070))   # cargo spawn band on the apron
    decoy_x: tuple = tunable((0.150, 0.185))   # decoy spawn band on the apron
    spawn_y: tuple = tunable((0.100, 0.185))   # shared y band on the apron

    # --- info: tunnel (env frame: z=0 ground, bore along x) --------------------------------------
    bore_x0: float = info(-0.040)         # bore interior x span
    bore_x1: float = info(0.442)
    bore_hy: float = info(0.050)          # bore interior half-width
    wall_t: float = info(0.008)
    floor_z0: float = info(0.115)         # floor slab (top = lane surface)
    floor_z1: float = info(0.130)
    roof_z0: float = info(0.220)          # roof slab (underside = bore ceiling)
    roof_z1: float = info(0.232)
    win_x0: float = info(0.062)           # loading window in the +y wall
    win_x1: float = info(0.127)
    win_z0: float = info(0.1315)          # sill top (BELOW the apron top, ABOVE the lane
    win_z1: float = info(0.200)           # floor: the transit surface descends monotonically)
    hole_x0: float = info(0.345)          # floor hole over the vault (full bore width)
    hole_x1: float = info(0.425)
    slit_x0: float = info(0.075)          # roof slit for the cage mast
    slit_x1: float = info(0.405)
    slit_hy: float = info(0.016)
    apron_x0: float = info(-0.010)        # apron table (+y side; its top sits 1.5 mm proud
    apron_x1: float = info(0.230)         # of the sill so a slid cube steps DOWN, never
    apron_y1: float = info(0.230)         # catching a flush proud edge)
    apron_z1: float = info(0.133)
    ped_x0: float = info(0.215)           # gantry pedestal + overhead arm
    ped_x1: float = info(0.265)
    ped_y0: float = info(-0.335)
    ped_y1: float = info(-0.295)
    arm_y1: float = info(0.045)
    arm_z0: float = info(0.470)
    arm_z1: float = info(0.500)
    # --- info: Scotch yoke (cage_x = crank_x + crank_r * cos(theta)) -----------------------------
    crank_x: float = info(0.240)          # crank axis x (y=0)
    crank_z: float = info(0.372)          # disc mid-plane height
    crank_r: float = info(0.145)          # pin/handle orbit radius
    disc_r: float = info(0.170)
    disc_t: float = info(0.020)
    pin_r: float = info(0.007)
    pin_h: float = info(0.054)
    pin_dz: float = info(-0.035)          # pin center below the disc plane
    handle_r: float = info(0.009)
    handle_h: float = info(0.078)
    handle_dz: float = info(0.049)        # handle peg spans z 0.382..0.460
    disc_mass: float = info(0.60)
    stroke: float = info(0.155)           # prismatic limits +/- about crank_x
    # --- info: cage ------------------------------------------------------------------------------
    cage_half_in: float = info(0.045)     # interior half-length (90 mm interior)
    cage_half_out: float = info(0.051)
    cage_wall_z0: float = info(0.132)     # walls float 2 mm above the floor
    cage_wall_z1: float = info(0.190)
    mast_hx: float = info(0.010)
    cap_z0: float = info(0.300)
    cap_z1: float = info(0.306)
    rail_z1: float = info(0.346)
    chan_hx: float = info(0.011)          # yoke channel half-gap (pin r + 4 mm slack)
    rail_hy: float = info(0.160)
    cage_mass: float = info(0.50)
    # --- info: cubes -----------------------------------------------------------------------------
    cargo_s: float = info(0.046)
    cargo_m: float = info(0.12)
    decoy_s: float = info(0.056)
    decoy_m: float = info(0.18)
    # --- info: materials -------------------------------------------------------------------------
    body_mu_s: float = info(0.35)
    body_mu_d: float = info(0.30)
    slide_mu_s: float = info(0.06)        # yoke + plow surfaces: default ~0.5 would fight it
    slide_mu_d: float = info(0.05)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (sum = 0.85 = the non-success cap) ---------------------------------
    w_align: float = info(0.10)
    w_loaded: float = info(0.20)
    w_ferry: float = info(0.25)
    w_drop: float = info(0.30)

    def __post_init__(self) -> None:
        xl = self.crank_x - self.crank_r   # 0.095: cage at theta=+/-180 (window dead center)
        xd = self.crank_x + self.crank_r   # 0.385: cage at theta=0/360 (over the hole)
        # window sits inside the aligned cage interior for every cage_x within align_tol
        assert self.win_x0 + 1e-9 >= xl + self.align_tol - self.cage_half_in, \
            "window left edge must stay inside the aligned cage"
        assert self.win_x1 <= xl - self.align_tol + self.cage_half_in + 1e-9, \
            "window right edge must stay inside the aligned cage"
        # both cubes pass the window; neither passes the roof slit; cargo passes the hole
        assert self.cargo_s + 0.014 <= self.win_x1 - self.win_x0, "cargo must clear the window width"
        assert self.cargo_s + 0.014 <= self.win_z1 - self.apron_z1, \
            "cargo (resting on the apron) must clear the window header"
        # the transit surface descends monotonically: apron -> sill -> lane floor
        # (a slid cube must never meet a proud edge — flush colliders catch it)
        assert self.floor_z1 < self.win_z0 < self.apron_z1, \
            "sill top must sit strictly between the lane floor and the apron top"
        assert self.apron_z1 - self.floor_z1 <= 0.006, "transit steps must stay small"
        assert self.decoy_s + 0.006 <= self.win_x1 - self.win_x0, \
            "decoy must also fit the window (identity, not geometry, rejects it)"
        assert 2 * self.slit_hy < self.cargo_s - 0.010, "roof slit must not pass the cargo"
        assert self.cargo_s * math.sqrt(2.0) <= (self.hole_x1 - self.hole_x0) - 0.008, \
            "cargo (any yaw) must drop through the floor hole"
        # cargo inside the cage stays below the wall top (contained while plowed)
        assert self.floor_z1 + self.cargo_s <= self.cage_wall_z1 - 0.010, \
            "cargo must ride below the cage wall top"
        # mis-ordered cargo is plowed into the near pocket — which must FIT it (no jam)
        assert (xl - self.cage_half_out) - self.bore_x0 >= max(self.cargo_s, self.decoy_s) + 0.010, \
            "near-end pocket must swallow a stranded cube without jamming"
        # far end clearance at full stroke
        assert self.bore_x1 - (xd + self.cage_half_out) >= 0.004, "cage must clear the far wall"
        # roof slit covers the mast at the prismatic limits, with lateral clearance
        assert self.slit_x0 <= self.crank_x - self.stroke - self.mast_hx + 1e-9
        assert self.slit_x1 >= self.crank_x + self.stroke + self.mast_hx - 1e-9
        assert self.mast_hx + 0.004 <= self.slit_hy, "mast needs lateral slack in the slit"
        # yoke: rails span the pin orbit; pin fits the channel; pin engages the rail band
        assert self.rail_hy >= self.crank_r + self.pin_r + 0.006, "rails must span the pin orbit"
        assert self.pin_r + 0.003 <= self.chan_hx, "pin needs slack in the channel"
        pin_z0 = self.crank_z + self.pin_dz - self.pin_h / 2
        pin_z1 = self.crank_z + self.pin_dz + self.pin_h / 2
        assert pin_z0 >= self.cap_z1 + 0.002, "pin tip must clear the cap"
        assert min(pin_z1, self.rail_z1) - max(pin_z0, self.cap_z1) >= 0.030, \
            "pin/rail engagement band must be deep"
        # heights: handle under the arm; everything's sweep clears the pedestal
        assert self.crank_z + self.handle_dz + self.handle_h / 2 <= self.arm_z0 - 0.008, \
            "handle top must clear the gantry arm"
        assert max(self.disc_r, self.crank_r + self.handle_r, self.rail_hy + 0.010) \
            <= -self.ped_y1 - 0.02, "rotating/sliding sweeps must clear the pedestal"
        assert self.crank_z - self.disc_t / 2 >= self.rail_z1 + 0.010, \
            "disc underside must clear the rails"
        # start band keeps the cage away from both dead centers
        x_lo = self.crank_x + self.crank_r * math.cos(math.radians(self.th0_hi))
        x_hi = self.crank_x + self.crank_r * math.cos(math.radians(self.th0_lo))
        assert 0.0 < self.th0_lo < self.th0_hi < 180.0
        assert x_lo - xl >= self.align_tol + 0.020, "start band must exclude the window"
        assert xd - x_hi >= 0.020, "start band must exclude the hole end"
        # spawn bands: cubes stay on the apron (any yaw), never overlapping each other
        rc = self.cargo_s * math.sqrt(2.0) / 2
        rd = self.decoy_s * math.sqrt(2.0) / 2
        assert self.cargo_x[0] - rc >= self.apron_x0 and self.decoy_x[1] + rd <= self.apron_x1
        assert self.decoy_x[0] - rd >= self.cargo_x[1] + rc + 0.004, "cube bands must not overlap"
        assert self.spawn_y[0] - rd >= self.bore_hy + self.wall_t + 0.002
        assert self.spawn_y[1] + rd <= self.apron_y1
        # rubric geometry: plowed cargo really passes the hole edge; goal box below the floor
        assert (xd - self.cage_half_in + self.cargo_s / 2) >= self.hole_x0 + 0.015, \
            "full stroke must plow the cargo past the hole edge"
        assert self.hole_x0 <= self.ferry_end <= self.hole_x0 + 0.010
        assert self.vault_gx0 <= self.hole_x0 and self.vault_gx1 >= self.hole_x1
        assert self.vault_gz < self.floor_z0 - 0.020, \
            "goal z band must exclude anything resting at lane level"
        assert self.cargo_s / 2 + 0.020 <= self.drop_z < self.floor_z0
        assert abs(self.w_align + self.w_loaded + self.w_ferry + self.w_drop - 0.85) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("crank_ferry")
class CrankFerryScene(BaseScene):
    cfg: CrankFerrySceneCfg

    def __init__(self, cfg: CrankFerrySceneCfg | None = None) -> None:
        super().__init__(cfg or CrankFerrySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        structure_spawn = cls["structure"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            bore_x0=c.bore_x0, bore_x1=c.bore_x1, bore_hy=c.bore_hy, wall_t=c.wall_t,
            floor_z0=c.floor_z0, floor_z1=c.floor_z1, roof_z0=c.roof_z0, roof_z1=c.roof_z1,
            win_x0=c.win_x0, win_x1=c.win_x1, win_z0=c.win_z0, win_z1=c.win_z1,
            hole_x0=c.hole_x0, hole_x1=c.hole_x1,
            slit_x0=c.slit_x0, slit_x1=c.slit_x1, slit_hy=c.slit_hy,
            apron_x0=c.apron_x0, apron_x1=c.apron_x1, apron_y1=c.apron_y1,
            apron_z1=c.apron_z1,
            ped_x0=c.ped_x0, ped_x1=c.ped_x1, ped_y0=c.ped_y0, ped_y1=c.ped_y1,
            arm_y1=c.arm_y1, arm_z0=c.arm_z0, arm_z1=c.arm_z1,
            contact_offset=c.contact_offset, body_mu_s=c.body_mu_s, body_mu_d=c.body_mu_d)
        cage_spawn = cls["cage"](
            half_in=c.cage_half_in, half_out=c.cage_half_out, wall_hy=c.bore_hy - 0.005,
            wall_z0=c.cage_wall_z0, wall_z1=c.cage_wall_z1, mast_hx=c.mast_hx,
            cap_z0=c.cap_z0, cap_z1=c.cap_z1, rail_z1=c.rail_z1, chan_hx=c.chan_hx,
            rail_hy=c.rail_hy, mass=c.cage_mass,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)
        disc_spawn = cls["disc"](
            disc_r=c.disc_r, disc_t=c.disc_t, crank_r=c.crank_r, pin_r=c.pin_r, pin_h=c.pin_h,
            pin_dz=c.pin_dz, handle_r=c.handle_r, handle_h=c.handle_h, handle_dz=c.handle_dz,
            mass=c.disc_mass,
            contact_offset=c.contact_offset, slide_mu_s=c.slide_mu_s, slide_mu_d=c.slide_mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        cube_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.body_mu_s, dynamic_friction=c.body_mu_d, restitution=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "structure": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Structure", spawn=structure_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # Cage and disc are authored at a CONSISTENT yoke pose (theta=90 deg,
            # cage_x=crank_x): the bind-time joints anchor at these authored poses.
            "cage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cage", spawn=cage_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.crank_x, 0.0, 0.0))),
            "disc": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Disc", spawn=disc_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.crank_x, 0.0, c.crank_z),
                    rot=(math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)))),
            "cargo": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cargo",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cargo_s, c.cargo_s, c.cargo_s),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cargo_m),
                    rigid_props=rigid, collision_props=coll, physics_material=cube_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.20, 0.10))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.05, 0.15, 0.16))),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sim_utils.CuboidCfg(
                    size=(c.decoy_s, c.decoy_s, c.decoy_s),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.decoy_m),
                    rigid_props=rigid, collision_props=coll, physics_material=cube_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.30, 0.80))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.17, 0.15, 0.16))),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.structure: RigidObject = env.iscene["structure"]
        self.cage: RigidObject = env.iscene["cage"]
        self.disc: RigidObject = env.iscene["disc"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        c = self.cfg
        self.XL = c.crank_x - c.crank_r    # cage_x at the window dead center
        self.XD = c.crank_x + c.crank_r    # cage_x at the hole end
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.th0 = torch.zeros(n, device=dev)     # sampled crank angle (rad)
        self.x0 = torch.zeros(n, device=dev)      # cage_x implied by th0
        # latches (partial credit survives transients; success is judged live)
        self._align = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ferry = torch.zeros(n, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env joints, authored ONCE at bind time against the AUTHORED poses
        (cage at cage_x=crank_x, disc at theta=90 deg — a consistent yoke pose).
        Joint collision filtering only disables the structure<->follower pair,
        so pin<->channel and cage<->cargo transmission contacts still collide.
        The revolute crank has NO limits (continuous rotation)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/ferry_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Structure"])
            j.CreateBody1Rel().SetTargets([f"{base}/Cage"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(float(c.crank_x), 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.stroke))
            j.CreateUpperLimitAttr(float(c.stroke))
            r = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/crank_pivot")
            r.CreateBody0Rel().SetTargets([f"{base}/Structure"])
            r.CreateBody1Rel().SetTargets([f"{base}/Disc"])
            r.CreateCollisionEnabledAttr(False)
            r.CreateAxisAttr("Z")
            r.CreateLocalPos0Attr(Gf.Vec3f(float(c.crank_x), 0.0, float(c.crank_z)))
            r.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            r.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            r.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # no limit attrs: continuous crank

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: structure re-asserted at its fixed pose, yoke written
        CONSISTENTLY (disc at sampled theta0, cage at the implied cage_x — one
        linkage, one write), cubes at sampled apron poses, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def place(body, dx, dy, dz, yaw=None) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = origin[:, 0] + dx
            s[:, 1] = origin[:, 1] + dy
            s[:, 2] = origin[:, 2] + dz
            if yaw is None:
                s[:, 3] = 1.0
            else:
                s[:, 3] = torch.cos(yaw / 2)
                s[:, 6] = torch.sin(yaw / 2)
            body.write_root_state_to_sim(s, env_ids)

        zeros = torch.zeros(m, device=dev)
        place(self.structure, zeros, zeros, zeros)

        _ = torch.rand(m, device=dev)  # burn: first post-seed draw is degenerate
        sgn = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0).to(dev)
        lo, hi = math.radians(c.th0_lo), math.radians(c.th0_hi)
        th0 = sgn * (lo + torch.rand(m, device=dev) * (hi - lo))
        self.th0[env_ids] = th0
        x0 = c.crank_x + c.crank_r * torch.cos(th0)
        self.x0[env_ids] = x0
        place(self.disc, zeros + c.crank_x, zeros, zeros + c.crank_z, yaw=th0)
        place(self.cage, x0, zeros, zeros)

        u = torch.rand(m, 6, device=dev)
        zc = c.apron_z1 + c.cargo_s / 2 + 0.003
        zd = c.apron_z1 + c.decoy_s / 2 + 0.003
        cx = c.cargo_x[0] + u[:, 0] * (c.cargo_x[1] - c.cargo_x[0])
        cy = c.spawn_y[0] + u[:, 1] * (c.spawn_y[1] - c.spawn_y[0])
        place(self.cargo, cx, cy, zeros + zc, yaw=(u[:, 2] * 2 - 1) * math.pi)
        dx = c.decoy_x[0] + u[:, 3] * (c.decoy_x[1] - c.decoy_x[0])
        dy = c.spawn_y[0] + u[:, 4] * (c.spawn_y[1] - c.spawn_y[0])
        place(self.decoy, dx, dy, zeros + zd, yaw=(u[:, 5] * 2 - 1) * math.pi)

        self._align[env_ids] = False
        self._loaded[env_ids] = False
        self._dropped[env_ids] = False
        self._ferry[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "structure": self.structure.data.root_state_w[env_ids].clone(),
            "cage": self.cage.data.root_state_w[env_ids].clone(),
            "disc": self.disc.data.root_state_w[env_ids].clone(),
            "cargo": self.cargo.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "th0": self.th0[env_ids].clone(),
            "x0": self.x0[env_ids].clone(),
            "align": self._align[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "dropped": self._dropped[env_ids].clone(),
            "ferry": self._ferry[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.structure.write_root_state_to_sim(state["structure"], env_ids)
        self.cage.write_root_state_to_sim(state["cage"], env_ids)
        self.disc.write_root_state_to_sim(state["disc"], env_ids)
        self.cargo.write_root_state_to_sim(state["cargo"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.th0[env_ids] = state["th0"]
        self.x0[env_ids] = state["x0"]
        self._align[env_ids] = state["align"]
        self._loaded[env_ids] = state["loaded"]
        self._dropped[env_ids] = state["dropped"]
        self._ferry[env_ids] = state["ferry"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A sealed steel transfer TUNNEL runs along x (bore interior x in "
            f"[{c.bore_x0:+.3f}, {c.bore_x1:+.3f}] m, width {2 * c.bore_hy * 100:.0f} cm, lane "
            f"floor top at z={c.floor_z1:.3f}, roof underside at z={c.roof_z0:.3f}; env frame, "
            f"z=0 ground). Its roof has only a narrow {2 * c.slit_hy * 1000:.0f} mm longitudinal "
            f"slit; its +y wall has one loading WINDOW at x in [{c.win_x0:.3f}, {c.win_x1:.3f}], "
            f"sill top z={c.win_z0:.4f}, header underside z={c.win_z1:.3f}, beside a wooden "
            f"APRON table (top z={c.apron_z1:.3f}, y from {c.bore_hy + c.wall_t:.3f} to "
            f"{c.apron_y1:.3f}; the surface steps DOWN apron->sill->lane floor, so a cube "
            f"slides cleanly through the window without catching an edge). On the "
            f"apron sit two free cubes: the RED cargo cube ({c.cargo_s * 1000:.0f} mm, the goal "
            f"object) and a larger BLUE decoy cube ({c.decoy_s * 1000:.0f} mm, a bystander — "
            f"leave it). The tunnel floor has a HOLE from x={c.hole_x0:.3f} to {c.hole_x1:.3f} "
            f"(full width) over a sealed under-floor VAULT — the vault's ONLY entry; the "
            f"{2 * c.slit_hy * 1000:.0f} mm roof slit passes neither cube, so nothing can be "
            f"dropped in from above.\n"
            f"Inside the tunnel a yellow FERRY CAGE slides along x on a hidden rail: a "
            f"three-sided open-bottom box (interior {2 * c.cage_half_in * 1000:.0f} mm along x, "
            f"walls {1000 * (c.cage_wall_z1 - c.cage_wall_z0):.0f} mm tall, OPEN toward the "
            f"window side and open underneath — cargo rides on the tunnel floor). Its mast "
            f"rises through the roof slit to an overhead crossbar with two rails forming a "
            f"transverse channel. A CRANK DISC (radius {c.disc_r * 100:.0f} cm) turns about a "
            f"vertical axle at (x={c.crank_x:.3f}, y=0) under a gantry arm; its downward pin "
            f"(orbit radius {c.crank_r:.3f} m) rides in that channel — a Scotch yoke: "
            f"cage_x = {c.crank_x:.3f} + {c.crank_r:.3f}*cos(theta). A vertical RED HANDLE peg "
            f"({2 * c.handle_r * 1000:.0f} mm dia, top at z="
            f"{c.crank_z + c.handle_dz + c.handle_h / 2:.3f}) stands on the disc at the same "
            f"radius: push it around its circle to crank. The crank is continuous (no stops) "
            f"and the mechanism is friction-held: it stays wherever it is left.\n"
            f"At theta=+/-180 deg the cage (cage_x={self.XL:.3f}) is centered on the window — "
            f"a DEAD CENTER: pushing the cage along its rail cannot turn the crank, so the "
            f"cage self-locks there for loading, and from that point EITHER crank direction "
            f"moves the cage toward the hole end (cage_x={self.XD:.3f} at theta=0/360). ORDER "
            f"MATTERS: push the red cube through the window ONLY when the cage is there — a "
            f"cube pushed through early lands loose in the lane, and the returning cage just "
            f"plows it into the near-end pocket (x < {self.XL - c.cage_half_out:.3f}), stranded "
            f"and unrecoverable. The episode starts with |theta| in [{c.th0_lo:.0f}, "
            f"{c.th0_hi:.0f}] deg (sign random; cage well away from the window) and both cube "
            f"poses randomized on the apron.\n"
            f"To solve: crank until the cage is centered on the window; push the red cube "
            f"across the apron through the window into the cage (the dead center holds the "
            f"cage still); then crank onward — the cage plows the cube down the sealed lane "
            f"until it tips over the hole edge and drops into the vault. Goal: the red cube "
            f"at rest on the vault floor (center x in [{c.vault_gx0:.3f}, {c.vault_gx1:.3f}], "
            f"|y| <= {c.vault_gy:.3f}, z <= {c.vault_gz:.3f}), everything settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Deliver the red cube into the sealed vault under the tunnel floor: "
            "crank the ferry cage to the loading window, push the red cube "
            "through the window into the cage, then crank onward so the cage "
            "carries the cube to the floor hole and drops it in. Leave the blue "
            "cube alone and finish with everything at rest."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the env frame (structure fixed at origin)."""
        return body.data.root_pos_w - self.env_origins

    def cage_x(self) -> torch.Tensor:
        """(N,) ferry cage position along the tunnel (env frame)."""
        return self._local(self.cage)[:, 0]

    def disc_yaw(self) -> torch.Tensor:
        """(N,) crank angle in (-pi, pi] (the yoke is symmetric in the sign)."""
        q = self.disc.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def settled(self) -> torch.Tensor:
        """(N,) bool: cubes and cage slow; disc rotation slow."""
        c = self.cfg
        return (self.cargo.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.cage.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.disc.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.cage.data.root_pos_w, self.disc.data.root_pos_w,
                         self.cargo.data.root_pos_w, self.decoy.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        cx = self.cage_x()
        self._align |= ((cx - self.XL).abs() <= c.align_tol) & fin
        p = self._local(self.cargo)
        near = ((p[:, 0] - cx).abs() <= c.aboard_x) & (p[:, 1].abs() <= c.aboard_y)
        aboard = near & (p[:, 2] >= c.aboard_z0) & (p[:, 2] <= c.aboard_z1)
        self._loaded |= aboard & fin
        ride = near & (p[:, 2] >= c.ferry_z0) & (p[:, 2] <= c.aboard_z1)
        prog = ((p[:, 0] - self.XL) / (c.ferry_end - self.XL)).clamp(0.0, 1.0)
        self._ferry = torch.maximum(
            self._ferry, torch.where(ride & fin, prog, torch.zeros_like(prog)))
        in_vault_xy = (p[:, 0] >= c.vault_gx0) & (p[:, 0] <= c.vault_gx1) \
            & (p[:, 1].abs() <= c.vault_gy)
        self._dropped |= (p[:, 2] < c.drop_z) & in_vault_xy & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the red cube rests on the vault floor — below lane level,
        inside the vault box — with everything settled and finite: a LIVE
        physical outcome. The vault is sealed except for the floor hole inside
        the sealed tunnel, so the only physical route is the ferry."""
        c = self.cfg
        self._update_latches()
        p = self._local(self.cargo)
        in_box = (p[:, 0] >= c.vault_gx0) & (p[:, 0] <= c.vault_gx1) \
            & (p[:, 1].abs() <= c.vault_gy) & (p[:, 2] <= c.vault_gz)
        return in_box & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 align + 0.20 loaded + 0.25 ferry progress
        + 0.30 dropped, all latched, capped at 0.85; exactly 1.0 iff success()
        holds live. Doing nothing scores ~0 (the sampled start keeps the cage
        off the dead center; the cubes start on the apron)."""
        c = self.cfg
        self._update_latches()
        base = (c.w_align * self._align.float() + c.w_loaded * self._loaded.float()
                + c.w_ferry * self._ferry + c.w_drop * self._dropped.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="crank_ferry", robot="null"))
