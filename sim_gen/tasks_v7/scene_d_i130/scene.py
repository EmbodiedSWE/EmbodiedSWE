"""CarouselVaultScene — index a turntable vault by its exposed rim, open the top
shutter, and lift the red cube out through the roof window onto the pad
(sim_gen task `scene_d_i130`).

Derived from calvin/scene_D but STRATEGICALLY different: the CALVIN table is a
menu of independent single-DOF primitives — press the button, flick the switch,
push the slider to its other end, pull the drawer, pick a block off the open
tabletop. Every seed strategy is "actuate one binary DOF" or "grasp a block in
free space". Here NOTHING is graspable in free space and no DOF is binary: both
cubes sit in pockets on a free-spinning TURNTABLE sealed inside a vault (solid
walls, a roof with one square access window, a sliding SHUTTER over the window).
The disc is reachable only where its rim protrudes through a front slit as an
exposed thumbwheel lip. The plan is a chain the seed never needs: (a) slide the
shutter clear of the window, (b) rotate the disc by its rim until the RED cube's
pocket indexes under the window — a CONTINUOUS angular alignment target, not an
end-stop push, randomized per episode over a ~260 degree band, (c) reach down
through the window into the pocket, lift the red cube out, and (d) set it on
the green pad on the ground. A blue decoy cube rides the opposite pocket; the
rubric keys the red body. Physics enforces the partial order: the closed
shutter provably caps any lift under the window, and everywhere else the roof
does (both driven with real forces in smoke).

No stored energy: the disc is a plain revolute rotor (viscous damping only),
the shutter a plain prismatic plate with hard stops, the cubes free bodies —
every outcome persists hands-off.

Assets are fully procedural (compound-spawner pattern; children of one body
never self-collide): housing (KINEMATIC: plinth deck, three walls, slitted
front wall, roof with window), disc (DYNAMIC cylinder + two pocket wall sets),
shutter (DYNAMIC plate + grip knob), two cubes, one kinematic goal pad. The
disc revolute and shutter prismatic joints are authored per-env at bind time
(housing = kinematic body0; collision filtering applies to those pairs only,
so cube<->disc, cube<->housing, cube<->shutter contacts — the containment —
still collide).

Per-episode randomization (readback-verified by smoke): the disc's initial
yaw theta0 (uniform over +/-[50, 310] deg — never starting indexed, nor even
window-extractable) which
carries both cubes with it, and the goal pad's ground xy.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.15  opened   — shutter ever slid past `open_thresh` (window clear)
  0.20  indexed  — red pocket centre ever within `index_tol` of the window centre
  0.25  out      — red cube ever above the roof plane (or clear of the footprint)
capped at 0.60; exactly 1.0 iff success(): the red cube resting ON the pad,
everything settled and finite. Null policy ~0 (shutter closed, disc parked off
window, cubes captive). The seed's strategies score at most the single latch
they mimic (shutter-only = its slider push -> 0.15) and placing the DECOY on
the pad scores 0.

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


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the vault housing: KINEMATIC compound. Local frame: disc axis at
    x=y=0, z=0 ground. Plinth deck, back/side walls, slitted front wall (the
    disc rim protrudes through the slit as the exposed thumbwheel), and the
    roof with one square access window centred over the pocket circle's front
    point (win_cx, 0)."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/deck", x=(c.deck_x0, c.front_out), y=(-c.deck_hy, c.deck_hy),
          z=(0.0, c.deck_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/wall_back", x=(c.deck_x0, c.back_in), y=(-c.deck_hy, c.deck_hy),
          z=(c.deck_z1, c.roof_z0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/wall_yp", x=(c.back_in, c.front_out), y=(c.side_in, c.deck_hy),
          z=(c.deck_z1, c.roof_z0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/wall_yn", x=(c.back_in, c.front_out), y=(-c.deck_hy, -c.side_in),
          z=(c.deck_z1, c.roof_z0), color=c.body_color, collide=collide)
    # front wall: full-width above the slit; side cheeks beside the slit
    _span(stage, f"{prim_path}/front_top", x=(c.front_in, c.front_out), y=(-c.deck_hy, c.deck_hy),
          z=(c.slit_z1, c.roof_z0), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/front_yp", x=(c.front_in, c.front_out), y=(c.slit_hy, c.deck_hy),
          z=(c.deck_z1, c.slit_z1), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/front_yn", x=(c.front_in, c.front_out), y=(-c.deck_hy, -c.slit_hy),
          z=(c.deck_z1, c.slit_z1), color=c.front_color, collide=collide)
    # roof with the access window (win_x0..win_x1, +/-win_hy)
    _span(stage, f"{prim_path}/roof_back", x=(c.deck_x0, c.win_x0), y=(-c.deck_hy, c.deck_hy),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    _span(stage, f"{prim_path}/roof_front", x=(c.win_x1, c.front_out), y=(-c.deck_hy, c.deck_hy),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    _span(stage, f"{prim_path}/roof_yp", x=(c.win_x0, c.win_x1), y=(c.win_hy, c.deck_hy),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    _span(stage, f"{prim_path}/roof_yn", x=(c.win_x0, c.win_x1), y=(-c.deck_hy, -c.win_hy),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    wood = _mk_material(prim_path, "wood", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawn_disc(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the turntable with root at the disc CENTRE (the revolute anchor):
    a cylinder plus two open-top pocket wall sets at local (+r_p, 0) (the RED
    pocket) and (-r_p, 0) (the BLUE pocket). Pocket walls rise from the disc
    top; the roof passes `roof gap` above their rims, so a pocketed cube is
    captive everywhere except under the window."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import Gf, UsdGeom

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/plate")
    cyl.CreateRadiusAttr(float(c.disc_r))
    cyl.CreateHeightAttr(float(c.disc_t))
    cyl.CreateAxisAttr("Z")
    cyl.CreateExtentAttr([Gf.Vec3f(-c.disc_r, -c.disc_r, -c.disc_t / 2),
                          Gf.Vec3f(c.disc_r, c.disc_r, c.disc_t / 2)])
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*c.disc_color)])
    collide(cyl.GetPrim())
    a, t = c.pock_a, c.pock_t
    z0, z1 = c.disc_t / 2, c.disc_t / 2 + c.pock_h
    for tag, cx in (("p", c.r_p), ("d", -c.r_p)):
        _span(stage, f"{prim_path}/{tag}_xp", x=(cx + a, cx + a + t), y=(-a - t, a + t),
              z=(z0, z1), color=c.pock_color, collide=collide)
        _span(stage, f"{prim_path}/{tag}_xn", x=(cx - a - t, cx - a), y=(-a - t, a + t),
              z=(z0, z1), color=c.pock_color, collide=collide)
        _span(stage, f"{prim_path}/{tag}_yp", x=(cx - a, cx + a), y=(a, a + t),
              z=(z0, z1), color=c.pock_color, collide=collide)
        _span(stage, f"{prim_path}/{tag}_yn", x=(cx - a, cx + a), y=(-a - t, -a),
              z=(z0, z1), color=c.pock_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    grippy = _mk_material(prim_path, "grippy", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, grippy)
    return root


def _spawn_shutter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the shutter with root at the plate CENTRE: a square plate that
    overlaps the roof window all round, plus the upright grip knob (a jaw-sized
    tab). It floats 2 mm above the roof on its prismatic joint — no rubbing."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/plate", x=(-c.plate_half, c.plate_half),
          y=(-c.plate_half, c.plate_half), z=(-c.plate_ht, c.plate_ht),
          color=c.plate_color, collide=collide)
    _span(stage, f"{prim_path}/knob", x=(-c.knob_hx, c.knob_hx), y=(-c.knob_hy, c.knob_hy),
          z=(c.plate_ht, c.plate_ht + c.knob_h), color=c.knob_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    slick = _mk_material(prim_path, "slick", 0.10, 0.08, "min")
    bind_physics_material(prim_path, slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            deck_x0: float = -0.23
            deck_hy: float = 0.23
            deck_z1: float = 0.10
            back_in: float = -0.21
            side_in: float = 0.21
            front_in: float = 0.15
            front_out: float = 0.17
            slit_hy: float = 0.14
            slit_z1: float = 0.14
            roof_z0: float = 0.19
            roof_z1: float = 0.21
            win_x0: float = 0.005
            win_x1: float = 0.145
            win_hy: float = 0.07
            body_color: tuple = (0.50, 0.37, 0.24)
            front_color: tuple = (0.44, 0.32, 0.20)
            roof_color: tuple = (0.36, 0.38, 0.42)
            contact_offset: float = 0.0015
            mu_s: float = 0.60
            mu_d: float = 0.55

        @configclass
        class DiscSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_disc)
            disc_r: float = 0.20
            disc_t: float = 0.03
            r_p: float = 0.075
            pock_a: float = 0.045
            pock_t: float = 0.010
            pock_h: float = 0.050
            mass: float = 1.2
            lin_damp: float = 2.0
            ang_damp: float = 4.0
            disc_color: tuple = (0.72, 0.72, 0.76)
            pock_color: tuple = (0.30, 0.30, 0.34)
            contact_offset: float = 0.0015
            mu_s: float = 0.60
            mu_d: float = 0.55

        @configclass
        class ShutterSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shutter)
            plate_half: float = 0.082
            plate_ht: float = 0.006
            knob_hx: float = 0.012
            knob_hy: float = 0.020
            knob_h: float = 0.048
            mass: float = 0.35
            lin_damp: float = 4.0
            ang_damp: float = 4.0
            plate_color: tuple = (0.16, 0.22, 0.34)
            knob_color: tuple = (0.95, 0.55, 0.10)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["housing"] = HousingSpawnerCfg
        _SPAWNER_CACHE["disc"] = DiscSpawnerCfg
        _SPAWNER_CACHE["shutter"] = ShutterSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CarouselVaultSceneCfg(BaseCfg):
    """Config for `CarouselVaultScene`. The containment contract is asserted in
    `__post_init__`: the pockets really sweep clear of the walls, a pocketed
    cube really cannot escape under the roof, through the slit, or past a
    closed shutter, the indexed pocket really presents the cube inside the
    window, and the exposed rim really protrudes far enough to pinch."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    open_thresh: float = tunable(0.150)   # shutter travel -> "opened" (window clear) (m)
    index_tol: float = tunable(0.030)     # red pocket centre within this of the window centre (m)
    out_z: float = tunable(0.225)         # red cube centre above this -> "out" latch (m)
    out_xy: float = tunable(0.27)         # |x| or |y| beyond this also counts as out (m)
    pad_xy_tol: float = tunable(0.050)    # cube centre within this of the pad centre (m)
    pad_z_tol: float = tunable(0.012)     # cube rest height tolerance on the pad (m)
    settle_lin: float = tunable(0.05)     # max |lin vel| of movers when judging (m/s)
    settle_ang: float = tunable(0.30)     # max disc |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    theta0_min_deg: float = tunable(50.0)   # |initial disc yaw| lower bound (never extractable)
    theta0_max_deg: float = tunable(310.0)  # upper bound (yaw in +/-[min, max] deg)
    pad_x: tuple = tunable((0.36, 0.54))    # pad centre x band on the ground (m)
    pad_y: tuple = tunable((-0.44, -0.14))  # pad centre y band on the ground (m)

    # --- info: housing (local frame: disc axis at x=y=0, z=0 ground) -----------------------------
    deck_x0: float = info(-0.23)
    deck_hy: float = info(0.23)
    deck_z1: float = info(0.10)           # plinth top (chamber floor)
    back_in: float = info(-0.21)
    side_in: float = info(0.21)
    front_in: float = info(0.15)          # front wall inner face
    front_out: float = info(0.17)         # front wall outer face (rim protrudes past it)
    slit_hy: float = info(0.14)           # front slit half-width
    slit_z1: float = info(0.14)           # slit top (slit spans deck_z1..slit_z1)
    roof_z0: float = info(0.19)
    roof_z1: float = info(0.21)
    win_x0: float = info(0.005)           # roof window (centred on (r_p, 0))
    win_x1: float = info(0.145)
    win_hy: float = info(0.07)
    # --- info: disc / pockets ---------------------------------------------------------------------
    disc_r: float = info(0.20)
    disc_t: float = info(0.03)
    disc_z: float = info(0.119)           # disc centre height (4 mm above the deck)
    r_p: float = info(0.075)              # pocket circle radius
    pock_a: float = info(0.045)           # pocket inner half-width
    pock_t: float = info(0.010)           # pocket wall thickness
    pock_h: float = info(0.050)           # pocket wall height above the disc top
    disc_mass: float = info(1.2)
    # --- info: shutter ------------------------------------------------------------------------------
    shut_cx: float = info(0.075)          # shutter root (= window centre x), closed pose
    shut_cz: float = info(0.218)          # plate mid-plane: floats 2 mm above the roof
    plate_half: float = info(0.082)
    plate_ht: float = info(0.006)
    knob_h: float = info(0.048)
    shut_travel: float = info(0.17)       # prismatic limits [0, travel] along +y
    shut_mass: float = info(0.35)
    # --- info: cubes / pad --------------------------------------------------------------------------
    block_s: float = info(0.045)          # cube edge
    block_mass: float = info(0.06)
    pad_half: float = info(0.07)
    pad_h: float = info(0.012)
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.15 + 0.20 + 0.25 = 0.60 = the non-success cap) ------------------
    w_open: float = info(0.15)
    w_index: float = info(0.20)
    w_out: float = info(0.25)

    def __post_init__(self) -> None:
        # -- pocket sweep: outer corner circle clears the front wall's inner face
        r_max = math.hypot(self.r_p + self.pock_a + self.pock_t, self.pock_a + self.pock_t)
        assert r_max <= self.front_in - 0.006, "pocket sweep must clear the front wall"
        # -- and stays on the disc
        assert r_max <= self.disc_r - 0.02, "pockets must sit on the disc"
        # -- window centred over the front pocket point
        assert abs((self.win_x0 + self.win_x1) / 2 - self.r_p) < 1e-9, "window centred on (r_p, 0)"
        # -- indexed pocket presents the whole footprint inside the window with margin
        assert self.r_p - (self.pock_a + self.pock_t) >= self.win_x0 + 0.005
        assert self.r_p + (self.pock_a + self.pock_t) <= self.win_x1 - 0.005
        assert self.pock_a + self.pock_t <= self.win_hy - 0.005
        # -- even at the index tolerance edge the CUBE is still fully under the window
        assert self.r_p + self.index_tol + self.block_s / 2 <= self.win_x1 - 0.005
        assert self.index_tol + self.block_s / 2 <= self.win_hy - 0.005
        # -- captive under the roof: wall-top..roof gap is a fraction of the cube
        wall_top = self.disc_z + self.disc_t / 2 + self.pock_h
        gap = self.roof_z0 - wall_top
        assert 0.004 <= gap <= self.block_s / 3, "roof gap must trap the cube"
        assert self.pock_h >= self.block_s + 0.004, "pocket walls must overtop the cube"
        # -- the cube fits the pocket at ANY yaw with clearance
        assert self.block_s * math.sqrt(2.0) / 2 <= self.pock_a - 0.002
        # -- slit: passes the disc freely, never the cube
        assert self.slit_z1 - self.deck_z1 >= self.disc_t + 0.008
        assert self.slit_z1 - self.deck_z1 <= self.block_s - 0.002, "cube must not pass the slit"
        assert self.disc_z - self.disc_t / 2 - self.deck_z1 >= 0.003, "disc floats off the deck"
        assert self.slit_z1 - (self.disc_z + self.disc_t / 2) >= 0.004, "disc clears the slit top"
        # -- slit width covers the disc chord at the inner wall plane
        assert math.sqrt(self.disc_r**2 - self.front_in**2) <= self.slit_hy - 0.006
        # -- exposed thumbwheel: rim protrudes pinchably past the outer face
        assert self.disc_r - self.front_out >= 0.015, "rim must protrude >= 15 mm"
        # -- shutter: overlaps the window closed, clears it at open_thresh
        assert self.plate_half >= max(self.win_x1 - self.win_x0, 2 * self.win_hy) / 2 + 0.008
        assert self.open_thresh - self.plate_half >= self.win_hy - 0.005, \
            "open_thresh must leave the window essentially clear"
        assert self.shut_travel >= self.open_thresh + 0.015
        # -- closed shutter really caps the window (small air gap, far under a cube)
        shut_air = (self.shut_cz - self.plate_ht) - self.roof_z1
        assert 0.0005 <= shut_air <= 0.004, "shutter must float just above the roof"
        # -- out latch fires only genuinely above the roof plane
        assert self.out_z >= self.roof_z1 + 0.010
        assert self.out_xy >= self.front_out + self.plate_half  # outside everything
        # -- start yaw band never starts indexed (index angle ~ 2 asin(tol / 2 r_p))
        idx_deg = math.degrees(2 * math.asin(self.index_tol / (2 * self.r_p)))
        assert self.theta0_min_deg >= idx_deg + 10.0, "theta0 band must start un-indexed"
        assert self.theta0_max_deg <= 360.0 - self.theta0_min_deg
        # -- reset never starts EXTRACTABLE either: the cube passes the window only
        #    if both offset components fit (|off| <= win_hy - s/2); at theta0_min the
        #    pocket's y-offset alone exceeds that bound by >= 8 mm of solid roof
        y_off_min = self.r_p * math.sin(math.radians(self.theta0_min_deg))
        assert y_off_min >= (self.win_hy - self.block_s / 2) + 0.008, \
            "theta0 band must start with the cube capped by solid roof"
        # -- Franka jaw feasibility: cube and knob fit the ~80 mm parallel jaw
        assert self.block_s <= 0.075 and 2 * 0.012 <= 0.075
        # -- on-pad predicate honest: an accepted cube really rests (mostly) on the pad
        assert self.pad_xy_tol + self.block_s / 2 <= self.pad_half + 0.010
        # -- pad band clear of the vault footprint and the protruding rim
        assert self.pad_x[0] - self.pad_half >= self.disc_r + 0.05
        assert self.pad_x[1] > self.pad_x[0] and self.pad_y[1] > self.pad_y[0]
        # -- rubric weights
        assert abs(self.w_open + self.w_index + self.w_out - 0.60) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("carousel_vault")
class CarouselVaultScene(BaseScene):
    cfg: CarouselVaultSceneCfg

    def __init__(self, cfg: CarouselVaultSceneCfg | None = None) -> None:
        super().__init__(cfg or CarouselVaultSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        housing_spawn = cls["housing"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            deck_x0=c.deck_x0, deck_hy=c.deck_hy, deck_z1=c.deck_z1, back_in=c.back_in,
            side_in=c.side_in, front_in=c.front_in, front_out=c.front_out, slit_hy=c.slit_hy,
            slit_z1=c.slit_z1, roof_z0=c.roof_z0, roof_z1=c.roof_z1, win_x0=c.win_x0,
            win_x1=c.win_x1, win_hy=c.win_hy, contact_offset=c.contact_offset)
        disc_spawn = cls["disc"](
            disc_r=c.disc_r, disc_t=c.disc_t, r_p=c.r_p, pock_a=c.pock_a, pock_t=c.pock_t,
            pock_h=c.pock_h, mass=c.disc_mass, contact_offset=c.contact_offset)
        shutter_spawn = cls["shutter"](
            plate_half=c.plate_half, plate_ht=c.plate_ht, knob_h=c.knob_h, mass=c.shut_mass,
            contact_offset=c.contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.60, dynamic_friction=0.55, restitution=0.0)

        def cube(color) -> Any:
            return sim_utils.CuboidCfg(
                size=(c.block_s, c.block_s, c.block_s),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                rigid_props=rigid, collision_props=coll, physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))

        bz = c.disc_z + c.disc_t / 2 + c.block_s / 2 + 0.002
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing", spawn=housing_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # disc and shutter are authored at the joint's zero pose: the bind-time
            # joints anchor at these authored positions.
            "disc": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Disc", spawn=disc_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.disc_z))),
            "shutter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shutter", spawn=shutter_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.shut_cx, 0.0, c.shut_cz))),
            "prize": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/PrizeCube", spawn=cube((0.85, 0.10, 0.08)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.r_p, 0.0, bz))),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/DecoyCube", spawn=cube((0.10, 0.22, 0.85)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-c.r_p, 0.0, bz))),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pad",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.pad_half, 2 * c.pad_half, c.pad_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll, physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.15))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.30, c.pad_h / 2))),
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
        from isaaclab.utils.math import quat_apply

        self._quat_apply = quat_apply
        self.housing: RigidObject = env.iscene["housing"]
        self.disc: RigidObject = env.iscene["disc"]
        self.shutter: RigidObject = env.iscene["shutter"]
        self.prize: RigidObject = env.iscene["prize"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.pad: RigidObject = env.iscene["pad"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke): sampled (theta0, pad_x, pad_y)
        self.layout = torch.zeros(n, 3, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._open_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._index_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._out_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env joints, authored ONCE at bind time against the AUTHORED poses:
        the disc's free revolute (axis z, no limits — a continuous rotor) and the
        shutter's prismatic (axis y, limits = hard stops). Joint collision
        filtering only disables the housing<->disc and housing<->shutter pairs,
        so cube<->everything contacts — the containment — still collide."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/disc_spin")
            j.CreateBody0Rel().SetTargets([f"{base}/Housing"])
            j.CreateBody1Rel().SetTargets([f"{base}/Disc"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.disc_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            k = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/shutter_slide")
            k.CreateBody0Rel().SetTargets([f"{base}/Housing"])
            k.CreateBody1Rel().SetTargets([f"{base}/Shutter"])
            k.CreateCollisionEnabledAttr(False)
            k.CreateAxisAttr("Y")
            k.CreateLocalPos0Attr(Gf.Vec3f(float(c.shut_cx), 0.0, float(c.shut_cz)))
            k.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            k.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            k.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            k.CreateLowerLimitAttr(0.0)
            k.CreateUpperLimitAttr(float(c.shut_travel))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: housing re-asserted at its fixed pose, disc written at a
        SAMPLED yaw theta0 (a rotation about its own joint axis — consistent with
        the revolute), both cubes seated in their pockets (carried to theta0 with
        the disc), shutter closed, pad at a SAMPLED ground xy, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        u = torch.rand(m, 4, device=dev)
        span = math.radians(c.theta0_max_deg - c.theta0_min_deg)
        theta = math.radians(c.theta0_min_deg) + u[:, 0] * span
        sign = torch.where(torch.rand(m, device=dev) < 0.5, -1.0, 1.0)
        theta = theta * sign  # yaw in +/-[min, max] deg — never indexed at start
        px = c.pad_x[0] + u[:, 1] * (c.pad_x[1] - c.pad_x[0])
        py = c.pad_y[0] + u[:, 2] * (c.pad_y[1] - c.pad_y[0])
        self.layout[env_ids, 0] = theta
        self.layout[env_ids, 1] = px
        self.layout[env_ids, 2] = py

        def write(body, dx, dy, dz, yaw=None) -> None:
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
        write(self.housing, zeros, zeros, zeros)
        write(self.disc, zeros, zeros, zeros + c.disc_z, yaw=theta)
        write(self.shutter, zeros + c.shut_cx, zeros, zeros + c.shut_cz)
        bz = c.disc_z + c.disc_t / 2 + c.block_s / 2 + 0.002
        write(self.prize, c.r_p * torch.cos(theta), c.r_p * torch.sin(theta),
              zeros + bz, yaw=theta)
        write(self.decoy, -c.r_p * torch.cos(theta), -c.r_p * torch.sin(theta),
              zeros + bz, yaw=theta)
        write(self.pad, px, py, zeros + c.pad_h / 2)

        self._open_l[env_ids] = False
        self._index_l[env_ids] = False
        self._out_l[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "disc": self.disc.data.root_state_w[env_ids].clone(),
            "shutter": self.shutter.data.root_state_w[env_ids].clone(),
            "prize": self.prize.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "layout": self.layout[env_ids].clone(),
            "open_l": self._open_l[env_ids].clone(),
            "index_l": self._index_l[env_ids].clone(),
            "out_l": self._out_l[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        self.disc.write_root_state_to_sim(state["disc"], env_ids)
        self.shutter.write_root_state_to_sim(state["shutter"], env_ids)
        self.prize.write_root_state_to_sim(state["prize"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.layout[env_ids] = state["layout"]
        self._open_l[env_ids] = state["open_l"]
        self._index_l[env_ids] = state["index_l"]
        self._out_l[env_ids] = state["out_l"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A squat wooden VAULT ({(c.front_out - c.deck_x0) * 100:.0f} x "
            f"{2 * c.deck_hy * 100:.0f} cm footprint, {c.roof_z1 * 100:.0f} cm tall) stands "
            f"on the ground. Inside it a grey TURNTABLE disc spins freely about a vertical "
            f"axis; the vault is sealed except for two openings:\n"
            f"- a square WINDOW ({(c.win_x1 - c.win_x0) * 100:.0f} x {2 * c.win_hy * 100:.0f} cm) "
            f"in the grey roof, on the vault's front half, covered at start by a dark-blue "
            f"sliding SHUTTER with an ORANGE grip knob (it slides sideways, toward the "
            f"vault's left, up to {c.shut_travel * 100:.0f} cm on rails);\n"
            f"- a thin horizontal SLIT across the front face through which the disc's rim "
            f"sticks out about {(c.disc_r - c.front_out) * 100:.0f} cm at "
            f"{c.disc_z * 100:.0f} cm height — an exposed thumbwheel lip: push or pinch-drag "
            f"it sideways to spin the disc either way (the slit is far too shallow for a "
            f"cube to pass).\n"
            f"The disc carries two open-top pockets on opposite sides, each holding one "
            f"{c.block_s * 100:.1f} cm cube: one RED, one BLUE. Their starting bearing varies "
            f"by episode. A pocket's cube can only leave the vault while that pocket sits "
            f"directly under the OPEN window (elsewhere the roof caps it, and a closed "
            f"shutter caps the window). A flat GREEN PAD ({2 * c.pad_half * 100:.0f} cm "
            f"square, position varies by episode) lies on the ground to the vault's "
            f"front-right.\n"
            f"Goal: the RED cube resting on the green pad, everything at rest. Slide the "
            f"shutter open and spin the disc by its exposed rim (either order) until the "
            f"red cube's pocket is centred under the window, then lift the red cube out "
            f"through the window and set it on the pad. The BLUE cube is a decoy — placing "
            f"it scores nothing; you may leave it anywhere."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the vault's blue roof shutter open by its orange knob, then spin the "
            "turntable by the rim lip sticking out of the front slit until the red cube's "
            "pocket sits under the roof window. Lift the red cube out through the window "
            "and set it on the green pad on the ground. The blue cube is a decoy — only "
            "the red cube on the pad counts."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the vault frame (housing fixed at the origin)."""
        return body.data.root_pos_w - self.env_origins

    def shutter_open_amt(self) -> torch.Tensor:
        """(N,) shutter travel along +y (its prismatic coordinate; 0 = closed)."""
        return self._local(self.shutter)[:, 1]

    def disc_yaw(self) -> torch.Tensor:
        """(N,) disc yaw about z (wrapped to [-pi, pi]; 0 = red pocket at the window)."""
        q = self.disc.data.root_quat_w
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                           1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def red_pocket_xy(self) -> torch.Tensor:
        """(N,2) the RED pocket centre in the vault frame."""
        c = self.cfg
        off = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        off[:, 0] = c.r_p
        w = self.disc.data.root_pos_w + self._quat_apply(self.disc.data.root_quat_w, off)
        return (w - self.env_origins)[:, :2]

    def opened(self) -> torch.Tensor:
        """(N,) bool: the shutter stands slid past `open_thresh` (window clear)."""
        return self.shutter_open_amt() >= self.cfg.open_thresh

    def indexed(self) -> torch.Tensor:
        """(N,) bool: the RED pocket centre within `index_tol` of the window centre."""
        c = self.cfg
        ctr = torch.tensor([c.r_p, 0.0], device=self.env.device)
        return (self.red_pocket_xy() - ctr).norm(dim=-1) < c.index_tol

    def prize_out(self) -> torch.Tensor:
        """(N,) bool: the red cube is OUT of the vault — above the roof plane, or
        clear of the footprint horizontally (e.g. resting on the ground/pad)."""
        c = self.cfg
        p = self._local(self.prize)
        return (p[:, 2] > c.out_z) | (p[:, 0].abs() > c.out_xy) | (p[:, 1].abs() > c.out_xy)

    def on_pad(self) -> torch.Tensor:
        """(N,) bool: red cube resting ON the pad (xy within tol, rest height)."""
        c = self.cfg
        p = self._local(self.prize)
        pd = self._local(self.pad)
        xy_ok = (p[:, :2] - pd[:, :2]).norm(dim=-1) < c.pad_xy_tol
        z_ok = (p[:, 2] - (c.pad_h + c.block_s / 2)).abs() < c.pad_z_tol
        return xy_ok & z_ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: cubes and shutter slow, disc rotation slow."""
        c = self.cfg
        return (self.prize.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.shutter.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.disc.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.disc.data.root_pos_w, self.shutter.data.root_pos_w,
                         self.prize.data.root_pos_w, self.decoy.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        fin = self._finite()
        self._open_l |= self.opened() & fin
        self._index_l |= self.indexed() & fin
        self._out_l |= self.prize_out() & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the RED cube rests on the green pad, everything settled and
        finite — a LIVE physical outcome. The cube is captive under the roof and
        the closed shutter, and cannot pass the front slit, so the only physical
        route here is: window uncovered, pocket indexed under it, cube lifted
        out through the aperture, set down on the pad."""
        self._update_latches()
        return self.on_pad() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 opened + 0.20 indexed + 0.25 out, all
        latched, capped at 0.60; exactly 1.0 iff success() holds live. Doing
        nothing scores ~0; the seed's slider-push analogue (shutter only) tops
        out at 0.15; placing the BLUE decoy on the pad scores 0."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._open_l.float() + c.w_index * self._index_l.float()
                + c.w_out * self._out_l.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="carousel_vault", robot="null"))
