"""TipFeederScene — load the white ball into an open-top TIPPING HOPPER hinged on a
stand, then actuate the hinge (pull the handle bar up and forward) so the hopper tips
and the ball rolls over the retaining lip, across the sill, through the ELEVATED
WINDOW into the sealed box; release the handle and let the hopper fall back to its
seat. Derived from rlbench/hockey, but no strike, no stick, no floor-level goal.

Seed (rlbench/hockey): grasp a hockey stick and STRIKE the ball across open floor into
an open-mouthed goal — one ballistic tool swing at a passively available receptacle.
Here the receptacle admits nothing at floor level and there is no tool:

- The goal becomes a fully SEALED BOX whose only ball-sized opening is an ELEVATED
  WINDOW (sill 13 cm up, 8.5 cm tall, 13 cm wide) in the near wall. A ball shot,
  rolled or pushed along the floor just bounces off the closed lower wall (smoke's
  seed-strategy probe fires the shot and watches it bounce). Nothing can be dropped
  in either: the roof seals the top and a HOOD (top plate + side plates) overhangs
  the window tunnel, so a ball released above the opening rests on the hood or roof.
- The only path through the window is MECHANICAL: a TIPPING HOPPER (an open-top
  basin with a low retaining lip) is hinged between two cheek plates on a heavy
  stand, its lip facing the window at sill height. Gravity holds the hopper SEATED
  (tilted 12 deg back). Pulling its handle bar up and forward rotates it on the
  hinge; past a modest tilt the loaded ball runs down the basin, hops the lip,
  crosses the sill and drops through the window into the box.
- Success is judged on the SETTLED MACHINE: white ball at rest on the box floor
  (below the window sill — inside the chamber, not perched in the window tunnel)
  AND the hopper back on its seat at rest (released, gravity-returned).

So a solver needs a different PLAN (stage the payload in a mechanism, actuate a
revolute joint through a handle, let gravity deliver the ball, then release the
mechanism) and different CODE STRUCTURE (hinge-angle control with load/release
phases instead of grasp-stick + swing). No strike anywhere: the ball is placed,
never hit.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide). The hinge is a spawn-authored USD revolute joint (stand = body0):
  - box: KINEMATIC compound — 4 walls, roof, window aperture in the front wall,
    sill shelf and hood (top + sides) framing the window tunnel. Origin at the
    interior floor centre; local -x faces the robot.
  - stand: heavy DYNAMIC compound (never kinematic: a kinematic body0 would leave
    the hinge anchor world-fixed after the randomization teleport) — base slab +
    two cheek plates. Origin on the ground under the hinge axis.
  - hopper: DYNAMIC compound — basin floor, low front lip, side walls, tall back
    wall, strut and handle BAR (yellow cylinder). Origin ON the hinge axis; mass,
    CoM and diagonal inertia authored explicitly (compound roots otherwise keep
    the CoM at the body origin and the gravity-return would be fake).
  - ball: DYNAMIC white sphere, 54 mm.

Per-episode randomization (readback-verifiable): the whole apparatus (box + stand +
hopper, one coherent linkage write) gets a planar offset + yaw about the hinge
point; the ball spawns on a Bernoulli LEFT/RIGHT side of the floor with xy jitter.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.20 * loaded    — ball ever in the hopper basin (hopper-frame readback)
  0.15 * tilted    — hinge ever rotated past +20 deg from level (real actuation)
  0.35 * delivered — ball ever inside the box below the sill (box-frame readback)
  1.0 iff success() — ball at rest on the box floor inside AND hopper seated
                      (< -6 deg) at rest. Non-success cap 0.70.

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


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    """Author one axis-aligned box collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _root_xform(prim_path: str, translation, orientation):
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


def _dynamic_body(root, *, mass: float, lin_damp: float, ang_damp: float,
                  com=None, inertia=None) -> None:
    """Author a dynamic rigid body: explicit mass (root-level mass_props on custom
    spawner cfgs is silently ignored), optional explicit CoM + diagonal inertia
    (compound roots keep the CoM at the ORIGIN unless authored), damping, zeroed
    sleep thresholds, solver velocity iterations 4 (GPU sphere-creep fix)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    massapi = UsdPhysics.MassAPI.Apply(root)
    massapi.CreateMassAttr(float(mass))
    if com is not None:
        massapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        massapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(lin_damp))
    pxrb.CreateAngularDampingAttr(float(ang_damp))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverVelocityIterationCountAttr(4)


def _spawn_box(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sealed box: KINEMATIC compound. Origin at the interior floor
    centre; interior |x| <= in_hx, |y| <= in_hy, roof underside at roof_z. The
    front wall (-x, facing the robot) carries the elevated window (sill at sill_z,
    top at win_top, half-width win_hy) with a sill shelf and an overhanging hood."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    hx, hy, t = c.in_hx, c.in_hy, c.wall_t
    top = c.roof_z + t          # walls reach the roof's top face
    fx0 = -hx - t               # front wall outer face
    ax0 = fx0 - c.apron_d       # sill/hood front face
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}", center=(0.0, sgn * (hy + t / 2), top / 2),
                 size=(2 * (hx + t), t, top), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back", center=(hx + t / 2, 0.0, top / 2),
             size=(t, 2 * hy, top), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(0.0, 0.0, c.roof_z + t / 2),
             size=(2 * (hx + t), 2 * hy, t), color=c.roof_color, collide=collide)
    # front wall: solid below the sill and above the window; side strips beside it
    _add_box(stage, f"{prim_path}/front_low", center=(fx0 + t / 2, 0.0, c.sill_z / 2),
             size=(t, 2 * (hy + t), c.sill_z), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/front_high",
             center=(fx0 + t / 2, 0.0, (c.win_top + top) / 2),
             size=(t, 2 * (hy + t), top - c.win_top), color=c.color, collide=collide)
    strip_w = hy + t - c.win_hy
    for sgn, nm in ((1.0, "front_l"), (-1.0, "front_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(fx0 + t / 2, sgn * (c.win_hy + strip_w / 2),
                         (c.sill_z + c.win_top) / 2),
                 size=(t, strip_w, c.win_top - c.sill_z), color=c.color,
                 collide=collide)
    # window tunnel furniture: sill shelf (floor), hood top (ceiling), hood sides
    tun_x = -hx - ax0  # tunnel depth (apron front to the wall's inner face)
    _add_box(stage, f"{prim_path}/sill", center=((ax0 - hx) / 2, 0.0, c.sill_z - t / 2),
             size=(tun_x, 2 * (c.win_hy + t), t), color=c.trim_color, collide=collide)
    _add_box(stage, f"{prim_path}/hood_top",
             center=((ax0 - hx) / 2, 0.0, c.win_top + t / 2),
             size=(tun_x, 2 * (c.win_hy + t), t), color=c.trim_color, collide=collide)
    for sgn, nm in ((1.0, "hood_l"), (-1.0, "hood_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((ax0 - hx) / 2, sgn * (c.win_hy + t / 2),
                         (c.sill_z + c.win_top + t) / 2),
                 size=(tun_x, t, c.win_top + t - c.sill_z), color=c.trim_color,
                 collide=collide)
    return root


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hinge stand: heavy DYNAMIC compound (body0 of the hinge — a
    kinematic body0 would leave the joint anchor world-fixed after the reset
    teleport). Origin on the ground under the hinge axis."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, mass=cfg.mass, lin_damp=0.5, ang_damp=0.5,
                  com=(-0.12, 0.0, 0.03))
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/slab", center=(-0.12, 0.0, 0.01),
             size=(0.28, 0.18, 0.02), color=cfg.color, collide=collide)
    for sgn, nm in ((1.0, "cheek_l"), (-1.0, "cheek_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(-0.085, sgn * (cfg.cheek_y + 0.004), 0.1275),
                 size=(0.19, 0.008, 0.215), color=cfg.color, collide=collide)
    return root


def _spawn_hopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tipping hopper: DYNAMIC compound with an explicit CoM behind the
    hinge (gravity-return) and authored diagonal inertia (shape-derived inertia is
    unauditable under external-wrench control). Origin ON the hinge axis. Also
    authors the revolute joint to the sibling Stand (axis Y, limits seat..dump);
    the joint pair keeps the default collision FILTERING — hopper/stand clearance
    is geometric (13 mm in y, an axis invariant under the hinge motion)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, mass=cfg.mass, lin_damp=0.05, ang_damp=cfg.ang_damp,
                  com=(-0.090, 0.0, 0.010),
                  inertia=(6.0e-4, 1.4e-3, 1.6e-3))
    collide = _make_collide(cfg)
    c = cfg
    # SWEEP CLEARANCE: every forward-sweeping part stays within |y| <= 0.058 so the
    # whole mouth passes THROUGH the 130 mm window aperture (win_hy 0.065) over the
    # full hinge travel. (First forge run: 136 mm-wide side walls jammed their
    # front-top corners — swing radius 55 mm vs the 14 mm hinge-to-apron gap —
    # against the hood side plates at exactly +11.7 deg, immovable at any torque.)
    _add_box(stage, f"{prim_path}/floor", center=(-0.085, 0.0, -0.006),
             size=(0.160, 0.116, 0.008), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/lip", center=(-0.001, 0.0, 0.0),
             size=(0.008, 0.116, 0.020), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "side_l"), (-1.0, "side_r")):
        _add_box(stage, f"{prim_path}/{nm}", center=(-0.085, sgn * 0.054, 0.0225),
                 size=(0.176, 0.008, 0.065), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/back", center=(-0.169, 0.0, 0.0375),
             size=(0.008, 0.116, 0.095), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/strut", center=(-0.188, 0.0, 0.075),
             size=(0.030, 0.040, 0.020), color=c.color, collide=collide)
    bar = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bar")
    bar.CreateAxisAttr("Y")
    bar.CreateHeightAttr(2 * c.bar_hl)
    bar.CreateRadiusAttr(c.bar_r)
    bar.CreateExtentAttr([Gf.Vec3f(-c.bar_r, -c.bar_hl, -c.bar_r),
                          Gf.Vec3f(c.bar_r, c.bar_hl, c.bar_r)])
    UsdGeom.Xformable(bar.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(-0.208, 0.0, 0.081))
    bar.CreateDisplayColorAttr([Gf.Vec3f(*c.bar_color)])
    collide(bar.GetPrim())
    # --- the hinge: revolute joint to the sibling Stand, authored at spawn ---
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(c.seat_lim_deg))
    j.CreateUpperLimitAttr(float(c.dump_lim_deg))
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the ball: DYNAMIC sphere with angular damping so it rolls to rest."""
    from pxr import Gf, UsdGeom

    stage, root = _root_xform(prim_path, translation, orientation)
    _dynamic_body(root, mass=cfg.mass, lin_damp=0.05, ang_damp=0.30)
    r = float(cfg.radius)
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    sph.CreateRadiusAttr(r)
    sph.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    sph.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _make_collide(cfg)(sph.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "box" not in _SPAWNER_CACHE:

        @configclass
        class BoxSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_box)
            in_hx: float = 0.080
            in_hy: float = 0.090
            roof_z: float = 0.250
            wall_t: float = 0.012
            win_hy: float = 0.065
            sill_z: float = 0.130
            win_top: float = 0.215
            apron_d: float = 0.025
            color: tuple = (0.30, 0.38, 0.55)
            roof_color: tuple = (0.22, 0.28, 0.42)
            trim_color: tuple = (0.55, 0.58, 0.62)
            contact_offset: float = 0.002

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            mass: float = 33.0
            cheek_y: float = 0.071
            color: tuple = (0.24, 0.25, 0.28)
            contact_offset: float = 0.002

        @configclass
        class HopperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hopper)
            mass: float = 0.25
            ang_damp: float = 1.5
            hinge_z: float = 0.145
            seat_lim_deg: float = -12.0
            dump_lim_deg: float = 42.0
            bar_r: float = 0.008
            bar_hl: float = 0.05
            color: tuple = (0.85, 0.45, 0.10)
            bar_color: tuple = (0.93, 0.80, 0.12)
            contact_offset: float = 0.002

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            radius: float = 0.027
            mass: float = 0.06
            color: tuple = (0.95, 0.95, 0.95)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(box=BoxSpawnerCfg, stand=StandSpawnerCfg,
                              hopper=HopperSpawnerCfg, ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TipFeederSceneCfg(BaseCfg):
    """Config for `TipFeederScene`. The interlocks are metric: every box opening
    except the window is smaller than the 54 mm ball (wall/roof/hood gaps <= 12 mm
    panel seams), the window floor (sill) is 130 mm up — unreachable by any rolled
    or shot ball — and the hood roofs the window tunnel so nothing can be dropped
    in from above. The hopper is the only machine that can put the ball through:
    its lip parks 5 mm short of the sill at full tilt and its basin, tilted, is a
    ramp whose run-up carries the ball over the lip and across the window."""

    # --- tunable: rubric thresholds ---------------------------------------------------------
    settle_speed: float = tunable(0.04)   # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.60)   # max |ang vel| when judging (rad/s)
    tilt_credit_deg: float = tunable(20.0)  # hinge angle that earns the tilt credit
    seat_deg: float = tunable(-6.0)       # hinge angle below which the hopper is seated
    inside_z_max: float = tunable(0.10)   # "inside" = ball CoM BELOW this (sill is 0.130)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    ens_dx_max: float = tunable(0.03)     # apparatus offset along x (+/- m)
    ens_dy_max: float = tunable(0.05)     # apparatus offset along y (+/- m)
    ens_yaw_max_deg: float = tunable(8.0)  # apparatus yaw about the hinge point (+/- deg)
    ball_x_min: float = tunable(0.15)     # ball spawn x band
    ball_x_max: float = tunable(0.25)
    ball_y_center: float = tunable(0.26)  # |y| centre of the two spawn sides
    ball_y_jitter: float = tunable(0.05)  # +/- jitter about the side centre
    side_swap: bool = tunable(True)       # Bernoulli left/right side

    # --- info: layout (single Franka base at the origin) -------------------------------------
    hinge_x: float = info(0.46)           # hinge point x (apparatus datum)
    hinge_z: float = info(0.145)          # hinge axis height
    box_dx: float = info(0.131)           # box origin (interior centre) x, hinge-relative

    # --- info: box structure ------------------------------------------------------------------
    in_hx: float = info(0.080)            # interior half-depth (x)
    in_hy: float = info(0.090)            # interior half-width (y)
    roof_z: float = info(0.250)           # roof underside
    wall_t: float = info(0.012)
    win_hy: float = info(0.065)           # window half-width (130 mm opening)
    sill_z: float = info(0.130)           # window floor (sill top)
    win_top: float = info(0.215)          # window ceiling (hood underside)
    apron_d: float = info(0.025)          # sill/hood protrusion beyond the wall face

    # --- info: stand + hopper -----------------------------------------------------------------
    stand_mass: float = info(33.0)
    hopper_mass: float = info(0.25)
    hopper_ang_damp: float = info(1.5)    # spawn-authored hinges have NO joint friction
    seat_lim_deg: float = info(-12.0)     # hinge lower limit (the seat)
    dump_lim_deg: float = info(42.0)      # hinge upper limit
    bar_r: float = info(0.008)            # handle bar radius (16 mm dia, jaw-sized)
    bar_hl: float = info(0.05)            # handle bar half-length

    # --- info: ball -----------------------------------------------------------------------------
    ball_r: float = info(0.027)           # 54 mm dia
    ball_mass: float = info(0.06)

    # --- info: colors + misc ---------------------------------------------------------------------
    box_color: tuple = info((0.30, 0.38, 0.55))
    roof_color: tuple = info((0.22, 0.28, 0.42))
    trim_color: tuple = info((0.55, 0.58, 0.62))
    stand_color: tuple = info((0.24, 0.25, 0.28))
    hopper_color: tuple = info((0.85, 0.45, 0.10))
    bar_color: tuple = info((0.93, 0.80, 0.12))
    white_color: tuple = info((0.95, 0.95, 0.95))
    contact_offset: float = info(0.002)
    # rubric weights (0.20 + 0.15 + 0.35 = 0.70 = the non-success cap)
    w_load: float = info(0.20)
    w_tilt: float = info(0.15)
    w_dlv: float = info(0.35)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tip_feeder")
class TipFeederScene(BaseScene):
    cfg: TipFeederSceneCfg

    def __init__(self, cfg: TipFeederSceneCfg | None = None) -> None:
        super().__init__(cfg or TipFeederSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        box_spawn = sp["box"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            in_hx=c.in_hx, in_hy=c.in_hy, roof_z=c.roof_z, wall_t=c.wall_t,
            win_hy=c.win_hy, sill_z=c.sill_z, win_top=c.win_top, apron_d=c.apron_d,
            color=c.box_color, roof_color=c.roof_color, trim_color=c.trim_color,
            contact_offset=c.contact_offset,
        )
        stand_spawn = sp["stand"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass=c.stand_mass, color=c.stand_color, contact_offset=c.contact_offset,
        )
        hopper_spawn = sp["hopper"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass=c.hopper_mass, ang_damp=c.hopper_ang_damp, hinge_z=c.hinge_z,
            seat_lim_deg=c.seat_lim_deg, dump_lim_deg=c.dump_lim_deg,
            bar_r=c.bar_r, bar_hl=c.bar_hl, color=c.hopper_color,
            bar_color=c.bar_color, contact_offset=c.contact_offset,
        )
        ball_spawn = sp["ball"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            radius=c.ball_r, mass=c.ball_mass, color=c.white_color,
            contact_offset=c.contact_offset,
        )
        # NOTE: Stand is declared BEFORE Hopper so the hopper's spawn-authored joint
        # can target the already-existing sibling /Stand prim. The template init
        # poses are hinge-consistent (hopper at joint angle 0 on the axis).
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "box": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Box",
                spawn=box_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hinge_x + c.box_dx, 0.0, 0.0)),
            ),
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.hinge_x, 0.0, 0.0)),
            ),
            "hopper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hopper",
                spawn=hopper_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hinge_x, 0.0, c.hinge_z)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=ball_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.20, c.ball_y_center, c.ball_r + 0.002)),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "enable_external_forces_every_iteration": True,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.box: RigidObject = env.iscene["box"]
        self.stand: RigidObject = env.iscene["stand"]
        self.hopper: RigidObject = env.iscene["hopper"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._load_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tilt_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dlv_ever = torch.zeros(n, dtype=torch.bool, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: the whole apparatus (box + stand + hopper) re-posed with a
        random planar offset + yaw about the hinge point — written as ONE coherent
        linkage (the hopper goes to its seated angle in the new frame; body0 is
        dynamic so the hinge anchor follows). Ball to a Bernoulli side + jitter."""
        from isaaclab.utils.math import quat_apply, quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        dx = (torch.rand(m, device=dev) * 2 - 1) * c.ens_dx_max
        dy = (torch.rand(m, device=dev) * 2 - 1) * c.ens_dy_max
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.ens_yaw_max_deg)
        q_yaw = torch.zeros(m, 4, device=dev)
        q_yaw[:, 0], q_yaw[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        ens = torch.zeros(m, 3, device=dev)
        ens[:, 0], ens[:, 1] = c.hinge_x + dx, dy

        # box (kinematic): origin at hinge + R_yaw * (box_dx, 0, 0)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.box_dx
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = ens + quat_apply(q_yaw, loc) + origin
        st[:, 3:7] = q_yaw
        self.box.write_root_state_to_sim(st, env_ids)

        # stand (dynamic body0): origin at the hinge point on the ground
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = ens + origin
        st[:, 3:7] = q_yaw
        self.stand.write_root_state_to_sim(st, env_ids)

        # hopper: ON the hinge axis, at the SEATED angle in the new frame
        seat = math.radians(c.seat_lim_deg)
        q_seat = torch.zeros(m, 4, device=dev)
        q_seat[:, 0], q_seat[:, 2] = math.cos(seat / 2), math.sin(seat / 2)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = ens + origin
        st[:, 2] += c.hinge_z
        st[:, 3:7] = quat_mul(q_yaw, q_seat)
        self.hopper.write_root_state_to_sim(st, env_ids)

        # ball: Bernoulli side +/- with xy jitter (torch.rand, not randint — the
        # first randint after manual_seed is degenerate across seeds)
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        bx = c.ball_x_min + torch.rand(m, device=dev) * (c.ball_x_max - c.ball_x_min)
        by = side * (c.ball_y_center
                     + (torch.rand(m, device=dev) * 2 - 1) * c.ball_y_jitter)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = bx, by, c.ball_r + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.ball.write_root_state_to_sim(st, env_ids)

        self._load_ever[env_ids] = False
        self._tilt_ever[env_ids] = False
        self._dlv_ever[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "box": self.box.data.root_state_w[env_ids].clone(),
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "hopper": self.hopper.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "load_ever": self._load_ever[env_ids].clone(),
            "tilt_ever": self._tilt_ever[env_ids].clone(),
            "dlv_ever": self._dlv_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.box.write_root_state_to_sim(state["box"], env_ids)
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.hopper.write_root_state_to_sim(state["hopper"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        self._load_ever[env_ids] = state["load_ever"]
        self._tilt_ever[env_ids] = state["tilt_ever"]
        self._dlv_ever[env_ids] = state["dlv_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stands a slate-blue SEALED BOX (interior "
            f"{2 * c.in_hx * 100:.0f} x {2 * c.in_hy * 100:.0f} cm, roof at "
            f"{c.roof_z * 100:.0f} cm) whose ONLY ball-sized opening is an ELEVATED "
            f"WINDOW in the wall facing the robot: sill {c.sill_z * 100:.0f} cm up, "
            f"{(c.win_top - c.sill_z) * 100:.0f} cm tall, {2 * c.win_hy * 100:.0f} cm "
            f"wide, framed by a gray sill shelf and an overhanging gray HOOD (top and "
            f"side plates), so nothing can be dropped in from above and nothing can "
            f"enter at floor level — a ball rolled or thrown at the box just bounces "
            f"off. In front of the window, hinged between the two cheek plates of a "
            f"heavy dark stand, sits an orange TIPPING HOPPER: an open-top basin "
            f"(floor about 16 cm long) with a low front lip, tall back wall, and a "
            f"yellow HANDLE BAR ({2 * c.bar_r * 1000:.0f} mm dia, "
            f"{2 * c.bar_hl * 100:.0f} cm long, horizontal, sticking back over the "
            f"back wall). Gravity holds the hopper tilted {-c.seat_lim_deg:.0f} deg "
            f"BACK onto its seat; its lip faces the window at sill height. Pulling "
            f"the handle bar UP and FORWARD (toward the box) rotates the hopper on "
            f"its hinge (hard stop at +{c.dump_lim_deg:.0f} deg); when released it "
            f"falls back to the seat on its own. A WHITE BALL "
            f"({2 * c.ball_r * 100:.1f} cm) lies on the open floor to the left or "
            f"right of the stand (side and position change between episodes).\n"
            f"Goal: the white ball must end up INSIDE the box — at rest on the box "
            f"floor, below the window sill — with the hopper RELEASED back on its "
            f"seat, at rest. The only working plan: place the ball into the hopper "
            f"basin (from above — the top is open between the cheek plates), then "
            f"pull the handle bar up and forward so the hopper tips well past level "
            f"(about +30 deg); the ball runs down the tilted basin, hops the low lip, "
            f"crosses the sill and drops through the window into the box. Then let "
            f"the handle go so the hopper reseats. Throwing or rolling the ball at "
            f"the box, dropping it onto the roof or hood, leaving it perched on the "
            f"sill or in the hopper, or holding the hopper tilted at the end — all "
            f"failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the white ball into the orange tipping hopper, then pull the yellow "
            "handle bar up and forward to tip the hopper so the ball rolls over the "
            "lip and through the elevated window into the sealed box. Release the "
            "handle and let the hopper fall back onto its seat."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def hinge_angle(self) -> torch.Tensor:
        """(N,) hinge angle in rad: + = dump (lip down toward the box), seat = -12 deg.
        Read from the stand->hopper relative quaternion (never from ang-vel)."""
        from isaaclab.utils.math import quat_inv, quat_mul

        q = quat_mul(quat_inv(self.stand.data.root_quat_w),
                     self.hopper.data.root_quat_w)
        ang = 2.0 * torch.atan2(q[:, 2], q[:, 0])
        return torch.atan2(torch.sin(ang), torch.cos(ang))

    def _local(self, ref: RigidObject, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in `ref`'s body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - ref.data.root_pos_w
        return quat_apply_inverse(ref.data.root_quat_w, rel)

    def _in_basin(self) -> torch.Tensor:
        """(N,) bool: ball CoM inside the hopper basin (hopper-frame readback)."""
        loc = self._local(self.hopper, self.ball)
        return ((loc[:, 0] > -0.170) & (loc[:, 0] < 0.005)
                & (loc[:, 1].abs() < 0.048)
                & (loc[:, 2] > -0.005) & (loc[:, 2] < 0.055))

    def _inside(self) -> torch.Tensor:
        """(N,) bool: ball CoM inside the box BELOW the sill (box-frame readback —
        a ball perched on the sill shelf or in the window tunnel is NOT inside)."""
        c = self.cfg
        loc = self._local(self.box, self.ball)
        return ((loc[:, 0].abs() < c.in_hx - 0.002)
                & (loc[:, 1].abs() < c.in_hy - 0.002)
                & (loc[:, 2] > 0.005) & (loc[:, 2] < c.inside_z_max))

    def _update_latches(self) -> None:
        c = self.cfg
        self._load_ever |= self._in_basin()
        self._tilt_ever |= self.hinge_angle() > math.radians(c.tilt_credit_deg)
        self._dlv_ever |= self._inside()

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant here (the hinge is passive: gravity + limits + damping) — just
        latch rubric progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: ball at rest on the box floor inside (below the sill) AND the
        hopper back on its seat at rest. Physical outcomes only — a held-tilted
        hopper (or a ball still moving) is not success."""
        c = self.cfg
        self._update_latches()
        b_still = ((self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                   & (self.ball.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))
        h_seated = self.hinge_angle() < math.radians(c.seat_deg)
        h_still = self.hopper.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega
        return self._inside() & b_still & h_seated & h_still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20*loaded + 0.15*tilted + 0.35*delivered — all
        latched, ~0 for doing nothing, capped 0.70 — and exactly 1.0 iff success()
        holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_load * self._load_ever.float()
                + c.w_tilt * self._tilt_ever.float()
                + c.w_dlv * self._dlv_ever.float()).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="tip_feeder", robot="null"))
