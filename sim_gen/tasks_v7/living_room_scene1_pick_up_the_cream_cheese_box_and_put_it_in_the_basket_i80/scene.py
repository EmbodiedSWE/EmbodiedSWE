"""CounterweighBasketScene — load the counterweight tray, then put the cheese box in the
basket so the beam balance settles LEVEL.

Derived from libero_90/living_room_scene1 "pick up the cream cheese box and put it in
the basket" (grasp the box among distractors, carry it over a passive open basket,
release; a bounding-box containment check ends the episode), but the receptacle is no
longer passive: the basket is one end of a FREE-SWINGING BEAM BALANCE (revolute joint
between two portal columns, hard stops at +/-20 deg, a keel CoM well below the pivot so
the empty beam self-levels). Success requires the box INSIDE the basket AND the beam
settled LEVEL (within `level_band_deg` of horizontal) — and a 0.20 kg box on a 0.22 m
arm tips the beam to ~16-20 deg, far outside the band, unless a matching counterweight
sits on the rimmed TRAY at the opposite end. Two same-size blocks lie in the scene: a
dark METAL block (0.20 kg — matches the box) and a white FOAM decoy (0.03 kg — leaves
the beam ~15 deg tipped). So the seed's ENTIRE plan — put the box in the basket —
produces a rejected end state here (smoke #6), and a solver needs a different plan:
reason about torque equilibrium, select the heavy block by material, load the tray,
and load the basket; the rubric judges the MECHANISM'S settled angle, not just
containment.

Assets are fully procedural (compound spawners, i33/i42 pattern):
  - stand: HEAVY DYNAMIC portal frame (25 kg — never kinematic: a kinematic body0's
    joint anchor stays world-fixed when the assembly is re-posed at reset). Foot slab,
    two columns, crossbar; pivot axis (local y) at z = `pivot_z`.
  - beam: one dynamic compound — bar, BASKET (floor + 4 walls, inner 130 x 100 mm,
    walls 55 mm) on the +x end, rimmed TRAY (inner 90 x 90 mm, rims 26 mm) on the -x
    end, a visual keel + axle. MassAPI mass with an EXPLICIT CoM 0.155 m below the
    pivot (with only a mass, PhysX leaves the CoM at the body origin and the pendulum
    never restores). Revolute joint to the stand, limits +/-20 deg; strong angular
    damping (plain-USD jointFriction is inert) so the balance rings down in seconds.
  - box: the blue cream-cheese carton, 96 x 64 x 36 mm, 0.20 kg.
  - metal / foam: two 45 mm cubes, 0.20 kg dark gray vs 0.03 kg white.

Per-episode randomization (readback-verifiable): stand xy + yaw, Bernoulli LEFT/RIGHT
slot swap of the two blocks + per-object xy jitter + free yaw.

Rubric (0..1; latched partial credit; still-ness via a consecutive-step counter in
post_step — instantaneous velocity gates pass spuriously at every swing turning point):
  0.25 * cw_latch   — metal block ever resting on the tray (latched)
  0.35 * box_latch  — box ever inside the basket volume (latched)
  1.0 iff success() — box inside the basket AND |beam pitch| <= level_band_deg AND
                      everything still for >= still_steps consecutive steps.
  Non-success cap 0.60; null policy scores 0.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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
_MAT_PATH = "/World/Materials/counterweigh_grip"


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             quat=None) -> None:
    """Author one box (optionally a collider). `quat` (w,x,y,z) is a local orientation
    applied between translate and scale (T * R * S)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
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


def _rigid_dynamic(root, mass: float, *, lin_damp: float, ang_damp: float,
                   iters: int = 16, com: tuple | None = None) -> None:
    """Dynamic rigid-body armor on a compound root. `com` authors an EXPLICIT local
    centre of mass — required for the beam: with only a mass on the root, PhysX keeps
    the CoM at the body origin (the pivot) and the self-levelling pendulum never
    restores."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(int(iters))
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _bind_grip(stage_path: str, static: float, dynamic: float) -> None:
    """Author (once) and bind a friction material — custom spawner colliders otherwise
    get the ~0.5 default and no restitution control."""
    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sim.utils import bind_physics_material

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(_MAT_PATH).IsValid():
        sim_utils.spawn_rigid_body_material(
            _MAT_PATH,
            sim_utils.RigidBodyMaterialCfg(static_friction=static, dynamic_friction=dynamic,
                                           restitution=0.0))
    bind_physics_material(stage_path, _MAT_PATH)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Portal frame: foot slab + two columns (straddling the beam in local y) +
    crossbar. HEAVY DYNAMIC (25 kg): a kinematic body0 leaves the joint anchor
    world-fixed at the spawn pose after reset teleports."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass_props.mass, lin_damp=0.5, ang_damp=0.5)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, 0.0, c.slab_t / 2), size=(c.slab_x, c.slab_y, c.slab_t),
             color=c.color, collide=collide)
    col_h = c.col_top - c.slab_t
    for sgn, nm in ((1.0, "col_l"), (-1.0, "col_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(0.0, sgn * c.col_y, c.slab_t + col_h / 2),
                 size=(0.048, 0.040, col_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/crossbar",
             center=(0.0, 0.0, c.col_top + 0.012),
             size=(0.048, 2 * c.col_y + 0.040, 0.024), color=c.color, collide=collide)
    _bind_grip(prim_path, c.mu_s, c.mu_d)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The balance beam: bar + basket (+x end) + rimmed tray (-x end) + keel + axle,
    one dynamic body with an EXPLICIT CoM `com_z` below the pivot (origin = pivot).
    Revolute joint (axis local y) to the sibling stand, hard limits +/-stop_deg."""
    from pxr import Gf, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass_props.mass, lin_damp=0.2, ang_damp=cfg.ang_damp,
                   iters=32, com=(0.0, 0.0, cfg.com_z))
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    fl_z = c.floor_z  # basket/tray floor-plate CENTRE z (top = fl_z + 0.004)
    # bar
    _add_box(stage, f"{prim_path}/bar",
             center=(0.0, 0.0, c.bar_z), size=(2 * c.bar_half, 0.05, 0.016),
             color=c.bar_color, collide=collide)
    # keel (visual mass story for the explicit CoM) + axle stubs into the columns
    _add_box(stage, f"{prim_path}/keel",
             center=(0.0, 0.0, -0.125), size=(0.024, 0.020, 0.130),
             color=c.bar_color, collide=collide)
    _add_box(stage, f"{prim_path}/axle",
             center=(0.0, 0.0, 0.0), size=(0.020, 0.200, 0.020),
             color=(0.35, 0.35, 0.38), collide=collide)
    # basket at +x: floor + 4 walls, inner bk_in_x x bk_in_y, walls wall_h tall
    bx, by = c.bk_in_x + 2 * c.t, c.bk_in_y + 2 * c.t
    _add_box(stage, f"{prim_path}/bk_floor",
             center=(c.arm, 0.0, fl_z), size=(bx, by, 0.008),
             color=c.basket_color, collide=collide)
    wz = fl_z + 0.004 + c.wall_h / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/bk_wx{'p' if sgn > 0 else 'n'}",
                 center=(c.arm + sgn * (bx / 2 - c.t / 2), 0.0, wz),
                 size=(c.t, by, c.wall_h), color=c.basket_color, collide=collide)
        _add_box(stage, f"{prim_path}/bk_wy{'p' if sgn > 0 else 'n'}",
                 center=(c.arm, sgn * (by / 2 - c.t / 2), wz),
                 size=(bx - 2 * c.t, c.t, c.wall_h), color=c.basket_color, collide=collide)
    # tray at -x: floor + 4 rims, inner tr_in x tr_in, rims rim_h tall
    tx = c.tr_in + 2 * c.t
    _add_box(stage, f"{prim_path}/tr_floor",
             center=(-c.arm, 0.0, fl_z), size=(tx, tx, 0.008),
             color=c.tray_color, collide=collide)
    rz = fl_z + 0.004 + c.rim_h / 2
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/tr_rx{'p' if sgn > 0 else 'n'}",
                 center=(-c.arm + sgn * (tx / 2 - c.t / 2), 0.0, rz),
                 size=(c.t, tx, c.rim_h), color=c.tray_color, collide=collide)
        _add_box(stage, f"{prim_path}/tr_ry{'p' if sgn > 0 else 'n'}",
                 center=(-c.arm, sgn * (tx / 2 - c.t / 2), rz),
                 size=(tx - 2 * c.t, c.t, c.rim_h), color=c.tray_color, collide=collide)
    _bind_grip(prim_path, c.mu_s, c.mu_d)

    # revolute hinge to the sibling stand (axis local y at the stand's pivot point).
    # Collision between beam and stand stays FILTERED (USD default for a joint pair) —
    # the axle may pass through the columns; the +/-stop_deg limits are the stops.
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/pivot")
    j.CreateBody0Rel().SetTargets([f"{base}/BalanceStand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(-c.stop_deg))
    j.CreateUpperLimitAttr(float(c.stop_deg))
    return root


def _spawn_brick(prim_path: str, cfg: Any, translation=None, orientation=None):
    """A simple dynamic box (the cheese carton or a block). Origin at its centre."""
    stage, root = _root_xform(prim_path, translation, orientation)
    _rigid_dynamic(root, cfg.mass_props.mass, lin_damp=0.05, ang_damp=0.10)
    collide = _make_collide(cfg.contact_offset)
    _add_box(stage, f"{prim_path}/body", center=(0.0, 0.0, 0.0), size=cfg.dims,
             color=cfg.color, collide=collide)
    _bind_grip(prim_path, cfg.mu_s, cfg.mu_d)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "stand" not in _SPAWNER_CACHE:

        @configclass
        class StandSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_stand)
            slab_x: float = 0.16
            slab_y: float = 0.30
            slab_t: float = 0.024
            col_y: float = 0.09
            col_top: float = 0.304
            color: tuple = (0.45, 0.38, 0.30)
            contact_offset: float = 0.002
            mu_s: float = 0.7
            mu_d: float = 0.6

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            arm: float = 0.22
            bar_half: float = 0.30
            bar_z: float = -0.060
            floor_z: float = -0.048
            bk_in_x: float = 0.130
            bk_in_y: float = 0.100
            wall_h: float = 0.055
            tr_in: float = 0.090
            rim_h: float = 0.026
            t: float = 0.008
            com_z: float = -0.155
            pivot_z: float = 0.24
            stop_deg: float = 20.0
            ang_damp: float = 3.0
            bar_color: tuple = (0.55, 0.50, 0.42)
            basket_color: tuple = (0.72, 0.55, 0.30)
            tray_color: tuple = (0.62, 0.20, 0.16)
            contact_offset: float = 0.002
            mu_s: float = 0.7
            mu_d: float = 0.6

        @configclass
        class BrickSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_brick)
            dims: tuple = (0.045, 0.045, 0.045)
            color: tuple = (0.5, 0.5, 0.5)
            contact_offset: float = 0.002
            mu_s: float = 0.7
            mu_d: float = 0.6

        _SPAWNER_CACHE.update(stand=StandSpawnerCfg, beam=BeamSpawnerCfg,
                              brick=BrickSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CounterweighBasketSceneCfg(BaseCfg):
    """Config for `CounterweighBasketScene`. Torque design (arm 0.22 m, beam CoM
    0.155 m below the pivot, beam 1.0 kg): box alone in the basket -> tan(theta) =
    0.2*0.22/0.155 ~= 0.28 -> ~16-20 deg (rejected); foam-only counterweight leaves
    ~15 deg (rejected); metal + box with worst-case in-receptacle scatter -> ~3 deg
    (inside the 8 deg band)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    level_band_deg: float = tunable(8.0)  # |beam pitch| for "level"
    still_omega: float = tunable(0.15)  # beam |ang vel| gate (rad/s)
    still_vel: float = tunable(0.05)  # box/blocks |lin vel| gate (m/s)
    still_steps: int = tunable(60)  # consecutive still steps (0.5 s at 120 Hz)
    latch_vel: float = tunable(0.10)  # max speed for latching a stage

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    stand_dx: float = tunable(0.03)  # stand x jitter (+/- m)
    stand_dy: float = tunable(0.06)  # stand y jitter (+/- m)
    stand_yaw_deg: float = tunable(12.0)  # stand yaw jitter (+/- deg) about the base yaw
    slot_jitter: float = tunable(0.025)  # per-block spawn xy jitter (+/- m)
    box_jitter: float = tunable(0.030)  # box spawn xy jitter (+/- m)
    obj_yaw_deg: float = tunable(45.0)  # free yaw for blocks and box (+/- deg)
    swap_slots: bool = tunable(True)  # Bernoulli metal/foam slot swap

    # --- info: layout (single Franka base at the origin; radii 0.20-0.55 m) ---------------------
    stand_x: float = info(0.42)  # stand centre distance from the base
    stand_yaw0_deg: float = info(90.0)  # base yaw: beam runs across the workspace (world y)
    slot_a: tuple = info((0.22, 0.21))  # block spawn slot A (left)
    slot_b: tuple = info((0.22, -0.21))  # block spawn slot B (right)
    box_slot: tuple = info((0.20, 0.0))  # box spawn slot

    # --- info: balance structure (mirrors the spawner defaults) ---------------------------------
    pivot_z: float = info(0.24)  # pivot height above the ground
    arm: float = info(0.22)  # lever arm to basket/tray centres
    bar_half: float = info(0.30)
    floor_z: float = info(-0.048)  # basket/tray floor-plate centre, beam frame
    bk_in_x: float = info(0.130)  # basket inner span along the bar
    bk_in_y: float = info(0.100)
    wall_h: float = info(0.055)
    tr_in: float = info(0.090)  # tray inner span
    rim_h: float = info(0.026)
    com_z: float = info(-0.155)  # beam CoM below the pivot (the keel)
    beam_mass: float = info(1.0)
    stand_mass: float = info(25.0)
    stop_deg: float = info(20.0)  # hard joint limits

    # --- info: objects --------------------------------------------------------------------------
    box_dims: tuple = info((0.096, 0.064, 0.036))
    box_mass: float = info(0.20)
    block_size: float = info(0.045)
    metal_mass: float = info(0.20)
    foam_mass: float = info(0.03)
    box_color: tuple = info((0.16, 0.35, 0.80))  # blue carton
    metal_color: tuple = info((0.22, 0.23, 0.27))  # dark gray metal
    foam_color: tuple = info((0.95, 0.95, 0.92))  # white foam

    # rubric weights (0.25 + 0.35 = 0.60 = the non-success cap)
    w_cw: float = info(0.25)
    w_box: float = info(0.35)
    contact_offset: float = info(0.002)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("counterweigh_basket")
class CounterweighBasketScene(BaseScene):
    cfg: CounterweighBasketSceneCfg

    def __init__(self, cfg: CounterweighBasketSceneCfg | None = None) -> None:
        super().__init__(cfg or CounterweighBasketSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        stand_spawn = spawners["stand"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.stand_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            contact_offset=c.contact_offset)
        beam_spawn = spawners["beam"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.beam_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            arm=c.arm, bar_half=c.bar_half, floor_z=c.floor_z, bk_in_x=c.bk_in_x,
            bk_in_y=c.bk_in_y, wall_h=c.wall_h, tr_in=c.tr_in, rim_h=c.rim_h,
            com_z=c.com_z, pivot_z=c.pivot_z, stop_deg=c.stop_deg,
            contact_offset=c.contact_offset)

        def brick(mass, dims, color):
            return spawners["brick"](
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                dims=dims, color=color, contact_offset=c.contact_offset)

        bs = (c.block_size,) * 3
        # NOTE: stand must precede beam (the beam's joint targets the sibling stand).
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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BalanceStand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_x, 0.0, 0.0),
                    rot=(math.cos(math.radians(c.stand_yaw0_deg) / 2), 0.0, 0.0,
                         math.sin(math.radians(c.stand_yaw0_deg) / 2))),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BalanceBeam",
                spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_x, 0.0, c.pivot_z),
                    rot=(math.cos(math.radians(c.stand_yaw0_deg) / 2), 0.0, 0.0,
                         math.sin(math.radians(c.stand_yaw0_deg) / 2))),
            ),
            "box": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/CheeseBox",
                spawn=brick(c.box_mass, c.box_dims, c.box_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.box_slot[0], c.box_slot[1], c.box_dims[2] / 2 + 0.002)),
            ),
            "metal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MetalBlock",
                spawn=brick(c.metal_mass, bs, c.metal_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.block_size / 2 + 0.002)),
            ),
            "foam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/FoamBlock",
                spawn=brick(c.foam_mass, bs, c.foam_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], c.block_size / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.stand: RigidObject = env.iscene["stand"]
        self.beam: RigidObject = env.iscene["beam"]
        self.box: RigidObject = env.iscene["box"]
        self.metal: RigidObject = env.iscene["metal"]
        self.foam: RigidObject = env.iscene["foam"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient regressions (rubric requirement)
        self._cw_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._box_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        # consecutive-still counter: instantaneous velocity gates pass spuriously at
        # every turning point of the damped swing — only >= still_steps in a row counts
        self._still_n = torch.zeros(n, dtype=torch.long, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: stand re-posed (xy + yaw), beam re-posed LEVEL onto the
        stand's pivot with the same yaw (both dynamic, so the joint anchor follows),
        blocks Bernoulli-assigned to the two slots (+ jitter + yaw), box at its slot,
        latches and the still counter cleared."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- stand: xy + yaw about the base yaw ---
        dx = (torch.rand(m, device=dev) * 2 - 1) * c.stand_dx
        dy = (torch.rand(m, device=dev) * 2 - 1) * c.stand_dy
        yaw = (math.radians(c.stand_yaw0_deg)
               + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.stand_x + dx
        st[:, 1] = dy
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.stand.write_root_state_to_sim(st, env_ids)
        s_pos, s_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- beam: LEVEL at the stand's pivot, same yaw ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = s_pos + quat_apply(
            s_quat, torch.tensor([0.0, 0.0, c.pivot_z], device=dev).expand(m, 3))
        st[:, 3:7] = s_quat
        self.beam.write_root_state_to_sim(st, env_ids)

        # --- blocks: Bernoulli slot swap + jitter + yaw; box at its slot ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        metal_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        foam_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        box_xy = torch.tensor(c.box_slot, device=dev).expand(m, 2) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.box_jitter
        for body, xy, z in ((self.metal, metal_xy, c.block_size / 2 + 0.002),
                            (self.foam, foam_xy, c.block_size / 2 + 0.002),
                            (self.box, box_xy, c.box_dims[2] / 2 + 0.002)):
            psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.obj_yaw_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = torch.cos(psi / 2)
            st[:, 6] = torch.sin(psi / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches + still counter ---
        self._cw_latch[env_ids] = False
        self._box_latch[env_ids] = False
        self._still_n[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "box": self.box.data.root_state_w[env_ids].clone(),
            "metal": self.metal.data.root_state_w[env_ids].clone(),
            "foam": self.foam.data.root_state_w[env_ids].clone(),
            "cw_latch": self._cw_latch[env_ids].clone(),
            "box_latch": self._box_latch[env_ids].clone(),
            "still_n": self._still_n[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        self.box.write_root_state_to_sim(state["box"], env_ids)
        self.metal.write_root_state_to_sim(state["metal"], env_ids)
        self.foam.write_root_state_to_sim(state["foam"], env_ids)
        self._cw_latch[env_ids] = state["cw_latch"]
        self._box_latch[env_ids] = state["box_latch"]
        self._still_n[env_ids] = state["still_n"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A BEAM BALANCE stands on the floor: a wooden portal frame whose two "
            f"columns carry a {2 * c.bar_half * 100:.0f} cm beam on a free-swinging "
            f"axle {c.pivot_z * 100:.0f} cm up (a keel under the middle self-levels "
            f"the EMPTY beam; hard stops at +/-{c.stop_deg:.0f} deg). One end of the "
            f"beam carries an open-topped tan BASKET (inner "
            f"{c.bk_in_x * 100:.0f} x {c.bk_in_y * 100:.0f} cm, walls "
            f"{c.wall_h * 100:.1f} cm); the other end carries a shallow RED-rimmed "
            f"TRAY (inner {c.tr_in * 100:.0f} x {c.tr_in * 100:.0f} cm, rim "
            f"{c.rim_h * 100:.1f} cm). On the floor in front of the balance lie a "
            f"BLUE cream-cheese box ({c.box_dims[0] * 100:.1f} x "
            f"{c.box_dims[1] * 100:.1f} x {c.box_dims[2] * 100:.1f} cm, "
            f"{c.box_mass * 1000:.0f} g) and two {c.block_size * 100:.1f} cm cube "
            f"blocks whose left/right positions swap between episodes — identify them "
            f"by look: a DARK GRAY METAL block ({c.metal_mass * 1000:.0f} g — it "
            f"matches the box's weight) and a WHITE FOAM block "
            f"({c.foam_mass * 1000:.0f} g — far too light).\n"
            f"Goal: the blue box must END UP INSIDE THE BASKET with the beam settled "
            f"LEVEL — within {c.level_band_deg:.0f} deg of horizontal, at rest. The "
            f"box's weight tips an uncounterweighted beam to ~{c.stop_deg:.0f} deg, "
            f"so you must counterweight the TRAY on the opposite end — the metal "
            f"block on the tray balances the box in the basket; the foam block is "
            f"too light to do it. Either order works (load the tray first or the "
            f"basket first); what is judged is the settled end state: box inside the "
            f"basket, beam level, everything at rest. A tipped beam, the box "
            f"anywhere but inside the basket, or the counterweight anywhere but on "
            f"the tray (e.g. dumped into the basket with the box) all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the blue cheese box in the balance's basket and set the dark metal "
            "block on the red tray at the other end so the beam settles level. The "
            "beam must end within 8 degrees of horizontal with the box inside the "
            "basket."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _beam_local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in the beam frame (+x toward the basket)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.beam.data.root_pos_w
        return quat_apply_inverse(self.beam.data.root_quat_w, rel)

    def beam_pitch_deg(self) -> torch.Tensor:
        """(N,) signed beam pitch in degrees: angle of the beam's local +x axis above
        the horizontal (positive = basket end UP). Yaw-independent."""
        from isaaclab.utils.math import quat_apply

        ex = quat_apply(self.beam.data.root_quat_w,
                        torch.tensor([1.0, 0.0, 0.0],
                                     device=self.env.device).expand(self.env.num_envs, 3))
        return torch.rad2deg(torch.asin(ex[:, 2].clamp(-1.0, 1.0)))

    def _in_basket(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM inside the basket volume (beam frame): over the basket
        floor, inside the walls, below the wall-top plane."""
        c = self.cfg
        loc = self._beam_local(body)
        return (((loc[:, 0] - c.arm).abs() <= 0.060)
                & (loc[:, 1].abs() <= 0.055)
                & (loc[:, 2] > c.floor_z - 0.003) & (loc[:, 2] <= 0.006))

    def _on_tray(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM over the tray floor, inside the rims, at resting height
        (a block perched on a rim top reads ~+0.005 and is rejected)."""
        c = self.cfg
        loc = self._beam_local(body)
        return (((loc[:, 0] + c.arm).abs() <= 0.045)
                & (loc[:, 1].abs() <= 0.045)
                & (loc[:, 2] > c.floor_z + 0.001) & (loc[:, 2] <= -0.008))

    def level(self) -> torch.Tensor:
        """(N,) bool: |beam pitch| within the level band."""
        return self.beam_pitch_deg().abs() <= self.cfg.level_band_deg

    def still(self) -> torch.Tensor:
        """(N,) bool: everything still for >= still_steps CONSECUTIVE steps."""
        return self._still_n >= self.cfg.still_steps

    # ----- step-coupled bookkeeping (every step) ---------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the consecutive-still counter and the stage latches. The counter
        must advance exactly once per step, so it lives here and NOWHERE else."""
        c = self.cfg
        inst = ((self.beam.data.root_ang_vel_w.norm(dim=-1) < c.still_omega)
                & (self.box.data.root_lin_vel_w.norm(dim=-1) < c.still_vel)
                & (self.metal.data.root_lin_vel_w.norm(dim=-1) < c.still_vel)
                & (self.foam.data.root_lin_vel_w.norm(dim=-1) < c.still_vel))
        self._still_n = torch.where(inst, self._still_n + 1,
                                    torch.zeros_like(self._still_n))
        self._cw_latch |= (self._on_tray(self.metal)
                           & (self.metal.data.root_lin_vel_w.norm(dim=-1) < c.latch_vel))
        self._box_latch |= (self._in_basket(self.box)
                            & (self.box.data.root_lin_vel_w.norm(dim=-1) < c.latch_vel))

    def success(self) -> torch.Tensor:
        """(N,) bool: the box INSIDE the basket, the beam LEVEL, everything still —
        settled physical outcome only (level physically requires the counterweight)."""
        return self._in_basket(self.box) & self.level() & self.still()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25*metal-on-tray + 0.35*box-in-basket (latched,
        ~0 for doing nothing, capped 0.60) — and exactly 1.0 iff success() holds."""
        c = self.cfg
        base = (c.w_cw * self._cw_latch.float()
                + c.w_box * self._box_latch.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through pose writes + physics.
register_env("simgen", lambda: EnvCfg(scene="counterweigh_basket", robot="null"))
