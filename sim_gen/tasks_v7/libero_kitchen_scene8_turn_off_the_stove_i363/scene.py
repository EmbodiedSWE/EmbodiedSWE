"""ClutchValveStoveScene — turn OFF the lit gas stove through a PULL-AND-TURN
child-safe clutch knob (sim_gen task `libero_kitchen_scene8_turn_off_the_stove_i363`).

Derived from libero_90/libero_kitchen_scene8_turn_off_the_stove, but
STRATEGICALLY different. The seed "turns off" the stove by rotating a stove
knob DOWN past a joint-angle threshold (`stove_joint_pos[:, 0] < 0.1`) — one
direct fixture-joint actuation the robot simply grasps and twists. Here the
valve dial cannot be turned by twisting anything directly:

  (1) the gas valve ROTOR (the true valve state) is sealed inside the stove's
      valve HOUSING — walls plus a roof whose only opening is a small square
      hole that passes nothing but the knob's shaft. Fingers, tools, and pots
      physically cannot reach the rotor; its angle is readable only through
      the burner FLAME (lit while the valve is open).
  (2) the knob on top is a CHILD-SAFE CLUTCH CAP riding a 2-DOF guide
      (vertical travel 16 mm + free spin). At rest its drive posts sit BELOW
      the rotor's hanging vanes with a 6 mm air gap: spinning the un-lifted
      knob is a FREE-SPIN DECOY — it transmits nothing (the seed strategy,
      "just twist the knob", is constructed and rejected in smoke).
  (3) only while the knob is PULLED UP >= 12 mm do its posts rise into the
      vane band and form a dog clutch; sustained lift-AND-turn is required —
      drop the knob and the clutch opens instantly (~43 deg of backlash also
      means the clutch must stay loaded through the turn).
  (4) success additionally requires RELEASING the knob at the end (lift back
      at the bottom stop): a hand holding the knob up at the OFF stop is not
      a finished, settled stove. Execution order is therefore physically
      inherent: engage (lift), then turn to the stop, then release.

Assets are fully procedural (native PhysX box colliders; both moving bodies
ride bind-time per-env USD joints anchored on the KINEMATIC stove, and both
joint pairs disable ONLY their own stove<->body collision — the knob<->rotor
CLUTCH contact stays live):
  - counter (KINEMATIC): 800 x 640 x 140 mm kitchen bench.
  - stove fixture (KINEMATIC, FIXED pose — it anchors both joints and
    kinematic joint anchors are world-fixed): valve housing (floor, four
    walls, roof with the shaft hole), gas burner box on the left with a
    flame pad that recolors with the LIVE valve state, brass supply pipe.
  - rotor (DYNAMIC, 0.25 kg, revolute-Z, limits [0, 90] deg, heavy angular
    damping): square dial plate with a central opening for the shaft plus
    four hanging drive VANES. Starts at 68..88 deg = valve OPEN, flame LIT.
  - knob (DYNAMIC, 0.35 kg, D6 guide: transZ in [0, 16] mm free-fall-closed,
    rotZ free, all else locked): hub + four arms carrying four drive POSTS,
    shaft up through the roof hole, cap disc and a red WING BAR (24 mm wide
    — a parallel-jaw gripper grasp feature).
  - 2 lidded stock POTS (DYNAMIC, decoys) shuffled over 3 front-apron slots.

Rubric (0..1; anchored in the demonstrated solve; monotone along it):
  0.15  engaging the clutch at least once (latched lift >= engage_z)
  +0.55 * progress of the LATCHED MINIMUM rotor angle from theta0 down to
        theta_off (reaching the OFF threshold -> 0.70 total)
  1.0   iff success(): rotor angle <= theta_off AND knob RELEASED (lift at
        the bottom stop) AND knob + rotor calm AND all bodies finite.
  Doing nothing scores ~0; free-spinning the un-lifted knob scores ~0;
  holding the knob up at the OFF stop caps at 0.70 until released.

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


_G = 9.81

# ----- procedural compound spawners -------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float | None) -> None:
    """Author one box child; `contact_offset=None` -> VISUAL ONLY (no collider)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _rigid_root(root, mass: float, kinematic: bool, lin_damp: float = 0.0,
                ang_damp: float = 0.0) -> None:
    from pxr import PhysxSchema, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:  # corpus-validated authoring: never author the attr False
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    if not kinematic:
        px.CreateSleepThresholdAttr(0.0)
        px.CreateStabilizationThresholdAttr(0.0)


def _spawn_counter(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC kitchen bench. Origin = deck centre at GROUND level; the solid
    counter block fills z in [0, deck_h]."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg  # unwrap: the spawner cfg carries the scene cfg in its `cfg` field
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 40.0, kinematic=True)
    _box(stage, f"{prim_path}/deck",
         (2 * cfg.counter_half_x, 2 * cfg.counter_half_y, cfg.deck_h),
         (0.0, 0.0, cfg.deck_h / 2), cfg.deck_color, cfg.contact_offset)
    return root


def _spawn_stove(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC gas stove fixture. Origin = the valve AXIS on the deck top.
    One kinematic body: the sealed valve HOUSING (floor plate, 4 walls, roof
    of 4 strips leaving only the square shaft hole), the burner box + top
    plate + VISUAL flame pad (recolored with the live valve state), and a
    brass supply pipe (visual). The stove anchors both joints, so it is NEVER
    moved after spawn (kinematic joint anchors are world-fixed)."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, 20.0, kinematic=True)
    co, c = cfg.contact_offset, cfg
    out_h = c.cham_half + c.wall_t          # housing outer half-extent

    # housing floor plate
    _box(stage, f"{prim_path}/floor", (2 * out_h, 2 * out_h, c.floor_t),
         (0.0, 0.0, c.floor_t / 2), c.housing_color, co)
    # 4 walls: floor_t .. wall_z_hi
    wh = c.wall_z_hi - c.floor_t
    wz = (c.floor_t + c.wall_z_hi) / 2
    for tag, sy in (("wall_n", 1.0), ("wall_s", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (2 * out_h, c.wall_t, wh),
             (0.0, sy * (c.cham_half + c.wall_t / 2), wz), c.housing_color, co)
    for tag, sx in (("wall_e", 1.0), ("wall_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (c.wall_t, 2 * c.cham_half, wh),
             (sx * (c.cham_half + c.wall_t / 2), 0.0, wz), c.housing_color, co)
    # roof: 4 strips leaving ONLY the central square shaft hole (half hole_half)
    rz = c.wall_z_hi + c.roof_t / 2
    span = out_h - c.hole_half
    for tag, sy in (("roof_n", 1.0), ("roof_s", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (2 * out_h, span, c.roof_t),
             (0.0, sy * (c.hole_half + span / 2), rz), c.roof_color, co)
    for tag, sx in (("roof_e", 1.0), ("roof_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}", (span, 2 * c.hole_half, c.roof_t),
             (sx * (c.hole_half + span / 2), 0.0, rz), c.roof_color, co)

    # burner box + top plate + flame pad (flame is VISUAL ONLY, recolored live)
    bx, bh = c.burner_x, c.burner_half
    _box(stage, f"{prim_path}/burner", (2 * bh, 2 * bh, c.burner_h),
         (bx, 0.0, c.burner_h / 2), c.burner_color, co)
    _box(stage, f"{prim_path}/burner_plate", (2 * bh - 0.010, 2 * bh - 0.010, 0.004),
         (bx, 0.0, c.burner_h + 0.002), c.plate_color, co)
    _box(stage, f"{prim_path}/flame", (0.060, 0.060, 0.010),
         (bx, 0.0, c.burner_h + 0.009), c.flame_on, None)

    # brass gas pipe from the housing to the burner (visual only)
    _box(stage, f"{prim_path}/pipe", (abs(bx) - bh - out_h + 0.030, 0.018, 0.018),
         ((bx + bh - out_h) / 2 + 0.005, 0.0, 0.030), c.pipe_color, None)
    return root


def _spawn_rotor(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC valve rotor. Origin = the valve AXIS on the deck top (the
    bind-time revolute Z joint anchors here). One compound body: a square
    dial PLATE (annulus of 4 strips around the central shaft opening) plus
    four hanging drive VANES below its rim. Mass properties are authored
    EXPLICITLY (MassAPI mass alone leaves the CoM at the body origin —
    corpus lesson): CoM on the axis so gravity exerts NO torque about the
    hinge — the rotor holds whatever angle it is left at."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.rotor_mass, kinematic=False, ang_damp=cfg.rotor_damp)
    mapi = UsdPhysics.MassAPI(root)
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.050))
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0006, 0.0006, 0.0005))
    mapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    co, c = cfg.contact_offset, cfg

    # dial plate: square annulus, opening half plate_open_half, outer half
    # plate_outer_half, z in [plate_z_lo, plate_z_lo + plate_t]
    pz = c.plate_z_lo + c.plate_t / 2
    strip = c.plate_outer_half - c.plate_open_half
    for tag, sy in (("plate_n", 1.0), ("plate_s", -1.0)):
        _box(stage, f"{prim_path}/{tag}",
             (2 * c.plate_outer_half, strip, c.plate_t),
             (0.0, sy * (c.plate_open_half + strip / 2), pz), c.rotor_color, co)
    for tag, sx in (("plate_e", 1.0), ("plate_w", -1.0)):
        _box(stage, f"{prim_path}/{tag}",
             (strip, 2 * c.plate_open_half, c.plate_t),
             (sx * (c.plate_open_half + strip / 2), 0.0, pz), c.rotor_color, co)

    # four hanging drive vanes below the rim (the clutch's driven faces)
    vr = (c.vane_r_lo + c.vane_r_hi) / 2
    vz = c.vane_z_lo + c.vane_h / 2
    rad, tan = c.vane_r_hi - c.vane_r_lo, 2 * c.vane_half_w
    for tag, ctr, size in (
        ("vane_e", (vr, 0.0, vz), (rad, tan, c.vane_h)),
        ("vane_w", (-vr, 0.0, vz), (rad, tan, c.vane_h)),
        ("vane_n", (0.0, vr, vz), (tan, rad, c.vane_h)),
        ("vane_s", (0.0, -vr, vz), (tan, rad, c.vane_h)),
    ):
        _box(stage, f"{prim_path}/{tag}", size, ctr, c.vane_color, co)
    return root


def _spawn_knob(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC clutch knob. Origin = the valve AXIS on the deck top at the
    BOTTOM of its travel (the bind-time D6 guide anchors here; gravity
    returns it to transZ = 0). One compound body, bottom to top: hub + four
    arms carrying four drive POSTS (the clutch's driving faces), square
    shaft rising through the roof hole, cap disc, and the red WING BAR (the
    grasp feature). Authored CoM on the axis; diagonal inertia."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.knob_mass, kinematic=False, lin_damp=0.5, ang_damp=0.5)
    mapi = UsdPhysics.MassAPI(root)
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.060))
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(0.002, 0.002, 0.0015))
    mapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    co, c = cfg.contact_offset, cfg

    hub_z = c.post_z_lo - 0.004               # hub/arm band: post_z_lo-8mm .. post_z_lo
    _box(stage, f"{prim_path}/hub", (0.030, 0.030, 0.008),
         (0.0, 0.0, hub_z), c.knob_color, co)
    arm_len = c.arm_r_out - 0.015             # arms: hub edge (r=0.015) .. arm_r_out
    for tag, ctr, size in (
        ("arm_e", ((0.015 + c.arm_r_out) / 2, 0.0, hub_z), (arm_len, 0.016, 0.008)),
        ("arm_w", (-(0.015 + c.arm_r_out) / 2, 0.0, hub_z), (arm_len, 0.016, 0.008)),
        ("arm_n", (0.0, (0.015 + c.arm_r_out) / 2, hub_z), (0.016, arm_len, 0.008)),
        ("arm_s", (0.0, -(0.015 + c.arm_r_out) / 2, hub_z), (0.016, arm_len, 0.008)),
    ):
        _box(stage, f"{prim_path}/{tag}", size, ctr, c.knob_color, co)
    # drive posts: raised half a step so their tops sit at post_z_lo + post_h
    pz = c.post_z_lo + c.post_h / 2
    for tag, ctr in (("post_e", (c.post_r, 0.0, pz)), ("post_w", (-c.post_r, 0.0, pz)),
                     ("post_n", (0.0, c.post_r, pz)), ("post_s", (0.0, -c.post_r, pz))):
        _box(stage, f"{prim_path}/{tag}", (c.post_s, c.post_s, c.post_h), ctr,
             c.post_color, co)
    # shaft: post band top .. disc bottom, through the roof hole
    sh = c.disc_z_lo - c.post_z_lo
    _box(stage, f"{prim_path}/shaft", (2 * c.shaft_half, 2 * c.shaft_half, sh),
         (0.0, 0.0, c.post_z_lo + sh / 2), c.knob_color, co)
    # cap disc + red wing bar (the grasp feature)
    _box(stage, f"{prim_path}/disc", (2 * c.disc_half, 2 * c.disc_half, c.disc_t),
         (0.0, 0.0, c.disc_z_lo + c.disc_t / 2), c.knob_color, co)
    _box(stage, f"{prim_path}/wing", (c.wing_len, c.wing_w, c.wing_t),
         (0.0, 0.0, c.disc_z_lo + c.disc_t + c.wing_t / 2), c.wing_color, co)
    return root


def _spawn_pot(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC lidded stock pot (decoy). Origin = base centre at its bottom."""
    import omni.usd
    from pxr import UsdGeom

    cfg = cfg.cfg
    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _rigid_root(root, cfg.pot_mass, kinematic=False, lin_damp=0.2, ang_damp=0.3)
    c = cfg
    _box(stage, f"{prim_path}/body", (2 * c.pot_half, 2 * c.pot_half, c.pot_h),
         (0.0, 0.0, c.pot_h / 2), c.pot_color, c.contact_offset)
    _box(stage, f"{prim_path}/lid", (2 * c.pot_half + 0.005, 2 * c.pot_half + 0.005, 0.008),
         (0.0, 0.0, c.pot_h + 0.004), c.lid_color, c.contact_offset)
    _box(stage, f"{prim_path}/handle", (0.016, 0.016, 0.012),
         (0.0, 0.0, c.pot_h + 0.014), c.lid_color, None)
    return root


def _compound_spawner_cfg(kind: str, spawn_fn, scene_cfg: Any, mass: float,
                          kinematic: bool, lin_damp: float = 0.0,
                          ang_damp: float = 0.0) -> Any:
    """Build (once) and instantiate a RigidObjectSpawnerCfg subclass wrapping
    `spawn_fn`, carrying the scene cfg through a single `cfg` field. The
    spawner-cfg rigid_props are applied by the isaaclab clone wrapper AFTER the
    spawn fn runs, so they are the authoritative word (sleep stays OFF: sleeping
    GPU bodies freeze mid-settle and ignore velocity writes — corpus lesson)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if kind not in _SPAWNER_CACHE:

        @configclass
        class _Cfg(RigidObjectSpawnerCfg):
            func: Callable = clone(spawn_fn)
            cfg: Any = None

        _Cfg.__name__ = f"{kind.title()}SpawnerCfg"
        _SPAWNER_CACHE[kind] = _Cfg

    if kinematic:
        rp = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    else:
        rp = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False, sleep_threshold=0.0, stabilization_threshold=0.0,
            max_depenetration_velocity=0.5,
            linear_damping=lin_damp, angular_damping=ang_damp,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
    return _SPAWNER_CACHE[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=rp,
        cfg=scene_cfg,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ClutchValveStoveSceneCfg(BaseCfg):
    """Config for `ClutchValveStoveScene`. The clutch geometry that the task's
    honesty rests on is asserted in __post_init__: the rest air gap keeps the
    un-lifted knob a true free-spin decoy (beyond speculative-contact reach);
    the lifted posts overlap the vanes by a solid band; the housing passes
    ONLY the shaft; the backlash and vane gaps admit the posts at any yaw."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    theta_off_deg: float = tunable(8.0)   # valve OFF once rotor angle <= this
    engage_z: float = tunable(0.012)      # lift that counts as clutch ENGAGED (m)
    disengage_z: float = tunable(0.004)   # released = lift at/below this (m)
    knob_calm: float = tunable(0.12)      # knob |lin vel| gate at success (m/s)
    knob_spin_calm: float = tunable(1.0)  # knob |ang vel| gate at success (rad/s)
    rotor_calm: float = tunable(0.6)      # rotor |ang vel| gate at success (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    theta0_lo_deg: float = tunable(68.0)  # initial valve angle window (deg)
    theta0_hi_deg: float = tunable(88.0)
    slot_xs: tuple = tunable((-0.24, -0.02, 0.20))  # 3 apron slots, 2 pots shuffled
    slot_y: float = tunable(-0.200)
    slot_jitter: float = tunable(0.012)

    # --- info: counter ---------------------------------------------------------------------------
    deck_h: float = info(0.140)
    counter_half_x: float = info(0.400)
    counter_half_y: float = info(0.320)
    # --- info: stove fixture (FIXED pose — joint anchor; never randomized) -----------------------
    stove_x: float = info(0.120)          # valve axis on the deck (world/deck xy)
    stove_y: float = info(0.080)
    floor_t: float = info(0.010)          # housing floor plate thickness
    cham_half: float = info(0.075)        # housing interior half-extent
    wall_t: float = info(0.012)
    wall_z_hi: float = info(0.070)        # wall top = roof bottom (stove-local z)
    roof_t: float = info(0.008)
    hole_half: float = info(0.022)        # roof shaft-hole half-extent
    burner_x: float = info(-0.240)        # burner centre (stove-local x)
    burner_half: float = info(0.080)
    burner_h: float = info(0.040)
    # --- info: rotor (the sealed valve dial) -----------------------------------------------------
    rotor_mass: float = info(0.25)
    rotor_damp: float = info(4.0)         # heavy: bounds post-drive coast to a few deg
    rotor_hi_deg: float = info(90.0)      # OPEN stop; lower stop 0 deg = fully shut
    plate_z_lo: float = info(0.056)
    plate_t: float = info(0.008)
    plate_outer_half: float = info(0.045)
    plate_open_half: float = info(0.022)  # central opening (shaft pass-through)
    vane_r_lo: float = info(0.036)        # hanging vane radial band
    vane_r_hi: float = info(0.046)
    vane_half_w: float = info(0.010)      # vane tangential half-width
    vane_z_lo: float = info(0.038)        # vane bottom (stove-local z)
    vane_h: float = info(0.018)
    # --- info: knob (the clutch cap) -------------------------------------------------------------
    knob_mass: float = info(0.35)
    knob_travel: float = info(0.016)      # D6 transZ range [0, travel]
    post_r: float = info(0.040)           # drive-post centre radius
    post_s: float = info(0.014)           # post square side
    post_z_lo: float = info(0.024)        # post bottom at rest (stove-local z)
    post_h: float = info(0.008)
    arm_r_out: float = info(0.031)        # carrier arms end well inside the vanes
    shaft_half: float = info(0.010)
    disc_z_lo: float = info(0.092)        # cap disc bottom (= shaft top)
    disc_half: float = info(0.035)
    disc_t: float = info(0.016)
    wing_len: float = info(0.110)
    wing_w: float = info(0.024)           # fits an 80 mm parallel jaw with margin
    wing_t: float = info(0.018)
    # --- info: pots ------------------------------------------------------------------------------
    n_pot: int = info(2)
    pot_half: float = info(0.0375)
    pot_h: float = info(0.090)
    pot_mass: float = info(0.40)
    # --- info: misc ------------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    deck_color: tuple = info((0.42, 0.44, 0.48))
    housing_color: tuple = info((0.30, 0.31, 0.35))
    roof_color: tuple = info((0.24, 0.25, 0.29))
    burner_color: tuple = info((0.16, 0.17, 0.19))
    plate_color: tuple = info((0.10, 0.10, 0.11))
    pipe_color: tuple = info((0.55, 0.45, 0.20))
    rotor_color: tuple = info((0.20, 0.35, 0.55))
    vane_color: tuple = info((0.16, 0.28, 0.45))
    knob_color: tuple = info((0.55, 0.56, 0.60))
    post_color: tuple = info((0.40, 0.41, 0.46))
    wing_color: tuple = info((0.85, 0.12, 0.10))
    pot_color: tuple = info((0.62, 0.60, 0.58))
    lid_color: tuple = info((0.45, 0.44, 0.43))
    flame_on: tuple = info((1.00, 0.50, 0.08))
    flame_off: tuple = info((0.15, 0.06, 0.05))

    # ----- derived -------------------------------------------------------------------------------
    @property
    def post_top_rest(self) -> float:
        """Drive-post TOP with the knob at the bottom stop (stove-local z)."""
        return self.post_z_lo + self.post_h

    @property
    def rest_gap(self) -> float:
        """Air gap post-top -> vane-bottom at rest: the free-spin decoy."""
        return self.vane_z_lo - self.post_top_rest

    @property
    def vane_half_deg(self) -> float:
        """Vane angular half-extent at its widest (inner corner), degrees."""
        return math.degrees(math.atan2(self.vane_half_w, self.vane_r_lo))

    @property
    def post_half_deg(self) -> float:
        """Post angular half-extent at its widest (inner corner), degrees."""
        return math.degrees(math.atan2(self.post_s / 2, self.post_r - self.post_s / 2))

    @property
    def backlash_deg(self) -> float:
        """Free play between a post and the next vane face, degrees."""
        return 90.0 - 2 * self.vane_half_deg - 2 * self.post_half_deg

    @property
    def pot_spawn_z(self) -> float:
        return self.deck_h + 0.002

    def __post_init__(self) -> None:
        c = self.contact_offset
        # -- the free-spin decoy is honest: rest air gap beyond speculative reach --
        assert self.rest_gap >= 0.005 and self.rest_gap > 2 * c + 0.002, \
            f"rest post->vane gap must beat speculative contact ({self.rest_gap:.4f})"
        # -- released = truly disengaged; engaged = solid axial overlap --
        assert self.disengage_z <= self.rest_gap - 0.002 + 1e-9, \
            "at the release gate the posts must still sit clear below the vanes"
        assert self.rest_gap + 0.004 < self.engage_z <= self.knob_travel - 0.003, \
            "the engage latch must fire only with real post/vane overlap, " \
            "and be reachable below the top stop"
        ov_engage = self.post_top_rest + self.engage_z - self.vane_z_lo
        assert ov_engage >= 0.005, f"overlap at engage_z too thin ({ov_engage:.4f})"
        ov_full = min(self.post_top_rest + self.knob_travel, self.vane_z_lo + self.vane_h) \
            - max(self.post_z_lo + self.knob_travel, self.vane_z_lo)
        assert ov_full >= 0.007, f"full-lift clutch overlap too thin ({ov_full:.4f})"
        # -- posts and vanes share a solid radial band; carrier arms stay clear --
        post_in, post_out = self.post_r - self.post_s / 2, self.post_r + self.post_s / 2
        rad_ov = min(post_out, self.vane_r_hi) - max(post_in, self.vane_r_lo)
        assert rad_ov >= 0.006, f"post/vane radial overlap too thin ({rad_ov:.4f})"
        assert self.arm_r_out + 0.004 <= self.vane_r_lo, \
            "the carrier arms must never brush the vanes, even fully lifted"
        # -- angular arithmetic: posts fit the vane gaps at ANY yaw; real backlash --
        gap_deg = 90.0 - 2 * self.vane_half_deg
        assert gap_deg >= 2 * self.post_half_deg + 10.0, \
            f"the vane gap must admit a post with >=10 deg to spare ({gap_deg:.1f})"
        assert self.backlash_deg >= 10.0, \
            f"the clutch needs real backlash ({self.backlash_deg:.1f} deg)"
        # -- the housing passes ONLY the shaft --
        shaft_diag = self.shaft_half * math.sqrt(2)
        assert shaft_diag + 0.005 <= self.hole_half, "shaft must clear the roof hole"
        assert shaft_diag + 0.005 <= self.plate_open_half, \
            "shaft must clear the rotor plate opening"
        assert self.hole_half <= 0.026, \
            "the roof hole must stay finger-proof (only the shaft passes)"
        # -- rotor swings free inside the chamber; knob stack clears the fixture --
        assert self.plate_outer_half * math.sqrt(2) <= self.cham_half - 0.004, \
            "the dial plate corner must clear the walls at every angle"
        assert math.hypot(post_out, self.post_s / 2) <= self.cham_half - 0.004 and \
            self.vane_r_hi <= self.cham_half - 0.004, \
            "posts and vanes must clear the walls at every angle"
        assert self.plate_z_lo + self.plate_t + 0.004 <= self.wall_z_hi, \
            "the dial plate must stay under the roof"
        assert self.vane_z_lo >= self.floor_t + 0.004 + self.knob_travel, \
            "the vane band must clear the floor plate"
        assert self.post_z_lo - 0.008 >= self.floor_t + 0.004, \
            "the hub/arm band must clear the housing floor at rest"
        assert self.post_top_rest + self.knob_travel + 0.004 <= self.plate_z_lo, \
            "fully lifted posts must never touch the dial plate"
        assert self.disc_z_lo - (self.wall_z_hi + self.roof_t) >= 0.010, \
            "the cap disc must ride clear above the roof"
        assert self.disc_z_lo >= self.post_z_lo + 0.050, "shaft must span roof height"
        # -- rubric window --
        assert 5.0 <= self.theta_off_deg <= self.theta0_lo_deg - 40.0, \
            "OFF threshold readable and far from every start angle"
        assert self.theta0_hi_deg <= self.rotor_hi_deg and \
            self.theta0_hi_deg - self.theta0_lo_deg >= 10.0, \
            "start-angle window must be real and inside the stops"
        # -- graspability --
        assert self.wing_w <= 0.075, "wing bar must fit a parallel-jaw gripper"
        assert self.wing_len >= 2 * self.disc_half + 0.020, \
            "wing bar must protrude past the cap disc for a clean grasp"
        # -- fixture inside the counter; burner clear of the housing --
        out_h = self.cham_half + self.wall_t
        assert self.stove_x + out_h < self.counter_half_x - 0.020
        assert self.stove_x + self.burner_x - self.burner_half > -self.counter_half_x + 0.020
        assert abs(self.stove_y) + out_h < self.counter_half_y - 0.020
        assert self.burner_x + self.burner_half <= -out_h - 0.020, \
            "burner box must sit clear west of the housing"
        # -- pot slots: clear of the fixture, each other, and the counter edge --
        pot_diag = self.pot_half * math.sqrt(2)
        reach = pot_diag + self.slot_jitter
        assert self.slot_y + reach < self.stove_y - out_h - 0.030, \
            "the pot row must sit clear (south) of the housing"
        assert self.slot_y + reach < self.stove_y - self.burner_half - 0.030, \
            "the pot row must sit clear (south) of the burner"
        xs = sorted(self.slot_xs)
        for a, b in zip(xs, xs[1:]):
            assert b - a >= 2 * reach + 0.010, "adjacent pot slots must never collide"
        assert max(abs(x) for x in xs) + reach < self.counter_half_x - 0.020
        assert abs(self.slot_y) + reach < self.counter_half_y - 0.020
        assert len(self.slot_xs) == self.n_pot + 1, "pots shuffle over n_pot+1 slots"


# ----- small helpers (wxyz quats, torch, batched) -----------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors `v` (N,3) into the body frame of quats `q` (N,4 wxyz)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("clutch_valve_stove")
class ClutchValveStoveScene(BaseScene):
    cfg: ClutchValveStoveSceneCfg

    def __init__(self, cfg: ClutchValveStoveSceneCfg | None = None) -> None:
        super().__init__(cfg or ClutchValveStoveSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        axis = (c.stove_x, c.stove_y, c.deck_h)  # valve axis on the deck top
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "counter": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Counter",
                spawn=_compound_spawner_cfg("counter", _spawn_counter, c, 40.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            # Stove authored at its FIXED pose; the bind-time joints anchor
            # here and the stove is NEVER moved (kinematic joint anchors are
            # world-fixed).
            "stove": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stove",
                spawn=_compound_spawner_cfg("stove", _spawn_stove, c, 20.0,
                                            kinematic=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=axis),
            ),
            "rotor": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rotor",
                spawn=_compound_spawner_cfg("rotor", _spawn_rotor, c, c.rotor_mass,
                                            kinematic=False, ang_damp=c.rotor_damp),
                init_state=RigidObjectCfg.InitialStateCfg(pos=axis),
            ),
            "knob": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Knob",
                spawn=_compound_spawner_cfg("knob", _spawn_knob, c, c.knob_mass,
                                            kinematic=False, lin_damp=0.5,
                                            ang_damp=0.5),
                init_state=RigidObjectCfg.InitialStateCfg(pos=axis),
            ),
        }
        for k in range(c.n_pot):
            out[f"pot{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pot" + str(k),
                spawn=_compound_spawner_cfg("pot", _spawn_pot, c, c.pot_mass,
                                            kinematic=False, lin_damp=0.2,
                                            ang_damp=0.3),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_xs[k], c.slot_y, c.pot_spawn_z)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # the solve/smoke drive the knob with external wrenches held
                # across substeps — without this flag the wrench only acts on
                # the first substep after each write (corpus lesson)
                "enable_external_forces_every_iteration": True,
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
        self.counter: RigidObject = env.iscene["counter"]
        self.stove: RigidObject = env.iscene["stove"]
        self.rotor: RigidObject = env.iscene["rotor"]
        self.knob: RigidObject = env.iscene["knob"]
        self.pots: list[RigidObject] = [
            env.iscene[f"pot{k}"] for k in range(self.cfg.n_pot)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._engaged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._theta0 = torch.full((n,), math.radians(self.cfg.theta0_hi_deg),
                                  device=dev)
        self._min_ang = self._theta0.clone()
        self._flame_shown: list[bool | None] = [None] * n
        self._flame_prims = None  # lazily grabbed in _refresh_flames
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env joints, authored ONCE at bind time against the AUTHORED
        poses (all bodies share the valve-axis origin, so every joint local
        frame is the identity):
          - rotor: revolute Z on the KINEMATIC stove; LIMITS [0, 90] deg are
            the valve stops (0 = fully shut, 90 = fully open).
          - knob: generic D6 `UsdPhysics.Joint` guide; per-axis LimitAPI locks
            transX/transY/rotX/rotY (low > high), bounds transZ to
            [0, knob_travel] metres, and leaves rotZ FREE (no LimitAPI).
        Each joint's collision filter disables ONLY its own stove<->body pair
        (this also kills roof-hole edge chatter on the sliding shaft); the
        knob<->rotor CLUTCH contact stays fully live."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        ident = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/rotor_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Stove"])
            j.CreateBody1Rel().SetTargets([f"{base}/Rotor"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(ident)
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(ident)
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.rotor_hi_deg))

            dj = UsdPhysics.Joint.Define(stage, f"{base}/knob_guide")
            dj.CreateBody0Rel().SetTargets([f"{base}/Stove"])
            dj.CreateBody1Rel().SetTargets([f"{base}/Knob"])
            dj.CreateCollisionEnabledAttr(False)
            dj.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            dj.CreateLocalRot0Attr(ident)
            dj.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            dj.CreateLocalRot1Attr(ident)
            prim = dj.GetPrim()
            for ax in ("transX", "transY", "rotX", "rotY"):
                la = UsdPhysics.LimitAPI.Apply(prim, ax)
                la.CreateLowAttr(1.0)   # low > high = axis LOCKED
                la.CreateHighAttr(-1.0)
            lz = UsdPhysics.LimitAPI.Apply(prim, "transZ")
            lz.CreateLowAttr(0.0)       # metres; gravity parks the knob here
            lz.CreateHighAttr(float(c.knob_travel))
            # rotZ: NO LimitAPI -> free spin

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: counter and stove re-asserted at their FIXED poses
        (the stove anchors both joints — kinematic joint anchors are
        world-fixed); the rotor is written to a random OPEN angle theta0 in
        [68, 88] deg (flame LIT); the knob drops to the bottom of its guide
        with a random free yaw; the 2 pots are PERMUTED over the 3 apron
        slots with xy jitter and free yaw (argsort of torch.rand — the FIRST
        post-seed draw is near-degenerate, so one draw is burned first);
        latches and flame cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(m, device=dev)  # burn the degenerate first post-seed draw

        def write(body, xy, z, quat=None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3:7] = quat if quat is not None else torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=dev).expand(m, 4)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.counter, torch.zeros(m, 2, device=dev),
              torch.zeros(m, device=dev))
        axis_xy = torch.tensor([c.stove_x, c.stove_y], device=dev).expand(m, 2)
        write(self.stove, axis_xy, c.deck_h)

        lo, hi = math.radians(c.theta0_lo_deg), math.radians(c.theta0_hi_deg)
        theta0 = lo + torch.rand(m, device=dev) * (hi - lo)
        write(self.rotor, axis_xy, c.deck_h, _qz(theta0))

        yaw = torch.rand(m, device=dev) * 2 * math.pi
        write(self.knob, axis_xy, c.deck_h, _qz(yaw))

        slot_xs = torch.tensor(c.slot_xs, device=dev)
        perm = torch.argsort(torch.rand(m, len(c.slot_xs), device=dev), dim=1)
        for i, body in enumerate(self.pots):
            sx = slot_xs[perm[:, i]] + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            sy = c.slot_y + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            pyaw = torch.rand(m, device=dev) * 2 * math.pi
            write(body, torch.stack([sx, sy], dim=-1), c.pot_spawn_z, _qz(pyaw))

        self._engaged[env_ids] = False
        self._theta0[env_ids] = theta0
        self._min_ang[env_ids] = theta0
        for e in env_ids.tolist():
            self._flame_shown[e] = None  # force a flame refresh next post_step

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "counter": self.counter.data.root_state_w[env_ids].clone(),
            "stove": self.stove.data.root_state_w[env_ids].clone(),
            "rotor": self.rotor.data.root_state_w[env_ids].clone(),
            "knob": self.knob.data.root_state_w[env_ids].clone(),
            "engaged": self._engaged[env_ids].clone(),
            "theta0": self._theta0[env_ids].clone(),
            "min_ang": self._min_ang[env_ids].clone(),
        }
        for k, b in enumerate(self.pots):
            out[f"pot{k}"] = b.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for name in ("counter", "stove", "rotor", "knob"):
            getattr(self, name).write_root_state_to_sim(state[name], env_ids)
        for k, b in enumerate(self.pots):
            b.write_root_state_to_sim(state[f"pot{k}"], env_ids)
        self._engaged[env_ids] = state["engaged"]
        self._theta0[env_ids] = state["theta0"]
        self._min_ang[env_ids] = state["min_ang"]
        for e in env_ids.tolist():
            self._flame_shown[e] = None

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A steel KITCHEN BENCH ({2 * c.counter_half_x * 100:.0f} x "
            f"{2 * c.counter_half_y * 100:.0f} cm, {c.deck_h * 100:.0f} cm tall). "
            f"On its back-right area sits a GAS STOVE. Its burner box on the "
            f"left is LIT — an orange flame pad burns on top, fed by a brass "
            f"pipe from the VALVE HOUSING on the right: a sealed steel box "
            f"({(c.cham_half + c.wall_t) * 200:.0f} mm square, walls and a "
            f"roof) whose only opening is a small square hole in the roof. "
            f"Inside — visible to no one and reachable by nothing wider than "
            f"the shaft — a blue VALVE DIAL holds the gas open (it starts "
            f"somewhere in the open range; the exact angle is randomized per "
            f"episode). Through the roof hole rises the square shaft of the "
            f"CHILD-SAFE CLUTCH KNOB: a grey cap disc with a RED WING BAR "
            f"({c.wing_len * 1000:.0f} x {c.wing_w * 1000:.0f} mm) on top. "
            f"The knob rides a 2-DOF guide: it can slide UP "
            f"{c.knob_travel * 1000:.0f} mm and it spins freely. At rest its "
            f"internal drive posts hang BELOW the dial's clutch vanes with an "
            f"air gap — twisting the un-lifted knob just spins it uselessly "
            f"(that is the child-safety). Only while the knob is PULLED UP "
            f"(≳{c.engage_z * 1000:.0f} mm) do the posts rise into the vanes "
            f"and form a dog clutch (with ~{c.backlash_deg:.0f} deg of free "
            f"backlash) that can wind the dial. Two lidded stock pots stand "
            f"on the front apron (positions shuffled per episode); they are "
            f"props and matter to nothing.\n"
            f"Goal: turn OFF the stove. Pull the clutch knob UP, keep it "
            f"lifted while twisting it CLOCKWISE (viewed from above) so the "
            f"dial winds down to its shut stop and the flame dies, then "
            f"RELEASE the knob so it drops back down and the clutch opens. "
            f"Success: the valve dial at/below {c.theta_off_deg:.0f} deg "
            f"(flame out), the knob released back at the bottom of its "
            f"travel, everything settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Turn off the gas stove: pull the red-barred clutch knob up, hold "
            "it up while twisting clockwise until the flame goes out, then "
            "release the knob."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def rotor_angle(self) -> torch.Tensor:
        """(N,) valve dial angle in radians (0 = shut stop, ~pi/2 = open stop).
        The rotor only ever rotates about the axis Z, so the quat is
        (cos a/2, 0, 0, sin a/2); limits [0, 90] deg keep it wrap-free."""
        q = self.rotor.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def knob_lift(self) -> torch.Tensor:
        """(N,) knob lift above the bottom stop (m)."""
        return self.knob.data.root_pos_w[:, 2] - self.env_origins[:, 2] \
            - self.cfg.deck_h

    def knob_yaw(self) -> torch.Tensor:
        """(N,) knob free-spin yaw in radians (wraps at +/-pi)."""
        q = self.knob.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def valve_off(self) -> torch.Tensor:
        """(N,) bool: dial at/below the OFF threshold (gas shut)."""
        return self.rotor_angle() <= math.radians(self.cfg.theta_off_deg)

    def released(self) -> torch.Tensor:
        """(N,) bool: knob back at the bottom of its travel (clutch open)."""
        return self.knob_lift() <= self.cfg.disengage_z

    def _update_latches(self) -> None:
        self._engaged |= self.knob_lift() >= self.cfg.engage_z
        self._min_ang = torch.minimum(self._min_ang, self.rotor_angle())

    def _refresh_flames(self) -> None:
        from pxr import Gf

        if self._flame_prims is None:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            self._flame_prims = [
                UsdGeom.Cube(stage.GetPrimAtPath(f"/World/envs/env_{i}/Stove/flame"))
                for i in range(self.env.num_envs)]
        c = self.cfg
        lit = ~self.valve_off()  # flame tracks the LIVE valve state
        for e in range(self.env.num_envs):
            key = bool(lit[e])
            if key != self._flame_shown[e]:
                self._flame_shown[e] = key
                col = c.flame_on if key else c.flame_off
                self._flame_prims[e].GetDisplayColorAttr().Set([Gf.Vec3f(*col)])

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Once per physics substep: advance the engage / min-angle latches
        and refresh the flames. The scene owns NO forces — the clutch is pure
        contact, the knob return is gravity, the dial holds by damping and
        its stops."""
        self._update_latches()
        self._refresh_flames()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the valve dial rests at/below theta_off (gas OFF, flame
        out) AND the knob is RELEASED back at the bottom stop (a hand holding
        the knob up at the OFF stop is not a finished stove) AND the knob and
        dial are calm AND all bodies finite. Held only by the stops and
        gravity, so it persists hands-off."""
        self._update_latches()
        c = self.cfg
        knob_calm = (self.knob.data.root_lin_vel_w.norm(dim=-1) < c.knob_calm) \
            & (self.knob.data.root_ang_vel_w.norm(dim=-1) < c.knob_spin_calm)
        rotor_calm = self.rotor.data.root_ang_vel_w.norm(dim=-1) < c.rotor_calm
        finite = torch.isfinite(self.rotor.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.knob.data.root_pos_w).all(dim=-1)
        for b in self.pots:
            finite &= torch.isfinite(b.data.root_pos_w).all(dim=-1)
        return self.valve_off() & self.released() & knob_calm & rotor_calm & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 for ever engaging the clutch, plus 0.55
        times the latched-minimum dial progress from theta0 to theta_off
        (max 0.70 without release) — and exactly 1.0 iff success() holds
        live. Doing nothing and free-spinning the un-lifted knob score ~0."""
        self._update_latches()
        c = self.cfg
        off = math.radians(c.theta_off_deg)
        span = (self._theta0 - off).clamp(min=1e-4)
        prog = ((self._theta0 - self._min_ang) / span).clamp(0.0, 1.0)
        base = 0.15 * self._engaged.float() + 0.55 * prog
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="clutch_valve_stove", robot="null"))
