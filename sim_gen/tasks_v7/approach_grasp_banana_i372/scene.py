"""CrankEjectorScene — eject the sealed red cargo cube by cranking a scotch-yoke ram
(sim_gen task `approach_grasp_banana_i372`).

Derived from pick_place/approach_grasp_banana, but STRATEGICALLY different: the seed is
a PREHENSILE PICK-AND-PLACE — a Franka approaches a banana lying in the open among
table clutter, closes its jaw around it and carries it to a basket; the judged payload
is exactly the thing held, and the plan is approach -> grasp -> transport -> release.
Here the judged payload CANNOT be grasped at all: a red cube (42 mm) sits deep inside a
roofed magazine tunnel whose two end mouths leave only 6 mm of side clearance — less
than a Franka fingertip (7 mm), so a jaw can never straddle it — and whose roof slot
(30 mm) is narrower than the cube. A grey DECOY cube occupies the opposite end of the
same tunnel (which end holds the red cargo is a per-episode coin flip). The only
intended handle is a YELLOW CRANK PEG on a side-mounted crank wheel: the crank's inner
drive peg rides a tall fork on a double-ended ram (a scotch yoke), so cranking one way
slides the ram toward the cargo end and cranking the other way slides it toward the
decoy end. The correct rotation direction plows the red cube out of its mouth, over a
drop lip, into the catch pocket sunk at that end. The WRONG direction irreversibly
ejects the grey decoy into the opposite pocket — it can never be pushed back in, and
the episode is spoiled (score caps low). Success = the red cube settled inside its
catch pocket, the grey decoy still housed in the tunnel, mechanism quiet. Plan
skeleton: read which end is red, choose a crank direction, drive a rotary-to-linear
mechanism until ejection, release — no grasp of the payload, no carry, no proximity
goal; nothing in common with the seed beyond "an object must end up in a container".

Mechanics (plain rigid bodies + authored USD joints; the crank "hand" is a bounded
external torque applied in `post_step`, which consumes the `crank_drive` buffer that
solve/smoke probes write): a fixed kinematic RIG (plinth, tunnel, two pockets, bearing
block) anchors a prismatic-jointed RAM (axis x, joint collision-filtered against the
rig) and a revolute-jointed CRANK (axis y, +/-92 deg stops, gravity-neutral authored
CoM at the axis). Crank/ram couple ONLY by contact: the crank's inner peg between the
ram's fork plates (3 mm backlash), x_ram = R * sin(theta). The cubes are free bodies.

Rubric (graded 0..1, latched credit anchored in the demonstrated solve.py trajectory;
cargo latches require PATH CONTINUITY — per-substep travel under `cont_max` — and ram
progress requires the ram itself to move continuously, so teleports earn nothing):
  - `prog` (latched max in [0,1]): correct-direction ram displacement,
    clamp(side * x_ram / prog_sat, 0, 1); cranking the wrong way earns 0.
  - `ejected` (latch): with prog > eject_gate, the cargo cube has continuously moved
    past its mouth (side * x beyond the lip) and dropped below the tunnel floor.
  - `seated` (latch): after ejecting, the cargo is INSIDE the goal pocket box (the z
    band rejects wall-top perches); snaps prog to 1.
  - `fouled` (permanent latch): the decoy left the tunnel — wrong-direction cranking;
    score caps at foul_cap forever (irreversible by construction: no handle reaches
    into a pocket-sunk cube and the ram cannot pull).
  - success(): `seated` AND the cargo currently rests in the goal pocket AND the decoy
    is still housed AND crank/ram/cargo are quiet — credit only for a released,
    settled end state.
  - score() = 0.20*prog + 0.25*ejected + 0.25*seated + 0.30*success -> exactly 1.0 iff
    success(); ~0 for the null policy; a fouled episode caps at 0.20; disturbing the
    mechanism after success falls back to 0.70, never 0.

Per-episode randomization (readback-verified in smoke.py): the cargo SIDE (the crank
direction decision flips), the crank's initial angle (so the ram's start offset
varies), and each cube's gap/lateral jitter. A memorized fixed crank schedule that
ignores the side is punished by the foul latch.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- USD authoring helpers --------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _decorate(prim, color, contact_offset: float, material=None) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None, visual_only: bool = False) -> None:
    """Author one box child prim (translate -> scale; authored once)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    if visual_only:
        UsdGeom.Gprim(seg.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    else:
        _decorate(seg.GetPrim(), color, contact_offset, material)


def _cyl(stage, path: str, radius: float, height: float, axis: str, center, color,
         contact_offset: float, material=None) -> None:
    """Author one cylinder child prim (round cross-section — rotation-invariant)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr(axis)
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -radius),
                          Gf.Vec3f(radius, radius, radius)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    _decorate(seg.GetPrim(), color, contact_offset, material)


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC rig. Local frame: origin on the ground at the plan center, tunnel
    along x, crank face toward +y. Plinth + spine, the roofed magazine tunnel (roof
    slot |y| < slot_hw for the ram's fork), one sunk catch pocket beyond each mouth
    (near side sealed by the spine face directly under the drop lip — no perch rim),
    and the pillow bearing block on the roof that visually carries the crank axle."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    hl, hw, wt = cfg.tunnel_hl, cfg.inner_hw, cfg.wall_t
    ft, rl, rt = cfg.floor_top, cfg.roof_lo, cfg.roof_top

    # plinth + spine (the spine face IS the pocket near wall, under the drop lip) ----------
    _box(stage, f"{prim_path}/slab", (0.62, 0.34, 0.020), (0.0, 0.0, 0.010),
         cfg.base_color, co, material=mat)
    _box(stage, f"{prim_path}/spine", (2 * cfg.spine_hl, 2 * (hw + wt), 0.068),
         (0.0, 0.0, 0.054), cfg.base_color, co, material=mat)
    # tunnel: floor, side walls, roof strips (slot open along the center) -----------------
    _box(stage, f"{prim_path}/floor", (2 * hl, 2 * (hw + wt), ft - 0.088),
         (0.0, 0.0, (ft + 0.088) / 2), cfg.floor_color, co, material=mat)
    for sy in (+1, -1):
        _box(stage, f"{prim_path}/wall_{'n' if sy > 0 else 's'}",
             (2 * hl, wt, rt - ft), (0.0, sy * (hw + wt / 2), (rt + ft) / 2),
             cfg.wall_color, co, material=mat)
        _box(stage, f"{prim_path}/roof_{'n' if sy > 0 else 's'}",
             (2 * hl, hw - cfg.slot_hw, rt - rl),
             (0.0, sy * (cfg.slot_hw + (hw - cfg.slot_hw) / 2), (rt + rl) / 2),
             cfg.roof_color, co, material=mat)
    # catch pockets (floor + far wall + side walls; near side = spine face) ---------------
    pf, pw = cfg.pk_floor_top, cfg.pk_wall_top
    x0, x1 = cfg.spine_hl, cfg.pk_x1
    for sx, nm in ((+1, "e"), (-1, "w")):
        _box(stage, f"{prim_path}/pkfloor_{nm}", (x1 - x0 + wt, 2 * cfg.pk_hw + 2 * wt, 0.010),
             (sx * (x0 + x1 + wt) / 2, 0.0, pf - 0.005), cfg.pocket_color, co, material=mat)
        _box(stage, f"{prim_path}/pkfar_{nm}", (wt, 2 * cfg.pk_hw + 2 * wt, pw - 0.020),
             (sx * (x1 + wt / 2), 0.0, (pw + 0.020) / 2), cfg.pocket_color, co, material=mat)
        for sy in (+1, -1):
            _box(stage, f"{prim_path}/pkside_{nm}{'n' if sy > 0 else 's'}",
                 (x1 - x0, wt, pw - 0.020),
                 (sx * (x0 + x1) / 2, sy * (cfg.pk_hw + wt / 2), (pw + 0.020) / 2),
                 cfg.pocket_color, co, material=mat)
    # bearing block on the roof + visual axle stub ----------------------------------------
    _box(stage, f"{prim_path}/bearing", (0.050, 0.027, 0.053), (0.0, 0.0305, 0.1885),
         cfg.base_color, co, material=mat)
    _box(stage, f"{prim_path}/axle", (0.014, 0.030, 0.014),
         (0.0, cfg.crank_y - 0.001, cfg.axis_z), cfg.steel_color, co, visual_only=True)
    return root


def _spawn_ram(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC double-ended ram. Local frame: origin at the ram body center. The body
    slides inside the tunnel; two fork plates rise through the roof slot and cage the
    crank's drive peg (the scotch-yoke follower). Explicit mass/CoM/inertia."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(0.05)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/body", (2 * cfg.half_len, 2 * cfg.half_w, cfg.height),
         (0.0, 0.0, 0.0), cfg.color, co, material=mat)
    for sx in (+1, -1):
        _box(stage, f"{prim_path}/fork_{'e' if sx > 0 else 'w'}",
             (cfg.fork_t, 2 * cfg.fork_hw, cfg.fork_h),
             (sx * (cfg.fork_gap_h + cfg.fork_t / 2), 0.0, cfg.height / 2 + cfg.fork_h / 2),
             cfg.fork_color, co, material=mat)
    return root


def _spawn_crank(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC crank. Local frame: origin at the AXIS (revolute y). Hub + inner drive
    arm ending in the drive peg (reaches toward -y between the fork plates) + outer
    handle arm ending in the YELLOW handle peg (toward +y, the robot's handle).
    Authored CoM AT THE AXIS -> gravity-neutral; explicit diagonal inertia."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.08)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mat = _friction_material(stage, f"{prim_path}/mat", cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    _box(stage, f"{prim_path}/hub", (0.050, 0.016, 0.050), (0.0, 0.0, 0.0),
         cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/drive_arm", (0.024, 0.016, cfg.peg_r - 0.02),
         (0.0, 0.0, (cfg.peg_r + 0.02) / 2), cfg.color, co, material=mat)
    # ROUND pin (cylinder, axis Y): a square peg's rotated x-extent peg_half*(cos+sin)
    # wedges the 15 mm fork gap at theta ~= 20.5 deg; a scotch-yoke pin must present a
    # rotation-invariant cross-section, so author it as a cylinder.
    _cyl(stage, f"{prim_path}/drive_peg", cfg.peg_half, cfg.peg_len, "Y",
         (0.0, -(cfg.peg_len / 2 + 0.008), cfg.peg_r), cfg.steel_color, co, material=mat)
    _box(stage, f"{prim_path}/handle_arm", (0.024, 0.016, cfg.handle_r - 0.01),
         (0.0, 0.0, -(cfg.handle_r + 0.01) / 2), cfg.color, co, material=mat)
    _box(stage, f"{prim_path}/handle_peg", (0.013, 0.045, 0.013),
         (0.0, 0.0305, -cfg.handle_r), cfg.handle_color, co, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            tunnel_hl: float = 0.150
            inner_hw: float = 0.027
            wall_t: float = 0.010
            floor_top: float = 0.100
            roof_lo: float = 0.150
            roof_top: float = 0.162
            slot_hw: float = 0.015
            spine_hl: float = 0.145
            pk_x1: float = 0.265
            pk_hw: float = 0.045
            pk_floor_top: float = 0.030
            pk_wall_top: float = 0.090
            crank_y: float = 0.053
            axis_z: float = 0.200
            mu_static: float = 0.35
            mu_dynamic: float = 0.30
            base_color: tuple = (0.16, 0.16, 0.18)
            floor_color: tuple = (0.72, 0.70, 0.64)
            wall_color: tuple = (0.40, 0.40, 0.44)
            roof_color: tuple = (0.24, 0.24, 0.28)
            pocket_color: tuple = (0.12, 0.35, 0.38)
            steel_color: tuple = (0.65, 0.66, 0.70)
            contact_offset: float = 0.0015

        @configclass
        class RamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ram)
            half_len: float = 0.055
            half_w: float = 0.021
            height: float = 0.046
            fork_t: float = 0.008
            fork_hw: float = 0.012
            fork_h: float = 0.198
            fork_gap_h: float = 0.0075
            mass: float = 0.30
            inertia: tuple = (0.004, 0.005, 0.002)
            mu_static: float = 0.30
            mu_dynamic: float = 0.25
            color: tuple = (0.30, 0.42, 0.65)
            fork_color: tuple = (0.30, 0.42, 0.65)
            contact_offset: float = 0.0015

        @configclass
        class CrankSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_crank)
            peg_r: float = 0.130
            handle_r: float = 0.100
            peg_half: float = 0.006
            peg_len: float = 0.063
            mass: float = 0.50
            inertia: tuple = (0.005, 0.006, 0.002)
            mu_static: float = 0.30
            mu_dynamic: float = 0.25
            color: tuple = (0.20, 0.20, 0.24)
            steel_color: tuple = (0.65, 0.66, 0.70)
            handle_color: tuple = (0.92, 0.80, 0.10)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg
        _SPAWNER_CACHE["ram"] = RamSpawnerCfg
        _SPAWNER_CACHE["crank"] = CrankSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CrankEjectorSceneCfg(BaseCfg):
    """Config for `CrankEjectorScene`. Honesty is asserted in __post_init__: no jaw can
    straddle a cube at either mouth and no cube fits through the roof slot; the full
    crank stroke carries the cargo CoM well past the drop lip; the drive peg clears the
    bearing block over its whole sweep; the pocket z band separates a floor rest from a
    wall-top perch; the drive torque has a large margin over the worst resisting load;
    one substep of free fall stays far under the continuity gate; the crank handle
    stays inside the documented arm envelope; the rubric weights sum to 1."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    prog_sat: float = tunable(0.075)     # ram |x| where correct-direction progress saturates
    eject_gate: float = tunable(0.92)    # `ejected` requires latched prog beyond this
    eject_x: float = tunable(0.155)      # `ejected`: side*x_cargo beyond this ...
    eject_z: float = tunable(0.095)      # ... AND cargo center below this (off the floor)
    cont_max: float = tunable(0.030)     # path continuity: max cargo travel per substep (m)
    ram_cont: float = tunable(0.020)     # ram continuity: max ram travel per substep (m)
    settle_v: float = tunable(0.10)      # success: max cargo |lin vel| (m/s)
    settle_w: float = tunable(0.60)      # success: max crank |FD rate| (rad/s)
    settle_ram: float = tunable(0.05)    # success: max ram |x vel| (m/s)
    foul_cap: float = tunable(0.20)      # score cap once the decoy is ejected

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    theta0_deg: float = tunable(6.0)     # crank initial angle jitter, +/- (deg)
    gap_lo: float = tunable(0.003)       # cube-to-ram-face gap range (m)
    gap_hi: float = tunable(0.007)
    y_jit: float = tunable(0.0015)       # cube lateral jitter, +/- (m)

    # --- tunable: plant (difficulty dials) ------------------------------------------------------
    drive_max: float = tunable(0.60)     # |crank_drive| clamp (N*m about the axis)
    rev_lim_deg: float = tunable(92.0)   # crank hard stops (deg; < 175 avoids the wrap trap)
    cube_mass: float = tunable(0.05)
    cube_mu: tuple = tunable((0.45, 0.40))

    # --- info: structure (env-local frame: rig plan center at the origin) -----------------------
    tunnel_hl: float = info(0.150)       # mouth planes at x = +/- this
    inner_hw: float = info(0.027)        # tunnel interior half-width (y)
    wall_t: float = info(0.010)
    floor_top: float = info(0.100)       # tunnel floor top (= the drop lip height)
    roof_lo: float = info(0.150)         # roof underside (interior height 50 mm)
    roof_top: float = info(0.162)
    slot_hw: float = info(0.015)         # roof slot half-width (fork passes; cube cannot)
    spine_hl: float = info(0.145)        # spine half-length (pocket near face, under the lip)
    pk_x1: float = info(0.265)           # pocket interior far edge
    pk_hw: float = info(0.045)           # pocket interior half-width
    pk_floor_top: float = info(0.030)
    pk_wall_top: float = info(0.090)
    ram_half: float = info(0.055)        # ram body half-length
    ram_hw: float = info(0.021)
    ram_h: float = info(0.046)
    ram_z: float = info(0.124)           # ram body center height (1 mm float above the floor)
    fork_gap_h: float = info(0.0075)     # fork inner half-gap (peg 12 mm + 3 mm backlash)
    fork_t: float = info(0.008)
    fork_hw: float = info(0.012)
    fork_h: float = info(0.198)
    crank_y: float = info(0.053)         # crank axis y (plane of the wheel)
    axis_z: float = info(0.200)          # crank axis height
    peg_r: float = info(0.130)           # drive-peg radius = the ram stroke R
    handle_r: float = info(0.100)        # handle-peg radius (the robot's crank handle)
    peg_half: float = info(0.006)
    cube: float = info(0.042)            # cargo/decoy cube edge
    ram_mass: float = info(0.30)
    crank_mass: float = info(0.50)
    crank_inertia: tuple = info((0.005, 0.006, 0.002))
    slide_lim: float = info(0.135)       # prismatic joint limit (> peg_r)
    finger_t: float = info(0.007)        # Franka fingertip min thickness (mouth must beat it)
    base_pos: tuple = info((0.0, 0.42))  # the documented Franka base xy (TASK.md)
    reach: float = info(0.72)            # documented comfortable arm envelope from base_pos
    contact_offset: float = info(0.0015)
    # rubric weights (sum with success weight = 1.0)
    w_prog: float = info(0.20)
    w_eject: float = info(0.25)
    w_seat: float = info(0.25)
    w_succ: float = info(0.30)

    # Derived (filled in __post_init__).
    cube_rest_z: float = field(default=None, init=False)   # cube center on the tunnel floor
    pocket_rest_z: float = field(default=None, init=False)  # cube center on a pocket floor
    seat_z_lo: float = field(default=None, init=False)
    seat_z_hi: float = field(default=None, init=False)
    cube_off: float = field(default=None, init=False)      # ram face -> cube center (no gap)

    def __post_init__(self) -> None:
        self.cube_rest_z = self.floor_top + self.cube / 2
        self.pocket_rest_z = self.pk_floor_top + self.cube / 2
        self.seat_z_lo = self.pk_floor_top + 0.006
        self.seat_z_hi = self.pk_floor_top + 0.050
        self.cube_off = self.ram_half + self.cube / 2
        # a jaw cannot straddle a cube at the mouth; the cube cannot leave through the slot
        assert (2 * self.inner_hw - self.cube) / 2 < self.finger_t, \
            "mouth side clearance admits a fingertip - the cargo would be graspable"
        assert 2 * self.slot_hw <= self.cube - 0.010, "roof slot admits the cube"
        assert self.roof_lo - self.floor_top >= self.cube + 0.006, "cube scrapes the roof"
        assert self.ram_z + self.ram_h / 2 < self.roof_lo - 0.002, "ram scrapes the roof"
        # full stroke carries the cargo CoM well past the drop lip (slow exits still fall in)
        carry = self.ram_half + self.peg_r + self.cube / 2 - self.tunnel_hl
        assert carry >= 0.045, f"full-stroke CoM carry only {carry:.3f} m"
        assert self.slide_lim > self.peg_r, "prismatic stops shorter than the crank stroke"
        assert self.rev_lim_deg < 175.0, "revolute limit in the 180-deg wrap zone"
        assert self.peg_r * math.sin(math.radians(self.rev_lim_deg)) > self.prog_sat / 0.9, \
            "crank stroke cannot even saturate progress"
        # drive peg never meets the bearing block (x-clear at low z, z-clear near center)
        peg_lo = self.axis_z + self.peg_r * math.cos(math.radians(self.rev_lim_deg)) \
            - self.peg_half
        assert peg_lo > self.roof_top + 0.005, "drive peg dips into the roof"
        assert self.peg_r * math.sin(math.radians(self.rev_lim_deg)) - self.peg_half \
            > 0.025 + 0.005, "drive peg reaches the bearing block laterally"
        s_at_block = (0.025 + self.peg_half + 0.004) / self.peg_r
        z_lo_at_block = self.axis_z + self.peg_r * math.sqrt(1 - s_at_block**2) - self.peg_half
        assert z_lo_at_block > 0.1885 + 0.053 / 2 + 0.004, "drive peg grazes the bearing block"
        # fork backlash small but real; fork spans the peg's whole z travel
        gap = 2 * self.fork_gap_h - 2 * self.peg_half
        assert 0.002 <= gap <= 0.004, "fork backlash out of range"
        fork_lo = self.ram_z + self.ram_h / 2
        fork_hi = fork_lo + self.fork_h
        assert fork_lo < peg_lo - 0.004, "fork plates too short at the bottom"
        assert fork_hi > self.axis_z + self.peg_r + self.peg_half + 0.006, "fork too short on top"
        # reset bands: cubes spawn >= 30 mm inside the mouths, >= 4 mm off the side walls
        x0_max = self.peg_r * math.sin(math.radians(self.theta0_deg))
        far_face = x0_max + self.cube_off + self.gap_hi + self.cube / 2
        assert far_face <= self.tunnel_hl - 0.030, "a cube can spawn too close to its mouth"
        assert self.inner_hw - self.cube / 2 - self.y_jit >= 0.004, "cube spawns on a wall"
        # pocket bands: floor rest inside the seat window, wall-top perch far outside it
        assert self.seat_z_lo + 0.004 < self.pocket_rest_z < self.seat_z_hi - 0.015
        perch_z = self.pk_wall_top + self.cube / 2
        assert perch_z > self.seat_z_hi + 0.02, "wall-top perch inside the seat z band"
        assert self.pk_wall_top < self.floor_top - 0.005, "pocket walls block the ejecta path"
        assert self.pk_x1 - self.tunnel_hl >= 2.2 * self.cube, "pocket too short"
        assert self.eject_z < self.floor_top - 0.004, "eject_z above the tunnel floor"
        assert self.eject_x > self.tunnel_hl, "eject_x inside the tunnel"
        # torque margin: worst resisting load vs the drive clamp (>= 5x)
        mu = 0.40  # worst pair-averaged cube/floor dynamic friction
        f_load = mu * (self.cube_mass * 2 + self.ram_mass * 0.1) * 9.81 + 0.3
        assert self.drive_max > 5.0 * f_load * self.peg_r / 1.0, "drive torque margin thin"
        # continuity gate: one substep of the worst free fall stays far under the gate
        v_fall = math.sqrt(2 * 9.81 * (self.cube_rest_z - self.pocket_rest_z))
        assert 2.0 * v_fall / 120.0 < self.cont_max, "continuity gate rejects the honest drop"
        # the handle circle stays inside the documented arm envelope
        bx, by = self.base_pos
        far = math.sqrt(self.handle_r**2
                        + (abs(by) - (self.crank_y + 0.008))**2
                        + (self.axis_z + self.handle_r)**2)
        assert far < self.reach, f"handle sweep {far:.3f} m outside reach {self.reach} m"
        # weights sum to exactly 1
        s = self.w_prog + self.w_eject + self.w_seat + self.w_succ
        assert abs(s - 1.0) < 1e-9, "rubric weights must sum to 1"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("crank_ejector")
class CrankEjectorScene(BaseScene):
    cfg: CrankEjectorSceneCfg

    def __init__(self, cfg: CrankEjectorSceneCfg | None = None) -> None:
        super().__init__(cfg or CrankEjectorSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic rig, the prismatic ram, the revolute crank
        (joints authored in bind()), and the two free cubes (red cargo, grey decoy)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)

        def cube_cfg(name: str, color, x: float):
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube, c.cube, c.cube),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=32,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.cube_mu[0], dynamic_friction=c.cube_mu[1]),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0.0, c.cube_rest_z + 0.002)),
            )

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sp["rig"](
                    tunnel_hl=c.tunnel_hl, inner_hw=c.inner_hw, wall_t=c.wall_t,
                    floor_top=c.floor_top, roof_lo=c.roof_lo, roof_top=c.roof_top,
                    slot_hw=c.slot_hw, spine_hl=c.spine_hl, pk_x1=c.pk_x1, pk_hw=c.pk_hw,
                    pk_floor_top=c.pk_floor_top, pk_wall_top=c.pk_wall_top,
                    crank_y=c.crank_y, axis_z=c.axis_z, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "ram": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ram",
                spawn=sp["ram"](
                    half_len=c.ram_half, half_w=c.ram_hw, height=c.ram_h,
                    fork_t=c.fork_t, fork_hw=c.fork_hw, fork_h=c.fork_h,
                    fork_gap_h=c.fork_gap_h, mass=c.ram_mass,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.ram_z)),
            ),
            "crank": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Crank",
                spawn=sp["crank"](
                    peg_r=c.peg_r, handle_r=c.handle_r, peg_half=c.peg_half,
                    mass=c.crank_mass, inertia=c.crank_inertia,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, c.crank_y, c.axis_z)),
            ),
            "cargo": cube_cfg("Cargo", (0.85, 0.10, 0.10), c.cube_off + 0.005),
            "decoy": cube_cfg("Decoy", (0.45, 0.45, 0.45), -(c.cube_off + 0.005)),
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
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        self._quat_apply = quat_apply
        self._quat_apply_inv = quat_apply_inverse
        n = env.num_envs
        dev = env.device
        self.rig: RigidObject = env.iscene["rig"]
        self.ram: RigidObject = env.iscene["ram"]
        self.crank: RigidObject = env.iscene["crank"]
        self.cargo: RigidObject = env.iscene["cargo"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        self._dt = 1.0 / 120.0
        # External drive input (solve/smoke write; post_step consumes + owns the wrench
        # slot — never call set_external_force_and_torque on the crank directly).
        self.crank_drive = torch.zeros(n, device=dev)  # torque about the crank axis (+y)
        # Rubric state.
        self.side = torch.ones(n, device=dev)  # +1 = cargo at the east end, -1 = west
        self.prog = torch.zeros(n, device=dev)
        self.ejected = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fouled = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cargo_prev = torch.zeros(n, 3, device=dev)
        self._ram_prev = torch.zeros(n, device=dev)
        self._prev_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._th_prev = torch.zeros(n, device=dev)
        self.crank_w = torch.zeros(n, device=dev)  # FD crank rate (root_ang_vel is phantom)

    def _author_joints(self) -> None:
        """Per env: rig --prismatic X--> ram (stops +/- slide_lim), rig --revolute Y-->
        crank (stops +/- rev_lim_deg). Joint pairs never collide (so the ram slides on
        the joint, not on tunnel friction, and the crank sweeps freely past the rig);
        the crank/ram couple only through the real peg-fork contact."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/ram_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Rig"])
            j.CreateBody1Rel().SetTargets([f"{base}/Ram"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.ram_z))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.slide_lim))
            j.CreateUpperLimitAttr(float(c.slide_lim))
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/crank_axle")
            j.CreateBody0Rel().SetTargets([f"{base}/Rig"])
            j.CreateBody1Rel().SetTargets([f"{base}/Crank"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, float(c.crank_y), float(c.axis_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.rev_lim_deg))
            j.CreateUpperLimitAttr(float(c.rev_lim_deg))

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the cargo SIDE (coin flip), the crank's initial angle
        (the ram follows, x0 = R sin theta0), and each cube's gap + lateral jitter;
        re-pose crank/ram/cubes about the never-moved kinematic rig; zero every latch
        and the drive buffer. Draws are burned first (degenerate-first-draw guard)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        for _ in range(4):
            torch.rand(2 * m + 4, device=dev)  # burn the correlated RNG prefix

        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           -torch.ones(m, device=dev), torch.ones(m, device=dev))
        th0 = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.theta0_deg)
        gap_c = c.gap_lo + torch.rand(m, device=dev) * (c.gap_hi - c.gap_lo)
        gap_d = c.gap_lo + torch.rand(m, device=dev) * (c.gap_hi - c.gap_lo)
        y_c = (torch.rand(m, device=dev) * 2 - 1) * c.y_jit
        y_d = (torch.rand(m, device=dev) * 2 - 1) * c.y_jit
        self.side[env_ids] = side
        x0 = c.peg_r * torch.sin(th0)

        st = torch.zeros(m, 13, device=dev)  # crank: rotated th0 about +y at the axis
        st[:, 0], st[:, 1], st[:, 2] = 0.0, c.crank_y, c.axis_z
        st[:, 0:3] += origin
        st[:, 3] = torch.cos(th0 / 2)
        st[:, 5] = torch.sin(th0 / 2)
        self.crank.write_root_state_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)  # ram follows the peg
        st[:, 0], st[:, 2] = x0, c.ram_z
        st[:, 0:3] += origin
        st[:, 3] = 1.0
        self.ram.write_root_state_to_sim(st, env_ids)

        for body, sgn, gap, yj in ((self.cargo, side, gap_c, y_c),
                                   (self.decoy, -side, gap_d, y_d)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = x0 + sgn * (c.cube_off + gap)
            st[:, 1] = yj
            st[:, 2] = c.cube_rest_z + 0.002
            st[:, 0:3] += origin
            st[:, 3] = 1.0
            body.write_root_state_to_sim(st, env_ids)

        self.prog[env_ids] = 0.0
        self.ejected[env_ids] = False
        self.seated[env_ids] = False
        self.fouled[env_ids] = False
        self._cargo_prev[env_ids] = 0.0
        self._ram_prev[env_ids] = 0.0
        self._prev_valid[env_ids] = False
        self._th_prev[env_ids] = th0
        self.crank_w[env_ids] = 0.0
        self.crank_drive[env_ids] = 0.0

    # ----- readings ------------------------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N, 3) body center in the env-local rig frame (rig is fixed at the origin)."""
        return body.data.root_pos_w - self.env_origins

    def crank_theta(self) -> torch.Tensor:
        """(N,) crank angle (rad): 0 = drive arm straight up, + tilts the peg to +x."""
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        v = self._quat_apply(self.crank.data.root_quat_w, ez)
        return torch.atan2(v[:, 0], v[:, 2])

    def ram_x(self) -> torch.Tensor:
        """(N,) ram displacement along the tunnel (env-local x)."""
        return self._local(self.ram)[:, 0]

    def in_goal_pocket(self) -> torch.Tensor:
        """(N,) bool: cargo currently inside the cargo-side catch pocket (physical,
        live; the z band accepts a floor rest and rejects a wall-top perch)."""
        c = self.cfg
        p = self._local(self.cargo)
        sx = self.side * p[:, 0]
        return ((sx > c.tunnel_hl) & (sx < c.pk_x1 - 0.002) & (p[:, 1].abs() < c.pk_hw - 0.001)
                & (p[:, 2] > c.seat_z_lo) & (p[:, 2] < c.seat_z_hi))

    def decoy_housed(self) -> torch.Tensor:
        """(N,) bool: decoy still inside the tunnel, up on the tunnel floor."""
        c = self.cfg
        p = self._local(self.decoy)
        return (p[:, 0].abs() < c.tunnel_hl - 0.005) & (p[:, 2] > c.eject_z)

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the cargo was cranked out along the latched path and now rests in
        the goal pocket, the decoy is still housed, and the mechanism is quiet."""
        c = self.cfg
        v = self.cargo.data.root_lin_vel_w.norm(dim=-1)
        vr = self.ram.data.root_lin_vel_w[:, 0].abs()
        return (self.seated & ~self.fouled & self.in_goal_pocket() & self.decoy_housed()
                & (v < c.settle_v) & (self.crank_w.abs() < c.settle_w) & (vr < c.settle_ram))

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched mechanism credit + live success. Exactly 1.0
        iff success(); ~0 for the null policy; a fouled (decoy-ejected) episode caps at
        foul_cap; disturbing the mechanism after success falls back to 0.70, never 0."""
        c = self.cfg
        s = (c.w_prog * self.prog + c.w_eject * self.ejected.float()
             + c.w_seat * self.seated.float() + c.w_succ * self.success().float())
        return torch.where(self.fouled, s.clamp(max=c.foul_cap), s)

    # ----- step-coupled mechanics (every substep) -------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Crank plant (clamped drive buffer applied as a body-frame torque about the
        axle), the FD crank rate, then the continuity-gated latches."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        # --- plant: bounded hand torque about the crank axis (+y) ---
        q = self.crank.data.root_quat_w
        tau = torch.zeros(n, 3, device=dev)
        tau[:, 1] = self.crank_drive.clamp(-c.drive_max, c.drive_max)
        tau_body = self._quat_apply_inv(q, tau)  # wrench slot takes BODY-frame torque
        self.crank.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=dev), tau_body.reshape(n, 1, 3))

        # --- FD crank rate (root_ang_vel is phantom under external wrenches) ---
        th = self.crank_theta()
        dth = th - self._th_prev
        dth = torch.where(dth > math.pi, dth - 2 * math.pi, dth)
        dth = torch.where(dth < -math.pi, dth + 2 * math.pi, dth)
        self.crank_w = dth / self._dt
        self._th_prev = th

        # --- latches (continuity-gated; a garbage/teleport frame earns NOTHING) ---
        p_c = self._local(self.cargo)
        p_c = torch.nan_to_num(p_c, nan=1e3, posinf=1e3, neginf=-1e3)
        cw = self.cargo.data.root_pos_w
        rx = self.ram_x()
        jump_c = (cw - self._cargo_prev).norm(dim=-1)
        jump_r = (rx - self._ram_prev).abs()
        cont_c = self._prev_valid & (jump_c < c.cont_max)
        cont_r = self._prev_valid & (jump_r < c.ram_cont)

        prog_now = (self.side * rx / c.prog_sat).clamp(0.0, 1.0)
        prog_now = torch.nan_to_num(prog_now, nan=0.0, posinf=0.0, neginf=0.0)
        self.prog = torch.maximum(
            self.prog, torch.where(cont_r, prog_now, torch.zeros_like(prog_now)))

        sx = self.side * p_c[:, 0]
        out_now = (sx > c.eject_x) & (p_c[:, 2] < c.eject_z)
        self.ejected = self.ejected | (cont_c & (self.prog > c.eject_gate) & out_now)
        self.seated = self.seated | (cont_c & self.ejected & self.in_goal_pocket())
        self.prog = torch.where(self.seated, torch.ones_like(self.prog), self.prog)

        p_d = self._local(self.decoy)
        p_d = torch.nan_to_num(p_d, nan=1e3, posinf=1e3, neginf=-1e3)
        dx = -self.side * p_d[:, 0]
        self.fouled = self.fouled | (dx > c.eject_x) | (p_d[:, 2] < c.eject_z)

        self._cargo_prev = cw.clone()
        self._ram_prev = rx.clone()
        self._prev_valid = torch.ones_like(self._prev_valid)

    # ----- state (full, restorable) --------------------------------------------------------------
    def _bodies(self) -> dict[str, Any]:
        return {"rig": self.rig, "ram": self.ram, "crank": self.crank,
                "cargo": self.cargo, "decoy": self.decoy}

    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone()
                       for nm, b in self._bodies().items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("side", "prog", "ejected", "seated", "fouled",
                               "_cargo_prev", "_ram_prev", "_prev_valid",
                               "_th_prev", "crank_w", "crank_drive")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm, b in self._bodies().items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low roofed magazine tunnel sits on a plinth in front of the robot, its "
            f"floor {c.floor_top * 100:.0f} cm up, with an open mouth at each end and a "
            f"sunk teal catch pocket below each mouth. Deep inside the tunnel a RED "
            f"cargo cube ({c.cube * 1000:.0f} mm) waits at one end and a GREY decoy "
            f"cube at the other (which end is red changes per episode — look through "
            f"the mouths or the narrow roof slot). The mouths leave only "
            f"{(2 * c.inner_hw - c.cube) / 2 * 1000:.0f} mm of side clearance and the "
            f"roof slot is {2 * c.slot_hw * 1000:.0f} mm wide: no jaw can straddle a "
            f"cube and no cube fits through the slot — the cubes cannot be grasped, "
            f"only pushed along the tunnel. Between the cubes rides a blue double-ended "
            f"ram whose tall fork rises through the roof slot and cages the inner peg "
            f"of a side-mounted crank wheel (a scotch yoke). The crank's outer YELLOW "
            f"handle peg, facing the robot on a {c.handle_r * 100:.0f} cm arm about an "
            f"axle {c.axis_z * 100:.0f} cm up, is the intended handle: rotating the "
            f"crank one way slides the ram toward the red end and plows the red cube "
            f"out over the drop lip into its pocket; rotating the other way ejects the "
            f"grey decoy instead — and a cube that falls into a pocket can never be "
            f"put back (nothing can grasp it and the ram cannot pull).\n"
            f"Goal: read which end holds the RED cube, crank the yellow handle in the "
            f"direction that ejects IT, let it drop into its catch pocket, then let go "
            f"and leave the mechanism at rest with the grey decoy still inside the "
            f"tunnel. Ejecting the grey decoy spoils the episode irreversibly."
        )

    def instruction(self) -> str:
        """SHORT imperative form for VLA training."""
        return (
            "Turn the yellow crank handle in the direction that drives the ram toward "
            "the red cube, ejecting it from the tunnel mouth into the sunk catch "
            "pocket at that end, then release the crank and let everything settle. Do "
            "not crank the other way: that irreversibly ejects the grey decoy, which "
            "must stay inside the tunnel. The cubes cannot be grasped — the mouths and "
            "roof slot are too narrow for the gripper."
        )


# ----- runnable env: scene physics only (NullRobot solve/smoke) -> "simgen.crank_ejector" ------
register_env("simgen", lambda: EnvCfg(scene="crank_ejector", robot="null"))
