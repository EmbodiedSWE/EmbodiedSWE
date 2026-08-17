"""VaultDepositScene — unlock a bolted sliding lid, deposit the red cube in the vault,
slide the lid shut again (sim_gen task `obstacle_i284`).

Derived from pick_place/obstacle, but STRATEGICALLY different: the seed's obstacle is a
free-standing wall between the payload and the goal zone — a detour in an otherwise open
transport (grasp the cube, arc OVER the wall, set it down; success is the cube's resting
pose in an open goal region). Here the goal region is the inside of a VAULT that is sealed
from above: a captive lid rides in rails over the bay, and the lid itself is blocked by a
cross-sliding BOLT whose blade stands in the lid's track. "Go over" does not exist — the
only way in is to RECONFIGURE the obstacle in a fixed order: (1) retract the bolt along its
channel until the blade clears the lid's corridor, (2) slide the lid open along its rails
to expose the bay mouth, (3) drop the RED cube (not the blue decoy) through the mouth,
(4) slide the lid SHUT again. Success reads the mechanism state as well as the payload:
red cube settled inside the bay AND the lid back in its closed window AND the decoy still
outside. The seed's whole plan (carry the payload over the blocker) delivers the cube onto
the CLOSED lid — explicitly rejected in smoke.

Strategy vs the corpus tasks read this session:
  - vs `obstacle_i17` (skittle gallery): i17 is a RELEASED ballistic strike through a
    pre-existing floor tunnel, judged on toppling an untouchable pin plus a perception
    coin-flip. Here nothing is thrown and nothing is judged on momentum: every interaction
    is sustained quasi-static contact (two slides + a drop), the judged payload is handled
    directly, and the blocking structure must be reconfigured (and RESTORED — the re-close
    clause has no i17 analogue).
  - vs `pull_cube_i20` (beam scale): i20's goal is a torque threshold on a body the robot
    never touches, with an episode-dependent stopping rule. Here there is no mass
    threshold and no indirection — the mechanism is driven by direct pushes/pulls, and the
    difficulty is the ORDERED INTERLOCK (bolt gates lid gates deposit), not equilibrium.
  - vs `pen_holder` (exemplar): open cup, multi-object tip-up insertion. Here a single
    deposit is gated by a locked, roofed aperture, plus a wrong-object decoy and the
    mechanism-restore end state.

Execution order is REQUIRED and PHYSICALLY enforced, not rubric-enforced: the lid jams on
the bolt blade after ~10 mm while locked (proved by a force probe in smoke), and no cube
fits the bay while the lid covers it — out-of-order end states are unreachable.

All geometry is procedural (compound spawners; children of one body never self-collide):
  - housing (KINEMATIC): open-top bay (160x160 mm cavity, 130 mm deep walls, rim at
    z=0.140) + a rail track running +x from the bay to an open shelf (support rails, guide
    fences and cap strips make the lid captive: it slides in x, cannot lift or leave), a
    closed-end stop and an open-end stop, and a raised bolt channel crossing the track's
    +y side just beyond the lid's closed position (pedestal + x-walls + y-end stops).
    Rails/fences/caps on the +y side are split around the channel so the blade and knob
    can pass. Slick physics material everywhere except the bay floor (grippy, so the
    delivered cube settles dead).
  - lid (dynamic, 350 g): 200 x 205 x 12 mm slab + a 30 mm handle knob on top; rides the
    rails at z=[0.140,0.152]; closed at x_local=0, fully open needs x_local >= 0.185.
  - bolt (dynamic, 120 g): base slab captive in the channel (slides in y), a blade rising
    through the rail-plane gap into the lid's corridor (locked: blade y=[0.030,0.090],
    inside the lid's swept corridor |y|<=0.108), and a graspable knob cylinder. Retracting
    the bolt by >= 0.082 m (+y, toward the channel's far stop) pulls the blade clear.
  - red cube (45 mm, 60 g): the payload. blue cube (45 mm, 60 g): the decoy — same size,
    different color; depositing it instead (or additionally) fails.

success(): red cube inside the bay window (|x|,|y| <= 0.065, z in [0.018, 0.105] in the
housing frame — honest by construction: any cube physically at rest in the bay counts, a
cube on the rim or outside the shell cannot enter the window) and settled; lid settled in
its closed window (|x_local| <= 0.020, resting at rail height); decoy NOT in the bay.
score(): latched stages the solution passes through — 0.10 * bolt-retraction progress
(latched max) + 0.15 * lid-open progress (latched max) + 0.30 once the red cube has been
inside the bay (latched); 1.0 iff success(). Latched credit never evaporates; the null
policy scores ~0 (bolt at its lock, lid closed, nothing deposited).

Per-episode randomization (readback-verified in smoke): housing xy jitter +/- 40 mm and
yaw = {0 or 180 deg} + [-25, 25] deg (the track/bolt side flips across the scene, so the
solver must read the layout), bolt lock-position jitter, lid closed-position jitter,
red/blue slot SWAP + polar jitter on both floor cubes. Heavy imports (isaaclab, pxr) are
deferred so importing this module stays app-free.
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


# ----- geometry constants (single source of truth: spawners + cfg asserts + rubric) ------------
# Housing frame: origin = bay center on the ground, +x = lid opening direction, +y = bolt side.
_BAY_HALF = 0.080          # bay cavity inner half extent (x and y)
_WALL_T = 0.012            # bay wall thickness
_Z_RIM = 0.140             # bay wall top = rail plane top = lid underside when seated
_LID_X, _LID_Y, _LID_T = 0.200, 0.205, 0.012   # lid slab
_LID_Z0 = _Z_RIM + _LID_T / 2                  # lid origin height when seated (0.146)
_X_TRACK_LO, _X_TRACK_HI = -0.115, 0.327       # rail span in x
_FENCE_IN = 0.108          # guide fence inner face |y| (lid y-play: 0.108 - 0.1025 = 5.5 mm/side)
_GAP_X = (0.098, 0.134)    # +y rail/fence/cap gap for the bolt blade + knob to pass
_LID_OPEN_NEED = 0.190     # lid x_local exposing the full bay mouth (0.190 - 0.100 >= 0.085 + margin)
_LID_X_MAX = 0.215         # open-end stop face 0.315 minus lid half length
_CLOSED_STOP_FACE = -0.102  # lid can reach x_local = -0.002

_CH_X = 0.116              # bolt channel centerline x
_CH_FLOOR_TOP = 0.100      # channel floor top (bolt base rides here)
_BOLT_BASE = (0.016, 0.120, 0.024)   # bolt base slab size
_BOLT_Y0 = 0.060           # bolt origin y when locked (base spans [0.0, 0.120])
_BOLT_STOP_TRAVEL = 0.092  # far-stop face 0.212 minus base +y half 0.060 minus y0
_BLADE_Y_HALF = 0.030      # blade y half (locked: blade spans y [0.030, 0.090])
_BLADE_Z = (0.124, 0.172)  # blade z span (covers the lid slab z [0.140, 0.152])
_BOLT_CLEAR = 0.082        # retraction that pulls the blade past the lid corridor + margin

_CUBE = 0.045              # payload / decoy edge


# ----- custom compound spawners ----------------------------------------------------------------
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
    """One USD physics material. Slick faces are load-bearing: PhysX pair friction is the
    MEAN of both prims, so lid/bolt AND the housing faces they ride must both be slick for
    a ~0.055 sliding mu."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, material=None, contact_offset: float = 0.0015):
    """Author one colliding box child prim (translate -> scale, authored once — the
    duplicate-xformOp trap)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return seg


def _cylinder(stage, path: str, radius: float, height: float, center, color, material=None):
    """Author one colliding z-axis cylinder child prim."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(0.0015)
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")
    return seg


def _spawn_housing(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC vault housing: bay box, rail track (+guides +caps +stops),
    bolt channel. The +y rail/fence/cap run is split around the channel (x in _GAP_X) so
    the bolt blade and knob can travel through the rail plane."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    from pxr import UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)

    slick = _friction_material(stage, f"{prim_path}/slick_mat", cfg.mu_slick_s, cfg.mu_slick_d)
    grip = _friction_material(stage, f"{prim_path}/grip_mat", cfg.mu_grip_s, cfg.mu_grip_d)
    body_c = (0.42, 0.45, 0.52)
    rail_c = (0.30, 0.32, 0.38)
    chan_c = (0.35, 0.30, 0.22)

    # --- base plate (looks + a hard floor under the track) ---
    _box(stage, f"{prim_path}/base", (0.454, 0.240, 0.008), (0.100, 0.0, 0.004), rail_c,
         material=slick)
    # --- bay: floor (grippy) + 4 walls to the rim ---
    _box(stage, f"{prim_path}/bay_floor", (0.184, 0.184, 0.010), (0.0, 0.0, 0.005), body_c,
         material=grip)
    for tag, sx in (("xp", 1.0), ("xn", -1.0)):
        _box(stage, f"{prim_path}/bay_w{tag}", (_WALL_T, 0.184, 0.130),
             (sx * 0.086, 0.0, 0.075), body_c, material=slick)
    for tag, sy in (("yp", 1.0), ("yn", -1.0)):
        _box(stage, f"{prim_path}/bay_w{tag}", (0.160, _WALL_T, 0.130),
             (0.0, sy * 0.086, 0.075), body_c, material=slick)

    # --- rail track: support beams (top = _Z_RIM), guide fences, cap strips ---
    x_lo, x_hi = _X_TRACK_LO, _X_TRACK_HI
    g0, g1 = _GAP_X

    def strip(tag: str, y_c: float, y_s: float, z_c: float, z_s: float, col, xr) -> None:
        for k, (a, b) in enumerate(xr):
            _box(stage, f"{prim_path}/{tag}_{k}", (b - a, y_s, z_s),
                 ((a + b) / 2, y_c, z_c), col, material=slick)

    full = [(x_lo, x_hi)]
    split = [(x_lo, g0), (g1, x_hi)]
    # support rails: y +/-[0.086, 0.110], z [0.126, 0.140]
    strip("rail_n", -0.098, 0.024, 0.133, 0.014, rail_c, full)
    strip("rail_p", 0.098, 0.024, 0.133, 0.014, rail_c, split)
    # guide fences: y +/-[0.108, 0.120], z [0.140, 0.178]
    strip("fence_n", -0.114, 0.012, 0.159, 0.038, rail_c, full)
    strip("fence_p", 0.114, 0.012, 0.159, 0.038, rail_c, split)
    # cap strips (hold the lid down): y +/-[0.092, 0.120], z [0.157, 0.165]
    strip("cap_n", -0.106, 0.028, 0.161, 0.008, rail_c, full)
    strip("cap_p", 0.106, 0.028, 0.161, 0.008, rail_c, split)
    # end stops: closed (face at -0.102) and open (face at 0.315)
    _box(stage, f"{prim_path}/stop_closed", (0.012, 0.240, 0.038), (-0.108, 0.0, 0.159),
         rail_c, material=slick)
    _box(stage, f"{prim_path}/stop_open", (0.012, 0.240, 0.038), (0.321, 0.0, 0.159),
         rail_c, material=slick)
    # cosmetic pillars under the far rail ends
    for tag, sy in (("p", 0.098), ("n", -0.098)):
        _box(stage, f"{prim_path}/pillar_{tag}", (0.030, 0.024, 0.118),
             (0.300, sy, 0.067), rail_c, material=slick)

    # --- bolt channel (+y side, crossing the track at x = _CH_X) ---
    # pedestal: solid block, top = channel floor the bolt base rides on
    _box(stage, f"{prim_path}/ch_pedestal", (0.044, 0.280, _CH_FLOOR_TOP),
         (_CH_X, 0.120, _CH_FLOOR_TOP / 2), chan_c, material=slick)
    # channel x-walls: inner faces at x 0.098 / 0.134 (the bolt's base FLANGES ride between
    # them with 2 mm play/side)
    _box(stage, f"{prim_path}/ch_wxn", (0.008, 0.280, 0.030), (0.094, 0.120, 0.115),
         chan_c, material=slick)
    _box(stage, f"{prim_path}/ch_wxp", (0.008, 0.280, 0.030), (0.138, 0.120, 0.115),
         chan_c, material=slick)
    # keeper lips: overhang strips on each wall's inner top, z [0.113, 0.130] — they trap
    # the bolt's base flanges (top 0.110) so a hard press on the blade CANNOT tip or fold
    # the bolt out of its channel (the interlock is force-proof, not just force-resistant).
    # The blade (x [0.110, 0.122]) and the raised knob (z >= 0.134) pass between/above them.
    _box(stage, f"{prim_path}/ch_lipn", (0.008, 0.280, 0.017), (0.102, 0.120, 0.1215),
         chan_c, material=slick)
    _box(stage, f"{prim_path}/ch_lipp", (0.008, 0.280, 0.017), (0.130, 0.120, 0.1215),
         chan_c, material=slick)
    # channel y end stops: lock-side face at y=-0.004, far face at y=0.212.
    # The lock-side stop crosses the lid corridor (y ~ -0.010), so it must stay BELOW the
    # rail plane (z <= 0.130 < 0.140): it only ever catches the bolt BASE (z [0.100, 0.124]).
    _box(stage, f"{prim_path}/ch_stop_lock", (0.040, 0.012, 0.030), (_CH_X, -0.010, 0.115),
         chan_c, material=slick)
    _box(stage, f"{prim_path}/ch_stop_far", (0.040, 0.012, 0.080), (_CH_X, 0.218, 0.140),
         chan_c, material=slick)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dynamic lid: slab + handle knob. Origin = slab center."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    slick = _friction_material(stage, f"{prim_path}/lid_mat", cfg.mu_slick_s, cfg.mu_slick_d)
    _box(stage, f"{prim_path}/slab", (_LID_X, _LID_Y, _LID_T), (0.0, 0.0, 0.0),
         (0.78, 0.78, 0.82), material=slick)
    _box(stage, f"{prim_path}/handle", (0.030, 0.030, 0.030), (0.0, 0.0, 0.021),
         (0.15, 0.15, 0.18), material=slick)
    return root


def _spawn_bolt(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the dynamic bolt: base slab (captive in the channel) + blade (rises through
    the rail-plane gap into the lid corridor) + knob cylinder. Origin = base center."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.2)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    slick = _friction_material(stage, f"{prim_path}/bolt_mat", cfg.mu_slick_s, cfg.mu_slick_d)
    brass = (0.80, 0.62, 0.18)
    _box(stage, f"{prim_path}/bolt_base", _BOLT_BASE, (0.0, 0.0, 0.0), brass,
         material=slick, contact_offset=0.001)
    # base flanges: bottom strips protruding +/-x, captive under the channel's keeper lips
    # (abs z [0.100, 0.110] when the base rides the floor -> 3 mm below the lips): the bolt
    # can slide along its channel but cannot be tipped, folded or lifted out.
    for tag, sx in (("fp", 1.0), ("fn", -1.0)):
        _box(stage, f"{prim_path}/flange_{tag}", (0.008, _BOLT_BASE[1], 0.010),
             (sx * 0.012, 0.0, -0.007), brass, material=slick, contact_offset=0.001)
    # blade: rel z center 0.036 -> abs [0.124, 0.172] when the base rides the channel floor
    _box(stage, f"{prim_path}/blade", (0.012, 2 * _BLADE_Y_HALF, _BLADE_Z[1] - _BLADE_Z[0]),
         (0.0, 0.0, (_BLADE_Z[0] + _BLADE_Z[1]) / 2 - (_CH_FLOOR_TOP + _BOLT_BASE[2] / 2)),
         brass, material=slick, contact_offset=0.001)
    # knob: raised so its underside (abs z 0.134) clears the keeper lips (top 0.130)
    _cylinder(stage, f"{prim_path}/knob", 0.011, 0.056, (0.0, 0.045, 0.050), brass,
              material=slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily, so imports stay app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "housing" not in _SPAWNER_CACHE:

        @configclass
        class HousingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_housing)
            mu_slick_s: float = 0.06
            mu_slick_d: float = 0.05
            mu_grip_s: float = 0.60
            mu_grip_d: float = 0.50

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            mu_slick_s: float = 0.06
            mu_slick_d: float = 0.05

        @configclass
        class BoltSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bolt)
            mu_slick_s: float = 0.06
            mu_slick_d: float = 0.05

        _SPAWNER_CACHE.update(housing=HousingSpawnerCfg, lid=LidSpawnerCfg, bolt=BoltSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class VaultDepositSceneCfg(BaseCfg):
    """Config for `VaultDepositScene`. `__post_init__` asserts the geometry that makes the
    task honest: the locked blade genuinely blocks the lid corridor, the retracted blade
    genuinely clears it (with the lid at its worst-case guide-play shift), the open lid
    genuinely exposes the whole mouth, and the containment window is unreachable from
    outside the bay."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    bay_xy_tol: float = tunable(0.065)   # cube center |x|,|y| bound, housing frame
    bay_z_win: tuple = tunable((0.018, 0.105))  # cube center z window (rejects rim/lid rests)
    lid_closed_tol: float = tunable(0.020)  # lid |x_local| bound for "closed" (still covers)
    lid_z_win: tuple = tunable((0.138, 0.158))  # lid center z window (seated on the rails)
    cube_settle_lin: float = tunable(0.08)  # judge gates (above the phantom-velocity band)
    cube_settle_ang: float = tunable(1.0)
    lid_settle_lin: float = tunable(0.06)
    lid_settle_ang: float = tunable(0.8)

    # --- tunable: score weights (latched stages of the demonstrated solution) ------------------
    w_bolt: float = tunable(0.10)   # x retraction progress (latched max)
    w_lid: float = tunable(0.15)    # x lid-open progress (latched max)
    w_deposit: float = tunable(0.30)  # red cube has been inside the bay (latched)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    house_jitter: float = tunable(0.04)   # uniform +/- xy jitter of the housing (m)
    house_yaw_deg: float = tunable(25.0)  # uniform +/- yaw jitter on top of the 0/180 flip
    bolt_y_jitter: float = tunable(0.004)  # lock-position jitter (m)
    lid_x_jitter: float = tunable(0.006)   # closed-position jitter (m)
    slot_radius: float = tunable(0.30)     # cube/decoy polar slots around the housing (m)
    slot_radius_jitter: float = tunable(0.03)
    slot_angles: tuple = tunable((150.0, 210.0))  # housing-local slot angles (deg, -x side)
    slot_angle_jitter: float = tunable(12.0)

    # --- info: structure ------------------------------------------------------------------------
    cube_size: float = info(_CUBE)   # graspable (< ~80 mm Franka jaw)
    cube_mass: float = info(0.06)
    lid_mass: float = info(0.35)
    bolt_mass: float = info(0.12)
    bolt_clear: float = info(_BOLT_CLEAR)      # retraction that frees the lid
    bolt_stop_travel: float = info(_BOLT_STOP_TRAVEL)
    lid_open_need: float = info(_LID_OPEN_NEED)
    mu_cube_s: float = info(0.50)
    mu_cube_d: float = info(0.40)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)

    # Derived (filled in __post_init__).
    lid_y_play: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.lid_y_play = _FENCE_IN - _LID_Y / 2  # per-side lid drift inside the guides
        assert 0.002 < self.lid_y_play < 0.012, "lid guide play out of range"
        # -- the locked blade blocks the lid corridor (blade inside |y| <= _FENCE_IN) --
        assert _BOLT_Y0 - _BLADE_Y_HALF >= -_FENCE_IN + 0.01
        assert _BOLT_Y0 + _BLADE_Y_HALF <= _FENCE_IN - 0.01, \
            "locked blade must stand fully inside the lid corridor"
        # -- the blade covers the lid slab's z span (the lid cannot ride over/under it) --
        assert _BLADE_Z[0] <= _Z_RIM - 0.005 and _BLADE_Z[1] >= _Z_RIM + _LID_T + 0.005
        # -- retracted blade clears the corridor even with the lid shifted to +y play --
        worst_lid_edge = _LID_Y / 2 + self.lid_y_play
        assert _BOLT_Y0 - _BLADE_Y_HALF + self.bolt_clear >= worst_lid_edge + 0.003, \
            "bolt_clear must pull the blade past the lid's worst-case swept corridor"
        assert self.bolt_stop_travel >= self.bolt_clear + 0.008, \
            "the channel must allow retraction beyond bolt_clear with margin"
        # -- blade and knob pass through the +y rail/fence/cap gap with margin --
        assert _GAP_X[0] <= _CH_X - 0.006 - 0.006 and _GAP_X[1] >= _CH_X + 0.006 + 0.006
        assert _GAP_X[0] <= _CH_X - 0.011 - 0.004 and _GAP_X[1] >= _CH_X + 0.011 + 0.004
        # -- keeper cage: flanges (x [0.100,0.108]/[0.124,0.132], top z 0.110) trapped under
        #    the lips (x [0.098,0.106]/[0.126,0.134], z [0.113,0.130]); blade and knob clear
        #    the lips; the knob's underside clears the lips' top. A press on the blade can
        #    tip the bolt at most ~3 mm before the flanges catch — it can never fold out. --
        assert 0.106 - 0.100 >= 0.004 and 0.132 - 0.126 >= 0.004, "flange/lip overlap"
        assert 0.113 - 0.110 >= 0.002, "flange-to-lip vertical play"
        assert 0.110 - 0.106 >= 0.003 and 0.126 - 0.122 >= 0.003, "blade must clear the lips"
        assert 0.134 >= 0.130 + 0.004, "knob underside must clear the lip tops"
        # -- fully open lid exposes the whole mouth; the stop allows it --
        assert self.lid_open_need - _LID_X / 2 >= _BAY_HALF + 0.005, \
            "lid at lid_open_need must clear the bay mouth entirely"
        assert _LID_X_MAX >= self.lid_open_need + 0.005
        # -- closed window still covers the mouth --
        assert _LID_X / 2 - self.lid_closed_tol >= _BAY_HALF, \
            "a lid anywhere in the closed window must fully cover the bay"
        # -- containment window honest by construction --
        in_max = _BAY_HALF - self.cube_size / 2          # deepest possible in-bay |center|
        out_min = _BAY_HALF + _WALL_T + self.cube_size / 2  # nearest outside |center|
        assert in_max < self.bay_xy_tol < out_min - 0.02, \
            "bay_xy_tol must accept every physical in-bay rest and reject wall-outside rests"
        floor_rest = 0.010 + self.cube_size / 2
        rim_rest = _Z_RIM + self.cube_size / 2
        assert self.bay_z_win[0] < floor_rest < self.bay_z_win[1] < rim_rest - 0.02, \
            "bay z window must accept floor rests and reject rim/lid rests"
        # -- embodiment: everything the arm must move fits the jaw --
        assert self.cube_size < 0.08 and 2 * 0.011 < 0.08 and 0.030 < 0.08
        # -- score: latched base credit stays below success --
        assert self.w_bolt + self.w_lid + self.w_deposit <= 0.65


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("vault_deposit")
class VaultDepositScene(BaseScene):
    cfg: VaultDepositSceneCfg

    def __init__(self, cfg: VaultDepositSceneCfg | None = None) -> None:
        super().__init__(cfg or VaultDepositSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        cube_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_cube_s, dynamic_friction=c.mu_cube_d, restitution=0.0)
        cube_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8, solver_velocity_iteration_count=4,
            max_depenetration_velocity=0.5, sleep_threshold=0.0, stabilization_threshold=0.0,
            linear_damping=0.05, angular_damping=0.05)
        cube_coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "housing": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Housing",
                spawn=sp["housing"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=sp["lid"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.lid_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _LID_Z0 + 0.002)),
            ),
            "bolt": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bolt",
                spawn=sp["bolt"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bolt_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg()),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(_CH_X, _BOLT_Y0, _CH_FLOOR_TOP + _BOLT_BASE[2] / 2 + 0.002)),
            ),
        }
        for nm, col, x0 in (("cube_red", (0.85, 0.10, 0.10), -0.26),
                            ("cube_blue", (0.10, 0.25, 0.85), -0.26)):
            out[nm] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + nm.title().replace("_", ""),
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_size,) * 3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=col),
                    physics_material=cube_mat, rigid_props=cube_rigid,
                    collision_props=cube_coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(x0, 0.15 if nm == "cube_red" else -0.15,
                         c.cube_size / 2 + 0.003)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.housing: RigidObject = env.iscene["housing"]
        self.lid: RigidObject = env.iscene["lid"]
        self.bolt: RigidObject = env.iscene["bolt"]
        self.cube: RigidObject = env.iscene["cube_red"]
        self.decoy: RigidObject = env.iscene["cube_blue"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # custom spawners author mass explicitly — assert it took (the density-mass trap)
        for body, want in ((self.lid, c.lid_mass), (self.bolt, c.bolt_mass)):
            got = float(body.root_physx_view.get_masses()[0])
            assert abs(got - want) < 0.02, f"authored mass mismatch: {got} vs {want}"
        # score latches
        self._bolt_prog = torch.zeros(n, device=dev)   # latched max retraction fraction
        self._lid_prog = torch.zeros(n, device=dev)    # latched max open fraction
        self._deposited = torch.zeros(n, dtype=torch.bool, device=dev)
        # per-episode references (world-invariant, housing frame)
        self._bolt_ref = torch.full((n,), _BOLT_Y0, device=dev)  # sampled lock y
        self._lid_ref = torch.zeros(n, device=dev)               # sampled closed x

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the housing (kinematic teleport) with xy jitter and a
        0/180-flip + jitter yaw, seat the lid closed and the bolt locked (jittered) in the
        HOUSING frame, scatter payload + decoy on jittered polar slots (random swap), zero
        the score latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- housing: xy jitter + (0 | 180) flip + yaw jitter ---
        hxy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.house_jitter
        flip = (torch.rand(m, device=dev) < 0.5).float() * math.pi
        yaw = flip + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.house_yaw_deg)
        half = yaw / 2
        zeros = torch.zeros(m, device=dev)
        q_yaw = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_world(local: torch.Tensor) -> torch.Tensor:
            wx = hxy[:, 0] + local[:, 0] * cy - local[:, 1] * sy
            wy = hxy[:, 1] + local[:, 0] * sy + local[:, 1] * cy
            return torch.stack([wx, wy, local[:, 2]], dim=-1)

        write(self.housing, torch.cat([hxy, zeros.unsqueeze(-1)], dim=-1), q_yaw)

        # --- lid: closed (jittered), seated on the rails ---
        lid_x = (torch.rand(m, device=dev) * 2 - 1) * c.lid_x_jitter
        self._lid_ref[env_ids] = lid_x
        write(self.lid, to_world(torch.stack(
            [lid_x, zeros, torch.full((m,), _LID_Z0 + 0.001, device=dev)], dim=-1)), q_yaw)

        # --- bolt: locked (jittered), base on the channel floor ---
        bolt_y = _BOLT_Y0 + (torch.rand(m, device=dev) * 2 - 1) * c.bolt_y_jitter
        self._bolt_ref[env_ids] = bolt_y
        write(self.bolt, to_world(torch.stack(
            [torch.full((m,), _CH_X, device=dev), bolt_y,
             torch.full((m,), _CH_FLOOR_TOP + _BOLT_BASE[2] / 2 + 0.001, device=dev)],
            dim=-1)), q_yaw)

        # --- payload + decoy: jittered polar slots, random SWAP, free yaw ---
        swap = torch.rand(m, device=dev) < 0.5
        for i, body in enumerate((self.cube, self.decoy)):
            k = torch.where(swap, torch.tensor(1 - i, device=dev), torch.tensor(i, device=dev))
            base_ang = torch.tensor([math.radians(a) for a in c.slot_angles], device=dev)[k]
            ang = base_ang + (torch.rand(m, device=dev) * 2 - 1) * math.radians(
                c.slot_angle_jitter)
            rad = c.slot_radius + (torch.rand(m, device=dev) * 2 - 1) * c.slot_radius_jitter
            local = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang),
                                 torch.full((m,), c.cube_size / 2 + 0.003, device=dev)],
                                dim=-1)
            cyawq = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
            chalf = cyawq / 2
            cq = torch.stack([torch.cos(chalf), zeros, zeros, torch.sin(chalf)], dim=-1)
            write(body, to_world(local), cq)

        # --- zero the latches ---
        self._bolt_prog[env_ids] = 0.0
        self._lid_prog[env_ids] = 0.0
        self._deposited[env_ids] = False

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "housing": self.housing.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "bolt": self.bolt.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "bolt_prog": self._bolt_prog[env_ids].clone(),
            "lid_prog": self._lid_prog[env_ids].clone(),
            "deposited": self._deposited[env_ids].clone(),
            "bolt_ref": self._bolt_ref[env_ids].clone(),
            "lid_ref": self._lid_ref[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.housing.write_root_state_to_sim(state["housing"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.bolt.write_root_state_to_sim(state["bolt"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self._bolt_prog[env_ids] = state["bolt_prog"]
        self._lid_prog[env_ids] = state["lid_prog"]
        self._deposited[env_ids] = state["deposited"]
        self._bolt_ref[env_ids] = state["bolt_ref"]
        self._lid_ref[env_ids] = state["lid_ref"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A vault stands on the floor: an open-top steel bay (16 x 16 cm cavity, 13 cm "
            "deep) sealed by a flat sliding LID that rides in rails. The rails run from "
            "the bay out over an open shelf on one side (call that the OPEN direction — "
            "read it from the track), and the lid is captive: it can only slide along the "
            "rails, between a closed stop (lid centered over the bay) and an open stop "
            "(bay mouth fully exposed). The lid carries a small dark handle block on top. "
            "Just past the lid's closed edge, a raised brass BOLT crosses the track: its "
            "blade stands up through a gap in the rails, directly in the lid's path, so "
            "the lid jams on it after about a centimeter — the vault is LOCKED. The bolt "
            "slides in its own channel, perpendicular to the lid's travel, and carries a "
            "round brass knob; pulling it along the channel AWAY from the track (toward "
            "the channel's far stop, about 9 cm of travel) retracts the blade clear of "
            "the lid. On the floor on the far side of the vault lie two cubes "
            f"({c.cube_size * 100:.1f} cm, both light and jaw-sized): one RED, one BLUE. "
            "Which slot each color occupies, the vault's position and heading (the track "
            "can point either way), the exact lock position and the cube spots change "
            "every episode — read the scene by looking.\n"
            "Goal, in the only order the mechanism admits: pull the brass knob to retract "
            "the bolt until its blade is clear of the lid's track; slide the lid along "
            "its rails to the open side until the bay mouth is exposed; put the RED cube "
            "into the bay (drop it through the mouth; it must come to rest on the bay "
            "floor, inside the walls); then slide the lid back until it fully covers the "
            "bay again. Finish with the red cube settled inside the closed vault. The "
            "BLUE cube is a decoy: it must stay OUT of the bay — depositing it (instead "
            "or additionally) fails. A cube left on the rim, on the lid, or beside the "
            "vault counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Unlock the vault: pull the brass bolt knob until the blade clears the lid's "
            "track, slide the lid open along its rails, drop the RED cube (never the "
            "blue one) into the bay, then slide the lid shut so it covers the bay again."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N, 3) body root position in the housing's frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.housing.data.root_quat_w,
                                  body.data.root_pos_w - self.housing.data.root_pos_w)

    def bolt_travel(self) -> torch.Tensor:
        """(N,) bolt retraction from its sampled lock position (housing-frame +y)."""
        return (self._local(self.bolt)[:, 1] - self._bolt_ref).clamp(min=0.0)

    def lid_open(self) -> torch.Tensor:
        """(N,) lid displacement from its sampled closed position (housing-frame +x)."""
        return (self._local(self.lid)[:, 0] - self._lid_ref).clamp(min=0.0)

    def in_bay(self, body) -> torch.Tensor:
        """(N,) bool: body center inside the bay containment window (housing frame)."""
        c = self.cfg
        loc = self._local(body)
        return (loc[:, 0].abs() <= c.bay_xy_tol) & (loc[:, 1].abs() <= c.bay_xy_tol) \
            & (loc[:, 2] >= c.bay_z_win[0]) & (loc[:, 2] <= c.bay_z_win[1])

    def lid_closed(self) -> torch.Tensor:
        """(N,) bool: lid seated on the rails inside its closed window (covers the bay)."""
        c = self.cfg
        loc = self._local(self.lid)
        return (loc[:, 0].abs() <= c.lid_closed_tol) \
            & (loc[:, 2] >= c.lid_z_win[0]) & (loc[:, 2] <= c.lid_z_win[1])

    def settled(self, body, lin: float, ang: float) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < ang)

    # ----- step-coupled latches ------------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch the demonstrated solution's stages every physics substep: max bolt
        retraction fraction, max lid-open fraction, and red-cube-was-in-the-bay."""
        c = self.cfg
        bp = (self.bolt_travel() / c.bolt_clear).clamp(0.0, 1.0)
        lp = (self.lid_open() / c.lid_open_need).clamp(0.0, 1.0)
        self._bolt_prog = torch.maximum(self._bolt_prog, bp)
        self._lid_prog = torch.maximum(self._lid_prog, lp)
        self._deposited |= self.in_bay(self.cube)

    # ----- rubric ----------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: red cube settled inside the bay, lid settled in its closed window,
        decoy NOT in the bay."""
        c = self.cfg
        return self.in_bay(self.cube) \
            & self.settled(self.cube, c.cube_settle_lin, c.cube_settle_ang) \
            & self.lid_closed() \
            & self.settled(self.lid, c.lid_settle_lin, c.lid_settle_ang) \
            & ~self.in_bay(self.decoy)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: latched stage credit (bolt retraction + lid opening +
        deposit), 1.0 iff success(). Latches only ever grow, so credit earned by correct
        behavior never evaporates; the null policy scores ~0."""
        c = self.cfg
        base = (c.w_bolt * self._bolt_prog + c.w_lid * self._lid_prog
                + c.w_deposit * self._deposited.float())
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="vault_deposit", robot="null", env_spacing=3.0))
