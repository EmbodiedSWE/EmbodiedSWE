"""WedgeHopperScene — tip a sealed hopper with a wedge jack so the ball inside drains
into the basin (sim_gen task `living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i18`).

Derived from libero_90 living_room_scene4 "pick up the salad dressing and put it in the
tray", but STRATEGICALLY different: the seed is a direct prehensile transport — grasp
the target bottle among distractors, carry it through free space, drop it inside a
tray's containment box. Here the payload is UNTOUCHABLE BY DESIGN: the amber ball sits
inside a covered hopper (roof + deep mouth — no grasp line reaches it), so "pick it up
and put it in" is physically impossible. The ONLY route into the basin is a FORCE
MACHINE: drive the blue WEDGE along a guided runway channel under the hopper's raised
back edge. The inclined plane converts the horizontal push into lift under load
(hopper + ball weight carried through wedge contact), the hopper pitches about its
pivot bar past level, and the ball rolls out of the mouth by gravity, drops over the
cradle onto a deflector and settles in the walled basin. A solver needs a different
PLAN (actuate a mechanism that moves the payload indirectly; never touch the payload)
and different CODE STRUCTURE (rig-frame push axis, pitch monitoring, latched drain
pathway) — not a grasp-carry-release routine.

Mechanics (all contact, no USD joints — corpus convention):
  - The HOPPER is a free rigid compound (floor + walls + roof + pivot bar + feet).
    Its pivot bar rests in a cradle slot (front); two feet rest on the runway (back),
    holding it ~2 degrees BACK-tilted, so the ball parks against the back wall and
    the null policy drains nothing. Stops in the cradle retain the bar while the
    wedge pushes.
  - The WEDGE (6 -> 62 mm over 130 mm, 23.3 degrees) enters a 10 mm designed gap
    under the hopper's back edge (feet create the gap; tip is 6 mm). Wedge-top /
    hopper-floor friction (avg ~0.7) exceeds tan(23.3 deg) = 0.43, so the inserted
    wedge SELF-LOCKS: the tilt survives hands-off. Wedge-bottom / runway friction is
    low (~0.15) so pushing is easy. Guide rails keep the push 1-DOF.
  - The red SHIM is a physically self-rejecting decoy: 9 mm thick < the 10 mm gap —
    it slides under the edge and lifts NOTHING.
  - Drain is IRREVERSIBLE: once the ball is in the basin (55 mm walls, deflector at
    the near side, worst-case roll-out climb 20 mm) nothing brings it back.

success(): the ball settled INSIDE the basin (rig-frame containment, resting height,
velocity gates) AND the drain went THROUGH THE MOUTH (latched pathway: the ball was
observed crossing the mouth window in the hopper's own frame — teleporting the ball
over the roof earns nothing). score() is latched credit that never evaporates:
0.10 wedge ever staged in the channel + 0.15 hopper ever pitched past +1.5 deg +
0.15 ever past +5 deg + 0.15 ball ever through the mouth + 0.20 ball ever inside the
basin (cap 0.75); exactly 1.0 iff success() live. Null policy scores 0.

Fully procedural geometry (boxes + one sphere; no external assets). Per-episode
randomization (readback-verified in smoke): rig yaw +/-30 deg + xy jitter (the push
axis must be READ from the scene), wedge / shim ground scatter with free yaw and
keep-out resampling, ball lateral start along the back wall. Heavy imports (isaaclab,
pxr) are deferred so importing this module stays app-free.
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


# ----- USD authoring helpers (compound spawners) ------------------------------------------------
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


def _box(stage, path: str, size, center, color, contact_offset: float,
         quat=None, material=None, collide: bool = True) -> None:
    """Author one box child prim (translate -> orient -> scale; authored once)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _dynamic_root(stage, root, mass: float, lin_damp: float = 0.05,
                  ang_damp: float = 0.05, iters: int = 32) -> None:
    """Rigid-body armor for a dynamic compound root."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateLinearDampingAttr(float(lin_damp))
    px.CreateAngularDampingAttr(float(ang_damp))
    px.CreateSolverPositionIterationCountAttr(int(iters))
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)


def _qy_t(half_ang: float) -> tuple:
    return (math.cos(half_ang), 0.0, math.sin(half_ang), 0.0)


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC rig. Local frame: origin on the GROUND at the cradle center,
    +x toward the basin (drain direction), -x toward the runway."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(40.0)

    co = cfg.contact_offset
    hi = _friction_material(stage, f"{prim_path}/mat_hi", 0.60, 0.50)
    lo = _friction_material(stage, f"{prim_path}/mat_lo", cfg.mu_runway, cfg.mu_runway * 0.85)
    gray = (0.45, 0.45, 0.48)
    dark = (0.28, 0.28, 0.32)

    # cradle pedestal + bar stops (retain the hopper's pivot bar; stop tops at 0.110
    # stay below the 0.120 exit lip so the draining ball clears them)
    _box(stage, f"{prim_path}/cradle", (0.060, 0.240, cfg.cradle_top),
         (0.0, 0.0, cfg.cradle_top / 2), gray, co, material=hi)
    _box(stage, f"{prim_path}/stop_front", (0.012, 0.240, 0.010),
         (0.014, 0.0, cfg.cradle_top + 0.005), dark, co, material=hi)
    _box(stage, f"{prim_path}/stop_back", (0.012, 0.240, 0.010),
         (-0.014, 0.0, cfg.cradle_top + 0.005), dark, co, material=hi)

    # runway (low-friction top: the wedge slides on it) — extends all the way to the
    # cradle pedestal so the advancing wedge always has support underneath, and the
    # pedestal wall is a natural hard stop. Rails end short of the hopper: rail tops
    # 0.113 would collide with the hopper floor underside.
    _box(stage, f"{prim_path}/runway", (0.640, 0.200, cfg.runway_top),
         (-0.353, 0.0, cfg.runway_top / 2), gray, co, material=lo)
    for s in (-1.0, 1.0):
        _box(stage, f"{prim_path}/rail_{'l' if s < 0 else 'r'}",
             (0.340, 0.012, 0.020), (-0.480, s * 0.040, cfg.runway_top + 0.010),
             dark, co, material=hi)

    # basin: floor, deflector (catches slow exits and sheds them forward), far and
    # side walls; the cradle face is the near containment
    _box(stage, f"{prim_path}/basin_floor", (0.200, 0.196, 0.012),
         (0.120, 0.0, 0.006), (0.62, 0.62, 0.66), co, material=hi)
    th_d = math.atan2(0.090, 0.085)  # deflector: (0.020, 0.110) -> (0.105, 0.020)
    ln_d = math.hypot(0.085, 0.090)
    nx, nz = math.sin(th_d), math.cos(th_d)
    _box(stage, f"{prim_path}/deflector", (ln_d, 0.180, 0.008),
         (0.0625 - nx * 0.004, 0.0, 0.065 - nz * 0.004), gray, co,
         quat=_qy_t(th_d / 2), material=hi)
    _box(stage, f"{prim_path}/basin_far", (0.012, 0.196, 0.055),
         (0.214, 0.0, 0.0275), dark, co, material=hi)
    for s in (-1.0, 1.0):
        _box(stage, f"{prim_path}/basin_{'l' if s < 0 else 'r'}",
             (0.212, 0.012, 0.055), (0.114, s * 0.098, 0.0275), dark, co, material=hi)
    return root


def _spawn_hopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC hopper compound. Local frame: origin at the front edge of the FLOOR
    TOP, mid-width; +x toward the mouth. Floor + 2 side walls + back wall + roof
    (sealed top: the ball cannot be grasped) + pivot bar (front, sits in the cradle)
    + 2 feet (back corners, create the 10 mm wedge-entry gap)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _dynamic_root(stage, root, cfg.mass)

    co = cfg.contact_offset
    hi = _friction_material(stage, f"{prim_path}/mat_hi", 0.60, 0.50)
    body = (0.10, 0.42, 0.24)
    lid = (0.14, 0.50, 0.30)
    L, W = cfg.floor_len, cfg.width

    _box(stage, f"{prim_path}/floor", (L, W, 0.008), (-L / 2, 0.0, -0.004),
         body, co, material=hi)
    _box(stage, f"{prim_path}/back", (0.008, W, 0.070), (-L + 0.004, 0.0, 0.035),
         body, co, material=hi)
    for s in (-1.0, 1.0):
        _box(stage, f"{prim_path}/side_{'l' if s < 0 else 'r'}",
             (L, 0.008, 0.070), (-L / 2, s * (W / 2 - 0.004), 0.035),
             body, co, material=hi)
    _box(stage, f"{prim_path}/roof", (L, W, 0.006), (-L / 2, 0.0, 0.073),
         lid, co, material=hi)
    _box(stage, f"{prim_path}/bar", (0.012, 0.176, 0.012), (-0.006, 0.0, -0.014),
         (0.05, 0.05, 0.06), co, material=hi)
    for s in (-1.0, 1.0):
        _box(stage, f"{prim_path}/foot_{'l' if s < 0 else 'r'}",
             (0.012, 0.012, cfg.feet_h),
             (-L + 0.006, s * (W / 2 - 0.006), -0.008 - cfg.feet_h / 2),
             (0.05, 0.05, 0.06), co, material=hi)
    return root


def _spawn_wedge(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC wedge. Local origin at the TIP, bottom; +x toward the tip (push
    direction). Low-friction base plate (slides on the runway), high-friction ramp
    plate (self-locks under the hopper), back block (the grasp/push feature)."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _dynamic_root(stage, root, cfg.wedge_mass, lin_damp=0.2, ang_damp=0.2)

    co = cfg.contact_offset
    top = _friction_material(stage, f"{prim_path}/mat_top", cfg.mu_wedge_top,
                             cfg.mu_wedge_top * 0.9)
    bot = _friction_material(stage, f"{prim_path}/mat_bot", cfg.mu_runway,
                             cfg.mu_runway * 0.85)
    blue = (0.15, 0.32, 0.85)
    Lw, tip, hmax = cfg.wedge_len, cfg.wedge_tip, cfg.wedge_h

    _box(stage, f"{prim_path}/base", (Lw, cfg.wedge_w, 0.005),
         (-Lw / 2, 0.0, 0.0025), blue, co, material=bot)
    # ramp plate: TOP surface through (0, tip) and (-Lw, hmax)
    ang = math.atan2(hmax - tip, Lw)
    ln = math.hypot(Lw, hmax - tip)
    nx, nz = math.sin(ang), math.cos(ang)
    _box(stage, f"{prim_path}/ramp", (ln, cfg.wedge_w, 0.006),
         (-Lw / 2 - nx * 0.003, 0.0, (tip + hmax) / 2 - nz * 0.003),
         blue, co, quat=_qy_t(ang / 2), material=top)
    _box(stage, f"{prim_path}/block", (0.024, cfg.wedge_w, hmax - 0.012),
         (-Lw + 0.012, 0.0, (hmax - 0.012) / 2 + 0.003), blue, co, material=top)
    return root


def _spawn_shim(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC decoy shim: a flat red plate THINNER than the wedge-entry gap — it
    slides under the hopper's back edge and lifts nothing."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _dynamic_root(stage, root, cfg.shim_mass, lin_damp=0.2, ang_damp=0.2)
    lo = _friction_material(stage, f"{prim_path}/mat", cfg.mu_runway,
                            cfg.mu_runway * 0.85)
    _box(stage, f"{prim_path}/plate", (cfg.wedge_len, cfg.wedge_w, cfg.shim_h),
         (-cfg.wedge_len / 2, 0.0, cfg.shim_h / 2), (0.85, 0.12, 0.12),
         cfg.contact_offset, material=lo)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rig" not in _SPAWNER_CACHE:

        @configclass
        class RigSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rig)
            cradle_top: float = 0.100
            runway_top: float = 0.093
            mu_runway: float = 0.15
            contact_offset: float = 0.0015

        @configclass
        class HopperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hopper)
            floor_len: float = 0.264
            width: float = 0.180
            feet_h: float = 0.010
            mass: float = 0.50
            contact_offset: float = 0.0015

        @configclass
        class WedgeSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_wedge)
            wedge_len: float = 0.130
            wedge_w: float = 0.060
            wedge_tip: float = 0.006
            wedge_h: float = 0.062
            wedge_mass: float = 0.25
            mu_wedge_top: float = 0.80
            mu_runway: float = 0.15
            contact_offset: float = 0.0015

        @configclass
        class ShimSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shim)
            wedge_len: float = 0.130
            wedge_w: float = 0.060
            shim_h: float = 0.009
            shim_mass: float = 0.15
            mu_runway: float = 0.15
            contact_offset: float = 0.0015

        _SPAWNER_CACHE.update(rig=RigSpawnerCfg, hopper=HopperSpawnerCfg,
                              wedge=WedgeSpawnerCfg, shim=ShimSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WedgeHopperSceneCfg(BaseCfg):
    """Config for `WedgeHopperScene`. Honesty asserted in __post_init__: the wedge
    reaches the drain tilt with margin, the shim physically cannot, the inserted
    wedge self-locks, the ball fits the mouth with headroom, everything a solver
    must move is jaw-sized, and the basin catches the whole exit-speed envelope."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(2.5)      # max |ang vel| when judging (rad/s; for the
    #                                       20 mm ball, 2.5 rad/s = 0.05 m/s rolling)
    lift_part_deg: float = tunable(1.5)   # latched partial-lift stage (drain tilt, deg)
    lift_full_deg: float = tunable(5.0)   # latched full-lift stage (deg)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    rig_yaw_deg: float = tunable(30.0)    # rig yaw, +/- deg (push axis must be read)
    rig_jitter: float = tunable(0.04)     # rig xy jitter (+/- m)
    ball_y_span: float = tunable(0.050)   # ball lateral start along the back wall (+/- m)
    spawn_r_min: float = tunable(0.22)    # wedge/shim ground annulus (m, from rig origin)
    spawn_r_max: float = tunable(0.62)
    spawn_sep: float = tunable(0.16)      # wedge-shim ground separation (m)

    # --- info: layout ---------------------------------------------------------------------------
    rig_pos: tuple = info((0.10, 0.0))    # cradle center on the ground (nominal)
    # --- info: rig geometry (rig frame: origin at cradle center, +x toward basin) ---------------
    cradle_top: float = info(0.100)       # cradle floor height (pivot bar rests here)
    runway_top: float = info(0.093)       # runway height (feet + wedge slide here)
    basin_x: tuple = info((0.025, 0.205))  # basin interior window, rig x
    basin_y: float = info(0.088)          # basin interior half-width
    basin_z: tuple = info((0.015, 0.062))  # ball-center height window inside the basin
    wall_h: float = info(0.055)
    # --- info: hopper (local frame: origin at floor-top front edge, +x out the mouth) -----------
    floor_len: float = info(0.264)
    width: float = info(0.180)
    feet_h: float = info(0.010)           # wedge-entry gap under the back edge
    hopper_mass: float = info(0.50)
    mouth_w: float = info(0.164)          # aperture between the side walls
    mouth_h: float = info(0.070)          # floor top to roof underside
    # --- info: wedge / shim ---------------------------------------------------------------------
    wedge_len: float = info(0.130)
    wedge_w: float = info(0.060)
    wedge_tip: float = info(0.006)
    wedge_h: float = info(0.062)
    wedge_mass: float = info(0.25)
    shim_h: float = info(0.009)           # < feet_h: the decoy lifts nothing
    shim_mass: float = info(0.15)
    mu_wedge_top: float = info(0.80)      # avg with hopper 0.60 -> 0.70 > tan(23.3 deg)
    mu_runway: float = info(0.15)
    # --- info: ball -----------------------------------------------------------------------------
    ball_r: float = info(0.020)
    ball_mass: float = info(0.060)
    ball_color: tuple = info((0.95, 0.62, 0.10))
    contact_offset: float = info(0.0015)
    # rubric weights (0.10 + 0.15 + 0.15 + 0.15 + 0.20 = 0.75 = the non-success cap)
    w_staged: float = info(0.10)
    w_part: float = info(0.15)
    w_full: float = info(0.15)
    w_mouth: float = info(0.15)
    w_basin: float = info(0.20)

    def __post_init__(self) -> None:
        span = self.floor_len - 0.012  # bar center to feet center
        pitch0 = math.asin(((self.cradle_top - self.runway_top) + 0.002) / span)
        assert 0.015 < pitch0 < 0.06, "resting back-tilt out of range"
        ang = math.atan2(self.wedge_h - self.wedge_tip, self.wedge_len)
        # the fully inserted wedge reaches well past the full-lift stage
        tilt_max = math.asin((self.wedge_h - self.feet_h) / span) - pitch0
        assert tilt_max > math.radians(self.lift_full_deg + 3.0), "wedge cannot lift enough"
        # the shim cannot even reach level (it fits under the gap and lifts nothing)
        assert self.shim_h < self.feet_h, "shim must slide under without lifting"
        # wedge tip enters the designed gap with clearance
        assert self.wedge_tip + 0.002 <= self.feet_h, "wedge tip does not fit the gap"
        # the inserted wedge SELF-LOCKS (avg-combined friction > tan(wedge angle))
        assert (self.mu_wedge_top + 0.60) / 2 > math.tan(ang) * 1.2, "wedge would squirt out"
        # ball fits the mouth with headroom, and two ball-widths of lateral play
        assert 2 * self.ball_r + 0.010 < self.mouth_h
        assert 4 * self.ball_r < self.mouth_w
        # basin catches the exit envelope: worst-case exit speed -> range inside the
        # interior; worst-case wall climb far below the wall height
        v_max = math.sqrt(2 * 9.81 * math.sin(tilt_max) * (self.floor_len - 0.03) * 5 / 7)
        t_fall = math.sqrt(2 * (self.cradle_top + 0.040 - self.basin_z[0]) / 9.81)
        assert v_max * t_fall < self.basin_x[1] - 0.05, "fast exit overshoots the basin"
        assert v_max * v_max / (2 * 9.81) * 5 / 7 < self.wall_h - 0.02, "ball could climb out"
        # everything the solver must move is jaw-sized (80 mm Franka stroke)
        assert self.wedge_w <= 0.078 and self.wedge_h <= 0.078 and self.shim_h <= 0.078


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("wedge_hopper")
class WedgeHopperScene(BaseScene):
    cfg: WedgeHopperSceneCfg

    def __init__(self, cfg: WedgeHopperSceneCfg | None = None) -> None:
        super().__init__(cfg or WedgeHopperSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        # angular damping 0.3 stands in for rolling resistance: an ideal PhysX
        # sphere on a flat floor would otherwise roll nearly forever in the basin
        ball_props = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.3,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.80, dynamic_friction=0.70, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sp["rig"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    cradle_top=c.cradle_top, runway_top=c.runway_top,
                    mu_runway=c.mu_runway, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.rig_pos[0], c.rig_pos[1], 0.0)),
            ),
            "hopper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hopper",
                spawn=sp["hopper"](
                    floor_len=c.floor_len, width=c.width, feet_h=c.feet_h,
                    mass=c.hopper_mass, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.106, 0.0, 0.122)),
            ),
            "wedge": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wedge",
                spawn=sp["wedge"](
                    wedge_len=c.wedge_len, wedge_w=c.wedge_w, wedge_tip=c.wedge_tip,
                    wedge_h=c.wedge_h, wedge_mass=c.wedge_mass,
                    mu_wedge_top=c.mu_wedge_top, mu_runway=c.mu_runway,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, -0.5, 0.002)),
            ),
            "shim": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shim",
                spawn=sp["shim"](
                    wedge_len=c.wedge_len, wedge_w=c.wedge_w, shim_h=c.shim_h,
                    shim_mass=c.shim_mass, mu_runway=c.mu_runway,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.5, 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    rigid_props=ball_props,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.13, 0.0, 0.145)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.rig: RigidObject = env.iscene["rig"]
        self.hopper: RigidObject = env.iscene["hopper"]
        self.wedge: RigidObject = env.iscene["wedge"]
        self.shim: RigidObject = env.iscene["shim"]
        self.ball: RigidObject = env.iscene["ball"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # latches (partial credit survives transients; success needs the mouth latch)
        self._staged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lift_part = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lift_full = torch.zeros(n, dtype=torch.bool, device=dev)
        self._via_mouth = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ever_basin = torch.zeros(n, dtype=torch.bool, device=dev)

    def _rest_pitch(self) -> float:
        c = self.cfg
        return math.asin(((c.cradle_top - c.runway_top) + 0.002) / (c.floor_len - 0.012))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the rig (yaw + xy jitter), seat the hopper on cradle +
        runway (2 mm drop), park the ball against its back wall (lateral random),
        scatter wedge and shim on the ground (keep-out resampled), clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        q_rig = _qz(psi)
        rp = torch.zeros(m, 3, device=dev)
        rp[:, 0] = c.rig_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        rp[:, 1] = c.rig_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin
        st[:, 3:7] = q_rig
        self.rig.write_root_state_to_sim(st, env_ids)

        # hopper: bar centered in the cradle, resting back-tilt, 2 mm drop
        from isaaclab.utils.math import quat_apply

        pitch0 = self._rest_pitch()
        q_hop = _qmul(q_rig, _qy(torch.full((m,), -pitch0, device=dev)))
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = 0.006
        loc[:, 2] = c.cradle_top + 0.020 + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = rp + origin + quat_apply(q_rig, loc)
        st[:, 3:7] = q_hop
        self.hopper.write_root_state_to_sim(st, env_ids)
        hop_pos, hop_q = st[:, 0:3].clone(), q_hop

        # ball: inside, near the back wall, lateral random
        by = (torch.rand(m, device=dev) * 2 - 1) * c.ball_y_span
        bloc = torch.zeros(m, 3, device=dev)
        bloc[:, 0] = -(c.floor_len - 0.008) + c.ball_r + 0.002
        bloc[:, 1] = by
        bloc[:, 2] = c.ball_r + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = hop_pos + quat_apply(hop_q, bloc)
        st[:, 3] = 1.0
        self.ball.write_root_state_to_sim(st, env_ids)

        # wedge + shim: ground scatter, free yaw, keep-out (annulus, rig rect, sep)
        cpsi, spsi = torch.cos(psi), torch.sin(psi)
        xy = torch.zeros(m, 2, 2, device=dev)
        bad = torch.ones(m, 2, dtype=torch.bool, device=dev)
        for _ in range(40):
            if not bad.any():
                break
            k = int(bad.sum())
            cand = torch.rand(k, 2, device=dev) * (2 * (c.spawn_r_max + 0.02)) \
                - (c.spawn_r_max + 0.02)
            xy[bad] = cand
            r = xy.norm(dim=-1)
            ok = (r > c.spawn_r_min) & (r < c.spawn_r_max)
            rel = xy - rp[:, None, 0:2]
            u = rel[..., 0] * cpsi[:, None] + rel[..., 1] * spsi[:, None]
            v = -rel[..., 0] * spsi[:, None] + rel[..., 1] * cpsi[:, None]
            ok &= ~((u > -0.72) & (u < 0.28) & (v.abs() < 0.18))
            d = (xy[:, 0, :] - xy[:, 1, :]).norm(dim=-1)
            ok &= (d > c.spawn_sep).unsqueeze(-1).expand(m, 2)
            bad = ~ok
        yaw = torch.rand(m, 2, device=dev) * 2 * math.pi
        for i, body in enumerate((self.wedge, self.shim)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = xy[:, i, 0]
            st[:, 1] = xy[:, i, 1]
            st[:, 2] = 0.002
            st[:, 3:7] = _qz(yaw[:, i])
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self._staged[env_ids] = False
        self._lift_part[env_ids] = False
        self._lift_full[env_ids] = False
        self._via_mouth[env_ids] = False
        self._ever_basin[env_ids] = False

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {n: getattr(self, n).data.root_state_w[env_ids].clone()
               for n in ("rig", "hopper", "wedge", "shim", "ball")}
        out["latches"] = torch.stack(
            [self._staged[env_ids], self._lift_part[env_ids], self._lift_full[env_ids],
             self._via_mouth[env_ids], self._ever_basin[env_ids]], dim=1).clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for n in ("rig", "hopper", "wedge", "shim", "ball"):
            getattr(self, n).write_root_state_to_sim(state[n], env_ids)
        lat = state["latches"]
        self._staged[env_ids] = lat[:, 0]
        self._lift_part[env_ids] = lat[:, 1]
        self._lift_full[env_ids] = lat[:, 2]
        self._via_mouth[env_ids] = lat[:, 3]
        self._ever_basin[env_ids] = lat[:, 4]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray machine stands on the floor; its position and heading change every "
            f"episode, so read its geometry from the scene. At its center, a green "
            f"COVERED HOPPER (a lidded box, {c.floor_len * 1000:.0f} x "
            f"{c.width * 1000:.0f} mm) rests tilted slightly BACKWARD: its front face is "
            f"an open MOUTH ({c.mouth_w * 1000:.0f} mm wide, {c.mouth_h * 1000:.0f} mm "
            f"tall), and an AMBER BALL ({2 * c.ball_r * 1000:.0f} mm) sits inside "
            f"against the back wall, visible through the mouth but far beyond any "
            f"gripper's reach — the lid seals the top. The hopper's front edge pivots "
            f"in a cradle; its raised BACK edge rests on two small feet over a low "
            f"RUNWAY that extends behind the machine, leaving a {c.feet_h * 1000:.0f} mm "
            f"slot under the back edge. Guide rails on the runway form a straight "
            f"channel aimed at that slot. In front of the machine, below the mouth, "
            f"lies a walled CATCH BASIN with a sloped gray deflector at its near side.\n"
            f"On the floor nearby lie two loose red-and-blue tools: a BLUE WEDGE "
            f"({c.wedge_len * 1000:.0f} mm long, rising {c.wedge_tip * 1000:.0f} to "
            f"{c.wedge_h * 1000:.0f} mm, {c.wedge_w * 1000:.0f} mm wide) and a flat "
            f"RED SHIM (same footprint, only {c.shim_h * 1000:.0f} mm thick — thinner "
            f"than the slot, so it lifts nothing).\n"
            f"Goal: get the amber ball to rest inside the catch basin. The ball cannot "
            f"be grasped or lifted out — the only way is to TIP THE HOPPER: set the "
            f"blue wedge in the runway channel, thin end toward the hopper, and drive "
            f"it under the raised back edge. The wedge lifts the back, the hopper "
            f"pitches forward past level, and the ball rolls out of the mouth, over "
            f"the cradle and down the deflector into the basin. The wedge self-locks "
            f"once driven home, so the tilt holds when you let go. The red shim "
            f"cannot tip the hopper; leave it. Success: the ball at rest inside the "
            f"basin, having exited through the mouth."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place the blue wedge in the runway channel behind the green hopper, thin "
            "end first, and push it under the hopper's raised back edge until the "
            "hopper tips forward and the amber ball rolls out of the mouth into the "
            "catch basin. The ball must exit through the mouth; the flat red shim "
            "cannot tip the hopper — do not use it."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _rig_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rig.data.root_quat_w,
                                  pos_w - self.rig.data.root_pos_w)

    def _hopper_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.hopper.data.root_quat_w,
                                  pos_w - self.hopper.data.root_pos_w)

    def drain_tilt(self) -> torch.Tensor:
        """(N,) drain tilt in radians: positive = mouth-down (draining), the resting
        pose reads ~-2 deg (back-tilted)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        x_axis = quat_apply(self.hopper.data.root_quat_w, ex)
        return -torch.asin(x_axis[:, 2].clamp(-1.0, 1.0))

    def _settled(self, body) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def ball_in_basin(self) -> torch.Tensor:
        """(N,) bool: ball center inside the basin interior (rig frame), at resting
        height — live geometric containment."""
        c = self.cfg
        loc = self._rig_local(self.ball.data.root_pos_w)
        return (loc[:, 0] > c.basin_x[0]) & (loc[:, 0] < c.basin_x[1]) \
            & (loc[:, 1].abs() < c.basin_y) \
            & (loc[:, 2] > c.basin_z[0]) & (loc[:, 2] < c.basin_z[1])

    def ball_in_hopper(self) -> torch.Tensor:
        """(N,) bool: ball center inside the hopper cavity (hopper frame)."""
        c = self.cfg
        loc = self._hopper_local(self.ball.data.root_pos_w)
        return (loc[:, 0] > -(c.floor_len - 0.008)) & (loc[:, 0] < 0.0) \
            & (loc[:, 1].abs() < c.width / 2 - 0.008) \
            & (loc[:, 2] > 0.0) & (loc[:, 2] < 0.070)

    def _mouth_window(self) -> torch.Tensor:
        """(N,) bool: ball center inside the mouth doorway (hopper frame) — the
        drain pathway a legitimate exit must cross."""
        loc = self._hopper_local(self.ball.data.root_pos_w)
        return (loc[:, 0] > -0.035) & (loc[:, 0] < 0.025) \
            & (loc[:, 1].abs() < 0.085) & (loc[:, 2] > 0.005) & (loc[:, 2] < 0.055)

    def _wedge_staged(self) -> torch.Tensor:
        """(N,) bool: wedge root inside the runway channel (rig frame)."""
        c = self.cfg
        loc = self._rig_local(self.wedge.data.root_pos_w)
        return (loc[:, 0] > -0.66) & (loc[:, 0] < -0.15) \
            & (loc[:, 1].abs() < 0.045) \
            & (loc[:, 2] > c.runway_top - 0.008) & (loc[:, 2] < c.runway_top + 0.07)

    def _update_latches(self) -> None:
        c = self.cfg
        tilt = self.drain_tilt()
        self._staged |= self._wedge_staged()
        self._lift_part |= tilt > math.radians(c.lift_part_deg)
        self._lift_full |= tilt > math.radians(c.lift_full_deg)
        self._via_mouth |= self._mouth_window()
        self._ever_basin |= self.ball_in_basin()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the ball settled inside the basin AND it got there through the
        mouth (latched pathway) — finite. The judged containment is live physics; the
        pathway latch rejects any over-the-roof bypass."""
        self._update_latches()
        finite = torch.isfinite(self.ball.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.hopper.data.root_pos_w).all(dim=-1)
        return self.ball_in_basin() & self._settled(self.ball) & self._via_mouth & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched stage credit (never evaporates) — wedge
        staged 0.10, lift past 1.5 deg 0.15, past 5 deg 0.15, ball through the mouth
        0.15, ball ever in the basin 0.20 (cap 0.75); exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_staged * self._staged.float()
                + c.w_part * self._lift_part.float()
                + c.w_full * self._lift_full.float()
                + c.w_mouth * self._via_mouth.float()
                + c.w_basin * self._ever_basin.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="wedge_hopper", robot="null"))
