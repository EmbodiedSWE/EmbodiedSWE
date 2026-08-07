"""CanBalanceScene — counterweigh a preloaded knife-edge beam balance with drink cans
(sim_gen task `coke_task_i15`).

Derived from simpler_env/coke_task, but STRATEGICALLY different: the seed is a
GRASP-AND-LIFT — one opened coke can on a table, success = a firm grasp plus a few
centimetres of height, evaluated by a grasp checker; the can is the goal and the episode
ends holding it in the air. Here no lift is ever judged and no single object is the
goal: the scene is a two-pan KNIFE-EDGE BEAM BALANCE whose loaded pan was preloaded (at
reset) with a random subset of black steel plates, so the beam rests TILTED against its
stop. The task is to SELECT, BY MASS, the unique subset of three drink cans (100 g /
200 g / 400 g — binary weights, every subset sum distinct) whose total equals the
preload, and place them on the empty pan so that GRAVITY ITSELF rotates the beam back
level about its knife edge. The judged outcome is the MECHANISM'S EQUILIBRIUM — a
physical null measurement: the rubric never sums masses, it reads the settled beam
angle, and only the correct subset can produce it (a 100 g error tilts the beam to
>= 13 deg, far outside the 4 deg gate; the statics are asserted in __post_init__).

A solver therefore needs a different PLAN each episode (read WHICH plates sit on the
loaded pan, infer the target mass, pick the matching can subset — or work closed-loop,
adding cans and watching the beam answer — while leaving the spare plates alone) and a
different CODE STRUCTURE (beam-frame pan predicates, a per-episode subset target, an
analog tilt readout) — not a grasp checker and a height threshold.

success(): beam seated on its knife edge (axle at the cradle, yaw aligned), settled and
LEVEL within `level_tol_deg`; every preload plate still resting on the loaded pan;
every spare plate clear of both pans; every can either resting fully on the empty pan
or clear of both pans; at least one can on the empty pan; everything finite. All
clauses are live physical outcomes — the equilibrium is maintained by contact and
gravity through the bearing alone (mass correctness is implied physically, never
bookkept).

score() is latched (credit never evaporates): 0.10 * a can ever resting on the empty
pan + 0.15 * each REQUIRED can (the subset matching the preload) ever resting there
+ 0.10 * the beam ever near-level (within 2x tolerance, preload intact, >= 1 can
loaded) — cap 0.65; exactly 1.0 iff success() live. Doing nothing scores 0 (the beam
starts hard against its stop).

Assets are fully procedural (no external files):
  - stand (KINEMATIC compound): ground plate, two posts carrying V-notch cradles (the
    knife edge, low-friction material — the bearing must not mask the measurement),
    and a stop platform under each pan that limits tilt to ~10.6 deg (cans stay far
    from their 22 deg topple angle) and leaves a 44 mm slot under a level pan — less
    than any can dimension, so the beam cannot be propped level from below.
  - beam (DYNAMIC compound, 0.8 kg): axle (r 5 mm, along the pivot axis) with end
    caps, a drop post, the arm, and two rimmed dish pans whose floors hang 58 mm
    BELOW the pivot; mass / CoM (50 mm below the axle) / inertia authored explicitly
    — the pendulum restoring torque IS the instrument.
  - plates (x3, dynamic): black steel squares 60 x 60 mm, thickness 8/16/32 mm for
    100/200/400 g. The sampled preload subset spawns stacked on the loaded pan; the
    complement spawns on the floor as decoys that must stay off the scale.
  - cans (x3, dynamic): upright cylinders d 56 mm (< the 80 mm Franka jaw): RED short
    62 mm = 100 g, BLUE mid 95 mm = 200 g, GREEN tall 135 mm = 400 g.

Per-episode randomization (readback-verified in smoke): the preload subset (7 choices
— the target mass and hence the correct can subset changes), stand xy jitter + yaw,
can and spare-plate floor spawns with free yaw and batched keep-out resampling. Heavy
imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- USD authoring helpers (stand + beam compound spawners) -----------------------------------
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


def _decorate(prim, color, contact_offset: float, material=None, collide: bool = True) -> None:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, size, center, color, contact_offset: float,
         quat=None, material=None, collide: bool = True) -> None:
    """Author one box child prim (translate -> orient -> scale; authored once)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if quat is not None:
        w, x, y, z = (float(v) for v in quat)
        sxf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _decorate(seg.GetPrim(), color, contact_offset, material, collide)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float, axis: str = "Z", material=None, collide: bool = True) -> None:
    """Author one cylinder child prim (PhysX collides it as a convex hull)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr(axis)
    lo = [-radius, -radius, -radius]
    hi = [radius, radius, radius]
    i = "XYZ".index(axis)
    lo[i], hi[i] = -height / 2, height / 2
    seg.CreateExtentAttr([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    _decorate(seg.GetPrim(), color, contact_offset, material, collide)


def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC stand at `prim_path`. Local frame: origin on the ground at
    the pivot's plan position; +x toward the EMPTY pan; the cradle apex (knife edge
    seat) at z = `apex_z`, so the axle (r `axle_r`) rests with its centre at
    `pivot_h` = apex_z + axle_r / sin(45 deg)."""
    import omni.usd
    from pxr import UsdPhysics

    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)

    slick = _friction_material(stage, f"{prim_path}/bearing_mat",
                               cfg.mu_bearing, cfg.mu_bearing)
    grippy = _friction_material(stage, f"{prim_path}/body_mat",
                                cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    # ground plate
    _box(stage, f"{prim_path}/base", (0.50, 0.14, 0.016), (0.0, 0.0, 0.008),
         cfg.stand_color, co, material=grippy)
    # stop platforms under the pans (limit tilt; slot under a level pan < any can)
    for s, nm in ((+1, "stop_e"), (-1, "stop_w")):
        _box(stage, f"{prim_path}/{nm}", (0.14, 0.13, cfg.stop_top - 0.016),
             (s * cfg.arm_len, 0.0, (cfg.stop_top + 0.016) / 2),
             cfg.stop_color, co, material=grippy)
    # posts + V-notch cradle plates (the knife edge; low-friction bearing material)
    c45 = math.cos(math.pi / 4)
    for sy in (+1, -1):
        py = sy * cfg.post_y
        _box(stage, f"{prim_path}/post_{'n' if sy > 0 else 's'}",
             (0.026, 0.024, cfg.apex_z - 0.020 - 0.016),
             (0.0, py, (cfg.apex_z - 0.020 + 0.016) / 2), cfg.stand_color, co,
             material=grippy)
        for sx in (+1, -1):
            # wall plate: box (w, d, t) rotated about y by sx*45 deg; its inner face
            # passes through the apex point (0, py, apex_z)
            w, t = 0.034, 0.010
            nx, nz = -sx * c45, c45          # inner-face normal (up, toward centre)
            fx, fz = sx * c45, c45           # in-face direction (up, outward)
            cx = fx * (w / 2 * 0.92) - nx * (t / 2)
            cz = cfg.apex_z + fz * (w / 2 * 0.92) - nz * (t / 2)
            # rotate -sx*45 deg about y: the box z (thickness) axis maps onto the
            # inner-face normal n, so the plate's broad face lines the V groove
            q = (math.cos(math.pi / 8), 0.0, -sx * math.sin(math.pi / 8), 0.0)
            _box(stage, f"{prim_path}/vee_{'n' if sy > 0 else 's'}_{'e' if sx > 0 else 'w'}",
                 (w, 0.024, t), (cx, py, cz), cfg.stand_color, co, quat=q,
                 material=slick)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC beam at `prim_path`. Local frame: origin at the AXLE CENTRE
    (the pivot), +x toward the EMPTY pan. Mass, CoM (below the pivot — the pendulum
    restoring torque is the instrument) and inertia are authored EXPLICITLY."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.beam_mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.beam_com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.beam_inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.2)
    px.CreateAngularDampingAttr(float(cfg.beam_ang_damping))
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    slick = _friction_material(stage, f"{prim_path}/bearing_mat",
                               cfg.mu_bearing, cfg.mu_bearing)
    grippy = _friction_material(stage, f"{prim_path}/body_mat",
                                cfg.mu_static, cfg.mu_dynamic)
    co = cfg.contact_offset
    L = cfg.arm_len
    # axle along local y (the pivot axis) + end caps (axial retention)
    _cyl(stage, f"{prim_path}/axle", cfg.axle_r, 0.130, (0.0, 0.0, 0.0),
         cfg.beam_color, co, axis="Y", material=slick)
    for sy, nm in ((+1, "cap_n"), (-1, "cap_s")):
        _cyl(stage, f"{prim_path}/{nm}", 0.013, 0.006, (0.0, sy * 0.065, 0.0),
             cfg.beam_color, co, axis="Y", material=slick)
    # drop post + arm (the CoM-lowering structure). The arm runs BELOW the pan
    # discs (top face touching their undersides) so the pan floors are clean flat
    # landing surfaces; the 33.5 mm arm-over-stop gap admits no can (d 56 mm).
    _box(stage, f"{prim_path}/drop", (0.024, 0.024, 0.072), (0.0, 0.0, -0.041),
         cfg.beam_color, co, material=grippy)
    _box(stage, f"{prim_path}/arm", (2 * L + 0.02, 0.028, 0.012), (0.0, 0.0, -0.072),
         cfg.beam_color, co, material=grippy)
    # pans: floor disc + 12-segment rim ring, floors at pan_floor_z below the pivot
    floor_c = cfg.pan_floor_z - 0.004  # disc centre (8 mm thick, top at pan_floor_z)
    for sx, col in ((+1, cfg.pan_e_color), (-1, cfg.pan_w_color)):
        nm = "pan_e" if sx > 0 else "pan_w"
        _cyl(stage, f"{prim_path}/{nm}_floor", cfg.pan_disc_r, 0.008,
             (sx * L, 0.0, floor_c), col, co, axis="Z", material=grippy)
        for k in range(12):
            a = 2 * math.pi * k / 12
            q = (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))
            _box(stage, f"{prim_path}/{nm}_rim_{k}",
                 (0.008, 2 * math.pi * cfg.rim_ring_r / 12 * 1.02, cfg.rim_h),
                 (sx * L + cfg.rim_ring_r * math.cos(a),
                  cfg.rim_ring_r * math.sin(a),
                  cfg.pan_floor_z + cfg.rim_h / 2),
                 col, co, quat=q, material=grippy)
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
            arm_len: float = 0.17
            apex_z: float = 0.158
            post_y: float = 0.048
            stop_top: float = 0.0535
            mu_bearing: float = 0.06
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            stand_color: tuple = (0.35, 0.34, 0.38)
            stop_color: tuple = (0.28, 0.27, 0.31)
            contact_offset: float = 0.0015

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            arm_len: float = 0.17
            axle_r: float = 0.005
            pan_floor_z: float = -0.058
            pan_disc_r: float = 0.075
            rim_ring_r: float = 0.071
            rim_h: float = 0.030
            beam_mass: float = 0.8
            beam_com_z: float = -0.050
            beam_inertia: tuple = (0.003, 0.016, 0.017)
            # a REAL balance is damped (air/magnetic dashpot): damping acts on velocity
            # only, so the settled angle — the judged quantity — is pure statics
            beam_ang_damping: float = 6.0
            mu_bearing: float = 0.06
            mu_static: float = 0.60
            mu_dynamic: float = 0.50
            beam_color: tuple = (0.72, 0.58, 0.20)
            pan_e_color: tuple = (0.80, 0.78, 0.72)
            pan_w_color: tuple = (0.80, 0.78, 0.72)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class CanBalanceSceneCfg(BaseCfg):
    """Config for `CanBalanceScene`. Honesty is asserted in __post_init__: the subset
    sums are all distinct (a unique correct answer), the balance is statically STABLE
    for every load configuration (CoM engineering), a 100 g error tilts the beam far
    outside the acceptance gate before hitting the stop, the stop caps tilt safely
    below the cans' topple angle, the slot under a level pan is too small to prop the
    beam with any object, and everything manipulated is jaw-sized."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    level_tol_deg: float = tunable(4.0)   # |beam tilt| gate for LEVEL (dead band ~1.5 deg;
    # the smallest wrong subset (100 g off) rests at >= 13 deg — asserted below)
    seat_tol: float = tunable(0.015)      # |axle centre - cradle seat| gate (m)
    yaw_tol_deg: float = tunable(15.0)    # beam yaw vs stand yaw gate (cradle slop ~8 deg)
    pan_fit_r: float = tunable(0.062)     # body centre within this of the pan axis = on pan
    clear_r: float = tunable(0.120)       # centre farther than this from BOTH pan axes = clear
    settle_lin: float = tunable(0.05)     # max |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.30)     # max |ang vel| when judging (rad/s)
    near_level_x: float = tunable(2.0)    # near-level latch gate = near_level_x * level_tol

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    stand_yaw_deg: float = tunable(35.0)  # stand yaw (+x = empty-pan side), +/- deg
    stand_jitter: float = tunable(0.03)   # stand xy jitter (+/- m)
    spawn_r_min: float = tunable(0.20)    # floor spawns: reach annulus (m)
    spawn_r_max: float = tunable(0.50)
    spawn_sep: float = tunable(0.115)     # pairwise floor-spawn separation (m)

    # --- info: layout ---------------------------------------------------------------------------
    stand_pos: tuple = info((0.28, 0.0))  # stand origin on the ground (nominal)
    # --- info: balance geometry (local frame: origin at pivot plan position; +x = empty pan) ----
    arm_len: float = info(0.17)           # pivot -> pan centre
    pivot_h: float = info(0.165)          # axle centre height (= apex_z + axle_r/sin45)
    apex_z: float = info(0.158)
    axle_r: float = info(0.005)
    pan_floor_z: float = info(-0.058)     # pan floor TOP, beam frame (below the pivot)
    pan_disc_r: float = info(0.075)
    rim_ring_r: float = info(0.071)
    rim_h: float = info(0.030)
    stop_top: float = info(0.0535)        # stop platform top (limits tilt to ~10.3 deg)
    beam_mass: float = info(0.8)
    beam_com_z: float = info(-0.050)      # beam CoM below the pivot (restoring arm d)
    beam_inertia: tuple = info((0.003, 0.016, 0.017))
    mu_bearing: float = info(0.06)        # knife-edge material (dead band ~1 deg)
    mu_static: float = info(0.60)
    mu_dynamic: float = info(0.50)
    contact_offset: float = info(0.0015)
    # --- info: bodies (binary masses -> unique subset sums) ------------------------------------
    masses: tuple = info((0.1, 0.2, 0.4))         # shared by plate i and can i
    can_r: float = info(0.028)
    can_h: tuple = info((0.062, 0.095, 0.135))    # short / mid / tall
    can_colors: tuple = info(((0.82, 0.10, 0.10), (0.10, 0.25, 0.85), (0.10, 0.62, 0.22)))
    can_names: tuple = info(("can_red", "can_blue", "can_green"))
    plate_side: float = info(0.060)
    plate_t: tuple = info((0.008, 0.016, 0.032))  # thickness encodes mass
    plate_color: tuple = info((0.10, 0.10, 0.12))
    plate_names: tuple = info(("plate_100", "plate_200", "plate_400"))
    # rubric weights (0.10 + 3*0.15 + 0.10 = 0.65 = the non-success cap)
    w_first: float = info(0.10)
    w_each: float = info(0.15)
    w_near: float = info(0.10)

    def __post_init__(self) -> None:
        # unique answer: all 8 subset sums distinct (binary weights)
        sums = set()
        for r in range(8):
            sums.add(round(sum(m for i, m in enumerate(self.masses) if r >> i & 1), 4))
        assert len(sums) == 8, "subset sums must be unique"
        g = 9.81
        d = -self.beam_com_z
        h_pan = -self.pan_floor_z
        # per-load pivot arm of a body resting on a pan (positive = below pivot = stabilizing)
        can_arm = [h_pan - hh / 2 for hh in self.can_h]
        plate_arm = h_pan - 0.016  # stack CoM (worst ~ full stack half height)
        # STABLE for every preload subset r (cans mirror the plates):
        # denom = m_b*d + sum(m_i * arm_i) must stay positive with margin
        denom_min, denom_max = 1e9, 0.0
        for r in range(1, 8):
            dn = self.beam_mass * d
            for i, m in enumerate(self.masses):
                if r >> i & 1:
                    dn += m * plate_arm + m * can_arm[i]
            denom_min = min(denom_min, dn)
            denom_max = max(denom_max, dn)
        assert denom_min > 0.015, f"balance not robustly stable (denom_min={denom_min:.4f})"
        # a 100 g error tilts beyond 2x the acceptance gate (before the stop stops it)
        tilt_err = math.degrees(math.atan(min(self.masses) * self.arm_len / denom_max))
        assert tilt_err > 2.5 * self.level_tol_deg, \
            f"100 g error only tilts {tilt_err:.1f} deg — gate not discriminating"
        # bearing dead band well inside the acceptance gate
        n_max = (self.beam_mass + 2 * sum(self.masses)) * g
        # linearized dead band: bearing friction torque / restoring torque gradient
        dead_deg = math.degrees(self.mu_bearing * n_max * self.axle_r / (denom_min * g))
        assert dead_deg < 0.6 * self.level_tol_deg, f"dead band {dead_deg:.2f} deg too wide"
        # stop caps tilt below the cans' topple angle
        # pan lowest point: outer floor edge at radius arm_len + pan_disc_r
        rr = self.arm_len + self.pan_disc_r
        z0 = self.pivot_h + self.pan_floor_z - 0.008
        th = 0.0
        for _ in range(60):  # solve z0*cos(th) ... simple fixed-point on contact eq
            th = math.asin(max(0.0, (z0 * math.cos(th) - self.stop_top)) / rr)
        self.max_tilt_deg = math.degrees(th)
        topple = math.degrees(math.atan(self.can_r / (max(self.can_h) / 2)))
        assert self.max_tilt_deg < topple - 6.0, \
            f"stop tilt {self.max_tilt_deg:.1f} vs can topple {topple:.1f}"
        assert self.max_tilt_deg > 2.5 * self.level_tol_deg, "stop tilt inside the gate"
        # the slot under a LEVEL pan admits no object (no propping from below)
        slot = (self.pivot_h + self.pan_floor_z - 0.008) - self.stop_top
        assert slot < min(2 * self.can_r, min(self.can_h)) - 0.008, "pan can be propped"
        assert slot > max(self.plate_t) + 0.008, "slot must clear the thickest plate"
        # three cans fit together on one pan; a can fits through the rim ring
        assert self.rim_ring_r - 0.004 > self.can_r * (1 + 2 / math.sqrt(3)) + 0.002
        # graspable by the 80 mm Franka jaw
        assert 2 * self.can_r <= 0.078 and self.plate_side <= 0.078
        # plates stack inside the rim region and under the pivot (stay below it)
        assert sum(self.plate_t) < h_pan - 0.001
        assert self.plate_side * math.sqrt(2) / 2 < self.rim_ring_r


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
@SCENES.register("can_balance")
class CanBalanceScene(BaseScene):
    cfg: CanBalanceSceneCfg

    def __init__(self, cfg: CanBalanceSceneCfg | None = None) -> None:
        super().__init__(cfg or CanBalanceSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        stand_spawn = sp["stand"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            arm_len=c.arm_len, apex_z=c.apex_z, stop_top=c.stop_top,
            mu_bearing=c.mu_bearing, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset)
        beam_spawn = sp["beam"](
            arm_len=c.arm_len, axle_r=c.axle_r, pan_floor_z=c.pan_floor_z,
            pan_disc_r=c.pan_disc_r, rim_ring_r=c.rim_ring_r, rim_h=c.rim_h,
            beam_mass=c.beam_mass, beam_com_z=c.beam_com_z, beam_inertia=c.beam_inertia,
            mu_bearing=c.mu_bearing, mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset)

        # Loose bodies (plates/cans) are damped like real steel-on-steel contacts:
        # with sleeping disabled, an undamped light plate atop a 3-deep stack keeps
        # micro-chattering above the settle gates forever. Damping acts on velocity
        # only — every judged equilibrium is pure statics.
        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.3, angular_damping=0.8,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        pmat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_static, dynamic_friction=c.mu_dynamic, restitution=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.8, dynamic_friction=0.7, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.stand_pos[0], c.stand_pos[1], 0.0)),
            ),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], c.pivot_h)),
            ),
        }
        for i, name in enumerate(c.plate_names):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.plate_side, c.plate_side, c.plate_t[i]),
                    rigid_props=rigid, collision_props=coll, physics_material=pmat,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.masses[i]),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.9 + 0.1 * i, -0.7, c.plate_t[i] / 2)),
            )
        for i, name in enumerate(c.can_names):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Can_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.can_r, height=c.can_h[i],
                    rigid_props=rigid, collision_props=coll, physics_material=pmat,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.masses[i]),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.can_colors[i])),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.9 + 0.1 * i, -0.9, c.can_h[i] / 2)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # kills the residual contact-velocity noise on light stacked bodies
                # (a chattering 8 mm plate flickers the settle gates otherwise)
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
        c = self.cfg
        self.stand: RigidObject = env.iscene["stand"]
        self.beam: RigidObject = env.iscene["beam"]
        self.plates: dict[str, RigidObject] = {n: env.iscene[n] for n in c.plate_names}
        self.cans: dict[str, RigidObject] = {n: env.iscene[n] for n in c.can_names}
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        # preload[e, i]: plate i starts on the loaded pan of episode e (=> can i required)
        self.preload = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._any_on_pan = torch.zeros(n, dtype=torch.bool, device=dev)
        self._req_on_pan = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self._near_level = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the stand (yaw + xy jitter), seat the LEVEL beam on
        its knife edge, sample the preload subset and stack it on the LOADED (-x)
        pan, scatter spare plates and all cans on the floor (free yaw, keep-out
        resampled), clear the latches. The caller settles; the beam then tips onto
        its stop under the preload — the tilted balance IS the initial condition."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- stand: kinematic, yaw + xy jitter ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        q_stand = _qz(psi)
        sp = torch.zeros(m, 3, device=dev)
        sp[:, 0] = c.stand_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        sp[:, 1] = c.stand_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + origin
        st[:, 3:7] = q_stand
        self.stand.write_root_state_to_sim(st, env_ids)

        # --- beam: seated level on the cradle, aligned with the stand ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + origin
        st[:, 2] += c.pivot_h
        st[:, 3:7] = q_stand
        self.beam.write_root_state_to_sim(st, env_ids)

        # --- preload subset: uniform over the 7 non-empty subsets ---
        r = torch.randint(1, 8, (m,), device=dev)
        for i in range(3):
            self.preload[env_ids, i] = (r >> i) & 1 == 1

        # --- floor spawns: 6 slots (3 cans + 3 plates), keep-out resampled ---
        cpsi, spsi = torch.cos(psi), torch.sin(psi)
        xy = torch.zeros(m, 6, 2, device=dev)
        bad = torch.ones(m, 6, dtype=torch.bool, device=dev)
        for _ in range(40):
            if not bad.any():
                break
            k = int(bad.sum())
            cand = torch.rand(k, 2, device=dev) * (2 * (c.spawn_r_max + 0.02)) \
                - (c.spawn_r_max + 0.02)
            xy[bad] = cand
            rr = xy.norm(dim=-1)
            ok = (rr > c.spawn_r_min) & (rr < c.spawn_r_max)
            # outside the stand footprint (stand frame, with margin)
            rel = xy - sp[:, None, 0:2]
            u = rel[..., 0] * cpsi[:, None] + rel[..., 1] * spsi[:, None]
            v = -rel[..., 0] * spsi[:, None] + rel[..., 1] * cpsi[:, None]
            ok &= ~((u.abs() < c.arm_len + c.pan_disc_r + 0.09) & (v.abs() < 0.19))
            # pairwise separation
            d = (xy[:, :, None, :] - xy[:, None, :, :]).norm(dim=-1)
            d += torch.eye(6, device=dev) * 10.0
            ok &= d.min(dim=-1).values > c.spawn_sep
            bad = ~ok
        yaw = torch.rand(m, 6, device=dev) * 2 * math.pi

        # --- cans: floor slots 0..2, upright, free yaw ---
        for i, name in enumerate(c.can_names):
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:2] = xy[:, i]
            s[:, 2] = c.can_h[i] / 2 + 0.002
            s[:, 3:7] = _qz(yaw[:, i])
            s[:, 0:3] += origin
            self.cans[name].write_root_state_to_sim(s, env_ids)

        # --- plates: preload -> stacked on the LOADED (-x) pan; spare -> floor slot ---
        # stack thickest-first so the pile is stable
        pan_c = torch.zeros(m, 3, device=dev)
        pan_c[:, 0] = -c.arm_len
        pan_c[:, 2] = c.pivot_h + c.pan_floor_z
        from isaaclab.utils.math import quat_apply

        pan_w = sp + quat_apply(q_stand, pan_c)  # loaded-pan floor centre, world (level beam)
        z_cursor = torch.zeros(m, device=dev)
        for i in (2, 1, 0):  # thickest at the bottom of the stack
            name = c.plate_names[i]
            pre = self.preload[env_ids, i]
            s = torch.zeros(m, 13, device=dev)
            # on-pan pose
            on = torch.zeros(m, 3, device=dev)
            on[:, 0:2] = pan_w[:, 0:2]
            on[:, 2] = pan_w[:, 2] + z_cursor + c.plate_t[i] / 2 + 0.002
            # floor pose (slot 3 + i)
            fl = torch.zeros(m, 3, device=dev)
            fl[:, 0:2] = xy[:, 3 + i]
            fl[:, 2] = c.plate_t[i] / 2 + 0.002
            s[:, 0:3] = torch.where(pre.unsqueeze(1), on, fl) + origin
            s[:, 3:7] = torch.where(pre.unsqueeze(1), q_stand, _qz(yaw[:, 3 + i]))
            self.plates[name].write_root_state_to_sim(s, env_ids)
            z_cursor = z_cursor + torch.where(pre, torch.full_like(z_cursor, c.plate_t[i] + 0.004),
                                              torch.zeros_like(z_cursor))

        # --- clear latches ---
        self._any_on_pan[env_ids] = False
        self._req_on_pan[env_ids] = False
        self._near_level[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "plates": {n: b.data.root_state_w[env_ids].clone() for n, b in self.plates.items()},
            "cans": {n: b.data.root_state_w[env_ids].clone() for n, b in self.cans.items()},
            "preload": self.preload[env_ids].clone(),
            "any_on_pan": self._any_on_pan[env_ids].clone(),
            "req_on_pan": self._req_on_pan[env_ids].clone(),
            "near_level": self._near_level[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for n, b in self.plates.items():
            b.write_root_state_to_sim(state["plates"][n], env_ids)
        for n, b in self.cans.items():
            b.write_root_state_to_sim(state["cans"][n], env_ids)
        self.preload[env_ids] = state["preload"]
        self._any_on_pan[env_ids] = state["any_on_pan"]
        self._req_on_pan[env_ids] = state["req_on_pan"]
        self._near_level[env_ids] = state["near_level"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A two-pan BEAM BALANCE stands on the floor: a golden beam rocks on a "
            f"knife-edge pivot {c.pivot_h * 100:.0f} cm up, carrying a round rimmed "
            f"pan (radius ~{c.pan_disc_r * 100:.0f} cm) at each end, with a gray stop "
            f"platform under each pan limiting the swing. One pan — call it the LOADED "
            f"pan — carries a stack of square BLACK STEEL PLATES and hangs low against "
            f"its stop; the other pan is EMPTY and rides high. The plates weigh 100, "
            f"200 or 400 g and their thickness shows it (8 / 16 / 32 mm); any plates "
            f"not on the balance lie loose on the floor and are SPARES. Also on the "
            f"floor stand three drink cans, {2 * c.can_r * 1000:.0f} mm across: a "
            f"short RED can (100 g), a mid-size BLUE can (200 g) and a tall GREEN can "
            f"(400 g). The balance position/heading, the preloaded plate subset and "
            f"all floor spawns change every episode.\n"
            f"Goal: counterweigh the balance USING ONLY THE CANS. Read the plates on "
            f"the loaded pan, and set the combination of cans whose total mass equals "
            f"theirs onto the EMPTY pan (each can resting inside the pan's rim; any "
            f"order; masses 100/200/400 g make exactly one combination correct) so "
            f"the beam swings back and rests LEVEL (within ~{c.level_tol_deg:.0f} "
            f"degrees). The preloaded plates must stay on their pan, the spare plates "
            f"must stay off the balance entirely, unused cans must be left clear of "
            f"the balance, and the beam must remain seated on its knife edge. A wrong "
            f"can total leaves the beam visibly tilted — watch it respond and correct "
            f"if needed."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Balance the scale using only the drink cans: place the cans whose total "
            "mass equals the black plates on the loaded pan (red 100 g, blue 200 g, "
            "green 400 g) onto the empty pan, so the beam rests level. Do not move "
            "the plates and keep everything else off the scale."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _stand_yaw(self) -> torch.Tensor:
        q = self.stand.data.root_quat_w
        return 2.0 * torch.atan2(q[:, 3], q[:, 0])

    def _beam_axes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(x_axis_w, tilt_deg): the beam's long axis in world and its tilt angle
        (positive = empty-pan end high)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(n, 3)
        xw = quat_apply(self.beam.data.root_quat_w, ex)
        tilt = torch.rad2deg(torch.asin(xw[:, 2].clamp(-1.0, 1.0)))
        return xw, tilt

    def _pan_centers_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(loaded_pan_floor_w, empty_pan_floor_w), each (N,3) — live beam frame."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        loc_e = torch.tensor([c.arm_len, 0.0, c.pan_floor_z], device=dev).expand(n, 3)
        loc_w = torch.tensor([-c.arm_len, 0.0, c.pan_floor_z], device=dev).expand(n, 3)
        p = self.beam.data.root_pos_w
        q = self.beam.data.root_quat_w
        return p + quat_apply(q, loc_w), p + quat_apply(q, loc_e)

    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  pos_w - self.beam.data.root_pos_w)

    def _settled(self, body) -> torch.Tensor:
        c = self.cfg
        return (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _on_pan(self, body, side: int) -> torch.Tensor:
        """(N,) bool: body resting fully on the pan at beam-local side*arm_len —
        centre within `pan_fit_r` of the pan axis and in the height band above the
        pan floor, settled. side=-1 loaded pan, +1 empty pan."""
        c = self.cfg
        loc = self._beam_local(body.data.root_pos_w)
        dx = loc[:, 0] - side * c.arm_len
        radial = torch.sqrt(dx * dx + loc[:, 1] * loc[:, 1])
        dz = loc[:, 2] - c.pan_floor_z
        return (radial < c.pan_fit_r) & (dz > 0.001) & (dz < 0.17) & self._settled(body)

    def _clear_of_balance(self, body) -> torch.Tensor:
        """(N,) bool: horizontal distance from BOTH pan axes exceeds `clear_r` (no
        wedging under a pan, no leaning on a rim, no riding the stop platforms)."""
        c = self.cfg
        pw, pe = self._pan_centers_w()
        p = body.data.root_pos_w
        dw = (p[:, 0:2] - pw[:, 0:2]).norm(dim=-1)
        de = (p[:, 0:2] - pe[:, 0:2]).norm(dim=-1)
        return (dw > c.clear_r) & (de > c.clear_r)

    def _seated(self) -> torch.Tensor:
        """(N,) bool: axle centre at the cradle seat, beam yaw aligned with the stand."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        seat_loc = torch.tensor([0.0, 0.0, c.pivot_h], device=self.env.device).expand(n, 3)
        seat_w = self.stand.data.root_pos_w + quat_apply(self.stand.data.root_quat_w, seat_loc)
        near = (self.beam.data.root_pos_w - seat_w).norm(dim=-1) < c.seat_tol
        xw, _tilt = self._beam_axes()
        psi = self._stand_yaw()
        sx = torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1)
        hx = xw[:, 0:2]
        hx = hx / hx.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        aligned = (hx * sx).sum(dim=-1) > math.cos(math.radians(c.yaw_tol_deg))
        return near & aligned

    def _status(self) -> dict[str, torch.Tensor]:
        """Live geometric predicates (all (N,) or (N,3))."""
        c = self.cfg
        _xw, tilt = self._beam_axes()
        seated = self._seated()
        beam_still = self._settled(self.beam)
        plates_on = torch.stack([self._on_pan(self.plates[n], -1) for n in c.plate_names], dim=1)
        plates_clear = torch.stack([self._clear_of_balance(self.plates[n])
                                    for n in c.plate_names], dim=1)
        cans_on = torch.stack([self._on_pan(self.cans[n], +1) for n in c.can_names], dim=1)
        cans_clear = torch.stack([self._clear_of_balance(self.cans[n])
                                  for n in c.can_names], dim=1)
        preload_ok = (plates_on | ~self.preload).all(dim=1)
        spares_ok = (plates_clear | self.preload).all(dim=1)
        cans_ok = (cans_on | cans_clear).all(dim=1)
        return {"tilt": tilt, "seated": seated, "beam_still": beam_still,
                "plates_on": plates_on, "cans_on": cans_on,
                "preload_ok": preload_ok, "spares_ok": spares_ok, "cans_ok": cans_ok}

    def _update_latches(self) -> None:
        c = self.cfg
        s = self._status()
        loaded = s["cans_on"].any(dim=1)
        self._any_on_pan |= loaded & s["seated"]
        self._req_on_pan |= s["cans_on"] & self.preload & s["seated"].unsqueeze(-1)
        self._near_level |= (s["tilt"].abs() < c.near_level_x * c.level_tol_deg) \
            & s["seated"] & s["preload_ok"] & loaded
        # a can on the pan of an unseated beam earns nothing (no off-cradle credit)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: beam seated on its knife edge, settled, LEVEL within
        `level_tol_deg`; every preload plate resting on the loaded pan; every spare
        plate clear of both pans; every can on the empty pan or clear; at least one
        can on the empty pan; everything finite. Mass correctness is implied
        PHYSICALLY: with the preload intact and only cans on the empty pan, level is
        reachable only by the unique matching subset (statics asserted in cfg)."""
        c = self.cfg
        self._update_latches()
        s = self._status()
        pos = torch.stack([self.beam.data.root_pos_w]
                          + [b.data.root_pos_w for b in self.plates.values()]
                          + [b.data.root_pos_w for b in self.cans.values()], dim=1)
        finite = torch.isfinite(pos).all(dim=-1).all(dim=-1)
        return s["seated"] & s["beam_still"] & (s["tilt"].abs() < c.level_tol_deg) \
            & s["preload_ok"] & s["spares_ok"] & s["cans_ok"] \
            & s["cans_on"].any(dim=1) & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10*any-can-ever-on-pan + 0.15*each required can
        ever on the pan + 0.10*beam ever near-level (all latched; 0 for doing
        nothing), capped at 0.65 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_first * self._any_on_pan.float()
                + c.w_each * (self._req_on_pan & self.preload).float().sum(dim=1)
                + c.w_near * self._near_level.float()).clamp(max=0.65)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="can_balance", robot="null"))
