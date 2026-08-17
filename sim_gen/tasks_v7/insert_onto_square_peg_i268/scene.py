"""RingBalanceScene — counterweight a beam balance by threading the RIGHT NUMBER of
square rings onto the empty pan's post until the beam settles LEVEL.

Derived from rlbench/insert_onto_square_peg ("pick up the square ring and put it on
the <color> spoke": one free square ring, three colored pegs; ONE vertical drop onto
the peg matching a named color; judged by the ring encircling the chosen peg; more
insertion can never hurt). Here the seed's whole plan is insufficient by
construction, not re-parameterized:

  1. the CHOICE moved from color perception to PHYSICAL REASONING: nothing is
     color-coded to a target. A beam balance rests tipped onto a hard stop because
     one pan carries a stack of 1..4 black ballast rings (count randomized per
     episode). The task is to discover HOW MANY gold rings counterweight it —
     by counting the black stack or by watching the beam's response ring by ring —
     and to load exactly that many;
  2. the judged predicate is a MECHANISM EQUILIBRIUM, not object containment: the
     beam angle within +/-4 deg of level, settled, with the ballast stack intact.
     No single placement is ever success; success emerges from the loaded beam's
     own statics;
  3. insertion is DEMOTED to a placement primitive and OVER-insertion FAILS: one
     ring too many tips the beam onto the OPPOSITE stop (the smoke battery threads
     all five gold rings — flawless seed-style insertion mastery — and shows it
     scores no success). The seed has no notion of "too much";
  4. a null/hands-off policy reads pinned at a stop (score ~0) and the
     ballast-intact clause rejects the degenerate "balance by unloading the loaded
     pan" solution.

The balance is built so the physics is honest by construction (asserted numerically
in cfg.__post_init__): a one-ring imbalance ALWAYS presses the beam onto a stop
(net torque margin at the stop, worst load case), while the worst-case bore-slack
bias of a balanced load stays a degree inside the +/-4 deg success band. The stops
are JOINT LIMITS (beam<->stand contact is joint-filtered and the pans ride ~0.27 m
above the floor), so there is no geometry to wedge or prop: only real mass in the
pans moves the settled angle.

Assets are fully procedural (compound spawners; custom spawn funcs apply no cfg
schemas, so mass/density/damping/collision are authored inside the funcs):
  - stand: heavy DYNAMIC pedestal (base + column + fork plates + visual axle pin),
    root MassAPI 30 kg with CoM at ground level. Dynamic, not kinematic: a joint
    anchored to a kinematic body stays world-fixed when teleported at reset.
  - beam: DYNAMIC rotor, body origin ON the axle (pure-quat angle writes; the
    stand+beam linkage is always written together). SHORT crossbar at axle
    height (ends well inboard of the pans), a hanger dropping at each bar end,
    and an under-arm running BELOW the pan floor out to each square-rimmed PAN
    with its square center POST (24 mm shaft, stepped 20/13 mm tip = a
    self-centering funnel for the 30 mm ring bore). The airspace above each
    post is completely open, so rings can descend the full bore travel. All
    restoring moment comes from below-pivot beam mass + loads.
    Spawn-authored RevoluteJoint to the stand (axis Y, limits +/-12 deg).
  - rings: 9 identical square rings (70 mm outer, 30 mm bore, 12 mm thick, steel
    density -> 0.377 kg): 5 GOLD (the supply, scattered on the floor), 4 BLACK
    (ballast; k active on the ballast post, the rest parked far off-field).

Per-episode randomization (readback-verifiable): stand xy + yaw, ballast side
(left/right pan), ballast count k in 1..4, gold ring scatter slots + jitter + yaw.

Rubric (0..1; latched stage credit anchored in the demonstrated solve):
  0.10 * lifted  — any gold ring ever above lift_z (latched: a ring got picked up)
  0.50 * load    — latched max of min(n_gold_on_free_post, k)/k (progress;
                   over-loading never earns beyond k and never evaporates)
  1.0 iff success() — |beam angle| <= 5.5 deg with a 1 s settled STREAK (the
                   streak gate rejects a beam swinging through level), ballast
                   stack intact on its post, >= 1 ring threaded on the free post,
                   everything still and finite. Non-success capped at 0.60; null
                   policy ~0 (beam pinned on its stop, nothing lifted).

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


# ----- USD authoring helpers --------------------------------------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable | None,
             density: float | None = None):
    """One box child: translate + scale, displayColor, collider, density."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(box.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable | None,
             axis: str = "Z", density: float | None = None):
    """One cylinder child along `axis`."""
    from pxr import Gf, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    h2 = height / 2
    ext = {"Z": (Gf.Vec3f(-radius, -radius, -h2), Gf.Vec3f(radius, radius, h2)),
           "Y": (Gf.Vec3f(-radius, -h2, -radius), Gf.Vec3f(radius, h2, radius)),
           "X": (Gf.Vec3f(-h2, -radius, -radius), Gf.Vec3f(h2, radius, radius))}[axis]
    cyl.CreateExtentAttr(list(ext))
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide is not None:
        collide(cyl.GetPrim())
    if density is not None:
        UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateDensityAttr(float(density))
    return cyl.GetPrim()


# ----- compound spawn funcs ---------------------------------------------------------------------
def _spawn_stand(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The balance pedestal: heavy DYNAMIC base + column + fork plates + visual axle
    pin. Local frame: origin at the footprint centre on the ground, beam axle along
    local y through (0, 0, pivot_z). Root MassAPI mass (CoM at the body origin =
    ground level) keeps it planted."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.stand_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    col = c.stand_color
    _add_box(stage, f"{prim_path}/base", center=(0.0, 0.0, 0.02),
             size=(0.34, 0.26, 0.04), color=col, collide=collide)
    _add_box(stage, f"{prim_path}/column", center=(0.0, 0.0, 0.165),
             size=(0.06, 0.05, 0.25), color=col, collide=collide)
    zf = (0.29 + c.pivot_z + 0.012) / 2
    hf = c.pivot_z + 0.012 - 0.29
    for name, sgn in (("fork_p", 1.0), ("fork_n", -1.0)):
        _add_box(stage, f"{prim_path}/{name}", center=(0.0, sgn * 0.035, zf),
                 size=(0.05, 0.010, hf), color=col, collide=collide)
    _add_cyl(stage, f"{prim_path}/axle_vis", center=(0.0, 0.0, c.pivot_z),
             radius=0.007, height=0.085, color=(0.62, 0.62, 0.65),
             collide=None, axis="Y")
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The balance beam: DYNAMIC rotor, body origin ON the axle. A SHORT crossbar
    through the origin whose ends stay inboard of the ring descent corridors; at
    each end a hanger drops to a horizontal under-arm that carries, FROM BELOW, a
    square-rimmed pan centred at local +/-arm, its floor pan_drop below the axle,
    with a square centre post (shaft + two centering steps) rising from the pan
    floor. The airspace above each post is completely open. All mass via per-child density
    so the restoring moment (all pan/hanger mass below the pivot) and the rotor
    inertia are real. Spawn-authored REVOLUTE joint to the sibling stand (axis Y
    through the origin, hard limits +/-stop_deg; joint-pair collision filtered —
    the beam never touches the stand, the stops are the joint limits)."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(c.beam_ang_damping))
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    den = c.alu_density
    # the bar ENDS well inboard of the pan axes: the ring's descent corridor over
    # each post (outer 70 mm around x=+/-arm) must contain NO beam geometry above
    # the funnel tip. The pan hangs from an outboard drop at the bar end plus a
    # horizontal under-arm BELOW the pan floor.
    bx = c.arm - 0.060                       # bar half-length: the bar end stays
    # clear of the ring outer HALF-DIAGONAL (a 45deg-yawed ring reaches
    # arm - ring_outer*sqrt(2)/2 = 0.1105 inboard), so no admissible ring pose
    # can catch the bar or the hanger drop
    _add_box(stage, f"{prim_path}/bar", center=(0.0, 0.0, 0.0),
             size=(2 * bx, 0.030, 0.020), color=c.bar_color,
             collide=collide, density=den)
    pf_top = -c.pan_drop                    # pan floor TOP (rings rest here)
    pf_c = pf_top - c.pan_floor_t / 2
    pf_bot = pf_top - c.pan_floor_t         # pan floor bottom (-0.075)
    ua_c = pf_bot - 0.004                   # under-arm centre z (-0.079)
    for side, sx in (("l", -1.0), ("r", 1.0)):
        x0 = sx * c.arm
        hx = sx * (bx - 0.005)              # hanger drop at the bar end
        _add_box(stage, f"{prim_path}/hang_{side}",
                 center=(hx, 0.0, (0.008 + (ua_c - 0.004)) / 2),
                 size=(0.010, 0.024, 0.008 - (ua_c - 0.004)), color=c.pan_color,
                 collide=collide, density=den)
        ua_lo, ua_hi = sorted((sx * (bx - 0.010), sx * (c.arm + c.pan_w / 2)))
        _add_box(stage, f"{prim_path}/underarm_{side}",
                 center=((ua_lo + ua_hi) / 2, 0.0, ua_c),
                 size=(ua_hi - ua_lo, 0.024, 0.008), color=c.pan_color,
                 collide=collide, density=den)
        _add_box(stage, f"{prim_path}/pan_{side}",
                 center=(x0, 0.0, pf_c),
                 size=(c.pan_w, c.pan_w, c.pan_floor_t), color=c.pan_color,
                 collide=collide, density=den)
        rim_c = pf_top + c.rim_h / 2 - 0.001
        d = c.pan_w / 2 + 0.004
        _add_box(stage, f"{prim_path}/rim_{side}_xp", center=(x0 + d, 0.0, rim_c),
                 size=(0.008, c.pan_w + 0.016, c.rim_h), color=c.pan_color,
                 collide=collide, density=den)
        _add_box(stage, f"{prim_path}/rim_{side}_xn", center=(x0 - d, 0.0, rim_c),
                 size=(0.008, c.pan_w + 0.016, c.rim_h), color=c.pan_color,
                 collide=collide, density=den)
        _add_box(stage, f"{prim_path}/rim_{side}_yp", center=(x0, d, rim_c),
                 size=(c.pan_w + 0.016, 0.008, c.rim_h), color=c.pan_color,
                 collide=collide, density=den)
        _add_box(stage, f"{prim_path}/rim_{side}_yn", center=(x0, -d, rim_c),
                 size=(c.pan_w + 0.016, 0.008, c.rim_h), color=c.pan_color,
                 collide=collide, density=den)
        # centre post: shaft + two self-centering steps (a stepped funnel tip)
        _add_box(stage, f"{prim_path}/post_{side}",
                 center=(x0, 0.0, pf_top + c.post_h / 2),
                 size=(2 * c.post_half, 2 * c.post_half, c.post_h),
                 color=c.post_color, collide=collide, density=den)
        z1 = pf_top + c.post_h
        _add_box(stage, f"{prim_path}/step1_{side}",
                 center=(x0, 0.0, z1 + c.step_h / 2),
                 size=(0.020, 0.020, c.step_h), color=c.step_color,
                 collide=collide, density=den)
        _add_box(stage, f"{prim_path}/step2_{side}",
                 center=(x0, 0.0, z1 + c.step_h + c.step_h / 2),
                 size=(0.013, 0.013, c.step_h), color=c.step_color,
                 collide=collide, density=den)

    # revolute joint to the sibling stand, axis Y through the axle
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/axle")
    j.CreateBody0Rel().SetTargets([f"{base}/Stand"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_z)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(-c.stop_deg))
    j.CreateUpperLimitAttr(float(c.stop_deg))
    return root


def _spawn_ring(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One square ring: 4 steel boxes forming a closed square band (outer
    ring_outer, bore 2*bore_half, thickness ring_t), body origin at the centre,
    bore along local z. Per-child density -> the real 0.377 kg ring mass."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    c = cfg
    UsdPhysics.RigidBodyAPI.Apply(root)
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(c.contact_offset)
    band = (c.ring_outer - 2 * c.bore_half) / 2
    off = c.bore_half + band / 2
    _add_box(stage, f"{prim_path}/b_yp", center=(0.0, off, 0.0),
             size=(c.ring_outer, band, c.ring_t), color=c.ring_color,
             collide=collide, density=c.steel_density)
    _add_box(stage, f"{prim_path}/b_yn", center=(0.0, -off, 0.0),
             size=(c.ring_outer, band, c.ring_t), color=c.ring_color,
             collide=collide, density=c.steel_density)
    _add_box(stage, f"{prim_path}/b_xp", center=(off, 0.0, 0.0),
             size=(band, 2 * c.bore_half, c.ring_t), color=c.ring_color,
             collide=collide, density=c.steel_density)
    _add_box(stage, f"{prim_path}/b_xn", center=(-off, 0.0, 0.0),
             size=(band, 2 * c.bore_half, c.ring_t), color=c.ring_color,
             collide=collide, density=c.steel_density)
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
            stand_mass: float = 30.0
            pivot_z: float = 0.35
            stand_color: tuple = (0.16, 0.18, 0.24)
            contact_offset: float = 0.0008

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            arm: float = 0.16
            pivot_z: float = 0.35
            pan_drop: float = 0.065
            pan_w: float = 0.100
            pan_floor_t: float = 0.010
            rim_h: float = 0.014
            post_half: float = 0.012
            post_h: float = 0.072
            step_h: float = 0.008
            stop_deg: float = 12.0
            beam_ang_damping: float = 3.0
            alu_density: float = 2700.0
            bar_color: tuple = (0.72, 0.58, 0.28)
            pan_color: tuple = (0.55, 0.57, 0.62)
            post_color: tuple = (0.42, 0.44, 0.50)
            step_color: tuple = (0.80, 0.80, 0.84)
            contact_offset: float = 0.0008

        @configclass
        class RingSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ring)
            ring_outer: float = 0.070
            bore_half: float = 0.015
            ring_t: float = 0.012
            steel_density: float = 7850.0
            ring_color: tuple = (0.85, 0.68, 0.20)
            contact_offset: float = 0.0008

        _SPAWNER_CACHE["stand"] = StandSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
        _SPAWNER_CACHE["ring"] = RingSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class RingBalanceSceneCfg(BaseCfg):
    """Config for `RingBalanceScene`. The equilibrium geometry is honest by
    construction — every load-bearing clause is asserted numerically in
    __post_init__: a one-ring imbalance always presses the beam onto a stop; the
    worst-case bore-slack bias of a balanced load stays a degree inside the
    success band; the stepped post tip funnels the bore; the post retains a full
    stack at the stop angle; no prop can reach the airborne pans."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    succ_tol_deg: float = tunable(5.5)       # |beam angle| below this counts as level (above the
    #   natural hands-off settle envelope: worst bore-slack bias ~2.6 deg plus the measured
    #   decaying overshoot ~2.6 deg past it grazes ~5.3 deg; wrong counts press the 12 deg stops)
    beam_settle_avel: float = tunable(0.25)  # max beam |ang vel| when judging (rad/s)
    settle_speed: float = tunable(0.10)      # max ring/stand |lin vel| when judging (m/s)
    succ_streak: int = tunable(120)          # consecutive settled-level steps (1 s at 120 Hz)
    capture_xy: float = tunable(0.035)       # ring centre within this of a post axis (beam frame)
    capture_z: tuple = tunable((-0.075, 0.020))  # ring centre z window on a post (beam frame)
    lift_z: float = tunable(0.15)            # a gold ring above this (env z) latches `lifted`

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    stand_yaw_deg: float = tunable(25.0)     # stand yaw about nominal (+/- deg)
    stand_jitter: float = tunable(0.05)      # stand xy jitter (+/- m)
    k_min: int = tunable(1)                  # ballast count lower bound
    k_max: int = tunable(4)                  # ballast count upper bound
    ring_jitter: float = tunable(0.03)       # gold ring slot xy jitter (+/- m)
    scatter_r: float = tunable(0.42)         # gold ring scatter arc radius (m)
    scatter_arc: tuple = tunable((-145.0, -35.0))  # scatter arc (deg, stand frame)

    # --- info: geometry (kept in sync with the spawner defaults) --------------------------------
    stand_pos: tuple = info((0.0, 0.0))
    stand_mass: float = info(30.0)
    pivot_z: float = info(0.35)              # axle height (stand local, axis = local y)
    arm: float = info(0.16)                  # pan centre offset along the beam (+/- x)
    pan_drop: float = info(0.065)            # axle to pan-floor TOP
    pan_w: float = info(0.100)
    pan_floor_t: float = info(0.010)
    rim_h: float = info(0.014)
    post_half: float = info(0.012)           # 24 mm square post shaft
    post_h: float = info(0.072)              # shaft height above the pan floor
    step_h: float = info(0.008)              # each of the two centering steps
    stop_deg: float = info(12.0)             # joint hard limits (the stops)
    beam_ang_damping: float = info(3.0)
    alu_density: float = info(2700.0)
    ring_outer: float = info(0.070)          # square ring outer side
    bore_half: float = info(0.015)           # 30 mm square bore
    ring_t: float = info(0.012)
    steel_density: float = info(7850.0)
    n_gold: int = info(5)
    n_black: int = info(4)
    gold_color: tuple = info((0.85, 0.68, 0.20))
    black_color: tuple = info((0.10, 0.10, 0.11))
    park_pos: tuple = info((1.6, 1.6))       # off-field depot for inactive black rings
    stack_gap: float = info(0.0006)          # reset stack clearance (depenetration hygiene)
    # rubric weights (0.10 + 0.50 = 0.60 = the non-success cap)
    w_lift: float = info(0.10)
    w_load: float = info(0.50)
    contact_offset: float = info(0.0008)

    def __post_init__(self) -> None:
        """Audit the equilibrium / funnel / retention geometry (metres, kg, deg)."""
        band = (self.ring_outer - 2 * self.bore_half) / 2
        vol = 2 * (self.ring_outer * band * self.ring_t) \
            + 2 * (band * 2 * self.bore_half * self.ring_t)
        m = self.steel_density * vol                     # one ring (~0.377 kg)
        self.ring_mass = m
        g = 9.81
        # beam restoring coefficient sum(m_i * h_i) below the pivot, mirroring
        # the spawner's children exactly (bar at pivot height contributes 0;
        # symmetric +/-x offsets cancel — only the vertical offsets matter):
        # hanger drops, under-arms, pan floors, rims, posts, steps (steps sit
        # ABOVE the pivot line? no — above the pan, still below the pivot by
        # -z1; their tiny above/below split is carried with signed h)
        bx = self.arm - 0.060
        pf_top = -self.pan_drop
        pf_bot = pf_top - self.pan_floor_t
        ua_c = pf_bot - 0.004
        hang_hgt = 0.008 - (ua_c - 0.004)
        m_hang = 0.010 * 0.024 * hang_hgt * self.alu_density
        h_hang = -(0.008 + (ua_c - 0.004)) / 2
        ua_len = (self.arm + self.pan_w / 2) - (bx - 0.010)
        m_ua = ua_len * 0.024 * 0.008 * self.alu_density
        m_pan = self.pan_w * self.pan_w * self.pan_floor_t * self.alu_density
        m_rim = 0.008 * (self.pan_w + 0.016) * self.rim_h * self.alu_density
        m_post = (2 * self.post_half) ** 2 * self.post_h * self.alu_density
        z1 = pf_top + self.post_h                       # shaft top (near +0.007)
        m_s1 = 0.020 * 0.020 * self.step_h * self.alu_density
        m_s2 = 0.013 * 0.013 * self.step_h * self.alu_density
        coef_beam = 2 * (m_hang * h_hang + m_ua * (-ua_c)
                         + m_pan * (self.pan_drop + self.pan_floor_t / 2)
                         + 4 * m_rim * (self.pan_drop - self.rim_h / 2 + 0.001)
                         + m_post * (self.pan_drop - self.post_h / 2)
                         - m_s1 * (z1 + self.step_h / 2)
                         - m_s2 * (z1 + 1.5 * self.step_h))
        self.coef_beam = coef_beam
        ring_h0 = self.pan_drop - self.ring_t / 2       # first ring CoM below pivot
        pitch = self.ring_t + self.stack_gap
        # (1) one-ring imbalance presses the stop: worst case k=4 vs n=3 (max
        # restoring per unit imbalance)
        def stack_coef(n: int) -> float:
            return sum(m * (ring_h0 - i * pitch) for i in range(n))
        stop = math.radians(self.stop_deg)
        c_max = g * (coef_beam + stack_coef(4) + stack_coef(3))
        drive = m * g * self.arm * math.cos(stop)
        assert drive - c_max * math.sin(stop) > 0.10, \
            (drive, c_max * math.sin(stop))
        # (2) worst-case balanced bias (all rings shifted bore-slack the same
        # way, both stacks full) PLUS the measured hands-off settle overshoot
        # (~2.6 deg decaying swing past the rest point, forge seed-0 telemetry)
        # stays inside the success band — a genuinely balanced beam can never
        # graze outside it while settling
        slack = self.bore_half - self.post_half
        c_bal = g * (coef_beam + 2 * stack_coef(4))
        bias = math.degrees(math.atan2(8 * m * g * slack, c_bal))
        assert bias + 2.8 < self.succ_tol_deg, (bias, self.succ_tol_deg)
        # (3) funnel steps: bore clears both steps and the shaft with real slack
        # AND the slack clears the speculative-contact envelope (PhysX holds an
        # aperture crossing at contact_offset_A + contact_offset_B standoff, so
        # per-side slack must exceed 2*contact_offset with margin or the ring
        # deterministically floats at the shaft entrance)
        assert slack >= 0.0015 and slack > 2 * self.contact_offset + 0.001, slack
        assert self.bore_half - 0.010 > 2 * self.contact_offset + 0.001  # step1
        assert 2 * self.bore_half > 0.020 + 0.004 and 2 * self.bore_half > 0.013 + 0.008
        # (4) the post retains a full stack: shaft top stands proud of 5 rings
        assert self.post_h > 5 * pitch + 0.004
        # (5) capture window: a threaded ring is inside; the funnel hover is not
        assert self.capture_xy > slack + 0.010            # threaded ring xy always in
        assert self.capture_z[1] < self.pan_drop + 0.001  # hover above tip never in
        assert self.capture_z[0] < -self.pan_drop + 0.002
        # (6) nothing can prop the pans: max ring-stack height on the floor is
        # far below the lowest pan-bottom excursion at the stop
        pan_bot = self.pivot_z - self.arm * math.sin(stop) - self.pan_drop - 0.02
        assert 5 * self.ring_t + 0.02 < pan_bot, pan_bot
        # (7) jaw feasibility: the grasped band fits a Franka parallel jaw
        assert band < 0.06 and self.ring_t < 0.06
        # (8) joint travel far from the PhysX +/-180 wrap
        assert 2 * self.stop_deg <= 175.0
        # (9) rubric weights fill the non-success cap
        assert abs(self.w_lift + self.w_load - 0.60) < 1e-9
        # (11) OPEN descent corridor: no beam geometry (bar end, hanger drop)
        # intrudes into the vertical prism swept by ANY admissible ring pose
        # over a post — worst case is the 45deg-yawed outer half-diagonal
        assert (self.arm - self.ring_outer * math.sqrt(2) / 2) - bx >= 0.008, bx
        # and the under-arm rides fully BELOW the pan floor (never in the pocket)
        assert ua_c + 0.004 <= pf_bot + 1e-9
        # (10) scatter slots cannot overlap: slot spacing minus jitter exceeds
        # the worst diagonal-to-diagonal ring contact distance
        a0, a1 = (math.radians(v) for v in self.scatter_arc)
        spacing = self.scatter_r * abs(a1 - a0) / max(self.n_gold - 1, 1)
        diag = self.ring_outer * math.sqrt(2.0)
        assert spacing - 2 * self.ring_jitter > diag + 0.005, spacing


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


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("ring_balance")
class RingBalanceScene(BaseScene):
    cfg: RingBalanceSceneCfg

    def __init__(self, cfg: RingBalanceSceneCfg | None = None) -> None:
        super().__init__(cfg or RingBalanceSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        stand_spawn = cls["stand"](stand_mass=c.stand_mass, pivot_z=c.pivot_z,
                                   contact_offset=c.contact_offset)
        beam_spawn = cls["beam"](
            arm=c.arm, pivot_z=c.pivot_z, pan_drop=c.pan_drop, pan_w=c.pan_w,
            pan_floor_t=c.pan_floor_t, rim_h=c.rim_h, post_half=c.post_half,
            post_h=c.post_h, step_h=c.step_h, stop_deg=c.stop_deg,
            beam_ang_damping=c.beam_ang_damping, alu_density=c.alu_density,
            contact_offset=c.contact_offset)

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
            "stand": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Stand",
                spawn=stand_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(*c.stand_pos, 0.0)),
            ),
            # the beam MUST spawn consistent with its authored joint frames
            # (stand at nominal, beam at angle 0 on the axle)
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam",
                spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_pos[0], c.stand_pos[1], c.pivot_z)),
            ),
        }
        for i in range(c.n_gold):
            spawn = cls["ring"](ring_outer=c.ring_outer, bore_half=c.bore_half,
                                ring_t=c.ring_t, steel_density=c.steel_density,
                                ring_color=c.gold_color,
                                contact_offset=c.contact_offset)
            out[f"gold_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gold_" + str(i),
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.5 + 0.12 * i, -0.6, c.ring_t / 2 + 0.002)),
            )
        for i in range(c.n_black):
            spawn = cls["ring"](ring_outer=c.ring_outer, bore_half=c.bore_half,
                                ring_t=c.ring_t, steel_density=c.steel_density,
                                ring_color=c.black_color,
                                contact_offset=c.contact_offset)
            out[f"black_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Black_" + str(i),
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.park_pos[0] + 0.12 * i, c.park_pos[1],
                         c.ring_t / 2 + 0.002)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                # solve/smoke drive rings via set_external_force_and_torque;
                # without this flag wrenches are under-applied across TGS iterations
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.stand: RigidObject = env.iscene["stand"]
        self.beam: RigidObject = env.iscene["beam"]
        self.gold: list[RigidObject] = [env.iscene[f"gold_{i}"] for i in range(c.n_gold)]
        self.black: list[RigidObject] = [env.iscene[f"black_{i}"] for i in range(c.n_black)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.ballast_sign = torch.ones(n, device=dev)      # +1: ballast on the +x pan
        self.k_count = torch.ones(n, dtype=torch.long, device=dev)
        self.black_active = torch.zeros(n, c.n_black, dtype=torch.bool, device=dev)
        # latches (partial credit survives transients; success is judged live+streak)
        self._lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        self._load_max = torch.zeros(n, device=dev)
        self._streak = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the stand (yaw + xy jitter) and the beam ON ITS AXLE
        pre-tipped toward the sampled ballast side in the same write (the jointed
        linkage always moves together), thread k black rings onto the ballast post
        (0.6 mm stack gaps — depenetration hygiene), park the inactive black rings
        off-field, scatter the gold rings on their floor arc, clear the latches."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.stand_yaw_deg)
        q_s = _qz(yaw)
        sp = torch.zeros(m, 3, device=dev)
        sp[:, 0] = c.stand_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        sp[:, 1] = c.stand_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.stand_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + origin
        st[:, 3:7] = q_s
        self.stand.write_root_state_to_sim(st, env_ids)

        # ballast side and count (torch.rand draws — first-randint-after-seed is
        # degenerate); beam written pre-tipped 1 deg inside the ballast-side stop
        side_p = torch.rand(m, device=dev) < 0.5
        sgn = torch.where(side_p, torch.ones(m, device=dev), -torch.ones(m, device=dev))
        self.ballast_sign[env_ids] = sgn
        span = c.k_max - c.k_min + 1
        k = c.k_min + (torch.rand(m, device=dev) * span * 0.99999).floor().long()
        self.k_count[env_ids] = k

        theta0 = sgn * math.radians(c.stop_deg - 1.0)
        q_b = _qmul(q_s, _qy(theta0))
        axle = torch.zeros(m, 3, device=dev)
        axle[:, 2] = c.pivot_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = sp + quat_apply(q_s, axle) + origin
        st[:, 3:7] = q_b
        self.beam.write_root_state_to_sim(st, env_ids)

        # black rings: the first k of each env threaded on the ballast post,
        # the rest parked off-field on the ground
        pitch = c.ring_t + c.stack_gap
        beam_pos = st[:, 0:3].clone()
        for i, body in enumerate(self.black):
            active = k > i
            self.black_active[env_ids, i] = active
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = sgn * c.arm
            loc[:, 2] = -c.pan_drop + c.ring_t / 2 + 0.001 + i * pitch
            on_post = beam_pos + quat_apply(q_b, loc)
            park = torch.zeros(m, 3, device=dev)
            park[:, 0] = c.park_pos[0] + 0.12 * i
            park[:, 1] = c.park_pos[1]
            park[:, 2] = c.ring_t / 2 + 0.002
            park += origin
            stb = torch.zeros(m, 13, device=dev)
            stb[:, 0:3] = torch.where(active.unsqueeze(1), on_post, park)
            stb[:, 3:7] = torch.where(active.unsqueeze(1), q_b,
                                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
                                      .expand(m, 4))
            body.write_root_state_to_sim(stb, env_ids)

        # gold rings: floor scatter arc in front of the stand, slot + jitter + yaw
        a0, a1 = (math.radians(v) for v in c.scatter_arc)
        for i, body in enumerate(self.gold):
            frac = i / max(c.n_gold - 1, 1)
            ang = yaw + a0 + (a1 - a0) * frac
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = sp[:, 0] + c.scatter_r * torch.cos(ang) \
                + (torch.rand(m, device=dev) * 2 - 1) * c.ring_jitter
            loc[:, 1] = sp[:, 1] + c.scatter_r * torch.sin(ang) \
                + (torch.rand(m, device=dev) * 2 - 1) * c.ring_jitter
            loc[:, 2] = c.ring_t / 2 + 0.002
            stg = torch.zeros(m, 13, device=dev)
            stg[:, 0:3] = loc + origin
            stg[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            body.write_root_state_to_sim(stg, env_ids)

        self._lifted[env_ids] = False
        self._load_max[env_ids] = 0.0
        self._streak[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "stand": self.stand.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "gold": [b.data.root_state_w[env_ids].clone() for b in self.gold],
            "black": [b.data.root_state_w[env_ids].clone() for b in self.black],
            "ballast_sign": self.ballast_sign[env_ids].clone(),
            "k_count": self.k_count[env_ids].clone(),
            "black_active": self.black_active[env_ids].clone(),
            "lifted": self._lifted[env_ids].clone(),
            "load_max": self._load_max[env_ids].clone(),
            "streak": self._streak[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.stand.write_root_state_to_sim(state["stand"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for b, s in zip(self.gold, state["gold"]):
            b.write_root_state_to_sim(s, env_ids)
        for b, s in zip(self.black, state["black"]):
            b.write_root_state_to_sim(s, env_ids)
        self.ballast_sign[env_ids] = state["ballast_sign"]
        self.k_count[env_ids] = state["k_count"]
        self.black_active[env_ids] = state["black_active"]
        self._lifted[env_ids] = state["lifted"]
        self._load_max[env_ids] = state["load_max"]
        self._streak[env_ids] = state["streak"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A BEAM BALANCE stands on the floor: a heavy dark-blue pedestal whose "
            "forked column carries, on a horizontal axle, a swinging brass BEAM "
            "with a square-rimmed steel PAN hanging from each end. Each pan has a "
            "square centre POST (24 mm) with a stepped, self-centering tip. The "
            f"axle has hard stops at +/-{c.stop_deg:.0f} degrees; a level beam "
            "reads 0. On ONE pan — which side varies per episode — a stack of 1 "
            "to 4 heavy BLACK square rings sits threaded on the post (the "
            "ballast), so the beam rests tipped down onto that side's stop, "
            "leaving the other pan raised and EMPTY. Five identical GOLD square "
            f"rings ({c.ring_outer * 1000:.0f} mm outer, "
            f"{2 * c.bore_half * 1000:.0f} mm square bore, "
            f"{c.ring_t * 1000:.0f} mm thick — each weighing exactly the same as "
            "one black ring) lie scattered flat on the floor in front of the "
            "balance. The pedestal's position and heading, the ballast side and "
            "count, and the gold rings' places all vary per episode.\n"
            "Goal: counterweight the ballast EXACTLY. Pick up gold rings one at a "
            "time and thread each over the EMPTY pan's post (the stepped tip "
            "funnels the 30 mm bore over the 24 mm shaft) until both pans carry "
            "equal weight and the beam floats LEVEL — settled within about "
            f"{c.succ_tol_deg:g} degrees of horizontal. Count the black stack, "
            "or use the beam itself as feedback: with too few gold rings it stays "
            "pressed on the ballast-side stop, with one too many it tips onto "
            "the OPPOSITE stop; only the exact count balances. The black stack "
            "must stay on its post untouched — unloading the ballast instead of "
            "counterweighting it fails. Success: the beam level and still with "
            "at least one gold ring threaded on the free post and the black "
            "stack intact."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Balance the beam scale: thread gold rings onto the empty pan's post, "
            "one for each black ring stacked on the loaded pan, until the beam "
            "rests level. Do not disturb the black stack; too many or too few "
            "gold rings leaves the beam tipped on a stop and fails."
        )

    # ----- frames / live predicates -------------------------------------------------------------
    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  pos_w - self.beam.data.root_pos_w)

    def beam_angle_deg(self) -> torch.Tensor:
        """(N,) float: beam angle about the axle in degrees (0 = level; positive =
        the +x pan is DOWN). No joint-state API exists on a plain spawn-authored
        USD joint; this is the axle readout from the stand-relative quaternion."""
        qs = self.stand.data.root_quat_w
        qb = self.beam.data.root_quat_w
        qs_inv = qs * torch.tensor([1.0, -1.0, -1.0, -1.0], device=qs.device)
        rel = _qmul(qs_inv, qb)
        ang = torch.rad2deg(2.0 * torch.atan2(rel[:, 2], rel[:, 0]))
        ang = torch.where(ang > 180.0, ang - 360.0, ang)
        ang = torch.where(ang < -180.0, ang + 360.0, ang)
        return ang

    def _on_post(self, body, sgn: torch.Tensor) -> torch.Tensor:
        """(N,) bool: `body`'s centre inside the post capture cylinder of the pan
        at beam-local x = sgn*arm (xy window wider than the bore slack, z window
        from the pan floor to just under the funnel tip — a hover above the tip
        never counts)."""
        c = self.cfg
        loc = self._beam_local(body.data.root_pos_w)
        dx = loc[:, 0] - sgn * c.arm
        return (dx.abs() < c.capture_xy) & (loc[:, 1].abs() < c.capture_xy) \
            & (loc[:, 2] > c.capture_z[0]) & (loc[:, 2] < c.capture_z[1])

    def n_free(self) -> torch.Tensor:
        """(N,) long: GOLD rings captured on the FREE pan's post. Gold-only by
        design: the described success demands gold — a black ring smuggled onto
        the free post carries weight (the beam levels) but earns nothing."""
        sgn = -self.ballast_sign
        cnt = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)
        for body in self.gold:
            cnt = cnt + self._on_post(body, sgn).long()
        return cnt

    def ballast_intact(self) -> torch.Tensor:
        """(N,) bool: every ACTIVE black ring still captured on the ballast post."""
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for i, body in enumerate(self.black):
            on = self._on_post(body, self.ballast_sign)
            ok = ok & (on | ~self.black_active[:, i])
        return ok

    def _finite(self) -> torch.Tensor:
        bodies = [self.stand, self.beam, *self.gold, *self.black]
        p = torch.stack([b.data.root_pos_w for b in bodies], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _settled(self) -> torch.Tensor:
        """(N,) bool: beam angular rate and every ring/stand linear rate small."""
        c = self.cfg
        ok = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.beam_settle_avel
        ok = ok & (self.stand.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        for body in (*self.gold, *self.black):
            ok = ok & (body.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return ok

    def _raw_success(self) -> torch.Tensor:
        c = self.cfg
        return (self.beam_angle_deg().abs() <= c.succ_tol_deg) & self._settled() \
            & self.ballast_intact() & (self.n_free() >= 1) & self._finite()

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the latches ONCE per physics step."""
        c = self.cfg
        fin = self._finite()
        gz = torch.stack([b.data.root_pos_w[:, 2] for b in self.gold], dim=1) \
            - self.env_origins[:, 2].unsqueeze(1)
        self._lifted |= (gz > c.lift_z).any(dim=1) & fin
        frac = torch.minimum(self.n_free(), self.k_count).float() \
            / self.k_count.float()
        self._load_max = torch.where(fin, torch.maximum(self._load_max, frac),
                                     self._load_max)
        raw = self._raw_success()
        self._streak = torch.where(raw, self._streak + 1,
                                   torch.zeros_like(self._streak))

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the beam LEVEL (within succ_tol_deg) and settled for a
        continuous succ_streak steps (the streak gate rejects a beam swinging
        through level), with the black ballast stack intact on its post and at
        least one ring threaded on the free post. Judged live: removing a ring
        or re-tipping the beam revokes it."""
        return self._raw_success() & (self._streak >= self.cfg.succ_streak)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10*lifted + 0.50*load_frac (latched; ~0 for the
        null policy — the beam rests pinned on its ballast stop and no gold ring
        moves), capped at 0.60 — and exactly 1.0 iff success() holds."""
        c = self.cfg
        base = (c.w_lift * self._lifted.float()
                + c.w_load * self._load_max).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ring_balance", robot="null"))
