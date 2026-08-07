"""TunnelShuttleScene — pull the gate pin, slide the trapped shuttle out of the covered
tunnel by its knob, then bin it (sim_gen task `pick_single_egad_i3`).

Derived from maniskill/pick_single_egad, but STRATEGICALLY different: the seed is a
single free-space pick — grasp one loose EGAD object resting in the open and raise it
7.5 cm; the checker is a pure z-position shift and the whole plan is one grasp + one
lift. Here the target CANNOT be lifted at the start: the blue shuttle block rides
inside a covered channel (a tunnel), its only graspable feature — a square knob —
poking up through a narrow slot in the cover, and a yellow gate pin dropped through a
cross-slot blocks the channel ahead of it. A solver needs a different PLAN (extract the
gate pin vertically out of its slot FIRST, then DRAG the shuttle along the slot — a
constrained horizontal slide under contact, through the position the gate used to
occupy — until it emerges from the open end, and only then lift it and place it in the
green bin) and a different code structure (an ordered mechanism: a pin-extraction
stage, a guided-slide stage with progress measured along the fixture's randomized
axis, and a final containment predicate — not a z-shift check). The seed's own plan —
"grasp the nearest loose object and lift it" — is expressible here and rejected: a
loose red decoy cube sits in the open, and lifting it or binning it scores ~0 (smoke
control); the shuttle itself physically cannot rise more than ~8 mm while under the
cover (smoke proves this with an applied upward force).

Execution order is enforced by GEOMETRY, not by rubric fiat: the knob-in-slot makes
vertical extraction impossible inside the tunnel (the cover catches the shuttle body),
the seated gate makes forward sliding impossible (smoke proves both), and the closed
rear end leaves exactly one path. The rubric judges physical outcomes only.

success(): the shuttle rests INSIDE the green bin (bin body frame: |xy| <= bin_xy_tol,
root height <= bin_z_max — a shuttle on the rim reads ~0.056 and is rejected) and is
settled. score() is graded and latched (credit never evaporates): 0.15 * best gate-pin
extraction (gated near its slot) + 0.35 * best slide progress along the channel
(normalized by the episode's own randomized start, gated inside the channel) + 0.15 *
best approach to the bin (gated once clear of the tunnel), capped at 0.65; 0.9 once in
the bin; 1.0 iff success(). The null policy scores ~0.

Assets are fully procedural, one rigid body each (compound spawners, the bayonet-lock
pattern — child colliders of one body never self-collide):
  - fixture: KINEMATIC — floor slab, two side walls, a closed rear end wall, and four
    ORANGE cover strips forming a 24 mm slot along the channel axis; the strips are
    interrupted by a 20 mm cross-gap where the gate pin drops through. The last 60 mm
    of the channel are uncovered (the open exit).
  - shuttle: BLUE 40 mm cube + a 16 mm square knob post through the slot (top ~32 mm
    above the cover — the jaw grasp feature). 8 mm headroom under the cover: it cannot
    leave the tunnel vertically.
  - gate: YELLOW blade (16 x 46 x 86 mm) filling the channel cross-section, dropped
    through the cover gap, with a T-head above the cover to grasp (jaw across the head
    ends — the blade below is thin along that axis, so fingers pass beside it).
  - bin: KINEMATIC green open box (120 mm interior, 30 mm walls) placed at a random
    bearing off the channel exit.
  - decoy: a loose RED 40 mm cube in the open — the seed's kind of object; worthless.
Contact offsets are explicit and small (1 mm): the default ~2 cm offset would eat the
2-8 mm working clearances that make the mechanism real.

Per-episode randomization (readback-verified in smoke): fixture xy jitter + yaw (the
slide direction moves), shuttle start depth along the tunnel (the slide length
changes), bin bearing + yaw off the exit, decoy pose. Heavy imports (isaaclab, pxr)
are deferred so importing this module — and registering the scene — stays app-free.
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


# ----- custom compound spawners ---------------------------------------------------------------
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


def _box(stage, path: str, size, center, color, contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _dynamic_armor(root, mass: float) -> None:
    """Rigid-body armor for the two dynamic puzzle pieces: depenetration cap, light
    damping, and ZERO sleep/stabilization thresholds (a sleeping body silently ignores
    applied external wrenches)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC tunnel fixture at `prim_path`. Origin = channel-floor
    centre at GROUND level (z=0); the channel runs along local +x, open at +x, closed
    at -x. Cover strips leave a slot |y| <= slot_half, interrupted by the gate
    cross-gap at gate_x; the channel is uncovered beyond cover_end."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(6.0)

    ch, cw, wt, ft = cfg.chan_half, cfg.chan_w_half, cfg.wall_t, cfg.floor_t
    wh, ct, sh = cfg.wall_h, cfg.cover_t, cfg.slot_half
    co, color, ccolor = cfg.contact_offset, cfg.color, cfg.cover_color
    x_lo = -(ch + wt)  # rear outer face
    # floor slab (full footprint)
    _box(stage, f"{prim_path}/floor", (ch + wt + ch + 0.005, 2 * (cw + wt), ft),
         ((x_lo + ch + 0.005) / 2, 0.0, ft / 2), color, co)
    # side walls (rear outer face to the channel end)
    wl = ch + wt + ch
    _box(stage, f"{prim_path}/wall_yn", (wl, wt, wh),
         (x_lo + wl / 2, -(cw + wt / 2), ft + wh / 2), color, co)
    _box(stage, f"{prim_path}/wall_yp", (wl, wt, wh),
         (x_lo + wl / 2, cw + wt / 2, ft + wh / 2), color, co)
    # rear end wall (closes the channel at -x)
    _box(stage, f"{prim_path}/wall_rear", (wt, 2 * cw, wh),
         (-(ch + wt / 2), 0.0, ft + wh / 2), color, co)
    # cover strips: two x-segments (split by the gate cross-gap) on each side of the slot
    zc = ft + wh + ct / 2
    strip_w = cw + wt - sh
    cy = sh + strip_w / 2
    seg_a_lo, seg_a_hi = x_lo, cfg.gate_x - cfg.gate_gap_half
    seg_b_lo, seg_b_hi = cfg.gate_x + cfg.gate_gap_half, cfg.cover_end
    for tag, lo, hi in (("a", seg_a_lo, seg_a_hi), ("b", seg_b_lo, seg_b_hi)):
        for side, sgn in (("yn", -1.0), ("yp", 1.0)):
            _box(stage, f"{prim_path}/cover_{tag}_{side}", (hi - lo, strip_w, ct),
                 ((lo + hi) / 2, sgn * cy, zc), ccolor, co)
    return root


def _spawn_shuttle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the shuttle at `prim_path`: one rigid body = 40 mm base cube + a square
    knob post on top (the through-the-slot grasp feature). Origin = base centre."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _dynamic_armor(root, cfg.mass_props.mass)
    s, kw, kh = cfg.base_size, cfg.knob_w, cfg.knob_h
    _box(stage, f"{prim_path}/base", (s, s, s), (0.0, 0.0, 0.0),
         cfg.color, cfg.contact_offset)
    _box(stage, f"{prim_path}/knob", (kw, kw, kh), (0.0, 0.0, s / 2 + kh / 2),
         cfg.knob_color, cfg.contact_offset)
    return root


def _spawn_gate(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gate pin at `prim_path`: one rigid body = thin blade filling the
    channel cross-section + a T-head above the cover (jaw closes across the head's
    long-axis end faces; the blade below is thin along that axis, so the fingers pass
    beside it). Origin = blade centre."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    _dynamic_armor(root, cfg.mass_props.mass)
    t, w, h = cfg.blade_t, cfg.blade_w, cfg.blade_h
    _box(stage, f"{prim_path}/blade", (t, w, h), (0.0, 0.0, 0.0),
         cfg.color, cfg.contact_offset)
    _box(stage, f"{prim_path}/head", (cfg.head_l, cfg.head_w, cfg.head_h),
         (0.0, 0.0, h / 2 + cfg.head_h / 2), cfg.head_color, cfg.contact_offset)
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC open goal bin at `prim_path`. Origin = bin-floor centre at
    GROUND level."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(2.0)
    ih, wt, wh, ft = cfg.inner_half, cfg.bin_wall_t, cfg.bin_wall_h, cfg.bin_floor_t
    co, color = cfg.contact_offset, cfg.color
    outer = 2 * (ih + wt)
    _box(stage, f"{prim_path}/floor", (outer, outer, ft), (0.0, 0.0, ft / 2), color, co)
    for tag, cx, cy, sx, sy in (
        ("xn", -(ih + wt / 2), 0.0, wt, outer),
        ("xp", ih + wt / 2, 0.0, wt, outer),
        ("yn", 0.0, -(ih + wt / 2), 2 * ih, wt),
        ("yp", 0.0, ih + wt / 2, 2 * ih, wt),
    ):
        _box(stage, f"{prim_path}/wall_{tag}", (sx, sy, wh),
             (cx, cy, ft + wh / 2), color, co)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the four compound-spawner cfg classes (the bayonet-lock pattern:
    explicit @configclass subclasses of RigidObjectSpawnerCfg, defined lazily so the
    module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class TunnelFixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            chan_half: float = 0.11
            chan_w_half: float = 0.025
            wall_t: float = 0.010
            floor_t: float = 0.006
            wall_h: float = 0.048
            cover_t: float = 0.008
            slot_half: float = 0.012
            cover_end: float = 0.05
            gate_x: float = -0.02
            gate_gap_half: float = 0.010
            color: tuple = (0.35, 0.35, 0.38)
            cover_color: tuple = (0.90, 0.45, 0.10)
            contact_offset: float = 0.001

        @configclass
        class ShuttleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shuttle)
            base_size: float = 0.040
            knob_w: float = 0.016
            knob_h: float = 0.048
            color: tuple = (0.15, 0.30, 0.85)
            knob_color: tuple = (0.25, 0.50, 0.95)
            contact_offset: float = 0.001

        @configclass
        class GatePinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gate)
            blade_t: float = 0.016
            blade_w: float = 0.046
            blade_h: float = 0.086
            head_l: float = 0.032
            head_w: float = 0.016
            head_h: float = 0.020
            color: tuple = (0.92, 0.80, 0.12)
            head_color: tuple = (0.98, 0.88, 0.25)
            contact_offset: float = 0.001

        @configclass
        class GoalBinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            inner_half: float = 0.060
            bin_wall_t: float = 0.008
            bin_wall_h: float = 0.030
            bin_floor_t: float = 0.006
            color: tuple = (0.15, 0.60, 0.20)
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(fixture=TunnelFixtureSpawnerCfg, shuttle=ShuttleSpawnerCfg,
                              gate=GatePinSpawnerCfg, goalbin=GoalBinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class TunnelShuttleSceneCfg(BaseCfg):
    """Config for `TunnelShuttleScene`. The strategic honesty knobs are geometric and
    asserted in `__post_init__`: the knob fits the slot but the base does NOT fit the
    gate gap (no vertical escape), the shuttle has real side/top clearance (the slide
    is feasible), the seated gate spans the channel (the block is real), and the bin's
    height threshold rejects a shuttle perched on the rim."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    bin_xy_tol: float = tunable(0.045)  # shuttle root within this of the bin axis (bin frame)
    bin_z_max: float = tunable(0.042)  # shuttle root height above bin origin: inside, not rim
    settle_lin: float = tunable(0.05)  # max |lin vel| when judging success (m/s)
    settle_ang: float = tunable(0.5)  # max |ang vel| when judging success (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fixture_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the fixture at reset (m)
    fixture_yaw_deg: float = tunable(25.0)  # uniform +/- fixture yaw at reset (deg)
    start_x_range: tuple = tunable((-0.095, -0.055))  # shuttle start x, fixture frame (m)
    bin_radius: float = tunable(0.18)  # bin centre distance from the channel exit (m)
    bin_angle_deg: float = tunable(45.0)  # uniform +/- bin bearing off the channel axis (deg)
    bin_yaw_deg: float = tunable(180.0)  # uniform +/- bin yaw at reset (free)
    decoy_jitter: float = tunable(0.02)  # uniform +/- xy jitter of the red decoy (m)

    # --- tunable: placement ------------------------------------------------------------------
    fixture_pos: tuple = tunable((0.0, 0.0))  # fixture origin, nominal
    decoy_local: tuple = tunable((0.05, -0.14))  # decoy spot, fixture frame

    # --- info: fixture structure -------------------------------------------------------------
    chan_half: float = info(0.11)  # channel interior half-length (x: -chan_half..+chan_half)
    chan_w_half: float = info(0.025)  # channel interior half-width
    wall_t: float = info(0.010)
    floor_t: float = info(0.006)  # channel floor top above ground
    wall_h: float = info(0.048)  # wall height above the channel floor
    cover_t: float = info(0.008)
    slot_half: float = info(0.012)  # cover slot half-width (knob passage)
    cover_end: float = info(0.05)  # cover extends to this x; beyond is the open exit
    gate_x: float = info(-0.02)  # gate cross-gap centre, fixture frame
    gate_gap_half: float = info(0.010)  # cover cross-gap half-width along x
    fixture_color: tuple = info((0.35, 0.35, 0.38))
    cover_color: tuple = info((0.90, 0.45, 0.10))
    # --- info: shuttle -----------------------------------------------------------------------
    base_size: float = info(0.040)  # shuttle base cube edge
    knob_w: float = info(0.016)  # knob square cross-section (through the 24 mm slot)
    knob_h: float = info(0.048)  # knob height above the base top
    shuttle_mass: float = info(0.10)
    shuttle_color: tuple = info((0.15, 0.30, 0.85))
    knob_color: tuple = info((0.25, 0.50, 0.95))
    # --- info: gate pin ----------------------------------------------------------------------
    blade_t: float = info(0.016)  # blade thickness along the channel axis
    blade_w: float = info(0.046)  # blade width across the channel (spans it)
    blade_h: float = info(0.086)  # blade height (floor to above the cover)
    head_l: float = info(0.032)  # T-head length (grasp across these end faces)
    head_w: float = info(0.016)
    head_h: float = info(0.020)
    gate_mass: float = info(0.05)
    gate_color: tuple = info((0.92, 0.80, 0.12))
    gate_head_color: tuple = info((0.98, 0.88, 0.25))
    gate_park_local: tuple = info((-0.10, -0.16))  # where solve parks the extracted pin
    # --- info: bin + decoy -------------------------------------------------------------------
    bin_inner_half: float = info(0.060)
    bin_wall_t: float = info(0.008)
    bin_wall_h: float = info(0.030)
    bin_floor_t: float = info(0.006)
    bin_color: tuple = info((0.15, 0.60, 0.20))
    decoy_size: float = info(0.040)
    decoy_mass: float = info(0.08)
    decoy_color: tuple = info((0.85, 0.15, 0.12))
    # Explicit small offsets: the ~2 cm default would eat the 2-8 mm working clearances.
    contact_offset: float = info(0.001)

    # Derived (filled in __post_init__).
    cover_top: float = field(default=None, init=False)  # cover top above ground (m)
    wall_top: float = field(default=None, init=False)
    shuttle_root_z: float = field(default=None, init=False)  # base centre when on the floor
    gate_root_z: float = field(default=None, init=False)  # blade centre when seated
    exit_x: float = field(default=None, init=False)  # slide target: shuttle clear of the cover

    def __post_init__(self) -> None:
        self.wall_top = self.floor_t + self.wall_h
        self.cover_top = self.wall_top + self.cover_t
        self.shuttle_root_z = self.floor_t + self.base_size / 2
        self.gate_root_z = self.floor_t + self.blade_h / 2
        self.exit_x = self.cover_end + self.base_size / 2 + 0.012
        # -- the mechanism must be real (geometry asserts) --
        assert 2 * self.chan_w_half - self.base_size >= 0.008, "shuttle needs side clearance"
        assert 2 * self.slot_half - self.knob_w >= 0.006, "knob must pass the slot"
        assert self.wall_h - self.base_size >= 0.006, "shuttle needs headroom under the cover"
        knob_top = self.floor_t + self.base_size + self.knob_h
        assert knob_top - self.cover_top >= 0.025, "knob must protrude enough to grasp"
        assert 2 * self.gate_gap_half - self.blade_t >= 0.003, "gate must drop through its gap"
        assert 2 * self.gate_gap_half < self.base_size, "shuttle must NOT fit up the gate gap"
        assert 2 * self.chan_w_half - self.blade_w >= 0.003, "gate must drop between the walls"
        assert self.blade_w > self.base_size, "seated gate must span the shuttle's path"
        assert self.floor_t + self.blade_h - self.cover_top >= 0.02, "gate head must stand proud"
        hi = self.start_x_range[1]
        assert hi + self.base_size / 2 <= self.gate_x - self.blade_t / 2 - 0.005, \
            "shuttle spawn must sit clear behind the seated gate"
        assert self.exit_x + self.base_size / 2 <= self.chan_half, "exit target inside the channel"
        rim_root = self.bin_floor_t + self.bin_wall_h + self.base_size / 2
        assert rim_root > self.bin_z_max + 0.01, "bin_z_max must reject a shuttle on the rim"
        assert 2 * self.bin_inner_half >= self.base_size + self.knob_h + 0.02, \
            "a toppled shuttle must still fit inside the bin"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("tunnel_shuttle")
class TunnelShuttleScene(BaseScene):
    cfg: TunnelShuttleSceneCfg

    def __init__(self, cfg: TunnelShuttleSceneCfg | None = None) -> None:
        super().__init__(cfg or TunnelShuttleSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        fixture_cls, shuttle_cls = spawners["fixture"], spawners["shuttle"]
        gate_cls, bin_cls = spawners["gate"], spawners["goalbin"]

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
            "fixture": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Fixture",
                spawn=fixture_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=6.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    chan_half=c.chan_half, chan_w_half=c.chan_w_half, wall_t=c.wall_t,
                    floor_t=c.floor_t, wall_h=c.wall_h, cover_t=c.cover_t,
                    slot_half=c.slot_half, cover_end=c.cover_end, gate_x=c.gate_x,
                    gate_gap_half=c.gate_gap_half, color=c.fixture_color,
                    cover_color=c.cover_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fixture_pos[0], c.fixture_pos[1], 0.0)),
            ),
            "shuttle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shuttle",
                spawn=shuttle_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.shuttle_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    base_size=c.base_size, knob_w=c.knob_w, knob_h=c.knob_h,
                    color=c.shuttle_color, knob_color=c.knob_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fixture_pos[0] - 0.08, c.fixture_pos[1], c.shuttle_root_z)),
            ),
            "gate": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gate",
                spawn=gate_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.gate_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    blade_t=c.blade_t, blade_w=c.blade_w, blade_h=c.blade_h,
                    head_l=c.head_l, head_w=c.head_w, head_h=c.head_h,
                    color=c.gate_color, head_color=c.gate_head_color,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fixture_pos[0] + c.gate_x, c.fixture_pos[1], c.gate_root_z)),
            ),
            "goal_bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/GoalBin",
                spawn=bin_cls(
                    mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    inner_half=c.bin_inner_half, bin_wall_t=c.bin_wall_t,
                    bin_wall_h=c.bin_wall_h, bin_floor_t=c.bin_floor_t,
                    color=c.bin_color, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fixture_pos[0] + c.chan_half + c.bin_radius,
                         c.fixture_pos[1], 0.0)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Decoy",
                spawn=sim_utils.CuboidCfg(
                    size=(c.decoy_size, c.decoy_size, c.decoy_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.decoy_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.decoy_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fixture_pos[0] + c.decoy_local[0],
                         c.fixture_pos[1] + c.decoy_local[1], c.decoy_size / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.fixture: RigidObject = env.iscene["fixture"]
        self.shuttle: RigidObject = env.iscene["shuttle"]
        self.gate: RigidObject = env.iscene["gate"]
        self.goal_bin: RigidObject = env.iscene["goal_bin"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.start_x = torch.full((n,), -0.08, device=dev)  # shuttle start, fixture frame
        self.d_ref = torch.full((n,), 0.3, device=dev)  # exit->bin distance at reset
        self.gate_latch = torch.zeros(n, device=dev)
        self.slide_latch = torch.zeros(n, device=dev)
        self.approach_latch = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: fixture with xy jitter + yaw (the slide direction moves),
        gate seated in its cross-gap, shuttle at a random depth behind the gate, bin at
        a random bearing off the exit, decoy in the open; latches zeroed and the
        approach baseline `d_ref` captured from the sampled poses."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        fx = torch.tensor(c.fixture_pos, device=dev).expand(m, 2).clone()
        fx += (torch.rand(m, 2, device=dev) * 2 - 1) * c.fixture_jitter
        fyaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fixture_yaw_deg)
        cosy, siny = torch.cos(fyaw), torch.sin(fyaw)

        def to_world(x_l: torch.Tensor, y_l: torch.Tensor) -> torch.Tensor:
            """(m,) fixture-frame xy -> (m, 2) world xy."""
            return torch.stack([fx[:, 0] + cosy * x_l - siny * y_l,
                                fx[:, 1] + siny * x_l + cosy * y_l], dim=-1)

        def write(body, xy: torch.Tensor, z: float, yaw: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zero = torch.zeros(m, device=dev)
        write(self.fixture, fx, 0.0, fyaw)
        # gate seated in its cross-gap
        write(self.gate, to_world(torch.full((m,), c.gate_x, device=dev), zero),
              c.gate_root_z, fyaw)
        # shuttle at a random depth behind the gate, aligned with the channel
        lo, hi = c.start_x_range
        sx = lo + torch.rand(m, device=dev) * (hi - lo)
        write(self.shuttle, to_world(sx, zero), c.shuttle_root_z, fyaw)
        # bin at a random bearing off the channel exit, free yaw
        ang = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bin_angle_deg)
        bx = c.chan_half + c.bin_radius * torch.cos(ang)
        by = c.bin_radius * torch.sin(ang)
        bin_xy = to_world(bx, by)
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.bin_yaw_deg)
        write(self.goal_bin, bin_xy, 0.0, byaw)
        # decoy in the open, with jitter + free yaw
        dx = torch.full((m,), c.decoy_local[0], device=dev) \
            + (torch.rand(m, device=dev) * 2 - 1) * c.decoy_jitter
        dy = torch.full((m,), c.decoy_local[1], device=dev) \
            + (torch.rand(m, device=dev) * 2 - 1) * c.decoy_jitter
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.decoy, to_world(dx, dy), c.decoy_size / 2 + 0.002, dyaw)

        # baselines + latches
        self.start_x[env_ids] = sx
        exit_xy = to_world(torch.full((m,), c.chan_half, device=dev), zero)
        self.d_ref[env_ids] = (bin_xy - exit_xy).norm(dim=-1).clamp(min=0.05)
        self.gate_latch[env_ids] = 0.0
        self.slide_latch[env_ids] = 0.0
        self.approach_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "fixture": self.fixture.data.root_state_w[env_ids].clone(),
            "shuttle": self.shuttle.data.root_state_w[env_ids].clone(),
            "gate": self.gate.data.root_state_w[env_ids].clone(),
            "goal_bin": self.goal_bin.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "start_x": self.start_x[env_ids].clone(),
            "d_ref": self.d_ref[env_ids].clone(),
            "gate_latch": self.gate_latch[env_ids].clone(),
            "slide_latch": self.slide_latch[env_ids].clone(),
            "approach_latch": self.approach_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.fixture.write_root_state_to_sim(state["fixture"], env_ids)
        self.shuttle.write_root_state_to_sim(state["shuttle"], env_ids)
        self.gate.write_root_state_to_sim(state["gate"], env_ids)
        self.goal_bin.write_root_state_to_sim(state["goal_bin"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.start_x[env_ids] = state["start_x"]
        self.d_ref[env_ids] = state["d_ref"]
        self.gate_latch[env_ids] = state["gate_latch"]
        self.slide_latch[env_ids] = state["slide_latch"]
        self.approach_latch[env_ids] = state["approach_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A gray channel fixture lies on the ground: a straight tunnel "
            f"({2 * c.chan_half * 1000:.0f} mm long inside, "
            f"{2 * c.chan_w_half * 1000:.0f} mm wide), closed at one end, OPEN at the "
            f"other, roofed by ORANGE cover strips that leave a "
            f"{2 * c.slot_half * 1000:.0f} mm slot running along the top; the last "
            f"{(c.chan_half - c.cover_end) * 1000:.0f} mm before the open end are "
            f"uncovered. Inside the tunnel sits a BLUE shuttle block "
            f"({c.base_size * 1000:.0f} mm cube) whose square blue knob sticks up "
            f"through the slot — the knob is its only reachable part while the shuttle "
            f"is under the cover, and the cover makes it impossible to lift the shuttle "
            f"out anywhere along the tunnel. Between the shuttle and the open end a "
            f"YELLOW gate pin is dropped through a gap in the cover: its blade fills "
            f"the channel cross-section and blocks the shuttle's path, and its yellow "
            f"T-head stands above the cover. A GREEN open bin sits on the ground beyond "
            f"the tunnel's open end (its bearing varies), and a loose RED cube lies "
            f"near the fixture — the red cube is a decoy and counts for nothing.\n"
            f"Goal: get the blue shuttle to rest inside the green bin. The only way: "
            f"first pull the yellow gate pin STRAIGHT UP by its T-head until its blade "
            f"clears the cover, and set it aside anywhere out of the way (not in the "
            f"bin); then grasp the blue knob and SLIDE the shuttle along the slot, "
            f"through where the gate stood, until the shuttle emerges from under the "
            f"cover at the open end; then lift it out of the channel and place it in "
            f"the green bin. Success requires the blue shuttle settled inside the bin "
            f"(resting on the bin floor, not perched on a wall). Putting the red decoy "
            f"or the yellow pin in the bin counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the yellow gate pin straight up out of the tunnel cover and set it "
            "aside, slide the blue shuttle by its knob along the slot to the tunnel's "
            "open end, then lift it out and place it inside the green bin. Only the "
            "blue shuttle in the bin counts; the red cube is a decoy."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body centre in the FIXTURE body frame (origin = channel-floor centre
        at ground level; +x = toward the open exit)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.fixture.data.root_pos_w
        return quat_apply_inverse(self.fixture.data.root_quat_w, rel)

    def _bin_local(self, body) -> torch.Tensor:
        """(N,3) body centre in the BIN body frame (origin = bin-floor centre)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.goal_bin.data.root_pos_w
        return quat_apply_inverse(self.goal_bin.data.root_quat_w, rel)

    def in_channel(self) -> torch.Tensor:
        """(N,) bool: shuttle riding inside the channel (between the walls, on the
        floor level, within the tunnel's x span) — the gate for slide credit."""
        c = self.cfg
        loc = self._local(self.shuttle)
        return (loc[:, 1].abs() <= c.chan_w_half) \
            & (loc[:, 2] <= c.wall_top) & (loc[:, 2] >= 0.0) \
            & (loc[:, 0] >= -c.chan_half - 0.01) & (loc[:, 0] <= c.chan_half + 0.01)

    def clear_of_tunnel(self) -> torch.Tensor:
        """(N,) bool: shuttle no longer confined by the cover (past the covered
        section, outside the walls, or lifted above them) — the gate for approach
        credit."""
        c = self.cfg
        loc = self._local(self.shuttle)
        return (loc[:, 0] > c.cover_end + c.base_size / 2 + 0.002) \
            | (loc[:, 1].abs() > c.chan_w_half + 0.02) | (loc[:, 2] > c.wall_top + 0.02)

    def gate_lift_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: gate-pin extraction progress — blade bottom risen from the
        channel floor to above the cover top — gated near its cross-gap (lifting the
        pin only counts where the pin actually was)."""
        c = self.cfg
        loc = self._local(self.gate)
        bottom = loc[:, 2] - c.blade_h / 2
        raw = ((bottom - c.floor_t) / (c.cover_top + 0.004 - c.floor_t)).clamp(0.0, 1.0)
        near = (loc[:, 0] - c.gate_x).abs() < 0.05
        near &= loc[:, 1].abs() < 0.06
        return raw * near.float()

    def slide_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: slide progress along the channel from the episode's own
        randomized start to the exit threshold, gated on riding inside the channel."""
        c = self.cfg
        loc = self._local(self.shuttle)
        raw = ((loc[:, 0] - self.start_x) / (c.exit_x - self.start_x)).clamp(0.0, 1.0)
        return raw * self.in_channel().float()

    def approach_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: shuttle xy approach to the bin, normalized by the episode's
        exit->bin distance, gated on being clear of the tunnel."""
        d = (self.shuttle.data.root_pos_w - self.goal_bin.data.root_pos_w)[:, :2].norm(dim=-1)
        raw = (1.0 - d / self.d_ref).clamp(0.0, 1.0)
        return raw * self.clear_of_tunnel().float()

    def in_bin(self) -> torch.Tensor:
        """(N,) bool: shuttle centre inside the bin footprint, LOW (resting on the bin
        floor — a shuttle perched on the rim reads root z ~= 0.056 and is rejected)."""
        c = self.cfg
        loc = self._bin_local(self.shuttle)
        return (loc[:, 0].abs() <= c.bin_xy_tol) & (loc[:, 1].abs() <= c.bin_xy_tol) \
            & (loc[:, 2] > 0.004) & (loc[:, 2] <= c.bin_z_max)

    def settled(self) -> torch.Tensor:
        return (self.shuttle.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (self.shuttle.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch best gate extraction, best slide progress and best bin approach each
        physics substep, so transient progress keeps its credit."""
        self.gate_latch = torch.maximum(self.gate_latch, self.gate_lift_frac())
        self.slide_latch = torch.maximum(self.slide_latch, self.slide_frac())
        self.approach_latch = torch.maximum(self.approach_latch, self.approach_frac())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the blue shuttle settled inside the green bin."""
        return self.in_bin() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 * latched gate extraction + 0.35 * latched slide
        progress + 0.15 * latched bin approach (max 0.65), 0.9 once the shuttle is in
        the bin, 1.0 iff success. Doing nothing scores ~0; the seed's plan (lift a
        loose object / bin the decoy) scores ~0."""
        base = (0.15 * self.gate_latch + 0.35 * self.slide_latch
                + 0.15 * self.approach_latch).clamp(0.0, 0.65)
        s = torch.where(self.in_bin(), torch.maximum(base, base.new_tensor(0.9)), base)
        return torch.where(self.success(), s.new_tensor(1.0), s)


register_env("simgen", lambda: EnvCfg(scene="tunnel_shuttle", robot="null"))
