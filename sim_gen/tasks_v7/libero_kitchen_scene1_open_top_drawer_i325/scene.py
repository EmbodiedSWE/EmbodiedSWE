"""RatchetRampScene — push cargo balls UPHILL through two passive one-way flap gates
into a roofed sunken catch basin
(sim_gen task `libero_kitchen_scene1_open_top_drawer_i325`).

Derived from libero_90/libero_kitchen_scene1_open_top_drawer, but STRATEGICALLY
different: the seed is one prehensile articulation act — grasp the drawer handle and
PULL the panel along its built prismatic joint until a joint readout crosses a
threshold. Here the articulated parts (two hinged pet-door FLAPS) are judged NOWHERE
as goals and moving them by hand earns NOTHING (the smoke battery torques one open,
lets it fall shut, and shows the score stays ~0). The judged outcome is uphill
TRANSPORT AGAINST GRAVITY of free cargo: two blue balls must be driven from open
ground, through the open lower mouth of an enclosed 12-degree ramp channel, THROUGH
both one-way flaps, over the crest, and into a sunken roofed catch basin where a
step retains them. The flaps are passive ratchet pawls: a ball pushed uphill shoves
each flap open and the flap falls closed behind it, so progress is CHECKPOINTED — a
ball released mid-climb rolls back only to the last flap it passed, never to the
bottom. A red decoy ball must be left OUT of the basin.

Plan-level contrast with the seed: the seed's skill is one guided translation of the
judged articulated part, terminated by a joint readout. Here no joint readout is
ever judged; the joints are unpowered checkpoint hardware the cargo itself operates
in passing, the judged bodies are free balls, the work is a sustained non-prehensile
push against gravity (the seed's pull is quasi-horizontal and ends in free space),
and the goal predicate is containment at elevation with the mechanism RE-closed.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - rig (KINEMATIC compound, world-fixed per env — the flap joints anchor to it and
    kinematic joint anchors must never be teleported): a tilted ramp slab climbing
    +x at 12 deg from a ground-level lip to a crest; vertical side walls forming one
    0.090 m channel; a roof over everything downhill... uphill of x=roof_x0 so the
    only free-space entry is the open stretch at the BOTTOM mouth; two header bars
    above the flap hinges so nothing passes over a flap; past the crest a sunken
    basin floor one ball-radius BELOW the crest (a retaining step) and an end wall.
  - flap_1 / flap_2 (DYNAMIC, one thin blade each, spawn-authored D6 hinge to the
    rig): hung from a cross-channel axis at the top, 5 mm floor gap, joint limits
    [-open_deg, +0.5 deg]: swinging UPHILL (ball passing under, negative angle) is
    free to -105 deg; swinging DOWNHILL is stopped at +0.5 deg, so a ball rolling
    back presses the flap against its stop and is held. Gravity (density-authored
    CoM below the hinge) returns each flap to hanging closed.
  - balls (DYNAMIC spheres): two BLUE cargo balls and one RED decoy, dealt to three
    ground bays downhill of the mouth by a per-episode permutation with jitter.

Per-episode randomization (readback-verifiable): which ball starts in which bay
(3-permutation) plus per-ball xy jitter. The rig cannot move (joint anchors), so all
episode variation lives in the ball deal.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  per cargo ball: 0.10 once past flap 1 (in-channel beyond it), 0.15 once past
  flap 2, 0.10 once settled inside the basin — latched, cap 0.70. Exactly 1.0 iff
  success(): both cargo balls inside the basin, the decoy NOT in the basin, both
  flaps hanging closed, everything settled and finite. Null policy scores ~0;
  waving a flap open scores ~0; the decoy latches nothing anywhere.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             orient=None, density=None):
    """One box child: translate (+ optional orient) + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if density is not None:
        # density on the CHILD gives the true CoM (MassAPI mass on the root would
        # park the CoM at the body origin = the hinge -> no gravity return).
        UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateDensityAttr(float(density))
    return box.GetPrim()


def _spawn_rig(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rig at `prim_path`: KINEMATIC compound. Local frame: origin on the
    ground at the ramp's downhill end, +x uphill, +z up. All boxes are axis-aligned
    except the tilted ramp slab (oriented by quat)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    # tilted ramp slab (top surface: z = z_lo + (x - x_lo) * tan(tilt))
    _add_box(stage, f"{prim_path}/slab", center=c.slab_center, size=c.slab_size,
             color=c.body_color, collide=collide, orient=c.slab_quat)
    # side walls
    wx = (c.wall_x0 + c.wall_x1) / 2
    for sy in (1.0, -1.0):
        t = "p" if sy > 0 else "n"
        _add_box(stage, f"{prim_path}/wall_{t}",
                 center=(wx, sy * (c.chan_w / 2 + c.wall_t / 2), c.wall_top / 2),
                 size=(c.wall_x1 - c.wall_x0, c.wall_t, c.wall_top),
                 color=c.body_color, collide=collide)
    # roof: covers everything uphill of roof_x0 (the mouth stretch stays open-top)
    _add_box(stage, f"{prim_path}/roof",
             center=((c.roof_x0 + c.wall_x1) / 2, 0.0, c.wall_top + c.roof_t / 2),
             size=(c.wall_x1 - c.roof_x0, c.chan_w + 2 * c.wall_t, c.roof_t),
             color=c.roof_color, collide=collide)
    # header bars above each flap hinge (nothing passes over a flap)
    for i, (fx, hz) in enumerate(zip(c.flap_x, c.hinge_z)):
        h0 = hz + c.header_clear
        _add_box(stage, f"{prim_path}/header_{i}",
                 center=(fx, 0.0, (h0 + c.wall_top) / 2),
                 size=(0.012, c.chan_w + 2 * c.wall_t, c.wall_top - h0),
                 color=c.roof_color, collide=collide)
    # sunken basin floor (solid to the ground) and the end wall
    _add_box(stage, f"{prim_path}/basin_floor",
             center=((c.basin_x0 + c.basin_x1) / 2, 0.0, c.basin_floor_top / 2),
             size=(c.basin_x1 - c.basin_x0, c.chan_w, c.basin_floor_top),
             color=c.body_color, collide=collide)
    _add_box(stage, f"{prim_path}/end_wall",
             center=((c.basin_x1 + c.wall_x1) / 2, 0.0, c.wall_top / 2),
             size=(c.wall_x1 - c.basin_x1, c.chan_w, c.wall_top),
             color=c.body_color, collide=collide)
    return root


def _spawn_flap(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one flap at `prim_path`: DYNAMIC body whose origin is ON the hinge
    axis, blade hanging below (density-authored CoM), plus a spawn-authored D6
    hinge to the sibling kinematic `/Rig`. Joint frame: LocalRot = q_x(-90 deg), so
    the joint Z axis is world +y; rotZ limits [-open_deg, +0.5] make it one-way."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.10)
    px.CreateSolverPositionIterationCountAttr(32)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _add_box(stage, f"{prim_path}/blade",
             center=(0.0, 0.0, -c.flap_len / 2),
             size=(c.flap_t, c.flap_w, c.flap_len),
             color=c.color, collide=collide, density=c.density)
    # hinge to the sibling rig
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.Joint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Rig"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateCollisionEnabledAttr(True)
    qx = Gf.Quatf(math.cos(-math.pi / 4), Gf.Vec3f(math.sin(-math.pi / 4), 0.0, 0.0))
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x), 0.0, float(c.hinge_z)))
    j.CreateLocalRot0Attr(qx)
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(qx)
    jp = j.GetPrim()
    for axis in ("transX", "transY", "transZ", "rotX", "rotY"):
        lim = UsdPhysics.LimitAPI.Apply(jp, axis)
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)  # low > high = locked
    lim = UsdPhysics.LimitAPI.Apply(jp, "rotZ")
    lim.CreateLowAttr(-float(c.open_deg))
    lim.CreateHighAttr(0.5)
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
            slab_center: tuple = (0.0, 0.0, 0.0)
            slab_size: tuple = (0.1, 0.1, 0.1)
            slab_quat: tuple = (1.0, 0.0, 0.0, 0.0)
            chan_w: float = 0.090
            wall_t: float = 0.012
            wall_x0: float = -0.020
            wall_x1: float = 0.617
            wall_top: float = 0.170
            roof_t: float = 0.010
            roof_x0: float = 0.115
            flap_x: tuple = (0.165, 0.305)
            hinge_z: tuple = (0.0, 0.0)
            header_clear: float = 0.006
            basin_x0: float = 0.452
            basin_x1: float = 0.605
            basin_floor_top: float = 0.070
            body_color: tuple = (0.42, 0.38, 0.30)
            roof_color: tuple = (0.55, 0.52, 0.46)
            contact_offset: float = 0.0015

        @configclass
        class FlapSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_flap)
            hinge_x: float = 0.0
            hinge_z: float = 0.0
            flap_t: float = 0.006
            flap_w: float = 0.084
            flap_len: float = 0.070
            open_deg: float = 88.0
            density: float = 500.0
            color: tuple = (0.85, 0.65, 0.10)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["rig"] = RigSpawnerCfg
        _SPAWNER_CACHE["flap"] = FlapSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class RatchetRampSceneCfg(BaseCfg):
    """Config for `RatchetRampScene`. Every opening a ball is NOT meant to pass is
    sized below the ball diameter (asserted below): flap floor gap, flap side gaps,
    flap-to-header gap. The roof starts uphill of nothing-but-the-mouth, so the only
    free-space way into the system is the open bottom mouth, and the only way into
    the basin is the full climb past both flaps."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_lin: float = tunable(0.05)      # max |lin vel| of every ball when judging (m/s)
    ball_slow: float = tunable(0.25)       # per-ball speed gate for the basin latch (m/s)
    closed_deg: float = tunable(10.0)      # flap counts as closed within this of hanging (deg)
    flap_still: float = tunable(2.0)       # max flap |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    bay_x_jit: float = tunable(0.030)      # per-ball bay x jitter (+/- m)
    bay_y_jit: float = tunable(0.015)      # per-ball bay y jitter (+/- m); < bay pitch safety

    # --- info: ramp (rig local frame: origin on the ground at the downhill lip, +x uphill) -------
    tilt_deg: float = info(12.0)
    ramp_x_lo: float = info(-0.010)        # ramp top-surface lower edge (x)
    ramp_z_lo: float = info(0.001)         # ramp top-surface height at the lower edge
    crest_x: float = info(0.4575)          # ramp top-surface upper edge (the drop lip)
    slab_t: float = info(0.024)
    # --- info: channel / roof --------------------------------------------------------------------
    chan_w: float = info(0.090)            # clear channel width (> ball diameter + 0.02)
    wall_t: float = info(0.012)
    wall_x0: float = info(-0.020)
    wall_x1: float = info(0.617)
    wall_top: float = info(0.170)          # roof underside (> crest ball top + 8 mm)
    roof_t: float = info(0.010)
    roof_x0: float = info(0.115)           # roof starts well uphill of the mouth, downhill of flap 1
    # --- info: flaps -----------------------------------------------------------------------------
    flap_x: tuple = info((0.165, 0.305))   # hinge stations
    flap_len: float = info(0.075)
    flap_t: float = info(0.006)
    flap_w: float = info(0.084)            # 3 mm side gap to each wall
    flap_gap: float = info(0.005)          # blade bottom to ramp surface
    open_deg: float = info(105.0)          # uphill swing limit (past horizontal: the
    #   blade must clear a ball at the TIP's x, where the inclined floor is higher)
    header_clear: float = info(0.006)      # hinge to header-bar underside
    flap_density: float = info(500.0)      # blade density -> ~18 g, CoM below hinge
    # --- info: basin -----------------------------------------------------------------------------
    basin_x0: float = info(0.452)          # basin floor starts under the crest lip
    basin_x1: float = info(0.605)          # interior face of the end wall
    step_h: float = info(0.030)            # crest lip above the basin floor (= ball_r: no vault)
    # --- info: balls / bays ----------------------------------------------------------------------
    ball_r: float = info(0.030)
    ball_mass: float = info(0.15)
    bay_x: float = info(-0.170)            # bay centre line (downhill of the mouth)
    bay_ys: tuple = info((-0.100, 0.0, 0.100))
    blue: tuple = info((0.15, 0.25, 0.85))
    red: tuple = info((0.75, 0.10, 0.10))
    contact_offset: float = info(0.0015)
    # --- info: rubric ----------------------------------------------------------------------------
    past_margin: float = info(0.025)       # "past flap i" = x > flap_x[i] + this
    w_p1: float = info(0.10)               # per cargo ball: past flap 1
    w_p2: float = info(0.15)               # per cargo ball: past flap 2
    w_bin: float = info(0.10)              # per cargo ball: settled in the basin
    cap: float = info(0.70)

    # Derived (filled in __post_init__).
    tan_t: float = field(default=0.0, init=False)
    crest_z: float = field(default=0.0, init=False)
    basin_floor_top: float = field(default=0.0, init=False)
    hinge_z: tuple = field(default=(), init=False)
    slab_center: tuple = field(default=(), init=False)
    slab_size: tuple = field(default=(), init=False)
    slab_quat: tuple = field(default=(), init=False)

    def floor_top(self, x: float) -> float:
        return self.ramp_z_lo + (x - self.ramp_x_lo) * self.tan_t

    def __post_init__(self) -> None:
        th = math.radians(self.tilt_deg)
        self.tan_t = math.tan(th)
        self.crest_z = self.floor_top(self.crest_x)
        self.basin_floor_top = self.crest_z - self.step_h
        self.hinge_z = tuple(self.floor_top(fx) + self.flap_gap + self.flap_len
                             for fx in self.flap_x)
        # tilted slab: top surface from (ramp_x_lo, ramp_z_lo) to (crest_x, crest_z)
        xm = (self.ramp_x_lo + self.crest_x) / 2
        zm = self.floor_top(xm)
        d = self.slab_t / 2
        self.slab_center = (xm + math.sin(th) * d, 0.0, zm - math.cos(th) * d)
        self.slab_size = ((self.crest_x - self.ramp_x_lo) / math.cos(th),
                          self.chan_w + 2 * self.wall_t, self.slab_t)
        self.slab_quat = (math.cos(th / 2), 0.0, -math.sin(th / 2), 0.0)

        dia = 2 * self.ball_r
        # Every unintended opening is smaller than the ball; intended ones larger.
        assert 0.004 <= self.flap_gap < dia, "flap floor gap: passable by nothing"
        assert self.chan_w - self.flap_w < 0.010, "flap side gaps must not pass a ball"
        assert self.header_clear > self.flap_t / 2 + 0.002, \
            "header must clear the flap's top-edge sweep"
        assert self.header_clear + self.flap_t < dia, "flap-to-header gap must not pass a ball"
        assert self.chan_w > dia + 0.02, "channel must pass a ball with slack"
        # A ball fits under a fully-open flap ON THE INCLINE: the binding point is
        # the blade TIP at x = hinge + L*sin(open), where the floor has risen by
        # tan(tilt)*L*sin(open). Tip height over the local floor there:
        #   gap + L*(1 - cos(open) - tan(tilt)*sin(open))
        op = math.radians(self.open_deg)
        tip_clear = self.flap_gap + self.flap_len * (
            1.0 - math.cos(op) - self.tan_t * math.sin(op)) - self.flap_t / 2
        assert tip_clear > dia + 0.005, \
            f"open flap must clear a passing ball at the tip (clear={tip_clear:.4f})"
        # The roof clears a ball riding the crest; top entry exists only at the mouth.
        assert self.wall_top >= self.crest_z + dia + 0.008, "roof must clear the crest ball"
        assert self.roof_x0 < self.flap_x[0] - 0.02, \
            "the open-top stretch must end well downhill of flap 1"
        assert self.roof_x0 > self.wall_x0, "some stretch of the mouth must be open-top"
        # Basin: retaining step and capacity.
        assert abs((self.crest_z - self.basin_floor_top) - self.step_h) < 1e-9
        assert self.step_h >= self.ball_r - 1e-9, \
            "step corner must sit at/below ball centre height (no pivot-out)"
        assert self.basin_x0 <= self.crest_x <= self.basin_x1, \
            "the crest lip must overhang the basin floor"
        assert self.basin_x1 - self.crest_x > 2 * dia + 0.015, \
            "basin must hold both cargo balls beyond the crest"
        assert self.wall_x1 > self.basin_x1, "end wall must exist"
        # Flap stations and thresholds are ordered and separated.
        assert self.flap_x[0] + self.past_margin < self.flap_x[1] - self.flap_len, \
            "flap sweeps and thresholds must not overlap"
        assert self.flap_x[1] + self.flap_len < self.basin_x0, \
            "flap 2's sweep must end before the basin"
        # A ball inserted through the open top can centre at most at roof_x0 + r,
        # which must stay upstream of flap 1's blade — the flap is the barrier.
        assert self.roof_x0 + self.ball_r < self.flap_x[0] - self.flap_t / 2 - 0.005, \
            "no top drop can land past flap 1"
        # Bays: always mutually clear and clear of the structure.
        pitch = min(abs(self.bay_ys[i + 1] - self.bay_ys[i])
                    for i in range(len(self.bay_ys) - 1))
        assert pitch - 2 * self.bay_y_jit > dia + 0.005, "jittered bays must never touch"
        assert self.bay_x + self.bay_x_jit + self.ball_r < self.wall_x0 - 0.005, \
            "bays must stay clear of the rig"
        # Rubric bookkeeping.
        assert abs(2 * (self.w_p1 + self.w_p2 + self.w_bin) - self.cap) < 1e-9


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("ratchet_ramp")
class RatchetRampScene(BaseScene):
    cfg: RatchetRampSceneCfg

    BALLS = ("cargo_0", "cargo_1", "decoy_0")
    CARGO = ("cargo_0", "cargo_1")
    FLAPS = ("flap_1", "flap_2")

    def __init__(self, cfg: RatchetRampSceneCfg | None = None) -> None:
        super().__init__(cfg or RatchetRampSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        rig_spawn = cls["rig"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            slab_center=c.slab_center, slab_size=c.slab_size, slab_quat=c.slab_quat,
            chan_w=c.chan_w, wall_t=c.wall_t, wall_x0=c.wall_x0, wall_x1=c.wall_x1,
            wall_top=c.wall_top, roof_t=c.roof_t, roof_x0=c.roof_x0,
            flap_x=c.flap_x, hinge_z=c.hinge_z, header_clear=c.header_clear,
            basin_x0=c.basin_x0, basin_x1=c.basin_x1,
            basin_floor_top=c.basin_floor_top, contact_offset=c.contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.05,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig", spawn=rig_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
        }
        for i, name in enumerate(self.FLAPS):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Flap" + str(i + 1),
                spawn=cls["flap"](
                    hinge_x=c.flap_x[i], hinge_z=c.hinge_z[i], flap_t=c.flap_t,
                    flap_w=c.flap_w, flap_len=c.flap_len, open_deg=c.open_deg,
                    density=c.flap_density, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.flap_x[i], 0.0, c.hinge_z[i])))
        for name, color in (("cargo_0", c.blue), ("cargo_1", c.blue),
                            ("decoy_0", c.red)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball_" + name,
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.60, dynamic_friction=0.50, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.6, 0.3 * i, c.ball_r)))
        return out

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
        self.rig: RigidObject = env.iscene["rig"]
        self.flaps: dict[str, RigidObject] = {n: env.iscene[n] for n in self.FLAPS}
        self.balls: dict[str, RigidObject] = {n: env.iscene[n] for n in self.BALLS}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.bay_perm = torch.zeros(n, len(self.BALLS), dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._p1 = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._p2 = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._bin = torch.zeros(n, 2, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: hang both flaps closed at their hinges (the rig itself is
        never teleported — the joint anchors live on it), deal the three balls to
        the three ground bays by a random permutation with jitter, clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        for i, name in enumerate(self.FLAPS):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.flap_x[i]
            st[:, 2] = c.hinge_z[i]
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.flaps[name].write_root_state_to_sim(st, env_ids)

        # burn one draw (the first post-seed draw is near-degenerate across seeds)
        _ = torch.rand(m, device=dev)
        perm = torch.rand(m, len(self.BALLS), device=dev).argsort(dim=1)
        self.bay_perm[env_ids] = perm
        bays = torch.tensor(c.bay_ys, device=dev)
        for i, name in enumerate(self.BALLS):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.bay_x + (torch.rand(m, device=dev) * 2 - 1) * c.bay_x_jit
            st[:, 1] = bays[perm[:, i]] + (torch.rand(m, device=dev) * 2 - 1) * c.bay_y_jit
            st[:, 2] = c.ball_r + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.balls[name].write_root_state_to_sim(st, env_ids)

        self._p1[env_ids] = False
        self._p2[env_ids] = False
        self._bin[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "flaps": {k: f.data.root_state_w[env_ids].clone()
                      for k, f in self.flaps.items()},
            "balls": {k: b.data.root_state_w[env_ids].clone()
                      for k, b in self.balls.items()},
            "bay_perm": self.bay_perm[env_ids].clone(),
            "p1": self._p1[env_ids].clone(), "p2": self._p2[env_ids].clone(),
            "bin": self._bin[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for k, f in self.flaps.items():
            f.write_root_state_to_sim(state["flaps"][k], env_ids)
        for k, b in self.balls.items():
            b.write_root_state_to_sim(state["balls"][k], env_ids)
        self.bay_perm[env_ids] = state["bay_perm"]
        self._p1[env_ids] = state["p1"]
        self._p2[env_ids] = state["p2"]
        self._bin[env_ids] = state["bin"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A one-lane RAMP CHANNEL climbs a {c.tilt_deg:.0f}-degree slope between "
            f"two walls. Its bottom MOUTH opens at ground level (the first "
            f"{(c.roof_x0 - c.wall_x0) * 1000:.0f} mm of the channel are open on "
            f"top); everything further uphill is ROOFED. Two yellow one-way FLAPS "
            f"hang across the channel from hinges at the top, like pet doors: a "
            f"ball pushed UPHILL shoves a flap open and passes under it, and the "
            f"flap falls shut behind the ball; a ball rolling back DOWNHILL presses "
            f"the flap against its stop and is HELD there — so climbing progress is "
            f"kept one flap at a time. Past the second flap the ramp ends at a "
            f"crest with a covered CATCH BASIN sunk one ball-radius below it: a "
            f"ball rolling over the crest drops in and the step keeps it there. On "
            f"the ground downhill of the mouth lie three loose balls "
            f"({2 * c.ball_r * 1000:.0f} mm): two BLUE and one RED. Which ball "
            f"starts where varies every episode; read the layout, do not memorize "
            f"it.\n"
            f"Goal: drive BOTH blue balls up the channel — in through the mouth, "
            f"through both one-way flaps, over the crest — so they end up resting "
            f"inside the catch basin. The RED ball is a decoy: leave it OUT of the "
            f"basin. The flaps are not the goal: opening or waving them by hand "
            f"achieves nothing — only balls delivered into the basin count. Final "
            f"state: both blue balls at rest inside the basin, the red ball not in "
            f"the basin, both flaps hanging closed, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Push the two blue balls up the ramp channel, through both one-way "
            "flaps, and over the crest so they drop into the covered catch basin. "
            "Leave the red ball out of the basin."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World -> rig-local (the rig sits at the env origin, identity rotation)."""
        return pos_w - self.env_origins

    def _ball_loc(self) -> torch.Tensor:
        """(N, 3, 3) rig-local positions of (cargo_0, cargo_1, decoy_0)."""
        return torch.stack([self._local(b.data.root_pos_w)
                            for b in self.balls.values()], dim=1)

    def _floor_top_t(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return c.ramp_z_lo + (x - c.ramp_x_lo) * c.tan_t

    def balls_past(self, x_min: float) -> torch.Tensor:
        """(N, 3) bool: ball centre rolling IN the channel beyond x_min (on the ramp
        surface band, between the walls), or already inside the basin."""
        c = self.cfg
        loc = self._ball_loc()
        x, y, z = loc[:, :, 0], loc[:, :, 1], loc[:, :, 2]
        floor = self._floor_top_t(x)
        on_ramp = (x > x_min) & (x < c.crest_x + 0.01) & (y.abs() < c.chan_w / 2) \
            & (z > floor + 0.005) & (z < floor + c.ball_r + 0.025)
        return on_ramp | self.balls_in_basin()

    def balls_in_basin(self) -> torch.Tensor:
        """(N, 3) bool: ball centre inside the basin's interior air volume."""
        c = self.cfg
        loc = self._ball_loc()
        x, y, z = loc[:, :, 0], loc[:, :, 1], loc[:, :, 2]
        return (x > c.crest_x + 0.010) & (x < c.basin_x1 - 0.004) \
            & (y.abs() < c.chan_w / 2 - 0.005) \
            & (z > c.basin_floor_top + 0.005) & (z < c.basin_floor_top + c.ball_r + 0.025)

    def flap_angle(self) -> torch.Tensor:
        """(N, 2) hinge angle in radians (rotation about world +y; negative = swung
        uphill/open, ~0 = hanging closed)."""
        q = torch.stack([f.data.root_quat_w for f in self.flaps.values()], dim=1)
        return 2.0 * torch.atan2(q[:, :, 2], q[:, :, 0])

    def flaps_closed(self) -> torch.Tensor:
        """(N,) bool: both flaps hanging closed at their hinges."""
        c = self.cfg
        ang_ok = self.flap_angle().abs() < math.radians(c.closed_deg)
        tgt = torch.tensor([[c.flap_x[0], 0.0, c.hinge_z[0]],
                            [c.flap_x[1], 0.0, c.hinge_z[1]]],
                           device=self.env.device)
        pos = torch.stack([self._local(f.data.root_pos_w)
                           for f in self.flaps.values()], dim=1)
        pos_ok = (pos - tgt.unsqueeze(0)).norm(dim=-1) < 0.010
        return (ang_ok & pos_ok).all(dim=1)

    def settled(self) -> torch.Tensor:
        """(N,) bool: every ball slow and both flaps not swinging."""
        c = self.cfg
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in self.balls.values()], dim=1)
        w = torch.stack([f.data.root_ang_vel_w.norm(dim=-1)
                         for f in self.flaps.values()], dim=1)
        return (v < c.settle_lin).all(dim=1) & (w < c.flap_still).all(dim=1)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in self.balls.values()]
                        + [f.data.root_pos_w for f in self.flaps.values()], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        past1 = self.balls_past(c.flap_x[0] + c.past_margin)
        past2 = self.balls_past(c.flap_x[1] + c.past_margin)
        v = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                         for b in self.balls.values()], dim=1)
        in_bin = self.balls_in_basin() & (v < c.ball_slow)
        # cargo balls only — the decoy latches nothing anywhere
        self._p1 |= past1[:, 0:2] & fin.unsqueeze(1)
        self._p2 |= past2[:, 0:2] & fin.unsqueeze(1)
        self._bin |= in_bin[:, 0:2] & fin.unsqueeze(1)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both cargo balls inside the basin, the decoy NOT in the basin,
        both flaps hanging closed, everything settled and finite. All clauses are
        live physical outcomes: the roof and the one-way flaps leave the full climb
        as the only path into the basin."""
        self._update_latches()
        in_bin = self.balls_in_basin()
        cargo_ok = in_bin[:, 0] & in_bin[:, 1]
        decoy_out = ~in_bin[:, 2]
        return cargo_ok & decoy_out & self.flaps_closed() & self.settled() \
            & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: per cargo ball 0.10 past flap 1 + 0.15 past flap 2
        + 0.10 settled in the basin (all latched), capped at 0.70; exactly 1.0 iff
        success() holds live. Doing nothing scores ~0; swinging a flap (the seed's
        move-the-articulated-part skill) latches nothing; the decoy latches
        nothing."""
        c = self.cfg
        self._update_latches()
        base = (c.w_p1 * self._p1.float().sum(dim=1)
                + c.w_p2 * self._p2.float().sum(dim=1)
                + c.w_bin * self._bin.float().sum(dim=1)).clamp(max=c.cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ratchet_ramp", robot="null"))
