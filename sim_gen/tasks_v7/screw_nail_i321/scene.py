"""RamDispenserScene — operate a gravity-fed block dispenser: stage the catch bin
under the spout, then PUMP the machine's reciprocating ram (pull fully back to feed
one cube, push fully forward to eject it) until every cube is in the bin
(sim_gen task `screw_nail_i321`).

Derived from rlbench/screw_nail, but STRATEGICALLY different: the seed is
tool-mediated rotary FASTENING — grasp a screwdriver, mate its tip to a nail and
drive the fastener IN by continuous rotation about a vertical axis (one tool grasp,
one long twisting motion, judged by fastener depth). Here nothing rotates, nothing is
fastened, and the robot NEVER touches the cargo: 1-3 cubes are SEALED inside a capped
magazine tower (force-probed in smoke — they can be rattled inside the shaft but can
never leave it except through the machine), and the only way to move them is to
operate a horizontal reciprocating ram through its full stroke, one CYCLE per cube:

  PULL the ram back to its rear stop  -> the bottom cube gravity-feeds from the
                                         tower onto the channel floor;
  PUSH the ram forward to its front stop -> the ram nose shoves that cube down the
                                         roofed channel and off the muzzle lip, and
                                         it falls into whatever waits below.

The receiver is the solver's job too: an open catch bin starts parked elsewhere on
the floor and must first be STAGED under the muzzle, or the dispensed cubes are lost
on the floor (unrecoverable — the machine has no second chance on a lost cube). A
solver therefore needs a different PLAN from the seed (stage a receiver, then a
counted loop of full-stroke reciprocating cycles, one per cube, each cycle two
opposite strokes against mechanical stops) and a different CODE STRUCTURE (a cyclic
feed-then-eject state machine over an indirect transport channel plus a bin-frame
containment count — not a single fastener-depth readout).

The machine is real contact geometry, not scripted: the tower's wall bottoms clear
the channel floor by exactly ONE cube height + 6 mm, so the fed cube slides out
under them while the next cube stays penned (a two-high ride-through is
geometrically impossible, asserted in `__post_init__`); the ram's stroke is bounded
by real stops (rear stop wall behind the tail, end-plate ahead of the handle post);
retracting past the tower mouth is what lets the next cube drop (it falls onto the
channel floor and can only leave forward — rearward it jams against the ram nose,
which backs onto its own stop). The full forward stroke carries the cube's center
35 mm past the muzzle lip, so it MUST tumble off the edge.

success(): every PRESENT cube rests INSIDE the upright bin — bin-frame containment
window that by construction accepts every physically-in-bin resting pose (including
2- and 3-high pileups) and rejects wall-top perches and outside poses (asserted) —
with the cubes and bin settled.

score() is graded and latched (credit never evaporates): 0.10 bin ever staged under
the muzzle + 0.50 * (fraction of cubes ever ejected out of the machine) + 0.30 *
(fraction of cubes ever resting in the bin), capped at 0.90; 1.0 iff success(). The
null policy scores ~0 (cubes spawn sealed in the tower, bin parked away).

Per-episode randomization (readback-verified in smoke): cube count 1-3 and which
cubes those are, ram start pose BIMODAL (retracted near the rear stop, or parked
forward under the tower so the cubes rest ON the ram nose), cube jitter in the
shaft, bin park slot (4) + jitter + free yaw. Assets are fully procedural
compound-box spawners; heavy imports (isaaclab, pxr) are deferred so importing this
module — and registering the scene — stays app-free.
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


# ----- generic compound-box spawner -------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _material(stage, path: str, static: float = 0.6, dynamic: float = 0.5):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — the duplicate
    xformOp trap is avoided by never re-authoring an existing prim's ops)."""
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


def _spawn_compound(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body assembled from `cfg.boxes` = ((name, size, center, color), ...)
    with the standard physics armor (zero sleep/stabilization thresholds: a sleeping
    body silently ignores applied wrenches, which solve/smoke force probes depend
    on). Mass + friction are authored HERE — custom spawn funcs ignore the isaaclab
    mass_props/rigid_props schemas."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if cfg.kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateLinearDampingAttr(float(cfg.lin_damp))
    pxrb.CreateAngularDampingAttr(float(cfg.ang_damp))
    mat = _material(stage, f"{prim_path}/phys_mat")
    for name, size, center, color in cfg.boxes:
        _box(stage, f"{prim_path}/{name}", size, center, color, cfg.contact_offset,
             material=mat)
    return root


def _spawner_class() -> Any:
    """Define (once) the compound-spawner cfg class (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "compound" not in _SPAWNER_CACHE:

        @configclass
        class CompoundSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_compound)
            boxes: tuple = ()
            mass: float = 1.0
            kinematic: bool = False
            lin_damp: float = 0.1
            ang_damp: float = 0.1
            contact_offset: float = 0.001

        _SPAWNER_CACHE["compound"] = CompoundSpawnerCfg
    return _SPAWNER_CACHE["compound"]


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class RamDispenserSceneCfg(BaseCfg):
    """Config for `RamDispenserScene`. The machine's honesty is asserted in
    `__post_init__`: exactly one cube fits through the tower underpass (a fed cube
    escapes forward, a two-high ride-through cannot), the sealed tower headroom is
    real, the full forward stroke must carry a cube's center well past the muzzle
    lip into the staged bin's interior, and the bin containment window accepts every
    physically-in-bin resting pose (1-3 high) while rejecting wall-top and outside
    poses."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    bin_xy_tol: float = tunable(0.062)  # |cube - bin|, bin frame, per axis (m)
    bin_z_lo: float = tunable(0.010)  # cube center height window in the bin frame:
    bin_z_hi: float = tunable(0.087)  # ... floor rest 0.023, 3-high 0.083; rim 0.090
    bin_tilt_max_deg: float = tunable(15.0)  # bin up-axis within this of world-up
    settle_lin: float = tunable(0.05)  # max |lin vel| (cubes AND bin) at judging (m/s)
    settle_ang: float = tunable(1.5)  # max |ang vel| (cubes) at judging (rad/s)
    eject_z: float = tunable(0.095)  # cube center below this = out of the machine (m)
    stage_r: float = tunable(0.06)  # bin center within this of the catch point (m)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    max_blocks: int = tunable(3)  # cube count k is sampled in 1..max_blocks
    ram_ret_band: tuple = tunable((0.032, 0.038))  # retracted-start ram tip x (rig frame)
    ram_fwd_band: tuple = tunable((0.085, 0.110))  # forward-parked ram tip x (rig frame)
    ram_fwd_prob: float = tunable(0.4)  # probability of the forward-parked start
    bin_slots: tuple = tunable(((0.18, 0.30), (0.18, -0.30), (0.66, 0.30), (0.66, -0.30)))
    bin_jitter: float = tunable(0.03)  # +-xy jitter of the bin at its park slot (m)
    block_xy_jitter: float = tunable(0.002)  # +-xy jitter of each cube in the shaft (m)

    # --- info: structure (the geometry the spawners author; rig frame = env frame - rig_pos)
    rig_pos: tuple = info((0.42, 0.0))  # machine base center (env frame; never moves)
    block_edge: float = info(0.030)
    chan_floor_z: float = info(0.100)  # channel floor top (pedestal height)
    chan_half_w: float = info(0.017)  # channel interior half-width (y)
    chan_h: float = info(0.036)  # channel interior height = tower underpass gap
    slot_x: tuple = info((0.040, 0.076))  # tower mouth span along the stroke (rig x)
    muzzle_x: float = info(0.150)  # pedestal front face = drop lip (rig x)
    tower_top: float = info(0.260)  # tower interior ceiling = cap underside
    sight_w: float = info(0.006)  # sight-slot width in the tower side walls
    ram_len: float = info(0.241)  # ram bar length; tip x = bar center x + len/2
    ram_w: float = info(0.028)
    ram_h: float = info(0.030)
    post_h: float = info(0.080)  # handle post height above the bar top
    tip_ret: float = info(0.032)  # ram tip at the rear stop (rig x)
    tip_adv: float = info(0.185)  # ram tip at the front stop (rig x)
    catch_x: float = info(0.245)  # bin center under the muzzle (rig x; catch y = 0)
    bin_inner_half: float = info(0.070)
    bin_wall_t: float = info(0.010)
    bin_wall_top: float = info(0.075)
    bin_floor_t: float = info(0.008)
    depot: tuple = info((1.20, 1.20))  # parking for ABSENT cubes (env frame)
    ram_mass: float = info(0.6)
    bin_mass: float = info(0.25)
    block_mass: float = info(0.03)
    contact_offset: float = info(0.001)

    def __post_init__(self) -> None:
        c = self
        e = c.block_edge
        # -- the underpass admits EXACTLY one cube (the feed mechanism is real) --
        assert c.chan_h >= e + 0.004, "a fed cube must pass under the tower walls"
        assert c.chan_h <= 2 * e - 0.020, \
            "two stacked cubes must NOT fit through the underpass (one-at-a-time feed)"
        assert c.ram_h <= c.chan_h - 0.004, "the ram nose must pass under the tower walls"
        assert abs(c.ram_h - e) <= 0.001, \
            "penned cubes must hand over flush from cube-top to ram-top"
        # -- the stroke really cycles the feed --
        assert c.ram_ret_band[1] <= c.slot_x[0] - 0.002, \
            "a retracted ram must fully clear the tower mouth (the feed gap opens)"
        assert c.ram_fwd_band[0] >= c.slot_x[1] + 0.005, \
            "a forward-parked ram nose must fully cover the tower mouth"
        assert c.tip_adv >= c.muzzle_x + e / 2 + 0.020, \
            "the full stroke must carry a cube's center well past the muzzle lip"
        assert c.slot_x[1] - c.slot_x[0] >= e + 0.004, "the tower mouth admits a cube"
        assert 2 * c.chan_half_w >= e + 0.003, "the channel admits a cube"
        assert 2 * c.chan_half_w >= c.ram_w + 0.004, "the channel admits the ram bar"
        # -- the tower really seals its magazine --
        assert c.tower_top - (c.chan_floor_z + c.ram_h + c.max_blocks * e) >= 0.02, \
            "stacked cubes need headroom below the cap (never wedged against it)"
        assert c.sight_w <= e / 3, "sight slots must be far too narrow for a cube"
        # -- the bin containment window is honest by construction --
        assert c.bin_xy_tol >= c.bin_inner_half - e / 2 + 0.004, \
            "every physically-in-bin pose (incl. wall-leaning) must be inside the window"
        assert c.bin_xy_tol <= c.bin_inner_half - 0.005, \
            "a cube balanced on the wall top (CoM >= inner face) must be outside the window"
        assert c.bin_z_lo <= c.bin_floor_t + e / 2 - 0.005, "floor rest inside the window"
        assert c.bin_z_hi >= c.bin_floor_t + e / 2 + (c.max_blocks - 1) * e + 0.003, \
            "a full in-bin pileup (3-high) must be inside the window"
        assert c.bin_z_hi <= c.bin_wall_top + e / 2 - 0.002, \
            "a cube centered on the wall top must be above the height window"
        # -- the drop geometry lands inside the staged bin --
        assert c.bin_wall_top <= c.chan_floor_z - 0.020, \
            "the bin rim must sit well below the muzzle lip"
        assert c.catch_x - c.bin_inner_half - c.bin_wall_t >= c.muzzle_x + 0.010, \
            "the staged bin must not collide with the pedestal"
        assert c.tip_adv + e / 2 >= c.catch_x - c.bin_inner_half + e / 2 + 0.005, \
            "the carried cube must be released over the bin interior (rear margin)"
        assert c.tip_adv + e / 2 <= c.catch_x + c.bin_inner_half - e / 2 - 0.020, \
            "the carried cube must be released over the bin interior (forward flight margin)"
        # -- layout: park slots clear of the machine footprint and of the catch point --
        rx, ry = c.rig_pos
        rig_x0 = rx + c.tip_ret - c.ram_len - 0.042  # behind the rear stop wall
        rig_x1 = rx + c.muzzle_x
        for sx, sy in c.bin_slots:
            reach = c.bin_inner_half + c.bin_wall_t + c.bin_jitter
            clear_x = sx + reach < rig_x0 or sx - reach > rig_x1
            clear_y = abs(sy - ry) - reach > 0.045
            assert clear_x or clear_y, f"bin slot ({sx},{sy}) collides with the machine"
            d = math.hypot(sx - (rx + c.catch_x), sy - ry)
            assert d > c.stage_r + 0.12, f"bin slot ({sx},{sy}) too close to the catch point"
        assert math.hypot(c.depot[0] - rx, c.depot[1] - ry) > 0.9, "depot far from the rig"


# ----- procedural geometry (box lists, rig-local coordinates) ------------------------------------
def _rig_boxes(c: RamDispenserSceneCfg) -> tuple:
    """The kinematic machine: pedestal + roofed channel + sealed magazine tower +
    stroke stops. Origin = rig base center at ground level; the stroke runs +x."""
    grey = (0.55, 0.55, 0.58)
    dark = (0.30, 0.30, 0.34)
    amber = (0.85, 0.65, 0.15)
    tail_ret = c.tip_ret - c.ram_len  # bar tail x at the rear stop (-0.209)
    tail_adv = c.tip_adv - c.ram_len  # bar tail x at the front stop (-0.056)
    ped_x0 = tail_ret - 0.030
    wall_z0 = c.chan_floor_z
    roof_z0 = c.chan_floor_z + c.chan_h  # tower underpass ceiling / roof underside
    s0, s1 = c.slot_x
    mid = (s0 + s1) / 2
    tw_h = c.tower_top - roof_z0  # tower wall height
    tw_zc = (roof_z0 + c.tower_top) / 2
    yw = c.chan_half_w + 0.012  # channel wall outer face
    boxes = [
        # pedestal deck: rear stop wall -> muzzle lip (the drop edge is this face)
        ("pedestal", (c.muzzle_x - ped_x0, 0.090, c.chan_floor_z),
         ((c.muzzle_x + ped_x0) / 2, 0.0, c.chan_floor_z / 2), grey),
        # rear stop wall (retraction stop: the bar tail bottoms out on its inner face)
        ("rear_stop", (0.012, 0.090, 0.080), (tail_ret - 0.006, 0.0, wall_z0 + 0.040), dark),
        # rear guide rails (guide the exposed bar over the full stroke; below bar top)
        ("rail_p", (-0.030 - tail_ret, 0.012, 0.026),
         ((tail_ret - 0.030) / 2, +c.chan_half_w + 0.006, wall_z0 + 0.013), dark),
        ("rail_n", (-0.030 - tail_ret, 0.012, 0.026),
         ((tail_ret - 0.030) / 2, -c.chan_half_w - 0.006, wall_z0 + 0.013), dark),
        # channel side walls
        ("wall_p", (c.muzzle_x + 0.030, 0.012, c.chan_h + 0.012),
         ((c.muzzle_x - 0.030) / 2, +c.chan_half_w + 0.006, wall_z0 + (c.chan_h + 0.012) / 2),
         grey),
        ("wall_n", (c.muzzle_x + 0.030, 0.012, c.chan_h + 0.012),
         ((c.muzzle_x - 0.030) / 2, -c.chan_half_w - 0.006, wall_z0 + (c.chan_h + 0.012) / 2),
         grey),
        # end-plate (advance stop for the handle post; also caps ram lift at the rear)
        ("end_plate", (0.012, 2 * yw, 0.046),
         (tail_adv + 0.014 + 0.006, 0.0, c.chan_floor_z + c.ram_h + 0.004 + 0.023), dark),
        # channel roof, split by the tower mouth
        ("roof_rear", (s0 + 0.030, 2 * yw, 0.012), ((s0 - 0.030) / 2, 0.0, roof_z0 + 0.006),
         grey),
        ("roof_front", (c.muzzle_x - s1, 2 * yw, 0.012),
         ((s1 + c.muzzle_x) / 2, 0.0, roof_z0 + 0.006), grey),
        # magazine tower shaft (wall bottoms at roof_z0 = the one-cube underpass)
        ("shaft_rear", (0.012, 2 * yw, tw_h), (s0 - 0.006, 0.0, tw_zc), grey),
        ("shaft_front", (0.012, 2 * yw, tw_h), (s1 + 0.006, 0.0, tw_zc), grey),
        # tower side walls with a vertical sight slot (cube count is visible)
        ("shaft_pa", (mid - c.sight_w / 2 - (s0 - 0.012), 0.012, tw_h),
         ((s0 - 0.012 + mid - c.sight_w / 2) / 2, +c.chan_half_w + 0.006, tw_zc), grey),
        ("shaft_pb", ((s1 + 0.012) - (mid + c.sight_w / 2), 0.012, tw_h),
         ((mid + c.sight_w / 2 + s1 + 0.012) / 2, +c.chan_half_w + 0.006, tw_zc), grey),
        ("shaft_na", (mid - c.sight_w / 2 - (s0 - 0.012), 0.012, tw_h),
         ((s0 - 0.012 + mid - c.sight_w / 2) / 2, -c.chan_half_w - 0.006, tw_zc), grey),
        ("shaft_nb", ((s1 + 0.012) - (mid + c.sight_w / 2), 0.012, tw_h),
         ((mid + c.sight_w / 2 + s1 + 0.012) / 2, -c.chan_half_w - 0.006, tw_zc), grey),
        # tower cap: the magazine is SEALED from above
        ("cap", (s1 - s0 + 0.036, 2 * yw + 0.012, 0.012),
         (mid, 0.0, c.tower_top + 0.006), amber),
    ]
    return tuple(boxes)


def _ram_boxes(c: RamDispenserSceneCfg) -> tuple:
    """The free (captive) ram: horizontal bar + upright handle post at the tail.
    Origin = bar center; tip x = origin x + ram_len/2."""
    steel = (0.45, 0.48, 0.55)
    red = (0.80, 0.15, 0.12)
    return (
        ("bar", (c.ram_len, c.ram_w, c.ram_h), (0.0, 0.0, 0.0), steel),
        ("post", (0.014, 0.014, c.post_h),
         (-c.ram_len / 2 + 0.007, 0.0, c.ram_h / 2 + c.post_h / 2), red),
    )


def _bin_boxes(c: RamDispenserSceneCfg) -> tuple:
    """The open catch bin: floor plate + four walls. Origin = floor center, bottom."""
    white = (0.92, 0.92, 0.95)
    ih, t = c.bin_inner_half, c.bin_wall_t
    wall_h = c.bin_wall_top - c.bin_floor_t
    wz = c.bin_floor_t + wall_h / 2
    return (
        ("floor", (2 * ih + 2 * t, 2 * ih + 2 * t, c.bin_floor_t),
         (0.0, 0.0, c.bin_floor_t / 2), white),
        ("wall_px", (t, 2 * ih + 2 * t, wall_h), (+ih + t / 2, 0.0, wz), white),
        ("wall_nx", (t, 2 * ih + 2 * t, wall_h), (-ih - t / 2, 0.0, wz), white),
        ("wall_py", (2 * ih, t, wall_h), (0.0, +ih + t / 2, wz), white),
        ("wall_ny", (2 * ih, t, wall_h), (0.0, -ih - t / 2, wz), white),
    )


_BLOCK_COLORS = ((0.85, 0.15, 0.15), (0.10, 0.75, 0.20), (0.15, 0.35, 0.85))
_BLOCK_NAMES = ("RED", "GREEN", "BLUE")


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("ram_dispenser")
class RamDispenserScene(BaseScene):
    cfg: RamDispenserSceneCfg

    def __init__(self, cfg: RamDispenserSceneCfg | None = None) -> None:
        super().__init__(cfg or RamDispenserSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_class()
        rx, ry = c.rig_pos
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.6, dynamic_friction=0.5, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rig": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig",
                spawn=sp(boxes=_rig_boxes(c), mass=8.0, kinematic=True,
                         contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(rx, ry, 0.0)),
            ),
            "ram": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ram",
                spawn=sp(boxes=_ram_boxes(c), mass=c.ram_mass, lin_damp=1.0,
                         ang_damp=2.0, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(rx + c.tip_ret + 0.003 - c.ram_len / 2, ry,
                         c.chan_floor_z + c.ram_h / 2 + 0.0015)),
            ),
            "bin": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bin",
                spawn=sp(boxes=_bin_boxes(c), mass=c.bin_mass, lin_damp=0.5,
                         ang_damp=0.5, contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bin_slots[0][0], c.bin_slots[0][1], 0.001)),
            ),
        }
        e = c.block_edge
        mid = (c.slot_x[0] + c.slot_x[1]) / 2
        for i in range(3):
            out[f"block{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block" + str(i),
                spawn=sp(boxes=(("body", (e, e, e), (0.0, 0.0, 0.0), _BLOCK_COLORS[i]),),
                         mass=c.block_mass, lin_damp=0.1, ang_damp=0.1,
                         contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(rx + mid, ry, c.chan_floor_z + e / 2 + 0.002 + i * (e + 0.002))),
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

    # ----- lifecycle ---------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.rig: RigidObject = env.iscene["rig"]
        self.ram: RigidObject = env.iscene["ram"]
        self.bin: RigidObject = env.iscene["bin"]
        self.blocks: list[RigidObject] = [env.iscene[f"block{i}"] for i in range(3)]
        self.env_origins = env.iscene.env_origins
        self.present = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        # progress latches (post_step): bin ever staged; per-cube ever ejected / in bin
        self.stage_latch = torch.zeros(n, device=dev)
        self.eject_latch = torch.zeros(n, 3, device=dev)
        self.inbin_latch = torch.zeros(n, 3, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: k in 1..max_blocks cubes (a random SUBSET of the three
        colors) stacked sealed in the tower; ram start BIMODAL (retracted near the
        rear stop, or forward-parked with the nose under the tower so cubes rest ON
        it); bin at a sampled park slot + jitter + free yaw; absent cubes parked in
        the far depot; latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        rx, ry = c.rig_pos
        e = c.block_edge

        # burn draws: the FIRST post-seed draw (rand AND randint) is near-constant
        # across seeds — never feed it to a discrete choice.
        _ = torch.rand(m, 7, device=dev)
        _ = torch.randint(0, 997, (m, 3), device=dev)

        def write(body, pos: torch.Tensor, quat: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            if quat is None:
                st[:, 3] = 1.0
            else:
                st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # which cubes exist, and their stack order
        k = torch.randint(1, c.max_blocks + 1, (m,), device=dev)
        order = torch.rand(m, 3, device=dev).argsort(dim=1)  # block index at each level
        level_of = order.argsort(dim=1)  # stack level of each block index
        present = level_of < k.unsqueeze(1)
        self.present[env_ids] = present

        # ram: bimodal start along the stroke, resting on the channel floor
        fwd = torch.rand(m, device=dev) < c.ram_fwd_prob
        u = torch.rand(m, device=dev)
        tip = torch.where(
            fwd, c.ram_fwd_band[0] + u * (c.ram_fwd_band[1] - c.ram_fwd_band[0]),
            c.ram_ret_band[0] + u * (c.ram_ret_band[1] - c.ram_ret_band[0]))
        ram_pos = torch.zeros(m, 3, device=dev)
        ram_pos[:, 0] = rx + tip - c.ram_len / 2
        ram_pos[:, 1] = ry
        ram_pos[:, 2] = c.chan_floor_z + c.ram_h / 2 + 0.0015
        write(self.ram, ram_pos)

        # cubes: stacked in the shaft (on the floor, or on the forward-parked nose);
        # absent cubes park in the far depot
        mid = (c.slot_x[0] + c.slot_x[1]) / 2
        base_z = torch.where(fwd, c.chan_floor_z + c.ram_h, c.chan_floor_z) + e / 2 + 0.002
        for i, body in enumerate(self.blocks):
            lvl = level_of[:, i].float()
            pos = torch.zeros(m, 3, device=dev)
            pos[:, 0] = rx + mid + (torch.rand(m, device=dev) * 2 - 1) * c.block_xy_jitter
            pos[:, 1] = ry + (torch.rand(m, device=dev) * 2 - 1) * c.block_xy_jitter
            pos[:, 2] = base_z + lvl * (e + 0.002)
            dep = torch.tensor([c.depot[0], c.depot[1] + 0.10 * i, e / 2 + 0.002],
                               device=dev).expand(m, 3)
            pos = torch.where(present[:, i].unsqueeze(1), pos, dep)
            write(body, pos)

        # bin: sampled park slot + jitter + free yaw
        slots = torch.tensor(c.bin_slots, device=dev)
        pick = torch.randint(0, len(c.bin_slots), (m,), device=dev)
        bpos = torch.zeros(m, 3, device=dev)
        bpos[:, 0:2] = slots[pick] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.bin_jitter
        bpos[:, 2] = 0.001
        half_yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi / 2
        quat = torch.stack([torch.cos(half_yaw), torch.zeros(m, device=dev),
                            torch.zeros(m, device=dev), torch.sin(half_yaw)], dim=-1)
        write(self.bin, bpos, quat)

        self.stage_latch[env_ids] = 0.0
        self.eject_latch[env_ids] = 0.0
        self.inbin_latch[env_ids] = 0.0

    # ----- state (full, restorable) ------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"ram": self.ram, "bin": self.bin,
                  **{f"block{i}": b for i, b in enumerate(self.blocks)}}
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "present": self.present[env_ids].clone(),
            "stage_latch": self.stage_latch[env_ids].clone(),
            "eject_latch": self.eject_latch[env_ids].clone(),
            "inbin_latch": self.inbin_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"ram": self.ram, "bin": self.bin,
                  **{f"block{i}": b for i, b in enumerate(self.blocks)}}
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        self.present[env_ids] = state["present"]
        self.stage_latch[env_ids] = state["stage_latch"]
        self.eject_latch[env_ids] = state["eject_latch"]
        self.inbin_latch[env_ids] = state["inbin_latch"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A grey dispenser machine stands bolted to the floor: a raised, roofed "
            "channel runs along it, a sealed magazine tower with an AMBER cap rises "
            "from the channel's middle, and a steel RAM slides lengthwise through the "
            "channel, its upright RED handle post sticking up at the rear. Inside the "
            f"tower, 1 to 3 colored cubes ({c.block_edge * 1000:.0f} mm) are stacked — "
            "count them through the narrow sight slots. The tower is capped and its "
            "slots are far too narrow: NO cube can be reached, lifted out, or freed by "
            "hand. The only way a cube leaves is by pumping the ram through one full "
            "cycle per cube: PULL the red handle all the way back to its stop and the "
            "bottom cube drops out of the tower onto the channel floor; PUSH the "
            "handle all the way forward to its stop and the ram nose shoves that cube "
            "down the roofed channel and off the drop lip at the muzzle, where it "
            "falls and is lost on the floor — unless the open WHITE catch bin has "
            "first been placed under the muzzle. The bin starts parked somewhere else "
            "on the floor. The cube count, the ram's starting position along its "
            "stroke, and the bin's parking spot and heading change every episode: "
            "read them by looking.\n"
            "Goal: put EVERY cube in this episode inside the white bin — stage the "
            "bin under the muzzle, then pump the ram full-stroke, back then forward, "
            "once per cube. Cubes resting inside the upright bin count (stacked is "
            "fine); a cube on the floor, on the machine, or balanced on the bin wall "
            "counts for nothing, and a tipped-over bin or still-moving cubes never "
            "count. A cube dispensed with no bin waiting cannot be recovered."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place the white bin under the dispenser's muzzle, then pump the red ram "
            "handle — fully back to drop a cube from the tower, fully forward to "
            "eject it into the bin — once per cube, until every cube rests in the bin."
        )

    # ----- geometry helpers ----------------------------------------------------------------------
    def ram_tip(self) -> torch.Tensor:
        """(N,) ram tip x in the rig frame: tip_ret at the rear stop, tip_adv front."""
        c = self.cfg
        return (self.ram.data.root_pos_w[:, 0] - self.env_origins[:, 0]
                - c.rig_pos[0] + c.ram_len / 2)

    def block_pos(self) -> torch.Tensor:
        """(N, 3, 3) cube centers relative to the env origin."""
        return torch.stack([b.data.root_pos_w - self.env_origins for b in self.blocks],
                           dim=1)

    def _blocks_in_bin_frame(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        q = self.bin.data.root_quat_w
        p = self.bin.data.root_pos_w
        return torch.stack(
            [quat_apply_inverse(q, b.data.root_pos_w - p) for b in self.blocks], dim=1)

    def bin_upright(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = quat_apply(self.bin.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.bin_tilt_max_deg))

    def block_in_bin(self) -> torch.Tensor:
        """(N, 3) bool, geometric: cube inside the upright bin — bin-frame containment
        (accepts every physically-in-bin resting pose incl. pileups, rejects wall-top
        and outside poses; asserted in __post_init__)."""
        c = self.cfg
        rel = self._blocks_in_bin_frame()
        geo = (rel[..., 0].abs() <= c.bin_xy_tol) & (rel[..., 1].abs() <= c.bin_xy_tol) \
            & (rel[..., 2] >= c.bin_z_lo) & (rel[..., 2] <= c.bin_z_hi)
        return geo & self.bin_upright().unsqueeze(1)

    def ejected_now(self) -> torch.Tensor:
        """(N, 3) bool: cube center below the machine deck — it has left the machine
        (the muzzle is the only exit; the tower is sealed and the channel's rear is
        plugged by the ram against its stop)."""
        z = torch.stack([b.data.root_pos_w[:, 2] for b in self.blocks], dim=1) \
            - self.env_origins[:, 2].unsqueeze(1)
        return z < self.cfg.eject_z

    def bin_staged_now(self) -> torch.Tensor:
        """(N,) bool: upright bin centered under the muzzle catch point."""
        c = self.cfg
        catch = torch.tensor([c.rig_pos[0] + c.catch_x, c.rig_pos[1]],
                             device=self.env.device)
        d = (self.bin.data.root_pos_w[:, 0:2] - self.env_origins[:, 0:2] - catch).norm(dim=-1)
        return (d <= c.stage_r) & self.bin_upright()

    def settled(self) -> torch.Tensor:
        """(N,) bool: every PRESENT cube and the bin at rest."""
        c = self.cfg
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.blocks], dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1) for b in self.blocks], dim=1)
        blocks_ok = (((lin < c.settle_lin) & (ang < c.settle_ang)) | ~self.present).all(dim=1)
        return blocks_ok & (self.bin.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    # ----- progress latches (step-coupled) -------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch each demonstrated stage every physics substep: bin ever staged under
        the muzzle; per-cube ever ejected out of the machine; per-cube ever resting
        inside the bin (absent cubes never latch anything)."""
        p = self.present.float()
        self.stage_latch = torch.maximum(self.stage_latch, self.bin_staged_now().float())
        self.eject_latch = torch.maximum(self.eject_latch, self.ejected_now().float() * p)
        self.inbin_latch = torch.maximum(self.inbin_latch, self.block_in_bin().float() * p)

    # ----- rubric ---------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: EVERY present cube rests inside the upright bin, cubes and bin
        settled. The sealed tower + roofed channel make the pump cycle (stage bin ->
        pull to feed -> push to eject, once per cube) physically necessary."""
        ok = (self.block_in_bin() | ~self.present).all(dim=1)
        return ok & self.present.any(dim=1) & self.bin_upright() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10 bin ever staged + 0.50 * fraction of cubes ever
        ejected + 0.30 * fraction ever in the bin (cap 0.90); 1.0 iff success().
        Latched — credit never evaporates; the null policy scores ~0 (cubes sealed
        in the tower, bin parked away from the catch point)."""
        p = self.present.float()
        np_ = p.sum(dim=1).clamp(min=1.0)
        frac_e = (self.eject_latch * p).sum(dim=1) / np_
        frac_i = (self.inbin_latch * p).sum(dim=1) / np_
        base = (0.10 * self.stage_latch + 0.50 * frac_e + 0.30 * frac_i).clamp(0.0, 0.90)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="ram_dispenser", robot="null", env_spacing=3.0))
