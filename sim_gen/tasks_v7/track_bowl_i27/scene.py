"""BellHerdScene — trap the ungraspable RED ball under the knobbed bell, drag the
loaded bell across the board until the ball drops into the sunken pocket, then park
the bell clear so the delivery is revealed (sim_gen task `track_bowl_i27`).

Derived from pick_place/track_bowl, but STRATEGICALLY different: the seed's plan is
pure prescribed TRANSPORT of the bowl itself — grasp the bowl, lift it, and carry it
through free space along a dense waypoint trajectory (reward = per-step distance to
the current waypoint; the bowl is the payload and the hand holds it the whole way).
Here the bowl-shaped object is INVERTED and becomes a TOOL, and the judged payload is
something the hand can never hold: an 85 mm ball, wider than the parallel jaw's 80 mm
span, free to roll on a fenced board. The plan is CAPTURE -> HERD -> DELIVER ->
REVEAL: lower the open-mouthed bell over the ball to cage it, then DRAG the bell
flat across the board so its inner skirt wall pushes the caged, rolling ball along
the surface (sliding-under-contact, not a carry — the bell never leaves the board
while loaded), steer the loaded bell over the dark sunken pocket so the ball falls
through the board's opening, and finally lift the empty bell away and park it clear
of the pocket. There is no prescribed path and no per-step tracking: any drag route
works, and success is a settled terminal state, not trajectory conformance. A solver
therefore needs a different PLAN (tool acquisition, containment, contact-dragging,
gravity delivery, tool removal) and a different code structure (containment predicates
and drag-control code instead of a waypoint follower).

Judged in the BOARD's body frame (kinematic fixture, xy + yaw randomized). success()
iff, settled:
  - the RED ball rests INSIDE the pocket (board-local xy within the pocket square,
    centre below the play surface);
  - the BLUE distractor ball is NOT in the pocket (dropping it in is a wrong-object
    failure that is practically irreversible: neither ball can be grasped);
  - the bell is parked CLEAR of the pocket (axis > `bell_clear` from the pocket
    centre) — a delivery still covered by the bell does not count as revealed.
score() is latched every physics substep: 0.15 * the red ball was ever CAGED under
the grounded bell + 0.25 * best fractional progress of the red ball toward the
pocket (normalized by its own spawn distance) + 0.30 * the red ball ever IN the
pocket, capped at 0.70; exactly 1.0 iff success(). Doing nothing scores ~0.

Assets are fully procedural (no external files):
  - board: ONE kinematic fixture — a 0.70 x 0.70 m raised play surface (top 80 mm
    above the floor) with a low fence around the rim, pierced by a 107 x 107 mm
    square POCKET sunk 60 mm below the surface near the +x side; the pocket floor is
    painted YELLOW and reads as a dark square opening with a yellow bottom;
  - bell: the only DYNAMIC tool — an open-bottomed octagonal bell (inner mouth
    inradius 75 mm, skirt 100 mm tall, ~180 mm across) closed by a round top plate
    and topped by a 22 mm grasp knob with a 36 mm cap flange; 0.5 kg, damped so a
    dragged bell glides instead of wobbling. Its mouth swallows the ball with 32 mm
    of radial slack, and its rim ring (min outer inradius 83 mm) is wider than the
    pocket's half-diagonal, so the bell can NEVER fall into the pocket it feeds;
  - balls: RED (target) and BLUE (distractor) 85 mm spheres, 150 g — both wider
    than an 80 mm parallel-jaw span: neither can be grasped, only caged or nudged.
Contact offsets are explicit and small (2 mm): the default would eat the 11 mm
pocket drop clearance and the bell's 3 mm anti-fall margin.

Per-episode randomization (verified by readback in smoke): board xy + yaw, a coin
flip for which side of the board holds the red ball (blue mirrors it), xy scatter
for both balls and the bell with keep-outs (pocket, mutual). Heavy imports
(isaaclab, pxr) are deferred so importing this module — and registering the scene —
stays app-free.
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


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float,
         yaw_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        sxf.AddRotateZOp().Set(float(yaw_deg))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _cyl(stage, path: str, radius: float, height: float, center, color,
         contact_offset: float) -> None:
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cylinder.Define(stage, path)
    seg.CreateRadiusAttr(float(radius))
    seg.CreateHeightAttr(float(height))
    seg.CreateAxisAttr("Z")
    seg.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(seg.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collide(seg.GetPrim(), contact_offset)


def _spawn_board(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC board at `prim_path`. Local origin = centre of the PLAY
    SURFACE (z = 0 local is the top the balls roll on). Four surface plates leave the
    square pocket hole open at (pit_x, 0); the pocket floor (YELLOW) sits `pit_depth`
    below; a low fence rings the field so nothing rolls off."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(30.0)

    co = cfg.contact_offset
    b, p, px = cfg.half_ext, cfg.pit_half, cfg.pit_x
    t = cfg.pit_depth  # plates are exactly pocket-deep: hole sides are the pit walls
    col, dark, yel = cfg.color, cfg.fence_color, cfg.pit_floor_color

    # --- surface plates around the pocket hole (tops at local z = 0) ---
    zc = -t / 2
    _box(stage, f"{prim_path}/plate_w", (px - p + b, 2 * b, t),
         ((-b + px - p) / 2, 0.0, zc), col, co)
    _box(stage, f"{prim_path}/plate_e", (b - px - p, 2 * b, t),
         ((px + p + b) / 2, 0.0, zc), col, co)
    _box(stage, f"{prim_path}/plate_n", (2 * p, b - p, t),
         (px, (p + b) / 2, zc), col, co)
    _box(stage, f"{prim_path}/plate_s", (2 * p, b - p, t),
         (px, -(p + b) / 2, zc), col, co)

    # --- pocket floor: YELLOW, slightly oversize footprint, top at local -pit_depth ---
    _box(stage, f"{prim_path}/pit_floor", (2 * p + 0.04, 2 * p + 0.04, 0.02),
         (px, 0.0, -t - 0.01), yel, co)

    # --- perimeter fence (dark), straddling the surface ---
    fh = cfg.fence_h + t  # from plate bottom to fence_h above the surface
    zf = (cfg.fence_h - t) / 2
    ft = cfg.fence_t
    _box(stage, f"{prim_path}/fence_n", (2 * b + 2 * ft, ft, fh),
         (0.0, b + ft / 2, zf), dark, co)
    _box(stage, f"{prim_path}/fence_s", (2 * b + 2 * ft, ft, fh),
         (0.0, -b - ft / 2, zf), dark, co)
    _box(stage, f"{prim_path}/fence_e", (ft, 2 * b, fh),
         (b + ft / 2, 0.0, zf), dark, co)
    _box(stage, f"{prim_path}/fence_w", (ft, 2 * b, fh),
         (-b - ft / 2, 0.0, zf), dark, co)
    return root


def _board_spawner_cfg(*, half_ext: float, pit_half: float, pit_x: float,
                       pit_depth: float, fence_h: float, fence_t: float, color: tuple,
                       fence_color: tuple, pit_floor_color: tuple,
                       contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "board" not in _SPAWNER_CACHE:

        @configclass
        class HerdBoardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_board)
            half_ext: float = 0.35
            pit_half: float = 0.0535
            pit_x: float = 0.20
            pit_depth: float = 0.06
            fence_h: float = 0.06
            fence_t: float = 0.02
            color: tuple = (0.62, 0.62, 0.65)
            fence_color: tuple = (0.30, 0.30, 0.34)
            pit_floor_color: tuple = (0.95, 0.85, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["board"] = HerdBoardSpawnerCfg

    return _SPAWNER_CACHE["board"](
        mass_props=sim_utils.MassPropertiesCfg(mass=30.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        half_ext=half_ext, pit_half=pit_half, pit_x=pit_x, pit_depth=pit_depth,
        fence_h=fence_h, fence_t=fence_t, color=color, fence_color=fence_color,
        pit_floor_color=pit_floor_color, contact_offset=contact_offset,
    )


def _spawn_bell(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC bell at `prim_path`. Local origin = centre of the OPEN
    MOUTH (rim plane): resting on the board puts the origin at surface height. An
    octagonal skirt of 8 yawed wall boxes rises to a round top plate; above it a
    grasp knob (22 mm shaft, 36 mm cap flange) for a parallel jaw. Uniform collider
    density keeps the COM low in the skirt — a dragged bell slides, it does not
    topple."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.10)
    px.CreateAngularDampingAttr(2.0)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    co = cfg.contact_offset
    r_in, t, h = cfg.mouth_r, cfg.wall_t, cfg.skirt_h
    w_flat = 2.0 * (r_in + t) * math.tan(math.radians(22.5)) + 0.002
    for k in range(8):
        ang = k * 45.0
        rad = math.radians(ang)
        cx = (r_in + t / 2) * math.cos(rad)
        cy = (r_in + t / 2) * math.sin(rad)
        _box(stage, f"{prim_path}/skirt_{k}", (t, w_flat, h),
             (cx, cy, h / 2), cfg.color, co, yaw_deg=ang)
    _cyl(stage, f"{prim_path}/top", cfg.top_r, cfg.top_t,
         (0.0, 0.0, h + cfg.top_t / 2), cfg.color, co)
    _cyl(stage, f"{prim_path}/knob", cfg.knob_r, cfg.knob_h,
         (0.0, 0.0, h + cfg.top_t + cfg.knob_h / 2), cfg.knob_color, co)
    _cyl(stage, f"{prim_path}/knob_cap", cfg.cap_r, cfg.cap_t,
         (0.0, 0.0, h + cfg.top_t + cfg.knob_h + cfg.cap_t / 2),
         cfg.knob_color, co)
    return root


def _bell_spawner_cfg(*, mouth_r: float, wall_t: float, skirt_h: float, top_r: float,
                      top_t: float, knob_r: float, knob_h: float, cap_r: float,
                      cap_t: float, mass: float, color: tuple, knob_color: tuple,
                      contact_offset: float) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bell" not in _SPAWNER_CACHE:

        @configclass
        class HerdBellSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bell)
            mouth_r: float = 0.075
            wall_t: float = 0.008
            skirt_h: float = 0.100
            top_r: float = 0.090
            top_t: float = 0.008
            knob_r: float = 0.011
            knob_h: float = 0.035
            cap_r: float = 0.018
            cap_t: float = 0.008
            mass: float = 0.5
            color: tuple = (0.35, 0.45, 0.55)
            knob_color: tuple = (0.20, 0.22, 0.26)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bell"] = HerdBellSpawnerCfg

    return _SPAWNER_CACHE["bell"](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mouth_r=mouth_r, wall_t=wall_t, skirt_h=skirt_h, top_r=top_r, top_t=top_t,
        knob_r=knob_r, knob_h=knob_h, cap_r=cap_r, cap_t=cap_t, mass=mass,
        color=color, knob_color=knob_color, contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BellHerdCfg(BaseCfg):
    """Config for `BellHerdScene`. Honesty knobs asserted in `__post_init__`: neither
    ball fits a parallel jaw, the ball fits the bell mouth with real slack, the ball
    falls through the pocket with real clearance, the bell's rim ring is wider than
    the pocket's half-diagonal (the tool can never fall into the pocket it feeds),
    and the in-pocket / on-surface heights are cleanly separated."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    drop_min: float = tunable(0.010)  # in-pocket: ball centre below surface by > this (m)
    pit_margin: float = tunable(0.005)  # in-pocket: xy inside the pocket square by this (m)
    bell_clear: float = tunable(0.20)  # bell axis farther than this from the pocket centre
    settle_lin: float = tunable(0.05)  # max |lin vel| of balls + bell when judging (m/s)
    cage_slack: float = tunable(0.005)  # caged: ball axis within mouth_r - ball_r + this

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fix_jitter: float = tunable(0.04)  # uniform +/- xy jitter of the board (m)
    fix_yaw_deg: float = tunable(10.0)  # uniform +/- board yaw (judge in the board frame)
    side_flip: bool = tunable(True)  # coin-flip which y-side holds the RED ball
    ball_scatter: tuple = tunable((0.06, 0.05))  # +/- xy scatter of each ball spawn (m)
    bell_scatter: tuple = tunable((0.05, 0.05))  # +/- xy scatter of the bell spawn (m)

    # --- tunable: placement (board-local xy) -------------------------------------------------
    fix_pos: tuple = tunable((0.40, 0.0))  # board centre, WORLD xy nominal
    red_pos: tuple = tunable((-0.16, 0.13))  # red ball nominal (y * side per episode)
    blue_pos: tuple = tunable((-0.16, -0.13))  # blue ball nominal (mirrored side)
    bell_pos: tuple = tunable((0.10, -0.18))  # bell nominal (y * side per episode)
    keepout_pit: float = tunable(0.14)  # min ball-spawn distance from the pocket centre
    keepout_mutual: float = tunable(0.15)  # min spawn distance between the three bodies

    # --- info: board structure ---------------------------------------------------------------
    surface_h: float = info(0.08)  # play-surface height above the floor (m)
    half_ext: float = info(0.35)  # board half-extent (0.70 x 0.70 m field)
    pit_half: float = info(0.0535)  # pocket half-width (107 mm square hole)
    pit_x: float = info(0.20)  # pocket centre, board-local x (y = 0)
    pit_depth: float = info(0.06)  # pocket floor below the play surface
    fence_h: float = info(0.06)  # fence above the surface: an 85 mm ball cannot escape
    fence_t: float = info(0.02)
    # --- info: bell (the tool) ---------------------------------------------------------------
    mouth_r: float = info(0.075)  # inner mouth inradius: swallows the ball with slack
    wall_t: float = info(0.008)
    skirt_h: float = info(0.100)  # taller than the ball: the cage fully encloses it
    top_r: float = info(0.090)
    top_t: float = info(0.008)
    knob_r: float = info(0.011)  # 22 mm knob shaft: a comfortable parallel-jaw grasp
    knob_h: float = info(0.035)
    cap_r: float = info(0.018)  # 36 mm cap flange above the shaft (hook-proof grasp)
    cap_t: float = info(0.008)
    bell_mass: float = info(0.5)
    # --- info: balls -------------------------------------------------------------------------
    ball_r: float = info(0.0425)  # 85 mm: wider than an 80 mm parallel-jaw span
    ball_mass: float = info(0.15)
    jaw_span: float = info(0.080)  # the Franka jaw the balls must defeat
    board_color: tuple = info((0.62, 0.62, 0.65))
    fence_color: tuple = info((0.30, 0.30, 0.34))
    pit_floor_color: tuple = info((0.95, 0.85, 0.10))
    bell_color: tuple = info((0.35, 0.45, 0.55))
    knob_color: tuple = info((0.20, 0.22, 0.26))
    red_color: tuple = info((0.85, 0.12, 0.10))
    blue_color: tuple = info((0.15, 0.35, 0.90))
    # Explicit small offsets: the default ~2 cm would eat the 11 mm drop clearance.
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    cage_r: float = field(default=None, init=False)  # max ball-axis offset when caged

    def __post_init__(self) -> None:
        self.cage_r = self.mouth_r - self.ball_r  # 32.5 mm of radial slack

        # neither ball fits the jaw: caging/nudging are the only handles on them
        assert 2 * self.ball_r >= self.jaw_span + 0.004, "balls must defeat the jaw span"
        # the bell mouth swallows the ball with real slack (arm-precision friendly)
        assert self.cage_r >= 0.025, "cage slack must exceed closed-loop arm precision"
        assert self.skirt_h >= 2 * self.ball_r + 0.010, "the cage must fully enclose the ball"
        # the ball falls through the pocket with real clearance
        assert self.pit_half >= self.ball_r + 0.010, "ball must drop through the pocket"
        # the bell can NEVER fall into the pocket: its min outer inradius (skirt flat
        # ring) exceeds the pocket's half-diagonal
        assert self.mouth_r + self.wall_t >= self.pit_half * math.sqrt(2) + 0.003, (
            "bell rim ring must always bridge the pocket")
        # a caged ball centred over the pocket is centred over the HOLE
        assert self.cage_r <= self.pit_half, "a caged ball must end up over the hole"
        # in-pocket vs on-surface heights are cleanly separated
        in_pit_z = self.ball_r - self.pit_depth  # settled in-pocket centre (local z)
        assert in_pit_z < -self.drop_min - 0.005, "in-pocket height must clear drop_min"
        assert self.ball_r > self.drop_min + 0.02, "on-surface height must clear drop_min"
        # the fence holds the balls (centre below the fence top)
        assert self.fence_h > self.ball_r + 0.015, "fence must contain a rolling ball"
        # the parked-bell clause is geometrically meaningful: clear means the whole
        # rim ring is off the pocket
        assert self.bell_clear >= self.top_r + self.pit_half * math.sqrt(2) + 0.02, (
            "bell_clear must put the whole bell off the pocket")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("bell_herd")
class BellHerdScene(BaseScene):
    cfg: BellHerdCfg

    def __init__(self, cfg: BellHerdCfg | None = None) -> None:
        super().__init__(cfg or BellHerdCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def ball_cfg(color: tuple) -> Any:
            return sim_utils.SphereCfg(
                radius=c.ball_r,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.05, angular_damping=0.20,
                    max_depenetration_velocity=0.5,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    sleep_threshold=0.0, stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.5, dynamic_friction=0.4, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        fx, fy = c.fix_pos
        z0 = c.surface_h
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
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                spawn=_board_spawner_cfg(
                    half_ext=c.half_ext, pit_half=c.pit_half, pit_x=c.pit_x,
                    pit_depth=c.pit_depth, fence_h=c.fence_h, fence_t=c.fence_t,
                    color=c.board_color, fence_color=c.fence_color,
                    pit_floor_color=c.pit_floor_color, contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(fx, fy, z0)),
            ),
            "bell": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bell",
                spawn=_bell_spawner_cfg(
                    mouth_r=c.mouth_r, wall_t=c.wall_t, skirt_h=c.skirt_h,
                    top_r=c.top_r, top_t=c.top_t, knob_r=c.knob_r, knob_h=c.knob_h,
                    cap_r=c.cap_r, cap_t=c.cap_t, mass=c.bell_mass,
                    color=c.bell_color, knob_color=c.knob_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx + c.bell_pos[0], fy + c.bell_pos[1], z0 + 0.002)),
            ),
            "red": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedBall",
                spawn=ball_cfg(c.red_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx + c.red_pos[0], fy + c.red_pos[1], z0 + c.ball_r + 0.002)),
            ),
            "blue": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BlueBall",
                spawn=ball_cfg(c.blue_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx + c.blue_pos[0], fy + c.blue_pos[1], z0 + c.ball_r + 0.002)),
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
        self.board: RigidObject = env.iscene["board"]
        self.bell: RigidObject = env.iscene["bell"]
        self.red: RigidObject = env.iscene["red"]
        self.blue: RigidObject = env.iscene["blue"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.red_side = torch.ones(n, device=dev)  # +1: red on +y side; -1: -y side
        self.d0 = torch.full((n,), 0.40, device=dev)  # red-spawn -> pocket distance
        self.cage_latch = torch.zeros(n, device=dev)
        self.prog_latch = torch.zeros(n, device=dev)
        self.pit_latch = torch.zeros(n, device=dev)

    def _scatter(self, m: int, nominal: torch.Tensor, scatter: tuple,
                 keepouts: list[tuple[torch.Tensor, float]]) -> torch.Tensor:
        """(m,2) board-LOCAL xy around per-env `nominal` (m,2), resampled (12 tries,
        batched) until outside every (centre, radius) keep-out."""
        dev = self.env.device
        jit = torch.tensor(scatter, device=dev)
        xy = nominal + (torch.rand(m, 2, device=dev) * 2 - 1) * jit
        for _ in range(12):
            bad = torch.zeros(m, dtype=torch.bool, device=dev)
            for ctr, rad in keepouts:
                bad |= (xy - ctr).norm(dim=-1) < rad
            if not bad.any():
                break
            k = int(bad.sum())
            xy[bad] = nominal[bad] + (torch.rand(k, 2, device=dev) * 2 - 1) * jit
        return xy

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: board with xy jitter + yaw (the pocket moves — read the pose
        from the scene), a coin flip for which y-side holds the RED ball (blue and the
        bell mirror it), xy scatter with keep-outs (pocket, mutual); latches zeroed
        and the progress baseline `d0` captured."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- board: xy jitter + yaw ---
        fix_xy = torch.tensor(c.fix_pos, device=dev).expand(m, 2).clone()
        fix_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        qw, qz = torch.cos(yaw / 2), torch.sin(yaw / 2)

        def write(body, local_xy: torch.Tensor, z: float, with_yaw: bool) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = fix_xy[:, 0] + local_xy[:, 0] * cy - local_xy[:, 1] * sy
            st[:, 1] = fix_xy[:, 1] + local_xy[:, 0] * sy + local_xy[:, 1] * cy
            st[:, 2] = z
            if with_yaw:
                st[:, 3], st[:, 6] = qw, qz
            else:
                st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        write(self.board, torch.zeros(m, 2, device=dev), c.surface_h, with_yaw=True)

        # --- side coin flip + scattered spawns with keep-outs ---
        if c.side_flip:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               -torch.ones(m, device=dev), torch.ones(m, device=dev))
        else:
            side = torch.ones(m, device=dev)
        pit_c = torch.tensor([c.pit_x, 0.0], device=dev).expand(m, 2)

        def nom(pos: tuple) -> torch.Tensor:
            out = torch.tensor(pos, device=dev).expand(m, 2).clone()
            out[:, 1] *= side
            return out

        red_xy = self._scatter(m, nom(c.red_pos), c.ball_scatter,
                               [(pit_c, c.keepout_pit)])
        blue_xy = self._scatter(m, nom(c.blue_pos), c.ball_scatter,
                                [(pit_c, c.keepout_pit), (red_xy, c.keepout_mutual)])
        bell_xy = self._scatter(m, nom(c.bell_pos), c.bell_scatter,
                                [(pit_c, c.keepout_pit), (red_xy, c.keepout_mutual),
                                 (blue_xy, c.keepout_mutual)])
        z0 = c.surface_h
        write(self.red, red_xy, z0 + c.ball_r + 0.003, with_yaw=False)
        write(self.blue, blue_xy, z0 + c.ball_r + 0.003, with_yaw=False)
        write(self.bell, bell_xy, z0 + 0.003, with_yaw=True)

        # --- baselines + latches ---
        self.red_side[env_ids] = side
        self.d0[env_ids] = (red_xy - pit_c).norm(dim=-1).clamp(min=0.10)
        self.cage_latch[env_ids] = 0.0
        self.prog_latch[env_ids] = 0.0
        self.pit_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "board": self.board.data.root_state_w[env_ids].clone(),
            "bell": self.bell.data.root_state_w[env_ids].clone(),
            "red": self.red.data.root_state_w[env_ids].clone(),
            "blue": self.blue.data.root_state_w[env_ids].clone(),
            "red_side": self.red_side[env_ids].clone(),
            "d0": self.d0[env_ids].clone(),
            "cage_latch": self.cage_latch[env_ids].clone(),
            "prog_latch": self.prog_latch[env_ids].clone(),
            "pit_latch": self.pit_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.board.write_root_state_to_sim(state["board"], env_ids)
        self.bell.write_root_state_to_sim(state["bell"], env_ids)
        self.red.write_root_state_to_sim(state["red"], env_ids)
        self.blue.write_root_state_to_sim(state["blue"], env_ids)
        self.red_side[env_ids] = state["red_side"]
        self.d0[env_ids] = state["d0"]
        self.cage_latch[env_ids] = state["cage_latch"]
        self.prog_latch[env_ids] = state["prog_latch"]
        self.pit_latch[env_ids] = state["pit_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A raised square BOARD ({2 * c.half_ext * 1000:.0f} x "
            f"{2 * c.half_ext * 1000:.0f} mm, play surface {c.surface_h * 1000:.0f} mm "
            f"above the floor) is ringed by a low dark fence "
            f"({c.fence_h * 1000:.0f} mm) so nothing rolls off. Near one side of the "
            f"board a square POCKET ({2 * c.pit_half * 1000:.0f} x "
            f"{2 * c.pit_half * 1000:.0f} mm) is sunk {c.pit_depth * 1000:.0f} mm below "
            f"the surface: it reads as a dark square opening with a YELLOW floor — the "
            f"only opening in the board. On the surface sit two loose balls, one RED and "
            f"one BLUE ({2 * c.ball_r * 1000:.0f} mm diameter, "
            f"{c.ball_mass * 1000:.0f} g each): both are WIDER than a parallel-jaw "
            f"gripper's {c.jaw_span * 1000:.0f} mm span, so neither ball can be picked "
            f"up — they can only be caged or nudged, and they roll freely. Also on the "
            f"board stands an open-bottomed steel-blue BELL (~{2 * c.top_r * 1000:.0f} "
            f"mm across, {c.skirt_h * 1000:.0f} mm tall skirt) with a "
            f"{2 * c.knob_r * 1000:.0f} mm grasp KNOB and cap on top — its open mouth "
            f"(inner width {2 * c.mouth_r * 1000:.0f} mm) swallows a ball with "
            f"{c.cage_r * 1000:.0f} mm of slack, and its rim is wider than the pocket, "
            f"so the bell itself can never fall in.\n"
            f"Goal: get the RED ball to rest INSIDE the yellow-floored pocket, keep the "
            f"BLUE ball OUT of it, and finish with the bell parked well clear of the "
            f"pocket (at least {c.bell_clear * 1000:.0f} mm away) so the delivery is "
            f"visible. The intended tool-use: grasp the bell by its knob, lower it over "
            f"the red ball to cage it, DRAG the loaded bell flat across the surface — "
            f"the skirt wall herds the rolling ball — until the cage crosses the pocket "
            f"and the ball drops through, then lift the empty bell away and set it down "
            f"clear. There is no prescribed path and no required order, but dropping the "
            f"BLUE ball into the pocket is a failure you cannot practically undo (no "
            f"ball can be grasped). Judged only when everything is settled."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Cage the red ball under the knobbed bell, drag the bell across the board "
            "until the ball drops into the yellow-floored pocket, then lift the bell "
            "away and set it down clear of the pocket. Keep the blue ball out of the "
            "pocket."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _board_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> board body frame (origin = play-surface centre,
        z = 0 at the surface)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.board.data.root_quat_w,
                                  p_w - self.board.data.root_pos_w)

    def _pit_dist(self, body) -> torch.Tensor:
        """(N,) horizontal distance of the body's centre from the pocket centre
        (board frame)."""
        c = self.cfg
        loc = self._board_local(body.data.root_pos_w)
        pit = torch.tensor([c.pit_x, 0.0], device=loc.device)
        return (loc[:, :2] - pit).norm(dim=-1)

    # ----- predicates -------------------------------------------------------------------------
    def in_pocket(self, body) -> torch.Tensor:
        """(N,) bool: body centre inside the pocket square AND below the play surface
        (board frame) — real containment, not proximity."""
        c = self.cfg
        loc = self._board_local(body.data.root_pos_w)
        return ((loc[:, 0] - c.pit_x).abs() < c.pit_half - c.pit_margin) \
            & (loc[:, 1].abs() < c.pit_half - c.pit_margin) \
            & (loc[:, 2] < -c.drop_min)

    def caged(self) -> torch.Tensor:
        """(N,) bool: the RED ball is enclosed by the GROUNDED bell — bell mouth down
        on the surface, ball axis within the mouth, ball on the surface (not perched
        on top of the bell)."""
        c = self.cfg
        bell_loc = self._board_local(self.bell.data.root_pos_w)
        ball_loc = self._board_local(self.red.data.root_pos_w)
        bell_down = bell_loc[:, 2] < 0.015  # mouth (origin) at the surface
        d = (ball_loc[:, :2] - bell_loc[:, :2]).norm(dim=-1)
        inside = d < c.cage_r + c.cage_slack
        on_surface = (ball_loc[:, 2] > 0.0) & (ball_loc[:, 2] < c.skirt_h - c.ball_r)
        return bell_down & inside & on_surface

    def bell_is_clear(self) -> torch.Tensor:
        """(N,) bool: bell axis farther than `bell_clear` from the pocket centre."""
        return self._pit_dist(self.bell) > self.cfg.bell_clear

    def settled(self) -> torch.Tensor:
        """(N,) bool: both balls and the bell below `settle_lin`."""
        c = self.cfg
        return ((self.red.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.blue.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                & (self.bell.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))

    # ----- graded progress --------------------------------------------------------------------
    def progress_frac(self) -> torch.Tensor:
        """(N,) in [0,1]: red-ball progress toward the pocket, normalized by the
        episode's own spawn distance."""
        return (1.0 - self._pit_dist(self.red) / self.d0).clamp(0.0, 1.0)

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch caging, best pocket progress, and pocket entry each physics substep,
        so transient progress keeps its credit."""
        self.cage_latch = torch.maximum(self.cage_latch, self.caged().float())
        self.prog_latch = torch.maximum(self.prog_latch, self.progress_frac())
        self.pit_latch = torch.maximum(self.pit_latch, self.in_pocket(self.red).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: RED ball settled inside the pocket + BLUE ball out of it + the
        bell parked clear of the pocket, everything settled."""
        return (self.in_pocket(self.red) & ~self.in_pocket(self.blue)
                & self.bell_is_clear() & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 * red ball ever caged under the grounded bell +
        0.25 * latched pocket progress + 0.30 * red ball ever in the pocket, capped at
        0.70; exactly 1.0 iff success(). Doing nothing scores ~0; the seed's strategy
        (carry the bowl-shaped object itself along a path and put it somewhere) moves
        no ball and earns ~0."""
        base = (0.15 * self.cage_latch + 0.25 * self.prog_latch
                + 0.30 * self.pit_latch).clamp(0.0, 0.70)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="bell_herd", robot="null"))
