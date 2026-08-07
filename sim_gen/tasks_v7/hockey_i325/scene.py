"""GateGoalScene — lift the captive gate board out of the goal mouth's channel slots,
then convey the white ball along the floor through the low mouth into the covered goal
chamber. Derived from rlbench/hockey but the goal is SEALED and ROOFED, so the seed's
plan is dead on arrival.

Seed (rlbench/hockey): grasp a hockey stick and STRIKE the ball across open floor into
an open-mouthed goal — one ballistic tool swing at a passively available receptacle.
Here the receptacle fights back and the tool is gone:

- The goal becomes a ROOFED CHAMBER whose only ball-sized opening is a floor-level
  MOUTH on the near side. The roof covers the whole interior: a lobbed or dropped ball
  just rests on the roof (smoke #7).
- The mouth is sealed by a PORTCULLIS: a yellow GATE BOARD standing in a vertical
  channel (front/rear post pairs + end caps). The board is CAPTIVE — it cannot tip
  over, cannot slide sideways (end caps), cannot be pushed through (rear posts); the
  ONLY way out is a straight vertical extraction of ~13 cm until its bottom edge
  clears the channel posts. The seed's move — drive the ball at the goal — bounces off
  the closed gate and scores nothing (smoke #6 fires the shot and watches it bounce).
- With the gate out, the open top slot above the channel is a 50 mm strip — narrower
  than the 60 mm ball, so drop-in stays impossible everywhere (smoke #8); the ball
  must TRAVEL THE FLOOR through the mouth. No stick exists; the ball is pushed with
  the hand (or carried to the mouth — but never through the roof).
- A same-size BLACK distractor ball (slot-swapped with the white target every episode)
  adds a color-grounded identification clause and an exclusion clause (smoke #11-12).

So a solver needs a different PLAN (constrained vertical extraction of a captive
barrier, park it, then a controlled floor-level conveyance through a low aperture —
no tool, no strike) and different CODE STRUCTURE (channel-extraction control + push
control instead of grasp-stick + swing). Execution order is geometry-forced: gate
first, ball second — enforced by collision, not by rubric timestamps.

Assets are fully procedural (pen_holder-pattern compound spawners; child colliders of
one body never self-collide):
  - goal: KINEMATIC compound — two side walls, back wall, full roof (underside
    110 mm), and the gate channel: 20x20 mm front/rear post pairs (130 mm tall)
    flanking the mouth plus two end caps sealing the channel ends. Origin at the
    MOUTH CENTRE on the floor; local +u points INTO the chamber.
  - gate board: DYNAMIC yellow slab 8 x 200 x 170 mm seated in the channel (bottom on
    the floor, top edge 40 mm proud of the posts — the pinch feature).
  - white / black balls: DYNAMIC 60 mm spheres, angular damping so they settle.

Per-episode randomization (readback-verifiable): goal lateral offset + yaw (the board
and channel follow the goal frame), Bernoulli WHITE/BLACK slot swap + per-ball xy
jitter.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * gate_out   — board ever extracted out of the gate zone (latched; a captive
                      board cannot fake this — only a real vertical extraction, or a
                      probe teleport, ever moves it)
  0.15 * approach   — white-ball approach to the mouth centre, gated on gate_out
                      (latched running max; ~0 for doing nothing)
  0.45 * inside     — white ball ever inside the chamber past the mouth plane, under
                      the roof (latched)
  1.0 iff success() — white ball inside the chamber resting on the floor, at rest,
                      black ball NOT inside. Non-success cap 0.75.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- custom compound spawners ---------------------------------------------------------------
# One rigid body per object, several child colliders, authored with raw pxr APIs; only
# `isaaclab.sim.utils.clone` is borrowed (regex-resolve + per-env replication).

_SPAWNER_CACHE: dict[str, Any] = {}


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    """Author one axis-aligned box collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _root_xform(prim_path: str, translation, orientation):
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


def _spawn_goal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the goal: KINEMATIC compound. Origin at the mouth centre on the floor;
    interior spans u in [0, in_d], |v| <= in_w/2, roof underside at roof_z. The gate
    channel (front/rear post pairs + end caps) stands in front of the mouth around
    u = slot_u."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    wall_h = c.roof_z + c.roof_t  # walls reach the roof's top face
    u_len = c.in_d + c.t  # walls run from the mouth plane to the back wall's outside
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(u_len / 2, sgn * (c.in_w / 2 + c.t / 2), wall_h / 2),
                 size=(u_len, c.t, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back",
             center=(c.in_d + c.t / 2, 0.0, wall_h / 2),
             size=(c.t, c.in_w + 2 * c.t, wall_h), color=c.color, collide=collide)
    # roof: covers the whole interior, leading edge AT the mouth plane (u = 0)
    _add_box(stage, f"{prim_path}/roof",
             center=(u_len / 2, 0.0, c.roof_z + c.roof_t / 2),
             size=(u_len, c.in_w + 2 * c.t, c.roof_t),
             color=c.roof_color, collide=collide)
    # gate channel: front/rear post pairs flanking the mouth (the vertical slot the
    # board rides in) + end caps sealing the channel ends (the board cannot leave
    # sideways). The open-top strip between the front posts and the roof edge is
    # post-to-roof = 50 mm — narrower than the ball.
    half_gap = c.board_t / 2 + c.slot_play  # slot inner half-gap
    for du, tag in ((-half_gap - c.post_w / 2, "f"), (half_gap + c.post_w / 2, "r")):
        for sgn, nm in ((1.0, "l"), (-1.0, "r")):
            _add_box(stage, f"{prim_path}/post_{tag}{nm}",
                     center=(c.slot_u + du, sgn * (c.in_w / 2 + c.post_w / 2),
                             c.post_h / 2),
                     size=(c.post_w, c.post_w, c.post_h), color=c.channel_color,
                     collide=collide)
    cap_len = 2 * half_gap + 2 * c.post_w + 0.004
    for sgn, nm in ((1.0, "cap_l"), (-1.0, "cap_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(c.slot_u, sgn * (c.board_w / 2 + c.cap_play + c.t / 2),
                         c.post_h / 2),
                 size=(cap_len, c.t, c.post_h), color=c.channel_color, collide=collide)
    return root


def _spawn_board(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the gate board: DYNAMIC thin slab standing on its bottom edge. Origin at
    the slab's volumetric centre."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    _add_box(stage, f"{prim_path}/slab", center=(0.0, 0.0, 0.0),
             size=(cfg.board_t, cfg.board_w, cfg.board_h), color=cfg.color,
             collide=collide)
    return root


def _spawn_ball(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a ball: DYNAMIC sphere. Angular damping so it rolls to rest instead of
    circling; sleep thresholds zeroed (force-driven and judged for stillness)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.30)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    r = float(cfg.radius)
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    sph.CreateRadiusAttr(r)
    sph.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    sph.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _make_collide(cfg)(sph.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "goal" not in _SPAWNER_CACHE:

        @configclass
        class GoalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_goal)
            in_w: float = 0.16
            in_d: float = 0.16
            roof_z: float = 0.11
            roof_t: float = 0.012
            t: float = 0.012
            slot_u: float = -0.020
            slot_play: float = 0.006
            post_w: float = 0.020
            post_h: float = 0.13
            board_t: float = 0.008
            board_w: float = 0.20
            cap_play: float = 0.005
            color: tuple = (0.30, 0.38, 0.55)
            roof_color: tuple = (0.22, 0.28, 0.42)
            channel_color: tuple = (0.55, 0.58, 0.62)
            contact_offset: float = 0.002

        @configclass
        class BoardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_board)
            board_t: float = 0.008
            board_w: float = 0.20
            board_h: float = 0.17
            color: tuple = (0.93, 0.80, 0.12)
            contact_offset: float = 0.002

        @configclass
        class BallSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_ball)
            radius: float = 0.03
            color: tuple = (0.95, 0.95, 0.95)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(goal=GoalSpawnerCfg, board=BoardSpawnerCfg,
                              ball=BallSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class GateGoalSceneCfg(BaseCfg):
    """Config for `GateGoalScene`. The interlocks are metric: the roof covers the whole
    interior (drop-in past the mouth plane is impossible), the seated board covers the
    full mouth (200 mm board vs 160 mm mouth, bottom on the floor — no ball-sized gap
    anywhere), the channel makes the board captive (end caps sideways, post pairs
    fore/aft; the only exit is ~130 mm straight up), and the open-top strip left after
    extraction (50 mm front-post-to-roof) is narrower than the 60 mm ball."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.04)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    approach_d0: float = tunable(0.40)  # approach ramp: p = 1 - d/approach_d0
    in_u_min: float = tunable(0.045)  # "inside" begins at mouth plane + ball r + 15 mm

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    goal_dy_max: float = tunable(0.05)  # goal lateral offset (+/- m)
    goal_yaw_max_deg: float = tunable(10.0)  # goal yaw (+/- deg)
    slot_jitter: float = tunable(0.03)  # per-ball spawn xy jitter (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli white/black slot swap

    # --- info: layout (single Franka base at the origin; radii 0.30-0.60 m) ---------------------
    goal_x: float = info(0.50)  # mouth plane distance from the base
    slot_a: tuple = info((0.32, 0.14))  # ball spawn slot A (left)
    slot_b: tuple = info((0.32, -0.14))  # ball spawn slot B (right)
    park_spot: tuple = info((0.16, 0.42))  # a free floor patch (describe() suggests it)

    # --- info: goal structure --------------------------------------------------------------------
    in_w: float = info(0.16)  # interior width (v)
    in_d: float = info(0.16)  # interior depth (u), mouth plane to back wall
    roof_z: float = info(0.11)  # roof underside height
    roof_t: float = info(0.012)
    t: float = info(0.012)  # wall thickness
    slot_u: float = info(-0.020)  # gate slot centre (u, in front of the mouth plane)
    slot_play: float = info(0.006)  # slot fore/aft clearance around the board
    post_w: float = info(0.020)  # channel post cross-section
    post_h: float = info(0.13)  # channel post height (extraction stroke)
    cap_play: float = info(0.005)  # sideways clearance board <-> end caps
    goal_color: tuple = info((0.30, 0.38, 0.55))  # slate blue
    roof_color: tuple = info((0.22, 0.28, 0.42))
    channel_color: tuple = info((0.55, 0.58, 0.62))  # gray

    # --- info: gate board ------------------------------------------------------------------------
    board_t: float = info(0.008)  # thickness (u) — the pinch span
    board_w: float = info(0.20)  # width (v): full mouth + 20 mm into each channel
    board_h: float = info(0.17)  # height: top edge 40 mm proud of the posts
    board_mass: float = info(0.12)
    board_color: tuple = info((0.93, 0.80, 0.12))  # yellow

    # --- info: balls -----------------------------------------------------------------------------
    ball_r: float = info(0.03)  # 60 mm dia — wider than every non-mouth opening
    ball_mass: float = info(0.08)
    white_color: tuple = info((0.95, 0.95, 0.95))
    black_color: tuple = info((0.08, 0.08, 0.08))

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.15 + 0.45 = 0.75 = the non-success cap)
    w_gate: float = info(0.15)
    w_app: float = info(0.15)
    w_in: float = info(0.45)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("gate_goal")
class GateGoalScene(BaseScene):
    cfg: GateGoalSceneCfg

    def __init__(self, cfg: GateGoalSceneCfg | None = None) -> None:
        super().__init__(cfg or GateGoalSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        goal_spawn = spawners["goal"](
            mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            in_w=c.in_w, in_d=c.in_d, roof_z=c.roof_z, roof_t=c.roof_t, t=c.t,
            slot_u=c.slot_u, slot_play=c.slot_play, post_w=c.post_w, post_h=c.post_h,
            board_t=c.board_t, board_w=c.board_w, cap_play=c.cap_play,
            color=c.goal_color, roof_color=c.roof_color,
            channel_color=c.channel_color, contact_offset=c.contact_offset,
        )
        board_spawn = spawners["board"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.board_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            board_t=c.board_t, board_w=c.board_w, board_h=c.board_h,
            color=c.board_color, contact_offset=c.contact_offset,
        )

        def ball_spawn(color):
            return spawners["ball"](
                mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                radius=c.ball_r, color=color, contact_offset=c.contact_offset,
            )

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
            "goal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Goal",
                spawn=goal_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.goal_x, 0.0, 0.0)),
            ),
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                spawn=board_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.goal_x + c.slot_u, 0.0, c.board_h / 2 + 0.002)),
            ),
            "white": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/WhiteBall",
                spawn=ball_spawn(c.white_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.ball_r + 0.002)),
            ),
            "black": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/BlackBall",
                spawn=ball_spawn(c.black_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_b[0], c.slot_b[1], c.ball_r + 0.002)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.goal: RigidObject = env.iscene["goal"]
        self.board: RigidObject = env.iscene["board"]
        self.white: RigidObject = env.iscene["white"]
        self.black: RigidObject = env.iscene["black"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._gate_out = torch.zeros(n, dtype=torch.bool, device=dev)  # board ever out
        self._app_max = torch.zeros(n, device=dev)  # mouth approach, running max
        self._inside_ever = torch.zeros(n, dtype=torch.bool, device=dev)  # ever inside

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: goal re-posed with a random lateral offset + yaw, gate board
        re-seated in the channel (it follows the goal frame), balls randomly ASSIGNED
        to the two spawn slots (+ xy jitter), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- goal (kinematic): lateral offset + yaw ---
        dy = (torch.rand(m, device=dev) * 2 - 1) * c.goal_dy_max
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.goal_yaw_max_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = c.goal_x, dy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.goal.write_root_state_to_sim(st, env_ids)
        g_pos, g_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- gate board: seated in the channel, expressed in the goal frame ---
        from isaaclab.utils.math import quat_apply

        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0], loc[:, 2] = c.slot_u, c.board_h / 2 + 0.002
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = g_pos + quat_apply(g_quat, loc)
        st[:, 3:7] = g_quat
        self.board.write_root_state_to_sim(st, env_ids)

        # --- balls: Bernoulli slot swap + jitter ---
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        w_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        b_xy = torch.where(swap.unsqueeze(1), slot_a, slot_b) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        for body, xy in ((self.white, w_xy), (self.black, b_xy)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = c.ball_r + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._gate_out[env_ids] = False
        self._app_max[env_ids] = 0.0
        self._inside_ever[env_ids] = False

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "goal": self.goal.data.root_state_w[env_ids].clone(),
            "board": self.board.data.root_state_w[env_ids].clone(),
            "white": self.white.data.root_state_w[env_ids].clone(),
            "black": self.black.data.root_state_w[env_ids].clone(),
            "gate_out": self._gate_out[env_ids].clone(),
            "app_max": self._app_max[env_ids].clone(),
            "inside_ever": self._inside_ever[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.goal.write_root_state_to_sim(state["goal"], env_ids)
        self.board.write_root_state_to_sim(state["board"], env_ids)
        self.white.write_root_state_to_sim(state["white"], env_ids)
        self.black.write_root_state_to_sim(state["black"], env_ids)
        self._gate_out[env_ids] = state["gate_out"]
        self._app_max[env_ids] = state["app_max"]
        self._inside_ever[env_ids] = state["inside_ever"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"On the floor stands a slate-blue GOAL BOX: a covered chamber (interior "
            f"{c.in_w * 100:.0f} cm wide x {c.in_d * 100:.0f} cm deep, roof underside "
            f"{c.roof_z * 100:.0f} cm) whose ONLY ball-sized opening is a floor-level "
            f"MOUTH on the side facing the robot. The roof covers the entire interior, "
            f"so nothing can be dropped in from above — a ball released over the box "
            f"just rests on the roof. The mouth is SEALED by a bright YELLOW GATE "
            f"BOARD ({c.board_w * 100:.0f} x {c.board_h * 100:.0f} cm, "
            f"{c.board_t * 1000:.0f} mm thick) standing in a gray vertical CHANNEL "
            f"(post pairs in front of and behind the board, end caps at its sides): "
            f"the board cannot tip over, cannot slide sideways and cannot be pushed "
            f"through — the ONLY way to open the goal is to grip the board's exposed "
            f"top edge (it sticks up {100 * (c.board_h - c.post_h):.0f} cm above the "
            f"posts) and slide it STRAIGHT UP about {c.post_h * 100:.0f} cm until it "
            f"leaves the channel, then set it down somewhere clear (e.g. the open "
            f"floor at ({c.park_spot[0]:.2f}, {c.park_spot[1]:.2f})). Even with the "
            f"gate out, the open slot above the channel is only ~5 cm across — "
            f"narrower than the ball — so the ball can only enter along "
            f"the floor, through the mouth. In front of the box lie TWO "
            f"{2 * c.ball_r * 100:.0f} cm balls (positions swap between episodes — "
            f"identify by COLOR): a WHITE ball and a BLACK ball.\n"
            f"Goal: the WHITE ball must end up INSIDE the goal chamber — through the "
            f"mouth, past the mouth plane — resting on the floor at rest; the BLACK "
            f"ball must remain OUTSIDE. First extract the yellow gate board straight "
            f"up out of its channel and park it clear, then move the white ball along "
            f"the floor through the mouth (push it in; a gentle rolling push is "
            f"enough — there is no stick and none is needed). Shooting the ball at "
            f"the closed gate, leaving it against the gate or short of the mouth "
            f"plane, dropping it on the roof or into the narrow gate slot, putting "
            f"the BLACK ball in (alone or additionally) — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the yellow gate board straight up out of its channel on the goal "
            "box and set it aside, then push the white ball along the floor through "
            "the open mouth so it rests inside the covered box. Keep the black ball "
            "outside."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _goal_local(self, body: RigidObject) -> torch.Tensor:
        """(N, 3) body CoM position in the goal frame (u into the chamber, v across
        the mouth, z up)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = body.data.root_pos_w - self.goal.data.root_pos_w
        return quat_apply_inverse(self.goal.data.root_quat_w, rel)

    def _board_in_gate(self) -> torch.Tensor:
        """(N,) bool: board CoM inside the gate zone — seated (or nearly seated) in
        the channel, still low enough to bar the mouth. The captive channel means the
        board can only leave this zone through a real ~13 cm vertical extraction."""
        loc = self._goal_local(self.board)
        return ((loc[:, 0] >= -0.075) & (loc[:, 0] <= 0.035)
                & (loc[:, 1].abs() <= 0.135)
                & (loc[:, 2] >= 0.025) & (loc[:, 2] <= 0.155))

    def _inside(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool: body CoM inside the chamber — past the mouth plane by ball r +
        15 mm, between the walls, resting height under the roof."""
        c = self.cfg
        loc = self._goal_local(body)
        return ((loc[:, 0] >= c.in_u_min) & (loc[:, 0] <= c.in_d - 0.005)
                & (loc[:, 1].abs() <= c.in_w / 2 - 0.005)
                & (loc[:, 2] > 0.005) & (loc[:, 2] < 0.095))

    def _update_latches(self) -> None:
        c = self.cfg
        self._gate_out |= ~self._board_in_gate()
        # mouth approach, gated on the gate being out first
        mouth = self.goal.data.root_pos_w.clone()
        mouth[:, 2] += c.ball_r
        d = (self.white.data.root_pos_w - mouth).norm(dim=-1)
        app = (1.0 - d / c.approach_d0).clamp(0.0, 1.0) * self._gate_out.float()
        app = torch.nan_to_num(app, nan=0.0, posinf=0.0, neginf=0.0)
        self._app_max = torch.maximum(self._app_max, app)
        self._inside_ever |= self._inside(self.white)

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant here (the goal is kinematic and jointless) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: white ball inside the chamber (past the mouth plane, resting on
        the floor), at rest, black ball NOT inside. Physical outcomes only."""
        c = self.cfg
        self._update_latches()
        w_still = ((self.white.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                   & (self.white.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))
        return self._inside(self.white) & w_still & ~self._inside(self.black)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*gate_out + 0.15*mouth-approach (gated on
        gate_out) + 0.45*inside — all latched, ~0 for doing nothing, capped 0.75 —
        and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_gate * self._gate_out.float() + c.w_app * self._app_max
                + c.w_in * self._inside_ever.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="gate_goal", robot="null"))
