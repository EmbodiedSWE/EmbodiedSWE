"""BoardLeanScene — prop the board LEANING against the display wall (sim_gen task
`pick_single_ycb_i53`, derived from maniskill/pick_single_ycb).

The seed is a prehensile HOIST: grasp the single free object and raise it 7.5 cm (its
checker is literally a +z position-shift on the object; the dense variant adds "hold it
at a goal point, grasped, arm static"). The object's final support is the GRIPPER — the
task ends with the object held in the air, and any grasp that lifts wins.

This task keeps the same minimal object world (ONE free body, one fixture, a ground
plane) but replaces the transport/hoist goal with a PLACEMENT-QUALITY goal that the
gripper cannot provide and holding cannot satisfy: the board must end in a metastable
two-point FRICTION EQUILIBRIUM — bottom edge on the floor inside a marked lane, top edge
resting against a slick vertical display wall, tilted 45-80 deg — released and settled,
supported by nothing but gravity, floor friction and the wall. The judged quantity is
the final unpowered equilibrium, not a displacement:

  - the seed's own plan (lift the board and hold it up) scores ~0 — a held/hovering
    board fails the on-floor + settled gates and earns no latch (tested control);
  - the generic pick-and-place instinct (set the object down flat AT the target) scores
    only the 0.2 staging credit — a flat board in the lane, even touching the wall, is
    not a lean (tested control);
  - parking the board near-vertical against the wall fails the lean band (>80 deg /
    overhang < 40 mm) — the tolerance near-miss control;
  - leaning it too shallow (~25 deg) is rejected by PHYSICS, not code: below the slip
    angle (analytic tan(th) = (1 - mu_f*mu_w)/(2*mu_f) ~ 38 deg for floor mu 0.6, slick
    wall mu 0.1) the bottom edge slides out and the board falls flat — the calibration
    probe measures this cliff.

Success is honest BY CONSTRUCTION: a settled board whose bottom end is on the floor and
whose top end is raised, tilted toward the wall with >= 40 mm of horizontal overhang and
its top end at the wall plane, has its CoM outside the floor-contact support — no such
pose is a static equilibrium without the wall actually carrying load.

Rubric (graded, latched transients):
  0.0  nothing / held aloft / board anywhere outside the lane
  0.2  `staged` latch: the board's bottom end entered the lane on the floor (any pose)
  0.6  `propped` latch: a full valid lean was achieved (settled) at some point
  1.0  iff success() NOW: valid lean, settled, in lane

Per-episode randomization (READBACK-verified in the smoke): the wall rack is kinematic
and re-posed every reset (xy jitter + yaw), and the board spawns flat at a sampled
distance/lateral offset/free yaw in front of the rack, always OUTSIDE the lane.

Assets are fully procedural: the board is a plain CuboidCfg (local +z = long axis); the
rack is one kinematic compound spawner (wall panel collider with a slick physics
material bound at the root, plus a visual-only floor lane mat and wall target stripe —
the pen-holder cone-tip pattern). Heavy imports (isaaclab, pxr) are deferred so
importing this module stays app-free.
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


# ----- rack compound spawner ---------------------------------------------------------------
# One KINEMATIC rigid body: the vertical wall panel (the only collider, bound to a slick
# physics material so the lean's slip cliff is deterministic) plus visual-only decorations:
# a floor lane mat in front of the wall and a target stripe on the wall face.

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the display rack at `prim_path`. Root Xform with a KINEMATIC RigidBodyAPI.
    Rack local frame: wall face normal = +x (the lane side), wall face plane at local
    x = wall_t/2, root at the wall panel center (so the floor is local z = -wall_h/2)."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    # wall panel — the only collider
    wall = UsdGeom.Cube.Define(stage, f"{prim_path}/wall")
    wall.CreateSizeAttr(1.0)
    wxf = UsdGeom.Xformable(wall.GetPrim())
    wxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, cfg.wall_w, cfg.wall_h))
    wall.CreateDisplayColorAttr([Gf.Vec3f(*cfg.wall_color)])
    UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(wall.GetPrim())
    px.CreateContactOffsetAttr(float(cfg.contact_offset))
    px.CreateRestOffsetAttr(0.0)

    # visual-only lane mat on the floor in front of the wall (NO CollisionAPI)
    mat = UsdGeom.Cube.Define(stage, f"{prim_path}/lane_mat")
    mat.CreateSizeAttr(1.0)
    mxf = UsdGeom.Xformable(mat.GetPrim())
    mxf.AddTranslateOp().Set(Gf.Vec3d(cfg.wall_t / 2 + cfg.lane_depth / 2, 0.0,
                                      -cfg.wall_h / 2 + 0.001))
    mxf.AddScaleOp().Set(Gf.Vec3f(cfg.lane_depth, cfg.wall_w, 0.002))
    mat.CreateDisplayColorAttr([Gf.Vec3f(*cfg.lane_color)])

    # visual-only target stripe on the wall face (NO CollisionAPI)
    stripe = UsdGeom.Cube.Define(stage, f"{prim_path}/stripe")
    stripe.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(stripe.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(cfg.wall_t / 2 + 0.001, 0.0, cfg.wall_h * 0.12))
    sxf.AddScaleOp().Set(Gf.Vec3f(0.002, cfg.wall_w * 0.9, cfg.wall_h * 0.30))
    stripe.CreateDisplayColorAttr([Gf.Vec3f(*cfg.stripe_color)])

    # slick physics material bound at the root (combine-mode "min" -> the board-wall pair
    # friction is the wall's mu no matter what the board declares: the slip-angle cliff
    # tan(th) = (1 - mu_f*mu_w)/(2*mu_f) stays where the calibration probe expects it).
    # Bind AFTER the collider children exist (the i33 lesson: earlier binds silently no-op).
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(cfg.wall_mu), dynamic_friction=float(cfg.wall_mu) * 0.9,
        restitution=0.0, friction_combine_mode="min", restitution_combine_mode="min")
    mat_cfg.func(f"{prim_path}/physmat", mat_cfg)
    bind_physics_material(prim_path, f"{prim_path}/physmat")
    return root


def _rack_spawner_cfg(*, wall_t: float, wall_w: float, wall_h: float, lane_depth: float,
                      wall_mu: float, wall_color: tuple, lane_color: tuple,
                      stripe_color: tuple, contact_offset: float) -> Any:
    """Build (lazily, app required) the rack spawner cfg — `clone` wraps `_spawn_rack`
    exactly like `spawn_cuboid` is wrapped."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class LeanRackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            wall_t: float = 0.05
            wall_w: float = 0.48
            wall_h: float = 0.50
            lane_depth: float = 0.35
            wall_mu: float = 0.10
            wall_color: tuple = (0.30, 0.34, 0.42)
            lane_color: tuple = (0.85, 0.70, 0.15)
            stripe_color: tuple = (0.90, 0.30, 0.10)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["rack"] = LeanRackSpawnerCfg

    return _SPAWNER_CACHE["rack"](
        mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        wall_t=wall_t, wall_w=wall_w, wall_h=wall_h, lane_depth=lane_depth,
        wall_mu=wall_mu, wall_color=wall_color, lane_color=lane_color,
        stripe_color=stripe_color, contact_offset=contact_offset,
    )


# ----- scene cfg -----------------------------------------------------------------------------
@dataclass
class BoardLeanSceneCfg(BaseCfg):
    """Config for `BoardLeanScene`. Rack local frame: face normal = +x, face plane at
    local x = wall_t/2, floor at local z = -wall_h/2."""

    # --- tunable: rubric thresholds --------------------------------------------------------
    theta_min_deg: float = tunable(45.0)   # lean band: tilt from the floor, lower edge
    theta_max_deg: float = tunable(80.0)   # lean band upper edge (steeper = "parked", fails)
    min_overhang: float = tunable(0.040)   # top end must overhang the bottom end toward the
    # wall by this much horizontally — with the on-floor gate this makes wall support
    # physically NECESSARY for a settled pose (honesty by construction).
    top_gap_lo: float = tunable(-0.030)    # top end center to wall face plane, near band (m)
    top_gap_hi: float = tunable(0.050)
    bottom_z_max: float = tunable(0.060)   # bottom end center height: resting on the floor
    top_z_min: float = tunable(0.280)      # top end must be genuinely raised (m)
    lane_top_margin: float = tunable(0.040)  # top end |y| <= wall_w/2 - this (no side-edge leans)
    settle_lin: float = tunable(0.05)      # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.50)      # max |ang vel| when judging settled (rad/s)
    stage_z_max: float = tunable(0.050)    # staged latch: bottom end near the floor
    stage_lin_max: float = tunable(0.50)   # staged latch: not flying through

    # --- tunable: randomization (the task-family knobs) ------------------------------------
    rack_jitter: float = tunable(0.05)     # uniform +/- xy jitter of the rack at reset (m)
    rack_yaw_deg: float = tunable(25.0)    # uniform +/- rack yaw at reset
    board_dist_range: tuple = tunable((0.64, 0.78))  # board spawn distance from the wall
    # face along rack +x; lower bound keeps the whole flat board OUTSIDE the 0.35 m lane
    # (0.64 - 0.04 jitter - 0.21 half-length = 0.39 > lane_depth) so null scores exactly 0.
    board_lat: float = tunable(0.12)       # uniform +/- lateral offset of the board spawn
    board_yaw_deg: float = tunable(180.0)  # uniform +/- board yaw at reset (lying flat)
    board_jitter: float = tunable(0.04)    # extra uniform +/- xy jitter of the board spawn

    # --- info: structure --------------------------------------------------------------------
    wall_h: float = info(0.50)     # taller than board_l * sin(80 deg) = 0.414: face contact
    wall_w: float = info(0.48)
    wall_t: float = info(0.05)
    lane_depth: float = info(0.35)  # board_l * cos(45 deg) = 0.297 < this: 45 deg fits
    wall_mu: float = info(0.10)    # slick wall (combine "min") -> slip cliff ~38 deg with
    board_mu: float = info(0.60)   # floor mu 0.6: tan(th_c) = (1 - 0.06)/1.2 = 0.783
    board_l: float = info(0.42)    # board local +z = long axis
    board_w: float = info(0.12)
    board_t: float = info(0.03)
    board_mass: float = info(0.30)
    board_color: tuple = info((0.75, 0.55, 0.25))
    wall_color: tuple = info((0.30, 0.34, 0.42))
    lane_color: tuple = info((0.85, 0.70, 0.15))
    stripe_color: tuple = info((0.90, 0.30, 0.10))
    contact_offset: float = info(0.003)


# ----- scene ---------------------------------------------------------------------------------
@SCENES.register("board_lean")
class BoardLeanScene(BaseScene):
    cfg: BoardLeanSceneCfg

    def __init__(self, cfg: BoardLeanSceneCfg | None = None) -> None:
        super().__init__(cfg or BoardLeanSceneCfg())

    # ----- assets -------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground (explicit 0.6 friction so the slip cliff is calibrated), light, the
        kinematic rack (re-posed by reset) and the free board lying flat in front."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.60, dynamic_friction=0.55, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=_rack_spawner_cfg(
                    wall_t=c.wall_t, wall_w=c.wall_w, wall_h=c.wall_h,
                    lane_depth=c.lane_depth, wall_mu=c.wall_mu, wall_color=c.wall_color,
                    lane_color=c.lane_color, stripe_color=c.stripe_color,
                    contact_offset=c.contact_offset,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.wall_h / 2)),
            ),
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                spawn=sim_utils.CuboidCfg(
                    size=(c.board_t, c.board_w, c.board_l),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.10),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.board_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.board_mu, dynamic_friction=c.board_mu - 0.05,
                        restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.board_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.70, 0.0, c.board_t / 2 + 0.003),
                    rot=(math.cos(math.pi / 4), 0.0, -math.sin(math.pi / 4), 0.0)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode latches."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.board: RigidObject = env.iscene["board"]
        self.rack: RigidObject = env.iscene["rack"]
        self.env_origins = env.iscene.env_origins
        self.staged = torch.zeros(n, dtype=torch.bool, device=dev)   # bottom end entered lane
        self.propped = torch.zeros(n, dtype=torch.bool, device=dev)  # full lean achieved

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: re-pose the kinematic rack (xy jitter + yaw), lay the board flat
        at a sampled distance / lateral offset / free yaw in front of it (always outside
        the lane), clear the latches."""
        from isaaclab.utils.math import quat_mul

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        rack_xy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)

        st = torch.zeros(m, 7, device=dev)
        st[:, 0:2] = rack_xy
        st[:, 2] = c.wall_h / 2
        st[:, 3] = torch.cos(psi / 2)
        st[:, 6] = torch.sin(psi / 2)
        st[:, 0:3] += origin
        self.rack.write_root_pose_to_sim(st, env_ids)

        # board: flat (thickness vertical), in front of the rack along its +x
        lo, hi = c.board_dist_range
        d = lo + (hi - lo) * torch.rand(m, device=dev)
        lat = (torch.rand(m, device=dev) * 2 - 1) * c.board_lat
        cx = rack_xy[:, 0] + d * torch.cos(psi) - lat * torch.sin(psi)
        cy = rack_xy[:, 1] + d * torch.sin(psi) + lat * torch.cos(psi)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = cx
        st[:, 1] = cy
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.board_jitter
        st[:, 2] = c.board_t / 2 + 0.003
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.board_yaw_deg) / 2
        q_yaw = torch.zeros(m, 4, device=dev)
        q_yaw[:, 0] = torch.cos(half)
        q_yaw[:, 3] = torch.sin(half)
        q_flat = torch.zeros(m, 4, device=dev)  # pitch -90 deg about y: local +z -> horizontal
        q_flat[:, 0] = math.cos(-math.pi / 4)
        q_flat[:, 2] = math.sin(-math.pi / 4)
        st[:, 3:7] = quat_mul(q_yaw, q_flat)
        st[:, 0:3] += origin
        self.board.write_root_state_to_sim(st, env_ids)

        self.staged[env_ids] = False
        self.propped[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Every physics substep: latch `staged` (EITHER board end entered the lane on the
        floor, not flying — a flat board's bottom/top labels are arbitrary) and `propped`
        (a full valid lean, settled, was achieved)."""
        c = self.cfg
        bot, top = self.board_ends()
        bot_loc, top_loc = self._ends_rack_frame()
        lin = self.board.data.root_lin_vel_w.norm(dim=-1)

        def _in_lane_low(loc: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
            in_lane = ((loc[:, 0] > c.wall_t / 2)
                       & (loc[:, 0] - c.wall_t / 2 <= c.lane_depth)
                       & (loc[:, 1].abs() <= c.wall_w / 2))
            return in_lane & ((world - self.env_origins)[:, 2] < c.stage_z_max)

        reached = _in_lane_low(bot_loc, bot) | _in_lane_low(top_loc, top)
        self.staged |= reached & (lin < c.stage_lin_max)
        self.propped |= self.lean_valid() & self.settled()

    # ----- state (full, restorable) ---------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "board": self.board.data.root_state_w[env_ids].clone(),
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "staged": self.staged[env_ids].clone(),
            "propped": self.propped[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.board.write_root_state_to_sim(state["board"], env_ids)
        self.rack.write_root_pose_to_sim(state["rack"][:, 0:7], env_ids)
        self.staged[env_ids] = state["staged"]
        self.propped[env_ids] = state["propped"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden board ({c.board_l * 1000:.0f} x {c.board_w * 1000:.0f} x "
            f"{c.board_t * 1000:.0f} mm) lies flat on the ground in front of a free-standing "
            f"display wall ({c.wall_h * 1000:.0f} mm tall, slick face with an orange target "
            f"stripe). A yellow lane mat on the floor marks the display zone at the foot of "
            f"the wall; the wall's position and heading change every episode.\n"
            f"Goal: prop the board up LEANING against the wall face — bottom end resting on "
            f"the floor inside the lane, top end against the wall, tilted between "
            f"{c.theta_min_deg:.0f} and {c.theta_max_deg:.0f} degrees from the floor — and "
            f"let go, so friction and the wall alone hold it. Holding it up scores nothing; "
            f"laying it flat in the lane is only staging; parking it near-vertical is not a "
            f"lean; too shallow and it will slide out and fall on its own."
        )

    # ----- progress / rubric --------------------------------------------------------------------
    def board_ends(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(bottom_end (N,3), top_end (N,3)) world-frame end-center points of the board's
        long axis; "top" is the end with the greater z."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        axis = quat_apply(self.board.data.root_quat_w, ez)
        p = self.board.data.root_pos_w
        e1 = p - axis * (self.cfg.board_l / 2)
        e2 = p + axis * (self.cfg.board_l / 2)
        hi = (e2[:, 2] >= e1[:, 2]).unsqueeze(-1)
        top = torch.where(hi, e2, e1)
        bot = torch.where(hi, e1, e2)
        return bot, top

    def _ends_rack_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Board end points in the RACK body frame (face normal = +x, face at x = wall_t/2)."""
        from isaaclab.utils.math import quat_apply_inverse

        bot, top = self.board_ends()
        rq = self.rack.data.root_quat_w
        rp = self.rack.data.root_pos_w
        return quat_apply_inverse(rq, bot - rp), quat_apply_inverse(rq, top - rp)

    def theta_deg(self) -> torch.Tensor:
        """(N,) board tilt from the floor plane, degrees (0 = flat, 90 = vertical)."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        axis = quat_apply(self.board.data.root_quat_w, ez)
        return torch.rad2deg(torch.asin(axis[:, 2].abs().clamp(max=1.0)))

    def settled(self) -> torch.Tensor:
        c = self.cfg
        lin = self.board.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        ang = self.board.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        return lin & ang

    def lean_valid(self) -> torch.Tensor:
        """(N,) bool, geometric: bottom end on the floor in the lane, top end raised, near
        the wall plane, tilt in the lean band, top overhanging the bottom toward the wall.
        (Instantaneous — combine with settled() to judge.)"""
        c = self.cfg
        bot, top = self.board_ends()
        bot_loc, top_loc = self._ends_rack_frame()
        bot_z = (bot - self.env_origins)[:, 2]
        top_z = (top - self.env_origins)[:, 2]
        th = self.theta_deg()
        in_lane = ((bot_loc[:, 0] > c.wall_t / 2)
                   & (bot_loc[:, 0] - c.wall_t / 2 <= c.lane_depth)
                   & (bot_loc[:, 1].abs() <= c.wall_w / 2))
        top_ok = ((top_loc[:, 0] - c.wall_t / 2 >= c.top_gap_lo)
                  & (top_loc[:, 0] - c.wall_t / 2 <= c.top_gap_hi)
                  & (top_loc[:, 1].abs() <= c.wall_w / 2 - c.lane_top_margin))
        overhang = (bot_loc[:, 0] - top_loc[:, 0]) >= c.min_overhang
        band = (th >= c.theta_min_deg) & (th <= c.theta_max_deg)
        return (in_lane & top_ok & overhang & band
                & (bot_z < c.bottom_z_max) & (top_z > c.top_z_min))

    def success(self) -> torch.Tensor:
        """(N,) bool: the board is CURRENTLY a settled, valid lean against the wall."""
        return self.lean_valid() & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0 for doing nothing; 0.2 once the board's bottom end has
        entered the lane on the floor (staged); 0.6 once a full settled lean was ever
        achieved (propped, latched); exactly 1.0 iff success() now."""
        n = self.env.num_envs
        s = torch.zeros(n, device=self.env.device)
        s = torch.where(self.staged, torch.full_like(s, 0.2), s)
        s = torch.where(self.propped, torch.full_like(s, 0.6), s)
        return torch.where(self.success(), torch.ones_like(s), s)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="board_lean", robot="null", env_spacing=4.0))
