"""ChuteDispatchScene — dispatch a cube DOWN a gravity chute into a far, unreachable,
walled pen marked by a green mat (sim_gen task `pull_cube_tool_i1`, derived from
maniskill/pull_cube_tool).

The seed pulls an out-of-reach cube INWARD with a grasped L-shaped tool (tool-mediated
reach extension toward the robot base). This task inverts the transfer direction and
removes the tool entirely: the cube starts WITHIN easy reach, and the goal — a walled
catch pen — lies far BEYOND reach. Two identical slick gravity chutes descend away from
the robot, each emptying into its own pen; one pen floor carries a GREEN mat (the
delivery target), the other a gray mat (a dead-end decoy), and which side is green is
sampled per episode. The only physical way into a pen is down its chute, so the solver
must read the marker, pick the cube up, and commit it to the correct chute mouth —
environmental (gravity) transport plus goal selection, instead of the seed's
tool-extended dragging. The seed's own plan direction ("bring the object toward your
base") is exactly wrong here, and proximity to the target counts for nothing: a cube
parked at the reachable limit next to the pen scores ~0.

Judged on PHYSICAL outcomes:
  - success() : the cube rests settled on the floor INSIDE the green-matted pen, having
    physically transited that pen's chute this episode (a latch set only by a
    continuous, in-channel crossing of a gate plane on the lower ramp — beyond arm
    reach — between consecutive physics steps; a kinematic teleport into the pen does
    NOT latch, so "teleport it into the goal" is not a solution).
  - score()   : 0 for doing nothing; 0.15 for a (continuously) lifted cube, +0.35 for
    the latched correct-chute transit; exactly 1.0 iff success. Wrong-chute transit
    earns nothing and is unrecoverable (the pens are out of reach). Max non-success
    score = 0.5.

Per-episode randomization: which pen is the green target (left/right), and the cube
spawn pose (xy + yaw) — so a memorized fixed drop fails; the solver must read the mats.

Assets are fully procedural (plain CuboidCfg boxes: two pitched ramp floors with side
walls and a mouth backstop, two walled pens, two thin floor mats, the cube). Heavy
imports (isaaclab) are deferred so importing this module — and registering the scene —
stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class ChuteDispatchSceneCfg(BaseCfg):
    """Config for `ChuteDispatchScene`. World frame: +x = away from the robot (down the
    chutes), y = across (left lane at +y, right lane at -y), ground top at z = 0."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    settle_lin: float = tunable(0.04)   # max |lin vel| when judging settled (m/s)
    settle_ang: float = tunable(0.50)   # max |ang vel| when judging settled (rad/s)
    rest_z_tol: float = tunable(0.012)  # cube-center height tolerance for "resting on floor"
    pen_margin: float = tunable(0.032)  # full-containment margin inside the pen: the cube's
    # half-DIAGONAL (0.045/sqrt(2) = 31.8 mm), so any yaw'd cube judged inside is inside.
    lift_z: float = tunable(0.12)       # cube-center height that latches "lifted"
    lift_dz_max: float = tunable(0.03)  # per-step rise cap for the lift latch (anti-teleport)
    gate_dx_max: float = tunable(0.05)  # per-step advance cap for the transit latch

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    spawn_x_range: tuple = tunable((0.02, 0.14))   # cube spawn x band (near the robot)
    spawn_y_range: tuple = tunable((-0.12, 0.12))  # cube spawn y band (between the mouths)
    spawn_yaw_deg: float = tunable(180.0)          # uniform +/- yaw (cube is symmetric)

    # --- tunable: difficulty geometry ---------------------------------------------------------
    chute_mu: float = tunable(0.06)   # chute surface friction (combine "min") — guarantees a
    # cube slides from rest anywhere on the 13.5 deg ramp (tan = 0.24 >> mu).
    floor_mu: float = tunable(0.45)   # ground / pen-floor friction — sets the run-out length.

    # --- info: structure ----------------------------------------------------------------------
    cube_size: float = info(0.045)
    cube_mass: float = info(0.04)
    lane_dy: float = info(0.17)       # lane centerlines at y = +/- lane_dy
    chan_w: float = info(0.095)       # channel inner width (cube 45 -> +/-25 mm capture)
    wall_t: float = info(0.012)
    chan_wall_h: float = info(0.06)   # channel side-wall height, normal to the ramp floor
    ramp_x0: float = info(0.26)       # mouth (top) end of the ramp floor's top surface
    ramp_x1: float = info(0.68)       # exit (bottom) end
    ramp_z0: float = info(0.105)      # top-surface height at the mouth end
    ramp_z1: float = info(0.004)      # top-surface height at the exit end (4 mm ground step)
    ramp_t: float = info(0.03)        # ramp floor thickness (no tunneling at drop-in impacts)
    stop_h: float = info(0.10)        # mouth backstop height (blocks backward bounce-outs)
    pen_x0: float = info(0.70)        # pen interior near edge — 1.02 m from the arm base:
    pen_x1: float = info(0.92)        # the whole pen is beyond a Franka's ~0.85 m reach
    pen_half_w: float = info(0.08)    # pen interior half-width
    pen_wall_h: float = info(0.10)
    gate_x: float = info(0.60)        # transit-latch plane on the lower ramp (also beyond
    # reach from the recorded base pose: 0.92 m from base — the cube must SLIDE across it)
    mat_t: float = info(0.004)        # floor-mat slab, sunk flush (top +0.2 mm)
    mat_size: tuple = info((0.20, 0.14))
    green: tuple = info((0.10, 0.65, 0.20))
    gray: tuple = info((0.45, 0.45, 0.47))
    contact_offset: float = info(0.002)  # small: default ~2 cm offsets would eat the 25 mm
    # cube-to-wall clearance inside the channel.

    # Derived (filled in __post_init__).
    ramp_pitch: float = field(default=None, init=False)   # ramp tilt about +y (rad, +x down)
    ramp_len: float = field(default=None, init=False)     # slope length of the ramp floor
    rest_z: float = field(default=None, init=False)       # cube center height at rest

    def __post_init__(self) -> None:
        drop = self.ramp_z0 - self.ramp_z1
        run = self.ramp_x1 - self.ramp_x0
        self.ramp_pitch = math.atan2(drop, run)
        self.ramp_len = math.hypot(drop, run)
        self.rest_z = self.cube_size / 2


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("chute_dispatch")
class ChuteDispatchScene(BaseScene):
    cfg: ChuteDispatchSceneCfg

    def __init__(self, cfg: ChuteDispatchSceneCfg | None = None) -> None:
        super().__init__(cfg or ChuteDispatchSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, per lane: pitched ramp floor + 2 pitched side walls + mouth
        backstop + walled pen (far / left / right / 2 near-wall flanks) + floor mat, and
        the cube. Only the mats and the cube are re-placed by reset()."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        pitch = c.ramp_pitch
        sp, cp = math.sin(pitch), math.cos(pitch)
        # quaternion for rotation about +y by +pitch (tilts local +x downward)
        rot = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)

        slick = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.chute_mu, dynamic_friction=c.chute_mu, restitution=0.0,
            friction_combine_mode="min", restitution_combine_mode="min")
        grippy = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.floor_mu, dynamic_friction=c.floor_mu, restitution=0.0)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        def box(name: str, size, pos, color, mat, rot4=None) -> RigidObjectCfg:
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=tuple(size), rigid_props=kin, collision_props=coll,
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(pos), rot=tuple(rot4) if rot4 is not None else (1, 0, 0, 0)),
            )

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(physics_material=grippy),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }

        # top-surface midpoint of the ramp floor
        mx = (c.ramp_x0 + c.ramp_x1) / 2
        mz = (c.ramp_z0 + c.ramp_z1) / 2
        chan_outer = c.chan_w + 2 * c.wall_t
        chute_col = (0.75, 0.55, 0.20)
        wall_col = (0.30, 0.30, 0.34)

        for tag, side in (("l", 1.0), ("r", -1.0)):
            y0 = side * c.lane_dy
            # ramp floor: center = surface midpoint - normal * t/2
            out[f"ramp_{tag}"] = box(
                f"Ramp_{tag.upper()}",
                (c.ramp_len, chan_outer, c.ramp_t),
                (mx - sp * c.ramp_t / 2, y0, mz - cp * c.ramp_t / 2),
                chute_col, slick, rot)
            # channel side walls: center = surface midpoint + normal * h/2, offset in y
            for wtag, wside in (("a", 1.0), ("b", -1.0)):
                out[f"chanwall_{tag}{wtag}"] = box(
                    f"Chanwall_{tag.upper()}{wtag.upper()}",
                    (c.ramp_len, c.wall_t, c.chan_wall_h),
                    (mx + sp * c.chan_wall_h / 2,
                     y0 + wside * (c.chan_w / 2 + c.wall_t / 2),
                     mz + cp * c.chan_wall_h / 2),
                    wall_col, slick, rot)
            # mouth backstop: upright, just up-slope of the top end (kinematic pairs
            # don't interact, so the slight overlap with the ramp floor is harmless)
            out[f"stop_{tag}"] = box(
                f"Stop_{tag.upper()}",
                (c.wall_t, chan_outer, c.stop_h),
                (c.ramp_x0 - c.wall_t / 2 + 0.002, y0, c.ramp_z0 - 0.01 + c.stop_h / 2),
                wall_col, slick)
            # pen: far wall + 2 side walls + 2 near-wall flanks around the chute inlet
            pen_cx = (c.pen_x0 + c.pen_x1) / 2
            out[f"pen_{tag}_far"] = box(
                f"Pen_{tag.upper()}_far",
                (c.wall_t, 2 * c.pen_half_w + 2 * c.wall_t, c.pen_wall_h),
                (c.pen_x1 + c.wall_t / 2, y0, c.pen_wall_h / 2),
                wall_col, grippy)
            for wtag, wside in (("a", 1.0), ("b", -1.0)):
                out[f"pen_{tag}_side{wtag}"] = box(
                    f"Pen_{tag.upper()}_side{wtag.upper()}",
                    (c.pen_x1 - c.pen_x0 + 2 * c.wall_t, c.wall_t, c.pen_wall_h),
                    (pen_cx, y0 + wside * (c.pen_half_w + c.wall_t / 2), c.pen_wall_h / 2),
                    wall_col, grippy)
                flank_w = c.pen_half_w - chan_outer / 2
                out[f"pen_{tag}_near{wtag}"] = box(
                    f"Pen_{tag.upper()}_near{wtag.upper()}",
                    (c.wall_t, flank_w, c.pen_wall_h),
                    (c.pen_x0 - c.wall_t / 2,
                     y0 + wside * (chan_outer / 2 + flank_w / 2), c.pen_wall_h / 2),
                    wall_col, grippy)

        # floor mats (green target / gray decoy) — thin kinematic slabs sunk flush into
        # the pen floors, swapped between pens by reset() (pose write only).
        pen_cx = (c.pen_x0 + c.pen_x1) / 2
        for name, color, side in (("mat_green", c.green, 1.0), ("mat_gray", c.gray, -1.0)):
            out[name] = box(
                name.capitalize(), (c.mat_size[0], c.mat_size[1], c.mat_t),
                (pen_cx, side * c.lane_dy, 0.0002 - c.mat_t / 2), color, grippy)

        out["cube"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(c.cube_size, c.cube_size, c.cube_size),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.0, angular_damping=0.05),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                collision_props=coll, physics_material=grippy,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, c.rest_z + 0.002)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode buffers the rubric depends on."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.cube: RigidObject = env.iscene["cube"]
        self.mats = {"green": env.iscene["mat_green"], "gray": env.iscene["mat_gray"]}
        self.env_origins = env.iscene.env_origins
        # target_side[e]: +1 = left lane (y > 0) is the green target, -1 = right lane.
        self.target_side = torch.ones(n, device=dev)
        self.lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        # transit[e, k]: chute k (0 = left/+y, 1 = right/-y) was physically transited.
        self.transit = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self._prev = torch.zeros(n, 3, device=dev)

    def _cube_local(self) -> torch.Tensor:
        """Cube center position in env-local coords, (N, 3)."""
        return self.cube.data.root_pos_w - self.env_origins

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the green-target side (mats swap by pose write) and the
        cube spawn pose; clear the latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        side = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        self.target_side[env_ids] = side
        pen_cx = (c.pen_x0 + c.pen_x1) / 2
        for name, s in (("green", side), ("gray", -side)):
            st = torch.zeros(m, 7, device=dev)
            st[:, 0] = pen_cx
            st[:, 1] = s * c.lane_dy
            st[:, 2] = 0.0002 - c.mat_t / 2
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.mats[name].write_root_pose_to_sim(st, env_ids)

        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.spawn_x_range[0] + (c.spawn_x_range[1] - c.spawn_x_range[0]) * torch.rand(m, device=dev)
        st[:, 1] = c.spawn_y_range[0] + (c.spawn_y_range[1] - c.spawn_y_range[0]) * torch.rand(m, device=dev)
        st[:, 2] = c.rest_z + 0.002
        half = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.spawn_yaw_deg) / 2
        st[:, 3] = torch.cos(half)
        st[:, 6] = torch.sin(half)
        st[:, 0:3] += origin
        self.cube.write_root_state_to_sim(st, env_ids)

        self.lifted[env_ids] = False
        self.transit[env_ids] = False
        self._prev[env_ids] = st[:, 0:3] - origin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch detection, every physics step. Lift: cube center rises past `lift_z`
        with a small per-step rise (a teleport jumps too far and does not latch).
        Transit: the cube crosses the gate plane on lane k moving down-lane between two
        CONSECUTIVE steps (small dx), inside that lane's channel cross-section."""
        c = self.cfg
        p = self._cube_local()
        dz = p[:, 2] - self._prev[:, 2]
        # RISING steps only (dz > 0): a teleport jumps too far in one step, and a body
        # FALLING past lift_z (e.g. dropped there kinematically) only has dz <= 0.
        self.lifted |= (p[:, 2] > c.lift_z) & (dz > 0.0005) & (dz < c.lift_dz_max)

        dx = p[:, 0] - self._prev[:, 0]
        base = ((self._prev[:, 0] < c.gate_x) & (p[:, 0] >= c.gate_x)
                & (dx > 0) & (dx < c.gate_dx_max) & (p[:, 2] < 0.14))
        for k, side in ((0, 1.0), (1, -1.0)):
            in_chan = (p[:, 1] - side * c.lane_dy).abs() < c.chan_w / 2
            self.transit[:, k] |= base & in_chan
        self._prev = p.clone()

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "mat_green": self.mats["green"].data.root_state_w[env_ids].clone(),
            "mat_gray": self.mats["gray"].data.root_state_w[env_ids].clone(),
            "target_side": self.target_side[env_ids].clone(),
            "lifted": self.lifted[env_ids].clone(),
            "transit": self.transit[env_ids].clone(),
            "prev": self._prev[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.mats["green"].write_root_pose_to_sim(state["mat_green"][:, 0:7], env_ids)
        self.mats["gray"].write_root_pose_to_sim(state["mat_gray"][:, 0:7], env_ids)
        self.target_side[env_ids] = state["target_side"]
        self.lifted[env_ids] = state["lifted"]
        self.transit[env_ids] = state["transit"]
        self._prev[env_ids] = state["prev"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A red cube ({c.cube_size * 1000:.0f} mm) rests on the ground within easy "
            f"reach. Two identical slick delivery chutes descend away from you, side by "
            f"side — one on your left, one on your right. Each chute is an open-top "
            f"walled channel ({c.chan_w * 1000:.0f} mm wide) starting at a raised mouth "
            f"(floor {c.ramp_z0 * 1000:.0f} mm high, near you) and sloping down into its "
            f"own walled catch pen far beyond your reach. One pen's floor carries a GREEN "
            f"mat — that pen is the delivery target; the other pen's mat is GRAY and that "
            f"pen is a dead-end decoy. Which side is green changes every episode: look at "
            f"the mats.\n"
            f"Goal: the red cube at rest on the floor INSIDE the green-matted pen. The "
            f"pens and the lower chutes are out of reach and fully walled — the only way "
            f"in is down a chute: drop the cube between the side walls near the mouth of "
            f"the chute on the green pen's side and let gravity deliver it. A cube sent "
            f"down the wrong chute ends in the decoy pen and cannot be recovered; a cube "
            f"left anywhere outside the green pen — however close — does not count."
        )

    # ----- progress / rubric ------------------------------------------------------------------
    def _target_idx(self) -> torch.Tensor:
        """(N,) long: 0 if the left lane (+y) is the target, 1 if the right lane."""
        return (self.target_side < 0).long()

    def settled(self) -> torch.Tensor:
        c = self.cfg
        lin = self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        ang = self.cube.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
        return lin & ang

    def in_pen(self, side: torch.Tensor | float) -> torch.Tensor:
        """(N,) bool: cube fully inside the pen on `side` (+1 left / -1 right), resting
        on its floor (full-containment margin = cube half-diagonal)."""
        c = self.cfg
        p = self._cube_local()
        if not torch.is_tensor(side):
            side = torch.full((p.shape[0],), float(side), device=p.device)
        x_ok = (p[:, 0] > c.pen_x0 + c.pen_margin) & (p[:, 0] < c.pen_x1 - c.pen_margin)
        y_ok = (p[:, 1] - side * c.lane_dy).abs() < c.pen_half_w - c.pen_margin
        z_ok = (p[:, 2] - c.rest_z).abs() < c.rest_z_tol
        return x_ok & y_ok & z_ok

    def in_target_pen(self) -> torch.Tensor:
        return self.in_pen(self.target_side)

    def transit_correct(self) -> torch.Tensor:
        """(N,) bool: the target-side chute was physically transited this episode."""
        return self.transit.gather(1, self._target_idx().unsqueeze(1)).squeeze(1)

    def success(self) -> torch.Tensor:
        """(N,) bool: cube settled on the floor fully inside the green-matted pen, having
        physically ridden that pen's chute (transit latch) this episode."""
        return self.in_target_pen() & self.settled() & self.transit_correct()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 for the latched lift + 0.35 for the latched
        correct-chute transit; exactly 1.0 iff success. Null policy scores 0; a wrong-
        chute delivery keeps only the lift credit (<= 0.15); latched credit never
        evaporates along a correct trajectory."""
        part = 0.15 * self.lifted.float() + 0.35 * self.transit_correct().float()
        return torch.where(self.success(), torch.ones_like(part), part)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="chute_dispatch", robot="null", env_spacing=4.0))
