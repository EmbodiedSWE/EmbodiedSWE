"""OvenDialsScene — set the oven's two control dials to their indicated settings.

Derived from the RLBench `open_oven` seed but STRATEGICALLY DIFFERENT (see TASK.md): the
seed's plan is "grasp the door handle, pull the hinged door open through a large arc" — a
binary, fixed-goal, gross-motion pull on the oven's one moving panel. Here the oven front
is a FIXED fascia (kinematic — there is nothing to open; pulling on it is a measured
negative control) carrying two spring-detented control KNOBS on revolute spindles. Every
episode samples, per knob, a random START setting and a random TARGET setting (>= 2 detents
apart); the target is shown physically by an amber lamp sitting at that setting's tick
mark. **Goal (carried here, no task layer): rotate each knob until its white-tipped
pointer rests on its glowing amber mark.** Goal-conditioned precision rotary positioning
with a tolerance stop — a different plan skeleton, not different numbers.

Mechanics (the proven combination_safe pattern — plain rigid bodies + authored USD
revolute joints; knob "friction" and the detent springs are external torques applied in
`post_step`, always overwritten each substep): each knob is a cylinder on a Y-axis
spindle through the fascia, with a symmetric grip bar fixed-jointed to its face (the
pointer; symmetric so the centre of mass stays ON the spindle — no gravity pendulum —
with a visual-only bright cap marking the pointing end, the balance_scale needle
pattern). `post_step` applies viscous friction plus a spring toward the NEAREST setting
(a real detent dial: every rest position is a setting; capture basin = half the 40 deg
spacing), and consumes external drive/force buffers (`knob_drive` / `knob_force` /
`knob_torque_ext`) that smokes or robot layers may write.

Rubric (graded 0..1, latching transient achievement):
  - per knob: err = |pointer - target| (deg, unwrapped — the 256 deg joint range cannot
    wrap), err0 = |start setting - target setting| (authored, >= 80 deg);
  - `best_prog[k]` latches max(1 - err/err0), forced to 1.0 the moment the pointer ever
    enters the +-`stop_tol_deg` target zone;
  - `at_target[k]` (current, physical): err <= stop_tol AND spindle rate settled;
  - score = 0.35*(best_prog0 + best_prog1) + 0.15*(at0 + at1) -> exactly 1.0 iff both
    dials currently rest on target (= success()); ~0 when nothing acts (start jitter is
    +-4 deg against err0 >= 80 deg); 0.85 if a set dial is later knocked off (latched).

Per-episode randomization: start setting index, target setting index (both per knob,
independent, |start - target| >= `min_sep_idx`), start jitter about the spindle, and the
lamp pose follows the target — a memorized fixed rotation fails across episodes.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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
class OvenDialsSceneCfg(BaseCfg):
    """Config for `OvenDialsScene`. Geometry is derived once in `__post_init__` so the scene
    AND the smoke read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    stop_tol_deg: float = tunable(8.0)  # pointer within this of the target = on the mark
    settle_omega: float = tunable(0.15)  # max |spindle rate| (rad/s) when judging at_target

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    start_jitter_deg: float = tunable(4.0)  # +- jitter about the start setting at reset
    min_sep_idx: int = tunable(2)  # min |start_idx - target_idx| -> err0 >= 2*spacing = 80 deg

    # --- tunable: knob plant (difficulty dials) ----------------------------------------------
    detent_k: float = tunable(0.4)  # detent spring (N*m/rad toward the nearest setting)
    # Discrete-stability audit (explicit external torques at 120 Hz): spindle inertia
    # I ~ 4.4e-4 (knob 0.5*m*r^2 + bar), so friction*dt/I ~ 0.96 < 1 (monotone decay) and
    # omega_n*dt = sqrt(k/I)/120 ~ 0.25 << 2; zeta ~ 1.9 -> overdamped snap, no ringing.
    knob_friction: float = tunable(0.05)  # viscous spindle friction (N*m*s/rad)
    knob_mass: float = tunable(0.40)  # heavy, well-damped knob: finger torques -> finger speeds

    # --- info: structure ----------------------------------------------------------------------
    n_settings: int = info(7)  # detents per knob
    spacing_deg: float = info(40.0)  # detent spacing; settings span -120..+120 deg
    joint_limit_deg: float = info(128.0)  # revolute stops just past the extreme settings
    panel_pos: tuple = info((0.0, 0.30, 0.28))  # fascia centre; front face toward -y
    panel_size: tuple = info((0.60, 0.05, 0.56))
    knob_x: tuple = info((-0.14, 0.14))  # spindle x per knob (z shared)
    knob_z: float = info(0.30)
    knob_r: float = info(0.045)
    knob_th: float = info(0.035)
    knob_gap: float = info(0.006)  # knob back face to fascia front (joint pair non-colliding)
    bar_len: float = info(0.085)  # symmetric grip bar (CoM on the spindle -> no pendulum)
    bar_w: float = info(0.016)
    bar_th: float = info(0.014)
    bar_mass: float = info(0.05)
    tick_r_off: float = info(0.014)  # tick ring radius = knob_r + this
    lamp_size: float = info(0.020)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    settings_deg: tuple = field(default=None, init=False)  # detent angles, index order
    panel_front_y: float = field(default=None, init=False)
    knob_y: float = field(default=None, init=False)  # knob centre y
    bar_y: float = field(default=None, init=False)  # grip-bar centre y
    tick_r: float = field(default=None, init=False)
    lamp_r: float = field(default=None, init=False)  # lamp ring radius (outside the ticks)

    def __post_init__(self) -> None:
        half_span = (self.n_settings - 1) / 2 * self.spacing_deg
        self.settings_deg = tuple(-half_span + self.spacing_deg * i for i in range(self.n_settings))
        self.panel_front_y = self.panel_pos[1] - self.panel_size[1] / 2
        self.knob_y = self.panel_front_y - self.knob_th / 2 - self.knob_gap
        self.bar_y = self.knob_y - self.knob_th / 2 - self.bar_th / 2 - 0.002
        self.tick_r = self.knob_r + self.tick_r_off
        self.lamp_r = self.tick_r + 0.022

    # -- shared geometry helpers (scene + smoke read the same numbers) -------------------------
    def knob_center(self, k: int) -> tuple:
        return (self.knob_x[k], self.knob_y, self.knob_z)

    def bar_center(self, k: int) -> tuple:
        return (self.knob_x[k], self.bar_y, self.knob_z)

    def lamp_pos(self, k: int, setting_idx: int) -> tuple:
        """Lamp centre for knob k pointing at setting `setting_idx` (angle 0 = up, +about +y)."""
        a = math.radians(self.settings_deg[setting_idx])
        return (self.knob_x[k] + self.lamp_r * math.sin(a),
                self.panel_front_y - self.lamp_size / 2 - 0.012,
                self.knob_z + self.lamp_r * math.cos(a))


def _quat_y(deg: float) -> tuple:
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, math.sin(h), 0.0)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("oven_dials")
class OvenDialsScene(BaseScene):
    cfg: OvenDialsSceneCfg

    def __init__(self, cfg: OvenDialsSceneCfg | None = None) -> None:
        super().__init__(cfg or OvenDialsSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic fascia, two knobs + grip bars, two target lamps."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cream = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.80, 0.74))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.16, 0.18))
        brass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.60, 0.28))
        amber = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.62, 0.05))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)

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
            "panel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Panel",
                spawn=sim_utils.CuboidCfg(
                    size=c.panel_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=cream,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.panel_pos),
            ),
        }
        for k in range(2):
            out[f"knob_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Knob_" + str(k),
                spawn=sim_utils.CylinderCfg(
                    radius=c.knob_r, height=c.knob_th, axis="Y",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.knob_mass),
                    collision_props=coll,
                    visual_material=dark,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.knob_center(k)),
            )
            out[f"bar_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar_" + str(k),
                spawn=sim_utils.CuboidCfg(
                    size=(c.bar_w, c.bar_th, c.bar_len),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                    collision_props=coll,
                    visual_material=brass,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.bar_center(k)),
            )
            out[f"lamp_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lamp_" + str(k),
                spawn=sim_utils.CuboidCfg(
                    size=(c.lamp_size,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                    collision_props=coll,
                    visual_material=amber,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.lamp_pos(k, 0)),
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
        n = env.num_envs
        dev = env.device
        self.panel: RigidObject = env.iscene["panel"]
        self.knobs: list[RigidObject] = [env.iscene[f"knob_{k}"] for k in range(2)]
        self.bars: list[RigidObject] = [env.iscene[f"bar_{k}"] for k in range(2)]
        self.lamps: list[RigidObject] = [env.iscene[f"lamp_{k}"] for k in range(2)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        self._author_decorations()
        self._settings = torch.tensor(c.settings_deg, device=dev)
        # Episode state.
        self.start_idx = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.target_idx = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.err0 = torch.full((n, 2), 1.0, device=dev)  # authored start error (deg)
        self.best_prog = torch.zeros(n, 2, device=dev)  # latched progress in [0, 1]
        # External drive inputs (smokes / robot layers write; post_step consumes + owns the
        # force buffer — never call set_external_force_and_torque on the knobs directly).
        self.knob_drive = torch.zeros(n, 2, device=dev)  # torque about the spindle (N*m)
        self.knob_force = torch.zeros(n, 2, 3, device=dev)  # world force at the knob CoM
        self.knob_torque_ext = torch.zeros(n, 2, 3, device=dev)  # world torque (off-axis probes)

    def _author_joints(self) -> None:
        """Per env: a Y-axis revolute spindle fascia->knob (limits just past the extreme
        settings) and a fixed joint knob->grip bar. Joint pairs never collide."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        px, py, pz = c.panel_pos
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for k in range(2):
                kx, ky, kz = c.knob_center(k)
                j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/knob_spindle_{k}")
                j.CreateBody0Rel().SetTargets([f"{base}/Panel"])
                j.CreateBody1Rel().SetTargets([f"{base}/Knob_{k}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("Y")
                j.CreateLocalPos0Attr(Gf.Vec3f(kx - px, ky - py, kz - pz))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(-c.joint_limit_deg)
                j.CreateUpperLimitAttr(c.joint_limit_deg)

                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/bar_fix_{k}")
                j.CreateBody0Rel().SetTargets([f"{base}/Knob_{k}"])
                j.CreateBody1Rel().SetTargets([f"{base}/Bar_{k}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, c.bar_y - c.knob_y, 0.0))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def _author_decorations(self) -> None:
        """Visual-only prims (displayColor, NO CollisionAPI), env_0 only — with num_envs>1
        isaaclab composes env_1.. from env_0 by reference, so authoring env_0 once is exactly
        right (the combination_safe idempotency precedent): tick marks on the fascia at every
        setting angle around each knob, and a white cap on the pointing (+z-at-zero) end of
        each grip bar."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Panel/tick_0_0").IsValid():
            return
        px, py, pz = c.panel_pos
        dark = Gf.Vec3f(0.10, 0.10, 0.12)
        for k in range(2):
            kx, _ky, kz = c.knob_center(k)
            for i, ang_deg in enumerate(c.settings_deg):
                a = math.radians(ang_deg)
                tick = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Panel/tick_{k}_{i}")
                tick.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(tick.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    kx - px + c.tick_r * math.sin(a),
                    c.panel_front_y - py,
                    kz - pz + c.tick_r * math.cos(a)))
                xf.AddRotateYOp().Set(ang_deg)
                xf.AddScaleOp().Set(Gf.Vec3f(0.006, 0.004, 0.018))
                tick.CreateDisplayColorAttr([dark])
            cap = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Bar_{k}/cap")
            cap.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cap.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, c.bar_len / 2 - 0.008))
            xf.AddScaleOp().Set(Gf.Vec3f(0.020, 0.016, 0.016))
            cap.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.98)])

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: per knob, sample start + target settings (>= min_sep_idx apart),
        place knob + bar at the jittered start angle, park the amber lamp at the target tick."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        for row, _e in enumerate(env_ids.tolist()):
            for k in range(2):
                while True:
                    s = int(torch.randint(0, c.n_settings, (1,)))
                    t = int(torch.randint(0, c.n_settings, (1,)))
                    if abs(s - t) >= c.min_sep_idx:
                        break
                self.start_idx[env_ids[row], k] = s
                self.target_idx[env_ids[row], k] = t

        starts = self._settings[self.start_idx[env_ids]]  # (m, 2) deg
        targets = self._settings[self.target_idx[env_ids]]
        self.err0[env_ids] = (starts - targets).abs().clamp(min=1.0)
        self.best_prog[env_ids] = 0.0
        self.knob_drive[env_ids] = 0.0
        self.knob_force[env_ids] = 0.0
        self.knob_torque_ext[env_ids] = 0.0

        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.start_jitter_deg
        ang = starts + jit
        half = torch.deg2rad(ang) / 2
        for k in range(2):
            for body, center in ((self.knobs[k], c.knob_center(k)),
                                 (self.bars[k], c.bar_center(k))):
                st = torch.zeros(m, 13, device=dev)
                st[:, 0:3] = origin + torch.tensor(center, device=dev)
                st[:, 3] = torch.cos(half[:, k])
                st[:, 5] = torch.sin(half[:, k])
                body.write_root_state_to_sim(st, env_ids)
            st = torch.zeros(m, 13, device=dev)
            st[:, 3] = 1.0
            for row in range(m):
                pos = c.lamp_pos(k, int(self.target_idx[env_ids[row], k]))
                st[row, 0:3] = origin[row] + torch.tensor(pos, device=dev)
            self.lamps[k].write_root_state_to_sim(st, env_ids)

    # ----- readings / rubric --------------------------------------------------------------------
    def readings_deg(self) -> torch.Tensor:
        """(N, 2) pointer angle per knob (deg, wrapped [-180, 180) — the 256 deg joint range
        cannot alias). Twist about +y of the knob root quat; the fascia never moves."""
        out = []
        for k in range(2):
            q = self.knobs[k].data.root_quat_w
            ang = torch.rad2deg(2.0 * torch.atan2(q[:, 2], q[:, 0]))
            out.append((ang + 180.0) % 360.0 - 180.0)
        return torch.stack(out, dim=1)

    def target_deg(self) -> torch.Tensor:
        return self._settings[self.target_idx]

    def errors_deg(self) -> torch.Tensor:
        """(N, 2) |pointer - target| in deg (unwrapped: physical along the limited spindle)."""
        return (self.readings_deg() - self.target_deg()).abs()

    def spindle_rate(self) -> torch.Tensor:
        """(N, 2) signed spindle rate (rad/s) — the world-y angular velocity."""
        return torch.stack([k.data.root_ang_vel_w[:, 1] for k in self.knobs], dim=1)

    def at_target(self) -> torch.Tensor:
        """(N, 2) bool, physical + current: pointer within `stop_tol_deg` of ITS target and
        the spindle settled."""
        c = self.cfg
        return (self.errors_deg() <= c.stop_tol_deg) & (self.spindle_rate().abs() < c.settle_omega)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.35 per knob of latched progress (best_prog, forced to 1.0
        once the target zone was ever reached) + 0.15 per knob currently resting on target.
        Exactly 1.0 iff success(); ~0 for doing nothing; 0.85 if a set dial was knocked off."""
        at = self.at_target().float()
        return 0.35 * self.best_prog.sum(dim=1) + 0.15 * at.sum(dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: both pointers resting on their glowing marks (current, settled state)."""
        return self.at_target().all(dim=1)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Knob plant: viscous friction + detent spring toward the NEAREST setting + external
        drive/force buffers; then latch rubric progress. Owns the knobs' external-wrench slot."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        ey = torch.tensor([0.0, 1.0, 0.0], device=dev)

        read = self.readings_deg()  # (N, 2)
        rate = self.spindle_rate()  # (N, 2) rad/s
        half_span = (c.n_settings - 1) / 2 * c.spacing_deg
        near = ((read + half_span) / c.spacing_deg).round().clamp(0, c.n_settings - 1)
        err_near = torch.deg2rad(read - (near * c.spacing_deg - half_span))
        tq = self.knob_drive - c.knob_friction * rate - c.detent_k * err_near
        for k in range(2):
            # reshape, not view: column slices of (N, 2[, 3]) buffers are non-contiguous
            torque = (tq[:, k].reshape(n, 1, 1) * ey.view(1, 1, 3)
                      + self.knob_torque_ext[:, k].reshape(n, 1, 3))
            self.knobs[k].set_external_force_and_torque(
                self.knob_force[:, k].reshape(n, 1, 3), torque)

        err = (read - self.target_deg()).abs()
        prog = (1.0 - err / self.err0).clamp(0.0, 1.0)
        prog = torch.where(err <= c.stop_tol_deg, torch.ones_like(prog), prog)
        # A diverged substep must not latch. `read` comes from the knob root quat, so a
        # non-finite sim state (agent code has raw sim access and can write a NaN force
        # or velocity) makes `err` -> `prog` NaN; torch.maximum PROPAGATES NaN, so one
        # bad frame would pin best_prog at NaN for the rest of the episode -- and
        # best_prog is in get_state/set_state, so the NaN is checkpointed and restored
        # by goto, poisoning every node below it. Observed on job f1175e3855f6fc25:
        # simgen.oven_dials.franka returned final=nan, which then killed the run in
        # wandb.Histogram. A garbage frame earns NO progress; the latch keeps its last
        # good value. (torch.clamp does NOT filter NaN, which is why this is needed.)
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        self.best_prog = torch.maximum(self.best_prog, prog)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {f"knob_{k}": self.knobs[k] for k in range(2)}
        bodies.update({f"bar_{k}": self.bars[k] for k in range(2)})
        bodies.update({f"lamp_{k}": self.lamps[k] for k in range(2)})
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("start_idx", "target_idx", "err0", "best_prog",
                               "knob_drive", "knob_force", "knob_torque_ext")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {f"knob_{k}": self.knobs[k] for k in range(2)}
        bodies.update({f"bar_{k}": self.bars[k] for k in range(2)})
        bodies.update({f"lamp_{k}": self.lamps[k] for k in range(2)})
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"An oven control fascia (a fixed {c.panel_size[0]:.2f} x {c.panel_size[2]:.2f} m "
            f"panel, front toward -y) carries two round control knobs "
            f"({2 * c.knob_r * 100:.0f} cm across), each with a brass grip bar whose "
            f"white-capped end is the pointer. Around each knob, {c.n_settings} tick marks "
            f"({c.spacing_deg:.0f} deg apart) are the detent settings; the knobs click from "
            f"setting to setting. Beside ONE tick of each knob sits a glowing amber lamp: "
            f"that is this episode's target setting for that knob (it changes every episode "
            f"— read it from the scene).\n"
            f"Goal: rotate each knob until its white pointer tip rests on the tick with the "
            f"amber lamp (within {c.stop_tol_deg:.0f} deg, at rest). Either knob first — "
            f"order is free. Nothing on this oven opens: the fascia is fixed, and pulling or "
            f"prying on a knob does not turn it."
        )


# ----- runnable env: scene physics only (NullRobot smoke/oracle) -> "simgen.oven_dials" --------
register_env("simgen", lambda: EnvCfg(scene="oven_dials", robot="null"))
