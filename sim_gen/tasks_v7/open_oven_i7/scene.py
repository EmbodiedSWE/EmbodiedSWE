"""OvenDialsScene — set the oven's two control dials to their indicated settings.

Derived from the RLBench `open_oven` seed but STRATEGICALLY DIFFERENT (see TASK.md): the
seed's plan is "grasp the door handle, pull the hinged door open through a large arc" — a
binary, fixed-goal, gross-motion pull on the oven's one moving panel. Here NOTHING opens:
the oven's control deck (a fixed, kinematic slab — the top console of a range) carries two
spring-detented control KNOBS on vertical revolute spindles. Every episode samples, per
knob, a random START setting and a random TARGET setting (>= 2 detents apart); the target
is shown physically by an amber lamp sitting at that setting's tick mark. **Goal (carried
here, no task layer): rotate each knob until its white-capped pointer rests on its glowing
amber mark.** Goal-conditioned precision rotary positioning with a tolerance stop — a
different plan skeleton, not different numbers.

Mechanics (the proven combination_safe / oven_dials-v1 pattern — plain rigid bodies +
authored USD revolute joints; knob "feel" and the detent springs are external torques
applied in `post_step`, always overwritten each substep): each knob is a cylinder on a
Z-axis spindle rising from the deck, with a symmetric grip bar fixed-jointed across its
top face (the pointer; symmetric so the centre of mass stays ON the spindle, with a
visual-only white cap marking the pointing end — the balance_scale needle pattern).
`post_step` applies viscous friction plus a spring toward the NEAREST setting (a real
detent dial: every rest position is a setting; capture basin = half the 40 deg spacing),
and consumes external drive/force buffers (`knob_drive` / `knob_force` /
`knob_torque_ext`). Writers of those buffers: solve.py's fingertip-scale spindle drive
(the v4 teleport-contract stand-in for the arm's pinch-turn — clamped to the measured
fingertip authority, so every turn still goes through the live detent dynamics) and
smoke's rubric probes. Nothing else touches the knobs' wrench slot.

Rubric (graded 0..1, latching transient achievement — anchored in the demonstrated
solve.py trajectory, which passes through per-knob approach/turn/settle stages):
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
    """Config for `OvenDialsScene`. Geometry is derived once in `__post_init__` so the scene,
    the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    stop_tol_deg: float = tunable(8.0)  # pointer within this of the target = on the mark
    settle_omega: float = tunable(0.15)  # max |spindle rate| (rad/s) when judging at_target

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    start_jitter_deg: float = tunable(4.0)  # +- jitter about the start setting at reset
    min_sep_idx: int = tunable(2)  # min |start_idx - target_idx| -> err0 >= 2*spacing = 80 deg

    # --- tunable: knob plant (difficulty dials) ----------------------------------------------
    # Detent spring sized against BOTH sides of the task: peak resist k*(spacing/2) =
    # 0.087 N*m (~2.6 N at the grip bar) sits well under a fingertip push's ~3-5 N OSC
    # authority (solve run 2 measured a 0.14 N*m detent stalling the push), while the
    # capture snap stays crisp (probe: 15 deg offsets snap in, 25 deg fall out).
    detent_k: float = tunable(0.25)  # detent spring (N*m/rad toward the nearest setting)
    # Discrete-stability audit (explicit external torques at 120 Hz): spindle inertia
    # I ~ 4.4e-4 (knob 0.5*m*r^2 + bar m*L^2/12), so friction*dt/I ~ 0.95 < 1 (monotone
    # decay) and omega_n*dt = sqrt(k/I)/120 ~ 0.20 << 2; zeta ~ 2.4 -> overdamped snap.
    knob_friction: float = tunable(0.05)  # viscous spindle friction (N*m*s/rad)
    knob_mass: float = tunable(0.40)  # heavy, well-damped knob: finger wrenches -> slow slews

    # --- info: structure ----------------------------------------------------------------------
    n_settings: int = info(7)  # detents per knob
    spacing_deg: float = info(40.0)  # detent spacing; settings span -120..+120 deg
    joint_limit_deg: float = info(128.0)  # revolute stops just past the extreme settings
    deck_pos: tuple = info((0.46, 0.0, 0.08))  # kinematic control-deck slab centre
    deck_size: tuple = info((0.60, 0.44, 0.16))  # top face = the work surface (z = 0.16)
    knob_dy: float = info(0.14)  # knob spindles at (deck_x, -dy) and (deck_x, +dy)
    knob_r: float = info(0.045)
    knob_th: float = info(0.036)
    knob_gap: float = info(0.005)  # knob bottom face to deck top (joint pair non-colliding)
    bar_len: float = info(0.085)  # symmetric grip bar (CoM on the spindle)
    bar_w: float = info(0.016)
    bar_h: float = info(0.032)  # proud enough for a fingertip to push its side face
    bar_mass: float = info(0.05)
    tick_r_off: float = info(0.014)  # tick ring radius = knob_r + this
    lamp_size: float = info(0.020)
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    settings_deg: tuple = field(default=None, init=False)  # detent angles, index order
    deck_top_z: float = field(default=None, init=False)
    knob_z: float = field(default=None, init=False)  # knob centre z
    bar_z: float = field(default=None, init=False)  # grip-bar centre z
    tick_r: float = field(default=None, init=False)
    lamp_r: float = field(default=None, init=False)  # lamp ring radius (outside the ticks)

    def __post_init__(self) -> None:
        half_span = (self.n_settings - 1) / 2 * self.spacing_deg
        self.settings_deg = tuple(-half_span + self.spacing_deg * i for i in range(self.n_settings))
        self.deck_top_z = self.deck_pos[2] + self.deck_size[2] / 2
        self.knob_z = self.deck_top_z + self.knob_gap + self.knob_th / 2
        self.bar_z = self.knob_z + self.knob_th / 2 + self.bar_h / 2 + 0.002
        self.tick_r = self.knob_r + self.tick_r_off
        self.lamp_r = self.tick_r + 0.022

    # -- shared geometry helpers (scene + smoke + solver read the same numbers) ----------------
    def knob_center(self, k: int) -> tuple:
        return (self.deck_pos[0], -self.knob_dy if k == 0 else self.knob_dy, self.knob_z)

    def bar_center(self, k: int) -> tuple:
        x, y, _ = self.knob_center(k)
        return (x, y, self.bar_z)

    def lamp_pos(self, k: int, setting_idx: int) -> tuple:
        """Lamp centre for knob k pointing at setting `setting_idx` (angle 0 = +x, + about
        world +z, i.e. counter-clockwise seen from above)."""
        a = math.radians(self.settings_deg[setting_idx])
        x, y, _ = self.knob_center(k)
        return (x + self.lamp_r * math.cos(a),
                y + self.lamp_r * math.sin(a),
                self.deck_top_z + self.lamp_size / 2 + 0.001)


def _quat_z(deg: float) -> tuple:
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("oven_dials")
class OvenDialsScene(BaseScene):
    cfg: OvenDialsSceneCfg

    def __init__(self, cfg: OvenDialsSceneCfg | None = None) -> None:
        super().__init__(cfg or OvenDialsSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic control deck, two knobs + grip bars, two target
        lamps (kinematic — a brushed lamp must stay a landmark, not become a puck)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cream = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.80, 0.74))
        dark = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.16, 0.18))
        brass = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.60, 0.28))
        amber = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.62, 0.05))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        # post_step drives the knobs with (deprecated) external wrenches that do NOT wake a
        # sleeping body — sleep_threshold=0 keeps the plant live (drawer_stash lesson).
        live = dict(sleep_threshold=0.0, stabilization_threshold=0.0)

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
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=sim_utils.CuboidCfg(
                    size=c.deck_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=cream,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.deck_pos),
            ),
        }
        for k in range(2):
            out[f"knob_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Knob_" + str(k),
                spawn=sim_utils.CylinderCfg(
                    radius=c.knob_r, height=c.knob_th, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.knob_mass),
                    collision_props=coll,
                    visual_material=dark,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.knob_center(k)),
            )
            out[f"bar_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bar_" + str(k),
                spawn=sim_utils.CuboidCfg(
                    size=(c.bar_len, c.bar_w, c.bar_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(**live),
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
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
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
        self.deck: RigidObject = env.iscene["deck"]
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
        # External drive inputs (solve.py's fingertip-scale drive and smoke probes write;
        # post_step consumes + owns the knobs' force buffer — never call
        # set_external_force_and_torque on the knobs directly).
        self.knob_drive = torch.zeros(n, 2, device=dev)  # torque about the spindle (N*m)
        self.knob_force = torch.zeros(n, 2, 3, device=dev)  # world force at the knob CoM
        self.knob_torque_ext = torch.zeros(n, 2, 3, device=dev)  # world torque (off-axis probes)

    def _author_joints(self) -> None:
        """Per env: a Z-axis revolute spindle deck->knob (limits just past the extreme
        settings) and a fixed joint knob->grip bar. Joint pairs never collide."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        dx, dy, dz = c.deck_pos
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for k in range(2):
                kx, ky, kz = c.knob_center(k)
                j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/knob_spindle_{k}")
                j.CreateBody0Rel().SetTargets([f"{base}/Deck"])
                j.CreateBody1Rel().SetTargets([f"{base}/Knob_{k}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateAxisAttr("Z")
                j.CreateLocalPos0Attr(Gf.Vec3f(kx - dx, ky - dy, kz - dz))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLowerLimitAttr(-c.joint_limit_deg)
                j.CreateUpperLimitAttr(c.joint_limit_deg)

                j = UsdPhysics.FixedJoint.Define(stage, f"{base}/bar_fix_{k}")
                j.CreateBody0Rel().SetTargets([f"{base}/Knob_{k}"])
                j.CreateBody1Rel().SetTargets([f"{base}/Bar_{k}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.bar_z - c.knob_z))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def _author_decorations(self) -> None:
        """Visual-only prims (displayColor, NO CollisionAPI), env_0 only — with num_envs>1
        isaaclab composes env_1.. from env_0 by reference, so authoring env_0 once is exactly
        right (the combination_safe idempotency precedent): tick marks on the deck at every
        setting angle around each knob, and a white cap on the pointing (+x-at-zero) end of
        each grip bar."""
        import omni.usd
        from pxr import Gf, UsdGeom

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Deck/tick_0_0").IsValid():
            return
        dx, dy, dz = c.deck_pos
        dark = Gf.Vec3f(0.10, 0.10, 0.12)
        for k in range(2):
            kx, ky, _kz = c.knob_center(k)
            for i, ang_deg in enumerate(c.settings_deg):
                a = math.radians(ang_deg)
                tick = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Deck/tick_{k}_{i}")
                tick.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(tick.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(
                    kx - dx + c.tick_r * math.cos(a),
                    ky - dy + c.tick_r * math.sin(a),
                    c.deck_top_z - dz + 0.002))
                xf.AddRotateZOp().Set(ang_deg)
                xf.AddScaleOp().Set(Gf.Vec3f(0.018, 0.006, 0.004))
                tick.CreateDisplayColorAttr([dark])
            cap = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Bar_{k}/cap")
            cap.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cap.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(c.bar_len / 2 - 0.010, 0.0, c.bar_h / 2 - 0.002))
            xf.AddScaleOp().Set(Gf.Vec3f(0.020, 0.017, 0.012))
            cap.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.98)])

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: per knob, sample start + target settings (>= min_sep_idx apart),
        place knob + bar at the jittered start angle (a pure joint-coordinate re-pose of the
        followers about the unchanged spindle — the fridge_clearway-proven safe teleport),
        park the amber lamp at the target tick."""
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
                st[:, 6] = torch.sin(half[:, k])
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
        cannot alias). Yaw of the knob root quat; the deck never moves and the knob only
        ever rotates about +z."""
        out = []
        for k in range(2):
            q = self.knobs[k].data.root_quat_w
            ang = torch.rad2deg(2.0 * torch.atan2(q[:, 3], q[:, 0]))
            out.append((ang + 180.0) % 360.0 - 180.0)
        return torch.stack(out, dim=1)

    def target_deg(self) -> torch.Tensor:
        return self._settings[self.target_idx]

    def errors_deg(self) -> torch.Tensor:
        """(N, 2) |pointer - target| in deg (unwrapped: physical along the limited spindle)."""
        return (self.readings_deg() - self.target_deg()).abs()

    def spindle_rate(self) -> torch.Tensor:
        """(N, 2) signed spindle rate (rad/s) — the world-z angular velocity."""
        return torch.stack([k.data.root_ang_vel_w[:, 2] for k in self.knobs], dim=1)

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
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev)

        read = self.readings_deg()  # (N, 2)
        rate = self.spindle_rate()  # (N, 2) rad/s
        half_span = (c.n_settings - 1) / 2 * c.spacing_deg
        near = ((read + half_span) / c.spacing_deg).round().clamp(0, c.n_settings - 1)
        err_near = torch.deg2rad(read - (near * c.spacing_deg - half_span))
        tq = self.knob_drive - c.knob_friction * rate - c.detent_k * err_near
        for k in range(2):
            # reshape, not view: column slices of (N, 2[, 3]) buffers are non-contiguous
            torque = (tq[:, k].reshape(n, 1, 1) * ez.view(1, 1, 3)
                      + self.knob_torque_ext[:, k].reshape(n, 1, 3))
            self.knobs[k].set_external_force_and_torque(
                self.knob_force[:, k].reshape(n, 1, 3), torque)

        err = (read - self.target_deg()).abs()
        prog = (1.0 - err / self.err0).clamp(0.0, 1.0)
        prog = torch.where(err <= c.stop_tol_deg, torch.ones_like(prog), prog)
        # A diverged substep must not latch: torch.maximum PROPAGATES NaN and best_prog is
        # checkpointed via get_state/set_state, so one bad frame would poison the episode
        # (observed in the v1 lineage). A garbage frame earns NO progress.
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
        x, _, _ = c.deck_pos
        return (
            f"An oven's control deck (a fixed {c.deck_size[0]:.2f} x {c.deck_size[1]:.2f} m "
            f"console slab, top face at height {c.deck_top_z:.2f} m) carries two round "
            f"control knobs ({2 * c.knob_r * 100:.0f} cm across, one left, one right), each "
            f"turning about a vertical spindle and carrying a brass grip bar across its top; "
            f"the END of the bar with the WHITE cap is the pointer. Around each knob, "
            f"{c.n_settings} dark tick marks on the deck ({c.spacing_deg:.0f} deg apart) are "
            f"the detent settings; the knobs click from setting to setting and rest only on "
            f"settings. Next to ONE tick of each knob sits a glowing amber lamp: that is "
            f"this episode's target setting for that knob (it is sampled fresh every "
            f"episode — read it from the scene; the two knobs' targets are independent).\n"
            f"Goal: rotate each knob about its spindle until its white pointer cap rests on "
            f"the tick marked by that knob's amber lamp (within {c.stop_tol_deg:.0f} deg, at "
            f"rest). Either knob may be set first — order is free. Nothing on this oven "
            f"opens or slides: the deck is fixed, the lamps are fixed, and pulling, prying "
            f"or pressing a knob does not turn it — only torque about the vertical spindle "
            f"does. Turning past the target loses nothing: back up and stop on the mark."
        )

    def instruction(self) -> str:
        return (
            "Turn each of the two control knobs about its vertical spindle until the "
            "white-capped end of its grip bar rests on the tick mark beside that knob's "
            "glowing amber lamp. Set both knobs; either order works."
        )


# ----- runnable env: scene physics only (NullRobot smoke) -> "simgen.oven_dials" ---------------
register_env("simgen", lambda: EnvCfg(scene="oven_dials", robot="null"))
