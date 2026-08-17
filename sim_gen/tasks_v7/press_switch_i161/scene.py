"""DialSetpointScene — set BOTH console dials to their sampled target flags.

Derived from the RLBench `press_switch` seed but STRATEGICALLY DIFFERENT (see
TASK.md): the seed's plan is ONE straight poke on a binary wall lever — a single
contact, a single direction, no magnitude, no stopping problem. Here nothing is
binary and nothing is a press: a charcoal console carries two identical
free-spinning dials, each with a red pointer and — standing on that dial's own
rim — an orange target flag whose angle is SAMPLED per episode (up to +-85 deg).
Each knob also STARTS at a sampled angle at least 40 deg away from its flag. The
task is continuous setpoint regulation, twice over: read each flag's bearing,
rotate that knob the right DIRECTION and the right AMOUNT (both sampled, so no
memorized motion works), and STOP inside a +-6 deg band — the dial has no detent
at the flag, so overshoot is a real failure mode, and sweeping through the target
at speed earns nothing (progress latches only through a slow-gate counter).
Success needs BOTH pointers resting on their flags simultaneously, settled. No
execution order is required.

Mechanics (plain rigid bodies + authored USD joints — the proven D6 pattern):
each knob is ONE dynamic compound body (cylinder + grip blade + pointer visual)
hung on a D6 joint (console -> knob) that frees exactly rotZ within +-92 deg
(physical end stops); translation is locked. The spin axis is vertical, so the
plant is gravity-neutral: angular damping parks the knob wherever it is released
— whatever angle you leave is the angle that gets judged. `post_step` owns both
knobs' wrench slots: it applies the `drive_t` / `drive_f` buffers (solve.py's
stand-in for the gripper's wrist roll on the grip blade) and latches rubric
progress through the slow-gate counter.

Rubric (graded 0..1, anchored in the demonstrated solve.py trajectory):
  - 0.30 * mean(align_latch): per-dial best QUASI-STATIC alignment progress,
    (err0 - err)/(err0 - tol) clamped to [0,1], latched only while the knob has
    been slower than `slow_gate` for `latch_steps` consecutive substeps — a fast
    sweep through the target latches nothing (smoke proves it);
  - 0.30 * mean(aligned_now): per-dial CURRENT |err| < tol and slow;
  - 0.40 * success(): both dials aligned_now and everything settled.
  score == 1.0 iff success() has held for a beat (the latches mature); ~0 for the
  null policy; latched credit never evaporates under correct behavior.

Per-episode randomization (readback-verified in smoke): each dial's target flag
angle (sign AND magnitude, flag physically re-posed on the rim) and each knob's
start angle (min 40 deg separation from its target, rejection-sampled with a
deterministic fallback).

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable
from robobench.core.registries import ENVS

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DialSetpointSceneCfg(BaseCfg):
    """Config for `DialSetpointScene`. Geometry is derived once in `__post_init__` so the
    scene, the smoke AND the solver read the same numbers."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tol_deg: float = tunable(6.0)  # |pointer - flag| within this = on target
    slow_gate: float = tunable(0.30)  # rad/s: knob counts as slow below this
    latch_steps: int = tunable(24)  # consecutive slow substeps before progress latches (0.2 s)
    settle_lin: float = tunable(0.05)  # max knob |lin vel| (m/s) when judging
    settle_ang: float = tunable(0.30)  # max knob |ang vel| (rad/s) when judging

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    target_max_deg: float = tunable(85.0)  # flag angle sampled in +-this
    start_max_deg: float = tunable(88.0)  # knob start angle sampled in +-this
    min_sep_deg: float = tunable(40.0)  # |start - target| at least this (>> tol)

    # --- tunable: plant ----------------------------------------------------------------------
    knob_mass: float = tunable(0.10)
    knob_lin_damp: float = tunable(2.0)
    knob_ang_damp: float = tunable(6.0)  # parks the knob where released (coast ~w/damp rad)

    # --- info: structure (env-local coordinates; dial axes are vertical) ---------------------
    table_center: tuple = info((0.05, 0.0, 0.36))
    table_size: tuple = info((0.90, 1.00, 0.08))  # top at z = 0.40
    console_center: tuple = info((0.0, 0.0, 0.43))
    console_size: tuple = info((0.30, 0.62, 0.06))  # top at z = 0.46
    dial_y: tuple = info((-0.155, 0.155))  # dial 0 / dial 1 axes (x = 0)
    bezel_r: float = info(0.088)  # visual rim disc on the console top
    bezel_h: float = info(0.006)
    knob_r: float = info(0.035)  # knob cylinder (collides)
    knob_h: float = info(0.030)
    knob_gap: float = info(0.002)  # clearance above the bezel top (no rubbing contact)
    blade_size: tuple = info((0.064, 0.014, 0.026))  # grip blade across the knob top (collides)
    pointer_size: tuple = info((0.046, 0.010, 0.008))  # red pointer, visual only
    pointer_cx: float = info(0.056)  # pointer centre offset along knob-local +x (tip at 79 mm)
    flag_r: float = info(0.100)  # flag stands on the rim at this radius (10 mm off the tip)
    flag_size: tuple = info((0.022, 0.016, 0.024))
    rot_lim_deg: float = info(92.0)  # D6 rotZ limit — the physical end stops
    tick_r: float = info(0.076)  # decor dots on the bezel every 45 deg
    contact_offset: float = info(0.002)

    # Derived (filled in __post_init__).
    console_top_z: float = field(default=None, init=False)
    knob_z: float = field(default=None, init=False)  # knob body centre height
    flag_z: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.console_top_z = self.console_center[2] + self.console_size[2] / 2
        self.knob_z = self.console_top_z + self.bezel_h + self.knob_gap + self.knob_h / 2
        self.flag_z = self.console_top_z + self.flag_size[2] / 2


def _quat_z(rad: torch.Tensor) -> torch.Tensor:
    """(N,) angle about +z -> (N, 4) wxyz."""
    half = rad / 2
    q = torch.zeros(rad.shape[0], 4, device=rad.device)
    q[:, 0] = torch.cos(half)
    q[:, 3] = torch.sin(half)
    return q


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


# ----- scene -----------------------------------------------------------------------------------
class DialSetpointScene(BaseScene):
    cfg: DialSetpointSceneCfg

    def __init__(self, cfg: DialSetpointSceneCfg | None = None) -> None:
        super().__init__(cfg or DialSetpointSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, kinematic table + console, two dynamic knobs (compound children
        authored in bind), two kinematic target flags (re-posed at reset to the episode's
        sampled angles)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.35, 0.20))
        charcoal = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.13, 0.13, 0.15))
        knob_gray = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58))
        orange = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.05))
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset, rest_offset=0.0)
        kin = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        # post_step drives the knobs with external wrenches that do NOT wake a sleeping
        # body — sleep_threshold=0 keeps the plant live.
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
            "table": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=c.table_size, rigid_props=kin, collision_props=coll,
                    visual_material=wood),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.table_center),
            ),
            "console": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Console",
                spawn=sim_utils.CuboidCfg(
                    size=c.console_size, rigid_props=kin, collision_props=coll,
                    visual_material=charcoal),
                init_state=RigidObjectCfg.InitialStateCfg(pos=c.console_center),
            ),
        }
        # --- the two knobs: root prim = the knob cylinder (children authored in bind) ---
        for d in (0, 1):
            out[f"knob{d}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Knob{d}",
                spawn=sim_utils.CylinderCfg(
                    radius=c.knob_r, height=c.knob_h, axis="Z",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        linear_damping=c.knob_lin_damp,
                        angular_damping=c.knob_ang_damp,
                        **live),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.knob_mass),
                    collision_props=coll,
                    visual_material=knob_gray,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, c.dial_y[d], c.knob_z)),
            )
            # --- the target flag: kinematic post on this dial's rim (re-posed at reset) ---
            out[f"flag{d}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"Flag{d}",
                spawn=sim_utils.CuboidCfg(
                    size=c.flag_size, rigid_props=kin, collision_props=coll,
                    visual_material=orange),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.flag_r, c.dial_y[d], c.flag_z)),
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
        n = env.num_envs
        dev = env.device
        self.knobs: list[RigidObject] = [env.iscene[f"knob{d}"] for d in (0, 1)]
        self.flags: list[RigidObject] = [env.iscene[f"flag{d}"] for d in (0, 1)]
        self.env_origins = env.iscene.env_origins
        self._author_dressing()
        self._author_d6()
        # Episode state.
        self.target = torch.zeros(n, 2, device=dev)  # flag angle per dial (rad)
        self.start = torch.zeros(n, 2, device=dev)  # knob start angle per dial (rad)
        self.err0 = torch.zeros(n, 2, device=dev)  # |start - target| at reset (rad)
        self.align_latch = torch.zeros(n, 2, device=dev)
        self.slow_ctr = torch.zeros(n, 2, dtype=torch.long, device=dev)
        # External drive input (solve.py and smoke probes write; post_step consumes + owns
        # both knobs' wrench slots — never call set_external_force_and_torque directly).
        self.drive_t = torch.zeros(n, 2, device=dev)  # torque about the dial axis (N*m)
        self.drive_f = torch.zeros(n, 2, 3, device=dev)  # force on the knob (N) — probes

    def _author_dressing(self) -> None:
        """Visual/compound children (env_0 only; authored idempotently): per dial a bezel
        disc + tick dots on the console, and per knob the GRIP BLADE (collides — the jaw
        target) + the red pointer (visual only)."""
        import omni.usd
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath("/World/envs/env_0/Knob0/blade").IsValid():
            return

        def box(root: str, name: str, center: tuple, size: tuple, color: tuple,
                collide: bool = False) -> None:
            cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(*center))
            xf.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            if collide:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                px.CreateContactOffsetAttr(c.contact_offset)
                px.CreateRestOffsetAttr(0.0)

        def disc(root: str, name: str, center: tuple, radius: float, height: float,
                 color: tuple) -> None:
            cy = UsdGeom.Cylinder.Define(stage, f"{root}/{name}")
            cy.CreateAxisAttr("Z")
            cy.CreateRadiusAttr(radius)
            cy.CreateHeightAttr(height)
            UsdGeom.Xformable(cy.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*center))
            cy.CreateDisplayColorAttr([Gf.Vec3f(*color)])

        console_root = "/World/envs/env_0/Console"
        cz = c.console_center[2]
        for d in (0, 1):
            # bezel disc on the console top (visual)
            disc(console_root, f"bezel{d}",
                 (0.0, c.dial_y[d], c.console_top_z - cz + c.bezel_h / 2),
                 c.bezel_r, c.bezel_h, (0.72, 0.72, 0.74))
            # tick dots every 45 deg (visual angle cues for a solver)
            for k, ang in enumerate((-90.0, -45.0, 0.0, 45.0, 90.0)):
                a = math.radians(ang)
                box(console_root, f"tick{d}_{k}",
                    (c.tick_r * math.cos(a), c.dial_y[d] + c.tick_r * math.sin(a),
                     c.console_top_z - cz + c.bezel_h + 0.002),
                    (0.006, 0.006, 0.004), (0.95, 0.95, 0.95))
        for d in (0, 1):
            root = f"/World/envs/env_0/Knob{d}"
            # grip blade across the knob top — THE jaw target (collides)
            box(root, "blade", (0.0, 0.0, c.knob_h / 2 + c.blade_size[2] / 2),
                c.blade_size, (0.25, 0.25, 0.28), collide=True)
            # red pointer toward the rim (visual only; tip 10 mm short of the flag ring)
            box(root, "pointer",
                (c.pointer_cx, 0.0, c.knob_h / 2 - c.pointer_size[2] / 2 + 0.004),
                c.pointer_size, (0.78, 0.05, 0.05))

    def _author_d6(self) -> None:
        """Per env and per dial: a D6 joint console -> knob freeing exactly rotZ within
        +-rot_lim_deg (physical end stops); all translation locked. The joint pair never
        collides; the flag is a SEPARATE kinematic body outside the pointer's sweep."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        cc = c.console_center
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            for d in (0, 1):
                j = UsdPhysics.Joint.Define(stage, f"{base}/dial_joint{d}")
                j.CreateBody0Rel().SetTargets([f"{base}/Console"])
                j.CreateBody1Rel().SetTargets([f"{base}/Knob{d}"])
                j.CreateCollisionEnabledAttr(False)
                j.CreateLocalPos0Attr(Gf.Vec3f(0.0, c.dial_y[d] - cc[1], c.knob_z - cc[2]))
                j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                for axis in ("transX", "transY", "transZ", "rotX", "rotY"):
                    lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), axis)
                    lim.CreateLowAttr(1.0)  # low > high = locked
                    lim.CreateHighAttr(-1.0)
                lim = UsdPhysics.LimitAPI.Apply(j.GetPrim(), "rotZ")
                lim.CreateLowAttr(-c.rot_lim_deg)
                lim.CreateHighAttr(c.rot_lim_deg)

    # ----- reset --------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample each dial's target flag angle and knob start angle
        (min separation enforced; deterministic fallback after rejection sampling), pose
        the knobs at their starts and the flags at their targets, clear latches and
        drives. Uses torch.rand throughout (the first randint after manual_seed is
        degenerate on this stack)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        t = (torch.rand(m, 2, device=dev) * 2 - 1) * math.radians(c.target_max_deg)
        s = (torch.rand(m, 2, device=dev) * 2 - 1) * math.radians(c.start_max_deg)
        min_sep = math.radians(c.min_sep_deg)
        for _ in range(12):
            bad = (s - t).abs() < min_sep
            if not bad.any():
                break
            rs = (torch.rand(m, 2, device=dev) * 2 - 1) * math.radians(c.start_max_deg)
            s = torch.where(bad, rs, s)
        # Deterministic fallback: 64 deg toward the roomier side (always in range, > min_sep).
        bad = (s - t).abs() < min_sep
        fb = torch.where(t <= 0, t + math.radians(64.0), t - math.radians(64.0))
        s = torch.where(bad, fb, s)

        self.target[env_ids] = t
        self.start[env_ids] = s
        self.err0[env_ids] = (s - t).abs()
        self.align_latch[env_ids] = 0.0
        self.slow_ctr[env_ids] = 0
        self.drive_t[env_ids] = 0.0
        self.drive_f[env_ids] = 0.0

        for d in (0, 1):
            # knob at its start angle
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = 0.0
            st[:, 1] = c.dial_y[d]
            st[:, 2] = c.knob_z
            st[:, 3:7] = _quat_z(s[:, d])
            st[:, 0:3] += origin
            self.knobs[d].write_root_state_to_sim(st, env_ids)
            # flag on this dial's rim at the target angle
            a = t[:, d]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = c.flag_r * torch.cos(a)
            st[:, 1] = c.dial_y[d] + c.flag_r * torch.sin(a)
            st[:, 2] = c.flag_z
            st[:, 3:7] = _quat_z(a)
            st[:, 0:3] += origin
            self.flags[d].write_root_state_to_sim(st, env_ids)

    # ----- readings -----------------------------------------------------------------------------
    def knob_yaw(self) -> torch.Tensor:
        """(N, 2) knob rotation about +z in rad (the D6 frees only rotZ)."""
        y = []
        for k in self.knobs:
            q = k.data.root_quat_w
            y.append(_wrap(2.0 * torch.atan2(q[:, 3], q[:, 0])))
        return torch.stack(y, dim=1)

    def knob_speed(self) -> torch.Tensor:
        """(N, 2) |angular velocity| about z (rad/s)."""
        return torch.stack([k.data.root_ang_vel_w[:, 2].abs() for k in self.knobs], dim=1)

    def align_err(self) -> torch.Tensor:
        """(N, 2) |pointer angle - flag angle| per dial (rad)."""
        return _wrap(self.knob_yaw() - self.target).abs()

    def aligned_now(self) -> torch.Tensor:
        """(N, 2) bool: CURRENT |err| < tol and the knob is slow."""
        return (self.align_err() < math.radians(self.cfg.tol_deg)) \
            & (self.knob_speed() < self.cfg.slow_gate)

    def settled(self) -> torch.Tensor:
        """(N,) bool: both knobs at rest (lin + ang)."""
        c = self.cfg
        ok = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for k in self.knobs:
            ok &= (k.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
                & (k.data.root_ang_vel_w[:, 2].abs() < c.settle_ang)
        return ok

    def success(self) -> torch.Tensor:
        """(N,) bool: BOTH pointers resting within tol of their own flags, settled
        (current, physical state — judged on the settled knob orientations)."""
        return self.aligned_now().all(dim=1) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 * mean(align_latch) + 0.30 * mean(aligned_now)
        + 0.40 * success. Latches mature only through the slow-gate counter, so a fast
        sweep through the target earns nothing; ~0 for the null policy; exactly 1.0
        once success holds for a beat (the latches saturate at the target)."""
        return 0.30 * self.align_latch.mean(dim=1) \
            + 0.30 * self.aligned_now().float().mean(dim=1) \
            + 0.40 * self.success().float()

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self) -> None:
        """Knob plant: consume the `drive_t` / `drive_f` buffers (owns both knobs' wrench
        slots; the drive torque is about the spin axis, which z-rotation never skews),
        then latch quasi-static alignment progress through the slow-gate counter."""
        n = self.env.num_envs
        dev = self.env.device
        c = self.cfg
        for d in (0, 1):
            f = torch.zeros(n, 1, 3, device=dev)
            t = torch.zeros(n, 1, 3, device=dev)
            f[:, 0, :] = self.drive_f[:, d, :]
            t[:, 0, 2] = self.drive_t[:, d]
            self.knobs[d].set_external_force_and_torque(f, t)

        err = self.align_err()
        slow = self.knob_speed() < c.slow_gate
        self.slow_ctr = torch.where(slow, self.slow_ctr + 1, torch.zeros_like(self.slow_ctr))
        mature = (self.slow_ctr >= c.latch_steps).float()
        tol = math.radians(c.tol_deg)
        prog = ((self.err0 - err) / (self.err0 - tol).clamp(min=1e-6)).clamp(0.0, 1.0)
        # A diverged substep must not latch: torch.maximum propagates NaN. Garbage earns 0.
        prog = torch.nan_to_num(prog, nan=0.0, posinf=0.0, neginf=0.0)
        self.align_latch = torch.maximum(self.align_latch, prog * mature)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = self._bodies()
        return {
            "bodies": {nm: b.data.root_state_w[env_ids].clone() for nm, b in bodies.items()},
            "task": {k: getattr(self, k)[env_ids].clone()
                     for k in ("target", "start", "err0", "align_latch", "slow_ctr",
                               "drive_t", "drive_f")},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = self._bodies()
        for nm, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][nm], env_ids)
        for k, v in state["task"].items():
            getattr(self, k)[env_ids] = v

    def _bodies(self) -> dict[str, Any]:
        return {"knob0": self.knobs[0], "knob1": self.knobs[1],
                "flag0": self.flags[0], "flag1": self.flags[1]}

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A charcoal control console sits on a table, its face pointing UP. It carries "
            f"two identical round dials side by side, {abs(c.dial_y[0]) * 200:.0f} cm apart: "
            f"each is a gray knob ({c.knob_r * 200:.0f} cm across) with a dark GRIP BLADE "
            f"standing across its top and a RED POINTER reaching toward a pale bezel ring "
            f"with white tick dots. On each dial's own rim stands a single small ORANGE "
            f"FLAG; its bearing is sampled fresh every episode (anywhere up to 85 degrees "
            f"either side of the dial's zero tick), and each knob also STARTS at a random "
            f"angle at least 40 degrees away from its flag — so the direction AND amount "
            f"to turn differ every episode and per dial.\n"
            f"Goal: rotate EACH knob about its vertical axis (grip the blade and turn; "
            f"either dial first, either direction) until its red pointer points at that "
            f"dial's orange flag, within about {c.tol_deg:.0f} degrees, then let go. The "
            f"dials spin freely between end stops at +-{c.rot_lim_deg:.0f} degrees and "
            f"stay wherever they are released — there is NO detent at the flag, so "
            f"overshooting past it and stopping there fails just like stopping short. "
            f"Sweeping the pointer through the flag without stopping counts for nothing: "
            f"only where the pointer RESTS is judged. The task is complete when BOTH "
            f"pointers rest on their own flags at the same time, hands off. Pressing or "
            f"pushing a knob does nothing — the reading only changes by rotation."
        )

    def instruction(self) -> str:
        return (
            "Rotate each of the two console dials until its red pointer rests on the "
            "orange flag on that dial's rim, within about 6 degrees. The dials stay "
            "where released; leave both on their flags at the same time."
        )


# Guarded registration: the forge may import this module under two names.
if "dial_setpoints" not in SCENES.list():
    SCENES.register("dial_setpoints", DialSetpointScene)
if "simgen.dial_setpoints" not in ENVS.list():
    register_env("simgen", lambda: EnvCfg(scene="dial_setpoints", robot="null"))
