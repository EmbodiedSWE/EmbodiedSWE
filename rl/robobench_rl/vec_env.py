"""RoboBenchEnv — the base RL env over a robobench BaseEnv (an rsl_rl VecEnv), with task hooks.

Isaac Lab's DirectRLEnv pattern: a task is a subclass overriding a fixed set of hooks —
    defaults()           config merged UNDER the yaml (horizon, action scaling, reward weights, PPO tweaks)
    _setup()             one-off writes after the env exists (default: `robot.gripper_stiffness`)
    _process_actions(a)  policy [-1,1]^A -> controller action (default: ActionMap from `action.*`)
    _get_extra_obs()     appended to the generic observation (default: none)
    _get_terminated()    early termination (default: none — episodes run to the time limit)
    _hover_target()      warm-start pose for the curriculum (default: the dense reward's, if any)
    _on_reset(ids)       per-env hook after the scene reset
    reward_cls           the dense-reward class (default: registry lookup by scene)
The base behaviour is the uniform baseline: config-driven action scaling, the observation rule in obs.py,
progress or dense per-step reward, fixed-length episodes, no termination.

Two modes: `Env(cfg)` builds the Isaac env (training); `Env(cfg, env=existing)` ATTACHES to a graded env
(the exported solve.py) — no grader, no reward — and exposes compute_obs() / process_actions(). Only the
torch-only parts are needed for that, so rsl_rl/tensordict imports are optional here."""
from __future__ import annotations

import copy
import math
import time

import torch

try:  # training only; the grading container has neither
    from rsl_rl.env import VecEnv
except ImportError:  # pragma: no cover
    class VecEnv:  # type: ignore[no-redef]
        pass

from .obs import StateObs


def _merge_under(cfg: dict, defaults: dict) -> dict:
    """cfg wins over defaults, recursively."""
    out = copy.deepcopy(defaults)
    for k, v in cfg.items():
        out[k] = _merge_under(v, out[k]) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def set_gripper_stiffness(env, kp: float) -> None:
    """Write `kp` as the PD stiffness of the robot's declared GRIPPER_JOINTS (no-op without a gripper)."""
    art = getattr(env.robot, "articulation", None)
    pats = getattr(env.robot, "GRIPPER_JOINTS", None)
    if art is None or not pats:
        return
    art.write_joint_stiffness_to_sim(kp, joint_ids=art.find_joints(list(pats))[0])


class ActionMap:
    """Policy [-1,1]^A -> env action: mirror (one scalar -> several dims), affine [lo, hi] per dim, joint_delta
    dims (target = clamp(q + a*scale, limits)). Built from `action.*` config; serialisable to env.json."""

    def __init__(self, env_action_dim: int, lo, hi, mirror_dims=(), jd_dims=(), jd_joint_ids=(), jd_scale=0.0,
                 jd_lim=None, device="cpu") -> None:
        self.env_action_dim = int(env_action_dim)
        self.lo = torch.as_tensor(lo, dtype=torch.float32, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float32, device=device)
        self.mirror_dims = list(mirror_dims)
        self.keep_dims = [d for d in range(self.env_action_dim) if d not in self.mirror_dims]
        self.num_actions = self.env_action_dim - max(0, len(self.mirror_dims) - 1)
        self.jd_dims, self.jd_joint_ids, self.jd_scale = list(jd_dims), list(jd_joint_ids), float(jd_scale)
        self.jd_lim = None if jd_lim is None else torch.as_tensor(jd_lim, dtype=torch.float32, device=device)

    @classmethod
    def from_cfg(cls, env, cfg: dict) -> "ActionMap":
        E, dev = env.robot.action_dim, env.device
        lo, hi = torch.full((E,), -1.0, device=dev), torch.full((E,), 1.0, device=dev)
        act = cfg.get("action", {}) or {}
        jd_dims, jd_ids, jd_scale, jd_lim = [], [], 0.0, None
        for spec in act.get("affine", []) or []:
            dims = list(spec["dims"])
            if spec.get("from") in ("joint_delta", "joint_limits"):
                ctrl = env.robot.controller
                arm = ctrl.controllers[0] if hasattr(ctrl, "controllers") else ctrl
                lim = env.robot.articulation.data.soft_joint_pos_limits[0, arm.joint_ids]
                if len(dims) != lim.shape[0]:
                    raise ValueError(f"action.affine {spec['from']}: {len(dims)} dims vs {lim.shape[0]} arm joints")
                if spec["from"] == "joint_delta":
                    jd_dims, jd_ids, jd_scale, jd_lim = dims, list(arm.joint_ids), float(spec["scale"]), lim.clone()
                else:
                    for d, (a, b) in zip(dims, lim.tolist()):
                        lo[d], hi[d] = a, b
            else:
                for d in dims:
                    lo[d], hi[d] = float(spec["lo"]), float(spec["hi"])
        return cls(E, lo, hi, act.get("gripper_mirror") or [], jd_dims, jd_ids, jd_scale, jd_lim, device=dev)

    def __call__(self, env, a: torch.Tensor) -> torch.Tensor:
        a = a.clamp(-1.0, 1.0)
        if self.mirror_dims:
            full = torch.zeros(a.shape[0], self.env_action_dim, device=a.device, dtype=a.dtype)
            full[:, self.keep_dims] = a[:, : len(self.keep_dims)]
            full[:, self.mirror_dims] = a[:, -1:].expand(-1, len(self.mirror_dims))
            a = full
        out = self.lo + (a + 1.0) * 0.5 * (self.hi - self.lo)
        if self.jd_dims:
            q = env.robot.articulation.data.joint_pos[:, self.jd_joint_ids]
            out = out.clone()
            out[:, self.jd_dims] = (q + a[:, self.jd_dims] * self.jd_scale).clamp(self.jd_lim[:, 0], self.jd_lim[:, 1])
        return out

    def describe(self) -> str:
        s = (f"action dim {self.num_actions} (env {self.env_action_dim}, mirror {self.mirror_dims or 'off'}) "
             f"lo {[round(x, 3) for x in self.lo.tolist()]} hi {[round(x, 3) for x in self.hi.tolist()]}")
        return s + (f"  joint_delta dims {self.jd_dims} scale {self.jd_scale} rad/step" if self.jd_dims else "")


class RoboBenchEnv(VecEnv):
    name = "generic"
    reward_cls = None  # type: ignore[assignment]

    @classmethod
    def defaults(cls) -> dict:
        return {}

    def __init__(self, cfg: dict, env=None, device: str | None = None) -> None:
        self.cfg = cfg = _merge_under(cfg, self.defaults())
        t = cfg["task"]
        self.attached = env is not None
        if env is None:
            env = self._build(cfg, device)
        self.env = env
        self.device = torch.device(env.device)
        self.num_envs = env.num_envs
        self._setup()
        self.action_map = ActionMap.from_cfg(env, cfg)
        self.num_actions = self.action_map.num_actions
        self.step_dt = env.dt * env.robot.control_period
        self.max_episode_length = int(math.ceil(float(t["episode_seconds"]) / self.step_dt))
        self.last_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        o = cfg.get("obs", {})
        self.obs_fn = StateObs(env, drop_keys=tuple(o.get("drop_keys", ())), env_local=o.get("env_local", True),
                               ee_pose=o.get("ee_pose", True), last_action=o.get("last_action", True), action_dim=self.num_actions)
        extra = self._get_extra_obs()
        self.num_extra_obs = 0 if extra is None else int(extra.shape[1])
        self.num_obs = self.obs_fn.dim + self.num_extra_obs
        if self.attached:
            return  # grading: no grader, no reward, no episode bookkeeping
        from .reward import GraderReward, load_grader_cls

        r = cfg.get("reward", {})
        self.reward_fn = GraderReward(env, load_grader_cls(t["preset"], self._scene_name), self._scene_name,
                                      mode=r.get("mode", "progress"), success_bonus=float(r.get("success_bonus", 1.0)),
                                      weights=r.get("weights"), task_cls=self.reward_cls)
        cur = cfg.get("curriculum", {}) or {}
        self.hover_frac = float(cur.get("hover_start_frac", 0.0))
        self.early_termination = bool((cfg.get("task") or {}).get("early_termination", True))  # task env's _get_terminated on/off
        self.hover_steps = int(cur.get("hover_steps", 90))
        self.hover_jitter = float(cur.get("hover_jitter", 0.03))
        self._in_preroll = False
        self.preroll_warm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ep_ret = torch.zeros(self.num_envs, device=self.device)
        self._ep_peak = torch.zeros(self.num_envs, device=self.device)
        self.total_episodes, self.total_successes = 0, 0

    # ----- hooks (override in a task env) --------------------------------------------------------
    def _setup(self) -> None:
        kp = (self.cfg.get("robot") or {}).get("gripper_stiffness")
        if kp:
            set_gripper_stiffness(self.env, float(kp))

    def _process_actions(self, a: torch.Tensor) -> torch.Tensor:
        return self.action_map(self.env, a)

    def _get_extra_obs(self) -> torch.Tensor | None:
        return None

    def _get_terminated(self) -> torch.Tensor | None:
        return None

    def _hover_target(self) -> torch.Tensor | None:
        task = getattr(getattr(self, "reward_fn", None), "task", None)
        return task.hover_target() if task is not None else None

    def _on_reset(self, ids: torch.Tensor) -> None:
        pass

    def _hover_action(self, target: torch.Tensor, warm: torch.Tensor) -> torch.Tensor:
        """One pre-roll action in policy space: P-servo the hand to `target` for the `warm` envs, zero for
        the rest, fingers open. Default assumes pose-delta arm actions (OSC/IK presets); joint-mode envs
        override with a Jacobian step."""
        art = self.env.robot.articulation
        ee = list(art.data.body_names).index(self.env.robot.EE_BODY)
        ctrl = self.env.robot.controller
        arm = ctrl.controllers[0] if hasattr(ctrl, "controllers") else ctrl
        pos_scale = float(getattr(getattr(arm, "cfg", None), "pos_scale", 0.02))
        a = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        a[:, 0:3] = ((target - art.data.body_pos_w[:, ee]) / pos_scale).clamp(-1.0, 1.0) * warm[:, None].float()
        a[:, 6:] = 1.0  # fingers open (arm dims first, gripper last — every preset here)
        return a

    # ----- build ---------------------------------------------------------------------------------
    def _build(self, cfg: dict, device: str | None):
        import robobench
        from robobench.core import ENVS, SCENES

        robobench.discover()
        t = cfg["task"]
        env_cfg = ENVS.get(t["preset"])()
        self._scene_name = env_cfg.scene
        n = int(t["num_envs"])
        # PhysX GPU buffers are sized by the scene for a few envs; scale them with the batch (1024 bulb
        # envs overflowed the scene's collision stack). Explicit `sim.physx` config entries win.
        try:
            scene_cls = SCENES.get(env_cfg.scene)
            base = (scene_cls(env_cfg.scene_cfg) if env_cfg.scene_cfg is not None else scene_cls()).sim_cfg().physx
        except Exception:  # noqa: BLE001 — sizing is best-effort
            base = {}
        per_env = {"gpu_collision_stack_size": 2**21, "gpu_max_rigid_contact_count": 2**15, "gpu_max_rigid_patch_count": 2**15}
        physx = {k: max(int(base.get(k, 0)), n * v) for k, v in per_env.items()}
        physx.update(cfg.get("sim", {}).get("physx", {}) or {})
        sim_overrides = {**env_cfg.sim_overrides, "physx": {**env_cfg.sim_overrides.get("physx", {}), **physx}}
        t0 = time.time()
        env = env_cfg.build(num_envs=n, device=device or cfg.get("device", "cuda:0"), seed=cfg.get("seed", 0),
                            sim_overrides=sim_overrides)
        print(f"[rl] env build ({n} envs) {time.time() - t0:.1f} s", flush=True)
        return env

    # ----- observation / action (shared by training and the exported solve) -----------------------
    def compute_obs(self) -> torch.Tensor:
        obs = self.obs_fn.compute(self.last_action)
        extra = self._get_extra_obs()
        return obs if extra is None else torch.cat([obs, extra.to(obs.device).float()], dim=1)

    def process_actions(self, a: torch.Tensor) -> torch.Tensor:
        a = a.to(self.device).clamp(-1.0, 1.0)
        self.last_action = a
        return self._process_actions(a)

    # ----- rsl_rl VecEnv API (training) ------------------------------------------------------------
    def get_observations(self):
        from tensordict import TensorDict

        obs = self.compute_obs()
        return TensorDict({"policy": obs, "critic": obs}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        self.env.step(self.process_actions(actions))
        self.episode_length_buf += 1
        reward, succ, stages = self.reward_fn.compute()
        time_out = self.episode_length_buf >= self.max_episode_length
        term = self._get_terminated() if self.early_termination else None
        done = time_out if term is None else (time_out | term.to(self.device))
        self._ep_ret += reward
        self._ep_peak = torch.maximum(self._ep_peak, stages["progress"])
        log = {f"/stage/{k}": v.mean() for k, v in stages.items()}
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.total_episodes += int(ids.numel())
            self.total_successes += int(succ[ids].sum())
            log["/episode/return"] = self._ep_ret[ids].mean()
            log["/episode/peak_progress"] = self._ep_peak[ids].mean()
            log["/episode/success"] = succ[ids].float().mean()
            if term is not None:
                log["/episode/terminated"] = term[ids].float().mean()
            self._reset_idx(ids)
        return self.get_observations(), reward, done, {"time_outs": time_out, "log": log, "success": succ}

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def _reset_idx(self, ids: torch.Tensor) -> None:
        # With the warm-start curriculum every env must reset together (the pre-roll steps physics for ALL envs).
        # An early-terminated env therefore restarts from home but keeps the SHARED episode clock, so the batch
        # stays in lockstep and the next full reset still gets its pre-roll. (Measured on slice tuned2: one knocked-off
        # knife desynced one env, every later reset was partial, and the curriculum silently ran for one episode only.)
        lockstep_partial = self.hover_frac > 0 and ids.numel() != self.num_envs
        self.env.reset(ids)
        self._on_reset(ids)
        if self.hover_frac > 0 and not self._in_preroll and not lockstep_partial:
            self._hover_preroll(ids)
        self.reward_fn.reset(ids)
        if not lockstep_partial:
            self.episode_length_buf[ids] = 0
        self.last_action[ids] = 0.0
        self._ep_ret[ids] = 0.0
        self._ep_peak[ids] = 0.0

    @torch.no_grad()
    def _hover_preroll(self, ids: torch.Tensor) -> None:
        """Servo a random `hover_frac` of the envs to the hover target with P-control on the controller's
        position-delta actions, fingers open. Physics advances for ALL envs, so every env must be resetting
        together (fixed-length episodes, init_at_random_ep_len off)."""
        target = self._hover_target()
        if target is None:
            return
        if ids.numel() != self.num_envs:  # cannot happen via _reset_idx any more; keep as a guard
            return
        self._in_preroll = True
        n = self.num_envs
        warm = torch.rand(n, device=self.device) < self.hover_frac
        target = target + (torch.rand(n, 3, device=self.device) * 2 - 1) * self.hover_jitter * torch.tensor([1.0, 1.0, 0.0], device=self.device)
        for _ in range(self.hover_steps):
            self.env.step(self._process_actions(self._hover_action(target, warm)))
        self.preroll_warm = warm
        self._in_preroll = False

    def close(self) -> None:
        self.env.close()

    def describe(self) -> str:
        t = self.cfg["task"]
        return "\n".join([
            f"preset {t['preset']}  task_env {self.name}  num_envs {self.num_envs}  device {self.device}",
            f"control dt {self.step_dt:.4f} s ({1 / self.step_dt:.1f} Hz), physics dt {self.env.dt:.5f}",
            f"episode {t['episode_seconds']} s = {self.max_episode_length} steps"
            + (f"; hover curriculum {self.hover_frac:.0%} x {self.hover_steps} steps" if getattr(self, "hover_frac", 0) > 0 else ""),
            self.action_map.describe(),
            f"reward {self.reward_fn.describe()}",
            self.obs_fn.describe() + (f"\n  extra_obs                        {self.num_extra_obs}" if self.num_extra_obs else ""),
        ])


def make_vec_env(cfg: dict, device: str | None = None) -> RoboBenchEnv:
    """Instantiate the task env named by `task.task_env` (generic RoboBenchEnv when absent)."""
    from .tasks import load_task_env_cls

    return load_task_env_cls(cfg["task"].get("task_env"))(cfg, device=device)
