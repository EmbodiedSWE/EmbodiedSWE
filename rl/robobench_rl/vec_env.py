"""rsl_rl VecEnv over a robobench BaseEnv.

Actions: the policy emits [-1, 1]^action_dim on the task's FROZEN controller preset — the same
`env.step(action)` a delivered solve(env) calls. Dims listed under `action.affine` are mapped to
[lo, hi] (e.g. the Franka finger targets in metres); all others pass through. Episode = the task's
horizon in seconds, always run to the limit (dense level reward; success is logged, not terminal);
the time-limit done is reported via extras["time_outs"] so PPO bootstraps it. Every env runs
the scene's nominal physics; initial-condition variety comes only from the scene's own reset."""
from __future__ import annotations

import math
import time

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .obs import StateObs
from .reward import GraderReward, load_grader_cls


class RoboBenchVecEnv(VecEnv):
    def __init__(self, cfg: dict, device: str | None = None) -> None:
        import robobench
        from robobench.core import ENVS

        if hasattr(robobench, "discover"):
            robobench.discover()
        self.cfg = cfg
        t = cfg["task"]
        self.device = torch.device(device or cfg.get("device", "cuda:0"))
        env_cfg = ENVS.get(t["preset"])()
        n = int(t["num_envs"])
        # PhysX GPU buffers are sized by the scene for a few envs; scale them with the batch. The
        # collision stack overflowed at ~0.8 MB/env on bulb (1024 envs) -> 2 MB/env, floor at the
        # scene's own value. Explicit `sim.physx` entries in the config win.
        try:  # the scene's own PhysX sizing is the floor
            from robobench.core import SCENES
            scene_cls = SCENES.get(env_cfg.scene)
            base = (scene_cls(env_cfg.scene_cfg) if env_cfg.scene_cfg is not None else scene_cls()).sim_cfg().physx
        except Exception:  # noqa: BLE001 — sizing is best-effort; the build itself reports real failures
            base = {}
        per_env = {"gpu_collision_stack_size": 2**21, "gpu_max_rigid_contact_count": 2**15,
                   "gpu_max_rigid_patch_count": 2**15}
        physx = {k: max(int(base.get(k, 0)), n * v) for k, v in per_env.items()}
        physx.update(cfg.get("sim", {}).get("physx", {}) or {})
        sim_overrides = {**env_cfg.sim_overrides, "physx": {**env_cfg.sim_overrides.get("physx", {}), **physx}}
        t0 = time.time()
        self.env = env_cfg.build(num_envs=n, device=str(self.device), seed=cfg.get("seed", 0),
                                 sim_overrides=sim_overrides)
        print(f"[rl] env build ({n} envs) {time.time() - t0:.1f} s  physx {physx}", flush=True)
        self.num_envs = self.env.num_envs
        self.num_actions = self.env.robot.action_dim
        self.step_dt = self.env.dt * self.env.robot.control_period
        self.max_episode_length = int(math.ceil(float(t["episode_seconds"]) / self.step_dt))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)

        # action mapping: affine dims -> [lo, hi], others pass through in [-1, 1]. A spec with
        # `from: joint_limits` takes lo/hi from the robot's arm-joint position limits (joint-mode
        # presets, where the controller writes raw joint targets).
        self._aff_lo = torch.full((self.num_actions,), -1.0, device=self.device)
        self._aff_hi = torch.full((self.num_actions,), 1.0, device=self.device)
        for spec in cfg.get("action", {}).get("affine", []) or []:
            dims = list(spec["dims"])
            if spec.get("from") == "joint_limits":
                ctrl = self.env.robot.controller
                arm = ctrl.controllers[0] if hasattr(ctrl, "controllers") else ctrl
                lim = self.env.robot.articulation.data.soft_joint_pos_limits[0, arm.joint_ids]  # (k, 2)
                if len(dims) != lim.shape[0]:
                    raise ValueError(f"action.affine joint_limits: {len(dims)} dims vs {lim.shape[0]} arm joints")
                for d, (lo, hi) in zip(dims, lim.tolist()):
                    self._aff_lo[d], self._aff_hi[d] = lo, hi
            else:
                for d in dims:
                    self._aff_lo[d], self._aff_hi[d] = float(spec["lo"]), float(spec["hi"])

        o = cfg.get("obs", {})
        self.obs_fn = StateObs(self.env, drop_keys=tuple(o.get("drop_keys", ())),
                               env_local=o.get("env_local", True), ee_pose=o.get("ee_pose", True),
                               last_action=o.get("last_action", True))
        self.num_obs = self.obs_fn.dim
        print(f"[rl] obs rule ready (dim {self.num_obs})", flush=True)
        r = cfg.get("reward", {})
        self.reward_fn = GraderReward(self.env, load_grader_cls(t["preset"], env_cfg.scene), env_cfg.scene,
                                      mode=r.get("mode", "progress"),
                                      progress_scale=float(r.get("progress_scale", 1.0)),
                                      success_bonus=float(r.get("success_bonus", 1.0)),
                                      action_penalty=float(r.get("action_penalty", 0.0)))
        print("[rl] grader reward ready", flush=True)
        # episode bookkeeping for logs
        self._ep_ret = torch.zeros(self.num_envs, device=self.device)
        self._ep_peak = torch.zeros(self.num_envs, device=self.device)
        self.total_episodes, self.total_successes = 0, 0

    # ----- VecEnv API --------------------------------------------------------------------------
    def get_observations(self) -> TensorDict:
        obs = self.obs_fn.compute(self.last_action)
        return TensorDict({"policy": obs, "critic": obs}, batch_size=[self.num_envs])

    def map_action(self, a: torch.Tensor) -> torch.Tensor:
        a = a.clamp(-1.0, 1.0)
        return self._aff_lo + (a + 1.0) * 0.5 * (self._aff_hi - self._aff_lo)

    def step(self, actions: torch.Tensor):
        a = actions.to(self.device).clamp(-1.0, 1.0)
        self.env.step(self.map_action(a))
        self.episode_length_buf += 1
        self.last_action = a
        reward, succ, stages = self.reward_fn.compute(a)
        time_out = self.episode_length_buf >= self.max_episode_length
        done = time_out  # fixed-length episodes: success is logged and paid, never terminal
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
            log["/episode/length_s"] = self.episode_length_buf[ids].float().mean() * self.step_dt
            self._reset_idx(ids)
        extras = {"time_outs": time_out, "log": log, "success": succ}
        return self.get_observations(), reward, done, extras

    def reset(self) -> TensorDict:
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def _reset_idx(self, ids: torch.Tensor) -> None:
        self.env.reset(ids)
        self.reward_fn.reset(ids)
        self.episode_length_buf[ids] = 0
        self.last_action[ids] = 0.0
        self._ep_ret[ids] = 0.0
        self._ep_peak[ids] = 0.0

    def close(self) -> None:
        self.env.close()

    def describe(self) -> str:
        t = self.cfg["task"]
        lines = [f"preset {t['preset']}  num_envs {self.num_envs}  device {self.device}",
                 f"control dt {self.step_dt:.4f} s ({1 / self.step_dt:.1f} Hz), physics dt {self.env.dt:.5f}",
                 f"episode {t['episode_seconds']} s = {self.max_episode_length} steps",
                 f"action dim {self.num_actions}  lo {self._aff_lo.tolist()}  hi {self._aff_hi.tolist()}",
                 f"reward {self.reward_fn.describe()}",
                 self.obs_fn.describe()]
        return "\n".join(lines)
