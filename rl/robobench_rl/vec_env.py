"""rsl_rl VecEnv over a robobench BaseEnv.

Fixed-length episodes at the task's horizon (dense per-step reward; success is logged, not terminal).
The policy emits [-1, 1]^A on the task's FROZEN controller preset; `action.affine` maps gripper dims to
[lo, hi] metres or arm dims to small joint deltas, `action.gripper_mirror` drives several finger dims from
one policy scalar. Observation = the state rule in obs.py. Optional `robot.gripper_stiffness` (finger PD)
and a warm-start curriculum: at each lockstep reset a fraction of envs is servoed to the shaped reward's
hover target before the policy takes over (Factory-style); grading always starts from home."""
from __future__ import annotations

import math
import time

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .obs import StateObs
from .reward import GraderReward, load_grader_cls


def set_gripper_stiffness(env, kp: float) -> None:
    """Write `kp` as the PD stiffness of the robot's declared GRIPPER_JOINTS (no-op without a gripper)."""
    art = getattr(env.robot, "articulation", None)
    pats = getattr(env.robot, "GRIPPER_JOINTS", None)
    if art is None or not pats:
        return
    art.write_joint_stiffness_to_sim(kp, joint_ids=art.find_joints(list(pats))[0])


class RoboBenchVecEnv(VecEnv):
    def __init__(self, cfg: dict, device: str | None = None) -> None:
        import robobench
        from robobench.core import ENVS, SCENES

        robobench.discover()
        self.cfg = cfg
        t = cfg["task"]
        self.device = torch.device(device or cfg.get("device", "cuda:0"))
        env_cfg = ENVS.get(t["preset"])()
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
        self.env = env_cfg.build(num_envs=n, device=str(self.device), seed=cfg.get("seed", 0), sim_overrides=sim_overrides)
        print(f"[rl] env build ({n} envs) {time.time() - t0:.1f} s", flush=True)
        self.num_envs = self.env.num_envs
        self.gripper_stiffness = (cfg.get("robot") or {}).get("gripper_stiffness")
        if self.gripper_stiffness:
            set_gripper_stiffness(self.env, float(self.gripper_stiffness))

        # ----- action mapping (serialised to env.json; the exported solve.py replays it) -----
        act = cfg.get("action", {}) or {}
        E = self.env.robot.action_dim
        self.mirror_dims = list(act.get("gripper_mirror") or [])
        self.keep_dims = [d for d in range(E) if d not in self.mirror_dims]
        self.num_actions = E - max(0, len(self.mirror_dims) - 1)
        self.env_action_dim = E
        self.lo = torch.full((E,), -1.0, device=self.device)
        self.hi = torch.full((E,), 1.0, device=self.device)
        self.jd_dims, self.jd_joint_ids, self.jd_scale, self.jd_lim = [], [], 0.0, None
        for spec in act.get("affine", []) or []:
            dims = list(spec["dims"])
            if spec.get("from") in ("joint_delta", "joint_limits"):
                ctrl = self.env.robot.controller
                arm = ctrl.controllers[0] if hasattr(ctrl, "controllers") else ctrl
                lim = self.env.robot.articulation.data.soft_joint_pos_limits[0, arm.joint_ids]
                if len(dims) != lim.shape[0]:
                    raise ValueError(f"action.affine {spec['from']}: {len(dims)} dims vs {lim.shape[0]} arm joints")
                if spec["from"] == "joint_delta":  # target = clamp(q + a*scale): small steps, never lunges
                    self.jd_dims, self.jd_joint_ids, self.jd_scale, self.jd_lim = dims, list(arm.joint_ids), float(spec["scale"]), lim.clone()
                else:
                    for d, (a, b) in zip(dims, lim.tolist()):
                        self.lo[d], self.hi[d] = a, b
            else:
                for d in dims:
                    self.lo[d], self.hi[d] = float(spec["lo"]), float(spec["hi"])

        self.step_dt = self.env.dt * self.env.robot.control_period
        self.max_episode_length = int(math.ceil(float(t["episode_seconds"]) / self.step_dt))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        o = cfg.get("obs", {})
        self.obs_fn = StateObs(self.env, drop_keys=tuple(o.get("drop_keys", ())), env_local=o.get("env_local", True),
                               ee_pose=o.get("ee_pose", True), last_action=o.get("last_action", True), action_dim=self.num_actions)
        self.num_obs = self.obs_fn.dim
        r = cfg.get("reward", {})
        self.reward_fn = GraderReward(self.env, load_grader_cls(t["preset"], env_cfg.scene), env_cfg.scene,
                                      mode=r.get("mode", "progress"), success_bonus=float(r.get("success_bonus", 1.0)),
                                      weights=r.get("weights"))
        cur = cfg.get("curriculum", {}) or {}
        self.hover_frac = float(cur.get("hover_start_frac", 0.0))
        self.hover_steps = int(cur.get("hover_steps", 90))
        self.hover_jitter = float(cur.get("hover_jitter", 0.03))
        self._in_preroll = False
        self.preroll_warm = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_ret = torch.zeros(self.num_envs, device=self.device)
        self._ep_peak = torch.zeros(self.num_envs, device=self.device)
        self.total_episodes, self.total_successes = 0, 0

    # ----- actions -----------------------------------------------------------------------------
    def action_map_json(self) -> dict:
        return {"env_action_dim": self.env_action_dim, "lo": self.lo.tolist(), "hi": self.hi.tolist(),
                "mirror_dims": self.mirror_dims, "jd_dims": self.jd_dims, "jd_joint_ids": self.jd_joint_ids,
                "jd_scale": self.jd_scale, "jd_lim": None if self.jd_lim is None else self.jd_lim.tolist()}

    def map_action(self, a: torch.Tensor) -> torch.Tensor:
        return map_action(self.env, a.clamp(-1.0, 1.0), self.action_map_json(), self.device)

    # ----- VecEnv API --------------------------------------------------------------------------
    def get_observations(self) -> TensorDict:
        obs = self.obs_fn.compute(self.last_action)
        return TensorDict({"policy": obs, "critic": obs}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        a = actions.to(self.device).clamp(-1.0, 1.0)
        self.env.step(self.map_action(a))
        self.episode_length_buf += 1
        self.last_action = a
        reward, succ, stages = self.reward_fn.compute()
        done = self.episode_length_buf >= self.max_episode_length
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
            self._reset_idx(ids)
        return self.get_observations(), reward, done, {"time_outs": done, "log": log, "success": succ}

    def reset(self) -> TensorDict:
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def _reset_idx(self, ids: torch.Tensor) -> None:
        self.env.reset(ids)
        if self.hover_frac > 0 and not self._in_preroll:
            self._hover_preroll(ids)
        self.reward_fn.reset(ids)
        self.episode_length_buf[ids] = 0
        self.last_action[ids] = 0.0
        self._ep_ret[ids] = 0.0
        self._ep_peak[ids] = 0.0

    @torch.no_grad()
    def _hover_preroll(self, ids: torch.Tensor) -> None:
        """Servo a random `hover_frac` of the envs to the shaped reward's hover target with P-control on the
        controller's position-delta actions, fingers open. Physics advances for ALL envs, so every env
        must be resetting together (fixed-length episodes, init_at_random_ep_len off)."""
        task = self.reward_fn.task
        target = task.hover_target() if task is not None else None
        if target is None:
            return
        if ids.numel() != self.num_envs:
            print("[rl] hover curriculum needs lockstep resets; partial reset -> skipped", flush=True)
            return
        self._in_preroll = True
        n = self.num_envs
        warm = torch.rand(n, device=self.device) < self.hover_frac
        target = target + (torch.rand(n, 3, device=self.device) * 2 - 1) * self.hover_jitter * torch.tensor([1.0, 1.0, 0.0], device=self.device)
        art = self.env.robot.articulation
        ee = list(art.data.body_names).index(self.env.robot.EE_BODY)
        ctrl = self.env.robot.controller
        arm = ctrl.controllers[0] if hasattr(ctrl, "controllers") else ctrl
        pos_scale = float(getattr(getattr(arm, "cfg", None), "pos_scale", 0.02))
        for _ in range(self.hover_steps):
            a = torch.zeros(n, self.num_actions, device=self.device)
            a[:, 0:3] = ((target - art.data.body_pos_w[:, ee]) / pos_scale).clamp(-1.0, 1.0) * warm[:, None].float()
            a[:, 6:] = 1.0  # fingers open (arm dims first, gripper last — every preset here)
            self.env.step(self.map_action(a))
        self.preroll_warm = warm
        self._in_preroll = False

    def close(self) -> None:
        self.env.close()

    def describe(self) -> str:
        t = self.cfg["task"]
        return "\n".join([
            f"preset {t['preset']}  num_envs {self.num_envs}  device {self.device}",
            f"control dt {self.step_dt:.4f} s ({1 / self.step_dt:.1f} Hz), physics dt {self.env.dt:.5f}",
            f"episode {t['episode_seconds']} s = {self.max_episode_length} steps"
            + (f"; hover curriculum {self.hover_frac:.0%} x {self.hover_steps} steps" if self.hover_frac > 0 else ""),
            f"action dim {self.num_actions} (env {self.env_action_dim}, mirror {self.mirror_dims or 'off'}) "
            f"lo {[round(x, 3) for x in self.lo.tolist()]} hi {[round(x, 3) for x in self.hi.tolist()]}"
            + (f"  joint_delta dims {self.jd_dims} scale {self.jd_scale} rad/step" if self.jd_dims else ""),
            f"reward {self.reward_fn.describe()}",
            self.obs_fn.describe(),
        ])


def map_action(env, a: torch.Tensor, m: dict, device) -> torch.Tensor:
    """Policy [-1,1]^A -> env action from the serialised mapping `m` (shared with the exported solve.py)."""
    E, mirror = int(m["env_action_dim"]), list(m.get("mirror_dims") or [])
    if mirror:
        keep = [d for d in range(E) if d not in mirror]
        full = torch.zeros(a.shape[0], E, device=a.device, dtype=a.dtype)
        full[:, keep] = a[:, : len(keep)]
        full[:, mirror] = a[:, -1:].expand(-1, len(mirror))
        a = full
    lo = torch.as_tensor(m["lo"], dtype=torch.float32, device=device)
    hi = torch.as_tensor(m["hi"], dtype=torch.float32, device=device)
    out = lo + (a + 1.0) * 0.5 * (hi - lo)
    if m.get("jd_dims"):
        q = env.robot.articulation.data.joint_pos[:, m["jd_joint_ids"]]
        lim = torch.as_tensor(m["jd_lim"], dtype=torch.float32, device=device)
        out = out.clone()
        out[:, m["jd_dims"]] = (q + a[:, m["jd_dims"]] * float(m["jd_scale"])).clamp(lim[:, 0], lim[:, 1])
    return out
