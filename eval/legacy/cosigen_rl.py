"""Minimal PPO core for AGENT-TRAINED sub-policies (runs ON the GPU render server).

The coding agent defines a sub-task MDP inside its own program (obs_fn / reward_fn /
done_fn as closures + a declarative action_spec) and calls `rl_train` (see
`cosigen_loop.AssemblyApi`). This module supplies the machinery:

  - small MLP actor-critic (agent-choosable hidden sizes, bounded by the caller)
  - PPO with GAE over BATCHED envs of the ONE warm robobench sim (num_envs = N)
  - episodes = fixed-horizon segments re-spawned from a broadcast env-0 snapshot
    (+ declarative start-state jitter), so training starts exactly at the hard moment
  - a policy registry (in-process) + save/load to HDFS

The policy's OUTPUT connects to the robobench CONTROLLER: actions are residual deltas
on the api's control targets (wrist pose targets + hand fraction), which the active
controller (pink_ik / joint) maps to joint commands. Never raw sim-state writes.

Design notes:
  - training steps the env DIRECTLY (env.step) so the episode step budget
    (api._steps_used) is not consumed and frame recording stays off;
  - obs normalization is built in (agent obs mix meters/radians/velocities);
  - everything is wall-clock budgeted (budget_s) and exception-safe for the caller.
"""
from __future__ import annotations

import os
import time
import uuid

import numpy as np
import torch
import torch.nn as nn

REGISTRY: dict[str, dict] = {}  # pid -> {"actor","critic","norm","spec","curve",...}

POLICY_HDFS_DIR = os.environ.get(
    "CAPX_COSIGEN_POLICY_DIR", "hdfs://haruna/tmp/zeyu.shen/cosigen_policies")


# --------------------------- small nets ---------------------------
def _mlp(sizes, act=nn.ELU, out_act=None):
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian policy (tanh-squashed to [-1,1]) + value head on separate trunks."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.actor = _mlp([obs_dim, *hidden, act_dim])
        self.critic = _mlp([obs_dim, *hidden, 1])
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs: torch.Tensor):
        mu = self.actor(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().clamp(1e-3, 2.0))

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        d = self.dist(obs)
        u = d.mean if deterministic else d.rsample()
        a = torch.tanh(u)
        # log-prob with tanh correction (SAC-style), summed over action dims
        logp = (d.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        return a, logp, self.critic(obs).squeeze(-1)


class RunningNorm:
    """Streaming mean/var obs normalizer (numpy)."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def norm(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


# --------------------------- action adapter ---------------------------
def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _quat_from_aa(aa: torch.Tensor) -> torch.Tensor:
    """(N,3) axis-angle -> (N,4) wxyz."""
    ang = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / ang
    half = ang * 0.5
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


class ActionAdapter:
    """Map policy outputs in [-1,1]^D to RESIDUAL deltas on the api's control targets
    (which the active robobench controller consumes). Declarative spec:

        {"arm": 1, "wrist_pos": 0.01, "wrist_rot": 0.0, "wrist_yaw": 0.0, "hand": 0.1}

    scale = max delta per CONTROL step (m / rad / hand-fraction); 0 disables the block.
    dims: wrist_pos->3, wrist_rot->3 (axis-angle), wrist_yaw->1 (rotation about the
    WORLD vertical axis only -- the mechanism-aligned DOF for screwing/threading:
    full 3-axis rotation exploration mostly produces task-destroying tilt there),
    hand->1. wrist_rot and wrist_yaw are mutually exclusive.
    arm may be 0, 1, or 'both' (bimanual: the pos/rot blocks are repeated per arm, in
    order [arm0, arm1]; the hand block stays a single shared dim)."""

    def __init__(self, api, spec: dict | None):
        spec = dict(spec or {})
        self.api = api
        arm = spec.get("arm", 1)
        self.arms = [0, 1] if arm in ("both", "dual", 2) else [int(arm)]
        self.s_pos = float(spec.get("wrist_pos", 0.01))
        self.s_rot = float(spec.get("wrist_rot", 0.0))
        self.s_yaw = float(spec.get("wrist_yaw", 0.0))
        self.s_hand = float(spec.get("hand", 0.1))
        if any(a not in (0, 1) for a in self.arms):
            raise ValueError("action_spec.arm must be 0, 1 or 'both'")
        if self.s_rot > 0 and self.s_yaw > 0:
            raise ValueError("wrist_rot and wrist_yaw are mutually exclusive")
        per_arm = (3 * (self.s_pos > 0) + 3 * (self.s_rot > 0)
                   + 1 * (self.s_yaw > 0))
        self.dim = per_arm * len(self.arms) + (self.s_hand > 0)
        if self.dim == 0:
            raise ValueError("action_spec enables no action dims (all scales are 0)")

    def apply(self, a: torch.Tensor) -> None:
        """a: (N, dim) in [-1,1]. Mutates the api's targets (all envs, per-env rows)."""
        api, k = self.api, 0
        for arm in self.arms:
            tgt = api._tgt[arm]
            if self.s_pos > 0:
                tgt[:, :3] += a[:, k:k + 3] * self.s_pos
                k += 3
            if self.s_rot > 0:
                dq = _quat_from_aa(a[:, k:k + 3] * self.s_rot)
                tgt[:, 3:7] = _quat_mul(dq, tgt[:, 3:7])
                tgt[:, 3:7] /= tgt[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
                k += 3
            if self.s_yaw > 0:
                aa = torch.zeros(a.shape[0], 3, device=a.device, dtype=a.dtype)
                aa[:, 2] = a[:, k] * self.s_yaw  # rotation about WORLD z only
                dq = _quat_from_aa(aa)
                tgt[:, 3:7] = _quat_mul(dq, tgt[:, 3:7])
                tgt[:, 3:7] /= tgt[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)
                k += 1
        if self.s_hand > 0:
            new = (api._hand_frac + a[:, k] * self.s_hand).clamp(0.0, 1.0)
            if hasattr(api, "set_hand_frac_tensor_arm"):
                # Bimanual apis keep PER-ARM gripper structures; drive only the arms
                # this policy controls (the inherited single-arm set_hand_frac_tensor
                # crashed on IkeaBimanualApi: no _hand_upper there, 2026-07-23).
                api.set_hand_frac_tensor_arm(new, self.arms)
            else:
                api.set_hand_frac_tensor(new)


# --------------------------- start-state handling ---------------------------
def _shift_root_states(tree, delta: torch.Tensor) -> None:
    """Add per-env origin deltas (n,3) to the POSITION columns of every root-state
    tensor (last dim 13 = pos3+quat4+vel6) in an expanded state tree, in place.

    Scene/robot get_state trees store WORLD-frame root states, and the env replicas
    live on an origin grid (env_spacing). Broadcasting env 0's state verbatim parks
    every replica's objects in env 0's grid cell — each replica's arm (fixed base in
    its own cell) is then tens of meters from its objects, so 511/512 episodes are
    born broken and terminate immediately (the 2026-07-24 'success=0.998 from iter 0,
    flat curve' RL bug). Re-offsetting positions by (origin_i - origin_0) gives every
    replica a faithful copy of env 0's state in its own cell."""
    if isinstance(tree, dict):
        for v in tree.values():
            _shift_root_states(v, delta)
    elif (torch.is_tensor(tree) and tree.dim() >= 2 and tree.shape[-1] == 13
          and tree.dtype.is_floating_point):
        d = delta.view(delta.shape[0], *([1] * (tree.dim() - 2)), 3).to(tree.device)
        tree[..., 0:3] += d


def _jitter_scene(state_n: dict, randomize: dict | None, gen: torch.Generator) -> None:
    """Apply the DECLARATIVE start-state jitter to an expanded scene state, in place.
    randomize = {"leg_2": {"pos": 0.01, "yaw": 0.3, "z": 0.02}, "table": {...}} — σ of xy
    noise (m) and yaw noise (rad) per object; "z" raises the object by Uniform(0, z) m at
    episode start (reverse curriculum: some episodes begin nearer success, e.g. already
    in-hand, so sparse hold/lift rewards are sampled from iteration one).
    This is the only sanctioned state perturbation."""
    if not randomize:
        return
    scene = state_n["scene"]

    def _first_tensor(tree):
        if torch.is_tensor(tree):
            return tree
        if isinstance(tree, dict):
            for v in tree.values():
                t = _first_tensor(v)
                if t is not None:
                    return t
        return None

    n = _first_tensor(scene).shape[0]

    def _apply(root: torch.Tensor, cfg: dict) -> None:
        s_pos = float(cfg.get("pos", 0.0))
        s_yaw = float(cfg.get("yaw", 0.0))
        s_z = float(cfg.get("z", 0.0))
        if s_pos > 0:
            root[:, :2] += torch.randn(n, 2, generator=gen, device=root.device) * s_pos
        if s_z > 0:
            root[:, 2] += torch.rand(n, generator=gen, device=root.device) * s_z
        if s_yaw > 0:
            yaw = torch.randn(n, generator=gen, device=root.device) * s_yaw
            dq = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw),
                              torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=-1)
            root[:, 3:7] = _quat_mul(dq, root[:, 3:7])

    for name, cfg in randomize.items():
        if name == "table" and "table" in scene:
            _apply(scene["table"], cfg)
        elif name.startswith("leg_") and "legs" in scene:
            _apply(scene["legs"][:, int(name.split("_")[1])], cfg)
        elif "cargo" in scene and name in scene["cargo"]:  # packing suite: cargo by name
            _apply(scene["cargo"][name], cfg)
        elif name == "lid" and "lid" in scene:
            _apply(scene["lid"], cfg)
        else:
            raise ValueError(f"randomize: unknown object {name!r} for this scene's state")


# --------------------------- training ---------------------------
def train(env, api, *, obs_fn, reward_fn, done_fn=None, action_spec=None,
          randomize=None, hidden_sizes=(256, 256), iters=40, budget_s=300.0,
          horizon=64, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4,
          minibatches=4, entropy_coef=0.005, seed=0, init_from: str | None = None,
          probe_fn=None, plateau_patience: int = 0) -> dict:
    """PPO-train a sub-policy from the CURRENT env-0 state (broadcast to all envs).
    Returns {"pid", "curve", "iters_run", "obs_dim", "act_dim", "improved", ...}. The env
    is left in the state the caller snapshotted (caller restores).

    Iteration-friendly extras:
      init_from=pid          warm-start weights/normalizer from an earlier policy (same
                             obs/act dims) -- train in SHORT bursts, continue only if the
                             curve climbs, instead of one long blind run;
      probe_fn(v)->dict      optional per-step diagnostics (floats); quantile stats are
                             returned under "probe" so dead reward terms are visible;
      plateau_patience=K     stop early if the best iteration reward hasn't improved in
                             K iterations (0 = off)."""
    from cosigen_loop import AssemblyView, expand_state_tree  # lazy: avoids import cycle

    t0 = time.time()
    n = env.num_envs
    device = env.device
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    torch.manual_seed(int(seed))

    adapter = ActionAdapter(api, action_spec)
    views = [AssemblyView(api, i) for i in range(n)]
    start = {"sim": env.get_states(torch.tensor([0], device=device)),
             "api": api._api_state_env0()}
    # Per-replica origin deltas: replicas get env 0's state re-offset into their own
    # grid cell (see _shift_root_states), and the api's WORLD-frame wrist targets get
    # the same shift (otherwise every replica's OSC controller pulls its arm toward
    # env 0's cell and the batch flails).
    origin_delta = (env.iscene.env_origins - env.iscene.env_origins[0:1]).to(device)

    def reset_to_start() -> None:
        state_n = expand_state_tree(start["sim"], n)
        _shift_root_states(state_n, origin_delta)
        _jitter_scene(state_n, randomize, gen)
        env.set_states(state_n)
        api._restore_api_state(start["api"])
        if not getattr(api, "_targets_env0_world", False):
            for tgt in getattr(api, "_tgt", []):  # world-frame wrist targets, one per arm
                tgt[:, :3] += origin_delta

    def get_obs() -> np.ndarray:
        rows = [np.asarray(obs_fn(v), dtype=np.float32).ravel() for v in views]
        d0 = rows[0].shape[0]
        if any(r.shape[0] != d0 for r in rows):
            raise ValueError("obs_fn returned inconsistent dims across envs")
        return np.stack(rows)

    reset_to_start()
    obs_dim = get_obs().shape[1]
    if init_from:
        prev = REGISTRY.get(init_from)
        if prev is None:
            raise KeyError(f"init_from: unknown policy {init_from!r}")
        if prev["obs_dim"] != obs_dim or prev["act_dim"] != adapter.dim:
            raise ValueError(f"init_from dims mismatch: policy is "
                             f"({prev['obs_dim']},{prev['act_dim']}), MDP is ({obs_dim},{adapter.dim})")
        hidden_sizes = tuple(prev["hidden_sizes"])
        net = ActorCritic(obs_dim, adapter.dim, hidden_sizes).to(device)
        net.load_state_dict(prev["net"].state_dict())
        norm = RunningNorm(obs_dim)
        norm.mean, norm.var, norm.count = (prev["norm"].mean.copy(),
                                           prev["norm"].var.copy(), prev["norm"].count)
    else:
        net = ActorCritic(obs_dim, adapter.dim, tuple(hidden_sizes)).to(device)
        norm = RunningNorm(obs_dim)
    opt = torch.optim.Adam(net.parameters(), lr=float(lr))
    curve: list[dict] = []
    probe_acc: dict[str, list] = {}
    best_state = None  # (ep_reward, state_dict, iter) of the best iteration

    iters_run = 0
    for it in range(int(iters)):
        if time.time() - t0 > float(budget_s):
            break
        reset_to_start()
        api._policy_last_action = np.zeros(adapter.dim, dtype=np.float32)

        obs_buf = np.zeros((horizon, n, obs_dim), dtype=np.float32)
        act_buf = torch.zeros(horizon, n, adapter.dim, device=device)
        logp_buf = torch.zeros(horizon, n, device=device)
        val_buf = torch.zeros(horizon + 1, n, device=device)
        rew_buf = torch.zeros(horizon, n, device=device)
        alive_buf = torch.ones(horizon, n, device=device)      # alive when ACTING at step t
        nextalive_buf = torch.ones(horizon, n, device=device)  # alive AFTER step t's done check
        alive = torch.ones(n, device=device)

        raw = get_obs()
        norm.update(raw)
        for t in range(horizon):
            ob = torch.as_tensor(norm.norm(raw), device=device)
            with torch.no_grad():
                a, logp, v = net.act(ob)
            adapter.apply(a * alive.unsqueeze(1))  # frozen (done) envs hold targets
            env.step(api._action(), render=False)

            rew = torch.zeros(n, device=device)
            done = torch.zeros(n, device=device)
            for i, view in enumerate(views):
                view.last_action = a[i].detach().cpu().numpy()
                rew[i] = float(reward_fn(view))
                if done_fn is not None and bool(done_fn(view)):
                    done[i] = 1.0
                if probe_fn is not None and alive[i] > 0:
                    try:
                        for key, val in (probe_fn(view) or {}).items():
                            probe_acc.setdefault(key, []).append(float(val))
                    except Exception:
                        probe_fn = None  # broken probe: disable, never kill training
            obs_buf[t] = norm.norm(raw)
            act_buf[t], logp_buf[t], val_buf[t] = a, logp, v
            rew_buf[t] = rew * alive
            alive_buf[t] = alive
            alive = alive * (1.0 - done)
            nextalive_buf[t] = alive
            raw = get_obs()
            norm.update(raw)

        with torch.no_grad():
            _, _, v_last = net.act(torch.as_tensor(norm.norm(raw), device=device))
        val_buf[horizon] = v_last

        # GAE over the segment; a done at step t is terminal (no bootstrap past it)
        adv = torch.zeros(horizon, n, device=device)
        last = torch.zeros(n, device=device)
        for t in reversed(range(horizon)):
            nonterm = nextalive_buf[t]  # 0 if the episode ended AT step t
            delta = rew_buf[t] + gamma * val_buf[t + 1] * nonterm - val_buf[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + val_buf[:horizon]

        mask = alive_buf.reshape(-1) > 0
        b_obs = torch.as_tensor(obs_buf.reshape(-1, obs_dim), device=device)[mask]
        b_act = act_buf.reshape(-1, adapter.dim)[mask]
        b_logp = logp_buf.reshape(-1)[mask]
        b_adv = adv.reshape(-1)[mask]
        b_ret = ret.reshape(-1)[mask]
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-6)

        idx = torch.randperm(b_obs.shape[0], device=device)
        mb = max(1, b_obs.shape[0] // int(minibatches))
        for _ in range(int(epochs)):
            for s in range(0, b_obs.shape[0], mb):
                j = idx[s:s + mb]
                d = net.dist(b_obs[j])
                u = torch.atanh(b_act[j].clamp(-0.999, 0.999))
                logp = (d.log_prob(u) - torch.log(1 - b_act[j].pow(2) + 1e-6)).sum(-1)
                ratio = (logp - b_logp[j]).exp()
                pol = -torch.min(ratio * b_adv[j],
                                 ratio.clamp(1 - clip, 1 + clip) * b_adv[j]).mean()
                vf = (net.critic(b_obs[j]).squeeze(-1) - b_ret[j]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pol + 0.5 * vf - float(entropy_coef) * ent
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()

        ep_rew = float(rew_buf.sum(0).mean())
        succ = float((1.0 - alive).mean())
        curve.append({"iter": it, "ep_reward": round(ep_rew, 4), "success": round(succ, 3),
                      "t": round(time.time() - t0, 1)})
        iters_run = it + 1
        # BEST-ITERATE retention: PPO on contact-rich tasks can catastrophically forget a
        # lucky-lift behavior in later iterations; keep the weights of the best iteration,
        # not the last one.
        if best_state is None or ep_rew > best_state[0]:
            best_state = (ep_rew, {k: v.detach().clone() for k, v in net.state_dict().items()},
                          it)
        if plateau_patience and len(curve) > int(plateau_patience):
            rewards = [c["ep_reward"] for c in curve]
            best_at = int(np.argmax(rewards))
            if len(rewards) - 1 - best_at >= int(plateau_patience):
                break

    if best_state is not None and best_state[2] != iters_run - 1:
        net.load_state_dict(best_state[1])
    pid = "pi_" + uuid.uuid4().hex[:8]
    REGISTRY[pid] = {
        "net": net, "norm": norm, "obs_dim": obs_dim, "act_dim": adapter.dim,
        "action_spec": dict(action_spec or {}), "hidden_sizes": list(hidden_sizes),
        "obs_fn": obs_fn, "done_fn": done_fn, "curve": curve,
        "best_iter": None if best_state is None else best_state[2],
    }
    rewards = [c["ep_reward"] for c in curve]
    k = max(1, len(rewards) // 5)
    improved = bool(len(rewards) >= 4 and
                    float(np.mean(rewards[-k:])) > float(np.mean(rewards[:k])) + 1e-6)
    out = {"pid": pid, "iters_run": iters_run, "obs_dim": obs_dim,
           "act_dim": adapter.dim, "seconds": round(time.time() - t0, 1),
           "curve": curve, "improved": improved,
           "stopped": "plateau" if iters_run < int(iters) and
                      time.time() - t0 <= float(budget_s) else "budget_or_iters"}
    if probe_acc:
        out["probe"] = {
            key: {"mean": round(float(np.mean(vals)), 4),
                  "p50": round(float(np.percentile(vals, 50)), 4),
                  "p95": round(float(np.percentile(vals, 95)), 4),
                  "max": round(float(np.max(vals)), 4),
                  "frac_pos": round(float(np.mean(np.asarray(vals) > 0)), 4)}
            for key, vals in probe_acc.items()
        }
    return out


# --------------------------- execution ---------------------------
def run(env, api, pid: str, *, max_steps=300, obs_fn=None, stop_when=None,
        deterministic=True) -> dict:
    """Roll a trained policy from the CURRENT state, env-0-authoritative (same action
    broadcast to all envs, so mirrors stay in sync). Steps via api.step -> consumes the
    episode budget and records video frames like any control call."""
    from cosigen_loop import AssemblyView

    rec = REGISTRY.get(pid)
    if rec is None:
        raise KeyError(f"unknown policy {pid!r}; trained this session: {list(REGISTRY)}")
    net, norm = rec["net"], rec["norm"]
    obs_fn = obs_fn or rec.get("obs_fn")
    if obs_fn is None:
        raise ValueError("policy has no stored obs_fn (loaded from disk?) — pass obs_fn=")
    adapter = ActionAdapter(api, rec["action_spec"])
    v0 = AssemblyView(api, 0)
    n = env.num_envs
    steps = 0
    for steps in range(1, int(max_steps) + 1):
        raw = np.asarray(obs_fn(v0), dtype=np.float32).ravel()[None]
        ob = torch.as_tensor(norm.norm(raw), device=env.device)
        with torch.no_grad():
            a, _, _ = net.act(ob, deterministic=deterministic)
        adapter.apply(a.repeat(n, 1))
        api.step(1)
        v0.last_action = a[0].detach().cpu().numpy()
        if stop_when is not None and bool(stop_when(v0)):
            return {"pid": pid, "steps": steps, "stopped": "stop_when"}
        if rec.get("done_fn") is not None and bool(rec["done_fn"](v0)):
            return {"pid": pid, "steps": steps, "stopped": "done_fn"}
        if api._steps_used >= api.max_steps:
            return {"pid": pid, "steps": steps, "stopped": "episode_budget"}
    return {"pid": pid, "steps": steps, "stopped": "max_steps"}


# --------------------------- persistence ---------------------------
def save(pid: str, name: str) -> str:
    rec = REGISTRY.get(pid)
    if rec is None:
        raise KeyError(f"unknown policy {pid!r}")
    payload = {
        "actor_critic": rec["net"].state_dict(),
        "norm": {"mean": rec["norm"].mean, "var": rec["norm"].var, "count": rec["norm"].count},
        "obs_dim": rec["obs_dim"], "act_dim": rec["act_dim"],
        "action_spec": rec["action_spec"], "hidden_sizes": rec["hidden_sizes"],
        "curve": rec["curve"],
    }
    local = f"/tmp/cosigen_policy_{name}.pt"
    torch.save(payload, local)
    dst = POLICY_HDFS_DIR.rstrip("/") + f"/{name}.pt"
    rc = os.system(f"hdfs dfs -mkdir -p {POLICY_HDFS_DIR} 2>/dev/null; "
                   f"hdfs dfs -put -f {local} {dst}")
    if rc != 0:
        raise RuntimeError(f"hdfs put failed rc={rc} dst={dst}")
    return dst


def load(name: str, device: str = "cuda:0") -> str:
    src = POLICY_HDFS_DIR.rstrip("/") + f"/{name}.pt"
    local = f"/tmp/cosigen_policy_{name}.pt"
    if os.system(f"hdfs dfs -get -f {src} {local}") != 0:
        raise RuntimeError(f"hdfs get failed: {src}")
    p = torch.load(local, map_location=device, weights_only=False)
    net = ActorCritic(p["obs_dim"], p["act_dim"], tuple(p["hidden_sizes"])).to(device)
    net.load_state_dict(p["actor_critic"])
    norm = RunningNorm(p["obs_dim"])
    norm.mean, norm.var, norm.count = p["norm"]["mean"], p["norm"]["var"], p["norm"]["count"]
    pid = "pi_" + uuid.uuid4().hex[:8]
    REGISTRY[pid] = {"net": net, "norm": norm, "obs_dim": p["obs_dim"],
                     "act_dim": p["act_dim"], "action_spec": p["action_spec"],
                     "hidden_sizes": p["hidden_sizes"], "obs_fn": None, "done_fn": None,
                     "curve": p.get("curve", [])}
    return pid
