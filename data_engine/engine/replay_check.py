"""Action replay check — the last gate an episode passes before it can be delivered.

A recorded success is only training data if its recorded actions CAUSE the
success: replayed open-loop in the same world, they must pass the grader again.
An episode that fails this depended on something the labels do not carry (a
lucky contact, timing jitter, non-determinism) and is discarded.

What is replayed: the EXECUTED action sequence — `action` (the clean label)
plus `action_noise` (the perturbation the recorder executed on noisy batches).
For clean episodes the two coincide; for noisy ones this tests that what
actually happened reproduces, not that the clean intent alone would.

How the world is rebuilt: the WHOLE recording batch, whichever subset was asked
for — same scene, same width, same seed, the batch's recorded per-env physics
draws — every recorded episode restored into ITS OWN slot. Recorded states are
world-frame (they carry the env's grid origin), so an episode can only be
restored into the slot it was recorded in: packing episodes of different batches
into one world put every object a grid cell away from its robot (0/8 reproduced
where per-batch replay gave 7/8 — measured 2026-09-05). And on GPU PhysX each
env's contacts shape every other env's, so the verdict of an episode is defined
as its verdict in the full-batch replay: a 6-episode subset with idle slots
running filler actions and the 399-episode remainder of one batch disagreed
6/6 vs 10/399. Every episode of the batch gets its verdict written in one pass.

One recording batch per process (a scene cannot be rebuilt inside a running
Isaac app: "A prim already exists at path '/World/ground'"). Results land in
each episode's meta.json under `replay`:
    {"success": bool, "score": float, "steps": T, "checked_at": iso}
The log traces the most-diverging recorded state leaves at ~8 checkpoints
(REPLAY_TRACE_EVERY=<steps> for a denser trace when diagnosing a failure).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .generation import build_env, load_grader_cls

_NOT_STATE = ("action", "action_noise", "robot/joint_target")


def _read(p: Path) -> dict:
    return json.loads(p.read_text())


def _clone_tree(d: dict) -> dict:
    return {k: _clone_tree(v) if isinstance(v, dict) else v.detach().clone()
            for k, v in d.items()}


def _leaf(tree: dict, key: str):
    """The tensor at a recorded flat key ("scene/nut/root_state") in the live
    get_states tree, or None when the live tree has no such leaf."""
    node = tree
    for part in key.split("/"):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return None if isinstance(node, dict) else node


# ---- the recorded control law -------------------------------------------------------------
# A solve is free to set controller/articulation parameters (gains, control period, nullspace
# posture, gripper PD) — every reference solve does, at the top of solve(). The recorder stamps
# the EFFECTIVE law into the episode meta (`controller`, captured after the solve, plus
# `controller_changes` for anything that moved mid-solve). Until 2026-09-06 the replay rebuilt
# the preset env and never re-applied any of it, so the recorded actions — computed for
# rot-kp 600 at 240 Hz — were executed by rot-kp 30 at 15 Hz and diverged from step 1: every
# retuning solve was "success does not REPLAY", and four campaigns spent 12 h of repair
# rounds trying to make the preset gains do a wrist reorientation that robobench itself
# documents they cannot (robots/franka.py `task_prop_gains`). The replay now runs under the
# recorded law: the same env, the same parameters, the same actions.

def _leaves(robot) -> list:
    return list(getattr(robot.controller, "controllers", None) or [robot.controller])


def _set_ctrl_path(robot, path: str, value, device) -> None:
    """Write one `_controller_info` path back onto the live robot.
    Paths: leaves[i].kp | leaves[i].kd | leaves[i].q_default | leaves[i].control_period |
    leaves[i].cfg.<field> | joint_stiffness | joint_damping."""
    import re
    import torch

    art = robot.articulation
    if path in ("joint_stiffness", "joint_damping"):
        t = torch.tensor(value, device=device, dtype=torch.float32)
        t = t.unsqueeze(0).expand(art.num_instances, -1).clone()
        (art.write_joint_stiffness_to_sim if path == "joint_stiffness"
         else art.write_joint_damping_to_sim)(t)
        return
    m = re.match(r"leaves\[(\d+)\]\.(.+)", path)
    if not m:
        raise RuntimeError(f"unknown controller path in recording: {path!r}")
    leaf = _leaves(robot)[int(m.group(1))]
    attr = m.group(2)
    if attr in ("kp", "kd", "q_default"):
        setattr(leaf, "_" + attr, torch.tensor(value, device=device, dtype=torch.float32))
    elif attr == "control_period":
        leaf._control_period = int(value)
    elif attr.startswith("cfg."):
        field = attr[4:]
        if not hasattr(leaf.cfg, field):
            raise RuntimeError(f"recorded cfg field {field!r} does not exist on {type(leaf).__name__}")
        setattr(leaf.cfg, field, tuple(value) if isinstance(value, list) else value)
    elif attr in ("class", "control_dt"):
        return                                   # descriptive, not a parameter
    else:
        raise RuntimeError(f"unknown controller leaf attribute in recording: {attr!r}")


def _law_at_start(ctrl: dict, changes: list[dict]) -> dict[str, object]:
    """{path: value} of the law the episode STARTED under: the final block, with every
    mid-solve change undone (the earliest change of a path carries its original value)."""
    flat: dict[str, object] = {}
    for i, leaf in enumerate(ctrl.get("leaves", [])):
        for k in ("kp", "kd", "q_default", "control_period"):
            if leaf.get(k) is not None:
                flat[f"leaves[{i}].{k}"] = leaf[k]
        for k, v in (leaf.get("cfg") or {}).items():
            if isinstance(v, (int, float, bool, list)):   # numeric/bool/list fields; strings
                flat[f"leaves[{i}].cfg.{k}"] = v          # (ee_body, names) are bind-time only
    for k in ("joint_stiffness", "joint_damping"):
        if ctrl.get(k):
            flat[k] = ctrl[k]
    first_old: dict[str, object] = {}
    for ch in sorted(changes, key=lambda c: c["step"]):
        for path, (old, _new) in ch["changed"].items():
            first_old.setdefault(path, old)
    flat.update(first_old)
    return flat


def apply_control_law(env, meta: dict, device) -> tuple[list[dict], int]:
    """Put the live robot under the law the episode was recorded with (see above) and return
    the mid-solve change schedule [(step, {path: new})] plus how many parameters were set."""
    ctrl = meta.get("controller") or {}
    if not ctrl or ctrl.get("children"):
        # no block (pre-2026-09 recording) or a composite of robots (bimanual) — the latter is
        # not re-applied yet; say so instead of silently replaying under the preset
        if ctrl.get("children"):
            print("[replay-check] WARNING: composite-robot controller block not re-applied "
                  "(replay runs under the preset law)", flush=True)
        return [], 0
    robot = env.robot
    leaves = _leaves(robot)
    rec_leaves = ctrl.get("leaves", [])
    if len(rec_leaves) != len(leaves) or any(
            r.get("class") != type(c).__name__ for r, c in zip(rec_leaves, leaves)):
        raise RuntimeError("recorded controller does not match the preset's: "
                           f"{[r.get('class') for r in rec_leaves]} vs {[type(c).__name__ for c in leaves]}")
    n = 0
    for path, value in _law_at_start(ctrl, meta.get("controller_changes") or []).items():
        _set_ctrl_path(robot, path, value, device)
        n += 1
    schedule = [(int(ch["step"]), {p: nv for p, (_o, nv) in ch["changed"].items()})
                for ch in sorted(meta.get("controller_changes") or [], key=lambda c: c["step"])]
    return schedule, n


def replay_episodes(gen_root: str | Path, eps: list[Path], device: str = "cuda:0") -> dict[str, bool]:
    """Replay the whole batch the episode dirs `eps` belong to (one batch per call);
    returns {episode dir: replay success} for every recorded episode of the batch and
    writes the `replay` verdict into each episode's meta.json."""
    import numpy as np
    import torch

    gen_root = Path(gen_root)
    batches = {ep.parent for ep in eps}
    if len(batches) != 1:
        raise SystemExit(f"replay check takes episodes of one batch per process, got {sorted(batches)}")
    batch_dir = batches.pop()
    eps = sorted(p for p in batch_dir.iterdir() if p.name.startswith("ep_") and (p / "traj.npz").is_file())
    bmeta = _read(batch_dir / "meta.json")
    scene = bmeta["cell"].split("/")[0]
    scene_dir = gen_root / "scenes" / scene
    num_envs = int(bmeta["num_envs"])
    env, _, _, _ = build_env(scene_dir, num_envs, device, seed=int(bmeta["seed"]), nominal=True)
    per_env = (bmeta.get("params") or {}).get("per_env") or []
    if per_env:
        # the batch's recorded draws, slot for slot (slot 0 / missing = nominal)
        names = sorted({n for d in per_env for n in d})
        cfg = env.scene.cfg
        env.scene.apply_physical_params(
            env, {n: [d.get(n, getattr(cfg, n)) for d in per_env] for n in names})
    grader_cls = load_grader_cls(scene_dir)

    results: dict[str, bool] = {}
    # episodes of one batch share the world; one rollout (= one reset seed) at a time
    by_rollout: dict[int, list[Path]] = {}
    for ep in eps:
        by_rollout.setdefault(int(_read(ep / "meta.json")["rollout"]), []).append(ep)
    for rollout, r_eps in sorted(by_rollout.items()):
        metas = {ep: _read(ep / "meta.json") for ep in r_eps}
        trajs = {ep: dict(np.load(ep / "traj.npz")) for ep in r_eps}
        slots = {ep: int(metas[ep]["env_index"]) for ep in r_eps}
        env.reset(seed=int(metas[r_eps[0]]["seed"]))

        # the law the episodes ran under (one per batch: all its episodes ran the same code);
        # its mid-solve changes are re-applied at their recorded steps in the loop below
        schedule, n_params = apply_control_law(env, metas[r_eps[0]], device)
        print(f"[replay-check {batch_dir.name}] rollout {rollout}: control law restored "
              f"({n_params} parameters, {len(schedule)} mid-solve changes)", flush=True)

        # Restore each episode's recorded step-0 state into its own slot — but ONLY the slots
        # whose live state (after reset(seed)) differs from the recording. `set_states` is a
        # write into PhysX on top of an already-identical world: it perturbs the solver's
        # contact/warm-start state, and a friction-held manipulation (bulb: mu 0.01, no grasp
        # weld) drifts from it 20 s later and fails, while the same recording replays
        # BIT-EXACTLY for all 18 540 steps when the redundant restore is skipped (measured
        # 2026-09-07). Episodes that did not start from the reset world (phase cells, noisy
        # or physically-drawn slots) still get restored, slot by slot.
        state = _clone_tree(env.get_states())          # full tree: for the drift trace + shapes
        differing: list[Path] = []
        for ep in r_eps:
            e = slots[ep]
            for k, v in trajs[ep].items():
                leaf = None if k in _NOT_STATE else _leaf(state, k)
                if leaf is None:
                    continue
                rec = torch.as_tensor(v[0], device=leaf.device, dtype=leaf.dtype)
                if rec.shape == leaf[e].shape and not torch.allclose(leaf[e], rec, rtol=0.0, atol=1e-6):
                    if ep not in differing:
                        differing.append(ep)
                leaf[e] = rec
        if differing:
            # set_states(states, env_ids) takes tensors already sliced to env_ids (row i <-> env_ids[i])
            ids = torch.tensor([slots[ep] for ep in differing], device=device, dtype=torch.long)
            sub = _clone_tree(env.get_states(env_ids=ids))
            for i, ep in enumerate(differing):
                for k, v in trajs[ep].items():
                    leaf = None if k in _NOT_STATE else _leaf(sub, k)
                    if leaf is not None:
                        leaf[i] = torch.as_tensor(v[0], device=leaf.device, dtype=leaf.dtype)
            env.set_states(sub, env_ids=ids)
        print(f"[replay-check {batch_dir.name}] rollout {rollout}: step-0 state restored for "
              f"{len(differing)}/{len(r_eps)} slots (the rest already match reset(seed))", flush=True)
        grader = grader_cls(env)
        grader.setup()
        entry_success = [bool(v["success"]) for v in grader.verdict()]

        executed = {}
        for ep in r_eps:
            a = trajs[ep]["action"].astype(np.float32)
            if "action_noise" in trajs[ep]:
                a = a + trajs[ep]["action_noise"].astype(np.float32)
            executed[ep] = torch.as_tensor(a, device=device)
        T = max(int(metas[ep]["steps"]) for ep in r_eps)
        print(f"[replay-check {batch_dir.name}] rollout {rollout}: {len(r_eps)} episodes, "
              f"{T} steps, {num_envs} envs", flush=True)
        sample = trajs[r_eps[0]]
        leaves = [k for k in sample if k not in _NOT_STATE and _leaf(state, k) is not None]
        every = int(os.environ.get("REPLAY_TRACE_EVERY", 0)) or max(1, T // 8)

        def trace(t: int) -> None:
            live = env.get_states()
            for ep in r_eps:
                e, tt = slots[ep], min(t + 1, int(metas[ep]["steps"]) - 1)
                diffs = []
                for k in leaves:
                    cur = _leaf(live, k)[e].detach().cpu().numpy().astype(np.float64)
                    rec = np.asarray(trajs[ep][k][tt], dtype=np.float64)
                    if cur.shape == rec.shape:
                        diffs.append((float(np.abs(cur - rec).max()), k))
                shown = [x for x in sorted(diffs, reverse=True) if x[0] > 1e-6][:8] or sorted(diffs, reverse=True)[:2]
                top = ", ".join(f"{k}={d:.4g}" for d, k in shown)
                print(f"[replay-check {batch_dir.name}] {ep.name} step {t + 1}/{T} "
                      f"largest drifts: {top}", flush=True)

        # a finished episode holds its last action (its trajectory was saved to sustained
        # success + one probe interval, so holding keeps it there); a slot with no
        # recorded episode in this rollout holds the first episode's action
        filler = executed[r_eps[0]]
        pending_changes = list(schedule)
        for t in range(T):
            # a change recorded at step s was in effect for action s onwards
            while pending_changes and pending_changes[0][0] <= t:
                _, values = pending_changes.pop(0)
                for path, value in values.items():
                    _set_ctrl_path(env.robot, path, value, device)
            row = filler[min(t, filler.shape[0] - 1)]
            act = row.unsqueeze(0).expand(num_envs, *row.shape).clone()
            for ep in r_eps:
                seq = executed[ep]
                act[slots[ep]] = seq[min(t, seq.shape[0] - 1)]
            env.step(act)
            if t % every == 0 or t == T - 1:
                trace(t)

        verdicts = grader.verdict()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for ep in r_eps:
            e = slots[ep]
            ok = bool(verdicts[e]["success"]) and not entry_success[e]
            metas[ep]["replay"] = {"success": ok, "score": verdicts[e].get("score"),
                                   "steps": int(metas[ep]["steps"]), "checked_at": now}
            (ep / "meta.json").write_text(json.dumps(metas[ep], indent=2) + "\n")
            results[str(ep)] = ok
        n_ok = sum(results[str(ep)] for ep in r_eps)
        print(f"[replay-check {batch_dir.name}] rollout {rollout}: {n_ok}/{len(r_eps)} "
              "reproduce their success", flush=True)
    return results
