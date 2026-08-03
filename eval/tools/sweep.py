"""Sweep — run the candidates you name SIMULTANEOUSLY, each from the same saved state, and
rank them. A candidate is (function, params): your function, with its own control flow —
the same function with different parameters, different functions, any mix — all stepping
one simulator together, each on its own slice of the parallel envs.

THE PROTOCOL — your candidate is a generator:

    def my_approach(env, ids, **params):
        ...

  * `ids` is the tensor of env indices the tool assigned to you (your group — your
    repeats). With N candidates on a 512-wide env, each gets ~512/N envs.
  * Write your normal control loop. Wherever you would have called `env.step(act)`,
    instead `yield act` — action rows for YOUR envs only, shape (len(ids), action_dim),
    on `env.device`.
  * Each `yield` advances the WHOLE world by one step; when your function resumes, state
    is fresh — read it sliced by your group (e.g. `scene.part.data.root_pos_w[ids]`).
  * When your function returns, your group is done: its envs receive zero actions (hold,
    for delta-style controllers) until every candidate finishes.
  * NEVER: call `env.step`/`env.reset`/`env.set_states` (the tool owns stepping and
    state), touch envs outside `ids`, or set SIM-GLOBAL state (controller gains, torque
    limits, physics settings) — a global write applies to every candidate's envs
    instantly and silently corrupts the comparison. To compare global settings, run
    separate sweeps.

E.g. (an example — the functions, names and numbers are all yours; *_rows are action
builders you write, each returning (len(ids), action_dim) rows for your group):

    from sweep import sweep

    def one_shot(env, ids, gain):                # one phase, early exit
        for k in range(200):
            yield drive_rows(env, ids, gain)     # world steps once per yield
            if done_mask(env)[ids].all():
                break

    def two_stage(env, ids, gain):               # structurally different: two phases
        for k in range(80):
            yield align_rows(env, ids)
        while not settled(env)[ids].all():       # wait, then finish
            yield hold_rows(env, ids)
        for k in range(150):
            yield finish_rows(env, ids, gain)

    def objective(env):                          # per-env scores, LOWER IS BETTER
        return goal_error(env)                   # shape (num_envs,)

    out = sweep(PRESET,                          # your task's preset (task.md names it)
                candidates=[(one_shot,  {"gain": 0.5}),
                            (one_shot,  {"gain": 2.0}),
                            (two_stage, {"gain": 0.5})],
                objective=objective,
                start_state="/workspace/.checkpoints/n2.pt",   # anchor: a saved tree node
                randomize={"part_0": {"pos": 0.005}})          # optional start jitter
    # -> {"ranking": [best candidate's index first, ...], "best": (name, params),
    #     "scores": [...], "stds": [...]} — scores stay in your candidate order

All candidates run in the wall-clock of the LONGEST one, each scored over its whole group
(jittered starts when `randomize` is set) — a distribution per approach, not one
anecdote. `parameter_search.search` remains the tool when you want values INVENTED and
tuned; sweep runs what you name.

Sweeps run like searches: from a script, via `parameter_search.launch()` — never in the
foreground.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

# Self-declaration, read by eval/tools/__init__.py::discover(). A plain dict on purpose: this
# file is also installed standalone on the agent's PYTHONPATH, where a relative import of the
# tools package would fail. Imports engine pieces from parameter_search, so a condition that
# grants sweep must grant parameter_search too.
TOOL = {
    "name": "sweep",
    "description": (
        "Run an EXPLICIT list of (function, params) candidates — approaches you wrote, "
        "settings you chose — SIMULTANEOUSLY in one lockstep pass, each on its own slice "
        "of the parallel envs from the same saved state, and rank them: sweep runs what "
        "you name, search invents values."
    ),
    "exports": ("sweep",),
    "prompt_doc": "tools/sweep.md",
    "requires_features": ("set_states",),
}


def sweep(
    env,
    candidates: list,
    objective: Callable[[Any], Any],
    *,
    randomize: dict | None = None,
    start_state=None,
    num_envs: int = 512,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run every `(function, params)` candidate simultaneously from the anchor state —
    each on its own group of envs — and rank them.

    Returns {"candidates": [(name, params), ...], "scores", "stds", "ranking", "best",
    "best_score", "group_sizes", "steps", "evaluations", "seconds"}; scores follow the
    order candidates were given. The world is restored to its pre-sweep states before
    returning.

    env          a live env, or a preset name for the tool to build one (512 wide default)
    candidates   [(generator_fn, params_dict), ...] — called as fn(env, ids, **params);
                 must be a GENERATOR yielding (len(ids), action_dim) action rows (see the
                 module docstring for the full protocol)
    objective    per-env scores, shape (num_envs,), LOWER is better
    randomize    start-state jitter per named object, same semantics as parameter_search
    start_state  anchor: a get_states() tree or a saved checkpoint .pt path (env 0's state
                 is replicated into every env)
    """
    import numpy as np
    import torch

    from parameter_search import (_slice_env0, build_wide_env, expand_state_tree,
                                  jitter_scene, shift_root_states, zero_root_velocities)

    if not candidates:
        raise ValueError("candidates is empty — give [(function, params), ...]")
    for i, c in enumerate(candidates):
        if not (isinstance(c, (tuple, list)) and len(c) == 2
                and callable(c[0]) and isinstance(c[1], dict)):
            raise ValueError(f"candidate {i} must be (function, params_dict), got {c!r}")

    if isinstance(env, str):
        env = build_wide_env(env, int(num_envs), verbose=verbose)

    n_env = int(env.num_envs)
    n_cand = len(candidates)
    if n_cand > n_env:
        raise ValueError(f"{n_cand} candidates but only {n_env} env(s) — every candidate "
                         f"needs at least one env")
    names = [f"{i}:{fn.__name__}" for i, (fn, _) in enumerate(candidates)]

    # Contiguous groups; the remainder spreads one extra env over the first groups.
    base, rem = divmod(n_env, n_cand)
    ids_list, start = [], 0
    for i in range(n_cand):
        size = base + (1 if i < rem else 0)
        ids_list.append(torch.arange(start, start + size, device=env.device,
                                     dtype=torch.long))
        start += size
    if verbose:
        print(f"[sweep] {n_cand} candidate(s) running simultaneously on {n_env} env(s): "
              f"groups of {[len(g) for g in ids_list]}"
              f"{' with start jitter' if randomize else ''}", flush=True)

    gen_t = torch.Generator(device=env.device)
    gen_t.manual_seed(int(seed))

    pre = env.get_states()
    if start_state is not None:
        if isinstance(start_state, (str, Path)):
            start_state = torch.load(str(start_state), map_location=env.device,
                                     weights_only=False)
        start0 = _slice_env0(start_state)
    else:
        start0 = env.get_states(torch.tensor([0], device=env.device, dtype=torch.long))
    origin_delta = (env.iscene.env_origins - env.iscene.env_origins[0:1]).to(env.device)

    state_n = expand_state_tree(start0, n_env)
    shift_root_states(state_n, origin_delta)
    jitter_scene(state_n, randomize, gen_t)
    zero_root_velocities(state_n)

    act_dim_seen = [None]  # set by the first yield; every later yield must match it

    def check_block(i: int, block, step: int):
        """One candidate's yielded action rows, validated loudly at the exact yield."""
        try:
            block = torch.as_tensor(block, device=env.device)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"candidate {names[i]} yielded something that is not an "
                             f"action array at step {step}: {exc!r}") from exc
        want = len(ids_list[i])
        if block.dim() != 2 or block.shape[0] != want:
            raise ValueError(
                f"candidate {names[i]} yielded shape {tuple(block.shape)} at step {step} "
                f"— it must yield rows for ITS group only: ({want}, action_dim)")
        if act_dim_seen[0] is None:
            act_dim_seen[0] = int(block.shape[1])
        elif block.shape[1] != act_dim_seen[0]:
            raise ValueError(
                f"candidate {names[i]} yielded action_dim {block.shape[1]} at step {step} "
                f"but earlier yields used {act_dim_seen[0]} — all candidates step ONE "
                f"simulator and must share action_dim")
        return block

    def resume(i: int, g, step: int):
        """Advance one candidate to its next yield; None means it returned (finished).
        Only next(g) is inside the catch, so tool-raised ValueErrors stay unwrapped."""
        try:
            raw = next(g)
        except StopIteration:
            return None
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"candidate {names[i]} raised at step {step} — a program "
                               f"error in your function, not a verdict on the approach: "
                               f"{exc!r}") from exc
        return check_block(i, raw, step)

    t0 = time.time()
    steps = 0
    try:
        env.set_states(state_n)
        gens, blocks, done = [], [None] * n_cand, [False] * n_cand
        for i, (fn, params) in enumerate(candidates):
            g = fn(env, ids_list[i], **params)
            if not inspect.isgenerator(g):
                raise ValueError(
                    f"candidate {names[i]} is not a generator — its function must "
                    f"`yield act` where it would have called env.step(act)")
            gens.append(g)
        for i, g in enumerate(gens):     # prime: run each to its first yield (no step yet)
            blocks[i] = resume(i, g, 0)
            done[i] = blocks[i] is None
        act_dim = act_dim_seen[0]
        last_beat = time.time()
        while not all(done):
            # Heartbeat: long candidates make the loop run for a while, and a silent log
            # reads as a hang — callers kill healthy work over exactly that silence.
            if verbose and time.time() - last_beat >= 60.0:
                last_beat = time.time()
                print(f"[sweep] running — {steps} step(s), {sum(done)}/{n_cand} "
                      f"candidate(s) finished, {int(time.time() - t0)}s", flush=True)
            full = torch.zeros(n_env, act_dim, device=env.device,
                               dtype=next(b.dtype for b in blocks if b is not None))
            for i in range(n_cand):
                if not done[i]:
                    full[ids_list[i]] = blocks[i]
            env.step(full)
            steps += 1
            for i, g in enumerate(gens):
                if done[i]:
                    continue
                blocks[i] = resume(i, g, steps)
                if blocks[i] is None:
                    done[i] = True
                    if verbose:
                        print(f"[sweep] {names[i]} finished after {steps} step(s)",
                              flush=True)

        scored = objective(env)
        if torch.is_tensor(scored):
            scored = scored.detach().cpu().numpy()
        per_env = np.asarray(scored, dtype=np.float64).reshape(-1)
        if per_env.shape[0] < n_env:
            raise ValueError(f"objective returned {per_env.shape[0]} scores for {n_env} "
                             f"envs — it must return one score per env")
    finally:
        env.set_states(pre)  # every env back exactly as the sweep found it

    scores = [float(np.nanmean(per_env[ids.cpu().numpy()])) for ids in ids_list]
    stds = [float(np.nanstd(per_env[ids.cpu().numpy()])) for ids in ids_list]
    order = sorted(range(n_cand), key=lambda i: (not np.isfinite(scores[i]), scores[i]))
    best_i = order[0]
    best_ok = np.isfinite(scores[best_i])
    out = {
        "candidates": [(names[i], candidates[i][1]) for i in range(n_cand)],
        "scores": scores,
        "stds": stds,
        "ranking": order,
        "best": (names[best_i], candidates[best_i][1]) if best_ok else None,
        "best_score": scores[best_i] if best_ok else None,
        "group_sizes": [len(g) for g in ids_list],
        "steps": steps,
        "evaluations": n_env,
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"[sweep] done: {n_cand} candidate(s) in {steps} lockstep step(s), "
              f"{out['seconds']}s, ranked best to worst:", flush=True)
        for rank, i in enumerate(order, start=1):
            print(f"  {rank}. score {scores[i]:.5g} (std {stds[i]:.4g} over "
                  f"{len(ids_list[i])} envs)  {names[i]} {candidates[i][1]}", flush=True)
    result_path = os.environ.get("PSEARCH_RESULT")
    if result_path:
        try:
            Path(result_path).write_text(json.dumps(out, indent=2, default=float) + "\n")
        except Exception as exc:  # noqa: BLE001 -- result file must never kill a sweep
            print(f"[sweep] result write failed: {exc!r}", flush=True)
    return out
