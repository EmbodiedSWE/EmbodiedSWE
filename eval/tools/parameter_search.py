"""Parameter search — tune the constants in your maneuver by PARALLEL evaluation + CMA-ES.

    from parameter_search import search

    # rollout drives ALL envs at once: p["NAME"] is an array with one value per env
    def rollout(env, p):
        for t in range(240):
            action = build_batched_action(env, dz=p["PRESS_DZ"], yaw=p["YAW_STEP_DEG"])
            env.step(action)                      # action: (num_envs, action_dim)

    def objective(env):                           # per-env scores, LOWER IS BETTER
        return goal_error_per_env(env)            # (num_envs,) tensor/array

    out = search(PRESET,                               # your task's preset (see task.md);
                 rollout, objective,                   # the tool builds the search env:
                                                       # 512 PARALLEL ENVS BY DEFAULT
                 space={"PRESS_DZ": (0.001, 0.02), "YAW_STEP_DEG": (2, 25)},
                 seed_values={"PRESS_DZ": 0.006, "YAW_STEP_DEG": 8},
                 start_state="/workspace/.checkpoints/n3.pt",   # anchor: a saved node
                 generations=8, repeats=1)

    # or pass an env you already built — then your env and your width are used as-is:
    out = search(my_env, rollout, objective, space={...})

One generation evaluates the whole population, in passes of `num_envs`: every env is reset
to the same anchor state (re-offset to its own grid cell), runs its own candidate's values,
and the end states are scored per env. A wide env (the tool builds 512 by default) does a
generation in one batched rollout; a 1-env session evaluates candidates one at a time with
the same rollout and objective. Width is purely a speed choice; setup time is printed.

The optimizer is CMA-ES (rank-weighted recombination, cumulative step-size adaptation,
rank-mu + rank-one covariance): these objectives are the correlated kind — a grasp offset
trades off against an approach angle — which plain random/grid search models not at all.
`seed_values` (the constants you use now) are always evaluated verbatim as one candidate of
the first generation, so the result is never worse than the incumbent.

Writing the objective decides whether the search helps or misleads:
  * LOWER is better, per env, and it must score the WHOLE goal of the maneuver — a lift that
    must also stay upright and keep hold scores all three, or the best "lift" found is a
    tilted one that dropped the part.
  * Cap credit at the physically achievable value and penalise ruined outcomes.
  * Put `float('inf')`/`nan` in an env's slot when that candidate must never be chosen.

Robustness variant: `randomize` jitters named objects' start poses per env, e.g.
`{"part_0": {"pos": 0.005, "yaw": 0.2}}` — values are SIGMAS (meters/radians), sized from the
task's real start variation — so the winning constants hold up under start variation instead
of overfitting one exact pose. Jitter needs `repeats` > 1, so each candidate is averaged over
several different starts rather than scored on one.

Never block on a search: put it in a script file and hand it to `launch(path)` — the tool
runs it detached on the freest GPU and `status()` reports generations done / best so far /
the final result, instantly, from a machine-written progress file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

# Self-declaration, read by eval/tools/__init__.py::discover(). A plain dict on purpose: this
# file is also installed standalone on the agent's PYTHONPATH, where a relative import of the
# tools package would fail.
TOOL = {
    "name": "parameter_search",
    "description": (
        "CMA-ES over named constants with PARALLEL population evaluation: every env carries "
        "one candidate, a generation is one batched lockstep rollout, scores adapt the "
        "distribution — offsets, depths, angles and timings measured, not guessed."
    ),
    "exports": ("search", "launch", "status"),
    "skill": "cosigen-parameter-optimize",
    "prompt_doc": "tools/parameter_search.md",
    "requires_features": ("set_states",),
}


# ----------------------- state utilities (ported verbatim from the original) ---------------
def expand_state_tree(tree, n: int):
    """Deep-copy a get_states() tree captured for ONE env, repeated to n envs (dim0 1->n)."""
    import copy

    import torch

    if isinstance(tree, dict):
        return {k: expand_state_tree(v, n) for k, v in tree.items()}
    if torch.is_tensor(tree):
        assert tree.shape[0] == 1, f"snapshot tensor dim0 != 1 ({tuple(tree.shape)})"
        return tree.repeat(n, *([1] * (tree.dim() - 1))).clone()
    if isinstance(tree, list):
        # per-env list state: repeat the single-env row so scene.set_state can index one
        # entry per env (was an IndexError with n>1).
        assert len(tree) == 1, f"snapshot list len != 1 ({len(tree)})"
        return [copy.deepcopy(tree[0]) for _ in range(n)]
    return tree


def _walk_root_states(tree):
    """Yield every tensor stored under a `root_state` key, at any depth. Root states are
    identified BY NAME — the state dict's own convention, the same one `jitter_scene` and
    checkpoint_tree's state diff use — never by tensor width, which corrupts any other
    tensor that happens to be 13 wide and silently skips a pose that is not."""
    import torch

    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == "root_state" and torch.is_tensor(v):
                yield v
            else:
                yield from _walk_root_states(v)


def shift_root_states(tree, delta) -> None:
    """Add per-env origin deltas (n,3) to the POSITION columns (0:3) of every `root_state`
    tensor in an expanded state tree, in place.

    Root states are WORLD-frame and env replicas live on an origin grid: broadcasting env 0's
    state verbatim parks every replica's objects in env 0's cell, so each replica's arm is
    tens of meters from its objects and every candidate but one is born broken."""
    for t in _walk_root_states(tree):
        if t.dim() >= 2 and t.shape[-1] >= 3 and t.dtype.is_floating_point:
            d = delta.view(delta.shape[0], *([1] * (t.dim() - 2)), 3).to(t.device)
            t[..., 0:3] += d


def zero_root_velocities(tree) -> None:
    """Zero the velocity columns (7:13, Isaac Lab's pos3+quat4+vel6 layout) of every
    `root_state` tensor in a state tree, in place. Restore hygiene: a state captured near
    motion carries residual velocities, and writing it into many envs at once seeds
    simultaneous jitter that puts PhysX in a pathological slow state."""
    for t in _walk_root_states(tree):
        if t.dim() >= 2 and t.shape[-1] >= 13 and t.dtype.is_floating_point:
            t[..., 7:13] = 0.0


def jitter_scene(state_n: dict, randomize: dict | None, gen) -> None:
    """Declarative start-state jitter, in place: {"part_0": {"pos": 0.01, "yaw": 0.3}} —
    per named object, "pos" is a gaussian xy sigma (m) and "yaw" a gaussian sigma (rad).
    An optional "z" is a UNIFORM upward lift in [0, z) meters — one-sided on purpose, so
    jitter never spawns an object below its support surface."""
    import torch

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

    def _apply(root, cfg: dict) -> None:
        s_pos, s_yaw = float(cfg.get("pos", 0.0)), float(cfg.get("yaw", 0.0))
        s_z = float(cfg.get("z", 0.0))
        if s_pos > 0:
            root[:, :2] += torch.randn(n, 2, generator=gen, device=root.device) * s_pos
        if s_z > 0:
            root[:, 2] += torch.rand(n, generator=gen, device=root.device) * s_z
        if s_yaw > 0:
            ang = torch.randn(n, generator=gen, device=root.device) * s_yaw
            half = ang * 0.5
            qz = torch.zeros(n, 4, device=root.device)
            qz[:, 0], qz[:, 3] = torch.cos(half), torch.sin(half)
            w1, x1, y1, z1 = qz[:, 0], qz[:, 1], qz[:, 2], qz[:, 3]
            w2, x2, y2, z2 = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
            root[:, 3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            root[:, 4] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            root[:, 5] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            root[:, 6] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    for name, cfg in randomize.items():
        for key, sub in scene.items():
            if key != name:
                continue
            root = sub.get("root_state") if isinstance(sub, dict) else None
            if torch.is_tensor(root):
                _apply(root, cfg)


# ----------------------- search space (verbatim) -------------------------------------------
class Space:
    """The search box. `{'NAME': (lo, hi)}` searches a float; a third element 'int' or 'log'
    makes it integral or log-scaled; `{'NAME': ('choices', (45, 90, 135))}` searches a fixed
    set. Everything is searched in a normalized [0,1] cube."""

    def __init__(self, space: dict[str, Any], fix: dict[str, Any] | None = None):
        import numpy as np

        if not space:
            raise ValueError("space is empty: give at least one parameter with bounds")
        self.fix = dict(fix or {})
        self.names: list[str] = []
        self.kind: list[str] = []
        self.choices: dict[str, list] = {}
        lo_list: list[float] = []
        hi_list: list[float] = []
        for name, spec in space.items():
            if name in self.fix:
                continue
            if str(spec[0]) == "choices":
                values = list(spec[1])
                if len(values) < 2:
                    raise ValueError(f"{name}: give at least two choices")
                self.names.append(name)
                self.choices[name] = values
                lo_list.append(0.0)
                hi_list.append(float(len(values) - 1))
                self.kind.append("choices")
                continue
            lo, hi = float(spec[0]), float(spec[1])
            kind = str(spec[2]) if len(spec) > 2 else "float"
            if not hi > lo:
                raise ValueError(f"{name}: upper bound must exceed lower ({lo}, {hi})")
            if kind == "log" and lo <= 0:
                raise ValueError(f"{name}: log-scale bounds must be positive")
            self.names.append(name)
            lo_list.append(float(np.log(lo)) if kind == "log" else lo)
            hi_list.append(float(np.log(hi)) if kind == "log" else hi)
            self.kind.append(kind)
        self.lo = np.asarray(lo_list, dtype=np.float64)
        self.hi = np.asarray(hi_list, dtype=np.float64)
        self.dim = len(self.names)

    def denorm(self, u):
        import numpy as np

        x = self.lo + np.clip(u, 0.0, 1.0) * (self.hi - self.lo)
        out = x.copy()
        for j, kind in enumerate(self.kind):
            if kind == "log":
                out[:, j] = np.exp(x[:, j])
            elif kind in ("int", "choices"):
                out[:, j] = np.round(x[:, j])
        return out

    def norm_values(self, values: dict):
        import numpy as np

        u = np.zeros(self.dim)
        for j, (name, kind) in enumerate(zip(self.names, self.kind)):
            if name not in values:
                return None
            v = values[name]
            if kind == "choices":
                vals = self.choices[name]
                idx = vals.index(v) if v in vals else int(
                    np.argmin([abs(float(c) - float(v)) for c in vals]))
                x = float(idx)
            elif kind == "log":
                if float(v) <= 0:
                    return None
                x = float(np.log(float(v)))
            else:
                x = float(v)
            span = self.hi[j] - self.lo[j]
            u[j] = float(np.clip((x - self.lo[j]) / span, 0.0, 1.0)) if span > 0 else 0.5
        return u

    def as_dict(self, row) -> dict:
        import numpy as np

        d: dict[str, Any] = {}
        for n, v, k in zip(self.names, row, self.kind):
            if k == "choices":
                vals = self.choices[n]
                d[n] = vals[int(np.clip(round(float(v)), 0, len(vals) - 1))]
            elif k == "int":
                d[n] = int(v)
            else:
                d[n] = float(v)
        d.update(self.fix)
        return d


# ----------------------- proposal distribution: CMA-ES (verbatim) --------------------------
class CMAES:
    """CMA-ES over the normalized cube (Hansen's formulation). Samples are clipped into the
    box and `tell` learns from the CLIPPED points; the sigma floor keeps integer / choice
    axes from collapsing between two levels."""

    def __init__(self, dim: int, rng, popsize: int, sigma0: float = 0.3, sigma_floor: float = 0.02):
        import numpy as np

        self.dim = int(dim)
        self.rng = rng
        self.lam = max(4, int(popsize))
        self.sigma = float(sigma0)
        self.sigma_floor = float(sigma_floor)
        self.mean = np.full(self.dim, 0.5)

        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)

        n = self.dim
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.gen = 0
        self._eigen_gen = 0

    def _update_eigen(self) -> None:
        import numpy as np

        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        vals, vecs = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(vals, 1e-20))
        self.B = vecs
        self._eigen_gen = self.gen

    def ask(self, n: int):
        import numpy as np

        # BUG FIX (2026-08-01): the lazy-eigen schedule `lam // (10*dim)` deferred the first
        # decomposition to generation ~17 at pop 512 / dim 3, so every 5-8 generation search
        # ever run sampled from B=D=identity — the C matrix was learned and NEVER used
        # (isotropic sampling throughout; only CSA sigma adaptation was live). Hansen's lazy
        # rule exists to amortize an O(n^3) cost that is microseconds at our dims: refresh
        # whenever C changed instead.
        if self.gen != self._eigen_gen:
            self._update_eigen()
        z = self.rng.standard_normal((n, self.dim))
        y = z @ (self.B * self.D).T
        return np.clip(self.mean[None, :] + self.sigma * y, 0.0, 1.0)

    def tell(self, u, scores) -> None:
        import numpy as np

        finite = np.isfinite(scores)
        if finite.sum() < 2:
            return
        u, scores = np.clip(u[finite], 0.0, 1.0), scores[finite]
        order = np.argsort(scores)               # minimization
        mu = min(self.mu, len(order))
        w = self.w[:mu] / self.w[:mu].sum()
        elite = u[order[:mu]]

        old_mean = self.mean.copy()
        self.mean = w @ elite
        if self.sigma <= 0:
            return
        y_w = (self.mean - old_mean) / self.sigma

        C_invsqrt_yw = self.B @ ((self.B.T @ y_w) / self.D)
        mueff = 1.0 / np.sum(w ** 2)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * mueff) * C_invsqrt_yw
        self.gen += 1
        hsig = (np.linalg.norm(self.ps)
                / np.sqrt(1 - (1 - self.cs) ** (2 * self.gen))
                / self.chiN) < (1.4 + 2 / (self.dim + 1))
        self.pc = (1 - self.cc) * self.pc + (
            np.sqrt(self.cc * (2 - self.cc) * mueff) * y_w if hsig else 0.0)

        ys = (elite - old_mean[None, :]) / self.sigma
        rank_mu = (ys * w[:, None]).T @ ys
        c1a = self.c1 * (0.0 if hsig else self.cc * (2 - self.cc))
        self.C = ((1 - self.c1 - self.cmu + c1a) * self.C
                  + self.c1 * np.outer(self.pc, self.pc)
                  + self.cmu * rank_mu)
        self.sigma = float(max(
            self.sigma_floor,
            self.sigma * np.exp((self.cs / self.damps)
                                * (np.linalg.norm(self.ps) / self.chiN - 1))))


def build_wide_env(preset: str, num_envs: int, *, verbose: bool = True):
    """Build a wide env from a preset name (an agent that builds and passes its own env
    decides its own width instead). Setup is timed and printed so overhead stays visible.

    The PhysX GPU collision stack is floored at 2**29: at 512 envs a contact-rich scene
    overflows smaller stacks (allen_bolt asked for ~437 MB against the 2**26 Isaac Lab
    default) and PhysX DROPS CONTACTS instead of failing, silently corrupting every score.
    Merged INTO the binding's own overrides, never lowering an explicit larger value."""
    import time as _t

    t_setup = _t.time()
    import robobench

    robobench.discover()
    from robobench.core.registries import ENVS

    cfg = ENVS.get(preset)()
    sim_ov = dict(cfg.sim_overrides)
    physx_ov = dict(sim_ov.get("physx", {}))
    physx_ov["gpu_collision_stack_size"] = max(
        int(physx_ov.get("gpu_collision_stack_size", 0)), 2 ** 29)
    sim_ov["physx"] = physx_ov
    env = cfg.build(num_envs=int(num_envs), sim_overrides=sim_ov)
    env.reset()
    if verbose:
        print(f"[parameter_search] built the search env ({int(num_envs)} parallel envs) "
              f"in {_t.time() - t_setup:.1f}s", flush=True)
    return env


# ----------------------- the search --------------------------------------------------------
def _slice_env0(tree):
    """Env 0's rows of a get_states() tree (dim0 -> 1), for anchoring from a saved state."""
    import copy

    import torch

    if isinstance(tree, dict):
        return {k: _slice_env0(v) for k, v in tree.items()}
    if torch.is_tensor(tree):
        return tree[0:1].clone()
    if isinstance(tree, list):
        return [copy.deepcopy(tree[0])]
    return tree


def search(
    env,
    rollout: Callable[[Any, dict], Any],
    objective: Callable[[Any], Any],
    space: dict[str, Any],
    *,
    num_envs: int = 512,
    start_state=None,
    seed_values: dict[str, Any] | None = None,
    generations: int = 8,
    popsize: int | None = None,
    repeats: int = 1,
    randomize: dict | None = None,
    fix: dict[str, Any] | None = None,
    budget_s: float = 3600.0,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Tune `space` for `rollout`, scored per env by `objective` (lower is better).

    env               a PRESET NAME (the "suite.scene.robot.mode" string your task.md build
                      snippet uses) — the tool then builds its own search env, 512 parallel
                      envs by default — or an env you already built, in which case your env
                      and your width are used as-is.
    num_envs          width of the tool-built env (preset path only). Default 512.
    start_state       the anchor to search from, when the tool builds the env: a states dict
                      or a path to a saved one (e.g. a checkpoint node's
                      /workspace/.checkpoints/n3.pt). Default: the fresh reset state.
    rollout(env, P)   drives ALL envs simultaneously; P maps each name to a torch tensor on
                      env.device, shape (num_envs,) — env i runs candidate i//repeats's
                      values, so P[name] drops straight into per-env GPU math (a scalar count
                      is int(P[name][0])). Non-numeric `choices` values stay a numpy array.
    objective(env)    returns per-env scores, shape (num_envs,) (tensor, array or list).
    popsize           candidates per generation; default num_envs // repeats (full width).
    repeats           envs per candidate; scores are averaged over them (noisy objectives).
    randomize         start-state jitter per object, e.g. {"part_0": {"pos": 0.005, "yaw": 0.2}}.
    budget_s          wall-clock cutoff; the search returns what it has when it trips.

    Returns {"best": {...}, "best_score": float, "history": [...], "evaluations": int,
    "generations_run": int, "truncated": bool}. The world is restored to its pre-search
    states (all envs) before returning.
    """
    import time as _setup_time

    import numpy as np
    import torch

    # Say who else is on the GPU before the search starts, so a slow search is never a
    # mystery. Facts only; what to do about a squatter is the caller's decision.
    if verbose:
        try:
            import subprocess
            me = str(os.getpid())
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            # exact pid-field compare — a prefix match would swallow pid 1234 under pid 123
            others = [l for l in apps.splitlines()
                      if l.strip() and l.split(",")[0].strip() != me]
            if others:
                print(f"[parameter_search] note: {len(others)} other compute process(es) on "
                      f"this GPU — generations will contend with them:\n      "
                      + "\n      ".join(others), flush=True)
        except Exception as exc:  # noqa: BLE001 -- a missing nvidia-smi must not block a search
            print(f"[parameter_search] (gpu process listing unavailable: {exc})", flush=True)

    if isinstance(env, str):
        env = build_wide_env(env, int(num_envs), verbose=verbose)

    sp = Space(space, fix)
    n_env = int(env.num_envs)
    reps = max(1, int(repeats))
    pop = int(popsize) if popsize else max(4, n_env // reps)
    # Candidates are evaluated in PASSES of num_envs: a wide env does the whole population
    # in one batched rollout; a 1-env session runs fully sequentially (the same rollout and
    # objective, just one candidate at a time — handy for smoke-testing a rollout or reusing
    # serial code); anything between takes more passes. One mechanism, no width refusals.
    n_run = pop * reps
    passes = (n_run + n_env - 1) // n_env
    if verbose and passes > 1:
        print(f"[parameter_search] {n_run} evaluations per generation in {passes} passes of "
              f"{n_env} env(s) — a wider env would need fewer passes", flush=True)
    if randomize and reps == 1 and verbose:
        # ADDITION (2026-07-31), not in the original, which had the same hazard silently:
        # with jitter on and one env per candidate, every candidate is scored on a DIFFERENT
        # random start, so the ranking CMA-ES learns from is start-state noise as much as
        # parameter effect. Averaging over repeats is what makes a jittered search comparable.
        print(f"[parameter_search] randomize= is on with repeats=1: every candidate is "
              f"scored on its own single random start, so the ranking mixes start-state "
              f"noise with parameter effect — raise repeats so each candidate is averaged "
              f"over several starts", flush=True)

    rng = np.random.default_rng(int(seed))
    gen_t = torch.Generator(device=env.device)
    gen_t.manual_seed(int(seed))

    # Full pre-search restoration target (every env, exactly as we found it), plus env 0's
    # state as the population anchor each generation replicates.
    pre = env.get_states()
    if start_state is not None:
        if isinstance(start_state, (str, Path)):
            start_state = torch.load(str(start_state), map_location=env.device,
                                     weights_only=False)
        start0 = _slice_env0(start_state)
    else:
        start0 = env.get_states(torch.tensor([0], device=env.device, dtype=torch.long))
    origin_delta = (env.iscene.env_origins - env.iscene.env_origins[0:1]).to(env.device)

    def _dev(a):
        """A searched parameter delivered where the rollout actually computes: a tensor on
        env.device. The rollout builds per-env action/target tensors on the GPU, so a param
        it can drop straight into that math (`origin[:, 2] + P['GRASP_Z']`) is the whole
        point. A CPU numpy array here forces `cuda_tensor + param` through numpy's __array__
        and dies with 'can't convert cuda:0 tensor to numpy' (bulb_c4 hit exactly this on
        2026-08-01)."""
        return torch.as_tensor(a, device=env.device)

    def reset_population() -> None:
        state_n = expand_state_tree(start0, n_env)
        shift_root_states(state_n, origin_delta)
        jitter_scene(state_n, randomize, gen_t)
        zero_root_velocities(state_n)
        env.set_states(state_n)

    opt = CMAES(sp.dim, rng, popsize=pop)
    u_seed = sp.norm_values(seed_values) if seed_values else None
    if u_seed is not None:
        opt.mean = u_seed.copy()
        if verbose:
            print(f"[parameter_search] seeding with the current values: "
                  f"{sp.as_dict(sp.denorm(u_seed[None, :])[0])}", flush=True)

    best: dict[str, Any] = {"best": None, "best_score": float("inf"), "gen": -1}
    seed_score: float | None = None
    history: list[dict] = []
    all_u: list = []
    all_scores: list = []
    all_raw: list = []      # per-ENV scores, kept for the repeats-based noise estimate
    beat_holder: list = [None]   # the live generation heartbeat, stopped in finally
    import time as _time

    t0 = _time.time()
    g = -1
    try:
        for g in range(int(generations)):
            if _time.time() - t0 > budget_s:
                g -= 1
                break
            u = opt.ask(pop)
            if g == 0 and u_seed is not None:
                u[0] = u_seed          # the incumbent runs as candidate 0 of generation 0
            values = sp.denorm(u)      # (pop, dim)
            values_env = np.repeat(values, reps, axis=0)           # (n_run, dim)
            raw = np.empty(n_run, dtype=np.float64)
            # Heartbeat while the generation evaluates: a batched rollout is one long
            # opaque call, and a log that stops moving reads as a hang — callers have
            # killed healthy searches over exactly that silence. Stopped after the passes;
            # the function's finally stops it on every exception path too.
            if verbose:
                beat_holder[0] = _heartbeat(
                    lambda dt, _g=g: f"[parameter_search] gen {_g + 1}/{int(generations)} "
                                     f"evaluating — {dt}s", 60.0)
            for lo in range(0, n_run, n_env):
                hi = min(lo + n_env, n_run)
                chunk = values_env[lo:hi]
                if hi - lo < n_env:    # idle tail envs replay the last candidate; not scored
                    pad = np.repeat(chunk[-1:], n_env - (hi - lo), axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)
                # Each param arrives as an env.device tensor, shape (n_env,) — one value per
                # env (env i runs candidate i//reps). Numeric only; a non-numeric `choices`
                # value (e.g. string labels) can't be a tensor, so it stays a numpy array.
                P = {}
                for j, name in enumerate(sp.names):
                    col = chunk[:, j]
                    if sp.kind[j] == "choices":
                        # BUG FIX (2026-08-01): denorm yields the choice INDEX; the rollout
                        # was handed that index while the reported best translated it to the
                        # choice VALUE — the searched behavior and the reported constants
                        # disagreed. Deliver the value on both surfaces.
                        vals = np.asarray(sp.choices[name])
                        picked = vals[np.clip(col.astype(np.int64), 0, len(vals) - 1)]
                        P[name] = _dev(picked) if picked.dtype.kind in "iuf" else picked
                    elif sp.kind[j] == "int":
                        # int64 tensor (agents use these as counts: int(P[name][0]))
                        P[name] = _dev(col.astype(np.int64))
                    else:
                        P[name] = _dev(col.astype(np.float32))
                for name, v in (sp.fix or {}).items():
                    fv = np.full(n_env, v)
                    if fv.dtype.kind == "f":
                        fv = fv.astype(np.float32)
                    P[name] = _dev(fv) if fv.dtype.kind in "iuf" else fv

                reset_population()
                try:
                    rollout(env, P)
                except Exception as exc:
                    # One rollout serves every candidate in the pass, so an exception here
                    # is a program error, not a bad parameter: surface it instead of
                    # returning a meaningless "search".
                    raise RuntimeError(
                        f"the rollout raised — this is a program error, not a parameter "
                        f"problem: {exc!r}") from exc

                scored = objective(env)
                if torch.is_tensor(scored):
                    scored = scored.detach().cpu().numpy()
                part = np.asarray(scored, dtype=np.float64).reshape(-1)
                if part.shape[0] < min(n_env, hi - lo):
                    raise ValueError(
                        f"objective returned {part.shape[0]} scores for {n_env} envs — it "
                        f"must return one score per env (shape ({n_env},))")
                raw[lo:hi] = part[:hi - lo]
            if beat_holder[0] is not None:
                beat_holder[0].set()
                beat_holder[0] = None
            scores = np.array([np.nanmean(raw[j * reps:(j + 1) * reps]) if reps > 1
                               else raw[j] for j in range(pop)])
            # ADDITION (2026-08-01): the incumbent's own score. It always ran as candidate 0
            # of generation 0, but was never reported — so no log could say what the search
            # actually GAINED over the values the agent already had. Report it once and
            # return it, so gain-vs-incumbent is measurable from every search log.
            if g == 0 and u_seed is not None:
                seed_score = float(scores[0])
                if verbose:
                    print(f"[parameter_search] incumbent (seed_values) scored "
                          f"{seed_score:.5g} — the search must beat this to be worth "
                          f"adopting", flush=True)
            opt.tell(u, scores)
            all_u.append(u)
            all_scores.append(scores)
            all_raw.append(raw)

            finite = scores[np.isfinite(scores)]
            j = int(np.nanargmin(scores)) if len(finite) else 0
            if len(finite) and scores[j] < best["best_score"]:
                best = {"best": sp.as_dict(values[j]), "best_score": float(scores[j]), "gen": g}
            rec = {"generation": g + 1,
                   "best_in_gen": float(np.nanmin(scores)) if len(finite) else None,
                   "median": float(np.nanmedian(scores)) if len(finite) else None,
                   "best_score": best["best_score"],
                   "sigma": float(opt.sigma),
                   "t": round(_time.time() - t0, 1)}
            history.append(rec)
            if verbose:
                print(f"[parameter_search] gen {g + 1}/{generations}: "
                      f"best_in_gen={rec['best_in_gen']} median={rec['median']} | "
                      f"best so far {best['best_score']:.5g} at {best['best']} | "
                      f"sigma {opt.sigma:.3f} | {rec['t']}s", flush=True)
            # Machine-readable progress for launch()/status(): one JSON line per generation,
            # so a poller never has to parse the human log. Written only when launch() (or
            # anything else) points PSEARCH_PROGRESS at a file.
            progress_path = os.environ.get("PSEARCH_PROGRESS")
            if progress_path:
                try:
                    with open(progress_path, "a") as fh:
                        fh.write(json.dumps({**rec, "best": best["best"],
                                             "seed_score": seed_score},
                                            default=float) + "\n")
                except Exception as exc:  # noqa: BLE001 -- progress must never kill a search
                    print(f"[parameter_search] progress write failed: {exc!r}", flush=True)
    finally:
        if beat_holder[0] is not None:   # never leave a ticker printing after a raise
            beat_holder[0].set()
        env.set_states(pre)  # every env back exactly as the search found it

    # Per-parameter sensitivity: |correlation| between a parameter's sampled values and the
    # scores. Near-zero = the objective barely responds to it.
    #
    # DELIBERATE FIX vs the original (2026-07-31), which pooled every evaluation from every
    # generation into one correlation. That pooling confounds CROSS-GENERATION DRIFT: the
    # CMA-ES mean moves between generations while the score distribution may not, and a chance
    # alignment of the two drifts manufactures a correlation. Measured: a parameter with
    # mathematically ZERO effect on the score reported sensitivity 0.87 that way over 4
    # generations. Correlating WITHIN each generation and averaging removes the confound (same
    # setup then reports < 0.17, matching the controlled expectation).
    sens: dict[str, float] = {}
    if all_u:
        for j, name in enumerate(sp.names):
            per_gen = []
            for u_g, s_g in zip(all_u, all_scores):
                okm = np.isfinite(s_g)
                if okm.sum() < 3:
                    continue
                col, sc = u_g[okm, j], s_g[okm]
                if col.std() < 1e-9 or sc.std() < 1e-9:
                    per_gen.append(0.0)
                else:
                    per_gen.append(abs(float(np.corrcoef(col, sc)[0, 1])))
            sens[name] = float(np.mean(per_gen)) if per_gen else 0.0
    # Red-team findings from the first agent-driven search (2026-07-26): a best value sitting
    # on a declared bound, and a budget-truncated search, were both silent. Say them out loud
    # so the agent knows to widen the range / raise budget_s.
    # Samples are clipped into the box, so an optimizer pushing past a bound leaves the best
    # sitting EXACTLY on it (0/1 normalized, up to float round-trip) — that, and only that,
    # is evidence the true optimum may lie outside. No invented edge margins.
    at_bounds: dict[str, str] = {}
    if best["best"]:
        u_best = sp.norm_values(best["best"])
        if u_best is not None:
            for j, name in enumerate(sp.names):
                if sp.kind[j] == "choices":
                    continue
                if u_best[j] <= 1e-9:
                    at_bounds[name] = "low"
                elif u_best[j] >= 1.0 - 1e-9:
                    at_bounds[name] = "high"
    truncated = len(history) < int(generations)
    # ADDITION (2026-07-31), not in the original: a signal-to-noise test on the result. The
    # gain the search achieved is compared against the score scatter WITHIN a generation; when
    # the gain is smaller than the noise it is fitting, the "best" values are arbitrary. This
    # is the case the original returned silently — and the one that produced a 1.5%
    # "improvement" on a saturated objective in a live run.
    # The gain the search achieved, and whether it beats what pure luck would produce.
    #
    # Noise can ONLY be measured across repeats of the SAME candidate. The scatter between
    # different candidates in a generation is mostly SIGNAL (that is the parameter effect the
    # search exists to find), so using it as a noise estimate flags every successful search as
    # noise — measured while building this: it fired on 20/20 genuinely converging runs.
    # With repeats == 1 there is no noise estimate at all, and the honest output says so
    # instead of asserting a verdict.
    gain = noise = luck = float("nan")
    if history:
        first_med = history[0]["median"]
        if first_med is not None and np.isfinite(best["best_score"]):
            gain = abs(first_med - best["best_score"])
    if reps >= 2 and all_raw:
        # within-candidate spread across its repeats, pooled over every candidate
        per_cand = [float(np.nanstd(r[j * reps:(j + 1) * reps]))
                    for r in all_raw for j in range(pop)
                    if np.isfinite(r[j * reps:(j + 1) * reps]).sum() >= 2]
        if per_cand:
            noise = float(np.nanmean(per_cand))
            # the minimum of `pop` noisy draws sits ~sqrt(2 ln pop) sigma below their median,
            # so that much "gain" appears even with no parameter effect whatsoever
            luck = noise * float(np.sqrt(2.0 * np.log(max(pop, 2))))
    if verbose and np.isfinite(gain) and np.isfinite(luck) and gain <= luck:
        print(f"[parameter_search] WARNING: the gain is inside the noise — total improvement "
              f"{gain:.4g}, while repeat-to-repeat noise ({noise:.4g}) would hand a search of "
              f"this width ~{luck:.4g} by luck alone, so these 'best' values are not "
              f"distinguishable from chance. Root causes to check, in order: the objective no "
              f"longer discriminates where the candidates land (e.g. a threshold every "
              f"candidate clears — score a continuous quality instead); the searched "
              f"constants do not drive this outcome (see sensitivity); the outcome is "
              f"noise-dominated (raise repeats). Sensitivity per parameter: {sens}",
              flush=True)
    elif verbose and reps == 1 and np.isfinite(gain):
        print(f"[parameter_search] gain {gain:.4g} (median of generation 1 -> best). With "
              f"repeats=1 the noise in your objective is unmeasured, so whether this gain is "
              f"real cannot be told from this run: if it looks small, re-run with repeats>1 "
              f"(which both measures the noise and averages it out). Sensitivity per "
              f"parameter: {sens}", flush=True)
    if verbose and at_bounds:
        print(f"[parameter_search] note: best value at the {at_bounds} edge of your search "
              f"range — the true optimum may lie beyond it; consider widening", flush=True)
    if verbose and truncated:
        print(f"[parameter_search] note: stopped by budget_s after {len(history)}/"
              f"{int(generations)} generations", flush=True)

    out = {"best": best["best"], "best_score": best["best_score"], "best_gen": best["gen"],
           "history": history, "sensitivity": sens, "at_bounds": at_bounds,
           "evaluations": n_run * len(history), "generations_run": len(history),
           "generations_planned": int(generations), "truncated": truncated,
           "seconds": round(_time.time() - t0, 1),
           "seed_score": seed_score,
           "gain_vs_seed": (float(seed_score - best["best_score"])
                            if seed_score is not None and np.isfinite(seed_score)
                            and np.isfinite(best["best_score"]) else None)}
    if verbose:
        vs = (f" | incumbent {seed_score:.5g}, gain {out['gain_vs_seed']:.5g}"
              if out["gain_vs_seed"] is not None else "")
        print(f"[parameter_search] done: {out['evaluations']} evaluations in "
              f"{len(history)} generation(s), best {out['best_score']:.5g} at {out['best']}"
              f"{vs} | sensitivity {sens}", flush=True)
    result_path = os.environ.get("PSEARCH_RESULT")
    if result_path:
        try:
            Path(result_path).write_text(json.dumps(out, indent=2, default=float) + "\n")
        except Exception as exc:  # noqa: BLE001 -- result file must never kill a search
            print(f"[parameter_search] result write failed: {exc!r}", flush=True)
    return out


# ----------------------- background execution: launch() / status() -------------------------
# A search must never block the agent, and the agent must never have to hand-roll nohup, GPU
# placement and polling — hand-rolling is exactly where live searches died (foreground
# timeouts killed four launches; a process-cleanup sweep killed a fifth). launch() owns the
# process: detached session (no foreground timeout can exist, an untargeted pkill sweep does
# not reap it), pinned to the freest GPU, progress machine-readable, pid on record.
_PSEARCH_DIR = Path(os.environ.get("PSEARCH_DIR", "/workspace/.psearch"))


def _heartbeat(message: Callable[[int], str], interval: float = 60.0):
    """Print `message(elapsed_seconds)` every `interval` seconds until the returned event
    is set. A long batched evaluation is one opaque call; without a pulse the log looks
    hung and callers kill healthy work."""
    import threading
    import time as _t

    stop = threading.Event()
    t0 = _t.time()

    def _loop():
        while not stop.wait(interval):
            print(message(int(_t.time() - t0)), flush=True)

    threading.Thread(target=_loop, daemon=True).start()
    return stop


def _freest_gpu() -> str | None:
    """Index of the GPU with the most free memory — ties broken AWAY from device 0.

    Everything interactive defaults to cuda:0, so on a multi-GPU box device 0 only LOOKS
    free between the caller's own tool calls; a search placed there gets buried the moment
    the session starts its next sim (2026-08-01: two live searches picked 0 on the free-at-
    that-instant reading, crawled under the session's Isaac, and were killed as stuck, while
    the one search that landed on device 1 ran to convergence). Preferring the higher index
    on near-ties is what actually reserves the second GPU for background work."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        rows = [(r[0].strip(), int(r[1].strip()))
                for r in (line.split(",") for line in out.splitlines() if "," in line)]
        if not rows:
            return None
        # freest wins; within 10% of the freest, the HIGHEST index wins (device 0 last)
        best_free = max(free for _, free in rows)
        contenders = [(idx, free) for idx, free in rows if free >= 0.9 * best_free]
        return max(contenders, key=lambda r: int(r[0]))[0]
    except Exception as exc:  # noqa: BLE001 -- placement is best-effort, never a blocker
        print(f"[parameter_search] gpu pick failed ({exc!r}); leaving placement to CUDA",
              flush=True)
        return None


def _registry_path() -> Path:
    _PSEARCH_DIR.mkdir(parents=True, exist_ok=True)
    return _PSEARCH_DIR / "registry.json"


def _registry_read() -> list[dict]:
    path = _registry_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 -- a corrupt registry is reported, not fatal
        print(f"[parameter_search] registry unreadable ({exc!r}); starting fresh", flush=True)
        return []


def launch(script: str | Path, *, gpu: str | int | None = None, name: str = "") -> dict:
    """Run a search SCRIPT in the background, owned by the tool. Returns immediately.

    The script is any python file that calls `search(...)` (your rollout + objective + the
    call, exactly as you would run it in the foreground). launch() runs it in its own
    detached session — it cannot hit a foreground command timeout and survives shell
    cleanup sweeps that are not aimed at its pid — pinned via CUDA_VISIBLE_DEVICES to the
    GPU with the most free memory (override with `gpu=`). Poll it with `status()`; it costs
    nothing and returns instantly.

        h = launch("/workspace/search_press.py")
        ...keep working...
        status(h["id"])     # -> generations done, best so far, or the final result
    """
    import subprocess
    import sys

    src = Path(script).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"launch() needs a script file; {src} does not exist")
    run_id = name or f"{src.stem}_{time.strftime('%H%M%S')}"
    run_dir = _PSEARCH_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PSEARCH_PROGRESS"] = str(run_dir / "progress.jsonl")
    env["PSEARCH_RESULT"] = str(run_dir / "result.json")
    picked = str(gpu) if gpu is not None else (_freest_gpu() or env.get("CUDA_VISIBLE_DEVICES", "0"))
    env["CUDA_VISIBLE_DEVICES"] = picked

    log = run_dir / "log"
    with log.open("w") as fh:
        proc = subprocess.Popen([sys.executable, "-u", str(src)], stdout=fh, stderr=fh,
                                start_new_session=True, env=env, cwd=str(src.parent))

    entry = {"id": run_id, "pid": proc.pid, "script": str(src), "dir": str(run_dir),
             "log": str(log), "progress": env["PSEARCH_PROGRESS"],
             "result": env["PSEARCH_RESULT"], "gpu": picked, "started": time.time()}
    entries = _registry_read()
    entries.append(entry)
    _registry_path().write_text(json.dumps(entries, indent=2) + "\n")
    print(f"[parameter_search] launched '{run_id}' (pid {proc.pid}, gpu {picked}); "
          f"log {log} — poll with status('{run_id}')", flush=True)
    return entry


def status(run_id: str | dict | None = None) -> dict:
    """Progress of a launched search, instantly: generations done, best so far, final result.

    No argument = the most recently launched. Prints one human line and returns the full
    picture: {"running": bool, "generations_done": int, "last": <latest generation record>,
    "result": <search() result dict when finished>, plus the launch entry}. A search that
    stopped without a result crashed — the "log" path holds why.
    """
    if isinstance(run_id, dict):
        run_id = run_id.get("id")
    entries = _registry_read()
    if not entries:
        print("[parameter_search] nothing launched yet", flush=True)
        return {"running": False, "error": "nothing launched"}
    entry = next((e for e in reversed(entries) if e["id"] == run_id), None) \
        if run_id else entries[-1]
    if entry is None:
        known = [e["id"] for e in entries]
        print(f"[parameter_search] no launch '{run_id}'; known: {known}", flush=True)
        return {"running": False, "error": f"unknown id {run_id}", "known": known}

    # /proc/<pid> also exists for a ZOMBIE (finished but unreaped because the launching
    # process is still alive) — state 'Z' in /proc/<pid>/stat means finished, not running.
    running = False
    try:
        stat = Path(f"/proc/{entry['pid']}/stat").read_text()
        running = stat.rsplit(")", 1)[-1].split()[0] != "Z"
    except OSError:
        running = False
    gens: list[dict] = []
    progress = Path(entry["progress"])
    if progress.exists():
        for line in progress.read_text().splitlines():
            try:
                gens.append(json.loads(line))
            except Exception as exc:  # noqa: BLE001 -- a torn line is reported, not fatal
                print(f"[parameter_search] torn progress line ignored ({exc!r})", flush=True)
    result = None
    result_path = Path(entry["result"])
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[parameter_search] result unreadable: {exc!r}", flush=True)

    out = {**entry, "running": running, "generations_done": len(gens),
           "last": gens[-1] if gens else None, "result": result}
    if result is not None:
        gain = result.get("gain_vs_seed")
        bs = result.get("best_score")
        bs_txt = f"{bs:.5g}" if isinstance(bs, (int, float)) else str(bs)
        print(f"[parameter_search] '{entry['id']}' FINISHED: best {bs_txt} "
              f"at {result['best']}"
              + (f", gain vs incumbent {gain:.5g}" if gain is not None else ""), flush=True)
    elif running and gens:
        last = gens[-1]
        print(f"[parameter_search] '{entry['id']}' running: gen {last['generation']} done, "
              f"best so far {last['best_score']:.5g} at {last['best']} ({last['t']}s)",
              flush=True)
    elif running:
        print(f"[parameter_search] '{entry['id']}' running, nothing scored yet — the log "
              f"shows how far it is (log: {entry['log']})", flush=True)
    else:
        print(f"[parameter_search] '{entry['id']}' STOPPED WITHOUT A RESULT — it crashed or "
              f"was killed; read {entry['log']}", flush=True)
    return out
