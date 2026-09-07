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

import hashlib
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
    "exports": ("search", "tune", "launch", "status"),
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


def _walk_state_tensors(tree, path=()):
    """Yield ``(path, tensor)`` for tensors in a public get_states() tree."""
    import torch

    if torch.is_tensor(tree):
        yield path, tree
    elif isinstance(tree, dict):
        for key, value in tree.items():
            yield from _walk_state_tensors(value, path + (str(key),))
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            yield from _walk_state_tensors(value, path + (str(i),))


def _walk_named_root_states(tree, path=()):
    """Yield floating 13-wide rigid-body states from the public scene state tree.

    Robobench scenes do not use one leaf name: real states include
    ``items.stapler``, ``bodies.bread_m``, ``ram_0`` and ``box_root`` in addition
    to nested ``root_state`` leaves. Restricting the walk to ``state["scene"]``
    makes the Isaac 13-value pose/velocity layout unambiguous without mistaking
    robot controller vectors for bodies.
    """
    import torch

    if not path and isinstance(tree, dict) and isinstance(tree.get("scene"), dict):
        tree = tree["scene"]
        path = ("scene",)
    if not isinstance(tree, dict):
        return
    for key, value in tree.items():
        child_path = path + (str(key),)
        if (torch.is_tensor(value) and value.dim() >= 2
                and value.shape[-1] == 13 and value.dtype.is_floating_point):
            yield child_path, value
        else:
            yield from _walk_named_root_states(value, child_path)


def _clone_state_tree(tree):
    """Clone a state tree, including per-env list state."""
    import copy

    import torch

    if torch.is_tensor(tree):
        return tree.clone()
    if isinstance(tree, dict):
        return {key: _clone_state_tree(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_clone_state_tree(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_clone_state_tree(value) for value in tree)
    return copy.deepcopy(tree)


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


def _rows_to_parameters(sp: Space, rows, n_env: int, device):
    """Convert denormalized search rows to the rollout's batched parameter mapping."""
    import numpy as np
    import torch

    rows = np.asarray(rows)
    if rows.shape != (n_env, sp.dim):
        raise ValueError(f"parameter rows have shape {rows.shape}, expected {(n_env, sp.dim)}")

    def _dev(value):
        return torch.as_tensor(value, device=device)

    params: dict[str, Any] = {}
    for j, name in enumerate(sp.names):
        col = rows[:, j]
        if sp.kind[j] == "choices":
            vals = np.asarray(sp.choices[name])
            picked = vals[np.clip(col.astype(np.int64), 0, len(vals) - 1)]
            params[name] = _dev(picked) if picked.dtype.kind in "iuf" else picked
        elif sp.kind[j] == "int":
            params[name] = _dev(col.astype(np.int64))
        else:
            params[name] = _dev(col.astype(np.float32))
    for name, value in (sp.fix or {}).items():
        fixed = np.full(n_env, value)
        if fixed.dtype.kind == "f":
            fixed = fixed.astype(np.float32)
        params[name] = _dev(fixed) if fixed.dtype.kind in "iuf" else fixed
    return params


def _values_to_parameters(sp: Space, values: dict[str, Any], n_env: int, device):
    """Repeat one parameter dictionary over every rollout environment."""
    import numpy as np

    normalized = sp.norm_values(values)
    if normalized is None:
        missing = [name for name in sp.names if name not in values]
        raise ValueError(f"parameter values are incomplete; missing searched values {missing}")
    row = sp.denorm(normalized[None, :])
    return _rows_to_parameters(sp, np.repeat(row, n_env, axis=0), n_env, device)


def _numpy_rows(value, n_env: int, label: str):
    """Convert a per-env metric to a two-dimensional float64 array."""
    import numpy as np
    import torch

    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim == 0:
        if n_env != 1:
            raise ValueError(f"{label} returned a scalar for {n_env} envs; return one value per env")
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr[:, None]
    else:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] < n_env:
        raise ValueError(f"{label} returned {arr.shape[0]} rows for {n_env} envs — it must "
                         f"return one value per env")
    try:
        return np.asarray(arr[:n_env], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric per-env values: {exc}") from exc


def _boolean_rows(value, n_env: int, label: str):
    """Return a strict per-env boolean vector and a mask of non-finite entries."""
    import numpy as np

    rows = _numpy_rows(value, n_env, label)
    if rows.shape[1] != 1:
        raise ValueError(f"{label} returned shape {rows.shape}; expected one boolean per env")
    raw = rows[:, 0]
    nonfinite = ~np.isfinite(raw)
    finite = raw[~nonfinite]
    if finite.size and not np.all((finite == 0.0) | (finite == 1.0)):
        bad = finite[(finite != 0.0) & (finite != 1.0)][:3].tolist()
        raise ValueError(f"{label} must be boolean (or numeric 0/1), got values such as {bad}")
    return np.where(nonfinite, False, raw != 0.0), nonfinite


def _quality_columns(value, n_env: int, label: str = "quality"):
    """Normalize one or several lower-is-better quality metrics, preserving dict order."""
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{label} is empty")
        columns: dict[str, Any] = {}
        for name, metric in value.items():
            rows = _numpy_rows(metric, n_env, f"{label}[{name!r}]")
            if rows.shape[1] != 1:
                raise ValueError(f"{label}[{name!r}] returned {rows.shape[1]} columns; each "
                                 f"named quality metric must return one value per env")
            columns[str(name)] = rows[:, 0]
        return columns

    rows = _numpy_rows(value, n_env, label)
    if rows.shape[1] == 1:
        return {"quality": rows[:, 0]}
    return {f"quality_{j}": rows[:, j] for j in range(rows.shape[1])}


def _parse_objective_output(scored, n_env: int):
    """Parse legacy scalar objectives or guarded structured objective output."""
    import numpy as np

    if not isinstance(scored, dict) or "quality" not in scored:
        rows = _numpy_rows(scored, n_env, "objective")
        if rows.shape[1] != 1:
            raise ValueError(f"objective returned {rows.shape[1]} values per env; legacy "
                             f"search() objectives must return shape ({n_env},)")
        return {"structured": False, "quality_names": ["objective"],
                "quality": rows, "goal": None, "invalid": None,
                "generic_invalid": None, "explicit_invalid": None}

    if "goal" not in scored:
        raise ValueError("structured objective output must contain a per-env 'goal' mask")
    quality = _quality_columns(scored["quality"], n_env, "objective['quality']")
    names = list(quality)
    quality_rows = np.column_stack([quality[name] for name in names])
    goal, goal_nonfinite = _boolean_rows(scored["goal"], n_env, "objective['goal']")
    invalid_value = scored.get("invalid", np.zeros(n_env, dtype=bool))
    invalid, invalid_nonfinite = _boolean_rows(
        invalid_value, n_env, "objective['invalid']")
    invalid = invalid | invalid_nonfinite | goal_nonfinite | ~np.isfinite(quality_rows).all(axis=1)

    def _optional_mask(name):
        if name not in scored:
            return np.zeros(n_env, dtype=bool)
        mask, nonfinite = _boolean_rows(scored[name], n_env, f"objective[{name!r}]")
        return mask | nonfinite

    return {
        "structured": True,
        "quality_names": names,
        "quality": quality_rows,
        "goal": goal,
        "invalid": invalid,
        "generic_invalid": _optional_mask("generic_invalid"),
        "explicit_invalid": _optional_mask("explicit_invalid"),
    }


def _lexicographic_order(invalid, goal_rate, quality):
    """Best-first candidate order: validity, goal rate, then lower quality columns."""
    import numpy as np

    invalid = np.asarray(invalid, dtype=bool).reshape(-1)
    goal_rate = np.asarray(goal_rate, dtype=np.float64).reshape(-1)
    quality = np.asarray(quality, dtype=np.float64)
    if quality.ndim == 1:
        quality = quality[:, None]
    if len(invalid) != len(goal_rate) or quality.shape[0] != len(invalid):
        raise ValueError("lexicographic metric lengths do not match")
    safe_quality = np.where(np.isfinite(quality), quality, np.inf)
    return sorted(
        range(len(invalid)),
        key=lambda i: (bool(invalid[i]), -float(goal_rate[i]),
                       *(float(v) for v in safe_quality[i])),
    )


def _generic_physical_invalid(anchor_state, live_state, n_env: int):
    """Generic, deliberately generous physical-invalid checks over public state trees.

    The check catches non-finite state, tracked ``root_state`` bodies more than five metres
    outside their anchor bounding box, bodies more than two metres below the anchor floor,
    and absolute linear/angular root velocity above 100 SI units.
    """
    import numpy as np
    import torch

    masks = {
        "nonfinite": torch.zeros(n_env, dtype=torch.bool),
        "outside_anchor": torch.zeros(n_env, dtype=torch.bool),
        "below_anchor_floor": torch.zeros(n_env, dtype=torch.bool),
        "extreme_velocity": torch.zeros(n_env, dtype=torch.bool),
    }

    for tree in (anchor_state, live_state):
        for _, tensor in _walk_state_tensors(tree):
            if (tensor.dtype.is_floating_point and tensor.dim() >= 1
                    and tensor.shape[0] == n_env):
                bad = ~torch.isfinite(tensor).reshape(n_env, -1).all(dim=1)
                masks["nonfinite"] |= bad.detach().cpu()

    anchors = dict(_walk_named_root_states(anchor_state))
    live_roots = dict(_walk_named_root_states(live_state))
    anchor_positions, live_positions, live_velocities = [], [], []
    for path, anchor in anchors.items():
        live = live_roots.get(path)
        if (live is None or anchor.shape[0] != n_env or live.shape[0] != n_env
                or anchor.shape[:-1] != live.shape[:-1]):
            continue
        anchor_flat = anchor[..., :3].reshape(n_env, -1, 3).detach().cpu()
        live_flat = live[..., :3].reshape(n_env, -1, 3).detach().cpu()
        velocity_flat = live[..., 7:13].reshape(n_env, -1, 6).detach().cpu()
        anchor_positions.append(anchor_flat)
        live_positions.append(live_flat)
        live_velocities.append(velocity_flat)

    if anchor_positions:
        anchor_pos = torch.cat(anchor_positions, dim=1)
        live_pos = torch.cat(live_positions, dim=1)
        live_vel = torch.cat(live_velocities, dim=1)
        finite_anchor = torch.where(torch.isfinite(anchor_pos), anchor_pos,
                                    torch.zeros_like(anchor_pos))
        lo = finite_anchor.amin(dim=1)
        hi = finite_anchor.amax(dim=1)
        outside = ((live_pos < (lo[:, None, :] - 5.0))
                   | (live_pos > (hi[:, None, :] + 5.0))).any(dim=(1, 2))
        below = (live_pos[..., 2] < (lo[:, None, 2] - 2.0)).any(dim=1)
        extreme = (live_vel.abs() > 100.0).any(dim=(1, 2))
        masks["outside_anchor"] |= outside
        masks["below_anchor_floor"] |= below
        masks["extreme_velocity"] |= extreme

    result = {name: mask.numpy().astype(bool, copy=False) for name, mask in masks.items()}
    result["combined"] = np.logical_or.reduce(list(result.values()))
    return result


def _call_goal(env, goal):
    if isinstance(goal, str):
        if not goal or goal.startswith("_"):
            raise ValueError("goal method name must name a public scene method")
        method = getattr(env.scene, goal, None)
        if not callable(method):
            raise AttributeError(f"scene has no public callable goal method {goal!r}")
        return method()
    if not callable(goal):
        raise TypeError("goal must be a callable or a public scene method name")
    return goal(env)


def _call_quality(env, quality):
    if callable(quality):
        return quality(env)
    if isinstance(quality, dict):
        if not quality:
            raise ValueError("quality mapping is empty")
        evaluated = {}
        for name, metric in quality.items():
            if not callable(metric):
                raise TypeError(f"quality[{name!r}] must be callable")
            evaluated[name] = metric(env)
        return evaluated
    raise TypeError("quality must be callable or a mapping of named callables")


class _TuneObjective:
    """Agent-metric-only objective used by tune(); no hidden grader is consulted."""

    def __init__(self, goal, quality, invalid):
        self.goal = goal
        self.quality = quality
        self.invalid = invalid
        self._anchor_state = None
        self.last_components: dict[str, Any] | None = None

    def _set_anchor_state(self, anchor_state) -> None:
        self._anchor_state = _clone_state_tree(anchor_state)

    def __call__(self, env):
        import numpy as np

        if self._anchor_state is None:
            raise RuntimeError("guarded objective has no anchor state for physical checks")
        n_env = int(env.num_envs)
        goal_raw = _call_goal(env, self.goal)
        quality_raw = _call_quality(env, self.quality)
        qualities = _quality_columns(quality_raw, n_env)
        goal, goal_nonfinite = _boolean_rows(goal_raw, n_env, "goal")

        if self.invalid is None:
            explicit = np.zeros(n_env, dtype=bool)
            explicit_nonfinite = np.zeros(n_env, dtype=bool)
        else:
            invalid_raw = self.invalid(env) if callable(self.invalid) else self.invalid
            explicit, explicit_nonfinite = _boolean_rows(
                invalid_raw, n_env, "explicit invalid mask")

        physical = _generic_physical_invalid(
            self._anchor_state, env.get_states(), n_env)
        quality_nonfinite = ~np.isfinite(np.column_stack(
            [qualities[name] for name in qualities])).all(axis=1)
        merged = (physical["combined"] | explicit | explicit_nonfinite
                  | goal_nonfinite | quality_nonfinite)
        self.last_components = {
            "generic_invalid": physical["combined"],
            "explicit_invalid": explicit | explicit_nonfinite,
            "invalid": merged,
            "goal": goal,
            "quality": qualities,
            "physical_reasons": physical,
        }
        return {
            "quality": qualities,
            "goal": goal,
            "invalid": merged,
            "generic_invalid": physical["combined"],
            "explicit_invalid": explicit | explicit_nonfinite,
        }


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
    deadline: float | None = None,
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
                      tune() also uses a structured goal/quality/invalid form internally.
    popsize           candidates per generation; default num_envs // repeats (full width).
    repeats           envs per candidate; scores are averaged over them (noisy objectives).
    randomize         start-state jitter per object, e.g. {"part_0": {"pos": 0.005, "yaw": 0.2}}.
    budget_s          cooperative wall-clock cutoff, checked between rollout passes.
    deadline          optional absolute Unix timestamp cutoff, also checked cooperatively.

    Returns {"best": {...}, "best_score": float, "history": [...], "evaluations": int,
    "generations_run": int, "truncated": bool}. The world is restored to its pre-search
    states (all envs) before returning.
    """
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

    def reset_population() -> None:
        state_n = expand_state_tree(start0, n_env)
        shift_root_states(state_n, origin_delta)
        jitter_scene(state_n, randomize, gen_t)
        zero_root_velocities(state_n)
        set_anchor = getattr(objective, "_set_anchor_state", None)
        if callable(set_anchor):
            set_anchor(state_n)
        env.set_states(state_n)

    opt = CMAES(sp.dim, rng, popsize=pop)
    u_seed = sp.norm_values(seed_values) if seed_values else None
    if u_seed is not None:
        opt.mean = u_seed.copy()
        if verbose:
            print(f"[parameter_search] seeding with the current values: "
                  f"{sp.as_dict(sp.denorm(u_seed[None, :])[0])}", flush=True)

    best: dict[str, Any] = {
        "best": None,
        "best_score": float("inf"),
        "best_quality": None,
        "goal_rate": None,
        "invalid_fraction": None,
        "key": None,
        "gen": -1,
    }
    seed_score: float | None = None
    seed_quality: dict[str, float] | None = None
    seed_goal_rate: float | None = None
    seed_invalid_fraction: float | None = None
    history: list[dict] = []
    all_u: list = []
    all_scores: list = []
    all_raw: list = []      # per-ENV scores, kept for the repeats-based noise estimate
    beat_holder: list = [None]   # the live generation heartbeat, stopped in finally
    import time as _time

    t0 = _time.time()
    stop_reason: str | None = None
    evaluations_done = 0
    structured_mode: bool | None = None
    quality_names: list[str] = []

    def cutoff_reason() -> str | None:
        now = _time.time()
        if deadline is not None and now >= float(deadline):
            return "deadline"
        if budget_s is not None and now - t0 >= float(budget_s):
            return "budget_s"
        return None

    try:
        for g in range(int(generations)):
            stop_reason = cutoff_reason()
            if stop_reason is not None:
                break
            u = opt.ask(pop)
            if g == 0 and u_seed is not None:
                u[0] = u_seed          # the incumbent runs as candidate 0 of generation 0
            values = sp.denorm(u)      # (pop, dim)
            values_env = np.repeat(values, reps, axis=0)           # (n_run, dim)
            raw_quality = None
            raw_goal = raw_invalid = None
            raw_generic_invalid = raw_explicit_invalid = None
            generation_complete = True
            # Heartbeat while the generation evaluates: a batched rollout is one long
            # opaque call, and a log that stops moving reads as a hang — callers have
            # killed healthy searches over exactly that silence. Stopped after the passes;
            # the function's finally stops it on every exception path too.
            if verbose:
                beat_holder[0] = _heartbeat(
                    lambda dt, _g=g: f"[parameter_search] gen {_g + 1}/{int(generations)} "
                                     f"evaluating — {dt}s", 60.0)
            for lo in range(0, n_run, n_env):
                stop_reason = cutoff_reason()
                if stop_reason is not None:
                    generation_complete = False
                    break
                hi = min(lo + n_env, n_run)
                chunk = values_env[lo:hi]
                if hi - lo < n_env:    # idle tail envs replay the last candidate; not scored
                    pad = np.repeat(chunk[-1:], n_env - (hi - lo), axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)
                # Each param arrives as an env.device tensor, shape (n_env,) — one value per
                # env (env i runs candidate i//reps). Numeric only; a non-numeric `choices`
                # value (e.g. string labels) can't be a tensor, so it stays a numpy array.
                P = _rows_to_parameters(sp, chunk, n_env, env.device)

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
                parsed = _parse_objective_output(scored, n_env)
                if structured_mode is None:
                    structured_mode = bool(parsed["structured"])
                    quality_names = list(parsed["quality_names"])
                elif structured_mode != bool(parsed["structured"]):
                    raise ValueError("objective changed between scalar and structured output")
                elif quality_names != list(parsed["quality_names"]):
                    raise ValueError("structured objective changed quality fields between passes: "
                                     f"{quality_names} -> {parsed['quality_names']}")

                if raw_quality is None:
                    raw_quality = np.empty((n_run, len(quality_names)), dtype=np.float64)
                    if structured_mode:
                        raw_goal = np.empty(n_run, dtype=bool)
                        raw_invalid = np.empty(n_run, dtype=bool)
                        raw_generic_invalid = np.empty(n_run, dtype=bool)
                        raw_explicit_invalid = np.empty(n_run, dtype=bool)
                take = hi - lo
                raw_quality[lo:hi] = parsed["quality"][:take]
                if structured_mode:
                    raw_goal[lo:hi] = parsed["goal"][:take]
                    raw_invalid[lo:hi] = parsed["invalid"][:take]
                    raw_generic_invalid[lo:hi] = parsed["generic_invalid"][:take]
                    raw_explicit_invalid[lo:hi] = parsed["explicit_invalid"][:take]
                evaluations_done += take
            if beat_holder[0] is not None:
                beat_holder[0].set()
                beat_holder[0] = None
            if not generation_complete:
                if verbose:
                    print(f"[parameter_search] cooperative {stop_reason} cutoff reached "
                          f"between rollout passes; preserving {len(history)} complete "
                          f"generation(s) and {evaluations_done} completed evaluations",
                          flush=True)
                break

            def _finite_mean(rows):
                finite_rows = rows[np.isfinite(rows)]
                return float(np.mean(finite_rows)) if finite_rows.size else float("inf")

            quality_scores = np.empty((pop, len(quality_names)), dtype=np.float64)
            for candidate_i in range(pop):
                candidate_rows = raw_quality[candidate_i * reps:(candidate_i + 1) * reps]
                for quality_i in range(len(quality_names)):
                    column = candidate_rows[:, quality_i]
                    quality_scores[candidate_i, quality_i] = (
                        _finite_mean(column) if structured_mode
                        else float(np.nanmean(column))
                    )
            scores = quality_scores[:, 0]

            if structured_mode:
                raw_invalid |= ~np.isfinite(raw_quality).all(axis=1)
                invalid_fraction = np.array([
                    float(np.mean(raw_invalid[j * reps:(j + 1) * reps]))
                    for j in range(pop)
                ])
                candidate_invalid = invalid_fraction > 0.0
                goal_rate = np.array([
                    float(np.mean(raw_goal[j * reps:(j + 1) * reps]))
                    for j in range(pop)
                ])
                generic_invalid_fraction = float(np.mean(raw_generic_invalid))
                explicit_invalid_fraction = float(np.mean(raw_explicit_invalid))
                order = _lexicographic_order(candidate_invalid, goal_rate, quality_scores)
                optimizer_scores = np.empty(pop, dtype=np.float64)
                optimizer_scores[order] = np.arange(pop, dtype=np.float64)
                j = int(order[0])
                candidate_key = (
                    bool(candidate_invalid[j]),
                    -float(goal_rate[j]),
                    *(float(v) if np.isfinite(v) else float("inf")
                      for v in quality_scores[j]),
                )
            else:
                invalid_fraction = goal_rate = candidate_invalid = None
                generic_invalid_fraction = explicit_invalid_fraction = None
                optimizer_scores = scores
                finite_indices = np.flatnonzero(np.isfinite(scores))
                j = int(finite_indices[np.argmin(scores[finite_indices])]) \
                    if finite_indices.size else 0
                candidate_key = (float(scores[j]),) if finite_indices.size else None

            # The incumbent's own metrics. It always runs as candidate 0 in generation 0.
            if g == 0 and u_seed is not None:
                seed_score = float(scores[0])
                seed_quality = {
                    name: float(quality_scores[0, k])
                    for k, name in enumerate(quality_names)
                }
                if structured_mode:
                    seed_goal_rate = float(goal_rate[0])
                    seed_invalid_fraction = float(invalid_fraction[0])
                if verbose:
                    guarded = (
                        f", goal_rate={seed_goal_rate:.1%}, "
                        f"invalid_fraction={seed_invalid_fraction:.1%}"
                        if structured_mode else ""
                    )
                    print(f"[parameter_search] incumbent (seed_values) quality "
                          f"{seed_score:.5g}{guarded}", flush=True)
            opt.tell(u, optimizer_scores)
            all_u.append(u)
            all_scores.append(optimizer_scores)
            all_raw.append(raw_quality[:, 0].copy())

            if candidate_key is not None and (
                    best["key"] is None or candidate_key < best["key"]):
                best = {
                    "best": sp.as_dict(values[j]),
                    "best_score": float(scores[j]),
                    "best_quality": {
                        name: float(quality_scores[j, k])
                        for k, name in enumerate(quality_names)
                    },
                    "goal_rate": float(goal_rate[j]) if structured_mode else None,
                    "invalid_fraction": (
                        float(invalid_fraction[j]) if structured_mode else None),
                    "key": candidate_key,
                    "gen": g,
                }
            finite = scores[np.isfinite(scores)]
            rec = {"generation": g + 1,
                   "best_in_gen": float(scores[j]) if np.isfinite(scores[j]) else None,
                   "median": float(np.median(finite)) if len(finite) else None,
                   "best_score": best["best_score"],
                   "sigma": float(opt.sigma),
                   "t": round(_time.time() - t0, 1)}
            if structured_mode:
                rec.update({
                    "invalid_fraction": float(np.mean(raw_invalid)),
                    "generic_invalid_fraction": generic_invalid_fraction,
                    "explicit_invalid_fraction": explicit_invalid_fraction,
                    "best_in_gen_goal_rate": float(goal_rate[j]),
                    "best_in_gen_invalid_fraction": float(invalid_fraction[j]),
                    "best_goal_rate": best["goal_rate"],
                    "best_invalid_fraction": best["invalid_fraction"],
                })
            history.append(rec)
            if verbose:
                guarded = (
                    f" | invalid_fraction={rec['invalid_fraction']:.1%}, "
                    f"best_goal_rate={rec['best_in_gen_goal_rate']:.1%}"
                    if structured_mode else ""
                )
                print(f"[parameter_search] gen {g + 1}/{generations}: "
                      f"best_in_gen={rec['best_in_gen']} median={rec['median']}{guarded} | "
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
                                             "seed_score": seed_score,
                                             "seed_goal_rate": seed_goal_rate,
                                             "seed_invalid_fraction": seed_invalid_fraction},
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
    # Population spread from the first generation's median to the eventual best, and whether
    # repeat noise is large enough to manufacture that spread by luck. This is deliberately
    # not called "gain": only the signed best-vs-seed comparison is an improvement measure.
    #
    # Noise can ONLY be measured across repeats of the SAME candidate. The scatter between
    # different candidates in a generation is mostly SIGNAL (that is the parameter effect the
    # search exists to find), so using it as a noise estimate flags every successful search as
    # noise — measured while building this: it fired on 20/20 genuinely converging runs.
    # With repeats == 1 there is no noise estimate at all, and the honest output says so
    # instead of asserting a verdict.
    population_spread = noise = luck = float("nan")
    if history:
        first_med = history[0]["median"]
        if first_med is not None and np.isfinite(best["best_score"]):
            population_spread = abs(first_med - best["best_score"])
    if reps >= 2 and all_raw:
        # within-candidate spread across its repeats, pooled over every candidate
        per_cand = [float(np.nanstd(r[j * reps:(j + 1) * reps]))
                    for r in all_raw for j in range(pop)
                    if np.isfinite(r[j * reps:(j + 1) * reps]).sum() >= 2]
        if per_cand:
            noise = float(np.nanmean(per_cand))
            # the minimum of `pop` noisy draws sits ~sqrt(2 ln pop) sigma below their median,
            # so that much spread appears even with no parameter effect whatsoever
            luck = noise * float(np.sqrt(2.0 * np.log(max(pop, 2))))
    if (verbose and np.isfinite(population_spread) and np.isfinite(luck)
            and population_spread <= luck):
        print(f"[parameter_search] WARNING: population spread is inside the noise — spread "
              f"{population_spread:.4g}, while repeat-to-repeat noise ({noise:.4g}) would "
              f"hand a search of this width ~{luck:.4g} by luck alone, so these 'best' values are not "
              f"distinguishable from chance. Root causes to check, in order: the objective no "
              f"longer discriminates where the candidates land (e.g. a threshold every "
              f"candidate clears — score a continuous quality instead); the searched "
              f"constants do not drive this outcome (see sensitivity); the outcome is "
              f"noise-dominated (raise repeats). Sensitivity per parameter: {sens}",
              flush=True)
    elif verbose and reps == 1 and np.isfinite(population_spread):
        print(f"[parameter_search] population_spread {population_spread:.4g} "
              f"(generation-1 median to best). With repeats=1 the objective noise is "
              f"unmeasured, so whether this spread is stable cannot be told from this run: "
              f"if it looks small, re-run with repeats>1 "
              f"(which both measures the noise and averages it out). Sensitivity per "
              f"parameter: {sens}", flush=True)
    if verbose and at_bounds:
        print(f"[parameter_search] note: best value at the {at_bounds} edge of your search "
              f"range — the true optimum may lie beyond it; consider widening", flush=True)
    if verbose and truncated:
        reason = stop_reason or "before all planned generations completed"
        print(f"[parameter_search] note: stopped by {reason} after {len(history)}/"
              f"{int(generations)} complete generations", flush=True)

    improvement_vs_seed = (
        float(seed_score - best["best_score"])
        if seed_score is not None and np.isfinite(seed_score)
        and np.isfinite(best["best_score"]) else None
    )
    out = {"best": best["best"], "best_score": best["best_score"], "best_gen": best["gen"],
           "best_quality": best["best_quality"], "best_goal_rate": best["goal_rate"],
           "best_invalid_fraction": best["invalid_fraction"],
           "history": history, "sensitivity": sens, "at_bounds": at_bounds,
           "evaluations": evaluations_done, "generations_run": len(history),
           "generations_planned": int(generations), "truncated": truncated,
           "stop_reason": stop_reason,
           "seconds": round(_time.time() - t0, 1),
           "seed_score": seed_score, "seed_quality": seed_quality,
           "seed_goal_rate": seed_goal_rate,
           "seed_invalid_fraction": seed_invalid_fraction,
           "population_spread": (float(population_spread)
                                 if np.isfinite(population_spread) else None),
           "improvement_vs_seed": improvement_vs_seed,
           # Compatibility only: new callers should use improvement_vs_seed.
           "gain_vs_seed": improvement_vs_seed,
           "gain_vs_seed_is_compatibility_alias": True,
           "objective_mode": ("guarded"
                              if structured_mode or isinstance(objective, _TuneObjective)
                              else "scalar")}
    if verbose:
        vs = (f" | incumbent {seed_score:.5g}, improvement_vs_seed "
              f"{improvement_vs_seed:.5g}"
              if improvement_vs_seed is not None else "")
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


def _guarded_anchor(env, start0, randomize, seed: int):
    """Build one full-width anchor that can be cloned for paired replay."""
    import torch

    n_env = int(env.num_envs)
    origin_delta = (env.iscene.env_origins - env.iscene.env_origins[0:1]).to(env.device)
    state_n = expand_state_tree(start0, n_env)
    shift_root_states(state_n, origin_delta)
    generator = torch.Generator(device=env.device)
    generator.manual_seed(int(seed))
    jitter_scene(state_n, randomize, generator)
    zero_root_velocities(state_n)
    return state_n


def _guarded_fixed_rollout(
    env,
    rollout,
    sp: Space,
    values: dict[str, Any],
    objective: _TuneObjective,
    anchor_state,
    sample_count: int,
    label: str,
):
    """Run one fixed parameter set and return all sampled guarded metrics."""
    import numpy as np

    env.set_states(_clone_state_tree(anchor_state))
    objective._set_anchor_state(anchor_state)
    params = _values_to_parameters(sp, values, int(env.num_envs), env.device)
    rollout(env, params)
    parsed = _parse_objective_output(objective(env), int(env.num_envs))
    if not parsed["structured"]:
        raise RuntimeError("internal error: tune objective did not return guarded metrics")

    count = min(int(sample_count), int(env.num_envs))
    sample = slice(0, count)
    quality = {
        name: [float(value) for value in parsed["quality"][sample, i]]
        for i, name in enumerate(parsed["quality_names"])
    }
    quality_means = {
        name: float(np.mean(values_per_env))
        for name, values_per_env in quality.items()
    }
    details = objective.last_components or {}
    physical_reasons = details.get("physical_reasons", {})
    reason_fractions = {
        name: float(np.mean(np.asarray(mask, dtype=bool)[sample]))
        for name, mask in physical_reasons.items()
        if name != "combined"
    }
    out = {
        "label": label,
        "instances": count,
        "score": quality_means[parsed["quality_names"][0]],
        "quality": quality_means,
        "quality_per_env": quality,
        "goal_rate": float(np.mean(parsed["goal"][sample])),
        "goal_per_env": [bool(value) for value in parsed["goal"][sample]],
        "invalid_fraction": float(np.mean(parsed["invalid"][sample])),
        "invalid_per_env": [bool(value) for value in parsed["invalid"][sample]],
        "generic_invalid_fraction": float(np.mean(parsed["generic_invalid"][sample])),
        "explicit_invalid_fraction": float(np.mean(parsed["explicit_invalid"][sample])),
        "physical_invalid_reason_fractions": reason_fractions,
    }
    return out


def _guarded_verdict(winner, incumbent, at_bounds, search_invalid_fraction=0.0):
    """Return a conservative replay verdict without weighted penalties."""
    import math

    if winner is None:
        return "INCONCLUSIVE", "the search produced no winner to replay"
    if float(search_invalid_fraction or 0.0) > 0.0:
        return "REJECT", "the winning candidate was physically invalid during search"
    if winner["invalid_fraction"] > 0.0:
        return "REJECT", "the winning candidate was invalid in paired replay"
    if not math.isfinite(float(winner["score"])):
        return "REJECT", "the winning candidate's replay quality was non-finite"
    if incumbent is None:
        return "INCONCLUSIVE", "the seed could not be replayed for a paired comparison"

    winner_key = (
        winner["invalid_fraction"] > 0.0,
        -float(winner["goal_rate"]),
        *(float(value) for value in winner["quality"].values()),
    )
    incumbent_key = (
        incumbent["invalid_fraction"] > 0.0,
        -float(incumbent["goal_rate"]),
        *(float(value) for value in incumbent["quality"].values()),
    )
    if not winner_key < incumbent_key:
        return "REJECT", "paired replay did not rank the winner ahead of seed_values"
    if winner["goal_rate"] < 1.0:
        if at_bounds:
            return "WIDEN", "the better candidate hit a search bound but did not satisfy every replay"
        return "INCONCLUSIVE", "the winner improved lexicographically but did not satisfy every replay"
    if at_bounds:
        return "WIDEN", "the replayed winner was better but lies on a search bound"
    return "ADOPT", "the winner was valid, satisfied every replay, and beat seed_values"


def tune(
    env,
    rollout: Callable[[Any, dict], Any],
    goal,
    quality,
    space: dict[str, Any],
    *,
    invalid=None,
    num_envs: int = 512,
    start_state=None,
    seed_values: dict[str, Any] | None = None,
    generations: int = 8,
    popsize: int | None = None,
    repeats: int = 1,
    randomize: dict | None = None,
    fix: dict[str, Any] | None = None,
    budget_s: float = 3600.0,
    deadline: float | None = None,
    seed: int = 0,
    validation_instances: int = 3,
    verbose: bool = True,
) -> dict:
    """Guarded parameter tuning with lexicographic validity/goal/quality ranking.

    ``goal`` is either ``goal(env)`` or a public no-argument scene method name. ``quality``
    is a lower-is-better callable, a callable returning an ordered mapping, or an ordered
    mapping of names to callables. ``invalid`` may be a per-env mask or ``invalid(env)``.
    The seed is smoke-tested before CMA-ES, and seed/winner are replayed from the same anchor.
    This function reports a verdict; it never edits solution code.
    """
    import numpy as np
    import torch

    if seed_values is None:
        raise ValueError("tune() requires seed_values so every search has a validated incumbent")
    if not isinstance(seed_values, dict):
        raise TypeError("seed_values must be a parameter dictionary")
    if int(validation_instances) not in (2, 3):
        raise ValueError("validation_instances must be 2 or 3")
    if deadline is not None and time.time() >= float(deadline):
        raise TimeoutError("tune() deadline has already passed")

    if isinstance(env, str):
        env = build_wide_env(env, int(num_envs), verbose=verbose)
    n_env = int(env.num_envs)
    sample_count = min(int(validation_instances), n_env)
    if sample_count < 2 and verbose:
        print("[parameter_search] only one env is available; seed/replay validation uses "
              "one instance instead of the requested 2 or 3", flush=True)

    sp = Space(space, fix)
    normalized_seed = sp.norm_values(seed_values)
    if normalized_seed is None:
        missing = [name for name in sp.names if name not in seed_values]
        raise ValueError(f"seed_values is missing searched parameters: {missing}")
    canonical_seed = sp.as_dict(sp.denorm(normalized_seed[None, :])[0])
    coerced = {}
    for name, kind in zip(sp.names, sp.kind):
        original = seed_values[name]
        canonical = canonical_seed[name]
        if kind == "choices":
            same = original in sp.choices[name] and original == canonical
        else:
            try:
                same = bool(np.isfinite(float(original))
                            and np.isclose(float(original), float(canonical),
                                           rtol=1e-12, atol=1e-12))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"seed_values[{name!r}] is not numeric: {exc}") from exc
        if not same:
            coerced[name] = {"given": original, "in_space": canonical}
    if coerced:
        raise ValueError(
            "tune() must smoke-test seed_values verbatim, but these values are outside or "
            f"incompatible with the declared space: {coerced}")
    guarded_objective = _TuneObjective(goal, quality, invalid)

    pre = env.get_states()
    if start_state is not None:
        if isinstance(start_state, (str, Path)):
            start_state = torch.load(str(start_state), map_location=env.device,
                                     weights_only=False)
        start0 = _slice_env0(start_state)
    else:
        start0 = env.get_states(torch.tensor([0], device=env.device, dtype=torch.long))

    smoke_anchor = _guarded_anchor(env, start0, randomize, int(seed))
    try:
        smoke = _guarded_fixed_rollout(
            env, rollout, sp, canonical_seed, guarded_objective, smoke_anchor,
            sample_count, "seed_smoke")
    finally:
        env.set_states(pre)
    if verbose:
        print(f"[parameter_search] seed smoke: {smoke['instances']} instance(s), "
              f"quality={smoke['score']:.5g}, goal_rate={smoke['goal_rate']:.1%}, "
              f"invalid_fraction={smoke['invalid_fraction']:.1%}", flush=True)
    if smoke["invalid_fraction"] > 0.0:
        raise RuntimeError(
            "seed smoke validation failed before search: "
            f"invalid_fraction={smoke['invalid_fraction']:.1%}, details={smoke}")
    if not np.isfinite(smoke["score"]):
        raise RuntimeError(
            f"seed smoke validation failed before search: non-finite quality, details={smoke}")

    # search() normally writes its return immediately for launch()/status(). Suppress that
    # intermediate file so status cannot mistake the pre-replay result for tune() completion.
    result_path = os.environ.pop("PSEARCH_RESULT", None)
    try:
        out = search(
            env, rollout, guarded_objective, space,
            start_state=start_state if start_state is not None else start0,
            seed_values=canonical_seed,
            generations=generations,
            popsize=popsize,
            repeats=repeats,
            randomize=randomize,
            fix=fix,
            budget_s=budget_s,
            deadline=deadline,
            seed=seed,
            verbose=verbose,
        )
    finally:
        if result_path is not None:
            os.environ["PSEARCH_RESULT"] = result_path

    replay_anchor = _guarded_anchor(env, start0, randomize, int(seed) + 1_000_003)
    replay_pre = env.get_states()
    winner_replay = None
    seed_replay = None
    try:
        seed_replay = _guarded_fixed_rollout(
            env, rollout, sp, canonical_seed, guarded_objective, replay_anchor,
            sample_count, "seed_replay")
        if out["best"] is not None:
            winner_replay = _guarded_fixed_rollout(
                env, rollout, sp, out["best"], guarded_objective, replay_anchor,
                sample_count, "winner_replay")
    finally:
        env.set_states(replay_pre)

    verdict, verdict_reason = _guarded_verdict(
        winner_replay, seed_replay, out.get("at_bounds", {}),
        out.get("best_invalid_fraction") or 0.0)
    replay_scores = {
        "winner": winner_replay["score"] if winner_replay is not None else None,
        "seed": seed_replay["score"] if seed_replay is not None else None,
    }
    replay_goal_rates = {
        "winner": winner_replay["goal_rate"] if winner_replay is not None else None,
        "seed": seed_replay["goal_rate"] if seed_replay is not None else None,
    }
    invalid_fractions = {
        "seed_smoke": smoke["invalid_fraction"],
        "search_winner": out.get("best_invalid_fraction"),
        "search_seed": out.get("seed_invalid_fraction"),
        "winner_replay": (
            winner_replay["invalid_fraction"] if winner_replay is not None else None),
        "seed_replay": seed_replay["invalid_fraction"] if seed_replay is not None else None,
    }
    replay_improvement = (
        float(seed_replay["score"] - winner_replay["score"])
        if winner_replay is not None and seed_replay is not None
        and np.isfinite(seed_replay["score"]) and np.isfinite(winner_replay["score"])
        else None
    )
    out.update({
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "adoptable": verdict == "ADOPT",
        "seed_smoke": smoke,
        "replay": {"winner": winner_replay, "seed": seed_replay},
        "replay_scores": replay_scores,
        "replay_goal_rates": replay_goal_rates,
        "invalid_fractions": invalid_fractions,
        "winner_replay_score": replay_scores["winner"],
        "seed_replay_score": replay_scores["seed"],
        "winner_replay_goal_rate": replay_goal_rates["winner"],
        "seed_replay_goal_rate": replay_goal_rates["seed"],
        "replay_improvement_vs_seed": replay_improvement,
    })
    if verbose:
        winner_invalid = invalid_fractions["winner_replay"]
        winner_invalid_text = (
            f"{winner_invalid:.1%}" if winner_invalid is not None else "unavailable")
        winner_score_text = (
            f"{replay_scores['winner']:.5g}"
            if replay_scores["winner"] is not None else "unavailable")
        winner_goal_text = (
            f"{replay_goal_rates['winner']:.1%}"
            if replay_goal_rates["winner"] is not None else "unavailable")
        print(f"[parameter_search] replay: seed quality={replay_scores['seed']:.5g}, "
              f"winner quality={winner_score_text}, "
              f"seed goal_rate={replay_goal_rates['seed']:.1%}, "
              f"winner goal_rate={winner_goal_text}, "
              f"winner invalid_fraction={winner_invalid_text}", flush=True)
        print(f"[parameter_search] VERDICT {verdict}: {verdict_reason}", flush=True)

    if result_path:
        try:
            Path(result_path).write_text(json.dumps(out, indent=2, default=float) + "\n")
        except Exception as exc:  # noqa: BLE001 -- reporting failure stays visible
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
        entries = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"parameter-search registry is unreadable: {path}: {exc!r}") from exc
    if not isinstance(entries, list):
        raise RuntimeError(f"parameter-search registry must contain a list: {path}")
    return entries


def _registry_write(entries: list[dict]) -> None:
    """Atomically replace the launch registry while its caller holds the lock."""
    path = _registry_path()
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(entries, indent=2) + "\n")
    temporary.replace(path)


def _pid_running(pid) -> bool:
    """Cross-platform liveness check; Linux zombies are finished."""
    import subprocess

    try:
        pid = int(pid)
    except (TypeError, ValueError) as exc:
        print(f"[parameter_search] invalid registry pid {pid!r}: {exc!r}", flush=True)
        return False
    if pid <= 0:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            return stat_path.read_text().rsplit(")", 1)[-1].split()[0] != "Z"
        except (OSError, IndexError) as exc:
            print(f"[parameter_search] cannot inspect pid {pid}: {exc!r}", flush=True)
            return False
    try:
        probe = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5)
    except Exception as exc:  # noqa: BLE001 -- status reports the failed liveness probe
        print(f"[parameter_search] cannot inspect pid {pid}: {exc!r}", flush=True)
        return False
    state = probe.stdout.strip()
    return probe.returncode == 0 and bool(state) and not state.startswith("Z")


def launch(
    script: str | Path,
    *,
    gpu: str | int | None = None,
    name: str = "",
    max_concurrent_per_gpu: int = 1,
    stall_after_s: float = 600.0,
    force: bool = False,
) -> dict:
    """Run a search SCRIPT in the background, owned by the tool. Returns immediately.

    The script is any python file that calls `search(...)` (your rollout + objective + the
    call, exactly as you would run it in the foreground). launch() runs it in its own
    detached session — it cannot hit a foreground command timeout and survives shell
    cleanup sweeps that are not aimed at its pid — pinned via CUDA_VISIBLE_DEVICES to the
    GPU with the most free memory (override with `gpu=`). A content hash blocks duplicate
    active scripts, and at most ``max_concurrent_per_gpu`` searches may occupy one GPU
    (default one). ``force=True`` explicitly overrides those two guards. No job is killed.
    Poll with `status()`; it costs nothing and returns instantly.

        h = launch("/workspace/search_press.py")
        ...keep working...
        status(h["id"])     # -> generations done, best so far, or the final result
    """
    import fcntl
    import subprocess
    import sys

    src = Path(script).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"launch() needs a script file; {src} does not exist")
    if int(max_concurrent_per_gpu) < 1:
        raise ValueError("max_concurrent_per_gpu must be at least 1")
    if float(stall_after_s) <= 0:
        raise ValueError("stall_after_s must be positive")
    content_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    run_id = name or f"{src.stem}_{time.strftime('%H%M%S')}"

    env = dict(os.environ)
    picked = str(gpu) if gpu is not None else (_freest_gpu() or env.get("CUDA_VISIBLE_DEVICES", "0"))
    env["CUDA_VISIBLE_DEVICES"] = picked

    _PSEARCH_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _PSEARCH_DIR / "registry.lock"
    with lock_path.open("a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        entries = _registry_read()
        active = [entry for entry in entries if _pid_running(entry.get("pid"))]
        active_ids = [entry.get("id") for entry in active]
        duplicates = [
            entry for entry in active if entry.get("content_hash") == content_hash
        ]
        on_gpu = [entry for entry in active if str(entry.get("gpu")) == picked]
        reasons = []
        if duplicates:
            reasons.append(
                "duplicate active content " + str([entry.get("id") for entry in duplicates]))
        if len(on_gpu) >= int(max_concurrent_per_gpu):
            reasons.append(
                f"GPU {picked} already has {len(on_gpu)} active search(es), cap "
                f"{int(max_concurrent_per_gpu)}")
        if reasons and not force:
            raise RuntimeError(
                "launch refused: " + "; ".join(reasons)
                + f". Active IDs: {active_ids}. Pass force=True only for an explicit override.")
        requested_id = run_id
        suffix = 2
        known_ids = {entry.get("id") for entry in entries}
        while run_id in known_ids:
            run_id = f"{requested_id}_{suffix}"
            suffix += 1
        if run_id != requested_id:
            print(f"[parameter_search] launch id {requested_id!r} already exists; "
                  f"recording this run as {run_id!r}", flush=True)

        run_dir = _PSEARCH_DIR / run_id
        run_dir.mkdir(parents=False, exist_ok=False)
        env["PSEARCH_PROGRESS"] = str(run_dir / "progress.jsonl")
        env["PSEARCH_RESULT"] = str(run_dir / "result.json")
        log = run_dir / "log"
        with log.open("w") as fh:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(src)], stdout=fh, stderr=fh,
                start_new_session=True, env=env, cwd=str(src.parent))

        entry = {
            "id": run_id,
            "pid": proc.pid,
            "script": str(src),
            "content_hash": content_hash,
            "dir": str(run_dir),
            "log": str(log),
            "progress": env["PSEARCH_PROGRESS"],
            "result": env["PSEARCH_RESULT"],
            "gpu": picked,
            "started": time.time(),
            "stall_after_s": float(stall_after_s),
            "max_concurrent_per_gpu": int(max_concurrent_per_gpu),
            "forced": bool(force),
        }
        entries.append(entry)
        _registry_write(entries)
    print(f"[parameter_search] launched '{run_id}' (pid {proc.pid}, gpu {picked}); "
          f"log {log} — poll with status('{run_id}')", flush=True)
    return entry


def status(
    run_id: str | dict | None = None,
    *,
    stalled_after_s: float | None = None,
) -> dict:
    """Progress of a launched search, instantly: generations done, best so far, final result.

    No argument = the most recently launched. Prints the verdict first when one exists and
    returns the full
    picture: {"running": bool, "generations_done": int, "last": <latest generation record>,
    "stalled": bool, "result": <search() result dict when finished>, plus the launch entry}.
    A search that stopped without a result crashed — the "log" path holds why.
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

    running = _pid_running(entry.get("pid"))
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

    threshold = float(
        stalled_after_s if stalled_after_s is not None else entry.get("stall_after_s", 600.0))
    if threshold <= 0:
        raise ValueError("stalled_after_s must be positive")
    activity_times = [float(entry.get("started", 0.0))]
    if progress.exists():
        try:
            activity_times.append(progress.stat().st_mtime)
        except OSError as exc:
            print(f"[parameter_search] cannot read activity time for {progress}: "
                  f"{exc!r}", flush=True)
    stalled_for_s = max(0.0, time.time() - max(activity_times))
    stalled = bool(running and stalled_for_s >= threshold)
    out = {**entry, "running": running, "stalled": stalled,
           "stalled_for_s": round(stalled_for_s, 1), "stall_after_s": threshold,
           "generations_done": len(gens),
           "last": gens[-1] if gens else None, "result": result}
    if result is not None:
        verdict = result.get("verdict")
        if verdict is not None:
            print(f"[parameter_search] VERDICT {verdict}: "
                  f"{result.get('verdict_reason', '')}", flush=True)
        improvement = result.get("improvement_vs_seed", result.get("gain_vs_seed"))
        bs = result.get("best_score")
        bs_txt = f"{bs:.5g}" if isinstance(bs, (int, float)) else str(bs)
        print(f"[parameter_search] '{entry['id']}' FINISHED: best {bs_txt} "
              f"at {result['best']}"
              + (f", improvement vs seed {improvement:.5g}"
                 if improvement is not None else ""), flush=True)
    elif stalled:
        print(f"[parameter_search] '{entry['id']}' STALLED: no completed-generation progress for "
              f"{stalled_for_s:.0f}s (threshold {threshold:.0f}s); it is still running and "
              f"will not be killed automatically. The log may still show rollout heartbeats: "
              f"{entry['log']}",
              flush=True)
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
