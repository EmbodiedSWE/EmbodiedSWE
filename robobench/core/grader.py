"""Grader — privileged, hidden scorer, applied by wrapping the graded env.

A grader is a rubric: weighted stages, each an individual method on the
concrete grader. `GradedEnv` wraps the env handed to the delivered
`solve(env)` and records the rubric after every step, so scoring happens
automatically while the delivery runs; when the harness is done stepping,
`verdict()` reads the final state. Never exposed to the agent.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch

    from .env import BaseEnv


class BaseGrader(ABC):
    """A concrete grader MUST provide: a docstring, `SCENE` (the scene class
    it grades — the env is type-checked at construction), `RUBRIC`, one
    method per stage, `setup()`, and `check_success()` — anything missing or
    mismatched fails loudly at construction.

    `RUBRIC` = ((stage, weight) | (stage, weight, "once"), ...). Each stage
    names a method on the grader returning its instantaneous PER-ENV value in
    [0, 1] — a `(num_envs,)` tensor (a scalar broadcasts to every env);
    discrete 0/1 for a satisfied-or-not check, continuous for a fraction.
    A plain (stage, weight) entry is scored LIVE — its current value, so the
    curve dips when the state is lost; a "once" entry is a milestone — scored
    on the best value ever achieved per env (e.g. a pick that ends when the
    part is placed). `progress()` = the weighted sum normalized by the weight
    total, so weights are relative and need not sum to 1. The grader is
    PER-TRAJECTORY throughout: every record and verdict is reported per
    env, in the same format regardless of `num_envs` — aggregation
    (statistics over the batch) is the harness scripts' job, never the
    class's.
    """

    SCENE: type | None = None
    RUBRIC: tuple[tuple, ...] = ()

    def __init__(self, env: BaseEnv) -> None:
        if self.SCENE is None:
            raise TypeError(f"{type(self).__name__} declares no SCENE")
        if not isinstance(env.scene, self.SCENE):
            raise TypeError(f"{type(self).__name__} grades {self.SCENE.__name__}, "
                            f"got {type(env.scene).__name__}")
        if not self.RUBRIC:
            raise TypeError(f"{type(self).__name__} declares no RUBRIC")
        self._stages = [(e[0], e[1], e[2] if len(e) > 2 else "live") for e in self.RUBRIC]
        bad = [n for n, _, mode in self._stages if mode not in ("live", "once")]
        if bad:
            raise TypeError(f"{type(self).__name__} has unknown stage modes on: {bad}")
        missing = [n for n, _, _ in self._stages if not callable(getattr(self, n, None))]
        if missing:
            raise TypeError(f"{type(self).__name__} is missing stage methods: {missing}")
        if any(w <= 0 for _, w, _ in self._stages):
            raise TypeError(f"{type(self).__name__} has non-positive rubric weights")
        import torch

        self.num_envs = env.num_envs
        self._best = {n: torch.zeros(self.num_envs)
                      for n, _, mode in self._stages if mode == "once"}
        self.env = env
        self.scene = env.scene  # children re-annotate with their SCENE type
        self.steps = 0
        self.history: list[list[dict]] = []  # per step: one record per trajectory (the curves)
        self._t0 = time.monotonic()
        self.setup()

    @abstractmethod
    def setup(self) -> None:
        """Capture whatever the rubric needs from the just-reset env (start
        poses, references). Runs once at construction."""

    @property
    def sim_time(self) -> float:
        return round(self.steps * self.env.dt * self.env.robot.control_period, 4)

    def _per_env(self, value, name: str):
        """Coerce a stage/success return to a `(num_envs,)` CPU tensor —
        scalars broadcast, anything else must match the env count."""
        import torch

        v = torch.as_tensor(value).detach().reshape(-1).cpu()
        if v.numel() == 1 and self.num_envs > 1:
            v = v.expand(self.num_envs).clone()
        if v.numel() != self.num_envs:
            raise ValueError(f"{type(self).__name__}.{name} returned {v.numel()} "
                             f"values for {self.num_envs} envs")
        return v

    def measure(self) -> dict:
        """Every stage's scored per-env value (`(num_envs,)` CPU tensor): the
        method's clamped return for a live stage, the best ever achieved per
        env for a "once" milestone."""
        import torch

        values = {}
        for name, _, mode in self._stages:
            v = self._per_env(getattr(self, name)(), name).float().clamp(0.0, 1.0)
            if mode == "once":
                v = self._best[name] = torch.maximum(self._best[name], v)
            values[name] = v
        return values

    def progress(self, values: dict | None = None):
        """The weighted rubric total per env — a `(num_envs,)` tensor."""
        values = self.measure() if values is None else values
        total = sum(w for _, w, _ in self._stages)
        return sum(w * values[n] for n, w, _ in self._stages) / total

    @abstractmethod
    def check_success(self) -> "torch.Tensor | bool":
        """Instantaneous, authoritative PER-ENV goal test (e.g.
        `scene.seated().all(dim=1)` -> `(num_envs,)` bools; a scalar
        broadcasts) — the ladder scores, this decides."""

    def record(self) -> list[dict]:
        """One curve entry PER TRAJECTORY — entry e is env e's ladder +
        stage values, in the same format regardless of `num_envs`;
        GradedEnv calls this after every step."""
        self.steps += 1
        m = self.measure()
        p = self.progress(m)
        wall = round(time.monotonic() - self._t0, 3)
        recs = [{"sim_time_s": self.sim_time, "wall_s": wall,
                 "progress": round(float(p[e]), 4),
                 "stages": {k: round(float(v[e]), 4) for k, v in m.items()}}
                for e in range(self.num_envs)]
        self.history.append(recs)
        return recs

    def verdict(self) -> list[dict]:
        """One verdict PER TRAJECTORY — a pure read, no stepping. For env e:
        success = the authoritative test on its final state; score = its
        highest recorded progress. Statistics over the batch are the
        caller's job."""
        succ = self._per_env(self.check_success(), "check_success")
        if self.history:
            peaks = [max(step[e]["progress"] for step in self.history)
                     for e in range(self.num_envs)]
        else:
            peaks = [round(float(x), 4) for x in self.progress()]
        return [{"env": e, "success": bool(succ[e]), "score": peaks[e]}
                for e in range(self.num_envs)]

    def describe(self) -> str:
        """Criteria provenance — written next to every verdict."""
        doc = (self.__doc__ or "").strip().splitlines()[0]
        weights = " · ".join(f"{n} ×{w:g}" + (" (once)" if mode == "once" else "")
                             for n, w, mode in self._stages)
        return f"{type(self).__name__}: {doc} — rubric: {weights}"


class GradedEnv:
    """What the delivered solve(env) receives instead of the raw env.

    Same surface — every attribute delegates to the wrapped env — except:
    step() also records the grader's rubric, and the grading-time shortcuts,
    reset and set_states, are blocked. The harness keeps the raw env."""

    def __init__(self, env: BaseEnv, grader: BaseGrader,
                 on_record: Callable[[list[dict]], None] | None = None) -> None:
        self._env, self._grader, self._on_record = env, grader, on_record

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action, render: bool = False):
        out = self._env.step(action, render)
        recs = self._grader.record()  # one record per trajectory
        if self._on_record is not None:
            self._on_record(recs)
        return out

    def reset(self, *args, **kwargs):
        raise PermissionError("blocked during grading — the graded env was reset "
                              "by the grader; solve may only step forward")

    def set_states(self, *args, **kwargs):
        raise PermissionError("blocked during grading — solve may only step forward")

    def verdict(self) -> list[dict]:
        """Per-trajectory verdicts — aggregation is the caller's job."""
        return self._grader.verdict()
