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
    from .env import BaseEnv


class BaseGrader(ABC):
    """A concrete grader MUST provide: a docstring, `SCENE` (the scene class
    it grades — the env is type-checked at construction), `RUBRIC`, one
    method per stage, `setup()`, and `check_success()` — anything missing or
    mismatched fails loudly at construction.

    `RUBRIC` = ((stage, weight) | (stage, weight, "once"), ...). Each stage
    names a method on the grader returning its instantaneous value in [0, 1]
    (discrete 0/1 for a satisfied-or-not check, continuous for a fraction).
    A plain (stage, weight) entry is scored LIVE — its current value, so the
    curve dips when the state is lost; a "once" entry is a milestone — scored
    on the best value ever achieved (e.g. a pick that ends when the part is
    placed). `progress()` = the weighted sum normalized by the weight total,
    so weights are relative and need not sum to 1.
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
        self._best = {n: 0.0 for n, _, mode in self._stages if mode == "once"}
        self.env = env
        self.scene = env.scene  # children re-annotate with their SCENE type
        self.steps = 0
        self.history: list[dict] = []  # one record per graded step — the score-vs-time curve
        self._t0 = time.monotonic()
        self.setup()

    @abstractmethod
    def setup(self) -> None:
        """Capture whatever the rubric needs from the just-reset env (start
        poses, references). Runs once at construction."""

    @property
    def sim_time(self) -> float:
        return round(self.steps * self.env.dt * self.env.robot.control_period, 4)

    def measure(self) -> dict[str, float]:
        """Every stage's scored value: the method's clamped return for a live
        stage, the best ever achieved for a "once" milestone."""
        values = {}
        for name, _, mode in self._stages:
            v = min(max(getattr(self, name)(), 0.0), 1.0)
            if mode == "once":
                v = self._best[name] = max(self._best[name], v)
            values[name] = v
        return values

    def progress(self, values: dict[str, float] | None = None) -> float:
        values = self.measure() if values is None else values
        total = sum(w for _, w, _ in self._stages)
        return sum(w * values[n] for n, w, _ in self._stages) / total

    @abstractmethod
    def check_success(self) -> bool:
        """Instantaneous, authoritative goal test (e.g. `scene.seated()`) —
        the ladder scores, this decides."""

    def record(self) -> dict:
        """One curve entry — the ladder value AND every stage's own value;
        GradedEnv calls this after every step."""
        self.steps += 1
        m = self.measure()
        rec = {"sim_time_s": self.sim_time,
               "wall_s": round(time.monotonic() - self._t0, 3),
               "progress": round(self.progress(m), 4),
               "stages": {k: round(v, 4) for k, v in m.items()}}
        self.history.append(rec)
        return rec

    def verdict(self) -> tuple[bool, float]:
        """(success, score) — a pure read, no stepping. success = the
        authoritative test on the final state; score = the highest progress
        recorded over the run."""
        peak = max((r["progress"] for r in self.history), default=self.progress())
        return self.check_success(), peak

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
                 on_record: Callable[[dict], None] | None = None) -> None:
        self._env, self._grader, self._on_record = env, grader, on_record

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action, render: bool = False):
        out = self._env.step(action, render)
        rec = self._grader.record()
        if self._on_record is not None:
            self._on_record(rec)
        return out

    def reset(self, *args, **kwargs):
        raise PermissionError("blocked during grading — the graded env was reset "
                              "by the grader; solve may only step forward")

    def set_states(self, *args, **kwargs):
        raise PermissionError("blocked during grading — solve may only step forward")

    def verdict(self) -> tuple[bool, float]:
        return self._grader.verdict()
