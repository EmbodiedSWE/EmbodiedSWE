#!/usr/bin/env python3
"""Replay one delivered solve.py from a FRESH reset and report the real oracle score.

Self-contained (does not touch the docker-only grade.py). Runs inside a pod's Isaac venv.
Given a preset and a solution dir, it builds the registered env, resets it, and runs the
delivery's solve(env) with reset()/set_states() blocked (a solve may only step forward, so
it cannot teleport parts into place). After every step it reads the scene's OWN authoritative
scorer and keeps the peak:

    RUBRIC grader for the scene (robobench/suites/<suite>/grader) if one exists  -> progress()
    else scene.score()   (int 0..100 for the 5 scenes that define partial credit) -> /100
    else scene.success() (every scene; binary)                                    -> 0/1

Prints exactly one line:  VERDICT {"preset":..., "success":bool, "score":float, "steps":int}
Only that line matters to the caller; Isaac chatter goes to stderr.

    python grade_replay.py --preset assembly.nut_thread.franka.osc --solution /sol --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from pathlib import Path


def _emit(v: dict) -> None:
    print("VERDICT " + json.dumps(v), flush=True)
    import os
    os._exit(0)  # Kit hangs on close; the verdict is already printed


class _StepGraded:
    """Wraps the env for the delivery: reset/set_states blocked; every step measures the
    scene's real scorer and updates the running peak held on the closure."""

    def __init__(self, env, measure, peak):
        self._env, self._measure, self._peak = env, measure, peak

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action, render: bool = False):
        out = self._env.step(action, render)
        self._peak[0] = max(self._peak[0], self._measure())
        return out

    def reset(self, *a, **k):
        raise PermissionError("blocked during grading — solve may only step forward")

    def set_states(self, *a, **k):
        raise PermissionError("blocked during grading — solve may only step forward")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=1800)
    args = ap.parse_args()

    def _timed_out():
        _emit({"preset": args.preset, "success": False, "score": 0.0, "steps": -1,
               "reason": f"exceeded --max-seconds {args.max_seconds:.0f}"})
    wd = threading.Timer(args.max_seconds, _timed_out)
    wd.daemon = True
    wd.start()

    solve_py = Path(args.solution) / "solve.py"
    if not solve_py.is_file():
        _emit({"preset": args.preset, "success": False, "score": 0.0, "steps": 0,
               "reason": "no solve.py"})

    try:
        from isaaclab.app import AppLauncher
        AppLauncher(headless=True)
        import robobench
        from robobench.core.registries import ENVS
        robobench.discover()

        env = ENVS.get(args.preset)().build(num_envs=args.num_envs, seed=args.seed)
        env.reset(seed=args.seed)
        scene = env.scene

        # Pick the scene's authoritative scorer, best available first.
        suite, scene_name = args.preset.split(".")[0], args.preset.split(".")[1]
        grader = None
        try:
            import importlib
            gmod = importlib.import_module(f"robobench.suites.{suite}.grader")
            gcls = getattr(gmod, "GRADERS", {}).get(scene_name)
            if gcls is not None:
                grader = gcls(env)
        except Exception as exc:  # noqa: BLE001 -- fall back to score()/success()
            print(f"[grade_replay] no rubric grader for {scene_name}: {exc!r}", file=sys.stderr)

        def measure() -> float:
            import torch
            if grader is not None:
                v = grader.progress()
                return float(v.max() if hasattr(v, "max") else v)
            if hasattr(scene, "score"):
                s = scene.score()
                s = s.float().max() if hasattr(s, "max") else float(s)
                return float(s) / 100.0
            s = scene.success()
            return float(s.max() if hasattr(s, "max") else s)

        peak = [measure()]  # score at the reset state (usually 0)
        graded = _StepGraded(env, measure, peak)

        sys.path.insert(0, str(Path(args.solution).resolve()))
        import importlib.util
        spec = importlib.util.spec_from_file_location("delivered_solve", solve_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "solve"):
            _emit({"preset": args.preset, "success": False, "score": round(peak[0], 4),
                   "steps": 0, "reason": "solve.py defines no solve(env)"})

        steps_before = int(getattr(env, "step_count", 0) or 0)
        mod.solve(graded)
        # final authoritative success on the end state
        succ = scene.success()
        success = bool(succ.all() if hasattr(succ, "all") else succ)
        peak[0] = max(peak[0], measure())
        _emit({"preset": args.preset, "success": success, "score": round(peak[0], 4),
               "steps": int(getattr(env, "step_count", 0) or 0) - steps_before})
    except Exception as exc:  # noqa: BLE001 -- a broken delivery is a real (failed) grade
        _emit({"preset": args.preset, "success": False,
               "score": round(peak[0], 4) if "peak" in dir() else 0.0, "steps": -1,
               "reason": f"{type(exc).__name__}: {exc}"[:300],
               "traceback": traceback.format_exc(limit=3)[-500:]})


if __name__ == "__main__":
    main()
