#!/usr/bin/env python3
"""Did the delivered solution solve the task? Harness-side, never seen by the agent.

Between legs of a long run we need to know whether the agent may stop. This asks the
question under the rules the task text states: build the env, reset it, wrap it so that
`reset` and `set_states` are blocked exactly as the text promises, and hand THAT to the
delivered `solve(env)`. Without the wrapper a solve that teleports the parts into
place would be declared solved and end the run, which is the one outcome this whole
setup exists to exclude.

This is orchestration, not grading. It asks one question — does the scene's own
success() pass — and asks it identically for every task. Graders are deliberately not
consulted: rubrics and partial credit are judged offline, and mixing them in here would
make the stop condition mean different things on different tasks.

Runs in its own process because Isaac must be booted fresh for an honest verdict, and
prints a single JSON line so the caller does not have to parse Isaac's chatter.

Usage:
  verify_solution.py --preset assembly.nut_thread.franka.osc --solution /workspace/solution
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from pathlib import Path


class _NoShortcuts:
    """The env the delivery gets: the same surface, with the two doors shut.

    Without this, "solved" could mean "wrote the goal state into the sim".
    """

    def __init__(self, env) -> None:
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        raise PermissionError("blocked during grading — solve may only step forward")

    def set_states(self, *args, **kwargs):
        raise PermissionError("blocked during grading — solve may only step forward")


def finish(verdict: dict) -> None:
    """Print the verdict and exit with the code the run loop reads.

    Every exit goes through here. An early `return` for one case used to skip the exit
    below and hand the loop python's default 0 — "solved" — so a run whose agent had not
    written solve.py yet was declared finished at its first leg boundary.
    """
    print("VERDICT " + json.dumps(verdict), flush=True)
    import os  # noqa: PLC0415

    # Only 0 means "solved". A scene we cannot evaluate is NOT a failure: it gets its own
    # code so a caller can never mistake "no predicate" for "the agent failed" — that
    # mistake cost ikea11 22 pointless legs.
    code = 0 if verdict.get("success") else (20 if verdict.get("unverifiable") else 10)
    os._exit(code)  # kit hangs on close; the verdict is already printed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--solution", required=True)
    ap.add_argument("--max-seconds", type=float, default=1800)
    args = ap.parse_args()

    # ENFORCE the budget. This argument existed but was never read, and an unbounded check is
    # not a slow check — it is a dead run: a delivered solve() that hangs (or just simulates for
    # hours) freezes agent-entry.sh's keep-going loop inside solved(), so no further leg ever
    # starts while the runner still reports the run as alive. Observed on nut_thread_c1, which
    # sat in one success check for 2h11m of its 24h budget with a frozen trajectory.
    #
    # A daemon timer, not signal.alarm: the hang is inside Isaac's C extensions, where a Python
    # signal handler would not run until control returned — which is exactly what never happens.
    # os._exit from the timer thread is immediate and needs no interpreter cooperation.
    def _timed_out() -> None:
        print(
            "VERDICT " + json.dumps({
                "preset": args.preset, "success": False, "timed_out": True,
                "reason": f"success check exceeded --max-seconds {args.max_seconds:.0f}",
            }),
            flush=True,
        )
        import os  # noqa: PLC0415

        os._exit(10)  # not solved, and NOT 'unverifiable' — the scene was never reached

    watchdog = threading.Timer(args.max_seconds, _timed_out)
    watchdog.daemon = True
    watchdog.start()

    verdict = {"preset": args.preset, "success": False, "reason": ""}
    solve_py = Path(args.solution) / "solve.py"
    if not solve_py.is_file():
        verdict["reason"] = "no solution/solve.py delivered yet"
        finish(verdict)

    try:
        from isaaclab.app import AppLauncher  # noqa: PLC0415

        AppLauncher(headless=True)  # must precede the isaaclab imports below
        import robobench  # noqa: PLC0415
        from robobench.core.registries import ENVS  # noqa: PLC0415

        robobench.discover()
        env = ENVS.get(args.preset)().build(num_envs=1)
        env.reset()  # ours, before the wrapper — the delivery may not reset

        graded = _NoShortcuts(env)

        sys.path.insert(0, str(Path(args.solution).resolve()))
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.spec_from_file_location("delivered_solve", solve_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "solve"):
            verdict["reason"] = "solve.py defines no solve(env)"
        else:
            mod.solve(graded)
            predicate = getattr(env.scene, "success", None)
            if callable(predicate):
                ok = predicate()
                verdict["success"] = bool(ok.all() if hasattr(ok, "all") else ok)
                verdict["reason"] = "scene success predicate after the delivered solve"
            else:
                # Most scenes define neither. Saying so is the honest answer; reporting it
                # as a failed check made every leg look like an error and churned the run.
                verdict.update(success=False, unverifiable=True,
                               reason=f"{type(env.scene).__name__} defines no success(): "
                                      f"nothing to orchestrate on")
    except Exception as exc:  # noqa: BLE001 -- a broken delivery is a failed verify
        verdict["reason"] = f"{type(exc).__name__}: {exc}"[:300]
        verdict["traceback"] = traceback.format_exc(limit=3)[-600:]
    finish(verdict)


if __name__ == "__main__":
    main()
