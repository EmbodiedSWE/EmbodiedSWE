#!/usr/bin/env python3
"""Grade one delivered solve.py — runs INSIDE the grading container.

Builds the registered preset from /bench (the same read-only tree the agent
had), wraps it in GradedEnv with the scene's grader from /graders, runs the
delivery's solve(env), writes verdict.json + progress.jsonl to /out. One
trajectory, no seeds yet. verdict.json is always written — success False
with the traceback if solve raises; a hang is the host launcher's budget
to kill.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

SOLUTION = "/solution/solve.py"  # fixed mount points: eval/scripts/run_grade.py
GRADERS_DIR = Path("/graders")
OUT = Path("/out")


def load_graders(grader_dir: Path) -> dict:
    """Import the suite's grader package from a path outside the /bench tree."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "suite_grader", grader_dir / "__init__.py",
        submodule_search_locations=[str(grader_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["suite_grader"] = mod  # registered first so its relative imports resolve
    spec.loader.exec_module(mod)
    return mod.GRADERS


def load_solve(path: str):
    """Import the delivery and hand back its solve(env)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("solve", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "solve", None)):
        raise AttributeError(f"{path} does not expose solve(env)")
    return mod.solve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app  # noqa: F841 — before any isaaclab.sim import

    import robobench

    robobench.discover()
    from robobench.core import GradedEnv
    from robobench.core.registries import ENVS

    grader_cls = load_graders(GRADERS_DIR)[args.scene]
    env = ENVS.get(args.preset)().build(num_envs=1)
    env.reset()
    grader = grader_cls(env)  # one grader instance = this trajectory

    def on_record(rec: dict) -> None:
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    genv = GradedEnv(env, grader, on_record=on_record)

    result = {"preset": args.preset, "scene": args.scene, "criteria": grader.describe()}
    try:
        solve = load_solve(SOLUTION)  # the delivery

        t0 = time.monotonic()
        solve(genv)
        result["solve_wall_s"] = round(time.monotonic() - t0, 3)
        result["solve_sim_steps"] = grader.steps

        success, score = genv.verdict()
        result.update(success=success, score=score)
    except Exception:
        result.update(success=False, score=0.0, error=traceback.format_exc())

    (out / "verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"GRADE_DONE success={result['success']} score={result['score']}", flush=True)
    os._exit(0)  # Kit sometimes hangs on close; the verdict is on disk


if __name__ == "__main__":
    main()
