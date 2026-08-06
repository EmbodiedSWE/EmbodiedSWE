"""Task validation (stage 4): generic physical gates + the task's own smoke battery.

Usage:  python sim_gen/pipeline/validate.py --task <task_name>
Writes a JSON report to artifacts/reports/<task_name>.validate.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SIM_GEN_ROOT = Path(__file__).resolve().parent.parent
COSIGEN_ROOT = SIM_GEN_ROOT.parent
PYTHON = os.environ.get("SIM_GEN_PYTHON") or sys.executable
REPORTS = SIM_GEN_ROOT / "artifacts" / "reports"


def load_scene(task_name: str):
    """Import the task package and return an instance of its registered scene."""
    from sim_gen.core import SCENES

    before = set(SCENES)
    importlib.import_module(f"sim_gen.tasks.{task_name}.scene")
    new = [SCENES[k] for k in SCENES if k not in before]
    if len(new) == 1:
        return new[0]()
    if not new:  # scene may already be imported (idempotent runs) — match by module
        for cls in SCENES.values():
            if cls.__module__ == f"sim_gen.tasks.{task_name}.scene":
                return cls()
    raise RuntimeError(f"expected exactly 1 scene registered by task {task_name!r}")


def generic_gates(task_name: str) -> list[dict]:
    """The task-independent physical/semantic gates."""
    results = []

    def gate(name: str, ok: bool, detail: str = ""):
        results.append({"gate": name, "pass": bool(ok), "detail": detail})
        print(f"[gate] {'PASS' if ok else 'FAIL':4s} {name} {detail}", flush=True)

    # 1. build
    try:
        scene = load_scene(task_name)
        gate("build: scene compiles", True)
    except Exception as e:  # noqa: BLE001 — the gate's whole job is to report this
        gate("build: scene compiles", False, repr(e))
        return results

    # 2. settle: zero-action rollout comes to rest, no NaN
    scene.reset(seed=0)
    scene.settle(500)
    ke = float(np.sum(scene.data.qvel**2))
    gate("settle: no NaN", bool(np.isfinite(scene.data.qpos).all()
                                and np.isfinite(scene.data.qvel).all()))
    gate("settle: kinetic energy decays", ke < 1e-2, f"sum qvel^2 = {ke:.2e}")

    # 3. determinism + state round-trip
    scene.reset(seed=0)
    scene.step(300)
    h1 = scene.state_hash()
    snap = scene.get_state()
    scene.reset(seed=0)
    scene.step(300)
    gate("determinism: same seed => same trajectory", h1 == scene.state_hash())
    scene.step(100)
    scene.set_state(snap)
    gate("state restore round-trips", scene.state_hash() == h1)

    # 4. randomization is real
    hashes = set()
    for s in range(5):
        scene.reset(seed=s)
        hashes.add(scene.state_hash())
    gate("randomization: 5 seeds => distinct instances", len(hashes) == 5,
         f"{len(hashes)}/5 unique")

    # 5. null policy
    scene.reset(seed=0)
    scene.settle(scene.cfg.episode_steps)
    gate("null policy: success() is False", not scene.success())
    gate("null policy: score() ~ 0", scene.score() < 0.05, f"score {scene.score():.3f}")

    # 6. describe() exists and is substantive
    desc = scene.describe()
    gate("describe(): substantive task statement", isinstance(desc, str) and len(desc) > 60,
         f"{len(desc)} chars")
    return results


def post_smoke_gates(task_name: str) -> dict:
    """Re-run the smoke's exported oracle_solution under the validator's own
    instrumentation (don't trust the smoke's self-reported results alone)."""
    import importlib

    result: dict = {"gates": []}

    def gate(name: str, ok: bool, detail: str = ""):
        result["gates"].append({"gate": name, "pass": bool(ok), "detail": detail})
        print(f"[gate] {'PASS' if ok else 'FAIL':4s} {name} {detail}", flush=True)

    smoke_mod = importlib.import_module(f"sim_gen.tasks.{task_name}.smoke")
    oracle_solution = getattr(smoke_mod, "oracle_solution", None)
    if oracle_solution is None:
        gate("oracle_solution exported by smoke.py", False,
             "cannot run validator-side persistence gate")
        return result

    scene = load_scene(task_name)
    scene.reset(seed=0)
    oracle_solution(scene)
    gate("oracle solution reaches success under validator", bool(scene.success()))

    # success persistence: no single-frame flicker success (also rejects success
    # triggered by transient fly-through states, so no separate quiescence gate)
    ok_hold = True
    for _ in range(10):
        scene.step(30)
        ok_hold = ok_hold and bool(scene.success())
    gate("success persists over 300-step hold window", ok_hold)
    return result


def run_smoke(task_name: str) -> dict:
    """Run the task's own smoke (its oracle solution + test battery), require ALL PASS."""
    video = SIM_GEN_ROOT / "artifacts" / f"{task_name}_smoke.mp4"
    cmd = [PYTHON, "-m", f"sim_gen.tasks.{task_name}.smoke", "--video", str(video)]
    env = dict(os.environ, MUJOCO_GL="osmesa")
    proc = subprocess.run(cmd, cwd=str(COSIGEN_ROOT), env=env,
                          capture_output=True, text=True, timeout=1800)
    out = proc.stdout + proc.stderr
    all_pass = "SIM_GEN_SMOKE: ALL PASS" in out
    print(out[-2000:], flush=True)
    return {"all_pass": all_pass, "rc": proc.returncode,
            "video": str(video) if video.exists() else None,
            "tail": out[-2000:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    gates = generic_gates(args.task)
    gates_ok = all(g["pass"] for g in gates)
    smoke = run_smoke(args.task) if gates_ok else {"all_pass": False,
                                                   "skipped": "generic gates failed"}
    post = (post_smoke_gates(args.task) if gates_ok and smoke["all_pass"]
            else {"gates": [], "skipped": "smoke failed"})
    post_ok = all(g["pass"] for g in post["gates"])
    verdict = gates_ok and smoke["all_pass"] and post_ok

    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / f"{args.task}.validate.json"
    report_path.write_text(json.dumps(
        {"task": args.task, "verdict": "PASS" if verdict else "FAIL",
         "generic_gates": gates, "smoke": smoke, "post_smoke": post}, indent=2))
    print(f"[validate] {'PASS' if verdict else 'FAIL'} — report -> {report_path}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    sys.path.insert(0, str(COSIGEN_ROOT))
    main()
