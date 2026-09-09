#!/usr/bin/env python3
"""Stage datagen seeds from the CoSiGen_Solutions reference repo (the held-out oracles).

Every campaign multiplies ONE verified solve. The references in CoSiGen_Solutions are the
originals every task was solved with (review item 5.4: v2 mixed v1 outputs and raw eval solves
— the fix was "restore originals for all 11 tasks"), so this writes each of them into the
eval-run shape runpod_dgen_campaign.py / orchestrate.py consume:

    /tmp/dgen_inputs/stage/<task>/run.json                 preset + provenance
    /tmp/dgen_inputs/stage/<task>/workspace/solution/*     solve.py + sibling modules

    python data_engine/scripts/stage_seeds.py --solutions /tmp/CoSiGen_Solutions \\
        --tasks allen_bolt,bulb,coffee,...            # franka presets; controller from MANIFEST

`source_status` is "completed": a reference solution is verified by definition (its MANIFEST
carries the CoSiGen rev it passed at), which is what the launcher's seed guard asks for. The
MANIFEST rev is recorded in run.json so a pre-check failure can be read as drift of the live
tree away from that rev, not as a broken seed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

STAGE = Path("/tmp/dgen_inputs/stage")
ROBOT = "franka"


def manifest(path: Path) -> dict:
    facts: dict[str, str] = {}
    key = None
    for line in path.read_text().splitlines():
        if line.startswith((" ", "\t")) and key:          # wrapped continuation of the last value
            facts[key] += " " + line.strip()
        elif ":" in line and not line.startswith("#"):
            key, _, v = line.partition(":")
            key = key.strip()
            facts[key] = v.strip()
    return facts


def find_solution(solutions: Path, task: str) -> Path:
    """<suite>/<task>/franka/<controller>/ — exactly one controller folder per task is expected."""
    hits = [p for p in solutions.glob(f"*/{task}/{ROBOT}/*/") if (p / "solve.py").is_file()]
    if len(hits) != 1:
        raise SystemExit(f"{task}: expected exactly one {ROBOT} solution folder, found "
                         f"{[str(h.relative_to(solutions)) for h in hits]}")
    return hits[0]


def stage_one(solutions: Path, sha: str, task: str) -> dict:
    src = find_solution(solutions, task)
    m = manifest(src / "MANIFEST")
    preset = m.get("preset") or ".".join(src.relative_to(solutions).parts)
    dst = STAGE / task
    if dst.exists():
        shutil.rmtree(dst)
    sol = dst / "workspace" / "solution"
    sol.mkdir(parents=True)
    for f in src.iterdir():
        if f.name in ("MANIFEST", "solve.mp4") or f.name.startswith("."):
            continue
        if f.is_dir():
            shutil.copytree(f, sol / f.name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(f, sol / f.name)
    run = {
        "preset": preset,
        "source_run": f"CoSiGen_Solutions/{src.relative_to(solutions)}@{sha}",
        "source_status": "completed",          # a verified reference (see module docstring)
        "model": "reference",
        "reference": {"solved_by": m.get("solved_by", ""),
                      "cosigen_rev": m.get("cosigen_rev", ""),
                      "run_wall_time": m.get("run_wall_time", ""),
                      "agent_solve_time": m.get("agent_solve_time", "")},
    }
    (dst / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return {"task": task, "preset": preset, "files": sorted(p.name for p in sol.iterdir())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solutions", required=True, help="checkout of CoSiGen_Solutions")
    ap.add_argument("--tasks", required=True, help="comma-separated task names")
    a = ap.parse_args()
    solutions = Path(a.solutions).resolve()
    sha = subprocess.run(["git", "-C", str(solutions), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip() or "nogit"
    STAGE.mkdir(parents=True, exist_ok=True)
    for task in [t for t in a.tasks.split(",") if t]:
        info = stage_one(solutions, sha, task)
        print(f"{info['task']:15s} {info['preset']:40s} {','.join(info['files'])}", flush=True)
    print(f"staged under {STAGE} from CoSiGen_Solutions@{sha}", flush=True)


if __name__ == "__main__":
    main()
