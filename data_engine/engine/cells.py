"""cells — create a new cell in a data_gen campaign.

One cell per proposed variant: a scene copy to edit, an empty strategy, or a
phase cell — one PROPOSAL of how to divide/enter the strategy's solve, holding
its own solve_by_phase.py + reset/ conditions. Used by the agent
(agent/cli/create_cell.py) to propose many variants in one session.
"""


from __future__ import annotations

import json
import shutil
from pathlib import Path


RESET_PY_STUB = '''"""One file per phase, named as the phase: the sampled file chooses the
entry and builds its state via reset_0(env), reset_1(env), … (generate
targets one by index, default 0)."""

from __future__ import annotations


def reset_0(env) -> None:
    return
'''

META_STUB = {"episodes": 0, "successes": 0, "success_rate": None, "batches": [], "refreshed": None}

def next_name(parent: Path, prefix: str) -> str:
    """Always one past the HIGHEST existing index — gaps are never refilled, so a
    deleted cell's name is never reused (batch metas in the pool may still
    reference it) and repeated create_cell calls count strictly onward."""
    ns = [int(p.name.rsplit("_", 1)[1]) for p in parent.glob(f"{prefix}_*")
          if p.name.rsplit("_", 1)[1].isdigit()]
    return f"{prefix}_{max(ns) + 1 if ns else 0}"


def create_cell(gen_root: Path, level: str, scene: str, strategy: str, name: str | None) -> tuple[Path, str, str]:
    """Create the target cell; returns (target_dir, base_reference, new_name)."""
    scenes = gen_root / "scenes"
    if level == "scene":
        name = name or next_name(scenes, "scene")
        src, dst = scenes / scene, scenes / name
        if dst.exists():
            raise SystemExit(f"{dst} already exists")
        shutil.copytree(src, dst, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.rmtree(dst / ".agent", ignore_errors=True)  # fresh trace, fresh metas
        (dst / ".agent").mkdir()
        for meta in dst.rglob("meta.json"):
            meta.write_text(json.dumps(META_STUB, indent=2) + "\n")
        return dst, scene, name
    if level == "strategy":
        name = name or next_name(scenes / scene / "strategies", "strategy")
        dst = scenes / scene / "strategies" / name
        if dst.exists():
            raise SystemExit(f"{dst} already exists")
        (dst / ".agent").mkdir(parents=True)
        (dst / "meta.json").write_text(json.dumps(META_STUB, indent=2) + "\n")
        return dst, strategy, name
    # phase: one proposal of how to divide/enter the strategy's solve
    dst = scenes / scene / "strategies" / strategy
    if not dst.is_dir():
        raise SystemExit(f"no such strategy: {dst}")
    name = name or next_name(dst / "phases", "phase")
    phase_dir = dst / "phases" / name
    if phase_dir.exists():
        raise SystemExit(f"{phase_dir} already exists")
    phase_dir.mkdir(parents=True)
    (phase_dir / "reset").mkdir()
    (phase_dir / "reset" / "scene_default.py").write_text(RESET_PY_STUB)
    (phase_dir / "meta.json").write_text(json.dumps(META_STUB, indent=2) + "\n")
    return phase_dir, strategy, name

