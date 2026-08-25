"""Orchestrator: CLI intent -> a built experiment folder.

    experiments/<name>/
    ├── resolved.json  MANIFEST        (provenance receipt)
    └── stages/01_<scene>_<robot>/
        ├── bench/                     (mounts ro at /bench)
        └── describe.md                (scene/robot/controller, harvested at boot)

The build produces the WORLD only. The per-run /task folder (instructions,
task text, rules, skills, tool docs) is assembled by the run launcher via
envbuild/prompts.py — one built world hosts many prompt conditions.

Code gating is the exception: a blocked feature is patched into the extracted tree HERE, at
build time, because /bench mounts read-only and the agent must not be able to revert it. The
condition's `features:` names which patches apply (design note §2: a built world already
embodies its condition; ablating a feature means two builds, never one world toggled at run
time).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from . import extract, manifest, patch, validate
from .condition import Condition
from .resolve import resolve_preset


@dataclass
class StageSpec:
    scene: str
    robot: str
    controller: str | None = None
    suite: str = "assembly"

    @classmethod
    def parse(cls, arg: str) -> "StageSpec":
        """'[suite.]scene:robot[:controller]' — the suite defaults to assembly, and is
        written the way preset names write it, so packing.crate:franka:osc works."""
        parts = arg.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(f"--stage wants [suite.]scene:robot[:controller], got '{arg}'")
        suite, _, scene = parts[0].rpartition(".")
        return cls(scene=scene, robot=parts[1],
                   controller=parts[2] if len(parts) == 3 else None,
                   suite=suite or "assembly")


def build_experiment(
    name: str,
    stages: list[StageSpec],
    condition: Condition,
    out_root: Path = Path("experiments"),
    seed: int = 0,
) -> Path:
    """Build one world under `condition`. Which code features are blocked comes from the
    condition's `features:`; each blocked feature maps to one `patch.py` function."""
    exp_dir = (out_root / name).resolve()
    if exp_dir.exists():
        raise SystemExit(f"refusing to overwrite existing {exp_dir}")
    set_states = condition.enabled("set_states")
    freeze_controller = not condition.enabled("control_mode")

    # resolve everything FIRST — fail before any copying
    presets = [resolve_preset(s.scene, s.robot, s.controller, s.suite) for s in stages]

    records = []
    for i, (s, preset) in enumerate(zip(stages, presets), start=1):
        stage_dir = exp_dir / "stages" / f"{i:02d}_{s.scene}_{s.robot}"
        bench = stage_dir / "bench"
        print(f"=== stage {stage_dir.name}: {preset} ===")
        assets = extract.minimal_tree(bench, s.suite, s.scene, s.robot)
        # One patch per feature the condition blocks. `condition.patches()` names them, so a new
        # gated feature is a FEATURE_PATCHES entry plus a patch function — no branch here.
        for fn_name in condition.patches():
            getattr(patch, fn_name)(bench)
        # boot validation is NOT optional: no bundle ships unbooted. When it fails on
        # an asset the extractor could not see, supply it and boot again — asset
        # detection is a heuristic, so the prover is also what repairs it. The boot runs
        # at the build seed, so describe.md harvests the same initial condition grading uses.
        # 12 boots, not 4: scenes that check their own assets reveal exactly ONE missing
        # directory per boot (the FileNotFoundError path), and the richer scenes reference
        # more than three undetected asset dirs — tool_packing (chest, stapler, scissors,
        # knife, ...) and coffee (machine, cup, tray, capsule, ...) both exhausted the old
        # cap while repairing perfectly well (2026-08-14).
        for _ in range(12):
            try:
                describe_text = validate.boot_preset(bench, preset, seed=seed)
                break
            except validate.MissingAssets as exc:
                supplied = extract.copy_missing(bench, exc.paths)
                if not supplied:
                    raise
                print(f"[build] assets the scene file does not name: {', '.join(supplied)}")
                assets += supplied
        else:
            raise SystemExit(f"'{preset}' still misses assets after repair; see above")
        (stage_dir / "describe.md").write_text(describe_text + "\n")
        records.append({
            "dir": stage_dir.name, "preset": preset, "assets": assets,
            "control_mode_frozen": freeze_controller, "boot_checked": True,
            "tree_sha256": manifest.tree_hash(bench),
        })

    manifest.write_receipt(exp_dir, {
        "name": name, "argv": sys.argv, "set_states": set_states, "boot_seed": seed,
        # The condition is part of the world's provenance: a built world embodies it, so the
        # receipt cites the exact file and its selections.
        "condition": condition.as_record(),
        "stages": records,
    })
    print(f"built: {exp_dir}")
    return exp_dir
