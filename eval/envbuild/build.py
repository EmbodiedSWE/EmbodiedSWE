"""Orchestrator: CLI intent -> a built experiment folder.

    experiments/<name>/
    ├── resolved.json  MANIFEST        (provenance receipt)
    └── stages/01_<scene>_<robot>/
        ├── bench/                     (mounts ro at /bench)
        └── describe.md                (scene/robot/controller, harvested at boot)

The build produces the WORLD only. The per-run /task folder (instructions,
task text, rules, hints) is assembled by the run launcher via
envbuild/prompts.py — one built world hosts many prompt conditions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from . import extract, manifest, patch, validate
from .resolve import resolve_preset


@dataclass
class StageSpec:
    scene: str
    robot: str
    controller: str | None = None
    suite: str = "assembly"

    @classmethod
    def parse(cls, arg: str) -> "StageSpec":
        """'scene:robot[:controller]'"""
        parts = arg.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(f"--stage wants scene:robot[:controller], got '{arg}'")
        return cls(scene=parts[0], robot=parts[1], controller=parts[2] if len(parts) == 3 else None)


def build_experiment(
    name: str,
    stages: list[StageSpec],
    set_states: bool = True,
    out_root: Path = Path("experiments"),
) -> Path:
    exp_dir = (out_root / name).resolve()
    if exp_dir.exists():
        raise SystemExit(f"refusing to overwrite existing {exp_dir}")

    # resolve everything FIRST — fail before any copying
    presets = [resolve_preset(s.scene, s.robot, s.controller, s.suite) for s in stages]

    records = []
    for i, (s, preset) in enumerate(zip(stages, presets), start=1):
        stage_dir = exp_dir / "stages" / f"{i:02d}_{s.scene}_{s.robot}"
        bench = stage_dir / "bench"
        print(f"=== stage {stage_dir.name}: {preset} ===")
        assets = extract.minimal_tree(bench, s.suite, s.scene, s.robot)
        if not set_states:
            patch.disable_set_states(bench)
        # boot validation is NOT optional: no bundle ships unbooted
        describe_text = validate.boot_preset(bench, preset)
        (stage_dir / "describe.md").write_text(describe_text + "\n")
        records.append({
            "dir": stage_dir.name, "preset": preset, "assets": assets,
            "boot_checked": True, "tree_sha256": manifest.tree_hash(bench),
        })

    manifest.write_receipt(exp_dir, {
        "name": name, "argv": sys.argv, "set_states": set_states, "stages": records,
    })
    print(f"built: {exp_dir}")
    return exp_dir
