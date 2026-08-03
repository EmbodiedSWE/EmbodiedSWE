"""Minimal robobench tree for one stage: over-approximate code, under-approximate assets.

All .py of core/controllers/robots/<suite> ship (their __init__ files hard-import
everything, and .py is cheap); other suites and every smokes/ dir are dropped;
assets are copied only for what the stage's scene file references plus the
stage's robot (composites resolved). Solve-provenance comments are scrubbed.
Asset detection is a heuristic — validate.boot_preset() is what proves it.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "robobench"

# composite robots pull other robots' assets (see robobench/robots/multi.py)
ROBOT_ASSET_ALIASES = {
    "aloha": ["wxai"],
    "bimanual_franka": ["franka"],
    "bimanual_piper": ["piper"],
    "null": [],
}

LEAK_RE = re.compile(
    r"experiments/|solve[-_ ]?(verified|calibrated)|BUILD_LOG|SESSION_TRACE|solve_four", re.I
)
ASSET_REF_RE = re.compile(r'assets"?\s*/\s*"([A-Za-z0-9_\-]+)"')


def minimal_tree(dst: Path, suite: str, scene: str, robot: str, keep_smokes: bool = False) -> list[str]:
    """Copy the minimal tree into dst/robobench; return the asset subtrees copied."""

    def ignore(dirpath: str, names: list[str]) -> set[str]:
        d = Path(dirpath)
        skip = {n for n in names if n == "__pycache__" or n.endswith(".egg-info")}
        if d == SRC / "suites":
            skip |= {n for n in names if (d / n).is_dir() and n != suite}
        if d.name == suite and d.parent == SRC / "suites" and not keep_smokes:
            skip |= {n for n in names if n == "smokes"}
        # graders are privileged: never in an agent-facing tree, no opt-out
        if d.parent == SRC / "suites":
            skip |= {n for n in names if n == "grader"}
        if d == SRC / "robots" and "assets" in names:
            skip.add("assets")
        if d == SRC / "suites" / suite and "assets" in names:
            skip.add("assets")
        return skip

    shutil.copytree(SRC, dst / "robobench", ignore=ignore)

    scene_files = list((dst / "robobench" / "suites" / suite / "scenes").glob(f"*{scene}*.py"))
    if not scene_files:
        raise SystemExit(f"no scene file matching '*{scene}*.py' in suite '{suite}'")
    scene_assets: set[str] = set()
    for f in scene_files:
        scene_assets |= set(ASSET_REF_RE.findall(f.read_text()))

    copied = []
    for sub in sorted(scene_assets):
        src = SRC / "suites" / suite / "assets" / sub
        if src.is_dir():
            shutil.copytree(src, dst / "robobench" / "suites" / suite / "assets" / sub)
            copied.append(f"suites/{suite}/assets/{sub}")
    for r in sorted(set(ROBOT_ASSET_ALIASES.get(robot, [robot]))):
        src = SRC / "robots" / "assets" / r
        if src.is_dir():
            shutil.copytree(src, dst / "robobench" / "robots" / "assets" / r)
            copied.append(f"robots/assets/{r}")

    _scrub_comments(dst / "robobench")
    return copied


def copy_missing(dst: Path, paths: list[str]) -> list[str]:
    """Copy the asset subtrees behind `paths`; return what was added.

    minimal_tree only sees the directories spelled out in the scene's Python, so
    assets that reference each other are invisible to it (ikea's leg.usd reaches
    for ../factory/factory_nut_m16.usd, and nothing in the scene file says
    "factory"). boot_preset reports what failed to resolve and this brings those
    trees over, keeping to the same granularity as minimal_tree: assets/<sub>.
    """
    added = []
    for p in paths:
        if Path(p).exists():  # an earlier subtree in this pass already brought it
            continue
        try:
            rel = Path(p).relative_to(dst)
        except ValueError:
            print(f"[extract] missing asset outside the tree, cannot supply it: {p}")
            continue
        if "assets" not in rel.parts:
            print(f"[extract] missing asset is not under an assets/ dir: {rel}")
            continue
        sub = Path(*rel.parts[: rel.parts.index("assets") + 2])
        src, out = SRC.parent / sub, dst / sub
        if out.exists():
            print(f"[extract] {sub} is already here yet {Path(p).name} is not — check the source")
            continue
        if not src.is_dir():
            print(f"[extract] no source for {sub} in {SRC}")
            continue
        shutil.copytree(src, out)
        added.append(str(sub.relative_to("robobench")))
    return added


def _scrub_comments(root: Path) -> None:
    """Drop full-line comments matching LEAK_RE; truncate matching inline ones; fail on survivors."""
    for f in root.rglob("*.py"):
        out, changed = [], False
        for line in f.read_text().splitlines(keepends=True):
            if LEAK_RE.search(line):
                code, sep, comment = line.partition("#")
                if sep and LEAK_RE.search(comment) and not LEAK_RE.search(code):
                    changed = True
                    if code.strip():
                        out.append(code.rstrip() + "\n")
                    continue
            out.append(line)
        if changed:
            f.write_text("".join(out))
    leaks = [str(f) for f in root.rglob("*.py") if LEAK_RE.search(f.read_text())]
    if leaks:
        raise SystemExit(f"LEAK: references survive in code, review by hand: {leaks[:5]}")
