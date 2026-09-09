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
SIM_GEN = REPO / "sim_gen"

# The generated-task corpora hold more than the scene, and the rest states the answer: TASK.md
# writes out the rubric and the intended strategy, smoke.py the rejection battery, and a stale
# __pycache__ can still hold a compiled solve.py. Only scene.py is agent-facing.
SIMGEN_SCENE_FILES = ("scene.py", "__init__.py")

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
# `assets / "name"` (or `ASSETS / "name"`): a subtree (assets/<sub>/) or one file straight under
# assets/ (assets/<file>.usd)
ASSET_REF_RE = re.compile(r'assets"?\s*/\s*"([A-Za-z0-9_\-.]+)"', re.I)


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

    if suite == "simgen":
        scene_files = [_stage_simgen_scene(dst, scene)]
    else:
        scene_files = list((dst / "robobench" / "suites" / suite / "scenes").glob(f"*{scene}*.py"))
        if not scene_files:  # a cfg-only variant name (EnvCfg.variant): glob its real scene's file
            real = _scene_of_variant(suite, scene)
            scene_files = list((dst / "robobench" / "suites" / suite / "scenes").glob(f"*{real}*.py"))
    if not scene_files:
        raise SystemExit(f"no scene file matching '*{scene}*.py' in suite '{suite}'")
    scene_assets: set[str] = set()
    suite_assets = SRC / "suites" / suite / "assets"
    entries = [e.name for e in suite_assets.iterdir()] if suite_assets.is_dir() else []
    for f in scene_files:
        text = f.read_text()
        scene_assets |= set(ASSET_REF_RE.findall(text))
        # a name reached through a constant or cfg field (`ASSETS / self.KNIFE`, `assets / cfg.food`)
        # still appears as a string literal somewhere in the scene file: any quoted literal that is an
        # entry of the suite's assets/ counts (a small over-approximation beats a boot per miss)
        scene_assets |= {e for e in entries if re.search(rf'["\']{re.escape(e)}["\']', text)}

    copied = []
    for sub in sorted(scene_assets):
        src = SRC / "suites" / suite / "assets" / sub
        out = dst / "robobench" / "suites" / suite / "assets" / sub
        if src.is_dir():
            shutil.copytree(src, out)
            copied.append(f"suites/{suite}/assets/{sub}")
        elif src.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            copied.append(f"suites/{suite}/assets/{sub}")
    for r in sorted(set(ROBOT_ASSET_ALIASES.get(robot, [robot]))):
        src = SRC / "robots" / "assets" / r
        if src.is_dir():
            shutil.copytree(src, dst / "robobench" / "robots" / "assets" / r)
            copied.append(f"robots/assets/{r}")

    _scrub_comments(dst / "robobench")
    if (dst / "sim_gen").is_dir():
        _scrub_comments(dst / "sim_gen")
    return copied


def _scene_of_variant(suite: str, name: str) -> str:
    """The SCENES name behind a preset's scene segment `name` — identical unless the segment is a
    cfg-only variant (`EnvCfg.variant`, e.g. `slice_banana` -> `slice`), read off the registered
    EnvCfg itself so the mapping can never drift from the convention."""
    import robobench

    robobench.discover()  # app-free
    from robobench.core.registries import ENVS

    for env_name in ENVS.list():
        parts = env_name.split(".")
        if parts[0] == suite and parts[1] == name:
            return ENVS.get(env_name)().scene
    raise SystemExit(f"no registered env under '{suite}.{name}' — nothing to extract a scene for")


def _stage_simgen_scene(dst: Path, scene: str) -> Path:
    """Stage the one generated task scene that registers `scene`; return its staged file.

    The simgen suite package is only a shim (robobench/suites/simgen/__init__.py): the scenes
    themselves are generated task packages living OUTSIDE robobench, at
    sim_gen/tasks*/<task>/scene.py, so the glob over suites/simgen/scenes finds nothing. They
    are copied as a sibling of robobench/, which is where the shim looks (its SIM_GEN_ROOT is
    parents[3]/"sim_gen"), so the shim needs no knowledge of this.

    Newest corpus wins on a name collision, matching the shim's own resolution order. Only
    SIMGEN_SCENE_FILES travel — everything else in a task folder gives the answer away.
    """
    corpora = ([SIM_GEN / "tasks"] if (SIM_GEN / "tasks").is_dir() else []) + sorted(
        p for p in SIM_GEN.glob("tasks_*") if p.is_dir())
    hits = [d for corpus in corpora for d in sorted(corpus.iterdir())
            if (d / "scene.py").is_file()
            and re.search(rf'SCENES\.register\(\s*["\']{re.escape(scene)}["\']',
                          (d / "scene.py").read_text())]
    if not hits:
        raise SystemExit(f"no generated task registers scene '{scene}' under {SIM_GEN}")
    task = hits[-1]
    out = dst / "sim_gen" / task.parent.name / task.name
    out.mkdir(parents=True)
    (dst / "sim_gen" / "__init__.py").write_bytes((SIM_GEN / "__init__.py").read_bytes())
    for fname in SIMGEN_SCENE_FILES:
        if (task / fname).is_file():
            shutil.copy2(task / fname, out / fname)
    if len(hits) > 1:
        print(f"[extract] scene '{scene}' exists in {len(hits)} corpora; staged {task.parent.name}")
    print(f"[extract] simgen scene: {task.parent.name}/{task.name}/scene.py")
    _assert_no_solutions(dst / "sim_gen")
    return out / "scene.py"


def _assert_no_solutions(root: Path) -> None:
    """Nothing under `root` may state how the task is solved."""
    banned = [str(p) for p in root.rglob("*")
              if p.is_file() and p.name not in SIMGEN_SCENE_FILES]
    if banned:
        raise SystemExit(f"LEAK: a staged generated task carries more than its scene: {banned}")


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
        if src.is_dir():
            shutil.copytree(src, out)
        elif src.is_file():  # a single file straight under assets/ (e.g. cutting's kitchen_island.usd)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
        else:
            print(f"[extract] no source for {sub} in {SRC}")
            continue
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
