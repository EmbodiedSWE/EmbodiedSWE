"""init — bake a data_gen campaign alongside an eval run.

Scaffolds `<run_dir>/data_gen/<gen_name>/` from the run's own artifacts: run.json
names the preset, workspace/solution/ is the solve, and the scene + grader + assets
come from the live repo suite — ONE tree, so scene, grader and runtime core are
consistent by construction. If the run's stage bench is present, init diffs the
scene against it and warns on drift; a nominal batch is the real check.

    data_gen/<gen_name>/
    ├─ gen.yaml                     campaign facts: preset, provenance, git sha
    ├─ scenes/scene_0/              the unmodified benchmark scene — nominal control
    │   ├─ meta.json
    │   ├─ scene/scene.py           repo suite scene copy; registrations renamed *_local so
    │   │                           the copy loads beside the suite and EDITS TAKE EFFECT
    │   ├─ assets/<sub> -> …        symlinks, only the asset subdirs this scene references
    │   │                           (scenes resolve assets relative to their own file)
    │   ├─ grader/grader.py         suite grader copy, re-pointed at the local scene class
    │   └─ strategies/strategy_0/
    │       ├─ meta.json
    │       ├─ solve.py             the eval solution (whole solution folder) — never modified
    │       └─ (phases/, solve_by_phase.py)
    │                               OPTIONAL — created together by an L3 session; until
    │                               then the solve runs from the scene's own reset
    └─ data/                        raw episodes only; batches land here

Everything is resolved statically (regex over suite sources) — init never boots Isaac.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

META_STUB = {"episodes": 0, "successes": 0, "success_rate": None, "batches": [], "refreshed": None}


# ---- resolution ----------------------------------------------------------------------------------

def resolve_run(run_dir: Path) -> dict:
    """The facts init needs from the eval run folder."""
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        raise SystemExit(f"not an eval run (no run.json): {run_dir}")
    run = json.loads(run_json.read_text())
    solution = run_dir / "workspace" / "solution"
    if not (solution / "solve.py").is_file():
        raise SystemExit(f"no delivered solution: {solution}/solve.py missing")
    stage_bench = run_dir.parents[1] / "stages" / run.get("stage", "") / "bench" / "robobench"
    return {"preset": run["preset"], "stage": run.get("stage"), "solution": solution,
            "stage_bench": stage_bench if stage_bench.is_dir() else None}


def resolve_suite(preset: str) -> dict:
    """preset name -> source files, statically (never boots Isaac).

    Scene AND grader come from the live repo suite — one tree, so scene, grader and
    runtime core are consistent by construction. Drift since the eval run is the
    nominal batch's job to catch (and init warns if the stage bench disagrees).
    """
    suite, scene = preset.split(".")[0:2]
    scenes_dir = REPO_ROOT / "robobench" / "suites" / suite / "scenes"
    scene_path = None
    for p in sorted(scenes_dir.glob("*.py")):
        if re.search(rf'@SCENES\.register\("{re.escape(scene)}"\)', p.read_text()):
            scene_path = p
            break
    if scene_path is None:
        raise SystemExit(f"no suite scene registers '{scene}' under {scenes_dir}")

    grader_init = REPO_ROOT / "robobench" / "suites" / suite / "grader" / "__init__.py"
    grader_path = None
    if grader_init.is_file():
        text = grader_init.read_text()
        m = re.search(rf'"{re.escape(scene)}":\s*(\w+)', text)
        if m:
            im = re.search(rf"from \.(\w+) import [\w, ]*\b{m.group(1)}\b", text)
            if im:
                grader_path = grader_init.parent / f"{im.group(1)}.py"
    if grader_path is None or not grader_path.is_file():
        raise SystemExit(f"no grader registered for scene '{scene}' in {grader_init}")

    return {"suite": suite, "scene": scene, "scene_path": scene_path, "grader_path": grader_path}


# ---- bakes ---------------------------------------------------------------------------------------

# `assets / "<sub>"` references in a scene file -> the top-level asset dirs it needs
# (same pattern as eval/envbuild/extract.py)
ASSET_REF_RE = re.compile(r'assets"?\s*/\s*"([A-Za-z0-9_\-]+)"')


def bake_scene(src: Path, dst: Path) -> None:
    """Copy the suite scene, renaming every SCENES registration to `<name>_local` —
    generation loads the copy beside the suite's registrations (discover() is needed
    for the preset binding, and the registry raises on duplicate names)."""
    text = src.read_text()
    for n in re.findall(r'@SCENES\.register\("([\w.]+)"\)', text):
        text = text.replace(f'@SCENES.register("{n}")', f'@SCENES.register("{n}_local")')
    dst.write_text(text)


def bake_grader(src: Path, dst: Path) -> None:
    """Copy the suite grader, re-pointing its scene-class import at the local copy,
    so its isinstance check matches envs built from scene.py."""
    text = src.read_text()
    m = re.search(r"^from robobench\.suites\.\w+\.scenes[\w.]* import ([\w, ]+)$", text, re.M)
    if m is None:
        raise SystemExit(f"cannot find the scene-class import to re-point in {src}")
    classes = [c.strip() for c in m.group(1).split(",")]
    shim = "\n".join(
        ["# baked by data_engine init: grade against the LOCAL scene copy (../scene/scene.py),",
         "# exec'd at most once per process under a fixed module key.",
         "import importlib.util as _ilu",
         "import sys as _sys",
         "from pathlib import Path as _Path",
         "",
         'if "datagen_local_scene" not in _sys.modules:',
         "    import robobench as _rb",
         "    _rb.discover()",
         '    _p = _Path(__file__).resolve().parent.parent / "scene" / "scene.py"',
         '    _spec = _ilu.spec_from_file_location("datagen_local_scene", _p)',
         "    _m = _ilu.module_from_spec(_spec)",
         '    _sys.modules["datagen_local_scene"] = _m',
         "    _spec.loader.exec_module(_m)"]
        + [f'{c} = _sys.modules["datagen_local_scene"].{c}' for c in classes]
    )
    dst.write_text(text.replace(m.group(0), shim))


# ---- the verb ------------------------------------------------------------------------------------

def init(run_dir: str | Path, name: str | None = None, force: bool = False) -> Path:
    run_dir = Path(run_dir).resolve()
    if not name:
        name = datetime.now().strftime("gen_%Y%m%d_%H%M%S")
    run = resolve_run(run_dir)
    suite = resolve_suite(run["preset"])
    # drift check: if the run's stage bench is present, compare its scene source
    if run["stage_bench"] is not None:
        staged = run["stage_bench"] / suite["scene_path"].relative_to(REPO_ROOT / "robobench")
        if staged.is_file() and staged.read_text() != suite["scene_path"].read_text():
            print(f"WARNING: {suite['scene_path'].name} moved since this run solved "
                  f"(stage {run['stage']}) — run a nominal batch to check")

    gen = run_dir / "data_gen" / name
    if gen.exists():
        if not force:
            raise SystemExit(f"{gen} already exists (pass --force to overwrite)")
        shutil.rmtree(gen)

    scene0 = gen / "scenes" / "scene_0"
    strat0 = scene0 / "strategies" / "strategy_0"
    for d in (scene0 / "scene", scene0 / "grader", strat0, gen / "data"):
        d.mkdir(parents=True, exist_ok=True)

    bake_scene(suite["scene_path"], scene0 / "scene" / "scene.py")
    # suite scenes resolve assets as `Path(__file__).parents[1] / "assets" / "<sub>"` — for
    # the copy that is scene_0/assets/. Link only the subdirs this scene references (the
    # same under-approximation as eval's extractor), each a relative symlink to the suite's.
    subs = sorted(set(ASSET_REF_RE.findall((scene0 / "scene" / "scene.py").read_text())))
    suite_assets = suite["scene_path"].parents[1] / "assets"
    for sub in subs:
        if (suite_assets / sub).is_dir():
            (scene0 / "assets").mkdir(exist_ok=True)
            (scene0 / "assets" / sub).symlink_to(
                os.path.relpath(suite_assets / sub, scene0 / "assets"), target_is_directory=True)
    bake_grader(suite["grader_path"], scene0 / "grader" / "grader.py")

    # the delivered solution folder becomes strategy_0 — solve.py plus its siblings
    for p in run["solution"].iterdir():
        if p.name == "__pycache__":
            continue
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, strat0 / p.name)

    for d in (scene0, strat0):
        (d / "meta.json").write_text(json.dumps(META_STUB, indent=2) + "\n")

    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    gen_yaml = "\n".join([
        f"preset: {run['preset']}",
        *([f"stage: {run['stage']}"] if run["stage"] else []),
        f"source_run: ../..            # the eval run this campaign multiplies",
        f"solution: workspace/solution # what strategy_0 was ported from",
        f"scene_source: {os.path.relpath(suite['scene_path'], REPO_ROOT)}   # live repo: one tree with grader + core",
        f"grader_source: {os.path.relpath(suite['grader_path'], REPO_ROOT)}",
        f"created: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"git_sha: {sha}",
    ]) + "\n"
    (gen / "gen.yaml").write_text(gen_yaml)

    print(f"baked {gen.relative_to(run_dir.parent.parent)}")
    for p in sorted(gen.rglob("*")):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(gen)
        print(f"  {rel}{'/' if p.is_dir() else ''}")
    return gen
