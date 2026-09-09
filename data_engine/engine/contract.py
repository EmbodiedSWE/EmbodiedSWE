"""contract — the exact-artifact dataset contract, as pure logic.

The campaign's deliverable is a LADDER of EXACT, nested artifacts (1 -> 5 -> 20
-> 50 -> 200 -> 600), each one either complete or loudly INCOMPLETE — never an
overshoot reported as a feature, never a shortfall papered over with another
clock:

    <level>_set.json     exactly <level>_target replay-verified episodes per
                         authoring stage (scene 5, strategy 20, phase 50): the
                         previous rung's set whole + a round-robin fill across
                         that level's cells (select_stage_set)
    physics_set.json     exactly physics_target verified episodes = the last
                         authoring set plus the earliest harvest survivors,
                         deterministic order; surplus successes stay on disk as
                         diagnostics
    render manifest      exactly physics_target x visual_draws (episode, look)
                         items — a durable work queue: items are marked complete
                         atomically in the manifest ledger, resume renders ONLY the
                         missing ones, and shards are sized by estimated frame count
                         (a fixed episodes-per-invocation chunk let 192 x ~7,800
                         frames blow a 3 h budget in v28)

Everything here is a pure function of on-disk state — no Isaac, no subprocesses —
so the whole contract is unit-testable (see data_engine/tests/test_contract.py).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_DRAW_RE = re.compile(r"_draw\d+$")


# ---------------------------------------------------------------- code identity

def code_hash() -> str:
    """The code identity every ledger/marker records. Launchers that ship immutable
    bundles name them by commit hash and export DGEN_CODE_HASH; a plain checkout
    falls back to git. 'unknown' (no git, no env) is loud in every artifact."""
    env = os.environ.get("DGEN_CODE_HASH", "").strip()
    if env:
        return env
    p = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return p.stdout.strip() or "unknown"


# ---------------------------------------------------------------- code identity of a cell

def cell_fingerprint(gen_root: Path, cell: str) -> str:
    """Identity of the code a cell's episodes were produced by: every .py in the
    strategy dir (solve + siblings), the scene, the grader, plus the phase port and
    reset builders when the cell has a phase. Recorded into every batch meta by
    generation and compared by the ladder gate, so an episode recorded by code that
    was later edited never vouches for the edited code (pen_holder once shipped a
    set containing episodes of a pre-fix strategy while the fixed one was never gated)."""
    import hashlib
    parts = cell.split("/")
    scene, strategy = parts[0], parts[1]
    phase = parts[2] if len(parts) > 2 else None
    strat_dir = gen_root / "scenes" / scene / "strategies" / strategy
    paths = sorted(strat_dir.glob("*.py"))
    paths += [gen_root / "scenes" / scene / "scene" / "scene.py",
              gen_root / "scenes" / scene / "grader" / "grader.py"]
    if phase:
        phase_dir = strat_dir / "phases" / phase
        paths.append(phase_dir / "solve_by_phase.py")
        paths += sorted((phase_dir / "reset").glob("*.py"))
    h = hashlib.sha256()
    for q in paths:
        h.update(q.read_bytes() if q.is_file() else b"<missing>")
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- ladder sets

def _round_robin(prev: list[str], by_cell: dict[str, list[str]], groups: list[list[str]],
                 target: int, identity) -> list[str]:
    """Fill `prev` up to `target` taking one episode per cell in turn, group by
    group; `identity(ep)` keys byte-identical trajectories so a duplicate recording
    is never counted twice (spatula once shipped 4 identical seed-0 runs in a 20)."""
    picked = list(prev)
    seen_ids = {identity(e) for e in prev}
    for group in groups:
        queues = [list(by_cell[c]) for c in group]
        while len(picked) < target and any(queues):
            for q in queues:
                if len(picked) >= target:
                    break
                while q:
                    ep = q.pop(0)
                    key = identity(ep)
                    if key not in seen_ids:
                        seen_ids.add(key)
                        picked.append(ep)
                        break
    return picked


def select_stage_set(prev_set: list[str], verified: dict[str, str],
                     new_cells: set[str], target: int,
                     identity=lambda ep: ep) -> tuple[list[str], int]:
    """(the stage's set, shortfall): EXACTLY `target` episodes. The previous
    rung's set enters whole (the ladder is nested), then the fill is a
    round-robin over cells — this level's new cells first, then the rest — so
    the spread the gate demanded is what actually ships, not one wide batch of
    a single cell. `verified` maps episode path -> cell key for every episode
    that is graded successful AND replay-verified. Deterministic: episodes are
    taken in sorted path order within each cell. `identity` (default: the path)
    dedups byte-identical trajectories."""
    seen = set(prev_set)
    by_cell: dict[str, list[str]] = {}
    for ep in sorted(verified):
        if ep not in seen:
            by_cell.setdefault(verified[ep], []).append(ep)
    groups = [sorted(c for c in by_cell if c in new_cells),
              sorted(c for c in by_cell if c not in new_cells)]
    picked = _round_robin(prev_set, by_cell, groups, target, identity)
    return picked[:target], max(0, target - len(picked))


# ---------------------------------------------------------------- physics set

def dynamics_survivors(gen_root: Path) -> list[str]:
    """Verified harvest episodes, campaign-relative, in DETERMINISTIC harvest
    order (batch name, then episode name): the physics set must select the same
    episodes on every resume and every machine."""
    out = []
    for meta_p in sorted(gen_root.glob("data/batch_dynamics_*/ep_*/meta.json")):
        meta = json.loads(meta_p.read_text())
        if meta.get("success"):
            out.append(str(meta_p.parent.relative_to(gen_root)))
    return out


def select_physics_set(base_set: list[str], survivors: list[str], target: int,
                       cell_of=None, identity=lambda ep: ep) -> tuple[list[str], int]:
    """(the physics set, shortfall). The base set enters whole, then harvest
    survivors fill up to exactly `target` ROUND-ROBIN ACROSS CELLS (with `cell_of`;
    without it, in harvest order) — the first 512-env batch of one cell must not
    become 150 of the 200 (coffee: 151/200 from scene_0/strategy_0). shortfall > 0
    = the stage is INCOMPLETE by that many episodes and no set is written."""
    seen = set(base_set)
    fresh = [e for e in survivors if e not in seen]
    if cell_of is None:
        picked = list(base_set) + fresh
        return picked[:target], max(0, target - len(picked))
    by_cell: dict[str, list[str]] = {}
    for e in fresh:
        by_cell.setdefault(cell_of(e), []).append(e)
    picked = _round_robin(base_set, by_cell, [sorted(by_cell)], target, identity)
    return picked[:target], max(0, target - len(picked))


# ---------------------------------------------------------------- render manifest

def build_manifest(gen_root: Path, physics_set: list[str],
                   draws: int) -> dict:
    """The exact (episode, pass) work queue: len(physics_set) x draws items, each
    carrying the facts sharding needs (scene, estimated frames). Items start
    incomplete; mark_items_complete flips them atomically after disk verification."""
    items = []
    for ep in physics_set:
        meta = json.loads((gen_root / ep / "meta.json").read_text())
        scene = str(meta.get("cell", "scene_0/strategy_0")).split("/")[0]
        for j in range(draws):
            items.append({"episode": ep, "pass": j, "scene": scene,
                          "frames": estimate_frames(meta), "complete": False})
    return {"schema_version": 1, "code_hash": code_hash(),
            "physics_set_size": len(physics_set), "draws": draws,
            "items": items}


def estimate_frames(ep_meta: dict, trim_margin: int = -1) -> int:
    """Frames one render of this episode costs (the sharding weight). Mirrors
    replay's trim rule: a trimmed episode renders only to its sustained-success
    step plus the margin."""
    rows = int(ep_meta.get("steps") or 0)
    if trim_margin >= 0 and ep_meta.get("success_step") is not None:
        rows = min(rows, int(ep_meta["success_step"]) + trim_margin)
    return max(rows, 1)


def pass_views(ep_dir: Path, j: int) -> list[str]:
    """Views pass j owns in an episode's render contract (imgs/render_<view>.json):
    pass 0 = the declared views under their own names, pass j >= 1 = the same views
    under the _draw<j> suffix. `_probe` views (the scene-contract render probe's
    suffix) belong to no pass — a probe must never vouch for a dataset render."""
    views = [p.stem.removeprefix("render_")
             for p in (ep_dir / "imgs").glob("render_*.json")]
    if j:
        return [v for v in views if v.endswith(f"_draw{j}")]
    return [v for v in views
            if not _DRAW_RE.search(v) and not v.endswith("_probe")]


def item_complete_on_disk(gen_root: Path, item: dict) -> bool:
    """Disk evidence one (episode, pass) item finished: at least one contract json
    with its video, and NO partial video for this pass (a `.part.mp4` is a killed
    writer — replay renames on close, so its presence proves an unfinished pass)."""
    ep_dir = gen_root / item["episode"]
    j = item["pass"]
    views = pass_views(ep_dir, j)
    if not views:
        return False
    for v in views:
        if not (ep_dir / "imgs" / f"{v}.mp4").is_file():
            return False
    for part in (ep_dir / "imgs").glob("*.part.mp4"):
        stem = part.name.removesuffix(".part.mp4")
        if j:
            owned = stem.endswith(f"_draw{j}")
        else:
            owned = not _DRAW_RE.search(stem) and not stem.endswith("_probe")
        if owned:
            return False
    return True


def plan_shards(items: list[dict], budget_frames: int) -> list[dict]:
    """Group INCOMPLETE items into render invocations: one shard = one (scene, pass)
    — replay boots one scene per process and one pass = one look/suffix — packed by
    estimated frame count up to `budget_frames`. Episodes sort by frame count inside
    a group so same-length episodes chunk together (replay pads mixed chunks to the
    longest)."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for it in items:
        if not it.get("complete"):
            groups.setdefault((it["scene"], it["pass"]), []).append(it)
    shards = []
    # pass-major: every episode's pass 0 (the declared cameras) renders before any
    # pass 1 — a clock cut then leaves complete looks, not scene_0's three looks
    # while scene_1..4's base episodes were never rendered (coffee, 228/600)
    for (scene, j), group in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        group.sort(key=lambda it: (it["frames"], it["episode"]))
        cur: list[dict] = []
        cur_frames = 0
        for it in group:
            if cur and cur_frames + it["frames"] > budget_frames:
                shards.append({"scene": scene, "pass": j,
                               "episodes": [x["episode"] for x in cur],
                               "frames": cur_frames})
                cur, cur_frames = [], 0
            cur.append(it)
            cur_frames += it["frames"]
        if cur:
            shards.append({"scene": scene, "pass": j,
                           "episodes": [x["episode"] for x in cur],
                           "frames": cur_frames})
    return shards


def mark_items_complete(manifest: dict, gen_root: Path,
                        episodes: list[str], j: int) -> int:
    """Flip `complete` for the given (episode, pass) items IF the disk evidence
    holds; returns how many flipped. The caller persists the manifest atomically —
    the manifest is the record, the disk is the evidence at marking time."""
    eps = set(episodes)
    flipped = 0
    for it in manifest["items"]:
        if (it["pass"] == j and it["episode"] in eps and not it["complete"]
                and item_complete_on_disk(gen_root, it)):
            it["complete"] = True
            flipped += 1
    return flipped
