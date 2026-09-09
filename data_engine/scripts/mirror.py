"""mirror — copy a campaign to durable storage, payloads only for what the dataset needs.

    python data_engine/scripts/mirror.py <work>/<task> <mirror>/<task> [--renders-only]

Copies everything under the task dir EXCEPT the payload (traj.npz, imgs/) of episodes
that no delivered set references: the agents' experimental pools ran to 30k episodes
(70 GB) per task in v2 and filled the shared volume; their meta.json (verdicts) are kept,
their trajectories are diagnostics that are regenerable. Referenced = every episode in
any <level>_set.json / physics_set.json of any campaign under data_gen/, plus any
episode of a nominal or wide-probe batch (the ladder's first rung and the pre-check
evidence), plus EVERY episode of a batch whose stage has not delivered its set yet (the
candidates a resumed campaign gates and replays — a mirror that dropped them would leave
a resume unable to prove anything). Deleted source files are removed from the mirror (rsync --delete semantics
for the copied subset) EXCEPT render output (`imgs/`), which render workers push to the
same mirror and the campaign pulls back — never deleted by anyone. `--renders-only` is
the render workers' mode: push `imgs/` files and nothing else (a worker must never
overwrite the campaign's ledger, status or manifest). Idempotent; safe every few minutes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def referenced_episodes(gen: Path) -> set[Path]:
    keep: set[Path] = set()
    for f in gen.glob("*_set.json"):
        try:
            keep.update(gen / e for e in json.loads(f.read_text()).get("episodes", []))
        except (OSError, ValueError) as exc:
            print(f"[mirror] unreadable {f}: {exc}", flush=True)
    for batch in gen.glob("data/batch_nominal_*"):
        keep.update(p for p in batch.glob("ep_*") if p.is_dir())
    for batch in gen.glob("data/batch_wide_probe_*"):
        keep.update(p for p in batch.glob("ep_*") if p.is_dir())
    # candidates of a stage still in progress: every episode of a batch whose cell's
    # level has no delivered set yet (the base cell's own batches feed the physics set)
    for bmeta in gen.glob("data/*/meta.json"):
        try:
            cell = json.loads(bmeta.read_text()).get("cell") or ""
        except (OSError, ValueError) as exc:
            print(f"[mirror] unreadable {bmeta}: {exc}", flush=True)
            continue
        parts = cell.split("/")
        if not cell:
            continue
        # same attribution as the orchestrator's _authored_at (a cell may count at two levels)
        sets = ([] if parts[0] == "scene_0" else ["scene_set.json"]) \
            + (["strategy_set.json"] if parts[0] == "scene_0" and len(parts) == 2 and parts[1] != "strategy_0" else []) \
            + (["phase_set.json"] if len(parts) == 3 else [])
        if not sets:
            sets = ["physics_set.json"]        # the base cell's own batches feed the physics set
        if not all((gen / s).is_file() for s in sets):
            keep.update(p for p in bmeta.parent.glob("ep_*") if p.is_dir())
    return keep


def is_render(path: Path, task_dir: Path) -> bool:
    return "imgs" in path.relative_to(task_dir).parts


def wanted(path: Path, task_dir: Path, keep: set[Path], renders_only: bool = False) -> bool:
    """Is this source file part of the mirror?"""
    if renders_only:
        return is_render(path, task_dir)
    rel = path.relative_to(task_dir).parts
    # <work>/<task>/data_gen/<gen>/data/<batch>/ep_NNNN/<...>
    if len(rel) >= 6 and rel[0] == "data_gen" and rel[2] == "data" and rel[4].startswith("ep_"):
        ep_dir = task_dir.joinpath(*rel[:5])
        if ep_dir in keep or rel[5] == "meta.json":
            return True
        return False
    if "__pycache__" in rel:
        return False
    return True


def mirror(task_dir: Path, out_dir: Path, renders_only: bool = False) -> tuple[int, int]:
    keep: set[Path] = set()
    for gen in (task_dir / "data_gen").glob("*"):
        if gen.is_dir():
            keep |= referenced_episodes(gen)
    copied = removed = 0
    seen: set[Path] = set()
    for src in task_dir.rglob("*"):
        dst = out_dir / src.relative_to(task_dir)
        if src.is_symlink() and not renders_only:
            # Recreate symlinks AS symlinks (same link text): the baked scene's `assets/<sub>`
            # entries are relative links into /repo/robobench (initialization.py:216-225).
            # They were skipped here (a dir symlink is not a file), so a campaign RESUMED from
            # the mirror had no assets and every Isaac boot died on `default_ground.usd`
            # (bulb, 2026-09-07). The link text is relative to the campaign dir, so it is
            # dangling ON the mirror and valid again once copied back under /work/<task>.
            seen.add(dst)
            try:
                target = os.readlink(src)
                if dst.is_symlink() and os.readlink(dst) == target:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                os.symlink(target, dst)
                copied += 1
            except OSError as exc:
                print(f"[mirror] FAILED symlink {src}: {exc}", flush=True)
            continue
        if not src.is_file() or not wanted(src, task_dir, keep, renders_only):
            continue
        seen.add(dst)
        try:
            st = src.stat()
            if dst.exists() and dst.stat().st_size == st.st_size and dst.stat().st_mtime >= st.st_mtime:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied += 1
        except OSError as exc:
            print(f"[mirror] FAILED {src}: {exc}", flush=True)
    if not renders_only:
        # Prune ONLY inside campaigns this pod owns (a data_gen/<gen> that exists at the
        # source). The mirror holds every campaign ever run for the task; a fresh campaign's
        # work dir holds only its own, so "absent at the source" was true of EVERY file of
        # every earlier campaign — the gen_o50_v3 launch (2026-09-06) deleted thousands of
        # gen_o50_v2 files per task in its first three-minute lap before the loops were
        # stopped. Files outside data_gen/ are per-task and always present at the source.
        owned = {g.name for g in (task_dir / "data_gen").glob("*") if g.is_dir()}
        top = {p.name for p in task_dir.iterdir()}
        for dst in out_dir.rglob("*"):
            if not dst.is_file() or dst in seen or is_render(dst, out_dir):
                continue
            rel = dst.relative_to(out_dir).parts
            # Prune only inside top-level entries THIS pod has, and inside data_gen only the
            # campaigns it owns. The first scoping (data_gen only) still let a fresh pod
            # empty `attempt1_precheck_failed/` — an archive placed beside data_gen by the
            # operator, which no pod has at its source (2026-09-06, five tasks' diagnostics).
            if rel[0] not in top or (rel[0] == "data_gen" and (len(rel) < 2 or rel[1] not in owned)):
                continue                       # not this pod's to prune
            src = task_dir / dst.relative_to(out_dir)
            if not src.exists():               # deleted at the source (e.g. a pruned cell)
                dst.unlink()
                removed += 1
    return copied, removed


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    c, r = mirror(Path(args[0]), Path(args[1]), renders_only="--renders-only" in sys.argv)
    print(f"[mirror] {c} files copied, {r} removed", flush=True)
