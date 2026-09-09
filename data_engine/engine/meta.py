"""meta — the derived meta.json cascade, recomputed from the raw data pool.

Scene, strategy and phase meta.json files are CACHES: every number in them is
re-derivable from the batch metas under data/. refresh_metas() recomputes the
whole cascade — the launcher calls it after each batch, and it can be run any
time to rebuild the metas from the pool alone.

Attribution is by the batch meta's `cell` ("scene_0/strategy_0[/phase_1]"):
a scene meta aggregates every batch in that scene, a strategy meta every batch
of that strategy (any phase), a phase meta only its own batches.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def refresh_metas(gen_root: str | Path) -> None:
    gen_root = Path(gen_root)
    batches = [json.loads(p.read_text()) for p in sorted(gen_root.glob("data/*/meta.json"))]

    def write(level_dir: Path, match) -> None:
        sel = [m for m in batches if match(m.get("cell", "").split("/"))]
        eps = sum(m["episodes"] for m in sel)
        ok = sum(m["successes"] for m in sel)
        try:
            # atomic: concurrent refreshes (parallel batches both finishing) must
            # never leave a half-written cache for a reader to json-crash on
            tmp = level_dir / "meta.json.tmp"
            tmp.write_text(json.dumps({
                "episodes": eps, "successes": ok,
                "success_rate": round(ok / eps, 4) if eps else None,
                "batches": [m["batch"] for m in sel],
                "refreshed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, indent=2) + "\n")
            tmp.replace(level_dir / "meta.json")
        except OSError:
            pass  # a fenced session's read-only base cell: its meta stays stale; the pool is truth

    for scene_dir in sorted(gen_root.glob("scenes/scene_*")):
        # only cells: init also links every sibling suite under scenes/<suite> (cross-suite asset
        # borrowing) and a glob("scenes/*") wrote meta.json caches INTO the live repo suites
        if scene_dir.is_symlink() or not scene_dir.is_dir():
            continue
        s = scene_dir.name
        write(scene_dir, lambda c, s=s: c[:1] == [s])
        for strat_dir in sorted(scene_dir.glob("strategies/*")):
            t = strat_dir.name
            write(strat_dir, lambda c, s=s, t=t: c[:2] == [s, t])
            for phase_dir in sorted(strat_dir.glob("phases/*")):
                p = phase_dir.name
                write(phase_dir, lambda c, s=s, t=t, p=p: c == [s, t, p])
