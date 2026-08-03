"""Seed catalog — curated RoboVerse task files suitable as construction seeds.

Seeds are handed to the construction agent as SOURCE TEXT (the semantics to mutate
away from); the new task is implemented from scratch against sim_gen.core, so no
metasim/RoboVerse dependency leaks into generated tasks.
"""

from __future__ import annotations

from pathlib import Path

SIM_GEN_ROOT = Path(__file__).resolve().parent.parent
ROBOVERSE_TASKS = SIM_GEN_ROOT / "RoboVerse" / "roboverse_pack" / "tasks"


def get_seed(seed_id: str) -> tuple[Path, str]:
    """Return (path, source_text) for a pool seed id (\"family/stem\")."""
    path = enumerate_seed_pool()[seed_id]
    return path, path.read_text()


# ---- the deduplicated sampling pool --------------------------------------------------
#
# The raw corpus is heavily over-represented by a few bulk-variant files (e.g.
# pick_single_egad.py registers ~1,589 asset variants of ONE task; peg_insertion_side.py
# ~1,000 geometry variants). Sampling is therefore done at the TASK-FILE level: one
# hand-ported task = one file = one seed, regardless of how many variants it registers.
# This collapses ~2,900 registrations to ~210 distinct tasks.

_EXCLUDE_PARTS = ("_passthrough", "_native", "__init__", "_base", "base_table",
                  "task_template", "_convert", "_locator", "_util")

# Locomotion / whole-body-imitation families: not manipulation seeds, out of scope for
# the arm-embodiment pipeline (stage 6 skips humanoids).
_EXCLUDE_FAMILIES = ("humanoid", "humanoid_bench", "beyondmimic", "mjlab")


def enumerate_seed_pool() -> dict[str, Path]:
    """One seed per distinct task file: {\"family/stem\": path}."""
    pool: dict[str, Path] = {}
    for path in sorted(ROBOVERSE_TASKS.rglob("*.py")):
        rel = path.relative_to(ROBOVERSE_TASKS)
        if rel.parts[0] in _EXCLUDE_FAMILIES:
            continue
        if any(x in str(rel) for x in _EXCLUDE_PARTS):
            continue
        if "register_task(" not in path.read_text():
            continue
        pool[f"{rel.parts[0]}/{path.stem}"] = path
    return pool


def sample_seeds(n: int, seed: int = 0) -> list[str]:
    """Uniform sample (without replacement) over the deduplicated pool."""
    import random

    ids = sorted(enumerate_seed_pool())
    return random.Random(seed).sample(ids, n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print the seed pool")
    parser.add_argument("--sample", type=int, default=0, help="sample N seed ids")
    parser.add_argument("--rng-seed", type=int, default=0)
    args = parser.parse_args()
    pool = enumerate_seed_pool()
    if args.list:
        for k in sorted(pool):
            print(k)
    print(f"pool size: {len(pool)} distinct task files")
    if args.sample:
        for s in sample_seeds(args.sample, args.rng_seed):
            print("sampled:", s)
