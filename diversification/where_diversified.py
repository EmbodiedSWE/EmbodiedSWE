"""Map every diversified quantity to the exact line that consumes it.

Answers "what is actually randomized, and where" without trusting the schema: the schema says what
SHOULD vary, this reads the code and reports what genuinely does. A key declared in the YAML but
never read is reported as UNUSED — that check already caught three constants (carry_gain,
carry_clamp, centred_tol) that params_pen_holder.yaml sampled while the program ignored them.

    python where_diversified.py [--task nut_thread|pen_holder] [--md]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
TASKS = {
    "nut_thread": {"spec": "params_nut_thread.yaml",
                   "files": ["run_episode.py", "nut_thread_policy.py", "nut_thread_batch.py"]},
    "pen_holder": {"spec": "params_pen_holder.yaml",
                   "files": ["run_pen_episode.py", "pen_holder_policy.py"]},
}
CLASS = {"env": "1. WORLD", "policy": "2. PARAMETERS", "strategy": "3. PROGRAM STRUCTURE"}


def scan(files: list[str]) -> dict[str, list[tuple[str, int, str]]]:
    """{theta key: [(file, line no, source line)]} for every theta read in the code."""
    hits: dict[str, list[tuple[str, int, str]]] = {}
    pat = re.compile(r"""theta(?:\.get)?[\[(]\s*["'](\w+)["']""")
    for fn in files:
        p = HERE / fn
        if not p.exists():
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for key in pat.findall(line):
                hits.setdefault(key, []).append((fn, n, line.strip()))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="", help="default: every task")
    ap.add_argument("--md", action="store_true", help="markdown table instead of text")
    a = ap.parse_args()
    for task, cfg in TASKS.items():
        if a.task and task != a.task:
            continue
        spec = yaml.safe_load((HERE / cfg["spec"]).read_text())
        hits = scan(cfg["files"])
        print(f"\n{'=' * 78}\n{task}   (schema: {cfg['spec']})\n{'=' * 78}")
        seen = set()
        for section in ("env", "policy", "strategy"):
            keys = [k for k in (spec.get(section) or {}) if k != "seed"]
            if not keys:
                continue
            print(f"\n{CLASS[section]}  — {len(keys)} dimensions")
            for k in keys:
                seen.add(k)
                d = spec[section][k]
                rng = (f"{d['range']}" if "range" in d else
                       f"choices {d['choices']}" if "choices" in d else "")
                where = hits.get(k, [])
                if not where:
                    print(f"  {k:20s} {rng:34s} !! UNUSED — sampled but never read")
                    continue
                f0, n0, src = where[0]
                print(f"  {k:20s} {rng:34s} {f0}:{n0}")
                if not a.md:
                    print(f"  {'':20s} {'':34s} {src[:76]}")
                for f1, n1, _ in where[1:]:
                    print(f"  {'':20s} {'':34s} {f1}:{n1}")
        extra = sorted(set(hits) - seen - {"seed", "_index", "_nominal", "_fixed_env", "_vary"})
        if extra:
            print(f"\nread from theta but NOT in the schema: {', '.join(extra)}")
        frozen = spec.get("frozen") or {}
        print(f"\nFROZEN ({len(frozen)}) — each with the reason it is load-bearing:")
        for k, why in frozen.items():
            print(f"  {k:20s} {why}")


if __name__ == "__main__":
    main()
