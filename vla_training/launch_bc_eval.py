#!/usr/bin/env python3
"""Fan out BC checkpoint evals: self-contained 1-GPU L20 pods, sharded per checkpoint.

Each pod boots Isaac + the openpi policy server side by side (mlx_bc_eval.yaml,
the template of the run that worked on 2026-08-05) and plays `--episodes`
seeded episodes. `--shards N` splits a checkpoint's episodes over N parallel
pods with disjoint seed ranges (shard k gets seeds k*episodes..k*episodes+episodes-1),
so every checkpoint is scored on the SAME layouts and shards never overlap.

pi05_eval.py + the scene file are pushed once per sweep to <sweep>/_code/;
pods pull from there. Per-shard artifacts land in <sweep>/plated_meal_s<step>_k<shard>/.

  python3 CoSiGen/vla_training/launch_bc_eval.py --episodes 5 --shards 10   # 50/ckpt
  python3 CoSiGen/vla_training/launch_bc_eval.py --steps 25000 --episodes 5
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPX = HERE.parents[1]
TMPL = HERE / "mlx_bc_eval.yaml"
EVAL_PY = CAPX / "CoSiGen/eval/scripts/pi05_eval.py"
SCENE_PY = (CAPX / "CoSiGen/sim_gen/data_gen/plated_meal/gen_v1/"
                   "scenes/scene_0/scene/scene.py")
CKPTS = ("hdfs://harunawl/home/byte_data_seed_wl/code_agent/zeyu.shen/"
         "simgen_bc/checkpoints_pytorch/pi05_simgen_bc_v1/bc_v1_h20x64")
OUT_BASE = "hdfs://haruna/tmp/zeyu.shen/pi05_eval"


def sh(cmd: list[str], timeout: float = 240) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def checkpoints() -> list[int]:
    out = sh(["hdfs", "dfs", "-ls", CKPTS]).stdout
    return sorted(int(m) for m in re.findall(r"/(\d+)\s*$", out, re.M))


def submit_one(tmpl: str, ts: str, step: int, shard: int, episodes: int) -> dict:
    run = f"s{step}_k{shard}"
    out_dir = f"{OUT_BASE}/{ts}/plated_meal_{run}"
    yaml = (tmpl.replace("__OUT__", out_dir)
                .replace("__CODE__", f"{OUT_BASE}/{ts}/_code")
                .replace("__RUN__", run)
                .replace("__STEP__", str(step))
                .replace("__SEED0__", str(shard * episodes))
                .replace("__EPISODES__", str(episodes)))
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     prefix=f"pi05_bc_{run}_") as f:
        f.write(yaml)
        path = f.name
    # mlx rate-limits bursts (a 290-job launch lost 163 submissions on
    # 2026-08-05) and exits 0 even then -- retry with backoff, verify by job id
    for attempt in range(5):
        out = sh(["mlx", "job", "submitv2", "-p", path])
        m = re.search(r"jobs/(\w+)", out.stdout + out.stderr)
        if m:
            break
        time.sleep(5 * (attempt + 1))
    if not m:
        print(f"  {run}: SUBMIT FAILED ({path}):\n"
              f"{(out.stdout + out.stderr).strip()[-400:]}", flush=True)
    return {"run": run, "step": step, "shard": shard, "out": out_dir,
            "job": m.group(1) if m else None, "rc": out.returncode if m else 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="", help="comma list; default = all on HDFS")
    ap.add_argument("--episodes", type=int, default=5, help="episodes per shard")
    ap.add_argument("--shards", type=int, default=1, help="parallel pods per checkpoint")
    ap.add_argument("--ts", default="", help="join an existing sweep dir (relaunches)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    steps = ([int(s) for s in args.steps.split(",") if s]
             if args.steps else checkpoints())
    if not steps:
        raise SystemExit("no checkpoints found")
    ts = args.ts or time.strftime("%m%d_%H%M")
    jobs = [(step, shard) for step in steps for shard in range(args.shards)]
    print(f"[launch] {len(steps)} ckpt(s) x {args.shards} shard(s) x "
          f"{args.episodes} eps = {len(jobs)} jobs, "
          f"{args.shards * args.episodes} eps/ckpt, ts={ts}")
    if args.dry_run:
        for step, shard in jobs:
            print(f"  s{step}_k{shard}: seeds {shard*args.episodes}.."
                  f"{(shard+1)*args.episodes-1}")
        return

    code = f"{OUT_BASE}/{ts}/_code"
    r = sh(["hdfs", "dfs", "-mkdir", "-p", code])
    assert r.returncode == 0, f"hdfs mkdir failed: {r.stderr}"
    for src, name in ((EVAL_PY, "pi05_eval.py"), (SCENE_PY, "plated_meal_scene.py")):
        r = sh(["hdfs", "dfs", "-put", "-f", str(src), f"{code}/{name}"])
        assert r.returncode == 0, f"hdfs put {name} failed: {r.stderr}"

    tmpl = TMPL.read_text()
    with ThreadPoolExecutor(max_workers=3) as pool:
        launched = list(pool.map(
            lambda js: submit_one(tmpl, ts, js[0], js[1], args.episodes), jobs))
    ok = [l for l in launched if l["job"]]
    bad = [l for l in launched if not l["job"]]
    print(f"[launch] submitted {len(ok)}/{len(launched)} ok; {len(bad)} failed")
    for l in bad:
        print(f"  FAILED: {l['run']}")
    mf = CAPX / f"simgen_bc/eval_artifacts/launched_{ts}.json"
    mf.parent.mkdir(parents=True, exist_ok=True)
    # merge: relaunches into an existing sweep must not clobber earlier entries
    prev = json.loads(mf.read_text()) if mf.exists() else []
    by_run = {row["run"]: row for row in prev}
    for row in launched:
        by_run[row["run"]] = row
    mf.write_text(json.dumps(list(by_run.values()), indent=1))
    print(f"[launch] manifest: {mf} ({len(by_run)} entries)")


if __name__ == "__main__":
    main()
