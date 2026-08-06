#!/usr/bin/env python3
"""Ship the pi05 BC training artifacts to HDFS for the merlin job.

What the job pod needs, and where it goes under <BASE>:

    openpi_repo.tgz   the openpi checkout (with the pi05_simgen_bc TrainConfig
                      + assets/<config>/<repo_id>/norm_stats.json), .venv and
                      .git excluded
    openpi_venv.tgz   the uv-built venv (system /usr/bin/python3.11 base --
                      same interpreter family as the training image, so the
                      venv transplants; jax cuda12 wheels bundle their CUDA)
    pi05_base/        12GB Orbax base checkpoint (params/ + assets/)
    lerobot_home/     the LeRobot dataset root (HF_LEROBOT_HOME on the pod)

Re-run after ANY change to the openpi checkout (config edits!), the venv, or
the dataset. Checkpoint and dataset uploads are skipped when already present
unless --force; the repo/venv tarballs are always rebuilt (cheap, and staleness
there is exactly what bites).

  python3 CoSiGen/vla_training/sync_to_hdfs.py [--force-ckpt] [--force-data]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CAPX = Path(__file__).resolve().parents[2]
OPENPI = CAPX / "relevant_repos" / "openpi"
PI05 = CAPX / "relevant_repos" / "pi05_base"
LEROBOT_HOME = CAPX / "simgen_bc" / "lerobot_home"
BASE = "hdfs://harunawl/home/byte_data_seed_wl/code_agent/zeyu.shen/simgen_bc"


def run(cmd: list[str]) -> None:
    print("[sync]", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(str(c) for c in cmd)}")


def exists(path: str) -> bool:
    return subprocess.run(["hdfs", "dfs", "-test", "-e", path],
                          capture_output=True).returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-ckpt", action="store_true")
    ap.add_argument("--force-data", action="store_true")
    args = ap.parse_args()

    run(["hdfs", "dfs", "-mkdir", "-p", BASE])

    run(["tar", "--exclude=.venv", "--exclude=.git", "--exclude=__pycache__",
         "-czf", "/tmp/openpi_repo.tgz", "-C", str(OPENPI.parent), "openpi"])
    run(["hdfs", "dfs", "-put", "-f", "/tmp/openpi_repo.tgz", f"{BASE}/"])

    run(["tar", "--exclude=__pycache__", "-czf", "/tmp/openpi_venv.tgz",
         "-C", str(OPENPI), ".venv"])
    run(["hdfs", "dfs", "-put", "-f", "/tmp/openpi_venv.tgz", f"{BASE}/"])

    if args.force_ckpt or not exists(f"{BASE}/pi05_base/params"):
        run(["hdfs", "dfs", "-put", "-f", str(PI05), f"{BASE}/pi05_base"])
    else:
        print("[sync] pi05_base already on HDFS -- skipped (--force-ckpt to redo)")

    # PyTorch conversion of the same base (multi-node torchrun path).
    pt = PI05.parent / "pi05_base_pytorch"
    if pt.exists():
        if args.force_ckpt or not exists(f"{BASE}/pi05_base_pytorch"):
            run(["hdfs", "dfs", "-put", "-f", str(pt), f"{BASE}/pi05_base_pytorch"])
        else:
            print("[sync] pi05_base_pytorch already on HDFS -- skipped")
    else:
        print("[sync] NOTE: no local pi05_base_pytorch (conversion not done) -- skipped")

    if args.force_data or not exists(f"{BASE}/lerobot_home/simgen/mock_bc"):
        run(["hdfs", "dfs", "-rm", "-r", "-f", "-skipTrash", f"{BASE}/lerobot_home"])
        run(["hdfs", "dfs", "-put", str(LEROBOT_HOME), f"{BASE}/lerobot_home"])
    else:
        print("[sync] lerobot_home already on HDFS -- skipped (--force-data to redo)")

    # BC EVAL package: forge-side module + scene/strategy VERBATIM from the
    # generator (train/eval parity) + the pod-side driver and policy server.
    vla = CAPX / "CoSiGen" / "vla_training"
    gen = CAPX / "CoSiGen" / "sim_gen" / "data_gen" / "plated_meal" / "gen_v1"
    eval_pkg = [
        vla / "eval_batch.py",
        vla / "eval_driver.py",
        vla / "policy_http_server.py",
        gen / "scenes" / "scene_0" / "scene" / "scene.py",
        gen / "scenes" / "scene_0" / "strategies" / "strategy_1" / "gen_strategy.py",
        CAPX / "CoSiGen" / "sim_gen" / "pipeline" / "forge_client.py",
    ]
    for p in eval_pkg:
        if not p.is_file():
            sys.exit(f"eval_pkg source missing: {p}")
    run(["hdfs", "dfs", "-mkdir", "-p", f"{BASE}/eval_pkg"])
    run(["hdfs", "dfs", "-put", "-f", *[str(p) for p in eval_pkg], f"{BASE}/eval_pkg/"])

    print(f"[sync] done -> {BASE}", flush=True)


if __name__ == "__main__":
    main()
