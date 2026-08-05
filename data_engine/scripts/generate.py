"""Generate one batch of graded episodes from a baked cell (boots Isaac).

    .venv/bin/python data_engine/scripts/generate.py --headless \\
        <…/data_gen/<gen_name>> [--scene scene_0] [--strategy strategy_0] [--phase phase_1] \\
        [--batch default] [--num_envs 4] [--rounds 1] [--seed 0] \\
        [--sigma 0.05 --prob 0.01 --duration 0.5 --dims 0:6]

The cell is the (scene × strategy × phase) triple; phase is optional — without it
the strategy's solve.py runs from scratch, with it solve_by_phase.py enters at the
phase's declared entries: each round sweeps ALL the cell's reset/ files, one rollout
per file, and within a rollout every reset_N builder in the file shapes an even share
of the envs. Noise defaults to off (the
nominal configuration); --dims (e.g. 0:6 = franka-osc arm) is required when
sigma > 0 — gripper dims are never noised.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="generate one batch of graded episodes")
parser.add_argument("gen_root", help="the campaign: …/<run>/data_gen/<gen_name>")
parser.add_argument("--scene", default="scene_0")
parser.add_argument("--strategy", default="strategy_0")
parser.add_argument("--phase", default=None,
                    help="phase cell under the strategy's phases/ (no cell = from scratch)")
parser.add_argument("--batch", default=None, help="batch name under data/ (default: batch_<timestamp>)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--rounds", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sigma", type=float, default=0.0, help="action-noise sigma (0 = nominal)")
parser.add_argument("--prob", type=float, default=1.0, help="noise-window start prob per step")
parser.add_argument("--duration", type=float, default=0.0, help="noise-window length (sim-seconds; 0 = a single step)")
parser.add_argument("--dims", default="", help="noised action dims as a:b (required if sigma > 0)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.sigma > 0 and not args.dims:
    parser.error("--dims is required when --sigma > 0")

app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.generation import run_batch  # noqa: E402

noise = {"sigma": args.sigma, "prob": args.prob, "duration": args.duration,
         "dims": tuple(int(x) for x in args.dims.split(":")) if args.dims else None}
run_batch(args.gen_root, batch=args.batch, scene=args.scene, strategy=args.strategy,
          phase=args.phase, num_envs=args.num_envs, rounds=args.rounds, seed=args.seed,
          noise=noise, device="cuda:0" if torch.cuda.is_available() else "cpu")

# Kit teardown regularly hangs inside app.close() (100% CPU spin, holds GPU memory) —
# same watchdog hard-exit as robobench/scripts/smoke.py; the batch is fully written by now.
import threading  # noqa: E402

watchdog = threading.Timer(10.0, lambda: os._exit(0))
watchdog.daemon = True
watchdog.start()
app.close()
os._exit(0)
