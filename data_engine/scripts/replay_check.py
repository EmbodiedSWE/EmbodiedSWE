"""Replay-check episodes of ONE batch: do their recorded actions reproduce success? (boots Isaac)

    .venv/bin/python data_engine/scripts/replay_check.py --headless \\
        <…/data_gen/<gen_name>> --episodes data/<batch>/ep_0003 data/<batch>/ep_0007 …

Writes the verdict into each episode's meta.json under `replay` (see
engine/replay_check.py). Episodes must all belong to the same batch: one Isaac
sim per process, and the batch is what gets rebuilt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

DATA_ENGINE_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="replay-check the whole batch of the given episodes")
parser.add_argument("gen_root", help="the campaign: …/<run>/data_gen/<gen_name>")
parser.add_argument("--episodes", nargs="+", required=True,
                    help="episode dirs (relative to gen_root or absolute), all from one batch")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, str(DATA_ENGINE_ROOT.parent))
sys.path.insert(0, str(DATA_ENGINE_ROOT))
from engine.replay_check import replay_episodes  # noqa: E402

gen_root = Path(args.gen_root)
eps = [Path(e) if Path(e).is_absolute() else gen_root / e for e in args.episodes]
results = replay_episodes(gen_root, eps,
                          device="cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[replay-check] {sum(results.values())}/{len(results)} episodes reproduce their success",
      flush=True)

# Kit teardown regularly hangs inside app.close() — same watchdog hard-exit as generate.py.
import threading  # noqa: E402

watchdog = threading.Timer(10.0, lambda: os._exit(0))
watchdog.daemon = True
watchdog.start()
app.close()
os._exit(0)
