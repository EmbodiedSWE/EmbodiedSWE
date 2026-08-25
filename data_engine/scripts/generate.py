"""Generate one batch of graded episodes from a baked cell (boots Isaac).

    .venv/bin/python data_engine/scripts/generate.py --headless \\
        <…/data_gen/<gen_name>> [--scene scene_0] [--strategy strategy_0] [--phase phase_1] \\
        [--batch default] [--num_envs 4] [--seed 0] [--noise_scale 1.0] \\
        [--render] [--render_args "--cams front --fps 30"]

The cell is the (scene × strategy × phase) triple; phase is optional — without it
the strategy's solve.py runs from scratch, with it solve_by_phase.py enters at the
phase's declared entries: the batch sweeps ALL the cell's reset/ files, one rollout
per file, and within a rollout every reset_N builder in the file shapes an even share
of the envs.

Noise is SOLVE-AUTHORED: a solve passes per-step perturbations through
`env.step(action, noise=...)` (its own phase knowledge decides where and how
much), and `--noise_scale` is the pipeline's master switch — 0 (default) executes
every batch clean even if the solve offers noise; 1.0 executes the offered noise
at authored strength. Recorded action labels are clean by construction either
way; the executed perturbations are recorded verbatim in traj.npz `action_noise`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

DATA_ENGINE_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="generate one batch of graded episodes")
parser.add_argument("gen_root", help="the campaign: …/<run>/data_gen/<gen_name>")
parser.add_argument("--scene", default="scene_0")
parser.add_argument("--strategy", default="strategy_0")
parser.add_argument("--phase", default=None,
                    help="phase cell under the strategy's phases/ (no cell = from scratch)")
parser.add_argument("--batch", default=None, help="batch name under data/ (default: batch_<timestamp>)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env_draw", type=int, default=0,
                    help="physical-param draw slice start: env slot e samples index env_draw+e-1 "
                         "from the scene's PHYSICAL_PARAMS bands (slot 0 stays nominal)")
parser.add_argument("--solve_draw", type=int, default=0,
                    help="solve-hyperparameter draw index: ONE set from the solve's "
                         "SOLVE_PARAMS bands for the whole batch")
parser.add_argument("--nominal", action="store_true",
                    help="no sampling at all (baseline batch: plain world, bare solve)")
parser.add_argument("--render", action="store_true",
                    help="after the batch is graded, replay it to RGB frames + previews "
                         "(chains scripts/render.py in its own process — generation itself "
                         "stays camera-free)")
parser.add_argument("--render_args", default="",
                    help='extra args forwarded to render.py, e.g. "--fps 30 --eye 1.0 -0.7 0.5"')
parser.add_argument("--noise_scale", type=float, default=0.0,
                    help="master switch for solve-authored noise (env.step(..., noise=…)): "
                         "0 = execute clean (default; probes/farm), 1.0 = execute the "
                         "authored perturbations (compound)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, str(DATA_ENGINE_ROOT.parent))
sys.path.insert(0, str(DATA_ENGINE_ROOT))
from engine.generation import run_batch  # noqa: E402

out = run_batch(args.gen_root, batch=args.batch, scene=args.scene, strategy=args.strategy,
                phase=args.phase, num_envs=args.num_envs, seed=args.seed,
                noise_scale=args.noise_scale,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                env_draw=args.env_draw, solve_draw=args.solve_draw, nominal=args.nominal)

if args.render:
    # A separate process on purpose: rendering needs --enable_cameras (a different,
    # flakier Kit config) and replays only the recorded states — generation stays
    # camera-free and pixel decisions stay re-renderable. out = real_root/data/<batch>.
    import shlex  # noqa: E402
    import subprocess  # noqa: E402

    cmd = [sys.executable, str(DATA_ENGINE_ROOT / "scripts" / "render.py"),
           str(out.parent.parent), "--batches", out.name, "--headless",
           *shlex.split(args.render_args)]
    print(f"[generate] chaining render: {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd).returncode
    print(f"[generate] render {'DONE' if rc == 0 else f'FAILED (exit {rc})'}", flush=True)

# Kit teardown regularly hangs inside app.close() (100% CPU spin, holds GPU memory) —
# same watchdog hard-exit as robobench/scripts/smoke.py; the batch is fully written by now.
import threading  # noqa: E402

watchdog = threading.Timer(10.0, lambda: os._exit(0))
watchdog.daemon = True
watchdog.start()
app.close()
os._exit(0)
