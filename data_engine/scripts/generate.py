"""Generate one batch of graded episodes from a baked cell (boots Isaac).

    .venv/bin/python data_engine/scripts/generate.py --headless \\
        <…/data_gen/<gen_name>> [--scene scene_0] [--strategy strategy_0] [--phase phase_1] \\
        [--batch default] [--num_envs 4] [--seed 0] \\
        [--sigma 0.05 --prob 0.01 --duration 0.5 --dims 0:6] \\
        [--render] [--render_args "--cams front --fps 30"]

The cell is the (scene × strategy × phase) triple; phase is optional — without it
the strategy's solve.py runs from scratch, with it solve_by_phase.py enters at the
phase's declared entries: the batch sweeps ALL the cell's reset/ files, one rollout
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
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--env_draw", type=int, default=0,
                    help="physical-param draw slice start: env slot e samples index env_draw+e-1 "
                         "from the scene's PHYSICAL_PARAMS bands (slot 0 stays nominal)")
parser.add_argument("--solve_draw", type=int, default=0,
                    help="solve-hyperparameter draw index: ONE set from the solve's "
                         "SOLVE_PARAMS bands for the whole batch")
parser.add_argument("--solo-draw", dest="solo_draw", action="store_true",
                    help="every env slot (incl. slot 0) takes a PHYSICAL_PARAMS draw — for "
                         "single-env diversified batches that avoid the lockstep phase coupling "
                         "(num_envs=1, one draw per boot). Default off keeps slot 0 the nominal canary.")
parser.add_argument("--nominal", action="store_true",
                    help="no sampling at all (baseline batch: plain world, bare solve)")
parser.add_argument("--phys-nominal", dest="phys_nominal", action="store_true",
                    help="skip PHYSICAL_PARAMS sampling only (file-value world) — isolates the "
                         "solve/noise axes, and keeps num_envs>1 batches lockstep-identical")
parser.add_argument("--solve-nominal", dest="solve_nominal", action="store_true",
                    help="skip SOLVE_PARAMS sampling only (file-value solve constants) — "
                         "isolates the physics/noise axes")
parser.add_argument("--render", action="store_true",
                    help="after the batch is graded, replay it to RGB frames + previews "
                         "(chains scripts/render.py in its own process — generation itself "
                         "stays camera-free)")
parser.add_argument("--render_args", default="",
                    help='extra args forwarded to render.py, e.g. "--fps 30 --eye 1.0 -0.7 0.5"')
parser.add_argument("--sigma", type=float, default=0.0, help="action-noise sigma (0 = nominal)")
parser.add_argument("--prob", type=float, default=1.0, help="noise-window start prob per step")
parser.add_argument("--duration", type=float, default=0.0, help="noise-window length (sim-seconds; 0 = a single step)")
parser.add_argument("--dims", default="", help="noised action dims as a:b (required if sigma > 0)")
parser.add_argument("--noise-gate-z", type=float, default=0.0, dest="noise_gate_z",
                    help="height gate (m): noise applies only while the hand is ABOVE this — "
                         "perturb transport, never the low precision phases (0 = ungated)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.sigma > 0 and not args.dims:
    parser.error("--dims is required when --sigma > 0")

app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.generation import run_batch  # noqa: E402

noise = {"sigma": args.sigma, "prob": args.prob, "duration": args.duration,
         "gate_z": args.noise_gate_z,
         "dims": tuple(int(x) for x in args.dims.split(":")) if args.dims else None}
out = run_batch(args.gen_root, batch=args.batch, scene=args.scene, strategy=args.strategy,
                phase=args.phase, num_envs=args.num_envs, seed=args.seed,
                noise=noise, device="cuda:0" if torch.cuda.is_available() else "cpu",
                env_draw=args.env_draw, solve_draw=args.solve_draw, nominal=args.nominal,
                solo_draw=args.solo_draw, phys_nominal=args.phys_nominal,
                solve_nominal=args.solve_nominal)

if args.render:
    # A separate process on purpose: rendering needs --enable_cameras (a different,
    # flakier Kit config) and replays only the recorded states — generation stays
    # camera-free and pixel decisions stay re-renderable. out = real_root/data/<batch>.
    import shlex  # noqa: E402
    import subprocess  # noqa: E402

    cmd = [sys.executable, str(Path(__file__).resolve().parent / "render.py"),
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
