"""Step-1 smoke: build the wrapped env, print the observation layout, run random actions, measure
throughput, and exercise the per-env partial reset. No learning.

    python rl/scripts/smoke.py --task bulb_franka_osc --num_envs 16 --steps 200 --headless
"""
from __future__ import annotations

import argparse
import faulthandler
import signal
import sys
import time
from pathlib import Path

faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1 <pid>` dumps every thread's stack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/ on the path -> robobench_rl

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="bulb_franka_osc")
ap.add_argument("--reward", default="progress")
ap.add_argument("--num_envs", type=int, default=None)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--action_scale", type=float, default=0.3, help="random action magnitude in [-1,1] units")
ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
ap.add_argument("--profile", action="store_true", help="cProfile the stepping loop; print the top entries")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from robobench_rl.config import load_config  # noqa: E402
from robobench_rl.vec_env import RoboBenchVecEnv  # noqa: E402


def main() -> None:
    ov = list(args.set)
    if args.num_envs is not None:
        ov.append(f"task.num_envs={args.num_envs}")
    cfg = load_config(args.task, args.reward, ov)
    t0 = time.time()
    venv = RoboBenchVecEnv(cfg)
    print(f"[smoke] build {time.time() - t0:.1f} s", flush=True)
    print(venv.describe(), flush=True)
    obs = venv.get_observations()
    print(f"[smoke] obs policy shape {tuple(obs['policy'].shape)}  finite {bool(torch.isfinite(obs['policy']).all())}")
    print(f"[smoke] obs env0 first 16: {[round(x, 3) for x in obs['policy'][0, :16].tolist()]}")

    n, A = venv.num_envs, venv.num_actions
    # warm-up
    for i in range(5):
        t1 = time.time()
        venv.step(torch.zeros(n, A, device=venv.device))
        torch.cuda.synchronize()
        print(f"[smoke] warm-up step {i} {time.time() - t1:.3f} s", flush=True)
    prof = None
    if args.profile:
        import cProfile
        prof = cProfile.Profile()
        prof.enable()
    t0 = time.time()
    rews, succs = [], 0
    for k in range(args.steps):
        a = (torch.randn(n, A, device=venv.device) * args.action_scale).clamp(-1, 1)
        obs, r, done, extras = venv.step(a)
        if k % 25 == 0:
            print(f"[smoke] step {k}", flush=True)
        rews.append(r)
        succs += int(extras["success"].sum())
        if k == args.steps // 2:  # exercise the partial reset on half the envs
            ids = torch.arange(0, n, 2, device=venv.device)
            venv._reset_idx(ids)
            p_after = venv.reward_fn.prev
            print(f"[smoke] partial reset of {len(ids)} envs: progress baseline after reset "
                  f"reset-envs mean {p_after[ids].mean():.4f}, others mean "
                  f"{p_after[torch.arange(1, n, 2, device=venv.device)].mean() if n > 1 else float('nan'):.4f}")
    torch.cuda.synchronize()
    dt = time.time() - t0
    if prof is not None:
        import io
        import pstats
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(35)
        print("[smoke] profile (cumulative):\n" + buf.getvalue(), flush=True)
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(20)
        print("[smoke] profile (tottime):\n" + buf.getvalue(), flush=True)
    R = torch.stack(rews)
    print(f"[smoke] {args.steps} control steps x {n} envs in {dt:.1f} s -> "
          f"{args.steps * n / dt:,.0f} env-steps/s, {args.steps / dt:.1f} batched steps/s, "
          f"{args.steps * n * venv.step_dt / dt:,.1f}x realtime (sim-s per wall-s)")
    print(f"[smoke] reward mean {R.mean():.5f} min {R.min():.4f} max {R.max():.4f}  successes {succs}")
    p, stages = venv.reward_fn.measure()
    print("[smoke] grader stage means now: " + ", ".join(f"{k}={v.mean():.4f}" for k, v in stages.items()) +
          f", progress={p.mean():.4f}")
    if venv.reward_fn.task is not None:
        terms = venv.reward_fn.task.terms()
        print("[smoke] shaped terms now: " + ", ".join(f"{k}={v.mean():.4f}" for k, v in terms.items()) +
              f", potential={venv.reward_fn.task.potential(terms).mean():.4f}")
    print(f"[smoke] obs finite {bool(torch.isfinite(obs['policy']).all())}  "
          f"gpu mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB (torch) ")
    import subprocess
    try:
        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().strip()
        print(f"[smoke] gpu mem used (nvidia-smi): {used} MiB")
    except Exception:
        pass
    import os
    sys.stdout.flush()
    os._exit(0)  # never call env.close(): kit hangs in sim.stop(); everything is printed already


if __name__ == "__main__":
    main()
