"""Step 2: train PPO (rsl_rl OnPolicyRunner) on a robobench task through the RoboBenchVecEnv wrapper.

    python rl/scripts/train.py --task slice_franka_joint --reward shaped --headless \
        --set task.num_envs=256 task.episode_seconds=20 ppo.max_iterations=30

Run dir: rl/runs/<task>/<reward>/<stamp>[_<name>]/ with config.yaml (the merged config actually
used), env.json (obs layout, action bounds, control rate — what an exported policy needs to rebuild
its input pipeline), tensorboard events, and model_<iter>.pt checkpoints on ppo.save_interval.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/ -> robobench_rl

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--reward", default="progress")
ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
ap.add_argument("--name", default="", help="suffix for the run dir")
ap.add_argument("--runs_dir", default=str(Path(__file__).resolve().parents[1] / "runs"))
ap.add_argument("--resume", default="", help="model_<it>.pt to load before training")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from robobench_rl.config import dump_config, load_config  # noqa: E402
from robobench_rl.vec_env import RoboBenchVecEnv  # noqa: E402


def make_train_cfg(cfg: dict) -> tuple[dict, int]:
    """rsl_rl train_cfg from cfg['ppo']; max_iterations is ours, everything else is rsl_rl's."""
    ppo = copy.deepcopy(cfg["ppo"])
    max_iterations = int(ppo.pop("max_iterations"))
    ppo.setdefault("logger", "tensorboard")
    return ppo, max_iterations


def main() -> None:
    cfg = load_config(args.task, args.reward, list(args.set))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{args.name}" if args.name else "")
    log_dir = Path(args.runs_dir) / cfg["task"]["name"] / cfg["reward"]["mode"] / stamp
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"[train] run dir {log_dir}", flush=True)

    t0 = time.time()
    venv = RoboBenchVecEnv(cfg)
    dump_config(cfg, log_dir / "config.yaml")
    print(f"[train] env ready in {time.time() - t0:.1f} s\n{venv.describe()}", flush=True)
    env_meta = {
        "preset": cfg["task"]["preset"], "num_envs": venv.num_envs, "num_obs": venv.num_obs, "num_actions": venv.num_actions,
        "step_dt": venv.step_dt, "max_episode_length": venv.max_episode_length, "obs": cfg.get("obs", {}),
        "obs_layout": venv.obs_fn.layout, "action_map": venv.action_map_json(), "gripper_stiffness": venv.gripper_stiffness,
        "reward": cfg["reward"], "ppo": cfg["ppo"],
    }
    (log_dir / "env.json").write_text(json.dumps(env_meta, indent=2))

    train_cfg, max_iterations = make_train_cfg(cfg)
    runner = OnPolicyRunner(venv, train_cfg, log_dir=str(log_dir), device=str(venv.device))
    if args.resume:
        runner.load(args.resume)
        print(f"[train] resumed from {args.resume}", flush=True)
    print(f"[train] PPO for {max_iterations} iterations x {train_cfg['num_steps_per_env']} steps x "
          f"{venv.num_envs} envs = {max_iterations * train_cfg['num_steps_per_env'] * venv.num_envs:,} env-steps",
          flush=True)
    t0 = time.time()
    lockstep = venv.hover_frac > 0  # the hover curriculum needs every env to reset together
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=not lockstep)
    wall = time.time() - t0
    summary = {"iterations": max_iterations, "wall_s": round(wall, 1),
               "env_steps": max_iterations * train_cfg["num_steps_per_env"] * venv.num_envs,
               "episodes": venv.total_episodes, "successes": venv.total_successes,
               "final_model": f"model_{runner.current_learning_iteration}.pt"}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done: {summary}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # kit hangs on close


if __name__ == "__main__":
    main()
