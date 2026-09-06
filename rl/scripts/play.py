"""Run an exported solution/ folder on a fresh env under the grading wrapper; print the grader's verdict.

    python rl/scripts/play.py --solution <run>/solutions/model_300 --num_envs 8 --headless [--trace]
    python rl/scripts/play.py --solution ... --warm --episodes 5 --headless      # warm-start diagnostic
    python scripts/record_video.py rl/scripts/play.py --video v.mp4 --eye .95 .95 .75 --target-at .26 .25 .12 \
        --solution <dir> --num_envs 1 [--trace]                                   # mp4 of env 0

Default = the grading protocol: reset by us, GradedEnv (reset/set_states blocked), the exported solve()
steps, the suite grader scores. `--warm` instead evaluates from the curriculum's warm start (hand
pre-rolled to the hover pose) — a diagnostic to separate "the pick skill exists" from "the approach is
too slow"; it loops `--episodes` times until env 0 earns the pick, which makes recordable episodes."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--solution", required=True)
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--trace", action="store_true", help="print env 0's dense terms + grader stages once per second")
ap.add_argument("--warm", action="store_true", help="warm-start diagnostic instead of the grading protocol")
ap.add_argument("--episodes", type=int, default=1, help="--warm: repeat until env 0 picks")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

SOL = Path(args.solution).resolve()
META = json.loads((SOL / "env.json").read_text())


def graded() -> None:
    import robobench
    from robobench.core import ENVS
    from robobench.core.grader import GradedEnv

    robobench.discover()
    preset = META["preset"]
    env = ENVS.get(preset)().build(num_envs=args.num_envs, seed=args.seed)
    env.reset()
    suite, scene = preset.split(".")[0], ENVS.get(preset)().scene
    grader = importlib.import_module(f"robobench.suites.{suite}.grader").GRADERS[scene](env)
    peaks: dict[str, torch.Tensor] = {}
    tracer = None
    if args.trace:
        sys.path.insert(0, str(SOL))
        from robobench_rl.task_envs import load_task_env_cls  # the exported copy
        from robobench_rl.task_rewards import load_task_reward
        task = (load_task_env_cls(META.get("task_env")).reward_cls or load_task_reward(scene))(env)
        every = max(1, int(round(1.0 / (env.dt * env.robot.control_period))))
        k = [0]

        def tracer(recs):
            k[0] += 1
            if k[0] % every == 0:
                t = task.terms() if task else {}
                print(f"[trace] t={k[0] * env.dt * env.robot.control_period:5.1f}s " + " ".join(f"{n}={float(v[0]):.3f}" for n, v in t.items())
                      + " | grader " + " ".join(f"{n}={v:.2f}" for n, v in recs[0]["stages"].items()), flush=True)

    def on_record(recs):
        for e, r in enumerate(recs):
            for n, v in r["stages"].items():
                peaks.setdefault(n, torch.zeros(len(recs)))
                peaks[n][e] = max(peaks[n][e], v)
        if tracer:
            tracer(recs)

    sys.path.insert(0, str(SOL))
    spec = importlib.util.spec_from_file_location("rl_solve", SOL / "solve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"[play] {preset} x{args.num_envs}  {SOL.name} ({META.get('checkpoint')}, iter {META.get('train_iteration')})", flush=True)
    mod.solve(GradedEnv(env, grader, on_record=on_record))
    verdicts = grader.verdict()
    for v in verdicts:
        print(f"[play] env {v['env']}: success {v['success']}  score {v['score']:.3f}  peaks "
              + ", ".join(f"{n}={float(peaks[n][v['env']]):.3f}" for n in peaks), flush=True)
    print(f"[play] success {sum(v['success'] for v in verdicts)}/{len(verdicts)}  mean score "
          f"{sum(v['score'] for v in verdicts) / len(verdicts):.3f}", flush=True)


def warm() -> None:
    from robobench_rl.vec_env import make_vec_env

    tc = json.loads(json.dumps(META["cfg"]))
    tc["task"]["num_envs"] = args.num_envs
    tc.setdefault("curriculum", {})["hover_start_frac"] = 1.0
    venv = make_vec_env(tc)
    policy = torch.jit.load(str(SOL / "policy.pt"), map_location=str(venv.device)).eval()
    for ep in range(args.episodes):
        venv.reset()
        peaks: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for k in range(venv.max_episode_length):
                a = policy(venv.get_observations()["policy"]).clamp(-1, 1)
                venv.env.step(venv.process_actions(a))
                _, stages = venv.reward_fn.measure()
                for n, v in stages.items():
                    peaks[n] = torch.maximum(peaks.get(n, torch.zeros_like(v)), v)
        picked = peaks.get("picked", peaks.get("knife_taken"))
        print(f"[warm] ep{ep} per-env peak pick stage: {[round(float(x), 2) for x in picked]}", flush=True)
        if float(picked[0]) >= 0.5:
            print("[warm] env 0 PICKED", flush=True)
            break


if __name__ == "__main__":
    warm() if args.warm else graded()
    sys.stdout.flush()
    os._exit(0)  # kit hangs on close
