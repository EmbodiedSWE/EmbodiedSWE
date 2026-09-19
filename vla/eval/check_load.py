"""check_load — stage-0 loader check: build an eval sim, hold, snapshot obs (boots Isaac).

    .venv/bin/python vla/eval/check_load.py assembly.bulb.franka.osc --headless
    .venv/bin/python vla/eval/check_load.py bulb_jointpd_60hz --headless
    .venv/bin/python vla/eval/check_load.py <…/datasets/<id>/meta/bake.json> --headless \\
        [--episode <…/data/<batch>/ep_0000>]
    .venv/bin/python vla/eval/check_load.py <bake.json> --cell <gen_root>/scenes/scene_59 \\
        --phase strategy_0/phase_0 --resets 4 --headless      # a cell world + its phase start

Loads the sim (registered name | preset | bake), resets, runs the warmup +
`--steps` hold-actions, and writes per-view PNGs + a report.json under
`--out/<tag>/`. With --episode it also restores that episode's first recorded
state (dataset init) and snapshots again. Everything a loader can get wrong —
preset resolution, camera injection, executor install, stamp validation, obs
layout — fails or shows up here, with no policy involved.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="stage-0 eval-sim loader check")
parser.add_argument("source", help="registered sim name | ENVS preset | bake.json path | cell dir")
parser.add_argument("--cell", default="", help="world from a data-engine cell dir (see sim.py)")
parser.add_argument("--phase", default="", help="phase reset file / dir / '<strategy>/<phase>' on the cell")
parser.add_argument("--resets", type=int, default=0,
                    help="extra resets at seed+1.. logging every non-robot root pose (start-distribution check)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=24, help="hold-action steps after warmup")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episode", default="", help="ep dir for an init_from_episode snapshot")
parser.add_argument("--out", default=str(Path(__file__).parent / "_out"))

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import load_sim  # noqa: E402

tag = re.sub(r"[^\w.-]+", "_", args.source).strip("_")[-80:]
out = Path(args.out) / tag
out.mkdir(parents=True, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

overrides = {k: v for k, v in (("cell", args.cell and str(Path(args.cell).resolve())),
                               ("phase", args.phase)) if v}
sim = load_sim(args.source, num_envs=args.num_envs, device=device, **overrides)
if sim.spec.provenance:
    print(f"[check] provenance: {json.dumps(sim.spec.provenance)}", flush=True)
print(f"[check] task: {sim.task}", flush=True)
print(f"[check] rate {sim.rate_hz:.0f} Hz, control_period {sim.env.robot.control_period}, "
      f"sim dt {sim.env.dt:.5f}", flush=True)
print(f"[check] state ({len(sim.state_names)}): {sim.state_names}", flush=True)


def snap(obs: dict, label: str) -> dict:
    for view, imgs in obs["images"].items():
        imageio.imwrite(str(out / f"{label}_{view}.png"), imgs[0])
    row = {
        "state": [round(float(x), 4) for x in obs["state"][0]],
        "success": obs["success"].tolist(),
        "images": {v: [list(i.shape), str(i.dtype)] for v, i in obs["images"].items()},
    }
    print(f"[check] {label}: state {row['state']} success {row['success']}", flush=True)
    return row


def roots() -> dict:
    """World-frame xyz + yaw (deg) of every recorded 13-vector root except the robot's."""
    import math

    from engine.generation import _flat

    def pose(row):
        x, y, z, qw, qx, qy, qz = [float(t) for t in row[:7]]
        yaw = math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        return [round(x, 4), round(y, 4), round(z, 4), round(yaw, 1)]

    out_ = {}
    for k, v in _flat(sim.env.get_states()).items():
        if k.startswith("robot") or v.shape[-1] != 13:
            continue
        if v.dim() == 2:
            out_[k] = pose(v[0])
        elif v.dim() == 3:  # (E, N, 13) multi-body asset: centroid xyz + yaw of body 0 + N
            c = v[0, :, :3].mean(dim=0)
            out_[k] = [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4),
                       pose(v[0, 0])[3], int(v.shape[1])]
    return out_


report = {"source": args.source, "spec": {k: v for k, v in vars(sim.spec).items() if k != "stamp"},
          "task": sim.task, "rate_hz": sim.rate_hz, "state_names": sim.state_names,
          "control_period": sim.env.robot.control_period, "sim_dt": sim.env.dt}

obs = sim.reset(seed=args.seed)
report["reset"] = snap(obs, "reset")
report["reset"]["roots"], report["reset"]["reset_fn"] = roots(), sim.reset_fn
print(f"[check] reset seed={args.seed} reset_fn={sim.reset_fn} roots {report['reset']['roots']}", flush=True)

start_q = obs["state"][0].copy()
for _ in range(args.steps):
    obs = sim.step(sim.hold_action())
report["hold"] = snap(obs, "hold")
drift = np.abs(obs["state"][0] - start_q)
report["hold_drift_max"] = round(float(drift.max()), 5)
print(f"[check] hold drift over {args.steps} steps: max {drift.max():.5f} "
      f"(per-dim {[round(float(d), 4) for d in drift]})", flush=True)

report["resets"] = []
for i in range(1, args.resets + 1):
    obs = sim.reset(seed=args.seed + i)
    report["resets"].append({"seed": args.seed + i, "reset_fn": sim.reset_fn, "roots": roots(),
                             "state": [round(float(x), 4) for x in obs["state"][0]]})
    print(f"[check] reset seed={args.seed + i} reset_fn={sim.reset_fn} roots {report['resets'][-1]['roots']}",
          flush=True)

if args.episode:
    obs = sim.init_from_episode(Path(args.episode))
    report["episode_roots"] = roots()
    print(f"[check] episode-init roots {report['episode_roots']}", flush=True)
    report["episode_init"] = snap(obs, "episode_init")
    q0 = np.load(Path(args.episode) / "traj.npz")["robot/joint_pos"][0]
    report["episode_q_err_max"] = round(float(
        np.abs(obs["state"][0][:-1] - q0[sim.arm_ids]).max()), 5)
    print(f"[check] episode-init arm-q error vs traj row0: {report['episode_q_err_max']}",
          flush=True)

(out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
print(f"[check] DONE -> {out}", flush=True)

# Kit teardown hangs under --enable_cameras; everything is written by now (render.py's recipe).
import os  # noqa: E402
import threading  # noqa: E402

watchdog = threading.Timer(10.0, lambda: os._exit(0))
watchdog.daemon = True
watchdog.start()
app.close()
os._exit(0)
