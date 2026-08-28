"""How low can the franka put its FINGERTIPS in the clear_organics cell, in FREE SPACE?

Every grasp probe floored at tip_z ~= 0.60 (about 5 cm above the 0.55 m table) regardless of
base height, wrist tilt or xy gain — while the produce centres sit at 0.57-0.58, so the jaw
only ever reached the top dome of a fruit and slid off. Two very different causes remain:

  * kinematics / controller  -> the floor is the same with NOTHING under the hand;
  * contact with the fruit   -> in free space the hand goes much lower.

This drives the hand straight down at a set of radii over EMPTY table and reports the lowest
fingertip height actually achieved at each, which decides it. Also sweeps the wrist tilt,
because a leaning wrist reaches lower than a strict top-down one.

Run:
    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python -u scripts/probe_organics_reach.py --headless
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--base_z", type=float, default=-1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_matrix,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

OPEN, FINGER_LEN = 0.04, 0.112


def main() -> None:
    robobench.discover()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfgf = ENVS.get("packing.clear_organics.franka.osc")()
    sc = cfgf.scene_cfg
    # park every object far away: this measures the ARM, not contact
    sc.subset_sample = False
    sc.shuffle_slots = False
    sc.reset_pos_jitter = 0.0
    sc.reset_yaw_deg = 0.0
    sc.scatter_center = (0.0, 3.0)
    sc.bin_pos = (3.0, 3.0)
    if args.base_z >= 0:
        bp = list(cfgf.robot_cfg.base_pos)
        bp[2] = args.base_z
        cfgf.robot_cfg.base_pos = tuple(bp)
    env = cfgf.build(num_envs=1, device=dev)
    scene, robot = env.scene, env.robot
    art = robot.articulation
    ee = art.body_names.index("panda_hand")
    osc = robot.controller.controllers[0]
    osc._kp = torch.tensor([220.0] * 3 + [600.0] * 3, device=dev)
    osc._kd = 2.0 * osc._kp.sqrt()
    osc.cfg.rot_scale = 0.15
    osc.cfg.kp_null, osc.cfg.kd_null = 1.0, 2.0
    n_act = robot.action_dim
    hz = 1.0 / (env.dt * robot.control_period)
    env.reset()
    Z0 = scene.cfg.surface_z
    base = art.data.root_pos_w[0].clone()
    print(f"[reach] surface_z={Z0}  base=({float(base[0]):.2f},{float(base[1]):.2f},"
          f"{float(base[2]):.2f})", flush=True)

    def V3(x, y, z):
        return torch.tensor([float(x), float(y), float(z)], device=dev)

    def pose():
        return art.data.body_pos_w[0, ee], art.data.body_quat_w[0, ee]

    def tipz():
        p, q = pose()
        return float((p + FINGER_LEN * quat_apply(q.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0])[2])

    def jaw(az, tilt, u):
        yh = V3(math.cos(az), math.sin(az), 0.0)
        zh = V3(0.0, 0.0, -1.0)
        gq = quat_from_matrix(torch.stack([torch.cross(yh, zh, dim=0), yh, zh], 1).unsqueeze(0))[0]
        if tilt > 0:
            axis = V3(-float(u[1]), float(u[0]), 0.0)
            gq = quat_mul(quat_from_angle_axis(torch.tensor([tilt], device=dev),
                                              axis.unsqueeze(0))[0].unsqueeze(0), gq.unsqueeze(0))[0]
        return gq

    def drive(target, gq, secs):
        for _ in range(max(1, round(secs * hz))):
            a = torch.zeros(1, n_act, device=dev)
            p, q = pose()
            err = target - p
            a[0, 0:3] = (torch.cat([err[:2] * 1.6, err[2:3]]) / osc.cfg.pos_scale).clamp(-1, 1)
            qe = quat_mul(gq.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
            a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1, 1)
            a[0, 6:8] = OPEN
            env.step(a)

    print(f"[reach] {'radius':>7s} {'tilt':>5s}  {'lowest tip_z':>12s} {'above table':>11s} "
          f"{'q4':>6s}", flush=True)
    for radius in (0.35, 0.45, 0.55):
        for tilt_deg in (0, 25, 40):
            robot.reset(torch.tensor([0], device=dev, dtype=torch.long))
            drive(V3(float(base[0]), float(base[1]) + 0.30, Z0 + 0.30),
                  jaw(0.0, 0.0, V3(0, 1, 0)), 2.0)
            tx = float(base[0])
            ty = float(base[1]) + radius
            u = V3(0.0, 1.0, 0.0)
            gq = jaw(0.0, math.radians(tilt_deg), u)
            drive(V3(tx, ty, Z0 + 0.25), gq, 3.0)          # get above the spot
            drive(V3(tx, ty, Z0 - 0.05), gq, 6.0)          # then push BELOW the table
            lo = tipz()
            print(f"[reach] {radius:7.2f} {tilt_deg:5d}  {lo:12.3f} "
                  f"{(lo - Z0) * 1000:9.0f}mm {float(art.data.joint_pos[0, 3]):6.2f}", flush=True)
    print("REACH_DONE", flush=True)


if __name__ == "__main__":
    main()
    import os
    import threading
    threading.Timer(8.0, lambda: os._exit(0)).start()
    app.close()
    os._exit(0)
