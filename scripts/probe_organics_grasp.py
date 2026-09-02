"""Probe ONE grasp on clear_organic_objects, step by step, to find why lifts fail.

A scripted pick reports `lift failed` with the jaw measurably closed ON the fruit
(w=61 mm on a 59 mm onion), which rules out both an empty jaw and slip-at-contact. This
prints, every few control ticks through a single hover -> descend -> close -> lift, the
quantities that discriminate the remaining hypotheses:

  hand z / tip z      - is the ARM actually rising, or is the OSC stalled?
  item z              - is the FRUIT rising with it?
  width               - is the jaw holding its commanded aperture or being forced open?
  |item - tip| xy     - is the fruit centred between the fingers or off to one side?
  q4                  - elbow joint, to catch the wound-arm / singular case.

Deterministic scene (no jitter, no shuffle, no subset) so the geometry is identical run to run.

Run:
    OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python -u scripts/probe_organics_grasp.py --headless --item red_onion
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--item", type=str, default="red_onion")
parser.add_argument("--depth_frac", type=float, default=0.25)
parser.add_argument("--indent", type=float, default=0.0055)
parser.add_argument("--near", type=float, default=0.0, help="tilt the hand for reaches inside "
                    "this radius of the base (0 = strict top-down, the jamming case)")
parser.add_argument("--max_tilt", type=float, default=0.35)
parser.add_argument("--base_z", type=float, default=-1.0,
                    help="override the franka base height (default: keep the binding's)")
parser.add_argument("--cage_open", action="store_true",
                    help="descend with the jaw FULLY OPEN instead of caged")
parser.add_argument("--hover_s", type=float, default=4.0)
parser.add_argument("--boost", type=float, default=1.0,
                    help="xy_boost during hover/descend (pen_holder solve uses 1.6)")
parser.add_argument("--scatter", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="override scatter_center, to test the reach annulus")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

OPEN, FINGER_LEN = 0.04, 0.112


def main() -> None:
    robobench.discover()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfgf = ENVS.get("packing.clear_organic_objects.franka.osc")()
    # freeze the layout so the probe is repeatable
    sc = cfgf.scene_cfg
    sc.shuffle_slots = False
    sc.reset_pos_jitter = 0.0
    sc.reset_yaw_deg = 0.0
    sc.subset_sample = False
    if args.scatter is not None:
        sc.scatter_center = tuple(args.scatter)
        print(f"[probe] scatter_center -> {sc.scatter_center}", flush=True)
    if args.base_z >= 0:
        bp = list(cfgf.robot_cfg.base_pos)
        bp[2] = args.base_z
        cfgf.robot_cfg.base_pos = tuple(bp)
        print(f"[probe] base_pos -> {cfgf.robot_cfg.base_pos}", flush=True)
    env = cfgf.build(num_envs=1, device=dev)
    scene, robot = env.scene, env.robot
    art = robot.articulation
    ee = art.body_names.index("panda_hand")
    fj1 = art.find_joints(["panda_finger_joint1"])[0]
    osc = robot.controller.controllers[0]
    osc._kp = torch.tensor([220.0] * 3 + [600.0] * 3, device=dev)
    osc._kd = 2.0 * osc._kp.sqrt()
    osc.cfg.rot_scale = 0.15
    osc.cfg.kp_null, osc.cfg.kd_null = 1.0, 2.0
    n_act = robot.action_dim
    hz = 1.0 / (env.dt * robot.control_period)
    env.reset()

    name = args.item
    k, _o, s = {n: (kk, o, ss) for n, kk, o, ss, _m in scene.cfg.MANIFEST}[name]
    import json
    from pathlib import Path
    ext = json.loads((Path(scene.cfg.asset_dir) / "extents.json").read_text())
    bb = [v * s for v in ext[k]["bbox_m"]]
    print(f"[probe] {name}: bbox(m)={[round(v,4) for v in bb]} scale={s}", flush=True)

    def V3(x, y, z):
        return torch.tensor([float(x), float(y), float(z)], device=dev)

    def pose():
        return art.data.body_pos_w[0, ee], art.data.body_quat_w[0, ee]

    def tip():
        p, q = pose()
        return p + FINGER_LEN * quat_apply(q.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]

    def width():
        return 2.0 * art.data.joint_pos[0, fj1].item()

    def item_p():
        return scene.items[name].data.root_pos_w[0]

    def jaw_quat(az):
        yh = V3(math.cos(az), math.sin(az), 0.0)
        zh = V3(0.0, 0.0, -1.0)
        return quat_from_matrix(torch.stack([torch.cross(yh, zh, dim=0), yh, zh], 1).unsqueeze(0))[0]

    # narrowest horizontal span of the live OBB
    q0 = scene.items[name].data.root_quat_w[0]
    axes = [quat_apply(q0.unsqueeze(0), V3(*e).unsqueeze(0))[0] * (b / 2)
            for e, b in zip(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), bb)]
    best = min(((2 * sum(abs(float(torch.dot(a, V3(math.cos(math.pi * i / 36), math.sin(math.pi * i / 36), 0)))) for a in axes), math.pi * i / 36)
                for i in range(36)))
    gw, az = best
    gq = jaw_quat(az)
    # optional lean-away-from-base for close/low reaches: a strict top-down wrist at ~0.36 m
    # with the base at tabletop height folds q4 onto its limit (see the module docstring)
    if args.near > 0:
        from isaaclab.utils.math import quat_from_angle_axis
        base_xy = art.data.root_pos_w[0, :2]
        tgt_xy = scene.items[name].data.root_pos_w[0, :2]
        d = float((tgt_xy - base_xy).norm())
        if d < args.near:
            tilt = min(args.max_tilt, (args.near - d) * 5.0)
            u = (tgt_xy - base_xy) / max(d, 1e-6)
            axis = V3(-float(u[1]), float(u[0]), 0.0)
            gq = quat_mul(quat_from_angle_axis(torch.tensor([tilt], device=dev),
                                               axis.unsqueeze(0))[0].unsqueeze(0),
                          gq.unsqueeze(0))[0]
            print(f"[probe] d_base={d:.3f} tilt={math.degrees(tilt):.0f}deg", flush=True)
        else:
            print(f"[probe] d_base={d:.3f} (>= near, no tilt)", flush=True)
    cage = OPEN if args.cage_open else min(0.04, gw / 2 + 0.008)
    grip = max(0.003, gw / 2 - args.indent)
    print(f"[probe] span={gw*1000:.1f}mm az={math.degrees(az):.0f}deg "
          f"cage={cage*2000:.0f}mm grip_cmd={grip*2000:.0f}mm", flush=True)

    t = {"n": 0}

    def servo(gp, grip_v, boost=1.0):
        a = torch.zeros(1, n_act, device=dev)
        p, q = pose()
        err = gp - p
        a[0, 0:3] = (torch.cat([err[:2] * boost, err[2:3]]) / osc.cfg.pos_scale).clamp(-1, 1)
        qe = quat_mul(gq.unsqueeze(0), quat_conjugate(q.unsqueeze(0)))
        a[0, 3:6] = (axis_angle_from_quat(qe)[0] / osc.cfg.rot_scale).clamp(-1, 1)
        a[0, 6:8] = grip_v
        env.step(a)
        t["n"] += 1

    def log(tag):
        ip, tp = item_p(), tip()
        print(f"[probe] {tag:9s} t={t['n']:4d} hand_z={float(pose()[0][2]):.3f} "
              f"tip_z={float(tp[2]):.3f} item_z={float(ip[2]):.3f} "
              f"w={width()*1000:5.1f}mm dxy={float((ip[:2]-tp[:2]).norm())*1000:5.1f}mm "
              f"q4={float(art.data.joint_pos[0,3]):.2f}", flush=True)

    def hold(gp, grip_v, secs, boost=1.0, tag=None, every=None):
        n = max(1, round(secs * hz))
        for i in range(n):
            servo(gp, grip_v, boost)
            if every and i % every == 0 and tag:
                log(tag)

    Z0 = scene.cfg.surface_z
    def goal(z):
        return V3(item_p()[0], item_p()[1], z) - FINGER_LEN * quat_apply(
            gq.unsqueeze(0), V3(0, 0, 1).unsqueeze(0))[0]

    log("start")
    hold(goal(Z0 + 0.26), OPEN, args.hover_s, boost=args.boost, tag="hover", every=15)
    log("hovered")
    gz = float(item_p()[2]) - args.depth_frac * bb[2]
    hold(goal(gz), cage, 4.0, boost=args.boost, tag="descend", every=15)
    log("descended")
    # ramped close, logging every few ticks so the moment of loss is visible
    n = max(1, round(1.5 * hz))
    for i in range(n):
        f = min(1.0, (i + 1) / (n * 0.8))
        servo(goal(gz), cage + (grip - cage) * f, args.boost)
        if i % 5 == 0:
            log("closing")
    log("closed")
    z_before = float(item_p()[2])
    # slow lift in small increments, logging each
    for dz in (0.02, 0.05, 0.09, 0.14, 0.20):
        hold(V3(pose()[0][0], pose()[0][1], (Z0 + 0.02) + dz + FINGER_LEN), grip, 0.8)
        log(f"lift{int(dz*100):02d}")
    rise = float(item_p()[2]) - z_before
    print(f"[probe] VERDICT rise={rise*1000:.0f}mm  {'HELD' if rise > 0.03 else 'FAILED'}", flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
    import os
    import threading
    threading.Timer(8.0, lambda: os._exit(0)).start()
    app.close()
    os._exit(0)
