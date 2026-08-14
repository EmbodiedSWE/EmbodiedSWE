"""Probe: build assembly.chair.multi.osc exactly as the bifranka smoke does, reset, and
track every body's pose through settle — isolates WHERE the chair goes at reset and when.

    /opt/venv/bin/python -m robobench.scripts.probe_chair_multi_reset --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--yaw", type=float, default=None,
                    help="override seat_yaw_base (deg) to isolate yaw-dependent resets")
parser.add_argument("--seat-x", type=float, default=None,
                    help="override seat_pos x to isolate position-dependent ejections")
parser.add_argument("--park-back", action="store_true",
                    help="spawn the backrest far off-table so only chair-table contacts remain")
parser.add_argument("--rest-back", action="store_true",
                    help="drop the backrest high over open table and measure its true rest pose")
parser.add_argument("--rest-yaw", type=float, default=-90.0,
                    help="slot yaw (deg) for the rest-pose drop — measure AT the deployed yaw")
parser.add_argument("--record", type=str, default="",
                    help="save an npz of rendered frames of reset+settle to this path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record:
    args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"
app = AppLauncher(args).app

import numpy as np
import torch

import robobench
from robobench.core import ENVS


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    from robobench.suites.assembly.configs.envs import _chair_multi_cfg

    scene_cfg = _chair_multi_cfg()
    scene_cfg.reset_pos_jitter = 0.0
    scene_cfg.reset_yaw_deg = 0.0
    scene_cfg.seat_jitter = 0.0
    scene_cfg.seat_yaw_deg = 0.0
    scene_cfg.shuffle_slots = False
    if args.yaw is not None:
        scene_cfg.seat_yaw_base = args.yaw
        print(f"[probe] seat_yaw_base OVERRIDE -> {args.yaw}", flush=True)
    if args.seat_x is not None:
        scene_cfg.seat_pos = (args.seat_x, scene_cfg.seat_pos[1])
        print(f"[probe] seat_pos OVERRIDE -> {scene_cfg.seat_pos}", flush=True)
    if args.park_back:
        s = scene_cfg.spawn_slots
        scene_cfg.spawn_slots = ((4.0, 0.0, -90.0),) + tuple(s[1:])
        print("[probe] back PARKED off-table", flush=True)
    if args.rest_back:
        scene_cfg.seat_pos = (0.9, 0.0)
        scene_cfg.spawn_slots = ((-0.6, 0.0, -90.0), (3.0, -0.3, 0.0), (3.0, 0.3, 0.0))
        print("[probe] rest-pose mode: chair parked at +0.9, nuts off-table", flush=True)
    env = ENVS.get("assembly.chair.multi.osc")().build(
        num_envs=1, device=device, scene_cfg=scene_cfg)
    scene = env.scene
    c = scene.cfg
    no_op = torch.zeros(1, env.robot.action_dim, device=device)

    frames: list[np.ndarray] = []
    annot = None
    if args.record:
        import omni.replicator.core as rep

        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((2.15, -2.35, 1.50)) + o),
                                tuple(np.array((0.0, -0.05, c.surface_z + 0.30)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[probe] camera ready {np.asarray(annot.get_data()).shape}", flush=True)

    def grab() -> None:
        if annot is None:
            return
        for _ in range(3):
            env.sim.render()
        arr = np.asarray(annot.get_data())
        if arr.size:
            frames.append(arr[..., :3].astype(np.uint8).copy())

    def snap(tag: str) -> None:
        parts = {"base": scene.base, "back": scene.back,
                 "nut_0": scene.nuts[0], "nut_1": scene.nuts[1]}
        bits = []
        for name, obj in parts.items():
            p = obj.data.root_pos_w[0]
            v = obj.data.root_lin_vel_w[0]
            bits.append(f"{name}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f} "
                        f"|v|={float(v.norm()):.2f})")
        print(f"[probe] {tag:10s} " + "  ".join(bits), flush=True)
        bv = scene.back.data.root_lin_vel_w[0]
        if float(bv.norm()) > 1.0:
            print(f"[probe]   back VEL VECTOR = ({bv[0]:+.2f},{bv[1]:+.2f},{bv[2]:+.2f})",
                  flush=True)

    def hands(tag: str) -> None:
        """Measured hand/pinch/finger state + pinch->site distances (the engage inputs)."""
        if not getattr(scene, "_gw_on", False) or not scene._gw_resolve_hands():
            print("[probe]   grasp contract OFF", flush=True)
            return
        from isaaclab.utils.math import quat_apply
        for hi, (art, hand_i, fingers) in enumerate(scene._gw_arts):
            hp = art.data.body_pos_w[0, hand_i]
            hq = art.data.body_quat_w[0, hand_i]
            gap = art.data.joint_pos[0, fingers].sum()
            vel = art.data.joint_vel[0, fingers].abs().sum()
            approach = torch.zeros(1, 3, device=device)
            approach[0, 2] = scene.GRASP_PINCH_OFFSET
            pinch = (hp.unsqueeze(0) + quat_apply(hq.unsqueeze(0), approach))
            dists = scene._gw_site_dists(pinch.expand(env.num_envs, 3))[0]
            print(f"[probe]   {tag} hand{hi}: pos=({hp[0]:+.3f},{hp[1]:+.3f},{hp[2]:+.3f}) "
                  f"gap={float(gap) * 1000:.1f}mm fvel={float(vel) * 1000:.2f} "
                  f"site_d={[f'{v * 1000:.0f}' for v in dists.tolist()]}mm", flush=True)

    env.reset()
    if args.rest_back:  # teleport the panel HIGH over open table, face-up AT THE DEPLOYED
        # YAW, and let it find its true rest (the roll direction does NOT rotate cleanly
        # with yaw — landing dynamics dominate — so measure at the yaw the scene uses)
        import math as _math

        half = _math.radians(args.rest_yaw) / 2
        cy_r, sy_r = _math.cos(half), _math.sin(half)
        c45 = _math.cos(_math.pi / 4)
        # q = qz(rest_yaw) * qx(-90)
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1], st[0, 2] = 0.30, 0.0, scene.cfg.surface_z + 0.30
        st[0, 3], st[0, 4] = cy_r * c45, -cy_r * c45
        st[0, 5], st[0, 6] = -sy_r * c45, sy_r * c45
        st[0, 0:3] += env.iscene.env_origins[0]
        scene.back.write_root_state_to_sim(st, torch.tensor([0], device=device))
        print(f"[probe] rest drop at yaw {args.rest_yaw}", flush=True)

    def base_contact(tag: str) -> None:
        try:
            f = scene.base.root_physx_view.get_net_contact_forces(env.sim.get_physics_dt())[0]
            print(f"[probe]   {tag} base net contact F=({f[0]:+.1f},{f[1]:+.1f},{f[2]:+.1f}) N",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[probe]   contact force read failed: {exc!r}", flush=True)

    snap("post-reset")
    hands("post-reset")
    grab()
    n_steps = 400 if args.rest_back else (150 if args.record else 90)
    for i in range(n_steps):
        env.step(no_op, render=bool(args.record))
        if i % 10 == 0 or i < 5:
            snap(f"step {i}")
            if i < 4:
                base_contact(f"step {i}")
        if i % 2 == 0:
            grab()
    if args.rest_back:
        import math as _math

        from isaaclab.utils.math import quat_mul as _qm

        bp = scene.back.data.root_pos_w[0] - env.iscene.env_origins[0]
        bq = scene.back.data.root_quat_w[0]
        half = _math.radians(-args.rest_yaw) / 2  # slot-frame quat: qz(yaw)^-1 * q_rest
        qinv = torch.tensor([_math.cos(half), 0.0, 0.0, _math.sin(half)], device=device)
        ql = _qm(qinv.unsqueeze(0), bq.unsqueeze(0))[0]
        print(f"[probe] REST POSE: pos=({bp[0]:+.4f},{bp[1]:+.4f},{bp[2]:+.4f}) "
              f"quat=({bq[0]:+.4f},{bq[1]:+.4f},{bq[2]:+.4f},{bq[3]:+.4f}) "
              f"slot_local=({ql[0]:+.4f},{ql[1]:+.4f},{ql[2]:+.4f},{ql[3]:+.4f}) "
              f"drift_xy=({float(bp[0]) - 0.30:+.4f},{float(bp[1]):+.4f}) "
              f"z_above_surface={float(bp[2]) - scene.cfg.surface_z:+.4f}", flush=True)
    snap("settled")
    if args.record and frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.record, frames=arr, env="assembly.chair.multi.osc")
        print(f"[probe] saved {arr.shape} -> {args.record}", flush=True)
    print("PROBE_DONE", flush=True)
    # no env.close(): it hangs this Isaac build's teardown (measured 14+ min at 100% CPU);
    # _hard_exit's watchdog nukes the process instead


def _hard_exit() -> None:
    import os as _os
    import threading as _threading

    _threading.Timer(10.0, lambda: _os._exit(0)).start()
    app.close()
    _os._exit(0)


if __name__ == "__main__":
    main()
    _hard_exit()
