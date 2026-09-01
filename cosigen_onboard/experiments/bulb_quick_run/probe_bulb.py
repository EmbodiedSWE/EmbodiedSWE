"""probe_bulb — the three go/no-go questions for the Franka bulb solve, no robot.

1. GEOMETRY: bulb + socket extents/origin offsets (BBoxCache on the authored lying pose) — feeds
   the grasp / stage / touch-detect math.
2. RELEASE STABILITY: teleport each bulb upright with its cap resting in the socket bore (per-env
   initial tilt 0/3/6/9/12 deg), zero velocity, NO forces, settle ~2 s. If the bulb stays upright,
   the place-then-regrasp plan is viable; if it tips, the plan changes.
3. PRESS WINDOW: re-stage upright, then per-env press (N) + capped twist (-0.15 N*m, cap -3 rad/s)
   at the scene's dt=1/240. Real threading <=> descent/turn ~= the thread pitch (the nut session
   caught its scene tunneling exactly this way).

    cd <repo root>
    .venv/bin/python experiments/bulb_franka_osc_fable/probe_bulb.py --headless
"""

from __future__ import annotations

import argparse
import math
import os
import threading

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--presses", type=str, default="-0.3,-0.6,-1.2,-2.5,-5.0", help="per-env -z force (N); also sets num_envs")
parser.add_argument("--tilts", type=str, default="0,3,6,9,12", help="per-env initial tilt (deg) for the release-stability phase")
parser.add_argument("--twist", type=float, default=-0.15, help="tighten torque about z (N*m)")
parser.add_argument("--target_wz", type=float, default=-3.0, help="spin-rate cap (rad/s)")
parser.add_argument("--bulb_friction", type=float, default=-1.0, help="override bulb friction dial (<0 = scene default 0.01)")
parser.add_argument("--dt", type=float, default=0.0, help="sim dt override (0 = scene default 1/240)")
parser.add_argument("--settle_sec", type=float, default=2.5, help="release-stability settle time (s)")
parser.add_argument("--screw_sec", type=float, default=30.0, help="press+twist phase (s)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402
from robobench.suites.assembly.scenes import BulbAssemblySceneCfg  # noqa: E402

GAP = 0.001          # staged free-end height above the bore mouth (stock smoke value)
BULB_FREE_END = 0.004  # thread free end above the bulb origin (stock smoke value)
SETTLE = 30


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    presses = [float(s) for s in args.presses.split(",")]
    tilts = [float(s) for s in args.tilts.split(",")]
    n = len(presses)
    assert len(tilts) == n, "tilts and presses must have the same count (both are per-env)"
    robobench.discover()
    scene_kw = dict(surface_z=0.20, bulb_row_x0=-0.12)  # the solve's layout (reach band + bulb at 0.43 m)
    if args.bulb_friction >= 0:
        scene_kw["bulb_friction"] = args.bulb_friction
    overrides = {"scene_cfg": BulbAssemblySceneCfg(**scene_kw)}
    if args.dt > 0:
        overrides["sim_overrides"] = {"dt": args.dt}
    env = ENVS.get("assembly.bulb")().build(num_envs=n, device=device, **overrides)
    print(f"[cfg] dt={env.dt:.6f} ({1 / env.dt:.0f} Hz) twist={args.twist} cap={args.target_wz} "
          f"bulb_friction={env.scene.cfg.bulb_friction}", flush=True)
    scene = env.scene
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    env.reset()

    bulb, socket = scene.bulbs[0], scene.sockets[0]
    c = scene.cfg
    stage_gap = c.socket_opening_z + GAP - BULB_FREE_END  # bulb-origin height above the socket origin at staging

    # ---- per-shape materials: how many colliders does each part expose? (per-shape override lever)
    mats_b = bulb.root_physx_view.get_material_properties()
    mats_s = socket.root_physx_view.get_material_properties()
    print(f"[mat] bulb shapes={tuple(mats_b.shape)} vals={mats_b[0].tolist()}", flush=True)
    print(f"[mat] socket shapes={tuple(mats_s.shape)} vals={mats_s[0].tolist()}", flush=True)

    # ---- geometry (authored lying pose; offsets are pose-invariant) ----------------------------
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def box(path):
        rng = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return rng.GetMin(), rng.GetMax()

    o0 = env.iscene.env_origins[0].tolist()
    wx, wy = c.workbench_pos
    bx, by = c.bulb_init_xy[0]
    bulb_auth = (o0[0] + wx + bx, o0[1] + wy + by, c.surface_z + c.bulb_init_z)  # authored origin (pre-jitter)
    sx, sy = c.socket_slots[0]
    sock_auth = (o0[0] + wx + sx, o0[1] + wy + sy, c.surface_z)
    (bl, bh) = box("/World/envs/env_0/Bulb_0")
    (sl, sh) = box("/World/envs/env_0/Socket_0")
    print(f"[geom] bulb lying bbox mm: dx={1e3 * (bh[0] - bl[0]):.1f} dy={1e3 * (bh[1] - bl[1]):.1f} "
          f"dz={1e3 * (bh[2] - bl[2]):.1f} | origin: y_min-y0={1e3 * (bl[1] - bulb_auth[1]):.1f} "
          f"y_max-y0={1e3 * (bh[1] - bulb_auth[1]):.1f} z_min-z0={1e3 * (bl[2] - bulb_auth[2]):.1f} "
          f"z_max-z0={1e3 * (bh[2] - bulb_auth[2]):.1f}", flush=True)
    print(f"[geom] socket bbox mm: dx={1e3 * (sh[0] - sl[0]):.1f} dy={1e3 * (sh[1] - sl[1]):.1f} "
          f"top-above-origin={1e3 * (sh[2] - sock_auth[2]):.1f} (cfg bore mouth {1e3 * c.socket_opening_z:.1f})", flush=True)
    print(f"[pin] socket root xy per env: {[f'({p[0]:+.2f},{p[1]:+.2f})' for p in socket.data.root_pos_w.tolist()]}", flush=True)

    import isaaclab.utils.math as lm

    def tilt_deg():  # bulb local +z vs world +z
        ez = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3)
        up = lm.quat_apply(bulb.data.root_quat_w, ez)
        return torch.rad2deg(torch.acos(up[:, 2].clamp(-1, 1)))

    def height_mm():
        return (bulb.data.root_pos_w - socket.data.root_pos_w)[:, 2] * 1e3

    def stage_upright(tilt_list):
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = socket.data.root_pos_w
        st[:, 2] += stage_gap
        for k, tdeg in enumerate(tilt_list):
            a = math.radians(tdeg) / 2
            st[k, 3], st[k, 4] = math.cos(a), math.sin(a)  # tilt about x
        bulb.write_root_state_to_sim(st, all_ids)

    settle_steps = round(args.settle_sec / env.dt)
    screw_steps = round(args.screw_sec / env.dt)
    report = max(1, round(1.25 / env.dt))

    # ---- phase 1: reset layout settles briefly (also confirms the lying spawn is at rest) -------
    for i in range(SETTLE):
        env.step(no_action, render=False)
    print(f"[lying] bulb z above surface mm: {[f'{v:.1f}' for v in ((bulb.data.root_pos_w[:, 2] - (env.iscene.env_origins[:, 2] + c.surface_z)) * 1e3).tolist()]}", flush=True)

    # ---- phase 2: RELEASE STABILITY — rest the cap in the bore, no forces, per-env tilt ---------
    stage_upright(tilts)
    for i in range(1, settle_steps + 1):
        env.step(no_action, render=False)
        if i % report == 0:
            print(f"  [rel] t={i * env.dt:4.1f}s h_mm: " + " ".join(f"{v:6.1f}" for v in height_mm().tolist())
                  + " | tilt_deg: " + " ".join(f"{v:5.1f}" for v in tilt_deg().tolist()), flush=True)
    h, t = height_mm(), tilt_deg()
    print("RELEASE VERDICT (init tilt -> final height mm, tilt deg, upright=tilt<15):", flush=True)
    for k in range(n):
        print(f"  tilt0 {tilts[k]:4.1f} -> h {h[k]:6.1f} mm  tilt {t[k]:5.1f} deg  upright={int(t[k] < 15)}", flush=True)

    # ---- phase 3: PRESS WINDOW — re-stage upright, per-env press + capped twist ----------------
    stage_upright([0.0] * n)
    for i in range(SETTLE):
        env.step(no_action, render=False)
    press = torch.tensor(presses, device=device)
    f = torch.zeros(n, 1, 3, device=device)
    f[:, 0, 2] = press
    ex = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3)

    def yaw_now():
        e = lm.quat_apply(bulb.data.root_quat_w, ex)
        return torch.atan2(e[:, 1], e[:, 0])

    yaw_prev, yaw_acc = None, torch.zeros(n, device=device)
    h0 = height_mm().clone()
    for i in range(1, screw_steps + 1):
        tq = torch.zeros(n, 1, 3, device=device)
        tq[bulb.data.root_ang_vel_w[:, 2] > args.target_wz, 0, 2] = args.twist
        bulb.set_external_force_and_torque(f, tq, is_global=True)
        env.step(no_action, render=False)
        y = yaw_now()
        if yaw_prev is not None:
            yaw_acc += (y - yaw_prev + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_prev = y
        if i % report == 0:
            turns = yaw_acc / (2 * torch.pi)
            print(f"  [scr] t={i * env.dt:4.1f}s h_mm: " + " ".join(f"{v:6.1f}" for v in height_mm().tolist())
                  + " | turns: " + " ".join(f"{v:+6.2f}" for v in turns.tolist()), flush=True)

    seated = scene.seated()
    h, lat = height_mm(), ((bulb.data.root_pos_w - socket.data.root_pos_w)[:, :2].norm(dim=-1) * 1e3)
    turns = yaw_acc / (2 * torch.pi)
    drop = h0 - h
    print("PRESS VERDICT (press -> final h, lat, turns, mm-descended-per-turn [~pitch = real threading]):", flush=True)
    for k in range(n):
        mmpt = (drop[k] / -turns[k]).item() if turns[k] < -0.25 else float("nan")
        print(f"  press {presses[k]:+5.2f} N -> h {h[k]:6.1f} mm  lat {lat[k]:5.1f} mm  turns {turns[k]:+6.2f} "
              f" mm/turn {mmpt:5.2f}  seated={int(seated[k, 0])}", flush=True)
    threading.Timer(10.0, lambda: os._exit(0)).start()  # Isaac teardown hangs; free the GPU regardless
    env.close()


if __name__ == "__main__":
    main()
    app.close()
    os._exit(0)
