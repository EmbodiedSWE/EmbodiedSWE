"""Smoke / oracle test for CratePackingScene — NullRobot, teleport staging, RECORDED.

One linear run:
  1. show    — settle the reset layout (lid open, cargo scattered) so you can see it;
  2. stage   — teleport the cargo into the REFERENCE PACKING inside the crate
               (slab flat on the floor -> 4 tubes side by side on the slab -> brick on
               the tubes), settling after each layer;
  3. close   — torque the lid down about its hinge until it falls onto the rim;
  4. verify  — report inside()/lid_angle/lid_seated()/packed() every phase, final verdict;
  5. (--jitter N) — optional calibration: re-stage with per-part xy jitter of N mm and
               report whether the lid still seats (the measured-tolerance sweep).

ALWAYS records video: frames are captured through the viewport rgb annotator (same
mechanism as the render server) and saved to a npz (and HDFS if reachable) for the
standard stitch-to-h264 pipeline. Bodies are driven straight through scene handles; the
NullRobot applies nothing.

Run (on a GPU node with the isaaclab env):
    python -m robobench.suites.packing.smokes.crate_packing_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo", action="store_true", default=False,
                    help="record ONE clean successful run only (no sweep, no negative "
                         "control) — the user-facing deliverable video")
parser.add_argument("--jitter", type=float, default=0.0, help="calibration xy jitter per part (m)")
parser.add_argument("--jitter_sweep", action="store_true", default=False,
                    help="run the calibration sweep: stage+close at increasing jitter, "
                         "3 seeds each, and report the lid-seat rate per jitter level")
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--out", type=str, default="crate_smoke_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (same as the render server): kit mis-decodes the L20 driver version and
# silently rejects RTX -> annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("packing.crate")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.3, -1.3, 1.1)) + o),
                                tuple(np.array((0.0, 0.0, 0.15)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check RTX recipe "
                  "(driver-version override, NVIDIA_DRIVER_CAPABILITIES)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int, render: bool = True) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=render)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                data = annot.get_data()
                arr = np.asarray(data)
                if step_i == 0:
                    print(f"[smoke] first capture: dtype={arr.dtype} shape={arr.shape}", flush=True)
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        inside = scene.inside()[0].int().tolist()
        print(f"[smoke] {tag:8s} | inside={inside} lid_deg={scene.lid_angle_deg()[0]:.1f} "
              f"lid_seated={bool(scene.lid_seated()[0])} packed={bool(scene.packed()[0])} "
              f"frames={len(frames)}",
              flush=True)

    jitter_amp = [args.jitter]  # mutable so the sweep can vary it
    # Drop height for staging. 0 = teleport straight to the in-crate pose. The CALIBRATION
    # sweep sets this >0: teleporting a jittered part INTO the wall makes PhysX resolve the
    # penetration by pushing it back inside (the walls self-center everything), which
    # measured 3/3 lid-seat at 25mm jitter even on a crate with only ±12mm slab slack.
    # Releasing from above the rim — like a robot placing the part — is the honest measure.
    drop_h = [0.0]

    # Yaw jitter (rad). Translation error self-centers off the wall funnel (measured 3/3
    # lid-seat at 25mm drop-in), but a part released with a YAW error jams diagonally —
    # rotational tolerance is the binding precision requirement for this crate.
    yaw_amp = [0.0]

    def teleport(name: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        body = scene.cargo[name]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = env.iscene.env_origins + torch.tensor(pos, device=device)
        if drop_h[0] > 0:  # release above the rim so jittered parts can't spawn inside walls
            rim_z = scene.cfg.surface_z + scene.cfg.wall_t + scene.cfg.interior[2]
            st[:, 2] = torch.maximum(
                st[:, 2], env.iscene.env_origins[:, 2] + rim_z + drop_h[0])
        q = torch.tensor(quat, device=device).expand(n, 4).clone()
        if yaw_amp[0] > 0:
            half = (torch.rand(n, device=device) * 2 - 1) * yaw_amp[0] / 2
            qz = torch.stack([torch.cos(half), torch.zeros_like(half),
                              torch.zeros_like(half), torch.sin(half)], dim=1)
            w1, x1, y1, z1 = qz.unbind(1)
            w2, x2, y2, z2 = q.unbind(1)
            q = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                             w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                             w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dim=1)
        st[:, 3:7] = q
        if jitter_amp[0] > 0:
            st[:, 0:2] += (torch.rand(n, 2, device=device) * 2 - 1) * jitter_amp[0]
        body.write_root_state_to_sim(st, all_ids)

    # --- staged reference packing + lid close, reusable for the calibration sweep ---
    cx_, cy_ = scene.cfg.crate_pos
    t_ = scene.cfg.wall_t
    z0_ = scene.cfg.surface_z
    slab_ = scene.cfg.manifest[0][2]
    tube_r_, _tube_l = scene.cfg.manifest[1][2]
    brick_ = scene.cfg.manifest[5][2]
    q_tube_ = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)

    def stage_reference() -> None:
        teleport("slab", (cx_, cy_, z0_ + t_ + slab_[2] / 2 + 0.003))
        step(40)
        zt = z0_ + t_ + slab_[2] + tube_r_ + 0.003
        for i in range(4):
            y = cy_ - 1.5 * (2 * tube_r_ + 0.006) + i * (2 * tube_r_ + 0.006)
            teleport(f"tube_{i}", (cx_, y, zt), q_tube_)
            step(20)
        step(40)
        teleport("brick", (cx_, cy_, z0_ + t_ + slab_[2] + 2 * tube_r_ + brick_[2] / 2 + 0.004))
        step(50)

    def close_lid(target_deg: float = 0.5) -> None:
        for _k in range(600):
            ang = float(scene.lid_angle_deg()[0])
            if ang <= target_deg and float(scene.lid.data.root_lin_vel_w[0].norm()) < 0.02:
                break
            tq = torch.zeros(n, 1, 3, device=device)
            tq[:, 0, 0] = 0.8 if ang > 8.0 else 0.15
            scene.lid.set_external_force_and_torque(torch.zeros(n, 1, 3, device=device), tq)
            step(1)
        scene.lid.set_external_force_and_torque(torch.zeros(n, 1, 3, device=device),
                                                torch.zeros(n, 1, 3, device=device))
        step(80)

    if args.jitter_sweep:
        print("[smoke] CALIBRATION SWEEP (drop-in release @10mm xy, yaw -> lid-seat rate, "
              "3 seeds each)", flush=True)
        results = {}
        for yaw_deg in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0):
            hits = 0
            for seed in range(3):
                torch.manual_seed(seed)
                jitter_amp[0] = 0.010
                yaw_amp[0] = math.radians(yaw_deg)
                drop_h[0] = 0.03  # release each part 3cm above the rim, let it fall in
                env.reset()
                step(30)
                stage_reference()
                drop_h[0] = 0.0
                yaw_amp[0] = 0.0
                step(60)  # extra settle: dropped parts bounce/roll before they come to rest
                close_lid()
                seated = bool(scene.lid_seated()[0]) and bool(scene.inside().all(dim=1)[0])
                hits += int(seated)
                print(f"[smoke]   yaw={yaw_deg:.0f}deg seed={seed}: packed={seated} "
                      f"lid_deg={float(scene.lid_angle_deg()[0]):.1f}", flush=True)
            results[yaw_deg] = hits
        print("[smoke] SWEEP RESULT: " +
              " | ".join(f"{k:.0f}deg: {v}/3" for k, v in results.items()), flush=True)
        jitter_amp[0] = 0.0

    env.reset()
    report("reset")
    step(60)
    report("show")
    # The lid MUST still be open after settling — if gravity slammed it shut, the hinge
    # convention regressed and everything downstream is meaningless. Fail loudly.
    assert float(scene.lid_angle_deg()[0]) > 60.0, \
        f"lid fell closed during show (deg={float(scene.lid_angle_deg()[0]):.1f}) — hinge broken"

    # --- stage: reference packing, then close (shared helpers) ---
    cx, cy = cx_, cy_
    t = t_
    slab = slab_
    tube_r = tube_r_
    brick = brick_
    stage_reference()
    report("staged")
    close_lid()
    report("close")

    ok = bool(scene.packed()[0])
    print(f"[smoke] RESULT: {'PACKED — SUCCESS' if ok else 'NOT PACKED — FAIL'}", flush=True)

    if args.demo:  # deliverable video = the one clean run above; stop here
        if frames:
            arr = np.stack(frames, axis=0)
            np.savez_compressed(args.out, frames=arr, env="packing.crate")
            print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
            os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                      f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print("CRATE_SMOKE_DONE", flush=True)
        env.close()
        return

    # --- negative control: a BAD packing must BLOCK the lid (the lid is the judge) ---
    # Re-open the lid, stand the brick UPRIGHT on the tube layer (stack 0.04+0.05+0.16 =
    # 0.25 m > interior 0.18 m -> pokes above the rim), close again: packed MUST be False
    # and the lid must rest visibly above the rim.
    for k in range(500):
        if float(scene.lid_angle_deg()[0]) >= 100.0:
            break
        tq = torch.zeros(n, 1, 3, device=device)
        tq[:, 0, 0] = -1.0  # opening torque (negative x in the new convention)
        scene.lid.set_external_force_and_torque(torch.zeros(n, 1, 3, device=device), tq)
        step(1)
    scene.lid.set_external_force_and_torque(torch.zeros(n, 1, 3, device=device),
                                            torch.zeros(n, 1, 3, device=device))
    step(30)
    q_up = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # brick long axis up
    z_upright = z0_ + t + slab[2] + 2 * tube_r + brick[0] / 2 + 0.004
    teleport("brick", (cx, cy, z_upright), q_up)
    step(50)
    report("bad-stage")
    for k in range(600):
        ang = float(scene.lid_angle_deg()[0])
        lid_v = float(scene.lid.data.root_lin_vel_w[0].norm())
        if ang <= 25.0 and lid_v < 0.02:  # resting on the proud brick
            break
        tq = torch.zeros(n, 1, 3, device=device)
        tq[:, 0, 0] = 0.8 if ang > 30.0 else 0.15
        scene.lid.set_external_force_and_torque(torch.zeros(n, 1, 3, device=device), tq)
        step(1)
    scene.lid.set_external_force_and_torque(torch.zeros(n, 1, 3, device=device),
                                            torch.zeros(n, 1, 3, device=device))
    step(80)
    report("bad-close")
    bad_ok = bool(scene.packed()[0])
    print(f"[smoke] NEGATIVE CONTROL: packed={bad_ok} "
          f"({'FAIL — bad packing passed!' if bad_ok else 'correctly rejected'})", flush=True)

    # --- save frames ---
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="packing.crate")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/"
                       f"{os.path.basename(args.out)}")
        print(f"[smoke] hdfs upload rc={rc} -> {args.hdfs_dir}/{os.path.basename(args.out)}",
              flush=True)
    # Combined verdict: the oracle pack must succeed AND the negative control must be
    # rejected (an `ok and not bad_ok` bool used to be computed here and then never
    # read — a failing negative control could not fail the smoke).
    all_ok = ok and not bad_ok
    print(f"[smoke] RESULT: {'ALL PASS' if all_ok else 'FAIL'} "
          f"(oracle packed={ok}, negative control rejected={not bad_ok})", flush=True)
    print("CRATE_SMOKE_DONE", flush=True)
    env.close()


def _hard_exit_teardown() -> None:
    """Kit teardown regularly hangs inside env.close()/app.close() (100% CPU spin),
    wedging headless runs after everything is printed — the repo's standard hard-exit
    (see robobench/scripts/smoke.py): a watchdog guarantees the process ends."""
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)


if __name__ == "__main__":
    main()
    _hard_exit_teardown()
