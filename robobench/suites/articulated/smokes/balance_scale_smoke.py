"""Smoke / oracle test for BalanceScaleScene — NullRobot, RECORDED.

The full weigh-and-sort pipeline, exactly the strategy an agent should discover:
  1. boot — beam level and settled with empty trays;
  2. CALIBRATION (the designed-property proof):
     (a) mass-gap sweep: pairs at the minimum gap and above, boxes CENTRED — the
         verdict must match the true heavier side every time, within a bounded settle
         time;
     (b) off-centre sweep: a LIGHTER box placed progressively off-centre outward —
         beyond some offset the lever arm wins and the reading flips (misleading by
         design); report the flip point;
  3. SORT — insertion-sort the five boxes using only scale verdicts (teleport a pair
     onto the trays, wait for settle, read, remove), counting weighings; then place
     the boxes on the shelf in the inferred order and verify success();
  4. negative control — reshuffle two adjacent boxes on the shelf: success() must go
     False.

The smoke may read `scene._masses` ONLY to verify verdicts (privileged, smoke-only);
the sort itself uses nothing but the beam.

    python -m robobench.suites.articulated.smokes.balance_scale_smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=6)
parser.add_argument("--out", type=str, default="scale_smoke_frames.npz")
parser.add_argument("--hdfs_dir", type=str, default="",
                    help="optional HDFS dir to upload the frames npz to ('' = no upload)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import os

import numpy as np
import torch

import robobench
from robobench.core import ENVS

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[scale-smoke] {'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robobench.discover()
    env = ENVS.get("articulated.scale")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.arange(env.num_envs, device=device)
    o = env.iscene.env_origins

    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o0 = o[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.85, 0.60)) + o0),
                                tuple(np.array((0.0, 0.12, 0.12)) + o0),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        print(f"[scale-smoke] camera ready shape={np.asarray(annot.get_data()).shape}",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[scale-smoke] camera setup FAILED ({exc!r})", flush=True)

    step_i = 0

    def step(k: int = 1) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=True)
            if annot is not None and step_i % args.record_every == 0:
                for _f in range(3):
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    z0 = c.surface_z
    sx, sy = c.scale_pos

    def put_box(k: int, pos) -> None:
        st = torch.zeros(env.num_envs, 13, device=device)
        st[:, 0:3] = o + torch.tensor(pos, device=device)
        st[:, 3] = 1.0
        scene.boxes[k].write_root_state_to_sim(st, ids)

    def park_box(k: int) -> None:
        bx = (k - (c.n_boxes - 1) / 2) * c.box_gap
        put_box(k, (bx, c.box_row_y, z0 + c.box_size[2] / 2 + 0.002))

    def tray_top_z(side: int) -> float:
        return float(scene.trays[side].data.root_pos_w[0, 2] - o[0, 2]) + c.tray_size[2] / 2

    def on_tray(i: int, j: int) -> tuple[bool, bool]:
        """Are boxes i (left) and j (right) still sitting on their trays?"""
        outs = []
        for k, side in ((i, 0), (j, 1)):
            p = (scene.boxes[k].data.root_pos_w - o)[0]
            t = (scene.trays[side].data.root_pos_w - o)[0]
            outs.append(bool(abs(float(p[0] - t[0])) < 0.09
                             and abs(float(p[1] - t[1])) < 0.09
                             and float(p[2]) > float(t[2])))
        return outs[0], outs[1]

    def weigh(i: int, j: int, off_i: float = 0.0, off_j: float = 0.0,
              max_wait: int = 480) -> tuple[int, int]:
        """Load box i LEFT, box j RIGHT (optional outward x offsets) with the beam
        ARRESTED, release, wait for a settled reading, re-arrest, unload. Returns
        (verdict, settle_steps): +1 = right (j) heavier, -1 = left (i), 0 = level."""
        scene.arrest(True)
        put_box(i, (sx - c.lever - off_i, sy, tray_top_z(0) + c.box_size[2] / 2 + 0.006))
        put_box(j, (sx + c.lever + off_j, sy, tray_top_z(1) + c.box_size[2] / 2 + 0.006))
        step(40)  # boxes settle onto arrested (level) trays
        scene.arrest(False)
        t = 0
        for t in range(max_wait):
            step(1)
            if t > 40 and bool(scene.beam_settled()[0]):
                break
        v = int(scene.verdict()[0])
        ang = float(scene.beam_angle_deg()[0])
        bx_i = float((scene.boxes[i].data.root_pos_w - o)[0, 0]) - sx
        bx_j = float((scene.boxes[j].data.root_pos_w - o)[0, 0]) - sx
        tx_l = float((scene.trays[0].data.root_pos_w - o)[0, 0]) - sx
        tx_r = float((scene.trays[1].data.root_pos_w - o)[0, 0]) - sx
        print(f"[scale-smoke]   verdict-geom: box_l_x={bx_i * 1000:+.0f}mm "
              f"box_r_x={bx_j * 1000:+.0f}mm tray_l={tx_l * 1000:+.0f} "
              f"tray_r={tx_r * 1000:+.0f} (lever={c.lever * 1000:.0f})", flush=True)
        oi, oj = on_tray(i, j)
        if not (oi and oj):
            print(f"[scale-smoke]   WARNING box fell off during weigh({i},{j}): "
                  f"left_on={oi} right_on={oj}", flush=True)
        print(f"[scale-smoke]   weigh({i},{j}) angle={ang:+.2f}deg settle={t}", flush=True)
        scene.arrest(True)
        step(30)
        park_box(i)
        park_box(j)
        step(20)
        return v, t

    env.reset()
    step(60)
    scene.arrest(False)
    step(90)  # empty beam, released: must sit level and settle
    ang0 = float(scene.beam_angle_deg()[0])
    check("boot-level", abs(ang0) < c.read_min_deg and bool(scene.beam_settled()[0]),
          f"released empty beam angle {ang0:.2f} deg "
          f"w={float(scene.beam.data.root_ang_vel_w[0, 1]):+.3f}")
    # JOINT DIAGNOSTIC: drive the released beam with a big pivot torque both ways.
    # A working revolute slams to each +-8 deg stop in ~0.1 s; a locked joint stays
    # put and tells us the pivot itself (not the loads) is the problem.
    jp = env.stage.GetPrimAtPath("/World/envs/env_0/beam_pivot")
    print(f"[scale-smoke] joint prim valid={jp.IsValid()} type={jp.GetTypeName()}",
          flush=True)
    for tq in (1.5, -1.5):
        scene.dbg_torque[0] = tq
        step(90)
        q = [round(float(v), 4) for v in scene.beam.data.root_quat_w[0]]
        print(f"[scale-smoke]   diag torque {tq:+.1f}Nm -> angle="
              f"{float(scene.beam_angle_deg()[0]):+.2f}deg w_y="
              f"{float(scene.beam.data.root_ang_vel_w[0, 1]):+.3f} quat={q}", flush=True)
    scene.dbg_torque[0] = 0.0
    step(60)
    scene.arrest(True)
    truth = scene._masses[0].tolist()
    print(f"[scale-smoke] hidden masses (privileged, smoke-only): "
          f"{[round(v, 3) for v in truth]}", flush=True)

    # 2a. mass-gap calibration (centred): every verdict must match the truth
    print("[scale-smoke] CALIBRATION A: centred verdicts vs true masses", flush=True)
    good, total, settle_times = 0, 0, []
    order = np.argsort(truth)
    pairs = [(int(order[a]), int(order[a + 1])) for a in range(c.n_boxes - 1)]  # min gaps
    pairs += [(int(order[0]), int(order[-1]))]  # max gap
    for i, j in pairs:
        v, t = weigh(i, j)
        expect = 1 if truth[j] > truth[i] else -1
        ok = v == expect
        good += int(ok)
        total += 1
        settle_times.append(t)
        print(f"[scale-smoke]   weigh(box{i} {truth[i]:.2f}kg | box{j} {truth[j]:.2f}kg) "
              f"-> v={v} expect={expect} settle={t} steps {'OK' if ok else 'WRONG'}",
              flush=True)
    check("calibration-centred", good == total,
          f"{good}/{total} correct, settle median {int(np.median(settle_times))} steps")

    # 2b. off-centre misleading sweep with CONTROLLED masses (privileged, smoke-only,
    # restored after). Flip needs off > lever*(hi/lo - 1) and must stay clear of the
    # tray edge: v14 telemetry showed ratio 1.2's flip (~60 mm with the heavy box's
    # own 6 mm outboard drift) collides with box fall-off (~75 mm) — the box balanced
    # at 65 and DROPPED OFF the tray at 80 (box_r_x jumped to +417). Ratio 1.1 flips
    # at ~30 mm, comfortably inside the tray.
    saved_masses = scene._masses.clone()
    hi, lo = int(order[-1]), int(order[0])
    scene._masses[0, hi] = 0.55
    scene._masses[0, lo] = 0.50
    scene._apply_masses(torch.arange(env.num_envs, device=device))
    truth[hi], truth[lo] = 0.55, 0.50
    print("[scale-smoke] CALIBRATION B: off-centre flip sweep "
          f"(left box{hi} {truth[hi]:.2f}kg vs right box{lo} {truth[lo]:.2f}kg)", flush=True)
    flip_at = None
    for off_mm in (0, 10, 20, 30, 40, 50, 60):
        v, t = weigh(hi, lo, off_j=off_mm / 1000.0)
        print(f"[scale-smoke]   right offset {off_mm}mm -> v={v}", flush=True)
        if v == 1 and flip_at is None:
            flip_at = off_mm
    check("calibration-offcentre", flip_at is not None and flip_at >= 10,
          f"verdict flips at +{flip_at}mm outward offset (misleading-by-design)")
    scene._masses.copy_(saved_masses)
    scene._apply_masses(torch.arange(env.num_envs, device=device))
    truth = scene._masses[0].tolist()

    # 3. SORT using only the scale: insertion sort on verdicts
    print("[scale-smoke] SORT (insertion, scale verdicts only)", flush=True)
    weighings = 0
    sorted_idx: list[int] = []
    for k in range(c.n_boxes):
        pos = 0
        for pos in range(len(sorted_idx) + 1):
            if pos == len(sorted_idx):
                break
            v, _ = weigh(k, sorted_idx[pos])
            weighings += 1
            if v == 1:  # right (sorted_idx[pos]) heavier -> k goes before it
                break
        sorted_idx.insert(pos, k)
    print(f"[scale-smoke] inferred order (light->heavy): {sorted_idx} in {weighings} "
          f"weighings; true order {list(map(int, order))}", flush=True)
    check("sort-inference", sorted_idx == list(map(int, order)),
          f"{weighings} weighings")

    # shelf placement in inferred order
    for slot, k in enumerate(sorted_idx):
        put_box(k, (scene.slot_x(slot), c.shelf_pos[1], z0 + 0.012 + c.box_size[2] / 2 + 0.004))
        step(20)
    step(60)
    check("success", bool(scene.success()[0]),
          f"slots={scene.slot_of_box()[0].tolist()}")

    # 4. negative control: swap two adjacent boxes -> success must fail
    a, b = sorted_idx[1], sorted_idx[2]
    put_box(a, (scene.slot_x(2), c.shelf_pos[1], z0 + 0.012 + c.box_size[2] / 2 + 0.004))
    put_box(b, (scene.slot_x(1), c.shelf_pos[1], z0 + 0.012 + c.box_size[2] / 2 + 0.004))
    step(60)
    check("negative-control", not bool(scene.success()[0]),
          f"slots={scene.slot_of_box()[0].tolist()}")

    print(f"[scale-smoke] RESULT: {'ALL PASS' if not FAILS else 'FAILED: ' + ','.join(FAILS)}",
          flush=True)

    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="articulated.scale")
        print(f"[scale-smoke] saved {arr.shape} -> {args.out}", flush=True)
        rc = args.hdfs_dir and os.system(f"hdfs dfs -mkdir -p {args.hdfs_dir} 2>/dev/null; "
                       f"hdfs dfs -put -f {args.out} {args.hdfs_dir}/{os.path.basename(args.out)}")
        print(f"[scale-smoke] hdfs upload rc={rc}", flush=True)
    print("SCALE_SMOKE_DONE", flush=True)
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
