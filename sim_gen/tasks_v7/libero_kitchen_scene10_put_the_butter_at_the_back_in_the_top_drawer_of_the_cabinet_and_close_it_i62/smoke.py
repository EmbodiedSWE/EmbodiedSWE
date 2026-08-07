"""Smoke / rubric-REJECTION battery for RammerGalleryScene (sim_gen task
`libero_kitchen_scene10_..._i62`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — butter laid on the apron by contact drop, the
trolley velocity-servo ram down the covered channel into the sunken well, the gate
servo to its closed stop — is the acceptance evidence that the rubric ACCEPTS a
correct outcome). Every teleport here is instrumentation that CONSTRUCTS a wrong
(or partial) outcome as a settled state and asserts the rubric REJECTS it; no probe
in this battery ever reaches success(), and a final audit check asserts exactly that.

  1-2.  settle/no-NaN     — reset layout settles finite: trolley parked outboard in
                            its randomized zone, gate open in its randomized range,
                            both blocks outside on the floor, everything still,
                            score ~0 at rest;
  3-5.  randomization     — READBACK over 8 seeded resets: butter/brick slot
                            assignment varies; per-slot xy jitter and spawn yaw vary;
                            the gate opening y0 and trolley carriage x0 vary;
  6.   null policy        — 240 idle steps -> score ~0, no success;
  7.   seed strategy      — the end state the seed's plan (lower the item into the
                            receptacle from above, then restore it "shut") produces
                            here: butter dropped over the well lands ON THE ROOF
                            (bridging the 26 mm slot), gate slid closed -> rejected;
  8.   mouth camp         — butter just inside the mouth but NOT fully entered ->
                            approach credit only, score <= 0.12;
  9.   mid-channel        — butter half-way down the bore: entered + depth credit,
                            but no success and score <= 0.58 (the well is the goal);
  10.  brick smuggled     — butter settled in the well AND gate closed, but the RED
                            brick inside the bore -> the brick-out clause rejects at
                            the 0.85 cap;
  11.  gate ajar          — butter in the well but the gate parked 20 mm short of
                            its stop -> the closed-band clause rejects;
  12.  latched credit     — from state 11 the butter is yanked back to the floor:
                            the latched score holds (credit does not evaporate),
                            still no success;
  13.  forced order       — gate CLOSED first, butter laid in the loading zone, the
                            trolley rammed by a constant force: the blade advances,
                            engages the butter, and the ram train STALLS at the
                            closed gate plane with the butter outside (the plate
                            covers the whole aperture — there is no path in), the
                            gate holds -> the deposit-before-seal order is
                            physically forced, no success, score stays at the
                            approach sliver;
  14.  rejection audit    — success() was never True at ANY judged point;
  15.  final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_i62.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rammer_gallery")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.85, -0.85, 0.80)) + o),
                                tuple(np.array((-0.05, 0.00, 0.12)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def butter_rel() -> tuple[float, float, float]:
        r = scene._rel(scene.butter)[0]
        return float(r[0]), float(r[1]), float(r[2])

    def report(tag: str) -> None:
        bx, by, bz = butter_rel()
        tx = float(scene.trolley_x()[0])
        gy = float(scene.gate_y()[0])
        s, ok = judge()
        print(f"[smoke] {tag:16s} | butter=({bx:+.3f},{by:+.3f},{bz:.3f}) "
              f"trolley_x={tx:+.3f} gate_y={gy:+.3f} "
              f"inside={bool(scene.inside_gallery(scene.butter)[0])} "
              f"entered={bool(scene._entered[0])} in_well={bool(scene.in_well()[0])} "
              f"closed={bool(scene.gate_closed()[0])} "
              f"brick_out={bool(scene.brick_out()[0])} "
              f"welled={bool(scene._welled[0])} sealed={bool(scene._sealed[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += env.iscene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def write_gate(y: float) -> None:
        """Follower-only re-pose of the gate along its unchanged transverse slide
        (probe constructor; the gallery is never moved)."""
        place(scene.gate, c.gallery_pos[0] + c.gate_x, c.gallery_pos[1] + y, c.gate_z0)

    gx, gy0 = c.gallery_pos
    bh = c.block_size[2] / 2  # 0.0225

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(60)
    report("show")
    fin0 = (torch.isfinite(scene.butter.data.root_state_w).all()
            and torch.isfinite(scene.brick.data.root_state_w).all()
            and torch.isfinite(scene.trolley.data.root_state_w).all()
            and torch.isfinite(scene.gate.data.root_state_w).all())
    tx = float(scene.trolley_x()[0])
    gy = float(scene.gate_y()[0])
    still = (float(scene.butter.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.trolley.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.gate.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, trolley parked outboard in its zone, gate open in "
          "its range, both blocks outside on the floor, everything still",
          bool(fin0) and c.trolley_x_range[0] - 0.005 < tx < c.trolley_x_range[1] + 0.005
          and c.gate_open_range[0] - 0.005 < gy < c.gate_open_range[1] + 0.005
          and still and not bool(scene.inside_gallery(scene.butter)[0])
          and not bool(scene.inside_gallery(scene.brick)[0])
          and butter_rel()[2] < 0.05)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-5. randomization is real =================================
    slots = np.array(c.spawn_slots)
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        row = []
        for body in (scene.butter, scene.brick):
            p = scene._rel(body)[0]
            px, py = float(p[0]), float(p[1])
            d = np.hypot(slots[:, 0] - px, slots[:, 1] - py)
            row += [px, py, int(d.argmin())]
        q = scene.butter.data.root_quat_w[0]
        row.append(math.degrees(2.0 * math.atan2(float(q[3]), float(q[0]))))
        row.append(float(scene.gate_y()[0]))
        row.append(float(scene.trolley_x()[0]))
        reads.append(row)
    arr = np.array(reads)
    print(f"[smoke] randomization readback (bx, by, b_slot, rx, ry, r_slot, "
          f"b_yaw_deg, gate_y0, trolley_x0):\n{arr}", flush=True)
    sigs = {tuple(r[[2, 5]].astype(int)) for r in arr}
    check("randomization: the butter/brick slot assignment varies across seeded "
          f"resets (readback: {len(sigs)} distinct assignments)", len(sigs) >= 2)
    jit = 0.0
    for col, slot_col in ((0, 2), (3, 5)):
        for s_id in range(2):
            grp = arr[arr[:, slot_col] == s_id]
            if len(grp) >= 2:
                jit = max(jit, float((grp[:, col:col + 2].max(axis=0)
                                      - grp[:, col:col + 2].min(axis=0)).max()))
    yaw_spread = float(arr[:, 6].max() - arr[:, 6].min())
    check("randomization: per-slot xy jitter and spawn yaw vary (readback)",
          jit > 0.004 and yaw_spread > 20.0)
    gate_spread = float(arr[:, 7].max() - arr[:, 7].min())
    trol_spread = float(arr[:, 8].max() - arr[:, 8].min())
    check("randomization: the gate opening y0 and trolley carriage x0 vary (readback: "
          f"gate {gate_spread * 1000:.1f} mm, trolley {trol_spread * 1000:.1f} mm)",
          gate_spread > 0.004 and trol_spread > 0.002)

    # =========================== 6. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 7. seed strategy: place from above, close ==================
    # The seed's plan — open, lower the butter in from above over "the back", push
    # shut — maps here to dropping the butter over the well and sliding the gate
    # closed. The gallery is ROOFED: the butter lands on the roof, bridging the 26 mm
    # trolley slot, and never enters. Constructed settled: rejected.
    torch.manual_seed(41)
    env.reset()
    step(10)
    place(scene.butter, gx + 0.125, gy0, c.roof_top + bh + 0.004)
    write_gate(0.0)
    step(90)
    report("seed-strategy")
    s, ok = judge()
    bz = butter_rel()[2]
    check("seed strategy: butter dropped over the well lands ON THE ROOF "
          f"(z={bz:.3f}, bridging the slot), gate slid closed — never inside, no "
          "success, score <= 0.15",
          bz > 0.20 and not bool(scene.inside_gallery(scene.butter)[0])
          and bool(scene.gate_closed()[0]) and not ok and s <= 0.15)

    # =========================== 8-9. short-of-the-well near misses =========================
    torch.manual_seed(51)
    env.reset()
    step(10)
    place(scene.butter, gx - 0.150, gy0, 0.1235)  # just inside the mouth
    step(90)
    report("mouth-camp")
    s, ok = judge()
    check("mouth camp: butter just inside the mouth but NOT fully entered — approach "
          "credit only, no success, score <= 0.12",
          bool(scene.inside_gallery(scene.butter)[0]) and not bool(scene._entered[0])
          and not ok and s <= 0.12)

    place(scene.butter, gx + 0.045, gy0, 0.1235)  # half-way down the bore
    step(90)
    report("mid-channel")
    s, ok = judge()
    check("mid-channel: butter half-way down the bore — entered + depth credit but "
          "not in the well, no success, score <= 0.58",
          bool(scene._entered[0]) and not bool(scene.in_well()[0])
          and not ok and s <= 0.58)

    # =========================== 10. brick smuggled in ======================================
    # Everything else right — butter settled in the well, gate at its stop — but the
    # RED brick inside the bore: the brick-out clause rejects at the 0.85 cap.
    torch.manual_seed(61)
    env.reset()
    step(10)
    place(scene.butter, gx + 0.128, gy0, c.well_top + bh + 0.002)
    place(scene.brick, gx - 0.050, gy0, 0.1235)
    write_gate(0.0)
    step(120)
    report("brick-smuggled")
    s, ok = judge()
    check("brick smuggled: butter in the well AND gate closed but the RED brick "
          "inside the bore — no success, score <= 0.85 (color identification and "
          "the brick-out clause are load-bearing)",
          bool(scene.in_well()[0]) and bool(scene.gate_closed()[0])
          and not bool(scene.brick_out()[0]) and bool(scene._welled[0])
          and not ok and s <= 0.85)

    # =========================== 11. gate ajar ==============================================
    torch.manual_seed(71)
    env.reset()
    step(10)
    place(scene.butter, gx + 0.128, gy0, c.well_top + bh + 0.002)
    write_gate(0.020)  # 20 mm short of the closed stop (band is 12 mm)
    step(120)
    report("gate-ajar")
    s, ok = judge()
    check("gate ajar: butter in the well but the gate parked 20 mm short of its stop "
          "— the closed-band clause rejects, no success, score <= 0.80",
          bool(scene.in_well()[0]) and not bool(scene.gate_closed()[0])
          and not ok and s <= 0.80)

    # =========================== 12. latched credit survives regression =====================
    place(scene.butter, gx + 0.30, gy0 + 0.30, bh + 0.002)  # yank it back to the floor
    step(60)
    report("regressed")
    s_a, ok_a = judge()
    step(60)
    report("regressed2")
    s_b, ok_b = judge()
    check("latched credit: entered/depth/welled latches earned then the butter yanked "
          f"back to the floor — score holds ({s_a:.3f} -> {s_b:.3f}, >= 0.5), still "
          "no success",
          bool(scene._welled[0]) and abs(s_a - s_b) < 1e-3 and s_a >= 0.5
          and not ok_a and not ok_b)

    # =========================== 13. forced order: closed gate blocks the ram ===============
    # Gate closed FIRST, butter laid in the loading zone, then the trolley rammed by
    # a constant +x force: the blade advances and engages the butter, and the ram
    # train stalls at the closed gate plane with the butter OUTSIDE the gallery (the
    # plate covers the whole aperture — 3 mm under-gap vs a 45 mm stick, full width,
    # full height — so there is no path in at ANY force). The deposit-before-seal
    # order is forced by geometry, not rubric fiat.
    torch.manual_seed(81)
    env.reset()
    step(10)
    write_gate(0.0)
    # place at rest height (probe constructor — no bounce, clean blade clearance)
    place(scene.butter, gx + c.mouth_point[0], gy0, 0.1235)
    step(30)
    report("order-loaded")
    tx0 = float(scene.trolley_x()[0])
    fw = torch.zeros(n, 1, 3, device=device)
    fw[0, 0, 0] = 0.9  # enough to break the butter's deck friction, too weak to climb
    for _ in range(480):
        scene.trolley.set_external_force_and_torque(fw, zero_w, env_ids=all_ids,
                                                    is_global=True)
        step(1)
    scene.trolley.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids,
                                                is_global=True)
    step(90)
    report("order-violation")
    s, ok = judge()
    bx = butter_rel()[0]
    tx = float(scene.trolley_x()[0])
    check("forced order: the trolley DID ram forward and engage the butter "
          f"(tx {tx0:+.3f} -> {tx:+.3f}) but the ram train stalls at the CLOSED "
          f"gate plane with the butter outside (bx={bx:+.3f} <= -0.225, never "
          "inside, tx <= -0.27); the gate holds its stop — no success, score stays "
          "at the approach sliver (<= 0.15)",
          bx <= -0.225 and bx >= -0.30
          and not bool(scene.inside_gallery(scene.butter)[0])
          and not bool(scene._entered[0])
          and tx >= tx0 + 0.008 and tx <= -0.27
          and bool(scene.gate_closed()[0]) and not ok and s <= 0.15)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.butter.data.root_state_w).all()
           and torch.isfinite(scene.brick.data.root_state_w).all()
           and torch.isfinite(scene.trolley.data.root_state_w).all()
           and torch.isfinite(scene.gate.data.root_state_w).all()
           and torch.isfinite(scene.gallery.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.rammer_gallery")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
