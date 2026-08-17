"""Smoke / rubric-REJECTION battery for IdlerGearboxScene (sim_gen task
`approach_grasp_spoon_i396`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — idler dropped onto the peg, crank torqued
counter-clockwise so the gear train racks the cube out of the tunnel into the
pocket — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial)
outcome as a settled state and asserts the rubric REJECTS it; no probe in this
battery ever reaches success(), and a final audit check asserts exactly that.

  1-2.  settle/no-NaN    — reset layout settles finite: cube inside the roofed
                           tunnel, both wheels flat on the ground away from the
                           peg, rack at its q=0 stop, all still, score ~0;
  3-4.  randomization    — READBACK over 8 seeded resets: whole-machine yaw + xy
                           offset + crank start angle are real; idler/blank
                           ground-slot assignment flips (Bernoulli swap);
                           per-wheel jitter and cube tunnel depth are real;
  5.    null policy      — 240 idle steps -> score ~0, no success;
  6.    seed strategy    — the SEED's plan (grasp the payload and CARRY it to the
                           goal) realized as a magic carry: cube teleported into
                           the pocket and settled -> in_pocket reads true but the
                           transmission latch never fired -> success REJECTED,
                           score ~0 (the payload cannot be delivered by carrying);
  7.    crank w/o idler  — crank torqued 600 steps with the gear train BROKEN:
                           the crank demonstrably spins yet the rack never moves,
                           no transmission, score ~0;
  8.    wrong object     — the smooth BLANK physically dropped onto the peg (it
                           seats geometrically, verified) and the crank torqued:
                           the crank spins freely, the rack never moves — no
                           seating credit (idler-specific), no transmission,
                           score ~0 (tooth identification is load-bearing);
  9.    off-peg          — idler dropped on the well floor 60+ mm from the peg:
                           NOT seated, no credit;
  10.   wrong direction  — idler probe-seated, crank torqued CLOCKWISE: the train
                           locks the rack against its retracted stop — rack never
                           advances, no transmission, no success (only the
                           latched seat credit remains);
  11.   anti-cheat rack  — idler probe-seated, then the rack TELEPORTED to full
                           stroke in 9 jumps (a wrench/teleport cheat): per-step
                           clamp caps the credited advance far below trans_min ->
                           eject never latches, success REJECTED even though the
                           cube was physically shoved forward;
  12.   near-miss lip    — cube settled just short of the tunnel lip: not in the
                           pocket, no success;
  13.   beside-pocket    — cube settled on the ground OUTSIDE the pocket wall at
                           an in-range x: frame-local y band rejects, no success;
  14.   latched credit   — idler probe-seated -> score 0.25; idler then teleported
                           back to the ground -> latched credit survives (still
                           0.25), NOT success, non-success cap holds;
  15.   rejection audit  — success() was never True at ANY judged point;
  16.   final no-NaN     — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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
# RTX recipe: kit mis-decodes some driver versions and silently rejects RTX -> the
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
    env = ENVS.get("simgen.idler_gearbox_i396")().build(num_envs=args.num_envs,
                                                        device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply, quat_mul

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.85, -0.85, 0.70)) + o),
                                tuple(np.array((0.05, 0.00, 0.05)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

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

    def report(tag: str) -> None:
        il = scene._local(scene.idler)[0]
        bl = scene._local(scene.blank)[0]
        cl = scene._local(scene.cube)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | idler=({float(il[0]):+.3f},{float(il[1]):+.3f},"
              f"{float(il[2]):.3f}) blank=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) cube=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):.3f}) rack_q={float(scene.rack_q()[0]):+.4f} "
              f"adv={float(scene.geared_adv[0]):.4f} "
              f"seated={bool(scene.idler_seated()[0])} "
              f"seated_ever={bool(scene.seated_ever[0] > 0.5)} "
              f"eject_ever={bool(scene.eject_ever[0] > 0.5)} "
              f"in_pocket={bool(scene.cube_in_pocket()[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place_local(body, x: float, y: float, z: float, *, yaw: float = 0.0) -> None:
        """Teleport `body` to a frame-local point of the machine's CURRENT pose
        (probe constructor: builds hover/in-pocket/beside relations directly)."""
        fp = scene.frame.data.root_pos_w
        fq = scene.frame.data.root_quat_w
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = fp + quat_apply(fq, loc)
        if yaw:
            h = yaw / 2
            qy = torch.tensor([math.cos(h), 0.0, 0.0, math.sin(h)],
                              device=device).expand(n, 4)
            st[:, 3:7] = quat_mul(fq, qy)
        else:
            st[:, 3:7] = fq
        body.write_root_state_to_sim(st, all_ids)

    def teleport_rack(q: float) -> None:
        """Joint-consistent rack teleport to joint position q (the cheat probe)."""
        place_local(scene.rack, c.rack_x0 + q, c.rack_line_y, 0.030)

    def seat_idler_probe() -> bool:
        """Probe constructor: write the idler just above its seated pose and let it
        drop the last 1.5 mm; returns idler_seated() readback after latching."""
        place_local(scene.idler, 0.0, 0.0, 0.0315)
        step(30)
        return bool(scene.idler_seated()[0])

    def spin_crank(sign: float, steps: int, tau: float = 0.18,
                   w_des: float = 2.5) -> float:
        """Velocity-regulated torque about the crank's own body z axis (the hinge
        axis) for `steps` steps; returns the max |crank rate| observed."""
        w_max = 0.0
        for _ in range(steps):
            w = float(scene.crank.data.root_ang_vel_w[0, 2])
            w_max = max(w_max, abs(w))
            t = torch.zeros(n, 1, 3, device=device)
            t[:, 0, 2] = sign * tau if sign * w < w_des else 0.0
            scene.crank.set_external_force_and_torque(zero_w, t, env_ids=all_ids,
                                                      is_global=False)
            step(1)
        scene.crank.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids,
                                                  is_global=False)
        step(1)
        return w_max

    def finite_all() -> bool:
        ok = True
        for body in (scene.frame, scene.crank, scene.rack, scene.idler,
                     scene.blank, scene.cube):
            ok = ok and bool(torch.isfinite(body.data.root_state_w).all())
        return ok

    def frame_pose() -> tuple[float, float, float]:
        p = (scene.frame.data.root_pos_w - scene.env_origins)[0]
        q = scene.frame.data.root_quat_w[0]
        return (float(p[0]), float(p[1]),
                math.degrees(2.0 * math.atan2(float(q[3]), float(q[0]))))

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    il = scene._local(scene.idler)[0]
    bl = scene._local(scene.blank)[0]
    cl = scene._local(scene.cube)[0]
    still = (float(scene.cube.data.root_lin_vel_w[0].norm()) < c.settle_lin
             and float(scene.idler.data.root_lin_vel_w[0].norm()) < c.settle_lin
             and float(scene.blank.data.root_lin_vel_w[0].norm()) < c.settle_lin
             and float(scene.rack.data.root_lin_vel_w[0].norm()) < c.settle_lin)
    check("settle: states finite, cube inside the tunnel on its floor, both wheels "
          "flat on the ground far from the peg, rack at its q=0 stop, all still",
          finite_all()
          and c.cube_x_lo - 0.012 < float(cl[0]) < c.cube_x_hi + 0.012
          and 0.010 < float(cl[2]) < 0.050
          and float(il[2]) < 0.05 and float(bl[2]) < 0.05
          and float(il[0:2].norm()) > 0.10 and float(bl[0:2].norm()) > 0.10
          and float(scene.rack_q()[0]) < 0.008 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        fx, fy, fyaw = frame_pose()
        il = scene._local(scene.idler)[0]
        cl = scene._local(scene.cube)[0]
        crank_deg = math.degrees(float(scene.crank_angle()[0]))
        # idler on ground slot A (x=+0.10) vs slot B (x=-0.20), frame-local
        on_a = 1.0 if float(il[0]) > -0.05 else 0.0
        reads.append((fx, fy, fyaw, crank_deg, float(il[0]), float(il[1]), on_a,
                      float(cl[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (frame_x, frame_y, frame_yaw_deg, "
          f"crank_deg, idler_x, idler_y, idler_on_slot_a, cube_x):\n{arr}",
          flush=True)
    yaw_spread = float(arr[:, 2].max() - arr[:, 2].min())
    xy_spread = float((arr[:, 0:2].max(axis=0) - arr[:, 0:2].min(axis=0)).max())
    crank_spread = float(arr[:, 3].max() - arr[:, 3].min())
    check("randomization: whole-machine yaw spread (> 2 deg), xy offset spread "
          "(> 4 mm) and crank start-angle spread (> 10 deg) are real (readback)",
          yaw_spread > 2.0 and xy_spread > 0.004 and crank_spread > 10.0)
    flags = arr[:, 6]
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 4:6].max(axis=0)
                                  - grp[:, 4:6].min(axis=0)).max()))
    cube_spread = float(arr[:, 7].max() - arr[:, 7].min())
    check("randomization: idler/blank slot assignment flips across seeded resets, "
          "per-wheel jitter (> 4 mm) and cube tunnel depth (> 4 mm) are real "
          "(readback)", 0.0 < flags.mean() < 1.0 and jit > 0.004
          and cube_spread > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 6. seed strategy: carry the payload ========================
    # The seed's plan — grasp the payload and CARRY it to the goal — realized as the
    # strongest possible carry (a teleport into the pocket, settled). in_pocket reads
    # true, but the transmission latch (geared rack advance while the idler is
    # seated) never fired: the rubric REJECTS the delivery. The cube is sealed in a
    # roofed tunnel, so no real carry exists; this proves even a magic one scores ~0.
    torch.manual_seed(41)
    env.reset()
    step(30)
    place_local(scene.cube, 0.240, c.rack_line_y, 0.030)
    step(180)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy: cube MAGIC-CARRIED (teleported) into the pocket and "
          "settled — in_pocket true but the transmission latch never fired: "
          "success REJECTED, score ~0 (carrying the payload cannot win)",
          bool(scene.cube_in_pocket()[0]) and float(scene.geared_adv[0]) < 1e-4
          and s <= 0.02 and not ok)

    # =========================== 7. crank without the idler =================================
    torch.manual_seed(51)
    env.reset()
    step(30)
    w_max = spin_crank(+1.0, 600)
    report("no-idler-crank")
    s, ok = judge()
    check("crank without idler: 600 steps of crank torque with the train BROKEN — "
          f"the crank demonstrably spun (max rate {w_max:.2f} rad/s > 1.0) yet the "
          "rack never advanced (q < 8 mm), no transmission, score ~0, no success",
          w_max > 1.0 and float(scene.rack_q()[0]) < 0.008
          and float(scene.geared_adv[0]) < 1e-3 and s <= 0.02 and not ok)

    # =========================== 8. wrong object: the smooth blank ==========================
    # The BLANK physically dropped onto the peg exactly like a real install: it
    # seats geometrically (verified by readback) — and transmits NOTHING. Cranking
    # spins freely, the rack never moves; no seating credit (idler-specific).
    torch.manual_seed(61)
    env.reset()
    step(30)
    place_local(scene.blank, 0.002, 0.0, 0.092)
    step(300)
    bl = scene._local(scene.blank)[0]
    blank_seated_geom = (float(bl[0:2].norm()) < c.seat_xy_tol
                         and c.seat_z_lo < float(bl[2]) < c.seat_z_hi)
    w_max = spin_crank(+1.0, 600)
    report("blank-crank")
    s, ok = judge()
    check("wrong object: smooth BLANK dropped onto the peg seats geometrically "
          f"(readback: r={float(bl[0:2].norm()) * 1000:.1f} mm, z={float(bl[2]) * 1000:.1f} mm) "
          f"but cranking (max rate {w_max:.2f} rad/s) moves the rack NOTHING — no "
          "seating credit, no transmission, score ~0 (tooth identification is "
          "load-bearing)",
          blank_seated_geom and w_max > 1.0 and float(scene.rack_q()[0]) < 0.008
          and float(scene.seated_ever[0]) < 0.5
          and float(scene.geared_adv[0]) < 1e-3 and s <= 0.02 and not ok)

    # =========================== 9. off-peg: idler on the well floor ========================
    torch.manual_seed(71)
    env.reset()
    step(30)
    place_local(scene.idler, 0.040, -0.050, 0.050)
    step(240)
    report("off-peg")
    il = scene._local(scene.idler)[0]
    s, ok = judge()
    check("off-peg: idler dropped on the well floor 60+ mm from the peg — NOT "
          "seated (xy gate), no seating credit, score ~0, no success",
          float(il[0:2].norm()) > c.seat_xy_tol + 0.010
          and not bool(scene.idler_seated()[0])
          and float(scene.seated_ever[0]) < 0.5 and s <= 0.02 and not ok)

    # =========================== 10. wrong crank direction ==================================
    # Idler probe-seated (latching the 0.25 seat credit — probes construct partial
    # states), then the crank torqued CLOCKWISE: the completed train just presses
    # the rack against its retracted stop. No advance, no transmission, no success.
    torch.manual_seed(81)
    env.reset()
    step(30)
    seated = seat_idler_probe()
    w_max = spin_crank(-1.0, 600)
    report("wrong-direction")
    s, ok = judge()
    check("wrong direction: idler probe-seated, crank torqued CLOCKWISE — the "
          "train locks the rack against its q=0 stop (q < 10 mm), no transmission "
          "latch, score <= seat credit only (<= 0.27), no success",
          seated and float(scene.rack_q()[0]) < 0.010
          and float(scene.geared_adv[0]) < c.trans_min
          and s <= c.w_seat + 0.02 and not ok)

    # =========================== 11. anti-cheat: teleported rack ============================
    # The strongest transmission fake: idler SEATED (max credit opportunity) and the
    # rack teleported to full stroke in 9 jumps — a wrench/teleport cheat that
    # physically shoves the cube forward. Each jump credits at most one capped step
    # (2 mm) and the depenetration creep it excites peaks near ~31 mm — half of
    # trans_min (60 mm): the latch refuses, eject never arms, and success is
    # REJECTED regardless of where the cube ends up.
    torch.manual_seed(91)
    env.reset()
    step(30)
    seated = seat_idler_probe()
    for k in range(1, 10):
        teleport_rack(min(0.0143 * k, c.rack_stroke - 0.001))
        step(40)
    step(120)
    report("rack-teleport")
    cl = scene._local(scene.cube)[0]
    s, ok = judge()
    check("anti-cheat: rack TELEPORTED to full stroke in 9 jumps with the idler "
          f"seated — cube physically shoved to x={float(cl[0]):+.3f} but credited "
          f"advance {float(scene.geared_adv[0]) * 1000:.1f} mm stays far below "
          f"trans_min ({c.trans_min * 1000:.0f} mm): transmission refused, eject "
          "never latched, success REJECTED, score <= seat credit",
          seated and float(scene.rack_q()[0]) > 0.10
          and float(scene.geared_adv[0]) < c.trans_min
          and float(scene.eject_ever[0]) < 0.5
          and s <= c.w_seat + 0.02 and not ok)

    # =========================== 12. near-miss: cube short of the lip =======================
    torch.manual_seed(101)
    env.reset()
    step(30)
    place_local(scene.cube, 0.180, c.rack_line_y, 0.030)
    step(180)
    report("near-miss-lip")
    cl = scene._local(scene.cube)[0]
    s, ok = judge()
    check("near-miss: cube settled just short of the tunnel lip (x < pocket_x0) — "
          "NOT in the pocket, score ~0, no success",
          float(cl[0]) < c.pocket_x0 and not bool(scene.cube_in_pocket()[0])
          and s <= 0.02 and not ok)

    # =========================== 13. beside-pocket (frame math) =============================
    torch.manual_seed(111)
    env.reset()
    step(30)
    place_local(scene.cube, 0.240, c.rack_line_y + 0.105, 0.030)
    step(180)
    report("beside-pocket")
    cl = scene._local(scene.cube)[0]
    s, ok = judge()
    check("beside-pocket: cube settled on the ground OUTSIDE the pocket wall at an "
          "in-range x — frame-local y band rejects: NOT in pocket, no success",
          abs(float(cl[1]) - c.rack_line_y) > c.pocket_hw
          and not bool(scene.cube_in_pocket()[0]) and not ok)

    # =========================== 14. latched credit survives regression =====================
    torch.manual_seed(121)
    env.reset()
    step(30)
    seated = seat_idler_probe()
    report("probe-seated")
    s14a, ok = judge()
    got_credit = seated and abs(s14a - c.w_seat) < 0.02 and not ok
    place_local(scene.idler, 0.10, -0.22, 0.012)  # regression: back to the ground
    step(60)
    report("regressed")
    s14b, ok = judge()
    check("latched credit: probe-seated idler earns exactly the seat credit "
          f"(0.25, got {s14a:.3f}) with NO success; teleporting it back to the "
          f"ground leaves the latched score unchanged ({s14a:.3f} -> {s14b:.3f}), "
          "still no success (non-success cap holds)",
          got_credit and abs(s14b - s14a) < 1e-3
          and not bool(scene.idler_seated()[0]) and not ok)

    # =========================== 15-16. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this "
          "battery", not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.idler_gearbox_i396")
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc})", flush=True)
        os._exit(1)
