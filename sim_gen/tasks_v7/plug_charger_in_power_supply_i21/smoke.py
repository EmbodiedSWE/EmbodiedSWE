"""Smoke / rubric-REJECTION battery for KeyholeUnplugScene (sim_gen task
`plug_charger_in_power_supply_i21`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — slide-unlock and lift the plug out under a
floating-hand force controller, lay it in the tray — is the acceptance evidence that
the rubric ACCEPTS a correct outcome). Every teleport here is instrumentation that
CONSTRUCTS a wrong (or partial) outcome and asserts the rubric REJECTS it; no probe
in this battery ever reaches success(), and a final audit check asserts exactly that.

  1.  settle/no-NaN       — reset layout settles finite: both plugs verified at their
                            lock seats by strip-frame READBACK;
  2.  SEED strategy       — the seed task's GOAL state (a plug fully inserted and
                            seated in its socket) is exactly this task's RESET state:
                            score ~0, no success — the seed's whole plan earns nothing;
  3-4. randomization      — READBACK over 8 seeded resets: strip xy + yaw and tray
                            xy + yaw all move; the blue/black socket assignment takes
                            BOTH values;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6.  yank probe          — 2 simulated seconds of a 6 N straight-up pull on the
                            still-LOCKED plug (pure contact, no teleport): the wide
                            foot bears on the lid, the plug stays captive, extraction
                            never latches, and the free-standing 14.7 N strip is not
                            hoisted -> unlock-by-sliding is physically mandatory;
  7.  near-miss slide     — plug slid to the wide opening but NOT lifted out: slide
                            credit only (0.15 <= score <= 0.30), NOT success;
  8.  near-miss stow      — plug lying on the FLOOR just outside the tray wall: NOT
                            in the tray, NOT success, score <= 0.60;
  9.  rim percher         — plug balanced on the tray WALL TOP: tray-frame height
                            above the interior band -> NOT in the tray, NOT success
                            (transient judged probe);
  10. left on the strip   — plug standing UPRIGHT on the strip's lid: below the
                            extraction latch height (readback), NOT extracted, NOT
                            success, score ~0;
  11. wrong object        — the BLACK plug laid in the tray, blue untouched in its
                            socket: NOT success, score ~0 (identity matters);
  12. constraint: strip   — blue plug properly lying in the tray, but the strip
                            dragged 80 mm (its black plug re-seated relative to it):
                            NOT success, score <= 0.85;
  13. constraint: black   — blue plug properly lying in the tray, but the black plug
                            unseated onto the floor: NOT success, score <= 0.85;
  14. latched credit      — slide credit survives sliding the plug BACK to the lock
                            seat (credit never evaporates, success stays gone);
  15. monotonicity        — a deeper unlock slide latches strictly more credit;
  16. rejection audit     — success() was never True at ANY judged point;
  17. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.plug_charger_in_power_supply_i21.smoke --headless
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
    env = ENVS.get("simgen.keyhole_unplug")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.72, -0.72, 0.55)) + o),
                                tuple(np.array((0.02, -0.04, 0.03)) + o),
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

    def strip_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.strip.data.root_pos_w - scene.env_origins)[0]
        q = scene.strip.data.root_quat_w[0]
        return sp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def tray_pose() -> tuple[torch.Tensor, float]:
        tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
        q = scene.tray.data.root_quat_w[0]
        return tp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def report(tag: str) -> None:
        bl = scene._strip_local(scene.plug_blue.data.root_pos_w)[0]
        tl = scene._tray_local(scene.plug_blue.data.root_pos_w)[0]
        s, ok = judge()
        print(f"[smoke] {tag:16s} | blue_strip_local=({float(bl[0]):+.3f},"
              f"{float(bl[1]):+.3f},{float(bl[2]):.3f}) "
              f"blue_tray_local=({float(tl[0]):+.3f},{float(tl[1]):+.3f},{float(tl[2]):.3f}) "
              f"in_tray={bool(scene.blue_in_tray()[0])} "
              f"black_seated={bool(scene.black_seated()[0])} "
              f"strip_ok={bool(scene.strip_ok()[0])} "
              f"latches=({float(scene.slide_latch[0]):.2f},"
              f"{float(scene.extract_latch[0]):.0f},{float(scene.stow_latch[0]):.0f}) "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
        return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))

    def lying_quat(yaw: float) -> tuple[float, float, float, float]:
        """q = qz(yaw) * qy(90 deg): plug local +z -> horizontal."""
        c45 = math.cos(math.pi / 4)
        cy2, sy2 = math.cos(yaw / 2), math.sin(yaw / 2)
        return (cy2 * c45, -sy2 * c45, cy2 * c45, sy2 * c45)

    def write_pose(body, x: float, y: float, z: float,
                   quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Kinematic probe write (instrumentation, not a solution); no stepping."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def place_body(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0),
                   settle_steps: int = 30) -> None:
        """Probe write + REAL physics steps before judging (the zero-step trap)."""
        write_pose(body, x, y, z, quat)
        step(settle_steps)

    def place_in_channel(body, x0: float, ly: float, settle_steps: int = 20) -> None:
        """Place a plug inside a socket channel at strip-local (x0, ly), upright,
        yaw-aligned with the strip — the pose family reset itself uses."""
        sp, syaw = strip_pose()
        ca, sa = math.cos(syaw), math.sin(syaw)
        place_body(body,
                   float(sp[0]) + ca * x0 - sa * ly,
                   float(sp[1]) + sa * x0 + ca * ly,
                   c.seat_z + 0.0015, quat=yaw_quat(syaw), settle_steps=settle_steps)

    def stow_lying_in_tray(body, settle_steps: int = 120) -> None:
        """Lying hover just above the tray floor + gravity set-down (the same honest
        set-down solve.py performs)."""
        tp, tyaw = tray_pose()
        span_mid = (c.plug_len / 2) - c.foot_h / 2
        place_body(body,
                   float(tp[0]) - span_mid * math.cos(tyaw),
                   float(tp[1]) - span_mid * math.sin(tyaw),
                   c.tray_floor_t + c.head_w / 2 + 0.009,
                   quat=lying_quat(tyaw), settle_steps=settle_steps)

    # =========================== 1-2. settle / SEED strategy ================================
    env.reset(seed=11)
    step(60)
    report("reset-settled")
    bl = scene._strip_local(scene.plug_blue.data.root_pos_w)[0]
    kl = scene._strip_local(scene.plug_black.data.root_pos_w)[0]
    bx0 = float(scene.blue_x0[0])
    fin0 = bool(torch.isfinite(scene.strip.data.root_state_w).all()
                and torch.isfinite(scene.plug_blue.data.root_state_w).all()
                and torch.isfinite(scene.plug_black.data.root_state_w).all())
    check("settle: states finite; both plugs verified at their lock seats by "
          "strip-frame readback; everything settled",
          fin0 and bool(scene.settled()[0])
          and abs(float(bl[0]) - bx0) < 0.006 and abs(float(bl[1]) - c.y_lock) < 0.006
          and abs(float(bl[2]) - c.seat_z) < 0.006
          and abs(float(kl[0]) + bx0) < 0.006 and abs(float(kl[1]) - c.y_lock) < 0.006)
    s, ok = judge()
    check("SEED strategy: the seed task's goal state (plug inserted and seated in its "
          "socket) IS this task's reset state — score ~0 (<= 0.02), no success",
          s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        sp, syaw = strip_pose()
        tp, tyaw = tray_pose()
        reads.append((float(sp[0]), float(sp[1]), syaw, float(tp[0]), float(tp[1]), tyaw,
                      float(scene.blue_x0[0])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (strip_x, strip_y, strip_yaw, tray_x, "
          f"tray_y, tray_yaw, blue_x0):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: strip xy + yaw and tray xy + yaw all vary across 8 seeded "
          "resets (readback)",
          spread[0] > 0.005 and spread[1] > 0.005 and spread[2] > 0.3
          and spread[3] > 0.005 and spread[4] > 0.005 and spread[5] > 0.3)
    check("randomization: the blue/black socket assignment takes BOTH values across "
          "seeded resets (readback)",
          arr[:, 6].min() < 0.0 < arr[:, 6].max())

    # =========================== 5. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6. yank probe (contact, no teleport) =======================
    # 2 simulated seconds of a 6 N straight-up pull on the still-LOCKED blue plug —
    # 7.6x its own weight, but under the 14.7 N strip weight. The 24 mm foot bears on
    # the 20 mm slot rails: the plug must stay captive under the lid, extraction must
    # never latch, and the free-standing strip must not be hoisted or dragged.
    env.reset(seed=41)
    step(30)
    fup = torch.zeros(n, 1, 3, device=device)
    fup[:, 0, 2] = 6.0
    scene.plug_blue.set_external_force_and_torque(fup, zero_wrench, env_ids=all_ids,
                                                  is_global=True)
    step(240)
    scene.plug_blue.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(60)
    report("yank-locked")
    s, ok = judge()
    bl = scene._strip_local(scene.plug_blue.data.root_pos_w)[0]
    check("yank: 2 s of a 6 N straight-up pull on the LOCKED plug — foot still under "
          "the lid (readback), extraction never latched, strip not hoisted/dragged, "
          "score <= 0.05, no success",
          float(bl[2]) < c.cav_z1 + 0.002 and float(scene.extract_latch[0]) == 0.0
          and abs(float(bl[1]) - c.y_lock) < 0.012 and bool(scene.strip_ok()[0])
          and s <= 0.05 and not ok)

    # =========================== 7. near-miss: slid but not lifted ==========================
    env.reset(seed=51)
    step(10)
    place_in_channel(scene.plug_blue, float(scene.blue_x0[0]), c.y_free, settle_steps=40)
    report("slid-not-lifted")
    s, ok = judge()
    bl = scene._strip_local(scene.plug_blue.data.root_pos_w)[0]
    check("near-miss slide: plug at the wide opening but NOT lifted out — slide "
          "credit only (0.15 <= score <= 0.30), NOT extracted, NOT success",
          abs(float(bl[1]) - c.y_free) < 0.008 and float(bl[2]) < c.cav_z1
          and float(scene.extract_latch[0]) == 0.0 and 0.15 <= s <= 0.30 and not ok)

    # =========================== 8. near-miss: dropped beside the tray ======================
    env.reset(seed=61)
    step(10)
    tp, tyaw = tray_pose()
    out_r = c.tray_inner_half + c.tray_wall_t + 0.030
    place_body(scene.plug_blue,
               float(tp[0]) + out_r * math.cos(tyaw),
               float(tp[1]) + out_r * math.sin(tyaw),
               c.head_w / 2 + 0.002, quat=lying_quat(tyaw + math.pi / 2),
               settle_steps=90)
    report("beside-tray")
    s, ok = judge()
    tl = scene._tray_local(scene.plug_blue.data.root_pos_w)[0]
    check("near-miss stow: plug lying on the FLOOR just outside the tray wall "
          "(readback outside the interior) — NOT in the tray, NOT success, "
          "score <= 0.60",
          max(abs(float(tl[0])), abs(float(tl[1]))) > c.in_tray_xy
          and not bool(scene.blue_in_tray()[0]) and s <= 0.60 and not ok)

    # =========================== 9. rim percher (transient probe) ===========================
    env.reset(seed=71)
    step(10)
    tp, tyaw = tray_pose()
    rim_r = c.tray_inner_half + c.tray_wall_t / 2
    place_body(scene.plug_blue,
               float(tp[0]) + rim_r * math.cos(tyaw),
               float(tp[1]) + rim_r * math.sin(tyaw),
               c.tray_wall_h + c.head_w / 2 + 0.002, quat=lying_quat(tyaw + math.pi / 2),
               settle_steps=3)  # transient judged probe: balanced, not a settled outcome
    report("rim-percher")
    s, ok = judge()
    tl = scene._tray_local(scene.plug_blue.data.root_pos_w)[0]
    check("rim percher: plug balanced on the tray WALL TOP — tray-frame height above "
          "the interior band (readback) => NOT in the tray, NOT success",
          float(tl[2]) > c.in_tray_z_hi and not bool(scene.blue_in_tray()[0]) and not ok)

    # =========================== 10. left standing on the strip lid =========================
    env.reset(seed=81)
    step(10)
    sp, syaw = strip_pose()
    place_body(scene.plug_blue,
               float(sp[0]), float(sp[1]),  # solid lid between the two sockets
               c.lid_top + c.foot_h / 2 + 0.001, quat=yaw_quat(syaw), settle_steps=60)
    report("on-the-lid")
    s, ok = judge()
    bl = scene._strip_local(scene.plug_blue.data.root_pos_w)[0]
    check("left on the strip: plug standing UPRIGHT on the lid — below the extraction "
          "latch height (readback), NOT extracted, NOT success, score <= 0.05",
          float(bl[2]) < c.extract_z and float(scene.extract_latch[0]) == 0.0
          and s <= 0.05 and not ok)

    # =========================== 11. wrong object ===========================================
    env.reset(seed=91)
    step(10)
    stow_lying_in_tray(scene.plug_black)
    report("wrong-object")
    s, ok = judge()
    check("wrong object: the BLACK plug laid in the tray while the blue plug never "
          "left its socket — NOT success, score <= 0.02",
          not bool(scene.black_seated()[0]) and s <= 0.02 and not ok)

    # =========================== 12. constraint: strip disturbed ============================
    env.reset(seed=101)
    step(10)
    sp, syaw = strip_pose()
    ca, sa = math.cos(syaw), math.sin(syaw)
    write_pose(scene.strip, float(sp[0]) + 0.08, float(sp[1]), 0.001, quat=yaw_quat(syaw))
    bx0 = float(scene.blue_x0[0])
    write_pose(scene.plug_black,  # re-seat the black plug relative to the MOVED strip
               float(sp[0]) + 0.08 + ca * (-bx0) - sa * c.y_lock,
               float(sp[1]) + sa * (-bx0) + ca * c.y_lock,
               c.seat_z + 0.0015, quat=yaw_quat(syaw))
    stow_lying_in_tray(scene.plug_blue)
    report("strip-dragged")
    s, ok = judge()
    check("constraint: blue plug properly lying in the tray, black plug still seated, "
          "but the strip dragged 80 mm from its reset pose — NOT success, "
          "score <= 0.85",
          bool(scene.blue_in_tray()[0]) and bool(scene.black_seated()[0])
          and not bool(scene.strip_ok()[0]) and s <= 0.85 and not ok)

    # =========================== 13. constraint: black unseated =============================
    env.reset(seed=111)
    step(10)
    write_pose(scene.plug_black, -0.30, -0.30, c.head_w / 2 + 0.002, quat=lying_quat(0.7))
    stow_lying_in_tray(scene.plug_blue)
    report("black-unseated")
    s, ok = judge()
    check("constraint: blue plug properly lying in the tray, strip untouched, but the "
          "BLACK plug unseated onto the floor — NOT success, score <= 0.85",
          bool(scene.blue_in_tray()[0]) and bool(scene.strip_ok()[0])
          and not bool(scene.black_seated()[0]) and s <= 0.85 and not ok)

    # =========================== 14. latched credit survives regression =====================
    env.reset(seed=121)
    step(10)
    place_in_channel(scene.plug_blue, float(scene.blue_x0[0]), c.y_free, settle_steps=20)
    s_in, _ = judge()
    place_in_channel(scene.plug_blue, float(scene.blue_x0[0]), c.y_lock, settle_steps=20)
    report("slid-back")
    s_back, ok = judge()
    check("latched credit: sliding the plug BACK to the lock seat leaves the latched "
          f"slide score unchanged ({s_in:.3f} -> {s_back:.3f}), still no success",
          s_in >= 0.15 and abs(s_back - s_in) < 0.02 and not ok)

    # =========================== 15. slide monotonicity =====================================
    env.reset(seed=131)
    step(10)
    travel = c.y_free - c.y_lock
    place_in_channel(scene.plug_blue, float(scene.blue_x0[0]),
                     c.y_lock + 0.35 * travel, settle_steps=8)
    a_part = float(scene.slide_latch[0])
    place_in_channel(scene.plug_blue, float(scene.blue_x0[0]),
                     c.y_lock + 0.80 * travel, settle_steps=8)
    a_deep = float(scene.slide_latch[0])
    judge()
    check("monotonicity: a deeper unlock slide latches strictly more credit "
          f"({a_part:.3f} < {a_deep:.3f})", a_part + 0.10 < a_deep)

    # =========================== 16-17. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.strip.data.root_state_w).all()
           and torch.isfinite(scene.plug_blue.data.root_state_w).all()
           and torch.isfinite(scene.plug_black.data.root_state_w).all()
           and torch.isfinite(scene.tray.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.keyhole_unplug")
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
    main()
