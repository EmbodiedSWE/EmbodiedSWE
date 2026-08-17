"""Smoke / rubric-REJECTION battery for HitchTowScene (sim_gen task
`living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_
i391`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — mate the carts on the stop block, drop the pin
through both bores, tow the trailer onto the dock — is the acceptance evidence; it
passes on forge seeds). Every teleport here is instrumentation that CONSTRUCTS a
wrong (or partial) outcome and asserts the rubric REJECTS it — plus physical probes
that prove the stop-block mate and the shear-pin transmission are working mechanisms,
not props. No probe in this battery ever reaches success() at a judged point, and a
final audit check asserts exactly that.

   1. settle/no-NaN      — reset settles finite: carts in their spawn bands at ride
                           height, pin standing in its socket, settled;
   2. fresh reset        — score ~0, no success;
   3. randomization      — READBACK over 6 seeded resets: trailer park depth,
                           tractor start, pin yaw all vary;
   4. null policy        — 240 idle steps -> score ~0, no success;
   5. free-transport     — the SEED-STRATEGY end state: the trailer teleported
                           straight onto the dock pad (docked, settled — looks
                           delivered) is REJECTED: never mated, never coupled,
                           never towed -> score ~0;
   6. mate mechanism     — the solve's own -x servo genuinely drives the tractor
                           (displacement readback — vacuous-probe guard) and the
                           STOP BLOCK arrests it exactly at mate_dx (bores
                           coaxial); credit 0.25, NOT success;
   7. tow-without-pin    — from the mated state, pulling the tractor +x WITHOUT
                           the pin just drives the tractor away: the trailer stays
                           (readback), the odometer stays ~0 — the shear pin is the
                           ONLY transmission;
   8. honest hitch       — re-mate + the solve's own pin press: coupled, credit
                           0.60, still NOT success (not docked, not towed);
   9. pin-perch miss     — the pin standing ON the coupler plate BESIDE the bore
                           (placement near-miss) is not a coupling: score stays
                           0.25 in its episode;
  10. teleport-rig cheat — an honestly hitched rig teleported AS A UNIT onto the
                           dock (docked & coupled & settled all True) is REJECTED
                           by the tow odometer alone (per-step clamped credit ~0);
  11. stopped short      — an honest full run whose tow stops at x=0.15 < dock_lo:
                           coupled & towed & settled yet NOT success (dock clause);
  12. settle gate        — the same rig towed INTO the dock band judged WHILE
                           MOVING: docked & coupled & towed all True yet NOT
                           success (stillness must persist);
  13. arrive unhitched   — the pin yanked out during the final approach: the
                           trailer coasts onto the pad and rests there (docked,
                           towed, settled) but uncoupled -> REJECTED ("arrives and
                           stays hitched" clause);
  14. rejection audit    — success() was never True at ANY judged point;
  15. final no-NaN       — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.living_room_scene4_stack_the_left_bowl_on_the_
             right_bowl_and_place_them_in_the_tray_i391.smoke --headless
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

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older isaaclab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hitch_tow")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -1.10, 0.85)) + o),
                                tuple(np.array((0.00, 0.00, 0.10)) + o),
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

    def lx0() -> float:
        return float(scene.tractor_x()[0])

    def tx0() -> float:
        return float(scene.trailer_x()[0])

    def pin0() -> tuple[float, float, float]:
        p = scene.pin_loc()[0]
        return float(p[0]), float(p[1]), float(p[2])

    def pin_yaw0() -> float:
        q = scene.pin.data.root_quat_w[0]
        dx = 1.0 - 2.0 * (float(q[2]) ** 2 + float(q[3]) ** 2)
        dy = 2.0 * (float(q[1]) * float(q[2]) + float(q[0]) * float(q[3]))
        return math.atan2(dy, dx)

    def report(tag: str) -> None:
        s, ok = judge()
        px, py, pz = pin0()
        print(f"[smoke] {tag:16s} | tractor={lx0():+.4f} trailer={tx0():+.4f} "
              f"rel={float(scene.rel_dx()[0]):+.4f} pin=({px:+.3f},{py:+.3f},{pz:+.3f}) "
              f"mated={bool(scene.mated_now()[0])} coupled={bool(scene.coupled_now()[0])} "
              f"docked={bool(scene.docked_now()[0])} tow={float(scene.tow_travel[0]):.3f} "
              f"settled={bool(scene.settled()[0])} | m/p={float(scene.mate_latch[0]):.0f}/"
              f"{float(scene.pin_latch[0]):.0f} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # ----- teleport instrumentation (joint-consistent writes + REAL steps) --------------------
    def write_cart(body, x: float) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = float(x)
        st[:, 2] = c.ride_z
        st[:, 0:3] += scene.env_origins
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)

    def write_pin(x: float, y: float, z: float, settle_steps: int = 0) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = float(x)
        st[:, 1] = float(y)
        st[:, 2] = float(z)
        st[:, 0:3] += scene.env_origins
        st[:, 3] = 1.0
        scene.pin.write_root_state_to_sim(st, all_ids)
        if settle_steps:
            step(settle_steps)

    # ----- the solve's own servos (mechanism probes) ------------------------------------------
    def tractor_force(fx: float | torch.Tensor) -> None:
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = fx
        scene.tractor.set_external_force_and_torque(f, zero_w, env_ids=all_ids)

    def tractor_off() -> None:
        scene.tractor.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def pin_off() -> None:
        scene.pin.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def mate_servo(max_steps: int = 1800) -> int:
        streak, i = 0, 0
        for i in range(max_steps):
            x = scene.tractor_x()
            vx = scene.tractor.data.root_lin_vel_w[:, 0]
            x_tgt = scene.trailer_x() + c.mate_dx - 0.003
            v_des = (3.0 * (x_tgt - x)).clamp(-0.06, 0.06)
            tractor_force((12.0 * (v_des - vx)).clamp(-3.0, 3.0))
            env.step(no_action)
            streak = streak + 1 if bool(scene.mated_now()[0]) else 0
            if streak >= 40:
                break
        return i + 1

    def shove_to_stop(hold: bool = True) -> None:
        """Press the mated pair onto the trailer's hard stop so the bores are
        stationary; optionally keep the light hold press on."""
        streak = 0
        for _ in range(1200):
            vx = scene.tractor.data.root_lin_vel_w[:, 0]
            tractor_force((12.0 * (-0.04 - vx)).clamp(-3.0, 3.0))
            env.step(no_action)
            at_stop = (tx0() <= c.t_lo + 0.004
                       and abs(float(scene.trailer.data.root_lin_vel_w[0, 0])) < 0.005)
            streak = streak + 1 if at_stop else 0
            if streak >= 60:
                break
        if hold:
            tractor_force(-0.4)
        else:
            tractor_off()
        step(30)

    def hitch_pin() -> bool:
        """The solve's own hitch: transport-teleport the pin above the (stationary)
        bores, then contact press-down until coupled. Leaves ALL wrenches off."""
        hover_z = c.plate_z_hi + c.shank_len / 2 + 0.015
        write_pin(float(scene.tractor_bore_x()[0]), 0.0, hover_z)
        seated = False
        for _attempt in range(2):
            streak, stall_z, stall_i = 0, None, 0
            for k in range(900):
                tractor_force(-0.4)
                p = scene.pin_loc()
                v = scene.pin.data.root_lin_vel_w
                f_world = torch.zeros(n, 3, device=device)
                f_world[:, 0] = (4.0 * (scene.tractor_bore_x() - p[:, 0])
                                 - 0.4 * v[:, 0]).clamp(-0.5, 0.5)
                f_world[:, 1] = (4.0 * (0.0 - p[:, 1]) - 0.4 * v[:, 1]).clamp(-0.5, 0.5)
                f_world[:, 2] = (c.pin_mass * G
                                 + 1.0 * (-0.06 - v[:, 2])).clamp(-0.30, 0.80)
                f_body = quat_apply_inverse(scene.pin.data.root_quat_w, f_world)
                scene.pin.set_external_force_and_torque(f_body.unsqueeze(1), zero_w,
                                                        env_ids=all_ids)
                env.step(no_action)
                z = float(scene.pin_loc()[0][2])
                streak = streak + 1 if bool(scene.coupled_now()[0]) else 0
                if streak >= 30:
                    seated = True
                    break
                if stall_z is None or z < stall_z - 0.001:
                    stall_z, stall_i = z, k
                elif k - stall_i >= 200:
                    break
            if seated:
                break
            write_pin(float(scene.tractor_bore_x()[0]), 0.0, hover_z)
        pin_off()
        step(60)
        tractor_off()
        step(60)
        return seated

    def tow_servo(stop_x: float, brake: bool, settle_steps: int) -> int:
        """+x velocity servo (ramped) until trailer_x >= stop_x; optional brake."""
        i = 0
        for i in range(3000):
            x = scene.tractor_x()
            vx = scene.tractor.data.root_lin_vel_w[:, 0]
            vcap = min(0.10, 0.02 + 0.08 * (i / 200.0))
            v_des = (3.0 * ((stop_x + c.mate_dx + 0.02) - x)).clamp(0.0, vcap)
            tractor_force((12.0 * (v_des - vx)).clamp(-6.0, 6.0))
            env.step(no_action)
            if tx0() >= stop_x - 0.005:
                break
        if brake:
            for _ in range(150):
                vx = scene.tractor.data.root_lin_vel_w[:, 0]
                tractor_force((12.0 * (0.0 - vx)).clamp(-6.0, 6.0))
                env.step(no_action)
            tractor_off()
        if settle_steps:
            step(settle_steps)
        return i + 1

    bodies = (scene.bed, scene.yard, scene.tractor, scene.trailer, scene.pin)

    # =========================== 1-2. settle / no-NaN =======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    px, py, pz = pin0()
    check("settle: all states finite, trailer in its spawn band "
          f"(x={tx0():+.4f} in [{c.t_spawn_lo:+.2f},{c.t_spawn_hi:+.2f}]), tractor in "
          f"its band (x={lx0():+.4f} in [{c.l_spawn_lo:+.2f},{c.l_spawn_hi:+.2f}]), pin "
          f"standing in the socket (({px:+.3f},{py:+.3f},{pz:+.3f}) ~ "
          f"({c.stand_x:+.2f},{c.stand_y:+.2f},{c.pin_socket_z:.3f})), settled",
          fin and c.t_spawn_lo - 0.006 <= tx0() <= c.t_spawn_hi + 0.006
          and c.l_spawn_lo - 0.006 <= lx0() <= c.l_spawn_hi + 0.006
          and abs(px - c.stand_x) <= 0.006 and abs(py - c.stand_y) <= 0.006
          and abs(pz - c.pin_socket_z) <= 0.006 and bool(scene.settled()[0]))
    s, ok = judge()
    check("fresh reset: score ~0, no success", s <= 0.02 and not ok)

    # =========================== 3. randomization is real ===================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(20)
        reads.append((tx0(), lx0(), pin_yaw0()))
        print(f"[smoke] seed {sd}: trailer={tx0():+.4f} tractor={lx0():+.4f} "
              f"pin_yaw={pin_yaw0():+.3f}", flush=True)
    t_spread = max(r[0] for r in reads) - min(r[0] for r in reads)
    l_spread = max(r[1] for r in reads) - min(r[1] for r in reads)
    y_spread = max(r[2] for r in reads) - min(r[2] for r in reads)
    check("randomization: trailer park depth varies (readback spread "
          f"{t_spread * 1000:.1f} mm), tractor start varies ({l_spread * 1000:.1f} mm), "
          f"pin yaw varies ({y_spread:.2f} rad)",
          t_spread > 0.012 and l_spread > 0.03 and y_spread > 0.8)

    # =========================== 4. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps",
          s <= 0.02 and not ok)

    # =========================== 5. free-transport cheat (the seed strategy) ================
    # The seed task's strategy is "carry the object to the goal region". Teleport the
    # trailer straight onto the dock (tractor moved clear first, same write batch —
    # no intermediate mated pose is ever visible to post_step).
    env.reset(seed=41)
    step(60)
    write_cart(scene.tractor, 0.62)
    write_cart(scene.trailer, 0.30)
    step(180)
    report("free-transport")
    s, ok = judge()
    check("free-transport cheat: the trailer teleported straight onto the dock pad "
          f"(docked={bool(scene.docked_now()[0])}, settled={bool(scene.settled()[0])}) "
          "is REJECTED — never mated, never coupled, tow odometer "
          f"{float(scene.tow_travel[0]):.3f} ~ 0 -> score ~0, NOT success",
          bool(scene.docked_now()[0]) and bool(scene.settled()[0])
          and float(scene.tow_travel[0]) <= 0.02 and s <= 0.02 and not ok)

    # =========================== 6. mate mechanism ==========================================
    env.reset(seed=51)
    step(60)
    l_start = lx0()
    used = mate_servo()
    tractor_off()
    step(60)
    report("mate")
    s, ok = judge()
    check("mate mechanism: the -x servo genuinely drove the tractor "
          f"({l_start:+.3f} -> {lx0():+.3f}, {used} steps — vacuous-probe guard) and "
          f"the stop block arrested it at rel={float(scene.rel_dx()[0]):+.4f} ~ "
          f"mate_dx {c.mate_dx:.3f} (bores coaxial); credit 0.25, NOT success",
          l_start - lx0() >= 0.05 and bool(scene.mated_now()[0])
          and float(scene.mate_latch[0]) > 0.5 and 0.23 <= s <= 0.27 and not ok)

    # =========================== 7. tow-without-pin =========================================
    shove_to_stop(hold=False)
    t_before, l_before = tx0(), lx0()
    for i in range(400):
        vx = scene.tractor.data.root_lin_vel_w[:, 0]
        vcap = min(0.10, 0.02 + 0.08 * (i / 200.0))
        v_des = (3.0 * ((l_before + 0.30) - scene.tractor_x())).clamp(0.0, vcap)
        tractor_force((12.0 * (v_des - vx)).clamp(-6.0, 6.0))
        env.step(no_action)
    tractor_off()
    step(90)
    report("tow-no-pin")
    s, ok = judge()
    check("tow-without-pin: pulling the tractor +x with NO pin just drives it away "
          f"(tractor {l_before:+.3f} -> {lx0():+.3f}) while the trailer stays "
          f"({t_before:+.3f} -> {tx0():+.3f}, moved {abs(tx0() - t_before) * 1000:.1f} mm) "
          f"and the odometer stays ~0 ({float(scene.tow_travel[0]):.3f}) — the shear "
          "pin is the ONLY transmission",
          lx0() - l_before >= 0.15 and abs(tx0() - t_before) <= 0.02
          and float(scene.tow_travel[0]) <= 0.005 and 0.23 <= s <= 0.27 and not ok)

    # =========================== 8. honest hitch ============================================
    mate_servo()
    shove_to_stop(hold=True)
    seated = hitch_pin()
    report("hitched")
    s, ok = judge()
    check("honest hitch: re-mated + the solve's own pin press seats the pin through "
          f"both bores (seated={seated}, coupled={bool(scene.coupled_now()[0])}, "
          f"pin_z={pin0()[2]:+.4f} ~ seat {c.pin_seat_z:.4f}); credit 0.60, still NOT "
          "success (not docked, not towed)",
          seated and bool(scene.coupled_now()[0]) and float(scene.pin_latch[0]) > 0.5
          and 0.58 <= s <= 0.62 and not ok)

    # =========================== 9. pin-perch near-miss =====================================
    env.reset(seed=61)
    step(60)
    mate_servo()
    shove_to_stop(hold=False)
    bore_x = float(scene.tractor_bore_x()[0])
    write_pin(bore_x + 0.030, 0.0, c.plate_z_hi + c.shank_len / 2 + 0.001,
              settle_steps=240)
    report("pin-perch")
    s, ok = judge()
    px, py, pz = pin0()
    check("pin-perch near-miss: the pin resting ON the coupler plate BESIDE the bore "
          f"(({px:+.3f},{py:+.3f},{pz:+.3f}), bore at ({bore_x:+.3f}, +0.000, seat "
          f"{c.pin_seat_z:.3f})) never couples: couple latch never fires, score stays "
          "0.25, NOT success",
          not bool(scene.coupled_now()[0]) and float(scene.pin_latch[0]) < 0.5
          and 0.23 <= s <= 0.27 and not ok)

    # =========================== 10. teleport-rig cheat =====================================
    env.reset(seed=71)
    step(60)
    mate_servo()
    shove_to_stop(hold=True)
    seated = hitch_pin()
    assert seated, "probe rig failed to hitch (mechanism, not rubric)"
    # Teleport the WHOLE hitched linkage as a unit (one write batch, then step).
    rel = float(scene.rel_dx()[0])
    pin_dx = pin0()[0] - tx0()
    pin_z = pin0()[2]
    write_cart(scene.trailer, 0.26)
    write_cart(scene.tractor, 0.26 + rel)
    write_pin(0.26 + pin_dx, pin0()[1], pin_z)
    step(180)
    report("rig-teleport")
    s, ok = judge()
    check("teleport-rig cheat: the honestly hitched rig teleported AS A UNIT onto "
          f"the dock (docked={bool(scene.docked_now()[0])}, coupled="
          f"{bool(scene.coupled_now()[0])}, settled={bool(scene.settled()[0])} — "
          "looks perfect) is REJECTED by the tow odometer alone "
          f"(tow={float(scene.tow_travel[0]):.3f} < dist_min {c.dist_min:.2f})",
          bool(scene.docked_now()[0]) and bool(scene.coupled_now()[0])
          and bool(scene.settled()[0]) and float(scene.tow_travel[0]) <= 0.02
          and not bool(scene.towed()[0]) and s <= 0.62 and not ok)

    # =========================== 11. stopped short ==========================================
    env.reset(seed=81)
    step(60)
    mate_servo()
    shove_to_stop(hold=True)
    seated = hitch_pin()
    assert seated, "probe rig failed to hitch (mechanism, not rubric)"
    tow_servo(0.15, brake=True, settle_steps=240)
    report("stopped-short")
    s, ok = judge()
    check("stopped short: an honest tow halted at trailer "
          f"x={tx0():+.3f} < dock_lo {c.dock_lo:.2f}: coupled="
          f"{bool(scene.coupled_now()[0])}, towed={bool(scene.towed()[0])} "
          f"(tow={float(scene.tow_travel[0]):.3f}), settled={bool(scene.settled()[0])} "
          "yet NOT success — the dock clause alone rejects it",
          tx0() < c.dock_lo and bool(scene.coupled_now()[0]) and bool(scene.towed()[0])
          and bool(scene.settled()[0]) and not ok and 0.58 <= s <= 0.62)

    # =========================== 12. settle gate ============================================
    tow_servo(0.215, brake=False, settle_steps=0)  # judge IMMEDIATELY, still moving
    moving = float(scene.tractor.data.root_lin_vel_w[0, 0])
    docked_mid = bool(scene.docked_now()[0])
    coupled_mid = bool(scene.coupled_now()[0])
    towed_mid = bool(scene.towed()[0])
    settled_mid = bool(scene.settled()[0])
    s, ok = judge()
    report("settle-gate")
    check("settle gate: the rig towed INTO the dock band judged WHILE MOVING "
          f"(vx={moving:+.3f}, docked={docked_mid}, coupled={coupled_mid}, "
          f"towed={towed_mid}, settled={settled_mid}) is NOT success — stillness "
          "must persist",
          docked_mid and coupled_mid and towed_mid and not settled_mid and not ok)

    # =========================== 13. arrive unhitched (pin yanked) ==========================
    # Yank the pin OUT while the rig is still rolling: the trailer coasts onto the
    # pad and rests there — docked, towed, settled, but UNCOUPLED.
    write_pin(0.74, 0.0, c.bed_top + c.shank_len / 2 + 0.001)
    for _ in range(150):  # brake the tractor; the trailer coasts free
        vx = scene.tractor.data.root_lin_vel_w[:, 0]
        tractor_force((12.0 * (0.0 - vx)).clamp(-6.0, 6.0))
        env.step(no_action)
    tractor_off()
    step(300)
    report("pin-yanked")
    s, ok = judge()
    check("arrive unhitched: the pin yanked during the final approach — the trailer "
          f"rests on the pad (x={tx0():+.3f} >= {c.dock_lo:.2f}, towed="
          f"{bool(scene.towed()[0])}, settled={bool(scene.settled()[0])}) but "
          "uncoupled -> REJECTED by the arrives-and-stays-hitched clause",
          bool(scene.docked_now()[0]) and bool(scene.towed()[0])
          and bool(scene.settled()[0]) and not bool(scene.coupled_now()[0])
          and not ok and 0.58 <= s <= 0.62)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies)
    check("final: all task-object states finite (no NaN)", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.hitch_tow")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(okc for _nm, okc in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)
