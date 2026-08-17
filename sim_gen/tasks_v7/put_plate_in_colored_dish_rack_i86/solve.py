"""Physical solution for CarouselDishRackScene (sim_gen task
`put_plate_in_colored_dish_rack_i86`) — the task's legitimacy certificate.

This solve uses NO teleports at all: the plate starts on the feed shelf and every
phase is executed with applied forces/torques through contact dynamics.

PLAN (read from scene.describe(), which names the blue bay, the single gate, the
compass rose and the forced order):
  P1 ALIGN — drive the carousel with a z-torque velocity servo (the "push the blue
     pointer knob" act) until the BLUE bay's wedge center faces the gate. The initial
     offset is random-signed 55..180 deg, so direction and magnitude are read live
     from `bay_offset()`.
  P2 INSERT — shuffleboard push: a horizontal CoM force slides the plate FLAT along
     the shelf, through the gate, onto the blue bay floor (velocity-regulated toward
     the wedge center, lateral PD centering on the gate line, never lifted; the
     carousel is simultaneously held aligned by a gentle torque servo). The stop
     radius (0.100 m) is short of hub contact (0.085 m rest), so the push never
     presses the mechanism.
  P3 STOW — the same torque servo rotates the LOADED carousel to +90 deg; the plate
     is carried by platform friction (real transported contact — centripetal demand
     at 0.8 rad/s, r 0.1 is ~0.06 m/s^2, far under the mu*g ~5 m/s^2 budget).
  P4 hands-off persistence >= 3.3 simulated seconds.

Pod force-frame quirk: some pods rotate an applied wrench by the body's rotation
since reset (applied = R_now * R_ref^T * arg). The carousel torque is immune (pure
z-torque on a yaw-only body is invariant), but the plate push force is horizontal,
so the mode is PROBED from actual progress (moving AWAY toggles the encoding) —
never assumed.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds at the end.

Run (forge): python -u -m simgen_tasks.put_plate_in_colored_dish_rack_i86.solve \
    --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carousel_dish_rack")().build(num_envs=args.num_envs,
                                                        device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def off_deg() -> float:
        return math.degrees(float(scene.bay_offset()[0]))

    def car_w() -> float:
        return float(scene.carousel.data.root_ang_vel_w[0, 2])

    def clear_wrenches() -> None:
        scene.carousel.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids,
                                                     is_global=True)
        scene.plate.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids,
                                                  is_global=True)

    def report(tag: str) -> None:
        loc = scene.plate_local()[0]
        print(f"[solve] {tag:12s} | offset={off_deg():+7.2f}deg w={car_w():+.3f} | "
              f"plate_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):+.3f}) r={float(loc[:2].norm()):.3f} | "
              f"entered={bool(scene.entered()[0])} aligned={bool(scene.aligned()[0])} "
              f"in_blue={bool(scene.in_blue_bay()[0])} stowed={bool(scene.stowed()[0])} "
              f"settled={bool(scene.settled()[0])} | "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(max_steps: int = 720) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled()[0]):
                break

    # ----- carousel z-torque velocity servo (the "push a pointer knob" act) ----------------
    def spin_to(target_deg: float, *, tol_deg: float, w_max: float, tag: str,
                max_steps: int = 2400) -> bool:
        """Servo `bay_offset` to `target_deg` (shortest path). Pure z-torque on a
        yaw-only body — frame-drag-invariant. Escalates the GAIN on stall (a P
        velocity servo stalls at tau = k*w_des, not at the cap)."""
        k_w, k_t, cap = 3.0, 0.6, 0.5
        target = math.radians(target_deg)
        win_i, win_err = 0, abs(_wrap(target - math.radians(off_deg())))
        for i in range(max_steps):
            err = _wrap(target - math.radians(off_deg()))
            w = car_w()
            if abs(err) <= math.radians(tol_deg) and abs(w) < 0.08:
                clear_wrenches()
                print(f"[solve] {tag}: reached offset={off_deg():+.2f} deg in "
                      f"{i} steps (w={w:+.3f})", flush=True)
                return True
            w_des = max(-w_max, min(w_max, k_w * err))
            tau = max(-cap, min(cap, k_t * (w_des - w)))
            t = torch.zeros(n, 1, 3, device=device)
            t[0, 0, 2] = tau
            scene.carousel.set_external_force_and_torque(zero_w, t, env_ids=all_ids,
                                                         is_global=True)
            env.step(no_action)
            if i - win_i >= 240:  # stall probe: escalate the gain, not just the cap
                if abs(err) > win_err - math.radians(2.0):
                    k_t = min(k_t * 1.6, 3.0)
                    cap = min(cap + 0.3, 1.5)
                    print(f"[solve] {tag}: stalled at {off_deg():+.2f} deg; "
                          f"k_t -> {k_t:.2f}, cap -> {cap:.2f}", flush=True)
                win_i, win_err = i, abs(err)
        clear_wrenches()
        print(f"[solve] {tag}: TIMED OUT at offset={off_deg():+.2f} deg", flush=True)
        return False

    def hold_aligned_tau() -> torch.Tensor:
        """One gentle hold-at-zero torque sample (used inside the push loop)."""
        err = _wrap(-math.radians(off_deg()))
        w_des = max(-0.4, min(0.4, 3.0 * err))
        tau = max(-0.4, min(0.4, 0.6 * (w_des - car_w())))
        t = torch.zeros(n, 1, 3, device=device)
        t[0, 0, 2] = tau
        return t

    # ----- the shuffleboard push ------------------------------------------------------------
    def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                     f_world: torch.Tensor) -> torch.Tensor:
        """mode 0: pass the world force through; mode 1: pre-encode with
        R_ref * R_now^T (pods that rotate the wrench by rotation-since-reset)."""
        if mode == 0:
            return f_world
        return quat_apply(quat_mul(q_ref, quat_inv(q_now)), f_world)

    def push_plate(tag: str, max_steps: int = 3600) -> bool:
        """Velocity-regulated horizontal CoM push: slide the plate flat along the
        shelf line (env-frame y=0), through the gate, until its carousel-frame
        planar radius <= stop_r (short of hub contact at 0.085 — the push never
        presses the mechanism). The carousel is held aligned throughout."""
        nonlocal mode
        stop_r = 0.100
        v_des, kp_v, f_cap = -0.10, 10.0, 3.0  # x-velocity servo (N per m/s)
        win_i = 0
        win_r = float(scene.plate_local()[0, :2].norm())
        for i in range(max_steps):
            loc = scene.plate_local()[0]
            r = float(loc[:2].norm())
            if r <= stop_r:
                clear_wrenches()
                print(f"[solve] {tag}: plate at r={r:.3f} "
                      f"(z_loc={float(loc[2]):+.3f}) after {i} steps", flush=True)
                return True
            p = (scene.plate.data.root_pos_w - scene.env_origins)[0]
            if float(p[2]) > c.h0 + 0.05:
                clear_wrenches()
                print(f"[solve] {tag}: plate CLIMBED to z={float(p[2]):.3f} — abort",
                      flush=True)
                return False
            v = scene.plate.data.root_lin_vel_w[0]
            f_x = max(-f_cap, min(0.5, kp_v * (v_des - float(v[0]))))
            f_y = max(-1.5, min(1.5, -8.0 * float(p[1]) - 3.0 * float(v[1])))
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, 0], f_world[0, 1] = f_x, f_y
            f_arg = encode_force(mode, q_ref_plate, scene.plate.data.root_quat_w,
                                 f_world)
            scene.plate.set_external_force_and_torque(f_arg.view(n, 1, 3), zero_w,
                                                      env_ids=all_ids, is_global=True)
            scene.carousel.set_external_force_and_torque(zero_w, hold_aligned_tau(),
                                                         env_ids=all_ids,
                                                         is_global=True)
            env.step(no_action)
            if i - win_i >= 45:  # progress probe
                if r > win_r + 0.008:
                    mode = 1 - mode
                    print(f"[solve] {tag}: moving away (r {win_r:.3f} -> {r:.3f}); "
                          f"force-frame mode -> {mode}", flush=True)
                elif r > win_r - 0.004:
                    kp_v = min(kp_v + 5.0, 28.0)  # gain, not just cap
                    print(f"[solve] {tag}: stalled at r={r:.3f}; kp_v -> {kp_v:.0f}",
                          flush=True)
                win_i, win_r = i, r
        clear_wrenches()
        print(f"[solve] {tag}: TIMED OUT at r="
              f"{float(scene.plate_local()[0, :2].norm()):.3f}", flush=True)
        return False

    # ---------------- phase 0: reset, settle, plan readback --------------------------------
    step(90)
    off0 = math.degrees(float(scene.offset0[0]))
    p0 = (scene.plate.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] readback (seed {args.seed}): initial blue-bay offset "
          f"{off0:+.2f} deg (live {off_deg():+.2f}), plate at "
          f"({float(p0[0]):+.3f},{float(p0[1]):+.3f},{float(p0[2]):+.3f})", flush=True)
    report("reset")
    assert abs(off_deg()) >= 40.0, "episode must start misaligned"
    assert not bool(scene.aligned()[0]), "fresh reset must not be aligned"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    q_ref_plate = scene.plate.data.root_quat_w.clone()
    mode = 0  # probed from actual progress during the push
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, f"score must start ~0, got {s0}"

    # ---------------- phase 1: ALIGN the blue bay with the gate ----------------------------
    ok = spin_to(0.0, tol_deg=6.0, w_max=1.5, tag="P1-align")
    settle()
    report("P1-align")
    assert ok and bool(scene.aligned()[0]), "blue bay must be aligned after P1"
    s1 = print_score("P1 blue bay aligned with the gate (torque servo)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.145, f"P1 score {s1} (expect aligned=0.15)"

    # ---------------- phase 2: INSERT the plate flat through the gate ----------------------
    ok = push_plate("P2-push")
    settle()
    # seat latch: `seat_steps` consecutive slow in-bay steps (no fly-through credit)
    for _ in range(12):
        if float(scene.inbay_latch[0]) > 0.5:
            break
        step(30)
    report("P2-push")
    assert ok and bool(scene.in_blue_bay()[0]), "plate must rest in the BLUE bay"
    assert float(scene.inbay_latch[0]) > 0.5, "seat latch must have fired"
    s2 = print_score("P2 plate seated on the blue bay floor (contact)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.545, f"P2 score {s2} (expect 0.55)"

    # ---------------- phase 3: STOW the loaded bay away from the gate ----------------------
    stow_target = 90.0 if off0 > 0 else -90.0  # keep turning the way it came
    ok = spin_to(stow_target, tol_deg=6.0, w_max=0.8, tag="P3-stow")
    report("P3-spin")
    assert ok, "stow rotation must complete"
    # Ring-down: wait for success() to hold CONTINUOUSLY for 1 s.
    consec = 0
    for _ in range(2400):
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P3-stow")
    s3 = print_score("P3 loaded blue bay stowed behind the wall")
    assert s3 >= s2 - 1e-6, "score decreased across P3"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after stow + ring-down)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                print(f"[solve] persist flicker @step {i}: offset={off_deg():+.2f} "
                      f"w={car_w():+.4f} in_blue={bool(scene.in_blue_bay()[0])} "
                      f"stowed={bool(scene.stowed()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
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
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
