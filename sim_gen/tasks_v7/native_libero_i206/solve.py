"""Teleport solution for TiltMazeScene (sim_gen task `native_libero_i206`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): NONE after reset. The scene's own reset places every body;
   the solver never writes a root state. The ball is captive under the cage roof and
   is never touched directly — every millimetre it travels is rolling contact.
2. STROKE 1 (applied torque + contact): a PD servo with gravity/keel feedforward
   applies a hinge-axis torque to the TRAY body (body-frame, along the body-y hinge
   axis — invariant under the pod's wrench-frame drag), the exact wrench of a finger
   pressing the BLUE paddle down (|tau| <= tau_cap ~ a 2 N fingertip press at the
   paddle arm). The tray tilts ~8 deg blue-end-down; GRAVITY rolls the ball east up
   the north lane, the diagonal BAFFLE deflects it around the divider tip, and it
   parks in the bay's south corner. The controller only watches and holds the angle.
3. STROKE 2 (applied torque + contact): the servo reverses to red-end-down (staged
   shallow so the ball arrives slow), gravity rolls the ball west along the south
   lane, and it DROPS into the recessed pocket — the sink is pure ballistics into the
   hole. Stalls escalate the tilt target; the hinge-travel limit bounds everything.
4. RELEASE (gravity): torque goes to zero; the KEEL swings the tray back level. The
   final judged state — ball sunk, tray level, all settled — is held hands-off.
5. ORDER: forced by topology. Red-first parks the ball at the start wall (the divider
   walls off the pocket lane); the roof denies any drop-in. Only blue-then-red works.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Controller audit (authored plant): I_yy = 0.020 kg m^2, restoring Mgd = 1.2 * 9.81 *
0.055 = 0.648 N m/rad_sin, kp = 0.6, kd = 0.05 (+ body angular damping 4.0 /s ->
c = 0.08): omega_n = sqrt(kp/I) = 5.5 rad/s, zeta ~ 0.6, kd*dt/I = 0.02 << 1 (one-
substep wrench delay safe). Hinge rate by finite difference (ang-vel readback is
phantom under external wrenches on this pod).

Run (forge): python -u -m simgen_tasks.native_libero_i206.solve --headless [--seed N]
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
    from . import scene as _scene_mod  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as _scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

DT = 1.0 / 120.0
KP, KD, MGD = 0.6, 0.05, 1.2 * 9.81 * 0.055


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tilt_maze")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_torque() -> None:
        scene.tray.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def tilt_deg() -> float:
        return math.degrees(float(scene.tilt()[0]))

    def ball_loc() -> torch.Tensor:
        return scene.ball_local()[0]

    def report(tag: str) -> None:
        b = ball_loc()
        print(f"[solve] {tag:12s} | tilt={tilt_deg():+6.2f} deg "
              f"ball_loc=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f}) "
              f"|v|={float(scene.ball.data.root_lin_vel_w[0].norm()):.3f} "
              f"bay={bool(scene._bay[0])} crossed={bool(scene._crossed[0])} "
              f"returned={bool(scene._returned[0])} "
              f"in_pocket={bool(scene.in_pocket()[0])} level={bool(scene.level()[0])} "
              f"spare={bool(scene.spare_home()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # Servo state (persists across phases): torque sign convention is PROBED from
    # measured tilt progress — +body-y should be blue-end-down, but never assumed.
    sv = {"sign": 1.0, "cap": 0.40, "prev": None}

    def servo_step(target_deg: float) -> None:
        """One physics step under the hinge servo: gravity feedforward + PD on the
        hinge angle, torque applied in the TRAY BODY frame along body y (the hinge
        axis — immune to the wrench-frame drag), rate by finite difference."""
        th = float(scene.tilt()[0])
        w = 0.0 if sv["prev"] is None else (th - sv["prev"]) / DT
        sv["prev"] = th
        tt = math.radians(target_deg)
        tau = MGD * math.sin(th) + KP * (tt - th) - KD * w
        tau = max(-sv["cap"], min(sv["cap"], tau))
        t_body = torch.zeros(n, 1, 3, device=device)
        t_body[0, 0, 1] = sv["sign"] * tau
        scene.tray.set_external_force_and_torque(zero_wrench, t_body, env_ids=all_ids)
        env.step(no_action)

    def hold_tilt(target_deg: float, stop_fn, max_steps: int, tag: str) -> bool:
        """Run the servo toward/at `target_deg` until stop_fn() (checked each step).
        Probes the torque SIGN from measured progress and escalates the cap on a
        true stall. Returns False on timeout."""
        sv["prev"] = None
        win_i, win_th = 0, tilt_deg()
        for i in range(max_steps):
            servo_step(target_deg)
            if stop_fn():
                return True
            if i - win_i >= 90:
                th = tilt_deg()
                err0, err1 = abs(target_deg - win_th), abs(target_deg - th)
                if err1 > err0 + 0.5:
                    sv["sign"] = -sv["sign"]
                    print(f"[solve] {tag}: tilt moving away ({win_th:+.2f} -> "
                          f"{th:+.2f} vs target {target_deg:+.1f}); torque sign -> "
                          f"{sv['sign']:+.0f}", flush=True)
                elif err1 > 1.0 and err1 > err0 - 0.3:
                    sv["cap"] = min(sv["cap"] + 0.15, 0.9)
                    print(f"[solve] {tag}: tilt stalled at {th:+.2f} deg "
                          f"(target {target_deg:+.1f}); cap -> {sv['cap']:.2f} N m",
                          flush=True)
                win_i, win_th = i, th
        print(f"[solve] {tag}: timed out at tilt {tilt_deg():+.2f} deg", flush=True)
        return False

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    b = ball_loc()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) tilt={tilt_deg():+.2f} "
          f"ball_loc=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f})",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert float(b[0]) < 0.0 and float(b[1]) > 0.0, "ball must start in the north lane west"
    assert abs(tilt_deg()) < c.level_tol_deg, "tray must start level (keel)"
    assert bool(scene.spare_home()[0]), "spare must start in its cradle"
    s0 = print_score("P0 reset+settle (ball at the north-lane start)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: stroke 1 — blue end down, ball around the baffle ------------
    def in_bay_corner() -> bool:
        loc = ball_loc()
        return (float(loc[1]) < c.lane_s_y and float(loc[0]) > c.bay_x
                and float(scene.ball.data.root_lin_vel_w[0].norm()) < 0.25)

    ok = hold_tilt(8.0, in_bay_corner, 1800, "stroke1")
    if not ok:
        # steeper retry within the hinge limit
        ok = hold_tilt(10.5, in_bay_corner, 1200, "stroke1-steep")
    report("stroke1")
    if not ok:
        clear_torque()
        print("SIM_GEN_SOLVE: FAIL (ball never rounded the baffle)", flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "cannot be success with the ball in the bay"
    s1 = print_score("P1 ball rolled east and rounded the baffle into the south lane")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_bay + c.w_cross - 1e-6, \
        f"P1 score {s1} (expect bay+crossed = 0.50)"

    # ---------------- phase 2: stroke 2 — red end down, ball west into the pocket ----------
    # Staged shallow so the ball arrives slower than the pocket lip can reject.
    def past_mid() -> bool:
        return float(ball_loc()[0]) < -0.04

    def sunk() -> bool:
        return bool(scene.in_pocket()[0])

    hold_tilt(-6.0, past_mid, 900, "stroke2-run")
    ok = sunk() or hold_tilt(-2.5, sunk, 600, "stroke2-coast")
    esc = -4.0
    while not ok and esc >= -10.0:
        ok = hold_tilt(esc, sunk, 600, f"stroke2-esc{esc:+.0f}")
        esc -= 2.0
    report("stroke2")
    if not ok:
        clear_torque()
        print("SIM_GEN_SOLVE: FAIL (ball never sank into the pocket)", flush=True)
        os._exit(1)
    s2 = print_score("P2 ball dropped into the sunken green pocket")
    assert s2 >= s1 - 1e-6 and s2 >= 0.75 - 1e-5, f"P2 score {s2} (expect latched 0.75)"

    # ---------------- phase 3: release — the keel returns the tray level -------------------
    clear_torque()
    calm = 0
    for _ in range(600):
        env.step(no_action)
        if bool(scene.level()[0]) and bool(scene.settled()[0]):
            calm += 1
            if calm >= 60:  # half a second of continuous level+still
                break
        else:
            calm = 0
    report("release")
    if not bool(scene.success()[0]):
        print("SIM_GEN_SOLVE: FAIL (tray did not settle level with the ball sunk)",
              flush=True)
        os._exit(1)
    s3 = print_score("P3 released; keel returned the tray level (success live)")
    assert s3 >= 1.0 - 1e-6, f"success must score 1.0, got {s3}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
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
    main()
