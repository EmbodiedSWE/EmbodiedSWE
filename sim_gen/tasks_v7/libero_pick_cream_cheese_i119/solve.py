"""Teleport solution for SiloTipPourScene (sim_gen task
`libero_pick_cream_cheese_i119`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. STAGING TRANSPORT (teleport): a single root-state write carries the BASKET from
   its ground start spot to free space 30 mm above the CREAM silo's catch pad,
   upright, zero velocity. The path is free air, the write satisfies no rubric
   clause by itself (the basket is airborne), and this is exactly the carry a
   Franka performs with a rim pinch. The basket FALLS the last 30 mm and settles
   on the pad by contact — the `staged` credit is produced by gravity, never
   written.
2. TIP THE SILO (applied torque + contact): a pure torque about the silo's hinge
   axis — the exact moment a fingertip pressing DOWN on the red lever paddle
   produces — tips the cream silo forward against its gravity-restoring bias.
   Bang-bang with a speed cap (quasi-static tip) and stall escalation. The torque
   axis IS the only axis the silo can rotate about, so the wrench is invariant
   under any force-frame drag convention (rotation-since-reset is a rotation about
   the axis itself); a sign probe at the start locks the direction. The box's
   slide down the tilted floor and out of the spout is pure friction contact
   (`poured` credit: exit while tipped — physics, never written).
3. DROP INTO THE BASKET (gravity + contact, hands-off): the torque is dropped
   shortly after the box exits; the box falls ~130 mm into the staged basket and
   the released silo swings back to its rest tilt on its own CoM bias. Every
   success clause (in-basket, upright, settled) is produced by ballistics and
   contact.
4. ORDER / IDENTITY: the basket is staged FIRST (a box poured without a receiver
   grounds — permanent fail), on the side read back from the box color assignment.
   The BROWN decoy and its silo are never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_pick_cream_cheese_i119.solve --headless [--seed N]
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
import traceback

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as _task_scene  # noqa: F401 - importing registers the scene/env
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as _task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.silo_tip_pour")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def cream_silo():
        return scene.silo_p if float(scene.cream_side[0]) > 0 else scene.silo_n

    def pitch() -> float:
        return float(scene.cream_silo_pitch_deg()[0])

    def report(tag: str) -> None:
        cl = scene._stand_local(scene.cream.data.root_pos_w)[0]
        bl = scene._stand_local(scene.basket.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | pitch={pitch():+.1f}deg "
              f"cream_std=({float(cl[0]):+.3f},{float(cl[1]):+.3f},{float(cl[2]):+.3f}) "
              f"basket_std=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
              f"in_silo={bool(scene.cream_in_silo()[0])} "
              f"staged={bool(scene._staged[0])} poured={bool(scene._poured[0])} "
              f"landed={bool(scene._landed[0])} breach={bool(scene._breach[0])} "
              f"grounded={bool(scene._grounded[0])} "
              f"in_basket={bool(scene.in_basket(scene.cream.data.root_pos_w)[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # silos settle onto their lower stops, boxes seat against the back walls
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    sq = scene.stand.data.root_quat_w[0]
    syaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    side = "+y" if float(scene.cream_side[0]) > 0 else "-y"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg "
          f"cream_side={side} pitch_p={float(scene.silo_pitch_deg(scene.silo_p)[0]):+.1f} "
          f"pitch_n={float(scene.silo_pitch_deg(scene.silo_n)[0]):+.1f}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(pitch() - c.rest_pitch_deg) < 3.0, \
        f"cream silo must rest near {c.rest_pitch_deg} deg, got {pitch():+.1f}"
    assert bool(scene.cream_in_silo()[0]), "cream box must start caged in its silo"
    p_brown = scene.brown.data.root_pos_w
    brown_in = scene._in_silo(scene.silo_n, p_brown) if side == "+y" \
        else scene._in_silo(scene.silo_p, p_brown)
    assert bool(brown_in[0]), "brown box must start caged in the other silo"
    assert not bool(scene.basket_staged()[0]), "basket must start off the pads"
    s0 = print_score("P0 reset+settle (silos at rest, boxes caged, basket away)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: basket -> cream pad (transport, gravity seats it) -----------
    pad = scene.pad_center_w(scene.cream_side)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = pad
    st[:, 2] += 0.030  # 30 mm free fall onto the pad
    st[:, 3:7] = scene.stand.data.root_quat_w  # square to the stand
    scene.basket.write_root_state_to_sim(st, all_ids)
    step(150)  # fall + settle, hands-off
    report("basket->pad")
    assert bool(scene.basket_staged()[0]), "basket must rest staged on the cream pad"
    assert bool(scene._staged[0]), "staged latch must be set"
    assert not bool(scene.success()[0]), "cannot be success with the box still caged"
    s1 = print_score("P1 basket staged on the cream-side catch pad")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_staged - 1e-6, \
        f"P1 score {s1} (expect staged={c.w_staged})"

    # ---------------- phase 2: torque the lever hinge; friction + gravity pour -------------
    silo = cream_silo()
    axis = quat_apply(scene.stand.data.root_quat_w,
                      torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))

    def apply_tau(mag: float, sign: float) -> None:
        if mag == 0.0:
            silo.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        else:
            t = (sign * mag) * axis.view(n, 1, 3)
            silo.set_external_force_and_torque(zero_wrench, t, env_ids=all_ids,
                                               is_global=True)

    def omega() -> float:
        return float((silo.data.root_ang_vel_w[0] * axis[0]).sum())

    # Gravity feedforward: the keel's restoring torque about the hinge decays as
    # sin(balance - p) from ~0.33 N*m at rest (p=-12.5) to zero at the ~55 deg
    # balance angle, so a CONSTANT "hold" torque is a runaway push at large pitch
    # (first forge run slammed to the +43 stop). Feed the exact decay forward and
    # close a near-critically-damped PD loop around it instead.
    _SIN0 = math.sin(math.radians(55.0 + 12.5))

    def grav_tau(p: float) -> float:
        return 0.34 * math.sin(math.radians(55.0 - p)) / _SIN0

    # sign probe: +torque about stand-local +y should RAISE the pitch. 0.55 N*m is
    # comfortably above the 0.33 rest holding torque; the silo rests on its lower
    # stop, so a wrong-signed push shows NO motion (not negative motion).
    sign = 1.0
    p_start = pitch()
    for _ in range(40):
        apply_tau(0.55, sign)
        env.step(no_action)
        if pitch() > p_start + 1.0:
            break
    if pitch() < p_start + 1.0:
        apply_tau(0.0, sign)
        for _ in range(30):
            env.step(no_action)
        for _ in range(40):
            apply_tau(0.55, -1.0)
            env.step(no_action)
            if pitch() > p_start + 1.0:
                sign = -1.0
                break
        print(f"[solve] torque probe: +y gave no lift; sign -> {sign:+.0f} "
              f"(pitch {pitch():+.1f})", flush=True)

    # PD servo about the hinge: I_hinge ~ 0.0105 kg*m^2, kp 0.010 N*m/deg
    # (0.57 N*m/rad -> ~7.4 rad/s), kd 0.15 N*m/(rad/s) ~ critical damping.
    # kd*dt/I ~ 0.12, safe against the one-substep wrench delay.
    target, kp, kd = 22.0, 0.010, 0.15
    tau_max = 0.80
    exit_step = -1
    last_probe_pitch, budget = pitch(), 1200
    for i in range(budget):
        if not bool(scene.cream_in_silo()[0]):
            exit_step = i
            print(f"[solve] box exited the spout at pitch {pitch():+.1f} deg "
                  f"(step {i}); holding the tilt while it clears the lip", flush=True)
            break
        p, w = pitch(), omega()
        tau = grav_tau(p) + kp * (target - p) - kd * (sign * w)
        apply_tau(max(0.0, min(tau, tau_max)), sign)
        env.step(no_action)
        if i % 90 == 89:
            if pitch() < last_probe_pitch + 1.0 and pitch() < target - 2.0:
                kp = min(kp + 0.005, 0.030)
                tau_max = min(tau_max + 0.20, 1.6)
                print(f"[solve] tip stalled at {pitch():+.1f} deg; "
                      f"kp -> {kp:.3f}, tau_max -> {tau_max:.2f}", flush=True)
            last_probe_pitch = pitch()
    if exit_step < 0:
        apply_tau(0.0, sign)
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (box never poured out)", flush=True)
        os._exit(1)
    hold_at = pitch()
    for _ in range(25):  # ~0.2 s: hold the current tilt while the box clears the lip
        p, w = pitch(), omega()
        tau = grav_tau(p) + kp * (hold_at - p) - kd * (sign * w)
        apply_tau(max(0.0, min(tau, tau_max)), sign)
        env.step(no_action)
    apply_tau(0.0, sign)
    print(f"[solve] torque released at pitch {pitch():+.1f} deg; hands off — "
          f"ballistics, the basket and the silo's own return bias take over", flush=True)

    ok_p2 = False
    for j in range(900):
        env.step(no_action)
        if bool(scene.success()[0]):
            ok_p2 = True
            break
        if j % 180 == 179:
            report(f"follow-{j + 1}")
    step(60)  # margin of settle, hands-off
    report("pour-drop")
    if not (ok_p2 or bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the pour)", flush=True)
        os._exit(1)
    s2 = print_score("P2 poured; cream box in the basket, silo returned")
    assert s2 >= s1 - 1e-6, "score decreased across the pour"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
