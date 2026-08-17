"""Teleport solution for HitchTowScene (sim_gen task
`living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_
i391`) — the task's legitimacy certificate.

ONE transport-only teleport: the shear pin is carried from its stand socket to a
hover pose above the aligned bores (zero velocity, upright). EVERY load-bearing
interaction runs through contact dynamics via velocity-regulated external wrenches,
re-set every step and zeroed before judging:
  - the TRACTOR cart is a transX D6 rider (rotation locked -> body frame == world
    frame): the MATE is a -x velocity-servo push until the coupler plate contacts the
    tongue's stop block (the hard stop IS the alignment jig); the TOW is a +x
    velocity-servo pull with the trailer dragged through the pin in shear;
  - the PIN descends with an xy PD onto the LIVE tractor-bore center + a z velocity
    servo pressing it down through both bores until the cap rests on the coupler
    plate. Gains respect the one-substep wrench delay (tractor KV*dt/m = 0.125,
    pin xy kp*dt/m = 0.67, pin z KV*dt/m = 0.17 — all < 1).

PLAN (read-only, from scene.describe()):
  P1 MATE — push the tractor rearward until mated_now (plate on the stop block,
     bores coaxial); hold a light press (score latch 0.25),
  P2 HITCH — teleport the pin above the bores (transport only), press it down
     through both plates until the cap seats; release (couple latch, score 0.60),
  P3 TOW — pull the tractor forward; the pin in shear drags the trailer out of the
     tunnel and onto the dock pad; brake, wrench off,
  P4 ring down to success (docked & coupled & towed & settled),
  P5 hands-off persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success() still holds
after the hands-off hold.

Run (forge): python -u -m simgen_tasks.living_room_scene4_stack_the_left_bowl_on_the_
             right_bowl_and_place_them_in_the_tray_i391.solve --headless [--seed N]
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

import os
import threading

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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81
# Tractor cart servo (mass 0.8, rotation joint-locked -> world force is body force;
# KV*dt/m = 12/(120*0.8) = 0.125 < 1).
C_KP = 3.0
C_KV = 12.0
MATE_VCAP = 0.06
MATE_FMAX = 3.0
MATE_PRESS = 0.4    # steady -x press holding the plates mated during the hitch (N)
TOW_VCAP = 0.10
TOW_FMAX = 6.0
TOW_TGT = 0.26      # trailer target x on the pad (dock_lo 0.20 + 60 mm margin)
# Pin press servo (mass 0.05: xy kp*dt/m = 4/(120*0.05) = 0.67; z KV*dt/m = 0.17).
PIN_KP = 4.0
PIN_KD = 0.4
PIN_FXY = 0.5
PIN_VDN = 0.06
PIN_KVZ = 1.0
PIN_FZ_LO, PIN_FZ_HI = -0.30, 0.80
PIN_HOVER = 0.015   # shank-tip hover height above the coupler-plate top (m)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hitch_tow")().build(num_envs=args.num_envs, device=device)
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

    def lx0() -> float:
        return float(scene.tractor_x()[0])

    def tx0() -> float:
        return float(scene.trailer_x()[0])

    def pin0() -> torch.Tensor:
        return scene.pin_loc()[0]

    def report(tag: str) -> None:
        p = pin0()
        print(f"[solve] {tag:12s} | tractor={lx0():+.4f} trailer={tx0():+.4f} "
              f"rel={float(scene.rel_dx()[0]):+.4f} "
              f"pin=({float(p[0]):+.4f},{float(p[1]):+.4f},{float(p[2]):+.4f}) | "
              f"mated={bool(scene.mated_now()[0])} coupled={bool(scene.coupled_now()[0])} "
              f"docked={bool(scene.docked_now()[0])} "
              f"tow={float(scene.tow_travel[0]):.3f}/{c.dist_min:.2f} "
              f"latches m/p={float(scene.mate_latch[0]):.0f}/{float(scene.pin_latch[0]):.0f} "
              f"settled={bool(scene.settled()[0])} | success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def tractor_force(fx: torch.Tensor | float) -> None:
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0] = fx
        scene.tractor.set_external_force_and_torque(f, zero_w, env_ids=all_ids)

    def tractor_off() -> None:
        scene.tractor.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def pin_off() -> None:
        scene.pin.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(90)
    p = pin0()
    print(f"[solve] readback (seed {args.seed}): tractor={lx0():+.4f} "
          f"(band [{c.l_spawn_lo:+.2f},{c.l_spawn_hi:+.2f}]), trailer={tx0():+.4f} "
          f"(band [{c.t_spawn_lo:+.2f},{c.t_spawn_hi:+.2f}]), "
          f"pin=({float(p[0]):+.4f},{float(p[1]):+.4f},{float(p[2]):+.4f}) "
          f"(socket ({c.stand_x:+.2f},{c.stand_y:+.2f},{c.pin_socket_z:.3f}))", flush=True)
    assert c.t_spawn_lo - 0.006 <= tx0() <= c.t_spawn_hi + 0.006, "trailer must spawn in band"
    assert c.l_spawn_lo - 0.006 <= lx0() <= c.l_spawn_hi + 0.006, "tractor must spawn in band"
    assert abs(float(p[0]) - c.stand_x) <= 0.006 and abs(float(p[1]) - c.stand_y) <= 0.006, \
        "pin must stand in its socket"
    assert abs(float(p[2]) - c.pin_socket_z) <= 0.006, "pin must rest tip-down in the socket"
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, "score must start ~0"

    # ---------------- phase 1: MATE — push the tractor onto the stop block -----------------
    # Velocity-servo toward (live) trailer_x + mate_dx, pressing 3 mm INTO the stop:
    # the hard stop catches the plate tip and jigs the bores coaxial.
    streak, i = 0, 0
    for i in range(1800):
        x = scene.tractor_x()
        vx = scene.tractor.data.root_lin_vel_w[:, 0]
        x_tgt = scene.trailer_x() + c.mate_dx - 0.003
        v_des = (C_KP * (x_tgt - x)).clamp(-MATE_VCAP, MATE_VCAP)
        tractor_force((C_KV * (v_des - vx)).clamp(-MATE_FMAX, MATE_FMAX))
        env.step(no_action)
        if (i + 1) % 200 == 0:
            print(f"[solve] P1 telemetry @{i + 1}: rel={float(scene.rel_dx()[0]):+.4f} "
                  f"tractor={lx0():+.4f} trailer={tx0():+.4f}", flush=True)
        streak = streak + 1 if bool(scene.mated_now()[0]) else 0
        if streak >= 40:
            break
    # Stage 2: shove the MATED pair to the trailer's hard stop and let it come to
    # rest there — otherwise the hold press keeps drifting the bores -x while the
    # pin descends (first forge run: the pin landed tip-on-tongue 8 mm off the
    # moving bore and wedged). At the stop the bores are stationary.
    streak = 0
    for j in range(1200):
        vx = scene.tractor.data.root_lin_vel_w[:, 0]
        tractor_force((C_KV * (-0.04 - vx)).clamp(-MATE_FMAX, MATE_FMAX))
        env.step(no_action)
        at_stop = (tx0() <= c.t_lo + 0.004
                   and abs(float(scene.trailer.data.root_lin_vel_w[0, 0])) < 0.005)
        streak = streak + 1 if at_stop else 0
        if streak >= 60:
            break
    tractor_force(-MATE_PRESS)  # keep the plates pinned on the stop through the hitch
    step(30)
    print(f"[solve] P1: mated in {i + 1} servo steps, shoved to the trailer stop in "
          f"{j + 1} (trailer={tx0():+.4f}, t_lo {c.t_lo:+.2f}), "
          f"rel={float(scene.rel_dx()[0]):+.4f} (mate_dx {c.mate_dx:.3f}) — press held",
          flush=True)
    assert tx0() <= c.t_lo + 0.006, "the mated pair must rest on the trailer stop"
    assert bool(scene.mated_now()[0]), "plates must be mated on the stop block"
    assert float(scene.mate_latch[0]) > 0.5, "mate latch must have fired"
    report("P1-done")
    s1 = print_score("P1 carts mated, bores aligned")
    assert s1 >= max(s0, 0.24), "mate credit missing"

    # ---------------- phase 2: HITCH — pin through both bores ------------------------------
    # Transport-only teleport: carry the pin from its socket to a hover pose above
    # the (now coaxial) bores — upright, zero velocity. Everything after is contact.
    hover_z = c.plate_z_hi + c.shank_len / 2 + PIN_HOVER
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = scene.tractor_bore_x()
    st[:, 2] = hover_z
    st[:, 0:3] += scene.env_origins
    st[:, 3] = 1.0
    scene.pin.write_root_state_to_sim(st, all_ids)
    print(f"[solve] P2: pin teleported to hover ({float(scene.tractor_bore_x()[0]):+.4f}, "
          f"+0.0000, {hover_z:.4f}) — press-down is contact physics", flush=True)

    def pin_servo(v_des_z: float, z_done: float, going_down: bool,
                  max_steps: int) -> bool:
        """xy PD onto the live bore center + z velocity servo; stall-aware when
        descending. Returns True when done (coupled streak going down, height
        reached going up)."""
        streak, stall_ref_z, stall_i = 0, None, 0
        for k in range(max_steps):
            tractor_force(-MATE_PRESS)  # re-set every step (wrench persistence trap)
            p = scene.pin_loc()
            v = scene.pin.data.root_lin_vel_w
            f_world = torch.zeros(n, 3, device=device)
            f_world[:, 0] = (PIN_KP * (scene.tractor_bore_x() - p[:, 0])
                             - PIN_KD * v[:, 0]).clamp(-PIN_FXY, PIN_FXY)
            f_world[:, 1] = (PIN_KP * (0.0 - p[:, 1])
                             - PIN_KD * v[:, 1]).clamp(-PIN_FXY, PIN_FXY)
            f_world[:, 2] = (c.pin_mass * G
                             + PIN_KVZ * (v_des_z - v[:, 2])).clamp(PIN_FZ_LO, PIN_FZ_HI)
            f_body = quat_apply_inverse(scene.pin.data.root_quat_w, f_world)
            scene.pin.set_external_force_and_torque(f_body.unsqueeze(1), zero_w,
                                                    env_ids=all_ids)
            env.step(no_action)
            z = float(pin0()[2])
            if (k + 1) % 150 == 0:
                pp = pin0()
                print(f"[solve] P2 telemetry @{k + 1}: pin_z={z:+.4f} "
                      f"(seat {c.pin_seat_z:.4f}) "
                      f"dx={float(pp[0]) - float(scene.tractor_bore_x()[0]):+.4f} "
                      f"coupled={bool(scene.coupled_now()[0])}", flush=True)
            if going_down:
                streak = streak + 1 if bool(scene.coupled_now()[0]) else 0
                if streak >= 30:
                    return True
                # stall detect: no descent for 200 steps and not coupled -> retry
                if stall_ref_z is None or z < stall_ref_z - 0.001:
                    stall_ref_z, stall_i = z, k
                elif k - stall_i >= 200:
                    print(f"[solve] P2: descent stalled at z={z:+.4f} — re-lift",
                          flush=True)
                    return False
            elif z >= z_done:
                return True
        return going_down and bool(scene.coupled_now()[0])

    seated = False
    for attempt in range(3):
        seated = pin_servo(-PIN_VDN, 0.0, going_down=True, max_steps=900)
        if seated:
            break
        # re-lift clear of the plates (xy PD recenters in the air), then retry
        pin_servo(+PIN_VDN, hover_z - 0.004, going_down=False, max_steps=600)
    pin_off()
    step(60)          # pin settles seated, press still holding the mate
    tractor_off()
    step(60)          # ALL wrenches off before judging
    print(f"[solve] P2: pin seated (attempt {attempt + 1}, seated={seated}), "
          f"pin_z={float(pin0()[2]):+.4f} (seat {c.pin_seat_z:.4f}) — wrenches OFF", flush=True)
    assert bool(scene.coupled_now()[0]), "pin must thread both bores, hands off"
    assert float(scene.pin_latch[0]) > 0.5, "couple latch must have fired"
    report("P2-done")
    s2 = print_score("P2 pin dropped through both bores — hitched")
    assert s2 >= max(s1, 0.59), "couple credit missing"

    # ---------------- phase 3: TOW — drag the trailer onto the dock pad --------------------
    # +x velocity servo on the tractor; the trailer is dragged through the pin in
    # shear. Speed ramps in so the lash takes up gently; break on the TRAILER
    # position readback (the towed thing, not the tow vehicle).
    x_tgt_l = TOW_TGT + c.mate_dx + 0.02  # tractor carrot past the trailer target
    i = 0
    for i in range(3000):
        x = scene.tractor_x()
        vx = scene.tractor.data.root_lin_vel_w[:, 0]
        vcap = min(TOW_VCAP, 0.02 + 0.08 * (i / 200.0))
        v_des = (C_KP * (x_tgt_l - x)).clamp(0.0, vcap)
        tractor_force((C_KV * (v_des - vx)).clamp(-TOW_FMAX, TOW_FMAX))
        env.step(no_action)
        if (i + 1) % 300 == 0:
            print(f"[solve] P3 telemetry @{i + 1}: trailer={tx0():+.4f} tractor={lx0():+.4f} "
                  f"tow={float(scene.tow_travel[0]):.3f} "
                  f"coupled={bool(scene.coupled_now()[0])}", flush=True)
        if tx0() >= TOW_TGT - 0.005:
            break
    # Brake: bleed the rig's momentum before releasing (coast v/damping otherwise).
    for _ in range(150):
        vx = scene.tractor.data.root_lin_vel_w[:, 0]
        tractor_force((C_KV * (0.0 - vx)).clamp(-TOW_FMAX, TOW_FMAX))
        env.step(no_action)
    tractor_off()
    step(60)
    print(f"[solve] P3: towed in {i + 1} servo steps, trailer={tx0():+.4f} "
          f"(dock_lo {c.dock_lo:.2f}), tow_travel={float(scene.tow_travel[0]):.3f} "
          f"(dist_min {c.dist_min:.2f}) — wrench OFF", flush=True)
    assert bool(scene.docked_now()[0]), "trailer must be on the dock pad"
    assert bool(scene.towed()[0]), "tow odometer must have accumulated dist_min"
    assert bool(scene.coupled_now()[0]), "the rig must arrive still hitched"
    report("P3-done")
    s3 = print_score("P3 trailer towed onto the dock pad")
    assert s3 >= s2 - 1e-6, "score decreased across P3"

    # ---------------- phase 4: ring down to success ----------------------------------------
    consec = 0
    for _ in range(2400):  # up to 20 s
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P4-ringdown")
    s4 = print_score("P4 at rest — docked, hitched, towed")
    assert s4 >= s3 - 1e-6, "score decreased across P4"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after ring-down)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"docked={bool(scene.docked_now()[0])} "
                      f"coupled={bool(scene.coupled_now()[0])} "
                      f"towed={bool(scene.towed()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"pin_z={float(pin0()[2]):+.4f}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
