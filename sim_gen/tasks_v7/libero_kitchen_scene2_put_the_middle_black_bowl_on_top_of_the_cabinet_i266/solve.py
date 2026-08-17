"""solve — hold-load + torque-crank solution for ScoopLiftCourtScene
(libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i266).

Scene-level env (robot="null"). Teleports are TRANSPORT ONLY; every credit-earning
interaction is contact dynamics driven by an applied hinge torque (what the Franka
does by pressing the lever paddle):

  PHASE H  HOLD   — PD + gravity-feedforward torque servo about the hinge axis
           raises the arm from its load stop (theta=-20 deg) to theta ~ 0 (mouth
           level) and holds it there. No credit moves.
  PHASE L  LOAD   — teleport the MIDDLE bowl (scene.target_idx) to the ARM frame
           point (0.283, 0, 0.100): centered over the scoop mouth's clear opening,
           ABOVE the scoop band (z 0.06 cap), zero velocity. Assert the score did
           not move — the teleport earned nothing. The bowl then FREE-FALLS ~4 cm
           through the mouth onto the slick cavity floor while the servo holds the
           arm level; the "loaded" latch fires on contact-settle.       -> 0.150
  PHASE U  LIFT   — slew the servo target to the dump stop (theta=120 deg,
           ~60 deg/s, velocity-damped). The bowl rides the pocket past the
           "lifted" gate (60 deg)                                        -> 0.300
           and the "delivered" gate (95 deg).                            -> 0.500
  PHASE P  POUR   — rate-servo the last leg (~103 -> 130 deg) at ~180 deg/s so the
           bowl has no time to dribble out against the wall face mid-swing (the
           slick pocket starts releasing near 104 deg), then press onto the dump
           stop (joint limit): the arm halts, the bowl keeps its tangential
           velocity, launches out of the mouth, flies through the WINDOW band
           (latch) and lands on the grippy court seat (latch), arrested
           inside the court.                                             -> 0.950
  PHASE R  RELEASE — cut all torque. The arm is gravity-bistable past vertical, so
           it stays pressed on the dump stop by itself; everything settles;
           success() -> score 1.000.
  PERSIST  >= 3.5 simulated seconds fully hands-off; success() must still hold
           before `SIM_GEN_SOLVE: SUCCESS` is printed.

The hinge axis is the arm body's OWN local +y at every angle, so the drive torque
is expressed as the body-frame vector (0, -tau, 0) (tau > 0 swings the scoop up):
correct under a body-frame wrench API, and off by at most the fixture yaw (<= 8
deg) under a world-frame one — the PD loop absorbs that. If headway stalls anyway,
the servo escalates its gain and torque cap (never the other way: a stalled
P-servo needs GAIN, not patience).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (latched credit — the printed
sequence never decreases) and exactly `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the persistence window. Hard exit (os._exit) with a watchdog Timer.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i266.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=900.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i266 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.scoop_lift_court")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    zero3 = torch.zeros(1, 1, 3, device=device)
    dt = env.dt

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def th_now() -> float:
        return float(scene.arm_deg()[0])

    def tidx() -> int:
        return int(scene.target_idx[0])

    def clear_torque() -> None:
        scene.arm.set_external_force_and_torque(zero3, zero3)

    def apply_tau(tau: float) -> None:
        """Hinge torque, up-swing-positive, as the body-frame vector (0, -tau, 0)
        (the hinge axis IS the arm's local +y; theta = -psi)."""
        t = torch.zeros(1, 1, 3, device=device)
        t[0, 0, 1] = -tau
        scene.arm.set_external_force_and_torque(zero3, t)

    def report(tag: str) -> None:
        ti = tidx()
        pa = scene.bowl_pos_arm()[0, ti]
        pf = scene.bowl_pos_fix()[0, ti]
        print(f"[solve] {tag:10s} theta={th_now():+7.2f} rate={float(scene.arm_rate[0]):+7.1f} "
              f"tgt={ti} tgt_arm=({pa[0]:+.3f},{pa[1]:+.3f},{pa[2]:+.3f}) "
              f"tgt_fix=({pf[0]:+.3f},{pf[1]:+.3f},{pf[2]:+.3f}) "
              f"L={scene._loaded_ever[0].tolist()} "
              f"U={scene._lifted_ever[0].tolist()} "
              f"D={scene._delivered_ever[0].tolist()} "
              f"W={scene._windowed_ever[0].tolist()} "
              f"C={scene._courted_ever[0].tolist()} "
              f"settled={bool(scene.settled()[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        if ok:
            print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        else:
            print("SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    # ----- PD + gravity-feedforward hinge servo ------------------------------------------------
    servo = {"kp": 1.8, "kd": 0.6, "cap": 4.0, "des": None}

    ARM_G = 0.941 * 9.81 * 0.074   # arm m*g*com_x  ~0.683 N*m at theta=0
    BOWL_G = c.bowl_mass * 9.81 * (c.cav_x0 + c.cav_w / 2)  # bowl aboard ~0.486

    def servo_tick(target: float, slew: float = 60.0) -> None:
        th = th_now()
        rate = float(scene.arm_rate[0])  # deg/s (FD in post_step)
        if servo["des"] is None:
            servo["des"] = th
        d = target - servo["des"]
        stp = slew * dt
        servo["des"] += max(-stp, min(stp, d))
        aboard = bool(scene.bowls_in_scoop()[0, tidx()])
        ff = (ARM_G + (BOWL_G if aboard else 0.0)) * math.cos(math.radians(th))
        tau = (ff + servo["kp"] * math.radians(servo["des"] - th)
               - servo["kd"] * math.radians(rate))
        tau = max(-servo["cap"], min(servo["cap"], tau))
        apply_tau(tau)
        step(1)

    def swing_to(target: float, tag: str, tol: float = 2.5,
                 max_steps: int = 2400) -> bool:
        """Slew the servo target to `target`; return once tracked within `tol` deg.
        Escalates gain + cap if headway stalls (stalled P-servo needs GAIN)."""
        probe_th, probe_i = th_now(), 0
        for i in range(max_steps):
            if abs(th_now() - target) < tol:
                return True
            servo_tick(target)
            if i - probe_i >= 90:
                cur = th_now()
                if abs(cur - probe_th) < 0.5 and abs(cur - target) > tol:
                    servo["kp"] = min(servo["kp"] * 1.5, 4.0)
                    servo["cap"] = min(servo["cap"] * 1.4, 7.0)
                    print(f"[solve] {tag}: stalled at {cur:+.1f} deg -> "
                          f"kp={servo['kp']:.2f} cap={servo['cap']:.1f}", flush=True)
                probe_th, probe_i = cur, i
        return abs(th_now() - target) < tol

    def hold_at(target: float, steps: int, done=None) -> bool:
        """Keep servoing at `target` for `steps` substeps (or until done())."""
        for _ in range(steps):
            if done is not None and done():
                return True
            servo_tick(target)
        return done() if done is not None else True

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    report("reset")
    assert not bool(scene._loaded_ever[0].any() | scene._lifted_ever[0].any()
                    | scene._delivered_ever[0].any() | scene._windowed_ever[0].any()
                    | scene._courted_ever[0].any()), "no latch may fire at reset"
    assert abs(th_now() + c.load_stop_deg) < 3.0, \
        f"arm must rest on the load stop (theta={th_now():+.1f})"
    phase_score("reset")  # ~0.000

    # ================= PHASE H: raise the arm to mouth-level and hold ==========================
    ok_h = swing_to(0.0, "hold", tol=2.0)
    if not ok_h:
        print("[solve] PHASE H FAILED: could not raise the arm to level", flush=True)
        report("hold_fail")
        verdict(False)
    hold_at(0.0, 60)  # let the hold quiesce
    report("hold")
    assert sc() <= last_score + 1e-6, "raising the empty arm must not earn credit"

    # ================= PHASE L: teleport the bowl above the mouth, free-fall in ================
    # Transport only: ARM frame (0.275, 0, 0.100) is centered over the open mouth
    # and ABOVE the scoop band (z cap 0.06); the teleport
    # must not move the score. The bowl free-falls ~4 cm into the slick pocket while
    # the servo holds the arm level; "loaded" latches on contact (theta ~ 0 < 10).
    tgt = scene.bowls[tidx()]
    drop_arm = torch.tensor([[float(c.cav_x0 + c.cav_w / 2), 0.0, 0.100]],
                            device=device)
    st = torch.zeros(1, 13, device=device)
    st[0, 0:3] = scene.arm_to_world(drop_arm)[0]
    st[0, 3:7] = scene.arm.data.root_quat_w[0]  # match the (near-level) pocket
    tgt.write_root_state_to_sim(st, torch.tensor([0], device=device))
    assert abs(sc() - last_score) < 1e-6, \
        f"teleport transport moved the score: {last_score} -> {sc():.3f}"

    ok_l = hold_at(0.0, 480,
                   done=lambda: bool(scene._loaded_ever[0, tidx()])
                   and float(tgt.data.root_lin_vel_w[0].norm()) < 0.05)
    report("load")
    if not ok_l:
        print("[solve] PHASE L FAILED: bowl did not settle into the scoop", flush=True)
        verdict(False)
    phase_score("load")  # 0.150

    # ================= PHASE U: crank the arm up past the lift/deliver gates ===================
    ok_u1 = swing_to(70.0, "lift", tol=8.0)
    if not (ok_u1 and bool(scene._lifted_ever[0, tidx()])):
        # give the latch a moment if we're hovering at the gate
        hold_at(70.0, 120, done=lambda: bool(scene._lifted_ever[0, tidx()]))
    report("lift")
    if not bool(scene._lifted_ever[0, tidx()]):
        print("[solve] PHASE U FAILED: bowl left the scoop before 60 deg", flush=True)
        verdict(False)
    phase_score("lift")  # 0.300

    ok_u2 = swing_to(103.0, "deliver", tol=8.0)
    if not (ok_u2 and bool(scene._delivered_ever[0, tidx()])):
        hold_at(103.0, 120, done=lambda: bool(scene._delivered_ever[0, tidx()]))
    report("deliver")
    if not bool(scene._delivered_ever[0, tidx()]):
        print("[solve] PHASE U FAILED: bowl left the scoop before 95 deg", flush=True)
        verdict(False)
    phase_score("deliver")  # 0.500

    # ================= PHASE P: fast final sweep, slam-launch at the dump stop =================
    # The slick pocket starts releasing the bowl near 104 deg; a slow approach would
    # dribble it out against the wall face. Rate-servo ~180 deg/s so the mouth reaches
    # the window in ~0.15 s (centrifugal force meanwhile pins the bowl to the floor),
    # then the joint-limit stop halts the arm and the bowl launches on its tangential
    # velocity, through the window, onto the seat.
    w_des = math.radians(180.0)
    for _ in range(240):
        if th_now() >= float(c.dump_stop_deg) - 2.0:
            break
        th = th_now()
        aboard = bool(scene.bowls_in_scoop()[0, tidx()])
        ff = (ARM_G + (BOWL_G if aboard else 0.0)) * math.cos(math.radians(th))
        tau = ff + 2.0 * (w_des - math.radians(float(scene.arm_rate[0])))
        apply_tau(max(-6.0, min(6.0, tau)))
        step(1)
    # press onto the stop while the pour completes
    ok_p = False
    for _ in range(720):
        if bool(scene._courted_ever[0, tidx()]):
            ok_p = True
            break
        apply_tau(1.2)
        step(1)
    report("pour")
    if not ok_p:
        print("[solve] PHASE P FAILED: bowl did not pour into the court", flush=True)
        verdict(False)
    assert bool(scene._windowed_ever[0, tidx()]), "window latch must fire mid-flight"
    phase_score("pour")  # 0.950 (or 1.000 if already settled)

    # ================= PHASE R: hands off — bistable arm stays on the dump stop ================
    clear_torque()
    ok_r = False
    for _ in range(10):  # up to ~5 s of settling
        step(60)
        if bool(scene.success()[0]):
            ok_r = True
            break
    report("release")
    if not ok_r:
        print("[solve] PHASE R FAILED: no settled success after release", flush=True)
        verdict(False)
    assert th_now() > 100.0, "arm must stay on the dump stop unpowered (bistable)"
    phase_score("release")  # 1.000

    # ================= persistence (>= 3.5 simulated seconds, hands off) =======================
    persist_steps = int(round(3.5 / dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
