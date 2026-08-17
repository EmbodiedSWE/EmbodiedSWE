"""Wrench solution for ClutchValveStoveScene (sim_gen task
`libero_kitchen_scene8_turn_off_the_stove_i363`) — the task's legitimacy
certificate.

NOTHING is ever teleported. The single manipulated body is the KNOB, driven
by an external wrench (the stand-in for a gripper holding the red wing bar):
a vertical force along the guide axis (grip lift) and a torque about it
(wrist roll). Every load-bearing interaction is contact dynamics:

1. ENGAGE (press-release-rephase — NEVER twist while pressed, a pressed
   twist friction-drags the dial through the vane-bottom thrust contact):
   press straight up with a gentle lift PD; if a post parks under a vane,
   release fully (the 6 mm air gap re-opens — zero dial coupling) and
   re-phase the knob ~24 deg while it is DOWN (the free-spin decoy motion),
   then press again. The clutch is formed by post/vane CONTACT, latched at
   lift >= engage_z, with the dial provably un-wound (asserted).
2. TURN: with the knob pressed against its TOP stop (press force goes into
   the joint limit — the posts never touch the dial plate, cfg-asserted), the
   yaw servo winds clockwise; the posts push the vane FACES and the sealed
   rotor follows through ~35 deg of backlash down to its 0-deg shut stop.
   The flame dies as the dial crosses theta_off. Success is asserted FALSE
   here on record: the valve is off but the knob is still held up.
3. RELEASE (the order-forced finish): the wrench is zeroed; gravity drops the
   knob down its guide, the clutch opens, the dial rests against its stop.
   Only now does success() turn True.
4. The pots are NEVER touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing along
the solve: the engage latch and the min-angle progress latch only ever grow),
then holds HANDS-OFF for >= 3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene8_turn_off_the_stove_i363.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401 — registers the scene
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WDT = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                        os._exit(3)))
_WDT.daemon = True
_WDT.start()

_G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.clutch_valve_stove")().build(num_envs=args.num_envs,
                                                        device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    mg = c.knob_mass * _G

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def deg() -> float:
        return math.degrees(float(scene.rotor_angle()[0]))

    def lift_mm() -> float:
        return 1000.0 * float(scene.knob_lift()[0])

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | dial={deg():+.1f}deg lift={lift_mm():.1f}mm "
              f"engaged={bool(scene._engaged[0])} "
              f"released={bool(scene.released()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # --- the "gripper": one wrench on the knob, re-set EVERY step -----------------------------
    Z_TGT = 0.019       # above the top stop -> steady press against the D6 limit
    KP, KD = 250.0, 8.0
    # Saturating P velocity servo (NO constant feedforward: a ff term
    # dominates the P term and the knob runs w_des - T_FF/KW past the
    # request — the near-miss probe then coasts through its target).
    # Gain bound: KW*dt/Izz = 0.44 < 1 (one-substep wrench delay).
    KW, T_MAX = 0.08, 0.40

    def drive(w_des: float, *, hold: bool, f_cap: float) -> None:
        """One substep of grip emulation: lift PD (if `hold`) + yaw servo.
        Force/torque are BODY-frame, but the D6 locks rotX/rotY so body-z ==
        world-z always (no is_global stale-reference risk)."""
        f = 0.0
        if hold:
            z = float(scene.knob_lift()[0])
            vz = float(scene.knob.data.root_lin_vel_w[0, 2])
            f = max(0.0, min(f_cap, mg + KP * (Z_TGT - z) - KD * vz))
        w = float(scene.knob.data.root_ang_vel_w[0, 2])
        t = 0.0
        if w_des != 0.0 or abs(w) > 0.05:
            t = max(-T_MAX, min(T_MAX, KW * (w_des - w)))
        fb = torch.zeros(n, 1, 3, device=device)
        tb = torch.zeros(n, 1, 3, device=device)
        fb[:, 0, 2] = f
        tb[:, 0, 2] = t
        scene.knob.set_external_force_and_torque(fb, tb, env_ids=all_ids)
        env.step(no_action)

    def hands_off() -> None:
        zero = torch.zeros(n, 1, 3, device=device)
        scene.knob.set_external_force_and_torque(zero, zero, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # knob seats on the bottom stop; pots seat on the deck
    print(f"[solve] layout readback (seed {args.seed}):", flush=True)
    for name, body in [("rotor", scene.rotor), ("knob", scene.knob),
                       ("pot0", scene.pots[0]), ("pot1", scene.pots[1])]:
        pp = (body.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve]   {name}: ({float(pp[0]):+.3f},{float(pp[1]):+.3f},"
              f"{float(pp[2]):.3f})", flush=True)
    report("reset")
    for b in [scene.rotor, scene.knob, *scene.pots]:
        assert torch.isfinite(b.data.root_pos_w).all(), "NaN/inf after settle"
    m_r = float(scene.rotor.root_physx_view.get_masses().sum())
    assert abs(m_r - c.rotor_mass) < 0.005, f"rotor mass readback {m_r}"
    m_k = float(scene.knob.root_physx_view.get_masses().sum())
    assert abs(m_k - c.knob_mass) < 0.005, f"knob mass readback {m_k}"
    th0 = deg()
    assert c.theta0_lo_deg - 3.0 <= th0 <= c.theta0_hi_deg + 3.0, \
        f"dial must start in the open window (saw {th0:+.1f} deg)"
    assert lift_mm() <= 2.0, f"knob must rest at the bottom stop ({lift_mm():.1f} mm)"
    assert not bool(scene.valve_off()[0]), "the stove must start ON"
    s0 = print_score("P0 reset+settle (flame lit, knob parked, clutch open)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.02, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: ENGAGE (press-release-rephase, never twist-while-pressed) ----
    # Twisting WHILE pressed would wind the dial before engagement: the 2 N
    # thrust contact on the vane BOTTOMS has a friction torque (mu*N*r) that
    # the tiny-inertia rotor tracks within milliseconds (seen on the forge:
    # dial at the stop by "engaged"). So: press straight up with NO twist
    # (no sliding -> no friction torque on the dial); if a post parks under
    # a vane, let go completely (the 6 mm air gap re-opens -> zero coupling),
    # re-phase the knob ~24 deg while it is DOWN — the free-spin decoy motion
    # — and press again. The 59 deg gaps admit the 24 deg posts over most
    # phases, so a few tries always find one.
    engaged_now = False
    for attempt in range(12):
        streak = 0
        for _ in range(60):
            drive(0.0, hold=True, f_cap=mg + 2.0)
            streak = streak + 1 if float(scene.knob_lift()[0]) >= c.engage_z + 0.0005 \
                else 0
            if streak >= 20:
                engaged_now = True
                break
        if engaged_now:
            break
        hands_off()
        for _ in range(40):  # fall back onto the bottom stop
            env.step(no_action)
            if float(scene.knob_lift()[0]) <= 0.002:
                break
        for _ in range(25):  # free twist at the bottom: dial untouched
            drive(-2.0, hold=False, f_cap=0.0)
        for _ in range(15):  # brake the wing before the next press
            drive(0.0, hold=False, f_cap=0.0)
        print(f"[solve]   press blocked (attempt {attempt + 1}) — released and "
              f"re-phased (lift {lift_mm():.1f} mm, dial {deg():+.1f} deg)",
              flush=True)
    if not engaged_now:
        report("engage-FAIL")
        print("SIM_GEN_SOLVE: FAIL (clutch never engaged)", flush=True)
        os._exit(1)
    report("engaged")
    assert bool(scene._engaged[0]), "engage latch must have fired"
    assert not bool(scene.success()[0]), "engaged-but-open must not be success"
    s1 = print_score("P1 clutch engaged (knob lifted into the vane band)")
    assert s1 >= 0.15 - 1e-6 and s1 >= s0 - 1e-6, f"expected >= 0.15, got {s1}"
    # engaging must NOT have wound the dial (the whole point of the phase
    # split): drag through the pop is bounded by the backlash + coast
    assert deg() >= th0 - 25.0, \
        f"engage dragged the dial ({th0:+.1f} -> {deg():+.1f} deg)"
    assert s1 <= 0.35, f"engage-only credit must stay partial, got {s1}"

    # ---------------- phase 2: TURN clockwise to the shut stop ------------------------------
    # Full press now: the force goes into the D6 TOP stop, not any contact
    # (fully lifted posts clear the dial plate by cfg assert), so the turn is
    # friction-free at the press interface.
    t_turn = 0
    last_deg, stall = deg(), 0
    while deg() > 1.5 and t_turn < 2400:
        drive(-1.2, hold=True, f_cap=10.0)
        t_turn += 1
        if t_turn % 240 == 0:
            report(f"turn t={t_turn}")
            stall = stall + 1 if deg() > last_deg - 2.0 else 0
            assert stall < 4, f"dial stalled at {deg():+.1f} deg"
            last_deg = deg()
    assert deg() <= 2.0, f"dial must reach the shut stop (saw {deg():+.1f} deg)"
    # stop the wrist: bring the knob's spin down while still holding it up
    for _ in range(80):
        drive(0.0, hold=True, f_cap=10.0)
    report("wound shut")
    assert bool(scene.valve_off()[0]), "valve must read OFF at the stop"
    assert lift_mm() >= 1000.0 * c.engage_z - 1.0, "knob must still be held up"
    # ON RECORD: valve off but knob still held -> NOT success (release clause)
    assert not bool(scene.success()[0]), \
        "success must be withheld while the knob is still held up"
    s2 = print_score("P2 dial wound to the shut stop (flame out, knob still held)")
    assert s2 >= s1 - 1e-6 and abs(s2 - 0.70) < 0.005, f"expected 0.70, got {s2}"

    # ---------------- phase 3: RELEASE (gravity closes the clutch) --------------------------
    hands_off()
    step(180)  # knob free-falls its 16 mm and seats; dial rests on its stop
    report("released")
    assert bool(scene.released()[0]), \
        f"knob must drop back to the bottom stop (lift {lift_mm():.1f} mm)"
    assert bool(scene.valve_off()[0]), "dial must stay at the shut stop"
    if not bool(scene.success()[0]):
        step(240)  # let everything calm
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after release)", flush=True)
        os._exit(1)
    s3 = print_score("P3 knob released; stove off and settled")
    assert s3 >= s2 - 1e-6 and abs(s3 - 1.0) < 1e-6, f"expected 1.0, got {s3}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s (gravity + stops hold the OFF state)")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6 \
        and abs(s4 - 1.0) < 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — fail fast, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)
