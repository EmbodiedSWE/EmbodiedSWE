"""solve — force-driven solution for SpringLatchFridgeScene (close_fridge_i360).

Scene-level env (robot="null"). NOTHING is teleported after reset: every phase runs
through the live plant via the scene's `bolt_force` buffer — a BODY-frame force on the
slide bolt's knob, i.e. the single fingertip contact the embodiment argument names
(body frame == door frame: +x presses the door closed through the prismatic joint,
+/-y slides the bolt along its travel).

  PHASE 1 (retract): pull the knob -y (2.5 N) until the bolt is fully retracted
  (ext <= 5 mm, against its own stop). The protruding tip no longer reaches the proud
  keeper housing, so the door's swing corridor is clear.

  PHASE 2 (press): a velocity-servo fingertip press: the +x force is
  clamp(-tau_des / L, 0, F_MAX) with tau_des = k*(theta - theta_eq) + KV*(w_des - w),
  w_des = -min(OMEGA_CAP, K_APPROACH * theta) — cancel the spring, close at
  <= 0.5 rad/s, land softly on the 0 deg stop. F_MAX = 8 N: a one-finger push.

  PHASE 3 (throw): KEEP the press servo running (the spring would reopen the door the
  moment the finger left) and add +y (3 N) on the knob: the bolt slides into the
  keeper bore (4 mm play — only enterable with the door held flush). When the tip is
  >= 18 mm past the mouth, ALL forces are zeroed at once: the spring shoves the door
  back a fraction of a degree until the bolt presses the bore wall — the bolt now
  holds the door, not the finger.

  PHASE 4 (persistence): with both drive buffers asserted zero, >= 3.5 simulated
  seconds hands-off; success() is live state (a bolt that slips out or a door that
  springs open would revert it). Only then `SIM_GEN_SOLVE: SUCCESS`.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (the printed sequence never
decreases — asserted). Hard exit (os._exit) after the verdict, with a watchdog Timer
as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.close_fridge_i360.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=600.0)
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
    from simgen_tasks.close_fridge_i360 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Fingertip-scale drive (the honesty argument): F_MAX 8 N on the knob at the ~0.39 m
# hinge lever = 3.1 N*m authority vs the spring's <= 2.0 N*m at closed — a one-finger
# OSC push. RETRACT/THROW pulls are 2.5-3 N along the bolt's own travel.
F_MAX = 8.0  # N, press force cap (door-local +x on the knob)
F_RETRACT = 2.5  # N, -y pull to retract the bolt
F_HOLD = 0.8  # N, -y bias during the press: the free-sliding bolt otherwise creeps
#              outward under centrifugal force (omega^2*r) and re-jams — the finger
#              pushes DIAGONALLY, pinning the knob against the retract stop
F_THROW = 3.0  # N, +y push to throw the bolt
KV = 3.0  # N*m*s/rad velocity-servo gain
TAU_BIAS = 0.4  # N*m constant closing bias: the finger leans in, so imperfect spring
#               feedforward (lever mismatch, hold-bias torque) can't stall the endgame
OMEGA_CAP = 0.5  # rad/s max closing rate
K_APPROACH = 1.5  # w_des = -min(OMEGA_CAP, K_APPROACH * theta) -> soft landing
LEVER = 0.39  # m, knob's y-distance from the hinge (retracted)
RETRACT_TO = 0.005  # m, phase-1 target extension
THROW_TO = 0.042  # m, phase-3 target extension (tip ~32 mm past the mouth)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spring_latch_fridge")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        th = math.degrees(float(scene.door_angle()[0]))
        w = float(scene.door_rate()[0])
        ext = float(scene.bolt_ext()[0])
        tip = scene.bolt_tip()[0].tolist()
        print(f"[solve] {tag:12s} door={th:6.1f}deg rate={w:+.3f}rad/s ext={ext * 1000:5.1f}mm "
              f"tip=({tip[0]:+.3f},{tip[1]:+.3f},{tip[2]:+.3f}) "
              f"engaged={bool(scene.engaged()[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def press_force() -> float:
        """One-finger press servo: desired hinge torque -> +x knob force (never pulls)."""
        theta = float(scene.door_angle()[0])
        w = float(scene.door_rate()[0])
        w_des = -min(OMEGA_CAP, K_APPROACH * theta)
        tau_des = float(scene.k_spring[0]) * (theta - float(scene.theta_eq[0])) \
            - TAU_BIAS + KV * (w_des - w)
        return max(0.0, min(F_MAX, -tau_des / LEVER))

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    print(f"[solve] seed={args.seed} "
          f"theta_eq={math.degrees(float(scene.theta_eq[0])):.1f}deg "
          f"k={float(scene.k_spring[0]):.2f}N*m/rad "
          f"ext0={float(scene.ext0[0]) * 1000:.1f}mm", flush=True)
    report("reset")
    assert float(scene.door_drive.abs().max()) == 0.0
    assert float(scene.bolt_force.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: retract the bolt ===============================================
    for i in range(600):  # 5 s budget; << 1 s expected
        if float(scene.bolt_ext()[0]) <= RETRACT_TO:
            break
        scene.bolt_force[0, 1] = -F_RETRACT
        step(1)
    scene.bolt_force[0] = 0.0
    step(30)
    if float(scene.bolt_ext()[0]) > RETRACT_TO + 0.003:
        report("retract-fail")
        print("[solve] PHASE 1 FAILED: bolt did not retract", flush=True)
        verdict(False)
    report("retracted")
    phase_score("phase1")  # 0.250 (retract latch)

    # ================= PHASE 2: press the door flush (and keep pressing) =======================
    flushed = False
    for i in range(2400):  # 20 s budget; ~4 s expected
        theta = float(scene.door_angle()[0])
        if theta <= math.radians(0.3):
            flushed = True
            break
        scene.bolt_force[0, 0] = press_force()
        scene.bolt_force[0, 1] = -F_HOLD  # diagonal push: pin the bolt retracted
        step(1)
        if i and i % 240 == 0:
            report("pressing")
    if not flushed:
        print("[solve] PHASE 2 FAILED: press servo timed out", flush=True)
        verdict(False)
    report("flush-held")
    phase_score("phase2")  # 0.600 (retract + close latches; door HELD, not yet secure)

    # ================= PHASE 3: throw the bolt while holding the press =========================
    thrown = False
    for i in range(1200):  # 10 s budget; ~1 s expected
        if float(scene.bolt_ext()[0]) >= THROW_TO and bool(scene.engaged()[0]):
            thrown = True
            break
        scene.bolt_force[0, 0] = press_force()
        scene.bolt_force[0, 1] = F_THROW
        step(1)
    scene.bolt_force[0] = 0.0  # RELEASE everything: the bolt holds the door now
    if not thrown:
        report("throw-fail")
        print("[solve] PHASE 3 FAILED: bolt did not engage the keeper", flush=True)
        verdict(False)
    print(f"[solve] released with ext={float(scene.bolt_ext()[0]) * 1000:.1f}mm — "
          f"the keeper holds the door hands-off", flush=True)
    ok_settle = False
    for _ in range(48):  # up to 4 s to settle onto the bolt
        step(10)
        if bool(scene.success()[0]):
            ok_settle = True
            break
    if not ok_settle:
        report("secure-fail")
        print("[solve] PHASE 3 FAILED: success() not reached after release", flush=True)
        verdict(False)
    report("secured")
    phase_score("phase3")  # 1.000

    # ================= PHASE 4: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.door_drive.abs().max()) == 0.0, "door drive must be zero"
    assert float(scene.bolt_force.abs().max()) == 0.0, "bolt force must be zero"
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — never leave a GPU zombie
        print(f"[solve] CRASH: {type(e).__name__}: {e}", flush=True)
        os._exit(1)
