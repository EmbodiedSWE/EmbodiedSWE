"""solve — demonstration solution for DialSetpointScene (press_switch_i161).

Scene-level env (robot="null"). NOTHING is teleported — there is no transport in
this task at all: both knobs start at their sampled angles (the reset state) and
every degree of progress flows through the live D6 + damping plant. The solver
only writes the scene's `drive_t` buffer — the stand-in for the Franka's wrist
roll on the grasped grip blade (TASK.md) — and post_step applies it as a torque
about the dial's ONE free axis, which z-rotation never skews.

  Per dial (0 then 1; the task declares no order): a cascaded velocity servo —
  outer rate command w_des = clamp(K_ANG * wrap(target - yaw), +-W_CAP), inner
  tau = KW * (w_des - w), |tau| <= TAU_MAX = 0.04 N*m (a fingertip on the 3 cm
  blade arm ~ 1.3 N — comfortable jaw authority). The knob's ~1e-4 kg*m^2
  inertia against the sampled loop wants the rate cascade (KW*dt/I ~ 0.3 < 1,
  and post_step wrenches act one substep late — gains stay well under 1). The
  servo approaches the flag, decelerates on the K_ANG cone and RELEASES only
  when slow and inside the band — the stopping problem IS the task (no detent
  catches the pointer; body damping parks it where released). If progress
  stalls the GAIN escalates, never the cap.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing: latched
credit must not evaporate). After success() first holds, keeps simulating >= 3.5
more simulated seconds with the drive buffers zero (asserted); only if success()
still holds prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after
the verdict, watchdog Timer as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.press_switch_i161.solve --headless [--seed N]
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
    from simgen_tasks.press_switch_i161 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Rotation servo (fingertip wrist-roll authority on the grip blade). Knob inertia
# about z ~ 1e-4 kg*m^2; the cascade stays discretely monotone (KW*dt/I ~ 0.3 < 1)
# against the one-substep wrench delay.
K_ANG = 4.0  # 1/s outer angle->rate gain (err decays at ~4/s)
W_CAP = 1.2  # rad/s max turn rate
KW0 = 4e-3  # N*m*s/rad inner rate gain (starting value)
KW_MAX = 1.0e-2  # escalation ceiling (stays delay-stable: KW*dt/I ~ 0.8)
TAU_MAX = 0.04  # N*m hard cap
ERR_DONE = math.radians(2.5)  # release: within 2.5 deg of the flag...
W_DONE = 0.10  # rad/s: ...and slow (never release mid-swing)


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.dial_setpoints")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        yaw = scene.knob_yaw()[0]
        err = scene.align_err()[0]
        print(f"[solve] {tag:12s} yaw=({math.degrees(float(yaw[0])):+6.1f}, "
              f"{math.degrees(float(yaw[1])):+6.1f})deg "
              f"err=({math.degrees(float(err[0])):5.1f}, "
              f"{math.degrees(float(err[1])):5.1f})deg "
              f"latch=({float(scene.align_latch[0, 0]):.2f}, "
              f"{float(scene.align_latch[0, 1]):.2f}) "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

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

    def servo_dial(d: int, budget: int = 2400) -> bool:
        """Rate-cascade torque servo: turn dial d to its sampled target, release only
        when slow and inside the band. Escalates GAIN (not the cap) on stalls."""
        tgt = float(scene.target[0, d])
        kw = KW0
        last_err = abs(wrap(tgt - float(scene.knob_yaw()[0, d])))
        last_ck = 0
        for i in range(budget):
            yaw = float(scene.knob_yaw()[0, d])
            w = float(scene.knobs[d].data.root_ang_vel_w[0, 2])
            e = wrap(tgt - yaw)
            if abs(e) < ERR_DONE and abs(w) < W_DONE:
                scene.drive_t[0, d] = 0.0
                return True
            w_des = max(-W_CAP, min(W_CAP, K_ANG * e))
            scene.drive_t[0, d] = max(-TAU_MAX, min(TAU_MAX, kw * (w_des - w)))
            step(1)
            if i - last_ck >= 240:  # stall watch: escalate GAIN, not the cap
                if abs(last_err) - abs(e) < math.radians(1.0) and kw < KW_MAX:
                    kw = min(KW_MAX, kw * 1.5)
                    print(f"[solve] dial{d} stalled at err="
                          f"{math.degrees(abs(e)):.1f}deg -> KW={kw:.4f}", flush=True)
                last_err, last_ck = e, i
            if i and i % 240 == 0:
                report(f"turning{d}")
        scene.drive_t[0, d] = 0.0
        return False

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    t0, t1 = (math.degrees(float(scene.target[0, d])) for d in (0, 1))
    s0, s1 = (math.degrees(float(scene.start[0, d])) for d in (0, 1))
    print(f"[solve] seed={args.seed} targets=({t0:+.1f}, {t1:+.1f})deg "
          f"starts=({s0:+.1f}, {s1:+.1f})deg", flush=True)
    report("reset")
    assert float(scene.drive_t.abs().max()) == 0.0 and float(scene.drive_f.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: dial 0 to its flag =============================================
    if not servo_dial(0):
        report("dial0-fail")
        print("[solve] PHASE 1 FAILED: dial 0 servo did not converge", flush=True)
        verdict(False)
    step(60)  # ring down; the latch counter matures
    report("dial0-set")
    phase_score("phase1")  # ~0.30 (dial 0 latched + aligned)

    # ================= PHASE 2: dial 1 to its flag =============================================
    if not servo_dial(1):
        report("dial1-fail")
        print("[solve] PHASE 2 FAILED: dial 1 servo did not converge", flush=True)
        verdict(False)
    step(60)
    report("dial1-set")

    settled = False
    for _ in range(48):  # up to 4 s for success (aligned + settled) to hold
        if bool(scene.success()[0]):
            settled = True
            break
        step(10)
    if not settled:
        report("settle-fail")
        print("[solve] PHASE 2 FAILED: success() not reached after release", flush=True)
        verdict(False)
    report("both-set")
    phase_score("phase2")  # 1.000

    # ================= PHASE 3: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.drive_t.abs().max()) == 0.0 and float(scene.drive_f.abs().max()) == 0.0, \
        "drives must be zero for persistence"
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
