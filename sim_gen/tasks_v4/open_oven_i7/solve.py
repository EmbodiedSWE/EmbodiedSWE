"""solve — TELEPORT-contract solution for OvenDialsScene (open_oven_i7).

Scene-level env (robot="null"). This task has NO transport component — nothing is
carried anywhere — so the teleport budget goes UNUSED: no root pose of any task object
is ever written by this solution. The load-bearing interaction (turning each detented
knob about its vertical spindle and stopping on the sampled target setting) is executed
entirely through the live dynamics:

  the solver writes the scene's `knob_drive` buffer — a torque about the spindle,
  clamped to TAU_MAX = 0.16 N*m. At the grip bar's half-length (42.5 mm) that is a
  ~3.8 N tangential fingertip push, i.e. exactly the authority the real Franka
  pinch-turn measured on this scene (3-5 N of OSC servo force at a 14-26 deg lead;
  see TASK.md). Every substep the scene's own post_step sums that drive with the
  viscous spindle friction and the detent spring toward the NEAREST setting, so the
  knob only moves the way physics lets it: the drive must dominate each detent's
  0.087 N*m peak resist to cross basins, and the pointer only rests where a detent
  holds it. Nothing is pinned, no velocity is written, no rubric state is touched.

Per knob (knob 0 then knob 1 — order is free, this is just a schedule):
  DRIVE    clamped PD toward the target angle (KP*err - KD*omega, |tau| <= TAU_MAX);
           terminal rate ~30 deg/s — a deliberate slow slew, not a flick;
  RELEASE  the moment the pointer is within RELEASE_DEG = 9 deg of the target (inside
           the measured 15 deg capture basin; asserted) the drive is zeroed — the
           detent spring alone seats the pointer on the mark, exactly the arm's
           release-in-basin endgame;
  SETTLE   hands off until the scene's own at_target() (8 deg AND spindle settled).

Prints the scene readouts and `SIM_GEN_SCORE <score>` at each phase boundary (the
printed sequence never decreases — asserted). After success() first holds, keeps
simulating >= 3 more simulated seconds with ALL drive buffers zero; only if success()
still holds (it is live state — a dial knocked off would revert it) prints exactly
`SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict, with a watchdog
Timer as backstop — Kit teardown hangs otherwise.

The intended single-Franka-arm strategy for the same plan (previously EXECUTED by a
real arm on this exact scene: straddle-pinch the grip bar, drag it along the
spindle-centred arc in <=100 deg chunks, release inside the basin) lives in TASK.md
as the embodiment argument.

Run (forge): python -u -m simgen_tasks.open_oven_i7.solve --headless
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
    from simgen_tasks.open_oven_i7 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Fingertip-scale drive (the whole honesty argument sits in these three numbers):
# TAU_MAX 0.16 N*m = ~3.8 N tangential at the bar half-length — the measured Franka
# pinch-turn authority; it exceeds the detent's 0.087 N*m peak resist (so basins can
# be crossed at all) but by less than 2x (so the plant, not the drive, owns the feel).
TAU_MAX = 0.16  # N*m about the spindle
KP = 1.5  # N*m/rad toward the target (clamped)
KD = 0.25  # N*m*s/rad damping -> terminal slew ~0.5 rad/s (~30 deg/s)
RELEASE_DEG = 9.0  # zero the drive here; must be inside the 15 deg capture basin


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.oven_dials")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    assert RELEASE_DEG < c.spacing_deg / 2 - 5.0, "release must sit inside the capture basin"

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        r = scene.readings_deg()[0].tolist()
        t = scene.target_deg()[0].tolist()
        w = scene.spindle_rate()[0].tolist()
        print(f"[solve] {tag:14s} read=({r[0]:7.1f},{r[1]:7.1f})deg "
              f"tgt=({t[0]:6.1f},{t[1]:6.1f}) rate=({w[0]:+.2f},{w[1]:+.2f})rad/s "
              f"at={scene.at_target()[0].int().tolist()} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

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
        threading.Timer(10.0, lambda: os._exit(code)).start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def settle_until(pred, max_steps: int = 360, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def turn_knob(k: int) -> bool:
        """Work knob k onto its target through the live detent plant: clamped PD drive
        about the spindle, release inside the capture basin, hands-off detent seating.
        Returns True iff the scene's own at_target() confirms the seated pointer."""
        tgt = float(scene.target_deg()[0, k])
        start = float(scene.readings_deg()[0, k])
        print(f"[solve] knob {k}: {start:+.1f} -> {tgt:+.1f} deg "
              f"({abs(tgt - start):.0f} deg traverse)", flush=True)
        released_at = None
        for i in range(2400):  # 20 s budget; a 240 deg traverse at ~30 deg/s needs ~8 s
            err_deg = tgt - float(scene.readings_deg()[0, k])
            if abs(err_deg) <= RELEASE_DEG:
                released_at = err_deg
                break
            w = float(scene.spindle_rate()[0, k])
            tau = KP * math.radians(err_deg) - KD * w
            scene.knob_drive[0, k] = max(-TAU_MAX, min(TAU_MAX, tau))
            step(1)
            if i and i % 300 == 0:
                report(f"knob{k}-drive")
        scene.knob_drive[0, k] = 0.0  # RELEASE: from here the detent does the seating
        if released_at is None:
            print(f"[solve] knob {k}: DRIVE TIMED OUT", flush=True)
            return False
        assert abs(released_at) < c.spacing_deg / 2, "release happened outside the basin"
        print(f"[solve] knob {k}: released at {released_at:+.1f} deg from target — "
              f"detent seats it hands-off", flush=True)
        ok = settle_until(lambda: bool(scene.at_target()[0, k]), max_steps=360)
        report(f"knob{k}-seated")
        return ok

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(90)
    report("reset")
    assert float(scene.knob_drive.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: knob 0 through the detent plant ================================
    if not turn_knob(0):
        print("[solve] PHASE 1 FAILED: knob 0 not seated on its target", flush=True)
        verdict(False)
    phase_score("phase1")  # ~0.500 (knob 0 latched + resting on target)

    # ================= PHASE 2: knob 1 through the detent plant ================================
    if not turn_knob(1):
        print("[solve] PHASE 2 FAILED: knob 1 not seated on its target", flush=True)
        verdict(False)
    if not settle_until(lambda: bool(scene.success()[0]), max_steps=240):
        print("[solve] PHASE 2 FAILED: success() not reached with both dials seated",
              flush=True)
        verdict(False)
    report("both-seated")
    phase_score("phase2")  # 1.000

    # ================= PHASE 3: persistence (>= 3 simulated seconds, hands off) ================
    assert float(scene.knob_drive.abs().max()) == 0.0, "drive must be zero for persistence"
    persist_steps = int(round(3.5 / env.dt))  # 420 physics steps at 1/120 s
    step(persist_steps)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
