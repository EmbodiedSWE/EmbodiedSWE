"""solve — TELEPORT solution for PuddingCarouselScene (…_i118).

Scene-level env (robot="null"). Teleportation handles TRANSPORT ONLY:

  PHASE 1 (align): pure contact/joint dynamics — a velocity-servoed torque on the
  drum axle (the scene's `drum_drive` buffer, capped at TAU_MAX = 0.8 N*m, the
  fingertip-scale push the red lever affords at its 0.19-0.28 m arm) walks the drum
  DOWN from its sampled 48-75 deg start to the 0 deg load stop; the drive is cut at
  2 deg and viscous axle friction parks it on the stop.

  PHASE 2 (load): the ONLY teleport of the target — one pose write moves the pudding
  box from its floor slot to a hover 3 mm above the SILL SHELF outside the window
  (free space; the box is still ~8 cm outside the bay and ~5 cm outside the drum
  footprint — nothing is bypassed). Gravity lands it; then a velocity-servoed WORLD
  force (the scene's `box_drive` buffer, capped at PUSH_MAX = 2.5 N — a light
  fingertip push) slides it through the window, over the 3 mm sill-rim gap, onto the
  bay floor; the force is CUT the moment the box centre crosses the deep-inside
  threshold and friction brings it to rest fully inside the drum footprint. During
  the push a small negative bias torque holds the drum against its load stop (the
  stop is one-sided; the bias stands in for the stiction a real detent would give —
  TASK.md discusses the arm equivalent).

  PHASE 3 (seal): joint/contact dynamics — the same velocity-servoed axle torque
  raises the drum from 0 to the 92 deg sealed stop WITH THE BOX RIDING INSIDE (held
  by nothing but friction and the bay walls); the drive is cut at 89 deg and the drum
  coasts (~2 deg) onto its stop. The payload's carriage is real contact dynamics; no
  velocity is ever written into a contact, no rubric state is touched.

Prints the scene readouts and `SIM_GEN_SCORE <score>` at each phase boundary (the
printed sequence never decreases — asserted). After success() first holds, keeps
simulating >= 3.5 more simulated seconds with both drive buffers zero; only if
success() still holds (live state — a box that creeps out or a drum that drifts off
its stop would revert it) prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit
(os._exit) after the verdict, with a watchdog Timer as backstop — Kit teardown hangs
otherwise.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i118.solve --headless [--seed N]
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
    from simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i118 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Fingertip-scale drives (the honesty argument, shared with smoke.py):
# TAU_MAX 0.8 N*m = ~3-4 N tangential at the lever's 0.19-0.28 m arm — an easy
# single-finger OSC push; OMEGA_CAP 0.45 rad/s keeps carriage accelerations tiny.
# PUSH_MAX 2.5 N on a 100 g box — a light fingertip slide.
TAU_MAX = 0.8  # N*m about the axle
KV = 2.0  # N*m*s/rad axle velocity-servo gain
OMEGA_CAP = 0.45  # rad/s max rotation rate
K_APPROACH = 1.5  # omega_des ramps down near the target angle -> soft landing
CUT_ALIGN = math.radians(2.0)  # cut the align drive here; friction parks on the stop
CUT_SEAL = math.radians(89.0)  # cut the seal drive here; ~2 deg coast onto the stop
HOLD_TAU = -0.15  # N*m bias holding the drum on its load stop during the push
PUSH_MAX = 2.5  # N force cap on the box
KP_V = 30.0  # N*s/m box velocity-servo gain
V_PUSH = 0.12  # m/s target slide speed
KY = 2.0  # 1/s lateral centering gain (v_y_des = -KY * y_err)
DEEP_X = -0.050  # drum-local x: cut the push once the box centre passes this


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pudding_carousel")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        th = math.degrees(float(scene.drum_angle()[0]))
        w = float(scene.drum_rate()[0])
        loc = scene.box_local()[0].tolist()
        print(f"[solve] {tag:12s} drum={th:6.1f}deg rate={w:+.3f}rad/s "
              f"box_local=({loc[0]:+.3f},{loc[1]:+.3f},{loc[2]:+.3f}) "
              f"in_bay={bool(scene.in_bay()[0])} score={sc():.3f} "
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

    def settle_until(pred, max_steps: int = 480, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def servo_drum(target_rad: float, cut_rad: float, direction: float,
                   hold: float = 0.0, budget: int = 3000) -> bool:
        """Velocity-servo the axle toward `target_rad`; cut the drive once the angle
        passes `cut_rad` in `direction` (+1 seals, -1 aligns); leave `hold` on the
        buffer afterwards. Returns True if the cut angle was reached."""
        for i in range(budget):
            th = float(scene.drum_angle()[0])
            if (direction > 0 and th >= cut_rad) or (direction < 0 and th <= cut_rad):
                scene.drum_drive[0] = hold
                return True
            w = float(scene.drum_rate()[0])
            gap = abs(th - target_rad)
            w_des = direction * min(OMEGA_CAP, K_APPROACH * gap)
            tau = KV * (w_des - w)
            scene.drum_drive[0] = max(-TAU_MAX, min(TAU_MAX, tau))
            step(1)
            if i and i % 300 == 0:
                report("rotating")
        scene.drum_drive[0] = hold
        return False

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)
    print(f"[solve] seed={args.seed} theta0={math.degrees(float(scene.theta0[0])):.1f}deg "
          f"pudding=({float(scene.pudding.data.root_pos_w[0, 0] - scene.env_origins[0, 0]):.3f},"
          f"{float(scene.pudding.data.root_pos_w[0, 1] - scene.env_origins[0, 1]):.3f})",
          flush=True)
    report("reset")
    assert float(scene.drum_drive.abs().max()) == 0.0
    assert float(scene.box_drive.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: align the drum (joint dynamics) ================================
    if not servo_drum(target_rad=0.0, cut_rad=CUT_ALIGN, direction=-1.0):
        print("[solve] PHASE 1 FAILED: align drive timed out", flush=True)
        verdict(False)
    if not settle_until(lambda: bool(scene.aligned()[0])
                        and abs(float(scene.drum_rate()[0])) < 0.05, max_steps=360):
        report("align-fail")
        print("[solve] PHASE 1 FAILED: drum did not park on the load stop", flush=True)
        verdict(False)
    report("aligned")
    phase_score("phase1")  # 0.100 (align latch)

    # ================= PHASE 2: load the box (transport + contact push) ========================
    # Teleport = transport only: hover 3 mm above the sill shelf, zero velocity, well
    # outside the bay; the landing, the slide through the window and the seating on
    # the bay floor are live contact dynamics.
    ax, ay = c.axis_xy
    st = torch.zeros(1, 13, device=device)
    st[0, 0] = c.sill_center[0]
    st[0, 1] = ay
    st[0, 2] = c.box_rest_z + 0.003
    st[0, 3] = 1.0  # yaw 0: faces flat to the window
    st[0, 0:3] += scene.env_origins[0]
    scene.pudding.write_root_state_to_sim(st, torch.tensor([0], device=device))
    step(2)
    scene.mark_box_ref()  # wrench-frame reference for the world-frame push
    step(20)
    report("on-sill")

    scene.drum_drive[0] = HOLD_TAU  # hold the drum on its stop while the box rubs the walls
    pushed_in = False
    for i in range(1800):  # 15 s budget; ~2 s expected
        loc = scene.box_local()[0]
        if float(loc[0]) >= DEEP_X:
            pushed_in = True
            break
        v = scene.pudding.data.root_lin_vel_w[0]
        fx = KP_V * (V_PUSH - float(v[0]))
        fy = KP_V * (-KY * float(loc[1]) - float(v[1]))
        cap = PUSH_MAX
        scene.box_drive[0, 0] = max(-cap, min(cap, fx))
        scene.box_drive[0, 1] = max(-cap, min(cap, fy))
        step(1)
        if i and i % 240 == 0:
            report("pushing")
    scene.box_drive[0] = 0.0  # CUT: friction seats the box the rest of the way
    scene.drum_drive[0] = 0.0
    if not pushed_in:
        report("push-fail")
        print("[solve] PHASE 2 FAILED: box never crossed the deep-inside threshold", flush=True)
        verdict(False)
    if not settle_until(lambda: bool((scene.in_bay()
                                      & (scene.pudding.data.root_lin_vel_w.norm(dim=-1)
                                         < c.settle_box))[0]), max_steps=480):
        report("seat-fail")
        print("[solve] PHASE 2 FAILED: box did not settle inside the bay", flush=True)
        verdict(False)
    report("loaded")
    phase_score("phase2")  # 0.450 (align + load latches)

    # ================= PHASE 3: seal (carry the payload through the rotation) ==================
    if not servo_drum(target_rad=math.radians(c.drum_limit_deg), cut_rad=CUT_SEAL,
                      direction=+1.0):
        print("[solve] PHASE 3 FAILED: seal drive timed out", flush=True)
        verdict(False)
    if not bool(scene.in_bay()[0]):
        report("cargo-lost")
        print("[solve] PHASE 3 FAILED: box left the bay during the carry", flush=True)
        verdict(False)
    print(f"[solve] drive cut at {math.degrees(float(scene.drum_angle()[0])):.2f} deg "
          f"— friction coasts the drum onto its stop hands-off", flush=True)
    if not settle_until(lambda: bool(scene.success()[0]), max_steps=600):
        report("seal-fail")
        print("[solve] PHASE 3 FAILED: success() not reached after the cut", flush=True)
        verdict(False)
    report("sealed")
    phase_score("phase3")  # 1.000

    # ================= PHASE 4: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.drum_drive.abs().max()) == 0.0, "drives must be zero for persistence"
    assert float(scene.box_drive.abs().max()) == 0.0
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
