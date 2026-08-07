"""Teleport solution for CamPressScene (sim_gen task `push_button_i35`) — the
task's legitimacy certificate.

There is NO teleport in the nominal path: nothing needs transporting. The entire
solution is one real actuation of the machine's only degree of freedom:

1. READ (perception): tower pose, staircase CHIRALITY (which mirror wheel is
   mounted — it flips the required rotation direction) and the wheel's jittered
   park angle are read back from the live scene, never assumed.
2. ROTATE (applied torque + contact): a pure z-axis torque, capped at 1.2 N*m —
   what a fingertip pushing ~8 N tangentially on a rim peg at r=0.148 m applies —
   drives the wheel in the descending direction under a velocity-limited PD law
   (spinning the wheel up and slamming the staircase under the button would be
   nothing like a hand on a peg). The BUTTON IS NEVER TOUCHED: every millimetre of
   its 70 mm travel is the staircase turning under its foot, riser by riser.
3. SELF-LOCKING CHECKPOINTS: at each rubric depth the torque is RELEASED and the
   machine holds by itself (risers + friction) — the score print at each phase
   boundary is taken hands-off.
4. ARREST (contact): the final park is decided by the wheel's underside pin
   meeting the tower's stop post — the PD aims slightly PAST the arrest angle and
   presses gently into the post, so geometry, not the controller, chooses where
   the wheel stops.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: depth
credit is latched and the risers make regression physically impossible), then
holds HANDS-OFF for >= 3 simulated seconds after success() first turns True and
prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.push_button_i35.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

TAU_MAX = 1.2      # N*m — fingertip ~8 N on a rim peg at r=0.148 m
OMEGA_MAX = 1.2    # rad/s — hand-on-peg rotation speed


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cam_press")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | psi={float(scene.psi_deg()[0]):+8.2f}deg "
              f"depth={float(scene.depth()[0]) * 1000:6.2f}mm "
              f"seated={bool(scene.disc_seated()[0])} "
              f"sleeve={bool(scene.in_sleeve()[0])} "
              f"at_low={bool(scene.at_low()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(120)  # wheel seats on the pedestal, follower foot lands on step 0
    s_chir = int(scene.chir[0])
    disc = scene.discs[s_chir]
    tp = (scene.tower.data.root_pos_w - scene.env_origins)[0]
    tyaw = math.degrees(float(scene.tower_yaw[0]))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"tower=({float(tp[0]):+.3f},{float(tp[1]):+.3f}) yaw~{tyaw:+.1f}deg "
          f"chirality={s_chir:+d} psi_start={float(scene.psi_start[0]):+.2f}deg "
          f"psi_live={float(scene.psi_deg()[0]):+.2f}deg "
          f"depth={float(scene.depth()[0]) * 1000:.2f}mm", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.disc_seated()[0]), "wheel must start seated on the pedestal"
    assert bool(scene.in_sleeve()[0]), "follower must start in the sleeve"
    assert float(scene.depth()[0]) < 0.005, "button must start at the top of travel"
    s0 = print_score("P0 reset+settle (wheel parked at the head of the staircase)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- the one degree of freedom: torque-driven descent ---------------------------
    # Unwrapped wheel angle (the wrapped readout crosses +/-180 during the 241 deg
    # journey; a wrapped PD would push the blocked, rising direction).
    psi_u = float(scene.psi_deg()[0])
    # Aim 3 deg PAST the pin/post arrest so contact, not the controller, parks the wheel.
    arrest = (c.pin_deg_s + 180.0) - c.post_deg - c.contact_gap_deg
    psi_tgt = -s_chir * (arrest + 3.0)
    zero = torch.zeros(n, 1, 3, device=device)
    tq = torch.zeros(n, 1, 3, device=device)

    def rotate_until(pred, tag: str, max_steps: int) -> None:
        """Drive the wheel toward the low stop until `pred()`; then RELEASE the
        torque and let the self-locking machine settle hands-off."""
        nonlocal psi_u
        done = False
        for _ in range(max_steps):
            now = float(scene.psi_deg()[0])
            d = (now - psi_u + 180.0) % 360.0 - 180.0
            psi_u += d
            err = math.radians(psi_tgt - psi_u)
            wz = float(disc.data.root_ang_vel_w[0, 2])
            w_des = max(-OMEGA_MAX, min(OMEGA_MAX, 2.0 * err))
            tau = max(-TAU_MAX, min(TAU_MAX, 3.0 * (w_des - wz)))
            # Pure z torque, BODY frame == world frame for a body rotating about z.
            tq[:, 0, 2] = tau
            disc.set_external_force_and_torque(zero, tq)
            env.step(no_action)
            if pred():
                done = True
                break
        disc.set_external_force_and_torque(zero, zero)
        step(60)  # hands-off: the staircase holds the button by itself
        assert done, f"{tag}: predicate not reached after {max_steps} steps " \
                     f"(psi_u={psi_u:+.1f} depth={float(scene.depth()[0]) * 1000:.1f}mm)"

    # ---------------- phase 1: descend to the first latch depth ----------------------------------
    rotate_until(lambda: bool(scene._l1[0]), "P1", 4000)
    report("latch-1")
    s1 = print_score("P1 button 25 mm down (torque released; machine self-locks)")
    assert s1 >= max(s0, c.w1) - 1e-6, f"P1 score {s1}"

    # ---------------- phase 2: descend to the second latch depth ---------------------------------
    rotate_until(lambda: bool(scene._l2[0]), "P2", 4000)
    report("latch-2")
    s2 = print_score("P2 button 45 mm down (torque released; machine self-locks)")
    assert s2 >= max(s1, c.w1 + c.w2) - 1e-6, f"P2 score {s2}"

    # ---------------- phase 3: descend to arrest at the low stop ---------------------------------
    # A momentary stall (the ball's 10 mm drop onto step 7 brakes the wheel for a
    # step or two) must not count as "parked": require the stall to PERSIST while
    # the torque is still pressing — only the pin held on the stop post does that.
    stall = {"k": 0}

    def parked() -> bool:
        wz = float(disc.data.root_ang_vel_w[0, 2])
        now = bool(scene._l3[0]) and bool(scene.at_low()[0]) and abs(wz) < 0.05
        stall["k"] = stall["k"] + 1 if now else 0
        return stall["k"] >= 30

    rotate_until(parked, "P3", 6000)
    travel = abs(psi_u - float(scene.psi_start[0]))
    print(f"[solve] total wheel travel {travel:.1f} deg "
          f"(direction {-s_chir:+d}, arrest predicted at {arrest:.1f} deg)", flush=True)
    assert travel >= 225.0, f"full descent requires ~230+ deg of rotation, got {travel:.1f}"
    report("arrest")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (wheel arrested but success() is False)", flush=True)
        os._exit(1)
    s3 = print_score("P3 wheel arrested on the stop post; button fully down")
    assert s3 >= s2 - 1e-6, "score decreased across the final descent"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------------
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
    try:
        main()
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
