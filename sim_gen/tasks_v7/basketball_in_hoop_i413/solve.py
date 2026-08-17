"""solve — demonstration solution for CaromCourtScene (basketball_in_hoop_i413).

Scene-level env (robot="null"). NOTHING is teleported — there is no transport in this
task at all: the ball starts parked on the chute behind the closed gate (the reset
state) and every metre it travels is rolled under gravity and banked off the driven
blade. The solver only writes the scene's two drive buffers — `drive_vane_t`, the
stand-in for the Franka's wrist-roll grip on the pointer bar, and `drive_gate_f`, the
stand-in for the jaw lifting the gate knob — and post_step applies them as wrenches on
the two D6-jointed controls.

  PHASE 1 (aim): P-servo on the vane bearing, tau = clamp(KP * err, +-TAU_MAX); the
  joint's own 0.35 N*m*s/rad damper is the D term (heavily overdamped, slowest pole
  ~ KP/damping ~ 1.7/s). Release only when close AND slow — the vane has no detent,
  so the stopping problem is real; the rubric's slow-gate makes a fast sweep worthless.
  PHASE 2 (release): hold ~1.5 N up on the gate (weight 0.49 N; the D6 stop caps the
  travel); the ball rolls out from rest, drops ~4.6 cm down the 15 deg chute, enters
  the court through the port at ~0.8 m/s. Drop the gate once the ball is inside.
  PHASE 3 (transit, hands off): the ball caroms off the aimed blade and rolls through
  the doorway into the bay; nothing is driven — physics finishes the task.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing: latched
credit must not evaporate). After success() first holds, keeps simulating >= 3.5 more
simulated seconds with the drive buffers zero (asserted); only if success() still
holds prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after the verdict,
daemon watchdog Timer as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.basketball_in_hoop_i413.solve --headless [--seed N]
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
    from simgen_tasks.basketball_in_hoop_i413 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()

# Vane aim servo (fingertip wrist-roll authority on the pointer bar). Vane inertia
# about z ~ 2.1e-3 kg*m^2; the joint's 0.35 N*m*s/rad damper is solver-implicit, so a
# pure P external torque is stable and heavily overdamped (zeta ~ 5 at KP = 0.6).
KP = 0.6  # N*m/rad
TAU_MAX = 0.45  # N*m hard cap (comfortable jaw authority on the 17 cm bar)
ERR_DONE = math.radians(1.5)  # release: within 1.5 deg of the bearing...
W_DONE = 0.05  # rad/s: ...and slow (never release mid-swing)
GATE_HOLD = 1.5  # N up (gate weight 0.49 N; the D6 stop caps the lift)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carom_court")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        p = scene.ball_local()[0]
        b = scene.bay_coords()[0]
        d_m = float((scene.ball_local()[0, :2] - scene.exit_m[0]).norm())
        print(f"[solve] {tag:12s} vane={math.degrees(float(scene.vane_yaw()[0])):+6.1f}deg "
              f"err={math.degrees(float(scene.aim_err()[0])):5.2f}deg "
              f"gate={float(scene.gate_lift()[0]) * 1000:5.1f}mm "
              f"ball=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"dM={d_m:.3f} bay=({float(b[0]):+.3f},{float(b[1]):+.3f}) "
              f"latches=({float(scene.aim_latch[0]):.0f},{float(scene.launch_latch[0]):.0f},"
              f"{float(scene.approach_latch[0]):.0f}) "
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
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    # ================= reset + settle ==========================================================
    env.reset(seed=args.seed)
    step(60)  # ball rolls ~2 cm down and parks against the closed gate tab
    th_star = math.degrees(float(scene.theta_star[0]))
    th0 = math.degrees(float(scene.theta0[0]))
    mx, my = (float(v) for v in scene.exit_m[0])
    print(f"[solve] seed={args.seed} theta*={th_star:+.1f}deg theta0={th0:+.1f}deg "
          f"M=({mx:+.3f},{my:+.3f})", flush=True)
    report("reset")
    assert float(scene.drive_vane_t.abs().max()) == 0.0
    assert float(scene.drive_gate_f.abs().max()) == 0.0
    phase_score("reset")  # ~0.000

    # ================= PHASE 1: aim the vane to theta* =========================================
    tgt = float(scene.theta_star[0])
    aimed = False
    for i in range(2400):  # 20 s budget
        yaw = float(scene.vane_yaw()[0])
        w = float(scene.vane.data.root_ang_vel_w[0, 2])
        e = math.atan2(math.sin(tgt - yaw), math.cos(tgt - yaw))
        if abs(e) < ERR_DONE and abs(w) < W_DONE:
            scene.drive_vane_t[0] = 0.0
            aimed = True
            break
        scene.drive_vane_t[0] = max(-TAU_MAX, min(TAU_MAX, KP * e))
        step(1)
        if i and i % 240 == 0:
            report("aiming")
    if not aimed:
        report("aim-fail")
        print("[solve] PHASE 1 FAILED: vane servo did not converge", flush=True)
        verdict(False)
    step(40)  # hands off; the slow-gate latch matures (24 substeps)
    report("aimed")
    if float(scene.aim_latch[0]) < 1.0:
        print("[solve] PHASE 1 FAILED: aim latch did not mature", flush=True)
        verdict(False)
    phase_score("phase1")  # ~0.20 (aim latched)

    # ================= PHASE 2: lift the gate, ball rolls in ===================================
    launched = False
    scene.drive_gate_f[0] = GATE_HOLD
    for i in range(600):  # 5 s budget: lift ~0.3 s, roll-out ~1 s
        step(1)
        p = scene.ball_local()[0]
        if float(p[:2].norm()) < 0.26 and float(p[2]) < 0.115:
            launched = True
            break
        if i and i % 120 == 0:
            report("releasing")
    scene.drive_gate_f[0] = 0.0  # hands off the gate — gravity re-closes it
    if not launched:
        report("launch-fail")
        print("[solve] PHASE 2 FAILED: ball did not enter the court", flush=True)
        verdict(False)
    report("launched")
    phase_score("phase2")  # ~0.45 (aim + launch latched)

    # ================= PHASE 3: hands-off transit into the bay =================================
    settled = False
    for i in range(960):  # 8 s budget for carom + roll-out + settle
        step(1)
        if bool(scene.success()[0]):
            settled = True
            break
        if i and i % 240 == 0:
            report("transit")
    if not settled:
        report("transit-fail")
        print("[solve] PHASE 3 FAILED: ball did not settle inside the bay", flush=True)
        verdict(False)
    report("in-bay")
    phase_score("phase3")  # 1.000

    # ================= PHASE 4: persistence (>= 3.5 simulated seconds, hands off) ==============
    assert float(scene.drive_vane_t.abs().max()) == 0.0
    assert float(scene.drive_gate_f.abs().max()) == 0.0
    assert float(scene.drive_ball_f.abs().max()) == 0.0
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
