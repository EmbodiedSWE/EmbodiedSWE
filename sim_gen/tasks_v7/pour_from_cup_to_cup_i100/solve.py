"""solve — TELEPORT-contract solution for TipDumpScene (pour_from_cup_to_cup_i100).

Scene-level env (robot="null"). This task has NO transport component — nothing is
carried anywhere — so the teleport budget goes UNUSED: no root pose of any task object
is ever written by this solution (the oven_dials precedent). The load-bearing
interaction (pressing the correct rocker pedal down against the return spring, HOLDING
it past the drain angle while gravity pours the balls out, then releasing) is executed
entirely through the live dynamics:

  the solver writes the scene's `press_tau` buffer — a torque about the trunnion
  hinge, self-clamped to TAU_MAX = 2.2 N*m (< the scene's own 2.5 N*m physical
  bound). At the pedal arm (155 mm) that is a ~14 N fingertip press — a light
  one-finger push-and-hold for a Franka. Every substep the scene's own post_step sums
  that press with the return spring and hinge damping, so the hopper only moves the
  way the plant lets it: stop pressing and the spring slams it back upright with the
  balls inside. The balls themselves are NEVER touched by the solver in any way —
  gravity rolls them over the spout lip and a real ballistic arc carries them into
  the basin. Nothing is pinned, no velocity is written, no rubric state is touched.

Phases (each ends with `SIM_GEN_SCORE`, printed sequence never decreases — asserted):
  P0 SETTLE   ~1.5 s hands-off; read back the randomized layout (basin side, present
              count, basin pose, start tilt); assert score ~0 and no success.
  P1 TIP      choose the pedal on the BASIN side, ramp a feedforward+PD press
              (tau = k*th_des + KP*err - KD*w, |tau| <= TAU_MAX, th_des slewed at
              ~40 deg/s) until the hinge passes 58 deg toward the basin — past the
              55 deg latch and the ~57 deg drain angle.
  P2 HOLD     keep holding at the 63 deg hold angle while the balls roll out and drop
              into the basin; if stragglers remain after ~3 s, rock the hold target
              +-4 deg at ~1.2 Hz (tilt modulation about the hinge — the axis that
              actually changes the roll-out force) until every present ball is in.
  P3 RELEASE  zero the press; the overdamped spring returns the hopper upright
              hands-off; wait for the scene's own success() (all present balls
              settled inside the basin, hopper upright and still).
  P4 PERSIST  >= 3.5 more simulated seconds with the press buffer asserted zero;
              only if success() still holds (live state — a ball bouncing out or the
              hopper swinging would revert it) print exactly `SIM_GEN_SOLVE: SUCCESS`.
Hard exit (os._exit) after the verdict, watchdog Timer as backstop.

The single-Franka-arm strategy for the same plan (press the 50 mm yellow pedal with
the closed-fingertip side of the gripper, follow the pedal's arc, hold ~9-14 N,
retract to release) lives in TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.pour_from_cup_to_cup_i100.solve --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=1350.0)
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
    from simgen_tasks.pour_from_cup_to_cup_i100 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                       os._exit(3))).start()

# Fingertip-scale press (the honesty argument in three numbers): TAU_MAX 2.2 N*m
# = ~14 N at the 155 mm pedal arm — a one-finger Franka press — UNDER the scene's own
# 2.5 N*m clamp, and only ~1.5x the ~1.45 N*m the spring + shifting balls demand at
# the hold angle, so the plant (not the press) owns the feel of the hold.
TAU_MAX = 2.2   # N*m about the trunnion hinge (scene clamps at 2.5 anyway)
KP = 6.0        # N*m/rad toward the slewed target angle
KD = 0.5        # N*m*s/rad hinge-rate damping (on top of the plant's own 0.30)
SLEW_DPS = 40.0  # deg/s target-angle slew — a deliberate controlled tip, not a flick
TIP_PASS_DEG = 58.0   # P1 exit: past the 55 deg latch and the ~57 deg drain angle
ROCK_AMP_DEG = 4.0    # straggler rock amplitude about the hold angle
ROCK_HZ = 1.2


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tip_dump_station")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    assert TAU_MAX <= c.press_tau_max, "solver press must respect the physical bound"
    assert TIP_PASS_DEG > max(c.drain_latch_deg, c.drain_deg), \
        "P1 exit must clear both the latch and the drain angle"
    assert c.hold_deg - ROCK_AMP_DEG > c.drain_deg, \
        "the rock must never dip below the drain angle"

    print("[solve] describe():", flush=True)
    print(scene.describe(), flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        th = float(scene.hinge_deg()[0])
        w = float(scene.hinge_rate()[0])
        pres = scene.present[0]
        n_pres = int(pres.sum())
        n_in = int((scene.in_basin()[0] & pres).sum())
        n_hop = int((scene.in_hopper()[0] & pres).sum())
        print(f"[solve] {tag:12s} hinge={th:+7.2f}deg rate={w:+.3f}rad/s "
              f"in_basin={n_in}/{n_pres} in_hopper={n_hop} "
              f"latch={bool(scene._tip_latch[0])} score={sc():.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> float:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)
        return s

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

    def press_toward(th_des_deg: float) -> None:
        """One substep of the pedal press: feedforward (cancel the spring at the
        target) + PD, clamped to the fingertip bound. Writes ONLY press_tau."""
        th = float(scene.hinge_rad()[0])
        w = float(scene.hinge_rate()[0])
        th_des = math.radians(th_des_deg)
        tau = c.spring_k * th_des + KP * (th_des - th) - KD * w
        scene.press_tau[0] = max(-TAU_MAX, min(TAU_MAX, tau))
        step(1)

    def settle_until(pred, max_steps: int, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    # ================= P0: reset + settle + layout readback ====================================
    env.reset(seed=args.seed)
    step(180)  # 1.5 s hands-off
    side = int(scene.side[0])
    n_pres = int(scene.present[0].sum())
    bp = scene.basin.data.root_pos_w[0] - scene.env_origins[0]
    print(f"[solve] layout: basin side={'+x' if side > 0 else '-x'} at "
          f"({float(bp[0]):+.3f},{float(bp[1]):+.3f}), present balls={n_pres}, "
          f"start tilt={float(scene.hinge_deg()[0]):+.2f} deg", flush=True)
    report("settled")
    assert float(scene.press_tau.abs().max()) == 0.0
    assert n_pres == int((scene.in_hopper()[0] & scene.present[0]).sum()), \
        "present balls must start inside the hopper"
    assert sc() <= 0.02, f"null-state score {sc():.3f} not ~0"
    assert not bool(scene.success()[0])
    phase_score("P0-settle")  # ~0.000

    # ================= P1: tip toward the basin past the drain angle ===========================
    print(f"[solve] env.dt={env.dt:.6f}", flush=True)
    hold = side * c.hold_deg
    th_des = float(scene.hinge_deg()[0])
    slew = SLEW_DPS * env.dt
    tipped = False
    for i in range(int(6.0 / env.dt)):
        th_des = min(th_des + slew, hold) if side > 0 else max(th_des - slew, hold)
        press_toward(th_des)
        th_now = float(scene.hinge_deg()[0])
        if i % 12 == 0:  # dense telemetry (diagnosing the hinge plant)
            hp = scene.hopper.data.root_pos_w[0] - scene.env_origins[0]
            print(f"[dbg] i={i:4d} th={th_now:+7.2f} des={th_des:+7.2f} "
                  f"w={float(scene.hinge_rate()[0]):+7.3f} "
                  f"press={float(scene.press_tau[0]):+6.3f} "
                  f"hp=({float(hp[0]):+.4f},{float(hp[1]):+.4f},{float(hp[2]):+.4f})",
                  flush=True)
        if side * th_now >= TIP_PASS_DEG:
            tipped = True
            break
        # fallback: the drain demonstrably happened (latch armed toward the basin AND
        # every present ball has already left the hopper on a fast swing)
        if bool(scene._tip_latch[0]) and \
                int((scene.in_hopper()[0] & scene.present[0]).sum()) == 0:
            tipped = True
            break
    report("P1-tipped")
    if not tipped:
        print("[solve] P1 FAILED: hinge never passed the drain angle", flush=True)
        verdict(False)
    s = phase_score("P1-tip")  # >= 0.15 (tip latch armed toward the basin)
    assert s >= c.w_latch - 1e-4, f"latch credit missing at P1: {s:.3f}"

    # ================= P2: hold (and rock for stragglers) until every ball is in ===============
    def all_in() -> bool:
        return bool(((scene.in_basin()[0] | ~scene.present[0])).all())

    t_hold = 0.0
    drained = False
    max_hold = 14.0  # s
    while t_hold < max_hold:
        if t_hold < 3.0:
            tgt = float(hold)
        else:  # stragglers: rock the tilt about the hold angle (never below drain)
            tgt = float(hold) + side * ROCK_AMP_DEG * math.sin(
                2 * math.pi * ROCK_HZ * (t_hold - 3.0))
        press_toward(tgt)
        t_hold += env.dt
        if all_in():
            drained = True
            break
        if int(t_hold / env.dt) % 240 == 0:
            report("P2-hold")
    report("P2-drained")
    if not drained:
        print("[solve] P2 FAILED: balls left un-drained after the hold", flush=True)
        verdict(False)
    s = phase_score("P2-hold")  # >= 0.75 (latch + full frac), success still False (held)
    assert s >= c.w_latch + c.w_frac - 1e-4, f"full-frac credit missing at P2: {s:.3f}"
    assert not bool(scene.success()[0]), "success must NOT hold while the pedal is pressed"

    # ================= P3: release — the spring returns the hopper hands-off ===================
    scene.press_tau[0] = 0.0
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=int(8.0 / env.dt))
    report("P3-released")
    if not ok:
        print("[solve] P3 FAILED: success() not reached after release", flush=True)
        verdict(False)
    s = phase_score("P3-release")  # 1.000
    assert s >= 1.0 - 1e-6

    # ================= P4: persistence (>= 3.5 simulated seconds, hands off) ===================
    assert float(scene.press_tau.abs().max()) == 0.0, "press must be zero for persistence"
    persist_steps = int(round(3.5 / env.dt))
    step(persist_steps)
    report("P4-final")
    phase_score("P4-final")
    still_ok = bool(scene.success()[0]) and sc() == 1.0
    print(f"[solve] persistence: {persist_steps} steps ({persist_steps * env.dt:.2f} s) "
          f"hands-off, success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()
