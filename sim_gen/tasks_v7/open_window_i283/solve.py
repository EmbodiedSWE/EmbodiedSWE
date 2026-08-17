"""Teleport solution for SashVentScene (sim_gen task `open_window_i283`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: the red parcel is teleported once (zero
velocity, upright) onto the INSIDE half of the sill — the staging spot a hand would
set it on, on the solver's own side of the wall, far outside every rubric window.
Nothing is ever teleported into the rubric state: the tray lies on the FAR side of
the wall, reachable only by sliding the parcel THROUGH the propped window aperture
(the through-latch requires crossing the wall plane below the fixed pane while the
sash is open, and success requires that latch).

The SASH and the PARCEL-through-the-gap are driven like a hand would drive them —
velocity-regulated external forces (body frame == world frame: the prismatic sash
never rotates, the parcel is pushed at its CoM), re-set every step, zeroed before
judging. Gains respect the one-substep wrench delay (KV*dt/m ~= 0.17 << 1). The
PAWL TAB is never touched at all: the rising lift rail cams it aside and gravity
returns it — the mechanism does its own work.

PLAN (read-only, from scene.describe()): the sash is gravity-loaded shut and only
the pawl can hold it; the parcel can only reach the tray through the open gap:
  P1 lift the sash (vertical force servo, gravity feedforward) past the pawl to the
     hold height,
  P2 HOLD there until the pawl tab gravity-returns to horizontal under the lip,
  P3 lower the sash to just above the tab, release — it lands propped (dead-man
     check: the wrench is OFF and it stays open),
  P4 transport: teleport the parcel onto the inside sill staging spot,
  P5 push the parcel -x through the propped gap; it crosses the wall plane and
     settles on the green tray; ring down to success,
  P6 hands-off persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success() still holds
after the hands-off hold.

Run (forge): python -u -m simgen_tasks.open_window_i283.solve --headless [--seed N]
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
# Sash lift servo (mass 1.2: KV*dt/m = 24/(120*1.2) ~= 0.17).
S_KP = 3.0      # outer position loop -> desired velocity (1/s)
S_VUP = 0.12    # desired-velocity cap going up (m/s)
S_VDN = 0.08    # desired-velocity cap going down (m/s)
S_KV = 24.0     # velocity loop -> force (N s/m)
S_FMIN, S_FMAX = -6.0, 45.0
# Parcel push servo (mass 0.15: KV*dt/m = 8/(120*0.15) ~= 0.44 — damped by sliding
# friction; the stall force KV*VCAP = 1.2 N must EXCEED static breakaway
# mu_s*m*g ~= 0.74 N, else the servo stalls at the slow-mode threshold).
P_KP = 3.0
P_VCAP = 0.15
P_KV = 8.0
P_FF = 0.60     # static-breakaway feedforward while the cube is not yet moving
P_FMAX = 2.0    # transient tip-climb bound ~ m*g = 1.5 N; brief tipping is harmless
# The servo goes quiescent where KP*KV*|x - x_tgt| + FF ~ breakaway: aim DEEPER than
# the resting spot so that stall point (x_tgt + 0.031) lies well past the break
# threshold; the end fence catches any overshoot inside the tray band.
P_XTGT = -0.135


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sash_vent")().build(num_envs=args.num_envs, device=device)
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

    def q0() -> float:
        return float(scene.sash_q()[0])

    def tab0() -> float:
        return float(scene.tab_angle()[0])

    def report(tag: str) -> None:
        p = scene.parcel_local()[0]
        print(f"[solve] {tag:12s} | q={q0():+.4f} tab={tab0():+.3f}rad "
              f"open={bool(scene.sash_open()[0])} | parcel=({float(p[0]):+.3f},"
              f"{float(p[1]):+.3f},{float(p[2]):+.3f}) through_now="
              f"{bool(scene.parcel_through_now()[0])} in_tray="
              f"{bool(scene.parcel_in_tray()[0])} | latches o/t/t="
              f"{float(scene.open_latch[0]):.0f}/{float(scene.through_latch[0]):.0f}/"
              f"{float(scene.tray_latch[0]):.0f} | success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def sash_wrench_off() -> None:
        scene.sash.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def sash_servo(target: float, v_cap: float, tag: str, max_steps: int = 900,
                   hold_until=None, hold_streak: int = 30) -> None:
        """Drive the sash along its prismatic track with a gravity-feedforward
        velocity-regulated vertical force (re-set EVERY step). If `hold_until` is
        given, keep servoing at the target until the predicate holds for
        `hold_streak` consecutive steps; otherwise park near the target."""
        f = torch.zeros(n, 1, 3, device=device)
        done = 0
        for i in range(max_steps):
            q = scene.sash_q()
            v = scene.sash.data.root_lin_vel_w[:, 2]
            v_des = (S_KP * (target - q)).clamp(-v_cap, v_cap)
            fz = (c.sash_mass * G + S_KV * (v_des - v)).clamp(S_FMIN, S_FMAX)
            f[:, 0, 2] = fz
            scene.sash.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
            env.step(no_action)
            near = abs(q0() - target) <= 0.006
            if hold_until is not None:
                ok = near and hold_until()
            else:
                ok = near and abs(float(v[0])) < 0.03
            done = done + 1 if ok else 0
            if done >= hold_streak:
                break
        print(f"[solve] {tag}: sash at q={q0():+.4f} (target {target:+.4f}, "
              f"{i + 1} servo steps, tab={tab0():+.3f}rad)", flush=True)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(90)
    print(f"[solve] readback (seed {args.seed}): q={q0():+.4f} (closed), "
          f"tab={tab0():+.3f} rad (horizontal stop), q_prop={c.q_prop:.4f}, "
          f"q_open_min={c.q_open_min:.4f}", flush=True)
    assert abs(q0()) <= 0.01, "sash must start closed on the bottom stop"
    assert abs(tab0()) <= 0.05, "tab must start on its horizontal stop"
    for nm, body in (("parcel", scene.parcel), ("distractor", scene.distractor)):
        p = (body.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback: {nm} at ({float(p[0]):+.3f},"
              f"{float(p[1]):+.3f},{float(p[2]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, "score must start ~0"

    # ---------------- phase 1: lift the sash past the pawl ---------------------------------
    # The rising lift rail cams the tab aside on the way up (one-way ratchet).
    sash_servo(c.lift_q_hold, S_VUP, "P1", max_steps=900)
    assert q0() >= c.lift_q_hold - 0.01, "sash must reach the hold height"
    s1 = print_score("P1 sash lifted past the pawl")
    assert s1 >= s0 - 1e-6, "score decreased across P1"

    # ---------------- phase 2: hold; the pawl gravity-returns under the lip ----------------
    sash_servo(c.lift_q_hold, S_VUP, "P2", max_steps=900,
               hold_until=lambda: (tab0() > -0.05
                                   and float(scene.tab.data.root_ang_vel_w[0].norm()) < 0.4),
               hold_streak=30)
    assert tab0() > -0.08, "tab must have gravity-returned to its horizontal stop"
    s2 = print_score("P2 pawl returned under the held sash")
    assert s2 >= s1 - 1e-6, "score decreased across P2"

    # ---------------- phase 3: set the sash down ON the pawl, hands off --------------------
    sash_servo(c.q_prop + 0.012, S_VDN, "P3", max_steps=900)
    sash_wrench_off()
    step(120)  # free fall 12 mm onto the tab, ring down — the DEAD-MAN moment
    print(f"[solve] P3 released: q={q0():+.4f} (pawl rest {c.q_prop:+.4f}), "
          f"tab={tab0():+.3f}rad — wrench OFF", flush=True)
    assert abs(q0() - c.q_prop) <= 0.010, "sash must rest ON the pawl, wrench off"
    assert bool(scene.sash_open()[0]), "propped sash must count as open"
    # open latch needs an open+still streak; give it its window
    step(60)
    assert float(scene.open_latch[0]) > 0.5, "open latch must have fired (propped+still)"
    s3 = print_score("P3 sash propped on the pawl (dead-man held)")
    assert s3 >= max(s2, 0.29), "propped-open credit missing"

    # ---------------- phase 4: TRANSPORT parcel to the inside sill staging spot ------------
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.env_origins
    st[:, 0] += 0.09
    st[:, 2] += c.sill_top + c.cube / 2 + 0.003
    st[:, 3] = 1.0
    scene.parcel.write_root_state_to_sim(st, all_ids)
    print("[solve] transport parcel -> inside sill staging (+0.090, 0.000) — still on "
          "the solver's side of the wall", flush=True)
    step(60)
    p = scene.parcel_local()[0]
    assert float(p[0]) > 0.05, "staged parcel must still be on the inside"
    assert not bool(scene.parcel_through_now()[0]) and float(scene.through_latch[0]) < 0.5
    s4 = print_score("P4 parcel staged on the inside sill")
    assert s4 >= s3 - 1e-6, "score decreased across P4"

    # ---------------- phase 5: push the parcel through the propped gap ---------------------
    x_break = -0.095
    done = 0
    for i in range(1800):
        x = scene.parcel_local()[:, 0]
        v = scene.parcel.data.root_lin_vel_w[:, 0]
        v_des = (P_KP * (P_XTGT - x)).clamp(-P_VCAP, P_VCAP)
        fx = P_KV * (v_des - v)
        fx = fx + torch.where(v.abs() < 0.02, P_FF * v_des.sign(), torch.zeros_like(fx))
        fx = fx.clamp(-P_FMAX, P_FMAX)
        # The wrench is applied in the BODY frame; the cube can tip/roll while
        # crossing, so rotate the desired WORLD-frame push into the body frame
        # every step (the external-force frame-drag trap).
        f_world = torch.zeros(n, 3, device=device)
        f_world[:, 0] = fx
        f_body = quat_apply_inverse(scene.parcel.data.root_quat_w, f_world)
        scene.parcel.set_external_force_and_torque(
            f_body.unsqueeze(1), zero_w, env_ids=all_ids)
        env.step(no_action)
        if (i + 1) % 300 == 0:
            p = scene.parcel_local()[0]
            print(f"[solve] P5 telemetry @{i + 1}: x={float(p[0]):+.4f} "
                  f"z={float(p[2]):+.4f} v={float(v[0]):+.4f} fx={float(fx[0]):+.3f}",
                  flush=True)
        done = done + 1 if float(x[0]) <= x_break else 0  # break on POSITION readback
        if done >= 3:
            break
    scene.parcel.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    print(f"[solve] P5: parcel pushed through in {i + 1} servo steps, "
          f"x={float(scene.parcel_local()[0, 0]):+.4f} — wrench OFF", flush=True)
    assert float(scene.through_latch[0]) > 0.5, "through latch must have fired"
    # Ring-down: wait for success() to hold CONTINUOUSLY for 1 s.
    consec = 0
    for _ in range(2400):  # up to 20 s
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P5-ringdown")
    s5 = print_score("P5 parcel through the gap, at rest on the tray")
    assert s5 >= s4 - 1e-6, "score decreased across P5"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after push + ring-down)", flush=True)
        os._exit(1)

    # ---------------- phase 6: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                pv = float(scene.parcel.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: q={q0():+.4f} "
                      f"parcel_v={pv:.4f} in_tray={bool(scene.parcel_in_tray()[0])} "
                      f"open={bool(scene.sash_open()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s6 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s6 >= s5 - 1e-6
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
