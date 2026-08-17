"""Teleport solution for PawlLadderScene (sim_gen task
`libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i431`) — the task's
legitimacy certificate.

This solve uses NO teleports at all, and NEVER touches the cart: the ONLY actuation
is a velocity-regulated vertical force on the PAWL (a hand on the red T-handle,
re-set every step, zeroed before judging). The cart is moved exclusively by ramp
gravity and arrested exclusively by the dropped pawl blade — exactly the metering
mechanism the task is about. Gains respect the one-substep wrench delay
(KV*dt/m ~= 0.17 << 1); forces act in the BODY frame along the body z axis (== the
vertical guide axis; the external-force frame-drag trap).

PLAN (read-only, from scene.describe()):
  P0 settle + readback: the spawned cart glides a few mm and is ARRESTED by the
     seated pawl at the station-0 rest (the dead-man demo); read the target k from
     the scene; score ~0,
  P1..Pk metering cycles, one notch each: LIFT the pawl ~22 mm by force (the blade
     clears the rack; the cart accelerates away), watch the cart displacement, and
     RELEASE EARLY (~16 mm of cart travel) so the blade lands on the passing land,
     rides it, and drops into the NEXT notch, arresting the cart one station
     further. A same-notch recatch (released too early) is detected by station
     readback and simply retried — the latches make score monotone,
  P(k+1) ring-down to success (fin at the green post, pawl seated, ball aboard),
  P(k+2) hands-off persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success() still holds
after the hands-off hold.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i431.solve --headless [--seed N]
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
# Pawl lift servo (mass 0.15: KV*dt/m = 3.0/(120*0.15) ~= 0.17).
P_KP = 8.0      # outer position loop -> desired velocity (1/s)
P_VCAP = 0.15   # desired-velocity cap (m/s)
P_KV = 3.0      # velocity loop -> force (N s/m)
P_FMIN, P_FMAX = -0.5, 6.0
LIFT_REF = 0.022     # pawl z target while lifting (m)
RELEASE_DELTA = 0.016  # cart travel from the station rest at which we let go (m)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pawl_ladder_i431")().build(num_envs=args.num_envs, device=device)
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

    def step(m: int) -> None:
        for _ in range(m):
            env.step(no_action)

    def q0() -> float:
        return float(scene.cart_q()[0])

    def pq0() -> float:
        return float(scene.pawl_q()[0])

    def station() -> int:
        return int(round((q0() - c.rest_off) / c.notch_pitch))

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | q={q0():+.4f} (st {station()}) pawl={pq0():+.4f} "
              f"| foul={float(scene.foul_latch[0]):.0f} lift={float(scene.lift_latch[0]):.0f} "
              f"arrive={[int(v) for v in scene.arrive_latch[0].tolist()]} "
              f"| ball={bool(scene.ball_in_tray()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def pawl_off() -> None:
        scene.pawl.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def pawl_lift(z_ref: float) -> None:
        """One step of the pawl lift servo: BODY-frame z force (the guide axis)
        with a feedforward cancelling gravity along it."""
        pq = scene.pawl_q()
        v_body = quat_apply_inverse(scene.pawl.data.root_quat_w,
                                    scene.pawl.data.root_lin_vel_w)
        v_des = (P_KP * (z_ref - pq)).clamp(-P_VCAP, P_VCAP)
        fz = (c.pawl_mass * G * math.cos(c.pitch)
              + P_KV * (v_des - v_body[:, 2])).clamp(P_FMIN, P_FMAX)
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 2] = fz
        scene.pawl.set_external_force_and_torque(f, zero_w, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)  # the cart glides a few mm and is arrested by the seated pawl
    masses = scene.cart.root_physx_view.get_masses().flatten()
    assert abs(float(masses[0]) - c.cart_mass) <= 0.2 * c.cart_mass, \
        f"authored cart mass missing (got {float(masses[0]):.3f})"
    k = int(scene.k[0])
    p_st = (scene.station.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] readback (seed {args.seed}): station=({float(p_st[0]):+.3f},"
          f"{float(p_st[1]):+.3f},{float(p_st[2]):+.3f}) q={q0():+.4f} "
          f"pawl={pq0():+.4f} k={k} q_target={float(scene.q_target()[0]):.4f} "
          f"v_term={c.v_term:.3f}", flush=True)
    assert abs(q0() - c.q_rest_i(0)) <= 0.004, \
        "cart must be arrested at the station-0 rest by the seated pawl (dead-man)"
    assert pq0() <= c.seat_tol, "pawl must spawn seated"
    assert 2 <= k <= 4, "target station out of the sampled range"
    report("reset")
    s_prev = print_score("P0 reset+settle (pawl arrests the spawn glide)")
    assert s_prev < 0.02, "score must start ~0"

    # ---------------- phases 1..k: metering cycles -----------------------------------------
    cur = 0  # station we are seated at
    while cur < k:
        target = cur + 1
        advanced = False
        for attempt in range(5):
            q_start = q0()
            # LIFT: servo the pawl up until the blade is clear
            lifted = False
            for i in range(400):
                pawl_lift(LIFT_REF)
                env.step(no_action)
                if pq0() >= c.lift_min + 0.003:
                    lifted = True
                    break
            assert lifted, f"pawl failed to lift (attempt {attempt}, pq={pq0():+.4f})"
            # HOLD the lift while the cart runs; RELEASE EARLY
            released = False
            for i in range(600):
                pawl_lift(LIFT_REF)
                env.step(no_action)
                if q0() - q_start >= RELEASE_DELTA:
                    released = True
                    break
            pawl_off()
            assert released, f"cart never ran after the lift (q={q0():+.4f})"
            # CATCH: the blade rides the land and drops into the next notch
            done = 0
            for i in range(700):
                env.step(no_action)
                v = float(scene.cart.data.root_lin_vel_w[0].norm())
                done = done + 1 if (pq0() <= c.seat_tol and v < 0.03) else 0
                if done >= 25:
                    break
            st_now = station()
            print(f"[solve] cycle->{target} attempt {attempt}: seated at station "
                  f"{st_now} (q={q0():+.4f} pawl={pq0():+.4f})", flush=True)
            assert st_now <= k, "overshot the target station — metering broke"
            assert float(scene.foul_latch[0]) < 0.5, "foul latched — metering broke"
            if st_now > cur:
                cur = st_now
                advanced = True
                break
            # same-notch recatch: released too early — just try again
        assert advanced, f"failed to advance past station {cur} in 5 attempts"
        s = print_score(f"P{cur} arrived at station {cur}/{k}")
        assert s >= s_prev - 1e-6, "score decreased across a metering cycle"
        assert s >= 0.10 + 0.50 * cur / k - 1e-6, "arrival credit missing"
        s_prev = s

    # ---------------- ring-down to success --------------------------------------------------
    pawl_off()
    consec = 0
    for _ in range(2400):  # up to 20 s
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("ringdown")
    s_rd = print_score(f"P{k + 1} ring-down at the green post")
    assert s_rd >= s_prev - 1e-6, "score decreased across the ring-down"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after ring-down)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.3 simulated seconds, hands off) --------------------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                cv = float(scene.cart.data.root_lin_vel_w[0].norm())
                bv = float(scene.ball.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: q={q0():+.4f} "
                      f"pawl={pq0():+.4f} cart_v={cv:.4f} ball_v={bv:.4f} "
                      f"ball={bool(scene.ball_in_tray()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s_p = print_score(f"P{k + 2} persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_p >= s_rd - 1e-6
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
