"""Teleport solution for DumpHopperScene (sim_gen task
`living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i177`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY: the BASKET is teleported once, from its floor
spawn to upright on the GREEN catch mark under the hopper's spout, with zero velocity
— exactly what a pick-and-carry of the empty basket delivers. The milk carton is
NEVER teleported and never touched: everything the rubric reads happens through the
mechanism and contact dynamics afterwards:

  press   — an applied PURE TORQUE about the hinge axis (a proxy for the gripper
            pressing the YELLOW lever paddle down: the paddle tip is 0.27 m from the
            hinge, so the held torque of ~2 N*m is a ~7 N fingertip press, and the
            wrench has no force component because a paddle press is a pure moment
            about the hinge as far as the tray body is concerned). The torque ramps
            up under a tilt-rate cap (bang-bang: high torque below the cap, near-zero
            above it) so the tray tips smoothly to its 35 deg joint stop instead of
            slamming, and HOLDS against the stop.
  slide   — at the stop, gravity beats the slick plate's friction cone
            (tan 35 deg = 0.70 vs mu ~ 0.225): the carton slides out through the
            discharge slot, pivots off the spout edge, and falls.
  catch   — the carton drops past the basket rim and lands inside; the high-grip
            basket interior kills the slide-out.
  release — the torque is cleared; the back-heavy tray swings back to level on its
            own (authored CoM behind the hinge). Nothing is touched again.

Force-frame note: some pods rotate applied wrenches by the body's rotation since a
reference orientation (applied = R_now * R_ref^T * arg). The lever torque here is
ALONG the hinge axis and the tray only ever rotates ABOUT that axis, so the drag
rotation leaves the torque vector INVARIANT — no mode probe is needed. The retry
ladder instead escalates the torque cap (and re-aims the basket if a discharge
misses, by rolling back to the pre-press snapshot and correcting the placement from
the observed landing point — a solver-side retry; the basket is legitimately
carriable).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing:
0 -> 0.15 basket on the mark -> >=0.70 carton discharged into the basket -> 1.0
settled success), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
    from .scene import BK_PLATE_T, TILT_MAX_DEG, _qapply, _qmul, _qz  # noqa: F401
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import BK_PLATE_T, TILT_MAX_DEG, _qapply, _qmul, _qz  # noqa: F401
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

BASKET_X0 = 0.325  # first basket target x (station frame; predicted landing ~0.30)
RATE_CAP = 0.33  # tilt-rate cap during the press, deg per step (~40 deg/s at 120 Hz)
TAU_LO = 0.25  # below-bias torque while over the rate cap (tray decelerates)
DT = 1.0 / 120.0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.dump_hopper")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def tilt() -> float:
        return float(scene.tilt_deg()[0])

    def milk_stn() -> torch.Tensor:
        return scene.stn_local(scene.milk.data.root_pos_w)[0]

    def clear_torque() -> None:
        scene.tray.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def report(tag: str) -> None:
        ms = milk_stn()
        bs = scene.stn_local(scene.basket.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | tilt={tilt():+6.2f}deg "
              f"milk_stn=({float(ms[0]):+.3f},{float(ms[1]):+.3f},{float(ms[2]):+.3f}) "
              f"basket_stn=({float(bs[0]):+.3f},{float(bs[1]):+.3f}) "
              f"caged={bool(scene.milk_in_cage()[0])} "
              f"released={bool(scene.milk_released()[0])} "
              f"in_basket={bool(scene.in_basket(scene.milk)[0])} "
              f"zone={bool(scene.basket_in_zone()[0])} "
              f"decoy_out={bool(scene.decoy_out()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wait_settled(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled(scene.milk)[0]) and bool(scene.settled(scene.basket)[0]) \
                    and bool(scene.settled(scene.tray)[0]):
                break

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled(360)
    stn_p = (scene.station.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.station.data.root_quat_w[0, 3]),
                           float(scene.station.data.root_quat_w[0, 0]))
    bk_p = scene.stn_local(scene.basket.data.root_pos_w)[0]
    ju_p = scene.stn_local(scene.juice.data.root_pos_w)[0]
    ml_t = scene.tray_local(scene.milk.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"station=({float(stn_p[0]):+.3f},{float(stn_p[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.0f}deg "
          f"milk_tray=({float(ml_t[0]):+.3f},{float(ml_t[1]):+.3f},{float(ml_t[2]):+.3f}) "
          f"basket_stn=({float(bk_p[0]):+.3f},{float(bk_p[1]):+.3f}) "
          f"juice_stn=({float(ju_p[0]):+.3f},{float(ju_p[1]):+.3f}) "
          f"tilt={tilt():+.2f}deg", flush=True)
    report("reset")
    assert abs(tilt()) < 2.0, f"tray must rest level, got {tilt():.2f} deg"
    assert bool(scene.milk_in_cage()[0]), "milk must start caged"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (tray level, milk caged, basket far)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT ONLY — carry the basket to the mark ---------------
    def teleport_basket(target_x: float, target_y: float) -> None:
        """Place the basket at rest, upright, on the catch mark: station-frame
        (target_x, target_y), station heading, floor height. Zero velocity, hands off
        — exactly the end state of carrying the empty basket by its rim."""
        q_stn = scene.station.data.root_quat_w
        local = torch.tensor([target_x, target_y, BK_PLATE_T + 0.003],
                             device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.station.data.root_pos_w + _qapply(q_stn, local)
        st[:, 3:7] = q_stn
        scene.basket.write_root_state_to_sim(st, all_ids)

    bx, by = BASKET_X0, 0.0
    teleport_basket(bx, by)
    wait_settled(240)
    report("placed")
    assert bool(scene.basket_in_zone()[0]), "basket must sit in the catch zone"
    s1 = print_score("P1 basket carried onto the catch mark (teleport transport)")
    assert s1 >= s0 - 1e-6, "score decreased across the carry"
    assert s1 >= 0.15 - 1e-6, f"zone credit missing, got {s1}"

    # ---------------- phase 2: press the lever — tip, slide, catch -------------------------
    snap = scene.get_state(all_ids)

    def press(tau_max: float, tag: str, frame: str = "body") -> str:
        """Press-and-hold: pure torque about the hinge axis, bang-bang under a
        tilt-rate cap (finite-difference rate — ang-vel readback is phantom under
        external wrenches), ramping to tau_max and holding against the 35 deg stop
        while the carton slides out. The torque is applied in the TRAY BODY frame
        (0, tau, 0) — the hinge axis is body-y by construction, at every tilt and
        every station yaw, which sidesteps the pod's global-wrench frame drag (its
        reference orientation is captured at the FIRST application and goes stale
        across resets); `frame="world"` is the station-frame fallback. Returns
        'caught' once the carton is released AND inside the basket for 20
        consecutive steps, 'missed' if it lands anywhere else, 'stall' on no tilt
        progress."""
        th_prev = tilt()
        in_streak = 0
        for i in range(2400):
            th = tilt()
            w = (th - th_prev) / 1.0  # deg per step
            th_prev = th
            tau_hi = min(0.4 + tau_max * (i / 300.0), tau_max)
            tau = TAU_LO if w > RATE_CAP else tau_hi
            t_vec = torch.tensor([0.0, tau, 0.0], device=device).expand(n, 3)
            if frame == "body":
                scene.tray.set_external_force_and_torque(
                    zero_wrench, t_vec.view(n, 1, 3), env_ids=all_ids)
            else:
                t_world = _qapply(scene.station.data.root_quat_w, t_vec)
                scene.tray.set_external_force_and_torque(
                    zero_wrench, t_world.view(n, 1, 3), env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i % 120 == 119:
                ms = milk_stn()
                print(f"[solve] {tag} @{i + 1}: tau={tau:4.2f}Nm tilt={tilt():+6.2f}deg "
                      f"milk_stn=({float(ms[0]):+.3f},{float(ms[1]):+.3f},"
                      f"{float(ms[2]):+.3f}) rel={bool(scene.milk_released()[0])} "
                      f"inb={bool(scene.in_basket(scene.milk)[0])}", flush=True)
            # stall probe: the tray never started tipping
            if i == 479 and tilt() < 5.0:
                clear_torque()
                print(f"[solve] {tag}: no tilt progress after 4 s "
                      f"(tilt {tilt():.2f} deg) — torque too low", flush=True)
                return "stall"
            if bool(scene.milk_released()[0]) and bool(scene.in_basket(scene.milk)[0]):
                in_streak += 1
                if in_streak >= 20:
                    clear_torque()
                    return "caught"
            else:
                in_streak = 0
            # miss probe: carton released, low, slow, and NOT in the basket
            if bool(scene.milk_released()[0]) \
                    and not bool(scene.in_basket(scene.milk)[0]):
                mz = float((scene.milk.data.root_pos_w - scene.env_origins)[0, 2])
                mv = float(scene.milk.data.root_lin_vel_w[0].norm())
                if mz < 0.10 and mv < 0.10:
                    clear_torque()
                    return "missed"
        clear_torque()
        print(f"[solve] {tag}: press budget exhausted (tilt {tilt():.2f} deg)",
              flush=True)
        return "stall"

    outcome = ""
    for attempt, (tau_max, frame) in enumerate(
            ((2.2, "body"), (2.2, "world"), (3.2, "body"), (4.5, "world"))):
        tag = f"press[a{attempt}/{frame}/tau{tau_max:.1f}]"
        outcome = press(tau_max, tag, frame=frame)
        if outcome == "caught":
            print(f"[solve] {tag}: carton discharged into the basket "
                  f"(tilt held {tilt():+.2f} deg)", flush=True)
            break
        if outcome == "missed":
            ms = milk_stn()
            miss_dx = float(ms[0]) - bx
            miss_dy = float(ms[1]) - by
            bx = min(max(bx + miss_dx, c.zone_x_win[0] + 0.020), c.zone_x_win[1] - 0.020)
            by = min(max(by + miss_dy, -c.zone_y_max + 0.010), c.zone_y_max - 0.010)
            print(f"[solve] {tag}: MISSED — carton landed at station "
                  f"({float(ms[0]):+.3f},{float(ms[1]):+.3f}); re-aiming basket to "
                  f"({bx:+.3f},{by:+.3f}) and rolling back", flush=True)
        scene.set_state(snap, all_ids)
        step(60)  # re-settle the rolled-back state
        teleport_basket(bx, by)
        wait_settled(240)
        assert bool(scene.basket_in_zone()[0]), "re-aimed basket left the catch zone"
    assert outcome == "caught", f"discharge failed on every attempt (last: {outcome})"
    report("caught")
    assert bool(scene.milk_released()[0]), "carton must be out of the cage"
    assert bool(scene.in_basket(scene.milk)[0]), "carton must be inside the basket"
    s2 = print_score("P2 lever pressed and held — carton slid out and dropped into "
                     "the basket (mechanism + gravity, milk never touched)")
    assert s2 >= s1 - 1e-6, "score decreased across the discharge"
    assert s2 >= 0.70 - 1e-6, f"in-basket credit missing, got {s2}"

    # ---------------- phase 3: release — the tray returns level, all settle ----------------
    for _ in range(20):  # up to 5 s hands-off return + settle
        step(30)
        if tilt() < 3.0 and bool(scene.success()[0]):
            break
    # Streak-gate the settle: a single instantaneous success() reading can land at a
    # velocity turning point while the carton is still creeping — require it to HOLD
    # for 60 consecutive steps (0.5 s) before starting the persistence clock.
    streak = 0
    for _ in range(1200):
        step(1)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 60:
            break
    report("released")
    assert tilt() < 3.0, f"tray must gravity-return to level, got {tilt():.2f} deg"
    s3 = print_score("P3 lever released — back-heavy tray returned level on its own, "
                     "everything at rest")
    assert s3 >= s2 - 1e-6, "score decreased across the release"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after release)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3.3 simulated seconds, no intervention) --------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                mv = float(scene.milk.data.root_lin_vel_w[0].norm())
                bv = float(scene.basket.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: "
                      f"inb={bool(scene.in_basket(scene.milk)[0])} "
                      f"upright={bool(scene.basket_upright_on_floor()[0])} "
                      f"milk_lin={mv:.4f} basket_lin={bv:.4f} "
                      f"decoy={bool(scene.decoy_out()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s (hands off)")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
