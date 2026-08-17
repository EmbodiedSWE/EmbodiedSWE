"""Teleport solution for DropChuteScene (sim_gen task
`living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i275`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY: the MILK CARTON is teleported once, from its
floor spawn to LYING on the yellow loading shelf with its long axis pointed at the
slot, at rest — exactly what a pick-carry-and-lay-down of a free-standing 60 mm
carton delivers. Everything the rubric reads after that happens through contact
dynamics:

  push  — an applied fingertip-scale FORCE on the carton (a proxy for the gripper
          pressing its rear face: ~1-3 N, velocity-servoed with a hard cap, well
          inside a Franka fingertip's ability). The force is recomputed every step
          in the BIN frame (drive along -x into the slot, small lateral centering)
          and applied in the CARTON BODY frame — body-frame wrenches follow the
          body, sidestepping the pod's global-wrench frame drag (its reference
          orientation is captured at the FIRST application and goes stale across
          resets).
  door  — the carton's nose meets the BLUE one-way flap and pushes it INWARD off
          its gravity bias (~0.4 N at the contact); the flap rides on the carton's
          top face while it transits the doorway. Nothing ever holds the flap.
  drop  — past the inner sill edge the carton tips nose-down and falls to the bin
          floor (the push cuts off as soon as the carton is committed past the
          edge); the grippy interior parks it.
  shut  — the flap, CoM authored below its hinge, swings closed on its own behind
          the carton. Nothing is touched again.

The retry ladder escalates the servo (higher target speed / stiffer gain / higher
cap, capped at 5 N) and, if the carton parks straddling the sill, re-pushes with a
deeper cutoff; every retry rolls back to the pre-push snapshot first, so no failed
attempt's debris pollutes the next.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing:
0 -> 0.20 staged on the shelf -> >=0.80 carton deep inside -> 1.0 settled success),
then holds HANDS-OFF for >= 3.3 simulated seconds after success() first turns True
and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
    from .scene import MILK_W, SILL, _qapply, _qinv, _qmul, _qy  # noqa: F401
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import MILK_W, SILL, _qapply, _qinv, _qmul, _qy  # noqa: F401
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

STAGE_X = 0.290  # staging x on the shelf (bin frame; carton tail at 0.37, nose 0.21)
STAGE_Z = SILL + MILK_W / 2 + 0.004  # lying carton CoM, 4 mm settle drop
CUT_X = 0.150  # push cutoff: CoM 20 mm past the inner sill edge — committed to tip
CUT_Z = 0.100  # ... or already dropping below the sill
LAT_KP = 6.0  # lateral centering spring (N/m) + damping (N s/m), clamped
LAT_KD = 1.5
LAT_CAP = 1.2
DT = 1.0 / 120.0


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drop_chute")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
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

    def flap() -> float:
        return float(scene.flap_deg()[0])

    def milk_loc() -> torch.Tensor:
        return scene.bin_local(scene.milk.data.root_pos_w)[0]

    def clear_force() -> None:
        scene.milk.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def report(tag: str) -> None:
        ml = milk_loc()
        ju = scene.bin_local(scene.juice.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | flap={flap():+6.2f}deg "
              f"milk_bin=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"juice_bin=({float(ju[0]):+.3f},{float(ju[1]):+.3f}) "
              f"shelf={bool(scene.milk_on_shelf()[0])} "
              f"entered={bool(scene.milk_entered()[0])} "
              f"inside={bool(scene.milk_inside()[0])} "
              f"closed={bool(scene.flap_closed()[0])} "
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
            if bool(scene.settled(scene.milk)[0]) and bool(scene.settled(scene.juice)[0]) \
                    and bool(scene.settled(scene.flap)[0]):
                break

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled(360)
    bin_p = (scene.bin.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.bin.data.root_quat_w[0, 3]),
                           float(scene.bin.data.root_quat_w[0, 0]))
    ml = milk_loc()
    ju = scene.bin_local(scene.juice.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"bin=({float(bin_p[0]):+.3f},{float(bin_p[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.0f}deg "
          f"milk_bin=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
          f"juice_bin=({float(ju[0]):+.3f},{float(ju[1]):+.3f}) "
          f"flap={flap():+.2f}deg", flush=True)
    report("reset")
    assert abs(flap()) < 3.0, f"flap must hang shut at reset, got {flap():.2f} deg"
    assert not bool(scene.milk_inside()[0]), "milk must start outside the bin"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (flap shut, cartons on the floor, bin empty)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT ONLY — lay the carton on the shelf ----------------
    def stage_milk() -> None:
        """Place the carton at rest, LYING on the yellow shelf, long axis into the
        slot, centered on the slot's midline — exactly the end state of picking up
        the free-standing carton and laying it down. Zero velocity, hands off."""
        q_bin = scene.bin.data.root_quat_w
        local = torch.tensor([STAGE_X, 0.0, STAGE_Z], device=device).expand(n, 3)
        half_pi = torch.full((n,), math.pi / 2, device=device)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.bin.data.root_pos_w + _qapply(q_bin, local)
        st[:, 3:7] = _qmul(q_bin, _qy(half_pi))  # carton long axis along bin x
        scene.milk.write_root_state_to_sim(st, all_ids)

    stage_milk()
    wait_settled(240)
    report("staged")
    assert bool(scene.milk_on_shelf()[0]), "carton must lie staged on the shelf"
    s1 = print_score("P1 carton carried onto the loading shelf (teleport transport)")
    assert s1 >= s0 - 1e-6, "score decreased across the carry"
    assert s1 >= 0.20 - 1e-6, f"shelf credit missing, got {s1}"

    # ---------------- phase 2: push it through the one-way flap ----------------------------
    snap = scene.get_state(all_ids)

    def push(v_des: float, kp: float, cap: float, tag: str,
             cut_x: float = CUT_X, cut_z: float = CUT_Z, budget: int = 1500) -> str:
        """Fingertip push: a velocity-servoed force driving the carton along bin -x
        into the slot, with a small lateral centering term. Desired force is built
        in the BIN frame each step and applied in the CARTON BODY frame (frame-drag
        safe; follows the carton if it starts tipping). Cuts off the instant the
        carton is committed past the inner sill edge. Returns 'in' on cutoff,
        'stall' if x progress freezes."""
        q_bin = scene.bin.data.root_quat_w
        best_x = float(milk_loc()[0])
        last_gain_x, stall_ticks = best_x, 0
        for i in range(budget):
            loc = milk_loc()
            if float(loc[0]) < cut_x or float(loc[2]) < cut_z:
                clear_force()
                return "in"
            v_bin = _qapply(_qinv(q_bin), scene.milk.data.root_lin_vel_w)[0]
            v_along = -float(v_bin[0])  # +ve = moving INTO the slot
            f_push = min(max(kp * (v_des - v_along), 0.0), cap)
            f_lat = min(max(-LAT_KP * float(loc[1]) - LAT_KD * float(v_bin[1]),
                            -LAT_CAP), LAT_CAP)
            f_bin = torch.tensor([-f_push, f_lat, 0.0], device=device).expand(n, 3)
            f_world = _qapply(q_bin, f_bin)
            f_body = _qapply(_qinv(scene.milk.data.root_quat_w), f_world)
            scene.milk.set_external_force_and_torque(f_body.view(n, 1, 3), zero_wrench,
                                                     env_ids=all_ids)
            env.step(no_action)
            if i % 120 == 119:
                lo = milk_loc()
                print(f"[solve] {tag} @{i + 1}: f={f_push:4.2f}N "
                      f"milk_bin=({float(lo[0]):+.3f},{float(lo[1]):+.3f},"
                      f"{float(lo[2]):+.3f}) flap={flap():+6.2f}deg "
                      f"v={v_along:+.3f}", flush=True)
            x_now = float(milk_loc()[0])
            if x_now < best_x - 0.002:
                best_x, last_gain_x, stall_ticks = x_now, x_now, 0
            else:
                stall_ticks += 1
                if stall_ticks >= 300:  # 2.5 s with zero forward progress
                    clear_force()
                    print(f"[solve] {tag}: stalled at x={x_now:+.3f} "
                          f"(flap {flap():+.2f} deg)", flush=True)
                    return "stall"
        clear_force()
        print(f"[solve] {tag}: push budget exhausted (x={float(milk_loc()[0]):+.3f})",
              flush=True)
        return "stall"

    delivered = False
    for attempt, (v_des, kp, cap) in enumerate(
            ((0.12, 15.0, 2.0), (0.20, 25.0, 3.0), (0.30, 40.0, 5.0))):
        tag = f"push[a{attempt}/v{v_des:.2f}/cap{cap:.1f}]"
        res = push(v_des, kp, cap, tag)
        if res == "in":
            wait_settled(420)
            if bool(scene.milk_inside()[0]):
                delivered = True
            else:
                # parked straddling the sill / leaning inside: re-push with a
                # deeper cutoff — still the same fingertip through the doorway
                lo = milk_loc()
                print(f"[solve] {tag}: committed but parked at "
                      f"({float(lo[0]):+.3f},{float(lo[1]):+.3f},{float(lo[2]):+.3f})"
                      f" — deep re-push", flush=True)
                res2 = push(v_des, kp, cap, tag + "/deep", cut_x=0.100, cut_z=0.060,
                            budget=900)
                wait_settled(420)
                delivered = bool(scene.milk_inside()[0])
                if not delivered:
                    print(f"[solve] {tag}: deep re-push -> {res2}, still not inside",
                          flush=True)
        if delivered:
            print(f"[solve] {tag}: carton delivered through the flap "
                  f"(flap now {flap():+.2f} deg)", flush=True)
            break
        # roll back to the staged snapshot and escalate
        scene.set_state(snap, all_ids)
        step(60)
        assert bool(scene.milk_on_shelf()[0]), "rollback lost the staged carton"
    assert delivered, "push-through failed on every servo setting"
    report("delivered")
    assert bool(scene.milk_inside()[0]), "carton must rest inside the bin"
    s2 = print_score("P2 carton pushed through the one-way flap and dropped to the "
                     "bin floor (contact push + gravity, door never held)")
    assert s2 >= s1 - 1e-6, "score decreased across the push"
    assert s2 >= 0.80 - 1e-6, f"deep-inside credit missing, got {s2}"

    # ---------------- phase 3: the flap swings shut on its own, all settle -----------------
    # If the carton parked STANDING right under the flap it props the door open (a
    # standing carton is taller than the closed flap's bottom edge). The doorway is
    # then wide open, so a fingertip reaches through the slot and nudges the carton —
    # it topples to lying (topple force ~1.2 N < slide force on the grippy floor) and
    # the flap is free. Same push servo, same fingertip force, deeper cutoff.
    for k in range(4):
        for _ in range(12):  # up to 3 s hands-off close + settle
            step(30)
            if bool(scene.flap_closed()[0]):
                break
        if bool(scene.flap_closed()[0]):
            break
        print(f"[solve] flap propped at {flap():+.2f} deg by the parked carton — "
              f"clear-nudge {k}", flush=True)
        push(0.10, 15.0, 2.0, f"clear[{k}]", cut_x=-0.040, cut_z=-1.0, budget=240)
        wait_settled(300)
        assert bool(scene.milk_inside()[0]), "clear-nudge pushed the carton out of window"
    for _ in range(8):  # settle out to full success
        step(30)
        if bool(scene.success()[0]):
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
    report("shut")
    assert bool(scene.flap_closed()[0]), \
        f"flap must gravity-close behind the carton, got {flap():.2f} deg"
    s3 = print_score("P3 flap swung shut on its own — milk inside, everything at rest")
    assert s3 >= s2 - 1e-6, "score decreased across the close"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after close)", flush=True)
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
                print(f"[solve] persist flicker @step {i}: "
                      f"inside={bool(scene.milk_inside()[0])} "
                      f"closed={bool(scene.flap_closed()[0])} "
                      f"milk_lin={mv:.4f} "
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
