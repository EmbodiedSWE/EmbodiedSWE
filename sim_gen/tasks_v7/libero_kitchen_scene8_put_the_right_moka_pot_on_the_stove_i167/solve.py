"""Teleport solution for StovePropScene (sim_gen task
`libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i167`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. LID OPENING (external wrench = the hand): a PD + gravity-feedforward torque
   about the hinge axis swings the lid open and holds it at ~68 degrees —
   exactly the wrench a gripper on the handle applies. The torque is about the
   lid's own hinge axis (body y == world y at every angle), so it is invariant
   under the forge pod's wrench frame drag.
2. TRANSPORT (teleport): single root-state writes carry ONE object at a time
   across FREE SPACE with zero velocity — exactly the carry a gripper performs.
   The rod is set with its base hovering just above the socket well and its
   tip aimed into the (held-open) lid's capture pocket; the pot is set at a
   hover a few millimetres above the burner pad, through the propped-open
   mouth. Nothing is ever written into contact.
3. BRACING (gravity + contact + the mechanism): the rod FALLS its last
   millimetres into the well; then the wrench RAMPS OFF and the freed lid
   swings down ~5 degrees onto the rod tip — the pocket walls capture it and
   the two-point brace (well + pocket) carries the lid from then on. Nothing
   about the lid is written after reset: whether the brace holds is 100 %
   contact dynamics.
4. The pot placement is only possible NOW: the drop corridor over the burner
   pad exists only under a braced-open lid (a closed lid seals the tub —
   asserted at P0).

If the brace fails (rod kicked out on release), the attempt is retried from a
re-opened lid — a retry, not a cheat: the final configuration is still 100 %
contact-made and the final 3.3 s are fully hands-off.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i167.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401 - registers the scene
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

HOLD_DEG = 68.0     # wrench hold angle (a few degrees above the braced angle)
G = 9.81


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stove_prop")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    zero3 = torch.zeros(n, 1, 3, device=device)
    r_com = c.lid_len / 2

    def lid_wrench(theta_des: float, scale: float = 1.0, kp: float = 2.0,
                   kd: float = 0.35) -> None:
        """One step of the lid servo: gravity feedforward + PD about the hinge
        axis (body y == world y — frame-drag invariant), then step."""
        th = scene.lid_angle()
        om_open = -scene.lid.data.root_ang_vel_w[:, 1]  # opening rate
        tau_open = (c.lid_mass * G * r_com * torch.cos(th)
                    + kp * (theta_des - th) - kd * om_open).clamp(-0.2, 1.5)
        t3 = zero3.clone()
        t3[:, 0, 1] = -tau_open * scale  # +y torque closes; open = negative
        scene.lid.set_external_force_and_torque(zero3, t3)
        env.step(no_action)

    def drop_wrench() -> None:
        scene.lid.set_external_force_and_torque(zero3, zero3)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        ang = math.degrees(float(scene.lid_angle()[0]))
        print(f"[solve] {tag:12s} | lid={ang:+6.2f}deg "
              f"tip_in_pocket={bool(scene.tip_in_pocket()[0])} "
              f"base_in_well={bool(scene.base_in_well()[0])} "
              f"propped={bool(scene.propped()[0])} "
              f"pot_on_burner={bool(scene.pot_on_burner()[0])} "
              f"wrong_in_tub={bool(scene.wrong_in_tub()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"latches=(o={bool(scene._l_open[0])},b={bool(scene._l_prop[0])},"
              f"p={bool(scene._l_pot[0])}) "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    last_printed = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= last_printed[0] - 1e-6, \
            f"score regressed: {last_printed[0]} -> {s}"
        last_printed[0] = s
        return s

    def write_pose(body, pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def open_lid(steps_ramp: int = 360, steps_hold: int = 240) -> None:
        """Servo the lid from wherever it is to HOLD_DEG and hold it there."""
        th0 = float(scene.lid_angle()[0])
        tgt = math.radians(HOLD_DEG)
        for i in range(steps_ramp):
            frac = (i + 1) / steps_ramp
            lid_wrench(th0 + (tgt - th0) * frac)
        for _ in range(steps_hold):
            lid_wrench(tgt)

    def place_rod() -> None:
        """Transport the rod: base hovering 1 mm above the well floor, axis
        aimed at a point just inside the (held-open) pocket cavity."""
        tp = scene.tub.data.root_pos_w
        base = tp.clone()
        base[:, 0] += c.well_x
        base[:, 1] += c.well_y
        base[:, 2] += c.floor_t + 0.001
        pocket_local = torch.tensor([c.pocket_r, c.pocket_y, -0.006],
                                    device=device).expand(n, 3)
        tip_tgt = scene.lid.data.root_pos_w + quat_apply(
            scene.lid.data.root_quat_w, pocket_local)
        d = tip_tgt - base
        d = d / d.norm(dim=-1, keepdim=True)
        # quaternion rotating local +z onto d (axis = ez x d, half-angle form)
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
        v = torch.cross(ez, d, dim=-1)
        w = 1.0 + d[:, 2:3]
        q = torch.cat([w, v], dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        write_pose(scene.rod, base, q)

    def hover_pot() -> None:
        """Transport the copper pot to a hover 8 mm above the burner pad,
        through the propped-open mouth (vertical drop corridor asserted)."""
        tp = scene.tub.data.root_pos_w
        pos = tp.clone()
        pos[:, 0] += c.pad_pos[0]
        pos[:, 1] += c.pad_pos[1]
        pos[:, 2] += c.floor_t + c.pad_h + 0.008
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        write_pose(scene.pot_t, pos, quat)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # everything seats (lid closed on the walls, rod/pots on the deck)
    report("reset")
    assert torch.isfinite(scene.lid.data.root_pos_w).all(), "NaN/inf after settle"
    ang0 = math.degrees(float(scene.lid_angle()[0]))
    assert abs(ang0) < 3.0, f"lid must start CLOSED (angle {ang0:.1f} deg)"
    assert not bool(scene.wrong_in_tub()[0]) and not bool(scene.pot_on_burner()[0]), \
        "pots must start outside the tub"
    s0 = print_score("P0 reset+settle (lid sealed shut, rod and pots on the counter)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: wrench the lid open -----------------------------------------
    open_lid()
    report("lid-held")
    ang1 = math.degrees(float(scene.lid_angle()[0]))
    assert ang1 > HOLD_DEG - 6.0, f"servo failed to open the lid (at {ang1:.1f} deg)"
    assert bool(scene._l_open[0]), "open latch must fire while the lid is held open"
    s1 = print_score("P1 lid held open by the wrench (the hand)")
    assert s1 >= 0.19, f"P1 score {s1} (expect open latch 0.20)"

    # ---------------- phases 2+3: brace the lid on the rod ---------------------------------
    braced = False
    for attempt in range(3):
        place_rod()
        for _ in range(240):  # rod falls 1 mm, seats in the well, tip in the pocket
            lid_wrench(math.radians(HOLD_DEG))
        report("rod-placed")
        if not (bool(scene.base_in_well()[0]) and bool(scene.tip_in_pocket()[0])):
            print(f"[solve] rod placement missed (attempt {attempt + 1}); retrying",
                  flush=True)
            continue
        # lower the lid QUASI-STATICALLY onto the rod tip (a free release from
        # the hold angle arrives at ~0.4 m/s and kicks the rod out): servo the
        # target angle down to just above the fore-wall CATCH angle — the
        # loaded tip slides fore along the panel until the deep fore fence
        # arrests it (the statically stable rest) — then bleed the wrench off
        th_now = float(scene.lid_angle()[0])
        tgt_low = c.catch_angle + math.radians(0.5)
        for i in range(360):
            frac = (i + 1) / 360
            lid_wrench(th_now + (tgt_low - th_now) * frac)
        report("lowered")
        for i in range(360):
            lid_wrench(tgt_low, scale=1.0 - (i + 1) / 360)
        drop_wrench()
        step(360)
        report("released")
        if bool(scene.propped()[0]):
            braced = True
            break
        print(f"[solve] brace failed on release (attempt {attempt + 1}); "
              f"re-opening the lid", flush=True)
        open_lid(steps_ramp=240, steps_hold=120)
    assert braced, "the rod brace never held the lid open hands-free"
    assert bool(scene._l_prop[0]), "propped latch must be set"
    ang2 = math.degrees(float(scene.lid_angle()[0]))
    th_c = math.degrees(c.catch_angle)
    assert th_c - 3.0 < ang2 < math.degrees(c.prop_angle) + 2.0, \
        f"braced angle {ang2:.1f} deg should rest between the catch ({th_c:.1f}) " \
        f"and the plumb design angle"
    s2 = print_score("P2+P3 hands off the lid — the rod brace carries it")
    assert s2 >= 0.49, f"brace score {s2} (expect 0.20 + 0.30)"

    # ---------------- phase 4: copper pot onto the burner ----------------------------------
    hover_pot()
    step(360)  # falls 8 mm, seats upright on the pad
    report("pot-set")
    assert bool(scene.pot_on_burner()[0]), "copper pot must rest upright on the pad"
    # wait for success to hold CONTINUOUSLY for 2 simulated seconds so the phase
    # boundary marks a genuinely settled state, not a transient
    streak = 0
    ok = False
    for _ in range(1200):
        env.step(no_action)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 240:
            ok = True
            break
    report("stable")
    assert ok and bool(scene.success()[0]), "success did not stabilize"
    s3 = print_score("P4 copper pot on the burner under the braced lid — success")
    assert s3 >= 0.99, f"success must score 1.0, got {s3}"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P5 persistence 3.3 s hands-off")
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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
