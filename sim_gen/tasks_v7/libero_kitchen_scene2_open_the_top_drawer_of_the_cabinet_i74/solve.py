"""Teleport solution for FoldCollarScene (sim_gen task
`libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i74`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport = transport only): one joint-consistent root-state write per
   leaf carries the whole articulated collar — PRESERVING its current fold angles —
   from its spawn to the BLUE column, base leaf parked a finger's width from the
   column with the hinged pocket facing it, 3 mm hover, zero velocity. The write
   satisfies nothing: the wings are still nearly straight, the free-edge gap is
   ~3 leaf-widths, `success()` is False and only the `approach` latch arms.
2. FOLDING (applied torque through the hinges): each wing is swung shut by a pure
   world-z torque on the wing body — the exact wrench of a palm pushing the leaf's
   outer face — velocity-regulated (~0.9 rad/s cruise, slowed near the target),
   BOTH wings driven simultaneously so their reactions on the base cancel. The
   torque is released just short of the target and the strongly-damped hinge plus
   ground friction park the leaf. The fold angle is judged from the resulting BODY
   poses. A pure z torque is invariant under the pod force-frame drag quirk (the
   drag rotates wrenches by the body's z-rotation since reset, which fixes the z
   axis). Stalls escalate the torque cap; if a wing cannot fold the run FAILS (it
   does not teleport the wing).
3. GAP TRIM (same torque channel): if the settled free-edge gap is still above the
   bound, the fold targets are deepened a few degrees and the same servo re-runs —
   closure is always produced by hinge travel against contact, never by a pose write.
4. ORDER (physically inherent): the collar must be AT the column before folding —
   folding first would close the pocket and the column could no longer enter it
   laterally (the gap bound is below the column diameter by construction).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet_i74.solve --headless [--seed N]
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
    from .scene import _qapply, _qz
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, _qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.fold_collar")().build(num_envs=args.num_envs, device=device)
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

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def folds_deg() -> tuple[float, float]:
        fl, fr = scene._folds()
        return float(torch.rad2deg(fl[0])), float(torch.rad2deg(fr[0]))

    def report(tag: str) -> None:
        fl, fr = folds_deg()
        gap = float(scene._gap()[0])
        print(f"[solve] {tag:12s} | folds=({fl:+6.1f},{fr:+6.1f})deg "
              f"gap={gap * 1000:6.1f}mm inside={bool(scene._blue_inside()[0])} "
              f"upright={bool(scene._upright_standing()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(150)
    base_p = (scene.base.data.root_pos_w - scene.env_origins)[0]
    qb = scene.base.data.root_quat_w[0]
    yaw0 = float(torch.rad2deg(2.0 * torch.atan2(qb[3], qb[0])))
    blue_xy = (scene.blue.data.root_pos_w - scene.env_origins)[0, :2]
    red_xy = (scene.red.data.root_pos_w - scene.env_origins)[0, :2]
    fl0, fr0 = folds_deg()
    slot = "slotA" if float(scene.blue_slot[0]) > 0 else "slotB"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"collar=({float(base_p[0]):+.3f},{float(base_p[1]):+.3f}) yaw={yaw0:+.1f}deg "
          f"folds=({fl0:+.1f},{fr0:+.1f})deg "
          f"blue=({float(blue_xy[0]):+.3f},{float(blue_xy[1]):+.3f}) [{slot}] "
          f"red=({float(red_xy[0]):+.3f},{float(red_xy[1]):+.3f})", flush=True)
    report("reset")
    assert 2.0 < fl0 < 34.0 and 2.0 < fr0 < 34.0, "wings must start in the zigzag window"
    assert bool(scene._upright_standing()[0]), "collar must stand upright after settle"
    s0 = print_score("P0 reset+settle (collar zigzagged, far from the columns)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: transport the collar to the BLUE column ---------------------
    # Rigid carry of the whole chain, fold angles preserved, pocket (+y) facing the
    # column; park the base leaf's inner face a finger's width from the column.
    fl_r, fr_r = scene._folds()
    d_park = c.col_r + c.panel_t / 2 + 0.008
    blue_w = scene.blue.data.root_pos_w[:, :2]
    psi = torch.zeros(n, device=device)  # carry heading: pocket faces world +y
    q_new = _qz(psi)
    base_new = torch.zeros(n, 3, device=device)
    base_new[:, 0] = blue_w[:, 0]
    base_new[:, 1] = blue_w[:, 1] - d_park
    base_new[:, 2] = scene.env_origins[:, 2] + 0.003
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = base_new
    st[:, 3:7] = q_new
    scene.base.write_root_state_to_sim(st, all_ids)
    for body, side, fold in ((scene.wing_l, -1.0, fl_r), (scene.wing_r, 1.0, fr_r)):
        hinge = torch.zeros(n, 3, device=device)
        hinge[:, 0] = side * c.panel_w / 2
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = base_new + _qapply(q_new, hinge)
        st[:, 3:7] = torch.stack([torch.cos(side * fold / 2),
                                  torch.zeros(n, device=device),
                                  torch.zeros(n, device=device),
                                  torch.sin(side * fold / 2)], dim=-1)
        body.write_root_state_to_sim(st, all_ids)
    step(90)  # drop the 3 mm hover, settle on the ground, hands-off
    report("transport")
    assert bool(scene._upright_standing()[0]), "collar must stand after transport"
    s1 = print_score("P1 collar carried to the blue column (still open)")
    assert s1 >= c.w_appr - 1e-6, f"P1 score {s1} (expect approach latch = {c.w_appr})"
    assert not bool(scene.success()[0]), "cannot be success with the collar open"

    # ---------------- phase 2: fold both wings shut (applied z torque) ---------------------
    # Gain sizing: wing inertia about its hinge is ~m*W^2/3 = 1.9e-3 kg*m^2, so
    # discrete stability at 120 Hz needs k/I * dt < 2 -> k < ~0.45. k = 0.12 cruises
    # smoothly; the cap escalates only if a real obstruction resists.
    def drive_folds(target_deg: float, max_steps: int, tau_cap0: float = 0.15) -> bool:
        # Stalls escalate BOTH the torque cap and the servo GAIN: near the target the
        # commanded speed is small, so a stationary wing only sees k*w_des of torque —
        # the gain (not the cap) is what must beat ground friction. Stability at
        # 120 Hz needs k < 2*I/dt ~ 0.46 (wing inertia ~1.9e-3 kg*m^2).
        caps = {"l": tau_cap0, "r": tau_cap0}
        gains = {"l": 0.12, "r": 0.12}
        done = {"l": False, "r": False}
        win = {"l": (0, folds_deg()[0]), "r": (0, folds_deg()[1])}
        for i in range(max_steps):
            fl, fr = folds_deg()
            cur = {"l": fl, "r": fr}
            for key, body, side in (("l", scene.wing_l, -1.0),
                                    ("r", scene.wing_r, 1.0)):
                if done[key]:
                    continue
                if cur[key] >= target_deg - 2.0:
                    clear_wrench(body)
                    done[key] = True
                    print(f"[solve] wing-{key}: released at {cur[key]:+.1f} deg",
                          flush=True)
                    continue
                w = float(body.data.root_ang_vel_w[0, 2])
                w_des = side * (0.9 if cur[key] < target_deg - 25.0 else 0.35)
                tau = max(-caps[key], min(caps[key], gains[key] * (w_des - w)))
                t_world = torch.zeros(n, 1, 3, device=device)
                t_world[0, 0, 2] = tau
                body.set_external_force_and_torque(zero_wrench, t_world,
                                                   env_ids=all_ids, is_global=True)
                wi, wth = win[key]
                if i - wi >= 90:  # stall probe every 0.75 s
                    if cur[key] < wth + 3.0:
                        caps[key] = min(caps[key] + 0.10, 0.45)
                        gains[key] = min(gains[key] + 0.10, 0.42)
                        print(f"[solve] wing-{key}: stalled at {cur[key]:+.1f} deg; "
                              f"cap -> {caps[key]:.2f} N*m gain -> {gains[key]:.2f}",
                              flush=True)
                    win[key] = (i, cur[key])
            if done["l"] and done["r"]:
                return True
            env.step(no_action)
        clear_wrench(scene.wing_l)
        clear_wrench(scene.wing_r)
        return done["l"] and done["r"]

    target = 112.0
    ok = drive_folds(target, 1500)
    step(90)  # damped-hinge park + settle, hands-off
    report("fold")
    if not ok:
        print("SIM_GEN_SOLVE: FAIL (a wing never reached the fold target)", flush=True)
        os._exit(1)
    s2 = print_score("P2 both wings folded around the column")
    assert s2 >= s1 - 1e-6 and s2 >= 0.70 - 1e-6, f"P2 score {s2} (expect cap 0.70)"

    # ---------------- phase 3: trim the free-edge gap under the bound ----------------------
    for round_ in range(3):
        gap = float(scene._gap()[0])
        if gap < c.gap_max - 0.004 and bool(scene.success()[0]):
            break
        target = min(target + 5.0, c.fold_limit_deg - 3.0)
        print(f"[solve] gap {gap * 1000:.1f} mm still above the bound; "
              f"deepening folds to {target:.0f} deg (round {round_})", flush=True)
        drive_folds(target, 600, tau_cap0=0.25)
        step(120)  # settle, hands-off
    report("trim")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (final state not successful)", flush=True)
        os._exit(1)
    s3 = print_score("P3 collar closed behind the column")
    assert s3 >= s2 - 1e-6, "score decreased across the trim phase"
    assert s3 >= 1.0 - 1e-6, f"success must score 1.0, got {s3}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
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
    main()
