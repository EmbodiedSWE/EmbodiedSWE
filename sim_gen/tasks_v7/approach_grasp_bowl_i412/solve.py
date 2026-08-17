"""Scripted solution for ChimneyCatchScene (sim_gen task approach_grasp_bowl_i412) —
the task's legitimacy certificate.

This solve uses NO teleports at all: every load-bearing interaction is force + contact,
exactly the pushes a Franka arm would exert.

1. COVER (forces + contact): the live tower is READ BACK per episode (which chimney the
   ball spawned on — visible in the real task by looking down the open chute tops). The
   bowl is slid across the deck by a horizontal velocity-servo force applied at its CoM
   (which sits at the base-disc centre — bottom-heavy, so the push cannot tip it): it is
   staged square-on beside the live slot, then pushed straight across the open hole so
   the flange meets the slot edges squarely (a stall against a slot edge escalates a
   breakaway bias that resets the moment the bowl moves). It parks covering the live
   floor slot directly under the chute, then brakes. The `covered` latch can only set
   here.
2. TRIGGER (forces + contact): a gentle +y velocity-servo force (with a friction
   feed-forward bias) pushes the live gate blade outboard along its table, exactly the
   knob push the arm would make. The blade slides out from under the ball; the chute
   walls guide the free-falling ball straight down into the covered bowl. The `caught`
   latch can only set here — gravity through the machine does the transfer.
3. DELIVER (forces + contact): the loaded bowl is servo-pushed out from under the chute
   (straight -y first, then to the dock disc centre, read back per episode) and braked.
4. HANDS OFF: all forces cleared; the bowl settles docked with the ball resting inside.
   Every success() clause (ball in bowl, bowl docked upright, settled, never lost) is a
   live physical pose produced by contact and gravity.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
partial credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.approach_grasp_bowl_i412.solve --headless [--seed N]
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

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

_ = scene_mod  # imported for its registrations


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chimney_catch")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero1 = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def bowl_p() -> torch.Tensor:
        return scene.bowl.data.root_pos_w - scene.env_origins

    def ball_p() -> torch.Tensor:
        return scene.ball.data.root_pos_w - scene.env_origins

    def report(tag: str) -> None:
        wp, bp = bowl_p()[0], ball_p()[0]
        vb = float(scene.bowl.data.root_lin_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:14s} | bowl=({float(wp[0]):+.3f},{float(wp[1]):+.3f},"
              f"{float(wp[2]):.3f}) v={vb:.3f} ball=({float(bp[0]):+.3f},"
              f"{float(bp[1]):+.3f},{float(bp[2]):.3f}) "
              f"covered={float(scene.covered[0]):.0f} caught={float(scene.caught[0]):.0f} "
              f"lost={bool(scene.lost[0])} in_bowl={bool(scene._in_bowl()[0])} "
              f"docked={bool(scene._docked()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_forces() -> None:
        scene.bowl.set_external_force_and_torque(zero1, zero1, env_ids=all_ids)
        for b in scene.blades.values():
            b.set_external_force_and_torque(zero1, zero1, env_ids=all_ids)

    def push_bowl(target_xy: torch.Tensor, *, mass: float, vmax: float, tol: float,
                  max_steps: int, tag: str) -> bool:
        """Slide the bowl to `target_xy` (env frame) with a horizontal velocity-servo
        force at the CoM, then brake. The CoM sits at the base centre, so the push
        cannot tip the bowl. The deck has REAL slot holes: the flat base flange can
        dip into one and catch on its far vertical edge (sill-step catch), so a stall
        (low speed while far from target) escalates a breakaway bias along the error
        direction — reset the moment the bowl moves again, so it can never fling.
        Returns True when parked within `tol`."""
        kp, kv, fmax = 3.0, 40.0, 8.0
        # Dry-friction feed-forward: the deck/bowl pair reads mu_eff ~0.5 in situ
        # (patch friction runs well above the authored materials), so the distance-
        # scaled servo alone parks short. Tapered near the target so it cannot hunt.
        ff0 = 0.55 * mass * 9.81
        stall, bias = 0, 0.0
        for i in range(max_steps):
            p = bowl_p()
            err = target_xy - p[:, :2]
            d = float(err.norm(dim=-1)[0])
            v = scene.bowl.data.root_lin_vel_w[:, :2]
            sp = float(v.norm(dim=-1)[0])
            if d < tol and sp < 0.03:
                break
            if sp < 0.015 and d > tol:
                stall += 1
                if stall >= 60 and stall % 60 == 0:
                    bias = min(bias + 3.0, 16.0)
                    print(f"[solve] {tag} i={i} d={d * 1000:.1f}mm stall -> "
                          f"breakaway bias {bias:.0f}N", flush=True)
            elif sp > 0.06:
                stall, bias = 0, 0.0
            v_des = kp * err
            nv = v_des.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            v_des = v_des / nv * nv.clamp(max=vmax)
            f_xy = mass * kv * (v_des - v)
            nf = f_xy.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_xy = f_xy / nf * nf.clamp(max=fmax)
            push = bias + (ff0 * min(max(d / 0.03, 0.4), 1.0) if d > tol else 0.0)
            if push > 0.0:
                f_xy = f_xy + push * err / err.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f = torch.zeros(n, 3, device=device)
            f[:, :2] = f_xy
            scene.bowl.set_external_force_and_torque(
                f.unsqueeze(1), zero1, env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i % 200 == 0:
                print(f"[solve] {tag} i={i} d={d * 1000:.1f}mm v={sp:.3f}", flush=True)
        # brake: servo to zero velocity, then hands off
        for _ in range(60):
            v = scene.bowl.data.root_lin_vel_w[:, :2]
            f = torch.zeros(n, 3, device=device)
            f[:, :2] = (mass * 40.0 * (-v)).clamp(min=-4.0, max=4.0)
            scene.bowl.set_external_force_and_torque(
                f.unsqueeze(1), zero1, env_ids=all_ids, is_global=True)
            env.step(no_action)
        clear_forces()
        step(60)
        d = float((target_xy - bowl_p()[:, :2]).norm(dim=-1)[0])
        print(f"[solve] {tag} parked: d={d * 1000:.1f}mm", flush=True)
        return d < tol + 0.004

    # ---------------- phase 0: reset, settle, layout readback -------------------------------
    step(180)
    side = float(scene.side[0])
    live = "p" if side > 0 else "n"
    slot_xy = scene._live_slot_xy()
    dock_xy = scene.dock_xy.clone()
    bp0, wp0 = ball_p()[0], bowl_p()[0]
    print(f"[solve] layout readback (seed {args.seed}): live side "
          f"{'+x' if side > 0 else '-x'} slot=({float(slot_xy[0, 0]):+.3f},"
          f"{float(slot_xy[0, 1]):+.3f}) dock=({float(dock_xy[0, 0]):+.3f},"
          f"{float(dock_xy[0, 1]):+.3f}) bowl=({float(wp0[0]):+.3f},{float(wp0[1]):+.3f}) "
          f"ball_z={float(bp0[2]):.3f}", flush=True)
    report("reset")
    assert torch.isfinite(scene.bowl.data.root_pos_w).all(), "NaN/inf after settle"
    assert abs(float(bp0[2]) - c.ball_blade_z) < 0.008, "ball must rest on the live blade"
    assert not bool(scene.lost[0]), "ball must not be lost at spawn"
    s0 = print_score("P0 reset+settle (ball on the live gate blade)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: COVER — slide the bowl over the live slot --------------------
    # Stage square-on first (same y as the slot, well before the hole), then cross the
    # open slot along the x axis so the flange meets the hole edges squarely.
    w1 = slot_xy.clone()
    w1[:, 0] -= scene.side * 0.155
    push_bowl(w1, mass=c.bowl_mass, vmax=0.25, tol=0.015, max_steps=600, tag="P1 stage")
    ok = push_bowl(slot_xy, mass=c.bowl_mass, vmax=0.14, tol=0.010,
                   max_steps=900, tag="P1 cover")
    report("covered")
    if not ok or not bool(scene.covered[0]):
        print("SIM_GEN_SOLVE: FAIL (bowl never covered the live slot)", flush=True)
        os._exit(1)
    assert not bool(scene.caught[0]), "caught latch must not set before the drop"
    s1 = print_score("P1 bowl covering the live floor slot under the chute")
    assert s1 >= s0 - 1e-6 and abs(s1 - 0.25) < 0.02, f"P1 score {s1} (expect covered=0.25)"

    # ---------------- phase 2: TRIGGER — push the gate blade, ball drops in -----------------
    blade = scene.blades[live]
    ff, kv, v_des_y = 0.35, 80.0, 0.06  # ff sized for the in-situ mu_eff ~0.5
    opened = False
    for i in range(900):
        by = float((blade.data.root_pos_w - scene.env_origins)[0, 1]) - c.slot_y
        if by >= c.blade_open_y + 0.003:
            opened = True
            break
        vy = blade.data.root_lin_vel_w[:, 1]
        fy = (ff + c.blade_mass * kv * (v_des_y - vy)).clamp(min=-0.4, max=1.5)
        f = torch.zeros(n, 3, device=device)
        f[:, 1] = fy
        blade.set_external_force_and_torque(
            f.unsqueeze(1), zero1, env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i % 150 == 0:
            print(f"[solve] P2 gate i={i} blade_y={by * 1000:.1f}mm "
                  f"ball_z={float(ball_p()[0, 2]):.3f}", flush=True)
    clear_forces()
    step(240)  # ball falls ~5 cm, lands in the bowl, settles
    report("caught")
    if not opened:
        print("SIM_GEN_SOLVE: FAIL (gate blade never opened)", flush=True)
        os._exit(1)
    if bool(scene.lost[0]):
        print("SIM_GEN_SOLVE: FAIL (ball lost to the plenum — cover failed?)", flush=True)
        os._exit(1)
    if not bool(scene._in_bowl()[0]):
        print("SIM_GEN_SOLVE: FAIL (ball did not land in the bowl)", flush=True)
        os._exit(1)
    assert bool(scene.caught[0]), "caught latch did not set during the drop"
    s2 = print_score("P2 gate opened — ball dropped into the covered bowl")
    assert s2 >= s1 - 1e-6 and abs(s2 - 0.60) < 0.02, f"P2 score {s2} (expect 0.60)"

    # ---------------- phase 3: DELIVER — slide the loaded bowl onto the dock ----------------
    m_tot = c.bowl_mass + c.ball_mass
    mid = slot_xy.clone()
    mid[:, 1] = 0.0  # straight -y out from under the chute first
    push_bowl(mid, mass=m_tot, vmax=0.20, tol=0.020, max_steps=600, tag="P3 exit")
    ok = push_bowl(dock_xy, mass=m_tot, vmax=0.20, tol=0.008,
                   max_steps=1200, tag="P3 dock")
    report("docked")
    if not ok or not bool(scene._docked()[0]):
        print("SIM_GEN_SOLVE: FAIL (bowl never docked)", flush=True)
        os._exit(1)
    if not bool(scene._in_bowl()[0]):
        print("SIM_GEN_SOLVE: FAIL (ball left the bowl during delivery)", flush=True)
        os._exit(1)
    step(180)
    ok_live = bool(scene.success()[0])
    for _ in range(6):  # allow extra settle if needed
        if ok_live:
            break
        step(120)
        ok_live = bool(scene.success()[0])
    if not ok_live:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after docking)", flush=True)
        os._exit(1)
    s3 = print_score("P3 loaded bowl docked on the red disc")
    assert s3 >= s2 - 1e-6, "score decreased across delivery"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) --------------
    holds = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        holds = holds and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = holds and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
