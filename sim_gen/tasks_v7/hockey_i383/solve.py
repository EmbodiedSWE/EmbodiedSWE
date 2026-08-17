"""Teleport solution for PolarityDockScene (sim_gen task `hockey_i383`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. PERCEIVE the per-bay terminal side. Nothing is memorized: the seat pose for
   each battery is computed FROM THE BAY CASSETTE'S OWN POSE READBACK (the
   Bernoulli 180 deg yaw that randomizes which end carries the silver terminal is
   part of that quaternion). The stdout layout report prints the recovered
   terminal side per bay so distinct seeds are provable.
2. SEAT EACH BATTERY (teleport = transport only): one pose write per battery
   stages it ABOVE its bay — lying flat, tip toward the terminal, bottom ~5 mm
   above the deck, exactly the arm's place-from-above after a wrist-yaw reorient.
   The battery then FALLS ~18 mm into the open-top trough and settles on the
   trough floor under gravity and contact; a short velocity-capped axial press
   (the fingertip nudge) squares the tip into the terminal slot. The seat
   verdict — flat on the floor, below deck, tip into the slot — is earned by
   contact dynamics, never written.
3. CLOSE THE LID (contact dynamics — never teleported): a world-frame force
   along the frame's local -y on the lid body (the applied-wrench emulation of
   the arm pushing the red handle) drives the D6 slide from its open park to the
   q = 0 stop. A velocity-capped bang-bang servo with a stall-escalating force
   cap (effective external wrench on the forge pods is a fraction of the
   commanded one) keeps the slide quasi-static. The force is CUT at the stop.
4. HANDS-OFF: from the force cut to the verdict nothing touches anything; the
   success state (both batteries seated below deck + lid at its closed stop) is
   a settled physical outcome.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.polarity_dock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    dt = 1.0 / 120.0

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def lidq() -> float:
        return float(scene.lid_q()[0])

    def bay_local_bat(i: int, j: int):
        """Battery i's centre in bay j's body frame (env 0)."""
        bay, bat = scene.bays[j], scene.bats[i]
        loc = quat_apply_inverse(bay.data.root_quat_w,
                                 bat.data.root_pos_w - bay.data.root_pos_w)[0]
        return loc

    def report(tag: str) -> None:
        seated = scene.seated()[0]
        print(f"[solve] {tag:12s} | lid_q={lidq() * 1000:+7.1f}mm "
              f"seat=({bool(seated[0])},{bool(seated[1])}) "
              f"seat_ever=({bool(scene._seat_ever[0, 0])},{bool(scene._seat_ever[0, 1])}) "
              f"lid_ever={bool(scene._lid_ever[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(90)
    fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
    fq = scene.frame.data.root_quat_w[0]
    f_yaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
    # Recover the per-bay terminal side from the cassette quaternion readback: the
    # relative yaw between bay and frame is ~0 (terminal at frame +x) or ~180 deg.
    sides = []
    for j, bay in enumerate(scene.bays):
        q_rel = quat_mul(
            torch.stack([fq[0], -fq[1], -fq[2], -fq[3]]).unsqueeze(0),
            bay.data.root_quat_w[0:1])[0]
        rel_yaw = 2.0 * math.atan2(float(q_rel[3]), float(q_rel[0]))
        sides.append("flipped" if abs(rel_yaw) > math.pi / 2 else "normal")
    b_str = " ".join(
        f"bat{i}=({float((scene.bats[i].data.root_pos_w - scene.env_origins)[0, 0]):+.3f},"
        f"{float((scene.bats[i].data.root_pos_w - scene.env_origins)[0, 1]):+.3f})"
        for i in range(2))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"frame=({float(fp[0]):+.3f},{float(fp[1]):+.3f}) yaw={math.degrees(f_yaw):+.1f}deg "
          f"lid_q={lidq() * 1000:.1f}mm terminal_side=(bay0:{sides[0]}, bay1:{sides[1]}) "
          f"{b_str}", flush=True)
    report("reset")
    assert lidq() > 0.090, f"lid should start parked open, q={lidq():.4f}"
    assert not bool(scene.seated()[0].any()), "a battery spawned already seated"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phases 1+2: seat each battery -----------------------------------------
    # Teleport = transport only: stage the battery ABOVE the open-top trough, lying
    # flat with the tip toward the terminal (orientation computed from the bay's own
    # quaternion — this is where the Bernoulli flip is read back). Gravity drops it
    # the last ~18 mm; a velocity-capped axial press squares the tip into the slot.
    qy90 = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],
                        device=device).expand(n, 4)
    prev = s0
    for i in (0, 1):
        bay = scene.bays[i]
        bat = scene.bats[i]
        q_bay = bay.data.root_quat_w
        # stage: bay-local (seat_x, 0, 0.055) -> bottom ~5 mm above the 37 mm deck
        drop_loc = torch.tensor([c.seat_x, 0.0, 0.055], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = bay.data.root_pos_w + quat_apply(q_bay, drop_loc)
        st[:, 3:7] = quat_mul(q_bay, qy90)  # battery +z (tip) -> bay-local +x
        bat.write_root_state_to_sim(st, all_ids)
        step(90)  # free fall into the trough + settle
        loc = bay_local_bat(i, i)
        print(f"[solve] bat{i} dropped into bay{i}: bay-local "
              f"({float(loc[0]) * 1000:+.1f},{float(loc[1]) * 1000:+.1f},"
              f"{float(loc[2]) * 1000:+.1f})mm", flush=True)
        # fingertip nudge: velocity-capped axial press toward the terminal slot
        push_dir = quat_apply(q_bay, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
        for _ in range(90):
            vel = bat.data.root_lin_vel_w[0]
            u = float((vel * push_dir[0]).sum())
            f = 0.35 if u < 0.05 else 0.0
            bat.set_external_force_and_torque(
                (push_dir * f).view(n, 1, 3).contiguous(), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        bat.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        step(60)  # settle
        report(f"seated{i}")
        loc = bay_local_bat(i, i)
        assert bool(scene.seated()[0, i]), (
            f"battery {i} did not seat: bay-local "
            f"({float(loc[0]):+.4f},{float(loc[1]):+.4f},{float(loc[2]):+.4f})")
        assert bool(scene._seat_ever[0, i]), f"seat latch {i} did not set"
        s = print_score(f"P{i + 1} battery {i} seated tip-to-terminal in bay {i}")
        assert s >= prev - 1e-6, "score decreased across seating"
        prev = s

    # ---------------- phase 3: slide the lid closed (contact dynamics) ----------------------
    # World-frame force along the frame's local -y on the lid body (the push on the
    # red handle). Bang-bang velocity servo, quasi-static; stall probe escalates the
    # force cap (effective wrench on the forge pods is a fraction of the commanded).
    fq_now = scene.frame.data.root_quat_w
    close_dir = quat_apply(fq_now, torch.tensor([0.0, -1.0, 0.0], device=device).expand(n, 3))
    ctl = {"cap": 2.0}
    ref_q, ref_i = lidq(), 0
    closed = False
    for i in range(720):
        q = lidq()
        if q < 0.0015:
            closed = True
            break
        v = -(scene.lid.data.root_lin_vel_w[0] * close_dir[0]).sum()  # closing speed
        f = ctl["cap"] if float(v) < 0.10 else 0.0
        scene.lid.set_external_force_and_torque(
            (close_dir * f).view(n, 1, 3).contiguous(), zero_wrench,
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i - ref_i >= 60:  # stall probe every 0.5 s
            q_now = lidq()
            if ref_q - q_now < 0.002 and q_now > 0.004:
                ctl["cap"] = min(ctl["cap"] * 1.7, 20.0)
                print(f"[solve] lid stalled at q={q_now * 1000:.1f}mm — force cap -> "
                      f"{ctl['cap']:.2f} N", flush=True)
            ref_q, ref_i = q_now, i
    scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(90)  # hands-off settle at the stop
    if lidq() > 0.002:  # limit-solver kick-back off the stop: press home once more
        print(f"[solve] lid drifted off the stop to q={lidq() * 1000:.1f}mm — "
              f"pressing home once more", flush=True)
        for _ in range(180):
            if lidq() < 0.0010:
                break
            v = -(scene.lid.data.root_lin_vel_w[0] * close_dir[0]).sum()
            f = ctl["cap"] if float(v) < 0.05 else 0.0
            scene.lid.set_external_force_and_torque(
                (close_dir * f).view(n, 1, 3).contiguous(), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        step(60)
    report("lid-closed")
    assert closed or lidq() < c.closed_tol, (
        f"lid never reached its closed stop, q={lidq() * 1000:.1f}mm")
    assert bool(scene.lid_closed()[0]), "lid_closed() false after the push"
    assert bool(scene._lid_ever[0]), "lid latch did not set"
    s3 = print_score("P3 lid slid fully closed over the seated pair")
    assert s3 >= prev - 1e-6, "score decreased across the closure"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) ------------
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
