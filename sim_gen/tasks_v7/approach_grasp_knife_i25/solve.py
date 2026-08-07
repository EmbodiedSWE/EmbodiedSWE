"""Teleport solution for UnderpinSwapScene (sim_gen task `approach_grasp_knife_i25`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY:
  P1 — one pose write carries the BLUE column across open space from its spawn to a
  staging pose beside the deck: on the ground at the slot's x station, block fully
  OUTSIDE the deck footprint on the +y side, handle trailing (exactly the pose an arm
  would arrive at after carrying the column over; nothing is under the deck yet and no
  scoring gate beyond the latched approach credit is satisfied).
  P4 — one pose write carries the RED column, already extracted and free-standing on
  open ground, over to the green pad (again pure carry across free space).
Every LOAD-BEARING interaction goes through CONTACT DYNAMICS:
  P2 — a floating-hand force controller (velocity-regulated push along the fixture -y
  axis, lateral PD holding the slot x station, yaw-hold torque) slides the blue column
  UNDER the raised deck through its 3 mm clearance until its bearing block stands
  beneath the blue band. Ground friction is real; the column is never teleported into
  the slot.
  P3 — the same controller drags the loaded RED column out sideways: the push works
  against ground friction AND the deck's weight bearing on the block, the deck loses
  its support and settles 3 mm onto the blue column purely under gravity/contact —
  the entire load transfer is contact transmission. Forces are then cut and everything
  settles on real friction.
The deck itself is never touched, teleported, or forced.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.approach_grasp_knife_i25.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.underpin_swap")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        bl = scene.blue_local()[0]
        rl = scene.red_local()[0]
        print(f"[solve] {tag:12s} | blue_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
              f"red_loc=({float(rl[0]):+.3f},{float(rl[1]):+.3f}) "
              f"end_z={float(scene.deck_end_z()[0]):.4f} "
              f"tilt={float(scene.deck_tilt_deg()[0]):.2f} "
              f"seated={bool(scene.blue_seated()[0])} clear={bool(scene.red_clear()[0])} "
              f"on_pad={bool(scene.red_on_pad()[0])} deck_ok={bool(scene.deck_ok()[0])} "
              f"spoiled={bool(scene.spoiled[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def fail(msg: str) -> None:
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL ({msg})", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    fq = scene.fixture.data.root_quat_w[0]
    fyaw = 2.0 * math.atan2(float(fq[3]), float(fq[0]))
    ca, sa = math.cos(fyaw), math.sin(fyaw)
    x_dir = torch.tensor([ca, sa, 0.0], device=device)  # fixture +x, world frame
    y_dir = torch.tensor([-sa, ca, 0.0], device=device)  # fixture +y, world frame
    push_dir = -y_dir  # both columns travel along fixture -y
    ap = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
    bl0 = (scene.blue.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): fixture=({float(ap[0]):+.3f},"
          f"{float(ap[1]):+.3f}) yaw={math.degrees(fyaw):+.1f}deg "
          f"blue_spawn=({float(bl0[0]):+.3f},{float(bl0[1]):+.3f}) "
          f"d0={float(scene.d0[0]):.3f} end_z0={float(scene.deck_end_z()[0]):.4f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    def col_wrench(body, *, x_hold: float, yaw_tgt: float, f_push: float,
                   v_des: float) -> None:
        """One floating-hand control tick on a column: velocity-regulated push along
        fixture -y, lateral PD holding the fixture-x station, yaw-hold torque."""
        loc = scene._to_local(body.data.root_pos_w)[0]
        v = body.data.root_lin_vel_w[0]
        w = body.data.root_ang_vel_w[0]
        v_along = float(torch.dot(v, push_dir))
        f_p = f_push if v_along < v_des else 0.0
        e_x = x_hold - float(loc[0])
        v_x = float(torch.dot(v, x_dir))
        f_lat = max(-0.8, min(0.8, c.col_mass * (60.0 * e_x - 15.0 * v_x) * 10.0))
        yaw = float(scene._yaw(body.data.root_quat_w)[0])
        tq_z = max(-0.06, min(0.06, -0.08 * _wrap(yaw - yaw_tgt) - 0.010 * float(w[2])))
        f_world = f_p * push_dir + f_lat * x_dir
        tq = torch.tensor([0.0, 0.0, tq_z], device=device)
        body.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)

    # ---------------- phase 1: TRANSPORT blue to the staging pose (teleport) ---------------
    # Staging: ground level at the slot's x station, block centre 160 mm out on the +y
    # side (fully outside the deck footprint), handle trailing +y, yaw = fixture yaw.
    stage_y = 0.160
    cy2, sy2 = math.cos(fyaw / 2), math.sin(fyaw / 2)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(ap[0]) + ca * c.slot_x - sa * stage_y
    st[:, 1] = float(ap[1]) + sa * c.slot_x + ca * stage_y
    st[:, 2] = 0.0005
    st[:, 3] = cy2
    st[:, 6] = sy2
    st[:, 0:3] += scene.env_origins
    scene.blue.write_root_state_to_sim(st, all_ids)
    step(30)
    report("staged")
    s1 = print_score("P1 transport blue to staging beside the deck")
    if s1 < s0 - 1e-6:
        fail("score decreased across transport")

    # ---------------- phase 2: SLIDE-IN (contact dynamics) ---------------------------------
    # Push the blue column along fixture -y through the 3 mm clearance under the deck
    # until its bearing block stands beneath the blue band. Ground friction is real.
    f_push, v_des = 1.5, 0.08
    best_y, last_bump, slide_steps = 1e9, 0, 0
    for i in range(2400):
        y_loc = float(scene.blue_local()[0][1])
        if y_loc <= 0.003:
            break
        if bool(scene.spoiled[0]):
            fail("deck spoiled during blue slide-in")
        col_wrench(scene.blue, x_hold=c.slot_x, yaw_tgt=fyaw, f_push=f_push, v_des=v_des)
        env.step(no_action)
        slide_steps += 1
        if y_loc < best_y - 0.002:
            best_y, last_bump = y_loc, i
        elif i - last_bump > 240:  # stalled: friction was underestimated
            f_push = min(f_push + 0.5, 5.0)
            last_bump = i
            print(f"[solve] slide-in stalled at y_loc={y_loc:.3f}, raising push to "
                  f"{f_push:.2f} N", flush=True)
    clear(scene.blue)
    step(90)
    report("seated")
    print(f"[solve] slide-in complete after {slide_steps} steps (push {f_push:.2f} N)",
          flush=True)
    if not bool(scene.blue_seated()[0]):
        fail("blue column did not seat in the slot")
    s2 = print_score("P2 blue slide-in under the deck (contact)")
    if s2 < s1 - 1e-6:
        fail("score decreased across slide-in")

    # ---------------- phase 3: EXTRACTION (contact dynamics, load-bearing) -----------------
    # Drag the loaded red column out along fixture -y by working against ground friction
    # plus the deck's weight on its block; the deck settles onto the blue column purely
    # through gravity and contact.
    f_push, v_des = 3.0, 0.10
    best_y, last_bump, drag_steps = 1e9, 0, 0
    for i in range(2400):
        y_loc = float(scene.red_local()[0][1])
        if y_loc <= -0.145:
            break
        if bool(scene.spoiled[0]):
            fail("deck spoiled during red extraction")
        col_wrench(scene.red, x_hold=c.red_x, yaw_tgt=fyaw + math.pi,
                   f_push=f_push, v_des=v_des)
        env.step(no_action)
        drag_steps += 1
        if y_loc < best_y - 0.002:
            best_y, last_bump = y_loc, i
        elif i - last_bump > 240:  # stalled: deck load friction was underestimated
            f_push = min(f_push + 0.75, 9.0)
            last_bump = i
            print(f"[solve] extraction stalled at y_loc={y_loc:.3f}, raising push to "
                  f"{f_push:.2f} N", flush=True)
    clear(scene.red)
    step(150)  # deck settles 3 mm onto the blue column, everything comes to rest
    report("extracted")
    print(f"[solve] extraction complete after {drag_steps} steps (push {f_push:.2f} N)",
          flush=True)
    if bool(scene.spoiled[0]):
        fail("deck spoiled while settling onto the blue column")
    if not bool(scene.red_clear()[0]):
        fail("red column not clear of the deck footprint")
    s3 = print_score("P3 red extraction + load transfer (contact)")
    if s3 < s2 - 1e-6:
        fail("score decreased across extraction")

    # ---------------- phase 4: TRANSPORT red to the pad (teleport) -------------------------
    # The red column stands free on open ground; carry it to the green pad (pure
    # transport, 2 mm drop onto the pad top under gravity).
    pad = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    half = (fyaw + math.pi) / 2
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(pad[0])
    st[:, 1] = float(pad[1])
    st[:, 2] = c.pad_t + 0.002
    st[:, 3] = math.cos(half)
    st[:, 6] = math.sin(half)
    st[:, 0:3] += scene.env_origins
    scene.red.write_root_state_to_sim(st, all_ids)
    step(90)
    report("parked")
    s4 = print_score("P4 transport red to the pad")
    if s4 < s3 - 1e-6:
        fail("score decreased across transport to pad")
    if not bool(scene.success()[0]):
        fail("no success after park+settle")

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
