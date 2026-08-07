"""Teleport solution for TunnelShuttleScene (sim_gen task `pick_single_egad_i3`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. The two LOAD-BEARING interactions go through
CONTACT DYNAMICS:
  1. GATE EXTRACTION — a regulated upward force (plus small fixture-frame centring)
     draws the yellow pin up through its cover cross-gap (2 mm clearances) until its
     blade clears the cover top under real sliding contact; only THEN is the freed pin
     teleported (transport across open air) to a parking spot on the ground.
  2. TUNNEL SLIDE — a velocity-regulated horizontal force along the fixture's
     (randomized) channel axis drags the shuttle along the floor, knob through the
     slot, THROUGH the cross-section the gate blocked, until it emerges past the
     covered section — friction, wall guidance and the cover overhead are all real.
The final placement is transport + gravity: the shuttle — by then in open air, exactly
where an arm would lift it — is teleported to a hover pose over the bin mouth and
DROPPED; it falls ~26 mm and settles on the bin floor under contact. Nothing is ever
written into a scoring state: the hover pose is above the bin walls and outside every
scoring gate except the (transport-only) approach term.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.pick_single_egad_i3.solve --headless [--seed N]
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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tunnel_shuttle")().build(num_envs=args.num_envs, device=device)
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

    def fixture_pose() -> tuple[torch.Tensor, float]:
        fp = (scene.fixture.data.root_pos_w - scene.env_origins)[0]
        q = scene.fixture.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return fp, yaw

    def to_world(x_l: float, y_l: float) -> tuple[float, float]:
        fp, fyaw = fixture_pose()
        cy, sy = math.cos(fyaw), math.sin(fyaw)
        return float(fp[0]) + cy * x_l - sy * y_l, float(fp[1]) + sy * x_l + cy * y_l

    def report(tag: str) -> None:
        sl = scene._local(scene.shuttle)[0]
        gl = scene._local(scene.gate)[0]
        print(f"[solve] {tag:12s} | shuttle_local=({float(sl[0]):+.3f},{float(sl[1]):+.3f},"
              f"{float(sl[2]):.3f}) gate_local=({float(gl[0]):+.3f},{float(gl[1]):+.3f},"
              f"{float(gl[2]):.3f}) latches=(g {float(scene.gate_latch[0]):.2f}, "
              f"s {float(scene.slide_latch[0]):.2f}, a {float(scene.approach_latch[0]):.2f}) "
              f"in_bin={bool(scene.in_bin()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_force(body, fx: float, fy: float, fz: float) -> None:
        f_w = torch.tensor([fx, fy, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
        body.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                           env_ids=all_ids, is_global=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    fp, fyaw = fixture_pose()
    sl0 = scene._local(scene.shuttle)[0]
    bp = (scene.goal_bin.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): fixture=({float(fp[0]):+.3f},"
          f"{float(fp[1]):+.3f}) yaw={math.degrees(fyaw):+.1f}deg "
          f"shuttle_start_x={float(sl0[0]):+.3f} bin=({float(bp[0]):+.3f},"
          f"{float(bp[1]):+.3f}) d_ref={float(scene.d_ref[0]):.3f} "
          f"exit_x={c.exit_x:.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: EXTRACT the gate pin (contact dynamics) ---------------------
    # Regulated up-force + small fixture-frame centring draws the blade up through the
    # 20 mm cover cross-gap (2 mm clearance each side) under real sliding contact.
    f_up, v_des = 1.5, 0.12
    grav_comp = 9.81 * c.gate_mass * 1.05
    lift_steps, best_bot, last_bump = 0, -1.0, 0
    fyaw_c, fyaw_s = math.cos(fyaw), math.sin(fyaw)
    for i in range(900):
        gl = scene._local(scene.gate)[0]
        bottom = float(gl[2]) - c.blade_h / 2
        if bottom > c.cover_top + 0.008:
            break
        vz = float(scene.gate.data.root_lin_vel_w[0, 2])
        fz = f_up if vz < v_des else grav_comp
        flx = max(-0.3, min(0.3, -8.0 * (float(gl[0]) - c.gate_x)))
        fly = max(-0.3, min(0.3, -8.0 * float(gl[1])))
        fx = fyaw_c * flx - fyaw_s * fly
        fy = fyaw_s * flx + fyaw_c * fly
        apply_force(scene.gate, fx, fy, fz)
        env.step(no_action)
        lift_steps += 1
        if bottom > best_bot + 0.002:
            best_bot, last_bump = bottom, i
        elif i - last_bump > 250:  # stalled: pull harder (friction/jam underestimated)
            f_up = min(f_up + 0.8, 6.0)
            last_bump = i
            print(f"[solve] gate lift stalled at bottom={bottom:.3f}, raising force to "
                  f"{f_up:.1f} N", flush=True)
    clear_wrench(scene.gate)
    print(f"[solve] gate extracted after {lift_steps} steps (force {f_up:.1f} N)", flush=True)
    # The freed pin is now in open air above the fixture — TRANSPORT it to the parking
    # spot on the ground (outside every scoring region), zero velocity, settle.
    px, py = to_world(*c.gate_park_local)
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = px, py, c.blade_h / 2 + 0.002
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.gate.write_root_state_to_sim(st, all_ids)
    step(60)
    report("gate-out")
    s1 = print_score("P1 extract gate pin (contact) + park (transport)")
    assert s1 >= s0 - 1e-6, "score decreased across gate extraction"

    # ---------------- phase 2: SLIDE the shuttle out (contact dynamics) --------------------
    # Velocity-regulated force along the channel axis + small centring drags the
    # shuttle along the floor, knob through the slot, through the gate's old station,
    # until it emerges past the covered section.
    f_fwd, v_slide = 1.0, 0.10
    slide_steps, best_x, last_bump = 0, -1.0, 0
    for i in range(1800):
        sl = scene._local(scene.shuttle)[0]
        if float(sl[0]) >= c.exit_x + 0.004:
            break
        v_w = scene.shuttle.data.root_lin_vel_w[0]
        vx_l = fyaw_c * float(v_w[0]) + fyaw_s * float(v_w[1])
        fl_x = f_fwd if vx_l < v_slide else 0.0
        fl_y = max(-0.4, min(0.4, -10.0 * float(sl[1])))
        fx = fyaw_c * fl_x - fyaw_s * fl_y
        fy = fyaw_s * fl_x + fyaw_c * fl_y
        apply_force(scene.shuttle, fx, fy, 0.0)
        env.step(no_action)
        slide_steps += 1
        if float(sl[0]) > best_x + 0.002:
            best_x, last_bump = float(sl[0]), i
        elif i - last_bump > 250:  # stalled: push harder
            f_fwd = min(f_fwd + 0.8, 5.0)
            last_bump = i
            print(f"[solve] slide stalled at x={float(sl[0]):+.3f}, raising force to "
                  f"{f_fwd:.1f} N", flush=True)
    clear_wrench(scene.shuttle)
    step(30)
    report("slid-out")
    print(f"[solve] slide complete after {slide_steps} steps (force {f_fwd:.1f} N)", flush=True)
    s2 = print_score("P2 slide shuttle out of the tunnel (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across the slide"

    # ---------------- phase 3: TRANSPORT to the bin + gravity drop -------------------------
    # The shuttle now stands in the open exit section — liftable straight up, exactly
    # where an arm would grasp the knob and carry it. One pose write moves it across
    # open air to a hover pose over the bin mouth, bottom ~8 mm above the wall top;
    # gravity and real contact do the placement.
    bp = (scene.goal_bin.data.root_pos_w - scene.env_origins)[0]
    hover_z = c.bin_floor_t + c.bin_wall_h + c.base_size / 2 + 0.008
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = float(bp[0]), float(bp[1]), hover_z
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.shuttle.write_root_state_to_sim(st, all_ids)
    step(240)  # 2 s: fall ~26 mm, land on the bin floor, ring down
    report("dropped")
    s3 = print_score("P3 transport to bin hover + gravity drop")
    assert s3 >= s2 - 1e-6, "score decreased across the drop"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after drop+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
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
