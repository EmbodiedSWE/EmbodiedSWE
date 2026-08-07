"""Teleport solution for TagHangersScene (sim_gen task `pick_single_egad_i4`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. The LOAD-BEARING interaction — threading the peg
through the tag's aperture and establishing real suspension — goes through CONTACT
DYNAMICS, once per tag:
  1. TRANSPORT: the tag is teleported from the ground to a free-air pose just off the
     MATCHING stand's peg tip, aperture axis lined up with the peg — exactly the pose
     an arm reaches after grasping the handle plate and orienting the washer.
  2. THREADING (contact): a gravity-compensated hold (what the arm's grip provides)
     plus a velocity-regulated axial push and weak centering springs drives the tag
     along the peg: the peg enters the 32 mm aperture at the tip and slides ~70 mm
     through it under real collision — any misalignment scrapes washer bars against
     the peg.
  3. RELEASE (contact): all forces are cleared mid-peg. The tag FALLS ~8 mm until the
     washer's inner top bar lands on the peg's top face, swings as a pendulum, and
     rings down to rest — the suspension the rubric judges is established by gravity
     and contact, never written.
Nothing is ever teleported into a scoring state: the pre-thread hover pose is outside
the peg span (and scores only transport-legal lift/approach latches).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.pick_single_egad_i4.solve --headless [--seed N]
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
    env = ENVS.get("simgen.tag_hangers")().build(num_envs=args.num_envs, device=device)
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

    def stand_pose(nm: str) -> tuple[torch.Tensor, float]:
        sp = (scene.stands[nm].data.root_pos_w - scene.env_origins)[0]
        q = scene.stands[nm].data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return sp, yaw

    def report(tag: str) -> None:
        bits = []
        for nm in scene.names:
            loc = scene._stand_local(nm, nm)[0]
            bits.append(f"{nm}_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                        f"{float(loc[2]):.3f}) hang={bool(scene.hanging(nm)[0])} "
                        f"settled={bool(scene.settled(nm)[0])}")
        print(f"[solve] {tag:12s} | " + " | ".join(bits)
              + f" | success={bool(scene.success()[0])} "
                f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def hang_tag(nm: str) -> None:
        """Thread `nm`'s tag onto its matching peg: transport to the tip, force-thread
        under contact, release, settle."""
        tag = scene.tags[nm]
        sp, yaw = stand_pose(nm)
        cy, sy = math.cos(yaw), math.sin(yaw)

        # -- transport: free-air pose just off the peg tip, aperture on the axis --
        tip_x = c.peg_tip_x + c.washer_t / 2 + 0.012
        tip_z = float(scene._peg_axis_z(torch.tensor([tip_x]))[0])
        wx, wy = float(sp[0]) + cy * tip_x, float(sp[1]) + sy * tip_x
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = wx, wy, tip_z
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        st[:, 0:3] += scene.env_origins
        tag.write_root_state_to_sim(st, all_ids)

        # -- contact threading: grav-comp hold + axial push + weak centering --
        f_ax, v_des = 0.8, 0.10
        grav = 9.81 * c.tag_mass
        steps, best_x, last_bump = 0, 10.0, 0
        for i in range(1500):
            loc = scene._stand_local(nm, nm)[0]
            x_l = float(loc[0])
            if x_l <= 0.032:  # deep: after release the washer rests near the post face,
                break         # whose second contact kills the edge-rocking mode
            v_w = tag.data.root_lin_vel_w[0]
            v_ax = -(cy * float(v_w[0]) + sy * float(v_w[1]))  # along -x_stand
            fl_x = -f_ax if v_ax < v_des else 0.0  # push toward the post
            fl_y = max(-0.4, min(0.4, -10.0 * float(loc[1])))
            dz = float(loc[2]) - float(scene._peg_axis_z(loc[0:1])[0])
            fz = grav + max(-0.4, min(0.4, -8.0 * dz))
            fx = cy * fl_x - sy * fl_y
            fy = sy * fl_x + cy * fl_y
            f_w = torch.tensor([fx, fy, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
            tag.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                              env_ids=all_ids, is_global=True)
            env.step(no_action)
            steps += 1
            if x_l < best_x - 0.002:
                best_x, last_bump = x_l, i
            elif i - last_bump > 250:  # stalled: push harder
                f_ax = min(f_ax + 0.6, 4.0)
                last_bump = i
                print(f"[solve] {nm} threading stalled at x={x_l:+.3f}, raising force "
                      f"to {f_ax:.1f} N", flush=True)
        clear_wrench(tag)
        print(f"[solve] {nm} threaded after {steps} steps (force {f_ax:.1f} N)", flush=True)

        # -- release: drop ~8 mm onto the peg, swing, ring down (all contact) --
        for _ in range(24):  # up to 24 x 30 = 720 steps = 6 s
            step(30)
            if bool((scene.hanging(nm) & scene.settled(nm))[0]):
                break
        report(f"{nm}-hung")

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    for nm in (*scene.names, "gray"):
        sp, yaw = stand_pose(nm)
        print(f"[solve] layout readback (seed {args.seed}): stand_{nm} at "
              f"({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg",
              flush=True)
    for i, nm in enumerate(scene.names):
        tp = (scene.tags[nm].data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback (seed {args.seed}): tag_{nm} at "
              f"({float(tp[0]):+.3f},{float(tp[1]):+.3f},{float(tp[2]):.3f}) "
              f"d_ref={float(scene.d_ref[0, i]):.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: hang the red tag --------------------------------------------
    hang_tag(scene.names[0])
    s1 = print_score("P1 thread+hang red (contact)")
    assert s1 >= s0 - 1e-6, "score decreased across red hang"

    # ---------------- phase 2: hang the blue tag -------------------------------------------
    hang_tag(scene.names[1])
    s2 = print_score("P2 thread+hang blue (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across blue hang"

    # ---------------- phase 3: settle to success -------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s3 = print_score("P3 all settled")
    assert s3 >= s2 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after threading+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                for nm in scene.names:
                    tag = scene.tags[nm]
                    lv = float(tag.data.root_lin_vel_w[0].norm())
                    av = float(tag.data.root_ang_vel_w[0].norm())
                    loc = scene._stand_local(nm, nm)[0]
                    dz = float(loc[2]) - float(scene._peg_axis_z(loc[0:1])[0])
                    print(f"[solve] persist flicker @step {i}: {nm} "
                          f"hang={bool(scene.hanging(nm)[0])} settled={bool(scene.settled(nm)[0])} "
                          f"lin={lv:.4f} ang={av:.4f} x={float(loc[0]):+.4f} "
                          f"y={float(loc[1]):+.4f} dz={dz:+.4f}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
