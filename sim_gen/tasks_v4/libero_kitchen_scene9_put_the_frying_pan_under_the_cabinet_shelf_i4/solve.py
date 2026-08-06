"""Teleport solution for PanUnderLowShelfScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i4`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY: one pose write carries the pan across open table
from its spawn to a staging spot on the alcove's axis, ~60 mm in front of the opening,
flat on the table, handle pointing away from the shelf (exactly where an arm would push
it from). The LOAD-BEARING interaction — sliding the pan under the 48 mm roof, through
the 18 mm clearance, against table friction, guided by the side walls, until the whole
disc is inside — is executed through CONTACT DYNAMICS: a horizontal external force
(velocity-regulated bang-bang, world frame) pushes the pan in; the simulator's friction,
roof and wall contacts do the rest. Nothing is ever spawned or written into the goal
region; the force is cut mid-band and the pan coasts/settles on real friction.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i4.solve --headless [--seed N]
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
    env = ENVS.get("simgen.pan_lowshelf_slide")().build(num_envs=args.num_envs, device=device)
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

    def shelf_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.shelf.data.root_pos_w - scene.env_origins)[0]
        q = scene.shelf.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return sp, yaw

    def report(tag: str) -> None:
        loc = scene._pan_local()[0]
        print(f"[solve] {tag:12s} | pan_local=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):.3f}) frac={float(scene.insertion_frac()[0]):.3f} "
              f"fully_in={bool(scene.fully_in()[0])} flat={bool(scene.flat()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.pan.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    sp, syaw = shelf_pose()
    pan0 = (scene.pan.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): shelf=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(syaw):+.1f}deg "
          f"pan_spawn=({float(pan0[0]):+.3f},{float(pan0[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: TRANSPORT (teleport across free space only) -----------------
    # Staging spot: on the alcove axis, disc leading edge ~60 mm in front of the opening,
    # flat on the table, handle pointing OUT (away from the back wall) — the pose an arm
    # would push from. This bypasses NO required interaction: the pan is still entirely
    # outside the alcove, in the open.
    stage_local_y = -(c.alcove_d / 2 + c.pan_r + 0.06)
    cy, sy = math.cos(syaw), math.sin(syaw)
    # world = shelf_pos + R(yaw) @ (0, stage_local_y)
    wx = float(sp[0]) + (-sy) * stage_local_y
    wy = float(sp[1]) + cy * stage_local_y
    pan_yaw = syaw - math.pi / 2  # pan local +x (handle) -> shelf local -y (out the front)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = wx
    st[:, 1] = wy
    st[:, 2] = c.pan_h / 2 + 0.002
    st[:, 3] = math.cos(pan_yaw / 2)
    st[:, 6] = math.sin(pan_yaw / 2)
    st[:, 0:3] += scene.env_origins
    scene.pan.write_root_state_to_sim(st, all_ids)
    step(30)  # real settle after the transport
    report("staged")
    s1 = print_score("P1 transport to staging")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: PUSH under the roof through contact dynamics ----------------
    # Velocity-regulated bang-bang horizontal force at the pan's CoM (world frame): the
    # slide under the roof is pure contact dynamics — table friction below, 18 mm roof
    # clearance above, side walls guiding. A small lateral correction keeps the approach
    # centred until the walls take over. Force is CUT once the disc centre crosses the
    # alcove midline; the pan coasts and settles on real friction.
    push_dir = torch.tensor([-sy, cy, 0.0], device=device)
    lat_dir = torch.tensor([cy, sy, 0.0], device=device)
    f_push, v_des = 5.0, 0.08
    best_y, last_bump = -1.0, 0
    pushed_steps = 0
    for i in range(2400):
        loc = scene._pan_local()[0]
        y_l, x_l = float(loc[1]), float(loc[0])
        if y_l >= 0.0:  # disc centre at the alcove midline: comfortably in the depth band
            break
        v_axis = float((scene.pan.data.root_lin_vel_w[0] * push_dir).sum())
        f_axis = f_push if v_axis < v_des else 0.0
        f_lat = max(-1.2, min(1.2, -30.0 * x_l)) if y_l < -c.alcove_d / 2 else 0.0
        f_w = (push_dir * f_axis + lat_dir * f_lat).view(1, 1, 3).expand(n, 1, 3)
        scene.pan.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)
        pushed_steps += 1
        if y_l > best_y + 0.005:
            best_y, last_bump = y_l, i
        elif i - last_bump > 240:  # stalled: push harder (friction was underestimated)
            f_push = min(f_push + 2.0, 12.0)
            last_bump = i
            print(f"[solve] push stalled at y_local={y_l:+.3f}, raising force to "
                  f"{f_push:.1f} N", flush=True)
    clear_force()
    step(2)
    report("pushed")
    print(f"[solve] push complete after {pushed_steps} steps, force {f_push:.1f} N", flush=True)
    s2 = print_score("P2 contact push under the roof")
    assert s2 >= s1 - 1e-6, "score decreased across push"

    # ---------------- phase 3: hands off, settle, judge ------------------------------------
    step(120)
    report("settled")
    s3 = print_score("P3 settle")
    assert s3 >= s2 - 1e-6, "score decreased across settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settle)", flush=True)
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
