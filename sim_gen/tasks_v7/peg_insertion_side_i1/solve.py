"""Teleport solution for RamEjectScene (sim_gen task `peg_insertion_side_i1`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: one pose write carries the rod across open space
from its rack to a hover pose on the bore axis, tip ~15 mm OUTSIDE the entry face
(exactly the pose an arm would arrive at after picking the rod and aligning it; the rod
is entirely outside the tube, the ball untouched, and hovering there satisfies no
scoring gate). The LOAD-BEARING interaction then goes through CONTACT DYNAMICS in one
continuous regulated stroke:
  ENTRY+RAM — a floating-hand force controller (gravity-compensating vertical PD,
  lateral centring in the assembly frame, an axis-alignment torque, and a
  velocity-regulated axial push) drives the rod INTO the bore under real contact; the
  rod tip meets the trapped ball and pushes it ~80-120 mm along the bore floor until
  the ball loses support at the muzzle, free-falls, and lands in the basin — every
  millimetre of the ball's motion is real contact transmission through the rod. Forces
  are then cut and everything settles on real friction. The ball is never teleported;
  the rod is never teleported INTO the tube.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.peg_insertion_side_i1.solve --headless [--seed N]
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
    env = ENVS.get("simgen.ram_eject")().build(num_envs=args.num_envs, device=device)
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

    def asm_pose() -> tuple[torch.Tensor, float]:
        ap = (scene.assembly.data.root_pos_w - scene.env_origins)[0]
        q = scene.assembly.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return ap, yaw

    def report(tag: str) -> None:
        tip = scene.rod_tip_local()[0]
        bl = scene.ball_local()[0]
        print(f"[solve] {tag:12s} | tip_local=({float(tip[0]):+.3f},{float(tip[1]):+.3f},"
              f"{float(tip[2]):.3f}) ball_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) in_basin={bool(scene.ball_in_basin()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench() -> None:
        scene.rod.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    ap, ayaw = asm_pose()
    rod0 = (scene.rod.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): assembly=({float(ap[0]):+.3f},"
          f"{float(ap[1]):+.3f}) yaw={math.degrees(ayaw):+.1f}deg "
          f"ball_depth_s0={float(scene.s0[0]):.3f} "
          f"rod_spawn=({float(rod0[0]):+.3f},{float(rod0[1]):+.3f},{float(rod0[2]):.3f}) "
          f"d0={float(scene.d0[0]):.3f} eject_tip_x={c.eject_tip_x:.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: TRANSPORT (teleport across free space only) -----------------
    # Hover pose: on the bore axis at bore-centre height, rod tip 15 mm OUTSIDE the
    # entry face, axis aligned with the bore. The rod is entirely outside the tube and
    # the ball is untouched — no scoring gate is satisfied, no interaction bypassed.
    tip_x0 = -0.015
    cx_local = tip_x0 - c.rod_len / 2
    ca, sa = math.cos(ayaw), math.sin(ayaw)
    cy2, sy2 = math.cos(ayaw / 2), math.sin(ayaw / 2)
    c45 = math.cos(math.pi / 4)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(ap[0]) + ca * cx_local
    st[:, 1] = float(ap[1]) + sa * cx_local
    st[:, 2] = c.bore_center_z
    # q = qz(yaw) * qy(90 deg): rod local +z -> assembly local +x
    st[:, 3] = cy2 * c45
    st[:, 4] = -sy2 * c45
    st[:, 5] = cy2 * c45
    st[:, 6] = sy2 * c45
    st[:, 0:3] += scene.env_origins
    scene.rod.write_root_state_to_sim(st, all_ids)
    step(1)
    report("staged")
    s1 = print_score("P1 transport to hover before the entry")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: ENTRY + RAM STROKE (contact dynamics) -----------------------
    # Floating-hand controller: vertical gravity-comp PD (until the tip is well inside,
    # then a light 0.6*m*g lift so the rod rides the bore floor), assembly-frame lateral
    # centring, an axis-alignment torque, and a velocity-regulated axial push. The rod
    # enters the bore, meets the ball, and drives it along the floor until it drops out
    # of the muzzle. Every contact is real; the loop ends on the BALL's ejection.
    m_rod = c.rod_mass
    g = 9.81
    x_dir = torch.tensor([ca, sa, 0.0], device=device)
    y_dir = torch.tensor([-sa, ca, 0.0], device=device)
    f_push, v_des = 2.0, 0.12
    ram_steps, last_tip, last_bump = 0, -1.0, 0
    ejected = False
    for i in range(3600):
        tip = scene.rod_tip_local()[0]
        bl = scene.ball_local()[0]
        if float(bl[0]) > c.tube_len + 0.002 or float(bl[2]) < c.bore_floor_z - 0.004:
            ejected = True
            break
        loc = scene._to_local(scene.rod.data.root_pos_w)[0]
        v_w = scene.rod.data.root_lin_vel_w[0]
        w_w = scene.rod.data.root_ang_vel_w[0]
        tip_x = float(tip[0])
        # vertical
        if tip_x < 0.035:
            e_z = c.bore_center_z - float(loc[2])
            fz = m_rod * (g + 80.0 * e_z - 20.0 * float(v_w[2]))
            fz = max(0.0, min(3.0, fz))
        else:
            fz = 0.6 * m_rod * g
        # lateral centring (assembly frame)
        v_y = float(torch.dot(v_w, y_dir))
        f_lat = max(-0.8, min(0.8, m_rod * (50.0 * -float(loc[1]) - 12.0 * v_y)))
        # axial push, velocity-regulated
        v_ax = float(torch.dot(v_w, x_dir))
        f_ax = f_push if v_ax < v_des else 0.0
        f_world = f_ax * x_dir + f_lat * y_dir + torch.tensor([0.0, 0.0, fz], device=device)
        # axis-alignment torque
        a = scene._rod_axis_w()[0]
        if float(torch.dot(a, x_dir)) < 0:
            a = -a
        tq = 0.03 * torch.linalg.cross(a, x_dir) - 0.006 * w_w
        tq = tq.clamp(-0.06, 0.06)
        scene.rod.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)
        env.step(no_action)
        ram_steps += 1
        if tip_x > last_tip + 0.002:
            last_tip, last_bump = tip_x, i
        elif i - last_bump > 300:  # stalled: push harder (friction was underestimated)
            f_push = min(f_push + 0.75, 6.0)
            last_bump = i
            print(f"[solve] ram stalled at tip_x={tip_x:.3f}, raising push to "
                  f"{f_push:.2f} N", flush=True)
    clear_wrench()
    step(240)  # 2 s: ball free-falls into the basin and settles, rod comes to rest
    report("rammed")
    print(f"[solve] ram stroke complete after {ram_steps} steps (ejected={ejected}, "
          f"push {f_push:.2f} N)", flush=True)
    s2 = print_score("P2 entry + ram stroke (contact) + settle")
    assert s2 >= s1 - 1e-6, "score decreased across the ram stroke"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after ram+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
