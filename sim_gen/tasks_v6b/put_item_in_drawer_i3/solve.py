"""Teleport solution for StopperVaultScene (sim_gen task `put_item_in_drawer_i3`) —
the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, cube): one root-state write carries the green cube from its
   ground spawn to a staging pose on the doorway axis, ~10 cm OUTSIDE the mouth on
   the ground — far outside the containment region, so the freshly-written state
   satisfies no deposit clause (only approach credit moves, as any real carry would).
2. PUSH-THROUGH (contact dynamics): the cube is driven inward by a horizontal
   external force at its CoM (world frame, aligned with the vault's doorway axis,
   velocity-regulated bang-bang with stall escalation, small lateral centring force)
   and must physically slide along the ground THROUGH the 104 mm doorway tunnel and
   past the stop ribs' 80 mm inner exit into the interior. No pose write crosses the
   doorway: the aperture, the tunnel walls and the ground do the constraining. The
   push stops just past the containment threshold — the depth a Franka fingertip
   reaching ~40 mm through the aperture would leave the cube at.
3. TRANSPORT (teleport, blue stopper): one root-state write carries the BLUE stopper
   to a staging pose upright on the doorway axis, slab facing the tunnel, its inner
   face ~5 mm outside the mouth — outside the seat band, so the freshly-written
   state satisfies no seating clause.
4. SEAT (contact dynamics): the stopper is driven inward by the same force scheme
   and must physically slide into the pocket — 6 mm of lateral clearance — until the
   stop ribs ARREST it (the arrest is the seating event; the final pose is decided
   by rib contact, not by any write).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_item_in_drawer_i3.solve --headless [--seed N]
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
    env = ENVS.get("simgen.stopper_vault")().build(num_envs=args.num_envs, device=device)
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

    def vault_pose() -> tuple[torch.Tensor, float]:
        vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
        q = scene.vault.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return vp, yaw

    def report(tag: str) -> None:
        cl = scene._local(scene.cube)[0]
        bl = scene._local(scene.blue)[0]
        print(f"[solve] {tag:12s} | cube_v=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):.3f}) blue_v=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) inside={bool(scene._cube_inside_now()[0])} "
              f"seated={bool(scene._blue_seated_now()[0])} "
              f"dep={bool(scene._deposited[0])} seal={float(scene._seal_max[0]):.3f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force(obj) -> None:
        obj.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def to_world(x_l: float, y_l: float) -> tuple[float, float, float]:
        vp, vyaw = vault_pose()
        wx = float(vp[0]) + math.cos(vyaw) * x_l - math.sin(vyaw) * y_l
        wy = float(vp[1]) + math.sin(vyaw) * x_l + math.cos(vyaw) * y_l
        return wx, wy, vyaw

    def push_in(obj, target_x: float, tag: str, v_des: float, f0: float,
                f_max: float, max_steps: int = 2400) -> None:
        """Drive `obj` inward along the vault's doorway axis with a world-frame
        horizontal force at its CoM (velocity-regulated bang-bang, stall escalation)
        plus a small lateral centring force, until its vault-frame x drops below
        `target_x`. Contact dynamics only — the tunnel walls, stop ribs and ground do
        the constraining; no pose write ever crosses the doorway."""
        _vp, vyaw = vault_pose()
        in_dir = torch.tensor([-math.cos(vyaw), -math.sin(vyaw), 0.0], device=device)
        lat_dir = torch.tensor([-math.sin(vyaw), math.cos(vyaw), 0.0], device=device)
        f_push = f0
        best, last_bump = -1e9, 0
        for i in range(max_steps):
            loc = scene._local(obj)[0]
            x_l = float(loc[0])
            if x_l < target_x:
                break
            v_axis = float((obj.data.root_lin_vel_w[0] * in_dir).sum())
            f_axis = f_push if v_axis < v_des else 0.0
            f_lat = max(-0.6, min(0.6, -8.0 * float(loc[1])))
            f_w = (in_dir * f_axis + lat_dir * f_lat).view(1, 1, 3).expand(n, 1, 3)
            obj.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                              env_ids=all_ids, is_global=True)
            env.step(no_action)
            prog = -x_l
            if prog > best + 0.003:
                best, last_bump = prog, i
            elif i - last_bump > 240:  # stalled: push harder (friction underestimated)
                f_push = min(f_push + 1.5, f_max)
                last_bump = i
                print(f"[solve] {tag}: stalled at x_v={x_l:+.3f}, raising force to "
                      f"{f_push:.1f} N", flush=True)
        clear_force(obj)
        step(60)  # coast + settle on real friction

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    vp, vyaw = vault_pose()
    cube0 = (scene.cube.data.root_pos_w - scene.env_origins)[0]
    blue0 = (scene.blue.data.root_pos_w - scene.env_origins)[0]
    red0 = (scene.red.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): vault=({float(vp[0]):+.3f},"
          f"{float(vp[1]):+.3f}) yaw={math.degrees(vyaw):+.1f}deg "
          f"cube=({float(cube0[0]):+.3f},{float(cube0[1]):+.3f}) "
          f"blue=({float(blue0[0]):+.3f},{float(blue0[1]):+.3f}) "
          f"red=({float(red0[0]):+.3f},{float(red0[1]):+.3f}) "
          f"d0_cube={float(scene._d0_cube[0]):.3f} d0_blue={float(scene._d0_blue[0]):.3f}",
          flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.02, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT cube (teleport across free ground only) -----------
    # One pose write carries the cube to the doorway axis, ~10 cm OUTSIDE the mouth,
    # standing on the ground, faces square to the vault — far outside containment.
    wx, wy, vyaw = to_world(c.mouth_x + 0.105, 0.0)
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.cube_s / 2 + 0.003
    st[:, 3], st[:, 6] = math.cos(vyaw / 2), math.sin(vyaw / 2)
    st[:, 0:3] += scene.env_origins
    scene.cube.write_root_state_to_sim(st, all_ids)
    step(30)
    assert not bool(scene._cube_inside_now()[0]), \
        "transport must not place the cube inside the vault"
    report("staged-cube")
    s1 = print_score("P1 transport cube to the doorway axis (outside)")
    assert s1 >= s0 - 1e-6, "score decreased across cube transport"

    # ---------------- phase 2: PUSH-THROUGH the doorway (contact dynamics) -----------------
    # Target 0.068: below inside_x_max 0.075 with margin, and reachable by a Franka
    # fingertip (cube rear face at 0.093, i.e. 42 mm past the mouth at 0.135).
    push_in(scene.cube, 0.068, "push-through", v_des=0.10, f0=3.0, f_max=8.0)
    report("deposited")
    assert bool(scene._cube_inside_now()[0]), "cube did not end fully inside the vault"
    assert bool(scene._deposited[0]), "deposit latch did not set"
    assert not bool(scene.success()[0]), "cannot be success before the stopper is seated"
    s2 = print_score("P2 contact-dynamics push-through the doorway")
    assert s2 >= s1 - 1e-6, "score decreased across push-through"

    # ---------------- phase 3: TRANSPORT blue stopper (teleport, staging only) --------------
    # Upright on the doorway axis, slab facing the tunnel (local +x = vault +x, boss
    # outward), slab inner face ~5 mm outside the mouth — outside the seat band.
    stage_x = c.mouth_x + 0.005 + c.slab_d / 2
    wx, wy, vyaw = to_world(stage_x, 0.0)
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = wx, wy, c.slab_h / 2 + 0.002
    st[:, 3], st[:, 6] = math.cos(vyaw / 2), math.sin(vyaw / 2)
    st[:, 0:3] += scene.env_origins
    scene.blue.write_root_state_to_sim(st, all_ids)
    step(30)
    assert not bool(scene._blue_seated_now()[0]), \
        "transport must not place the stopper inside the seat band"
    report("staged-blue")
    s3 = print_score("P3 transport blue stopper to the mouth (outside the seat band)")
    assert s3 >= s2 - 1e-6, "score decreased across stopper transport"

    # ---------------- phase 4: SEAT — slide into the pocket until the ribs arrest ----------
    push_in(scene.blue, c.seat_x + 0.002, "seat", v_des=0.06, f0=2.5, f_max=6.0,
            max_steps=1800)
    report("seated")
    assert bool(scene._blue_seated_now()[0]), "blue stopper did not seat in the pocket"
    s4 = print_score("P4 contact-dynamics seat against the stop ribs")
    assert s4 >= s3 - 1e-6, "score decreased across seating"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seating)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

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
