"""Teleport solution for UmbrellaRailScene (sim_gen task
`put_umbrella_in_umbrella_stand_i72`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. The load-bearing interaction — hooking the
crook over the bar and the free-hang itself — goes through contact dynamics:
  1. CARRY (transport): the umbrella is teleported from the ground to a hover pose,
     crook UP, with the mouth of the hook aligned 50 mm ABOVE the bar. Nothing is in
     contact; the bar is outside the hook.
  2. HOOK (dynamics): a velocity-regulated vertical force (gravity feed-forward + a
     PD to -0.10 m/s, force at the CoM only — no torque, no orientation pinning)
     lowers the umbrella so the bar passes through the open mouth of the crook; the
     wrench is DROPPED the moment the bar reads inside the arc. The catch — arc
     inner surface meeting the bar — is pure contact.
  3. HANG (dynamics): hands off. Gravity swings the umbrella about the bar to its
     pendulum equilibrium (~14 deg shaft tilt) and damping settles it. The final
     state is carried entirely by the crook-on-bar contact; success() reads the
     suspension (bar in arc + crook up + tip clear of the ground + settled). The
     umbrella is never teleported into contact, never welded, never held at the end.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_umbrella_in_umbrella_stand_i72.solve --headless [--seed N]
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
    env = ENVS.get("simgen.umbrella_rail")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f_w: torch.Tensor) -> None:
        """Apply a WORLD force (n,3) to `body`, expressed in its CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every step while pushing."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        u = rel(scene.umbrella)
        r = rel(scene.rack)
        print(f"[solve] {tag:14s} | umb=({float(u[0]):+.3f},{float(u[1]):+.3f},"
              f"{float(u[2]):.3f}) rack=({float(r[0]):+.3f},{float(r[1]):+.3f})"
              f" eng_d={float(scene.engage_dist()[0]):.4f}"
              f" tip_h={float(scene.tip_height()[0]):.3f}"
              f" up={bool(scene.crook_up()[0])} eng={bool(scene.engaged()[0])}"
              f" set={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    u, ca, r = rel(scene.umbrella), rel(scene.cane), rel(scene.rack)
    q = scene.rack.data.root_quat_w[0]
    ryaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rack=({float(r[0]):+.3f},{float(r[1]):+.3f}) yaw={math.degrees(ryaw):+.1f} "
          f"umb=({float(u[0]):+.3f},{float(u[1]):+.3f},{float(u[2]):.3f}) "
          f"cane=({float(ca[0]):+.3f},{float(ca[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: CARRY (transport only) --------------------------------------
    # Hover pose: crook UP, hook plane perpendicular to the bar, arc center 50 mm
    # ABOVE the bar's nearest point — the bar is OUTSIDE the hook, nothing touches.
    from isaaclab.utils.math import quat_apply

    p_r = scene.rack.data.root_pos_w
    q_r = scene.rack.data.root_quat_w
    yaw = 2.0 * torch.atan2(q_r[:, 3], q_r[:, 0])
    half = yaw / 2
    zc = torch.zeros_like(half)
    q_u = torch.stack([torch.cos(half), zc, zc, torch.sin(half)], dim=-1)
    bar_c = p_r + quat_apply(q_r, torch.tensor([0.0, 0.0, c.bar_z],
                                               device=device).expand(n, 3))
    gap = 0.050
    c_tgt = bar_c + torch.tensor([0.0, 0.0, gap], device=device)
    arc_loc = torch.tensor(c.arc_center_local, device=device).expand(n, 3)
    pos_u = c_tgt - quat_apply(q_u, arc_loc)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = pos_u
    st[:, 3:7] = q_u
    scene.umbrella.write_root_state_to_sim(st, all_ids)
    xy_hold = pos_u[:, 0:2].clone()
    # hold against gravity for a moment (regulated, zero-velocity target) while the
    # lift latch reads the raised pose
    for _ in range(10):
        v = scene.umbrella.data.root_lin_vel_w
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = (c.umb_mass * 9.81 + 6.0 * (0.0 - v[:, 2])).clamp(0.0, 2 * c.umb_mass * 9.81)
        f[:, 0:2] = 4.0 * (xy_hold - scene.umbrella.data.root_pos_w[:, 0:2]) \
            - 2.0 * v[:, 0:2]
        f[:, 0:2] = f[:, 0:2].clamp(-1.5, 1.5)
        wrench(scene.umbrella, f)
        env.step(no_action)
    report("hover")
    s1 = print_score("P1 carried to hover above the bar (transport)")
    assert s1 >= s0 - 1e-6
    assert float(scene.engage_dist()[0]) > c.engage_tol, \
        "hover must start with the bar OUTSIDE the hook"

    # ---------------- phase 2: HOOK (contact dynamics) -------------------------------------
    # Velocity-regulated lowering: the bar passes through the open mouth of the
    # crook; the wrench is dropped the moment the bar reads inside the arc.
    caught = False
    for i in range(600):
        d = float(scene.engage_dist()[0])
        cz = float(scene._arc_center_w()[0, 2] - bar_c[0, 2])
        if d < 0.027 or cz < -0.020:
            caught = True
            break
        v = scene.umbrella.data.root_lin_vel_w
        f = torch.zeros(n, 3, device=device)
        f[:, 2] = (c.umb_mass * 9.81 + 6.0 * (-0.10 - v[:, 2])).clamp(
            0.0, 2 * c.umb_mass * 9.81)
        f[:, 0:2] = 4.0 * (xy_hold - scene.umbrella.data.root_pos_w[:, 0:2]) \
            - 2.0 * v[:, 0:2]
        f[:, 0:2] = f[:, 0:2].clamp(-1.5, 1.5)
        wrench(scene.umbrella, f)
        env.step(no_action)
    wrench(scene.umbrella, zero3.expand(n, 3))
    report("hooked")
    assert caught, "lowering never brought the bar into the crook arc"
    s2 = print_score("P2 bar lowered through the hook mouth (contact dynamics)")
    assert s2 >= s1 - 1e-6

    # ---------------- phase 3: HANG (hands off, gravity + contact) -------------------------
    for _ in range(32):  # up to 8 s of hands-off pendulum settling
        if bool(scene.success()[0]):
            break
        step(30)
    if bool(scene.success()[0]):
        step(240)  # 2 s more hands-off: let the last dregs of sway die well below
        # the stillness thresholds before opening the strict persistence window
    report("settled")
    s3 = print_score("P3 free hang settled (gravity + crook-on-bar contact)")
    assert s3 >= s2 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after hook+hang+settle)", flush=True)
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
                u = scene.umbrella
                print(f"[solve] persist flicker @step {i}: "
                      f"eng={bool(scene.engaged()[0])} up={bool(scene.crook_up()[0])} "
                      f"susp={bool(scene.suspended()[0])} "
                      f"lin={float(u.data.root_lin_vel_w[0].norm()):.4f} "
                      f"ang={float(u.data.root_ang_vel_w[0].norm()):.4f}", flush=True)
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
