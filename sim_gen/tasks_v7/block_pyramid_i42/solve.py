"""Contact-dynamics solution for TiltLabyrinthScene (sim_gen task `block_pyramid_i42`)
— the task's legitimacy certificate.

There are NO teleports at all: the ball spawns on the tray at reset and is NEVER
written again. Every effect on the ball is earned through contact dynamics — the only
actuation is a bounded external torque on the TRAY body (capped at 0.40 N m, the
moment a fingertip pressing an amber rim tab at radius 0.228 m with <= 1.8 N applies).
A nested controller does the closed-loop plate control the task demands:
  - OUTER loop (ball position, tray frame): desired downhill direction
    d = clamp(kp*err - kd*vel, sin 5 deg) toward the current waypoint.
  - INNER loop (tray attitude): torque about the base's horizontal axes drives the
    tray's up-vector toward d against the bottom-heavy self-levelling moment
    (gravity feedforward kg*d), with angular-rate damping.
The waypoint route is read from the SCENE'S OWN frames (ball_tray_local / describe
readback), so it works at any randomized apparatus yaw: east along the start leg,
north through the first gap, WEST along the middle leg hugging the south wall to skirt
the red trap hole, north through the second gap, east along the last leg, and over the
green goal hole. The drop through the hole and the landing in the catch box are pure
gravity + contact; the torque is then cut and the tray self-levels.

Known forge quirk (documented in project memory): is_global external wrenches are
dragged by the body's rotation since reset. Reset writes the full yawed orientation,
so the residual drag here is only the tilt (<= 7 deg) — absorbed by the feedback loop.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.block_pyramid_i42.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tilt_labyrinth")().build(num_envs=args.num_envs, device=device)
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

    def clear_wrench() -> None:
        scene.tray.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        p = scene.ball_tray_local()[0]
        pb = scene.ball_base_local()[0]
        u = scene.tray_up_base()[0]
        print(f"[solve] {tag:12s} | ball_tray=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) ball_base_z={float(pb[2]):.3f} "
              f"u_xy=({float(u[0]):+.3f},{float(u[1]):+.3f}) "
              f"mid={bool(scene.lat_mid[0])} north={bool(scene.lat_north[0])} "
              f"near={bool(scene.lat_near[0])} trap={bool(scene.in_trap_box()[0])} "
              f"goal={bool(scene.in_goal_box()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def fail(msg: str) -> None:
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL ({msg})", flush=True)
        os._exit(1)

    # ---------------- controller ------------------------------------------------------------
    kp_b, kd_b = 0.9, 0.55          # outer: ball position -> desired downhill (tray frame)
    kt, kg, kdw = 3.0, 1.0, 0.15    # inner: attitude P, gravity feedforward, rate damping
    d_max = math.sin(math.radians(5.0))
    tau_cap = 0.40                  # N m (<= 1.8 N fingertip press at the 0.228 m tabs)

    def control_step(wp: tuple) -> None:
        """One physics step under the nested tilt controller steering toward `wp`."""
        qb = scene.base.data.root_quat_w
        p = scene.ball_tray_local()[0]
        v = quat_apply_inverse(qb, scene.ball.data.root_lin_vel_w)[0]
        u = scene.tray_up_base()[0]
        w = quat_apply_inverse(qb, scene.tray.data.root_ang_vel_w)[0]
        ex, ey = float(wp[0] - p[0]), float(wp[1] - p[1])
        dx = kp_b * ex - kd_b * float(v[0])
        dy = kp_b * ey - kd_b * float(v[1])
        dn = math.hypot(dx, dy)
        if dn > d_max:
            dx, dy = dx * d_max / dn, dy * d_max / dn
        # up-vector mapping: +tau_y raises u_x, -tau_x raises u_y
        tau_y = kt * (dx - float(u[0])) + kg * dx - kdw * float(w[1])
        tau_x = -kt * (dy - float(u[1])) - kg * dy - kdw * float(w[0])
        tau_x = max(-tau_cap, min(tau_cap, tau_x))
        tau_y = max(-tau_cap, min(tau_cap, tau_y))
        tq_b = torch.tensor([tau_x, tau_y, 0.0], device=device).unsqueeze(0)
        tq_w = quat_apply(qb, tq_b).view(n, 1, 3)
        scene.tray.set_external_force_and_torque(
            zero_wrench, tq_w.contiguous(), env_ids=all_ids, is_global=True)
        env.step(no_action)

    def drive_to(wp: tuple, tol: float, max_steps: int, tag: str, until=None) -> bool:
        """Steer toward `wp` until within `tol` (and, if given, `until()` also holds —
        used to gate leg completion on the scene's own latches, not just proximity)."""
        for i in range(max_steps):
            control_step(wp)
            p = scene.ball_tray_local()[0]
            close = math.hypot(float(wp[0] - p[0]), float(wp[1] - p[1])) < tol
            if close and (until is None or bool(until())):
                print(f"[solve] waypoint {tag} reached in {i + 1} steps", flush=True)
                return True
            if bool(scene.in_trap_box()[0]):
                fail("ball fell into the red trap box")
            if float(p[2]) < c.floor_top_local - 0.012 and not bool(scene.in_goal_box()[0]):
                # fell somewhere unexpected (goal handled by the caller of the last leg)
                fail(f"ball left the floor unexpectedly en route to {tag}")
        print(f"[solve] waypoint {tag} TIMEOUT (continuing)", flush=True)
        return False

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    bp = (scene.base.data.root_pos_w - scene.env_origins)[0]
    qb0 = scene.base.data.root_quat_w[0]
    yaw = math.atan2(2 * (float(qb0[0]) * float(qb0[3])), 1 - 2 * float(qb0[3]) ** 2)
    p0 = scene.ball_tray_local()[0]
    print(f"[solve] layout readback (seed {args.seed}): base=({float(bp[0]):+.3f},"
          f"{float(bp[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg "
          f"ball_tray=({float(p0[0]):+.3f},{float(p0[1]):+.3f},{float(p0[2]):+.3f})",
          flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: start leg -> first gap -> middle corridor --------------------
    drive_to((0.160, -0.140), 0.030, 1500, "L1 east end of start leg")
    drive_to((0.163, -0.040), 0.030, 1500, "L2 through gap, hug south wall",
             until=lambda: scene.lat_mid[0])
    if not bool(scene.lat_mid[0]):
        fail("middle corridor not latched after the first gap")
    report("mid-corridor")
    s1 = print_score("P1 first gap -> middle corridor")
    assert s1 >= s0 - 1e-6, "score decreased entering the middle corridor"

    # ---------------- phase 2: past the trap -> second gap -> north corridor ----------------
    drive_to((-0.145, -0.045), 0.030, 1800, "L3 west past the trap (wall hug)")
    drive_to((-0.160, 0.140), 0.030, 1500, "L4 through gap into north corridor",
             until=lambda: scene.lat_north[0])
    if not bool(scene.lat_north[0]):
        fail("north corridor not latched after the second gap")
    report("north")
    s2 = print_score("P2 trap skirted -> north corridor")
    assert s2 >= s1 - 1e-6, "score decreased crossing the middle corridor"

    # ---------------- phase 3: goal approach -> drop through the green hole -----------------
    drive_to((0.090, 0.148), 0.030, 1500, "L5 east along the last leg")
    dropped = False
    for i in range(1500):
        control_step((c.goal_hole_c[0], c.goal_hole_c[1]))
        p = scene.ball_tray_local()[0]
        if float(p[2]) < c.floor_top_local - 0.012 or bool(scene.in_goal_box()[0]):
            dropped = True
            break
        if bool(scene.in_trap_box()[0]):
            fail("ball fell into the red trap box during the goal approach")
    clear_wrench()
    print(f"[solve] goal-hole drop detected={dropped} (steps={i + 1}); torque cut, "
          "tray self-levels, gravity finishes the job", flush=True)
    # gravity landing: fall into the catch box, settle on real contact
    for _ in range(20):
        step(30)
        if bool(scene.success()[0]):
            break
    step(60)
    report("landed")
    if bool(scene.in_trap_box()[0]):
        fail("ball ended in the trap box")
    s3 = print_score("P3 drop through the goal hole + settle")
    assert s3 >= s2 - 1e-6, "score decreased across the drop"
    if not bool(scene.success()[0]):
        fail("no success after the drop")

    # ---------------- phase 4: persistence (>= 3.4 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.42 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.4 s")
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - die fast, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)
