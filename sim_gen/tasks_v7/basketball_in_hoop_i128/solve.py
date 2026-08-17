"""Teleport solution for CraterRunScene (sim_gen task `basketball_in_hoop_i128`) — the
task's legitimacy certificate.

This solve uses NO teleports at all: the ball starts in the foot bay already inside
the workspace, and every meter of its journey is closed-loop CONTACT/FORCE dynamics —
the stand-in for a Franka pushing the ungraspable 90 mm ball with its closed gripper:

  1. CLIMB A (dynamics): a horizontal velocity-servo force (world-frame, body-encoded
     per the house wrench convention, re-encoded every step because the ball tumbles)
     conveys the ball out of the blue foot bay and up the lower green incline against
     gravity to the pocket entry.
  2. TURN (dynamics): the same servo steers the ball around the 180-degree flat turn
     pocket into the upper lane.
  3. CLIMB B + LANDING (dynamics): up the upper incline and along the summit landing.
  4. LIP DROP (dynamics): a faster shove carries the ball over the 8 mm red lip; the
     wrench is CLEARED the moment the geometric basin window reads True, and gravity +
     contact finish the job. The final resting state is never spawned.

Servo notes: `enable_external_forces_every_iteration` is set in the scene; the gain
respects the one-substep wrench delay (kv*dt/m = 6/(120*0.25) = 0.20 << 1, escalated
cap 20 -> 0.67 < 1); on stall the GAIN escalates, not the force cap. The route is
read from the scene per episode (chirality + yaw + jitter) through the scene's
`canon_to_world`, so one canonical waypoint list serves both mirror twins.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.5 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.basketball_in_hoop_i128.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.crater_run")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def push(f_world: torch.Tensor) -> None:
        """World-frame force on the ball, encoded in its CURRENT body frame (the house
        wrench convention). The ball tumbles while rolling, so this is re-computed
        every single step."""
        from isaaclab.utils.math import quat_apply_inverse

        scene.ball.set_external_force_and_torque(
            quat_apply_inverse(scene.ball.data.root_link_quat_w,
                               f_world).unsqueeze(1),
            zero_rows, env_ids=all_ids)

    def clear_push() -> None:
        scene.ball.set_external_force_and_torque(zero_rows, zero_rows, env_ids=all_ids)

    def report(tag: str) -> None:
        p = scene.ball_canon()[0]
        print(f"[solve] {tag:12s} | canon=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f})"
              f" latches=({float(scene.climbA_latch[0]):.0f},"
              f"{float(scene.turn_latch[0]):.0f},{float(scene.climbB_latch[0]):.0f},"
              f"{float(scene.land_latch[0]):.0f})"
              f" in_basin={bool(scene.in_basin()[0])}"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def drive(waypoints: list[tuple[float, float]], tag: str, v_des: float,
              kv0: float = 6.0, cap: float = 4.0, max_steps: int = 3600,
              stop_in_basin: bool = False) -> bool:
        """Velocity-servo the ball through canonical-frame xy waypoints: horizontal
        world force F = kv * (v_des * dir - v_xy), norm-clamped, applied every step.
        The channel walls do the fine guidance. Stall -> escalate GAIN (kv), not the
        cap. Returns True when the last waypoint (or the basin, if `stop_in_basin`)
        is reached."""
        kv = kv0
        wp_i = 0
        last_dist, last_check = float("inf"), 0
        for i in range(max_steps):
            if stop_in_basin and bool(scene.in_basin()[0]):
                clear_push()
                print(f"[solve] drive {tag}: basin window entered @step {i}",
                      flush=True)
                return True
            p = scene.ball_canon()
            wp = waypoints[wp_i]
            dx, dy = wp[0] - float(p[0, 0]), wp[1] - float(p[0, 1])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 0.045:
                wp_i += 1
                last_dist, last_check = float("inf"), i
                if wp_i >= len(waypoints):
                    clear_push()
                    return True
                continue
            # canonical target -> world (chirality + yaw + jitter handled by scene)
            tgt_c = p.clone()
            tgt_c[:, 0] = wp[0]
            tgt_c[:, 1] = wp[1]
            tgt_w = scene.canon_to_world(tgt_c)
            d_w = tgt_w - scene.ball.data.root_pos_w
            d_w[:, 2] = 0.0
            d_w = d_w / d_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            v_w = scene.ball.data.root_lin_vel_w.clone()
            v_w[:, 2] = 0.0
            f = kv * (v_des * d_w - v_w)
            f_norm = f.norm(dim=-1, keepdim=True)
            f = f * (f_norm.clamp(max=cap) / f_norm.clamp_min(1e-9))
            push(f)
            env.step(no_action)
            if i - last_check >= 240:  # 2 s without a waypoint advance
                if last_dist - dist < 0.02:  # stalled: escalate GAIN, not cap
                    kv = min(kv * 1.6, 20.0)
                    print(f"[solve] drive {tag}: stall wp{wp_i} dist={dist:.3f}, "
                          f"kv->{kv:.1f}", flush=True)
                last_dist, last_check = dist, i
        clear_push()
        p = scene.ball_canon()[0]
        print(f"[solve] drive {tag}: TIMEOUT at wp{wp_i}, canon="
              f"({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})",
              flush=True)
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p0 = scene.ball_canon()[0]
    print(f"[solve] layout readback (seed {args.seed}): mirror="
          f"{float(scene.mirror[0]):+.0f} ball_canon=({float(p0[0]):+.3f},"
          f"{float(p0[1]):+.3f},{float(p0[2]):.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: convey up incline A (contact dynamics) ----------------------
    ok = drive([(-0.10, 0.0), (0.10, 0.0), (0.26, 0.0)], "climbA", v_des=0.18)
    report("climbA")
    assert ok, "ball never reached the pocket entry"
    s1 = print_score("P1 up the lower incline (force servo vs gravity)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.15 - 1e-6, "climbA stage credit missing"

    # ---------------- phase 2: around the turn pocket (contact dynamics) -------------------
    ok = drive([(0.29, 0.05), (0.29, 0.12), (0.25, 0.16)], "turn", v_des=0.15)
    report("turn")
    assert ok, "ball never rounded the turn pocket"
    s2 = print_score("P2 around the 180-degree turn pocket")
    assert s2 >= s1 - 1e-6 and s2 >= 0.30 - 1e-6, "turn stage credit missing"

    # ---------------- phase 3: up incline B onto the landing (contact dynamics) ------------
    ok = drive([(0.10, 0.16), (-0.10, 0.16), (-0.24, 0.16), (-0.29, 0.16)],
               "climbB", v_des=0.18)
    report("climbB")
    assert ok, "ball never reached the summit landing"
    s3 = print_score("P3 up the upper incline onto the summit landing")
    assert s3 >= s2 - 1e-6 and s3 >= 0.60 - 1e-6, "climbB/landing credit missing"

    # ---------------- phase 4: over the lip, release, settle (gravity + contact) -----------
    ok = drive([(-0.43, 0.16)], "lip", v_des=0.30, stop_in_basin=True)
    assert ok, "ball never crossed the lip into the basin"
    clear_push()
    for _ in range(16):  # up to 4 s hands-off settling on the basin floor
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s4 = print_score("P4 over the lip, at rest in the crater basin")
    assert s4 >= s3 - 1e-6, "score decreased across the drop"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after climb+turn+climb+drop)",
              flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.5 simulated seconds, hands-off) -----------
    hold, flickers = True, 0
    for i in range(420):  # 420 substeps = 3.5 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                p = scene.ball_canon()[0]
                print(f"[solve] persist flicker @step {i}: "
                      f"in_basin={bool(scene.in_basin()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"canon=({float(p[0]):+.3f},{float(p[1]):+.3f},"
                      f"{float(p[2]):+.3f}) "
                      f"blin={float(scene.ball.data.root_lin_vel_w[0].norm()):.4f} "
                      f"bang={float(scene.ball.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s5 = print_score("P5 persistence 3.5 s")
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
