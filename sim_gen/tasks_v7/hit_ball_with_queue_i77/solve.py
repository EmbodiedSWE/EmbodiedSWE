"""Teleport solution for SkywayBridgeScene (sim_gen task `hit_ball_with_queue_i77`) —
the task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. TRANSPORT THE BRIDGE (teleport): one pose write stages the 16 cm bridge span
   HOVERING 4 cm above its seat, axis-aligned with the track, with a deliberate 6 mm
   lateral offset (inside the guide slot, off the perfect centre — the seating must
   tolerate a realistic place, not a magic one). Free-space transport only: the span
   is NOT written into the seated state.
2. SEAT THE BRIDGE (contact dynamics — never teleported into place): the span is
   RELEASED and falls onto the two recessed support tabs under gravity, between the
   lateral guide walls; it lands, rocks, and settles into bearing on both tabs — the
   seating contact, the bearing heights and the final alignment are all physics. The
   scene's own `_bridge_seated` readback (pose + axes + stillness) must confirm it.
3. RELEASE THE BALL (contact dynamics): a velocity-limited horizontal force at the
   ball's CoM (the fingertip-nudge emulation of the arm's push — the 90 mm ball is
   wider than the jaws and can never be grasped) walks it over the 8 mm detent
   ridge. The force is CUT the moment the ball clears the ridge; a runtime probe
   toggles the wrench pre-encoding if the pod's external-force API drags the frame
   (both encodings are tried against measured progress).
4. HANDS-OFF DESCENT (pure physics — the heart of the task): from the force cut to
   the verdict NOTHING touches any body. The ball rolls down slope 1, rides across
   the seated bridge (the crossing latch requires the bridge seated UNDER the moving
   ball), rolls down slope 2, flies off the lip over the basin's low near wall,
   lands inside and settles. Every centimetre of the payload's journey to the goal
   is gravity and contact.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.skyway_bridge")().build(num_envs=args.num_envs,
                                                   device=device)
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

    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        bl = scene._local(scene.ball)[0]
        rl = scene._local(scene.bridge)[0]
        print(f"[solve] {tag:12s} | ball_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) bridge_loc=({float(rl[0]):+.3f},"
              f"{float(rl[1]):+.3f},{float(rl[2]):.3f}) "
              f"seated={bool(scene._bridge_seated()[0])} "
              f"seated_ever={bool(scene._seated_ever[0])} "
              f"crossed_ever={bool(scene._crossed_ever[0])} "
              f"in_basin={bool(scene._in_basin()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    sp = (scene.skyway.data.root_pos_w - scene.env_origins)[0]
    sq = scene.skyway.data.root_quat_w[0]
    s_yaw = 2.0 * math.atan2(float(sq[3]), float(sq[0]))
    bl0 = scene._local(scene.ball)[0]
    rl0 = scene._local(scene.bridge)[0]
    dl0 = scene._local(scene.decoy)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"skyway=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) "
          f"yaw={math.degrees(s_yaw):+.1f}deg "
          f"ball_loc=({float(bl0[0]):+.3f},{float(bl0[1]):+.3f}) "
          f"bridge_loc=({float(rl0[0]):+.3f},{float(rl0[1]):+.3f}) "
          f"decoy_loc=({float(dl0[0]):+.3f},{float(dl0[1]):+.3f})", flush=True)
    report("reset")
    assert float(bl0[2]) > c.deck_top, "ball did not spawn on the deck"
    assert not bool(scene._bridge_seated()[0]), "bridge spawned seated?!"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: transport the bridge (teleport, HOVER only) ------------------
    # One pose write: the span hovers 40 mm above its seat, axis-aligned, with a
    # deliberate 6 mm lateral offset (inside the guide slot). NOT written seated.
    s_pos = scene.skyway.data.root_pos_w
    s_quat = scene.skyway.data.root_quat_w
    hover_loc = torch.tensor([0.0, 0.006, c.seat_z + 0.040], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = s_pos + quat_apply(s_quat, hover_loc)
    st[:, 3:7] = s_quat  # axis-aligned with the track
    scene.bridge.write_root_state_to_sim(st, all_ids)
    step(2)
    assert not bool(scene._bridge_seated(require_still=False)[0]), \
        "hover pose already reads as seated (teleport must not seat the bridge)"
    s1 = print_score("P1 bridge transported to hover over the gap (not seated)")

    # ---------------- phase 2: SEAT THE BRIDGE (contact dynamics) ---------------------------
    # Release: the span falls onto the tabs and settles into bearing under gravity.
    seated = False
    for i in range(600):
        env.step(no_action)
        if bool(scene._bridge_seated()[0]):
            seated = True
            break
    report("seated")
    assert seated, "bridge did not settle into a seated bearing on both tabs"
    assert bool(scene._seated_ever[0]), "seating did not latch"
    s2 = print_score("P2 bridge dropped onto the tabs and settled seated")
    assert s2 >= s1 - 1e-6 and s2 >= 0.30 - 1e-6, "seating credit missing"

    # ---------------- phase 3: RELEASE THE BALL (contact dynamics) --------------------------
    # Velocity-limited horizontal force at the ball's CoM walks it over the detent
    # ridge; cut the moment it clears. Runtime probe: if the pod's force API drags
    # the wrench frame (rotation-since-reset), measured progress collapses — toggle
    # the pre-encoding and continue.
    u_dir = quat_apply(s_quat, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    q_ref = scene.ball.data.root_quat_w.clone()
    encode_mode = 0  # 0: raw world force; 1: pre-rotate by R_ref * R_now^T

    def encode(f_w: torch.Tensor) -> torch.Tensor:
        if encode_mode == 0:
            return f_w
        q_now = scene.ball.data.root_quat_w
        return quat_apply(quat_mul(q_ref, quat_conjugate(q_now)), f_w)

    f_push, v_des = 2.5, 0.30
    cut_u = c.ridge_u + 0.05  # past the ridge: gravity owns the rest
    best_u = float(scene._local(scene.ball)[0, 0])
    start_u, last_bump, stalls = best_u, 0, 0
    released = False
    for i in range(900):
        bl = scene._local(scene.ball)[0]
        u = float(bl[0])
        if u > cut_u:
            released = True
            break
        vel = scene.ball.data.root_lin_vel_w[0]
        u_vel = float((vel * u_dir[0]).sum())
        f_axis = f_push if u_vel < v_des else 0.0
        f_w = (u_dir * f_axis).view(n, 1, 3)
        scene.ball.set_external_force_and_torque(encode(f_w).contiguous(),
                                                 zero_wrench, env_ids=all_ids,
                                                 is_global=True)
        env.step(no_action)
        if u > best_u + 0.002:
            best_u, last_bump, stalls = u, i, 0
        if u < start_u - 0.01 and encode_mode == 0:
            encode_mode = 1  # pushed the wrong way: the pod drags the wrench frame
            print(f"[solve] frame-drag detected (u={u:+.3f} < start) -> "
                  f"pre-encode mode 1", flush=True)
            last_bump = i
        elif i - last_bump > 120:  # stalled against the ridge
            stalls += 1
            last_bump = i
            if stalls >= 2:
                # escalation did not help: a 1 N quasi-static climb needing > 4 N
                # means the wrench frame is dragged (the ball rotated ~70 deg
                # rolling to the ridge) — toggle the pre-encoding and retry gently
                encode_mode ^= 1
                f_push, stalls = 2.5, 0
                print(f"[solve] stall persists at u={u:+.3f} -> toggle pre-encode "
                      f"to mode {encode_mode}", flush=True)
            else:
                f_push = min(f_push + 0.8, 6.0)
                print(f"[solve] stall at u={u:+.3f} -> f_push={f_push:.1f} N",
                      flush=True)
    scene.ball.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    report("released")
    assert released, "nudge never carried the ball over the detent ridge"
    s3 = print_score("P3 ball nudged over the ridge, force cut — hands-off from here")
    assert s3 >= s2 - 1e-6, "score decreased across the release"

    # ---------------- phase 4: HANDS-OFF DESCENT (pure physics) -----------------------------
    # Roll, cross, fly, land, settle: nothing touches anything.
    quiet = 0
    landed = False
    for i in range(1440):  # up to 12 simulated seconds
        env.step(no_action)
        if i % 120 == 0:
            report(f"coast+{i}")
        still = (float(scene.ball.data.root_lin_vel_w[0].norm()) < c.settle_speed
                 and float(scene.ball.data.root_ang_vel_w[0].norm()) < c.settle_omega)
        quiet = quiet + 1 if (still and bool(scene._in_basin()[0])) else 0
        if quiet >= 30:
            landed = True
            break
    report("descended")
    assert bool(scene._crossed_ever[0]), \
        "the ball never crossed the gap over the seated bridge"
    assert landed and bool(scene._in_basin()[0]), \
        "the ball did not settle inside the basin"
    s4 = print_score("P4 gravity descent complete: crossed the bridge, settled in basin")
    assert s4 >= s3 - 1e-6, "score decreased across the descent"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, no intervention) ------
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
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
