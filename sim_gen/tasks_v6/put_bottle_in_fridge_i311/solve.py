"""Teleport solution for ChillRackScene (sim_gen task `put_bottle_in_fridge_i311`) —
the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, the only pose write on the bottle): one root-state write
   carries the amber bottle from its upright ground spawn across open air to a release
   pose LYING (axis along the trough) 20 mm ABOVE the drawn-out rack's cradle — above
   the trough z-band, so the freshly-teleported state earns no credit and cannot
   satisfy success(). Both endpoints are in free space; the reorientation happens in
   free air, which is exactly what a wrist rotation does.
2. LAY / BED-DOWN (contact dynamics): gravity drops the bottle the last 20 mm; it
   lands between the cradle ridges, rolls, beds down against bed + ridges and settles
   under physics. The `laid` latch first sets here, from real resting contact.
3. RIDE (contact dynamics, no teleport): the captive rack — with the bottle aboard,
   coupled to it ONLY by gravity, friction, ridges and chock — is driven inward by a
   horizontal external force at its CoM (world frame, aligned with the locker's
   channel axis, velocity-regulated bang-bang with stall escalation) through the
   letterbox mouth until it seats against the back wall. The bottle must physically
   ride the whole 20+ cm; if it toppled or slipped off, success() could never turn
   True. No pose write ever touches the rack.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.put_bottle_in_fridge_i311.solve --headless [--seed N]
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
    env = ENVS.get("simgen.chill_rack")().build(num_envs=args.num_envs, device=device)
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

    def locker_pose() -> tuple[torch.Tensor, float]:
        lp = (scene.locker.data.root_pos_w - scene.env_origins)[0]
        q = scene.locker.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return lp, yaw

    def report(tag: str) -> None:
        t = float(scene._travel()[0])
        bl = scene._bottle_in_rack()[0]
        print(f"[solve] {tag:12s} | travel={t:+.3f} bottle_rack=({float(bl[0]):+.3f},"
              f"{float(bl[1]):+.3f},{float(bl[2]):.3f}) in_trough={bool(scene._in_trough_now()[0])} "
              f"laid={bool(scene._laid[0])} ride={float(scene._ride_max[0]):.3f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force() -> None:
        scene.rack.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def push_rack(target_t: float, tag: str, v_des: float = 0.08,
                  max_steps: int = 2400) -> None:
        """Drive the rack inward along its channel with a world-frame horizontal force
        at its CoM (velocity-regulated bang-bang, stall escalation) until its travel
        drops below `target_t`. Contact dynamics only — the bottle rides through
        friction/ridges/chock; no pose write ever touches the rack."""
        _lp, lyaw = locker_pose()
        push_dir = torch.tensor([-math.cos(lyaw), -math.sin(lyaw), 0.0], device=device)
        f_push = 4.0
        best, last_bump = -1e9, 0
        for i in range(max_steps):
            t = float(scene._travel()[0])
            if t < target_t:
                break
            v_axis = float((scene.rack.data.root_lin_vel_w[0] * push_dir).sum())
            f_axis = f_push if v_axis < v_des else 0.0
            f_w = (push_dir * f_axis).view(1, 1, 3).expand(n, 1, 3)
            scene.rack.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                     env_ids=all_ids, is_global=True)
            env.step(no_action)
            prog = -t
            if prog > best + 0.004:
                best, last_bump = prog, i
            elif i - last_bump > 240:  # stalled: push harder (friction underestimated)
                f_push = min(f_push + 2.0, 12.0)
                last_bump = i
                print(f"[solve] {tag}: rack stalled at travel={t:+.3f}, raising force "
                      f"to {f_push:.1f} N", flush=True)
        clear_force()
        step(40)  # coast + settle on real friction

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    lp, lyaw = locker_pose()
    bot0 = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
    can0 = (scene.can.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): locker=({float(lp[0]):+.3f},"
          f"{float(lp[1]):+.3f}) yaw={math.degrees(lyaw):+.1f}deg "
          f"t0={float(scene._t0[0]):.3f} travel={float(scene._travel()[0]):+.3f} "
          f"bottle=({float(bot0[0]):+.3f},{float(bot0[1]):+.3f}) "
          f"can=({float(can0[0]):+.3f},{float(can0[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"

    # ---------------- phase 1: TRANSPORT (teleport across free air only) -------------------
    # One pose write carries the bottle from its upright ground spawn to a LYING pose
    # (axis along the trough, neck outward) 20 mm ABOVE the drawn-out rack's cradle —
    # above the trough z-band, so the freshly-written state earns no credit.
    from isaaclab.utils.math import quat_apply

    rack_pos = scene.rack.data.root_pos_w.clone()
    rack_q = scene.rack.data.root_quat_w[0]
    ryaw = 2.0 * math.atan2(float(rack_q[3]), float(rack_q[0]))
    rel = torch.tensor([-0.005, 0.0, c.cradle_z + 0.020], device=device).expand(n, 3)
    target = rack_pos + quat_apply(scene.rack.data.root_quat_w, rel)
    cy, sy, c45 = math.cos(ryaw / 2), math.sin(ryaw / 2), math.cos(math.pi / 4)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = target
    # lying along the trough: q = qz(yaw) * qy(90 deg)
    st[:, 3], st[:, 4], st[:, 5], st[:, 6] = cy * c45, -sy * c45, cy * c45, sy * c45
    scene.bottle.write_root_state_to_sim(st, all_ids)
    assert not bool(scene._in_trough_now()[0]), \
        "transport must not place the bottle inside the trough band"
    report("transported")
    s1 = print_score("P1 transport to hover pose above the drawn-out cradle")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: LAY through gravity + contact --------------------------------
    step(120)  # 1 s: drop 20 mm, land between the ridges, bed down, settle
    report("laid")
    assert bool(scene._laid[0]), "bottle did not bed down into the cradle trough"
    assert not bool(scene.success()[0]), "cannot be success while the rack is drawn out"
    s2 = print_score("P2 gravity bed-down into the cradle")
    assert s2 >= s1 - 1e-6, "score decreased across lay"

    # ---------------- phase 3: RIDE — push the loaded rack in through the mouth ------------
    push_rack(0.006, "ride")
    # trim: if the rack rebounded off the back wall past tolerance, nudge it back
    for _ in range(3):
        if float(scene._travel()[0]) <= c.seat_tol - 0.002:
            break
        push_rack(0.006, "ride-trim", v_des=0.04, max_steps=240)
    report("seated")
    assert bool(scene._in_trough_now()[0]), "bottle fell out of the trough during the ride"
    s3 = print_score("P3 contact-dynamics ride to seat")
    assert s3 >= s2 - 1e-6, "score decreased across ride"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seating)", flush=True)
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
