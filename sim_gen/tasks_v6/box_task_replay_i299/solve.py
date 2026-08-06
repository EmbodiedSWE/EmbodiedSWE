"""Teleport solution for FlapPostboxScene (sim_gen task `box_task_replay_i299`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): pose writes move each parcel ACROSS THE PORCH ONLY — from its
   spawn slot to the start of the push runway in front of the doorway (and the waiting
   parcel to a porch corner, out of the lane). Every teleport endpoint is OUTSIDE the
   postbox on the open porch: no containment, no doorway crossing, and no rubric gate
   is satisfied by any of these writes (the tiny approach term is transport progress by
   definition and is capped at 0.05/parcel).
2. FLAP PASSAGE (contact dynamics — the core interaction; teleporting past it would
   bypass the task and is never done): a regulated horizontal force (velocity-servoed,
   capped at 2 N) pushes the parcel along the porch INTO the hanging flap. The flap
   yields inward about its real hinge under the contact force, the parcel crosses the
   sill under the swinging plate, tips over the inner edge, and free-falls 70 mm into
   the interior pit. The force is CUT the moment the parcel crosses the inner face (or
   starts falling); landing, tumbling and settling are pure physics, and the flap falls
   shut again by gravity. The solve ASSERTS the flap actually swung open (> 15 deg)
   during each passage — the interaction demonstrably happened.
3. Repeat for the second parcel; then hands off: wait for both parcels, then the flap,
   to settle. success() judges settled containment + the flap hanging shut.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.box_task_replay_i299.solve --headless [--seed N]
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
    env = ENVS.get("simgen.flap_postbox")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    bx, by = c.box_pos

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        p_can, p_blk = loc(scene.can), loc(scene.block)
        print(f"[solve] {tag:12s} | can=({float(p_can[0]):+.3f},{float(p_can[1]):+.3f},"
              f"{float(p_can[2]):.3f}) block=({float(p_blk[0]):+.3f},"
              f"{float(p_blk[1]):+.3f},{float(p_blk[2]):.3f}) "
              f"flap={math.degrees(float(scene.flap_angle()[0])):+.1f}deg "
              f"in_can={bool(scene._inside(scene.can)[0])} "
              f"in_blk={bool(scene._inside(scene.block)[0])} "
              f"shut={bool(scene.flap_shut()[0])} "
              f"lat_in={scene._in[0].tolist()} shutld={bool(scene._shut_loaded[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def place(body, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        """Transport teleport: pose write to an open-porch pose (zero velocity)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def push_through(body, name: str) -> float:
        """Push the parcel from the runway INTO and THROUGH the flap with a regulated
        world-frame force; cut the force once it crosses the inner face or starts
        falling; let it land and settle under pure physics. Returns the max flap
        opening seen (deg) — the proof the flap interaction physically happened."""
        v_des, flap_max = 0.25, 0.0
        pushed_steps = 0
        for _ in range(900):
            p = loc(body)
            flap_max = max(flap_max, abs(math.degrees(float(scene.flap_angle()[0]))))
            if float(p[0]) > bx + c.wall_t + 0.030 or float(p[2]) < 0.050:
                break
            v = body.data.root_lin_vel_w[0]
            # force caps sized to displace the flap yet stay under the standing
            # parcels' quasi-static tipping accel (~6-8 m/s^2)
            fx = max(-0.5, min(0.9, 4.0 * (v_des - float(v[0]))))
            fy = max(-0.8, min(0.8, 5.0 * (by - float(p[1])) - 1.0 * float(v[1])))
            f_w = torch.tensor([fx, fy, 0.0], device=device).view(1, 1, 3).expand(n, 1, 3)
            body.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                               env_ids=all_ids, is_global=True)
            env.step(no_action)
            pushed_steps += 1
        clear_wrench(body)
        # hands off: land, tumble, settle under gravity + real contacts
        quiet = 0
        for _ in range(600):
            env.step(no_action)
            still = float(body.data.root_lin_vel_w[0].norm()) < 0.04
            quiet = quiet + 1 if still else 0
            if quiet >= 30:
                break
        print(f"[solve] {name}: pushed {pushed_steps} steps, flap swung to "
              f"{flap_max:.1f} deg during passage", flush=True)
        return flap_max

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p_can, p_blk = loc(scene.can), loc(scene.block)
    p_crate = loc(scene.crate)
    print(f"[solve] layout readback (seed {args.seed}): can=({float(p_can[0]):+.3f},"
          f"{float(p_can[1]):+.3f}) block=({float(p_blk[0]):+.3f},{float(p_blk[1]):+.3f}) "
          f"crate=({float(p_crate[0]):+.3f},{float(p_crate[1]):+.3f}) "
          f"door=({c.door_center[0]:.3f},{c.door_center[1]:.3f},{c.door_center[2]:.3f})",
          flush=True)
    report("reset")
    assert float(p_can[2]) > c.sill_h - 0.01 and float(p_blk[2]) > c.sill_h - 0.01, \
        "parcels did not settle standing on the porch"
    assert bool(scene.flap_shut()[0]), "flap did not hang shut at reset"
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: TRANSPORT (teleports on the open porch only) ----------------
    # The waiting block moves to a porch corner clear of the push lane; the can moves to
    # the runway start in front of the doorway. Both endpoints are on the open porch —
    # outside the postbox, touching nothing, satisfying no containment gate.
    place(scene.block, bx - 0.26, 0.145, c.sill_h + c.block_h / 2 + 0.003)
    place(scene.can, bx - 0.17, by, c.sill_h + c.can_h / 2 + 0.003)
    step(30)
    report("staged-can")
    s1 = print_score("P1 transport: can to the runway, block parked aside")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: push the CAN through the flap (contact dynamics) -------------
    flap_deg = push_through(scene.can, "can")
    report("can-delivered")
    assert flap_deg > 15.0, "flap never swung open during the can's passage"
    assert bool(scene._inside(scene.can)[0]), "can did not end up inside the postbox"
    s2 = print_score("P2 can pushed through the flap (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across the can delivery"

    # ---------------- phase 3: TRANSPORT + push the BLOCK through the flap ------------------
    place(scene.block, bx - 0.17, by, c.sill_h + c.block_h / 2 + 0.003)
    step(30)
    report("staged-block")
    s3a = print_score("P3a transport: block to the runway")
    assert s3a >= s2 - 1e-6, "score decreased across transport"
    flap_deg = push_through(scene.block, "block")
    report("block-delivered")
    assert flap_deg > 15.0, "flap never swung open during the block's passage"
    assert bool(scene._inside(scene.block)[0]), "block did not end up inside the postbox"
    s3 = print_score("P3b block pushed through the flap (contact)")
    assert s3 >= s3a - 1e-6, "score decreased across the block delivery"

    # ---------------- phase 4: hands off — flap falls shut, everything settles --------------
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        ok_now = (bool(scene.flap_shut()[0]) and bool(scene._parcels_still()[0])
                  and float(scene.flap.data.root_ang_vel_w[0].norm()) < c.settle_omega)
        quiet = quiet + 1 if ok_now else 0
        if quiet >= 30:
            break
    report("settled")
    s4 = print_score("P4 flap shut + everything settled")
    assert s4 >= s3 - 1e-6, "score decreased across the settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after both deliveries)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) --------
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
