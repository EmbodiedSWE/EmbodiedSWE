"""Teleport solution for RollerRelayScene (sim_gen task `pick_and_lift_i370`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT OF THE TOOL ROLLERS ONLY, always ending in FREE
SPACE: each 36 mm roller (jaw-sized, ~0.18 kg — exactly what a Franka would carry)
is moved from the staging area to a hover pose ~9 mm above the rubber channel and
dropped; later, spent rollers that have emerged CLEAR BEHIND the moving slab are
picked from free space and relaid ahead of it (the rolling relay), or carried back
to the staging area once the slab no longer needs them. The 7 kg judged SLAB is
NEVER teleported, lifted, or posed: every bit of its motion comes from CONTACT
DYNAMICS under a regulated horizontal push (<= 14 N at its centre — well inside
what a Franka pressing on the slab's rear face can do), riding on the rollers it
was given. The push force is a velocity servo (never a blind shove), with a light
lateral/yaw keeper, and it is CUT for parking, settling, and the whole persistence
window.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.pick_and_lift_i370.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.roller_relay_freight")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zeros3 = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)

    def slab_x() -> float:
        return float(scene.slab_pos()[0, 0])

    def slab_state() -> tuple:
        p = scene.slab_pos()[0]
        q = scene.slab.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        v = scene.slab.data.root_lin_vel_w[0]
        w = scene.slab.data.root_ang_vel_w[0]
        return p, yaw, v, w

    def report(tag: str) -> None:
        p, yaw, v, _ = slab_state()
        staged = int(scene.rollers_staged()[0].sum())
        rx = [round(float(x), 3) for x in scene.roller_pos()[0, :, 0]]
        print(f"[solve] {tag:12s} | slab=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) yaw={math.degrees(yaw):+.1f} vx={float(v[0]):+.3f} "
              f"staged={staged} roller_x={rx} aboard={bool(scene.aboard_now()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_wrench() -> None:
        scene.slab.set_external_force_and_torque(zeros3, zeros3, env_ids=all_ids,
                                                 is_global=True)

    def lay_roller(i: int, x: float, y: float = 0.0) -> None:
        """TRANSPORT teleport: roller i to a free-space hover ~9 mm above the mat,
        axis across the track, zero velocity. It seats by gravity + contact."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = x
        st[:, 1] = y
        st[:, 2] = c.mat_t + c.roller_r + 0.009
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.rollers[i].write_root_state_to_sim(st, all_ids)

    def park_roller(i: int, slot: int, side: float) -> None:
        """TRANSPORT teleport: spent roller back to the open staging floor."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = 0.10 + 0.09 * slot
        st[:, 1] = side * (c.stage_y + 0.06)
        st[:, 2] = c.roller_r + 0.006
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.rollers[i].write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p, yaw, _, _ = slab_state()
    side = float(scene.side[0])
    rpos = scene.roller_pos()[0]
    rollers = " ".join(f"r{i}=({float(q[0]):+.3f},{float(q[1]):+.3f})"
                       for i, q in enumerate(rpos))
    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)
    print(f"[solve] layout readback (seed {args.seed}): slab=({float(p[0]):+.3f},"
          f"{float(p[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg side={side:+.0f} "
          f"{rollers} x0={float(scene.x0[0]):+.3f}", flush=True)
    report("reset")
    prev = print_score("P0 reset+settle")

    # ---------------- phase 1: lay the three rollers across the channel --------------------
    lay_x = (0.035, 0.125, 0.215)
    for i, x in enumerate(lay_x):
        lay_roller(i, x)
        step(60)  # 0.5 s: drop 9 mm, seat on the rubber
    step(60)
    staged = scene.rollers_staged()[0]
    for i in range(c.n_rollers):  # one retry per roller if a seat was unlucky
        if not bool(staged[i]):
            lay_roller(i, lay_x[i])
            step(90)
    report("staged")
    if int(scene.rollers_staged()[0].sum()) < c.n_rollers:
        print("SIM_GEN_SOLVE: FAIL (rollers did not stage)", flush=True)
        os._exit(1)
    s = print_score("P1 rollers staged")
    assert s >= prev - 1e-6, "score decreased across staging"
    prev = s

    # ---------------- phases 2+3: push aboard, ride the relay, land on the dock ------------
    # Velocity-servo push at the slab centre (is_global): Fx tracks v_des with a 14 N
    # cap (slick dock needs ~4 N; rolling needs less), plus a light y/yaw keeper.
    # Spent rollers that emerge clear BEHIND the slab are relaid ahead of it while the
    # channel still needs them, then carried back to staging once the slab front is
    # over the goal dock (also keeps the pit from ever piling three rollers).
    parked = [False] * c.n_rollers
    park_slots = iter(range(c.n_rollers))
    aboard_reported = False
    stall = 0
    for it in range(4200):  # hard cap ~35 s sim
        p, yaw, v, w = slab_state()
        x = float(p[0])
        front, rear = x + c.slab_l / 2, x - c.slab_l / 2

        v_des = 0.10 if x < 0.38 else (0.06 if x < 0.55 else 0.04)
        # Gain 160 (not 70): at v_des=0.04 the old gain topped out at 2.8 N, under
        # the slick dock's ~4.1 N static breakaway -> servo gain stall short of goal.
        fx = max(-10.0, min(14.0, 160.0 * (v_des - float(v[0]))))
        fy = max(-5.0, min(5.0, -60.0 * float(p[1]) - 30.0 * float(v[1])))
        tz = max(-1.0, min(1.0, -4.0 * yaw - 1.5 * float(w[2])))
        f = torch.zeros(n, 1, 3, device=device)
        t = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0], f[:, 0, 1], t[:, 0, 2] = fx, fy, tz
        scene.slab.set_external_force_and_torque(f, t, env_ids=all_ids, is_global=True)
        env.step(no_action)

        # roller relay bookkeeping
        rp = scene.roller_pos()[0]
        for i in range(c.n_rollers):
            if parked[i]:
                continue
            rx, rz = float(rp[i, 0]), float(rp[i, 2])
            clear_behind = rx < rear - 0.012 and rz < 0.035
            if not clear_behind:
                continue
            if front < 0.44:  # channel still needs support ahead
                target = min(front + 0.055, c.mat_len - 0.030)
                if target > rx + 0.05:
                    lay_roller(i, target)
            else:  # slab front over the goal dock: carry the spent roller away
                park_roller(i, next(park_slots), side)
                parked[i] = True

        if not aboard_reported and bool(scene.aboard_now()[0]):
            aboard_reported = True
            report("aboard")
            s = print_score("P2 slab aboard the rollers")
            assert s >= prev - 1e-6, "score decreased across boarding"
            prev = s

        # parked against the bumper (or deep enough): stop pushing
        if x > 0.615 and abs(float(v[0])) < 0.010:
            stall += 1
            if stall >= 30:
                break
        else:
            stall = 0
    clear_wrench()
    step(180)  # 1.5 s settle, hands off
    report("landed")
    if not aboard_reported:
        print("SIM_GEN_SOLVE: FAIL (slab never rode the rollers)", flush=True)
        os._exit(1)

    if not bool(scene.success()[0]):
        step(240)  # grace: let everything finish settling
        report("grace")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (slab not settled flat on the goal dock)", flush=True)
        os._exit(1)
    s = print_score("P3 slab delivered on the goal dock")
    assert s >= prev - 1e-6, "score decreased across delivery"
    prev = s

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= prev - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
