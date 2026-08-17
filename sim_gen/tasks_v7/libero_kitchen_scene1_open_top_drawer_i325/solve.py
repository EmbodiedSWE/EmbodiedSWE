"""Teleport solution for RatchetRampScene (sim_gen task
`libero_kitchen_scene1_open_top_drawer_i325`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): each cargo ball is carried by a single root-state write from
   its ground bay to a free-space spot on the ground JUST DOWNHILL of the channel
   mouth, zero velocity. The write satisfies no rubric clause by itself: the ball is
   at ground level, outside the channel, with the whole climb ahead of it.
2. CLIMB (applied force + contact): a capped horizontal force (<= 2.5 N — a fingertip
   push on a 150 g ball) drives the ball through the mouth and up the 12-degree ramp
   under velocity servo with gravity feedforward, plus a small lateral centering
   term. Both one-way flaps are opened BY THE BALL in passing — the solver never
   touches a flap; each flap falls closed behind the ball (the ratchet click).
3. DELIVERY (gravity + contact, hands-off): the force is cut at the crest; the ball
   rolls over the drop lip, falls one ball-radius into the sunken basin, and the
   retaining step plus end wall settle it inside. The basin latch fires only for a
   slow ball inside the basin volume.
4. IDENTITY: the RED decoy ball is never touched — it stays in its bay.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: progress
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_top_drawer_i325.solve
             --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

F_CAP = 2.5        # max push force (N) — fingertip on a 150 g ball
V_DES = 0.22       # uphill speed setpoint (m/s)
STAGE_X = -0.070   # staging spot on the ground, downhill of the mouth


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ratchet_ramp")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        loc = scene._ball_loc()[0]
        ib = scene.balls_in_basin()[0]
        ang = scene.flap_angle()[0]
        print(f"[solve] {tag:12s} | "
              + " ".join(f"{nm}=({float(loc[i, 0]):+.3f},{float(loc[i, 1]):+.3f},"
                         f"{float(loc[i, 2]):+.3f}){'B' if bool(ib[i]) else ''}"
                         for i, nm in enumerate(scene.BALLS))
              + f" flaps=({math.degrees(float(ang[0])):+.1f},"
              f"{math.degrees(float(ang[1])):+.1f})deg "
              f"closed={bool(scene.flaps_closed()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)

    def deliver(ball: str, idx: int) -> None:
        """TRANSPORT then CLIMB then DELIVERY (see module docstring)."""
        body = scene.balls[ball]
        # TRANSPORT: one write to the staging spot on the ground before the mouth.
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = STAGE_X
        st[:, 2] = c.ball_r + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, torch.arange(n, device=device))
        step(30)  # let it touch down before pushing

        # CLIMB: capped-force velocity servo, world +x, gravity feedforward on the
        # slope; small lateral centering. Body-frame conversion EVERY step (the
        # rolling ball's body frame spins).
        ff = c.ball_mass * 9.81 * math.sin(math.radians(c.tilt_deg))
        cut_x = c.basin_x0  # crest lip: cut here, gravity finishes the job
        reached = False
        for _ in range(2400):
            loc = scene._ball_loc()[0, idx]
            if float(loc[0]) > cut_x:
                reached = True
                break
            v = body.data.root_lin_vel_w
            fx = (4.0 * (V_DES - v[:, 0]) + ff).clamp(-1.0, F_CAP)
            fy = (-6.0 * (body.data.root_pos_w[:, 1] - scene.env_origins[:, 1])
                  - 1.0 * v[:, 1]).clamp(-0.8, 0.8)
            fw = torch.stack([fx, fy, torch.zeros_like(fx)], dim=-1)
            fb = quat_apply_inverse(body.data.root_quat_w, fw)
            body.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            env.step(no_action)
        body.set_external_force_and_torque(zero, zero)
        assert reached, (f"{ball} never reached the crest "
                         f"(x={float(scene._ball_loc()[0, idx, 0]):+.3f}, "
                         f"flaps={[round(math.degrees(float(a)), 1) for a in scene.flap_angle()[0]]})")

        # DELIVERY: hands-off — over the lip, into the basin, settle.
        for _ in range(600):
            env.step(no_action)
            if bool(scene.balls_in_basin()[0, idx]) and bool(scene.settled()[0]):
                break
        step(60)
        assert bool(scene.balls_in_basin()[0, idx]), \
            f"{ball} did not settle inside the basin"
        assert bool(scene.flaps_closed()[0]), "flaps must fall closed behind the ball"

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(120)  # flaps hang closed; balls settle in their bays
    loc0 = scene._ball_loc()[0]
    ang0 = scene.flap_angle()[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"perm={scene.bay_perm[0].tolist()} "
          + " ".join(f"{nm}=({float(loc0[i, 0]):+.3f},{float(loc0[i, 1]):+.3f})"
                     for i, nm in enumerate(scene.BALLS))
          + f" flaps=({math.degrees(float(ang0[0])):+.2f},"
          f"{math.degrees(float(ang0[1])):+.2f})deg "
          f"masses={[round(float(b.data.default_mass.sum()), 3) for b in scene.balls.values()]}"
          f"+flaps={[round(float(f.data.default_mass.sum()), 3) for f in scene.flaps.values()]}",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.flaps_closed()[0]), "flaps must start hanging closed"
    s0 = print_score("P0 reset+settle (flaps closed, balls in their bays)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: first cargo ball up the ramp --------------------------------------
    deliver("cargo_0", 0)
    report("cargo-1")
    s1 = print_score("P1 first blue ball pushed through both flaps into the basin")
    assert s1 >= s0 - 1e-6, "score decreased"
    assert s1 >= c.w_p1 + c.w_p2 + c.w_bin - 1e-4, f"P1 score {s1}"

    # ---------------- phase 2: second cargo ball -------------------------------------------------
    deliver("cargo_1", 1)
    report("cargo-2")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (deliveries complete but success() is False)",
              flush=True)
        os._exit(1)
    s2 = print_score("P2 second blue ball delivered; decoy untouched; flaps closed")
    assert s2 >= s1 - 1e-6, "score decreased across the final delivery"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------------
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
    try:
        main()
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
