"""Teleport solution for HatchShelfScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i306`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. LID CLOSING (contact/joint dynamics — the core interaction; teleporting the lid
   closed would bypass the task and is never done): a regulated horizontal force
   (velocity-servoed on the hinge rate, capped at 5 N) pushes the standing lid's face
   forward. The lid rotates about its REAL hinge, passes the over-centre point, and
   gravity carries it down flat onto the hinge's closed stop; the force is CUT well
   before flat (at 25 deg) so the final fall and the slam onto the stop are pure
   physics. The solve ASSERTS the lid started fully open (> 95 deg) and ended settled
   closed — the mechanism demonstrably travelled its whole arc under dynamics.
2. TRANSPORT (teleport): ONE pose write moves the BLACK bowl across free space from
   its floor slot to a hover 5 cm ABOVE the closed lid's centre — touching nothing,
   inside no scoring band (the on-lid gate's z band tops out 2.5 cm over the lid;
   the hover is ABOVE it, so no gate and no success is satisfied by the teleport).
   The white decoy bowl is never touched.
3. PLACEMENT (contact dynamics): the bowl is released from the hover and falls onto
   the closed lid under gravity; landing, micro-bounce and settling are pure physics,
   and the closed lid physically CARRIES the bowl's weight on the hinge stop —
   the surface the task exists to create is demonstrably load-bearing.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i306.solve --headless [--seed N]
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
    env = ENVS.get("simgen.hatch_shelf")().build(num_envs=args.num_envs, device=device)
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

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        p_blk, p_wht = loc(scene.bowl_black), loc(scene.bowl_white)
        print(f"[solve] {tag:12s} | black=({float(p_blk[0]):+.3f},{float(p_blk[1]):+.3f},"
              f"{float(p_blk[2]):.3f}) white=({float(p_wht[0]):+.3f},"
              f"{float(p_wht[1]):+.3f},{float(p_wht[2]):.3f}) "
              f"lid={float(scene.lid_open_deg()[0]):+.1f}deg "
              f"closed={bool(scene.lid_closed()[0])} "
              f"on={bool(scene._on_lid(scene.bowl_black)[0])} "
              f"lat=[p{float(scene._prog_max[0]):.2f} c{int(scene._closed[0])} "
              f"k{float(scene._carry_max[0]):.2f} o{int(scene._on[0])}] "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p_blk, p_wht = loc(scene.bowl_black), loc(scene.bowl_white)
    lid0 = float(scene.lid_open_deg()[0])
    print(f"[solve] layout readback (seed {args.seed}): black=({float(p_blk[0]):+.3f},"
          f"{float(p_blk[1]):+.3f}) white=({float(p_wht[0]):+.3f},{float(p_wht[1]):+.3f}) "
          f"lid_open={lid0:.1f}deg goal=({c.goal_pt[0]:.3f},{c.goal_pt[1]:.3f},"
          f"{c.goal_pt[2]:.3f})", flush=True)
    report("reset")
    assert lid0 > 95.0, "lid did not rest fully open past vertical at reset"
    assert float(p_blk[2]) < 0.03 and float(p_wht[2]) < 0.03, \
        "bowls did not settle standing on the floor"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.05, "score not ~0 at reset"

    # ---------------- phase 1: CLOSE THE LID (contact/joint dynamics) ----------------------
    # Regulated +x force at the lid's CoM: drives the hinge at ~1.2 rad/s toward the
    # over-centre point. Cut at 25 deg — from there the fall and the slam onto the
    # closed stop are pure gravity + the joint limit.
    pushed = 0
    for _ in range(900):
        a = float(scene.lid_open_deg()[0])
        if a < 25.0:
            break
        w = float(scene.lid.data.root_ang_vel_w[0, 1])  # +y rate = closing
        fx = max(0.0, min(5.0, 4.0 * (1.2 - w)))
        f_w = torch.tensor([fx, 0.0, 0.0], device=device).view(1, 1, 3).expand(n, 1, 3)
        scene.lid.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                env_ids=all_ids, is_global=True)
        env.step(no_action)
        pushed += 1
    scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        ok_now = bool(scene.lid_closed()[0]) and bool(scene._lid_still()[0])
        quiet = quiet + 1 if ok_now else 0
        if quiet >= 30:
            break
    report("lid-closed")
    print(f"[solve] lid: pushed {pushed} steps from {lid0:.1f} deg, now "
          f"{float(scene.lid_open_deg()[0]):+.1f} deg", flush=True)
    assert bool(scene.lid_closed()[0]) and bool(scene._lid_still()[0]), \
        "lid did not settle closed after the over-centre push"
    s1 = print_score("P1 lid driven over-centre and fallen closed (contact)")
    assert s1 >= s0 - 1e-6, "score decreased across the lid closing"

    # ---------------- phase 2: TRANSPORT the black bowl (one free-space teleport) -----------
    # Endpoint: 5 cm ABOVE the closed lid's centre — free space, touching nothing,
    # ABOVE the on-lid gate's z band (which tops out 2.5 cm over the lid).
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1] = c.goal_pt[0], c.goal_pt[1]
    st[:, 2] = c.lid_top_z + 0.050
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.bowl_black.write_root_state_to_sim(st, all_ids)
    report("hover")
    assert not bool(scene._on_lid(scene.bowl_black)[0]) and not bool(scene.success()[0]), \
        "hovering above the lid must not satisfy the on-lid gate (teleport is transport only)"
    s2 = print_score("P2 transport: black bowl hovering above the closed lid")
    assert s2 >= s1 - 1e-6, "score decreased across transport"

    # ---------------- phase 3: PLACEMENT (gravity drop + settle, pure physics) --------------
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        still = bool(scene._bowl_still(scene.bowl_black)[0])
        quiet = quiet + 1 if still else 0
        if quiet >= 30:
            break
    report("placed")
    assert bool(scene._on_lid(scene.bowl_black)[0]), \
        "black bowl did not settle upright centered on the closed lid"
    assert bool(scene.lid_closed()[0]), "lid did not carry the bowl's weight"
    s3 = print_score("P3 bowl landed and settled on the lid (contact)")
    assert s3 >= s2 - 1e-6, "score decreased across the placement"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after placement)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) --------
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
