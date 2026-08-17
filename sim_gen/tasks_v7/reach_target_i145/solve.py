"""Teleport solution for ColorCarouselScene (sim_gen task `reach_target_i145`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction goes through
contact/constraint dynamics:
  1. ALIGN (dynamics): the carousel is rotated to put the TARGET-color cell under the
     loading chute by a velocity-regulated torque about its pivot axis (the same
     wrench a peg push produces about the pivot) — the revolute joint and the body's
     angular damping are what turn it and hold it. Alignment is then RELEASED and the
     rotor must stay inside the tolerance window on its own for the align latch to
     fire. Nothing is teleported.
  2. FEED (transport + dynamics): the white token is teleported (pure transport across
     free space) from its floor slot to hover inside the chute collar mouth — free
     space above the roof — and RELEASED. Gravity and contact do the insertion: the
     token falls through the chute, past the roof plane, into the aligned cell, and
     settles on the deck between the cell walls. It is never spawned inside the cell;
     if the wrong cell were under the chute (or none), this exact drop would land in
     the wrong place and score nothing (smoke proves both).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.reach_target_i145.solve --headless [--seed N]
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
    env = ENVS.get("simgen.color_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)
    hx, hy = c.hub_pos

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f3: torch.Tensor, t3: torch.Tensor) -> None:
        """Apply a WORLD wrench to `body`, expressed in its CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every step while pushing."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            quat_apply_inverse(q, t3.view(1, 3).expand(n, 3)).unsqueeze(1),
            env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        err = math.degrees(float(scene.align_err()[0]))
        w = float(scene.carousel.data.root_ang_vel_w[0, 2])
        tok = rel(scene.token)
        uvz = scene._token_cell_uvz(scene.target.float() * (math.pi / 2))[0]
        print(f"[solve] {tag:14s} | target={int(scene.target[0])}"
              f" err={err:+7.2f}deg w={w:+.3f}"
              f" tok=({float(tok[0]):+.3f},{float(tok[1]):+.3f},{float(tok[2]):.3f})"
              f" uvz=({float(uvz[0]):+.3f},{float(uvz[1]):+.3f},{float(uvz[2]):.3f})"
              f" aligned={bool(scene.aligned()[0])}"
              f" in_cell={bool(scene.token_in_target_cell()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- contact primitives ---------------------------------------------------
    def align_carousel() -> None:
        """Turn the rotor with a velocity-regulated torque about the pivot (PD to a
        capped angular-velocity setpoint; the servo torque always dominates the
        damping torque near the stop condition — no slow-mode stall), then RELEASE
        and let damping hold it inside the window."""
        t3 = torch.zeros(3, device=device)
        for _ in range(2400):
            e = float(scene.align_err()[0])
            w = float(scene.carousel.data.root_ang_vel_w[0, 2])
            if abs(e) < 0.030 and abs(w) < 0.05:
                break
            w_des = max(min(-2.5 * e, 0.45), -0.45)
            t3[2] = max(min(1.2 * (w_des - w), 0.25), -0.25)
            wrench(scene.carousel, zero3, t3)
            env.step(no_action)
        wrench(scene.carousel, zero3, zero3)
        for _ in range(120):  # hands-off: damping brings it to rest inside the window
            step(1)
            if float(scene.carousel.data.root_ang_vel_w[0].norm()) < 0.02:
                break
        step(20)  # let the 10-substep align latch counter mature

    def feed_token() -> None:
        """TRANSPORT the token to free space inside the chute collar mouth, release
        with zero velocity, and let gravity + contact insert it into the cell."""
        ap_x = hx + (c.ap_r_lo + c.ap_r_hi) / 2 * math.cos(c.ap_azimuth)
        ap_y = hy + (c.ap_r_lo + c.ap_r_hi) / 2 * math.sin(c.ap_azimuth)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = ap_x, ap_y, c.collar_top + 0.018
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.token.write_root_state_to_sim(st, all_ids)
        step(300)  # free fall through the chute + contact settling in the cell

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    tok = rel(scene.token)
    card = rel(scene.cards[scene_mod.CELL_NAMES[int(scene.target[0])]])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"target={int(scene.target[0])} ({scene_mod.CELL_NAMES[int(scene.target[0])]}) "
          f"err={math.degrees(float(scene.align_err()[0])):+.1f}deg "
          f"token=({float(tok[0]):+.3f},{float(tok[1]):+.3f}) "
          f"cue_card=({float(card[0]):+.3f},{float(card[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: align the target cell under the chute (dynamics) ------------
    align_carousel()
    report("aligned")
    s1 = print_score("P1 target cell aligned under the chute (torque about the pivot)")
    assert s1 >= s0 - 1e-6
    assert bool(scene.aligned()[0]), "carousel not aligned after servo + release"
    assert bool(scene.align_latch[0] > 0), "align latch did not fire"

    # ---------------- phase 2: feed the token through the chute (transport + gravity) ------
    feed_token()
    report("token fed")
    s2 = print_score("P2 token dropped through the chute into the cell (gravity+contact)")
    assert s2 >= s1 - 1e-6
    assert bool(scene.token_in_target_cell()[0]), \
        "token did not land inside the target cell"

    # ---------------- phase 3: settle to success -------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s3 = print_score("P3 all settled")
    assert s3 >= s2 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after align+feed+settle)", flush=True)
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
                print(f"[solve] persist flicker @step {i}: "
                      f"in_cell={bool(scene.token_in_target_cell()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"tok_lin={float(scene.token.data.root_lin_vel_w[0].norm()):.4f} "
                      f"car_ang={float(scene.carousel.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
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
