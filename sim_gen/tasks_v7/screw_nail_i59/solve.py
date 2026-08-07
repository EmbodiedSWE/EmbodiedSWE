"""Teleport solution for LatchVaultScene (sim_gen task `screw_nail_i59`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction goes through
contact/constraint dynamics:
  1. UNLOCK (dynamics): each latch is slid to its outer stop by a velocity-regulated
     horizontal force on the latch body — the prismatic joint and its limit are what
     move and stop it. Nothing is teleported.
  2. LID OFF (dynamics + transport): a velocity-regulated vertical force lifts the
     lid out of the pocket THROUGH the now-open latch gap — if a latch were still
     engaged this lift would jam at 4 mm (smoke proves it does). Only once the lid is
     dynamically clear of the pocket is it teleported (pure transport across free
     space) to a parking spot and settled.
  3. RETRIEVE (dynamics + transport): the green block is lifted dynamically straight
     up out of the open cavity by a vertical force (proving the mouth is open), then
     teleported (transport) to a spot above the dish, RELEASED, and dropped in — the
     final containment is made by gravity + contact with the dish, never by spawning
     the block seated.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.screw_nail_i59.solve --headless [--seed N]
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
    env = ENVS.get("simgen.latch_vault")().build(num_envs=args.num_envs, device=device)
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

    def wrench(body, f3: torch.Tensor) -> None:
        """Apply a WORLD force to `body`, expressed in its CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every step while pushing."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        ext = scene.bolt_extension()[0]
        lid = rel(scene.lid)
        gem = rel(scene.gem)
        dish = rel(scene.dish)
        print(f"[solve] {tag:14s} | bolts=({float(ext[0]):.3f},{float(ext[1]):.3f})"
              f" retracted=({bool(scene.bolt_retracted()[0, 0])},"
              f"{bool(scene.bolt_retracted()[0, 1])})"
              f" lid=({float(lid[0]):+.3f},{float(lid[1]):+.3f},{float(lid[2]):.3f})"
              f" gem=({float(gem[0]):+.3f},{float(gem[1]):+.3f},{float(gem[2]):.3f})"
              f" dish=({float(dish[0]):+.3f},{float(dish[1]):+.3f})"
              f" in_dish={bool(scene.gem_in_dish()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(body, max_steps: int = 360) -> None:
        wrench(body, zero3)
        for _ in range(max_steps // 30):
            step(30)
            if float(body.data.root_lin_vel_w[0].norm()) < 0.03:
                break

    # ---------------- contact primitives ---------------------------------------------------
    def retract_bolt(name: str, sgn: float) -> None:
        """Slide latch `name` OUTWARD (world +-x) to its stop with a velocity-regulated
        force. The joint constrains everything else; the limit is the stop."""
        body = scene.bolts[name]
        tgt = c.bolt_eng + c.bolt_stroke  # extension at the outer stop
        f3 = torch.zeros(3, device=device)
        for i in range(900):
            ext = float(scene.bolt_extension()[0, 0 if name == "bolt_px" else 1])
            if ext >= tgt - 0.002:
                break
            v = float(body.data.root_lin_vel_w[0, 0]) * sgn  # outward speed
            f = 60.0 * (0.10 - v)  # PD to 0.10 m/s outward
            f3[0] = sgn * max(min(f, 6.0), -6.0)
            wrench(body, f3)
            env.step(no_action)
        settle(body, 120)

    def lift_lid_off(park_xy: tuple) -> None:
        """Lift the lid dynamically out of the pocket (this is the interaction the
        latches gate), then TRANSPORT it (teleport) to `park_xy` and settle it."""
        f3 = torch.zeros(3, device=device)
        for i in range(900):
            z = float(rel(scene.lid)[2])
            if z > 0.16:
                break
            vz = float(scene.lid.data.root_lin_vel_w[0, 2])
            f = c.lid_mass * 9.81 + 8.0 * (0.12 - vz)  # gravity + PD to 0.12 m/s up
            f3[2] = max(min(f, 6.0), 0.0)
            wrench(scene.lid, f3)
            env.step(no_action)
        wrench(scene.lid, zero3)
        z = float(rel(scene.lid)[2])
        assert z > 0.15, f"lid did not lift clear (z={z:.3f}) — is a latch engaged?"
        # transport only: carry the free lid across open space to the parking spot
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = park_xy[0], park_xy[1], 0.05
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.lid.write_root_state_to_sim(st, all_ids)
        settle(scene.lid)

    def retrieve_gem() -> None:
        """Lift the block dynamically straight up out of the open cavity (proves the
        mouth is open), transport it above the dish, release, and let gravity +
        contact seat it in the dish."""
        f3 = torch.zeros(3, device=device)
        for i in range(900):
            z = float(rel(scene.gem)[2])
            if z > 0.18:
                break
            vz = float(scene.gem.data.root_lin_vel_w[0, 2])
            f = c.gem_mass * 9.81 + 2.0 * (0.15 - vz)
            f3[2] = max(min(f, 2.0), 0.0)
            wrench(scene.gem, f3)
            env.step(no_action)
        wrench(scene.gem, zero3)
        z = float(rel(scene.gem)[2])
        assert z > 0.17, f"block did not lift clear of the cavity (z={z:.3f})"
        # transport only: to 60 mm above the dish center, zero velocity, then DROP
        dish = rel(scene.dish)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1] = float(dish[0]), float(dish[1])
        st[:, 2] = 0.10
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        scene.gem.write_root_state_to_sim(st, all_ids)
        step(180)  # free fall + contact settling inside the dish

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    ext = scene.bolt_extension()[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"bolt ext=({float(ext[0]):.3f},{float(ext[1]):.3f}) "
          f"gem=({float(rel(scene.gem)[0]):+.3f},{float(rel(scene.gem)[1]):+.3f}) "
          f"dish=({float(rel(scene.dish)[0]):+.3f},{float(rel(scene.dish)[1]):+.3f}) "
          f"decoy=({float(rel(scene.decoy)[0]):+.3f},{float(rel(scene.decoy)[1]):+.3f})",
          flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: retract both latches (dynamics) -----------------------------
    retract_bolt("bolt_px", +1.0)
    report("bolt_px out")
    s1a = print_score("P1a latch +x retracted (contact/joint dynamics)")
    assert s1a >= s0 - 1e-6
    retract_bolt("bolt_nx", -1.0)
    report("bolt_nx out")
    s1 = print_score("P1b latch -x retracted (contact/joint dynamics)")
    assert s1 >= s1a - 1e-6
    assert bool(scene.bolt_retracted()[0].all()), "latches not both retracted"

    # ---------------- phase 2: lift the freed lid off (dynamics), park it (transport) ------
    dish = rel(scene.dish)
    cands = ((0.15, 0.32), (0.15, -0.32))
    park = max(cands, key=lambda p: (p[0] - float(dish[0])) ** 2
               + (p[1] - float(dish[1])) ** 2)
    lift_lid_off(park)
    report("lid off")
    s2 = print_score("P2 lid lifted out of the pocket (dynamics) and parked")
    assert s2 >= s1 - 1e-6
    assert bool(scene.lid_clear()[0]), "lid not clear of the case"

    # ---------------- phase 3: retrieve the block into the dish ----------------------------
    retrieve_gem()
    report("gem dropped")
    s3 = print_score("P3 block lifted out (dynamics), dropped into the dish (contact)")
    assert s3 >= s2 - 1e-6

    # ---------------- phase 4: settle to success -------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s4 = print_score("P4 all settled")
    assert s4 >= s3 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after unlock+lid+retrieve+settle)",
              flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                g = scene.gem
                print(f"[solve] persist flicker @step {i}: "
                      f"in_dish={bool(scene.gem_in_dish()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"lin={float(g.data.root_lin_vel_w[0].norm()):.4f} "
                      f"ang={float(g.data.root_ang_vel_w[0].norm()):.4f}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
