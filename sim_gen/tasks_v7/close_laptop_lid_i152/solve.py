"""Teleport solution for BallastLidChestScene (sim_gen task `close_laptop_lid_i152`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries one steel block from its
   ground scatter spot to a free-space point inside the tray airspace — centred
   between the tray walls, ~17 mm above the tray floor, oriented flush with the
   tilted lid, zero velocity (the same release a gripper performs after lowering
   the block between the walls; a release from above the wall tops is NOT used
   because a world-vertical fall in the 46-deg-tilted lid frame drifts forward by
   tan(46) per unit drop and carries the block over the front lip — observed).
   The tray mouth is open to the sky at the stop, so the transport bypasses no
   contact interaction, and the write satisfies no rubric clause by itself: the
   block is airborne above the tray floor.
2. BALLASTING (gravity + contact): the block FALLS onto the tray floor, slides down
   the tilted floor and rests against the tray's hinge-side wall. `in_tray` +
   settled are produced by contact, never written. After ONE block the lid must
   stay at its open stop (the counterweight wins by >= 0.35 N.m) — asserted.
3. CLOSURE (gravity + contact, hands-off): the SECOND block tips the torque
   balance; the lid swings shut on its hinge, carrying both blocks in the tray, and
   comes to rest with its slab on the chest rim. Every closure fact the rubric
   checks (opening angle, blocks riding in the tray, settled) is produced by the
   hinge dynamics; the lid is NEVER pushed or teleported after reset.
4. NO ORDER: the two drops are interchangeable (any two of the three blocks, either
   first). The third block is never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_laptop_lid_i152.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_lid_chest")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def ang() -> float:
        return float(scene.open_angle_deg()[0])

    def report(tag: str) -> None:
        bl = [scene._lid_local(b.data.root_pos_w)[0] for b in scene.blocks]
        bl_s = " ".join(f"b{i}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})"
                        for i, p in enumerate(bl))
        print(f"[solve] {tag:14s} | angle={ang():+6.2f}deg tray={int(scene.tray_count()[0])} "
              f"{bl_s} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def drop_block(i: int, y_loc: float, tag: str) -> None:
        """Teleport block i to free space above the tray mouth (flush with the lid
        tilt), then let gravity do the placement."""
        tray_pt = torch.zeros(n, 3, device=device)
        tray_pt[:, 0] = (c.tray_x0 + c.tray_x1) / 2
        tray_pt[:, 1] = y_loc
        tray_pt[:, 2] = 0.035  # INSIDE the tray airspace, ~17 mm above the floor
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.lid.data.root_pos_w \
            + quat_apply(scene.lid.data.root_quat_w, tray_pt)
        st[:, 3:7] = scene.lid.data.root_quat_w  # flush with the tilted tray
        scene.blocks[i].write_root_state_to_sim(st, all_ids)
        print(f"[solve] {tag}: block {i} released above the tray at "
              f"lid-local y {y_loc:+.3f}", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # lid settles onto its open stop, blocks onto the ground
    cp = (scene.chest.data.root_pos_w - scene.env_origins)[0]
    cq = scene.chest.data.root_quat_w[0]
    cyaw = math.degrees(2.0 * math.atan2(float(cq[3]), float(cq[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"chest=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) yaw={cyaw:+.1f}deg "
          f"angle={ang():+.2f}deg", flush=True)
    # mass/CoM readback: per-child densities must have produced the real lid mass
    # and a CoM BEHIND the hinge (root mass_props on custom spawners is ignored on
    # this stack — guard against silent regression)
    lm = float(scene.lid.root_physx_view.get_masses()[0].sum())
    lcom = scene.lid.root_physx_view.get_coms()[0]
    print(f"[solve] lid mass readback: {lm:.3f} kg (ref {c.lid_mass_ref}); "
          f"com=({float(lcom[0]):+.4f},{float(lcom[1]):+.4f},{float(lcom[2]):+.4f})",
          flush=True)
    assert abs(lm - c.lid_mass_ref) < 0.35, f"lid mass {lm} vs ref {c.lid_mass_ref}"
    assert float(lcom[0]) < c.lid_com_x_max, \
        f"lid CoM must sit behind the hinge, com_x={float(lcom[0]):+.4f}"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(ang() - c.open_stop_deg) < 3.0, \
        f"lid must rest at the open stop, angle={ang():.2f}"
    assert int(scene.tray_count()[0]) == 0, "no block may start in the tray"
    s0 = print_score("P0 reset+settle (lid open on its stop, blocks on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: first block -> tray (transport, gravity seats it) -----------
    drop_block(0, -0.050, "P1")
    step(240)  # fall, slide to the hinge-side wall, lid dips and returns, settle
    report("block0->tray")
    assert int(scene.tray_count()[0]) >= 1, "block 0 must ride inside the tray"
    assert ang() > c.open_stop_deg - 12.0, \
        f"ONE block must not close the lid (angle={ang():.2f})"
    assert not bool(scene.success()[0]), "cannot be success with one block"
    s1 = print_score("P1 one block ballasted (lid still open — counterweight wins)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_ballast1 - 1e-4, f"P1 score {s1}"

    # ---------------- phase 2: second block -> tray; gravity swings the lid shut -----------
    drop_block(1, +0.050, "P2")
    closed = False
    for i in range(720):
        env.step(no_action)
        if bool(scene.success()[0]):
            closed = True
            break
    step(60)  # extra settle, hands-off
    report("lid-closed")
    if not closed or not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (lid never closed under two-block ballast)", flush=True)
        os._exit(1)
    assert ang() <= c.closed_max_deg, f"angle={ang():.2f}"
    s2 = print_score("P2 two blocks ballasted; lid swung shut under gravity")
    assert s2 >= s1 - 1e-6 and s2 >= 0.999, f"P2 score {s2}"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
