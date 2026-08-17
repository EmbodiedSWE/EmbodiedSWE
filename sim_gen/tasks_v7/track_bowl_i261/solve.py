"""Teleport solution for CoveredDishScene (sim_gen task `track_bowl_i261`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, always ending in a non-contact hover in free
space; every load-bearing interaction is CONTACT DYNAMICS:
  P1 — SEAT: the bowl is carried from its scatter slot to a hover over the stand's
  socket (boss bottom 15 mm above the collar rim, axis offset a few mm) and then
  simply DROPS: the 76 mm boss falls into the 90 mm socket under gravity and the
  boss bottom lands on the socket floor. The mate is closed by gravity + contact.
  P2 — FILL: the egg is carried to a hover 40 mm above the seated bowl's rim plane
  (free space above the open mouth) and dropped: it falls past the rim, hits the
  bowl floor, rolls, and settles INSIDE — pure gravity + contact containment.
  P3 — CAP: the REAL lid (scene.lid — identified by construction; its scatter slot
  is randomized and printed) is carried to a hover with its skirt bottom 12 mm above
  the rim, slightly off-axis, and dropped: the octagonal skirt falls into the mouth
  (8.7 mm worst-case radial slack), self-centres on skirt/wall contact, and the
  cover disk lands on the rim ring. The decoy is never touched.
Nothing is ever teleported into contact, and the egg/lid never receive forces.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.track_bowl_i261.solve --headless [--seed N]
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
    env = ENVS.get("simgen.covered_dish")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def quat_apply_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        return quat_apply(q, v)

    def hover(body, pos_w: torch.Tensor, quat: torch.Tensor) -> None:
        """TRANSPORT ONLY: write a zero-velocity free-space pose (world incl. origin)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat
        body.write_root_state_to_sim(st, all_ids)

    def report(tag: str) -> None:
        bl = scene._stand_local(scene.bowl.data.root_pos_w)[0]
        el = scene._bowl_local(scene.egg.data.root_pos_w)[0]
        ll = scene._bowl_local(scene.lid.data.root_pos_w)[0]
        dl = scene._bowl_local(scene.decoy.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | bowl_stand=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):+.3f}) egg_bowl=({float(el[0]):+.3f},{float(el[1]):+.3f},"
              f"{float(el[2]):+.3f}) lid_bowl=({float(ll[0]):+.3f},{float(ll[1]):+.3f},"
              f"{float(ll[2]):+.3f}) decoy_bowl_z={float(dl[2]):+.3f} "
              f"seated={bool(scene.bowl_seated()[0])} "
              f"egg_in={bool(scene.egg_in_bowl()[0])} "
              f"lid_on={bool(scene.lid_seated()[0])} "
              f"decoy_in={bool(scene.decoy_in_bowl()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    sq = scene.stand.data.root_quat_w[0]
    syaw = 2.0 * math.atan2(float(sq[3]), float(sq[0]))
    b0 = scene._stand_local(scene.bowl.data.root_pos_w)[0]
    e0 = scene._stand_local(scene.egg.data.root_pos_w)[0]
    l0 = scene._stand_local(scene.lid.data.root_pos_w)[0]
    d0 = scene._stand_local(scene.decoy.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): stand=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(syaw):+.1f}deg "
          f"lid_slot={float(scene.lid_slot[0]):+.0f} "
          f"bowl_stand=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"egg_stand=({float(e0[0]):+.3f},{float(e0[1]):+.3f}) "
          f"lid_stand=({float(l0[0]):+.3f},{float(l0[1]):+.3f}) "
          f"decoy_stand=({float(d0[0]):+.3f},{float(d0[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: bowl TRANSPORT + gravity drop = SEAT ------------------------
    # Hover the bowl over the socket: boss bottom 15 mm above the collar rim, axis
    # offset 4 mm (boss radius 38 + 4 = 42 < 45 mm socket inradius: the hover and the
    # whole fall path intersect nothing). Upright, yaw = stand yaw (the boss is a
    # cylinder — yaw-free mate). Then let gravity close the mate.
    hover_z = c.collar_h + 0.015 + c.bowl_org_h  # stand-local bowl-origin height
    tgt_local = torch.tensor([0.004, 0.0, hover_z], device=device).expand(n, 3)
    pos_w = scene.stand.data.root_pos_w + quat_apply_batch(
        scene.stand.data.root_quat_w, tgt_local)
    hover(scene.bowl, pos_w, scene.stand.data.root_quat_w.clone())
    step(90)  # fall 15 mm, boss lands on the socket floor, settle
    report("seated")
    s1 = print_score("P1 bowl transport + gravity seat")
    assert s1 >= s0 - 1e-6, "score decreased across the seating phase"
    if not bool(scene.bowl_seated()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (bowl did not seat in the socket)", flush=True)
        os._exit(1)

    # ---------------- phase 2: egg TRANSPORT + gravity drop = FILL -------------------------
    # Hover the egg on the seated bowl's axis, centre 40 mm above the rim plane (free
    # space above the open 104 mm mouth), offset 5 mm so it lands on the floor and
    # rolls — containment closes by gravity + wall/floor contact only.
    tgt_local = torch.tensor([0.005, 0.0, c.wall_h + 0.040 + c.egg_r],
                             device=device).expand(n, 3)
    pos_w = scene.bowl.data.root_pos_w + quat_apply_batch(
        scene.bowl.data.root_quat_w, tgt_local)
    eye = torch.zeros(n, 4, device=device)
    eye[:, 0] = 1.0
    hover(scene.egg, pos_w, eye)
    step(120)  # fall ~95 mm to the floor, roll, settle
    report("filled")
    s2 = print_score("P2 egg transport + gravity fill")
    assert s2 >= s1 - 1e-6, "score decreased across the filling phase"
    if not (bool(scene.egg_in_bowl()[0]) and bool(scene.bowl_seated()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (egg not resting inside the seated bowl)", flush=True)
        os._exit(1)

    # ---------------- phase 3: lid TRANSPORT + gravity drop = CAP --------------------------
    # Hover the REAL lid over the bowl: skirt bottom 12 mm above the rim plane, axis
    # offset 4 mm, yaw = bowl yaw (flats aligned -> maximum 12 mm flat-to-flat slack;
    # worst-case corner-to-flat slack is still 8.7 mm, so any yaw would mate). The
    # skirt corners at 43.3 + 4 = 47.3 < 52 mm mouth inradius: the fall path is
    # clear. Gravity drops the skirt into the mouth; skirt/wall contact self-centres
    # the lid and the cover disk lands on the rim ring. The decoy is never touched.
    tgt_local = torch.tensor([0.004, 0.0, c.wall_h + c.lid_skirt_h + 0.012],
                             device=device).expand(n, 3)
    pos_w = scene.bowl.data.root_pos_w + quat_apply_batch(
        scene.bowl.data.root_quat_w, tgt_local)
    hover(scene.lid, pos_w, scene.bowl.data.root_quat_w.clone())
    step(120)  # fall 12 mm, skirt into the mouth, disk onto the rim, settle
    report("capped")
    s3 = print_score("P3 lid transport + gravity cap")
    assert s3 >= s2 - 1e-6, "score decreased across the capping phase"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the lid was capped)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) -----
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
