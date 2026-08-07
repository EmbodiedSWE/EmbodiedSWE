"""Teleport solution for FireCribScene (sim_gen task `close_grill_i8`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, one body at a time): a single root-state write carries a
   split (or the griddle) from its ground scatter slot to a free-space HOVER pose
   12 mm ABOVE where it will finally rest — over the hearth plate for the bottom
   layer, over the already-placed bottom pair for the top layer, over the completed
   crib for the griddle. Every hover pose is in open air (verified clearances in
   the module docstring of scene.py); the write satisfies no rubric clause by
   itself and sets zero velocity — it is exactly the carry a gripper performs.
2. SEATING (gravity + contact, hands-off): from the hover the body FALLS and lands
   on the real hearth plate / the real lower splits / the real top pair. Every
   structural fact the rubric checks (layer heights, parallel spacing, orthogonal
   crossings landing on both supports, the griddle's level rest in the success
   band) is produced by ballistics and contact, never written.
3. ORDER: bottom pair, then top pair, then griddle — each stage rests on the one
   below, so the build order is physically inherent. The OFFCUT is never touched
   (restraint) and stays where it spawned, off the hearth plate.

If a settle leaves a stage geometrically off (a bounce shifted a bar), the body is
picked up again (transport to the same free-space hover) and re-dropped — a retry,
not a cheat: the final configuration is still 100 % contact-made.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_grill_i8.solve --headless [--seed N]
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

_qmul, _qz = scene_mod._qmul, scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.fire_crib")().build(num_envs=args.num_envs, device=device)
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

    def report(tag: str) -> None:
        pos, _a, vel = scene._split_tensors()
        loc = scene._pad_local_xy(pos)[0]
        dz = (pos[0, :, 2] - scene._pad_top_z()[0])
        ss = " ".join(f"s{i}=({float(loc[i, 0]):+.3f},{float(loc[i, 1]):+.3f},"
                      f"dz{float(dz[i]):+.3f})" for i in range(4))
        base, span1, crib = scene._structure_now()
        print(f"[solve] {tag:12s} | {ss} vmax={float(vel[0].max()):.3f} "
              f"base={bool(base[0])} span1={bool(span1[0])} crib={bool(crib[0])} "
              f"griddle={bool(scene.griddle_on_crib()[0])} "
              f"offcut_clear={bool(scene.offcut_clear()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def pad_pose(local_xy, rest_dz: float, yaw_local: float):
        """World (pos (n,3), quat (n,4)) for a pad-frame target: xy in the hearth
        frame, centre height `rest_dz` above the hearth top PLUS the 12 mm hover."""
        pp = scene.pad.data.root_pos_w
        pq = scene.pad.data.root_quat_w
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1] = local_xy
        pos = pp + quat_apply(pq, loc)
        pos[:, 2] = scene._pad_top_z() + rest_dz + 0.012
        quat = _qmul(pq, _qz(torch.full((n,), yaw_local, device=device)))
        return pos, quat

    def drop(body, pos_w, quat_w, max_steps: int = 240) -> None:
        """TRANSPORT to the free-space hover (zero velocity), then hands-off fall
        and full settle — the seating itself is pure gravity + contact."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)
        for i in range(max_steps):
            env.step(no_action)
            if i > 20 and float(body.data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_speed:
                break
        step(60)  # extra hands-off settle

    def place(body, local_xy, rest_dz: float, yaw_local: float, check, tag: str) -> None:
        """Place with up to 3 attempts; `check()` is the geometric stage predicate.
        A retry is a fresh transport to the same free-space hover + re-drop."""
        for attempt in range(3):
            pos, quat = pad_pose(local_xy, rest_dz, yaw_local)
            drop(body, pos, quat)
            if bool(check()[0]):
                return
            print(f"[solve] {tag}: stage predicate not met after attempt "
                  f"{attempt + 1}; re-dropping", flush=True)
            report(f"{tag}-retry")
        report(f"{tag}-FAIL")
        print(f"SIM_GEN_SOLVE: FAIL ({tag} never seated)", flush=True)
        os._exit(1)

    s_gap = 0.095  # chosen pair spacing, mid-window (0.078 .. 0.128)
    z_bot = c.bar_w / 2                      # bottom split rest centre above pad top
    z_top = 1.5 * c.bar_w                    # top split rest centre
    z_gr = 2.0 * c.bar_w + c.plate_t / 2     # griddle rest centre

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything seats on the ground
    pp = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    pq = scene.pad.data.root_quat_w[0]
    pyaw = math.degrees(2.0 * math.atan2(float(pq[3]), float(pq[0])))
    slot_of = scene.slot_of[0].tolist()
    pos, axis, _v = scene._split_tensors()
    bars_str = " ".join(
        f"{nm}=({float(scene.bars[nm].data.root_pos_w[0, 0] - scene.env_origins[0, 0]):+.3f},"
        f"{float(scene.bars[nm].data.root_pos_w[0, 1] - scene.env_origins[0, 1]):+.3f})"
        for nm in scene.BAR_NAMES)
    gp = (scene.griddle.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"hearth=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) yaw={pyaw:+.1f}deg "
          f"slot_of={slot_of} {bars_str} "
          f"griddle=({float(gp[0]):+.3f},{float(gp[1]):+.3f})", flush=True)
    report("reset")
    assert torch.isfinite(pos).all(), "NaN/inf in split states after settle"
    hz = pos[0, :, 2] - scene.env_origins[0, 2]
    assert bool((hz > 0.005).all() and (hz < 0.03).all()), \
        f"splits must lie flat on the ground, z={hz.tolist()}"
    assert bool(scene.offcut_clear()[0]), "offcut must start off the hearth plate"
    s0 = print_score("P0 reset+settle (all stock scattered on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: bottom pair (splits 0, 1) -----------------------------------
    # Axes along the hearth-local y (yaw +90 deg), centres at hearth-local x = -/+ s/2.
    drop(scene.bars["split_0"], *pad_pose((-s_gap / 2, 0.0), z_bot, math.pi / 2))
    report("bot-A")
    place(scene.bars["split_1"], (+s_gap / 2, 0.0), z_bot, math.pi / 2,
          lambda: scene._structure_now()[0], "bottom-pair")
    report("bot-B")
    assert bool(scene._base[0]), "base latch did not set"
    assert not bool(scene.success()[0]), "cannot be success with only the base"
    s1 = print_score("P1 bottom pair laid parallel on the hearth (open air gap)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.34, f"P1 score {s1} (expect appr+base=0.35)"

    # ---------------- phase 2: first spanner (split 2) -------------------------------------
    # Axis along the hearth-local x (yaw 0), centre at hearth-local y = -s/2,
    # resting across BOTH bottom splits.
    place(scene.bars["split_2"], (0.0, -s_gap / 2), z_top, 0.0,
          lambda: scene._structure_now()[1], "first-spanner")
    report("span-A")
    assert bool(scene._span1[0]), "span1 latch did not set"
    assert not bool(scene.success()[0]), "cannot be success with three splits"
    s2 = print_score("P2 first spanner resting across both bottom splits")
    assert s2 >= s1 - 1e-6 and s2 >= 0.54, f"P2 score {s2} (expect 0.55)"

    # ---------------- phase 3: second spanner (split 3): crib complete ---------------------
    place(scene.bars["split_3"], (0.0, +s_gap / 2), z_top, 0.0,
          lambda: scene._structure_now()[2], "second-spanner")
    report("span-B")
    assert bool(scene._crib[0]), "crib latch did not set"
    assert not bool(scene.success()[0]), "cannot be success without the griddle"
    s3 = print_score("P3 two-layer log-cabin crib complete")
    assert s3 >= s2 - 1e-6 and s3 >= 0.74, f"P3 score {s3} (expect 0.75)"

    # ---------------- phase 4: the griddle proves the structure ----------------------------
    place(scene.griddle, (0.0, 0.0), z_gr, 0.0,
          lambda: scene.griddle_on_crib() & scene.success(), "griddle")
    report("griddle")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the griddle)", flush=True)
        os._exit(1)
    s4 = print_score("P4 griddle resting level on the crib: structure carries the load")
    assert s4 >= s3 - 1e-6, "score decreased across the griddle placement"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
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
