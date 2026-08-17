"""Teleport solution for BeamBalanceTrayScene (sim_gen task
`living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray_i139`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the PRESENT butter bar
   from its ground slot to free space 12 mm above its rest height over the TRAY
   pan's centre, zero velocity, bar aligned with the beam. The path is free air;
   the write satisfies no rubric clause by itself (the butter is airborne).
2. LOADING THE TRAY (gravity + contact): the butter FALLS the last 12 mm, lands on
   the pan floor, and the BEAM RESPONDS — it swings tray-side-down and rests at its
   +18 deg stop, carrying the butter with it. The `in_tray` credit (at rest inside
   the tray pan) and the tilt are produced entirely by contact and joint dynamics.
   This is also the demonstration that the SEED'S OWN PLAN (butter in tray, stop)
   is not success here: the solve asserts success() is False at this point.
3. COUNTERWEIGHT SELECTION (scene-state read = the oracle's stand-in for the size
   cue): the correct cube index equals the butter's size class (equal-density
   families; a visual solver reads the same fact off the object sizes, per
   describe()). The solve asserts the chosen cube's mass matches the butter's.
4. BALANCING (gravity + contact + pivot dynamics, hands-off): the chosen cube is
   teleported to free space 15 mm above its rest height over the (tilted!) BALLAST
   pan centre — computed in the beam's CURRENT frame — and dropped. High-friction
   pan floors hold both loads where they landed; the beam swings back through its
   damped pendulum dynamics and settles LEVEL. Every success clause (butter in
   tray, |tilt| < 12 deg, settled) is produced by physics; nothing is pinned,
   nothing is written into place.
5. No forces are ever applied, and no object is ever teleported into a scoring
   state: both drop points sit ABOVE the pan volume gate (z_loc = floor + 95 mm +
   half-height), so at the instant of the write the load is outside every rubric
   volume; it enters the gate by free fall at ~1.4 m/s (too fast for the at-rest
   latches) and scores only once contact has genuinely seated it.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray_i139.solve --headless [--seed N]
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
import traceback

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as _scene  # noqa: F401 - registers simgen.beam_balance_tray
except ImportError:  # pragma: no cover - direct-script fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as _scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.beam_balance_tray")().build(num_envs=args.num_envs,
                                                       device=device)
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
        bl = scene._beam_local(scene.butter_pos_w())[0]
        cob = scene.cubes_on_ballast()[0]
        print(f"[solve] {tag:16s} | butter_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):+.3f}) beam={float(scene.beam_deg()[0]):+.1f}deg "
              f"in_tray={bool(scene.butter_in_tray()[0])} "
              f"cubes_on_ballast={[bool(v) for v in cob]} "
              f"balanced={bool(scene.balanced()[0])} settled={bool(scene.settled()[0])} "
              f"latch_tray={bool(scene._in_tray[0])} latch_pan={bool(scene._on_pan[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def beam_point_w(loc_xyz) -> torch.Tensor:
        """Beam-local point -> world (uses the beam's CURRENT pose, incl. tilt)."""
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # the empty beam self-centers; ground objects settle
    bq = scene.base.data.root_quat_w[0]
    byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
    cls = int(scene.butter_class[0])
    print(f"[solve] layout readback (seed {args.seed}): base_yaw={byaw:+.1f}deg "
          f"beam={float(scene.beam_deg()[0]):+.1f}deg butter_class={cls} "
          f"(mass {c.masses[cls] * 1000:.0f} g, bar {c.butter_dims[cls][0] * 1000:.0f} mm)",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(float(scene.beam_deg()[0])) < 3.0, \
        f"empty beam must self-center, got {float(scene.beam_deg()[0]):+.1f} deg"
    assert not bool(scene.butter_in_tray()[0]), "butter must start on the ground"
    s0 = print_score("P0 reset+settle (empty beam level, butter on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: butter -> tray pan (transport; gravity loads it) ------------
    # Drop point: tray pan centre, ABOVE the pan volume gate (floor + 95 mm), in the
    # beam's CURRENT (level) frame; bar aligned with the beam so it fits the pan
    # with margin. The write itself is outside every rubric volume.
    h = c.butter_dims[cls][2]
    drop = beam_point_w((c.pan_x, 0.0, c.pan_floor_top + h / 2 + 0.095))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = drop
    st[:, 3:7] = scene.beam.data.root_quat_w
    scene.butters[cls].write_root_state_to_sim(st, all_ids)
    step(360)  # fall, beam swings tray-side-down to the stop, settle (hands-off)
    report("butter->tray")
    tilt1 = float(scene.beam_deg()[0])
    assert bool(scene.butter_in_tray()[0]), "butter must rest inside the tray pan"
    assert bool(scene._in_tray[0]), "in_tray latch must be set"
    assert tilt1 > c.balanced_deg, \
        f"butter alone must tip the beam past the gate, got {tilt1:+.1f} deg"
    assert not bool(scene.success()[0]), \
        "the seed's plan (butter in tray, beam at the stop) must NOT be success"
    s1 = print_score("P1 butter loaded in the tray pan; beam at the stop")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_in_tray - 1e-6, \
        f"P1 score {s1} (expect in_tray={c.w_in_tray})"

    # ---------------- phase 2: matching cube -> ballast pan; the beam levels itself --------
    # Selection: cube index == butter size class (equal-density families; the visual
    # cue is the size, per describe()). Sanity-check by physical mass, not by index:
    # the mass table entry the class points at must equal the butter's mass.
    butter_mass = c.masses[cls]
    pick = min(range(3), key=lambda i: abs(c.masses[i] - butter_mass))
    assert abs(c.masses[pick] - butter_mass) < 1e-9, "no cube matches the butter mass"
    print(f"[solve] selecting cube_{pick} ({c.masses[pick] * 1000:.0f} g, "
          f"{c.cube_sides[pick] * 1000:.0f} mm) to match the "
          f"{butter_mass * 1000:.0f} g butter", flush=True)
    s_cube = c.cube_sides[pick]
    # Ballast pan centre ABOVE the pan volume gate (floor + 95 mm), in the beam's
    # CURRENT (tilted) frame — the write is outside every rubric volume.
    # 12 mm outboard bias: on the 18deg-tilted pan the landing skid is inboard
    # (downhill); the bias keeps the seated cube near the pan centre.
    drop = beam_point_w((-(c.pan_x + 0.012), 0.0, c.pan_floor_top + s_cube / 2 + 0.095))
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = drop
    st[:, 3:7] = scene.beam.data.root_quat_w  # face-aligned with the tilted pan floor
    scene.cubes[pick].write_root_state_to_sim(st, all_ids)

    # hands-off: the cube lands, friction holds both loads, the damped pendulum
    # dynamics carry the beam back to level.
    ok_p2 = False
    for j in range(900):
        env.step(no_action)
        if bool(scene.success()[0]):
            ok_p2 = True
            break
        if j % 180 == 179:
            report(f"level-{j + 1}")
    report("balanced")
    if not ok_p2:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (beam never settled level)", flush=True)
        os._exit(1)
    s2 = print_score("P2 matching cube on the ballast pan; beam settled level")
    assert s2 >= s1 - 1e-6, "score decreased across the balancing"

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, hands-off) -----------
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
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
