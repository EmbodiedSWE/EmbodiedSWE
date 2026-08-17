"""Teleport solution for MissingRailRackScene (sim_gen task
`libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack_i345`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. REPAIR — TRANSPORT (teleport): one pose write stages the loose CROSSBAR level
   above the two empty bracket slots — rack-local x at the near-rail position (with
   a deliberate few-mm lateral offset and a ~1.5 deg yaw error, the imperfection a
   real place has), long axis along the rack y-axis, bottom face just above the
   guide-wall tops, zero velocity. This is the pose the arm reaches by carrying the
   grasped bar over the rack.
   REPAIR — SEAT (contact dynamics): the bar is released; it falls ~4 cm between
   each slot's guide walls onto the two tower tops and settles SEATED — the seating
   itself (drop-in, wall guidance, rest on the slot floors) is pure contact. If the
   first drop does not seat (a corner hangs on a wall top), one retry re-stages the
   hover with zero offsets.
2. RACK — TRANSPORT (teleport): one pose write stages the BOTTLE horizontal above
   the chock saddle — rack-local origin over the rail midline (x=0, a deliberate
   +6 mm y offset), axis along the rack x-axis (neck toward the front), body
   underside ~20 mm above the rail-top plane, zero velocity.
   RACK — REST (contact dynamics): released, the body drops onto the fixed rail and
   the freshly seated crossbar simultaneously, is arrested laterally by the chock
   saddle, and settles bridging the two rails — the rest is earned through contact
   with the rail the solver itself installed.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
    env = ENVS.get("simgen.missing_rail_rack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        bl = scene._bar_loc()[0]
        ol = scene._bottle_loc()[0]
        a = scene._bottle_axis_local()[0]
        print(f"[solve] {tag:12s} | bar_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) align={float(scene._bar_align()[0]):.3f} "
              f"seated={bool(scene.bar_seated()[0])} | "
              f"bottle_loc=({float(ol[0]):+.3f},{float(ol[1]):+.3f},{float(ol[2]):.3f}) "
              f"axis=({float(a[0]):+.2f},{float(a[2]):+.2f}) "
              f"racked={bool(scene.bottle_racked()[0])} | "
              f"latches=({bool(scene._bar_lift_ever[0])},{bool(scene._bar_seat_ever[0])},"
              f"{bool(scene._bottle_lift_ever[0])},{bool(scene._cradle_ever[0])}) "
              f"settled={bool(scene._settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def q_z(deg: float) -> torch.Tensor:
        h = math.radians(deg) / 2
        return torch.tensor([math.cos(h), 0.0, 0.0, math.sin(h)],
                            device=device).expand(n, 4)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    r = (scene.rack.data.root_pos_w - scene.env_origins)[0]
    rq = scene.rack.data.root_quat_w[0]
    r_yaw = 2.0 * math.atan2(float(rq[3]), float(rq[0]))
    b0 = (scene.bar.data.root_pos_w - scene.env_origins)[0]
    o0 = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rack=({float(r[0]):+.3f},{float(r[1]):+.3f}) yaw={math.degrees(r_yaw):+.1f}deg "
          f"bar=({float(b0[0]):+.3f},{float(b0[1]):+.3f},{float(b0[2]):.3f}) "
          f"side={'+' if float(b0[1]) > 0 else '-'} "
          f"bottle=({float(o0[0]):+.3f},{float(o0[1]):+.3f},{float(o0[2]):.3f})",
          flush=True)
    report("reset")
    assert abs(float(b0[2]) - c.bar_floor_z) < 0.01, "bar did not settle on the floor"
    assert abs(float(o0[2]) - c.stand_z) < 0.01, "bottle did not settle standing"
    assert float(b0[1]) * float(o0[1]) < 0, "bar and bottle should spawn on opposite sides"
    assert not bool(scene.success()[0]), "success at reset?!"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: REPAIR — transport the bar, drop it into the brackets --------
    # Hover: rack-local (rail_x + 4 mm, 0, bottom just above the guide-wall tops),
    # long axis along the rack y (a 1.5 deg deliberate yaw error). Released, it falls
    # between the walls onto the tower tops — the seating is pure contact.
    hover_z = c.tower_h + c.wall_h + c.bar_s / 2 + 0.006

    def stage_bar(dx: float, dyaw_deg: float) -> None:
        r_pos = scene.rack.data.root_pos_w
        r_quat = scene.rack.data.root_quat_w
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = c.rail_x + dx
        loc[:, 2] = hover_z
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = r_pos + quat_apply(r_quat, loc)
        st[:, 3:7] = quat_mul(r_quat, q_z(dyaw_deg))
        scene.bar.write_root_state_to_sim(st, all_ids)

    seated = False
    for attempt, (dx, dyaw) in enumerate(((0.004, 1.5), (0.0, 0.0))):
        stage_bar(dx, dyaw)
        for _ in range(300):
            env.step(no_action)
            if bool(scene.bar_seated()[0]) and bool(scene._settled()[0]):
                seated = True
                break
        if seated:
            break
        print(f"[solve] bar drop attempt {attempt} did not seat; retrying square",
              flush=True)
    report("bar seated")
    assert seated, "crossbar never seated in its brackets"
    bl = scene._bar_loc()[0]
    assert abs(float(bl[2]) - c.seat_z) <= c.bar_z_tol, \
        f"bar rest height off the slot floor (z_loc={float(bl[2]):.4f})"
    assert not bool(scene.success()[0]), "bar alone must NOT be success"
    s1 = print_score("P1 crossbar dropped into both brackets, seated")
    assert s1 >= s0 - 1e-6, "score decreased across the repair"
    assert 0.35 <= s1 <= 0.45, f"seated bar should score 0.40, got {s1}"

    # ---------------- phase 2: RACK — transport the bottle, lay it across the rails ---------
    # Hover: horizontal over the saddle — rack-local (0, +6 mm, body underside 20 mm
    # above the rail tops), axis along rack +x (neck toward the front). Released, the
    # body drops onto BOTH rails (the fixed one and the one just installed) and the
    # chock saddle arrests the roll.
    r_pos = scene.rack.data.root_pos_w
    r_quat = scene.rack.data.root_quat_w
    loc = torch.zeros(n, 3, device=device)
    loc[:, 1] = 0.006
    loc[:, 2] = c.rest_z + 0.020
    h = math.pi / 4  # rotate bottle +z onto +x: 90 deg about y
    q_y90 = torch.tensor([math.cos(h), 0.0, math.sin(h), 0.0], device=device).expand(n, 4)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = r_pos + quat_apply(r_quat, loc)
    st[:, 3:7] = quat_mul(r_quat, q_y90)
    scene.bottle.write_root_state_to_sim(st, all_ids)
    # hands-off: drop onto the rails, settle into the saddle
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        if bool(scene.success()[0]):
            quiet += 1
        else:
            quiet = 0
        if quiet >= 30:
            break
    report("racked")
    s2 = print_score("P2 bottle laid across both rails, settled in the saddle")
    assert s2 >= s1 - 1e-6, "score decreased across the racking"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) ------
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
    except BaseException:  # noqa: BLE001 - die fast, don't idle to the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(4)
