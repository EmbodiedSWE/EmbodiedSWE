"""Teleport solution for FlameKeeperScene (sim_gen task
`libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i375`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): single root-state writes carry ONE object at a time
   across FREE SPACE with zero velocity — exactly the carry a gripper performs.
   Every write ends at a HOVER a few millimetres above the destination surface,
   never intersecting anything.
2. THE DEAD-MAN MECHANISM (gravity + contact + the pilot spring): the plate's
   state is never written after reset. Whether it stays pressed is decided 100 %
   by the live force balance between the scene's pilot-spring force and whatever
   weight actually rests on the plate through contact. The solve honours the
   ORDER the mechanism enforces: pot onto the plate's FREE half FIRST (the two
   payloads share the plate; the flame never flickers), pan off to the counter
   SECOND (the pot alone keeps the plate pressed).
3. The choice of the free half is read from the live pan pose after settling —
   the side is randomized per episode.

If a drop leaves an object outside its target region (a bounce), it is picked
up again (fresh transport to the same hover) — a retry, not a cheat: the final
configuration is still 100 % contact-made and the final 3.3 s are hands-off.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i375.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401 - registers the scene
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.flame_keeper")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        q = float(scene.plate_q()[0]) * 1000
        print(f"[solve] {tag:12s} | q={q:+6.2f}mm pressed={bool(scene.plate_pressed()[0])} "
              f"armed={bool(scene._armed[0])} foul={bool(scene._foul[0])} "
              f"pot_on={bool(scene.pot_on_plate()[0])} pan_on={bool(scene.pan_on_plate()[0])} "
              f"pan_clear={bool(scene.pan_clear()[0])} settled={bool(scene.settled()[0])} "
              f"latches=(n={bool(scene._l_near[0])},o={bool(scene._l_on[0])},"
              f"s={bool(scene._l_swap[0])}) success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    last_printed = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= last_printed[0] - 1e-6, \
            f"score regressed: {last_printed[0]} -> {s}"
        last_printed[0] = s
        return s

    def write_pose(body, pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def settle(max_steps: int, min_steps: int = 60) -> None:
        for i in range(max_steps):
            env.step(no_action)
            if i >= min_steps and bool(scene.settled()[0]):
                break

    def qz(yaw: float) -> torch.Tensor:
        return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                            device=device).expand(n, 4)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # the pan seats on the plate; the foul arms (plate rested pressed)
    side = 1.0 if float(scene.pan.data.root_pos_w[0, 0]
                        - scene.plate.data.root_pos_w[0, 0]) > 0 else -1.0
    pot0 = (scene.pot.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): pan_side={side:+.0f} "
          f"pot=({float(pot0[0]):+.3f},{float(pot0[1]):+.3f}) "
          f"q={float(scene.plate_q()[0]) * 1000:+.2f}mm", flush=True)
    report("reset")
    assert torch.isfinite(scene.pot.data.root_pos_w).all(), "NaN/inf after settle"
    assert bool(scene.pan_on_plate()[0]), "pan must start resting on the plate"
    assert bool(scene.plate_pressed()[0]), "the occupied plate must start pressed"
    assert bool(scene._armed[0]), "the flame foul must have armed after settling"
    assert not bool(scene._foul[0]), "the flame must be alive at reset"
    assert not bool(scene.pot_near()[0]), "the pot must spawn outside the approach radius"
    s0 = print_score("P0 reset+settle (pan holds the plate down, flame alive)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.005, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: pot onto the plate's FREE half (pan still on!) --------------
    # hover a few mm above the plate top at the free slot; handle points away
    # from the pan; the drop and everything after is contact dynamics.
    pot_yaw = math.pi if side > 0 else 0.0
    for attempt in range(3):
        tgt = scene.plate.data.root_pos_w.clone()
        tgt[:, 0] += -side * c.pot_slot_x
        tgt[:, 2] += c.plate_size[2] + 0.010
        write_pose(scene.pot, tgt, qz(pot_yaw))
        settle(600, min_steps=90)
        if bool(scene.pot_on_plate()[0]) and bool(scene.pot_upright()[0]):
            break
        print(f"[solve] pot drop bounced (attempt {attempt + 1}); retrying", flush=True)
    report("pot-on-plate")
    assert bool(scene.pot_on_plate()[0]) and bool(scene.pot_upright()[0]), \
        "pot must rest upright on the plate's free half"
    assert bool(scene.pan_on_plate()[0]), "the pan is still on the plate (shared)"
    assert bool(scene.plate_pressed()[0]), "two payloads certainly press the plate"
    assert not bool(scene._foul[0]), "the flame must survive the pot's arrival"
    assert bool(scene._l_on[0]), "the pot-on-plate latch must have fired"
    s1 = print_score("P1 pot installed beside the pan (plate never unloaded)")
    assert s1 >= 0.345, f"P1 score {s1} (expect near 0.10+0.25)"

    # ---------------- phase 2: pan off to the counter (the pot keeps the flame) ------------
    park = torch.tensor([0.62, 0.18], device=device)
    for attempt in range(3):
        tgt = torch.zeros(n, 3, device=device)
        tgt[:, 0:2] = scene.env_origins[:, 0:2] + park
        tgt[:, 2] = scene.env_origins[:, 2] + c.deck_top + 0.006
        write_pose(scene.pan, tgt, qz(0.0))
        settle(480, min_steps=90)
        if bool(scene.pan_clear()[0]):
            break
        print(f"[solve] pan set-down bounced (attempt {attempt + 1}); retrying", flush=True)
    report("pan-off")
    assert bool(scene.pan_clear()[0]), "pan must rest on the counter clear of the stove"
    assert bool(scene.pot_on_plate()[0]), "pot must still hold the plate"
    assert bool(scene.plate_pressed()[0]), "the pot alone must keep the plate pressed"
    assert not bool(scene._foul[0]), "the flame must survive the swap"
    assert bool(scene._l_swap[0]), "the swap latch must have fired"
    s2 = print_score("P2 swap complete (pan on the counter, flame alive)")
    assert s2 >= 0.595, f"P2 score {s2} (expect 0.60)"

    # ---------------- phase 3: success holds continuously ----------------------------------
    ok = False
    streak = 0
    for _ in range(1200):
        env.step(no_action)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 240:  # 2 simulated seconds of continuous success
            ok = True
            break
    report("success-hold")
    assert ok and bool(scene.success()[0]), "success must hold continuously for 2 s"
    s3 = print_score("P3 success live (pot alone on the pressed plate, flame lit)")
    assert s3 >= 0.99, f"success must score 1.0, got {s3}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s hands-off")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    try:
        main()
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
