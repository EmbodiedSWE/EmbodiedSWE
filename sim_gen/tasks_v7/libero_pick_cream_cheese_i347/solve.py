"""Teleport solution for BalanceVerdictScene (sim_gen task
`libero_pick_cream_cheese_i347`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. LOAD THE BALANCE (teleport transport + gravity): two single root-state writes
   carry the cartons, one at a time, from their ground spots to free space 15 mm
   above a weighing pan each (poses read back LIVE from the beam, oriented with
   it). Assignment is IDENTITY-AGNOSTIC — each carton goes to the pan on its own
   start side; the solver never touches the hidden mass assignment. Each carton
   FALLS onto its pan and the beam responds by contact: the `loaded` and `weighed`
   credit is produced by gravity and the hinge, never written.
2. READ THE VERDICT (pure readback): after the beam settles against its stop the
   solver reads WHICH PAN IS LOWER (`down_side()`, a world-z comparison of the two
   pan centres — information any camera could extract) and selects the carton
   sitting on that pan. This is the experiment the task exists for: before the
   weighing the two cartons are indistinguishable.
3. DELIVER (teleport transport + gravity): one root-state write carries the
   selected carton from its pan to free space above the basket mouth; it falls in
   and settles by contact. The beam, freed of the heavy carton, swings to the
   other stop on its own. Nothing else is touched.
4. ORDER: the weighing strictly precedes the delivery (the scene latches a
   permanent FOUL on any earlier basket entry).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_pick_cream_cheese_i347.solve --headless [--seed N]
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
    from . import scene as _task_scene  # noqa: F401 - importing registers the scene/env
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as _task_scene  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.balance_verdict")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def stand_local(pos_w: torch.Tensor) -> torch.Tensor:
        return quat_apply_inverse(scene.stand.data.root_quat_w,
                                  pos_w - scene.stand.data.root_pos_w)

    def angle() -> float:
        return float(scene.beam_angle_deg()[0])

    def report(tag: str) -> None:
        hl = stand_local(scene.heavy.data.root_pos_w)[0]
        ll = stand_local(scene.light.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | angle={angle():+.1f}deg "
              f"down={float(scene.down_side()[0]):+.0f} "
              f"heavy_std=({float(hl[0]):+.3f},{float(hl[1]):+.3f},{float(hl[2]):+.3f}) "
              f"light_std=({float(ll[0]):+.3f},{float(ll[1]):+.3f},{float(ll[2]):+.3f}) "
              f"loaded={bool(scene._loaded[0])} weighed={bool(scene._weighed[0])} "
              f"delivered={bool(scene._delivered[0])} foul={bool(scene._foul[0])} "
              f"in_h={bool(scene.in_basket(scene.heavy.data.root_pos_w)[0])} "
              f"in_l={bool(scene.in_basket(scene.light.data.root_pos_w)[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def drop_on_pan(body, side: torch.Tensor) -> None:
        """Transport-only write: body to free space 15 mm above the (live) pan
        floor centre on `side`, oriented with the beam; it falls in by gravity."""
        pan = scene.pan_center_w(side)
        up = quat_apply(scene.beam.data.root_quat_w,
                        torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pan + up * (c.box_size[2] / 2 + 0.015)
        st[:, 3:7] = scene.beam.data.root_quat_w
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(150)  # keel self-levels the empty beam; cartons/basket seat on the ground
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    sq = scene.stand.data.root_quat_w[0]
    syaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    m_beam = float(scene.beam.root_physx_view.get_masses()[0].sum())
    m_h = float(scene.heavy.root_physx_view.get_masses()[0].sum())
    m_l = float(scene.light.root_physx_view.get_masses()[0].sum())
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg "
          f"heavy_start_side={float(scene.heavy_side_start[0]):+.0f} "
          f"masses beam={m_beam:.3f} heavy={m_h:.3f} light={m_l:.3f}", flush=True)
    assert abs(m_beam - c.beam_mass) < 0.05, f"beam mass readback {m_beam} (MassAPI not applied?)"
    assert abs(m_h - c.heavy_mass) < 0.01 and abs(m_l - c.light_mass) < 0.01, \
        f"carton mass readback {m_h}/{m_l}"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(angle()) < 3.0, f"empty beam must self-level, got {angle():+.1f} deg"
    hl = stand_local(scene.heavy.data.root_pos_w)[0]
    ll = stand_local(scene.light.data.root_pos_w)[0]
    assert abs(float(hl[0]) - c.spot_x) < 0.06 and abs(float(ll[0]) - c.spot_x) < 0.06, \
        "cartons must start on their ground spots"
    assert float(hl[1]) * float(scene.heavy_side_start[0]) > 0, "heavy carton on its drawn side"
    s0 = print_score("P0 reset+settle (beam level, cartons on the ground, basket empty)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: load the pans (identity-agnostic), gravity weighs -----------
    # Each carton goes to the pan on its OWN start side — the solver does not know
    # (and does not use) which one is heavy.
    y_h = stand_local(scene.heavy.data.root_pos_w)[:, 1]
    side_h = torch.where(y_h > 0, torch.ones(n, device=device), -torch.ones(n, device=device))
    first, second = (scene.heavy, scene.light) if float(y_h[0]) > 0 else (scene.light, scene.heavy)
    side_first = side_h if first is scene.heavy else -side_h

    drop_on_pan(first, side_first)
    step(150)  # fall + the beam takes its first swing to a stop
    report("first->pan")
    assert bool(scene.on_pan(first, side_first)[0]), "first carton must rest on its pan"
    drop_on_pan(second, -side_first)
    step(300)  # the balance renders and settles its verdict against a stop
    report("second->pan")
    assert bool(scene.loaded_live()[0]), "both cartons must rest on opposite pans"
    assert bool(scene._loaded[0]) and bool(scene._weighed[0]), \
        f"loaded/weighed latches must be set (angle {angle():+.1f})"
    assert abs(angle()) >= c.tilt_min_deg, f"beam must be tipped, got {angle():+.1f}"
    assert not bool(scene._foul[0]) and not bool(scene.success()[0])
    s1 = print_score("P1 weighed: beam verdict rendered and latched")
    assert s1 >= c.w_loaded + c.w_weighed - 1e-5, f"P1 score {s1} (expect 0.40)"

    # ---------------- phase 2: read the verdict, deliver the full carton -------------------
    down = scene.down_side()
    target_is_heavy = bool(scene.on_pan(scene.heavy, down)[0])
    target = scene.heavy if target_is_heavy else scene.light
    print(f"[solve] verdict: down pan = {float(down[0]):+.0f} -> carton on it is "
          f"{'heavy' if target_is_heavy else 'light'} (hidden truth: physics must agree)",
          flush=True)
    assert target_is_heavy, "the balance's verdict must point at the heavy carton"

    bq = scene.basket.data.root_quat_w
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.basket.data.root_pos_w
    st[:, 2] += 0.145  # bottom clears the 100 mm walls; ~110 mm free fall into the mouth
    st[:, 3:7] = bq  # square to the basket so it falls between the walls
    target.write_root_state_to_sim(st, all_ids)
    step(120)  # fall + first settle; the freed beam swings to the other stop
    report("deliver-drop")
    ok_p2 = False
    for j in range(600):
        env.step(no_action)
        if bool(scene.success()[0]):
            ok_p2 = True
            break
        if j % 180 == 179:
            report(f"follow-{j + 1}")
    step(60)  # margin of settle, hands-off
    report("delivered")
    if not (ok_p2 or bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the delivery)", flush=True)
        os._exit(1)
    s2 = print_score("P2 delivered: full carton in the basket after the weighing")
    assert s2 >= s1 - 1e-6, "score decreased across the delivery"

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
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
