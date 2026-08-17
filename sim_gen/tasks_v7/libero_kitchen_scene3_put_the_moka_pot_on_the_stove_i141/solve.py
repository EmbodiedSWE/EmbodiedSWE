"""Teleport solution for MokaBalanceScene (sim_gen task
`libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i141`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): single root-state writes carry ONE object at a time
   across FREE SPACE with zero velocity — exactly the carry a gripper performs.
   Every write ends at a HOVER a few millimetres above the destination pan's
   platform (inside the pan's open top, below its hanger crossbar), never
   intersecting anything.
2. WEIGHING (gravity + contact + the passive mechanism): after each release the
   object FALLS onto the pan, the hanging pan transmits the load to the beam
   through its free pivot, and the BEAM responds — slamming to its stop under
   the unmatched pot, then swinging back and settling LEVEL only once the blue
   pan carries the exact counterweight. Nothing about the balance is ever
   written after reset: the measurement outcome is 100 % contact dynamics.
3. The ORACLE part (allowed for the solver, hidden from a policy): the solve
   reads the pot's randomized mass from the scene's reset readback cache to
   pick the right weight combination directly instead of trial-and-error. The
   physics does not care how the combination was chosen — only whether it is
   exactly right.

If a release leaves an object outside its pan (a bounce), it is picked up again
(fresh transport to the same hover) — a retry, not a cheat: the final
configuration is still 100 % contact-made and the final 3.3 s are hands-off.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i141.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.moka_balance")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    from isaaclab.utils.math import quat_apply

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
        tilt = math.degrees(float(scene.beam_tilt()[0]))
        w = float(scene.beam.data.root_ang_vel_w.norm(dim=-1)[0])
        in_r = scene.in_basket(scene.basket_r)[0]
        in_b = scene.in_basket(scene.basket_b)[0]
        imb = float(scene.imbalance()[0]) * 1000
        print(f"[solve] {tag:12s} | tilt={tilt:+6.2f}deg w={w:.3f} "
              f"pot_in_red={bool(in_r[0])} wts_in_blue={int(in_b[1:7].sum())} "
              f"imb={imb:+.1f}g settled={bool(scene.settled()[0])} "
              f"latches=(p={bool(scene._l_pot[0])},w={bool(scene._l_wt[0])},"
              f"l={bool(scene._l_lvl[0])}) "
              f"success={bool(scene.success()[0])} "
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

    def hover_pose(basket, local_xy, hover_dz: float):
        """World pose (identity attitude) hovering `hover_dz` above the pan's
        platform at pan-local xy, computed from the LIVE pan pose."""
        lp = torch.tensor([local_xy[0], local_xy[1], -c.platform_drop + hover_dz],
                          device=device).expand(n, 3)
        pos = basket.data.root_pos_w + quat_apply(basket.data.root_quat_w, lp)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        return pos, quat

    def write_pose(body, pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def settle(max_steps: int, min_steps: int = 120) -> None:
        for i in range(max_steps):
            env.step(no_action)
            if i >= min_steps and bool(scene.settled()[0]) \
                    and bool(scene.beam_calm()[0]):
                break

    def drop_into(body, basket, local_xy, payload_idx: int,
                  settle_steps: int = 420) -> None:
        """Transport `body` to the in-pan hover, release, and let contact
        dynamics seat it. Retries a bounced-out release from the same hover."""
        for attempt in range(3):
            pos, quat = hover_pose(basket, local_xy, 0.008)
            write_pose(body, pos, quat)
            settle(settle_steps)
            if bool(scene.in_basket(basket)[0, payload_idx]):
                return
            print(f"[solve] drop bounced out (attempt {attempt + 1}); retrying",
                  flush=True)
        raise AssertionError(f"payload {payload_idx} would not seat in the pan")

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # everything seats (pot on the stove, weights on the deck)
    pot_g = round(float(scene.masses_cache[0, 0]) * 1000)
    sp = (scene.stove.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): pot_mass={pot_g}g "
          f"stove=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) "
          f"tilt={math.degrees(float(scene.beam_tilt()[0])):+.2f}deg", flush=True)
    report("reset")
    assert torch.isfinite(scene.pot.data.root_pos_w).all(), "NaN/inf after settle"
    assert bool(scene.pot_on_stove()[0]), "pot must start standing on the stove"
    assert abs(math.degrees(float(scene.beam_tilt()[0]))) < c.level_max_deg, \
        "empty balance must rest level"
    s0 = print_score("P0 reset+settle (pot on the unlit stove, balance empty)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: pot into the RED pan ----------------------------------------
    drop_into(scene.pot, scene.basket_r, (0.0, 0.0), payload_idx=0, settle_steps=600)
    report("pot-in-red")
    assert bool(scene._l_pot[0]), "pot latch must fire once the pot rests in red"
    tilt1 = math.degrees(float(scene.beam_tilt()[0]))
    assert tilt1 < -(c.beam_limit_deg - 3.0), \
        f"the unmatched pot must slam the beam to its red-side stop (tilt {tilt1:.1f})"
    s1 = print_score("P1 pot in the red pan (beam slammed to its stop)")
    assert s1 >= 0.19, f"P1 score {s1} (expect pot latch 0.20)"

    # ---------------- phase 2: counterweights into the BLUE pan ----------------------------
    # oracle: the exact combination for the hidden mass (weights list indices)
    combos = {150: [2, 0], 200: [4], 250: [4, 0], 300: [4, 2], 350: [4, 2, 0]}
    spots = {1: [(0.0, 0.0)],
             2: [(-0.032, 0.0), (0.032, 0.0)],
             3: [(-0.032, 0.016), (0.032, 0.016), (0.0, -0.032)]}
    combo = combos[pot_g]
    put = spots[len(combo)]
    print(f"[solve] combo for {pot_g}g: "
          f"{[scene.WEIGHT_KEYS[k] for k in combo]}", flush=True)
    for j, (widx, xy) in enumerate(zip(combo, put)):
        drop_into(scene.weights[widx], scene.basket_b, xy, payload_idx=1 + widx,
                  settle_steps=360)
        if j == 0:
            report("first-weight")
            assert bool(scene._l_wt[0]), "weight latch must fire in the blue pan"
            s2 = print_score("P2 first counterweight resting in the blue pan")
            assert s2 >= 0.39, f"P2 score {s2} (expect 0.40)"

    # ---------------- phase 3: the mechanism answers — beam returns level ------------------
    # Wait for success to hold CONTINUOUSLY for 2 simulated seconds (240
    # substeps) so the phase boundary marks a genuinely settled beam, not the
    # first mid-swing pass through the level band.
    ok = False
    streak = 0
    for _ in range(4800):  # up to 40 s: slow mode ~4.8 s period, zeta ~0.76
        env.step(no_action)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 240:
            ok = True
            break
    report("balanced")
    imb = abs(float(scene.imbalance()[0])) * 1000
    assert ok and bool(scene.success()[0]), \
        f"beam did not settle level (tilt " \
        f"{math.degrees(float(scene.beam_tilt()[0])):+.2f}deg, imb {imb:.1f}g)"
    assert bool(scene._l_lvl[0]), "balanced latch must be set"
    s3 = print_score("P3 free beam settled level — the balance certifies the mass")
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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
