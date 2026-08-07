"""Teleport solution for CounterweighBasketScene (sim_gen task
`living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket_i80`) — the
task's legitimacy certificate.

Teleports carry objects through FREE SPACE only; every load-bearing interaction — the
drops onto the beam, the beam's swings, and the final settle to level — is contact
dynamics on the free revolute joint:

1. LOAD THE TRAY (transport, teleport + drop): one pose write places the dark METAL
   block 3 cm ABOVE the tray at the beam's -x end (the arm's version is a pinch-lift of
   the 45 mm cube and a release above the tray). The drop, the tray catching the block,
   and the beam swinging tray-end-down to its ~16 deg loaded equilibrium are physics.
   Selecting THIS block is the task's reasoning step: the white foam decoy would leave
   the beam ~15 deg tipped.
2. LOAD THE BASKET (transport, teleport + drop): one pose write places the blue cheese
   box just above the basket floor at the +x end (now raised ~16 deg), oriented with
   the beam so it drops flat between the walls. The drop and the beam's counterswing
   back toward level are physics; nothing holds or guides the beam.
3. HANDS OFF: the loaded balance rings down (loaded balanced mechanisms ring longest —
   generous settle budget). success() needs box-in-basket AND |pitch| <= 8 deg AND a
   60-consecutive-step stillness latch, none of which a teleport can fake: the level
   condition is torque equilibrium, physically reachable only with the matching
   counterweight on the opposite end.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then, after success() first turns True, keeps simulating >= 5.3
simulated seconds hands-off and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds throughout the strict persistence window.

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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counterweigh_basket")().build(num_envs=args.num_envs,
                                                         device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def pitch() -> float:
        return float(scene.beam_pitch_deg()[0])

    def report(tag: str) -> None:
        bl = scene._beam_local(scene.box)[0]
        ml = scene._beam_local(scene.metal)[0]
        print(f"[solve] {tag:12s} | pitch={pitch():+6.2f}deg "
              f"box_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
              f"metal_loc=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"in_basket={bool(scene._in_basket(scene.box)[0])} "
              f"on_tray={bool(scene._on_tray(scene.metal)[0])} "
              f"cw_latch={bool(scene._cw_latch[0])} box_latch={bool(scene._box_latch[0])} "
              f"still_n={int(scene._still_n[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(budget: int, min_steps: int = 60) -> int:
        """Step until the scene's own stillness latch fires. Always steps at least
        `min_steps` — a freshly teleported body reads zero velocity, so consulting a
        stillness predicate before physics has run is vacuous."""
        for i in range(budget):
            env.step(no_action)
            if i + 1 >= min_steps and bool(scene.still()[0]):
                return i + 1
        return budget

    def beam_pose():
        return scene.beam.data.root_pos_w, scene.beam.data.root_quat_w

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    sq = scene.stand.data.root_quat_w[0]
    s_yaw = 2.0 * math.atan2(float(sq[3]), float(sq[0]))
    b0 = (scene.box.data.root_pos_w - scene.env_origins)[0]
    m0 = (scene.metal.data.root_pos_w - scene.env_origins)[0]
    f0 = (scene.foam.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={math.degrees(s_yaw):+.1f}deg "
          f"box=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"metal=({float(m0[0]):+.3f},{float(m0[1]):+.3f}) "
          f"foam=({float(f0[0]):+.3f},{float(f0[1]):+.3f})", flush=True)
    report("reset")
    assert abs(pitch()) <= 3.0, f"empty beam did not self-level, pitch={pitch():+.2f}"
    s0 = print_score("P0 reset+settle (empty beam self-levelled)")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: LOAD THE TRAY (teleport 3 cm above, drop) --------------------
    # One pose write: the METAL block (0.20 kg — the box's match; the foam decoy would
    # leave the beam tipped) to 3 cm above the tray centre, oriented with the beam.
    # Small -z velocity so the stillness counter hard-resets on the first post_step
    # (a zero-velocity teleport reads vacuously still before physics runs).
    b_pos, b_quat = beam_pose()
    rest_z = c.floor_z + 0.004 + c.block_size / 2  # block CoM resting on the tray floor
    hover = torch.tensor([-c.arm, 0.0, rest_z + 0.030], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = b_pos + quat_apply(b_quat, hover)
    st[:, 3:7] = b_quat
    st[:, 9] = -0.10
    scene.metal.write_root_state_to_sim(st, all_ids)
    k1 = settle(1800)
    report(f"tray({k1})")
    assert bool(scene._on_tray(scene.metal)[0]), "metal block did not settle on the tray"
    assert pitch() >= 10.0, \
        f"beam did not respond to the counterweight (pitch={pitch():+.2f}, expect ~+16)"
    s1 = print_score("P1 metal counterweight dropped onto the tray, beam swung")
    assert s1 >= c.w_cw - 1e-4, f"tray credit not latched, score={s1}"
    assert s1 >= s0 - 1e-6, "score decreased across P1"

    # ---------------- phase 2: LOAD THE BASKET (teleport just above the floor, drop) --------
    # One pose write: the box just above the basket floor at the raised +x end, oriented
    # WITH the beam so it drops flat between the walls. The counterswing back to level
    # is pure torque physics.
    b_pos, b_quat = beam_pose()
    box_rest = c.floor_z + 0.004 + c.box_dims[2] / 2
    drop = torch.tensor([c.arm, 0.0, box_rest + 0.020], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = b_pos + quat_apply(b_quat, drop)
    st[:, 3:7] = b_quat
    st[:, 9] = -0.10
    scene.box.write_root_state_to_sim(st, all_ids)
    k2 = settle(2400)
    report(f"basket({k2})")
    assert bool(scene._in_basket(scene.box)[0]), "box did not settle inside the basket"
    s2 = print_score("P2 box dropped into the basket, beam counterswung to level")
    assert s2 >= s1 - 1e-6, "score decreased across P2"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL (no success after settle; pitch={pitch():+.2f})",
              flush=True)
        os._exit(1)

    # ---------------- phase 3: extra hands-off ring-down (2 s) ------------------------------
    # The balance's residual sway can hover at the stillness threshold; give it 2 s of
    # extra hands-off settling BEFORE opening the strict persistence window.
    step(240)
    report("ringdown")
    s3 = print_score("P3 2 s extra hands-off settle")
    assert s3 >= s2 - 1e-6, "score decreased across the ring-down"

    # ---------------- phase 4: strict persistence (3.3 s, success at every block) -----------
    hold = bool(scene.success()[0])
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
