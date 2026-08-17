"""Teleport solution for SoundingWellsScene (sim_gen task
`living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i401`) — the task's
legitimacy certificate: a genuine SENSE-then-COMMIT loop.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the probe rod from
   wherever it rests to a free-space HOVER pose over a well mouth — upright, yaw
   aligned with the bore, shaft tip 10 mm ABOVE the rim (open air; nothing is
   inserted by the write, velocity is zero). This is exactly the carry a gripper
   performs — including the WITHDRAWAL from a stalled probe, where the real cap
   sits ~36 mm proud and graspable.
2. SOUNDING (gravity + contact, hands-off): from the hover the rod FALLS into the
   bore. What stops it is real contact: the hidden filler slug (cap stalls ~36 mm
   above the rim) or, in the one deep well, the rim itself (cap seats flush).
   The MEASUREMENT the plan branches on is the physical readback of where the cap
   came to rest — never the scene's hidden `true_idx`.
3. SEARCH POLICY: wells are probed in the fixed canonical order 0, 1, 2 until the
   cap seats flush; the hidden state is only used AFTER the fact, to verify that
   the physically-sensed answer matches ground truth. Different seeds hide the
   deep well in different places, so the number of probes differs per seed.
4. The DUMMY BAR is never touched (identity restraint).

If a drop leaves the tip outside the bore (cocked on the mouth), the rod is picked
up again (transport to the same free-space hover) and re-dropped — a retry, not a
cheat: every rest state is 100 % contact-made.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket_i401.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sounding_wells")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

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
        tip = scene._rod_tip_w()[0]
        gap = float(scene.cap_gap()[0])
        print(f"[solve] {tag:14s} | rod=({float(scene.rod.data.root_pos_w[0, 0]):+.3f},"
              f"{float(scene.rod.data.root_pos_w[0, 1]):+.3f},"
              f"{float(scene.rod.data.root_pos_w[0, 2]):+.3f}) "
              f"tip_z={float(tip[2]):+.3f} up_z={float(scene._rod_up_z()[0]):+.2f} "
              f"true_gap={gap * 1000:+.1f}mm "
              f"sounded={bool(scene.sounded_now()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hover_and_drop(i: int, max_steps: int = 240) -> None:
        """TRANSPORT the rod to the free-space hover over well i's mouth (upright,
        yaw-aligned, tip 10 mm above the rim, zero velocity), then hands-off fall
        and full settle — what stops the rod is pure contact."""
        wp = scene.wells[i].data.root_pos_w
        wq = scene.wells[i].data.root_quat_w
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = wp[:, 0:2]
        st[:, 2] = wp[:, 2] + c.base_t + c.wall_h + 0.010 + c.shaft_len
        st[:, 3:7] = wq
        scene.rod.write_root_state_to_sim(st, all_ids)
        for k in range(max_steps):
            env.step(no_action)
            if k > 30 and float(scene.rod.data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_speed:
                break
        step(40)  # extra hands-off settle

    def probe(i: int) -> float:
        """Sound well i: drop (with up to 3 re-drops if the rod cocks on the mouth)
        and return the MEASURED cap height above well i's rim (m)."""
        for attempt in range(3):
            hover_and_drop(i)
            if bool(scene._tip_in_bore()[0, i]):
                gap_i = float(scene.rod.data.root_pos_w[0, 2] - scene._rim_z()[0, i])
                print(f"[solve] probe well {i}: tip seated in bore, cap "
                      f"{gap_i * 1000:+.1f} mm above the rim", flush=True)
                return gap_i
            print(f"[solve] probe well {i}: rod cocked on the mouth (attempt "
                  f"{attempt + 1}); re-dropping", flush=True)
            report(f"probe{i}-retry")
        report(f"probe{i}-FAIL")
        print(f"SIM_GEN_SOLVE: FAIL (rod never entered well {i})", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything seats on the ground
    o = scene.env_origins[0]
    rp = scene.rod.data.root_pos_w[0] - o
    dp = scene.dummy.data.root_pos_w[0] - o
    wxy = [(float(w.data.root_pos_w[0, 0] - o[0]), float(w.data.root_pos_w[0, 1] - o[1]))
           for w in scene.wells]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"wells={[(f'{x:+.3f}', f'{y:+.3f}') for x, y in wxy]} "
          f"rod=({float(rp[0]):+.3f},{float(rp[1]):+.3f},{float(rp[2]):+.3f}) "
          f"dummy=({float(dp[0]):+.3f},{float(dp[1]):+.3f}) "
          f"[hidden true_idx={int(scene.true_idx[0])} — verification only]", flush=True)
    report("reset")
    assert torch.isfinite(scene.rod.data.root_state_w).all(), "NaN/inf after settle"
    assert float(rp[2]) < 0.06, f"rod must lie on the ground, z={float(rp[2]):.3f}"
    # slugs really sit inside the two decoy wells (bore-frame readback)
    for k, slug in enumerate(scene.slugs):
        decoy = int((scene.true_idx[0] + 1 + k) % 3)
        d = (slug.data.root_pos_w[0, :2] - scene.wells[decoy].data.root_pos_w[0, :2]).norm()
        sz = float(slug.data.root_pos_w[0, 2] - scene.wells[decoy].data.root_pos_w[0, 2])
        assert float(d) < 0.01 and c.base_t - 0.002 < sz < c.base_t + c.slug_h, \
            f"slug {k} not seated in decoy well {decoy}: dxy={float(d):.3f} z={sz:.3f}"
    s0 = print_score("P0 reset+settle (rod and dummy scattered on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1..: sound the wells in canonical order ------------------------
    flush_thresh = c.cap_gap_tol + 0.010  # measured decision line, far below the ~36 mm stall
    found = -1
    last_s = s0
    for i in range(3):
        gap_i = probe(i)
        report(f"after-probe-{i}")
        s = print_score(f"P{i + 1} sounded well {i}: cap rest {gap_i * 1000:+.1f} mm above rim")
        assert s >= last_s - 1e-6, f"score decreased across probe {i}"
        last_s = s
        if gap_i < flush_thresh:
            found = i
            break
        assert gap_i > 0.020, \
            f"stalled probe should sit clearly proud, measured {gap_i * 1000:.1f} mm"
        assert not bool(scene.success()[0]), "stalled probe must not be success"
    if found < 0:
        print("SIM_GEN_SOLVE: FAIL (no well accepted the rod flush)", flush=True)
        os._exit(1)

    # the physically-sensed answer must match the hidden assignment (verification only)
    assert found == int(scene.true_idx[0]), \
        f"sensed deep well {found} != hidden true_idx {int(scene.true_idx[0])}"
    step(60)
    report("seated")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (flush rest did not judge as success)", flush=True)
        os._exit(1)
    s_seat = print_score(f"P4 rod planted flush in the deep well {found}")
    assert s_seat >= last_s - 1e-6

    # ---------------- final phase: persistence (>= 3 simulated seconds, hands-off) ---------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_end = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_end >= s_seat - 1e-6
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
