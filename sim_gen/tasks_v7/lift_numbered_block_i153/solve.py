"""Teleport solution for BeamHoistScene (sim_gen task `lift_numbered_block_i153`) — the
task's legitimacy certificate.

Teleports are used for TRANSPORT ONLY (the stand-in for the Franka's pick-and-carry of
each easily-graspable 45 mm block); every load-bearing interaction is contact/joint
dynamics:

  1. SEAT TARGET (contact): the placard-matching block is carried to a hover pose over
     the amber cradle and RELEASED — gravity + contact seat it on the cradle floor of
     the still-down beam.
  2. COUNTERWEIGHT A (contact): the first distractor is carried over the weight pan and
     released into the +y pan slot. One counterweight is provably insufficient — the
     beam must stay cradle-down (asserted here against the scene's torque margins).
  3. COUNTERWEIGHT B (contact + hinge statics): the second distractor is released into
     the -y pan slot; the pan side now out-torques the cradle side and the HINGE does
     the hoisting — the beam tips against its upper stop, raising the cradle and the
     numbered block riding in it. No wrench ever touches the beam or the seated target;
     the success state is never spawned.
  4. PERSISTENCE: >= 3.5 simulated seconds hands-off; success must hold.

The target identity and every drop pose are READ from the scene per episode (placard /
target_idx, beam pose via `beam_to_world`), so one script serves all seeds. Prints
`SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's credit is
latched) and `SIM_GEN_SOLVE: SUCCESS` only if success still holds after the hold.

Run (forge): python -u -m simgen_tasks.lift_numbered_block_i153.solve --headless [--seed N]
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
    env = ENVS.get("simgen.beam_hoist")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def theta_deg() -> float:
        return math.degrees(float(scene.beam_theta()[0]))

    def report(tag: str) -> None:
        tgt, d1, d2 = scene._role_flags()
        print(f"[solve] {tag:12s} | theta={theta_deg():+.1f}deg"
              f" seated={bool(tgt[0])} panA={bool(d1[0])} panB={bool(d2[0])}"
              f" latches=({float(scene.seat_latch[0]):.0f},"
              f"{float(scene.panA_latch[0]):.0f},{float(scene.panB_latch[0]):.0f},"
              f"{float(scene.up_latch[0]):.0f})"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def carry_and_release(block_idx: int, local_xy: tuple, tag: str,
                          settle_max: int = 420) -> None:
        """TRANSPORT teleport: place block `block_idx` at a hover pose a few cm above
        the given beam-frame tray point (beam-aligned orientation, zero velocity),
        then release — gravity + contact do the seating. Waits until the whole scene
        is settled (or settle_max substeps)."""
        local = torch.tensor([[local_xy[0], local_xy[1], c.seat_z_b + 0.045]],
                             device=device).expand(n, 3)
        world = scene.beam_to_world(local)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = world
        st[:, 3:7] = scene.beam.data.root_quat_w
        scene.blocks[block_idx].write_root_state_to_sim(st, all_ids)
        for i in range(settle_max):
            env.step(no_action)
            if i > 60 and bool(scene.settled()[0]):
                break
        print(f"[solve] {tag}: released block_{block_idx} -> settled "
              f"@theta={theta_deg():+.1f}deg", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    tgt_i = int(scene.target_idx[0])
    d1_i, d2_i = int(scene.dist_idx[0, 0]), int(scene.dist_idx[0, 1])
    print(f"[solve] layout readback (seed {args.seed}): target=block_{tgt_i} "
          f"(pips {tgt_i + 1}) distractors=block_{d1_i},block_{d2_i} "
          f"theta={theta_deg():+.1f}deg", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"
    assert theta_deg() < -8.0, "empty beam must rest cradle-down"

    # ---------------- phase 1: seat the numbered block in the cradle (contact) -------------
    carry_and_release(tgt_i, (c.cradle_cx_b, 0.0), "P1 seat")
    report("seat")
    tgt, _d1, _d2 = scene._role_flags()
    assert bool(tgt[0]), "target block did not seat in the cradle"
    s1 = print_score("P1 numbered block seated in the cradle (gravity + contact)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.15 - 1e-6, "seat stage credit missing"

    # ---------------- phase 2: first counterweight (contact; provably insufficient) --------
    carry_and_release(d1_i, (c.pan_cx_b, 0.045), "P2 counterweight A")
    report("panA")
    _tgt, d1, _d2 = scene._role_flags()
    assert bool(d1[0]), "distractor A did not land in the pan"
    assert theta_deg() < -8.0, \
        "ONE counterweight tipped the beam — torque margins violated"
    s2 = print_score("P2 first counterweight in the pan (beam must stay down)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.30 - 1e-6, "panA stage credit missing"

    # ---------------- phase 3: second counterweight -> the hinge hoists the cradle ---------
    carry_and_release(d2_i, (c.pan_cx_b, -0.045), "P3 counterweight B",
                      settle_max=720)
    for _ in range(20):  # up to 5 s extra for the tip + settle
        if bool(scene.success()[0]):
            break
        step(30)
    report("hoisted")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after loading the beam)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)
    s3 = print_score("P3 second counterweight: beam tips, cradle hoisted")
    assert s3 >= s2 - 1e-6 and s3 >= 1.0 - 1e-6, "success credit missing"
    assert theta_deg() > c.theta_up_min_deg - 0.5, \
        "beam not raised past the threshold"

    # ---------------- phase 4: persistence (>= 3.5 simulated seconds, hands-off) -----------
    hold, flickers = True, 0
    for i in range(420):  # 420 substeps = 3.5 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                tgt, d1, d2 = scene._role_flags()
                print(f"[solve] persist flicker @step {i}: seated={bool(tgt[0])} "
                      f"panA={bool(d1[0])} panB={bool(d2[0])} "
                      f"up={bool(scene.beam_up()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"theta={theta_deg():+.2f} "
                      f"beam_w={float(scene.beam.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s4 = print_score("P4 persistence 3.5 s")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
