"""Teleport solution for RelayCascadeScene (sim_gen task `franka_handover_i356`) —
the task's legitimacy certificate.

Teleports are TRANSPORT ONLY (relocating a body the solver is already holding in
free air). Every load-bearing interaction is contact dynamics or applied torque:

1. PERCEPTION: the housing pose, WHICH colour the pedestal tile names, where that
   cube sits, and both tray rest angles are read back from the episode state —
   never hard-coded.
2. DROP (gravity + contact): the target cube is TELEPORTED from its apron slot to
   free air above the roof intake mouth (open sky above both endpoints) and
   RELEASED. Gravity drops it through the mouth onto tray 1; the tray deck and
   retainer — real contacts — cradle it at the rest tilt.
3. PRESS L1 (applied torque): the scene's sanctioned `lever_drive[:, 0]` torque
   (the stand-in for a fingertip press on the protruding rod, plant-clamped at
   0.5 N*m ~ 7 N at the rod) drives tray 1 to its +30 deg stop; the cube slides
   off the deck — pure contact dynamics — and lands on tray 2. Release: the
   ballast gravity-returns the tray.
4. PRESS L2: same, `lever_drive[:, 1]` to the -35 deg stop; the cube slides out
   over the partition into the sealed bin. Release; everything settles.
5. VERIFY: hands off (drives zero), success() must hold and keep holding for
   >= 3 more simulated seconds before SIM_GEN_SOLVE: SUCCESS prints.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches).

Run (forge): python -u -m simgen_tasks.franka_handover_i356.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.relay_cascade")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

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
        ph = scene.phi()[0]
        tgt_in, dec_in = scene._cube_flags()
        tgt = int(scene.target_idx[0])
        loc = scene._housing_local(scene.cubes[tgt].data.root_pos_w)[0]
        print(f"[solve] {tag:16s} | phi=({math.degrees(float(ph[0])):+6.1f},"
              f"{math.degrees(float(ph[1])):+6.1f})deg "
              f"cube_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"tgt_in_bin={bool(tgt_in[0])} dec_in_bin={bool(dec_in[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def press(k: int, sign: float, hold: int, tag: str) -> None:
        """Drive lever k to its stop with the sanctioned clamped torque, hold, release,
        gravity-return, settle. Asserts the tray actually moved to the stop."""
        lim = math.radians((c.t1_lim, c.t2_lim)[k][1 if sign > 0 else 0])
        rest = math.radians((c.t1_rest, c.t2_rest)[k])
        peak = torch.full((n,), math.inf if sign < 0 else -math.inf, device=device)
        scene.lever_drive[:, k] = sign * c.tau_max
        for _ in range(hold):
            env.step(no_action)
            ph = scene.phi()[:, k]
            peak = torch.minimum(peak, ph) if sign < 0 else torch.maximum(peak, ph)
        assert bool(((peak - lim).abs() < math.radians(4.0)).all()), \
            f"{tag}: tray must reach its pressed stop (peak={peak.tolist()})"
        scene.lever_drive[:, k] = 0.0
        step(240)  # gravity return + cargo settle
        ph = scene.phi()[:, k]
        assert bool(((ph - rest).abs() < math.radians(4.0)).all()), \
            f"{tag}: tray must gravity-return to rest (phi={ph.tolist()})"

    # ----- phase 0: settle + perception ----------------------------------------------------------
    step(60)
    report("settled start")
    tgt = scene.target_idx.clone()
    ph0 = scene.phi()
    print(f"[solve] perception: target colour = {c.color_names[int(tgt[0])]} "
          f"(tile readback), slots = {scene.slot_of[0].tolist()}, "
          f"tray rest = ({math.degrees(float(ph0[0, 0])):+.1f},"
          f"{math.degrees(float(ph0[0, 1])):+.1f}) deg", flush=True)
    assert abs(math.degrees(float(ph0[0, 0])) - c.t1_rest) < 3.0
    assert abs(math.degrees(float(ph0[0, 1])) - c.t2_rest) < 3.0
    s0 = print_score("start")
    assert s0 < 0.05, f"fresh episode must score ~0, got {s0}"

    # ----- phase 1: DROP — transport the target cube above the intake mouth, release -------------
    h_p = scene.housing.data.root_pos_w
    h_q = scene.housing.data.root_quat_w
    my = (c.mouth_y0 + c.mouth_y1) / 2
    local = torch.tensor([0.0, my, c.top_z1 + c.cube_size / 2 + 0.03],
                         device=device).expand(n, 3)
    drop_pos = h_p + quat_apply(h_q, local)
    for k in range(2):
        ids = torch.nonzero(tgt == k, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            continue
        st = torch.zeros(ids.numel(), 13, device=device)
        st[:, 0:3] = drop_pos[ids]
        st[:, 3:7] = h_q[ids]  # faces parallel to the shaft walls
        scene.cubes[k].write_root_state_to_sim(st, ids)
    step(180)  # free fall through the mouth + cradle on tray 1
    report("dropped")
    assert bool(scene._intake.all()), "target cube must land on tray 1"
    s1 = print_score("target on tray 1")
    assert s1 >= c.w_intake - 0.02, f"intake latch must pay, got {s1}"

    # ----- phase 2: PRESS L1 — tray 1 discharges the cube onto tray 2 ----------------------------
    press(0, +1.0, 300, "L1")
    report("after L1")
    assert bool(scene._stage.all()), "target cube must transfer to tray 2"
    s2 = print_score("target on tray 2")
    assert s2 >= s1, "score must not decrease"
    assert s2 >= c.w_intake + c.w_stage - 0.02, f"stage latch must pay, got {s2}"

    # ----- phase 3: PRESS L2 — tray 2 discharges the cube into the bin ---------------------------
    press(1, -1.0, 300, "L2")
    report("after L2")
    assert bool(scene._bin.all()), "target cube must reach the bin"
    s3 = print_score("target in bin")
    assert s3 >= s2, "score must not decrease"

    # ----- phase 4: VERIFY — hands off, success must hold and keep holding -----------------------
    step(120)
    report("verify")
    ok = scene.success()
    assert bool(ok.all()), "success() must hold before the persistence window"
    s4 = print_score("success reached")
    assert s4 >= 0.999, f"live success must score 1.0, got {s4}"

    step(400)  # >= 3.3 simulated seconds, hands off
    report("persistence")
    ok = scene.success()
    s5 = float(scene.score()[0])
    print(f"SIM_GEN_SCORE {s5:.4f}", flush=True)
    if bool(ok.all()) and s5 >= 0.999:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        os._exit(0)
    print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)
    os._exit(1)


try:
    main()
except Exception as e:  # noqa: BLE001
    print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
    os._exit(1)
