"""Teleport solution for WedgeJackScene (sim_gen task `two_robot_pick_cube_i315`) —
the task's legitimacy certificate.

Teleports are TRANSPORT ONLY (relocating the crate through free air). Every
load-bearing interaction is contact dynamics or the scene's sanctioned drive force:

1. PERCEPTION: the station pose, the sampled ram start and the crate's apron slot are
   read back from the episode state — never hard-coded (xw()/crate pose readbacks are
   cross-checked against the scene's recorded samples).
2. LOAD (gravity + contact): the crate is TELEPORTED from its apron slot to free air
   4 cm above the deck centre (the deck is open from above) and RELEASED. Gravity
   drops it in; the deck plate and rim — real contacts — seat it. The rubric judges
   the settled seated pose, not the carry.
3. JACK (applied force): a feedforward + velocity servo writes the scene's sanctioned
   `ram_drive` (the stand-in for a sustained push on the pale end plate, clamped at
   25 N by the plant): drive = clamp(f_ff + 60*(v_des - ins_rate), 0, 24) with
   v_des = clamp(2*(xw - xw_goal), 0.008, 0.05) m/s. f_ff starts at 8 N (~1.2x the
   analytic insertion load of 6.8 N) and escalates 1.5x on stall (memory: escalate
   the GAIN/authority, never mask a stall). Effective velocity gain
   (60 + 6)/(120 * 1.2) = 0.46 < 1 — discretely stable with one-substep-late
   wrenches. The wedge feeds under the foot and the platform + crate RIDE UP —
   pure contact mechanics; the drive never touches the car or the crate.
4. VERIFY: drive cut to zero — friction self-locks the wedge (mu 0.60 >= 2.24x
   tan 15 deg), so the raised state must persist hands-off. success() must hold and
   keep holding for >= 3.3 more simulated seconds before SIM_GEN_SOLVE: SUCCESS.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches): 0.00 start -> 0.30 loaded -> 1.00 jacked -> 1.00 persisted.

Run (forge): python -u -m simgen_tasks.two_robot_pick_cube_i315.solve --headless [--seed N]
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
    env = ENVS.get("simgen.wedge_jack")().build(num_envs=args.num_envs, device=device)
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
        print(f"[solve] {tag:16s} | q={float(scene.q()[0]) * 1000:+7.2f}mm "
              f"xw={float(scene.xw()[0]) * 1000:+7.1f}mm "
              f"q_ram={float(scene.q_from_ram()[0]) * 1000:+7.2f}mm "
              f"ins_rate={float(scene.ins_rate[0]):+.3f} "
              f"aboard={bool(scene.aboard()[0])} sup={bool(scene.supported()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- phase 0: settle + perception ----------------------------------------------------------
    step(60)
    report("settled start")
    xw_rb = scene.xw()
    err = (xw_rb - scene.xw0).abs().max()
    assert float(err) < 0.004, f"ram start readback must match the sample, err {float(err):.4f}"
    print(f"[solve] perception: ram start xw0 = {float(xw_rb[0]) * 1000:.1f} mm "
          f"(sampled {float(scene.xw0[0]) * 1000:.1f} mm), crate slot = "
          f"({float(scene.crate_slot[0, 0]):+.3f}, {float(scene.crate_slot[0, 1]):+.3f}) "
          f"station-local — all readbacks, nothing hard-coded", flush=True)
    s0 = print_score("start")
    assert s0 < 0.05, f"fresh episode must score ~0, got {s0}"

    # ----- phase 1: LOAD — transport the crate above the deck, release, gravity seats it ---------
    car_p = scene.car.data.root_pos_w
    car_q = scene.car.data.root_quat_w
    local = torch.tensor([0.0, 0.0, c.crate_size / 2 + 0.04], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = car_p + quat_apply(car_q, local)
    st[:, 3:7] = car_q  # faces parallel to the rim walls
    scene.crate.write_root_state_to_sim(st, torch.arange(n, device=device))
    step(150)  # free fall 4 cm + settle inside the rim
    report("loaded")
    assert bool(scene.aboard().all()), "crate must be seated on the deck after the drop"
    s1 = print_score("crate loaded on deck")
    assert s1 >= 0.29, f"aboard latch must pay, got {s1}"

    # ----- phase 2: JACK — feedforward + velocity servo on the sanctioned ram_drive --------------
    q0c, tan_t = scene.q_at_coeff()
    q_goal = c.q_req + 0.002                    # 2 mm past the requirement, 3.3 mm short of the stop
    xw_goal = (q0c - q_goal) / tan_t
    print(f"[solve] jack plan: xw {float(scene.xw()[0]) * 1000:+.1f} -> "
          f"{xw_goal * 1000:+.1f} mm for q >= {q_goal * 1000:.1f} mm", flush=True)
    f_ff = 8.0
    xw_ckpt = scene.xw().clone()
    for i in range(3000):
        rem = scene.xw() - xw_goal
        v_des = (2.0 * rem).clamp(0.008, 0.05)
        scene.ram_drive[:] = (f_ff + 60.0 * (v_des - scene.ins_rate)).clamp(0.0, c.f_max - 1.0)
        env.step(no_action)
        if bool((scene.q() >= q_goal).all()):
            break
        if i % 180 == 179:
            xw_now = scene.xw()
            if float((xw_ckpt - xw_now).min()) < 0.002:  # < 2 mm progress in 1.5 s
                f_ff = min(f_ff * 1.5, 18.0)
                print(f"[solve] stall — escalating feedforward to {f_ff:.1f} N", flush=True)
            xw_ckpt = xw_now.clone()
            report(f"jacking {i + 1}")
    scene.ram_drive[:] = 0.0
    step(120)  # self-locking wedge: the raised state must survive the drive cut
    report("jacked")
    assert bool((scene.q() >= c.q_req).all()), "deck must reach the delivery level"
    assert bool(scene.supported().all()), "platform must rest on the driven ram"
    s2 = print_score("jacked to delivery level")
    assert s2 >= s1, "score must not decrease"

    # ----- phase 3: VERIFY — hands off, success must hold and keep holding -----------------------
    step(60)
    report("verify")
    ok = scene.success()
    assert bool(ok.all()), "success() must hold before the persistence window"
    s3 = print_score("success reached")
    assert s3 >= 0.999, f"live success must score 1.0, got {s3}"

    step(400)  # >= 3.3 simulated seconds, hands off — self-lock must hold the lift
    report("persistence")
    ok = scene.success()
    s4 = print_score("after persistence")
    if bool(ok.all()) and s4 >= 0.999:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        os._exit(0)
    print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)
    os._exit(1)


try:
    main()
except Exception as e:  # noqa: BLE001
    print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
    os._exit(1)
