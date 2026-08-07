"""Teleport solution for CheckerSiloScene (sim_gen task `setup_checkers_i2`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, one disc at a time): a single root-state write carries a
   checker from the supply rack to the free-space RELEASE POINT above the funnel
   mouth (silo local (0, 0, z_mouth)), held on edge with its axis along the silo
   slit axis — exactly the pose a gripper would carry it in. The write puts the
   disc in open air with zero velocity; it satisfies no rubric clause by itself.
2. INSERTION (gravity + contact, hands-off): from the release point the disc FALLS —
   the funnel plates guide it through the throat, the one-disc-wide channel keeps it
   on edge, and it lands on the channel floor or on the previous disc. Every stack
   fact the rubric checks (seated, contiguous, color order, alignment) is produced
   by ballistics and contact, never written. If a disc wedges in the funnel, a small
   escalating downward force at its CoM (starting at ~1x its own weight) stands in
   for the fingertip poke a robot would use — contact-consistent, cleared at once.
3. ORDER: red_0, then white_0, then red_1 — the bottom-up pattern IS the drop
   order, which is the whole point of the task. white_1 is never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.setup_checkers_i2.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qmul, _qy = scene_mod._qmul, scene_mod._qy

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.checker_silo")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def stack_str() -> str:
        k, _zs, red = scene.stack_readout()
        return "".join("R" if bool(red[0, j]) else "W" for j in range(int(k[0])))

    def report(tag: str) -> None:
        pos, _q, vel = scene._disc_tensors()
        loc = scene._silo_local(pos)[0]
        zs = " ".join(f"{nm}=({float(loc[i, 0]):+.3f},{float(loc[i, 1]):+.3f},"
                      f"{float(loc[i, 2]):+.3f})"
                      for i, nm in enumerate(scene.DISC_NAMES))
        print(f"[solve] {tag:12s} | stack=[{stack_str()}] {zs} "
              f"vmax={float(vel[0].max()):.3f} "
              f"appr={bool(scene._appr[0])} p1={bool(scene._p1[0])} "
              f"p2={bool(scene._p2[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force(disc) -> None:
        disc.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def drop(name: str) -> None:
        """TRANSPORT the disc to the release point above the funnel mouth (on edge,
        axis along the silo slit axis, zero velocity), then let GRAVITY insert it.
        Escalating downward CoM nudge only if it wedges in the funnel."""
        disc = scene.discs[name]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.mouth_point_w()
        st[:, 3:7] = _qmul(scene.silo.data.root_quat_w,
                           _qy(torch.full((n,), math.pi / 2, device=device)))
        disc.write_root_state_to_sim(st, all_ids)
        # hands-off fall: funnel -> throat -> channel -> stack
        seated = False
        nudge = 0.0
        for i in range(720):
            if nudge > 0.0:
                f = torch.zeros(n, 1, 3, device=device)
                f[:, 0, 2] = -nudge
                disc.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                   is_global=True)
            env.step(no_action)
            zloc = float(scene._silo_local(disc.data.root_pos_w)[0, 2])
            v = float(disc.data.root_lin_vel_w.norm(dim=-1)[0])
            if i > 30 and zloc < 0.16 and v < c.settle_speed:
                seated = True
                break
            if i >= 150 and i % 60 == 0 and v < 0.03 and zloc > 0.16:
                nudge = min(nudge + 0.3, 1.5)  # ~1x disc weight per escalation
                print(f"[solve] {name} wedged in the funnel at z_loc={zloc:.3f}; "
                      f"downward nudge {nudge:.1f} N", flush=True)
        clear_force(disc)
        step(120)  # full settle, hands-off
        if not seated:
            report(f"{name}-STUCK")
            print(f"SIM_GEN_SOLVE: FAIL ({name} never seated in the channel)",
                  flush=True)
            os._exit(1)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # discs seat into the rack groove
    sp = (scene.silo.data.root_pos_w - scene.env_origins)[0]
    sq = scene.silo.data.root_quat_w[0]
    syaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    rp = (scene.rack.data.root_pos_w - scene.env_origins)[0]
    rq = scene.rack.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    slot_of = scene.slot_of[0].tolist()  # slot index of disc i (name order)
    slot_color = {slot_of[i]: ("R" if scene.IS_RED[i] else "W") for i in range(4)}
    rack_order = "".join(slot_color[s] for s in range(4))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"silo=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg "
          f"rack=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={ryaw:+.1f}deg "
          f"rack_slots(-x..+x)=[{rack_order}] slot_of={slot_of}", flush=True)
    report("reset")
    pos, _q, _v = scene._disc_tensors()
    assert torch.isfinite(pos).all(), "NaN/inf in disc states after settle"
    hz = (pos[0, :, 2] - scene.env_origins[0, 2])
    assert bool((hz > 0.015).all() and (hz < 0.06).all()), \
        f"discs must rest on edge in the rack, z={hz.tolist()}"
    assert int(scene.in_silo()[0].sum()) == 0, "no disc may start inside the silo"
    s0 = print_score("P0 reset+settle (all four checkers on the rack)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1-3: drop red, white, red -------------------------------------
    drop("red_0")
    report("drop-red_0")
    assert bool(scene.in_silo()[0, 0]), "red_0 must be inside the silo"
    assert stack_str() == "R", f"stack should read [R], got [{stack_str()}]"
    assert bool(scene._p1[0]), "prefix1 latch did not set"
    assert not bool(scene.success()[0]), "cannot be success after one disc"
    s1 = print_score("P1 red checker seated on the channel floor")
    assert s1 >= s0 - 1e-6 and s1 >= 0.44, f"P1 score {s1} (expect appr+p1=0.45)"

    drop("white_0")
    report("drop-white_0")
    assert bool(scene.in_silo()[0, 2]), "white_0 must be inside the silo"
    assert stack_str() == "RW", f"stack should read [RW], got [{stack_str()}]"
    assert bool(scene._p2[0]), "prefix2 latch did not set"
    assert not bool(scene.success()[0]), "cannot be success after two discs"
    s2 = print_score("P2 white checker stacked on the red")
    assert s2 >= s1 - 1e-6 and s2 >= 0.74, f"P2 score {s2} (expect 0.75)"

    drop("red_1")
    report("drop-red_1")
    assert bool(scene.in_silo()[0, 1]), "red_1 must be inside the silo"
    assert stack_str() == "RWR", f"stack should read [RWR], got [{stack_str()}]"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the third drop)", flush=True)
        os._exit(1)
    s3 = print_score("P3 second red tops the stack: red/white/red complete")
    assert s3 >= s2 - 1e-6, "score decreased across the third drop"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
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
