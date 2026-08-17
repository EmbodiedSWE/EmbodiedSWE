"""Teleport solution for RollAwayVaultScene (sim_gen task `open_box_i184`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. OPENING (applied force + gravity + contact): the crimson roller seals the slot,
   seated on the bridge bars behind the 4 mm detent lip. A horizontal force at the
   roller's CoM (vault-local +x, along the fences; ramping 3 -> 9 N against the
   ~5.9 N quasi-static lip threshold) pushes it over the lip — the exact fingertip
   push a gripper would perform. The instant the roller tips onto the ramp the force
   is CUT: the descent into the walled dock is pure gravity + rolling contact,
   hands-off. The roller is NEVER teleported. Every `cleared` fact the rubric checks
   (roller at rest, slot uncovered) is produced by physics.
2. TRANSPORT (teleport): one root-state write carries the GOLD prize cube from the
   cavity floor to the free-space point 30 mm above the blue pad, upright, zero
   velocity. The path is free space: the slot is UNCOVERED at this point (phase 1
   finished), so the transport bypasses no contact interaction — it stands in for
   the straight-up pinch-lift through the 80 x 144 mm centre opening that the
   embodiment argument describes. The write satisfies no placement clause by
   itself: the cube is airborne over the pad.
3. SEATING (gravity + contact): the cube FALLS the last 30 mm onto the pad and
   settles. `prize_on_pad` (xy on the pad axis, correct rest height, settled) is
   produced by contact, never written.
4. ORDER: the opening comes FIRST — the task's physically forced order (a seated
   roller seals the mouth by geometry: max crescent clearance 20 mm < the 40 mm
   cube; smoke check proves a hard shove cannot get the prize past it). The GREY
   distractor cube is never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.open_box_i184.solve --headless [--seed N]
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.roll_away_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

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

    def roller_loc() -> torch.Tensor:
        return scene._vault_local(scene.roller.data.root_pos_w)[0]

    def report(tag: str) -> None:
        rl = roller_loc()
        pl = scene._vault_local(scene.prize.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | roller_loc=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):+.3f}) prize_loc=({float(pl[0]):+.3f},{float(pl[1]):+.3f},"
              f"{float(pl[2]):+.3f}) covered={bool(scene.mouth_covered()[0])} "
              f"seated={bool(scene.roller_seated()[0])} out={bool(scene.prize_out()[0])} "
              f"on_pad={bool(scene.prize_on_pad()[0])} "
              f"distr_in={bool(scene.distractor_in()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # roller settles onto the bridge bars, cubes onto the cavity floor
    vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
    vq = scene.vault.data.root_quat_w[0]
    vyaw = math.degrees(2.0 * math.atan2(float(vq[3]), float(vq[0])))
    pd = (scene.pad.data.root_pos_w - scene.env_origins)[0]
    prize_side = "+y" if float(scene.prize_slot[0]) > 0 else "-y"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"vault=({float(vp[0]):+.3f},{float(vp[1]):+.3f}) yaw={vyaw:+.1f}deg "
          f"pad=({float(pd[0]):+.3f},{float(pd[1]):+.3f}) "
          f"roller_x={float(roller_loc()[0]):+.3f} prize_slot={prize_side}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.roller_seated()[0]), "roller must start seated over the slot"
    assert bool(scene.mouth_covered()[0]), "mouth must start covered"
    assert not bool(scene.prize_out()[0]), "prize must start inside the cavity"
    assert bool(scene.distractor_in()[0]), "distractor must start inside the cavity"
    s0 = print_score("P0 reset+settle (roller seated, cubes inside)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: push the roller over the lip; gravity rolls it home ---------
    fdir = quat_apply(scene.vault.data.root_quat_w,
                      torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    fmag = 3.0
    cut_at = None
    cleared = False
    for i in range(1500):
        rl = roller_loc()
        on_ramp = float(rl[0]) > c.box_hx + 0.005  # past the plate edge: gravity's job now
        if not on_ramp and cut_at is None:
            f = (fmag * fdir).view(n, 1, 3)
            scene.roller.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                       is_global=True)
            if i % 60 == 59:
                fmag = min(fmag + 1.0, 9.0)
                print(f"[solve] roller behind the lip (x_loc={float(rl[0]):+.3f}); "
                      f"push force -> {fmag:.0f} N", flush=True)
        elif cut_at is None:
            cut_at = i
            scene.roller.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                       env_ids=all_ids)
            print(f"[solve] roller over the lip and onto the ramp at step {i} "
                  f"(x_loc={float(rl[0]):+.3f}, z_loc={float(rl[2]):+.3f}); "
                  f"force cut, gravity takes it", flush=True)
        env.step(no_action)
        if cut_at is not None and bool(scene._cleared[0]) \
                and bool(scene.mouth_clear()[0]) \
                and float(scene.roller.data.root_lin_vel_w[0].norm()) < 0.03:
            cleared = True
            break
    scene.roller.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(120)  # full settle, hands-off
    report("roll-away")
    if not cleared or not bool(scene.mouth_clear()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (roller never cleared the slot)", flush=True)
        os._exit(1)
    rl = roller_loc()
    print(f"[solve] roller at rest in the dock: loc=({float(rl[0]):+.3f},"
          f"{float(rl[1]):+.3f},{float(rl[2]):+.3f})", flush=True)
    assert float(rl[2]) < c.roller_r + 0.020, "roller should rest at ground level"
    assert not bool(scene.prize_out()[0]), "prize untouched: still inside"
    s1 = print_score("P1 roller pushed over the lip, rolled to rest in the dock")
    assert s1 >= s0 - 1e-6 and s1 >= 0.34, f"P1 score {s1} (expect unseat+clear=0.35)"

    # ---------------- phase 2: prize -> pad (transport, then gravity seats it) -------------
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.pad.data.root_pos_w
    st[:, 2] += c.pad_t / 2 + c.cube_s / 2 + 0.030  # 30 mm free fall onto the pad
    st[:, 3] = 1.0
    scene.prize.write_root_state_to_sim(st, all_ids)
    step(150)  # fall + settle, hands-off
    report("prize->pad")
    assert bool(scene.prize_on_pad()[0]), "prize must be seated on the pad"
    assert bool(scene.distractor_in()[0]), "distractor must remain inside"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the prize seated)", flush=True)
        os._exit(1)
    s2 = print_score("P2 prize lifted out through the open slot and seated on the pad")
    assert s2 >= s1 - 1e-6 and s2 >= 0.99, f"P2 score {s2} (expect success=1.0)"

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
    main()
