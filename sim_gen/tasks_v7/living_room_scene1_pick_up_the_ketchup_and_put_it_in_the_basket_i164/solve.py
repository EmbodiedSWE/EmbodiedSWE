"""Teleport solution for HookHangBasketScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i164`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. HANG THE BASKET (transport teleport + contact dynamics): one pose write STAGES the
   basket in free air with its handle bar aligned 25 mm ABOVE the HIGH peg (identified
   by readback — the side is randomized), zero velocity, nothing touching. The actual
   hanging — the free fall, the bar-on-peg CATCH, the support transfer and the pendulum
   swing-and-settle on the point contact — is pure contact physics. The basket is never
   written into a hung state: if the catch failed it would simply fall to the floor.
2. DEPOSIT THE KETCHUP (transport teleport + contact dynamics): one pose write stages
   the red bottle upright just above the HANGING basket's opening (over the interior,
   on the board side of the handle bar, computed in the basket's CURRENT readback
   frame), zero velocity. The drop through the opening, the impact, the induced swing
   of the loaded pendulum and the joint ring-down to a settled suspended state are
   contact physics — exactly the compliance that makes this task not the seed.

From the second release to the verdict nothing touches any body. success() demands the
live physical state: bar on peg, basket dangling clear of the floor, bottle inside,
everything still — a state that only the peg's support can sustain.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hook_hang_basket")().build(num_envs=args.num_envs,
                                                      device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def vels(tag: str) -> None:
        bv = float(scene.basket.data.root_lin_vel_w[0].norm())
        bw = float(scene.basket.data.root_ang_vel_w[0].norm())
        kv = float(scene.ketchup.data.root_lin_vel_w[0].norm())
        kw = float(scene.ketchup.data.root_ang_vel_w[0].norm())
        print(f"[solve] {tag:12s} | basket v={bv:.4f} w={bw:.4f} "
              f"ketchup v={kv:.4f} w={kw:.4f}", flush=True)

    def report(tag: str) -> None:
        bz = float((scene.basket.data.root_pos_w - scene.env_origins)[0, 2])
        vels(tag)
        print(f"[solve] {tag:12s} | basket_z={bz:+.3f} "
              f"eng_a={bool(scene._engaged_on(scene.peg_a)[0])} "
              f"eng_b={bool(scene._engaged_on(scene.peg_b)[0])} "
              f"hung_ever={bool(scene._hung_ever[0])} "
              f"k_in={bool(scene.contained(scene.ketchup)[0])} "
              f"bbq_in={bool(scene.contained(scene.bbq)[0])} "
              f"in_ever={bool(scene._in_ever[0])} "
              f"loaded_ever={bool(scene._loaded_ever[0])} "
              f"still={bool(scene._still()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    za = float((scene.peg_a.data.root_pos_w - scene.env_origins)[0, 2])
    zb = float((scene.peg_b.data.root_pos_w - scene.env_origins)[0, 2])
    high = scene.peg_a if za > zb else scene.peg_b
    sq = scene.stand.data.root_quat_w[0]
    s_yaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    b0 = (scene.basket.data.root_pos_w - scene.env_origins)[0]
    k0 = (scene.ketchup.data.root_pos_w - scene.env_origins)[0]
    q0 = (scene.bbq.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): stand_yaw={s_yaw:+.1f}deg "
          f"peg_a_z={za:.3f} peg_b_z={zb:.3f} high={'a' if za > zb else 'b'} "
          f"basket=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"ketchup=({float(k0[0]):+.3f},{float(k0[1]):+.3f}) "
          f"bbq=({float(q0[0]):+.3f},{float(q0[1]):+.3f})", flush=True)
    report("reset")
    assert abs(za - zb) > 0.15, "peg heights did not differentiate"
    assert not bool(scene.engaged_hung()[0]), "basket spawned engaged?!"
    assert not bool(scene.contained(scene.ketchup)[0]), "ketchup spawned contained?!"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: HANG THE BASKET (stage above the peg; catch is physics) ------
    # One pose write: bar 25 mm above the high peg, bar axis across the peg, basket
    # dangling in free air (nothing touching). Then hands off: gravity drops it, the
    # bar CATCHES on the peg and the pendulum settles — all contact dynamics.
    p_pos = high.data.root_pos_w
    p_quat = high.data.root_quat_w
    bar_loc = torch.tensor([0.105, 0.0, c.peg_r + c.bar_t / 2 + 0.025],
                           device=device).expand(n, 3)
    bar_w = p_pos + quat_apply(p_quat, bar_loc)
    off = torch.tensor([0.0, 0.0, c.bar_z], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = bar_w - quat_apply(p_quat, off)  # basket aligned with the peg frame
    st[:, 3:7] = p_quat
    scene.basket.write_root_state_to_sim(st, all_ids)

    caught = False
    quiet = 0
    for i in range(720):
        env.step(no_action)
        eng = bool(scene.engaged_hung()[0])
        still = float(scene.basket.data.root_lin_vel_w[0].norm()) < c.settle_speed
        quiet = quiet + 1 if (eng and still) else 0
        if quiet >= 30:
            caught = True
            break
    report("hung")
    assert caught, "bar did not catch on the high peg / basket did not settle hung"
    assert bool(scene._hung_ever[0]), "hung latch did not set"
    s1 = print_score("P1 basket hung on the high peg (drop-catch, settled)")
    assert s1 >= s0 - 1e-6, "score decreased across the hang"
    assert s1 >= 0.24, f"hang credit missing, score {s1}"

    # ---------------- phase 2: DEPOSIT THE KETCHUP (stage above the opening; drop) ----------
    # One pose write in the HANGING basket's CURRENT frame: bottle upright over the
    # interior, on the board side of the handle bar, bottom ~8 mm above the rim.
    b_pos = scene.basket.data.root_pos_w
    b_quat = scene.basket.data.root_quat_w
    rim = c.floor_t + c.wall_h
    drop_loc = torch.tensor([-0.040, 0.0, rim], device=device).expand(n, 3)
    p_w = b_pos + quat_apply(b_quat, drop_loc)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = p_w[:, 0:2]
    st[:, 2] = p_w[:, 2] + 0.008 + c.body_h / 2  # bottom 8 mm above the rim, upright
    st[:, 3] = 1.0
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    assert not bool(scene.contained(scene.ketchup)[0]), \
        "staging pose is already contained (teleport must stay above the opening)"

    quiet = 0
    done = False
    for i in range(1200):
        env.step(no_action)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            done = True
            break
        if (i + 1) % 240 == 0:
            vels(f"P2 t+{(i + 1) / 120.0:.0f}s")
    report("loaded")
    assert bool(scene.contained(scene.ketchup)[0]), "ketchup did not land inside"
    assert bool(scene.engaged_hung()[0]), "the deposit knocked the basket off the peg"
    assert done, "loaded basket did not settle to success"
    s2 = print_score("P2 ketchup dropped into the hanging basket, settled")
    assert s2 >= s1 - 1e-6, "score decreased across the deposit"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after deposit)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) ------
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
    except Exception as e:  # noqa: BLE001 — fail fast, don't idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
