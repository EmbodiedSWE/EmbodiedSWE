"""Teleport solution for HookedBasketScene (sim_gen task
`living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i149`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. INSERT (teleport + gravity/contact): one pose write carries the RED can from its
   ground band to a release pose ABOVE the basket rim, centred over one of the two
   mouth openings beside the blue handle bar. The release pose is fully OUTSIDE the
   containment volume (basket-local z above the rim, above `inside_z_max`), so the
   freshly-teleported state earns no containment credit. Gravity then drops the can
   through the real 76 mm opening past the bar; it impacts the basket floor and
   settles under contact dynamics. Only the SETTLED state (slow, inside, below the
   rim) latches `_in`.
2. TRANSPORT (teleport, composite): one root-state write per body carries the LOADED
   basket — basket and can written together with the can's basket-relative pose
   preserved bitwise, exactly what a careful carry does — to a staging pose over the
   stand's orange hook arm: bar centred over the seat window (stand-local x =
   `seat_x_mid`, y = 0), bar yawed perpendicular to the arm, bar underside ~22 mm
   ABOVE the arm's top face. That pose is outside the seat z-window (`seat_z_hi`),
   so hang geometry is false at the moment of teleport and success() is unreachable
   there. Both endpoints are in free space.
3. SEAT (contact dynamics): gravity lowers the loaded basket; the hook arm passes
   through the arch under the bar and the bar lands ON the arm's 24 x 18 mm top
   patch. The loaded basket becomes a pendulum hanging from the seated bar (CoM
   0.22 m below the seat — passively stable) and rings down under angular damping
   and contact friction. success() first turns True only on this settled physical
   state: can inside, bar in the seat window, upright, elevated, beige can out,
   everything at rest.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i149.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hooked_basket")().build(num_envs=args.num_envs, device=device)
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

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        bl = scene._bar_stand_local()[0]
        tl = scene._basket_local(scene.tomato)[0]
        bz = float((scene.basket.data.root_pos_w - scene.env_origins)[0, 2])
        print(f"[solve] {tag:12s} | bar_stand=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) tomato_local=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):.3f}) basket_z={bz:.3f} up={float(scene._basket_up()[0]):+.3f} "
              f"in={bool(scene._in[0])} lift={bool(scene._lift[0])} hang={bool(scene._hang[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    sq = scene.stand.data.root_quat_w[0]
    syaw = 2.0 * math.atan2(float(sq[3]), float(sq[0]))
    bp0 = (scene.basket.data.root_pos_w - scene.env_origins)[0]
    tom0 = (scene.tomato.data.root_pos_w - scene.env_origins)[0]
    dis0 = (scene.distractor.data.root_pos_w - scene.env_origins)[0]
    masses = scene.basket.root_physx_view.get_masses()
    print(f"[solve] layout readback (seed {args.seed}): stand=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(syaw):+.1f}deg "
          f"basket=({float(bp0[0]):+.3f},{float(bp0[1]):+.3f}) "
          f"tomato=({float(tom0[0]):+.3f},{float(tom0[1]):+.3f}) "
          f"distractor=({float(dis0[0]):+.3f},{float(dis0[1]):+.3f}) "
          f"basket_mass={float(masses.reshape(-1)[0]):.3f}", flush=True)
    report("reset")
    assert float(scene.score()[0]) < 0.02, "score must start ~0"
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: INSERT the red can (teleport to free air above the mouth) ---
    # One pose write puts the can 6 cm above the rim, centred over the mouth opening
    # beside the handle bar (basket-local (0, +0.047)). Basket-local z = 0.180 is
    # above `inside_z_max` = 0.115: NO containment credit at the teleport instant.
    bq = scene.basket.data.root_quat_w
    drop_local = torch.zeros(n, 3, device=device)
    drop_local[:, 1] = (c.in_x / 2 + c.bar_w / 2) / 2  # centre of the mouth opening
    drop_local[:, 2] = 0.180
    drop_w = scene.basket.data.root_pos_w + quat_apply(bq, drop_local)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = drop_w
    st[:, 3] = 1.0  # upright in world
    scene.tomato.write_root_state_to_sim(st, all_ids)
    assert not bool(scene._inside_now(scene.tomato)[0]), \
        "teleport must not place the can inside the containment volume"
    report("released")
    step(180)  # 1.5 s: fall through the opening, impact the floor, settle
    report("inserted")
    assert bool(scene._inside_now(scene.tomato)[0]), "can did not come to rest inside"
    assert bool(scene._in[0]), "containment latch did not fire on the settled state"
    s1 = print_score("P1 gravity insertion through the mouth opening")
    assert s1 >= s0 - 1e-6, "score decreased across insertion"

    # ---------------- phase 2: TRANSPORT the loaded basket (composite teleport) ------------
    # Both bodies are written together with the can's basket-relative pose preserved —
    # a carry, not a shortcut. Target: bar centred over the seat window, perpendicular
    # to the hook arm, bar underside ~22 mm ABOVE the arm top face — OUTSIDE the seat
    # z-window, so hang geometry is false here and success() is unreachable.
    bq_old = scene.basket.data.root_quat_w.clone()
    bp_old = scene.basket.data.root_pos_w.clone()
    can_rel = quat_apply_inverse(bq_old, scene.tomato.data.root_pos_w - bp_old)
    can_q_rel = quat_mul(quat_inv(bq_old), scene.tomato.data.root_quat_w)

    stage_gap = 0.022  # bar underside this far above the peg top at staging
    byaw = syaw + math.pi / 2  # bar (basket local x) perpendicular to the arm (stand local x)
    bq_new = torch.zeros(n, 4, device=device)
    bq_new[:, 0], bq_new[:, 3] = math.cos(byaw / 2), math.sin(byaw / 2)
    # bar target in stand frame -> world
    bar_stand = torch.tensor([c.seat_x_mid, 0.0, c.peg_top_z + stage_gap + c.bar_t / 2],
                             device=device).expand(n, 3)
    bar_w = scene.stand.data.root_pos_w + quat_apply(scene.stand.data.root_quat_w, bar_stand)
    ez = torch.zeros(n, 3, device=device)
    ez[:, 2] = c.bar_z
    bp_new = bar_w - quat_apply(bq_new, ez)

    st = torch.zeros(n, 13, device=device)
    st[:, 0:3], st[:, 3:7] = bp_new, bq_new
    scene.basket.write_root_state_to_sim(st, all_ids)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = bp_new + quat_apply(bq_new, can_rel)
    st[:, 3:7] = quat_mul(bq_new, can_q_rel)
    scene.tomato.write_root_state_to_sim(st, all_ids)

    report("staged")
    assert bool(scene._inside_now(scene.tomato)[0]), "carry must preserve containment"
    assert not bool(scene._hang_geom()[0]), "staging pose must be outside the seat window"
    assert not bool(scene.success()[0]), "success unreachable at the staged pose"
    s2 = print_score("P2 composite transport to the staging pose above the arm")
    assert s2 >= s1 - 1e-6, "score decreased across transport"

    # ---------------- phase 3: SEAT through gravity + contact -------------------------------
    # Gravity drops the loaded basket; the arm threads the arch, the bar lands on the
    # arm's top patch, and the pendulum rings down. A single success() sample can land
    # on a swing turning point (everything momentarily at rest), so require success at
    # SIX consecutive samples (2 s) before declaring the hang settled.
    seated = False
    streak = 0
    for _ in range(60):  # up to 60 x 40 = 2400 steps = 20 s of ring-down
        step(40)
        if bool(scene.success()[0]):
            streak += 1
            if streak >= 6:
                seated = True
                break
        else:
            streak = 0
    report("seated")
    if not seated:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seating)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)
    assert bool(scene._hang[0]), "hang latch did not fire"
    s3 = print_score("P3 gravity seat onto the hook arm -> success")
    assert s3 >= s2 - 1e-6, "score decreased across seating"
    assert s3 >= 0.999, "success must score 1.0"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
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
