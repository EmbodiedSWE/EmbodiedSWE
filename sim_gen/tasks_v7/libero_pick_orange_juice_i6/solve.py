"""Teleport solution for LedgeCatchScene (sim_gen task `libero_pick_orange_juice_i6`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, basket): one root-state write carries the basket from its
   ground spawn to the catch pose — upright on the ground, centred under the slab
   edge nearest the ORANGE carton (~6 cm outboard of the edge), aligned to the shelf
   yaw. This is receptacle STAGING across free ground, exactly what a real arm does
   by grasping the basket rim and carrying it; the freshly-written state contains no
   carton and satisfies no containment clause (approach/staged credit moves, as any
   real carry would earn).
2. PUSH-OFF + BALLISTIC CATCH (contact dynamics): the ORANGE carton is driven toward
   the slab edge by a horizontal external force at its CoM (world frame, along the
   shelf's outboard axis for the target's side, velocity-regulated bang-bang kept
   BELOW the ~1.0 N tipping bound so the carton slides instead of toppling on the
   slab). The force is CUT the moment the carton's CoM crosses the slab edge; from
   there gravity alone tips it over the edge, it falls ~15 cm, and the basket walls
   and floor do the catching. No pose write touches the carton at any point: its
   entire trajectory from shelf slot to basket interior is contact dynamics and
   gravity. The catch only works because phase 1 staged the basket correctly — the
   aiming IS the task.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_pick_orange_juice_i6.solve --headless [--seed N]
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
    env = ENVS.get("simgen.ledge_catch")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def shelf_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.shelf.data.root_pos_w - scene.env_origins)[0]
        q = scene.shelf.data.root_quat_w[0]
        yaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        return sp, yaw

    def report(tag: str) -> None:
        tl = scene._shelf_local(scene.target)[0]
        bl = scene._shelf_local(scene.basket)[0]
        tb = scene._basket_local(scene.target)[0]
        print(f"[solve] {tag:12s} | target_s=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):.3f}) basket_s=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
              f"target_b=({float(tb[0]):+.3f},{float(tb[1]):+.3f},{float(tb[2]):.3f}) "
              f"on_shelf={bool(scene._on_shelf(scene.target)[0])} "
              f"inside={bool(scene._target_inside_now()[0])} "
              f"staged={bool(scene._staged[0])} caught={bool(scene._caught[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force(obj) -> None:
        obj.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def to_world(x_l: float, y_l: float) -> tuple[float, float, float]:
        sp, syaw = shelf_pose()
        wx = float(sp[0]) + math.cos(syaw) * x_l - math.sin(syaw) * y_l
        wy = float(sp[1]) + math.sin(syaw) * x_l + math.cos(syaw) * y_l
        return wx, wy, syaw

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    sp, syaw = shelf_pose()
    side = float(scene._side[0])
    t0 = (scene.target.data.root_pos_w - scene.env_origins)[0]
    d0 = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    b0 = (scene.basket.data.root_pos_w - scene.env_origins)[0]
    catch = scene._catch_local[0]
    print(f"[solve] layout readback (seed {args.seed}): shelf=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(syaw):+.1f}deg side={side:+.0f} "
          f"target=({float(t0[0]):+.3f},{float(t0[1]):+.3f},{float(t0[2]):.3f}) "
          f"decoy=({float(d0[0]):+.3f},{float(d0[1]):+.3f}) "
          f"basket=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"catch_local=({float(catch[0]):+.3f},{float(catch[1]):+.3f}) "
          f"d0_basket={float(scene._d0_basket[0]):.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.02, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT basket (teleport across free ground only) ---------
    # One pose write stages the basket at the catch pose: upright on the ground,
    # centred at the catch point (~6 cm outboard of the target's slab edge), aligned
    # to the shelf yaw. The write contains no carton — no containment clause is met.
    wx, wy, syaw = to_world(float(catch[0]), float(catch[1]))
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = wx, wy, 0.002
    st[:, 3], st[:, 6] = math.cos(syaw / 2), math.sin(syaw / 2)
    st[:, 0:3] += scene.env_origins
    scene.basket.write_root_state_to_sim(st, all_ids)
    step(60)
    assert not bool(scene._target_inside_now()[0]), \
        "transport must not place the carton inside the basket"
    assert bool(scene._staged[0]), "basket staging did not latch"
    report("staged")
    s1 = print_score("P1 transport basket to the catch pose")
    assert s1 >= s0 - 1e-6, "score decreased across basket transport"

    # ---------------- phase 2: PUSH-OFF + ballistic catch (contact dynamics) ---------------
    # Drive the carton outboard along the shelf's side axis with a horizontal force
    # at its CoM. Bang-bang below the ~1.0 N tip bound so it SLIDES on the slab; cut
    # the force the instant the CoM crosses the slab edge — gravity does the rest.
    _sp, syaw = shelf_pose()
    out_dir = torch.tensor([-math.sin(syaw) * side, math.cos(syaw) * side, 0.0],
                           device=device)
    v_des, f_push, f_max = 0.10, 0.55, 0.95
    launched = False
    for i in range(1800):
        loc = scene._shelf_local(scene.target)[0]
        if float(loc[1]) * side > c.hy:  # CoM past the slab edge: hands off
            launched = True
            break
        if float(loc[2]) < c.slab_top - 0.02:  # already falling (safety)
            launched = True
            break
        v_out = float((scene.target.data.root_lin_vel_w[0] * out_dir).sum())
        f = f_push if v_out < v_des else 0.0
        f_w = (out_dir * f).view(1, 1, 3).expand(n, 1, 3)
        scene.target.set_external_force_and_torque(f_w.contiguous(), zero_wrench,
                                                   env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i > 0 and i % 480 == 0 and float(loc[1]) * side < c.hy - c.edge_gap:
            f_push = min(f_push + 0.15, f_max)  # stalled: friction underestimated
            print(f"[solve] push-off: slow at y_s={float(loc[1]):+.3f}, raising force "
                  f"to {f_push:.2f} N", flush=True)
    clear_force(scene.target)
    print(f"[solve] push-off: force cut (launched={launched}), free fall + settle",
          flush=True)
    step(240)  # fall + land + damp out (2.0 s)
    report("landed")
    tb = scene._basket_local(scene.target)[0]
    print(f"[solve] landing: target in basket frame = ({float(tb[0]):+.3f},"
          f"{float(tb[1]):+.3f},{float(tb[2]):.3f})", flush=True)
    assert bool(scene._target_inside_now()[0]), "carton did not land inside the basket"
    assert bool(scene._caught[0]), "catch latch did not set"
    s2 = print_score("P2 contact-dynamics push-off + ballistic catch")
    assert s2 >= s1 - 1e-6, "score decreased across push-off"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after catch)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3 simulated seconds, no intervention) -------
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
