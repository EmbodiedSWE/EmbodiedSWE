"""Teleport solution for ButterSwitchyardScene (sim_gen task `libero_pick_butter_i172`)
— the task's legitimacy certificate.

This solution uses NO teleports at all: both blocks start captive in the trunk
channel and every metre of their journey is contact dynamics. All four phases are
velocity-regulated horizontal external forces at the CoM (the scene-level stand-in
for a fingertip push into the open channel top — the intended Franka strategy in
TASK.md), with the channel walls doing the guiding and gravity doing the delivery:

1. BRICK ALONG THE TRUNK (contact): push the grey brick forward (+x, deck frame)
   from its slot to the T-junction; the servo centres it in the junction box.
2. BRICK DISPOSAL (contact): push the brick down the crossbar AWAY from the basket
   and over that drop mouth; the force is cut the moment its CoM crosses the deck
   edge and gravity grounds it on the bare ground (the `cleared` latch).
3. BUTTER ALONG THE TRUNK (contact): only now is the trunk open; push the yellow
   butter forward through the junction box (the `junction` latch — the route
   credential success later requires).
4. BUTTER DELIVERY (contact): push the butter down the crossbar TOWARD the basket
   and over its mouth; force cut at the edge, free fall into the basket interior
   (`delivered` fires while descending), settle to success.

The push servo aims each stroke at a TARGET POINT in the deck's body frame, so wall
deflections self-correct, and the force-frame encoding is PROBED at runtime (some
pods rotate an `is_global=True` wrench by the body's rotation since reset — see the
`encode_force` note): if measured progress goes the wrong way or wedges at max
force, the encoding mode is flipped.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_pick_butter_i172.solve --headless [--seed N]
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
# DAEMON so a crashed main thread doesn't idle the process until the timer fires.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.butter_switchyard")().build(num_envs=args.num_envs, device=device)
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

    def rel_yaw(obj) -> float:
        """Object yaw relative to the deck (deg) — wall-rub spin telemetry."""
        q = quat_mul(quat_conjugate(scene.deck.data.root_quat_w),
                     obj.data.root_quat_w)[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def deck_to_world(tx: float, ty: float) -> torch.Tensor:
        """(n, 3) world point of a deck-frame target (z at channel-floor height)."""
        loc = torch.tensor([tx, ty, c.deck_top + c.block_h / 2],
                           device=device).expand(n, 3)
        return scene.deck.data.root_pos_w + quat_apply(scene.deck.data.root_quat_w, loc)

    def report(tag: str) -> None:
        bl = scene._deck_local(scene.butter)[0]
        kl = scene._deck_local(scene.brick)[0]
        bb = scene._basket_local(scene.butter)[0]
        print(f"[solve] {tag:12s} | butter_d=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) yaw={rel_yaw(scene.butter):+.0f} "
              f"brick_d=({float(kl[0]):+.3f},{float(kl[1]):+.3f},"
              f"{float(kl[2]):.3f}) yaw={rel_yaw(scene.brick):+.0f} "
              f"butter_b=({float(bb[0]):+.3f},{float(bb[1]):+.3f},"
              f"{float(bb[2]):.3f}) prog={float(scene._prog_max[0]):.3f} "
              f"cleared={bool(scene._cleared[0])} junction={bool(scene._junction[0])} "
              f"delivered={bool(scene._delivered[0])} dirty={bool(scene._dirty[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force(obj) -> None:
        obj.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # Force-frame encoding: some pods apply an `is_global=True` wrench rotated by the
    # body's rotation since reset (applied = R_now * R_ref^T * arg). Mode 0 passes the
    # world force through; mode 1 pre-encodes with R_ref * R_now^T. The right mode is
    # PROBED from measured progress, never assumed.
    q_ref: dict[str, torch.Tensor] = {}
    mode = {"brick": 0, "butter": 0}

    def encode_force(key: str, obj, f_world: torch.Tensor) -> torch.Tensor:
        if mode[key] == 0:
            return f_world
        return quat_apply(quat_mul(q_ref[key], quat_conjugate(obj.data.root_quat_w)),
                          f_world)

    def push_to(obj, key: str, tag: str, target_fn, done_fn, *, v_max: float = 0.10,
                floor0: float = 0.70, f_max: float = 2.0, max_steps: int = 3000) -> bool:
        """Velocity-regulated horizontal CoM push toward `target_fn()` (world (n,3));
        force cut the instant `done_fn()` holds. K*dt/m = 0.22 (stable under the
        one-substep wrench delay); a stiction floor breaks static friction; the cap
        stays under the 2.65 N CoM-height tipping bound. The progress probe flips the
        force-frame mode if the block moves AWAY, escalates the floor on a stall, and
        flips the mode if it stays wedged at max force. Returns done_fn() at exit."""
        K = 4.0
        floor_f = floor0
        stuck = 0
        win_i, win_dist = 0, None
        for i in range(max_steps):
            if done_fn():
                break
            d_vec = (target_fn() - obj.data.root_pos_w)[0, :2]
            dist = float(d_vec.norm())
            u = d_vec / max(dist, 1e-6)
            v_des = v_max if dist > 0.06 else 0.05
            v = obj.data.root_lin_vel_w[0, :2]
            f_xy = K * (u * v_des - v)
            f_along = float((f_xy * u).sum())
            if float(v.norm()) < 0.02 and f_along < floor_f:
                f_xy = f_xy + u * (floor_f - f_along)
            fn_ = float(f_xy.norm())
            if fn_ > f_max:
                f_xy = f_xy * (f_max / fn_)
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, :2] = f_xy
            obj.set_external_force_and_torque(
                encode_force(key, obj, f_world).view(n, 1, 3), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
            if win_dist is None:
                win_dist = dist
            if i - win_i >= 60:
                if dist > win_dist + 0.008:
                    mode[key] = 1 - mode[key]
                    stuck = 0
                    print(f"[solve] {tag}: moving away (dist {win_dist:.3f} -> "
                          f"{dist:.3f}); force-frame mode -> {mode[key]}", flush=True)
                elif dist > win_dist - 0.004:
                    stuck += 1
                    if floor_f < f_max:
                        floor_f = min(floor_f + 0.35, f_max)
                        print(f"[solve] {tag}: stalled at dist {dist:.3f}; "
                              f"stiction floor -> {floor_f:.2f} N", flush=True)
                    elif stuck >= 3:
                        mode[key] = 1 - mode[key]
                        floor_f = floor0
                        stuck = 0
                        print(f"[solve] {tag}: wedged at max force; force-frame "
                              f"mode -> {mode[key]}", flush=True)
                else:
                    stuck = 0
                win_i, win_dist = i, dist
        clear_force(obj)
        if not bool(done_fn()):
            print(f"[solve] {tag}: leg timed out", flush=True)
        return bool(done_fn())

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    q_ref["brick"] = scene.brick.data.root_quat_w.clone()
    q_ref["butter"] = scene.butter.data.root_quat_w.clone()
    side = float(scene._side[0])
    dl_bu = scene._deck_local(scene.butter)[0]
    dl_br = scene._deck_local(scene.brick)[0]
    dl_ba = scene._deck_local(scene.basket)[0]

    def deck_yaw_deg() -> float:
        q = scene.deck.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    print(f"[solve] layout readback (seed {args.seed}): deck_yaw={deck_yaw_deg():+.1f}deg "
          f"side={side:+.0f} butter_d=({float(dl_bu[0]):+.3f},{float(dl_bu[1]):+.3f}) "
          f"brick_d=({float(dl_br[0]):+.3f},{float(dl_br[1]):+.3f}) "
          f"basket_d=({float(dl_ba[0]):+.3f},{float(dl_ba[1]):+.3f}) "
          f"brick_x0={float(scene._brick_x0[0]):+.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.02, f"baseline score should be ~0, got {s0}"

    def brick_d() -> torch.Tensor:
        return scene._deck_local(scene.brick)[0]

    def butter_d() -> torch.Tensor:
        return scene._deck_local(scene.butter)[0]

    # ---------------- phase 1: brick along the trunk to the junction (contact) -------------
    ok = push_to(scene.brick, "brick", "brick-trunk", lambda: deck_to_world(0.0, 0.0),
                 lambda: float(brick_d()[0].abs()) < 0.020 and float(brick_d()[1].abs()) < 0.025)
    assert ok, "brick never centred in the junction"
    step(30)
    report("brick@T")
    s1 = print_score("P1 brick pushed along the trunk to the junction")
    assert s1 >= s0 - 1e-6, "score decreased across P1"

    # ---------------- phase 2: brick disposal out the AWAY mouth (contact) -----------------
    away = -side
    ok = push_to(scene.brick, "brick", "brick-cross",
                 lambda: deck_to_world(0.0, away * (c.cross_half + 0.12)),
                 lambda: float(brick_d()[1]) * away > c.cross_half, v_max=0.12)
    assert ok, "brick never reached the away mouth"
    print("[solve] brick disposal: force cut at the deck edge, free fall + settle", flush=True)
    step(240)
    report("brick-down")
    assert bool(scene._cleared[0]), "brick cleared latch did not set"
    assert not bool(scene._dirty[0]), "brick contaminated the basket"
    s2 = print_score("P2 brick dropped onto the bare ground (cleared)")
    assert s2 >= s1 - 1e-6, "score decreased across P2"

    # ---------------- phase 3: butter along the trunk, through the junction ----------------
    ok = push_to(scene.butter, "butter", "butter-trunk", lambda: deck_to_world(0.0, 0.0),
                 lambda: (float(butter_d()[0].abs()) < 0.020
                          and float(butter_d()[1].abs()) < 0.025
                          and bool(scene._junction[0])))
    assert ok, "butter never reached the junction"
    step(30)
    report("butter@T")
    assert bool(scene._junction[0]), "junction latch did not set"
    s3 = print_score("P3 butter pushed through the junction (route credential)")
    assert s3 >= s2 - 1e-6, "score decreased across P3"

    # ---------------- phase 4: butter delivery out the BASKET mouth (contact) --------------
    ok = push_to(scene.butter, "butter", "butter-cross",
                 lambda: deck_to_world(0.0, side * (c.cross_half + 0.12)),
                 lambda: float(butter_d()[1]) * side > c.cross_half, v_max=0.12)
    assert ok, "butter never reached the basket mouth"
    print("[solve] delivery: force cut at the deck edge, free fall into the basket", flush=True)
    step(240)
    report("delivered")
    bb = scene._basket_local(scene.butter)[0]
    print(f"[solve] landing: butter in basket frame = ({float(bb[0]):+.3f},"
          f"{float(bb[1]):+.3f},{float(bb[2]):.3f})", flush=True)
    assert bool(scene._delivered[0]), "delivered latch did not set"
    assert bool(scene._in_basket(scene.butter)[0]), "butter did not land inside the basket"
    s4 = print_score("P4 butter delivered into the basket")
    assert s4 >= s3 - 1e-6, "score decreased across P4"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after delivery)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) -------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001 — any crash must exit NOW, not idle to the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
