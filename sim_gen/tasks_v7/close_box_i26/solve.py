"""Teleport solution for TrapCrateScene (sim_gen task `close_box_i26`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the open-top CRATE from its
   spawn to a free-space hover 18 mm above the RED block, already inverted (opening
   down), zero velocity. The write satisfies no rubric clause by itself: the crate is
   airborne, `capped()` demands floor-height seating and `trapped()` demands the block
   inside the wall box of a SEATED crate.
2. CAPTURE (gravity + contact): the crate FALLS ~58 mm; its walls descend AROUND the
   block and the rim seats on the floor. Every `trapped()` fact (inverted attitude,
   floor-height rest, block inside the walls) is produced by ballistics and contact —
   never written. If a bounce spoils the capture, the crate is lifted (transport) and
   dropped again; the trap itself is always made by gravity.
3. HAUL (applied force + contact): the capped crate is SLID across the floor into the
   pen by a horizontal force at its CoM — the exact wrench of a fingertip push low on
   a wall — velocity-regulated (~0.12 m/s, cap 5.0 N, under the ~0.33 Nm tipping
   margin), with a tilt guard and a stiction floor. The trapped block is never touched:
   it is dragged along INSIDE by wall contact + friction, which is the mechanism the
   task is built on (lifting would free it). If the straight path grazes the BLUE
   block, the slide detours through a waypoint; the blue block is never touched either.
   Pod force-frame quirk: some pods rotate an applied wrench by the body's rotation
   since reset, so the desired world force is (de/en)coded per `encode_force` and the
   correct mode is PROBED from actual progress, not assumed.
4. ORDER: none is required by the task (cap-then-slide and move-block-first are both
   legal); this demonstration uses cap-then-slide.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_box_i26.solve --headless [--seed N]
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
    from .scene import _qx, encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qx, encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.trap_crate")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.crate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)

    def report(tag: str) -> None:
        cp = (scene.crate.data.root_pos_w - scene.env_origins)[0]
        up = float(scene.crate_up_z()[0])
        print(f"[solve] {tag:12s} | crate=({float(cp[0]):+.3f},{float(cp[1]):+.3f},"
              f"{float(cp[2]):+.3f}) up_z={up:+.3f} "
              f"capped={bool(scene.capped()[0])} trapped={bool(scene.trapped()[0])} "
              f"in_pen={bool(scene.in_pen()[0])} "
              f"blue_clear={bool(scene.distractor_clear()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)
    pen_xy = (scene.pen.data.root_pos_w - scene.env_origins)[0, :2]
    red_xy = (scene.red.data.root_pos_w - scene.env_origins)[0, :2]
    blue_xy = (scene.blue.data.root_pos_w - scene.env_origins)[0, :2]
    slot = "slot0" if float(scene.red_slot[0]) > 0 else "slot1"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"pen=({float(pen_xy[0]):+.3f},{float(pen_xy[1]):+.3f}) "
          f"red=({float(red_xy[0]):+.3f},{float(red_xy[1]):+.3f}) [{slot}] "
          f"blue=({float(blue_xy[0]):+.3f},{float(blue_xy[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert float(scene.crate_up_z()[0]) > 0.98, "crate must start UPRIGHT (opening up)"
    assert not bool(scene.trapped()[0]), "nothing may start trapped"
    s0 = print_score("P0 reset+settle (crate upright and empty)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # Force-frame reference orientation for `encode_force` (readback at reset).
    q_ref = scene.crate.data.root_quat_w.clone()
    mode = 0  # probed from actual progress during the slide

    # ---------------- phase 1: invert-and-drop capture (gravity makes the trap) ------------
    hover_z = c.height / 2 + c.block + 0.018  # rim ~18 mm above the block top
    trapped = False
    for attempt in range(3):
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = scene.red.data.root_pos_w[:, :2]  # centre the walls on the block
        st[:, 2] = scene.env_origins[:, 2] + hover_z
        st[:, 3:7] = _qx(torch.full((n,), math.pi, device=device))
        scene.crate.write_root_state_to_sim(st, all_ids)
        step(200)  # free fall + rim seating + settle, hands-off
        trapped = bool(scene.trapped()[0]) and bool(scene.settled()[0])
        if trapped:
            break
        print(f"[solve] capture attempt {attempt} missed "
              f"(trapped={bool(scene.trapped()[0])}); re-dropping", flush=True)
    report("capture")
    if not trapped:
        print("SIM_GEN_SOLVE: FAIL (capture drop never trapped the block)", flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "cannot be success away from the pen"
    s1 = print_score("P1 crate dropped over the red block; trap made by gravity")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_trap - 1e-6, f"P1 score {s1} (expect trapped=0.30)"

    # ---------------- phase 2: slide the trap into the pen (applied force + friction) ------
    def crate_xy() -> torch.Tensor:
        return scene.crate.data.root_pos_w[0, :2]

    def slide_to(target_xy: torch.Tensor, *, final: bool, tag: str) -> bool:
        """Velocity-regulated horizontal CoM push toward `target_xy` (world). Returns
        True when the stop condition is met with the block still trapped."""
        nonlocal mode
        floor_f = 2.6  # stiction floor (static drag of crate + dragged block ~2.2 N)
        win_i, win_dist = 0, float((target_xy - crate_xy()).norm())
        lost = 0
        for i in range(3600):
            d_vec = target_xy - crate_xy()
            dist = float(d_vec.norm())
            if final:
                d_pen = scene._pen_local_xy(scene.crate.data.root_pos_w)[0].abs()
                if bool((d_pen < 0.020).all()):
                    clear_forces()
                    return True
            elif dist < 0.030:
                clear_forces()
                return True
            # tilt guard: pushing paused while the crate rocks
            if float(scene.crate_up_z()[0]) > -0.98:
                clear_forces()
                step(30)
                continue
            # containment watch: the mechanism must hold; a lost block aborts the leg
            if not bool(scene._under_crate(scene.red, c.capture_xy_tol)[0]):
                lost += 1
                if lost > 90:
                    clear_forces()
                    print(f"[solve] {tag}: block escaped during the slide", flush=True)
                    return False
            else:
                lost = 0
            u = d_vec / max(dist, 1e-6)
            v_des = u * (0.12 if dist > 0.06 else 0.05)
            v = scene.crate.data.root_lin_vel_w[0, :2]
            f_xy = 9.0 * (v_des - v)
            f_along = float((f_xy * u).sum())
            if float(v.norm()) < 0.02 and f_along < floor_f:
                f_xy = f_xy + u * (floor_f - f_along)  # break stiction
            fn = float(f_xy.norm())
            if fn > 5.0:
                f_xy = f_xy * (5.0 / fn)  # stay under the tipping margin
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, :2] = f_xy
            f_arg = encode_force(mode, q_ref, scene.crate.data.root_quat_w, f_world)
            scene.crate.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
            env.step(no_action)
            # progress probe every 45 steps: wrong force-frame mode moves us AWAY
            if i - win_i >= 45:
                if dist > win_dist + 0.008:
                    mode = 1 - mode
                    print(f"[solve] {tag}: moving away (dist {win_dist:.3f} -> "
                          f"{dist:.3f}); force-frame mode -> {mode}", flush=True)
                elif dist > win_dist - 0.005:
                    floor_f = min(floor_f + 0.4, 3.6)
                    print(f"[solve] {tag}: stalled at dist {dist:.3f}; "
                          f"stiction floor -> {floor_f:.1f} N", flush=True)
                win_i, win_dist = i, dist
        clear_forces()
        print(f"[solve] {tag}: leg timed out at dist "
              f"{float((target_xy - crate_xy()).norm()):.3f}", flush=True)
        return False

    # Plan the path: detour if the straight line to the pen grazes the blue block.
    pen_w = scene.pen.data.root_pos_w[0, :2]
    blue_w = scene.blue.data.root_pos_w[0, :2]
    a, b = crate_xy().clone(), pen_w.clone()
    ab = b - a
    t = float(((blue_w - a) @ ab) / max(float(ab @ ab), 1e-9))
    t = min(max(t, 0.0), 1.0)
    proj = a + t * ab
    clearance = float((blue_w - proj).norm())
    legs: list[torch.Tensor] = []
    if clearance < 0.17:
        away = proj - blue_w
        away = away / max(float(away.norm()), 1e-6)
        wp = proj + away * 0.20
        print(f"[solve] blue block {clearance:.3f} m off the straight path; "
              f"detour via ({float(wp[0]):+.3f},{float(wp[1]):+.3f})", flush=True)
        legs.append(wp)
    legs.append(pen_w)

    ok = True
    for j, target in enumerate(legs):
        final = j == len(legs) - 1
        ok = slide_to(target, final=final, tag=f"leg{j}")
        if not ok:
            break
    if not ok:
        # one full recovery: re-make the trap by gravity where the block now lies,
        # then slide straight in (transport-only teleport, capture still by contact)
        print("[solve] recovery: re-dropping the crate over the block", flush=True)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = scene.red.data.root_pos_w[:, :2]
        st[:, 2] = scene.env_origins[:, 2] + hover_z
        st[:, 3:7] = _qx(torch.full((n,), math.pi, device=device))
        scene.crate.write_root_state_to_sim(st, all_ids)
        step(200)
        ok = bool(scene.trapped()[0]) and slide_to(pen_w, final=True, tag="recover")
    step(180)  # full settle, hands-off
    report("slide")
    if not ok or not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (trap not delivered to the pen)", flush=True)
        os._exit(1)
    s2 = print_score("P2 capped crate slid into the pen under containment")
    assert s2 >= s1 - 1e-6, "score decreased across the haul"
    assert s2 >= 1.0 - 1e-6, f"success must score 1.0, got {s2}"

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
