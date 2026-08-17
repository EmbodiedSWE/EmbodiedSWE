"""Teleport solution for GrooveLidChestScene (sim_gen task `close_box_i300`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the free LID PLATE from its
   floor spawn to a free-space hover 30 mm above the loading PORCH (chest-local
   x = drop_x, centred, knob up, long axis along the rails), zero velocity. The write
   satisfies no rubric clause by itself: the plate is airborne (not in the channel
   plane, not engaged, nothing latched — `seated` demands REST at shelf height).
2. SEAT (gravity + contact): the plate FALLS ~30 mm between the flared guide walls and
   lands on the two porch rails, settling in the channel plane. The `seated` fact is
   produced by ballistics and contact, never written. If a bounce spoils the seat, the
   plate is lifted (transport) and dropped again; the seat itself is always made by
   gravity.
3. THREAD + SLIDE (applied force + contact): a horizontal force at the plate's CoM —
   the exact wrench of a fingertip push on the red knob — drives it along chest-local
   -x, velocity-regulated (~0.10 m/s, cap 4.0 N, stiction floor with stall
   escalation). The leading edge threads into BOTH C-grooves at the slot mouths
   (guided by the flared walls + slot geometry, not by any write), then the plate runs
   captive down the channel until it hits the -x end STOP. A final gentle press
   (~1.2 N, 0.5 s) squares it against the stop; forces are cleared and everything
   settles hands-off. Every `engaged` / `deep` / `covered` fact comes from contact
   dynamics. The gold cube is never touched: it simply stays where it lies while the
   lid closes 22 mm above it.
   Pod force-frame quirk: some pods rotate an applied wrench by the body's rotation
   since its reference orientation, so the desired world force is pre-encoded per
   `encode_force` and the correct mode is PROBED from actual progress, not assumed.
4. ORDER: seat-then-slide is physically forced (the flanges make the grooves
   unreachable from above — asserted in the scene cfg); within that the task imposes
   no further order and this demonstration is the canonical porch entry.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_box_i300.solve --headless [--seed N]
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

import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from .scene import _qapply, encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.groove_lid_chest")().build(num_envs=args.num_envs,
                                                      device=device)
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
        scene.plate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)

    def report(tag: str) -> None:
        p = scene.plate_local()[0]
        print(f"[solve] {tag:12s} | plate_local=({float(p[0]):+.3f},"
              f"{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"depth={float(scene.lead_in()[0]):+.3f} "
              f"chan={bool(scene.in_channel()[0])} "
              f"seat={bool(scene._seated[0])} eng={bool(scene._engaged[0])} "
              f"deep={bool(scene._deep[0])} covered={bool(scene.covered()[0])} "
              f"cube_in={bool(scene.cube_in()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)
    q_chest = scene.chest.data.root_quat_w.clone()
    chest_xy = (scene.chest.data.root_pos_w - scene.env_origins)[0, :2]
    plate_xy = (scene.plate.data.root_pos_w - scene.env_origins)[0, :2]
    cube_loc = scene._chest_local(scene.cube.data.root_pos_w)[0]
    yaw = float(2.0 * torch.atan2(q_chest[0, 3], q_chest[0, 0]))
    side = "+y" if float(scene.plate_side[0]) > 0 else "-y"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"chest=({float(chest_xy[0]):+.3f},{float(chest_xy[1]):+.3f}) "
          f"yaw={yaw:+.2f} rad | plate=({float(plate_xy[0]):+.3f},"
          f"{float(plate_xy[1]):+.3f}) [{side}] | "
          f"cube_local=({float(cube_loc[0]):+.3f},{float(cube_loc[1]):+.3f},"
          f"{float(cube_loc[2]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.cube_in()[0]), "cube must start inside the cavity"
    assert not bool(scene.in_channel()[0]), "plate must start on the floor, not seated"
    s0 = print_score("P0 reset+settle (plate on the floor, chest open)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: porch drop (gravity makes the seat) -------------------------
    # Hover pose: chest-local (drop_x, 0, z_seat + 0.030), aligned with the chest,
    # knob up (trailing +x, away from the stop). Free space: the slab bottom clears
    # the guide-wall tops by ~10 mm.
    seated = False
    for attempt in range(3):
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = c.drop_x
        loc[:, 2] = c.z_seat + 0.030
        pos_w = scene.chest.data.root_pos_w + _qapply(q_chest, loc)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = q_chest
        scene.plate.write_root_state_to_sim(st, all_ids)
        step(180)  # free fall + rail seating + settle, hands-off
        seated = bool(scene.seated_now()[0]) and bool(scene.settled()[0])
        if seated:
            break
        print(f"[solve] seat attempt {attempt} missed "
              f"(chan={bool(scene.in_channel()[0])}); re-dropping", flush=True)
    report("seat")
    if not seated:
        print("SIM_GEN_SOLVE: FAIL (porch drop never seated the plate)", flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "cannot be success out on the porch"
    s1 = print_score("P1 plate dropped onto the porch; seat made by gravity")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_seat - 1e-6, f"P1 score {s1} (expect seated=0.15)"

    # Force-frame reference orientation for `encode_force` (readback at the seat).
    q_ref = scene.plate.data.root_quat_w.clone()
    mode = 0  # probed from actual progress during the slide

    # ---------------- phase 2: thread + slide to the stop (applied force + contact) --------
    # Push direction: chest-local -x expressed in world (the chest is kinematic).
    u3 = _qapply(q_chest, torch.tensor([[-1.0, 0.0, 0.0]], device=device).expand(n, 3))
    u_xy = u3[0, :2] / max(float(u3[0, :2].norm()), 1e-6)

    def px_now() -> float:
        return float(scene.plate_local()[0, 0])

    def slide_home(tag: str) -> bool:
        """Velocity-regulated chest-local -x CoM push until the plate reaches the
        stop (px <= px_home + 4 mm). Returns False on timeout/derail."""
        nonlocal mode
        floor_f = 0.8  # stiction floor (mu 0.15 * 0.18 kg * g ~ 0.27 N static drag)
        target = c.px_home + 0.004
        win_i, win_px = 0, px_now()
        for i in range(3000):
            px = px_now()
            if px <= target:
                clear_forces()
                return True
            # derail guard: the plate must stay in the channel bands once engaged
            p = scene.plate_local()[0]
            if float(p[2]) > c.z_seat + 0.020 or float(p[1].abs()) > c.wall_in:
                clear_forces()
                print(f"[solve] {tag}: derailed at local=({float(p[0]):+.3f},"
                      f"{float(p[1]):+.3f},{float(p[2]):+.3f})", flush=True)
                return False
            dist = px - c.px_home
            v_des_mag = 0.10 if dist > 0.050 else 0.04
            v = scene.plate.data.root_lin_vel_w[0, :2]
            v_along = float((v * u_xy).sum())
            f_along = 6.0 * (v_des_mag - v_along)
            if abs(v_along) < 0.02 and f_along < floor_f:
                f_along = floor_f  # break stiction
            f_along = min(max(f_along, -1.0), 4.0)
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, :2] = u_xy * f_along
            f_arg = encode_force(mode, q_ref, scene.plate.data.root_quat_w, f_world)
            scene.plate.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
            env.step(no_action)
            # progress probe every 45 steps: wrong force-frame mode moves us AWAY
            if i - win_i >= 45:
                px2 = px_now()
                if px2 > win_px + 0.008:
                    mode = 1 - mode
                    print(f"[solve] {tag}: moving away (px {win_px:.3f} -> "
                          f"{px2:.3f}); force-frame mode -> {mode}", flush=True)
                elif px2 > win_px - 0.004:
                    floor_f = min(floor_f + 0.4, 3.0)
                    print(f"[solve] {tag}: stalled at px {px2:.3f}; "
                          f"stiction floor -> {floor_f:.1f} N", flush=True)
                win_i, win_px = i, px2
        clear_forces()
        print(f"[solve] {tag}: leg timed out at px {px_now():.3f}", flush=True)
        return False

    ok = slide_home("slide")
    if not ok:
        # one full recovery: re-make the seat by gravity, then slide again
        print("[solve] recovery: re-dropping the plate onto the porch", flush=True)
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = c.drop_x
        loc[:, 2] = c.z_seat + 0.030
        pos_w = scene.chest.data.root_pos_w + _qapply(q_chest, loc)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = q_chest
        scene.plate.write_root_state_to_sim(st, all_ids)
        step(180)
        ok = bool(scene.seated_now()[0]) and slide_home("recover")
    if ok:
        # square against the stop: gentle press ~0.5 s, then hands-off settle
        f_world = torch.zeros(n, 3, device=device)
        f_world[0, :2] = u_xy * 1.2
        for _ in range(60):
            f_arg = encode_force(mode, q_ref, scene.plate.data.root_quat_w, f_world)
            scene.plate.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero_wrench, env_ids=all_ids, is_global=True)
            env.step(no_action)
        clear_forces()
        step(180)  # full settle, hands-off
    report("slide")
    if not ok or not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (lid not threaded home)", flush=True)
        os._exit(1)
    s2 = print_score("P2 lid threaded through the grooves to the stop; mouth covered")
    assert s2 >= s1 - 1e-6, "score decreased across the slide"
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
