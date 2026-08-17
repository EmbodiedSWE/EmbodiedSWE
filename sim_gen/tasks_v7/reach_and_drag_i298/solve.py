"""Teleport solution for DropCourierScene (sim_gen task `reach_and_drag_i298`) — the
task's legitimacy certificate.

There is NO teleport of any body in this solve after reset: every phase is executed
through contact dynamics.

1. ALIGN UNDER CONTACT (applied horizontal force + friction): the tray is pushed along
   its channel with a CoM-level force — the same shove a fingertip on the tray's end
   wall produces. A velocity servo (|v| capped at 0.12 m/s, slowed near the target)
   drives the rig-frame error |cargo_y - tray_y| below 8 mm; ground friction brakes and
   holds it there. A stiction floor escalates on measured stall (the break-away force is
   a physical unknown), and a force-frame probe toggles the encoding if the tray moves
   AWAY (some pods rotate applied wrenches by the body's rotation since reset).
2. DISPENSE UNDER CONTACT + GRAVITY: the cargo cube is pushed on the ledge top toward
   the drop edge (velocity-regulated, force capped at 2.2 N < the m*g = 2.45 N tipping
   bound), accelerated to ~0.3 m/s for the exit so its CoM carries past the tray's back
   wall, and RELEASED before the edge: it tips over the edge and GRAVITY delivers it
   into the waiting tray. Nothing steers the fall.
3. SHUTTLE UNDER CONTACT: the loaded tray (with the cube riding inside — containment is
   judged in the tray's body frame) is pushed along the channel to the green-post end
   until it rests in the bay band against/at the end stop.
4. ORDER: align FIRST, then drop — the task's forced order (a cube dropped with the
   tray elsewhere lands on the channel floor; it is 90 mm > the 80 mm jaw and the tray
   walls are above its centre, so it can never be loaded from the ground).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
partial credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still
holds.

Run (forge): python -u -m simgen_tasks.reach_and_drag_i298.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drop_courier")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

    def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                     f_world: torch.Tensor) -> torch.Tensor:
        """Pre-encode a desired WORLD force. mode 0: pass through. mode 1: premultiply
        by R_ref * R_now^T for pods that rotate wrenches by rotation-since-reset."""
        if mode == 0:
            return f_world
        return quat_apply(quat_mul(q_ref, quat_conjugate(q_now)), f_world)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.tray.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
        scene.cargo.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def rig_loc(body) -> torch.Tensor:
        return scene._rig_local(body.data.root_pos_w)[0]

    def report(tag: str) -> None:
        tl = rig_loc(scene.tray)
        cl = rig_loc(scene.cargo)
        print(f"[solve] {tag:12s} | tray=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):+.3f}) cargo=({float(cl[0]):+.3f},{float(cl[1]):+.3f},"
              f"{float(cl[2]):+.3f}) align_err={float(scene.align_err()[0]) * 1000:+.1f}mm "
              f"bay_err={float(scene.bay_err()[0]) * 1000:.1f}mm "
              f"on_ledge={bool(scene.cargo_on_ledge()[0])} "
              f"contained={bool(scene.contained()[0])} parked={bool(scene.parked()[0])} "
              f"aligned_l={bool(scene._aligned[0])} loaded_l={bool(scene._loaded[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    rig_q = scene.rig.data.root_quat_w  # kinematic: constant after reset
    ey_w = quat_apply(rig_q, torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))[0]
    ex_w = quat_apply(rig_q, torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]

    def push_tray_to(target_y: float, tol: float, max_steps: int, tag: str) -> bool:
        """Velocity-servo the tray along the channel (rig frame) to `target_y`.
        CoM-level horizontal force, capped; stiction floor escalates on measured
        stall; force-frame mode probed from progress. Returns True on arrival."""
        mode, floor_f = 0, 2.8  # mu*m*g ~ 2.5 N for the 0.5 kg tray (+ cargo later)
        q_ref = scene.tray.data.root_quat_w.clone()
        win_i, win_e = 0, abs(float(scene._rig_local(scene.tray.data.root_pos_w)[0, 1])
                              - target_y)
        for i in range(max_steps):
            ty = float(scene._rig_local(scene.tray.data.root_pos_w)[0, 1])
            err = target_y - ty
            if abs(err) < tol and float(scene.tray.data.root_lin_vel_w[0].norm()) < 0.03:
                clear_forces()
                return True
            v_cap = 0.12 if abs(err) > 0.06 else 0.05
            v_des = max(-v_cap, min(v_cap, 2.0 * err))
            v = scene.tray.data.root_lin_vel_w[0, :3]
            v_along = float((v[:2] * ey_w[:2]).sum())
            f_mag = 25.0 * (v_des - v_along)
            # Stiction/creep floor: a pure P velocity servo settles at the speed where
            # gain*(v_des - v) equals kinetic friction — for the LOADED tray that is a
            # ~0.01 m/s creep that quietly outlasts the step budget. Keep the force at
            # or above the floor until the tray actually moves at a useful fraction of
            # v_des (friction still brakes and holds it at the target).
            if abs(v_along) < max(0.04, 0.5 * abs(v_des)) and abs(err) > tol \
                    and 0.0 < abs(f_mag) < floor_f:
                f_mag = math.copysign(floor_f, f_mag)
            f_mag = max(-9.0, min(9.0, f_mag))
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, :3] = ey_w * f_mag
            f_arg = encode_force(mode, q_ref, scene.tray.data.root_quat_w, f_world)
            scene.tray.set_external_force_and_torque(f_arg.view(n, 1, 3), zero_wrench,
                                                     env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i - win_i >= 45:
                e_now = abs(float(scene._rig_local(scene.tray.data.root_pos_w)[0, 1])
                            - target_y)
                if e_now > win_e + 0.008:
                    mode = 1 - mode
                    print(f"[solve] {tag}: tray moving away "
                          f"(err {win_e * 1000:.0f} -> {e_now * 1000:.0f} mm); "
                          f"force-frame mode -> {mode}", flush=True)
                elif e_now > win_e - (0.015 if e_now > 0.06 else 0.004) and e_now > tol:
                    floor_f = min(floor_f + 0.6, 6.0)
                    print(f"[solve] {tag}: tray stalled at err={e_now * 1000:.0f}mm; "
                          f"stiction floor -> {floor_f:.1f} N", flush=True)
                win_i, win_e = i, e_now
        clear_forces()
        return False

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(180)  # everything seats
    tl0 = rig_loc(scene.tray)
    cl0 = rig_loc(scene.cargo)
    bq = scene.rig.data.root_quat_w[0]
    byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
    sgn = float(scene.bay_sign[0])
    pl = rig_loc(scene.post)
    print(f"[solve] layout readback (seed {args.seed}): rig_yaw={byaw:+.1f}deg "
          f"cargo_y={float(cl0[1]) * 1000:+.0f}mm tray_y={float(tl0[1]) * 1000:+.0f}mm "
          f"bay_sign={sgn:+.0f} post_y={float(pl[1]) * 1000:+.0f}mm "
          f"align_err={float(scene.align_err()[0]) * 1000:+.0f}mm "
          f"bay_err={float(scene.bay_err()[0]) * 1000:.0f}mm", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.cargo_on_ledge()[0]), "cargo must start on the ledge"
    assert bool(scene.tray_in_channel()[0]), "tray must start in the channel"
    assert sgn * float(pl[1]) > 0.4, "green post must stand at the bay end (readback)"
    assert abs(float(scene.align_err()[0])) > c.tray_cube_min - 0.02, \
        "tray must start off the drop line"
    assert float(scene.bay_err()[0]) > c.tray_bay_min - 0.02, \
        "tray must start off the bay"
    s0 = print_score("P0 reset+settle (tray off the drop line and off the bay)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: slide the tray under the drop line --------------------------
    target = float(rig_loc(scene.cargo)[1])
    ok = push_tray_to(target, 0.008, 3000, "align")
    step(150)  # friction brake + settle, hands-off
    report("aligned")
    if not ok:
        print("SIM_GEN_SOLVE: FAIL (tray align servo timed out)", flush=True)
        os._exit(1)
    assert abs(float(scene.align_err()[0])) < c.align_tol, "tray must sit on the drop line"
    assert bool(scene._aligned[0]), "aligned latch must have fired"
    assert not bool(scene.success()[0]), "cannot be success with the cargo on the ledge"
    s1 = print_score("P1 tray waiting under the drop line (slide under contact)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_aligned - 1e-6, \
        f"P1 score {s1} (expect aligned={c.w_aligned})"

    # ---------------- phase 2: push the cargo off the edge; gravity loads the tray ---------
    # Velocity-regulated CoM push toward local -x; force capped below the m*g tipping
    # bound; accelerate for the exit so the CoM carries past the tray's back wall;
    # RELEASE before the edge — the drop itself is pure gravity.
    mode, floor_f = 0, 1.4  # mu*m*g ~ 1.2 N for the 0.25 kg cube
    q_ref = scene.cargo.data.root_quat_w.clone()
    edge = c.ledge_face_x
    win_i, win_x = 0, float(rig_loc(scene.cargo)[0])
    released = False
    for i in range(2400):
        x_loc = float(rig_loc(scene.cargo)[0])
        z_loc = float(rig_loc(scene.cargo)[2])
        if x_loc < edge + 0.004 or z_loc < c.ledge_h + 0.030:
            released = True
            clear_forces()
            break
        v = scene.cargo.data.root_lin_vel_w[0, :3]
        v_along = float(-(v[:2] * ex_w[:2]).sum())  # speed toward -x
        v_des = 0.12 if x_loc > edge + 0.075 else 0.30
        f_mag = 6.0 * (v_des - v_along)
        if v_along < 0.02 and 0.0 < f_mag < floor_f:
            f_mag = floor_f
        f_mag = max(-2.2, min(2.2, f_mag))
        f_world = torch.zeros(n, 3, device=device)
        f_world[0, :3] = -ex_w * f_mag
        f_arg = encode_force(mode, q_ref, scene.cargo.data.root_quat_w, f_world)
        scene.cargo.set_external_force_and_torque(f_arg.view(n, 1, 3), zero_wrench,
                                                  env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i - win_i >= 45:
            if x_loc > win_x + 0.008:
                mode = 1 - mode
                print(f"[solve] dispense: cargo moving away (x {win_x:+.3f} -> "
                      f"{x_loc:+.3f}); force-frame mode -> {mode}", flush=True)
            elif x_loc > win_x - 0.004:
                floor_f = min(floor_f + 0.3, 2.1)
                print(f"[solve] dispense: stalled at x={x_loc:+.3f}; "
                      f"stiction floor -> {floor_f:.1f} N", flush=True)
            win_i, win_x = i, x_loc
    clear_forces()
    step(300)  # free fall + landing + full settle, hands-off
    report("dropped")
    if not released:
        print("SIM_GEN_SOLVE: FAIL (cargo never reached the drop edge)", flush=True)
        os._exit(1)
    if not bool(scene.contained()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (cargo did not land inside the tray)", flush=True)
        os._exit(1)
    assert bool(scene._loaded[0]), "loaded latch must have fired"
    s2 = print_score("P2 cargo dropped into the tray (gravity delivery)")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_aligned + c.w_loaded - 1e-6, \
        f"P2 score {s2} (expect {c.w_aligned + c.w_loaded})"

    # ---------------- phase 3: shuttle the loaded tray to the green bay --------------------
    bay_y = float(scene.bay_sign[0]) * c.bay_center_y
    ok = push_tray_to(bay_y, 0.012, 3600, "park")
    step(180)  # settle, hands-off
    report("parked")
    if not ok:
        print("SIM_GEN_SOLVE: FAIL (tray park servo timed out)", flush=True)
        os._exit(1)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after parking the loaded tray)", flush=True)
        os._exit(1)
    s3 = print_score("P3 loaded tray parked at the green bay")
    assert s3 >= s2 - 1e-6 and s3 >= 0.999, f"P3 score {s3} (expect success=1.0)"

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
    main()
