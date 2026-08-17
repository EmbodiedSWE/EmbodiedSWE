"""Teleport solution for ComboShutterPantryScene (sim_gen task
`libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i336`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in one write ending in FREE SPACE:
  P3a — the pan is carried from its random porch spawn to a staging pose on the bare
  counter in front of the shutter gate (8 cm clear of the front ridge), handle
  trailing — the reorient-and-set-down a robot performs before a planar push. It is
  released there and settles; it is never teleported into, past, or through the gate.
Everything else is DRIVEN, not bypassed:
  P1 — a force governor applies a bounded horizontal force to the REAR (orange)
  shutter plate along the cabinet's y axis (read from the scene: the cabinet yaw is
  randomized) — the force a hand pushing the plate's black tab applies — sliding it
  along its guide slot against friction until its cut-out registers with the doorway.
  P2 — the same governor aligns the FRONT (green) plate. Both plates are now
  registered SIMULTANEOUSLY: the combination is open.
  P3b — a bounded push drives the pan FLAT along the counter, through both plate
  cut-outs, under their bridges, and through the doorway into the alcove, with a small
  lateral servo keeping it centred — exactly the planar push an arm performs on a pan
  too wide to pass any other way. The force is cut and everything settles on real
  contact. The pan is never teleported past the gate; the alignment latches, the
  transit latch, and success() are all earned by physics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i336.solve --headless [--seed N]
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
from isaaclab.utils.math import quat_apply_inverse

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
    env = ENVS.get("simgen.combo_shutter_pantry")().build(num_envs=args.num_envs,
                                                          device=device)
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

    def cab_axes() -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame unit vectors of the cabinet's +x (push-through) and +y axes."""
        yaw = float(scene._yaw_of(scene.cabinet)[0])
        xd = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=device)
        yd = torch.tensor([-math.sin(yaw), math.cos(yaw), 0.0], device=device)
        return xd, yd

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_force(body, fvec: torch.Tensor) -> None:
        # Pre-encode the commanded WORLD force into the body's LIVE frame and use
        # the default (body-frame) call. On these pods `is_global=True` rotates
        # the wrench by the body's rotation since its FIRST-ever application (the
        # stale reference persists across env.reset), which mis-aims any later
        # push. quat_apply_inverse with the live quaternion is exact for these
        # upright, yaw-only sliding bodies (no rolling).
        fb = quat_apply_inverse(body.data.root_quat_w[0:1], fvec.view(1, 3))
        fw = fb.view(1, 1, 3).expand(n, 1, 3).contiguous()
        body.set_external_force_and_torque(fw, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        aF = float(scene.plate_offset(scene.plate_f)[0])
        aR = float(scene.plate_offset(scene.plate_r)[0])
        pl = scene.cab_local(scene.pan.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | aF={aF * 1000:+.1f}mm aR={aR * 1000:+.1f}mm "
              f"seatF={bool(scene.plate_seated(scene.plate_f, c.slot_xF)[0])} "
              f"seatR={bool(scene.plate_seated(scene.plate_r, c.slot_xR)[0])} "
              f"pan_cab=({float(pl[0]):+.3f},{float(pl[1]):+.3f},{float(pl[2]):.3f}) "
              f"inside={bool(scene.pan_inside()[0])} "
              f"upright={bool(scene.pan_upright()[0])} "
              f"lF={float(scene.alignF_latch[0]):.2f} lR={float(scene.alignR_latch[0]):.2f} "
              f"lB={float(scene.both_latch[0]):.2f} lT={float(scene.transit_latch[0]):.2f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def align_plate(body, slot_x: float, label: str, budget: int = 1500) -> bool:
        """Force-govern one plate along the cabinet's y axis until its cut-out
        registers with the doorway (|offset| < 5 mm). Bounded tab force, velocity
        servo with a taper near the target so it cannot overshoot past align_tol."""
        # Friction feedforward (~0.7 N > the ~0.6 N static friction of the 0.25 kg
        # plate) + a modest velocity servo (gain * dt / m ~ 0.67, safely under the
        # one-substep wrench delay bound). A plain velocity servo stalls where its
        # commanded force equals friction — the feedforward removes that stall point.
        gain, cap, ff = 20.0, 3.0, 0.70
        for i in range(budget):
            a = float(scene.plate_offset(body)[0])
            if abs(a) < 0.004:
                clear_wrench(body)
                return True
            _, yd = cab_axes()
            v = float(torch.dot(body.data.root_lin_vel_w[0], yd))
            v_des = -math.copysign(min(0.06, 2.0 * abs(a) + 0.015), a)
            f = max(-cap, min(cap, math.copysign(ff, v_des) + gain * (v_des - v)))
            apply_force(body, yd * f)
            env.step(no_action)
            if i == budget // 2 and abs(float(scene.plate_offset(body)[0])) > 0.008:
                ff, cap = 1.2, 4.5
                print(f"[solve] {label}: slow at step {i} "
                      f"(offset {a * 1000:+.1f}mm) — raising drive", flush=True)
        clear_wrench(body)
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    hp = (scene.cabinet.data.root_pos_w - scene.env_origins)[0]
    hyaw = float(scene._yaw_of(scene.cabinet)[0])
    aF0 = float(scene.plate_offset(scene.plate_f)[0])
    aR0 = float(scene.plate_offset(scene.plate_r)[0])
    pl0 = scene.cab_local(scene.pan.data.root_pos_w)[0]
    dl0 = scene.cab_local(scene.decoy.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): cabinet=({float(hp[0]):+.3f},"
          f"{float(hp[1]):+.3f}) yaw={math.degrees(hyaw):+.1f}deg "
          f"aF={aF0 * 1000:+.1f}mm aR={aR0 * 1000:+.1f}mm "
          f"pan_cab=({float(pl0[0]):+.3f},{float(pl0[1]):+.3f}) "
          f"decoy_cab=({float(dl0[0]):+.3f},{float(dl0[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: DRIVE the rear (orange) plate into registration -------------
    ok_r = align_plate(scene.plate_r, c.slot_xR, "align R")
    step(45)
    report("aligned R")
    s1 = print_score("P1 rear plate registered")
    assert s1 >= s0 - 1e-6, "score decreased across the rear alignment"
    if not ok_r or abs(float(scene.plate_offset(scene.plate_r)[0])) > c.align_tol:
        print("SIM_GEN_SOLVE: FAIL (rear plate did not register)", flush=True)
        os._exit(1)

    # ---------------- phase 2: DRIVE the front (green) plate into registration -------------
    ok_f = align_plate(scene.plate_f, c.slot_xF, "align F")
    step(45)
    report("aligned F")
    s2 = print_score("P2 front plate registered (combination open)")
    assert s2 >= s1 - 1e-6, "score decreased across the front alignment"
    if not ok_f or abs(float(scene.plate_offset(scene.plate_f)[0])) > c.align_tol \
            or float(scene.both_latch[0]) < 0.5:
        print("SIM_GEN_SOLVE: FAIL (front plate did not register)", flush=True)
        os._exit(1)

    # ---------------- phase 3: pan TRANSPORT to the staging pose + DRIVEN push through -----
    # Transport (free space): set the pan down on the bare counter 8 cm before the
    # front ridge, handle trailing, centred on the doorway line. Then a bounded force
    # pushes it flat through the registered gate; a small lateral servo keeps it
    # centred so the rim never snags the ledge ends.
    cab_p = scene.cabinet.data.root_pos_w[0:1]
    cab_q = scene.cabinet.data.root_quat_w[0:1]
    from isaaclab.utils.math import quat_apply, quat_mul

    stage_local = torch.tensor([[-0.150, 0.0, 0.004]], device=device)
    flip = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)  # yaw pi: handle -> -x
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = cab_p + quat_apply(cab_q, stage_local)
    st[:, 3:7] = quat_mul(cab_q, flip)
    scene.pan.write_root_state_to_sim(st, all_ids)
    step(60)  # release + settle on the counter
    report("staged")

    # Position-carrot push: a bounded spring toward a waypoint that advances along
    # the doorway centreline. Direction comes from POSITION error (GPU velocity
    # readbacks carry phantom noise that made velocity-referenced servos veer and
    # stall), so a lateral drift is always pulled back to y = 0. Force stays
    # capped at 5 N — when the pan is obstructed the carrot saturates into the
    # same bounded press as before.
    pushed = False
    x_ref = float(scene.cab_local(scene.pan.data.root_pos_w)[0][0])
    kx, dx_, cap = 120.0, 8.0, 5.0
    ky, dy_, capy = 250.0, 10.0, 4.5
    for i in range(2000):
        loc = scene.cab_local(scene.pan.data.root_pos_w)[0]
        x, y = float(loc[0]), float(loc[1])
        if x > 0.210:
            pushed = True
            break
        x_ref = min(x_ref + 0.05 / 120.0, 0.210 + cap / kx + 0.02)
        xd, yd = cab_axes()
        v = scene.pan.data.root_lin_vel_w[0]
        vx = float(torch.dot(v, xd))
        vy = float(torch.dot(v, yd))
        fx = max(-cap, min(cap, kx * (x_ref - x) - dx_ * vx))
        fy = max(-capy, min(capy, -ky * y - dy_ * vy))
        apply_force(scene.pan, xd * fx + yd * fy)
        env.step(no_action)
    clear_wrench(scene.pan)
    step(90)
    report("pushed in")
    s3 = print_score("P3 pan staged + pushed through the registered gate")
    assert s3 >= s2 - 1e-6, "score decreased across the push"
    if not pushed or not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (pan did not end inside the alcove)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) -----
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
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)
