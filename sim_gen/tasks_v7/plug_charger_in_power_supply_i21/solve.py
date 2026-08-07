"""Teleport solution for KeyholeUnplugScene (sim_gen task
`plug_charger_in_power_supply_i21`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY — exactly ONE pose write, and it happens when
the blue plug is already a FREE body in mid-air (freshly lifted out of the strip),
carrying it across open floor to a hover INSIDE the tray, above its floor. Every
load-bearing interaction goes through CONTACT DYNAMICS:
  P1 — UNLOCK SLIDE: a floating-hand force controller (velocity-regulated push along
  the strip-local slot direction, lateral PD centring the foot in its channel,
  upright/yaw-steadying torque, stall escalation) slides the still-captive plug from
  the lock seat to the wide opening. The foot drags on the real channel floor and is
  guided by the real cavity walls and slot rails the whole way; the free-standing
  strip is never held (the push force is far below its friction footprint).
  P2 — LIFT-OUT: the same controller switches to a velocity-regulated vertical pull
  (force capped at plug weight + 2.5 N — a locked plug CANNOT be freed this way; the
  cap is an order of magnitude below the 14.7 N strip weight, so the lift never
  hoists the strip) drawing the foot up through the wide opening under real contact
  guidance until the plug is clear of the lid.
  P3 — STOW: the freed plug is teleported (transport through free space) to a lying
  hover 9 mm above the tray floor and RELEASED: the set-down is a real gravity drop
  onto the tray floor, then everything settles.
The plug is never teleported out of (or into) the socket; the slide and the lift —
the two interactions the keyhole latch exists to force — are pure contact physics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.plug_charger_in_power_supply_i21.solve --headless [--seed N]
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
    env = ENVS.get("simgen.keyhole_unplug")().build(num_envs=args.num_envs, device=device)
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

    def strip_pose() -> tuple[torch.Tensor, float]:
        sp = (scene.strip.data.root_pos_w - scene.env_origins)[0]
        q = scene.strip.data.root_quat_w[0]
        return sp, 2.0 * math.atan2(float(q[3]), float(q[0]))

    def blue_loc() -> torch.Tensor:
        return scene._strip_local(scene.plug_blue.data.root_pos_w)[0]

    def report(tag: str) -> None:
        bl = blue_loc()
        tl = scene._tray_local(scene.plug_blue.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | blue_strip_local=({float(bl[0]):+.3f},"
              f"{float(bl[1]):+.3f},{float(bl[2]):.3f}) "
              f"blue_tray_local=({float(tl[0]):+.3f},{float(tl[1]):+.3f},{float(tl[2]):.3f}) "
              f"in_tray={bool(scene.blue_in_tray()[0])} "
              f"black_seated={bool(scene.black_seated()[0])} "
              f"strip_ok={bool(scene.strip_ok()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench() -> None:
        scene.plug_blue.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                      env_ids=all_ids)

    def apply_wrench(f_world: torch.Tensor, tq_world: torch.Tensor) -> None:
        scene.plug_blue.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)

    def steady_torque() -> torch.Tensor:
        """Upright + yaw-follow steadying torque (what a hand holding the knob does)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=device)
        axis = quat_apply(scene.plug_blue.data.root_quat_w, ez.expand(n, 3))[0]
        w_w = scene.plug_blue.data.root_ang_vel_w[0]
        _, syaw = strip_pose()
        q = scene.plug_blue.data.root_quat_w[0]
        pyaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        dyaw = math.atan2(math.sin(syaw - pyaw), math.cos(syaw - pyaw))
        tq = 0.05 * torch.linalg.cross(axis, ez) - 0.008 * w_w
        tq[2] += 0.01 * dyaw
        return tq.clamp(-0.06, 0.06)

    m_plug, g = c.plug_mass, 9.81

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    sp, syaw = strip_pose()
    tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
    bl = blue_loc()
    blue_side = "-x" if float(scene.blue_x0[0]) < 0 else "+x"
    print(f"[solve] layout readback (seed {args.seed}): strip=({float(sp[0]):+.3f},"
          f"{float(sp[1]):+.3f}) yaw={math.degrees(syaw):+.1f}deg blue_socket={blue_side} "
          f"tray=({float(tp[0]):+.3f},{float(tp[1]):+.3f}) "
          f"blue_seat_local=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):.3f})",
          flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: UNLOCK SLIDE (contact dynamics) -----------------------------
    # Velocity-regulated push along the strip-local +y slot direction; the foot drags
    # the real channel floor, guided by the cavity walls, until it sits under the
    # wide opening. The strip is NOT held: the push is ~1-2 N against its ~7 N
    # friction footprint (verified by the strip_ok readback below).
    from isaaclab.utils.math import quat_apply

    push_ff, v_des = 0.35, 0.10
    best_y, last_gain = -1.0, 0
    done_slide = False
    for i in range(2400):
        sp, syaw = strip_pose()
        loc = blue_loc()
        y = float(loc[1])
        if y >= c.y_free - 0.0015:
            done_slide = True
            break
        dir_w = torch.tensor([-math.sin(syaw), math.cos(syaw), 0.0], device=device)
        lat_w = torch.tensor([math.cos(syaw), math.sin(syaw), 0.0], device=device)
        v_w = scene.plug_blue.data.root_lin_vel_w[0]
        v_along = float((v_w * dir_w).sum())
        f_along = max(0.0, min(2.0, push_ff + m_plug * 60.0 * (v_des - v_along)))
        ex = float(scene.blue_x0[0]) - float(loc[0])
        v_lat = float((v_w * lat_w).sum())
        f_lat = max(-0.5, min(0.5, m_plug * (80.0 * ex - 20.0 * v_lat)))
        f_world = dir_w * f_along + lat_w * f_lat
        apply_wrench(f_world, steady_torque())
        env.step(no_action)
        if y > best_y + 0.001:
            best_y, last_gain = y, i
        elif i - last_gain > 240:  # stalled: lean a little harder
            push_ff = min(push_ff + 0.25, 1.8)
            last_gain = i
            print(f"[solve] slide stalled at y={y:+.4f}, push_ff={push_ff:.2f} N", flush=True)
    clear_wrench()
    step(30)
    report("slid")
    print(f"[solve] slide loop done (done_slide={done_slide}, y={float(blue_loc()[1]):+.4f})",
          flush=True)
    s1 = print_score("P1 unlock slide (contact)")
    assert s1 >= s0 - 1e-6, "score decreased across the unlock slide"
    if not done_slide and float(blue_loc()[1]) < c.y_free - 0.004:
        print("SIM_GEN_SOLVE: FAIL (slide did not reach the opening)", flush=True)
        os._exit(1)

    # ---------------- phase 2: LIFT-OUT (contact dynamics) ---------------------------------
    # Velocity-regulated vertical pull, force capped at m*g + 2.5 N (a LOCKED plug
    # cannot be freed by this force — its foot bears on the lid and the cap is far
    # below the 14.7 N strip weight). Lateral PD keeps the foot centred in the wide
    # opening while it rises through under real contact guidance.
    y_ap = (c.slot_y1 + c.ap_y1) / 2  # centre of the wide opening
    lift_extra, v_up = 0.0, 0.10
    best_z, last_gain = -1.0, 0
    done_lift = False
    for i in range(1800):
        sp, syaw = strip_pose()
        loc = blue_loc()
        z = float(loc[2])
        if z >= c.extract_z + 0.010:
            done_lift = True
            break
        dir_w = torch.tensor([-math.sin(syaw), math.cos(syaw), 0.0], device=device)
        lat_w = torch.tensor([math.cos(syaw), math.sin(syaw), 0.0], device=device)
        v_w = scene.plug_blue.data.root_lin_vel_w[0]
        fz = m_plug * (g + 30.0 * (v_up - float(v_w[2]))) + lift_extra
        fz = max(0.0, min(m_plug * g + 2.5, fz))
        ex = float(scene.blue_x0[0]) - float(loc[0])
        ey = y_ap - float(loc[1])
        v_lat = float((v_w * lat_w).sum())
        v_along = float((v_w * dir_w).sum())
        f_lat = max(-0.5, min(0.5, m_plug * (80.0 * ex - 20.0 * v_lat)))
        f_along = max(-0.5, min(0.5, m_plug * (80.0 * ey - 20.0 * v_along)))
        f_world = lat_w * f_lat + dir_w * f_along
        f_world = f_world + torch.tensor([0.0, 0.0, fz], device=device)
        apply_wrench(f_world, steady_torque())
        env.step(no_action)
        if z > best_z + 0.001:
            best_z, last_gain = z, i
        elif i - last_gain > 240:
            lift_extra = min(lift_extra + 0.3, 1.5)
            last_gain = i
            print(f"[solve] lift stalled at z={z:.4f}, lift_extra={lift_extra:.1f} N", flush=True)
    loc = blue_loc()
    print(f"[solve] lift loop done (done_lift={done_lift}, "
          f"local=({float(loc[0]):+.4f},{float(loc[1]):+.4f},{float(loc[2]):.4f}))", flush=True)
    if not done_lift:
        clear_wrench()
        print("SIM_GEN_SOLVE: FAIL (lift did not clear the lid)", flush=True)
        os._exit(1)
    s2 = print_score("P2 lift-out (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across the lift-out"

    # ---------------- phase 3: TRANSPORT (the one teleport) + gravity set-down -------------
    # The plug is now a free body in mid-air. Carry it (pose write, free space only)
    # to a lying hover 9 mm above the tray floor, then RELEASE: the set-down is a
    # real gravity drop onto the tray floor.
    clear_wrench()
    tq = scene.tray.data.root_quat_w[0]
    tyaw = 2.0 * math.atan2(float(tq[3]), float(tq[0]))
    tp = (scene.tray.data.root_pos_w - scene.env_origins)[0]
    span_mid = (c.plug_len / 2) - c.foot_h / 2  # origin -> mid-span offset along +axis
    c45 = math.cos(math.pi / 4)
    cy2, sy2 = math.cos(tyaw / 2), math.sin(tyaw / 2)
    axis_w = torch.tensor([math.cos(tyaw), math.sin(tyaw), 0.0], device=device)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(tp[0]) - span_mid * float(axis_w[0])
    st[:, 1] = float(tp[1]) - span_mid * float(axis_w[1])
    st[:, 2] = c.tray_floor_t + c.head_w / 2 + 0.009  # head side face 9 mm above the floor
    # lying: q = qz(tyaw) * qy(90 deg): plug local +z -> world horizontal
    st[:, 3] = cy2 * c45
    st[:, 4] = -sy2 * c45
    st[:, 5] = cy2 * c45
    st[:, 6] = sy2 * c45
    st[:, 0:3] += scene.env_origins
    scene.plug_blue.write_root_state_to_sim(st, all_ids)
    step(180)  # gravity drop + 1.5 s settle
    report("stowed")
    s3 = print_score("P3 transport + gravity set-down + settle")
    assert s3 >= s2 - 1e-6, "score decreased across the stow"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after stow+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.4 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.4 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.4 s")
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
