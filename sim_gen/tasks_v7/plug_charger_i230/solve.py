"""Teleport solution for ShutterGarageScene (sim_gen task `plug_charger_i230`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY — exactly ONE pose write, and it happens while
the charger brick is a FREE body loose on the open floor (where it spawned), carrying
it across free space to a staging spot on the floor in FRONT of the (already opened)
doorway, prongs facing in. Every load-bearing interaction goes through CONTACT
DYNAMICS:
  P1 — OPEN THE SHUTTER: a floating-hand force controller (velocity-regulated push
  along the garage-local +x rail direction on the shutter, yaw/upright steadying
  torque, friction-stall escalation) slides the captive slab sideways in its real
  rails — floor ridge, hanging lip, facade — until the doorway is fully clear.
  P2 — TRANSPORT: the single teleport, spawn floor -> staging floor spot outside the
  doorway. The brick is never teleported through the doorway or into the bay.
  P3 — PUSH THROUGH THE DOORWAY: the same controller pushes the brick along the
  garage-local +y axis THROUGH the real doorway along the real floor (lateral PD
  centring it on the bay axis, steadying torque) until both brass prongs press the
  copper contact plate at the back of the bay; then it releases and lets everything
  settle. The docking is pure contact: the plate is what stops the brick.
  P4 — CLOSE THE SHUTTER: the controller slides the slab back until it is centred
  over the doorway again, sealing the brick inside; release, settle.
The doorway transit and both shutter slides — the three interactions the mechanism
exists to force — are pure contact physics.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.plug_charger_i230.solve --headless [--seed N]
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
    env = ENVS.get("simgen.shutter_garage")().build(num_envs=args.num_envs, device=device)
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

    def dock_yaw() -> float:
        q = scene.garage.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def dock_axes() -> tuple[torch.Tensor, torch.Tensor]:
        """(lat_w, in_w): garage-local +x (rail) and +y (inward) axes in world."""
        gy = dock_yaw()
        lat_w = torch.tensor([math.cos(gy), math.sin(gy), 0.0], device=device)
        in_w = torch.tensor([-math.sin(gy), math.cos(gy), 0.0], device=device)
        return lat_w, in_w

    def report(tag: str) -> None:
        sd = scene.shutter_dock()[0]
        bd = scene.brick_dock()[0]
        td = scene.tip_dock()[0]
        print(f"[solve] {tag:12s} | shutter_x={float(sd[0]):+.4f} "
              f"brick_dock=({float(bd[0]):+.3f},{float(bd[1]):+.3f},{float(bd[2]):.3f}) "
              f"tip_y={float(td[1]):+.4f} "
              f"in_rails={bool(scene.shutter_in_rails()[0])} "
              f"closed={bool(scene.shutter_closed()[0])} "
              f"docked={bool(scene.docked()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_wrench(body, f_world: torch.Tensor, tq_world: torch.Tensor) -> None:
        body.set_external_force_and_torque(
            f_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq_world.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)

    def steady_torque(body, k_up: float, k_yaw: float, clamp: float) -> torch.Tensor:
        """Upright + dock-yaw-follow steadying torque (what a hand does for free)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=device)
        axis = quat_apply(body.data.root_quat_w, ez.expand(n, 3))[0]
        w_w = body.data.root_ang_vel_w[0]
        q = body.data.root_quat_w[0]
        byaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        gy = dock_yaw()
        dyaw = math.atan2(math.sin(gy - byaw), math.cos(gy - byaw))
        tq = k_up * torch.linalg.cross(axis, ez) - 0.008 * w_w
        tq[2] += k_yaw * dyaw
        return tq.clamp(-clamp, clamp)

    def shutter_steer_torque() -> torch.Tensor:
        """Yaw PD holding the slab parallel to its rails (the anti-drawer-jam moment
        a hand on the knob provides for free) + mild upright steadying. A slab that
        yaws in the channel wedges corner-to-corner between facade and lip; the CoM
        push cannot prevent that, the hand's moment can."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=device)
        axis = quat_apply(scene.shutter.data.root_quat_w, ez.expand(n, 3))[0]
        w_w = scene.shutter.data.root_ang_vel_w[0]
        q = scene.shutter.data.root_quat_w[0]
        byaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
        gy = dock_yaw()
        dyaw = math.atan2(math.sin(gy - byaw), math.cos(gy - byaw))
        tq = 0.05 * torch.linalg.cross(axis, ez) - 0.01 * w_w
        tq[0] = tq[0].clamp(-0.06, 0.06)
        tq[1] = tq[1].clamp(-0.06, 0.06)
        tq[2] = max(-0.25, min(0.25, 0.5 * dyaw - 0.03 * float(w_w[2])))
        return tq

    def slide_shutter(x_tgt: float, tol: float, tag: str, max_iter: int = 2400) -> bool:
        """Velocity-regulated slide of the shutter along its rails to garage-local
        x = x_tgt (contact dynamics: the slab drags the floor, guided by ridge, lip
        and facade). Returns True when |x - x_tgt| < tol."""
        m = c.door_mass
        ff, v_max = 1.0, 0.10
        best_err, last_gain = 1e9, 0
        done = False
        for i in range(max_iter):
            lat_w, in_w = dock_axes()
            loc = scene.shutter_dock()[0]
            x = float(loc[0])
            err = x_tgt - x
            if abs(err) < tol:
                done = True
                break
            v_des = max(-v_max, min(v_max, 2.0 * err))
            v_w = scene.shutter.data.root_lin_vel_w[0]
            v_along = float((v_w * lat_w).sum())
            f_along = m * 50.0 * (v_des - v_along) + ff * (1.0 if v_des > 0 else -1.0)
            f_along = max(-6.0, min(6.0, f_along))
            # y-PD toward the channel line — what the hand on the knob provides for
            # free; keeps the slab's bottom corner out of the ridge band so it never
            # snags a ridge-segment end face at the doorway gap.
            ey = c.door_yc - float(loc[1])
            v_in = float((v_w * in_w).sum())
            f_in = max(-1.5, min(1.5, m * (150.0 * ey - 25.0 * v_in)))
            apply_wrench(scene.shutter, lat_w * f_along + in_w * f_in,
                         shutter_steer_torque())
            env.step(no_action)
            if abs(err) < best_err - 0.001:
                best_err, last_gain = abs(err), i
            elif i - last_gain > 240:  # friction stall: lean harder
                ff = min(ff + 0.5, 5.0)
                last_gain = i
                print(f"[solve] {tag} stalled at x={x:+.4f}, ff={ff:.1f} N", flush=True)
        clear_wrench(scene.shutter)
        step(30)
        print(f"[solve] {tag} done (done={done}, x={float(scene.shutter_dock()[0, 0]):+.4f})",
              flush=True)
        return done

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    gp = (scene.garage.data.root_pos_w - scene.env_origins)[0]
    bd = scene.brick_dock()[0]
    q = scene.brick.data.root_quat_w[0]
    byaw = 2.0 * math.atan2(float(q[3]), float(q[0]))
    print(f"[solve] layout readback (seed {args.seed}): garage=({float(gp[0]):+.3f},"
          f"{float(gp[1]):+.3f}) yaw={math.degrees(dock_yaw()):+.1f}deg "
          f"brick_dock=({float(bd[0]):+.3f},{float(bd[1]):+.3f}) "
          f"brick_yaw={math.degrees(byaw):+.1f}deg", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: OPEN THE SHUTTER (contact dynamics) -------------------------
    # Slide the captive slab sideways in its real rails until the doorway is fully
    # clear. Open AWAY from the brick's side so the staging spot stays uncluttered.
    open_sign = -1.0 if float(bd[0]) > 0.0 else 1.0
    x_open = open_sign * (c.x_door_clear + 0.003)
    ok_open = slide_shutter(x_open, 0.0015, "open slide")
    report("opened")
    if not ok_open and abs(float(scene.shutter_dock()[0, 0])) < c.x_door_clear:
        print("SIM_GEN_SOLVE: FAIL (shutter did not clear the doorway)", flush=True)
        os._exit(1)
    s1 = print_score("P1 shutter opened (contact)")
    assert s1 >= s0 - 1e-6, "score decreased across the opening slide"

    # ---------------- phase 2: TRANSPORT (the one teleport) --------------------------------
    # The brick is a free body loose on the open floor. Carry it (pose write, free
    # space only) to a staging spot on the floor OUTSIDE the doorway, prongs facing
    # in. It is NOT teleported through the doorway: its prong tips stay ~47 mm in
    # front of the facade's outer plane.
    gy = dock_yaw()
    lat_w, in_w = dock_axes()
    gpos = scene.garage.data.root_pos_w[0] - scene.env_origins[0]
    stage_y = -0.10  # dock frame: tip at -0.047 < 0 (outside)
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = float(gpos[0]) + stage_y * float(in_w[0])
    st[:, 1] = float(gpos[1]) + stage_y * float(in_w[1])
    st[:, 2] = c.brick_h / 2 + 0.0015
    st[:, 3] = math.cos(gy / 2)
    st[:, 6] = math.sin(gy / 2)
    st[:, 0:3] += scene.env_origins
    scene.brick.write_root_state_to_sim(st, all_ids)
    step(60)
    report("staged")
    s2 = print_score("P2 transport to the doorway apron")
    assert s2 >= s1 - 1e-6, "score decreased across the transport"

    # ---------------- phase 3: PUSH THROUGH THE DOORWAY (contact dynamics) -----------------
    # Velocity-regulated push along the garage-local inward axis: the brick slides
    # the real floor, through the real doorway, into the bay, until both prongs
    # press the copper plate. Lateral PD centres it on the bay axis; the plate is
    # what stops it.
    m_b = c.brick_mass
    push_ff, v_des = 0.5, 0.08
    tip_goal = c.plate_y - 0.0005  # push until the PLATE stops the brick
    best_y, last_gain = -1.0, 0
    done_push = False
    for i in range(2400):
        lat_w, in_w = dock_axes()
        td = scene.tip_dock()[0]
        y = float(td[1])
        if y >= tip_goal:
            done_push = True
            break
        v_w = scene.brick.data.root_lin_vel_w[0]
        v_in = float((v_w * in_w).sum())
        f_in = max(0.0, min(4.0, push_ff + m_b * 60.0 * (v_des - v_in)))
        bx = float(scene.brick_dock()[0, 0])
        v_lat = float((v_w * lat_w).sum())
        f_lat = max(-1.0, min(1.0, m_b * (80.0 * (0.0 - bx) - 20.0 * v_lat)))
        apply_wrench(scene.brick, in_w * f_in + lat_w * f_lat,
                     steady_torque(scene.brick, 0.05, 0.02, 0.05))
        env.step(no_action)
        if y > best_y + 0.001:
            best_y, last_gain = y, i
        elif i - last_gain > 240:  # stalled
            if y >= c.y_seat_min + 0.001:
                # stalled INSIDE the seat band while still pressing forward: the
                # copper plate is what is stopping the brick — docked by contact.
                print(f"[solve] plate stop reached (stall at tip_y={y:+.4f} "
                      f"under forward press)", flush=True)
                done_push = True
                break
            push_ff = min(push_ff + 0.4, 3.0)
            last_gain = i
            print(f"[solve] push stalled at tip_y={y:+.4f}, push_ff={push_ff:.1f} N",
                  flush=True)
    clear_wrench(scene.brick)
    step(90)  # release + settle: the dock latch needs a SETTLED docked brick
    report("docked")
    print(f"[solve] push loop done (done_push={done_push}, "
          f"tip_y={float(scene.tip_dock()[0, 1]):+.4f})", flush=True)
    if not bool(scene.docked()[0]):
        print("SIM_GEN_SOLVE: FAIL (brick did not dock on the plate)", flush=True)
        os._exit(1)
    s3 = print_score("P3 doorway push + dock (contact)")
    assert s3 >= s2 - 1e-6, "score decreased across the doorway push"

    # ---------------- phase 4: CLOSE THE SHUTTER (contact dynamics) ------------------------
    # Slide the slab back until it is centred over the doorway again. The docked
    # brick sits entirely behind the facade, clear of the swept lane.
    ok_close = slide_shutter(0.0, 0.006, "close slide")
    step(60)
    report("closed")
    if not (ok_close and bool(scene.shutter_closed()[0])):
        print("SIM_GEN_SOLVE: FAIL (shutter did not re-close)", flush=True)
        os._exit(1)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after closing)", flush=True)
        os._exit(1)
    s4 = print_score("P4 shutter closed (contact) -> success")
    assert s4 >= s3 - 1e-6, "score decreased across the closing slide"

    # ---------------- phase 5: persistence (>= 3.4 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.4 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.4 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
