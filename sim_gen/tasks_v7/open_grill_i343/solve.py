"""Teleport SOLUTION for ShiftParkGrillScene — NullRobot, contact-driven end to end.

Phases (SIM_GEN_SCORE printed at each boundary, non-decreasing):
  P0  reset + settle: lid forward-closed under the catch, patty on the board; 0.
  P1  SLIDE BACK (applied force): a velocity-regulated CoM force along the grill's
      -x (force-frame mode probed from progress) drags the lid its full slide to
      the rear stop; `released` latches (0.15).
  P2  RAISE (applied vertical force): a world-vertical CoM force above the lid's
      weight, rate-servoed on the pitch readback, swings the lid up to ~82 deg at
      the rear (only there does the sweep clear the rest bar); `raised` (0.35).
  P3  SLIDE FORWARD + RELEASE: holding pitch with the vertical servo, a +x force
      slides the raised lid forward to the stop; all forces then ramp out and the
      lid falls ~20 deg onto the REST BAR, creeping back until the underside cleat
      seats — parked leaning at ~60 deg, hands off; `parked` (0.65).
  P4  DELIVER (teleport = transport only): the patty is teleported to a hover point
      above the open mouth (a live-computed corridor clear of the leaning plate)
      and DROPPED; gravity carries it through the mouth onto the grate. success()
      must hold (1.00).
  P5  persistence: >= 3.3 simulated seconds hands-off; success holds throughout.
      Then print `SIM_GEN_SOLVE: SUCCESS`.

The teleport moves the patty through free air only (board -> hover above the open
mouth); every rubric fact (released / raised / parked / on-grate) is produced by
applied forces, joint limits, gravity and contacts.

Run: python -m simgen_tasks.open_grill_i343.solve --headless [--seed N]
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

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qapply, encode_force = task_scene._qapply, task_scene.encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shift_park_grill")().build(num_envs=args.num_envs,
                                                      device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    ids = torch.arange(n, device=device)
    no_action = torch.empty(0, device=device)
    zero = torch.zeros(n, 1, 3, device=device)

    def step(k: int = 1) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.lid.set_external_force_and_torque(zero, zero, env_ids=ids)

    last_score = -1.0

    def print_score(tag: str) -> None:
        nonlocal last_score
        s = float(scene.score()[0])
        assert s >= last_score - 1e-6, f"score regressed at {tag}: {last_score} -> {s}"
        last_score = max(last_score, s)
        print(f"SIM_GEN_SCORE {s:.2f}", flush=True)

    def report(tag: str) -> None:
        p = scene.patty_local()[0]
        print(f"[solve] {tag:12s} | slide={float(scene.slide()[0]) * 1000:6.1f}mm "
              f"pitch={math.degrees(float(scene.pitch()[0])):+6.1f}deg "
              f"patty=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"settled={bool(scene.settled()[0])} "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    def apply_world_force(mode, q_ref, f_world) -> None:
        f_arg = encode_force(mode, q_ref, scene.lid.data.root_quat_w, f_world)
        scene.lid.set_external_force_and_torque(
            f_arg.view(n, 1, 3), zero, env_ids=ids, is_global=True)

    # ---------------- P0: reset + settle --------------------------------------------------------
    env.reset(seed=args.seed)
    step(90)
    report("P0 settle")
    sl0 = float(scene.slide()[0])
    th0 = math.degrees(float(scene.pitch()[0]))
    assert sl0 > c.slide_s - c.slide_jitter - 0.006, f"lid not forward at spawn ({sl0:.3f})"
    assert abs(th0) < 3.0, f"lid not closed at spawn ({th0:.1f} deg)"
    p0 = scene.patty_local()[0]
    assert c.board_y0 - 0.05 < float(p0[1]) < c.board_y1 + 0.05, "patty not on the board"
    assert float(p0[2]) > c.board_h - 0.02, "patty fell off the board"
    assert float(scene.score()[0]) <= 0.01, "nonzero score at spawn"
    print_score("P0")

    # grill-frame push directions in the world (grill is heavy: read once)
    q_g = scene.grill.data.root_quat_w.clone()
    ex = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3)
    u_fwd3 = _qapply(q_g, ex)
    u_fwd = torch.zeros(n, 3, device=device)
    u_fwd[:, :2] = u_fwd3[:, :2] / u_fwd3[0, :2].norm().clamp(min=1e-6)
    up = torch.zeros(n, 3, device=device)
    up[:, 2] = 1.0

    # ---------------- P1: SLIDE BACK (velocity-regulated CoM force) -----------------------------
    q_ref = scene.lid.data.root_quat_w.clone()
    mode = 0
    floor_f = 0.5
    win_i, win_sl = 0, float(scene.slide()[0])
    reached = False
    for i in range(900):
        sl = float(scene.slide()[0])
        if sl <= 0.004:
            reached = True
            break
        v = scene.lid.data.root_lin_vel_w[0, :3]
        v_along = float((v * u_fwd[0]).sum())  # + = forward
        f_mag = 8.0 * (-0.15 - v_along)        # want v_along = -0.15 m/s
        f_mag = min(max(f_mag, -3.0), 3.0)
        if abs(v_along) < 0.01 and abs(f_mag) < floor_f:
            f_mag = -floor_f
        apply_world_force(mode, q_ref, u_fwd * f_mag)
        step(1)
        if i - win_i >= 30:
            sl2 = float(scene.slide()[0])
            if sl2 > win_sl + 0.003:
                mode = 1 - mode
                print(f"[solve] slide moving forward; force-frame mode -> {mode}",
                      flush=True)
            elif sl2 > win_sl - 0.003:
                floor_f = min(floor_f + 0.5, 3.0)
                print(f"[solve] slide stalled at {sl2 * 1000:.1f}mm; floor -> "
                      f"{floor_f:.1f} N", flush=True)
            win_i, win_sl = i, sl2
    clear_forces()
    step(30)
    assert reached, f"slide-back never reached the rear stop ({float(scene.slide()[0]):.3f})"
    assert float(scene.slide()[0]) <= c.slide_rear_max, "lid crept off the rear stop"
    report("P1 back")
    print_score("P1")

    # ---------------- P2: RAISE (vertical force, rate-servoed on pitch) -------------------------
    m_lid = c.lid_mass
    g = 9.81
    L_c = c.lid_com_x
    th_ref = math.radians(82.0)
    th_prev = float(scene.pitch()[0])
    raised = False
    for i in range(900):
        th = float(scene.pitch()[0])
        if th >= math.radians(80.0):
            raised = True
            break
        w = (th - th_prev) * 120.0
        th_prev = th
        w_des = min(max(1.8 * (th_ref - th), 0.0), 1.2)
        lever = L_c * max(math.cos(th), 0.15)
        f_up = m_lid * g + 0.35 * (w_des - w) / lever
        f_up = min(max(f_up, 0.0), 4.0 * m_lid * g)
        apply_world_force(mode, q_ref, up * f_up + u_fwd * (-0.3))
        step(1)
    assert raised, f"raise stalled at {math.degrees(float(scene.pitch()[0])):.1f} deg"
    assert float(scene.slide()[0]) <= c.slide_rear_max + 0.01, "slide drifted during raise"
    report("P2 raised")
    print_score("P2")

    # ---------------- P3: SLIDE FORWARD raised, then RELEASE onto the bar -----------------------
    th_prev = float(scene.pitch()[0])
    floor_x = 0.8
    win_i, win_sl = 0, float(scene.slide()[0])
    fwd_done = False
    for i in range(900):
        sl = float(scene.slide()[0])
        if sl >= c.slide_s - 0.004:
            fwd_done = True
            break
        th = float(scene.pitch()[0])
        w = (th - th_prev) * 120.0
        th_prev = th
        v = scene.lid.data.root_lin_vel_w[0, :3]
        v_along = float((v * u_fwd[0]).sum())
        f_x = 8.0 * (0.06 - v_along)
        if abs(v_along) < 0.02 and f_x < floor_x:
            f_x = floor_x
        f_x = min(max(f_x, -2.0), 2.5)
        # hold pitch: exact gravity + slide-force closing-torque feedforward + PD trim
        w_des = min(max(1.8 * (th_ref - th), -0.8), 0.8)
        lever = L_c * max(math.cos(th), 0.15)
        f_up = m_lid * g + f_x * math.tan(min(max(th, 0.0), math.radians(84.0))) \
            + 0.35 * (w_des - w) / lever
        f_up = min(max(f_up, 0.0), 6.0 * m_lid * g)
        apply_world_force(mode, q_ref, up * f_up + u_fwd * f_x)
        step(1)
        if i - win_i >= 30:
            sl2 = float(scene.slide()[0])
            if sl2 < win_sl - 0.003:
                mode = 1 - mode
                print(f"[solve] fwd slide moving backward; force-frame mode -> {mode}",
                      flush=True)
            elif sl2 < win_sl + 0.003:
                if floor_x >= 2.9:
                    # maxed-out authority and still frozen: the force frame must be
                    # dragging with the (now large) pitch — flip the encoding.
                    mode = 1 - mode
                    floor_x = 1.0
                    print(f"[solve] fwd slide frozen at max floor; force-frame mode "
                          f"-> {mode}", flush=True)
                else:
                    floor_x = min(floor_x + 0.5, 3.0)
                    print(f"[solve] fwd slide stalled at {sl2 * 1000:.1f}mm "
                          f"(v_along={v_along:+.3f}, pitch="
                          f"{math.degrees(th):.1f}deg); floor -> {floor_x:.1f} N",
                          flush=True)
            win_i, win_sl = i, sl2
    assert fwd_done, f"forward slide stalled at {float(scene.slide()[0]) * 1000:.1f}mm"
    print(f"[solve] fwd stop reached; pitch={math.degrees(float(scene.pitch()[0])):.1f}deg "
          f"(mode {mode})", flush=True)
    # re-raise / steady the pitch at the forward stop before letting go: the lid must
    # fall ONTO the rest bar from above (~82 deg), not slip under it.
    th_prev = float(scene.pitch()[0])
    for i in range(240):
        th = float(scene.pitch()[0])
        w = (th - th_prev) * 120.0
        th_prev = th
        if th >= math.radians(78.0) and abs(w) < 0.3 and i > 30:
            break
        w_des = min(max(1.8 * (th_ref - th), -0.8), 1.2)
        lever = L_c * max(math.cos(th), 0.15)
        f_up = m_lid * g + 0.35 * (w_des - w) / lever
        f_up = min(max(f_up, 0.0), 6.0 * m_lid * g)
        apply_world_force(mode, q_ref, up * f_up + u_fwd * 0.6)
        step(1)
    th_rel = math.degrees(float(scene.pitch()[0]))
    assert th_rel > 66.0, f"pitch too low to catch the bar at release ({th_rel:.1f} deg)"
    # ramp the lift out over 0.5 s so the lid lowers onto the bar without a slam
    for i in range(60):
        f_up = m_lid * g * (1.0 - (i + 1) / 60.0)
        apply_world_force(mode, q_ref, up * f_up + u_fwd * (0.4 * (1.0 - (i + 1) / 60.0)))
        step(1)
    clear_forces()
    for i in range(600):
        step(1)
        if i > 90 and bool(scene.settled()[0]):
            break
    step(60)
    report("P3 parked")
    th = math.degrees(float(scene.pitch()[0]))
    sl = float(scene.slide()[0])
    assert c.park_lo_deg < th < c.park_hi_deg, f"lid not in the park band ({th:.1f} deg)"
    assert sl >= c.slide_fwd_min, f"park slide not forward ({sl * 1000:.1f}mm)"
    assert bool(scene.parked()[0]) and bool(scene.settled()[0]), "park not settled"
    print_score("P3")

    # ---------------- P4: DELIVER the patty through the open mouth (teleport = transport) -------
    th_r = float(scene.pitch()[0])
    plate_max_x = (c.pin_x_rear + float(scene.slide()[0])) \
        + c.plate_x1 * math.cos(th_r) + (c.lid_t / 2) * math.sin(th_r) + 0.004
    x_drop = plate_max_x + c.patty_w / 2 + 0.015
    x_hi = c.hx - c.wall_t - c.patty_w / 2 - 0.008
    assert x_drop <= x_hi, f"no drop corridor ({x_drop:.3f} > {x_hi:.3f})"
    x_drop = min(x_drop + 0.010, x_hi)  # centre the column in the free span
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0] = x_drop
    loc[:, 2] = 0.30
    q_gnow = scene.grill.data.root_quat_w
    pos_w = scene.grill.data.root_pos_w + _qapply(q_gnow, loc)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = pos_w
    st[:, 3:7] = q_gnow
    scene.patty.write_root_state_to_sim(st, ids)
    print(f"[solve] patty dropped at grill-x {x_drop * 1000:.1f}mm", flush=True)
    for i in range(480):
        step(1)
        if i > 60 and bool(scene.settled()[0]):
            break
    step(60)
    report("P4 deliver")
    assert bool(scene.patty_on_grate()[0]), "patty not resting on the grate"
    assert bool(scene.success()[0]), "success() does not hold after delivery"
    print_score("P4")

    # ---------------- P5: persistence ------------------------------------------------------------
    for k in range(10):
        step(40)
        assert bool(scene.success()[0]), f"success dropped in persistence block {k}"
    report("P5 persist")
    print_score("P5")
    print("SIM_GEN_SOLVE: SUCCESS", flush=True)

    t = threading.Timer(10.0, lambda: os._exit(0))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
