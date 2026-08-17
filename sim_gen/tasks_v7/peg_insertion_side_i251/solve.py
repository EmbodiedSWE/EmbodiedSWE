"""Teleport solution for QuarterLatchScene (sim_gen task `peg_insertion_side_i251`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, in two writes, each ending in FREE SPACE:
  P1 — the key is carried from its floor spawn to a hover in front of the porthole:
  tip 15 mm OUTSIDE the collar mouth, long axis on the bore line, blade rolled
  vertical to match the slot. Everything after that is CONTACT DYNAMICS.
  P4 — after the key has been withdrawn clear of the porthole under contact control,
  it is carried to a lying pose on the open floor (free space above the ground) and
  released.
The load-bearing interactions are all forces/torques a hand could apply to the KEY:
  P2 — guarded insertion: axial velocity servo + lateral PD toward the bore line +
  axis-alignment torque threads the blade through the collar into the dial's slot
  under real contact.
  P3 — the quarter turn: a ramped twist torque about the bore axis (bang-bang on the
  dial's measured rate) drives the DIAL through the key blade's contact with the slot
  cheeks, through the counterweight's over-center point; the torque is cut at ~84 deg
  and the counterweight itself carries the dial onto its 90-deg stop.
  P5 — withdrawal: axial velocity servo pulls the blade back out through the porthole.
The dial is NEVER teleported after reset; the key is never teleported into the
porthole or the slot.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.peg_insertion_side_i251.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.quarter_latch_drum")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_wrench() -> None:
        scene.key.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def h_frame() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        hp = scene.housing.data.root_pos_w[0]
        hq = scene.housing.data.root_quat_w[0]
        ax = quat_apply(hq.unsqueeze(0),
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        yaw = 2.0 * math.atan2(float(hq[3]), float(hq[0]))
        return hp, hq, ax, yaw

    def key_read() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = scene.key.data.root_pos_w[0]
        a = quat_apply(scene.key.data.root_quat_w, ez.expand(n, 3))[0]
        v = scene.key.data.root_lin_vel_w[0]
        w = scene.key.data.root_ang_vel_w[0]
        return p, a, v, w

    def theta_deg() -> float:
        return math.degrees(float(scene.dial_theta()[0]))

    def report(tag: str) -> None:
        _, tip, _ = scene._key_pts(scene.key)
        th = scene._h_local(tip)[0]
        kv = float(scene.key.data.root_lin_vel_w[0].norm())
        dv = float(scene.dial.data.root_lin_vel_w[0].norm())
        dw = float(scene.dial.data.root_ang_vel_w[0].norm())
        print(f"[solve] {tag:12s} | tip_h=({float(th[0]):+.3f},{float(th[1]):+.3f},"
              f"{float(th[2]):+.3f}) theta={theta_deg():+6.1f}deg "
              f"clear={bool(scene.key_clear()[0])} at_stop={bool(scene.dial_at_stop()[0])} "
              f"kset={bool(scene.key_settled()[0])}(|v|={kv:.3f}) "
              f"dset={bool(scene.dial_settled()[0])}(|v|={dv:.3f},|w|={dw:.3f}) "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def servo_step(v_des_ax: float, axial_bias: float, twist: float,
                   hp: torch.Tensor, ax: torch.Tensor) -> None:
        """One contact-control step on the KEY: axial velocity servo + lateral PD
        toward the bore line + gravity comp; axis-align torque (perpendicular only —
        the key's roll inertia is tiny) + optional twist torque about the bore axis."""
        m = c.key_mass
        p, a_k, v, w = key_read()
        v_ax = float(torch.dot(v, ax))
        f_ax = m * 25.0 * (v_des_ax - v_ax) + axial_bias
        f_ax = max(-2.0, min(2.0, f_ax))
        # lateral PD centres the TIP on the bore line (origin-centring leaves the tip
        # off-axis under a small load-tilt, and the blade then hooks the aperture jamb
        # on withdrawal — the blade has free play along the slot).
        _, tip_w, _ = scene._key_pts(scene.key)
        rel = tip_w[0] - hp
        e_perp = -(rel - torch.dot(rel, ax) * ax)
        v_perp = v - torch.dot(v, ax) * ax
        # stiff: w_n ~ 24.5 rad/s, zeta ~ 0.82 (must recenter the blade against
        # corner drag on the collar wall during withdrawal)
        f_lat = m * (600.0 * e_perp - 40.0 * v_perp)
        f_lat = f_lat.clamp(-3.0, 3.0)
        f = f_ax * ax + f_lat + m * 9.81 * ez
        # alignment PD (perpendicular only — the key's roll inertia is tiny):
        # I_perp ~ 1.4e-4, w_n = sqrt(0.04/I) ~ 17 rad/s, zeta ~ 0.85: near-critical.
        tq = 0.04 * torch.linalg.cross(a_k, -ax) - 0.004 * (w - torch.dot(w, a_k) * a_k)
        tq = tq.clamp(-0.08, 0.08)
        tq = tq + twist * ax
        scene.key.set_external_force_and_torque(
            f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)
        env.step(no_action)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    hp, hq, ax, hyaw = h_frame()
    hp_l = hp - scene.env_origins[0]
    k0 = (scene.key.data.root_pos_w - scene.env_origins)[0]
    de0 = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): housing=({float(hp_l[0]):+.3f},"
          f"{float(hp_l[1]):+.3f}) yaw={math.degrees(hyaw):+.1f}deg "
          f"key_spawn=({float(k0[0]):+.3f},{float(k0[1]):+.3f}) "
          f"decoy_spawn=({float(de0[0]):+.3f},{float(de0[1]):+.3f}) "
          f"d0={float(scene.d0[0]):.3f} theta0={theta_deg():+.1f}deg", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: key TRANSPORT (teleport to hover, free space) ---------------
    # Tip 15 mm OUTSIDE the collar mouth, on the bore line, blade rolled vertical.
    # Orientation: qz(housing yaw) * qy(-90): local +z -> -x_h, blade width -> world up.
    cy2, sy2 = math.cos(hyaw / 2), math.sin(hyaw / 2)
    c45 = math.cos(math.pi / 4)
    qk = (cy2 * c45, sy2 * c45, -cy2 * c45, sy2 * c45)  # qz(yaw)*qy(-90), wxyz
    org = hp + (c.mouth_x + 0.015 + c.key_half) * ax
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = org
    st[:, 3] = qk[0]
    st[:, 4] = qk[1]
    st[:, 5] = qk[2]
    st[:, 6] = qk[3]
    scene.key.write_root_state_to_sim(st, all_ids)
    for _ in range(30):  # actively-held hover: settle alignment before contact
        servo_step(0.0, 0.0, 0.0, hp, ax)
    report("key-hover")
    s1 = print_score("P1 key transport to hover")
    assert s1 >= s0 - 1e-6, "score decreased across key transport"

    # ---------------- phase 2: guarded insertion under contact -----------------------------
    best_x, last_gain, bias = 1.0, 0, 0.0
    tx = 1.0
    for i in range(2400):
        _, tip, _ = scene._key_pts(scene.key)
        th = scene._h_local(tip)[0]
        tx = float(th[0])
        if tx <= c.insert_tip_x + 0.001:
            break
        servo_step(-0.06, -bias, 0.0, hp, ax)
        if i % 150 == 149:
            _, a_k, _, _ = key_read()
            bx = quat_apply(scene.key.data.root_quat_w,
                            torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))[0]
            print(f"[solve] inserting: tip_h=({tx:+.3f},{float(th[1]):+.3f},"
                  f"{float(th[2]):+.3f}) axis_dot={float(torch.dot(a_k, -ax)):+.3f} "
                  f"roll_up_z={float(bx[2]):+.3f} bias={bias:.1f}", flush=True)
        if tx < best_x - 0.001:
            best_x, last_gain = tx, i
        elif i - last_gain > 240:  # stalled: lean in a little harder
            bias = min(bias + 0.3, 1.5)
            last_gain = i
            print(f"[solve] insertion stalled at tip_x={tx:.3f}, bias={bias:.1f} N",
                  flush=True)
    if tx > 0.005:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (insertion incomplete)", flush=True)
        os._exit(1)
    # HOLD the key seated (do not let go: unsupported, gravity pitches the key and
    # the slick slot cams the blade back out); score is read while held.
    for _ in range(30):
        servo_step(-0.01, -0.2, 0.0, hp, ax)
    report("inserted")
    s2 = print_score("P2 contact insertion")
    assert s2 >= s1 - 1e-6, "score decreased across insertion"

    # ---------------- phase 3: the quarter turn (ramped twist torque) ----------------------
    # Guards: never twist a disengaged blade (tip must be inside the slot) and never
    # feed torque into a free-spinning key (tiny roll inertia -> runaway).
    tau, reached = 0.02, False
    for i in range(3000):
        th = theta_deg()
        if th >= 84.0:
            reached = True
            break
        _, tip, _ = scene._key_pts(scene.key)
        tx = float(scene._h_local(tip)[0, 0])
        roll = float(torch.dot(scene.key.data.root_ang_vel_w[0], ax))
        w_par = float(torch.dot(scene.dial.data.root_ang_vel_w[0], ax))
        if w_par < 0.8:
            tau = min(tau * 1.06, 0.15)
        elif w_par > 1.8:
            tau = max(tau * 0.85, 0.002)
        t_cmd = tau if (tx <= -0.012 and abs(roll) <= 2.5) else 0.0
        servo_step(-0.02, -0.3, t_cmd, hp, ax)
        if i % 400 == 399:
            print(f"[solve] twisting: theta={th:+.1f}deg tau={tau:.3f} w={w_par:+.2f} "
                  f"tip_x={tx:+.3f} roll={roll:+.2f}", flush=True)
    # HOLD the key still (no twist) while the counterweight completes over-center
    # onto the 90-deg stop; letting go here would drag the dial via the slot.
    for _ in range(180):
        servo_step(0.0, 0.0, 0.0, hp, ax)
    report("turned")
    print(f"[solve] twist loop done (reached_84={reached}) theta={theta_deg():+.1f}deg",
          flush=True)
    s3 = print_score("P3 quarter turn + over-center settle")
    assert s3 >= s2 - 1e-6, "score decreased across the turn"
    if theta_deg() < c.success_lo_deg:
        print("SIM_GEN_SOLVE: FAIL (dial did not latch)", flush=True)
        os._exit(1)

    # ---------------- phase 4: withdrawal under contact (3 stages) -------------------------
    def pull(v_des: float, stop_x: float, iters: int, bias_cap: float) -> float:
        """Slow pull along +axis until the tip passes stop_x. A stiction-breaking bias
        may build up during a stall, but it is RESET the moment the key moves: a held
        bias yanks the freed blade out fast, yaws it, and wedges it in the collar."""
        best_out, last_gain, bias = -1.0, 0, 0.0
        tx = -1.0
        for i in range(iters):
            _, tip, _ = scene._key_pts(scene.key)
            th = scene._h_local(tip)[0]
            tx = float(th[0])
            if tx >= stop_x:
                break
            v_ax = float(torch.dot(scene.key.data.root_lin_vel_w[0], ax))
            if v_ax > 0.12:
                bias = 0.0
            servo_step(v_des, bias, 0.0, hp, ax)
            if tx > best_out + 0.001:
                best_out, last_gain = tx, i
            elif i - last_gain > 180:  # stalled: pull a little harder
                bias = min(bias + 0.3, bias_cap)
                last_gain = i
                print(f"[solve] withdrawal stalled at tip_h=({tx:+.3f},"
                      f"{float(th[1]):+.3f},{float(th[2]):+.3f}), bias={bias:.1f} N",
                      flush=True)
        return tx
    # stage A: grind the blade out of the slot (stick-slip against the cheek; keep the
    # bias gentle — a hard bias yaws the blade as it exits)
    tx = pull(+0.04, -0.009, 2000, 0.9)
    if tx < -0.009:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (withdrawal incomplete — will not teleport out of "
              "the mechanism)", flush=True)
        os._exit(1)
    # stage B/C: pull through the aperture + collar with drawer-jam recovery. A yawed
    # blade locks diagonally across the aperture window (both edges on opposite
    # jambs): first try the gentle recipe (back in, square up under the alignment
    # PD, re-pull at a low bias cap — frees the light jams without yanking), then
    # escalate the cap across attempts (a corner-hooked blade needs a harder grind;
    # the velocity-triggered bias reset in pull() prevents the freed-blade yank).
    done = False
    for attempt in range(10):
        for _ in range(75):  # square up: hold on-axis, alignment torque kills yaw
            servo_step(0.0, 0.0, 0.0, hp, ax)
        cap = 0.4 if attempt < 3 else (0.9 if attempt < 6 else 1.5)
        iters = 500 if attempt < 3 else 1200
        tx = pull(+0.04, c.mouth_x + 0.035, iters, cap)
        if tx >= c.mouth_x + 0.035:
            done = True
            break
        print(f"[solve] withdrawal jam at tip_x={tx:+.3f} (attempt {attempt}, "
              f"cap {cap:.1f} N): backing in to square up", flush=True)
        for _ in range(50):  # relieve the diagonal lock: push back in a touch
            servo_step(-0.04, 0.0, 0.0, hp, ax)
    if not done:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (withdrawal incomplete — will not teleport out of "
              "the mechanism)", flush=True)
        os._exit(1)
    clear_wrench()
    step(10)
    report("withdrawn")

    # ---------------- phase 5: key TRANSPORT (teleport to a clear floor pose) --------------
    # The key is in free air outside the porthole; carry it to open floor, lying flat,
    # 2 mm above the ground (free space), and let it settle.
    de_xy = scene.decoy.data.root_pos_w[0, :2]
    drop = None
    for off in ((0.30, 0.20), (0.30, -0.20), (0.42, 0.0)):
        cand = hp + off[0] * ax + off[1] * quat_apply(hq.unsqueeze(0), torch.tensor(
            [[0.0, 1.0, 0.0]], device=device))[0]
        if float((cand[:2] - de_xy).norm()) > 0.16:
            drop = cand
            break
    assert drop is not None, "no clear drop spot"
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = drop[:2]
    st[:, 2] = scene.env_origins[0, 2] + c.shaft_r + 0.0005  # 0.5 mm gap: no roll jolt
    yaw2 = hyaw + math.pi / 2
    cy, sy = math.cos(yaw2 / 2), math.sin(yaw2 / 2)
    st[:, 3] = cy * c45
    st[:, 4] = -cy * c45
    st[:, 5] = -sy * c45
    st[:, 6] = sy * c45
    scene.key.write_root_state_to_sim(st, all_ids)
    streak = 0  # streak-gated settle: wait for success to hold 30 consecutive steps
    for _ in range(900):
        env.step(no_action)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 30:
            break
    report("key-parked")
    s5 = print_score("P5 key transport to floor + settle")
    assert s5 >= s3 - 1e-6, "score decreased across withdrawal/parking"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after withdrawal)", flush=True)
        os._exit(1)

    # ---------------- phase 6: persistence (>= 3.3 simulated seconds, hands-off) -----------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s6 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s6 >= s5 - 1e-6
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
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {e})", flush=True)
        os._exit(1)
