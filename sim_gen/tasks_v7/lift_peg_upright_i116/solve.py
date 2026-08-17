"""Teleport solution for HoodPropScene (sim_gen task `lift_peg_upright_i116`) — the
task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: the long red peg is teleported once, from the
floor to a rest pose 3 mm above the socket collar cavity (exactly what a pick-and-
carry delivers), and released. EVERYTHING the rubric reads happens through contact
dynamics and applied wrenches:

1. OPEN (episodes that start closed): the lid is driven about its contact hinge by an
   external torque along the hinge axis — a gravity feedforward (m*g*com_x*cos(theta))
   plus a velocity servo on a finite-difference hinge rate (root_ang_vel readback is
   phantom under external wrenches). Torque is cut past ~96 deg and the lid falls
   back onto the BACK-STOP by gravity alone (the temporary hold). Episodes that start
   parked open skip this phase (and get no `opened` credit — the scene latches it
   only for closed starts).
2. PLACE: the peg is teleported above the socket and DROPPED; it seats in the collar
   under gravity.
3. LOWER: the lid is pulled off the stop with a closing torque and lowered
   quasi-statically; the feedforward carries most of the weight, so the underside
   meets the peg top at ~0.2 rad/s. Contact is detected from the FD rate (theta
   stops falling inside the expected window), then the feedforward is RAMPED to zero
   over 0.75 s so the lid's weight transfers onto the peg gently, and the wrench is
   cleared. The propped rest inside [band_min, band_max] is pure contact statics.

Pod force-frame quirk: a hinge-axis torque is invariant under the documented
rotation-since-reset drag (the drag rotation IS about the hinge axis), but the mode
is still guarded by a runtime progress probe that flips to a body-frame pre-encoding
if theta ever moves against the commanded rate.

Prints `SIM_GEN_SCORE <score>` at each settled phase boundary (non-decreasing: the
scene's stage credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.lift_peg_upright_i116.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

_qapply = scene_mod._qapply
_qmul = scene_mod._qmul

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

_DT = 1.0 / 120.0
_I_HINGE = 0.00376 + 0.5 * 0.152 ** 2  # lid inertia about the hinge axis (~0.01535)
_TAU_G = 0.5 * 9.81 * 0.152  # gravity feedforward amplitude m*g*com_x (~0.746 N*m)
_K_W = 0.5  # velocity-servo gain (K*dt/I ~ 0.27 << 1: stable under the 1-step delay)
_TAU_MAX = 2.0


def _qinv(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hood_prop")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def theta() -> float:
        return float(scene.lid_angle()[0])

    def clear_wrench() -> None:
        scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                env_ids=all_ids)

    def hinge_axis_w() -> torch.Tensor:
        """(3,) world hinge axis such that POSITIVE torque about it OPENS the lid."""
        _ex, ey = scene._chest_axes()
        return -ey[0]

    def report(tag: str) -> None:
        th = math.degrees(theta())
        lv = float(scene.lid.data.root_lin_vel_w[0].norm())
        av = float(scene.lid.data.root_ang_vel_w[0].norm())
        print(f"[solve] {tag:14s} | theta={th:+7.2f} deg hinge_ok={bool(scene.hinge_ok()[0])} "
              f"pegged={bool(scene.peg_socketed()[0])} in_band={bool(scene.in_band()[0])} "
              f"lid(lin={lv:.3f},ang={av:.3f}) | success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # --- hinge torque plant -------------------------------------------------------------
    mode = 0  # 0: raw world torque; 1: body-frame pre-encode (probed, see module doc)
    th_prev = [0.0]

    def apply_tau(tau: float) -> None:
        t_world = torch.zeros(n, 3, device=device)
        t_world[0] = hinge_axis_w() * tau
        if mode == 1:
            t_world = _qapply(_qinv(scene.lid.data.root_quat_w), t_world)
        scene.lid.set_external_force_and_torque(zero_wrench, t_world.view(n, 1, 3),
                                                env_ids=all_ids, is_global=True)

    def drive(w_des_of_theta, stop_fn, tag: str, max_steps: int) -> bool:
        """Servo the lid's hinge rate: tau = g-feedforward + K*(w_des - w_fd), with a
        wrong-frame progress probe. Returns True when stop_fn(theta) fires."""
        nonlocal mode
        th_prev[0] = theta()
        anchor_i, anchor_th = 0, th_prev[0]
        for i in range(max_steps):
            th = theta()
            w_fd = (th - th_prev[0]) / _DT
            th_prev[0] = th
            if stop_fn(th):
                return True
            w_des = w_des_of_theta(th)
            tau = _TAU_G * math.cos(th) + _K_W * (w_des - w_fd)
            tau = max(-_TAU_MAX, min(_TAU_MAX, tau))
            apply_tau(tau)
            env.step(no_action)
            if i - anchor_i >= 30:
                prog = (theta() - anchor_th) * (1.0 if w_des > 0 else -1.0)
                if prog < -0.025:  # moved against the command by > ~1.4 deg
                    mode = 1 - mode
                    print(f"[solve] {tag}: theta moving against command "
                          f"(prog {prog:+.4f}); torque-frame mode -> {mode}", flush=True)
                anchor_i, anchor_th = i, theta()
        return False

    def wait_lid_settled(max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.lid_settled()[0]):
                break

    # ---------------- phase 0: reset, settle, baseline ----------------------------------
    step(240)  # hands-off: closed lids seat on the rim, open starts fall onto the stop
    closed_start = bool(scene.started_closed[0])
    cp = (scene.chest.data.root_pos_w - scene.env_origins)[0]
    q = scene.chest.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))
    print(f"[solve] layout readback (seed {args.seed}): chest=({float(cp[0]):+.3f},"
          f"{float(cp[1]):+.3f}) yaw={yaw:+.1f} deg | lid start "
          f"{'CLOSED' if closed_start else 'OPEN (on back-stop)'} "
          f"theta={math.degrees(theta()):+.2f} deg", flush=True)
    for name, b in (("peg_long", scene.peg_long), ("peg_short", scene.peg_short)):
        p = (b.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback (seed {args.seed}): {name} at "
              f"({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})", flush=True)
    report("reset")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s_prev = print_score("P0 reset+settle")
    assert s_prev <= 0.03, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: swing the lid past vertical onto the back-stop -----------
    if closed_start:
        ok = drive(lambda th: 1.2, lambda th: th >= math.radians(96.0),
                   "open", max_steps=1200)
        assert ok, "lid never reached 96 deg during the open swing"
        clear_wrench()
        # gravity now pulls the lid BACK onto the stop (CoM behind the hinge)
        wait_lid_settled(600)
        report("P1-opened")
        th_deg = math.degrees(theta())
        assert 97.0 <= th_deg <= 112.0, \
            f"lid should rest on the back-stop (~{c.stop_deg:.0f} deg), got {th_deg:.1f}"
        s_now = print_score("P1 lid swung past vertical, resting on the back-stop")
        assert s_now >= s_prev - 1e-6, "score decreased across the open swing"
        s_prev = s_now
    else:
        print("[solve] lid already parked on the back-stop — skipping the open phase "
              "(no opened-credit on open starts)", flush=True)

    # ---------------- phase 2: stand the long peg in the socket (drop, not press) -------
    def teleport_peg() -> None:
        """TRANSPORT ONLY: hold the peg upright with its foot 3 mm above the collar
        cavity, zero velocity, then release — it seats under gravity."""
        ex, _ey = scene._chest_axes()
        pos = scene.chest.data.root_pos_w + ex * c.sock_x
        pos[:, 2] += 0.012 + c.peg_l_long / 2 + 0.003
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = scene.chest.data.root_quat_w
        scene.peg_long.write_root_state_to_sim(st, all_ids)

    seated = False
    for attempt in range(3):
        teleport_peg()
        step(180)  # 1.5 s: drop 3 mm, seat, ring out
        if bool(scene.peg_socketed()[0]) and bool(scene.peg_settled(scene.peg_long)[0]):
            seated = True
            break
        print(f"[solve] peg did not seat (attempt {attempt}) — re-dropping", flush=True)
    assert seated, "long peg failed to seat upright in the socket"
    report("P2-pegged")
    s_now = print_score("P2 long peg dropped into the socket, standing")
    assert s_now >= s_prev - 1e-6, "score decreased across the peg drop"
    s_prev = s_now

    # ---------------- phase 3: lower the lid onto the peg -------------------------------
    def w_des_lower(th: float) -> float:
        if th > math.radians(80.0):
            return -0.8
        if th > math.radians(65.0):
            return -0.4
        return -0.2

    # contact detection: theta stops falling inside the expected prop window
    hist: list[float] = []
    contact = [False]

    def stop_on_contact(th: float) -> bool:
        hist.append(th)
        if th < math.radians(50.0):  # sailed past the prop — peg must have failed
            return True
        if len(hist) >= 25 and th < math.radians(66.0) \
                and abs(hist[-1] - hist[-25]) < 0.004:
            contact[0] = True
            return True
        return False

    drive(w_des_lower, stop_on_contact, "lower", max_steps=1500)
    th_deg = math.degrees(theta())
    print(f"[solve] lowering stopped at theta={th_deg:.2f} deg "
          f"(contact={contact[0]}, expect ~{c.prop_deg_long:.1f})", flush=True)
    assert contact[0], f"no prop contact detected (theta={th_deg:.2f} deg)"

    # weight hand-off: ramp the feedforward to zero over 0.75 s (quasi-static release)
    for k in range(90):
        apply_tau(_TAU_G * math.cos(theta()) * (1.0 - (k + 1) / 90.0))
        env.step(no_action)
    clear_wrench()
    step(300)  # 2.5 s hands-off settle on the prop
    report("P3-propped")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after lowering onto the peg)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)
    s_now = print_score("P3 lid resting on the peg inside the band")
    assert s_now >= s_prev - 1e-6, "score decreased across the lowering"
    s_prev = s_now

    # ---------------- persistence (>= 3 simulated seconds, no intervention) -------------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                lv = float(scene.lid.data.root_lin_vel_w[0].norm())
                av = float(scene.lid.data.root_ang_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: "
                      f"theta={math.degrees(theta()):+.2f} "
                      f"in_band={bool(scene.in_band()[0])} "
                      f"hinge_ok={bool(scene.hinge_ok()[0])} "
                      f"pegged={bool(scene.peg_socketed()[0])} "
                      f"lid(lin={lv:.4f},ang={av:.4f}) "
                      f"peg_settled={bool(scene.peg_settled(scene.peg_long)[0])}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
