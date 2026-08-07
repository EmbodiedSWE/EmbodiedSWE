"""Teleport solution for PistonJarScene (sim_gen task `open_jar_i65`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. CARRY (teleport = transport only): one rigid translation carries the SEALED jar
   assembly (jar + captive piston + welded lid + riding ball, all at their current
   relative offsets, orientations untouched, velocities zeroed) to a hover with the
   jar's bottom 5 mm ABOVE the spike tip, xy on the spike axis. The write satisfies
   nothing: the tip is still outside the jar (`engaged` false), the seal intact.
2. GUARDED LOWER (applied wrench = the gripper surrogate): the jar is held by a
   velocity-regulated vertical force (support feedforward ~5.7 N = stack weight,
   press term capped at 3 N so the intact seal never sees more than a fraction of
   its 10 N shear threshold), an xy spring to the LIVE spike axis, and an attitude
   PD to the spawn orientation (the sleeved jar is an inverted pendulum — it must
   be held, exactly like a Franka wrist holds it). The spike passes through the
   bottom hole and lifts the internal piston until the ball presses the welded lid
   — the stack STALLS with the seal intact. Engagement is produced by contact.
3. PRESS TO SHEAR (applied force does the opening): the same servo with the press
   term released to 25 N (escalating on stall) drives the jar down the spike; the
   piston->ball->lid chain loads the weld past its 10 N breakForce, the seal
   SHEARS (a PhysX breakable joint — irreversible), the rising ball shoves the lid
   out of the collar pocket and its offset centre of mass tips it off the rim,
   and the jar bottoms out on the pedestal base with the ball PRESENTED above the
   rim. If the popped lid balances covering the mouth, small lateral force pulses
   on the (now free) lid shove it off — still pure applied force, judged by pose.
4. DELIVERY (teleport = transport only; containment by gravity): the freed ball is
   carried to a hover 75 mm above the dish centre — ABOVE `dish_z_hi`, so the
   write does not satisfy `ball_in_dish` — and dropped; gravity and the fence
   walls produce containment. Mis-drops are re-dropped.

Wrenches are pre-encoded with R_ref * R_now^T against the pod's force-frame drag
(applied wrenches are rotated by the body's rotation-since-reset); the attitude
hold keeps the jar's drag ~identity anyway.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it holds.

Run (forge): python -u -m simgen_tasks.open_jar_i65.solve --headless [--seed N]
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
    from .scene import _qapply, _qinv, _qmul
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, _qinv, _qmul

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- controller gains --------------------------------------------------------------------------
W_FF = 9.81 * (0.45 + 0.06 + 0.04 + 0.03)  # the held stack: jar + piston + lid + ball (N)
KZ = 40.0                                  # vertical velocity gain (N per m/s); < 2*m*hz
KP_XY, KD_XY, F_XY_MAX = 60.0, 10.0, 4.0   # xy spring to the live spike axis
KR, KW, TAU_MAX = 0.6, 0.03, 0.15          # attitude PD (TAU_MAX << breakTorque 0.5)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.piston_jar")().build(num_envs=args.num_envs, device=device)
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

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def enc(q_ref: torch.Tensor, q_now: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
        """Pre-encode a desired WORLD vector against the force-frame drag."""
        return _qapply(q_ref, _qapply(_qinv(q_now), vec_w))

    def jz(pos_w: torch.Tensor) -> torch.Tensor:
        return scene._jar_local(pos_w)

    def report(tag: str) -> None:
        jar_z = float(scene.jar.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        tip_l = jz(scene.spike_tip_w())[0]
        ball_l = jz(scene.ball.data.root_pos_w)[0]
        lid_l = jz(scene.lid.data.root_pos_w)[0]
        print(f"[solve] {tag:12s} | jar_z={jar_z:+.4f} "
              f"tip_l=({float(tip_l[0]):+.3f},{float(tip_l[1]):+.3f},{float(tip_l[2]):+.3f}) "
              f"ball_lz={float(ball_l[2]):+.3f} "
              f"lid_l=({float(lid_l[0]):+.3f},{float(lid_l[1]):+.3f},{float(lid_l[2]):+.3f}) "
              f"eng={bool(scene.engaged()[0])} cov={bool(scene.covered()[0])} "
              f"pres={bool(scene.presented()[0])} in_jar={bool(scene.ball_in_jar()[0])} "
              f"in_dish={bool(scene.ball_in_dish()[0])} still={bool(scene.still()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def teleport_stack(dxyz: torch.Tensor) -> None:
        """Rigid TRANSLATION of the sealed assembly (transport only): every body
        keeps its orientation and current relative offsets, velocities zeroed."""
        for body in (scene.jar, scene.piston, scene.lid, scene.ball):
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = body.data.root_pos_w + dxyz
            st[:, 3:7] = body.data.root_quat_w
            body.write_root_state_to_sim(st, all_ids)

    def jar_servo(tag: str, *, v_des: float, press_cap: float, max_steps: int,
                  target_z: float | None = None, esc_step: float = 0.0,
                  esc_max: float = 0.0) -> str:
        """Hold-and-lower the jar with an applied wrench (the gripper surrogate).
        Returns 'reached' (target_z hit) | 'stalled' | 'timeout'. Forces persist
        after return (the caller decides when to let go)."""
        press, cap = 0.0, press_cap
        win_i, win_z = 0, float(scene.jar.data.root_pos_w[0, 2])
        for i in range(max_steps):
            p = scene.jar.data.root_pos_w
            v = scene.jar.data.root_lin_vel_w
            q = scene.jar.data.root_quat_w
            w = scene.jar.data.root_ang_vel_w
            z0 = float(p[0, 2] - scene.env_origins[0, 2])
            vz0 = float(v[0, 2])
            if target_z is not None and z0 <= target_z:
                print(f"[solve] {tag}: reached z={z0:.4f} (target {target_z:.4f})",
                      flush=True)
                return "reached"
            # adaptive press feedforward: creep up while too slow, back off when fast
            if vz0 > v_des + 0.005:
                press = min(press + 0.08, cap)
            elif vz0 < v_des - 0.02:
                press = max(press - 0.3, 0.0)
            fz = (W_FF + KZ * (v_des - v[:, 2]) - press).clamp(W_FF - cap, W_FF + 6.0)
            tip = scene.spike_tip_w()
            fxy = KP_XY * (tip[:, :2] - p[:, :2]) - KD_XY * v[:, :2]
            fn = fxy.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            fxy = fxy * (fn.clamp(max=F_XY_MAX) / fn)
            f_w = torch.cat([fxy, fz.unsqueeze(-1)], dim=-1)
            qe = _qmul(jar_qref, _qinv(q))
            rotvec = 2.0 * qe[:, 1:4] * torch.sign(qe[:, 0:1])
            tau_w = KR * rotvec - KW * w
            tn = tau_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            tau_w = tau_w * (tn.clamp(max=TAU_MAX) / tn)
            scene.jar.set_external_force_and_torque(
                enc(jar_qref, q, f_w).unsqueeze(1), enc(jar_qref, q, tau_w).unsqueeze(1),
                env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i - win_i >= 90:  # 0.75 s stall window
                if win_z - z0 < 0.0008:
                    if esc_step > 0.0 and cap < esc_max:
                        cap = min(cap + esc_step, esc_max)
                        print(f"[solve] {tag}: stalled at z={z0:.4f}; "
                              f"press cap -> {cap:.0f} N", flush=True)
                    else:
                        print(f"[solve] {tag}: stalled at z={z0:.4f}", flush=True)
                        return "stalled"
                win_i, win_z = i, z0
        return "timeout"

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(150)
    orig = scene.env_origins[0]
    def yaw_of(q):
        return float(torch.rad2deg(2.0 * torch.atan2(q[3], q[0])))
    ped_p = (scene.pedestal.data.root_pos_w[0] - orig)
    jar_p = (scene.jar.data.root_pos_w[0] - orig)
    dish_p = (scene.dish.data.root_pos_w[0] - orig)
    tip_w = (scene.spike_tip_w()[0] - orig)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"pedestal=({float(ped_p[0]):+.3f},{float(ped_p[1]):+.3f}) "
          f"yaw={yaw_of(scene.pedestal.data.root_quat_w[0]):+.1f}deg "
          f"tip=({float(tip_w[0]):+.3f},{float(tip_w[1]):+.3f},{float(tip_w[2]):+.3f}) "
          f"jar=({float(jar_p[0]):+.3f},{float(jar_p[1]):+.3f}) "
          f"yaw={yaw_of(scene.jar.data.root_quat_w[0]):+.1f}deg "
          f"dish=({float(dish_p[0]):+.3f},{float(dish_p[1]):+.3f}) "
          f"yaw={yaw_of(scene.dish.data.root_quat_w[0]):+.1f}deg", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.covered()[0]), "lid must start sealing the mouth"
    assert bool(scene.ball_in_jar()[0]), "ball must start captive in the jar"
    assert not bool(scene.engaged()[0]), "spike must start outside the jar"
    s0 = print_score("P0 reset+settle (jar sealed, ball captive)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"
    jar_qref = scene.jar.data.root_quat_w.clone()   # attitude target + drag reference
    lid_qref = scene.lid.data.root_quat_w.clone()   # drag reference for lid nudges

    # ---------------- phase 1: carry over the spike, guarded lower to the stall ------------
    engaged_stall = False
    for attempt in range(3):
        tip = scene.spike_tip_w()
        target = tip.clone()
        target[:, 2] += 0.005          # jar bottom 5 mm ABOVE the tip: nothing engaged
        teleport_stack(target - scene.jar.data.root_pos_w)
        assert not bool(scene.engaged()[0]), "hover write must not engage the spike"
        status = jar_servo("P1-lower", v_des=-0.02, press_cap=3.0, max_steps=900)
        engaged_stall = status == "stalled" and bool(scene.engaged()[0]) \
            and bool(scene.covered()[0])
        if engaged_stall:
            break
        print(f"[solve] P1 attempt {attempt} failed (status={status}, "
              f"eng={bool(scene.engaged()[0])}); re-carrying", flush=True)
        clear_wrench(scene.jar)
    report("P1-stall")
    if not engaged_stall:
        print("SIM_GEN_SOLVE: FAIL (never stalled sleeved on the spike, seal intact)",
              flush=True)
        os._exit(1)
    s1 = print_score("P1 sleeved on the spike, seal intact (stalled at ball-lid contact)")
    assert s1 >= c.w_engage - 1e-6, f"P1 score {s1} (expect engage latch {c.w_engage})"
    assert bool(scene.covered()[0]), "the 3 N guarded lower must NOT shear the seal"
    assert not bool(scene.success()[0])

    # ---------------- phase 2: press to shear; jar bottoms out; lid tips off ---------------
    ped_top = float(scene.pedestal.data.root_pos_w[0, 2] - orig[2]) + c.base_h
    status = jar_servo("P2-press", v_des=-0.04, press_cap=25.0, max_steps=900,
                       target_z=ped_top + 0.004, esc_step=10.0, esc_max=45.0)
    step(60)          # sustain the press (last wrench persists) to fully seat
    clear_wrench(scene.jar)
    step(150)         # hands off; everything relaxes
    jar_z = float(scene.jar.data.root_pos_w[0, 2] - orig[2])
    if status not in ("reached",) and not (jar_z <= ped_top + 0.010):
        report("P2-fail")
        print("SIM_GEN_SOLVE: FAIL (press never bottomed out)", flush=True)
        os._exit(1)
    # contingency: the popped (now FREE) lid balanced covering the mouth (a thin
    # plate on a sphere top is statically STABLE, and an edge can wedge at the
    # collar) -> swipe it off: lateral push + partial gravity relief, direction
    # rotating 45 deg per attempt, escalating, with a live early-exit. This is
    # the fingertip swipe a Franka would use on the ungrippable loose plate.
    for k in range(8):
        if bool(scene.jar_open()[0]):
            break
        d = scene.lid.data.root_pos_w[:, :2] - scene.jar.data.root_pos_w[:, :2]
        if float(d[0].norm()) < 0.004:
            ex = torch.zeros(n, 3, device=device)
            ex[:, 0] = 1.0
            d = _qapply(scene.lid.data.root_quat_w, ex)[:, :2]  # CoM-offset direction
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        ang = torch.tensor(math.radians(45.0 * k), device=device)
        rot = torch.stack([d[:, 0] * torch.cos(ang) - d[:, 1] * torch.sin(ang),
                           d[:, 0] * torch.sin(ang) + d[:, 1] * torch.cos(ang)], dim=-1)
        mag = 0.15 + 0.06 * k          # lateral, N
        fup = min(0.30 + 0.04 * k, 0.55)  # partial gravity relief (lid weight 0.39 N)
        print(f"[solve] P2: lid still covering; swipe {k} "
              f"(lat {mag:.2f} N, up {fup:.2f} N)", flush=True)
        for _ in range(90):
            if bool(scene.jar_open()[0]):
                break
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, :2] = rot * mag
            f_w[:, 2] = fup
            scene.lid.set_external_force_and_torque(
                enc(lid_qref, scene.lid.data.root_quat_w, f_w).unsqueeze(1),
                zero_wrench, env_ids=all_ids, is_global=True)
            env.step(no_action)
        clear_wrench(scene.lid)
        step(90)
    report("P2-open")
    if not bool(scene.jar_open()[0]):
        print("SIM_GEN_SOLVE: FAIL (lid never cleared the mouth)", flush=True)
        os._exit(1)
    s2 = print_score("P2 seal sheared; lid off the mouth; ball presented above the rim")
    assert s2 >= s1 - 1e-6, "score decreased across the press phase"
    assert s2 >= c.w_engage + c.w_open + c.w_present - 1e-6, \
        f"P2 score {s2} (expect engaged+opened+presented = 0.60)"
    assert not bool(scene.success()[0]), "cannot be success with the ball not delivered"

    # ---------------- phase 3: deliver the freed ball to the dish (drop) -------------------
    got = False
    for attempt in range(3):
        st = torch.zeros(n, 13, device=device)
        hover = torch.zeros(n, 3, device=device)
        hover[:, 2] = 0.075            # ABOVE dish_z_hi: the write satisfies nothing
        st[:, 0:3] = scene.dish.data.root_pos_w + _qapply(
            scene.dish.data.root_quat_w, hover)
        st[:, 3] = 1.0
        scene.ball.write_root_state_to_sim(st, all_ids)
        step(180)                      # free fall + rattle + settle, hands-off
        got = bool(scene.ball_in_dish()[0]) and \
            float(scene.ball.data.root_lin_vel_w[0].norm()) < 0.05
        if got:
            break
        print(f"[solve] ball drop attempt {attempt} missed "
              f"(in_dish={bool(scene.ball_in_dish()[0])}); re-dropping", flush=True)
    report("P3-deliver")
    if not got:
        print("SIM_GEN_SOLVE: FAIL (ball never came to rest in the dish)", flush=True)
        os._exit(1)
    s3 = print_score("P3 red ball dropped into the dish")
    assert s3 >= s2 - 1e-6, "score decreased across the delivery phase"

    # ---------------- phase 4: settle to stillness -> success ------------------------------
    step(150)         # the still counter needs 60 consecutive quiet substeps
    report("P4-still")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (final state not successful)", flush=True)
        os._exit(1)
    s4 = print_score("P4 at rest: ball in dish, jar mouth open, all still")
    assert s4 >= 1.0 - 1e-6, f"success must score 1.0, got {s4}"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
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
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()
