"""Teleport solution for BarredGateScene (sim_gen task `close_door_i162`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. CLOSING the door (torque servo, real hinge + contact dynamics): a PD torque
   about the hinge axis (`set_external_force_and_torque` on the door body — pure
   z torque, what a hand on the panel edge applies) swings the door from its
   random opening onto its closed stop and presses it there. The joint limit and
   the frame geometry arrest it; the closure credit is produced by the hinge
   dynamics, never written.
2. LIFTING the rod (applied force): a PD force + gravity feedforward + righting
   torque (a firm force-limited grasp) draws the BLUE lock rod straight up out of
   its caddy well. The `lifted` latch fires DURING this force lift.
3. TRANSPORT (teleport): one root-state write carries the held rod through free
   air to a hover 30+ mm ABOVE the door's eyelet funnel mouth, upright, zero
   velocity. Open sky — no contact interaction is bypassed and no rubric clause
   is satisfied by the write (the rod is airborne, not threaded, not seated).
4. INSERTION (guided force descent + funnels + gravity): the same force-limited
   carry lowers the rod; the eyelet funnel, the eyelet channel, the staple funnel
   and the staple channel steer the shaft, and the tip comes to rest on the
   staple floor. `threaded` and `seated` are produced by contact. A light
   constant press torque holds the door on its stop during the descent (a hand
   on the panel) and is released before judging.
5. HANDS-OFF: all wrenches are zeroed; success must hold through >= 3.3 simulated
   seconds untouched — the seated rod alone bars the door.

Order: the door is closed BEFORE the rod ever approaches the channels (the scene
geometry arrests the door in both wrong orders; smoke proves it).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.close_door_i162.solve --headless [--seed N]
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
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.barred_gate")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def ang() -> float:
        return float(scene.open_angle_deg()[0])

    def report(tag: str) -> None:
        tip = scene._frame_local(scene.rod.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | angle={ang():+6.2f}deg "
              f"tip=({float(tip[0]):+.3f},{float(tip[1]):+.3f},{float(tip[2]):+.3f}) "
              f"seated={bool(scene.rod_seated()[0])} "
              f"thread={bool(scene.rod_threading_eyelet()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def door_press(tau: float) -> None:
        """Constant closing press about the hinge axis (pure z — immune to the
        pod's wrench frame-drag quirk). tau=0 releases the door."""
        t_w = torch.zeros(n, 3, device=device)
        t_w[:, 2] = tau
        t_b = quat_apply_inverse(scene.door.data.root_quat_w, t_w)
        scene.door.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))

    def close_door(*, kp: float = 8.0, kd: float = 2.0, clamp: float = 4.0,
                   a_tgt_deg: float = -2.0, steps: int = 900) -> None:
        """PD torque servo about the hinge: swing the door onto its closed stop
        and press. Positive z torque closes (open = -z rotation). Gains sized for
        the door's ~0.55 kg m^2 hinge inertia and the 1-substep wrench delay
        (kp*dt/I ~= 0.12)."""
        streak = 0
        for i in range(steps):
            a_rad = math.radians(ang() - a_tgt_deg)
            wz = float(scene.door.data.root_ang_vel_w[0, 2])
            # closing = +z angular velocity; damp against it
            tau = max(-clamp, min(clamp, kp * a_rad - kd * wz))
            door_press(tau)
            env.step(no_action)
            if ang() < 0.8 and abs(wz) < 0.25:
                streak += 1
                if streak >= 30:
                    break
            else:
                streak = 0
        print(f"[solve] close_door: angle={ang():+.2f}deg after <= {steps} steps",
              flush=True)

    def carry(body, mass: float, frame_body, tgt_local, *, kp: float, kd: float,
              clamp: float, ku: float, kw: float, steps: int, done=None,
              label: str = "") -> None:
        """Applied-force carry: PD toward a frame_body-local target + gravity
        feedforward (a firm force-limited grasp), plus a righting torque that
        stands in for the grasp's orientation constraint. Wrenches only THIS
        body."""
        tgt = torch.zeros(n, 3, device=device)
        tgt[:] = torch.tensor(tgt_local, device=device)
        for _ in range(steps):
            q = body.data.root_quat_w
            p = body.data.root_pos_w
            v = body.data.root_lin_vel_w
            w = body.data.root_ang_vel_w
            tgt_w = frame_body.data.root_pos_w \
                + quat_apply(frame_body.data.root_quat_w, tgt)
            f_w = mass * 9.81 * ez + kp * (tgt_w - p) - kd * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            f_b = quat_apply_inverse(q, f_w)
            axis = quat_apply(q, ez)
            # righting torque + damping of the TRANSVERSE spin only: the thin
            # rod's axial inertia is ~2e-5 kg m^2, so damping the axial
            # component violates the wrench-delay stability bound (kw*dt/I >> 1
            # -> NaN, observed); axial spin is symmetric and irrelevant
            w_perp = w - (w * axis).sum(dim=-1, keepdim=True) * axis
            t_w = ku * torch.cross(axis, ez, dim=-1) - kw * w_perp
            t_b = quat_apply_inverse(q, t_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
            env.step(no_action)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        loc = scene._frame_local(body.data.root_pos_w)[0]
        print(f"[solve] carry {label}: reached frame-local "
              f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"after <= {steps} steps", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # frame/caddy/rods settle; the damped door holds its random angle
    fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
    fq = scene.frame.data.root_quat_w[0]
    fyaw = math.degrees(2.0 * math.atan2(float(fq[3]), float(fq[0])))
    well = "+x" if bool(scene.rod_in_well_p[0]) else "-x"
    print(f"[solve] layout readback (seed {args.seed}): "
          f"frame=({float(fp[0]):+.3f},{float(fp[1]):+.3f}) yaw={fyaw:+.1f}deg "
          f"a0={float(scene.a0[0]):.1f}deg angle={ang():+.2f}deg rod_well={well}",
          flush=True)
    # mass readbacks: per-child density must have produced real masses (custom
    # spawners apply no cfg mass schemas — guard against silent regression)
    rod_m = float(scene.rod.root_physx_view.get_masses()[0].sum())
    door_m = float(scene.door.root_physx_view.get_masses()[0].sum())
    frame_m = float(scene.frame.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: rod={rod_m:.3f} kg door={door_m:.3f} kg "
          f"frame={frame_m:.1f} kg", flush=True)
    assert 0.25 < rod_m < 1.0, f"rod mass {rod_m} (density not applied?)"
    assert 3.0 < door_m < 12.0, f"door mass {door_m} (density not applied?)"
    assert frame_m > 30.0, f"frame mass {frame_m} (MassAPI not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(ang() - float(scene.a0[0])) < 4.0, \
        f"door must hold its spawned angle, angle={ang():.2f} vs a0={float(scene.a0[0]):.1f}"
    s0 = print_score("P0 reset+settle (door open, rods in the caddy)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: torque-servo the door onto its closed stop ------------------
    close_door()
    door_press(0.6)  # keep a light press while the rod is handled
    step(60)
    report("door-closed")
    assert ang() <= c.closed_max_deg, f"door not on its stop, angle={ang():.2f}"
    assert not bool(scene.success()[0]), \
        "a closed-but-unbarred door must NOT be success (the seed plan fails here)"
    s1 = print_score("P1 door closed on its stop (unbarred: no success)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_close - 0.02, f"P1 score {s1}"

    # ---------------- phase 2a: force-lift the lock rod out of its well --------------------
    rloc = scene._frame_local(scene.rod.data.root_pos_w)[0]
    carry(scene.rod, rod_m, scene.frame,
          (float(rloc[0]), float(rloc[1]), 0.34),
          kp=25.0, kd=8.0, clamp=8.0, ku=0.05, kw=0.05, steps=360,
          done=lambda: float(scene.rod.data.root_pos_w[0, 2]
                             - scene.env_origins[0, 2]) > 0.30,
          label="rod lift")
    assert bool(scene._lifted[0]), "lifted must latch during the force lift"
    s2a = print_score("P2a lock rod lifted clear of the caddy")
    assert s2a >= s1 - 1e-6 and s2a >= c.w_close + c.w_lift - 0.02, f"P2a score {s2a}"

    # ---------------- phase 2b: TRANSPORT to a hover above the eyelet funnel ---------------
    hover = torch.zeros(n, 3, device=device)
    hover[:] = torch.tensor([c.ch_x, c.ey_y, c.eyelet_fun_top + 0.032], device=device)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.door.data.root_pos_w \
        + quat_apply(scene.door.data.root_quat_w, hover)
    st[:, 3:7] = scene.door.data.root_quat_w  # upright (door quat is a pure yaw)
    scene.rod.write_root_state_to_sim(st, all_ids)
    print("[solve] P2b: rod transported to free air above the eyelet funnel "
          "(airborne: nothing judged is satisfied by the write)", flush=True)
    assert not bool(scene.rod_seated()[0]), "hover must not read as seated"

    # ---------------- phase 2c: guided force descent through eyelet -> staple --------------
    carry(scene.rod, rod_m, scene.door,
          (c.ch_x, c.ey_y, c.staple_z0 + 0.003),
          kp=25.0, kd=8.0, clamp=6.0, ku=0.05, kw=0.05, steps=720,
          done=lambda: bool(scene.rod_seated()[0]),
          label="rod descent")
    door_press(0.0)  # hands fully off
    step(90)
    report("rod-seated")
    assert bool(scene.rod_seated()[0]), "rod must rest seated in the staple"
    assert bool(scene.rod_threading_eyelet()[0]), "shaft must thread the eyelet"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (rod seated but success not reached)", flush=True)
        os._exit(1)
    s2 = print_score("P2 rod dropped through eyelet into staple; door barred")
    assert s2 >= s2a - 1e-6 and s2 >= 0.999, f"P2 score {s2}"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s hands-off")
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
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
