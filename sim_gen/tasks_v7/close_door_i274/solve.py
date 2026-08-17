"""Solution for SaggingGateScene (sim_gen task `close_door_i274`) — the task's
legitimacy certificate. NO teleports at all: the gate is the only moving body and
it never leaves its worn hinge, so every phase is applied wrenches through the
joint + contact dynamics (`set_external_force_and_torque` on the gate root — the
force a hand gripping the blue T-knob applies, plus the small hinge-axis moment
of pushing the panel).

1. LIFT (applied force): a PD heave servo with gravity feedforward raises the
   whole gate up its 50 mm hinge slack and holds it there. `lifted` latches
   DURING the force lift; this also proves the worn hinge's transZ DOF is live.
2. CARRY SHUT (force + torque servo): the heave hold continues while a PD torque
   about the hinge axis swings the gate from its random opening to ~2 deg — the
   raised shoe passes OVER the threshold apron (the same servo without the lift
   is arrested by the apron wall at ~52 deg; smoke proves it). Mid-solve the
   closed-but-held-high state is asserted NOT success.
3. RELEASE (guided descent + gravity + contact): the heave target ramps to the
   bottom of the slack while a light yaw hold keeps the shoe over the socket;
   the shoe drops into the green-rimmed well and the gate seats. All wrenches
   are then zeroed. The seat is produced by gravity and the well walls.
4. HANDS-OFF: success must hold through >= 3.3 simulated seconds untouched —
   the seated shoe alone keeps the gate closed.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.close_door_i274.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401 - registers the scene
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
    env = ENVS.get("simgen.sagging_gate")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply_inverse

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

    def heave() -> float:
        return float(scene.heave()[0])

    def report(tag: str) -> None:
        p = scene.shoe_frame_local()[0]
        print(f"[solve] {tag:14s} | angle={ang():+6.2f}deg heave={heave():+.4f} "
              f"shoe=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"seated={bool(scene.shoe_seated()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)
    gate_m = float(scene.gate.root_physx_view.get_masses()[0].sum())

    def drive(steps: int, *, h_tgt: float, a_tgt: float | None,
              kp_h: float = 100.0, kd_h: float = 22.0,
              kp_a: float = 1.6, kd_a: float = 0.5, t_clamp: float = 1.2,
              done=None, streak_n: int = 1, label: str = "") -> None:
        """One combined wrench servo tick loop on the gate root:
        - heave: world-z PD force toward h_tgt + gravity feedforward (the grip on
          the knob; kp_h*dt/m ~= 0.39, under the wrench-delay stability bound);
        - yaw (if a_tgt given): world-z PD torque toward a_tgt deg (positive z
          closes; kp_a*dt/I ~= 0.13 for the gate's ~0.10 kg m^2 hinge inertia).
        Wrenches are recomputed every step from live state and converted to the
        gate's BODY frame. Leaves the last wrench applied (the hold persists)."""
        streak = 0
        for _ in range(steps):
            q = scene.gate.data.root_quat_w
            vz = float(scene.gate.data.root_lin_vel_w[0, 2])
            wz = float(scene.gate.data.root_ang_vel_w[0, 2])
            fz = gate_m * 9.81 + kp_h * (h_tgt - heave()) - kd_h * vz
            fz = max(-10.0, min(35.0, fz))
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 2] = fz
            t_w = torch.zeros(n, 3, device=device)
            if a_tgt is not None:
                tau = kp_a * math.radians(ang() - a_tgt) - kd_a * wz
                t_w[:, 2] = max(-t_clamp, min(t_clamp, tau))
            scene.gate.set_external_force_and_torque(
                quat_apply_inverse(q, f_w).reshape(n, 1, 3),
                quat_apply_inverse(q, t_w).reshape(n, 1, 3))
            env.step(no_action)
            if done is not None:
                streak = streak + 1 if done() else 0
                if streak >= streak_n:
                    break
        print(f"[solve] drive {label}: angle={ang():+.2f}deg heave={heave():+.4f} "
              f"after <= {steps} steps", flush=True)

    def hands_off() -> None:
        scene.gate.set_external_force_and_torque(zero, zero)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # frame settles on its feet; the damped gate holds its random angle
    fp = (scene.frame.data.root_pos_w - scene.env_origins)[0]
    fq = scene.frame.data.root_quat_w[0]
    fyaw = math.degrees(2.0 * math.atan2(float(fq[3]), float(fq[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"frame=({float(fp[0]):+.3f},{float(fp[1]):+.3f}) yaw={fyaw:+.1f}deg "
          f"a0={float(scene.a0[0]):.1f}deg angle={ang():+.2f}deg heave={heave():+.4f}",
          flush=True)
    # mass readbacks: custom spawners apply no cfg mass schemas — the gate's
    # per-child density and the frame's root MassAPI must have taken.
    frame_m = float(scene.frame.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: gate={gate_m:.3f} kg frame={frame_m:.1f} kg", flush=True)
    assert 1.2 < gate_m < 2.6, f"gate mass {gate_m} (density not applied?)"
    assert frame_m > 30.0, f"frame mass {frame_m} (MassAPI not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    # the worn hinge must be live and correctly limited: the gate hangs at its
    # spawned angle, sagging at the BOTTOM of the vertical play
    assert abs(ang() - float(scene.a0[0])) < 4.0, \
        f"gate must hold its spawned angle, angle={ang():.2f} vs a0={float(scene.a0[0]):.1f}"
    assert heave() < 0.010, f"gate must sag at the bottom of the slack, heave={heave():.4f}"
    s0 = print_score("P0 reset+settle (gate open, sagging low)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: force-lift the gate up its hinge slack ----------------------
    drive(300, h_tgt=c.slack - 0.002, a_tgt=None,
          done=lambda: heave() > 0.044, streak_n=15, label="lift")
    report("lifted")
    assert heave() > 0.040, f"gate must ride up the slack, heave={heave():.4f}"
    assert bool(scene._lifted[0]), "lifted must latch during the force lift"
    assert not bool(scene.success()[0])
    s1 = print_score("P1 gate lifted up the worn-hinge slack (still open)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_lift - 0.02, f"P1 score {s1}"

    # ---------------- phase 2: carry it shut, held high ------------------------------------
    drive(900, h_tgt=c.slack - 0.002, a_tgt=1.9,
          done=lambda: ang() < 2.8 and abs(float(scene.gate.data.root_ang_vel_w[0, 2])) < 0.15
          and heave() > 0.040,
          streak_n=40, label="carry-shut")
    report("carried-shut")
    assert ang() <= 3.5, f"gate not carried shut, angle={ang():.2f}"
    assert heave() > 0.038, f"gate sagged during the carry, heave={heave():.4f}"
    assert bool(scene._carried[0]), "carried must latch during the raised swing"
    assert not bool(scene.shoe_seated()[0]), "held high: the shoe must NOT read seated"
    assert not bool(scene.success()[0]), \
        "closed-but-held-high must NOT be success (the angle alone is not the goal)"
    s2 = print_score("P2 gate carried shut over the apron (held high: no success)")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_lift + c.w_carry + c.w_close * 0.9 - 0.02, \
        f"P2 score {s2}"

    # ---------------- phase 3: guided release into the socket ------------------------------
    # lower the gate gently down the slack (a hand easing it down), light yaw
    # hold keeping the shoe over the well; the shoe enters the green rim walls
    # and the gate seats. Then hands fully off.
    drive(360, h_tgt=0.0, a_tgt=1.9, kp_h=60.0, kd_h=25.0, t_clamp=0.5,
          done=lambda: heave() < 0.006, streak_n=20, label="release-descent")
    hands_off()
    step(150)
    report("seated")
    assert bool(scene.shoe_seated()[0]), "shoe must rest inside the socket well"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (shoe seated but success not reached)", flush=True)
        os._exit(1)
    s3 = print_score("P3 shoe dropped into the socket; gate seated closed")
    assert s3 >= s2 - 1e-6 and s3 >= 0.999, f"P3 score {s3}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s hands-off")
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
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
