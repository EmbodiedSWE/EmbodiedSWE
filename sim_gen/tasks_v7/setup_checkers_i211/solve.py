"""Teleport solution for TiltLabyrinthScene (sim_gen task `setup_checkers_i211`)
— the task's legitimacy certificate.

There is NOTHING to teleport in this task: the checker is sealed under the
grille and must never be written to after reset, and the case never leaves its
stand. Every phase is executed through contact dynamics:

1. TILT LEG 1 (torque servo on the tray rim): a PD attitude servo maps the
   tray's world up-vector error to a torque on the TRAY body (exactly the wrench
   a hand pressing down on the 16 mm rim applies), tilting the play field toward
   the baffle (stand-local +x). Gravity slides the sealed disc through the
   funnel guides and the central baffle gap into the far chamber — the
   `gap_latch` credit is produced by the disc's own contact trajectory.
2. TILT LEG 2 (same servo, two straight sub-legs): first a pure +/-y tilt parks
   the disc against the BEACON-side rim wall, then a pure +x tilt slides it
   along that wall (no sideways press, so no wall friction) until it drops
   7 mm into the target corner pocket (`pocket_latch`).
3. RELEASE + LEVEL: the servo target goes to level, then all wrenches are
   zeroed — the spring-return hinges hold the case level, the pocket ledge
   holds the disc. success() = in target pocket + level + settled.
4. HANDS-OFF: success must hold through >= 3.3 simulated seconds untouched.

The wrench is torque-only on the tray (no forces on the disc, ever) and is
clamped to 1.5 N m — about a 10 N fingertip press at the rim lever arm.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.setup_checkers_i211.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
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
    env = ENVS.get("simgen.tilt_labyrinth")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

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

    def report(tag: str) -> None:
        p = scene.disc_local()[0]
        print(f"[solve] {tag:14s} | disc_loc=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) tilt={float(scene.tilt_deg()[0]):5.2f}deg "
              f"gap={bool(scene._gap_latch[0])} "
              f"pocket={bool(scene._pocket_latch[0])} "
              f"decoy={bool(scene.in_decoy_pocket()[0])} "
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

    def hands_off() -> None:
        scene.tray.set_external_force_and_torque(zero, zero)

    def tilt_servo(d_local, alpha: float, steps: int, *, done=None, label: str = "",
                   kp: float = 20.0, kd: float = 0.5, clamp: float = 1.5) -> None:
        """PD attitude servo: press the rim (a torque on the tray body) until the
        tray up-vector leans `alpha` rad toward the stand-local direction
        `d_local` (downhill = the direction the floor descends). tau_w =
        kp*(ez x err) - kd*w_xy is yaw-covariant, so it works at any stand
        heading. Torque only — the sealed disc is never touched. Gains sized for
        the ~0.02-0.09 kg m^2 hinge inertias and the 1-substep wrench delay
        (kp*dt^2/I ~= 0.07, kd*dt/I ~= 0.21)."""
        d = torch.zeros(n, 3, device=device)
        nm = math.hypot(d_local[0], d_local[1])
        if nm > 1e-9:
            d[:, 0] = d_local[0] / nm
            d[:, 1] = d_local[1] / nm
        # spring feedforward kills the proportional droop (theta would otherwise
        # settle at kp*alpha/(kp+k_spring)): drive_k is authored per-DEGREE
        k_ff = c.drive_k * 180.0 / math.pi
        for _ in range(steps):
            d_w = quat_apply(scene.stand.data.root_quat_w, d)
            err = alpha * d_w - scene.tray_up()
            err[:, 2] = 0.0
            w = scene.tray.data.root_ang_vel_w.clone()
            w[:, 2] = 0.0
            tau_w = kp * torch.cross(ez, err, dim=-1) - kd * w \
                + k_ff * alpha * torch.cross(ez, d_w, dim=-1)
            tn = tau_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            tau_w = tau_w * (tn.clamp(max=clamp) / tn)
            tau_b = quat_apply_inverse(scene.tray.data.root_quat_w, tau_w)
            scene.tray.set_external_force_and_torque(zero, tau_b.reshape(n, 1, 3))
            env.step(no_action)
            if done is not None and done():
                break
        hands_off()
        p = scene.disc_local()[0]
        print(f"[solve] tilt {label}: disc_loc=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) tilt={float(scene.tilt_deg()[0]):.2f}deg "
              f"after <= {steps} steps", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # springs centre the gimbal; the disc parks in its start cell
    sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
    sq = scene.stand.data.root_quat_w[0]
    syaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    sgn = float(scene.target_sign[0])
    p0 = scene.disc_local()[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"stand=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg "
          f"target_side={'+y' if sgn > 0 else '-y'} "
          f"disc_loc=({float(p0[0]):+.3f},{float(p0[1]):+.3f},{float(p0[2]):+.3f})",
          flush=True)
    # mass readbacks: custom spawners apply no cfg mass schemas — guard against
    # silent regression (density/MassAPI authored in the spawn funcs)
    disc_m = float(scene.disc.root_physx_view.get_masses()[0].sum())
    tray_m = float(scene.tray.root_physx_view.get_masses()[0].sum())
    frame_m = float(scene.frame.root_physx_view.get_masses()[0].sum())
    stand_m = float(scene.stand.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: disc={disc_m * 1000:.1f} g tray={tray_m:.2f} kg "
          f"frame={frame_m:.2f} kg stand={stand_m:.1f} kg", flush=True)
    assert 0.02 < disc_m < 0.05, f"disc mass {disc_m} (MassAPI not applied?)"
    assert 0.4 < tray_m < 5.0, f"tray mass {tray_m} (density not applied?)"
    assert 0.5 < frame_m < 5.0, f"frame mass {frame_m} (density not applied?)"
    assert stand_m > 30.0, f"stand mass {stand_m} (MassAPI not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    # spring-return calibration guard: with the disc off-centre the residual tilt
    # must stay inside the level band (catches a per-radian drive-unit surprise)
    t0 = float(scene.tilt_deg()[0])
    assert t0 < c.level_max_deg, \
        f"case must rest level under springs, tilt={t0:.2f}deg (drive_k units?)"
    assert abs(float(p0[0]) - (c.disc_x_range[0] + c.disc_x_range[1]) / 2) < 0.05 \
        and float(p0[2]) < 0.0, f"disc must rest in the start chamber, got {p0}"
    s0 = print_score("P0 reset+settle (case level, checker parked in the start chamber)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: tilt leg 1 — through the funnel and the baffle gap ----------
    tilt_servo((1.0, 0.0), 0.19, 1200,
               done=lambda: float(scene.disc_local()[0, 0]) > 0.10,
               label="leg1 (+x, through the gap)")
    step(60)  # let the disc finish its run to the far wall
    report("leg1-done")
    assert float(scene.disc_local()[0, 0]) > c.gap_x_min + 0.02, \
        "disc must be in the far chamber after leg 1"
    assert bool(scene._gap_latch[0]), "gap credit must latch during leg 1"
    assert not bool(scene.success()[0]), "far chamber alone must not be success"
    s1 = print_score("P1 checker through the baffle gap (far chamber)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_gap - 0.02, f"P1 score {s1}"

    # ---------------- phase 2: tilt leg 2 — to the beacon-side wall, then into the pocket --
    # 2a: pure +/-y tilt parks the disc against the beacon-side rim wall. A
    # diagonal would not work: the x downhill component cannot beat floor + wall
    # friction while the disc is pressed sideways (measured stall on the forge).
    tilt_servo((0.0, sgn), 0.17, 900,
               done=lambda: float(scene.disc_local()[0, 1]) * sgn > 0.070,
               label="leg2a (to the beacon-side wall)")
    step(60)
    p2a = scene.disc_local()[0]
    assert float(p2a[1]) * sgn > 0.060, \
        f"disc must reach the beacon-side wall band, got y={float(p2a[1]):+.3f}"
    # 2b: pure +x tilt slides the disc ALONG the wall (no y-press, no wall
    # friction) until it drops over the pocket ledge.
    tilt_servo((1.0, 0.0), 0.19, 1200,
               done=lambda: bool(scene._pocket_latch[0]),
               label="leg2b (along the wall into the pocket)")
    report("leg2-done")
    assert bool(scene._pocket_latch[0]), "pocket credit must latch during leg 2"
    assert bool(scene.in_target_pocket()[0]), "disc must sit in the TARGET pocket"
    assert not bool(scene.in_decoy_pocket()[0]), "disc must not be in the decoy"
    assert not bool(scene.success()[0]), \
        "pocketed but still tilted/held must NOT be success (release is judged)"
    s2 = print_score("P2 checker dropped into the beacon-side pocket (case still held)")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_gap + c.w_pocket - 0.02, f"P2 score {s2}"

    # ---------------- phase 3: release the rim — springs re-level, everything rests --------
    tilt_servo((0.0, 0.0), 0.0, 300,
               done=lambda: float(scene.tilt_deg()[0]) < 0.8,
               label="re-level")
    hands_off()
    step(180)  # fully hands-off settle
    report("released")
    assert bool(scene.in_target_pocket()[0]), "pocket must retain the disc through release"
    assert bool(scene.level()[0]), f"case must rest level, tilt={float(scene.tilt_deg()[0]):.2f}"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (released and level but success not reached)", flush=True)
        os._exit(1)
    s3 = print_score("P3 rim released; case level; checker settled in the marked pocket")
    assert s3 >= 0.999, f"P3 score {s3}"

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
