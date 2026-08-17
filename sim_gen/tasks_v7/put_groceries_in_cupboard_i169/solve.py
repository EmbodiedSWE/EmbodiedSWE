"""Teleport solution for CarouselCupboardScene (sim_gen task
`put_groceries_in_cupboard_i169`) — the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. ALIGN (torque servo, real free pivot + contact dynamics): a PD torque about
   the vertical pivot (`set_external_force_and_torque` on the turntable — pure
   z torque, what a hand pushing the crank KNOB in a circle applies) rotates
   the carousel from its random 75-170 deg offset until the EMPTY bay faces
   the window. The decoy cans ride their bays on friction. `aligned` latches
   DURING this servo; the damped pivot then holds the angle.
2. LIFT (applied force): a PD force + gravity feedforward + righting torque
   (a firm force-limited grasp) lifts the red carton straight up off its
   pickup stand. The `lifted` latch fires DURING this force lift.
3. TRANSPORT (teleport): one root-state write carries the held carton through
   free air to a hover OUTSIDE the window (cabinet-local x = 0.32, above the
   plinth edge, below the roof line, upright, zero velocity). Open air — no
   contact interaction is bypassed and no rubric clause is satisfied by the
   write (the carton is airborne, outside every bay).
4. INSERT (guided force carry + gravity): the same force-limited carry moves
   the carton horizontally in through the window and lowers it onto the
   carousel floor at the empty bay's slot radius; a light PD hold on the
   turntable (a hand steadying the knob) keeps the bay centred during the
   approach and is released before judging. `loaded` is produced by contact
   (3-step persistence of the upright, in-bay, slow carton).
5. STOW (torque servo again): the knob servo rotates the carousel a further
   ~150 deg so the loaded bay is behind the wall; the carton and both cans
   ride the disc on friction. Gains are sized so tangential/centripetal
   accelerations stay far below the friction cone (nothing slides or topples).
6. HANDS-OFF: all wrenches are zeroed; success must hold through >= 3.3
   simulated seconds untouched — the damped pivot alone keeps the bay stowed.

Order is forced by geometry: insertion is only possible through the window
into the aligned bay, and stowing is a rotation that must happen AFTER the
carton is aboard.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.put_groceries_in_cupboard_i169.solve --headless [--seed N]
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
    env = ENVS.get("simgen.carousel_cupboard")().build(num_envs=args.num_envs, device=device)
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

    def bearing() -> float:
        return float(scene.target_bearing_deg()[0])

    def report(tag: str) -> None:
        loc = scene._tt_local(scene.carton.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | bearing={bearing():+7.2f}deg "
              f"tt_w={float(scene.turntable.data.root_ang_vel_w[0, 2]):+.3f} "
              f"carton_tt=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"in_target={bool(scene.carton_in_target()[0])} "
              f"stowed={bool(scene.stowed()[0])} "
              f"decoys={bool(scene.decoys_home()[0])} "
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

    def tt_torque(tau: float) -> None:
        """Pure z torque on the turntable (what a hand on the knob applies —
        immune to the pod's wrench frame-drag quirk). tau=0 releases it."""
        t_w = torch.zeros(n, 3, device=device)
        t_w[:, 2] = tau
        t_b = quat_apply_inverse(scene.turntable.data.root_quat_w, t_w)
        scene.turntable.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))

    def tt_hold_tau(tgt_deg: float, *, kp: float = 0.6, kd: float = 0.25,
                    clamp: float = 0.30) -> float:
        err = math.radians((bearing() - tgt_deg + 180.0) % 360.0 - 180.0)
        wz = float(scene.turntable.data.root_ang_vel_w[0, 2])
        return max(-clamp, min(clamp, -(kp * err) - kd * wz))

    def index_to(tgt_deg: float, *, kp: float = 0.6, kd: float = 0.25,
                 clamp: float = 0.45, steps: int = 1200, tol: float = 4.0,
                 label: str = "") -> None:
        """PD torque servo on the free pivot: drive the empty-bay bearing to
        tgt_deg (shortest wrapped path). Gains sized for the ~0.045 kg m^2
        pivot inertia and the 1-substep wrench delay (kp*dt/I ~= 0.11); the
        torque clamp keeps tangential acceleration at the slot radius far
        below the friction cone so riders never slide."""
        streak = 0
        for _ in range(steps):
            tt_torque(tt_hold_tau(tgt_deg, kp=kp, kd=kd, clamp=clamp))
            env.step(no_action)
            err = abs((bearing() - tgt_deg + 180.0) % 360.0 - 180.0)
            wz = abs(float(scene.turntable.data.root_ang_vel_w[0, 2]))
            if err < tol and wz < 0.10:
                streak += 1
                if streak >= 30:
                    break
            else:
                streak = 0
        print(f"[solve] index_to {label}: bearing={bearing():+.2f}deg "
              f"(target {tgt_deg:+.1f}) after <= {steps} steps", flush=True)

    def carry(body, mass: float, frame_body, tgt_local, *, kp: float, kd: float,
              clamp: float, ku: float, kw: float, steps: int, done=None,
              also=None, label: str = "") -> None:
        """Applied-force carry: PD toward a frame_body-local target + gravity
        feedforward (a firm force-limited grasp), plus a righting torque that
        stands in for the grasp's orientation constraint. Transverse-only spin
        damping (the axial component of a near-symmetric prism is irrelevant
        and its small inertia violates the wrench-delay bound). `also`, if
        given, is called every step (e.g. the turntable hold servo)."""
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
            w_perp = w - (w * axis).sum(dim=-1, keepdim=True) * axis
            t_w = ku * torch.cross(axis, ez, dim=-1) - kw * w_perp
            t_b = quat_apply_inverse(q, t_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))
            if also is not None:
                also()
            env.step(no_action)
            if done is not None and done():
                break
        body.set_external_force_and_torque(zero, zero)
        loc = scene._tt_local(body.data.root_pos_w)[0]
        print(f"[solve] carry {label}: reached turntable-local "
              f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"after <= {steps} steps", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything settles; the damped free pivot holds its random angle
    cp = (scene.cabinet.data.root_pos_w - scene.env_origins)[0]
    cq = scene.cabinet.data.root_quat_w[0]
    cyaw = math.degrees(2.0 * math.atan2(float(cq[3]), float(cq[0])))
    e = int(scene.empty_bay[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cab=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) yaw={cyaw:+.1f}deg "
          f"empty_bay={e} blue_bay={int(scene.blue_bay[0])} "
          f"green_bay={int(scene.green_bay[0])} "
          f"bearing0={float(scene.bearing0[0]):+.1f} bearing={bearing():+.2f}deg",
          flush=True)
    # mass readbacks: authored MassAPI / per-child density must have produced
    # real masses (custom spawners apply no cfg mass schemas)
    cab_m = float(scene.cabinet.root_physx_view.get_masses()[0].sum())
    tt_m = float(scene.turntable.root_physx_view.get_masses()[0].sum())
    carton_m = float(scene.carton.root_physx_view.get_masses()[0].sum())
    can_m = float(scene.can_blue.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: cabinet={cab_m:.1f} kg turntable={tt_m:.2f} kg "
          f"carton={carton_m:.3f} kg can={can_m:.3f} kg", flush=True)
    assert cab_m > 30.0, f"cabinet mass {cab_m} (MassAPI not applied?)"
    assert 1.0 < tt_m < 8.0, f"turntable mass {tt_m} (density not applied?)"
    assert 0.15 < carton_m < 0.5, f"carton mass {carton_m} (density not applied?)"
    assert 0.08 < can_m < 0.4, f"can mass {can_m} (density not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(bearing() - float(scene.bearing0[0])) < 8.0, \
        f"carousel must hold its spawned bearing: {bearing():.2f} vs {float(scene.bearing0[0]):.1f}"
    assert bool(scene.decoys_home()[0]), "decoy cans must start seated in their bays"
    s0 = print_score("P0 reset+settle (empty bay rotated away, carton on its stand)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: index the carousel — empty bay to the window ----------------
    index_to(0.0, label="align")
    step(60)
    report("aligned")
    assert abs(bearing()) < c.align_tol_deg, f"bay not aligned, bearing={bearing():.2f}"
    assert bool(scene._aligned[0]), "aligned must latch during the knob servo"
    assert bool(scene.decoys_home()[0]), "decoys must ride the rotation in their bays"
    assert not bool(scene.success()[0]), "aligned-but-empty must not be success"
    s1 = print_score("P1 empty bay aligned with the window (knob torque servo)")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_align - 0.02, f"P1 score {s1}"

    # ---------------- phase 2a: force-lift the carton off its stand ------------------------
    carry(scene.carton, carton_m, scene.stand, (0.0, 0.0, 0.36),
          kp=15.0, kd=6.0, clamp=6.0, ku=0.03, kw=0.012, steps=360,
          done=lambda: float(scene.carton.data.root_pos_w[0, 2]
                             - scene.env_origins[0, 2]) > c.lift_z + 0.03,
          label="carton lift")
    assert bool(scene._lifted[0]), "lifted must latch during the force lift"
    s2a = print_score("P2a carton lifted off the pickup stand")
    assert s2a >= s1 - 1e-6 and s2a >= c.w_align + c.w_lift - 0.02, f"P2a score {s2a}"

    # ---------------- phase 2b: TRANSPORT to a hover outside the window --------------------
    hover = torch.zeros(n, 3, device=device)
    hover[:] = torch.tensor([0.32, 0.0, 0.20], device=device)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.cabinet.data.root_pos_w \
        + quat_apply(scene.cabinet.data.root_quat_w, hover)
    st[:, 3:7] = scene.cabinet.data.root_quat_w  # upright (cabinet quat is a pure yaw)
    scene.carton.write_root_state_to_sim(st, all_ids)
    print("[solve] P2b: carton transported to free air outside the window "
          "(airborne, outside every bay: nothing judged is satisfied by the write)",
          flush=True)
    assert not bool(scene.carton_in_target()[0]), "hover must not read as in-bay"

    # ---------------- phase 2c: guided force carry in through the window -------------------
    bay_ang = math.radians(120.0 * e)
    bx, by = c.slot_r * math.cos(bay_ang), c.slot_r * math.sin(bay_ang)
    # stage 1: through the window to above the bay (bottom clears the disc edge)
    carry(scene.carton, carton_m, scene.turntable, (bx, by, 0.215),
          kp=15.0, kd=6.0, clamp=5.0, ku=0.03, kw=0.012, steps=600,
          also=lambda: tt_torque(tt_hold_tau(0.0)),
          done=lambda: float(scene._tt_local(scene.carton.data.root_pos_w)[0, :2]
                             .norm()) < c.slot_r + 0.02,
          label="window entry")
    # stage 2: lower onto the carousel floor in the empty bay
    carry(scene.carton, carton_m, scene.turntable,
          (bx, by, c.disc_top + c.carton_h / 2 + 0.002),
          kp=15.0, kd=6.0, clamp=4.0, ku=0.03, kw=0.012, steps=600,
          also=lambda: tt_torque(tt_hold_tau(0.0)),
          done=lambda: bool(scene._loaded[0]),
          label="set-down")
    tt_torque(0.0)  # hands fully off the knob
    step(120)
    report("loaded")
    assert bool(scene.carton_in_target()[0]), "carton must rest upright in the empty bay"
    assert bool(scene._loaded[0]), "loaded must latch from contact"
    assert not bool(scene.success()[0]), \
        "loaded-but-facing-the-window must NOT be success (the seed plan ends here)"
    s2 = print_score("P2 carton set down in the empty bay through the window")
    assert s2 >= s2a - 1e-6 and s2 >= 0.70 - 1e-3, f"P2 score {s2}"

    # ---------------- phase 3: stow — rotate the loaded bay behind the wall ----------------
    index_to(150.0, kp=0.6, kd=0.30, clamp=0.30, steps=1500, tol=5.0,
             label="stow")
    tt_torque(0.0)
    step(120)
    report("stowed")
    assert bool(scene.stowed()[0]), f"bay not stowed, bearing={bearing():.2f}"
    assert bool(scene.carton_in_target()[0]), "carton must ride the rotation in its bay"
    assert bool(scene.decoys_home()[0]), "decoys must still be seated"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (stowed but success not reached)", flush=True)
        os._exit(1)
    s3 = print_score("P3 loaded bay stowed behind the wall")
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
