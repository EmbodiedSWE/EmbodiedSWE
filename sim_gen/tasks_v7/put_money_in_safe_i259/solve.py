"""Teleport solution for SeesawVaultScene (sim_gen task `put_money_in_safe_i259`) —
the task's legitimacy certificate.

The striking property of this solve: it needs NO applied wrench anywhere. Every
load-bearing interaction is gravity + contact through the see-saw pivot; the
teleports are pure TRANSPORT (each write places an object in free air with zero
velocity, satisfying nothing the rubric judges):

1. PARK (teleport + dynamics): one root-state write carries the brass
   counterweight from its pedestal to a 6 mm hover over the pedal pan's centre
   (free air inside the open-top cage — nothing judged). It falls onto the pan;
   its standing weight back-drives the see-saw and swings the lid up to the
   60 deg open stop. The open credit comes from the hinge dynamics, never written.
2. DEPOSIT (teleport + dynamics): one write carries the cash brick from its stand
   to a free-air hover 0.42 m up, over the REAR part of the mouth that the raised
   lid has uncovered, long axis across the mouth, zero velocity (outside the
   chamber — nothing judged). Released, it free-falls through the mouth and
   settles on the vault floor: the `in` latch is produced by contact geometry.
3. UNPARK (teleport + dynamics): one write lifts the counterweight out of the pan
   back onto its pedestal top (free air above a kinematic stand — nothing
   judged). With the toll removed, the lid's own gravity bias closes the see-saw
   back onto the lower joint limit: the reseal is pure mechanism dynamics.
4. HANDS-OFF: success() must hold through >= 3.3 simulated seconds untouched —
   the brick rests inside the sealed chamber, the lid on its closed stop.

Order is forced physically: the closed lid refuses the brick (13 mm hover-gap vs a
30 mm brick, and pushing presses the lid INTO its stop — smoke proves it with a
real drop); the brick itself cannot pay the toll (2x torque margin, smoke proves
it); the lid cannot be closed while the counterweight sits in the pan.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.put_money_in_safe_i259.solve --headless [--seed N]
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
    env = ENVS.get("simgen.seesaw_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    from isaaclab.utils.math import quat_apply

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
        return float(scene.lid_angle_deg()[0])

    def report(tag: str) -> None:
        pb = scene._vault_local(scene.brick.data.root_pos_w)[0]
        pw = scene._vault_local(scene.weight.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | lid={ang():+7.2f}deg "
              f"brick=({float(pb[0]):+.3f},{float(pb[1]):+.3f},{float(pb[2]):+.3f}) "
              f"weight=({float(pw[0]):+.3f},{float(pw[1]):+.3f},{float(pw[2]):+.3f}) "
              f"inside={bool(scene.brick_inside()[0])} "
              f"closed={bool(scene.lid_closed()[0])} "
              f"open={bool(scene.lid_open()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def vault_world(local_xyz) -> torch.Tensor:
        """(N,3) world position of a vault-local point."""
        t = torch.zeros(n, 3, device=device)
        t[:] = torch.tensor(local_xyz, device=device)
        return scene.vault.data.root_pos_w + quat_apply(scene.vault.data.root_quat_w, t)

    def wait_until(pred, max_steps: int, streak_need: int = 30) -> int:
        """Step until pred() holds for streak_need consecutive steps (or budget)."""
        streak = 0
        for i in range(max_steps):
            env.step(no_action)
            if pred():
                streak += 1
                if streak >= streak_need:
                    return i + 1
            else:
                streak = 0
        return max_steps

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything settles; the lid rests on its closed stop
    vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
    vq = scene.vault.data.root_quat_w[0]
    vyaw = math.degrees(2.0 * math.atan2(float(vq[3]), float(vq[0])))
    pa = (scene.stand_a.data.root_pos_w - scene.env_origins)[0]
    pb = (scene.stand_b.data.root_pos_w - scene.env_origins)[0]
    brick_on_a = bool(scene.brick_on_a[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"vault=({float(vp[0]):+.3f},{float(vp[1]):+.3f}) yaw={vyaw:+.1f}deg "
          f"lid={ang():+.2f}deg "
          f"standA=({float(pa[0]):+.3f},{float(pa[1]):+.3f}) "
          f"standB=({float(pb[0]):+.3f},{float(pb[1]):+.3f}) "
          f"brick_on_a={brick_on_a}", flush=True)
    # mass readbacks: per-child density / root MassAPI must have produced real
    # masses (custom spawners apply no cfg mass schemas — guard the regression).
    # The torque budget IS the mechanism, so the ratio matters, not just presence.
    brick_m = float(scene.brick.root_physx_view.get_masses()[0].sum())
    weight_m = float(scene.weight.root_physx_view.get_masses()[0].sum())
    lever_m = float(scene.lever.root_physx_view.get_masses()[0].sum())
    vault_m = float(scene.vault.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: brick={brick_m:.3f} kg weight={weight_m:.3f} kg "
          f"lever={lever_m:.3f} kg vault={vault_m:.1f} kg", flush=True)
    assert 0.15 < brick_m < 0.45, f"brick mass {brick_m} (density not applied?)"
    assert 1.00 < weight_m < 1.60, f"weight mass {weight_m} (density not applied?)"
    assert 0.60 < lever_m < 2.00, f"lever mass {lever_m} (density not applied?)"
    assert vault_m > 40.0, f"vault mass {vault_m} (MassAPI not applied?)"
    assert weight_m > 3.0 * brick_m, "counterweight must dominate the brick"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(ang()) < 3.0, f"lid must rest on its closed stop, lid={ang():.2f}"
    s0 = print_score("P0 reset+settle (lid closed on its stop, brick on its stand)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: PARK the counterweight — the toll opens the lid -------------
    # Transport teleport: free-air hover 6 mm over the pan floor, centred in the
    # open-top cage, zero velocity. Everything after the write is hinge dynamics.
    pan_cx = (c.pan_x0 + c.pan_x1) / 2
    pan_floor_top = c.axis_z - 0.020 + 0.004
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = vault_world((pan_cx, 0.0, pan_floor_top + 0.006))
    st[:, 3:7] = scene.vault.data.root_quat_w  # upright
    scene.weight.write_root_state_to_sim(st, all_ids)
    print("[solve] P1: counterweight transported to a hover inside the pan cage "
          "(free air: nothing judged is satisfied by the write)", flush=True)
    used = wait_until(lambda: ang() > c.open_min_deg
                      and float(scene.lever.data.root_ang_vel_w[0].norm()) < 0.25,
                      max_steps=720)
    step(60)
    report("toll-paid")
    assert ang() > c.open_min_deg, \
        f"counterweight must hold the lid open, lid={ang():.2f} (after {used} steps)"
    pw = scene._vault_local(scene.weight.data.root_pos_w)[0]
    assert c.pan_x0 - 0.02 < float(pw[0]) < c.pan_x1 + 0.02 and abs(float(pw[1])) < c.pan_half_y + 0.02, \
        f"weight must stay caged in the pan, weight={pw.tolist()}"
    assert not bool(scene.success()[0]), "an open lid with the brick outside is not success"
    s1 = print_score("P1 counterweight parked, lid held on the open stop")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_open - 0.02, f"P1 score {s1}"

    # ---------------- phase 2: DEPOSIT the brick through the uncovered mouth ---------------
    # Transport teleport: hover 0.42 m up over the drop zone the raised lid has
    # uncovered, long axis across the mouth (vault-local y), zero velocity.
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = vault_world((c.drop_x, 0.0, c.drop_hover_z))
    half_pi = torch.full((n,), math.pi / 2, device=device)
    qz90 = torch.zeros(n, 4, device=device)
    qz90[:, 0], qz90[:, 3] = torch.cos(half_pi / 2), torch.sin(half_pi / 2)
    st[:, 3:7] = scene_mod._qmul(scene.vault.data.root_quat_w, qz90)
    scene.brick.write_root_state_to_sim(st, all_ids)
    print("[solve] P2: brick transported to a free-air hover over the uncovered "
          "rear mouth (outside the chamber: nothing judged is satisfied)", flush=True)
    assert not bool(scene.brick_inside()[0]), "hover must not read as inside"
    used = wait_until(lambda: bool(scene.brick_inside()[0])
                      and float(scene.brick.data.root_lin_vel_w[0].norm()) < c.settle_speed,
                      max_steps=600)
    step(60)
    report("deposited")
    if not bool(scene.brick_inside()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (brick did not land inside the chamber)", flush=True)
        os._exit(1)
    assert bool(scene._in[0]), "the `in` latch must be set after the drop"
    assert ang() > c.open_min_deg - 6.0, f"lid must still be held open, lid={ang():.2f}"
    assert not bool(scene.success()[0]), "lid still open: must not be success yet"
    s2 = print_score("P2 brick dropped through the mouth, resting inside")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_open + c.w_in - 0.03, f"P2 score {s2}"

    # ---------------- phase 3: UNPARK the counterweight — gravity reseals ------------------
    # Transport teleport: lift the weight out of the cage back onto its pedestal
    # top (the stand NOT holding the brick), free air, zero velocity. The lid
    # falls shut on its own gravity bias — pure mechanism dynamics.
    home = scene.stand_b.data.root_pos_w if brick_on_a else scene.stand_a.data.root_pos_w
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = home
    st[:, 2] = home[:, 2] + c.stand_h + 0.004
    st[:, 3] = 1.0
    scene.weight.write_root_state_to_sim(st, all_ids)
    print("[solve] P3: counterweight transported back to its pedestal "
          "(free air above a kinematic stand: nothing judged is satisfied)", flush=True)
    used = wait_until(lambda: bool(scene.lid_closed()[0])
                      and float(scene.lever.data.root_ang_vel_w[0].norm()) < 0.2,
                      max_steps=720)
    step(90)
    report("resealed")
    assert bool(scene.lid_closed()[0]), \
        f"lid must fall shut once the toll is removed, lid={ang():.2f} (after {used} steps)"
    assert bool(scene._closed_after[0]), "closed_after latch must be set"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (banked + resealed but success not reached)", flush=True)
        os._exit(1)
    s3 = print_score("P3 counterweight unparked, lid closed — brick banked")
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
