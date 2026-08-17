"""Teleport solution for VaultLauncherScene (sim_gen task pick_single_egad_i328) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. AIMING (applied torque on the turret — the handle-drag surrogate): a
   velocity-servo yaw torque about the axle swivels the turret from its
   randomized misaim (18-55 deg off) onto the vault bearing and settles it
   within a fraction of a degree. The `aimed` latch fires while the servo
   parks the turret. Gravity exerts no yaw torque about a vertical axle, so
   the released turret stays put.
2. TRANSPORT (teleport): one root-state write carries the ball from its tray
   to a hover 40 mm ABOVE the chute floor at the load point — open air, zero
   velocity. Nothing judged is satisfied by the write (the `loaded` band
   requires the ball ON the floor between the rails).
3. THE LAUNCH (gravity only, hands off): the ball drops into the channel,
   rolls down the incline, exits the lip at ~1.4 m/s, flies the ~109 mm gap,
   drops ~30 mm into the snout mouth, carries the 12 mm one-way ridge with
   ~3x the required energy, and falls 72 mm onto the vault's interior floor.
   `loaded`, `entered` and success() are all produced by contact dynamics
   after the release — no wrench ever touches the ball.
4. HANDS-OFF: success must hold through >= 3.3 simulated seconds untouched —
   the sealed geometry alone keeps the ball delivered.

Order is forced physically: the ball leaves the chute ~0.5 s after loading,
so aiming must precede loading (the smoke battery launches at a misaimed
turret to show the aim is load-bearing).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's stage credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds after the hands-off persistence window.

Run (forge): python -u -m simgen_tasks.pick_single_egad_i328.solve --headless [--seed N]
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
    env = ENVS.get("simgen.vault_launcher")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

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

    def yaw() -> float:
        return float(scene.turret_yaw_deg()[0])

    def err() -> float:
        return float(scene.aim_err_deg()[0])

    def report(tag: str) -> None:
        bt = scene._turret_local(scene.ball.data.root_pos_w)[0]
        bv = scene._vault_local(scene.ball.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | yaw={yaw():+7.2f}deg err={err():+6.2f}deg "
              f"ball_tur=({float(bt[0]):+.3f},{float(bt[1]):+.3f},{float(bt[2]):+.3f}) "
              f"ball_vlt=({float(bv[0]):+.3f},{float(bv[1]):+.3f},{float(bv[2]):+.3f}) "
              f"|v|={float(scene.ball.data.root_lin_vel_w[0].norm()):.3f} "
              f"aimed={bool(scene._aimed[0])} loaded={bool(scene._loaded[0])} "
              f"entered={bool(scene._entered[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    zero = torch.zeros(n, 1, 3, device=device)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything settles: turret hangs on its joint, ball rests in the tray
    bear = math.degrees(float(scene.vault_bear[0]))
    yaw0 = math.degrees(float(scene.init_yaw[0]))
    print(f"[solve] layout readback (seed {args.seed}): vault_bear={bear:+.2f}deg "
          f"turret_yaw={yaw():+.2f}deg (written {yaw0:+.2f}) aim_err={err():+.2f}deg",
          flush=True)
    # mass readbacks: per-child density / root MassAPI must have produced real
    # masses (custom spawners apply no cfg mass schemas — guard against regression)
    ball_m = float(scene.ball.root_physx_view.get_masses()[0].sum())
    tur_m = float(scene.turret.root_physx_view.get_masses()[0].sum())
    base_m = float(scene.base.root_physx_view.get_masses()[0].sum())
    print(f"[solve] mass readback: ball={ball_m*1e3:.1f} g turret={tur_m:.2f} kg "
          f"base={base_m:.1f} kg", flush=True)
    assert 0.030 < ball_m < 0.070, f"ball mass {ball_m} (density not applied?)"
    assert 0.8 < tur_m < 12.0, f"turret mass {tur_m} (density not applied?)"
    assert base_m > 25.0, f"base mass {base_m} (MassAPI not applied?)"
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(err()) >= 12.0, f"turret must start misaimed >= 12 deg, err={err():.2f}"
    ball_z0 = float(scene.ball.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    assert 0.015 < ball_z0 < 0.06, f"ball must rest in the tray, z={ball_z0:.3f}"
    s0 = print_score("P0 reset+settle (misaimed, ball in tray)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: AIM the turret (velocity-servo yaw torque) ------------------
    kt, kt_max, tau_max = 0.6, 5.0, 2.0
    good, last_ck, stall_ref = 0, 0, abs(err())
    tq = torch.zeros(n, 1, 3, device=device)
    for i in range(1800):
        e = math.radians(err())
        w = float(scene.turret.data.root_ang_vel_w[0, 2])
        w_des = max(-1.2, min(1.2, -3.0 * e))
        tau = max(-tau_max, min(tau_max, kt * (w_des - w)))
        tq[:, 0, 2] = tau
        # turret body z == world z (yaw-only joint), so body-frame torque is exact
        scene.turret.set_external_force_and_torque(zero, tq)
        env.step(no_action)
        if abs(err()) < 0.6 and abs(float(scene.turret.data.root_ang_vel_w[0, 2])) < 0.04:
            good += 1
            if good >= 5:
                break
        else:
            good = 0
        if i - last_ck >= 150:  # stall watch: escalate the GAIN, not the cap
            if stall_ref - abs(err()) < 0.5 and kt < kt_max:
                kt *= 1.6
                print(f"[solve] aim stall at err={err():+.2f}deg -> kt={kt:.2f}", flush=True)
            last_ck, stall_ref = i, abs(err())
    scene.turret.set_external_force_and_torque(zero, zero)
    step(60)  # hands off: no gravity yaw torque about a vertical axle — it must stay
    report("aimed")
    assert abs(err()) < c.aim_tol_deg - 0.5, f"aim servo missed, err={err():+.2f}deg"
    assert bool(scene._aimed[0]), "aimed must latch while the servo parks the turret"
    s1 = print_score("P1 turret aimed at the vault bearing")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_aim - 0.02, f"P1 score {s1}"

    # ---------------- phase 2: TRANSPORT the ball to a hover above the chute ---------------
    drop = torch.zeros(n, 3, device=device)
    drop[:, 0] = c.drop_x
    drop[:, 2] = float(scene._chute_floor_top(torch.tensor([c.drop_x], device=device))[0]) \
        + c.ball_r + c.drop_hover
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.turret.data.root_pos_w + quat_apply(scene.turret.data.root_quat_w, drop)
    st[:, 3] = 1.0
    scene.ball.write_root_state_to_sim(st, torch.arange(n, device=device))
    assert not bool(scene.ball_loaded()[0]), "hover must not read as loaded"
    print("[solve] P2: ball transported to free air 40 mm above the chute floor "
          "(airborne: nothing judged is satisfied by the write)", flush=True)
    s2 = print_score("P2 ball hovering over the chute load point")

    # ---------------- phase 3: release — gravity launch, flight, capture -------------------
    v_exit = z_arr = None
    lip_seen = mouth_seen = False
    for i in range(720):
        env.step(no_action)
        bt = scene._turret_local(scene.ball.data.root_pos_w)[0]
        bv = scene._vault_local(scene.ball.data.root_pos_w)[0]
        if not lip_seen and float(bt[0]) > c.chan_lip_x:
            lip_seen = True
            v_exit = float(scene.ball.data.root_lin_vel_w[0].norm())
            print(f"[solve] lip crossing at step {i}: v_exit={v_exit:.3f} m/s "
                  f"(predicted 1.41)", flush=True)
        if not mouth_seen and float(bv[0]) > c.mouth_x:
            mouth_seen = True
            z_arr = float(bv[2])
            print(f"[solve] mouth crossing at step {i}: arrival z={z_arr:.4f} "
                  f"(window {c.tun_floor_z0 + c.ball_r:.3f}..{c.tun_ceil_z - c.ball_r:.3f})",
                  flush=True)
        if i % 60 == 59:
            report(f"launch+{i+1}")
        if bool(scene.ball_in_vault()[0]) \
                and float(scene.ball.data.root_lin_vel_w[0].norm()) < 0.5 * c.settle_speed:
            break
    report("delivered")
    assert lip_seen and v_exit is not None and v_exit > 1.0, \
        f"ball never launched cleanly (v_exit={v_exit})"
    assert mouth_seen, "ball never crossed the mouth plane"
    assert bool(scene._loaded[0]), "loaded must latch while the ball rides the chute"
    assert bool(scene._entered[0]), "entered must latch inside the snout"
    assert bool(scene.ball_in_vault()[0]), "ball must end inside the vault interior"

    # ---------------- phase 4: success ------------------------------------------------------
    step(120)  # settle margin
    report("settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (delivered but success not reached)", flush=True)
        os._exit(1)
    s4 = print_score("P4 ball at rest inside the sealed vault")
    assert s4 >= s2 - 1e-6 and s4 >= 0.999, f"P4 score {s4}"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s hands-off")
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
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - Kit teardown hangs; die loudly NOW
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        import traceback

        traceback.print_exc()
        os._exit(1)
