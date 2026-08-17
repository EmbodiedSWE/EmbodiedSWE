"""Teleport solution for StoneDoorVaultScene (sim_gen task `close_microwave_i306`) —
the task's legitimacy certificate.

This solution uses ZERO teleports: the blue stone starts standing in its channel, and
every load-bearing interaction is executed through contact dynamics:

1. ROLL (applied force + rolling/sliding contact): a horizontal velocity-servo force
   at the stone's CoM (a world-space vector, re-expressed in the stone's body frame
   every step — see the wrench-frame note in main() — along the channel toward the
   pocket, |F| capped at 4 N ~ 0.8 mg) drives the stone 14-26 cm along the channel bed — the exact
   steady drag/push a gripper on the stone's exposed top rim would perform — while a
   small yaw-keeping torque (<= 0.03 N m, the wrist holding the wheel square) prevents
   the 40 mm disc from cocking and drawer-jamming in the 48 mm channel. The servo is
   speed-capped so the stone approaches the pocket quasi-statically; on stall it backs
   off and squares up before resuming (a wedge only tightens under more force).
2. CAPTURE (gravity + contact, hands-off): the force is cut 90 mm short of the pocket
   center — the gripper releases and the stone COASTS the final stretch (rolling on a
   level bed is loss-free, so it arrives at ~the release speed). At the pocket gap it
   drops ~10 mm onto the pocket floor and the bed edges bound it laterally; the
   seating (position, height band, uprightness, settled) is produced entirely by
   ballistics and contact. At the release speed the stone's kinetic energy (~2.5 mJ)
   is ~20x short of the ~49 mJ needed to climb back out, so the capture is
   dynamically guaranteed — never written.
3. RESTRAINT: the red decoy disc is never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_microwave_i306.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stone_door_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def stone_loc() -> torch.Tensor:
        return scene._vault_local(scene.stone.data.root_pos_w)[0]

    def stone_z() -> float:
        return float(scene.stone.data.root_pos_w[0, 2] - scene.env_origins[0, 2])

    def report(tag: str) -> None:
        sl = stone_loc()
        dl = scene._vault_local(scene.decoy.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | stone_loc=({float(sl[0]):+.3f},{float(sl[1]):+.3f}) "
              f"z={stone_z():.3f} decoy_y={float(dl[1]):+.3f} "
              f"seated={bool(scene.stone_seated()[0])} settled={bool(scene.settled()[0])} "
              f"moved={bool(scene._moved[0])} near={bool(scene._near[0])} "
              f"in={bool(scene._in[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # discs settle onto the bed
    vp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
    vq = scene.vault.data.root_quat_w[0]
    vyaw = math.degrees(2.0 * math.atan2(float(vq[3]), float(vq[0])))
    sl = stone_loc()
    dl = scene._vault_local(scene.decoy.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"vault=({float(vp[0]):+.3f},{float(vp[1]):+.3f}) yaw={vyaw:+.1f}deg "
          f"stone_y={float(sl[1]):+.3f} decoy_y={float(dl[1]):+.3f} "
          f"side={'+' if float(scene.stone_side[0]) > 0 else '-'}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert stone_z() > 0.067, f"stone must start ON the bed, z={stone_z():.3f}"
    assert abs(float(sl[1])) > 0.10, "stone must start well away from the pocket"
    s0 = print_score("P0 reset+settle (stone on the bed, away from the pocket)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: roll the stone along the channel into the pocket ------------
    # The gripper's grip does two things: (a) a velocity-servo push at the CoM along the
    # track, and (b) a yaw-keeping torque that keeps the disc SQUARE to the channel.
    # (b) is load-bearing: the disc (40 mm) in the channel (48 mm) is a classic drawer
    # jam — at ~4 deg of yaw its rim wedges diagonally between the rails and a centered
    # push only tightens the wedge (self-locking, seen as a dead stop under 5 N on
    # seed 2). A hand rolling a wheel holds it square; the torque servo does the same.
    # IMPORTANT — wrench frame: `is_global=True` wrenches on a ROLLING body are applied
    # through a stale rotation reference, so the "world" force rotates with the disc's
    # roll angle (the stone becomes a driven pendulum in roll-angle space and parks
    # where the force points vertically — measured stall at exactly (pi/2)*r past the
    # start on seed 2; probe scenario D reproduces it, scenario E with per-step
    # body-frame wrenches rolls clean). ALL wrenches below are therefore converted to
    # the stone's body frame with the CURRENT quat, every step.
    ez = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3)
    ex = torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3)

    def apply_wrench_w(force_w: torch.Tensor, torque_w: torch.Tensor) -> None:
        """Apply a desired WORLD wrench via body-frame conversion at the current quat."""
        q = scene.stone.data.root_quat_w
        q_c = torch.cat([q[:, :1], -q[:, 1:]], dim=1)
        scene.stone.set_external_force_and_torque(
            quat_apply(q_c, force_w.view(n, 3)).view(n, 1, 3),
            quat_apply(q_c, torque_w.view(n, 3)).view(n, 1, 3), env_ids=all_ids)

    def yaw_error() -> tuple[float, float]:
        """Signed yaw of the disc axis vs the channel normal (rad), and wz."""
        ax = quat_apply(scene.stone.data.root_quat_w, ez)[0, :2]
        cn = quat_apply(scene.vault.data.root_quat_w, ex)[0, :2]
        if float((ax * cn).sum()) < 0:
            cn = -cn
        psi = math.atan2(float(cn[0] * ax[1] - cn[1] * ax[0]), float((ax * cn).sum()))
        return psi, float(scene.stone.data.root_ang_vel_w[0, 2])

    def square_torque(psi: float, wz: float) -> torch.Tensor:
        tau = max(-0.03, min(0.03, -(0.30 * psi + 0.010 * wz)))
        return (tau * ez).view(n, 1, 3)

    v_des, k_gain, f_max = 0.08, 5.0, 4.0
    near_printed = False
    s_near = s0
    last_absy = abs(float(sl[1]))
    last_check = 0
    entered = False
    for i in range(1800):
        loc = stone_loc()
        absy = abs(float(loc[1]))
        if stone_z() < c.in_z and absy < c.pocket_hl:
            entered = True
            scene.stone.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                      env_ids=all_ids)
            print(f"[solve] stone entered the pocket at step {i} "
                  f"(y={float(loc[1]):+.3f}, z={stone_z():.3f}); force cut", flush=True)
            break
        # velocity-servo force at the CoM, along the track toward the pocket.
        # Final approach (|y| < 0.09): FORCE-FREE COAST — the gripper releases and the
        # stone rolls the last few cm; the pocket's gravity detent does the capture.
        # (A driven entry excites a contact artifact that accelerates the disc and can
        # vault it back out of the pocket; a slow ballistic entry is 20x under the
        # climb-out energy and captures unconditionally.)
        sgn = -1.0 if float(loc[1]) > 0 else 1.0
        dir_w = quat_apply(scene.vault.data.root_quat_w,
                           torch.tensor([[0.0, sgn, 0.0]], device=device).expand(n, 3))
        v_along = float((scene.stone.data.root_lin_vel_w[0] * dir_w[0]).sum())
        psi, wz = yaw_error()
        if absy < 0.09:
            f_mag = 0.0
            tq = torch.zeros(n, 3, device=device)  # hands-off coast: no steering either
            if v_along < 0.01:  # stalled in the coast zone: gentle squared re-nudge
                f_mag = 0.8
                tq = square_torque(psi, wz).view(n, 3)
            apply_wrench_w(f_mag * dir_w, tq)
        else:
            f_mag = max(-f_max, min(f_max, k_gain * (v_des - v_along)))
            apply_wrench_w(f_mag * dir_w, square_torque(psi, wz).view(n, 3))
        env.step(no_action)
        dense = absy < 0.075  # per-step trace at the pocket edge: find the ejection event
        if i % 60 == 0 or dense:
            vel = scene.stone.data.root_lin_vel_w[0]
            wv = scene.stone.data.root_ang_vel_w[0]
            print(f"[solve]   dbg i={i:4d} y={float(loc[1]):+.4f} x={float(loc[0]):+.4f} "
                  f"z={stone_z():.4f} v={v_along:+.3f} vz={float(vel[2]):+.3f} "
                  f"F={f_mag:+.2f} psi={math.degrees(psi):+.1f} "
                  f"|w|={float(wv.norm()):.2f}", flush=True)
        if not near_printed and bool(scene._near[0]):
            near_printed = True
            s_near = print_score("P1a stone driven near the pocket")
            assert s_near >= s0 - 1e-6, "score decreased during the approach"
        if i - last_check >= 180:  # stall watch
            if last_absy - absy < 0.005:
                # A wedge only grinds in harder under more force: SQUARE first
                # (back off + zero the yaw with the wrist), then resume gently.
                psi, wz = yaw_error()
                print(f"[solve] stall at |y|={absy:.3f} psi={math.degrees(psi):+.1f}deg; "
                      f"squaring + backoff, then resume", flush=True)
                for _ in range(45):
                    psi, wz = yaw_error()
                    apply_wrench_w(-1.2 * dir_w, square_torque(psi, wz).view(n, 3))
                    env.step(no_action)
                scene.stone.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                          env_ids=all_ids)
                step(30)
                v_des = min(v_des + 0.02, 0.14)
                k_gain = min(k_gain * 1.2, 10.0)
                f_max = min(f_max * 1.25, 5.0)
                print(f"[solve] resume: v_des={v_des:.2f} gain={k_gain:.1f} "
                      f"f_max={f_max:.1f} psi={math.degrees(psi):+.1f}deg", flush=True)
            last_absy, last_check = absy, i
    scene.stone.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)
    step(150)  # drop + settle, hands-off
    report("pocket-entry")
    if not entered or not bool(scene.stone_seated()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (stone never seated in the pocket)", flush=True)
        os._exit(1)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after seating)", flush=True)
        os._exit(1)
    s1 = print_score("P1 stone dropped into the pocket and seated")
    assert s1 >= s_near - 1e-6, "score decreased across the capture"

    # ---------------- phase 2: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s2 = print_score("P2 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s2 >= s1 - 1e-6
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
