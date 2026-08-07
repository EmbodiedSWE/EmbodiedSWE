"""Teleport solution for FacingLaneScene (sim_gen task `put_groceries_in_cupboard_i329`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. TRANSPORT (teleport): one pose write stages the red tin hovering in the front
   window, 47 mm OUTSIDE the sill, bottom 4 mm above the sill top. Verifiably
   credit-free (asserted: no entry latch, zero compression, score unchanged).
2. PRESS THE SPRING OPEN (contact dynamics — never teleported): from the staged hover
   the tin is driven by an applied wrench that emulates the arm's grasp — gravity
   compensation + a z hold (the "hand" carrying the tin), a velocity-limited
   horizontal push, lateral centering, and a small attitude-hold torque (a gripper
   constrains orientation). The tin's face meets the orange pusher plate and every
   millimetre of cavity is opened by pressing the plate back against its spring
   THROUGH the tin — the spring resists the whole way (readback: compression grows
   from 0 to ~63 mm under contact). Nothing is bypassed: the slot is created by the
   press, exactly as the arm must create it.
3. LOWER AND RELEASE (contact dynamics): still holding the compression through the
   tin (feedforward = the live spring force), the z hold ramps the tin down behind
   the sill onto the lane floor; then ALL applied wrenches are cut at once. From that
   instant nothing touches the tin: the returning spring closes the remaining gap and
   clamps the tin's front face against the sill inner wall — the store-facing state
   the rubric reads back (plate compression + front gap + stillness).

The green and blue decoy tins are never touched. Prints `SIM_GEN_SCORE <score>` at
each phase boundary (non-decreasing: the scene's credit is latched), then holds
HANDS-OFF for >= 3.3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.facing_lane")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def red_loc() -> torch.Tensor:
        return scene._bay_local(scene.red)[0]

    def comp() -> float:
        return float(scene.compression()[0])

    def report(tag: str) -> None:
        rl = red_loc()
        print(f"[solve] {tag:12s} | red_loc=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):.3f}) comp={comp() * 1000:.1f}mm "
              f"entered={bool(scene._entered(scene.red)[0])} "
              f"seated={bool(scene._seated_now(scene.red)[0])} "
              f"clamped={bool(scene._clamped_now()[0])} "
              f"press_max={float(scene._press_max[0]):.3f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    bay = torch.tensor(c.bay_pos, device=device)

    def hold_wrench(z_des: float, fu: float, fv: float) -> None:
        """Apply the grasp-emulation wrench to the red tin: gravity compensation + z
        PD to z_des (bay-local), the given horizontal forces (bay frame), and a small
        attitude-hold torque toward upright/zero-yaw (world frame)."""
        pos = scene.red.data.root_pos_w - scene.env_origins
        vel = scene.red.data.root_lin_vel_w
        z = float(pos[0, 2]) - float(bay[2])
        fz = c.tin_mass * 9.81 + 300.0 * (z_des - z) - 8.0 * float(vel[0, 2])
        fz = max(0.0, min(8.0, fz))
        f = torch.tensor([fu, fv, fz], device=device).view(1, 1, 3).expand(n, 1, 3)
        q = scene.red.data.root_quat_w[0]
        sgn = 1.0 if float(q[0]) >= 0.0 else -1.0
        omega = scene.red.data.root_ang_vel_w[0]
        tau = (-0.8 * 2.0 * sgn * q[1:4] - 0.05 * omega).clamp(-0.25, 0.25)
        scene.red.set_external_force_and_torque(
            f.contiguous(), tau.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    r0 = (scene.red.data.root_pos_w - scene.env_origins)[0]
    g0 = (scene.green.data.root_pos_w - scene.env_origins)[0]
    b0 = (scene.blue.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"red=({float(r0[0]):+.3f},{float(r0[1]):+.3f}) "
          f"green=({float(g0[0]):+.3f},{float(g0[1]):+.3f}) "
          f"blue=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) comp={comp() * 1000:.1f}mm",
          flush=True)
    report("reset")
    assert comp() < 0.005, "plate did not rest at home after reset"
    assert not bool(scene._entered(scene.red)[0]), "red tin spawned inside the lane"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: TRANSPORT to the staged hover (teleport, outside) ------------
    # One pose write: red tin into the front window, 47 mm outside the sill, bottom
    # 4 mm above the sill top, square to the lane. Credit-free (asserted).
    glide_z = c.sill_h + 0.004 + c.tin_h / 2  # bay-local CoM height on the glide line
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = bay[0] - 0.075
    st[:, 1] = bay[1]
    st[:, 2] = bay[2] + glide_z
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.red.write_root_state_to_sim(st, all_ids)
    # hold immediately (the tin is "in hand"): one step to let buffers refresh
    hold_wrench(glide_z, 0.0, 0.0)
    step(5)
    report("staged")
    assert comp() < 0.004, "staging already compressed the spring (must be outside)"
    assert not bool(scene._entered(scene.red)[0]), \
        "staging pose already reads as entered (teleport must stay outside the sill)"
    s1 = print_score("P1 red tin transported to the staged hover (outside the sill)")
    assert s1 <= s0 + 1e-6, "staging earned credit (it must not)"

    # ---------------- phase 2: PRESS THE SPRING OPEN (contact dynamics) ---------------------
    # Velocity-limited horizontal push through the window; the tin's face meets the
    # plate and drives it back against the spring. Exit only on COMPRESSION readback:
    # the cavity provably exists and the tin is deep enough to descend behind the sill.
    comp_goal = c.tin_w + 0.008  # open 8 mm more than the tin needs
    v_des, f_cap = 0.10, 5.0
    best, last_bump = -1.0, 0
    pressed = False
    for i in range(1800):
        cm = comp()
        rl = red_loc()
        if cm >= comp_goal and float(rl[0]) - c.tin_w / 2 >= c.sill_t + 0.003:
            pressed = True
            break
        vel = scene.red.data.root_lin_vel_w[0]
        fu = min(f_cap, 2.5 + 32.0 * cm) if float(vel[0]) < v_des else 32.0 * cm
        fv = max(-2.0, min(2.0, -30.0 * float(rl[1]) - 3.0 * float(vel[1])))
        hold_wrench(glide_z, fu, fv)
        env.step(no_action)
        prog = cm + max(0.0, float(rl[0]))
        if prog > best + 0.002:
            best, last_bump = prog, i
        elif i - last_bump > 240:  # stalled: push harder
            f_cap = min(f_cap + 0.7, 9.0)
            last_bump = i
            print(f"[solve] stall at comp={cm * 1000:.1f}mm u={float(rl[0]):+.3f} -> "
                  f"f_cap={f_cap:.1f} N", flush=True)
    report("pressed")
    assert pressed, "horizontal press never opened the spring cavity to the goal depth"
    assert float(scene._press_max[0]) > 0.9, "press latch did not register the compression"
    s2 = print_score("P2 spring pressed open through the tin (cavity created)")
    assert s2 >= s1 - 1e-6, "score decreased across the press"

    # ---------------- phase 3: LOWER BEHIND THE SILL AND RELEASE (contact dynamics) ---------
    # Still holding the spring back through the tin (feedforward = live spring force),
    # ramp the z hold down until the tin stands on the lane floor; then cut everything.
    u_hold = float(red_loc()[0])
    seat_z = c.tin_h / 2 + 0.002
    for i in range(240):
        rl = red_loc()
        frac = min(1.0, i / 150.0)
        z_des = glide_z + (seat_z - glide_z) * frac
        vel = scene.red.data.root_lin_vel_w[0]
        fu = 32.0 * comp() + 200.0 * (u_hold - float(rl[0])) - 15.0 * float(vel[0])
        fu = max(-2.0, min(6.0, fu))
        fv = max(-2.0, min(2.0, -30.0 * float(rl[1]) - 3.0 * float(vel[1])))
        hold_wrench(z_des, fu, fv)
        env.step(no_action)
        if frac >= 1.0 and float(rl[2]) <= seat_z + 0.004:
            break
    report("lowered")
    rl = red_loc()
    assert float(rl[2]) <= seat_z + 0.010, "tin never reached the lane floor"
    assert float(rl[0]) - c.tin_w / 2 >= c.sill_t - 0.002, \
        "tin descended on the wrong side of the sill"
    # RELEASE: cut all applied wrenches at once — from here the spring does the clamp.
    scene.red.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
    step(240)
    report("released")
    s3 = print_score("P3 lowered behind the sill, released — spring clamped the tin")
    assert s3 >= s2 - 1e-6, "score decreased across the lower/release"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after release+settle)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) ------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
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
    main()
