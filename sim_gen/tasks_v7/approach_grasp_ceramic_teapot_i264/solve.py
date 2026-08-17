"""Teleport solution for TeapotBayonetScene (sim_gen task
`approach_grasp_ceramic_teapot_i264`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY, and the single lid teleport ends in FREE
SPACE: the lid is carried to a hover pose on the pot axis, ~1.8 cm above the rim,
yaw-matched to the pot (lugs over the notches), velocities zeroed. Everything
load-bearing happens through CONTACT DYNAMICS:
  - INSERT: a gentle velocity-servoed press (gravity feedforward + xy centering +
    a weak yaw-hold torque) lowers the lid through the keyed throat — the lugs pass
    the notch gaps, the plug rides the ring aperture, and the pilot nose lands on
    the internal ledge. If the descent wedges, the lid is re-hovered and re-pressed.
  - TWIST: with a light continued press keeping the nose on the ledge, an angle-PD
    yaw torque (gain-escalating on stall, clamped to +-0.15 N m) rotates the lid
    about the pot axis; the lugs sweep under the solid ring through the chamber
    until past the lock threshold (target ~43 deg, credit line 36 deg, internal
    stops at ~48 deg). If one direction jams, the other is tried — the bayonet is
    symmetric.
  - RELEASE: yaw braked, wrench cleared, and the lid left alone; it rests seated
    and locked. success() judges only this live state.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.approach_grasp_ceramic_teapot_i264.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.teapot_bayonet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    g = 9.81

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def lid_p() -> torch.Tensor:
        return scene._lid_pos()[0]

    def pot_p() -> torch.Tensor:
        return scene._pot_pos()[0]

    def pot_yaw() -> float:
        return float(scene._yaw(scene.pot.data.root_quat_w)[0])

    def signed_rel_yaw() -> float:
        """Signed lid-vs-pot yaw folded into (-pi/2, pi/2] — the working coordinate
        of the 2-fold-symmetric bayonet (0 = lugs over the notches)."""
        d = float(scene._yaw(scene.lid.data.root_quat_w)[0]) - pot_yaw()
        return (d + math.pi / 2) % math.pi - math.pi / 2

    def fold_deg() -> float:
        return float(torch.rad2deg(scene.rel_yaw_fold())[0])

    def clear_wrench() -> None:
        scene.lid.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def apply_wrench(f3, t3) -> None:
        f = torch.tensor(f3, device=device, dtype=torch.float32)
        tq = torch.tensor(t3, device=device, dtype=torch.float32)
        scene.lid.set_external_force_and_torque(
            f.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            tq.view(1, 1, 3).expand(n, 1, 3).contiguous(),
            env_ids=all_ids, is_global=True)

    def report(tag: str) -> None:
        lp, pp = lid_p(), pot_p()
        dxy = float((lp[:2] - pp[:2]).norm())
        print(f"[solve] {tag:14s} | lid=({float(lp[0]):+.3f},{float(lp[1]):+.3f},"
              f"{float(lp[2]):.4f}) dxy={dxy:.4f} fold={fold_deg():5.1f}deg "
              f"up_z={float(scene.lid_up_z()[0]):.4f} "
              f"carried={float(scene.carried[0]):.0f} entered={float(scene.entered[0]):.0f} "
              f"rot_max={math.degrees(float(scene.rot_max[0])):5.1f}deg "
              f"seated={bool(scene.seated()[0])} locked={bool(scene.locked()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    m = c.lid_mass
    K = 60.0  # velocity-servo gain; K*dt = 0.5 < 1 (wrench acts one substep late)

    # ----- contact primitives ---------------------------------------------------------------
    def hover(dz: float = 0.018) -> None:
        """TRANSPORT: one teleport to a free-space hover — lid origin on the pot axis
        with the nose bottom `dz` above the rim, yaw matched to the pot (lugs over
        the notches), velocities zero. Everything below the lid is open throat."""
        pp = pot_p()
        py = pot_yaw()
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1] = float(pp[0]), float(pp[1])
        st[:, 2] = c.ring_z1 + dz
        st[:, 3] = math.cos(py / 2)
        st[:, 6] = math.sin(py / 2)
        st[:, 0:3] += scene.env_origins
        scene.lid.write_root_state_to_sim(st, all_ids)

    def press(v_down: float, max_steps: int, stop_z: float | None,
              yaw_target: float | None = 0.0) -> bool:
        """CONTACT: velocity-servo the lid downward at `v_down` (gravity
        feedforward so the servo tracks), with xy centering onto the pot axis and a
        weak yaw-hold torque at `yaw_target`. Returns True once the origin is below
        `stop_z` and centred (or after `max_steps`, holding the press)."""
        for _ in range(max_steps):
            lp, pp = lid_p(), pot_p()
            if stop_z is not None and float(lp[2]) < stop_z and \
                    float((lp[:2] - pp[:2]).norm()) < 0.008:
                return True
            v = scene.lid.data.root_lin_vel_w[0]
            vx_des = min(max(3.0 * (float(pp[0]) - float(lp[0])), -0.06), 0.06)
            vy_des = min(max(3.0 * (float(pp[1]) - float(lp[1])), -0.06), 0.06)
            fx = m * K * (vx_des - float(v[0]))
            fy = m * K * (vy_des - float(v[1]))
            fz = m * K * (-v_down - float(v[2])) + m * g  # ff cancels gravity
            fx = min(max(fx, -3.0), 3.0)
            fy = min(max(fy, -3.0), 3.0)
            fz = min(max(fz, -3.0 + m * g), 3.0 + m * g)
            tz = 0.0
            if yaw_target is not None:
                wz = float(scene.lid.data.root_ang_vel_w[0, 2])
                tz = min(max(0.06 * (yaw_target - signed_rel_yaw()) - 0.006 * wz,
                             -0.05), 0.05)
            apply_wrench((fx, fy, fz), (0.0, 0.0, tz))
            env.step(no_action)
        return stop_z is None

    def insert() -> bool:
        """Hover-teleport aligned above the mouth, then press the lid down through
        the keyed throat onto the ledge. Re-hovers and retries on a wedge."""
        for attempt in range(5):
            hover()
            step(2)
            ok = press(v_down=0.08, max_steps=500, stop_z=c.seat_z + 0.006)
            lp = lid_p()
            print(f"[solve] insert attempt {attempt}: z={float(lp[2]):.4f} "
                  f"fold={fold_deg():4.1f}deg ok={ok}", flush=True)
            if ok:
                press(v_down=0.04, max_steps=60, stop_z=None)  # firm the seat
                return True
        return False

    def twist() -> bool:
        """Rotate the seated lid past the lock threshold with an angle-PD yaw
        torque under a light press. Escalates the gain on stall (velocity-servo
        stall trap: escalate GAIN, not the clamp); tries the other direction if
        one jams. Returns True once folded rotation >= lock_min + 3 deg."""
        goal = c.lock_min_deg + 3.0  # overshoot the credit line (latch-lag margin)
        for sgn in (+1.0, -1.0):
            target = sgn * math.radians(43.0)
            kp, kp_max = 0.08, 0.90
            last_fold, stall = fold_deg(), 0
            for i in range(1400):
                fd = fold_deg()
                if fd >= goal:
                    print(f"[solve] twist: reached {fd:.1f} deg (dir {sgn:+.0f}, "
                          f"kp={kp:.2f}, {i} steps)", flush=True)
                    return True
                lp, pp = lid_p(), pot_p()
                v = scene.lid.data.root_lin_vel_w[0]
                wz = float(scene.lid.data.root_ang_vel_w[0, 2])
                fx = min(max(m * K * (3.0 * (float(pp[0]) - float(lp[0])) - float(v[0])),
                             -2.0), 2.0)
                fy = min(max(m * K * (3.0 * (float(pp[1]) - float(lp[1])) - float(v[1])),
                             -2.0), 2.0)
                fz = m * K * (-0.04 - float(v[2])) + m * g  # light seat press
                tz = min(max(kp * (target - signed_rel_yaw()) - 0.010 * wz, -0.15), 0.15)
                apply_wrench((fx, fy, fz), (0.0, 0.0, tz))
                env.step(no_action)
                if i % 90 == 89:
                    if fold_deg() - last_fold < 1.5:
                        stall += 1
                        kp = min(kp * 1.7, kp_max)
                        print(f"[solve] twist stall {stall} (dir {sgn:+.0f}) at "
                              f"{fold_deg():.1f} deg -> kp={kp:.2f}", flush=True)
                    else:
                        stall = 0
                    last_fold = fold_deg()
                    if stall >= 5:
                        break  # gain maxed and still pinned — try the other way
            print(f"[solve] twist dir {sgn:+.0f} gave up at {fold_deg():.1f} deg",
                  flush=True)
        return False

    def brake() -> None:
        """Kill residual yaw rate (coast-overshoot trap), then go hands-off."""
        for _ in range(200):
            wz = float(scene.lid.data.root_ang_vel_w[0, 2])
            v = scene.lid.data.root_lin_vel_w[0]
            if abs(wz) < 0.08 and float(v.norm()) < 0.05:
                break
            fz = m * K * (0.0 - float(v[2])) + m * g
            apply_wrench((0.0, 0.0, fz), (0.0, 0.0, min(max(-0.02 * wz, -0.1), 0.1)))
            env.step(no_action)
        clear_wrench()

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(90)
    lp0, pp0 = lid_p(), pot_p()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"pot=({float(pp0[0]):+.3f},{float(pp0[1]):+.3f}) "
          f"pot_yaw={math.degrees(pot_yaw()):+.1f}deg "
          f"lid=({float(lp0[0]):+.3f},{float(lp0[1]):+.3f},{float(lp0[2]):.3f}) "
          f"fold={fold_deg():.1f}deg", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")

    # ---------------- phase 1: transport hover + keyed insertion ----------------------------
    if not insert():
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (insertion wedged after retries)", flush=True)
        os._exit(1)
    report("inserted")
    s1 = print_score("P1 hover + keyed insertion")
    assert s1 >= s0 - 1e-6, "score decreased across insertion"

    # ---------------- phase 2: twist to the stop --------------------------------------------
    if not twist():
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (twist never reached the lock window)", flush=True)
        os._exit(1)
    brake()
    report("twisted")
    s2 = print_score("P2 twist past the lock threshold")
    assert s2 >= s1 - 1e-6, "score decreased across the twist"

    # ---------------- phase 3: hands-off settle ---------------------------------------------
    step(180)
    report("settled")
    s3 = print_score("P3 release + settle")
    assert s3 >= s2 - 1e-6, "score decreased across the settle"
    if not bool(scene.success()[0]):
        step(240)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settle)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.4 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 41 = 410 substeps = 3.42 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.4 s")
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
