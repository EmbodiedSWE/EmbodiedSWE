"""Teleport solution for GuardCapScene (sim_gen task `light_bulb_in_i381`) — the
task's legitimacy certificate.

Teleports are TRANSPORT ONLY (the stand-ins for the arm's grasped carries):
  - the fresh bulb is teleported from its lying spawn to a hover 20 mm ABOVE the
    brass pad, upright — then RELEASED: gravity and contact seat it standing on the
    pad (the actual standing rest is settled physics, not a written pose);
  - the cage is teleported to a hover with its rim well ABOVE the standing bulb's
    knob-free descent line, key-aligned in yaw — then a 6-DOF external-wrench servo
    (gravity feed-forward + PD position + PD attitude) lowers it THROUGH the key
    bridges to the deck under live contact dynamics, and releases; the flush seat is
    whatever physics says it is.
No load-bearing state is ever written: both terminal rests (bulb standing, cage
seated) are reached by released/force-driven dynamics.

Servo notes (external-wrench plant recipe): the scene sets
`enable_external_forces_every_iteration`; the cage has authored CoM + diagonal
inertia; wrenches are encoded into the CURRENT body frame (the house convention).
Gains are discretely stable: kd*dt/m = 8/(120*0.25) = 0.27 < 1 and
kdw*dt/I = 0.015/(120*6e-4) = 0.21 < 1.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.5 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.light_bulb_in_i381.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bulb_guard_cap")().build(num_envs=args.num_envs,
                                                    device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)
    G = 9.81

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench_on(body, f_w: torch.Tensor, t_w: torch.Tensor) -> None:
        """World-frame CoM force + torque, encoded into the CURRENT body frame (the
        house convention: the engine treats the given wrench as body-frame)."""
        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).unsqueeze(1),
            quat_apply_inverse(q, t_w).unsqueeze(1), env_ids=all_ids)

    def release(body) -> None:
        body.set_external_force_and_torque(zero_rows, zero_rows, env_ids=all_ids)

    def loc_of(body) -> torch.Tensor:
        return scene._plinth_local(body.data.root_pos_w)[0]

    def plinth_to_world(lx: float, ly: float, lz: float) -> torch.Tensor:
        v = torch.tensor([[lx, ly, lz]], device=device)
        return scene.plinth.data.root_pos_w + quat_apply(
            scene.plinth.data.root_quat_w, v)

    def teleport(body, lx: float, ly: float, lz: float) -> None:
        """TRANSPORT teleport: place `body` at a plinth-frame point, upright with the
        plinth's yaw, zero velocity (the stand-in for a grasped carry)."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = plinth_to_world(lx, ly, lz)
        st[:, 3:7] = scene.plinth.data.root_quat_w
        body.write_root_state_to_sim(st, all_ids)

    def report(tag: str) -> None:
        b = loc_of(scene.bulb)
        k = loc_of(scene.cage)
        print(f"[solve] {tag:14s} | bulb_loc=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):+.3f}) cage_loc=({float(k[0]):+.3f},{float(k[1]):+.3f},"
              f"{float(k[2]):+.3f})"
              f" latches=({float(scene.stand_latch[0]):.0f},"
              f"{float(scene.cap_latch[0]):.0f})"
              f" standing={bool(scene.bulb_standing()[0])}"
              f" capped={bool(scene.cage_capped()[0])}"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    b, d, k = loc_of(scene.bulb), loc_of(scene.dead), loc_of(scene.cage)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"bulb=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
          f"dead=({float(d[0]):+.3f},{float(d[1]):+.3f}) "
          f"cage=({float(k[0]):+.3f},{float(k[1]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: stand the bulb on the pad (carry + released drop) -----------
    stood = False
    for attempt, hover in enumerate((0.020, 0.014, 0.026)):
        teleport(scene.bulb, 0.0, 0.0, c.pad_h + hover)
        step(2)  # hand it to the engine
        release(scene.bulb)
        step(90)  # gravity drop + settle: the standing rest is real contact physics
        stood = bool(scene.bulb_standing()[0])
        print(f"[solve] stand attempt {attempt} (hover {hover * 1000:.0f} mm): "
              f"standing={stood}", flush=True)
        if stood:
            break
    assert stood, "bulb never settled standing on the pad"
    report("stood")
    s1 = print_score("P1 bulb stood on the pad (carried, released, gravity-seated)")
    assert s1 >= 0.25 - 1e-6 and s1 >= s0 - 1e-6

    # ---------------- phase 2: key-align and lower the cage (carry + wrench servo) ---------
    KP, KD, FCAP = 60.0, 8.0, 8.0  # kd*dt/m = 0.27 < 1
    KR, KDW, TCAP = 0.10, 0.015, 0.05  # kdw*dt/I = 0.21 < 1
    LEAD = 0.025
    HOVER_Z = 0.105  # rim start: above the standing bulb's globe top (0.088)

    def descend() -> bool:
        """6-DOF wrench-servo descent from the hover to the flush seat: gravity
        feed-forward + PD position toward a reference that ramps down at 0.05 m/s +
        PD attitude toward the plinth's yaw. Releases when the rim is in the capped
        window and slow; returns capped-after-settle."""
        q_des = scene.plinth.data.root_quat_w
        z_ref = HOVER_Z
        for i in range(1400):
            kloc = loc_of(scene.cage)
            v = scene.cage.data.root_lin_vel_w[0]
            w = scene.cage.data.root_ang_vel_w[0]
            in_window = c.cap_z_lo <= float(kloc[2]) <= c.cap_z_hi
            if in_window and float(v.norm()) < 0.03 and float(w.norm()) < 0.3:
                break
            z_ref = max(z_ref - 0.05 / 120.0, c.cap_z_lo + 0.0005)
            p_des = plinth_to_world(0.0, 0.0, z_ref)[0]
            err = p_des - scene.cage.data.root_pos_w[0]
            err = err.clamp(-LEAD, LEAD)
            f = torch.tensor([0.0, 0.0, c.cage_mass * G], device=device) \
                + KP * err - KD * v
            fn = f.norm()
            if fn > FCAP:
                f = f * (FCAP / fn)
            q_err = quat_mul(q_des, quat_conjugate(scene.cage.data.root_quat_w))[0]
            sgn = 1.0 if float(q_err[0]) >= 0.0 else -1.0
            rot_vec = 2.0 * sgn * q_err[1:4]
            t = KR * rot_vec - KDW * w
            tn = t.norm()
            if tn > TCAP:
                t = t * (TCAP / tn)
            wrench_on(scene.cage, f.unsqueeze(0).expand(n, 3),
                      t.unsqueeze(0).expand(n, 3))
            env.step(no_action)
        release(scene.cage)
        step(90)
        kloc = loc_of(scene.cage)
        print(f"[solve] descent done after {i + 1} servo steps: rim z "
              f"{float(kloc[2]) * 1000:+.1f} mm, capped="
              f"{bool(scene.cage_capped()[0])}", flush=True)
        return bool(scene.cage_capped()[0])

    capped = False
    for attempt in range(3):
        teleport(scene.cage, 0.0, 0.0, HOVER_Z)
        step(2)
        print(f"[solve] cap attempt {attempt}: cage staged at rim z "
              f"{HOVER_Z * 1000:.0f} mm, key-aligned", flush=True)
        capped = descend()
        if capped and bool(scene.bulb_standing()[0]):
            break
    assert capped, "cage never seated flush inside the curb"
    assert bool(scene.bulb_standing()[0]), "bulb lost its stand during the descent"
    report("capped")
    s2 = print_score("P2 cage key-aligned and servo-lowered to the flush seat")
    assert s2 >= 0.45 - 1e-6 and s2 >= s1 - 1e-6

    # ---------------- phase 3: settle to success -------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s3 = print_score("P3 all settled")
    assert s3 >= s2 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after stand+cap+settle)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.5 simulated seconds, hands-off) -----------
    hold, flickers = True, 0
    for i in range(420):  # 420 substeps = 3.5 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"standing={bool(scene.bulb_standing()[0])} "
                      f"capped={bool(scene.cage_capped()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"blin={float(scene.bulb.data.root_lin_vel_w[0].norm()):.4f} "
                      f"klin={float(scene.cage.data.root_lin_vel_w[0].norm()):.4f} "
                      f"kang={float(scene.cage.data.root_ang_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s4 = print_score("P4 persistence 3.5 s")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
