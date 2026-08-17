"""Teleport solution for KnifeStandScene (sim_gen task `track_knife_i402`) — the
task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, pose writes that satisfy NO gate): one pose write lifts the
   cap panel off the rack to a free HOVER above the standing base panel (15+ mm above
   its top edge — far outside every mesh/engage clause); later one pose write lifts
   the knife off its pedestal to a free hover above the built stand (above the judged
   notch band). Neither teleport creates contact or credit.
2. PRESS-FIT MESH (contact + applied wrench): the hovering cap panel is HELD by a
   6-DOF PD wrench (the analog of a parallel-jaw grip on its exposed edge: xy/yaw/
   uprighting regulation + gravity feedforward) while its z target ramps down; the
   slot walls GUIDE the slab (slick joinery, ~2 mm play per side) until the cap seats
   on the floor through the base's foot channel. The wrench is then dropped and the
   joint must stand on its own.
3. KNIFE SEAT (contact + applied wrench, then gravity): the hovering knife is lowered
   by the same kind of gentle held wrench onto the LIVE notch line (recomputed from
   both panels' actual poses); the last millimetres are a free gravity settle into
   the two notches. Hands off before judging.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.track_knife_i402.solve --headless [--seed N]
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
    import scene as scene_mod

_wrap = scene_mod._wrap
_yaw_of = scene_mod._yaw_of

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.knife_stand")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        pa, pb, pk = loc(scene.panel_a), loc(scene.panel_b), loc(scene.knife)
        print(f"[solve] {tag:12s} | A=({float(pa[0]):+.3f},{float(pa[1]):+.3f},"
              f"{float(pa[2]):.3f}) B=({float(pb[0]):+.3f},{float(pb[1]):+.3f},"
              f"{float(pb[2]):.3f}) K=({float(pk[0]):+.3f},{float(pk[1]):+.3f},"
              f"{float(pk[2]):.3f}) engaged={bool(scene.engaged()[0])} "
              f"meshed={bool(scene.meshed()[0])} seated={bool(scene.knife_seated()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    last_score = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        assert s >= last_score[0] - 1e-6, f"score regressed at {tag}: {last_score[0]} -> {s}"
        last_score[0] = s
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def place(body, pos, quat) -> None:
        """TRANSPORT-ONLY pose write: free hover, zero velocity, satisfies no gate."""
        st = torch.zeros(n, 13, device=device)
        st[0, 0:3] = torch.tensor(pos, device=device)
        st[0, 3:7] = torch.tensor(quat, device=device)
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def hold(body, f_world: torch.Tensor, t_world: torch.Tensor) -> None:
        """Apply a desired WORLD wrench encoded in the BODY frame (is_global=False):
        immune to this stack's world-frame wrench drag."""
        q = body.data.root_quat_w
        f = torch.zeros(n, 1, 3, device=device)
        t = torch.zeros(n, 1, 3, device=device)
        f[:, 0, :] = quat_apply_inverse(q, f_world.unsqueeze(0))
        t[:, 0, :] = quat_apply_inverse(q, t_world.unsqueeze(0))
        body.set_external_force_and_torque(f.contiguous(), t.contiguous(),
                                           env_ids=all_ids, is_global=False)

    def release(body) -> None:
        body.set_external_force_and_torque(zero3, zero3, env_ids=all_ids, is_global=False)

    def held_move(body, *, mass: float, xy_t, yaw_t, z_from: float, z_to: float,
                  rate: float, kp_xy: float, kd_xy: float, kp_z: float, kd_z: float,
                  k_up: float, k_yaw: float, f_cap: float, press_cap: float,
                  lift_cap: float, exit_z: float, tag: str, max_steps: int = 2000,
                  wiggle: float = 0.0) -> bool:
        """Hold `body` level, at yaw_t(), over xy_t() (LIVE-tracking callables) with a
        PD wrench while ramping its z target from z_from toward z_to (rate signed by
        the direction). fz is clamped SYMMETRICALLY: net down <= press_cap, net up <=
        lift_cap, so overshoot is always corrected. Exit when root z crosses exit_z
        (from above when descending, from below when ascending) at low speed.
        ABORTS early (returns False) if the body tips past 35 deg, runs away in xy,
        or stalls > 300 steps — a failed press must never grind the scene apart."""
        g = 9.81
        dt = 1.0 / 120.0
        down = z_to < z_from
        stall, best_z = 0, float(loc(body)[2])
        for i in range(max_steps):
            p = loc(body)
            q = body.data.root_quat_w
            v = body.data.root_lin_vel_w[0]
            w = body.data.root_ang_vel_w[0]
            upz = float(scene_mod._up_z(q)[0])
            tx0, ty0 = xy_t()
            exy = math.hypot(tx0 - float(p[0]), ty0 - float(p[1]))
            if upz < math.cos(math.radians(35.0)):
                print(f"[solve] {tag}: ABORT tipped (up_z={upz:.2f}, i={i})", flush=True)
                return False
            if exy > 0.06:
                print(f"[solve] {tag}: ABORT runaway (exy={exy:.3f}, i={i})", flush=True)
                return False
            if stall > 300:
                print(f"[solve] {tag}: ABORT stalled (z={float(p[2]):.4f}, i={i})",
                      flush=True)
                return False
            if i % 150 == 0:
                dy0 = float(_wrap(torch.tensor([yaw_t()], device=device)
                                  - _yaw_of(q)[0:1])[0])
                print(f"[solve] {tag}: i={i} z={float(p[2]):.4f} "
                      f"exy={exy * 1000:.1f}mm dyaw={math.degrees(dy0):+.1f}deg "
                      f"up_z={upz:.3f} stall={stall}", flush=True)
            if down:
                z_t = max(z_to, z_from - rate * dt * i)
            else:
                z_t = min(z_to, z_from + rate * dt * i)
            tx, ty = xy_t()
            fx = kp_xy * (tx - float(p[0])) - kd_xy * float(v[0])
            fy = kp_xy * (ty - float(p[1])) - kd_xy * float(v[1])
            fz = mass * g + kp_z * (z_t - float(p[2])) - kd_z * float(v[2])
            fz = max(mass * g - press_cap, min(mass * g + lift_cap, fz))
            fxy = torch.tensor([fx, fy], device=device)
            if float(fxy.norm()) > f_cap:
                fxy = fxy * (f_cap / float(fxy.norm()))
            # uprighting torque: rotate the body z-axis onto world z; yaw PD on top
            zb = quat_apply(q, ez.unsqueeze(0))[0]
            t_up = k_up * torch.linalg.cross(zb, ez) - 0.15 * k_up * w
            dyaw = float(_wrap(torch.tensor([yaw_t()], device=device)
                               - _yaw_of(q)[0:1])[0])
            t_up[2] += k_yaw * dyaw - 0.3 * k_yaw * float(w[2])
            if wiggle > 0.0 and stall > 60:
                t_up[2] += wiggle * math.sin(i * 0.35)
            hold(body, torch.tensor([float(fxy[0]), float(fxy[1]), fz], device=device),
                 t_up)
            env.step(no_action)
            z = float(loc(body)[2])
            if (down and z < best_z - 0.0015) or (not down and z > best_z + 0.0015):
                best_z, stall = z, 0
            else:
                stall += 1
            done = (z <= exit_z) if down else (z >= exit_z)
            if done and abs(float(body.data.root_lin_vel_w[0, 2])) < 0.08 \
                    and ((z_t <= exit_z + 0.002) if down else (z_t >= exit_z - 0.002)):
                print(f"[solve] {tag}: reached z={z:.4f} in {i} steps", flush=True)
                return True
        print(f"[solve] {tag}: NOT reached (z={float(loc(body)[2]):.4f})", flush=True)
        return False

    # ----- P0: settle + readback ----------------------------------------------------------------
    step(60)
    report("P0 settle")
    print_score("P0 settle")

    # ----- P1: transport the cap panel to a free hover over the base ----------------------------
    pa = loc(scene.panel_a)
    ayaw = float(_yaw_of(scene.panel_a.data.root_quat_w)[0])
    byaw = ayaw + math.pi / 2
    hover_z = c.height + 0.012  # bottom edge 12 mm above the base's top edge: free air
    place(scene.panel_b, (float(pa[0]), float(pa[1]), hover_z),
          (math.cos(byaw / 2), 0.0, 0.0, math.sin(byaw / 2)))
    report("P1 hover B")
    print_score("P1 hover B")

    # ----- P2: press-fit the cross-lap (held wrench vs slot-wall contact) -----------------------
    # LIVE targets: always chase the base panel's actual centre/yaw so a nudged base
    # cannot decouple the press from the slot line.
    def a_xy():
        p = loc(scene.panel_a)
        return float(p[0]), float(p[1])

    def a_cross_yaw():
        return float(_yaw_of(scene.panel_a.data.root_quat_w)[0]) + math.pi / 2

    meshed = False
    for attempt in range(3):
        held_move(scene.panel_b, mass=c.b_mass, xy_t=a_xy, yaw_t=a_cross_yaw,
                  z_from=float(loc(scene.panel_b)[2]), z_to=-0.004, rate=0.020,
                  kp_xy=45.0 + 15.0 * attempt, kd_xy=8.0,
                  kp_z=60.0 + 30.0 * attempt, kd_z=10.0,
                  # discrete-damping stability: c*dt/I < 1 on every axis
                  # (yaw: 0.045*dt/6e-4 = 0.63; tilt: 0.075*dt/1.3e-3 = 0.47)
                  k_up=0.5, k_yaw=0.15, f_cap=3.5,
                  press_cap=2.0 + 2.0 * attempt, lift_cap=4.0,
                  exit_z=0.005, tag=f"press[{attempt}]", max_steps=1100,
                  wiggle=0.03 + 0.03 * attempt)
        release(scene.panel_b)
        step(150)
        report(f"P2 seat[{attempt}]")
        if bool(scene.meshed()[0]):
            meshed = True
            break
        if attempt < 2:
            # a failed press aborted with zero wrench; let everything settle, then
            # TRANSPORT the free cap panel back to a clean aligned hover and retry
            step(90)
            pa = loc(scene.panel_a)
            byaw = a_cross_yaw()
            place(scene.panel_b, (float(pa[0]), float(pa[1]), hover_z),
                  (math.cos(byaw / 2), 0.0, 0.0, math.sin(byaw / 2)))
            step(2)
    assert meshed, "cross-lap press-fit failed"
    print_score("P2 meshed")

    # ----- P3: transport the knife to a free hover over the LIVE notch line ---------------------
    na, nb = scene.notch_points()
    mid = (na[0] + nb[0]) / 2
    line = nb[0] - na[0]
    kyaw = math.atan2(float(line[1]), float(line[0]))
    place(scene.knife, (float(mid[0]), float(mid[1]), c.height + 0.025),
          (math.cos(kyaw / 2), 0.0, 0.0, math.sin(kyaw / 2)))
    report("P3 hover K")
    print_score("P3 hover K")

    # ----- P4: lower the knife into the notches, then hands-off gravity settle ------------------
    def notch_mid():
        a, b = scene.notch_points()
        m = (a[0] + b[0]) / 2
        return float(m[0]), float(m[1])

    def notch_yaw():
        a, b = scene.notch_points()
        line = b[0] - a[0]
        y = math.atan2(float(line[1]), float(line[0]))
        # pick the branch nearest the knife's current yaw (notch line is mod pi)
        ky = float(_yaw_of(scene.knife.data.root_quat_w)[0])
        while y - ky > math.pi / 2:
            y -= math.pi
        while y - ky < -math.pi / 2:
            y += math.pi
        return y

    held_move(scene.knife, mass=c.knife_mass, xy_t=notch_mid, yaw_t=notch_yaw,
              z_from=c.height + 0.025, z_to=c.knife_rest_z - 0.002,
              rate=0.020, kp_xy=25.0, kd_xy=4.0, kp_z=30.0, kd_z=6.0,
              # knife roll inertia ~2.5e-6: ANY explicit roll damping is discrete-
              # unstable, so no uprighting torque at all (force at the CoM keeps it
              # level); yaw PD kept tiny (c*dt/I = 0.008*dt/1.1e-4 = 0.6)
              k_up=0.0, k_yaw=0.04, f_cap=2.5, press_cap=0.8, lift_cap=1.5,
              exit_z=c.knife_rest_z + 0.004, tag="lower knife", max_steps=600)
    release(scene.knife)
    step(180)
    report("P4 seated")
    assert bool(scene.knife_seated()[0]), "knife did not seat on the notch line"
    print_score("P4 seated")

    # ----- P5: hands-off persistence ------------------------------------------------------------
    assert bool(scene.success()[0]), "success not reached before persistence"
    for k in range(10):
        step(41)  # 10 * 41 / 120 = 3.42 s hands-off
        if not bool(scene.success()[0]):
            report(f"P5 lost@{k}")
    report("P5 persist")
    final = bool(scene.success()[0])
    print_score("P5 persist")
    if final:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
        code = 0
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)
        code = 1
    threading.Timer(10.0, lambda: os._exit(code)).start()
    os._exit(code)


try:
    main()
except BaseException:  # noqa: BLE001 - report, then hard-exit (Kit teardown hangs)
    import traceback

    traceback.print_exc()
    print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
    threading.Timer(10.0, lambda: os._exit(2)).start()
    os._exit(2)
