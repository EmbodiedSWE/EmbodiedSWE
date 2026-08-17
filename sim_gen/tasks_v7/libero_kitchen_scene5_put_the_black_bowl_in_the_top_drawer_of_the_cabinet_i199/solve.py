"""Teleport solution for BatonStowScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i199`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. OPEN THE DRAWER (contact dynamics — never teleported): a bounded horizontal force
   on the drawer body (the applied-wrench emulation of the arm pulling the blue
   handle bar) servoes the drawer out along its real prismatic slide against the
   slide damping; the drive is then REMOVED and the drawer stays out on its own
   (springless slide). The drawer's pose is never written after reset.
2. STAGE (transport teleport): the black baton is teleported ONCE from the cabinet
   roof to a free-space pre-insertion pose — pitched ~50 deg, low end hovering over
   the exposed drawer mouth, overlapping nothing. This is pure transport; from here
   on the baton is "held" by a bounded wrench, never pose-written again.
3. THREAD (applied wrench + contact): a world-frame wrench servo (force PD with
   gravity feedforward, cap 6 N; attitude PD, cap 0.3 N.m — the emulation of the
   arm gripping the baton's protruding half) walks the low end aft UNDER the fixed
   roof lip along the clearance curve (p, theta) while the tip rides the drawer
   floor through real contact. The final aft push is closed-loop on the MEASURED
   aft-endpoint readback, then the wrench is CUT and the baton falls flat onto the
   drawer floor under gravity. The roof, drawer and floor react through contacts
   the whole way; nothing else is written.
4. SHUT THE DRAWER (contact dynamics): the same bounded slide force pushes the
   drawer back in; the drawer's front wall collects the baton and carries it the
   last few cm through real contact. Drive off; everything settles; success() first
   turns True here, judged on settled physical state. The white decoy baton is
   NEVER touched: it stays on the roof, outside the cavity, as required.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True  # never keep a dead interpreter alive waiting for the timer
_wd.start()

# ----- held-baton wrench servo gains (stability: Kd*dt/m = 4/(120*0.12) = 0.28 << 1;
# attitude about y: I = m(L^2+w^2)/12 = 4.1e-4, Kd_t*dt/I = 0.16 << 1; both ~0.8 damped)
KP_F, KD_F, F_CAP = 50.0, 4.0, 6.0
KP_T, KD_T, T_CAP = 0.08, 0.008, 0.3


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.baton_stow")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    g = 9.81

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    all_ids = torch.arange(n, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def open_m() -> float:
        return float(scene.drawer_open()[0])

    def black_p() -> list:
        return [float(v) for v in (scene.black.data.root_pos_w - scene.env_origins)[0]]

    def decoy_p() -> list:
        return [float(v) for v in (scene.decoy.data.root_pos_w - scene.env_origins)[0]]

    def ep_local() -> list:
        """Drawer-local x of the two black endpoints, sorted [fore, aft]."""
        loc = (scene.black_endpoints() - scene.drawer.data.root_pos_w.unsqueeze(1))[0]
        xs = sorted([float(loc[0, 0]), float(loc[1, 0])])
        return xs

    def report(tag: str) -> None:
        b = black_p()
        e = ep_local()
        print(f"[solve] {tag:12s} | open={open_m() * 1000:5.1f}mm "
              f"black=({b[0]:+.3f},{b[1]:+.3f},{b[2]:.3f}) ep_loc=({e[0]:+.3f},{e[1]:+.3f}) "
              f"inside={bool(scene.black_inside()[0])} decoy_in={bool(scene.decoy_in_cavity()[0])} "
              f"open_max={float(scene._open_max[0]):.2f} thread={bool(scene._thread[0])} "
              f"ins_latch={bool(scene._inside[0])} close_max={float(scene._close_max[0]):.2f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)  # batons settle onto the roof
    b0, d0 = black_p(), decoy_p()
    print(f"[solve] layout readback (seed {args.seed}): black=({b0[0]:+.3f},{b0[1]:+.3f}) "
          f"decoy=({d0[0]:+.3f},{d0[1]:+.3f}) drawer_open={open_m() * 1000:.1f}mm", flush=True)
    report("reset")
    assert open_m() < 0.005, "drawer did not start shut"
    assert b0[2] > c.roof_z1 - 0.01 and d0[2] > c.roof_z1 - 0.01, "batons not on the roof"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"score not ~0 at reset ({s0:.3f})"

    # ---------------- phase 1: OPEN the drawer through the slide (contact dynamics) ---------
    target = c.travel - 0.008  # 157 mm
    for _ in range(700):
        x = open_m()
        v = float(scene.drawer.data.root_lin_vel_w[0, 0])  # +x = closing
        f = -60.0 * (target - x) + 8.0 * (-v)  # pull toward -x
        scene.drawer_drive[:] = max(-10.0, min(10.0, f))
        env.step(no_action)
        if x >= target - 0.003:
            break
    scene.drawer_drive[:] = 0.0
    step(60)  # drive off: the springless slide stays where it was left
    report("opened")
    j_open = open_m()
    assert j_open >= 0.145, f"drawer did not stay open ({j_open * 1000:.1f}mm)"
    s1 = print_score("P1 drawer pulled open by slide force, resting out")
    assert s1 >= max(s0, 0.14) - 1e-6, "open credit missing"

    # ---------------- phase 2: STAGE (one transport teleport to free space) -----------------
    # Held-pose parametrization: the baton's +x axis points DOWN-AFT toward the low
    # tip; pitch theta below horizontal; low tip at (front_x + p, 0, tip_z).
    def pose_of(p: float, th_deg: float, tip_z: float):
        th = math.radians(th_deg)
        ax = (math.cos(th), 0.0, -math.sin(th))
        tip = (c.front_x + p, 0.0, tip_z)
        half = c.black_len / 2
        center = (tip[0] - half * ax[0], 0.0, tip[2] - half * ax[2])
        quat = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
        return center, quat

    stage_p, stage_th, stage_tipz = -0.030, 50.0, 0.178
    ctr, qt = pose_of(stage_p, stage_th, stage_tipz)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = torch.tensor(ctr, device=device) + scene.env_origins
    st[:, 3:7] = torch.tensor(qt, device=device)
    scene.black.write_root_state_to_sim(st, all_ids)  # the ONLY baton pose write
    # take hold with the wrench servo before gravity does anything
    def servo_step(p: float, th_deg: float, tip_z: float) -> None:
        ctr, qt = pose_of(p, th_deg, tip_z)
        pos = scene.black.data.root_pos_w - scene.env_origins
        vel = scene.black.data.root_lin_vel_w
        err = torch.tensor(ctr, device=device).expand(n, 3) - pos
        f = KP_F * err - KD_F * vel
        f[:, 2] += c.black_mass * g  # gravity feedforward
        nrm = f.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        f = f * (nrm.clamp(max=F_CAP) / nrm)
        # attitude PD toward the target pitch (rot about +y), world frame
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

        q_des = torch.tensor(qt, device=device).expand(n, 4)
        rotvec = axis_angle_from_quat(quat_mul(q_des, quat_conjugate(scene.black.data.root_quat_w)))
        t = KP_T * rotvec - KD_T * scene.black.data.root_ang_vel_w
        tn = t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        t = t * (tn.clamp(max=T_CAP) / tn)
        scene.black_force_w[:] = f
        scene.black_torque_w[:] = t
        env.step(no_action)

    for _ in range(90):  # grab-and-steady at the staging pose
        servo_step(stage_p, stage_th, stage_tipz)
    report("staged")
    bz = black_p()[2]
    assert bz > 0.20, f"baton not held aloft after staging (z={bz:.3f})"
    s2a = print_score("P2a baton staged at the pre-insertion pose (held by wrench)")
    assert s2a >= s1 - 1e-6, "score decreased across staging"

    # ---------------- phase 3: THREAD under the roof lip (wrench + contact) -----------------
    # Clearance-curve waypoints (p = low-tip penetration past the roof edge, theta):
    # z_top(roof plane) = tip_z + p*tan(th) + 0.015/cos(th) stays >= 5 mm under the
    # roof bottom (0.228) even with the tip riding the drawer floor.
    waypoints = [(-0.030, 50.0, 0.178), (0.0, 48.0, 0.170), (0.020, 41.0, 0.170),
                 (0.040, 34.0, 0.170), (0.060, 27.0, 0.170), (0.080, 20.0, 0.170),
                 (0.084, 14.0, 0.170)]
    for (p0, t0, z0), (p1, t1, z1) in zip(waypoints[:-1], waypoints[1:]):
        for i in range(150):
            s = (i + 1) / 150
            w = s * s * (3 - 2 * s)  # smoothstep
            servo_step(p0 + (p1 - p0) * w, t0 + (t1 - t0) * w, z0 + (z1 - z0) * w)
    # closed-loop final push: advance p until the MEASURED aft endpoint is deep enough
    p_now = waypoints[-1][0]
    for i in range(500):
        if ep_local()[1] >= 0.225:
            break
        p_now = min(p_now + 0.0004, 0.150)
        servo_step(p_now, 14.0, 0.170)
    report("threaded")
    assert ep_local()[1] >= 0.215, f"aft endpoint not deep enough ({ep_local()[1]:.3f})"
    assert bool(scene._thread[0]), "thread latch did not fire"
    # RELEASE: cut the wrench; the baton falls flat inside under gravity
    scene.black_force_w[:] = 0.0
    scene.black_torque_w[:] = 0.0
    for i in range(600):
        env.step(no_action)
        sp = float(scene.black.data.root_lin_vel_w[0].norm())
        if bool(scene.black_inside()[0]) and sp < 0.04 and i > 60:
            break
    report("released")
    assert bool(scene.black_inside()[0]), "baton did not settle flat inside the cavity"
    s3 = print_score("P3 baton threaded under the roof lip, released, lying inside")
    assert s3 >= max(s2a, 0.69) - 1e-6, "thread/inside credit missing"

    # ---------------- phase 4: SHUT the drawer (contact dynamics; baton rides/pushed) --------
    for _ in range(900):
        x = open_m()
        v = float(scene.drawer.data.root_lin_vel_w[0, 0])
        f = -60.0 * (0.002 - x) + 8.0 * (-v)  # push toward +x (shut)
        scene.drawer_drive[:] = max(-10.0, min(10.0, f))
        env.step(no_action)
        if x <= 0.005:
            break
    scene.drawer_drive[:] = 0.0
    step(120)  # drive off; everything settles
    report("shut")
    assert open_m() <= c.close_tol, f"drawer did not stay shut ({open_m() * 1000:.1f}mm)"
    assert bool(scene.black_inside()[0]), "baton no longer inside after shutting"
    assert not bool(scene.decoy_in_cavity()[0]), "decoy ended in the cavity"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after shutting)", flush=True)
        os._exit(1)
    s4 = print_score("P4 drawer pushed shut with the baton enclosed")
    assert s4 >= s3 - 1e-6, "score decreased across shutting"

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) --------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
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
    except BaseException:  # noqa: BLE001 - Kit threads would hang the interpreter
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
