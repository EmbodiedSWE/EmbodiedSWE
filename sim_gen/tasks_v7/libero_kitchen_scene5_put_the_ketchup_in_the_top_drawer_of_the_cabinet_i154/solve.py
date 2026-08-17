"""Teleport solution for CarouselCabinetScene (sim_gen task
`libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_i154`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. ALIGN (applied torque through the bearing, closed loop): the carousel starts with
   the green sector 40..150 deg away from the window (either side). A velocity-capped
   P-servo writes the scene's crank-drive buffer (a torque about the bearing axis —
   exactly what a hand on the green crank arm applies); the carousel rotates on its
   frictionless revolute bearing until the green sector faces the window. The
   carousel is NEVER teleported after reset.
2. TRANSPORT (teleport): a single root-state write carries the RED ketchup bottle
   from its ground slot to the free-space point 15 mm above the green mat, upright,
   zero velocity. The path is free space: with the green sector aligned, the window
   (~290 x 220 mm) is a straight-line corridor onto the mat for a 60 mm bottle —
   the transport bypasses no contact interaction. The write satisfies no rubric
   clause by itself: the bottle is airborne.
3. SEATING (gravity + contact): the bottle FALLS the last 15 mm onto the green mat
   and settles standing. `in_green_sector` (sector wedge, radial band, standing
   height, upright) is produced by contact, never written.
4. STOW (applied torque under load + friction): the same crank servo rotates the
   LOADED carousel half a turn so the green sector faces the BACK. The bottle rides
   the platform on friction contact the whole way — it is never touched again.
5. ORDER: physically forced — the roof and the 260-deg wall make the window the only
   way in, so the green sector must be AT the window before the bottle can enter it
   (smoke: insertion attempts into a non-aligned green sector are rejected by the
   geometry), and the divider walls make it impossible to move the bottle between
   sectors without rotating the carousel. The YELLOW mustard bottle is never touched.

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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def _wrap(a: float) -> float:
    """Wrap degrees to (-180, 180]."""
    return a - 360.0 * math.floor((a + 180.0) / 360.0)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carousel_cabinet")().build(num_envs=args.num_envs,
                                                     device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def az() -> float:
        return float(scene.green_az_deg()[0])

    def omega() -> float:
        return float(scene.carousel.data.root_ang_vel_w[0, 2])

    def report(tag: str) -> None:
        kk = (scene.ketchup.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | green_az={az():+7.1f}deg omega={omega():+.3f} "
              f"ketchup=({float(kk[0]):+.3f},{float(kk[1]):+.3f},{float(kk[2]):+.3f}) "
              f"aligned={bool(scene.aligned()[0])} in_green={bool(scene.in_green_sector()[0])} "
              f"stowed={bool(scene.stowed()[0])} must_out={bool(scene.mustard_outside()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def servo_to(target_deg: float, *, omega_cap: float, tau_max: float,
                 timeout: int, tag: str, stop_err_deg: float = 2.5) -> bool:
        """Velocity-capped P-servo on the crank torque: rotate the green sector to
        `target_deg` (housing-frame azimuth). PD in disguise (w_des = kp*err;
        tau = K*(w_des - w)), so it actively brakes into the target — no
        constant-torque coast. Gain escalates on stall; K*dt/I stays < 0.85."""
        kp = 2.0          # (rad/s) per rad of angle error
        gain = 0.25       # N*m per (rad/s) of velocity error; I=0.010, dt=1/120
        best_abs = 1e9
        last_gain_bump = 0
        for i in range(timeout):
            err = _wrap(target_deg - az())
            w = omega()
            if abs(err) < stop_err_deg and abs(w) < 0.02:
                scene.crank_tau[:] = 0.0
                print(f"[solve] {tag}: converged at step {i} "
                      f"(err={err:+.2f}deg, omega={w:+.3f})", flush=True)
                return True
            w_des = max(-omega_cap, min(omega_cap, kp * math.radians(err)))
            tau = max(-tau_max, min(tau_max, gain * (w_des - w)))
            scene.crank_tau[:] = tau
            env.step(no_action)
            a = abs(_wrap(target_deg - az()))
            best_abs = min(best_abs, a)
            if i % 120 == 119:
                print(f"[solve] {tag}: step {i} err={_wrap(target_deg - az()):+7.1f}deg "
                      f"omega={omega():+.3f} tau={tau:+.3f}", flush=True)
            # stall: barely moving, far from target, no recent progress -> more gain
            if (i - last_gain_bump) >= 240 and a > stop_err_deg + 2.0 \
                    and abs(omega()) < 0.05:
                gain = min(gain * 1.6, 0.85)
                last_gain_bump = i
                print(f"[solve] {tag}: stall at err={a:.1f}deg; gain -> {gain:.2f}",
                      flush=True)
        scene.crank_tau[:] = 0.0
        print(f"[solve] {tag}: TIMEOUT (best |err|={best_abs:.1f}deg)", flush=True)
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)
    # authored-mass readback: custom spawners apply no cfg schemas, so verify the
    # explicit MassAPI took effect (density-derived mass would break the servo gains)
    m_car = float(scene.carousel.root_physx_view.get_masses()[0].sum())
    print(f"[solve] carousel mass readback: {m_car:.3f} kg (authored {c.carousel_mass})",
          flush=True)
    assert abs(m_car - c.carousel_mass) < 0.05 * c.carousel_mass, \
        f"carousel mass {m_car} != authored {c.carousel_mass} (MassAPI did not take)"
    kp0 = (scene.ketchup.data.root_pos_w - scene.env_origins)[0]
    side = "+y" if float(scene.ketchup_side[0]) > 0 else "-y"
    print(f"[solve] layout readback (seed {args.seed}): green_az={az():+.1f}deg "
          f"ketchup_side={side} ketchup=({float(kp0[0]):+.3f},{float(kp0[1]):+.3f})",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(az()) >= c.yaw_min_deg - 8.0, f"start az {az():.1f} inside align band"
    assert 180.0 - abs(az()) > c.stow_tol_deg + 5.0, f"start az {az():.1f} near stow"
    assert not bool(scene.aligned()[0]), "must not start aligned"
    assert not bool(scene._inside(scene.ketchup)[0]), "ketchup must start outside"
    s0 = print_score("P0 reset+settle (green sector away from the window)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: crank the green sector to the window ------------------------
    ok = servo_to(0.0, omega_cap=0.80, tau_max=0.30, timeout=2400, tag="align")
    step(90)  # hands-off settle; residual spin decays
    report("align")
    if not ok or not bool(scene.aligned()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (green sector never aligned)", flush=True)
        os._exit(1)
    s1 = print_score("P1 green sector aligned with the window (crank torque servo)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.19, f"P1 score {s1} (expect aligned=0.20)"

    # ---------------- phase 2: ketchup -> hover over the green mat; gravity seats it -------
    yaw = math.radians(az())
    mat_r = 0.095  # green mat centre radius (carousel local +x)
    hover = torch.zeros(n, 13, device=device)
    hover[:, 0] = c.hub_pos[0] + mat_r * math.cos(yaw)
    hover[:, 1] = c.hub_pos[1] + mat_r * math.sin(yaw)
    hover[:, 2] = c.z_plat + c.mat_t + c.bot_h / 2 + 0.015  # 15 mm free fall
    hover[:, 3] = 1.0
    hover[:, 0:3] += scene.env_origins
    scene.ketchup.write_root_state_to_sim(hover, all_ids)
    step(150)  # fall + settle, hands-off
    report("insert")
    assert bool(scene.in_green_sector()[0]), \
        "ketchup must stand upright in the green sector"
    assert bool(scene.mustard_outside()[0]), "mustard must remain outside"
    assert not bool(scene.success()[0]), "cannot be success while green faces the window"
    s2 = print_score("P2 ketchup dropped through the window, seated on the green mat")
    assert s2 >= s1 - 1e-6 and s2 >= 0.59, f"P2 score {s2} (expect 0.60)"

    # ---------------- phase 3: crank the loaded green sector to the back -------------------
    # target +180 or -180: same point; servo the shorter signed direction from here.
    target = 180.0 if az() >= 0.0 else -180.0
    ok = servo_to(target, omega_cap=0.50, tau_max=0.30, timeout=3600, tag="stow",
                  stop_err_deg=3.0)
    step(150)  # hands-off settle
    report("stow")
    if not ok or not bool(scene.stowed()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (green sector never stowed at the back)", flush=True)
        os._exit(1)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after stow)", flush=True)
        os._exit(1)
    s3 = print_score("P3 loaded carousel rotated; green sector stowed at the back")
    assert s3 >= s2 - 1e-6, "score decreased across the stow"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
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
