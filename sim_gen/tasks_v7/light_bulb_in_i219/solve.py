"""Teleport solution for LanternShelterScene (sim_gen task `light_bulb_in_i219`) — the
task's legitimacy certificate.

NOTHING is teleported: the globe exceeds the jaw span, so the whole demonstration is
non-prehensile contact dynamics, exactly the strategy the task forces:
  1. STAGE (dynamics): a CoM velocity-servo force rolls the globe across the floor to
     a staging point on the doorway axis (the stand-in for the arm's guarded push on
     the globe's rear pole — the contact point stays outside the shelter throughout).
  2. ENTER + SEAT (dynamics): the servo drives the globe through the doorway and
     delivers a measured SHOVE; the globe pops over the four-post entry barrier
     (~10 mm center rise) and the posts + back wall capture it. On a stall the shove
     speed escalates and the globe is retreated and re-run.
  3. SHUTTER (dynamics): a velocity-servo force slides the free shutter slab along
     its curb/rail channel to the -y end stop, which by construction reads closed.

Servo notes (external-wrench plant recipe): the physx flag
`enable_external_forces_every_iteration` is set in the scene; globe and shutter have
authored diagonal inertia; wrenches are encoded into the CURRENT body frame (the
house convention — critical for the rolling globe, whose orientation is arbitrary).
Gains are discretely stable: globe kv*dt/m = 6/(120*0.15) = 0.33 < 1, shutter
kv*dt/m = 25/(120*0.30) = 0.69 < 1.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.5 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.light_bulb_in_i219.solve --headless [--seed N]
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
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.lantern_shelter")().build(num_envs=args.num_envs,
                                                     device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_rows = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def force_on(body, f_w: torch.Tensor) -> None:
        """World-frame force at the CoM, encoded into the CURRENT body frame (the
        house convention: the engine treats the given wrench as body-frame)."""
        body.set_external_force_and_torque(
            quat_apply_inverse(body.data.root_link_quat_w, f_w).unsqueeze(1),
            zero_rows, env_ids=all_ids)

    def release(body) -> None:
        body.set_external_force_and_torque(zero_rows, zero_rows, env_ids=all_ids)

    def loc_of(body) -> torch.Tensor:
        return scene._shelter_local(body.data.root_pos_w)[0]

    def to_world(v_local: torch.Tensor) -> torch.Tensor:
        """Shelter-frame vector -> world frame (rows of size 3)."""
        return quat_apply(scene.shelter.data.root_quat_w, v_local.unsqueeze(0))

    def report(tag: str) -> None:
        g = loc_of(scene.globe)
        d = loc_of(scene.shutter)
        print(f"[solve] {tag:14s} | globe_loc=({float(g[0]):+.3f},{float(g[1]):+.3f},"
              f"{float(g[2]):+.3f}) door_y={float(d[1]):+.3f}"
              f" latches=({float(scene.enter_latch[0]):.0f},"
              f"{float(scene.seat_latch[0]):.0f},{float(scene.shut_latch[0]):.0f})"
              f" in_seat={bool(scene.in_seat()[0])}"
              f" closed={bool(scene.door_closed()[0])}"
              f" settled={bool(scene.settled()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    KV_G, CAP_G = 6.0, 4.0  # globe servo: kv*dt/m = 0.33 < 1

    def globe_servo_step(v_des_local: torch.Tensor) -> None:
        v_des_w = to_world(v_des_local)[0]
        v = scene.globe.data.root_lin_vel_w[0]
        f = KV_G * (v_des_w - v)
        f[2] = 0.0  # ground supports the globe; never push into/out of the floor
        fn = f.norm()
        if fn > CAP_G:
            f = f * (CAP_G / fn)
        force_on(scene.globe, f.unsqueeze(0).expand(n, 3))
        env.step(no_action)

    def roll_to(lx: float, ly: float, tol: float, speed: float, tag: str,
                max_steps: int = 1800) -> None:
        """Roll the globe to a shelter-frame waypoint with a decelerating velocity
        servo; release and let the 0.02-damped sphere coast to rest."""
        for i in range(max_steps):
            g = loc_of(scene.globe)
            d = torch.tensor([lx - float(g[0]), ly - float(g[1]), 0.0], device=device)
            dist = float(d.norm())
            v = float(scene.globe.data.root_lin_vel_w[0].norm())
            if dist < tol and v < 0.08:
                break
            v_mag = min(speed, 2.0 * dist)
            globe_servo_step(d / max(dist, 1e-6) * v_mag)
        release(scene.globe)
        step(60)
        g = loc_of(scene.globe)
        print(f"[solve] roll {tag}: at ({float(g[0]):+.3f},{float(g[1]):+.3f}) "
              f"after {i + 1} servo steps", flush=True)

    def shove(speed: float) -> bool:
        """Drive the globe forward (+x, y-guarded) through the doorway and over the
        four-post entry barrier at `speed`; cut the force at the seat axis and let
        the posts + back wall capture it. Returns seated-after-settle."""
        for _ in range(720):
            g = loc_of(scene.globe)
            if float(g[0]) >= 0.0 or bool(scene.in_seat()[0]):
                break
            vy = max(min(-2.0 * float(g[1]), 0.08), -0.08)
            globe_servo_step(torch.tensor([speed, vy, 0.0], device=device))
        release(scene.globe)
        step(150)  # hands-off: capture or roll-back, decided by physics
        return bool(scene.in_seat()[0])

    def retreat() -> None:
        """Roll a stalled globe back out through the doorway for another run-up."""
        for _ in range(900):
            g = loc_of(scene.globe)
            if float(g[0]) <= -0.16:
                break
            vy = max(min(-2.0 * float(g[1]), 0.08), -0.08)
            globe_servo_step(torch.tensor([-0.22, vy, 0.0], device=device))
        release(scene.globe)
        step(60)

    KV_D, CAP_D = 25.0, 4.0  # shutter servo: kv*dt/m = 0.69 < 1

    def close_shutter() -> None:
        """Slide the free shutter along its channel to the -y end stop (push-to-stop
        reads closed by construction); release only when covering and slow."""
        for i in range(1500):
            d = loc_of(scene.shutter)
            v = float(scene.shutter.data.root_lin_vel_w[0].norm())
            if abs(float(d[1])) <= c.close_y_tol - 0.002 and v < 0.04:
                break
            v_des_w = to_world(torch.tensor([0.0, -0.12, 0.0], device=device))[0]
            vel = scene.shutter.data.root_lin_vel_w[0]
            f = KV_D * (v_des_w - vel)
            f[2] = 0.0
            fn = f.norm()
            if fn > CAP_D:
                f = f * (CAP_D / fn)
            force_on(scene.shutter, f.unsqueeze(0).expand(n, 3))
            env.step(no_action)
        release(scene.shutter)
        step(60)
        d = loc_of(scene.shutter)
        print(f"[solve] shutter parked at slide offset {float(d[1]):+.4f} "
              f"after {i + 1} servo steps", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    g, d, dc = loc_of(scene.globe), loc_of(scene.shutter), loc_of(scene.decoy)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"globe=({float(g[0]):+.3f},{float(g[1]):+.3f},{float(g[2]):.3f}) "
          f"decoy=({float(dc[0]):+.3f},{float(dc[1]):+.3f}) "
          f"door_open={float(d[1]):+.3f}", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: stage the globe on the doorway axis (dynamics) --------------
    roll_to(-0.20, 0.0, 0.006, 0.25, "stage")
    g = loc_of(scene.globe)
    assert abs(float(g[1])) < 0.012, "globe not aligned with the doorway axis"
    report("staged")
    s1 = print_score("P1 globe staged before the doorway (rolled, CoM servo)")
    assert s1 >= s0 - 1e-6

    # ---------------- phase 2: enter + shove onto the four-post seat (dynamics) ------------
    seated = False
    for speed in (0.55, 0.65, 0.78, 0.90):
        print(f"[solve] shove attempt at {speed:.2f} m/s", flush=True)
        seated = shove(speed)
        report(f"shove {speed:.2f}")
        if seated:
            break
        retreat()
        roll_to(-0.20, 0.0, 0.006, 0.25, "restage")
    assert seated, "globe never seated on the posts"
    assert float(scene.enter_latch[0]) == 1.0, "enter latch never fired"
    s2 = print_score("P2 globe shoved over the barrier, seated on the posts")
    assert s2 >= s1 - 1e-6 and s2 >= 0.50 - 1e-6

    # ---------------- phase 3: slide the shutter closed (dynamics) -------------------------
    close_shutter()
    report("shuttered")
    assert bool(scene.door_closed()[0]), "shutter does not read closed"
    s3 = print_score("P3 shutter slid closed along its channel")
    assert s3 >= s2 - 1e-6 and s3 >= 0.75 - 1e-6

    # ---------------- phase 4: settle to success -------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s4 = print_score("P4 all settled")
    assert s4 >= s3 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after stage+shove+shutter+settle)",
              flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.5 simulated seconds, hands-off) -----------
    hold, flickers = True, 0
    for i in range(420):  # 420 substeps = 3.5 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"in_seat={bool(scene.in_seat()[0])} "
                      f"closed={bool(scene.door_closed()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"glin={float(scene.globe.data.root_lin_vel_w[0].norm()):.4f} "
                      f"gang={float(scene.globe.data.root_ang_vel_w[0].norm()):.4f} "
                      f"dlin={float(scene.shutter.data.root_lin_vel_w[0].norm()):.4f}",
                      flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/420 steps", flush=True)
    report("persist")
    s5 = print_score("P5 persistence 3.5 s")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
