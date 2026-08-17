"""Teleport solution for MokaCarouselScene (sim_gen task
`libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i290`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): ONE root-state write carries the pot across FREE SPACE
   with zero velocity to a hover a few millimetres above the floor of an OPEN
   bay out at the loading front (azimuth pi, canopy-free sky) — exactly the
   carry a gripper performs. The write never intersects anything.
2. ROTATION (applied torque + the passive mechanism): the carousel is turned by
   a torque about its own hinge axis — the physical analogue of a hand pushing
   the brass posts sideways. A cascaded velocity servo (omega_des =
   clamp(-2*err, +/-0.8 rad/s); tau = 0.6*(omega_des - omega); |tau| <= 0.5 Nm)
   rotates the loaded bay from the loading front to the station azimuth and
   brakes. Stability audit: k_w*dt/I = 0.6/(120*0.035) = 0.14 << 1; coast after
   torque-cut ~ omega/ang_damp = 0.02/0.5 = 0.04 rad = 2.3 deg << the 15 deg
   window. The hinge rate is finite-differenced from the yaw readback
   (root_ang_vel_w is phantom under external wrenches), the torque rides the
   body's own rotation axis (drag-invariant encoding), and the physx cfg sets
   enable_external_forces_every_iteration.
3. DELIVERY is 100 % mechanism: the pot's final pose at the station is produced
   by the CAROUSEL carrying it under the canopy — no write ever places the pot
   in the success region (the canopy makes that geometrically impossible
   anyway: fence 35 + pot 76 > roof gap 100 mm).

If a drop leaves the pot outside its bay (a bounce), it is picked up again
(fresh transport to the same hover) — a retry, not a cheat: the final
configuration is still 100 % contact-made and the final 3.3 s are hands-off
with the external wrench zeroed.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i290.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401 - registers the scene
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.moka_carousel")().build(num_envs=args.num_envs, device=device)
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

    dt = 1.0 / 120.0
    zeros3 = torch.zeros(n, 1, 3, device=device)

    def apply_tau(tau: float) -> None:
        """Torque about the disc's own rotation axis (body z == world z for a
        yaw-only body; drag-invariant encoding)."""
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 2] = tau
        scene.disc.set_external_force_and_torque(zeros3, t)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def yaw() -> float:
        return float(scene.disc_yaw()[0])

    def bay_az(k: int) -> float:
        return float(scene.pocket_azimuths()[0, k])

    def report(tag: str) -> None:
        azs = scene.pocket_azimuths()[0]
        k = int(scene.pot_bay()[0])
        print(f"[solve] {tag:12s} | yaw={math.degrees(yaw()):+7.1f}deg "
              f"bays=({math.degrees(float(azs[0])):+6.1f},"
              f"{math.degrees(float(azs[1])):+6.1f},"
              f"{math.degrees(float(azs[2])):+6.1f}) pot_bay={k} "
              f"err={math.degrees(float(scene.bay_err()[0])):6.1f}deg "
              f"still={bool(scene.still()[0])} seat={bool(scene._l_seat[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    last_printed = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= last_printed[0] - 1e-6, \
            f"score regressed: {last_printed[0]} -> {s}"
        last_printed[0] = s
        return s

    def servo_bay(k: int, target: float, tag: str,
                  tol: float = math.radians(2.0), max_steps: int = 3600) -> None:
        """Rotate the carousel so bay k's azimuth reaches `target`, then brake
        and CUT the wrench. Cascaded velocity servo on the FD hinge rate."""
        prev = yaw()
        w_fd = 0.0
        for i in range(max_steps):
            err = _wrap(bay_az(k) - target)
            w_des = max(-0.8, min(0.8, -2.0 * err))
            tau = max(-0.5, min(0.5, 0.6 * (w_des - w_fd)))
            apply_tau(tau)
            env.step(no_action)
            y = yaw()
            w_fd = _wrap(y - prev) / dt
            prev = y
            if abs(err) < tol and abs(w_fd) < 0.02:
                break
        else:
            raise AssertionError(
                f"servo[{tag}] did not converge: err="
                f"{math.degrees(_wrap(bay_az(k) - target)):+.1f}deg w={w_fd:+.3f}")
        apply_tau(0.0)  # cut the wrench; coast ~ w/damp = 2.3 deg worst case
        step(60)
        print(f"[solve] servo[{tag}] done in {i + 1} steps: bay{k} at "
              f"{math.degrees(bay_az(k)):+.1f}deg (target "
              f"{math.degrees(target):+.1f}deg)", flush=True)

    def write_pot(pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        scene.pot.write_root_state_to_sim(st, all_ids)

    def hover_over_bay(k: int, dz: float = 0.015):
        """World pose (identity attitude) hovering `dz` above bay k's floor,
        computed from the LIVE disc pose. Pot max half-reach 47 mm < bay half
        60 mm: the hover fits at any bay yaw with 13 mm to spare."""
        phi = float(scene._phi[k])
        lp = torch.tensor([c.r_bay * math.cos(phi), c.r_bay * math.sin(phi),
                           c.slab_t / 2 + dz], device=device).expand(n, 3)
        pos = scene.disc.data.root_pos_w + quat_apply(scene.disc.data.root_quat_w, lp)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
        return pos, quat

    def settle(max_steps: int, min_steps: int = 90) -> None:
        for i in range(max_steps):
            env.step(no_action)
            if i >= min_steps and bool(scene.still()[0]):
                break

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # everything seats (pot on the counter, frypan in its bay)
    pan_bay = int(scene.pan_bay[0])
    pp = (scene.pot.data.root_pos_w - scene.env_origins)[0]
    fl = scene._local_of(scene.frypan)[0]
    print(f"[solve] layout readback (seed {args.seed}): yaw="
          f"{math.degrees(yaw()):+.1f}deg pan_bay={pan_bay} "
          f"pot=({float(pp[0]):+.3f},{float(pp[1]):+.3f},{float(pp[2]):.3f}) "
          f"pan_r={float(fl[:2].norm()):.3f}", flush=True)
    report("reset")
    assert torch.isfinite(scene.pot.data.root_pos_w).all(), "NaN/inf after settle"
    assert abs(float(pp[2]) - c.deck_top) < 0.01, "pot must start on the counter"
    assert int(scene.pot_bay()[0]) == -1, "pot must start outside every bay"
    assert abs(float(fl[:2].norm()) - c.r_bay) < 0.02, \
        "frypan must start seated in its bay"
    s0 = print_score("P0 reset+settle (pot on the counter, frypan riding one bay)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: stage an empty bay out to the loading front -----------------
    # Empty bay closest (by rotation) to the loading azimuth pi (open sky, in
    # front of the canopy's far side).
    LOAD = math.pi
    empties = [k for k in range(3) if k != pan_bay]
    k = min(empties, key=lambda j: abs(_wrap(bay_az(j) - LOAD)))
    print(f"[solve] loading bay: {k} (empty bays {empties}, "
          f"az now {math.degrees(bay_az(k)):+.1f}deg)", flush=True)
    servo_bay(k, LOAD, "stage")
    report("staged")
    assert abs(_wrap(bay_az(k) - LOAD)) < math.radians(8.0), \
        "empty bay must sit at the loading front"
    s1 = print_score("P1 empty bay staged at the loading front (azimuth pi)")

    # ---------------- phase 2: load the pot into the staged bay ----------------------------
    for attempt in range(3):
        pos, quat = hover_over_bay(k)
        write_pot(pos, quat)
        settle(420)
        if int(scene.pot_bay()[0]) == k:
            break
        print(f"[solve] drop bounced out (attempt {attempt + 1}); retrying", flush=True)
    report("loaded")
    assert int(scene.pot_bay()[0]) == k, "pot would not seat in the staged bay"
    assert bool(scene._l_seat[0]), "seat latch must fire once the pot rests in a bay"
    s2 = print_score("P2 pot seated upright in the staged bay")
    assert 0.24 <= s2 <= 0.33, f"P2 score {s2} (expect seat 0.25, ~zero progress)"

    # ---------------- phase 3: the mechanism delivers — rotate to the station --------------
    servo_bay(k, 0.0, "deliver")
    report("delivered")
    # Wait for success to hold CONTINUOUSLY for 2 simulated seconds (240
    # substeps) so the boundary marks a genuinely stopped carousel, not a
    # fly-through of the azimuth window.
    ok = False
    streak = 0
    for _ in range(2400):
        env.step(no_action)
        streak = streak + 1 if bool(scene.success()[0]) else 0
        if streak >= 240:
            ok = True
            break
    report("settled")
    assert ok and bool(scene.success()[0]), \
        f"delivery did not settle into success (err " \
        f"{math.degrees(float(scene.bay_err()[0])):.1f}deg, " \
        f"wz {float(scene.disc.data.root_ang_vel_w[0, 2]):+.3f})"
    s3 = print_score("P3 bay centred on the station, carousel stopped — success")
    assert s3 >= 0.99, f"success must score 1.0, got {s3}"

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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
