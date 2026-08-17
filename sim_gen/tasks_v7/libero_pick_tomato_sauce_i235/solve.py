"""Teleport solution for SauceCarouselScene (sim_gen task `libero_pick_tomato_sauce_i235`)
— the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. One teleport of one free body: the RED sauce can,
from its parked pose in the OPEN HATCH SECTOR (open sky above — the roof gap is 95
degrees, asserted clear of the aligned can's lift corridor in the scene cfg) to a
release pose centered inside the basket, 6 mm above the basket floor, zero velocity —
exactly what a pick-over-the-sill-and-carry delivers. The lift is attempted ONLY after
the alignment predicate reads True; anywhere else the can is under the roof and the
same lift is geometrically impossible (smoke constructs that bypass and rejects it).

Every load-bearing interaction happens through contact dynamics and applied wrenches:

  spin  — a pure world-z torque on the PLATTER (a proxy for a hand turning the green
          crank bar above the roof; a z-torque is invariant under the body's yaw, so
          the external-wrench frame quirk cannot misdirect it). A two-speed velocity
          servo (0.5 rad/s far out, 0.15 rad/s near the hatch) drives the carousel the
          SHORT way toward the hatch bearing; the cans ride the grippy platter top by
          friction alone — the cans are never touched during the spin.
  park  — the torque switches to an active brake (servo to zero rate) inside the
          alignment window; if the coast overshoots the window the servo re-engages at
          crawl speed (up to 4 park attempts, torque cap escalated on stall).
  place — the can is released 6 mm over the basket floor and lands by gravity.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing, latched by the
scene: 0 -> 0.40 parked in the hatch -> 0.60 extracted -> 1.0 in the basket), then
holds HANDS-OFF for >= 3 simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
    from .scene import BKT_FLOOR_T, CAN_H
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import BKT_FLOOR_T, CAN_H
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

FAST_W = 0.50  # coarse spin rate (rad/s)
SLOW_W = 0.15  # approach spin rate (rad/s)
SLOW_AT_DEG = 40.0  # switch to approach rate inside this |angle|
CUT_AT_DEG = 10.0  # cut drive and brake inside this |angle| (tolerance is 20)
KW = 1.5  # rate-servo gain (N m per rad/s)
KB = 3.0  # brake gain (N m per rad/s)
TAU0 = 0.40  # initial torque cap (N m)
TAU_MAX = 0.80  # escalation ceiling (N m)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sauce_carousel")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)
            scene.score()  # keep the spin-progress latch current

    def clear_forces() -> None:
        scene.platter.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)

    def apply_ztorque(tau: torch.Tensor) -> None:
        """Pure world-z torque on the platter (yaw-invariant under the frame quirk)."""
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 2] = tau
        scene.platter.set_external_force_and_torque(zero_wrench, t, env_ids=all_ids,
                                                    is_global=True)

    def th_deg() -> float:
        return math.degrees(float(scene.hatch_angle(scene.sauce)[0]))

    def plat_w() -> float:
        return float(scene.platter.data.root_ang_vel_w[0, 2])

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | th={th_deg():+7.1f}deg w={plat_w():+.3f} "
              f"riding={bool(scene.on_platter(scene.sauce)[0])} "
              f"aligned={bool(scene.aligned_now()[0])} "
              f"out={bool(scene.out_now()[0])} "
              f"in_bkt={bool(scene.in_basket()[0])} "
              f"decoy={bool(scene.decoy_ok()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wait_settled(bodies, max_steps: int = 600) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(b)[0]) for b in bodies):
                break

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled([scene.platter, scene.sauce, scene.decoy, scene.basket], 360)
    rot_p = (scene.rotunda.data.root_pos_w - scene.env_origins)[0]
    yaw = 2.0 * math.atan2(float(scene.rotunda.data.root_quat_w[0, 3]),
                           float(scene.rotunda.data.root_quat_w[0, 0]))
    bkt_p = (scene.basket.data.root_pos_w - scene.env_origins)[0]
    thd = math.degrees(float(scene.hatch_angle(scene.decoy)[0]))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rotunda=({float(rot_p[0]):+.3f},{float(rot_p[1]):+.3f}) "
          f"yaw={math.degrees(yaw):+.0f}deg th_sauce={th_deg():+.1f}deg "
          f"th_decoy={thd:+.1f}deg "
          f"basket=({float(bkt_p[0]):+.3f},{float(bkt_p[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene.on_platter(scene.sauce)[0]), "sauce must start riding the platter"
    assert bool(scene.decoy_ok()[0]), "decoy must start riding the platter"
    assert abs(th_deg()) >= 60.0, "sauce must start far from the hatch"
    assert not bool(scene.aligned_now()[0]), "sauce must not start aligned"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (cans riding deep under the roof)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: SPIN the carousel by the crank, park in the hatch -----------
    def drive(tau_cap: float, kw: float, budget: int, tag: str) -> str:
        """Two-speed rate servo toward the hatch; returns 'cut' when |th| enters the
        brake window, 'stall' if the approach stops progressing, '' on budget out."""
        best = abs(th_deg())
        since_best = 0
        for i in range(budget):
            th = scene.hatch_angle(scene.sauce)
            ath = th.abs()
            speed = torch.where(ath > math.radians(SLOW_AT_DEG),
                                torch.full_like(th, FAST_W),
                                torch.full_like(th, SLOW_W))
            w_des = -torch.sign(th) * speed
            w = scene.platter.data.root_ang_vel_w[:, 2]
            apply_ztorque((kw * (w_des - w)).clamp(-tau_cap, tau_cap))
            env.step(no_action)
            scene.score()
            a = abs(th_deg())
            if a <= CUT_AT_DEG:
                return "cut"
            if a < best - 2.0:
                best, since_best = a, 0
            else:
                since_best += 1
                if since_best > 240:
                    print(f"[solve] {tag} @{i + 1}: stalled at th={th_deg():+.1f}deg "
                          f"w={plat_w():+.3f} (cap {tau_cap:.2f} gain {kw:.1f})",
                          flush=True)
                    return "stall"
            if i % 240 == 239:
                print(f"[solve] {tag} @{i + 1}: th={th_deg():+.1f}deg "
                      f"w={plat_w():+.3f}", flush=True)
        return ""

    def brake() -> None:
        """Servo the platter rate to zero, then hands off."""
        for _ in range(240):
            w = scene.platter.data.root_ang_vel_w[:, 2]
            apply_ztorque((-KB * w).clamp(-TAU0, TAU0))
            env.step(no_action)
            scene.score()
            if abs(plat_w()) < 0.03:
                break
        clear_forces()
        step(90)

    tau_cap = TAU0
    kw = KW
    parked = False
    for attempt in range(4):
        res = drive(tau_cap, kw, 4800, f"spin[a{attempt}]")
        if res == "stall":
            # gain-limited at rest the servo can only exert kw*w_des, so a stall
            # needs a GAIN escalation, not just a cap escalation
            tau_cap = min(TAU_MAX, tau_cap * 1.5)
            kw = min(8.0, kw * 2.0)
            print(f"[solve] escalating: cap {tau_cap:.2f} N m, gain {kw:.1f}",
                  flush=True)
            continue
        brake()
        report(f"park[a{attempt}]")
        if bool(scene.aligned_now()[0]) and abs(th_deg()) <= 15.0:
            parked = True
            break
        print(f"[solve] park attempt {attempt}: th={th_deg():+.1f}deg after brake — "
              f"re-approaching", flush=True)
    assert parked, "failed to park the sauce can in the hatch sector"
    assert bool(scene.decoy_ok()[0]), "decoy must still be riding after the spin"
    s1 = print_score("P1 carousel spun by crank torque, RED can parked in the hatch "
                     "(cans carried by platter friction alone)")
    assert s1 >= 0.40 - 1e-6 and s1 >= s0, "parking credit missing"

    # ---------------- phase 2: TRANSPORT — lift the parked can out to the basket -----------
    # The can sits in the hatch sector: open sky above (the 95-degree roof gap clears
    # its lift corridor by >= 3 degrees, asserted in cfg), a 75-degree fence window
    # ahead, only the 15 cm sill to clear. This is a plain pick-over-the-sill carry.
    assert bool(scene.aligned_now()[0]), "lift attempted while not parked in the hatch"
    place = torch.zeros(n, 13, device=device)
    place[:, 0:2] = scene.basket.data.root_pos_w[:, :2]
    place[:, 2] = scene.basket.data.root_pos_w[:, 2] + BKT_FLOOR_T + CAN_H / 2 + 0.006
    place[:, 3] = 1.0
    scene.sauce.write_root_state_to_sim(place, all_ids)
    step(2)  # latch the extraction credit at the release pose
    report("carried")
    s2 = print_score("P2 parked can lifted over the sill and carried to the basket "
                     "(teleport transport, zero velocity, released 6 mm up)")
    assert s2 >= 0.60 - 1e-6 and s2 >= s1 - 1e-6, "extraction credit missing"

    # ---------------- phase 3: gravity landing + settle ------------------------------------
    for _ in range(10):
        step(30)
        if bool(scene.success()[0]):
            break
    report("placed")
    s3 = print_score("P3 can landed in the basket by gravity, all settled")
    assert s3 >= s2 - 1e-6, "score decreased across the placement"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after placement)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        env.step(no_action)
        scene.score()
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"in_bkt={bool(scene.in_basket()[0])} "
                      f"sauce_set={bool(scene.settled(scene.sauce)[0])} "
                      f"bkt_set={bool(scene.settled(scene.basket)[0])} "
                      f"plat_set={bool(scene.settled(scene.platter)[0])} "
                      f"decoy={bool(scene.decoy_ok()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s (can resting in the basket)")
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
