"""Teleport solution for PagodaRelayScene (sim_gen task `living_room_scene4_stack_the_
left_bowl_on_the_right_bowl_and_place_them_in_the_tray_i194`) — the task's legitimacy
certificate: the classic 7-move Tower-of-Hanoi relay, played move by legal move.

Teleportation handles TRANSPORT ONLY: each move lifts the CURRENT TOP tier of one post
(verified against the seated matrix before every move — exactly what the post
captivity allows a gripper to do) and releases it with zero velocity 12 mm ABOVE the
destination post's tip, centered on the post with a ~3 mm lateral offset. Everything
rubric-relevant then happens through contact dynamics: gravity threads the falling
tier's square hole down the round post, the post's geometry funnels and captures it,
and it lands and settles on the pad / the tier below at its level height. No tier is
ever written into a seated pose; seating is always produced by the drop. The rule
latches (one tier in flight, never bigger above smaller) run every physics substep in
scene.post_step throughout — the solve wins by SEQUENCING, the same way any policy
must:

  move 1  small  A -> tray   (unlocks the mid tier; latches 0.15)
  move 2  mid    A -> B
  move 3  small  tray -> B   (parks small on mid — legal: smaller above bigger)
  move 4  large  A -> tray   (the payoff move; latches 0.40)
  move 5  small  B -> A
  move 6  mid    B -> tray   (latches 0.70)
  move 7  small  A -> tray   (pagoda rebuilt; success -> 1.0)

Each drop is verified by READBACK (seated at the destination, at the expected level,
settled) with up to 3 re-carries of the same still-in-flight tier. Prints
`SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing, latched by the scene),
then holds HANDS-OFF for >= 3 simulated seconds after success() first turns True and
prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

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
    from .scene import PAD_H, POST_TOP, TIER_H, _qapply
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    from scene import PAD_H, POST_TOP, TIER_H, _qapply
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# (tier, dest station, expected level, minimum latched score after the move)
# tiers: 0=large 1=mid 2=small; stations: 0=A(start) 1=B(buffer) 2=tray(goal)
MOVES = (
    (2, 2, 0, 0.15), (1, 1, 0, 0.15), (2, 1, 1, 0.15), (0, 2, 0, 0.40),
    (2, 0, 0, 0.40), (1, 2, 1, 0.70), (2, 2, 2, 0.70),
)
HOVER = 0.012  # release height above the post tip (m)
DROP_OFF = 0.003  # lateral release offset (m) — well inside the 6 mm threading slack


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.pagoda_relay")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        seated, lz = scene.seat_state()
        stacks = []
        for s, nm in enumerate("ABC"):
            col = [("LMS"[t], float(lz[0, t, s]))
                   for t in range(3) if bool(seated[0, t, s])]
            col.sort(key=lambda p: p[1])
            stacks.append(nm + ":" + "".join(p[0] for p in col))
        print(f"[solve] {tag:10s} | {' '.join(stacks)} "
              f"violated={bool(scene._violated[0])} "
              f"carry_ctr={float(scene._carry_ctr[0]):.0f} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def wait_settled(max_steps: int = 480) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if all(bool(scene.settled(t)[0]) for t in scene.tiers):
                break

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    wait_settled(360)
    seated, lz = scene.seat_state()
    for s, nm in enumerate(scene.STATIONS):
        p = (scene.stations[s].data.root_pos_w - scene.env_origins)[0]
        yaw = 2.0 * math.atan2(float(scene.stations[s].data.root_quat_w[0, 3]),
                               float(scene.stations[s].data.root_quat_w[0, 0]))
        print(f"[solve] layout readback (seed {args.seed}): {nm} at "
              f"({float(p[0]):+.3f},{float(p[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg",
              flush=True)
    report("reset")
    for t in range(3):
        assert bool(seated[0, t, 0]), f"tier {t} must start threaded on the start post"
        want = PAD_H + t * TIER_H
        assert abs(float(lz[0, t, 0]) - want) <= float(scene.cfg.lvl_tol) + 0.004, \
            f"tier {t} must start at level {t}"
    assert not bool(scene._violated[0]), "fresh reset must not be violated"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (pagoda assembled on the start post)")
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1..7: the legal moves -----------------------------------------
    def assert_top(mi: int, t: int) -> None:
        """The move is legal only if tier t is the TOP of its pile — readback proof."""
        seated, lz = scene.seat_state()
        cur = -1
        for si in range(3):
            if bool(seated[0, t, si]):
                cur = si
        assert cur >= 0, f"move {mi}: tier {t} is not seated anywhere"
        for u in range(3):
            if u != t and bool(seated[0, u, cur]):
                assert float(lz[0, u, cur]) < float(lz[0, t, cur]) + 1e-4, \
                    f"move {mi}: tier {t} is not the top of its pile"

    def move(mi: int, t: int, s: int, lvl: int) -> None:
        """One legal move: hover-teleport tier t (zero velocity) 12 mm above station
        s's post tip; gravity threads it down the post onto the pile. Verify seated at
        the expected level and settled; up to 3 re-carries of the still-in-flight tier."""
        assert_top(mi, t)
        stn, tier = scene.stations[s], scene.tiers[t]
        want_z = PAD_H + lvl * TIER_H
        loc = torch.tensor([DROP_OFF, -DROP_OFF, POST_TOP + HOVER],
                           device=device).expand(n, 3)
        ok = False
        for attempt in range(4):
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = stn.data.root_pos_w + _qapply(stn.data.root_quat_w, loc)
            st[:, 3:7] = stn.data.root_quat_w
            tier.write_root_state_to_sim(st, all_ids)
            for _ in range(80):  # up to 2 s: fall + threading + settle
                step(3)
                seated, lz = scene.seat_state()
                if bool(seated[0, t, s]) \
                        and abs(float(lz[0, t, s]) - want_z) <= float(scene.cfg.lvl_tol) \
                        and bool(scene.settled(tier)[0]):
                    ok = True
                    break
            if ok:
                break
            seated, lz = scene.seat_state()
            print(f"[solve] move {mi} attempt {attempt + 1}: tier {'LMS'[t]} not "
                  f"seated at level {lvl} on {'ABT'[s]} — re-carrying", flush=True)
        assert ok, f"move {mi}: tier {'LMS'[t]} failed to seat on {'ABT'[s]} lvl {lvl}"
        assert not bool(scene._violated[0]), f"move {mi}: rule latch tripped"

    prev = s0
    for mi, (t, s, lvl, min_score) in enumerate(MOVES, start=1):
        move(mi, t, s, lvl)
        report(f"move {mi}")
        sc = print_score(f"P{mi} move {mi}: tier {'LMS'[t]} dropped onto "
                         f"{('post A', 'post B', 'the tray post')[s]} (level {lvl})")
        assert sc >= min_score - 1e-6, f"move {mi}: expected >= {min_score}, got {sc}"
        assert sc >= prev - 1e-6, f"move {mi}: score decreased {prev} -> {sc}"
        prev = sc

    # ---------------- final: settle + success ----------------------------------------------
    wait_settled(240)
    report("final")
    if not bool(scene.success()[0]):
        seated, lz = scene.seat_state()
        vels = [(round(float(t.data.root_lin_vel_w[0].norm()), 4),
                 round(float(t.data.root_ang_vel_w[0].norm()), 3)) for t in scene.tiers]
        print(f"[solve] FAIL-state: seated_C={[bool(seated[0, t, 2]) for t in range(3)]} "
              f"lz_C={[round(float(lz[0, t, 2]), 3) for t in range(3)]} "
              f"settled={[bool(scene.settled(t)[0]) for t in scene.tiers]} "
              f"vel(lin,ang)={vels} finite={bool(scene._finite()[0])} "
              f"violated={bool(scene._violated[0])}", flush=True)
        print("SIM_GEN_SOLVE: FAIL (no success after the 7 moves)", flush=True)
        os._exit(1)
    s8 = print_score("P8 pagoda rebuilt in the tray, settled")
    assert s8 >= 1.0 - 1e-6 and s8 >= prev - 1e-6, "success credit missing"

    # ---------------- persistence (>= 3 simulated seconds, no intervention) ----------------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                seated, lz = scene.seat_state()
                print(f"[solve] persist flicker @step {i}: "
                      f"seated_C={[bool(seated[0, t, 2]) for t in range(3)]} "
                      f"lz_C={[round(float(lz[0, t, 2]), 3) for t in range(3)]} "
                      f"settled={[bool(scene.settled(t)[0]) for t in scene.tiers]} "
                      f"violated={bool(scene._violated[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s9 = print_score("P-persist persistence 3.3 s (pagoda standing in the tray)")
    ok = hold and bool(scene.success()[0]) and s9 >= s8 - 1e-6
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
