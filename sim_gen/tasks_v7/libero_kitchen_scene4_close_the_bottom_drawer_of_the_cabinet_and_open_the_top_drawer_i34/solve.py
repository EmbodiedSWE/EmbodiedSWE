"""Teleport solution for GumballMeterScene (sim_gen task
`libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_i34`)
— the task's legitimacy certificate.

This solve uses NO teleports at all: every ball movement is metered by the shuttle
airlock under gravity and contact, and the shuttle itself is driven only by applied
forces (what a hand on the exposed T-knob would do). The load-bearing interactions:

1. PERCEPTION: the quota K is read back from the episode state (a policy would count
   the green marker posts); the number of solve cycles is derived from it, never
   hard-coded.
2. CYCLING (applied force + contact): each dispense is a PULL — a PD force along the
   tail axis until the shuttle is ARRESTED by the front wall (contact decides the
   stop; the pocket is then over the drop hole and the shuttle's solid rear seals
   the silo) — then a hands-off wait while gravity drops the pocketed ball through
   the hole onto the basin ramp and it rolls clear, then a PUSH back to the rear
   wall, where gravity reloads exactly one ball from the silo into the pocket.
3. RESTRAINT: exactly K cycles are executed and not one more; the solver then parks
   the shuttle IN and keeps hands off. The (K+1)-th ball stays in the mechanism.
4. IDENTITY: the red decoy ball is never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: pull and
count credit are latched), then holds HANDS-OFF for >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_i34.solve
             --headless [--seed N]
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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gumball_meter")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | shut_x={float(scene.shuttle_x()[0]):+.3f} "
              f"parked_in={bool(scene.shuttle_parked_in()[0])} "
              f"basin={int(scene.basin_count()[0])}/{int(scene.k_target[0])} "
              f"stowed={int((scene.balls_stowed() & scene.active).sum())} "
              f"over={bool(scene._over[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def tail_axis() -> torch.Tensor:
        """World direction of the tower's +x axis (the pull direction)."""
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        return quat_apply(scene.tower.data.root_quat_w, ex)

    def drive(target_x: float, clamp: float, kd: float, steps: int) -> bool:
        """CYCLING: drive the shuttle toward `target_x` (tower frame) with a PD force
        along the tail axis — the channel's end walls arrest it (contact decides the
        park). This is what a hand on the T-knob does; the balls are never forced.
        Returns True if the shuttle settled within 6 mm of the target."""
        zero = torch.zeros(n, 1, 3, device=device)
        ok_frames = 0
        reached = False
        for _ in range(steps):
            axis = tail_axis()
            v = (scene.shuttle.data.root_lin_vel_w * axis).sum(-1)
            f_mag = (200.0 * (target_x - scene.shuttle_x()) - kd * v).clamp(-clamp, clamp)
            fw = axis * f_mag.unsqueeze(-1)
            # set_external_force_and_torque takes BODY-frame forces: convert each step.
            fb = quat_apply_inverse(scene.shuttle.data.root_quat_w, fw)
            scene.shuttle.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            env.step(no_action)
            near = abs(float(scene.shuttle_x()[0]) - target_x) < 0.006 \
                and abs(float(v[0])) < 0.02
            ok_frames = ok_frames + 1 if near else 0
            if ok_frames >= 5:
                reached = True
                break
        scene.shuttle.set_external_force_and_torque(zero, zero)
        step(20)
        return reached

    def wiggle(cycles: int = 8, amp: float = 8.0, half: int = 5) -> None:
        """Rapid alternating force on the knob (~12 Hz dither): mechanical agitation
        breaks a one-ball hopper arch, exactly like jiggling a sticky gumball slide.
        Forces go through the shuttle only; the balls are never touched."""
        zero = torch.zeros(n, 1, 3, device=device)
        for i in range(cycles):
            sgn = 1.0 if i % 2 == 0 else -1.0
            for _ in range(half):
                axis = tail_axis()
                fb = quat_apply_inverse(scene.shuttle.data.root_quat_w, axis * (sgn * amp))
                scene.shuttle.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
                env.step(no_action)
        scene.shuttle.set_external_force_and_torque(zero, zero)
        step(15)

    def seek(target_x: float, label: str, back_off, clamp: float, kd: float) -> None:
        """Drive toward a waypoint or hard stop, with un-wedge retries: a ball caught
        mid-transition (stack wipe, hole tip-in, pocket reload) can transiently arch
        against the throat; jiggling the knob and easing off lets gravity finish the
        drop, exactly as a hand on a sticky slide would."""
        for _ in range(6):
            if drive(target_x, clamp=clamp, kd=kd, steps=300):
                break
            here = float(scene.shuttle_x()[0])
            print(f"[solve] {label}: stuck at x={here:+.3f}, jiggling", flush=True)
            for bi, name in enumerate(scene.BALLS):          # diagnostic: jam geometry
                if not bool(scene.active[0, bi]):
                    continue
                p = scene._tower_local(scene.balls[name].data.root_pos_w)[0]
                print(f"[solve]   {name}: x={float(p[0]):+.4f} y={float(p[1]):+.4f} "
                      f"z={float(p[2]):+.4f}", flush=True)
            wiggle()
            if drive(target_x, clamp=clamp, kd=kd, steps=150):
                break
            drive(back_off(here), clamp=5.0, kd=40.0, steps=120)
            step(30)   # hands off: let the caught ball finish dropping
        assert abs(float(scene.shuttle_x()[0]) - target_x) < 0.010, \
            f"{label}: shuttle at x={float(scene.shuttle_x()[0]):+.3f}, want {target_x:+.3f}"

    def pull_out() -> None:
        # leg 1 — through the silo-wipe transition (the stack is stripped off the
        # departing pocket ball): full force, stop before the hole tip-in (~0.045)
        seek(0.040, "pull past silo wipe",
             back_off=lambda x: max(x - 0.020, 0.001), clamp=10.0, kd=30.0)
        # leg 2 — gentle creep so the pocket ball tips into the drop hole without
        # jamming; aim just past the stop so the front wall (contact) defines the park
        seek(c.travel + 0.002, "pull to OUT stop",
             back_off=lambda x: max(x - 0.015, 0.025), clamp=3.0, kd=60.0)

    def push_in() -> None:
        # leg 1 — back through the empty channel, stop before the reload region
        seek(0.030, "push to reload approach",
             back_off=lambda x: min(x + 0.020, 0.060), clamp=8.0, kd=25.0)
        # leg 2 — gentle creep while the waiting ball drops into the pocket
        seek(-0.002, "push to IN stop",
             back_off=lambda x: min(x + 0.015, 0.045), clamp=4.0, kd=60.0)
        assert bool(scene.shuttle_parked_in()[0]), "shuttle failed to park IN"

    def wait_drop(expected: int) -> None:
        """Hands-off: gravity drops the dispensed ball through the hole onto the
        ramp; wait until the basin holds `expected` settled ambers."""
        for _ in range(720):
            env.step(no_action)
            if int(scene.basin_count()[0]) >= expected and bool(scene.settled()[0]):
                break
        step(30)
        got = int(scene.basin_count()[0])
        assert got == expected, f"basin holds {got} balls, expected {expected}"

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(120)   # stock stack + pocket ball + decoy settle
    tp = (scene.tower.data.root_pos_w - scene.env_origins)[0]
    tq = scene.tower.data.root_quat_w[0]
    tyaw = math.degrees(2.0 * math.atan2(float(tq[3]), float(tq[0])))
    K = int(scene.k_target[0])
    stock = int(scene.stock[0])
    dp = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"tower=({float(tp[0]):+.3f},{float(tp[1]):+.3f}) yaw~{tyaw:+.1f}deg "
          f"quota_K={K} stock={stock} "
          f"decoy=({float(dp[0]):+.2f},{float(dp[1]):+.2f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.shuttle_parked_in()[0]), "shuttle must start parked IN"
    assert int(scene.basin_count()[0]) == 0, "basin must start empty"
    assert int((scene.balls_stowed() & scene.active)[0].sum()) == stock, \
        "all stock must start stowed in the silo column"
    s_prev = print_score("P0 reset+settle (silo loaded, basin empty, shuttle IN)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.03, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phases 1..K: one airlock cycle per quota ball ------------------------------
    for cyc in range(1, K + 1):
        pull_out()          # pocket over the hole; solid rear seals the silo
        wait_drop(cyc)      # gravity delivers exactly this cycle's ball
        push_in()           # pocket back under the silo; gravity reloads one ball
        step(60)            # reload + stack settle
        report(f"cycle-{cyc}")
        s = print_score(f"P{cyc} airlock cycle {cyc}/{K}: ball {cyc} dispensed, shuttle back IN")
        want = c.w_pull + c.w_count * cyc / K
        assert s >= s_prev - 1e-6, f"score decreased across cycle {cyc}"
        assert bool(scene.success()[0]) or s >= want - 1e-6, \
            f"cycle {cyc}: score {s:.3f} < expected {want:.3f}"
        s_prev = s

    # ---------------- restraint: NOT one cycle more ----------------------------------------------
    assert int(scene.basin_count()[0]) == K, "exact quota must be in the basin"
    assert not bool(scene._over[0]), "overfill latch must never fire in the solve"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (quota dispensed but success() is False)", flush=True)
        os._exit(1)

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P-final persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
