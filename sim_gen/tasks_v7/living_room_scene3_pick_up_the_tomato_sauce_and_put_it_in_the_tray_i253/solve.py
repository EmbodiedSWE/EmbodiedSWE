"""Teleport solution for SauceBalanceScene (sim_gen task
`living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i253`) — the
task's legitimacy certificate.

Every teleport is TRANSPORT ONLY (carrying one free object across free space and
setting it down 3 mm above its rest, velocities zeroed); every load-bearing
outcome — the beam's tilt response, block stacking, the final equilibrium — is
pure joint + contact physics that runs hands-off after each set-down:

1. LOAD: one pose write stands the RED can 3 mm above the BROWN tray-pan floor
   (beam-frame placement, beam still level). It settles through contact; the beam
   then swings to its +12 deg hard stop under the can's torque BY ITSELF
   (asserted: the tilt readback pegs — the scene's own demonstration that the
   seed strategy alone earns only the 0.20 floor).
2. WEIGH (closed loop, tilt readback only): greedy exact counterweighting.
   Repeat: settle, read tilt. Tray side down -> transport the LARGEST untried
   block onto the blue-pan stack top (3 mm drop, beam-frame aligned so the block
   passes the one-block-wide pan). Counter side down -> the last block was too
   big: transport it back to its old floor slot and never retry it. |tilt|
   inside the level band -> stop. The per-episode can mass is NEVER read — the
   beam is the only scale, exactly as a robot would have to use it. (Greedy over
   {200,150,100,50} g terminates exactly for every sampled can mass in
   {150..350} g — asserted in the scene cfg's subset-sum check.)
3. HOLD: hands off. success() requires an unbroken 75-substep level streak with
   the can loaded and a counterweight present — sustainable only by the true
   torque equilibrium (an out-of-equilibrium beam accelerates out of the band in
   ~19 substeps by the authored dynamics).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: latched
credit), then keeps simulating >= 3.3 s after success() first turns True and
prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

The single-Franka-arm strategy for the same plan (side pinch of the can, top
pinch of each block, the same tilt-sign decision rule) lives in TASK.md.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i253.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sauce_balance")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    edges = [scene_mod._block_edge(m, c.block_density) for m in c.block_masses]

    # Seed AFTER build (the EnvCfg.build reseed trap); print describe() so distinct
    # seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def tilt() -> float:
        return float(scene.tilt_deg()[0])

    def report(tag: str) -> None:
        cp = (scene.can.data.root_pos_w - scene.env_origins)[0]
        inb = scene.blocks_in_counter()[0]
        print(f"[solve] {tag:12s} | tilt={tilt():+6.2f} deg "
              f"can=({float(cp[0]):+.3f},{float(cp[1]):+.3f},{float(cp[2]):.3f}) "
              f"in_tray={bool(scene.can_in_tray()[0])} "
              f"blocks_in_pan={[int(b) for b in inb]} "
              f"loaded={bool(scene._loaded[0])} counter={bool(scene._counter[0])} "
              f"near={bool(scene._near[0])} level={bool(scene._level[0])} "
              f"lvl_streak={int(scene._lvl_streak[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle_beam(cap: int = 720, min_steps: int = 180) -> None:
        """Run hands-off until the hinge is quiet: minimum dwell, then a 30-step
        streak of small hinge rate AND small tilt drift (a velocity threshold
        alone can fire at a swing's turning point)."""
        step(min_steps)
        streak, last = 0, tilt()
        for _ in range(cap):
            env.step(no_action)
            t_now = tilt()
            w = float(scene.beam.data.root_ang_vel_w[0].norm())
            streak = streak + 1 if (w < 0.06 and abs(t_now - last) < 0.02) else 0
            last = t_now
            if streak >= 30:
                break

    def place_on_beam(body, local_xyz: tuple[float, float, float]) -> None:
        """Transport teleport: set-down aligned with the (possibly tilted/yawed)
        beam frame, velocities zeroed. beam_point_w returns world incl. origins."""
        p_l = torch.tensor(local_xyz, device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.beam_point_w(p_l)
        st[:, 3:7] = scene.beam.data.root_quat_w
        body.write_root_state_to_sim(st, all_ids)

    def place_on_floor(body, world_xy: torch.Tensor, rest_z: float) -> None:
        """Transport teleport back to a free floor spot, upright, velocities zeroed."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:2] = world_xy
        st[:, 2] = rest_z + 0.003
        st[:, 3] = 1.0
        st[:, 2:3] += scene.env_origins[:, 2:3]
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline ---------------------------------------
    step(90)
    t0 = tilt()
    cp0 = (scene.can.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): tilt={t0:+.2f} "
          f"can=({float(cp0[0]):+.3f},{float(cp0[1]):+.3f},{float(cp0[2]):.3f})",
          flush=True)
    for i, nm in enumerate(scene.BLOCK_NAMES):
        bp = (scene.blocks[i].data.root_pos_w - scene.env_origins)[0]
        print(f"[solve]   block {nm}: ({float(bp[0]):+.3f},{float(bp[1]):+.3f},"
              f"{float(bp[2]):.3f})", flush=True)
    report("reset")
    assert abs(t0) < 1.0, f"empty beam not level at reset ({t0:+.2f} deg)"
    assert not bool(scene.can_in_tray()[0]), "can spawned inside the tray pan"
    assert float(cp0[2]) < 0.10, "can not resting on the floor"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"baseline score not ~0 ({s0:.3f})"

    # home slots for block removals (their own settled spawn spots, by then vacant)
    home_xy = [b.data.root_pos_w[:, 0:2].clone() for b in scene.blocks]

    # ---------------- phase 1: LOAD — stand the can in the brown tray pan --------------------
    place_on_beam(scene.can, (c.pan_x, 0.0, c.well_floor_top + c.can_h / 2 + 0.003))
    settle_beam()
    report("loaded")
    assert bool(scene.can_in_tray()[0]), "can did not seat in the tray pan"
    assert bool(scene._loaded[0]), "loaded latch not set"
    t1 = tilt()
    assert t1 > c.near_tol_deg, \
        f"the loaded beam should have swung to the tray-side stop ({t1:+.2f})"
    s1 = print_score("P1 can loaded, beam pegged tray-side")
    assert s1 >= s0 - 1e-6, "score decreased across the load"
    assert s1 >= c.w_load - 0.001, f"load credit missing ({s1:.3f})"

    # ---------------- phase 2: WEIGH — closed-loop greedy counterweighting -------------------
    # Decisions from the TILT READBACK ONLY (the can's sampled mass is never read).
    order = sorted(range(4), key=lambda i: -c.block_masses[i])  # heaviest first
    untried = list(order)
    stack: list[int] = []  # block indices bottom -> top

    for op in range(12):  # greedy bound: <= 4 adds + 4 removes (+ margin)
        t_now = tilt()
        if abs(t_now) <= c.level_tol_deg - 0.5 and stack:
            print(f"[solve] WEIGH: balanced at {t_now:+.2f} deg with "
                  f"{[scene.BLOCK_NAMES[i] for i in stack]}", flush=True)
            break
        if t_now > 0:  # tray side down: add the largest untried block
            assert untried, "tray side still down but no blocks left to try"
            i = untried.pop(0)
            z_loc = c.well_floor_top + sum(edges[j] for j in stack) + edges[i] / 2 + 0.003
            print(f"[solve] WEIGH op{op}: tilt {t_now:+.2f} -> ADD "
                  f"{scene.BLOCK_NAMES[i]} ({c.block_masses[i] * 1000:.0f} g)", flush=True)
            place_on_beam(scene.blocks[i], (-c.pan_x, 0.0, z_loc))
            stack.append(i)
        else:  # counter side down: last block too big — take it back out
            assert stack, "counter side down with an empty pan"
            i = stack.pop()
            print(f"[solve] WEIGH op{op}: tilt {t_now:+.2f} -> REMOVE "
                  f"{scene.BLOCK_NAMES[i]}", flush=True)
            place_on_floor(scene.blocks[i], home_xy[i], edges[i] / 2)
        settle_beam()
        report(f"weigh op{op}")
    else:
        raise AssertionError("greedy weighing did not terminate")

    assert bool(scene.can_in_tray()[0]), "can left the tray during weighing"
    assert bool(scene.blocks_in_counter()[0].any()), "no counterweight in the pan"
    assert bool(scene._counter[0]), "counterweight latch not set"
    s2 = print_score("P2 exact counterweight found (closed loop, tilt readback only)")
    assert s2 >= s1 - 1e-6, "score decreased across weighing"

    # ---------------- phase 3: HOLD — the level streak is pure hands-off physics -------------
    for _ in range(600):
        env.step(no_action)
        if bool(scene.success()[0]):
            break
    report("balanced")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (level streak never filled)", flush=True)
        os._exit(1)
    s3 = print_score("P3 sustained level equilibrium (success)")
    assert s3 >= 0.99, f"success did not map to score 1.0 ({s3:.3f})"

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 = 400 substeps = 3.33 s at 120 Hz
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
    # Hard exit: Kit teardown hangs — arm a timer, then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang to the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
