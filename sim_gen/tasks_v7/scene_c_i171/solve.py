"""Teleport solution for BeamBalanceScene (sim_gen task `scene_c_i171`) — the
task's legitimacy certificate.

Teleports are permitted for TRANSPORT only, and this solve uses them for
nothing else: each chosen candidate cube is teleported to where a gripper
would release it — inside the counter pocket's open mouth, orientation-matched
to the (tilted) beam, just above its seat — and then DROPPED. Every
load-bearing event is contact dynamics: the cube lands on the pocket floor
(or on the previous cube), the beam carries the new weight through the pocket
contacts and its bind-time revolute, swings up off its hard stop, and settles
at the equilibrium angle that the mass arithmetic dictates. No external
force/torque is ever applied to anything.

Plan per episode:
  P0  settle + perception: read the cargo band count k and side s back from
      the LIVE state (mass readback via get_masses guards the density-mass
      trap on every custom-spawned body); assert the beam rests ON its stop
      on the cargo side, score ~ 0.
  P1  decompose k into the unique subset of {1u, 2u, 4u} (binary digits),
      then drop the subset cubes one at a time into the counter pocket
      (heaviest first; later cubes stack on earlier ones). After the FIRST
      drop the `loaded` latch fires (score 0.20).
  P2  hands off: the beam swings up and settles level; success() turns True
      on the live state (score 1.0) and holds through a 3.3 s hands-off
      persistence window before `SIM_GEN_SOLVE: SUCCESS`.

The whole episode repeats on a second seed (fresh reset; the SIM_GEN_SCORE
stream is printed for the first seed only and is non-decreasing).

Run (forge): python -u -m simgen_tasks.scene_c_i171.solve --headless [--seed N]
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
    env = ENVS.get("simgen.beam_balance")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    stop = math.radians(c.stop_deg)

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        w = float(scene.beam.data.root_ang_vel_w[0].norm())
        print(f"[solve] {tag:12s} | tilt={float(scene.beam_tilt()[0]):+.4f} |w|={w:.3f} "
              f"seated={bool(scene.cargo_seated()[0])} "
              f"loaded={bool(scene.counter_loaded()[0])} "
              f"legal={bool(scene.cands_legal()[0])} "
              f"depot={bool(scene.spares_in_depot()[0])} "
              f"level={bool(scene.level()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def wait_settle(max_steps: int, min_steps: int = 60) -> None:
        """Step until the beam is slow and every mover slow (or the cap)."""
        for i in range(max_steps):
            env.step(no_action)
            if i >= min_steps and bool(scene.settled()[0]):
                break

    def drop_cand(units: int, seat_z: float) -> None:
        """The transport teleport: write candidate `units` to a release pose
        inside the counter pocket mouth — orientation-matched to the live beam,
        centre `seat_z` above the pocket floor plane (beam-local), zero
        velocity — then let contact dynamics carry the load."""
        cand = scene.cands[c.cand_units.index(units)]
        cs = -float(scene.cargo_side()[0])
        bq = scene.beam.data.root_quat_w
        lp = torch.zeros(n, 3, device=device)
        lp[:, 0] = cs * c.pock_x
        lp[:, 2] = c.floor_top + seat_z
        world = scene.beam.data.root_pos_w + scene._quat_apply(bq, lp)
        s13 = torch.zeros(n, 13, device=device)
        s13[:, :3] = world
        s13[:, 3:7] = bq
        cand.write_root_state_to_sim(s13, torch.arange(n, device=device))
        wait_settle(500)

    def episode(seed: int, announce: bool) -> bool:
        env.reset(seed=seed)
        s_prev = 0.0

        def print_score(tag: str) -> float:
            nonlocal s_prev
            s = float(scene.score()[0])
            print(f"[solve] phase boundary: {tag}", flush=True)
            if announce:
                print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
            assert s >= s_prev - 1e-6, f"score decreased: {s_prev:.4f} -> {s:.4f}"
            s_prev = s
            return s

        # ---------------- phase 0: settle, perception, mass audit -----------------------
        step(150)
        k = int(float(scene.layout[0, 0]))
        side = float(scene.layout[0, 1])
        tilt = float(scene.beam_tilt()[0])
        print(f"[solve] layout readback (seed {seed}): k={k} side={side:+.0f} "
              f"tilt={tilt:+.4f} (stop {stop:+.4f})", flush=True)
        # mass readback guards the density-mass trap on every custom-spawned body
        mb = float(scene.beam.root_physx_view.get_masses().sum())
        assert abs(mb - c.beam_mass) < 0.01, f"beam mass readback {mb:.3f}"
        for j, cargo in enumerate(scene.cargos, start=1):
            mj = float(cargo.root_physx_view.get_masses().sum())
            assert abs(mj - j * c.unit_mass) < 0.005, f"cargo_{j} mass readback {mj:.3f}"
        for u, cand in zip(c.cand_units, scene.cands):
            mu = float(cand.root_physx_view.get_masses().sum())
            assert abs(mu - u * c.unit_mass) < 0.005, f"cand_{u} mass readback {mu:.3f}"
        # candidates settled at their sampled scatter
        for i, cand in enumerate(scene.cands):
            live = (cand.data.root_pos_w - scene.env_origins)[0, :2]
            want = scene.layout[0, 2 + 2 * i:4 + 2 * i]
            assert float((live - want).norm()) < 0.05, \
                f"cand slot readback off: {live.tolist()} vs {want.tolist()}"
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert bool(scene.cargo_seated()[0]), "cargo must ride its start pocket"
        assert not bool(scene.counter_loaded()[0]), "counter pocket must start empty"
        assert bool(scene.spares_in_depot()[0]), "spares must start in the depot"
        # the un-countered cargo pins the beam at its stop, on the cargo side
        assert tilt * side > 0, "beam must tilt toward the cargo side"
        assert abs(tilt) > stop - 0.05, f"beam must rest on its stop: {tilt:+.4f}"
        assert not bool(scene.level()[0]) and not bool(scene.success()[0])
        s = print_score("P0 reset+settle (cargo pins the beam on its stop)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: drop the unique balancing subset ---------------------
        subset = [u for u in sorted(c.cand_units, reverse=True) if k & u]
        assert sum(subset) == k, f"subset arithmetic broke: {subset} for k={k}"
        print(f"[solve] balancing subset for k={k}: {subset} (units)", flush=True)
        stack = 0
        for i, u in enumerate(subset):
            drop_cand(u, seat_z=stack + c.cand_s / 2 + 0.006)
            stack += c.cand_s
            report(f"drop {u}u")
            assert bool(scene.counter_loaded()[0]), f"cand_{u} must ride the counter pocket"
            assert bool(scene.cargo_seated()[0]), "cargo must stay seated through the drop"
            if i == 0:
                s = print_score("P1 first candidate dropped into the counter pocket")
                assert s >= c.w_loaded - 1e-6, f"P1 score {s:.3f} below loaded credit"

        # ---------------- phase 2: hands off — swing up, settle level, success ----------
        won_at = None
        for i in range(1200):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("hands-off")
        assert won_at is not None and bool(scene.success()[0]), \
            f"success must turn True hands-off (tilt {float(scene.beam_tilt()[0]):+.4f})"
        assert abs(float(scene.beam_tilt()[0])) < c.theta_tol, "beam must stand level"
        s = print_score(f"P2 beam settled level hands-off ({won_at} steps)")
        assert s >= 1.0 - 1e-6, "success must score 1.0"

        # ---------------- persistence (>= 3 simulated seconds, hands-off) ---------------
        hold = True
        for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
            step(40)
            hold = hold and bool(scene.success()[0])
        report("persist")
        s_final = print_score("P-final persistence 3.3 s")
        ok = hold and bool(scene.success()[0]) and s_final >= 1.0 - 1e-6
        print(f"[solve] episode seed={seed}: {'OK' if ok else 'FAILED'}", flush=True)
        return ok

    ok = episode(args.seed, announce=True)
    ok = episode(args.seed + 1, announce=False) and ok

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
