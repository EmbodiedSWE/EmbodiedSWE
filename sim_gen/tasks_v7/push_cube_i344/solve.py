"""Teleport SOLUTION for BalanceTriageScene — transport-only teleports, every
load-bearing interaction (weighing, seating, settling) through contact dynamics.

Phases (SIM_GEN_SCORE printed at each boundary, non-decreasing):
  P0  reset + settle: beam level, both cubes at their floor spots, score ~0.
      Oracle mass readback (heavy > light) sanity-checks the hidden state.
  P1  WEIGH: teleport the heavy cube to a hover point just above pan A and let it
      FALL in (contact seats it; the beam tips to the pan-A stop under gravity);
      then hover-drop the light cube onto pan B. The beam holds its verdict —
      tipped toward the true heavy side — until the `weighed` streak latch fires.
      The tilt DIRECTION is cross-checked against the oracle masses. Score 0.45.
  P2  DELIVER LIGHT: hover-drop the light cube from the raised pan into the BLUE
      bin (the beam stays pinned at the heavy stop). Score 0.60.
  P3  DELIVER HEAVY: hover-drop the heavy cube into the RED bin; the unloaded beam
      self-levels on its own (CoM below hinge). success() must hold. Score 1.00.
  P4  persistence: >= 3.3 simulated seconds hands-off; success holds throughout.
      Then print `SIM_GEN_SOLVE: SUCCESS`.

Teleports only MOVE cubes through free space to hover points ~12 mm above the
target surface; the seating, the beam's verdict tilt, and all settling are pure
gravity + contact. Nothing is ever teleported INTO a rubric-satisfying pose.

Run: python -m simgen_tasks.push_cube_i344.solve --headless [--seed N]
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

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qapply = task_scene._qapply

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.balance_triage")().build(num_envs=args.num_envs,
                                                   device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    ids = torch.arange(n, device=device)
    no_action = torch.empty(0, device=device)

    def step(k: int = 1) -> None:
        for _ in range(k):
            env.step(no_action)

    def teleport(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        """Transport only: write pose with zero velocity; gravity does the rest."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, ids)

    last_score = -1.0

    def print_score(tag: str) -> None:
        nonlocal last_score
        s = float(scene.score()[0])
        assert s >= last_score - 1e-6, f"score regressed at {tag}: {last_score} -> {s}"
        last_score = max(last_score, s)
        print(f"SIM_GEN_SCORE {s:.2f}", flush=True)

    def report(tag: str) -> None:
        ang = math.degrees(float(scene.beam_angle()[0]))
        oph = scene.on_pan(scene.cube_h)[0]
        opl = scene.on_pan(scene.cube_l)[0]
        print(f"[solve] {tag:12s} | beam={ang:+.1f}deg "
              f"h_pan=({bool(oph[0])},{bool(oph[1])}) "
              f"l_pan=({bool(opl[0])},{bool(opl[1])}) "
              f"weighed={bool(scene._weighed[0])} streak={int(scene._streak[0])} "
              f"h_red={bool(scene.in_bin(scene.cube_h, scene.bin_red)[0])} "
              f"l_blue={bool(scene.in_bin(scene.cube_l, scene.bin_blue)[0])} "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    def pan_hover(side: float, clearance: float) -> tuple[torch.Tensor, torch.Tensor]:
        """World hover pose above the pan center at +x (side=+1) / -x (side=-1),
        computed from the LIVE beam pose so it tracks the current tilt."""
        q = scene.beam.data.root_quat_w.clone()
        local = torch.tensor([side * c.arm_l, 0.0, c.cube / 2 + clearance],
                             device=device).expand(n, 3)
        return scene.beam.data.root_pos_w + _qapply(q, local), q

    def bin_hover(bin_body, clearance: float) -> tuple[torch.Tensor, torch.Tensor]:
        q = bin_body.data.root_quat_w.clone()
        local = torch.tensor([0.0, 0.0, c.bin_floor_t + c.cube / 2 + clearance],
                             device=device).expand(n, 3)
        return bin_body.data.root_pos_w + _qapply(q, local), q

    def drop_on_pan(body, side: float, name: str) -> None:
        """Hover-drop a cube onto a pan; retry with a slightly higher release if a
        bounce takes it off the tray. Seating is pure gravity + contact."""
        pan_idx = 0 if side > 0 else 1
        for attempt in range(4):
            pos, q = pan_hover(side, 0.012 + 0.004 * attempt)
            teleport(body, pos, q)
            step(110)
            if bool(scene.on_pan(body)[0, pan_idx]):
                return
            print(f"[solve] {name} missed pan (attempt {attempt}); retrying",
                  flush=True)
        raise AssertionError(f"{name} never seated on its pan")

    def drop_in_bin(body, bin_body, name: str) -> None:
        for attempt in range(4):
            pos, q = bin_hover(bin_body, 0.012 + 0.004 * attempt)
            teleport(body, pos, q)
            step(130)
            if bool(scene.in_bin(body, bin_body)[0]):
                return
            print(f"[solve] {name} missed bin (attempt {attempt}); retrying",
                  flush=True)
        raise AssertionError(f"{name} never seated in its bin")

    # ---------------- P0: reset + settle --------------------------------------------------------
    env.reset(seed=args.seed)
    step(90)
    report("P0 settle")
    mh, ml = scene.masses()
    print(f"[solve] oracle masses: heavy={mh:.3f} kg light={ml:.3f} kg", flush=True)
    assert mh > ml + 0.15, f"hidden masses not decisive ({mh:.3f} vs {ml:.3f})"
    ang0 = abs(math.degrees(float(scene.beam_angle()[0])))
    assert ang0 < 3.0, f"beam not level at spawn ({ang0:.1f} deg)"
    assert not bool(scene.on_pan(scene.cube_h).any()), "heavy cube starts on a pan"
    assert not bool(scene.on_pan(scene.cube_l).any()), "light cube starts on a pan"
    assert float(scene.score()[0]) <= 0.01, "nonzero score at spawn"
    print_score("P0")

    # ---------------- P1: WEIGH (drop heavy on pan A, light on pan B) ---------------------------
    drop_on_pan(scene.cube_h, +1.0, "heavy cube")
    ang = math.degrees(float(scene.beam_angle()[0]))
    print(f"[solve] heavy seated on pan A; beam={ang:+.1f} deg", flush=True)
    drop_on_pan(scene.cube_l, -1.0, "light cube")

    weighed = False
    for i in range(900):
        step(1)
        if bool(scene._weighed[0]):
            weighed = True
            break
    report("P1 weigh")
    assert weighed, "the weighed latch never fired with both cubes on opposite pans"
    ang = math.degrees(float(scene.beam_angle()[0]))
    # cross-check the physical verdict against the oracle: heavy is on pan A (+x),
    # so the beam must be tipped pan-A-down (positive angle)
    assert ang >= c.theta_min_deg - 0.5, \
        f"beam verdict does not point at the true heavy side ({ang:+.1f} deg)"
    assert float(scene.score()[0]) >= 0.44, "weigh credit missing"
    print_score("P1")

    # ---------------- P2: DELIVER LIGHT -> BLUE bin ----------------------------------------------
    drop_in_bin(scene.cube_l, scene.bin_blue, "light cube")
    report("P2 light")
    assert bool(scene._lbin[0]), "light-in-blue latch did not fire"
    assert float(scene.score()[0]) >= 0.59, "light delivery credit missing"
    print_score("P2")

    # ---------------- P3: DELIVER HEAVY -> RED bin -----------------------------------------------
    drop_in_bin(scene.cube_h, scene.bin_red, "heavy cube")
    # let the unloaded beam finish self-leveling and everything go quiet
    step(120)
    report("P3 heavy")
    assert bool(scene._hbin[0]), "heavy-in-red latch did not fire"
    assert bool(scene.success()[0]), "success() does not hold after delivery"
    print_score("P3")

    # ---------------- P4: persistence ------------------------------------------------------------
    for k in range(10):
        step(40)
        assert bool(scene.success()[0]), f"success dropped in persistence block {k}"
    report("P4 persist")
    print_score("P4")
    print("SIM_GEN_SOLVE: SUCCESS", flush=True)

    threading.Timer(10.0, lambda: os._exit(0)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
