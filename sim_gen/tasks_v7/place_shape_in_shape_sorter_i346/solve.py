"""Teleport SOLUTION for CounterpoiseRackScene — transport-only teleports; the
seating, the beam's verdict, and all settling are pure gravity + contact.

Phases (SIM_GEN_SCORE printed at each boundary, non-decreasing):
  P0  reset + settle: plugs seated, beam level, cubes on the ground, score ~0.
      Audits the plug-mass readback (mass x radius = plug_c on both arms) and
      derives the unique torque-cancelling pocket assignment from the LIVE
      plug seats: orange (0.32 kg) at a free radius r on one arm, blue
      (0.16 kg) at the free radius 2r on the other arm.
  P1  LOAD: hover both cubes over their assigned pockets on the still-LEVEL
      beam (release z chosen ABOVE the seated z-band — asserted at the write,
      so the seat latches can only fire from physics) and release them one
      step apart; gravity drops each ~20 mm into its pocket before the beam
      can build any tilt. Retry ladder: if either cube misses, both go back
      to the ground, the beam re-levels on its own (the plugs still cancel),
      and the drop repeats slightly higher. Score 0.60 (both seat latches).
  P2  VERDICT: hands-off; the loaded beam swings, damps out and settles level
      (correct pairing => residual torque only from in-pocket play, <= ~2.2
      deg << 4.5 deg tolerance). Loop until success() holds. Score 1.00.
  P3  persistence: >= 3.3 simulated seconds untouched; success holds
      throughout. Then print `SIM_GEN_SOLVE: SUCCESS`.

Teleports only MOVE cubes through free space to hover poses above the beam;
nothing is ever teleported INTO a rubric-satisfying pose.

Run: python -m simgen_tasks.place_shape_in_shape_sorter_i346.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counterpoise_rack")().build(num_envs=args.num_envs,
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
        tl = math.degrees(float(scene.tilt()[0]))
        print(f"[solve] {tag:12s} | tilt={tl:+.2f}deg "
              f"h_seat={bool(scene.cube_seated(scene.cube_h)[0])} "
              f"l_seat={bool(scene.cube_seated(scene.cube_l)[0])} "
              f"plugs={bool(scene.plugs_seated()[0])} "
              f"level={bool(scene.level()[0])} still={int(scene._still_count[0])} "
              f"score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])}", flush=True)

    def beam_hover(x_local: float, z_local: float) -> tuple[torch.Tensor, torch.Tensor]:
        """World hover pose above the beam at beam-frame (x_local, 0, z_local),
        computed from the LIVE beam pose."""
        qb = scene.beam.data.root_quat_w.clone()
        local = torch.tensor([x_local, 0.0, z_local], device=device).expand(n, 3)
        return scene.beam.data.root_pos_w + _qapply(qb, local), qb

    def ground_park(body, dx: float) -> None:
        """Park a cube back on open ground (staging strip, clear of the rig)."""
        pos = scene.env_origins.clone()
        pos[:, 0] += 0.40 + dx
        pos[:, 1] += -0.45
        pos[:, 2] = c.cube / 2 + 0.003
        qid = torch.zeros(n, 4, device=device)
        qid[:, 0] = 1.0
        teleport(body, pos, qid)

    # ---------------- P0: reset + settle + audits ------------------------------------------------
    env.reset(seed=args.seed)
    step(90)
    report("P0 settle")

    m_l, m_r = scene.plug_masses()
    seat = scene._plug_seat[0]
    r_l, r_r = float(-seat[0]), float(seat[1])
    print(f"[solve] plug seats: left r={r_l:.3f} m={m_l:.3f} kg "
          f"(m*r={m_l * r_l:.5f}) | right r={r_r:.3f} m={m_r:.3f} kg "
          f"(m*r={m_r * r_r:.5f})", flush=True)
    assert abs(m_l * r_l - c.plug_c) < 1e-4, "left plug mass readback off ledger"
    assert abs(m_r * r_r - c.plug_c) < 1e-4, "right plug mass readback off ledger"
    assert r_l != r_r, "plug config violates left != right"
    assert bool(scene.plugs_seated()[0]), "plugs not seated at spawn"
    tilt0 = abs(math.degrees(float(scene.tilt()[0])))
    assert tilt0 < 2.0, f"locked beam not level at spawn ({tilt0:.2f} deg)"
    assert not bool(scene.cube_seated(scene.cube_h)[0]), "orange starts seated"
    assert not bool(scene.cube_seated(scene.cube_l)[0]), "blue starts seated"
    assert float(scene.score()[0]) <= 0.01, "nonzero score at spawn"
    print_score("P0")

    # -- derive the torque-cancelling assignment from the live plug seats ------------------------
    slots = tuple(c.slots)                      # (0.060, 0.120, 0.240)
    free = {-1.0: [s for s in slots if abs(s - r_l) > 1e-6],
            +1.0: [s for s in slots if abs(s - r_r) > 1e-6]}
    plans = []
    for side in (-1.0, +1.0):
        for r_h in free[side]:
            if any(abs(2 * r_h - s) < 1e-6 for s in free[-side]):
                plans.append((side, r_h))
    assert plans, f"no balancing assignment for plug config ({r_l}, {r_r})"
    side, r_h = plans[0]
    x_h, x_l = side * r_h, -side * 2 * r_h
    print(f"[solve] plan: orange at x={x_h:+.3f}, blue at x={x_l:+.3f} "
          f"(torque {c.cube_m_heavy * r_h:.5f} vs {c.cube_m_light * 2 * r_h:.5f})",
          flush=True)

    # ---------------- P1: LOAD (simultaneous hover-drops onto the level beam) --------------------
    # release z: cube bottom ~3.5 mm above the wall tops, centre ABOVE the seated
    # z-band [cube_z_lo, cube_z_hi] so the write itself can never satisfy a seat.
    seated = False
    for attempt in range(5):
        z_rel = 0.050 + 0.004 * attempt
        assert z_rel > c.cube_z_hi + 0.005, "release z would enter the seated band"
        pos_h, qb = beam_hover(x_h, z_rel)
        pos_l, _ = beam_hover(x_l, z_rel)
        teleport(scene.cube_h, pos_h, qb)
        assert not bool(scene.cube_seated(scene.cube_h)[0]), \
            "orange seated at the write — teleport touched the rubric band"
        step(1)
        teleport(scene.cube_l, pos_l, qb)
        assert not bool(scene.cube_seated(scene.cube_l)[0]), \
            "blue seated at the write — teleport touched the rubric band"
        step(150)
        if bool(scene.cube_seated(scene.cube_h)[0]) and \
                bool(scene.cube_seated(scene.cube_l)[0]):
            seated = True
            break
        print(f"[solve] drop attempt {attempt} missed "
              f"(h={bool(scene.cube_seated(scene.cube_h)[0])} "
              f"l={bool(scene.cube_seated(scene.cube_l)[0])}); re-staging",
              flush=True)
        ground_park(scene.cube_h, 0.15)
        ground_park(scene.cube_l, -0.15)
        # the plugs still cancel, so the unloaded beam re-levels on its own
        step(240)
    assert seated, "cubes never both seated after 5 drop attempts"
    report("P1 load")
    assert float(scene.score()[0]) >= 0.59, "seat credit missing after load"
    print_score("P1")

    # ---------------- P2: VERDICT (hands-off settle to level success) ----------------------------
    ok = False
    for i in range(2000):
        step(1)
        if bool(scene.success()[0]):
            ok = True
            break
    report("P2 verdict")
    tl = math.degrees(float(scene.tilt()[0]))
    assert ok, f"success never held after load (tilt={tl:+.2f} deg)"
    assert abs(tl) < c.level_tol_deg, "beam not level at success"
    print_score("P2")

    # ---------------- P3: persistence ------------------------------------------------------------
    for k in range(10):
        step(40)
        assert bool(scene.success()[0]), f"success dropped in persistence block {k}"
    report("P3 persist")
    print_score("P3")
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
