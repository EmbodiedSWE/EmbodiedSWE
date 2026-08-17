"""Teleport solution for ValveBeamStoveScene (sim_gen task
`libero_kitchen_scene3_turn_on_the_stove_i280`) — the task's legitimacy
certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, the pick-and-place a gripper would do): one root-state
   write carries a steel ingot from its apron slot to a RELEASE POSE in the
   air ABOVE the weigh basket — aligned with the (tilted) basket mouth,
   strictly OUTSIDE the judged in-basket window (asserted: not in-basket,
   nothing latched, score unchanged), zero velocity.
2. DEPOSIT (contact dynamics): the ingot free-falls into the basket, strikes
   the floor/walls, and settles. The scene's count comes from latched
   in-basket-and-calm readback of REAL positions; nothing is ever written
   inside the judged window.
3. WEIGHING (pure statics — the actual "turn on"): with the second ingot
   seated, the combined load out-torques the counterweight (cfg-asserted
   >= 15 % margin at cube-at-wall worst case) and the beam swings on its real
   revolute pivot from the closed stop (-16 deg) to the open stop (+16 deg),
   entirely under gravity — no force is EVER applied to the beam by this
   script. The valve "opens" because statics holds it open.
4. The wood decoys are NEVER touched (their irrelevance is demonstrated by
   the cfg torque asserts and the smoke's decoy-load rejection).

If a drop bounces out or perches on a wall (it should not: 24 mm mouth
clearance, walls taller than a cube), the ingot is re-transported to the
release pose and dropped again — a retry, not a cheat: the final
configuration is still 100 % contact-made.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing along
the solve: deposits only add load and the beam only sinks further), then holds
HANDS-OFF for >= 3 simulated seconds after success() first turns True and
prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_i280.solve --headless [--seed N]
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
import traceback

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_quat_rotate_inv = scene_mod._quat_rotate_inv

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WDT = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                        os._exit(3)))
_WDT.daemon = True
_WDT.start()


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate body vectors `v` (N,3) into world by quats `q` (N,4 wxyz)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.valve_beam_stove")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def deg() -> float:
        return math.degrees(float(scene.beam_angle()[0]))

    def report(tag: str) -> None:
        ins = scene.steels_in_basket()[0]
        print(f"[solve] {tag:12s} | count={int(scene.count()[0])} "
              f"in={[int(b) for b in ins]} "
              f"beam={deg():+.1f}deg "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # beam falls from level onto the CLOSED stop; cubes seat on the apron
    print(f"[solve] layout readback (seed {args.seed}):", flush=True)
    for name, body in [(f"steel{k}", b) for k, b in enumerate(scene.steels)] + \
            [(f"wood{k}", b) for k, b in enumerate(scene.woods)]:
        pp = (body.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve]   {name}: ({float(pp[0]):+.3f},{float(pp[1]):+.3f},"
              f"{float(pp[2]):.3f})", flush=True)
    report("reset")
    for b in [*scene.steels, *scene.woods, scene.beam]:
        assert torch.isfinite(b.data.root_pos_w).all(), "NaN/inf after settle"
    for b in scene.steels:
        m_s = float(b.root_physx_view.get_masses().sum())
        assert abs(m_s - c.steel_mass) < 0.005, f"steel mass readback {m_s}"
        pp = (b.data.root_pos_w - scene.env_origins)[0]
        assert abs(float(pp[1]) - c.slot_y) < c.slot_jitter + 0.02, \
            "steel must start on the apron row"
    for b in scene.woods:
        m_w = float(b.root_physx_view.get_masses().sum())
        assert abs(m_w - c.wood_mass) < 0.005, f"wood mass readback {m_w}"
    m_b = float(scene.beam.root_physx_view.get_masses().sum())
    assert abs(m_b - c.beam_mass) < 0.01, f"beam mass readback {m_b}"
    # the authored CoM is what makes the counterweight win: behavioral check —
    # the beam must have fallen onto the CLOSED stop, basket end up
    assert deg() <= c.beam_lo_deg + 3.0, \
        f"empty beam must rest on the closed stop (saw {deg():+.1f} deg)"
    assert int(scene.count()[0]) == 0, "basket must start empty"
    s0 = print_score("P0 reset+settle (beam closed, basket empty, cubes on the apron)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1..2: deposit the two steel ingots ----------------------------
    def release_pose(bias_x: float, off_y: float):
        """World release pose above the basket mouth at the beam's CURRENT
        tilt: beam-frame (basket_x + bias, off_y, drop_z) with the beam's own
        orientation, so the cube faces align with the tilted basket. drop_z
        is ABOVE the judged window (in_z_hi) — the teleport satisfies
        nothing; only the fall and contacts do."""
        drop_z = c.in_z_hi + 0.012
        off = torch.tensor([[c.basket_x + bias_x, off_y, drop_z]],
                           device=device).expand(n, 3)
        q = scene.beam.data.root_quat_w
        pos = scene.beam.data.root_pos_w + _quat_rotate(q, off)
        return pos, q.clone()

    def deposit(k: int) -> None:
        b = scene.steels[k]
        for attempt in range(3):
            # uphill bias: at the closed stop the mouth tilts basket-end-up, so
            # a falling cube drifts toward -x (downhill); +x bias re-centres it.
            # Side-by-side y offsets: the two ingots each get their own half of
            # the basket, so the second never lands ON the first (a stacked
            # cube tops the walls and can bounce out).
            bias = 0.010 if deg() < -5.0 else 0.0
            off_y = -0.022 if k == 0 else 0.022
            pos, quat = release_pose(bias, off_y)
            st = torch.zeros(n, 13, device=device)
            st[:, 0:3] = pos
            st[:, 3:7] = quat
            b.write_root_state_to_sim(st, all_ids)
            # transport satisfies nothing: outside the judged window, not counted
            assert not bool(scene.steels_in_basket()[0, k]), \
                "the release pose must NOT be inside the judged basket window"
            assert not bool(scene.counted()[0, k]), \
                "the teleport itself must not produce a counted deposit"
            step(30)   # free fall + first contacts
            step(150)  # settle (and let the beam swing if the quota is met)
            print(f"[solve]   steel{k} attempt {attempt + 1}: "
                  f"in={bool(scene.steels_in_basket()[0, k])} beam={deg():+.1f}deg",
                  flush=True)
            if bool(scene.counted()[0, k]):
                return
            report(f"drop{k}-retry")
        report(f"drop{k}-FAIL")
        print(f"SIM_GEN_SOLVE: FAIL (steel {k} never seated)", flush=True)
        os._exit(1)

    prev = s0
    # --- P1: first ingot — beam must STAY closed (one steel is not enough) ---
    deposit(0)
    report("fed 1/2")
    assert int(scene.count()[0]) == 1, "count must track the seated ingot"
    assert deg() <= c.beam_lo_deg + 4.0, \
        f"one steel must NOT lift the counterweight (saw {deg():+.1f} deg)"
    assert not bool(scene.success()[0]), "no success with one ingot"
    s1 = print_score("P1 first steel ingot seated (beam still closed)")
    assert s1 >= prev - 1e-6 and s1 >= 0.25 - 1e-6, f"expected >= 0.25, got {s1}"
    prev = s1

    # --- P2: second ingot — the load now out-torques the counterweight ---
    deposit(1)
    # wait for the swing to the open stop (pure gravity; no force on the beam)
    for _ in range(40):
        if deg() >= c.theta_on_deg and bool(scene.success()[0]):
            break
        step(20)
    report("fed 2/2")
    assert int(scene.count()[0]) == 2, "both ingots must be counted"
    assert deg() >= c.theta_on_deg, \
        f"the loaded beam must sink past theta_on (saw {deg():+.1f} deg)"
    if not bool(scene.success()[0]):
        step(240)  # let everything calm
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the quota)", flush=True)
        os._exit(1)
    s2 = print_score("P2 second ingot seated; beam sank to the open stop")
    assert s2 >= prev - 1e-6 and abs(s2 - 1.0) < 1e-6, f"expected 1.0, got {s2}"
    prev = s2

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s (statics holds the valve open)")
    ok = hold and bool(scene.success()[0]) and s3 >= prev - 1e-6 \
        and abs(s3 - 1.0) < 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — fail fast, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)
