"""Teleport solution for DeadmanStoveScene (sim_gen task
`libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i278`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. GAS (gravity + contact, the "turn on"): one root-state write carries the
   CAST-IRON BALLAST — selected by MASS over the identical-shape foam decoy —
   through a high waypoint (a real carry, clearly off the bench) to a
   free-space hover 10 mm above the dead-man pedal's top face, level, zero
   velocity. Then HANDS OFF: the ballast falls onto the pedal and its own
   1.2 kg weight overpowers the return spring, bottoms the pedal out past the
   15 mm gas threshold, and HOLDS it there. The scene's dead-man streak
   counter reads REAL pedal displacement made by contact; nothing is ever
   written to the pedal. Nothing latches — if the ballast were removed the
   spring would shut the valve (smoke proves exactly that).
2. TRANSPORT (teleport, the pan carry): one root-state write carries the pan
   to a free-space HOVER 20 mm above the cook plate, upright, handle south,
   zero velocity. The hover satisfies NOTHING (pan_on_burner needs the bottom
   within 12 mm of the plate top; asserted False at the hover).
3. SEATING (gravity + contact, hands-off): the pan falls onto the plate and
   settles flat. Every rubric fact (bottom at plate-top height, centre inside
   the radial window, upright, settled) is produced by ballistics and contact
   — and the gas stays open only because the ballast KEEPS resting on the
   pedal, untouched, through the whole seat + persistence window.

If a drop bounces the ballast off the pedal or a seat lands off-window, the
object is picked up again (fresh transport to the same free-space pose) and
re-dropped — a retry, not a cheat: the final configuration is still 100 %
contact-made.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds at every checkpoint.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i278.solve --headless [--seed N]
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

_qz = scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.deadman_stove")().build(num_envs=args.num_envs, device=device)
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

    def yaw_of(q) -> float:
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def depth() -> float:
        return float(scene.pedal_depth()[0])

    def report(tag: str) -> None:
        pp = (scene.pan.data.root_pos_w - scene.env_origins)[0]
        bp = (scene.ballast.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | depth={depth() * 1000:+.1f}mm "
              f"streak={int(scene._streak[0])} lit={bool(scene.lit()[0])} "
              f"ballast=({float(bp[0]):+.3f},{float(bp[1]):+.3f},{float(bp[2]):.3f}) "
              f"pan=({float(pp[0]):+.3f},{float(pp[1]):+.3f},{float(pp[2]):.3f}) "
              f"v_pan={float(scene.pan.data.root_lin_vel_w.norm(dim=-1)[0]):.3f} "
              f"on_burner={bool(scene.pan_on_burner()[0])} "
              f"latches=(carry={bool(scene._carry[0])},gas={bool(scene._gas[0])},"
              f"seat={bool(scene._seat[0])}) "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def write_pose(body, pos_w, quat_w) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)  # everything seats (pedal on its spring, objects on their slots)
    pp = (scene.pan.data.root_pos_w - scene.env_origins)[0]
    blp = (scene.ballast.data.root_pos_w - scene.env_origins)[0]
    dp = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    bu = (scene.burner.data.root_pos_w - scene.env_origins)[0]
    slot_x = [-c.slot_dx, 0.0, c.slot_dx]
    slots = scene.obj_slot[0].tolist()
    print(f"[solve] layout readback (seed {args.seed}): "
          f"burner=({float(bu[0]):+.3f},{float(bu[1]):+.3f}) "
          f"pan=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"pan_yaw={yaw_of(scene.pan.data.root_quat_w[0]):+.1f}deg "
          f"ballast=({float(blp[0]):+.3f},{float(blp[1]):+.3f}) "
          f"decoy=({float(dp[0]):+.3f},{float(dp[1]):+.3f}) "
          f"slots={slots} depth={depth() * 1000:+.2f}mm", flush=True)
    for name, body in (("pan", scene.pan), ("pedal", scene.pedal),
                       ("ballast", scene.ballast), ("decoy", scene.decoy)):
        assert torch.isfinite(body.data.root_pos_w).all(), f"NaN/inf {name} after settle"
    # masses readback (custom spawners: MassAPI must have won over density)
    for name, body, want, tol in (
            ("pan", scene.pan, c.pan_mass, 0.02),
            ("pedal", scene.pedal, c.pedal_mass, 0.005),
            ("ballast", scene.ballast, c.ballast_mass, 0.05),
            ("decoy", scene.decoy, c.decoy_mass, 0.005)):
        m = float(body.root_physx_view.get_masses().sum())
        assert abs(m - want) < tol, f"{name} mass readback {m} (want {want})"
    assert depth() < 0.002, f"pedal must rest at the TOP of its travel ({depth()})"
    assert not bool(scene.lit()[0]), "burner must start OFF"
    assert float(scene._up_w(scene.pan)[0, 2]) > 0.95, "pan must start upright"
    # each object on ITS slot per the permutation readback
    for j, (name, p) in enumerate((("pan", pp), ("ballast", blp), ("decoy", dp))):
        want_x = slot_x[slots[j]]
        assert abs(float(p[0]) - want_x) < c.slot_x_jitter + 0.02, \
            f"{name} must start on slot {slots[j]} (x={float(p[0]):+.3f}, want ~{want_x:+.3f})"
    report("reset")
    s0 = print_score("P0 reset+settle (gas shut, pedal up, objects on their slots)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: GAS — park the ballast on the dead-man pedal ----------------
    org = scene.env_origins[0]
    ped_top = float(org[2]) + c.pedal_top_rest_w

    def carry_waypoint():
        """A real high carry: clearly off the bench (latches the carry credit),
        clear of everything, then the release hover comes next."""
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0] = org[0] + c.pedal_x
        pos[:, 1] = org[1] + c.pedal_y - 0.10
        pos[:, 2] = org[2] + c.deck_h + 0.25
        return pos

    def park_pose():
        """Level hover, ballast bottom 10 mm above the pedal's top face, dead
        centre on the housing axis. The hover itself presses nothing."""
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0] = org[0] + c.pedal_x
        pos[:, 1] = org[1] + c.pedal_y
        pos[:, 2] = ped_top + c.block_h / 2 + 0.010
        return pos

    lit = False
    for attempt in range(3):
        write_pose(scene.ballast, carry_waypoint(), ident)
        env.step(no_action)  # the carry latch reads this height in post_step
        assert bool(scene._carry[0]), "high waypoint must latch the carry credit"
        write_pose(scene.ballast, park_pose(), ident)
        env.step(no_action)
        assert depth() < c.gas_on_depth, "the hover itself must not open the gas"
        max_depth = 0.0
        for _ in range(240):
            env.step(no_action)
            max_depth = max(max_depth, depth())
            if bool(scene.lit()[0]):
                lit = True
                break
        print(f"[solve] park attempt {attempt + 1}: lit={lit} "
              f"max_depth={max_depth * 1000:.1f}mm", flush=True)
        if lit:
            break
        report("park-retry")
    if not lit:
        report("park-FAIL")
        print("SIM_GEN_SOLVE: FAIL (ballast never held the gas open)", flush=True)
        os._exit(1)
    step(60)  # hands-off: the hold must be a stable rest, not a transient
    report("gas-open")
    assert bool(scene.lit()[0]), "the gas must STAY open under the resting ballast"
    assert depth() >= c.gas_on_depth + 0.002, \
        f"the ballast must hold the pedal well past the threshold ({depth() * 1000:.1f}mm)"
    assert not bool(scene.success()[0]), "no success before the pan is on the plate"
    s1 = print_score("P1 ballast parked; dead-man pedal held open by its weight")
    assert s1 >= s0 - 1e-6 and s1 >= 0.39, f"P1 score {s1} (expect 0.40 = carry+gas)"

    # ---------------- phase 2: pan onto the plate (transport hover + gravity seat) ---------
    def pan_hover_pose():
        """Hover 20 mm above the plate top, upright, handle SOUTH; target xy
        read back from the burner body itself. pan_on_burner() is False at the
        hover (z band is 12 mm)."""
        pos = scene.burner.data.root_pos_w.clone()
        pos[:, 2] = pos[:, 2] + c.plate_top_dz + c.pan_bottom_dz + 0.020
        quat = _qz(torch.full((n,), -math.pi / 2, device=device))
        return pos, quat

    pos, quat = pan_hover_pose()
    write_pose(scene.pan, pos, quat)
    env.step(no_action)  # refresh (falls < 1 mm in one substep)
    assert not bool(scene.pan_on_burner()[0]), \
        "the hover itself must NOT satisfy pan_on_burner()"
    for attempt in range(3):
        for i in range(240):
            env.step(no_action)
            if i > 30 and bool(scene.settled()[0]):
                break
        step(60)  # extra hands-off settle
        if bool(scene.pan_on_burner()[0]):
            break
        print(f"[solve] pan not seated after attempt {attempt + 1}; re-dropping",
              flush=True)
        report("pan-retry")
        pos, quat = pan_hover_pose()
        write_pose(scene.pan, pos, quat)
    else:
        if not bool(scene.pan_on_burner()[0]):
            report("pan-FAIL")
            print("SIM_GEN_SOLVE: FAIL (pan never seated on the plate)", flush=True)
            os._exit(1)
    report("seated")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the seat)", flush=True)
        os._exit(1)
    s2 = print_score("P2 pan seated on the plate; gas still held by the ballast")
    assert s2 >= s1 - 1e-6, "score decreased across the seat"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s (the dead-man hold never lapsed)")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — fail fast, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)
