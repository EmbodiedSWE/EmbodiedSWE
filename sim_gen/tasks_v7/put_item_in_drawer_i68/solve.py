"""Teleport solution for BallastLiftScene (sim_gen task `put_item_in_drawer_i68`) —
the task's legitimacy certificate.

Teleports move objects across free space; they never do the task. Every load-bearing
interaction — the cube entering the bucket, the steels entering the pan, and above all
the LIFT itself — goes through contact dynamics and the real revolute pivot:

1. TRANSPORT CUBE (teleport): one pose write carries the red cube from its floor slot
   to a hover ABOVE the low bucket's open mouth — asserted OUTSIDE the containment
   volume. No rubric term moves (there is no approach term; load credit requires
   actual containment).
2. DEPOSIT (contact dynamics): the cube is RELEASED — it free-falls through the open
   mouth onto the tilted bucket floor, slides to the low wall under real friction and
   settles INSIDE. The beam stays parked on its rest stop (the cube only adds
   keep-down torque). No pose write touches the cube afterwards.
3. BALLAST 1 (teleport + contact): one pose write stages steel_a lying above the pan
   mouth (outside the pan volume — asserted), then it free-falls in. The drop impulse
   kicks the beam transiently, but ONE steel's torque (<= 0.166 kg*m) cannot beat the
   beam's bucket-side bias + the loaded cube (>= 0.207 kg*m): the beam returns to and
   stays on the rest stop — asserted. (Any transient lift the kick latches is honest
   physical lift of the loaded bucket and is reported.)
4. BALLAST 2 -> LIFT (teleport + contact): steel_b is staged the same way and dropped.
   With BOTH steels in the pan the balance flips: gravity alone — through the pivot
   joint, against the viscous pivot damping — hoists the loaded bucket ~40 deg up to
   its +20 deg limit stop, docking it under the shroud plate. The cube rides inside
   through real contacts. success() first turns True here. No pose write moves the
   beam at any point after reset.
5. PERSISTENCE: >= 3.3 more simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS`
   only if success() still holds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: all scene
credit is latched).

Run (forge): python -u -m simgen_tasks.put_item_in_drawer_i68.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_lift")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def deg() -> float:
        return math.degrees(float(scene.beam_angle()[0]))

    def pos(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def in_pan(body) -> bool:
        return bool(scene._in_box(body, c.pan_lo, c.pan_hi)[0])

    def report(tag: str) -> None:
        cu = pos(scene.cube)
        sa = pos(scene.steel_a)
        sb = pos(scene.steel_b)
        print(f"[solve] {tag:12s} | beam={deg():+6.1f} deg "
              f"cube=({float(cu[0]):+.3f},{float(cu[1]):+.3f},{float(cu[2]):.3f}) "
              f"sa=({float(sa[0]):+.3f},{float(sa[1]):+.3f},{float(sa[2]):.3f}) "
              f"sb=({float(sb[0]):+.3f},{float(sb[1]):+.3f},{float(sb[2]):.3f}) "
              f"in_bucket={bool(scene.cube_in_bucket()[0])} "
              f"steels_in_pan={float(scene.steels_in_pan()[0]):.0f} "
              f"load={bool(scene._load[0])} ballast={float(scene._ballast_max[0]):.2f} "
              f"lift={float(scene._lift_max[0]):.2f} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def stage(body, loc, q_rel=None) -> None:
        """ONE pose write (the teleport): `body` to the beam-local point `loc`,
        orientation slaved to the beam (optionally composed with q_rel), all
        velocities zeroed. Free space only — every stage pose is asserted OUTSIDE
        the containment volumes right after the write.

        Aiming note: a world-vertical free fall drifts in beam-local y by
        +tan(20 deg) ~ +0.364 per unit of local-z drop (the beam rests at -20 deg),
        so release points are offset in -y to cross the mouth centered."""
        loc_t = torch.tensor(loc, device=device, dtype=torch.float32).expand(n, 3)
        p = scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc_t)
        q = scene.beam.data.root_quat_w
        if q_rel is not None:
            q = quat_mul(q, torch.tensor(q_rel, device=device,
                                         dtype=torch.float32).expand(n, 4))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p
        st[:, 3:7] = q
        body.write_root_state_to_sim(st, all_ids)

    def settle(bodies, max_steps: int, quiet_need: int = 60,
               beam_omega: float = 0.15) -> bool:
        """Hands-off stepping until `bodies` and the beam hold still for
        `quiet_need` consecutive substeps."""
        quiet = 0
        for _ in range(max_steps):
            env.step(no_action)
            still = float(scene.beam.data.root_ang_vel_w[0].norm()) < beam_omega
            for b in bodies:
                still = still and float(b.data.root_lin_vel_w[0].norm()) < 0.05
            quiet = quiet + 1 if still else 0
            if quiet >= quiet_need:
                return True
        return False

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(150)  # the randomly-ajar beam falls onto its -20 deg rest stop here
    print(f"[solve] layout readback (seed {args.seed}): "
          f"cube=({float(pos(scene.cube)[0]):+.3f},{float(pos(scene.cube)[1]):+.3f}) "
          f"steel_a=({float(pos(scene.steel_a)[0]):+.3f},{float(pos(scene.steel_a)[1]):+.3f}) "
          f"steel_b=({float(pos(scene.steel_b)[0]):+.3f},{float(pos(scene.steel_b)[1]):+.3f}) "
          f"foam=({float(pos(scene.foam)[0]):+.3f},{float(pos(scene.foam)[1]):+.3f}) "
          f"beam={deg():+.1f} deg", flush=True)
    report("reset")
    assert deg() <= -17.0, f"beam did not fall onto its rest stop ({deg():+.1f} deg)"
    assert not bool(scene.cube_in_bucket()[0]), "cube spawned inside the bucket"
    assert float(scene.steels_in_pan()[0]) == 0.0, "a steel spawned inside the pan"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"non-zero score at reset ({s0:.3f})"

    # ---------------- phase 1: TRANSPORT cube (teleport, provably outside) ------------------
    # Hover above the low bucket's open mouth. Bucket containment tops out at beam-local
    # z 0.095; this pose sits at local z 0.152 — outside, asserted. The -y offset
    # compensates the fall drift so the cube lands near the bucket floor's center.
    stage(scene.cube, (0.0, c.arm_len - 0.038, 0.152))
    report("transported")
    assert not bool(scene.cube_in_bucket()[0]), \
        "hover pose already inside the bucket volume (teleport must stay outside)"
    s1 = print_score("P1 cube teleported to a hover above the open mouth (outside)")
    assert s1 >= s0 - 1e-6, "score decreased across transport"

    # ---------------- phase 2: DEPOSIT through the open mouth (gravity + contact) -----------
    ok = settle([scene.cube], 420, quiet_need=40)
    report("deposited")
    assert ok and bool(scene.cube_in_bucket()[0]), \
        "cube did not settle inside the low bucket"
    assert deg() <= -17.0, "beam left its rest stop during the deposit"
    s2 = print_score("P2 cube free-fell into the bucket and settled")
    assert s2 >= 0.24, f"load credit missing ({s2:.3f})"

    # ---------------- phase 3: BALLAST 1 (teleport outside, then gravity) -------------------
    # Steel lying axis-along-beam-x, above the pan mouth (pan containment tops out at
    # beam-local z 0.100; this pose sits at local z 0.133 — outside, asserted). Aimed
    # so the fall crosses the rim plane centered in the 60 mm mouth; the drift then
    # carries it onto the inner wall face, which guides it down into the pan.
    r2 = math.sqrt(0.5)
    stage(scene.steel_a, (-0.037, -c.arm_len - 0.008, 0.133),
          q_rel=(r2, 0.0, r2, 0.0))
    assert not in_pan(scene.steel_a), \
        "steel_a stage pose already inside the pan volume (teleport must stay outside)"
    ok = settle([scene.steel_a, scene.cube], 720, quiet_need=60)
    report("ballast 1")
    assert ok, "steel_a did not settle"
    assert in_pan(scene.steel_a), "steel_a did not land in the pan"
    assert bool(scene.cube_in_bucket()[0]), "cube left the bucket during ballast 1"
    assert deg() <= -15.0, \
        f"ONE steel held the beam off its rest stop ({deg():+.1f} deg) — torque budget broken"
    s3 = print_score("P3 steel_a dropped into the pan; one steel: beam stays down")
    assert s3 >= 0.36, f"ballast credit missing ({s3:.3f})"

    # ---------------- phase 4: BALLAST 2 -> the balance flips and LIFTS ---------------------
    stage(scene.steel_b, (0.037, -c.arm_len - 0.008, 0.133),
          q_rel=(r2, 0.0, r2, 0.0))
    assert not in_pan(scene.steel_b), \
        "steel_b stage pose already inside the pan volume (teleport must stay outside)"
    ok = settle([scene.steel_a, scene.steel_b, scene.cube], 1100, quiet_need=60)
    report("lifted")
    assert ok, "beam did not settle after the flip"
    assert float(scene.steels_in_pan()[0]) >= 2.0, "steel_b did not land in the pan"
    assert bool(scene.cube_in_bucket()[0]), "cube fell out during the lift"
    assert deg() >= c.lift_min_deg, \
        f"two steels failed to hoist the bucket to the top stop ({deg():+.1f} deg)"
    s4 = print_score("P4 steel_b dropped: balance flips, loaded bucket hoisted to the stop")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the lift)", flush=True)
        os._exit(1)
    assert s4 >= 1.0 - 1e-6, f"success live but score {s4:.3f}"

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    except BaseException as exc:  # noqa: BLE001 - die fast: Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
