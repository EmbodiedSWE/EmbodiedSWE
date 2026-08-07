"""Teleport solution for IceDoserScene (sim_gen task `get_ice_from_fridge_i52`)
— the task's legitimacy certificate.

Teleports are TRANSPORT-ONLY: the single teleport moves the blue cup from its
random slot to the ground under the outlet (a pick-and-place a Franka does by
sliding/pinching the cup — nothing about the machine is touched by it). Every
load-bearing interaction is real contact dynamics:

1. READ (perception): housing pose (FREE yaw +/- 180 deg + xy jitter), shuttle
   stroke state, cup bearings — all read back from the live scene, never assumed.
2. STAGE (teleport, transport-only): blue cup to the drop line. Done FIRST —
   the scene is unrecoverable if a ball is dispensed onto open ground.
3. PUMP x2 (applied force + contact): a horizontal force capped at 8 N — a
   fingertip push on a knob plate — drives the shuttle under a velocity-limited
   PD law along the housing's +x axis. Each cycle: press the protruding knob to
   the OPEN hard stop (the pocket crosses the outlet and its ONE ball falls into
   the cup; the shuttle body seals the hopper throat behind it), then press the
   other knob back to the CLOSED hard stop (the next ball shears cleanly into
   the pocket). Both endpoints are decided by knob-vs-housing contact, not by
   the controller. The force is RELEASED at every phase boundary; the score
   prints are taken hands-off.
4. STOP: after exactly target_count cycles the machine is left at its CLOSED
   stop. A third cycle would overfill and fail — the solve demonstrates the
   count, not a hold.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: stage
credit is latched), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.get_ice_from_fridge_i52.solve --headless [--seed N]
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

F_MAX = 8.0    # N — fingertip push on a knob plate
V_MAX = 0.08   # m/s — hand-on-knob slide speed


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ice_doser")().build(num_envs=args.num_envs, device=device)
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

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | d={float(scene.shuttle_d()[0]) * 1000:+7.2f}mm "
              f"blue={int(scene.cup_count('blue')[0])} "
              f"red={int(scene.cup_count('red')[0])} "
              f"kept={int(scene.retained().sum(dim=1)[0])} "
              f"staged={bool(scene.staged()[0])} "
              f"closed={bool(scene.shuttle_closed()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(180)  # ball column settles down the shaft; bottom ball seats in the pocket
    yaw = float(scene.housing_yaw[0])
    hp = (scene.housing.data.root_pos_w - scene.env_origins)[0]
    ab = math.degrees(float(scene.slot_ang[0, 0]))
    ar = math.degrees(float(scene.slot_ang[0, 1]))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"housing=({float(hp[0]):+.3f},{float(hp[1]):+.3f}) yaw={math.degrees(yaw):+.1f}deg "
          f"blue_slot={ab:+.1f}deg red_slot={ar:+.1f}deg "
          f"d={float(scene.shuttle_d()[0]) * 1000:+.2f}mm", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.shuttle_closed()[0]), "shuttle must start at the CLOSED stop"
    assert int(scene.retained().sum(dim=1)[0]) == c.n_balls, \
        "all balls must start sealed inside the machine"
    assert int(scene.cup_count("blue")[0]) == 0 and int(scene.cup_count("red")[0]) == 0
    assert not bool(scene.staged()[0]), "no cup may spawn already staged"
    s0 = print_score("P0 reset+settle (machine loaded, cups in random slots)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: STAGE the blue cup (transport-only teleport) ----------------------
    # A Franka slides/pinches the cup here; the teleport only relocates it.
    hq = scene.housing.data.root_quat_w
    loc = torch.tensor([c.out_c, 0.0, 0.0], device=device).expand(n, 3)
    pos = scene.housing.data.root_pos_w + quat_apply(hq, loc)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = pos[:, 0:2]
    st[:, 2] = 0.002
    st[:, 3:7] = hq
    scene.cups["blue"].write_root_state_to_sim(st, all_ids)
    step(120)
    report("staged")
    assert bool(scene.staged()[0]), "blue cup must sit staged under the outlet"
    s1 = print_score("P1 blue cup staged on the drop line (hands off)")
    assert s1 >= max(s0, c.w_stage) - 1e-6, f"P1 score {s1}"

    # ---------------- the pump: force-driven strokes between the two hard stops ------------------
    # FORCE-FRAME TRAP: on the forge pods `set_external_force_and_torque` applies
    # the given vector in a rotating frame (default is_global=False: the BODY
    # frame — a world-frame push must be pre-encoded with R_now^-1; on some
    # pods/eras the drag is rotation-since-reset instead). The semantics are
    # pod-dependent, so PROBE at runtime: start body-encoded and FLIP to raw
    # world if the shuttle measurably moves away from (or refuses to move
    # toward) the stroke target.
    dir_w = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=device)
    zero = torch.zeros(n, 1, 3, device=device)
    force = torch.zeros(n, 1, 3, device=device)
    enc_mode = {"m": 0, "flips": 0}   # 0: body-encode R_now^-1 * world; 1: raw world

    def encode(fw: torch.Tensor) -> torch.Tensor:
        if enc_mode["m"] == 0:
            return quat_apply_inverse(scene.shuttle.data.root_quat_w,
                                      fw.unsqueeze(0).expand(n, 3))
        return fw.unsqueeze(0).expand(n, 3)

    def stroke_to(tgt: float, pred, tag: str, max_steps: int) -> None:
        """PD-push the shuttle along the housing axis toward `tgt` (aimed PAST a
        hard stop so knob-vs-housing contact parks it) until `pred()`; then
        RELEASE the force and settle hands-off."""
        done = False
        d0 = float(scene.shuttle_d()[0])
        for i in range(max_steps):
            d = float(scene.shuttle_d()[0])
            v = float(scene.shuttle.data.root_lin_vel_w[0].dot(dir_w))
            v_des = max(-V_MAX, min(V_MAX, 4.0 * (tgt - d)))
            f = max(-F_MAX, min(F_MAX, 25.0 * (v_des - v)))
            force[:, 0, :] = encode(f * dir_w)
            scene.shuttle.set_external_force_and_torque(force, zero)
            env.step(no_action)
            if pred():
                done = True
                break
            # runtime probe: no measurable progress toward a far target after
            # 0.75 s of full push means the frame encoding is wrong — flip it.
            if i in (90, 240) and enc_mode["flips"] < 2:
                moved = float(scene.shuttle_d()[0]) - d0
                want = tgt - d0
                if abs(want) > 0.010 and moved * math.copysign(1.0, want) < 0.002:
                    enc_mode["m"] ^= 1
                    enc_mode["flips"] += 1
                    d0 = float(scene.shuttle_d()[0])
                    print(f"[solve] {tag}: force-frame probe FLIPPED encoding to "
                          f"mode {enc_mode['m']} (moved {moved * 1000:+.1f}mm "
                          f"toward {want * 1000:+.1f}mm)", flush=True)
        scene.shuttle.set_external_force_and_torque(zero, zero)
        step(90)  # hands off: ball finishes falling / next ball loads the pocket
        assert done, f"{tag}: predicate not reached after {max_steps} steps " \
                     f"(d={float(scene.shuttle_d()[0]) * 1000:+.1f}mm " \
                     f"blue={int(scene.cup_count('blue')[0])})"

    def pump_cycle(k: int) -> None:
        """One full close->open->close cycle; exactly ball #k drops."""
        stroke_to(c.stroke + 0.012,
                  lambda: int(scene.cup_count("blue")[0]) >= k,
                  f"open-{k}", 2400)
        report(f"open-{k}")
        assert int(scene.cup_count("blue")[0]) == k, \
            f"cycle {k}: expected exactly {k} balls in the blue cup"
        stroke_to(-0.012,
                  lambda: float(scene.shuttle_d()[0]) <= 0.006,
                  f"close-{k}", 2400)
        report(f"close-{k}")
        assert bool(scene.shuttle_closed()[0]), f"cycle {k}: shuttle must re-park CLOSED"
        assert int(scene.cup_count("blue")[0]) == k, \
            f"cycle {k}: count changed during the return stroke"
        assert int(scene.cup_count("red")[0]) == 0, "red cup must stay empty"

    # ---------------- phase 2: cycle 1 — first ball metered into the cup -------------------------
    pump_cycle(1)
    assert int(scene.retained().sum(dim=1)[0]) == c.n_balls - 1
    s2 = print_score("P2 cycle 1 complete (1 ball in the blue cup, shuttle re-closed)")
    assert s2 >= max(s1, c.w_stage + c.w_one) - 1e-6, f"P2 score {s2}"

    # ---------------- phase 3: cycle 2 — target count reached, machine parked --------------------
    pump_cycle(2)
    assert int(scene.retained().sum(dim=1)[0]) == c.n_balls - 2
    step(120)  # full settle for the live success gate
    report("parked")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (target count reached but success() is False)", flush=True)
        os._exit(1)
    s3 = print_score("P3 cycle 2 complete: exactly 2 balls, shuttle CLOSED, machine sealed")
    assert s3 >= max(s2, 0.999) - 1e-6, f"P3 score {s3}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
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
