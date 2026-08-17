"""Teleport solution for MicrowaveBallastDoorScene (sim_gen task
`libero_kitchen_scene6_close_the_microwave_i114`) — the task's legitimacy certificate.

Teleports are TRANSPORT ONLY: each mug is moved once through free air to a hover pose
2 cm above its slot in the hanging tray (read in the LIVE pan frame), then released
with zero velocity. Everything load-bearing is contact dynamics: the mug falls onto
the tray floor and is held by friction; the tray self-levels on its free pivot; and
the door closure itself is pure mechanism physics — two mugs' weight overpowers the
counterweight and swings the door shut with NOTHING applied (no wrench, no pose write
touches the door or the pan, ever). After one mug the door provably stays at the open
stop (statics condition B, asserted in the scene cfg), which this script checks live.

Plant numbers (why this converges): with both mugs the closing torque beats the
counterweight by >= 1.25x at every angle (worst 1.49x at the 85 deg open stop,
~0.9 N m net); the pivot pin sits IN the blade plane 30 mm beyond the blade tip and
the open stop is 85 deg, so the blade stays clear of the tray for the whole swing
(the failure mode of v1); effective inertia about the hinge ~0.15 kg m^2 with the
hanging load, angular damping 1.5/s -> the door swings shut in a couple of seconds
and presses onto the closed stop (+0.057 N m net holding torque at flush), where
damping kills the residual swing. Empty-door margin 3.69x at flush; one mug still
loses at the stop by 1.22x.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: milestone
credit is latched in the scene's post_step), then holds HANDS-OFF for >= 3.3
simulated seconds after success() first turns True and prints `SIM_GEN_SOLVE:
SUCCESS` only if success() still holds at the end.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene6_close_the_microwave_i114.solve --headless [--seed N]
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

MUG_SLOT_Y = 0.048   # pan-frame y of each mug slot (two mugs side by side)
HOVER_LZ = -0.050    # pan-frame z of the hover release (~1 cm above resting; deep
#                      enough that the hovering mug top clears the blade tip plane)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.microwave_ballast_door")().build(num_envs=args.num_envs,
                                                            device=device)
    scene = env.scene
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        ang = math.degrees(float(scene.door_angle()[0]))
        cnt = int(scene.mugs_in_pan()[0])
        la = scene._mug_local(scene.mug_a)[0]
        lb = scene._mug_local(scene.mug_b)[0]
        print(f"[solve] {tag:16s} | door={ang:+7.1f}deg in_pan={cnt} "
              f"a=({float(la[0]):+.3f},{float(la[1]):+.3f},{float(la[2]):+.3f}) "
              f"b=({float(lb[0]):+.3f},{float(lb[1]):+.3f},{float(lb[2]):+.3f}) "
              f"settled={bool(scene.settled()[0])} "
              f"L=({float(scene.latch_one[0]):.0f},{float(scene.latch_two[0]):.0f},"
              f"{float(scene.latch_swing[0]):.0f}) "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def place_mug(body, side: float) -> None:
        """TRANSPORT teleport: hover the mug 2 cm above its tray slot, read in the
        LIVE pan frame, upright, zero velocity — then gravity and friction do the
        rest. The door and the pan are never written."""
        from isaaclab.utils.math import quat_apply

        local = torch.tensor([0.0, side * MUG_SLOT_Y, HOVER_LZ],
                             device=device).expand(n, 3)
        world = scene.pan.data.root_pos_w + quat_apply(scene.pan.data.root_quat_w,
                                                       local)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = world
        st[:, 3] = 1.0  # upright, handle along pan x (fits the tray depth)
        body.write_root_state_to_sim(st, all_ids)

    def drop_into_pan(body, side: float, tag: str) -> None:
        """Place + settle + verify containment, with up to 3 retries."""
        for attempt in range(3):
            place_mug(body, side)
            step(90)  # 0.75 s: fall 2 cm, land, friction grabs
            for _ in range(8):  # up to 2 s more for the pendulum jolt to fade
                if bool(scene.in_pan(body)[0]) \
                        and float(body.data.root_lin_vel_w[0].norm()) < 0.08:
                    break
                step(30)
            if bool(scene.in_pan(body)[0]):
                report(f"{tag} landed" + (f" (retry {attempt})" if attempt else ""))
                return
            report(f"{tag} MISSED (attempt {attempt})")
        raise AssertionError(f"{tag}: mug failed to land in the tray after 3 tries")

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(120)  # door drifts 82 -> 85 deg stop; mugs drop their 2 mm float
    ang0 = math.degrees(float(scene.door_angle()[0]))
    pa = scene.mug_a.data.root_pos_w[0] - scene.env_origins[0]
    pb = scene.mug_b.data.root_pos_w[0] - scene.env_origins[0]
    print(f"[solve] layout readback (seed {args.seed}): door={ang0:+.1f}deg "
          f"mugA=({float(pa[0]):+.3f},{float(pa[1]):+.3f}) "
          f"mugB=({float(pb[0]):+.3f},{float(pb[1]):+.3f})", flush=True)
    report("reset")
    assert ang0 >= 78.0, f"door did not rest at the open stop ({ang0:+.1f} deg)"
    assert int(scene.mugs_in_pan()[0]) == 0, "a mug spawned inside the tray"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: first mug (door must STAY open) -----------------------------
    drop_into_pan(scene.mug_a, +1.0, "mug A")
    step(240)  # 2 s: one-mug statics — the counterweight holds the open stop
    report("one mug")
    ang1 = math.degrees(float(scene.door_angle()[0]))
    assert ang1 >= 70.0, \
        f"one mug closed the door ({ang1:+.1f} deg) — statics condition B broken"
    assert float(scene.latch_one[0]) > 0.5, "latch_one did not latch"
    s1 = print_score("P1 first mug in the tray (door stays open)")
    assert s1 >= s0 - 1e-6

    # ---------------- phase 2: second mug -> the mechanism closes the door -----------------
    drop_into_pan(scene.mug_b, -1.0, "mug B")
    assert float(scene.latch_two[0]) > 0.5, "latch_two did not latch"
    s2a = print_score("P2a both mugs in the tray")
    assert s2a >= s1 - 1e-6
    # hands off: ballast beats the counterweight; the door swings shut on its own
    swing_seen = False
    s_prev = s2a
    for i in range(3600):  # 30 s hard budget; the swing needs a few seconds
        env.step(no_action)
        if not swing_seen and float(scene.latch_swing[0]) > 0.5:
            swing_seen = True
            report(f"swing gate @ {i}")
            s2b = print_score("P2b loaded door below 45 deg (pure mechanism)")
            assert s2b >= s_prev - 1e-6
            s_prev = s2b
        if bool(scene.success()[0]):
            break
    report("closed")
    assert swing_seen, "the loaded door never crossed the swing gate"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after ballast closing)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)
    s2 = print_score("P2c door shut, all settled")
    assert s2 >= s_prev - 1e-6

    # ---------------- phase 3: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                print(f"[solve] persist flicker @step {i}: "
                      f"door={math.degrees(float(scene.door_angle()[0])):+.2f} "
                      f"in_pan={int(scene.mugs_in_pan()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
