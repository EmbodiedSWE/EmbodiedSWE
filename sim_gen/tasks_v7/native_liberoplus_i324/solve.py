"""Teleport solution for BlockMagazineScene (sim_gen task `native_liberoplus_i324`)
— the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. DISPENSING (contact/joint dynamics — never teleported): each cycle, a regulated
   force on the plunger knob (position-servoed on the joint coordinate, error
   clamped to 6 cm, force capped at 8 N ~ 2.5x the measured push resistance) drives
   the rod through its D6 guide to the 8 cm hard stop, shoving the BOTTOM block of
   the stack out through the one-block-high port; the same servo then pulls the rod
   back to its retracted stop and the stack drops one step under gravity alone.
   Every block that leaves the magazine leaves THROUGH THE PORT, pushed by the rod.
2. ROUTING (transport teleports + gravity): the freshly dispensed block — lying on
   the open floor outside the port — is teleported across free space to a HOVER
   above its destination (gray: above the discard bin interior; red: above the
   green pad) and RELEASED; gravity and contact produce the settled contained /
   resting end state. No teleport ever creates a scored predicate directly: the
   port-transit latch fires only during the physical push (the hover points are
   far outside the port window), and containment/on-pad states are reached by
   dropping.
3. The loop length is decided by the episode: cycles repeat until the RED block
   emerges (its stack depth G in {1,2,3} is randomized), so the solve reads the
   scene, not a memorized script.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.native_liberoplus_i324.solve --headless [--seed N]
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.block_magazine")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    mx, my = c.mag_pos

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    bodies = {"red": scene.red, "gray0": scene.grays[0], "gray1": scene.grays[1],
              "gray2": scene.grays[2]}

    def report(tag: str) -> None:
        parts = []
        for nm, b in bodies.items():
            p = loc(b)
            parts.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):.3f})")
        print(f"[solve] {tag:14s} | ins={float(scene.rod_insertion()[0]):.4f} "
              + " ".join(parts)
              + f" lat=[e{int(scene._eject[0].sum())} b{int(scene._binned[0].sum())} "
              f"t{int(scene._transit[0])} p{int(scene._pad[0])}] G={int(scene._G[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_rod_force() -> None:
        scene.rod.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def drive_rod(y_ref: float, *, done, max_steps: int, kp0: float = 120.0) -> bool:
        """Position-servo the plunger joint coordinate toward `y_ref` with a force
        along +y (error clamp 6 cm, force cap 8 N, damped) until `done()` holds.
        Escalates the gain if the rod stalls (measured, bounded). Returns done()."""
        kp, kd, fmax, errc = kp0, 8.0, 8.0, 0.06
        last_ins, stall = float(scene.rod_insertion()[0]), 0
        for i in range(max_steps):
            if done():
                break
            ins = float(scene.rod_insertion()[0])
            v = float(scene.rod.data.root_lin_vel_w[0, 1])
            err = max(-errc, min(errc, y_ref - ins))
            f = max(-fmax, min(fmax, kp * err - kd * v))
            fw = torch.tensor([0.0, f, 0.0], device=device).view(1, 1, 3).expand(n, 1, 3)
            scene.rod.set_external_force_and_torque(fw.contiguous(), zero_wrench,
                                                    env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i % 120 == 119:
                if abs(float(scene.rod_insertion()[0]) - last_ins) < 0.002:
                    stall += 1
                    kp = min(kp * 1.5, 400.0)
                    print(f"[solve] rod stall #{stall}: ins={ins:.4f}, kp->{kp:.0f}",
                          flush=True)
                last_ins = float(scene.rod_insertion()[0])
        clear_rod_force()
        return bool(done())

    def settle(pred, max_steps: int = 600, streak: int = 30) -> bool:
        quiet = 0
        for _ in range(max_steps):
            env.step(no_action)
            quiet = quiet + 1 if bool(pred()) else 0
            if quiet >= streak:
                return True
        return False

    def hover_drop(body, x: float, y: float, z: float) -> None:
        """Transport teleport: pose the body at a free-space hover, zero velocity;
        gravity does the rest."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3] = 1.0
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: reset, settle, baseline ---------------------------------------
    step(60)
    G = int(scene._G[0])
    ins0 = float(scene.rod_insertion()[0])
    stack = sorted((float(loc(b)[2]), nm) for nm, b in bodies.items())
    print(f"[solve] layout readback (seed {args.seed}): G={G} (grays below red), "
          f"stack bottom->top={[nm for _, nm in stack]}, rod ins={ins0:.4f}, "
          f"bin=({float(loc(scene.bin)[0]):+.3f},{float(loc(scene.bin)[1]):+.3f}) "
          f"pad=({float(loc(scene.pad)[0]):+.3f},{float(loc(scene.pad)[1]):+.3f})",
          flush=True)
    report("reset")
    assert 1 <= G <= 3, "red stack slot out of range"
    assert ins0 < 0.01, "plunger did not rest retracted at reset"
    assert stack[0][1] != "red" or G == 0, "a gray must start below the red block"
    for nm, b in bodies.items():
        assert bool(scene.in_magazine(loc(b).unsqueeze(0))[0]), f"{nm} not in the magazine"
    s_prev = print_score("P0 reset+settle")
    assert s_prev <= 0.05, "score not ~0 at reset"

    # ---------------- dispense loop ----------------------------------------------------------
    routed: set = set()
    bin_spots = [(-0.032, -0.032), (0.032, 0.032), (-0.032, 0.032)]
    phase = 0
    red_out = False
    for cycle in range(1, 5):
        assert not red_out
        # -- push: drive the rod to its far hard stop; the bottom block exits the port.
        def block_clear() -> bool:
            for nm, b in bodies.items():
                if nm in routed:
                    continue
                p = loc(b)
                if float(p[1]) - my > 0.060 and float(p[2]) < 0.15:
                    return True
            return False

        ok = drive_rod(c.stroke, done=block_clear, max_steps=1500)
        if not ok:  # pressed on the stop but block not clear yet: let it coast/slide
            settle(lambda: True, max_steps=60, streak=60)
        report(f"cycle{cycle}-push")
        assert block_clear(), f"cycle {cycle}: no block cleared the port after full stroke"

        # -- retract: press the rod back against its near hard stop (the tip parks
        # behind the shaft's inner face); the stack drops one step under gravity.
        ok = drive_rod(-0.02, done=lambda: float(scene.rod_insertion()[0]) <= 0.002,
                       max_steps=1500)
        assert ok, f"cycle {cycle}: plunger did not retract"
        inmag = [b for nm, b in bodies.items()
                 if nm not in routed and bool(scene.in_magazine(loc(b).unsqueeze(0))[0])]
        settle(lambda: all(bool(scene._still(b)[0]) for b in inmag),
               max_steps=480, streak=30)
        report(f"cycle{cycle}-retract")

        # -- identify the dispensed block (exactly one new body outside the magazine).
        outs = [nm for nm, b in bodies.items()
                if nm not in routed
                and not bool(scene.in_magazine(loc(b).unsqueeze(0))[0])
                and float(loc(b)[2]) < 0.15]
        assert len(outs) == 1, f"cycle {cycle}: expected exactly one dispensed block, got {outs}"
        nm = outs[0]
        body = bodies[nm]
        print(f"[solve] cycle {cycle}: dispensed {nm}", flush=True)

        # -- route it (transport teleport to a hover, then gravity).
        if nm == "red":
            assert bool(scene._transit[0]), \
                "red exited without latching the port transit (impossible: port is the only way)"
            pad = loc(scene.pad)
            hover_drop(body, float(pad[0]), float(pad[1]), 0.10)
            ok = settle(lambda: bool(scene.red_on_pad()[0]) and bool(scene._still(body)[0]))
            assert ok, "red block did not settle on the pad"
            red_out = True
        else:
            k = len(routed)
            b0 = loc(scene.bin)
            hover_drop(body, float(b0[0]) + bin_spots[k][0],
                       float(b0[1]) + bin_spots[k][1], 0.15)
            ok = settle(lambda: bool(scene.in_bin(loc(body).unsqueeze(0))[0])
                        and bool(scene._still(body)[0]))
            assert ok, f"gray block {nm} did not settle inside the bin"
        routed.add(nm)
        phase += 1
        report(f"cycle{cycle}-routed")
        s = print_score(f"P{phase} cycle {cycle}: dispensed {nm} through the port and routed it")
        assert s >= s_prev - 1e-6, "score decreased across a dispense cycle"
        s_prev = s
        if red_out:
            break

    assert red_out, "red block never emerged within 4 cycles"
    assert len(routed) == G + 1, f"routed {len(routed)} blocks, expected G+1={G + 1}"

    # ---------------- success + persistence (>= 3 simulated seconds, hands-off) -------------
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after routing the red block)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s_end = print_score("P-final persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s_end >= s_prev - 1e-6
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
    except Exception:  # noqa: BLE001 — die loudly, never hang in Kit teardown
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
