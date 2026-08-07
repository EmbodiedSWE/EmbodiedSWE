"""Teleport solution for BalanceScaleScene (sim_gen task `place_cups_i82`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY: each weight is teleported (at zero velocity) to a
spot ~2.5 cm ABOVE its target pan floor — computed from the LIVE beam pose readback, so
the drop target rides the heeled beam — and released. EVERY load-bearing interaction is
contact dynamics: gravity seats the weight on the pan floor, the pan rims and friction
keep it aboard, the beam heels onto its hard stop under one-sided load and swings back
as the counter-loads arrive, and the final LEVEL state is the beam's own statics ringing
down (angular damping) with the correct mass partition riding the pans. Nothing is ever
teleported INTO the rubric state: `in_pan` demands pan-floor contact height, which only
gravity provides, and `level` + `settled` demand the mechanism itself comes to rest.

PLAN (read-only): scene.describe() states the invariant m_big = m_small + m_mid, so the
unique balancing partition is {big} alone vs {mid, small} together. Load big into pan +y,
then mid and small side by side (x offsets -0.026 / +0.030 — x is torque-neutral, the
lever arm is y) into pan -y. Order is free; the beam resting on its stop mid-load is
expected and harmless.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.place_cups_i82.solve --headless [--seed N]
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
    env = ENVS.get("simgen.balance_scale")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
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

    def tilt_deg() -> float:
        return math.degrees(float(scene.tilt()[0]))

    def report(tag: str) -> None:
        bits = []
        for nm in c.cup_names:
            loc = scene._cup_local(nm)[0]
            bits.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                        f"{float(loc[2]):+.3f})b in={bool(scene.in_pan(nm)[0])}")
        print(f"[solve] {tag:14s} | tilt={tilt_deg():+.2f}deg | " + " | ".join(bits)
              + f" | success={bool(scene.success()[0])} "
                f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(max_steps: int = 900) -> None:
        """Hands-off until the whole mechanism is quiet (or the step budget runs out —
        a beam resting on its hard stop under one-sided load also reads settled)."""
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled()[0]):
                break

    def drop_into_pan(nm: str, side: float, x_off: float) -> None:
        """TRANSPORT: teleport weight `nm` (zero velocity, beam-aligned) to 2.5 cm above
        the pan floor of pan `side`, computed from the LIVE beam pose — then hands off:
        gravity + pan contact seat it, the beam responds with its own dynamics."""
        from isaaclab.utils.math import quat_apply

        beam_pos = scene.beam.data.root_pos_w[0]
        beam_quat = scene.beam.data.root_quat_w[0]
        local = torch.tensor([x_off, side * c.pan_y, c.cup_rest_z + 0.025], device=device)
        world = beam_pos + quat_apply(beam_quat.unsqueeze(0), local.unsqueeze(0))[0]
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = world
        st[:, 3:7] = beam_quat  # beam-aligned: lands flush even on a heeled pan
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        print(f"[solve] transport {nm} -> pan {side:+.0f} x_off={x_off:+.3f} "
              f"(release 25 mm above the floor, beam at {tilt_deg():+.2f} deg)",
              flush=True)
        settle()

    # ---------------- phase 0: reset, settle, plan readback --------------------------------
    step(90)
    m = scene.cup_mass[0]
    print(f"[solve] mass readback (seed {args.seed}): small={float(m[0]):.4f} "
          f"mid={float(m[1]):.4f} big={float(m[2]):.4f} "
          f"(sum check |big-(small+mid)|={abs(float(m[2] - m[0] - m[1])):.2e} kg)",
          flush=True)
    for nm in c.cup_names:
        p = (scene.cups[nm].data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback: {nm} at ({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f})", flush=True)
    report("reset")
    assert abs(tilt_deg()) < 1.0, "empty beam must rest level"
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, "score must start ~0"

    # ---------------- phase 1: the widest weight alone into pan +y -------------------------
    drop_into_pan("big", +1.0, 0.0)
    report("P1-big")
    assert bool(scene.in_pan("big")[0]), "big must ride pan +y after the drop"
    assert tilt_deg() < -math.degrees(c.tilt_max_rad), \
        "one-sided load must heel the beam past tilt_max (mechanism reality)"
    s1 = print_score("P1 big seated in pan +y (contact)")
    assert s1 >= s0 - 1e-6, "score decreased across P1"

    # ---------------- phase 2: the middle weight into pan -y -------------------------------
    drop_into_pan("mid", -1.0, -0.026)
    report("P2-mid")
    assert bool(scene.in_pan("mid")[0]), "mid must ride pan -y after the drop"
    s2 = print_score("P2 mid seated in pan -y (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across P2"

    # ---------------- phase 3: the narrowest weight beside it — the beam levels ------------
    drop_into_pan("small", -1.0, +0.030)
    # Ring-down: wait for success() to hold CONTINUOUSLY for 1 s (the scene's settled()
    # is a persistence counter, but the swing amplitude must also decay enough that the
    # beam stays under the stillness gate for the whole hands-off hold).
    consec = 0
    for _ in range(2400):  # up to 20 s
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P3-small")
    assert bool(scene.in_pan("small")[0]), "small must ride pan -y after the drop"
    s3 = print_score("P3 all three riding, beam ringing down")
    assert s3 >= s2 - 1e-6, "score decreased across P3"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after loading + ring-down)", flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                bl = float(scene.beam.data.root_ang_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: tilt={tilt_deg():+.2f} "
                      f"beam_w={bl:.4f} loaded={bool(scene.loaded()[0])} "
                      f"level={bool(scene.level()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
