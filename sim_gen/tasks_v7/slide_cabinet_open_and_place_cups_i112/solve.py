"""Teleport solution for ShutterCabinetScene (sim_gen task
`slide_cabinet_open_and_place_cups_i112`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: each cup is teleported (zero velocity, upright)
to a spot ~4 cm ABOVE the currently-UNCOVERED strip of its bay's top opening and
released — gravity carries it through the opening and seats it on the bay floor.
Nothing is ever teleported INTO the rubric state: the membership z band demands
floor-rest height, which only a fall through the opening provides, and a cup dropped
over a COVERED opening would land on the shutter and be rejected by that same band.

The SHUTTER is never teleported. It is driven like a hand would drive it: a
velocity-regulated external force (body frame == world frame — a prismatic body never
rotates) pushes it along its spawn-authored track until the hard stop / target is
reached, then the wrench is zeroed and the plate rings down. Gains respect the
one-substep wrench delay (K*dt/m ~= 0.14 << 1) and release only when slow.

PLAN (read-only, from scene.describe()): the captive shutter never uncovers both
openings at once, and the end state must have it covering BLUE — so serve blue FIRST:
  P1 push the shutter to the -stop (blue opening fully exposed),
  P2 drop the BLUE cup through the exposed blue strip,
  P3 push the shutter to the +stop (blue sealed, red fully exposed — already the
     required end pose),
  P4 drop RED cup A through the red strip (x offset -0.031),
  P5 drop RED cup B beside it (x offset +0.031), ring down to success,
  P6 hands-off persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success() still holds after
the hands-off hold.

Run (forge): python -u -m simgen_tasks.slide_cabinet_open_and_place_cups_i112.solve --headless [--seed N]
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

# Shutter servo gains (wrench acts one substep late: KV*dt/m = 6/(120*0.35) ~= 0.14).
KP_POS = 3.0   # outer position loop -> desired velocity (1/s)
V_CAP = 0.12   # desired-velocity cap (m/s)
KV = 6.0       # velocity loop -> force (N s/m)
F_MAX = 3.0    # force clamp (N)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shutter_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def shutter_c() -> float:
        return float(scene.shutter_c()[0])

    def report(tag: str) -> None:
        bits = []
        for nm in c.cup_names:
            loc = scene._cup_local(nm)[0]
            bits.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                        f"{float(loc[2]):+.3f})c seated={bool(scene.seated(nm)[0])}")
        print(f"[solve] {tag:12s} | shutter c={shutter_c():+.4f} "
              f"covers_blue={bool(scene.covers_blue()[0])} | " + " | ".join(bits)
              + f" | success={bool(scene.success()[0])} "
                f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(max_steps: int = 600) -> None:
        """Hands-off until the scene reads persistently still (or budget runs out)."""
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled()[0]):
                break

    def push_shutter(target: float, tag: str, max_steps: int = 900) -> None:
        """Drive the shutter along its track with a velocity-regulated external force
        (re-set EVERY step; body frame == world frame for a non-rotating prismatic
        body). Zero the wrench once parked slow near the target, then ring down."""
        f = torch.zeros(n, 1, 3, device=device)
        done = 0
        for i in range(max_steps):
            cpos = scene.shutter_c()
            v = scene.shutter.data.root_lin_vel_w[:, 1]
            v_des = (KP_POS * (target - cpos)).clamp(-V_CAP, V_CAP)
            fy = (KV * (v_des - v)).clamp(-F_MAX, F_MAX)
            near = (cpos - target).abs() <= 0.005
            slow = v.abs() < 0.02
            fy = torch.where(near & slow, torch.zeros_like(fy), fy)
            f[:, 0, 1] = fy
            scene.shutter.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
            env.step(no_action)
            if bool(near[0]) and bool(slow[0]):
                done += 1
                if done >= 12:
                    break
            else:
                done = 0
        scene.shutter.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        step(30)
        print(f"[solve] {tag}: shutter pushed to c={shutter_c():+.4f} "
              f"(target {target:+.4f}, {i + 1} servo steps)", flush=True)

    def drop_cup(nm: str, x_loc: float, y_loc: float, tag: str) -> None:
        """TRANSPORT: teleport cup `nm` (zero velocity, upright) to 4 cm above the
        top plate over cabinet-frame (x_loc, y_loc) — an uncovered strip of the
        target opening — then hands off: gravity takes it through the opening."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.case_origin
        st[:, 0] += x_loc
        st[:, 1] += y_loc
        st[:, 2] += 0.04
        st[:, 3] = 1.0
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        print(f"[solve] transport {nm} -> release 4 cm above opening at "
              f"({x_loc:+.3f},{y_loc:+.3f})c, shutter at {shutter_c():+.4f}", flush=True)
        settle()
        report(tag)

    # ---------------- phase 0: reset, settle, plan readback --------------------------------
    step(90)
    c0 = shutter_c()
    print(f"[solve] readback (seed {args.seed}): shutter starts at c={c0:+.4f} "
          f"(range +/-{c.c0_range}), stroke +/-{c.stroke}", flush=True)
    assert abs(c0) <= c.stroke + 0.003, "shutter start must lie on the track"
    for nm in c.cup_names:
        p = (scene.cups[nm].data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback: {nm} at ({float(p[0]):+.3f},"
              f"{float(p[1]):+.3f},{float(p[2]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, "score must start ~0"

    # ---------------- phase 1: shutter to -stop (blue opening exposed) ---------------------
    push_shutter(-c.stroke, "P1")
    assert shutter_c() <= -c.stroke + 0.007, "shutter must park at the -stop"
    assert not bool(scene.covers_blue()[0]), "blue opening must be exposed at -stop"
    s1 = print_score("P1 shutter at -stop, blue exposed")
    assert s1 >= s0 - 1e-6, "score decreased across P1"

    # ---------------- phase 2: BLUE cup through the exposed blue strip ---------------------
    # Exposed blue strip at -stop: y in [+0.075, +0.180] -> drop center +0.128.
    drop_cup("blue", 0.0, +0.128, "P2-blue")
    assert bool(scene.seated("blue")[0]), "blue cup must seat on the blue bay floor"
    s2 = print_score("P2 blue cup seated (fell through the opening)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.10, "blue seat credit missing"

    # ---------------- phase 3: shutter to +stop (blue sealed, red exposed) -----------------
    push_shutter(+c.stroke, "P3")
    assert shutter_c() >= c.stroke - 0.007, "shutter must park at the +stop"
    assert bool(scene.covers_blue()[0]), "+stop must seal the blue opening"
    s3 = print_score("P3 shutter at +stop, red exposed, blue sealed")
    assert s3 >= s2 - 1e-6, "score decreased across P3"

    # ---------------- phases 4+5: both RED cups through the exposed red strip --------------
    # Exposed red strip at +stop: y in [-0.180, -0.075] -> drop center -0.128.
    drop_cup("red_a", -0.031, -0.128, "P4-red_a")
    assert bool(scene.seated("red_a")[0]), "red_a must seat on the red bay floor"
    s4 = print_score("P4 red_a seated")
    assert s4 >= s3 - 1e-6, "score decreased across P4"

    drop_cup("red_b", +0.031, -0.128, "P5-red_b")
    # Ring-down: wait for success() to hold CONTINUOUSLY for 1 s.
    consec = 0
    for _ in range(2400):  # up to 20 s
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P5-ringdown")
    assert bool(scene.seated("red_b")[0]), "red_b must seat on the red bay floor"
    s5 = print_score("P5 all cups seated, shutter covering blue, rung down")
    assert s5 >= s4 - 1e-6, "score decreased across P5"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after deposits + ring-down)", flush=True)
        os._exit(1)

    # ---------------- phase 6: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                sv = float(scene.shutter.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: c={shutter_c():+.4f} "
                      f"shutter_v={sv:.4f} all={bool(scene.all_seated()[0])} "
                      f"covers={bool(scene.covers_blue()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s6 = print_score("P6 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s6 >= s5 - 1e-6
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
