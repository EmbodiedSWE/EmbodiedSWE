"""Teleport solution for FerryDoorCabinetScene (sim_gen task
`slide_cabinet_open_and_place_cups_i432`) — the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY: each cup is teleported (zero velocity, upright)
~3.5 cm above the roof PORT and released — gravity carries it through the port onto the
corridor floor. Nothing is ever teleported into the rubric state: the bay membership
window demands floor-rest height at the FAR end of a roofed corridor that a falling cup
cannot reach (the port is ~13 cm short of the bay), so the only way any cup enters the
bay here is the door's own closing stroke physically ramming it there.

The DOOR is never teleported. It is driven the way a hand would drag its handle: a
velocity-regulated external force along its prismatic track, re-set every step
(body frame == world frame — the track locks rotation). The stall authority
KV*V_CAP = 1.2 N comfortably exceeds the worst-case breakaway (two cups' floor
friction ~0.6 N) and the gain respects the one-substep wrench delay
(KV*dt/m = 12/(120*0.35) ~= 0.29 < 0.37). The wrench is zeroed once parked slow at
the target, so every stop is a true hard-stop park, not a force-assisted hover.

PLAN (read-only, from scene.describe()): one cup per closing stroke —
  P1 drive the door to the +stop (retracted clear of the port),
  P2 drop cup_a through the port (lands staged on the corridor floor),
  P3 drive the door to the -stop: the leading face rams cup_a down the roofed
     corridor into the bay (cup center ~ -0.046),
  P4 re-open to the +stop (cup_a stays put in the bay — no spring, nothing touches it),
  P5 drop cup_b through the re-exposed port,
  P6 drive the door to the -stop again: it rams cup_b, which shunts cup_a deeper
     (chain push: cup_a -> ~-0.102, cup_b -> ~-0.046), and the door parks fully
     CLOSED — the goal pose,
  P7 ring down until success() holds 120 consecutive steps,
  P8 hands-off persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: latched
credit), then `SIM_GEN_SOLVE: SUCCESS` only if success() still holds after the hold.

Run (forge): python -u -m simgen_tasks.slide_cabinet_open_and_place_cups_i432.solve --headless [--seed N]
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

# Door servo gains (one-substep wrench delay: KV*dt/m = 12/(120*0.35) ~= 0.29 < 0.37).
# A pure P velocity servo ASYMPTOTES near a stop: v_des -> 0 and the stall force falls
# under the cups' static breakaway (~0.3 N/cup) just short of the target — so a constant
# feedforward bias toward the target (zeroed only once parked) supplies break-away force.
KP_POS = 3.0   # outer position loop -> desired velocity (1/s)
V_CAP = 0.10   # desired-velocity cap (m/s)
KV = 12.0      # velocity loop -> force (N s/m)
FF = 0.9       # constant travel-direction bias (N) > 2-cup breakaway ~0.6 N
F_MAX = 4.0    # total force clamp (N)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ferry_door_cabinet")().build(num_envs=args.num_envs, device=device)
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

    def door_d() -> float:
        return float(scene.door_d()[0])

    def report(tag: str) -> None:
        bits = []
        for nm in c.cup_names:
            loc = scene._cup_local(nm)[0]
            bits.append(f"{nm}=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                        f"{float(loc[2]):+.3f})c stg={bool(scene.staged(nm)[0])} "
                        f"bay={bool(scene.seated(nm)[0])}")
        print(f"[solve] {tag:12s} | door d={door_d():+.4f} "
              f"closed={bool(scene.door_closed()[0])} | " + " | ".join(bits)
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

    def push_door(target: float, tag: str, max_steps: int = 1100) -> None:
        """Drag the door along its track with a velocity-regulated external force
        (re-set EVERY step; body frame == world frame — the track locks rotation).
        A constant FF bias toward the target defeats the P-servo stall asymptote
        (the cups' static friction); `near` means AT/PAST the target band (the hard
        stop clips travel, so pressing toward it can never overshoot). Zero the
        wrench once parked slow, then ring down: every park is a true hard-stop
        rest, not a force-assisted hover."""
        sgn = 1.0 if target > 0 else -1.0  # targets are the track stops
        f = torch.zeros(n, 1, 3, device=device)
        done = 0
        for i in range(max_steps):
            d = scene.door_d()
            v = scene.door.data.root_lin_vel_w[:, 1]
            v_des = (KP_POS * (target - d)).clamp(-V_CAP, V_CAP)
            fy = (KV * (v_des - v) + sgn * FF).clamp(-F_MAX, F_MAX)
            near = (d - target) * sgn >= -0.004
            slow = v.abs() < 0.02
            fy = torch.where(near & slow, torch.zeros_like(fy), fy)
            f[:, 0, 1] = fy
            scene.door.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
            env.step(no_action)
            if bool(near[0]) and bool(slow[0]):
                done += 1
                if done >= 12:
                    break
            else:
                done = 0
        scene.door.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        step(30)
        print(f"[solve] {tag}: door driven to d={door_d():+.4f} "
              f"(target {target:+.4f}, {i + 1} servo steps)", flush=True)

    def drop_cup(nm: str, tag: str) -> None:
        """TRANSPORT: teleport cup `nm` (zero velocity, upright) to ~3.5 cm above
        the roof over the port center, then hands off — gravity takes it through
        the port onto the corridor floor. Requires the door retracted clear."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.case_origin
        st[:, 1] += (c.port_y0 + c.port_y1) / 2
        st[:, 2] += 0.125
        st[:, 3] = 1.0
        scene.cups[nm].write_root_state_to_sim(st, all_ids)
        print(f"[solve] transport {nm} -> release above the port "
              f"(y={(c.port_y0 + c.port_y1) / 2:+.3f}c), door at d={door_d():+.4f}",
              flush=True)
        settle()
        report(tag)

    # ---------------- phase 0: reset, settle, plan readback --------------------------------
    step(90)
    d0 = door_d()
    print(f"[solve] readback (seed {args.seed}): door starts at d={d0:+.4f} "
          f"(range +/-{c.d0_range}, stops +/-{c.half_stroke})", flush=True)
    assert abs(d0) <= c.half_stroke + 0.003, "door start must lie on the track"
    for nm in c.cup_names:
        p = (scene.cups[nm].data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback: {nm} at ({float(p[0]):+.3f},"
              f"{float(p[1]):+.3f},{float(p[2]):+.3f})", flush=True)
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, "score must start ~0"

    # ---------------- phase 1: door to the +stop (port exposed) ----------------------------
    push_door(+c.half_stroke, "P1")
    assert door_d() >= c.half_stroke - 0.007, "door must park at the open stop"
    s1 = print_score("P1 door open, port exposed")
    assert s1 >= s0 - 1e-6, "score decreased across P1"

    # ---------------- phase 2: cup_a through the port --------------------------------------
    drop_cup("cup_a", "P2-cup_a")
    assert bool(scene.staged("cup_a")[0]), "cup_a must land staged on the corridor floor"
    s2 = print_score("P2 cup_a staged through the port")
    assert s2 >= s1 - 1e-6 and s2 >= 0.10, "staging credit missing"

    # ---------------- phase 3: closing stroke rams cup_a into the bay ----------------------
    push_door(-c.half_stroke, "P3")
    assert door_d() <= -(c.half_stroke - 0.007), "door must park at the closed stop"
    assert bool(scene.seated("cup_a")[0]), "the closing stroke must ram cup_a into the bay"
    s3 = print_score("P3 cup_a rammed into the bay, door closed")
    assert s3 >= s2 - 1e-6 and s3 >= 0.30, "bay credit missing"

    # ---------------- phase 4: re-open (cup_a stays put in the bay) ------------------------
    push_door(+c.half_stroke, "P4")
    assert door_d() >= c.half_stroke - 0.007, "door must re-park at the open stop"
    assert bool(scene.seated("cup_a")[0]), "cup_a must stay in the bay when the door opens"
    s4 = print_score("P4 door re-opened, cup_a still in the bay")
    assert s4 >= s3 - 1e-6, "score decreased across P4"

    # ---------------- phase 5: cup_b through the re-exposed port ---------------------------
    drop_cup("cup_b", "P5-cup_b")
    assert bool(scene.staged("cup_b")[0]), "cup_b must land staged on the corridor floor"
    s5 = print_score("P5 cup_b staged through the port")
    assert s5 >= s4 - 1e-6 and s5 >= 0.40, "second staging credit missing"

    # ---------------- phase 6: final closing stroke (chain ram) + ring-down ----------------
    push_door(-c.half_stroke, "P6")
    assert door_d() <= -(c.half_stroke - 0.007), "door must park fully closed"
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
    report("P6-ringdown")
    assert bool(scene.seated("cup_a")[0]), "cup_a must remain in the bay after the chain ram"
    assert bool(scene.seated("cup_b")[0]), "cup_b must be rammed into the bay"
    s6 = print_score("P6 both cups in the bay, door fully closed, rung down")
    assert s6 >= s5 - 1e-6, "score decreased across P6"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after ferry cycle + ring-down)", flush=True)
        os._exit(1)

    # ---------------- phase 7: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                dv = float(scene.door.data.root_lin_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: d={door_d():+.4f} "
                      f"door_v={dv:.4f} both={bool(scene.both_seated()[0])} "
                      f"closed={bool(scene.door_closed()[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s7 = print_score("P7 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s7 >= s6 - 1e-6
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
