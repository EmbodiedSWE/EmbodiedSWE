"""Teleport solution for StokerStoveScene (sim_gen task
`libero_kitchen_scene3_turn_on_the_stove_i193`) — the task's legitimacy
certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, the pick-and-place a gripper would do): one root-state
   write carries a briquette from its apron slot into the loading CHUTE — set
   down flat in the channel, 100 mm short of the port, zero velocity. The
   chute pose satisfies NOTHING (asserted: not inside, count unchanged).
2. FEEDING (contact dynamics, the actual "turn on"): a velocity-servo push
   (external force on the briquette, capped well below anything that could
   fake containment) slides the briquette along the channel, THROUGH the
   one-way flap — the briquette's own nose rotates the flap about its real
   revolute hinge against gravity — and over the sill, where gravity topples
   it into the firebox. The wrench is then ZEROED and the briquette settles
   hands-off. The scene's count comes from latched inside-and-calm readback of
   REAL positions; nothing is ever written inside the firebox.
   The flap's max hinge angle during each feed is asserted > 15 deg — the
   mechanism visibly worked, every time.
3. Repeat for 3 of the 4 briquettes. The butter stick is NEVER touched (its
   exclusion clause holds by leaving it alone — the honest way).

If a push stalls (briquette tumbled in the channel, wedged under the flap, or
jammed against an earlier briquette), the servo escalates its target speed and
force cap; if it still fails, the briquette is picked up again (fresh
transport to the same chute start) and re-fed — a retry, not a cheat: the
final configuration is still 100 % contact-made, and feeding is irreversible
so earlier briquettes keep their credit.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: fed
briquettes physically cannot come back out), then holds HANDS-OFF for >= 3
simulated seconds after success() first turns True and prints
`SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_i193.solve --headless [--seed N]
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
    env = ENVS.get("simgen.stoker_stove")().build(num_envs=args.num_envs, device=device)
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
        ins = scene.pellets_inside()[0]
        print(f"[solve] {tag:12s} | count={int(scene.count()[0])} "
              f"inside={[int(b) for b in ins]} "
              f"butter_in={bool(scene.butter_inside()[0])} "
              f"flap={math.degrees(float(scene.flap_angle()[0])):+.1f}deg "
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

    def zero_wrench(body) -> None:
        body.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=device), torch.zeros(n, 1, 3, device=device))

    stove_w = scene.stove.data.root_pos_w[0]  # world (env origin included)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.stove.data.root_pos_w)[0]

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)  # everything seats on the apron / chute stays empty / flap hangs shut
    print("[solve] layout readback (seed {}):".format(args.seed), flush=True)
    for name, body in [("butter", scene.butter)] + [
            (f"pellet{k}", p) for k, p in enumerate(scene.pellets)]:
        pp = (body.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve]   {name}: ({float(pp[0]):+.3f},{float(pp[1]):+.3f},"
              f"{float(pp[2]):.3f})", flush=True)
    report("reset")
    for p in scene.pellets:
        assert torch.isfinite(p.data.root_pos_w).all(), "NaN/inf after settle"
        m_p = float(p.root_physx_view.get_masses().sum())
        assert abs(m_p - c.pellet_mass) < 0.005, f"pellet mass readback {m_p}"
        pp = (p.data.root_pos_w - scene.env_origins)[0]
        assert abs(float(pp[1]) - c.slot_y) < c.slot_jitter + 0.02, \
            "briquette must start on the apron row"
    m_b = float(scene.butter.root_physx_view.get_masses().sum())
    m_f = float(scene.flap.root_physx_view.get_masses().sum())
    assert abs(m_b - c.butter_mass) < 0.005, f"butter mass readback {m_b}"
    assert abs(m_f - c.flap_mass) < 0.005, f"flap mass readback {m_f}"
    assert abs(math.degrees(float(scene.flap_angle()[0]))) < 3.0, \
        "flap must hang shut at reset"
    assert int(scene.count()[0]) == 0, "firebox must start empty"
    assert not bool(scene.butter_inside()[0]), "butter must start outside"
    s0 = print_score("P0 reset+settle (firebox empty, flap shut, items on the apron)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1..3: feed briquettes through the one-way flap ----------------
    chute_start = torch.zeros(n, 3, device=device)
    chute_start[:, 0] = stove_w[0]
    chute_start[:, 1] = stove_w[1] - 0.190
    chute_start[:, 2] = stove_w[2] + c.sill_z + c.pellet_s / 2 + 0.002
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
    kv, kx, kw = 2.5, 4.0, 4.0e-4  # kv*dt/m = 0.42 < 1; kw*dt/I = 0.39 < 1

    def feed(k: int) -> float:
        """Feed briquette k. Returns the max flap angle (deg) seen while it
        went through — the proof the mechanism worked."""
        p = scene.pellets[k]
        for attempt in range(3):
            write_pose(p, chute_start, ident)
            step(20)  # set down in the channel (transport satisfies nothing)
            assert not bool(scene.pellets_inside()[0, k]), \
                "the chute start pose must NOT be inside the firebox"
            v_des, cap = 0.15, 0.55
            max_flap, stall = 0.0, 0
            done = False
            for i in range(900):
                pos = p.data.root_pos_w[0]
                vel = p.data.root_lin_vel_w[0]
                r = rel(p)
                if float(r[1]) > -0.055 or bool(scene.pellets_inside()[0, k]):
                    done = True
                    break
                f = torch.zeros(n, 1, 3, device=device)
                f[:, 0, 0] = max(-0.15, min(0.15, kx * (float(stove_w[0]) - float(pos[0]))))
                f[:, 0, 1] = max(-cap, min(cap, kv * (v_des - float(vel[1]))))
                tau = (-kw) * p.data.root_ang_vel_w.view(n, 1, 3)
                p.set_external_force_and_torque(f, tau)
                env.step(no_action)
                max_flap = max(max_flap, math.degrees(float(scene.flap_angle()[0])))
                stall = stall + 1 if float(vel[1]) < 0.02 else 0
                if stall > 90:  # tumbled / wedged / jammed on an earlier briquette
                    v_des, cap = min(v_des + 0.10, 0.50), min(cap + 0.30, 1.50)
                    stall = 0
                    print(f"[solve]   pellet{k}: stall escalation -> "
                          f"v_des={v_des:.2f} cap={cap:.2f}", flush=True)
            zero_wrench(p)
            step(90)  # hands-off: gravity topples it in, everything calms
            print(f"[solve]   pellet{k} attempt {attempt + 1}: done={done} "
                  f"max_flap={max_flap:.1f}deg inside="
                  f"{bool(scene.pellets_inside()[0, k])}", flush=True)
            if bool(scene.counted()[0, k]):
                return max_flap
            report(f"feed{k}-retry")
        report(f"feed{k}-FAIL")
        print(f"SIM_GEN_SOLVE: FAIL (briquette {k} never fed)", flush=True)
        os._exit(1)

    prev = s0
    fed = 0
    for k in range(int(c.need)):
        max_flap = feed(k)
        fed += 1
        report(f"fed {fed}/{int(c.need)}")
        assert max_flap > 15.0, \
            f"the flap must visibly swing during a feed (saw {max_flap:.1f} deg)"
        assert int(scene.count()[0]) == fed, "count must track the fed briquettes"
        assert not bool(scene.butter_inside()[0]), "butter must stay outside"
        if fed < int(c.need):
            assert not bool(scene.success()[0]), \
                "no success before the quota is fed"
        s = print_score(f"P{fed} briquette {fed}/{int(c.need)} fed through the flap")
        assert s >= prev - 1e-6, "score decreased across a feed (latch broke?)"
        assert s >= 0.25 * fed - 1e-6 or bool(scene.success()[0]), \
            f"expected >= {0.25 * fed} after {fed} feeds, got {s}"
        prev = s

    if not bool(scene.success()[0]):
        step(120)  # last briquette may still be rocking
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the quota)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= prev - 1e-6 \
        and abs(s4 - 1.0) < 1e-6
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
