"""Teleport solution for SluiceHopperCatchScene (sim_gen task
`living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket_i214`) — the task's
legitimacy certificate.

The load-bearing interactions and how they are executed:

1. STAGE THE RECEPTACLE (transport teleport): one pose write places the green basket
   on the floor under the hopper's discharge lip, centered on the hopper axis. The
   landing spot is computed in the hopper's CURRENT readback frame — the hopper's
   side and heading are randomized, so the layout must be read, not memorized.
2. EXTRACT THE SLUICE GATE (applied force — never a pose write while constrained):
   a velocity-capped vertical lift force on the gate body pulls it up its rails
   against gravity, rail friction and the bottle's ramp-pressure on its back face.
   The rail channel, the sliding contacts, the growing gap and the moment the bottle
   squeezes under the rising slab are all contact physics. Only once the slab is
   measurably CLEAR of the rails (readback) is the force dropped and the free gate
   parked on open floor by a transport teleport.
3. THE DISCHARGE IS NEVER TOUCHED: from the instant the gate clears, nothing is
   written to the ketchup. It rolls down the ramp under gravity, over the lip, falls
   into the staged basket, rattles and settles — pure contact dynamics. The solver
   never contacts the target bottle at any point in the demonstrated solution.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sluice_hopper_catch")().build(num_envs=args.num_envs,
                                                         device=device)
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

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def vels(tag: str) -> None:
        bv = float(scene.basket.data.root_lin_vel_w[0].norm())
        kv = float(scene.ketchup.data.root_lin_vel_w[0].norm())
        kw = float(scene.ketchup.data.root_ang_vel_w[0].norm())
        gv = float(scene.gate.data.root_lin_vel_w[0].norm())
        print(f"[solve] {tag:12s} | basket v={bv:.4f} ketchup v={kv:.4f} w={kw:.4f} "
              f"gate v={gv:.4f}", flush=True)

    def report(tag: str) -> None:
        g_loc = scene._hopper_local(scene.gate.data.root_pos_w)[0]
        k_loc = scene._hopper_local(scene.ketchup.data.root_pos_w)[0]
        vels(tag)
        print(f"[solve] {tag:12s} | gate_loc=({float(g_loc[0]):+.3f},"
              f"{float(g_loc[1]):+.3f},{float(g_loc[2]):+.3f}) "
              f"ketchup_loc_x={float(k_loc[0]):+.3f} "
              f"staged={bool(scene.basket_staged()[0])} "
              f"staged_ever={bool(scene._staged_ever[0])} "
              f"clear={bool(scene.gate_clear()[0])} "
              f"extracted_ever={bool(scene._extracted_ever[0])} "
              f"released_ever={bool(scene._released_ever[0])} "
              f"k_in={bool(scene.contained(scene.ketchup)[0])} "
              f"bbq_in={bool(scene.contained(scene.bbq)[0])} "
              f"still={bool(scene._still()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, layout readback -------------------------------
    # 2 s: the ketchup rolls down the ramp inside the sealed hopper and comes to rest
    # pressed against the gate's back face — the state the whole task starts from.
    step(240)
    side = float(scene._side[0])
    hq = scene.hopper.data.root_quat_w[0]
    h_yaw = math.degrees(2.0 * math.atan2(float(hq[3]), float(hq[0])))
    h0 = (scene.hopper.data.root_pos_w - scene.env_origins)[0]
    g_loc = scene._hopper_local(scene.gate.data.root_pos_w)[0]
    k_loc = scene._hopper_local(scene.ketchup.data.root_pos_w)[0]
    b0 = (scene.basket.data.root_pos_w - scene.env_origins)[0]
    q0 = (scene.bbq.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): side={side:+.0f} "
          f"hopper=({float(h0[0]):+.3f},{float(h0[1]):+.3f}) yaw={h_yaw:+.1f}deg "
          f"gate_loc=({float(g_loc[0]):+.3f},{float(g_loc[1]):+.3f},"
          f"{float(g_loc[2]):+.3f}) ketchup_loc_x={float(k_loc[0]):+.3f} "
          f"basket=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"bbq=({float(q0[0]):+.3f},{float(q0[1]):+.3f})", flush=True)
    report("reset")
    assert abs(float(g_loc[0]) - c.gate_seat_x) < 0.02, "gate not seated in x"
    assert abs(float(g_loc[2]) - c.gate_seat_z) < 0.02, "gate not seated in z"
    assert float(k_loc[0]) < c.lip_x, "ketchup escaped the sealed hopper at reset?!"
    assert not bool(scene.contained(scene.ketchup)[0]), "ketchup spawned contained?!"
    s0 = print_score("P0 reset+settle (bottle rolled down against the gate)")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: STAGE THE BASKET (transport teleport) ------------------------
    # One pose write: basket upright on the floor, centered under the discharge lip in
    # the hopper's CURRENT frame (readback — side and yaw are randomized), long side
    # across the discharge direction. Nothing else moves.
    h_pos = scene.hopper.data.root_pos_w
    h_quat = scene.hopper.data.root_quat_w
    tgt_loc = torch.tensor([c.lip_x + c.land_dx, 0.0, 0.0], device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = h_pos + quat_apply(h_quat, tgt_loc)
    st[:, 2] = 0.004
    st[:, 3:7] = h_quat
    scene.basket.write_root_state_to_sim(st, all_ids)
    step(90)  # settle + streak latch (10 consecutive steps)
    report("staged")
    assert bool(scene.basket_staged()[0]), "basket not staged in the catch zone"
    assert bool(scene._staged_ever[0]), "staged latch did not set"
    s1 = print_score("P1 basket staged under the discharge lip")
    assert s1 >= s0 - 1e-6, "score decreased across staging"
    assert s1 >= c.w_staged - 0.01, f"staging credit missing, score {s1}"

    # ---------------- phase 2: EXTRACT THE GATE (applied force, velocity-capped) ------------
    # Bang-bang vertical lift on the gate body: full force while below the speed cap,
    # sub-weight sustain above it. Escalates on stall (rail friction + the bottle's
    # ramp-pressure are real and vary with the wobble). The rails, the sliding
    # contacts and the bottle squeezing out under the rising slab are all contact
    # physics — the gate is NEVER pose-written while it is inside the channel.
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    lift_f = 2.2           # N; gate weighs ~0.98 N
    sustain_f = 0.6 * c.gate_mass * 9.81
    v_cap = 0.30           # m/s
    clear_z = 0.46         # root local z: slab bottom 0.38 >> rail top 0.33
    best_z = float(scene._hopper_local(scene.gate.data.root_pos_w)[0, 2])
    stall = 0
    freed = False
    for i in range(900):
        vz = float(scene.gate.data.root_lin_vel_w[0, 2])
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 2] = lift_f if vz < v_cap else sustain_f
        scene.gate.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                 is_global=True)
        env.step(no_action)
        z = float(scene._hopper_local(scene.gate.data.root_pos_w)[0, 2])
        if z > best_z + 0.005:
            best_z, stall = z, 0
        else:
            stall += 1
            if stall >= 240:  # 2 s without 5 mm of progress: jammed — pull harder
                lift_f = min(lift_f + 0.6, 8.0)
                stall = 0
                print(f"[solve] P2 stall at z={z:.3f}, escalate lift to "
                      f"{lift_f:.1f} N", flush=True)
        if z >= clear_z:
            freed = True
            break
        if (i + 1) % 240 == 0:
            print(f"[solve] P2 t+{(i + 1) / 120.0:.1f}s gate_z={z:.3f} "
                  f"vz={vz:+.3f} lift={lift_f:.1f}N", flush=True)
    # force off, then PARK the free gate on open floor (transport teleport of an
    # unconstrained body) far from the hopper, basket and decoy.
    scene.gate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                             env_ids=all_ids, is_global=True)
    report("lifted")
    assert freed, f"gate never cleared the rails (best z {best_z:.3f})"
    assert bool(scene._extracted_ever[0]), "extract latch did not set during the lift"
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = 0.10
    st[:, 1] = -side * 0.45
    st[:, 2] = 0.010
    st[:, 3], st[:, 5] = math.cos(math.pi / 4), math.sin(math.pi / 4)  # lying flat
    st[:, 0:3] += scene.env_origins
    scene.gate.write_root_state_to_sim(st, all_ids)
    s2a = print_score("P2a gate extracted from the rails and parked")
    assert s2a >= s1 - 1e-6, "score decreased across the extraction"
    assert s2a >= c.w_staged + c.w_extract - 0.01, f"extract credit missing: {s2a}"

    # Hands off: gravity discharges the bottle — roll, lip, free fall into the basket,
    # rattle, settle. Nothing touches anything from here to the verdict.
    quiet = 0
    done = False
    for i in range(1500):
        env.step(no_action)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            done = True
            break
        if (i + 1) % 240 == 0:
            vels(f"P2b t+{(i + 1) / 120.0:.0f}s")
    report("discharged")
    assert bool(scene._released_ever[0]), "ketchup never crossed the lip plane"
    assert bool(scene.contained(scene.ketchup)[0]), "ketchup did not land in the basket"
    assert done, "discharged bottle did not settle to success"
    s2 = print_score("P2 ketchup discharged into the staged basket, settled")
    assert s2 >= s2a - 1e-6, "score decreased across the discharge"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after discharge)", flush=True)
        os._exit(1)

    # ---------------- phase 3: persistence (>= 3.3 simulated seconds, no intervention) ------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
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
    except Exception as e:  # noqa: BLE001 — fail fast, don't idle until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
