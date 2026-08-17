"""Teleport solution for ShuttleVaultScene (sim_gen task
`libero_kitchen_scene10_..._i341`) — the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): one pose write carries the YELLOW butter from its floor spawn
   slot across free space to a hover 20 mm above the shuttle's porch shelf, long axis
   across the rail. It then FALLS under gravity and settles on the plate through real
   contact. The write satisfies no rubric clause: the hover endpoint is outside every
   band (asserted) — `boarded` demands a settled rest on the plate that only the
   contact landing produces, and `conveyed`/`vaulted` demand under-hood / in-cavity
   poses that only the shuttle stroke can produce.
2. DEPOSIT (applied force + contact — the core interaction): the shuttle is drawn
   rearward by a horizontal velocity-servo force at its CoM — the exact wrench of a
   fingertip drag on its green tower (the prismatic rail resists the small moment
   either way). Every millimetre of the butter's journey — the friction ride under
   the hood, the arrest at the orange stop bar, the floor withdrawing beneath it, the
   13 cm gravity drop through the hatch into the cavity — is plate/bar<->butter
   contact plus friction and gravity. The butter itself is NEVER forced or
   pose-written after the load drop.
3. SEAL (applied force + contact): the same servo pushes the shuttle forward to its
   closed joint stop, covering the hatch again. `resealed` latches only with the
   butter already settled in the cavity (geometry forces this order anyway: the plate
   IS the hatch cover — a closed shuttle admits nothing).
   Note on the pod force-frame quirk (wrenches rotated by the body's rotation since
   reset): the shuttle rides a prismatic joint and never rotates, so R_now = R_ref =
   identity and every mode coincides; raw world vectors are correct here.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

The intended single-Franka-arm strategy for the same plan (pinch the butter across
its 45 mm faces, lay it on the open porch shelf; hook-drag the green tower 0.21 m
rearward along the rail; push it 0.21 m back) lives in TASK.md as the embodiment
argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it_i341.solve --headless [--seed N]
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
    env = ENVS.get("simgen.shuttle_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    vx, vy = c.vault_pos

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.shuttle.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)

    def butter_rel() -> torch.Tensor:
        return scene._rel(scene.butter)[0]

    def report(tag: str) -> None:
        b = butter_rel()
        print(f"[solve] {tag:12s} | butter=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) op={float(scene.opening()[0]):+.4f} "
              f"|vb|={float(scene.butter.data.root_lin_vel_w[0].norm()):.4f} "
              f"|vs|={float(scene.shuttle.data.root_lin_vel_w[0].norm()):.4f} "
              f"on_plate={bool(scene.on_plate(scene.butter)[0])} "
              f"conv={bool(scene.conveyed_now()[0])} "
              f"in_vault={bool(scene.in_vault(scene.butter)[0])} "
              f"closed={bool(scene.closed()[0])} "
              f"brick_out={bool(scene.brick_out()[0])} "
              f"app={float(scene._app_max[0]):.3f} "
              f"boarded={bool(scene._boarded[0])} conveyed={bool(scene._conveyed[0])} "
              f"vaulted={bool(scene._vaulted[0])} resealed={bool(scene._resealed[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def fail(msg: str) -> None:
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL ({msg})", flush=True)
        os._exit(1)

    def servo(v_des: float, *, gain: float, floor: float, cap: float) -> float:
        """One-step velocity-servo world force on the shuttle along the rail (+x);
        returns the commanded force. Raw world vector is correct: prismatic bodies
        never rotate (see module docstring)."""
        v = float(scene.shuttle.data.root_lin_vel_w[0, 0])
        f = gain * (v_des - v)
        sgn = 1.0 if v_des >= 0 else -1.0
        if abs(v) < 0.02 and sgn * f < floor:
            f = sgn * floor  # break static friction / damping hold
        f = max(-cap, min(cap, f))
        fw = torch.zeros(n, 1, 3, device=device)
        fw[0, 0, 0] = f
        scene.shuttle.set_external_force_and_torque(fw, zero_wrench, env_ids=all_ids,
                                                    is_global=True)
        env.step(no_action)
        return f

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(60)
    b0 = butter_rel()
    k0 = scene._rel(scene.brick)[0]
    op0 = float(scene.opening()[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"butter=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"brick=({float(k0[0]):+.3f},{float(k0[1]):+.3f}) op0={op0:+.4f}", flush=True)
    report("reset")
    assert c.open_init_range[0] - 0.006 <= op0 <= c.open_init_range[1] + 0.006, \
        f"shuttle not parked ajar in its init range (op0={op0:+.4f})"
    assert not bool(scene.closed()[0]), "shuttle must start ajar, not closed"
    assert not bool(scene.in_vault(scene.butter)[0]), "butter spawned inside the vault"
    assert bool(scene.brick_out()[0]), "brick spawned inside the vault"
    assert not bool(scene.on_plate(scene.butter)[0]), "butter spawned on the plate"
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.02, f"baseline score not ~0 ({s0:.3f})"

    # ---------------- phase 1: TRANSPORT — lay the butter on the porch ----------------------
    # One pose write to a hover 20 mm above the porch centre, long axis across the
    # rail (yaw 90 deg); gravity lands it through real contact. The porch spans
    # [0.03 + op0, hood_x0]; its centre is 0.075 + op0/2. Asserted: the hover
    # endpoint satisfies no band (above the on-plate z band, outside conveyed/vault).
    lx = 0.075 + op0 / 2
    hover_z = c.ceil_top + c.plate_t + c.block_size[2] / 2 + 0.020
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = vx + lx, vy, hover_z
    st[:, 3] = 0.7071068  # yaw 90 deg: long axis across the rail
    st[:, 6] = 0.7071068
    st[:, 0:3] += scene.env_origins
    scene.butter.write_root_state_to_sim(st, all_ids)
    assert not bool(scene.on_plate(scene.butter)[0]), "hover endpoint already on-plate"
    assert not bool(scene.conveyed_now()[0]), "hover endpoint already conveyed"
    assert not bool(scene.in_vault(scene.butter)[0]), "hover endpoint already in-vault"
    step(90)
    report("loaded")
    b = butter_rel()
    assert abs(float(b[0]) - lx) < 0.020 and abs(float(b[1])) < 0.020, \
        "butter did not settle on the porch centre"
    assert c.plate_z_lo < float(b[2]) < c.plate_z_hi, \
        f"butter not flat on the plate (z={float(b[2]):.4f})"
    assert bool(scene.on_plate(scene.butter)[0]), "butter not on-plate after landing"
    assert bool(scene._boarded[0]), "boarded latch not set after the contact landing"
    s1 = print_score("P1 butter laid on the porch (contact landing)")
    assert s1 >= s0 - 1e-6, "score decreased across loading"
    assert float(scene._app_max[0]) > 0.90, "approach credit not earned at the porch"

    # ---------------- phase 2: DEPOSIT — draw the shuttle open; the floor leaves ------------
    # Velocity-servo force on the shuttle CoM only. The butter is untouched: it rides
    # the plate under the hood by friction, the orange stop bar arrests it over the
    # hatch, and the receding plate drops it through the hatch into the cavity. Stop
    # when the butter is inside the cavity or the carriage reaches its limit.
    conveyed_printed = False
    floor_f = 0.8
    win_i, win_op = 0, float(scene.opening()[0])
    stalled = 0
    done = False
    for i in range(3600):
        op = float(scene.opening()[0])
        if bool(scene.in_vault(scene.butter)[0]):
            done = True
            break
        if op >= c.stroke - 0.001:
            done = True
            break
        v_des = 0.08 if op < 0.11 else 0.04  # slow down once the hatch starts opening
        servo(v_des, gain=8.0, floor=floor_f, cap=4.0)
        if not conveyed_printed and bool(scene._conveyed[0]):
            clear_forces()
            report("conveyed")
            s_mid = print_score("P2a butter carried under the hood (friction ride)")
            assert s_mid >= s1 - 1e-6, "score decreased across the carry"
            conveyed_printed = True
        if i - win_i >= 60:
            if op < win_op + 0.002:
                stalled += 1
                floor_f = min(floor_f + 0.4, 2.8)
                print(f"[solve] draw stalled at op={op:+.4f} "
                      f"bx={float(butter_rel()[0]):+.3f}; "
                      f"floor -> {floor_f:.1f} N ({stalled})", flush=True)
                if stalled >= 6:
                    clear_forces()
                    fail("draw wedged — shuttle made no progress")
            else:
                stalled = 0
            win_i, win_op = i, op
    clear_forces()
    if not done:
        fail("draw timed out")
    step(150)  # hands-off settle: the butter comes to rest on the cavity floor
    report("dropped")
    if not bool(scene.in_vault(scene.butter)[0]):
        fail(f"butter not in the cavity after the draw "
             f"(b=({float(butter_rel()[0]):+.3f},{float(butter_rel()[1]):+.3f},"
             f"{float(butter_rel()[2]):.3f}), op={float(scene.opening()[0]):+.4f})")
    assert bool(scene._conveyed[0]), "conveyed latch never set"
    assert bool(scene._vaulted[0]), "vaulted latch not set"
    s2 = print_score("P2 butter dropped through the hatch into the vault (floor withdrawal)")
    assert s2 >= s1 - 1e-6, "score decreased across the deposit"

    # ---------------- phase 3: SEAL — push the shuttle back to its closed stop --------------
    done = False
    floor_f = 0.8
    for i in range(2400):
        op = float(scene.opening()[0])
        if op <= 0.004:
            done = True
            break
        servo(-0.08, gain=8.0, floor=floor_f, cap=3.0)
        if i and i % 300 == 0:
            floor_f = min(floor_f + 0.3, 2.0)
    clear_forces()
    if not done:
        fail(f"shuttle never reached its closed stop (op={float(scene.opening()[0]):+.4f})")
    step(120)  # hands-off settle at the stop
    report("sealed")
    assert bool(scene.closed()[0]), "shuttle not closed after the return stroke"
    assert bool(scene.in_vault(scene.butter)[0]), "butter left the cavity while sealing"
    if not bool(scene.success()[0]):
        fail("no success after the seal")
    s3 = print_score("P3 shuttle pushed to its closed stop (hatch fully covered)")
    assert s3 >= s2 - 1e-6, "score decreased across the seal"
    assert s3 >= 1.0 - 1e-6, f"success must score 1.0, got {s3}"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
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
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)
