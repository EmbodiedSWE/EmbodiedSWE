"""Teleport solution for RammerGalleryScene (sim_gen task
`libero_kitchen_scene10_..._i62`) — the task's legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): one pose write carries the YELLOW butter from its floor spawn
   slot across free space to a hover 20 mm above the apron loading zone, long axis
   down the channel, between the parked rammer blade and the open gate plane. It then
   FALLS under gravity and settles on the apron through real contact. The write
   satisfies no rubric clause: the endpoint is outside the gallery bore (asserted),
   `entered`/`welled` demand in-bore poses that only the ram can produce.
2. RAM (applied force + contact — the core interaction): the trolley is dragged
   rearward by a horizontal velocity-servo force at its CoM — the exact wrench of a
   fingertip drag on its green knob (the prismatic slide resists the small moment
   either way). Every millimetre of the butter's journey — across the apron, through
   the open gate plane, through the mouth, down the covered channel, over the lip,
   the 10 mm drop into the sunken well — is blade<->butter contact plus friction and
   gravity. The butter itself is NEVER forced or pose-written after the load drop.
3. SEAL (applied force + contact): the gate is slid along its transverse rail to its
   y=0 closed stop by the same kind of velocity-servo force on its own body — the
   wrench of a finger on its dark-blue knob. The stop is the joint limit; `sealed`
   latches only with the butter already settled in the well (geometry forces this
   order anyway: a closed gate stands in the ram's path).
   Note on the pod force-frame quirk (wrenches rotated by the body's rotation since
   reset): both driven bodies ride prismatic joints and never rotate, so R_now =
   R_ref = identity and every mode coincides; raw world vectors are correct here.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

The intended single-Franka-arm strategy for the same plan (pinch the butter across
its 45 mm faces, lay it on the open apron shelf; hook-drag the green knob 0.39 m
along the channel axis; side-push the blue knob 0.11 m along the rail) lives in
TASK.md as the embodiment argument.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_i62.solve --headless [--seed N]
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
    env = ENVS.get("simgen.rammer_gallery")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    gx, gy = c.gallery_pos

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.trolley.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                    env_ids=all_ids)
        scene.gate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    def butter_rel() -> torch.Tensor:
        return scene._rel(scene.butter)[0]

    def report(tag: str) -> None:
        b = butter_rel()
        print(f"[solve] {tag:12s} | butter=({float(b[0]):+.3f},{float(b[1]):+.3f},"
              f"{float(b[2]):.3f}) trolley_x={float(scene.trolley_x()[0]):+.3f} "
              f"gate_y={float(scene.gate_y()[0]):+.3f} "
              f"|vb|={float(scene.butter.data.root_lin_vel_w[0].norm()):.4f} "
              f"|vt|={float(scene.trolley.data.root_lin_vel_w[0].norm()):.4f} "
              f"|vg|={float(scene.gate.data.root_lin_vel_w[0].norm()):.4f} "
              f"inside={bool(scene.inside_gallery(scene.butter)[0])} "
              f"entered={bool(scene.entered()[0])} in_well={bool(scene.in_well()[0])} "
              f"closed={bool(scene.gate_closed()[0])} "
              f"brick_out={bool(scene.brick_out()[0])} "
              f"app={float(scene._app_max[0]):.3f} "
              f"depth={float(scene._depth_max[0]):.3f} "
              f"welled={bool(scene._welled[0])} sealed={bool(scene._sealed[0])} "
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

    def servo(body, axis: int, v_des: float, *, gain: float, floor: float,
              cap: float) -> float:
        """One-step velocity-servo world force on `body` along `axis`; returns the
        commanded force. Raw world vector is correct: prismatic bodies never rotate
        (see module docstring)."""
        v = float(body.data.root_lin_vel_w[0, axis])
        f = gain * (v_des - v)
        sgn = 1.0 if v_des >= 0 else -1.0
        if abs(v) < 0.02 and sgn * f < floor:
            f = sgn * floor  # break static friction / damping hold
        f = max(-cap, min(cap, f))
        fw = torch.zeros(n, 1, 3, device=device)
        fw[0, 0, axis] = f
        body.set_external_force_and_torque(fw, zero_wrench, env_ids=all_ids,
                                           is_global=True)
        env.step(no_action)
        return f

    # ---------------- phase 0: reset, settle, layout readback ------------------------------
    step(60)
    b0 = butter_rel()
    k0 = scene._rel(scene.brick)[0]
    tx0 = float(scene.trolley_x()[0])
    gy0 = float(scene.gate_y()[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"butter=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"brick=({float(k0[0]):+.3f},{float(k0[1]):+.3f}) "
          f"trolley_x={tx0:+.3f} gate_y={gy0:+.3f}", flush=True)
    report("reset")
    assert c.trolley_x_range[0] - 0.006 <= tx0 <= c.trolley_x_range[1] + 0.006, \
        f"trolley not parked in its outboard zone ({tx0:+.3f})"
    assert c.gate_open_range[0] - 0.008 <= gy0 <= c.gate_open_range[1] + 0.008, \
        f"gate not parked open ({gy0:+.3f})"
    assert not bool(scene.inside_gallery(scene.butter)[0]), "butter spawned inside"
    assert bool(scene.brick_out()[0]), "brick spawned inside"
    s0 = print_score("P0 reset+settle")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.02, f"baseline score not ~0 ({s0:.3f})"

    # ---------------- phase 1: TRANSPORT — lay the butter in the loading zone --------------
    # One pose write to a hover 20 mm above the apron, long axis down the channel,
    # between the parked blade (front face <= -0.297) and the gate plane (-0.196);
    # gravity lands it through real contact. Asserted: endpoint outside the bore.
    lx, ly, lz = c.mouth_point
    st = torch.zeros(n, 13, device=device)
    st[:, 0], st[:, 1], st[:, 2] = gx + lx, gy + ly, lz + 0.020
    st[:, 3] = 1.0
    st[:, 0:3] += scene.env_origins
    scene.butter.write_root_state_to_sim(st, all_ids)
    assert not bool(scene.inside_gallery(scene.butter)[0]), \
        "hover endpoint already inside the gallery bore"
    step(90)
    report("loaded")
    b = butter_rel()
    assert abs(float(b[0]) - lx) < 0.020 and abs(float(b[1])) < 0.020, \
        "butter did not settle in the loading zone"
    assert 0.115 < float(b[2]) < 0.130, f"butter not flat on the apron (z={float(b[2]):.3f})"
    assert not bool(scene.inside_gallery(scene.butter)[0]), "butter already inside"
    s1 = print_score("P1 butter laid on the apron (contact landing)")
    assert s1 >= s0 - 1e-6, "score decreased across loading"
    assert float(scene._app_max[0]) > 0.90, "approach credit not earned at the loading zone"

    # ---------------- phase 2: RAM — drag the trolley; the blade does the transport --------
    # Velocity-servo force on the trolley CoM only. The butter is untouched: it is
    # swept by blade contact through the gate plane, the mouth, the channel, and over
    # the lip into the well. Stop when the butter is deep in the well band or the
    # carriage reaches its limit.
    entered_printed = False
    floor_f = 0.8
    win_i, win_tx = 0, float(scene.trolley_x()[0])
    stalled = 0
    done = False
    for i in range(4200):
        bx = float(butter_rel()[0])
        tx = float(scene.trolley_x()[0])
        if bx >= 0.124 or tx >= c.trolley_hi - 0.002:
            done = True
            break
        v_des = 0.10 if bx < 0.06 else 0.05  # slow down for the lip drop
        servo(scene.trolley, 0, v_des, gain=6.0, floor=floor_f, cap=3.0)
        if not entered_printed and bool(scene._entered[0]):
            clear_forces()
            report("entered")
            s_mid = print_score("P2a butter swept fully into the bore (blade contact)")
            assert s_mid >= s1 - 1e-6, "score decreased across entry"
            entered_printed = True
        if i - win_i >= 60:
            if tx < win_tx + 0.002:
                stalled += 1
                floor_f = min(floor_f + 0.4, 2.6)
                print(f"[solve] ram stalled at tx={tx:+.3f} bx={bx:+.3f}; "
                      f"floor -> {floor_f:.1f} N ({stalled})", flush=True)
                if stalled >= 6:
                    clear_forces()
                    fail("ram wedged — trolley made no progress")
            else:
                stalled = 0
            win_i, win_tx = i, tx
    clear_forces()
    if not done:
        fail("ram timed out")
    step(150)  # hands-off settle: the butter flattens into the well
    report("rammed")
    assert bool(scene.entered()[0] if not entered_printed else True), "never entered"
    if not bool(scene.in_well()[0]):
        fail(f"butter not in the well after the ram (bx={float(butter_rel()[0]):+.3f}, "
             f"bz={float(butter_rel()[2]):.3f})")
    assert bool(scene._welled[0]), "welled latch not set"
    s2 = print_score("P2 butter rammed into the sunken well (blade contact + lip drop)")
    assert s2 >= s1 - 1e-6, "score decreased across the ram"

    # ---------------- phase 3: SEAL — slide the gate to its closed stop ---------------------
    done = False
    floor_f = 0.6
    for i in range(1800):
        gyv = float(scene.gate_y()[0])
        if gyv <= 0.005:
            done = True
            break
        servo(scene.gate, 1, -0.08, gain=6.0, floor=floor_f, cap=2.0)
        if i and i % 300 == 0:
            floor_f = min(floor_f + 0.3, 1.8)
    clear_forces()
    if not done:
        fail(f"gate never reached its closed stop (gate_y={float(scene.gate_y()[0]):+.3f})")
    step(120)  # hands-off settle at the stop
    report("sealed")
    assert bool(scene.gate_closed()[0]), "gate not closed after the slide"
    assert bool(scene.in_well()[0]), "butter left the well while sealing"
    if not bool(scene.success()[0]):
        fail("no success after the seal")
    s3 = print_score("P3 gate slid to its closed stop (mouth fully covered)")
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
