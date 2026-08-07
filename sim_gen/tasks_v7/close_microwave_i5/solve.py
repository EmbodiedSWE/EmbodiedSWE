"""Teleport solution for BayonetCanisterScene (sim_gen task `close_microwave_i5`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport): a single root-state write carries the BLUE lid from its
   ground spawn slot to the free-space point 30 mm above its seat, centred on the
   canister axis, wings rotated to the entry alignment (psi ~ +2 deg, read back from
   the canister's randomized yaw). At the hover height the lug feet are ABOVE the
   flange top, so the write bypasses no contact interaction and satisfies no rubric
   clause: the lid is airborne.
2. ENTRY + SEATING (gravity + contact): the lid FALLS; the three lug feet pass down
   through the three entry notches and the plate lands on the mouth rim. A short 3 N
   downward press (a fingertip press stand-in) confirms full seating. If the feet
   catch a flange edge, the lid perches high — detected by height readback and
   retried with a fresh hover (transport only). `lid_seated` (axis, seat height,
   upright) is produced by ballistics and contact, never written.
3. TWIST TO LOCK (applied torque + contact): with a 1 N hold-down force, a modest
   +z torque (0.06 N*m, escalating on stall, velocity-capped at 2 rad/s so the stop
   impact stays gentle) rotates the seated lid counterclockwise. The lug feet slide
   UNDER the flange through friction contact until they hit the under-flange lock
   stops at psi ~ +50 deg. The twist — the task's core interaction — happens entirely
   through contact dynamics; the lid is never teleported once it has entered.
4. The RED decoy lid is never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.close_microwave_i5.solve --headless [--seed N]
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qz = scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bayonet_canister")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def lid_loc():
        return scene._can_local(scene.blue.data.root_pos_w)[0]

    def report(tag: str) -> None:
        loc = lid_loc()
        print(f"[solve] {tag:14s} | lid_loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
              f"{float(loc[2]):+.3f}) psi={float(scene.psi_deg()[0]):+6.1f}deg "
              f"over={bool(scene.lid_over()[0])} seated={bool(scene.lid_seated()[0])} "
              f"locked={bool(scene.lid_locked()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # lids settle onto their feet
    cp = (scene.canister.data.root_pos_w - scene.env_origins)[0]
    cyaw = float(scene_mod._yaw_deg(scene.canister.data.root_quat_w)[0])
    bl = (scene.blue.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"canister=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) yaw={cyaw:+.1f}deg "
          f"blue_lid=({float(bl[0]):+.3f},{float(bl[1]):+.3f}) "
          f"blue_slot={int(scene.blue_slot[0])}", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene.lid_over()[0]), "blue lid must start away from the canister"
    s0 = print_score("P0 reset+settle (lids on the ground, canister open)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: transport to hover; gravity + press seats the lid -----------
    # Hover: on the canister axis, 30 mm above the seat, wings at entry alignment
    # (psi ~ +2 deg). At this height the feet are above the flange: free space.
    seated = False
    for attempt, psi0 in enumerate((2.0, 5.0, -1.0, 8.0)):
        can_yaw = scene_mod._yaw_deg(scene.canister.data.root_quat_w)  # (n,) deg
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.canister.data.root_pos_w
        st[:, 2] += c.z_seat + c.hover
        st[:, 3:7] = _qz(torch.deg2rad(can_yaw + psi0))
        scene.blue.write_root_state_to_sim(st, all_ids)
        step(60)  # free fall: feet drop through the notches, plate lands on the rim
        # fingertip-press stand-in: 3 N straight down to confirm full seating
        f = zero.clone()
        f[:, 0, 2] = -3.0
        for _ in range(60):
            scene.blue.set_external_force_and_torque(f, zero, env_ids=all_ids,
                                                     is_global=True)
            env.step(no_action)
        scene.blue.set_external_force_and_torque(zero, zero, env_ids=all_ids)
        step(30)
        report(f"drop-try{attempt}")
        if bool(scene.lid_seated()[0]):
            seated = True
            break
        loc = lid_loc()
        print(f"[solve] lid not seated (z_loc={float(loc[2]):.3f}, seat {c.z_seat:.3f})"
              f" — re-hovering (transport only)", flush=True)
    if not seated:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (lid never seated through the notches)", flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "seated-but-untwisted must not be success"
    s1 = print_score("P1 lid seated on the mouth (feet through the notches)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.44, f"P1 score {s1} (expect over+seated=0.45)"

    # ---------------- phase 2: twist to the lock stop (torque under contact) ---------------
    tau = 0.06
    last_psi = float(scene.psi_deg()[0])
    reached = False
    for i in range(2400):
        psi = float(scene.psi_deg()[0])
        if psi >= 46.0:
            reached = True
            break
        w = float(scene.blue.data.root_ang_vel_w[0, 2])
        f = zero.clone()
        f[:, 0, 2] = -1.0  # hold-down so the lid stays pressed while twisting
        t = zero.clone()
        if w < 2.0:  # velocity cap: keep the stop impact gentle
            t[:, 0, 2] = tau
        scene.blue.set_external_force_and_torque(f, t, env_ids=all_ids, is_global=True)
        env.step(no_action)
        if i % 120 == 119:
            if psi - last_psi < 2.0:
                tau = min(tau + 0.06, 0.50)
                print(f"[solve] twist stalled at psi={psi:+.1f}deg; torque -> "
                      f"{tau:.2f} N*m", flush=True)
            last_psi = psi
    scene.blue.set_external_force_and_torque(zero, zero, env_ids=all_ids)
    step(120)  # full settle, hands-off
    report("twist")
    psi = float(scene.psi_deg()[0])
    if not reached and psi < c.psi_lock_min:
        report("FAIL-state")
        print(f"SIM_GEN_SOLVE: FAIL (twist stalled at psi={psi:+.1f}deg)", flush=True)
        os._exit(1)
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the twist)", flush=True)
        os._exit(1)
    s2 = print_score("P2 lid twisted to the lock stop")
    assert s2 >= s1 - 1e-6, "score decreased across the twist"

    # ---------------- phase 3: persistence (>= 3 simulated seconds, hands-off) -------------
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
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)
