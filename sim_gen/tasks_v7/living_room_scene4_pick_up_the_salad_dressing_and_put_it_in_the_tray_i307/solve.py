"""Teleport solution for CrateDockScene (sim_gen task
`living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i307`) — the
task's legitimacy certificate.

This solution uses NO teleports at all: every state change is produced by contact
dynamics. Both crates are ungraspable by design (110 mm footprint > any parallel
jaw), so the certificate is exactly what the arm would do — push on faces:

1. CLEAR THE BLOCKER (applied CoM force, contact dynamics): a speed-capped,
   stall-escalated horizontal force along the rig's +y axis — the palm push a Franka
   would deliver against the blocker's exposed south face — slides the tall gray
   blocker out of the junction into the siding branch, past the corridor line. The
   force is cleared between bouts and dropped the moment the blocker is clear.
2. DOCK THE PAYLOAD (applied CoM force, contact dynamics): the same style of push
   along the rig's +x axis drives the amber payload down the entry lane, across the
   junction band, through the mouth window, under the roof, until it rests against
   the dock's back wall — fully inside the containment window. Jam recovery: if the
   crate wedges against a wall, a short reverse bout un-wedges it and the push
   resumes at the base force.

Both pushes are contact-consistent external forces at the crate CoM, capped well
below the tipping bound (F * h_com < m g w/2), speed-capped so nothing is thrown.
The rig, the roof, the walls are never touched; nothing is ever written to a pose.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i307.solve --headless [--seed N]
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
import sys
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # forge fallback: run as a plain script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.crate_dock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def rig_xyz(body) -> tuple[float, float, float]:
        p = scene._rig_local(body.data.root_pos_w)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def report(tag: str) -> None:
        px, py, pz = rig_xyz(scene.payload)
        bx, by, bz = rig_xyz(scene.blocker)
        print(f"[solve] {tag:14s} | payload=({px:+.3f},{py:+.3f},{pz:+.3f}) "
              f"blocker=({bx:+.3f},{by:+.3f},{bz:+.3f}) "
              f"in_dock={bool(scene.payload_in_dock()[0])} "
              f"cleared={bool(scene.blocker_cleared()[0])} "
              f"latch=[c {int(scene._cleared[0])} j {int(scene._junction[0])} "
              f"m {int(scene._mouth[0])} d {int(scene._dock_ever[0])}] "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def rig_axis(ax: float, ay: float) -> torch.Tensor:
        v = torch.tensor([ax, ay, 0.0], device=device).expand(n, 3)
        return quat_apply(scene.rig.data.root_quat_w, v)

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def speed(body) -> float:
        return float(body.data.root_lin_vel_w.norm(dim=-1)[0])

    def push_until(body, ax: float, ay: float, done, name: str,
                   f0: float, f_max: float, budget: int,
                   coord: int, v_cap: float = 0.08) -> bool:
        """Speed-capped CoM push along the rig-frame (ax, ay) axis, monitored every
        step: force cleared whenever |v| > v_cap (no momentum ram), escalated x1.4
        only on a genuine stall (no advance of the pushed coordinate in 45 steps),
        with one reverse un-wedge bout per hard stall at the cap. Stops (force
        cleared) the moment done() is True."""
        newtons = f0
        pushing = False
        stall_ref = rig_xyz(body)[coord]
        for i in range(budget):
            if done():
                clear_force(body)
                print(f"[solve] {name}: target reached at step {i}", flush=True)
                return True
            if speed(body) > v_cap:
                if pushing:
                    clear_force(body)
                    pushing = False
            else:
                f = rig_axis(ax, ay).reshape(n, 1, 3) * newtons
                body.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                   is_global=True)
                pushing = True
            env.step(no_action)
            if i % 45 == 44:
                cur = rig_xyz(body)[coord]
                if cur - stall_ref < 0.0015 and speed(body) < 0.02:
                    if newtons >= f_max - 1e-6:
                        # hard stall at the cap: reverse un-wedge bout, then resume
                        print(f"[solve] {name}: hard stall at {cur:+.3f}; "
                              f"reverse un-wedge bout", flush=True)
                        rf = rig_axis(-ax, -ay).reshape(n, 1, 3) * f0
                        body.set_external_force_and_torque(
                            rf, zero_wrench, env_ids=all_ids, is_global=True)
                        step(30)
                        clear_force(body)
                        step(20)
                        newtons = f0
                    else:
                        newtons = min(newtons * 1.4, f_max)
                stall_ref = cur
            if i % 90 == 0:
                px, py, pz = rig_xyz(body)
                print(f"[solve] {name} step {i:4d}: F={newtons:5.2f} N "
                      f"pos=({px:+.3f},{py:+.3f},{pz:+.3f}) v={speed(body):.3f}",
                      flush=True)
        clear_force(body)
        print(f"[solve] {name}: budget exhausted", flush=True)
        return False

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)
    rp = (scene.rig.data.root_pos_w - scene.env_origins)[0]
    rq = scene.rig.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    px, py, pz = rig_xyz(scene.payload)
    bx, by, bz = rig_xyz(scene.blocker)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={ryaw:+.1f}deg "
          f"payload_rig=({px:+.3f},{py:+.3f},{pz:+.3f}) "
          f"blocker_rig=({bx:+.3f},{by:+.3f},{bz:+.3f})", flush=True)
    report("reset")
    for body in (scene.payload, scene.blocker):
        assert torch.isfinite(body.data.root_pos_w).all(), "NaN/inf after settle"
    assert px < -0.15, "payload must start in the entry lane"
    assert 0.02 < bx < 0.16 and abs(by) < 0.08, "blocker must start in the junction"
    s0 = print_score("P0 reset+settle (blocker plugs the junction)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: CLEAR THE BLOCKER into the siding ----------------------------
    # Push along the rig +y axis until the blocker center is comfortably past the
    # cleared line (0.15) — target 0.20, well inside the siding.
    ok = push_until(
        scene.blocker, 0.0, 1.0,
        done=lambda: rig_xyz(scene.blocker)[1] >= 0.20,
        name="clear-blocker", f0=1.5, f_max=5.0, budget=1500, coord=1)
    step(60)  # hands-off settle
    report("cleared")
    if not (ok and bool(scene.blocker_cleared()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (blocker never cleared the junction)", flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "clearing the blocker cannot be success"
    s1 = print_score("P1 blocker pushed into the siding (corridor open)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.14, f"P1 score {s1} (expect 0.15)"

    # ---------------- phase 2: DOCK THE PAYLOAD through the mouth ---------------------------
    # Push along the rig +x axis until the payload is fully inside the containment
    # window; the dock back wall is the natural stop.
    ok = push_until(
        scene.payload, 1.0, 0.0,
        done=lambda: rig_xyz(scene.payload)[0] >= 0.255,
        name="dock-payload", f0=1.2, f_max=6.0, budget=2400, coord=0)
    step(90)  # hands-off settle
    report("docked")
    if not ok or not bool(scene.payload_in_dock()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (payload never reached the dock window)", flush=True)
        os._exit(1)
    assert bool(scene._junction[0]), "junction band must have been crossed"
    assert bool(scene._mouth[0]), "mouth window must have been crossed"
    s2 = print_score("P2 payload pushed through the mouth, fully inside the dock")
    assert s2 >= s1 - 1e-6, "score decreased during docking"

    # ---------------- phase 3: judged state, live ------------------------------------------
    step(60)
    report("judge")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success at the judged state)", flush=True)
        os._exit(1)
    s3 = print_score("P3 payload at rest fully inside the roofed dock")
    assert s3 >= s2 - 1e-6, "score decreased at the judged state"

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
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        print(f"[solve] EXCEPTION: {exc!r}", flush=True)
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
