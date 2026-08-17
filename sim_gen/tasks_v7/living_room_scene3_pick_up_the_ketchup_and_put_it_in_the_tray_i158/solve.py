"""Teleport solution for ShuttleHatchScene (sim_gen task
`living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i158`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. PUSH THE SHUTTLE IN (applied force + contact): a horizontal bang-bang force at
   the tray's CoM, directed station-local -x (into the channel), slides the shuttle
   along the slick plate until its back wall meets the BACK STOP. This is exactly
   the push a fingertip on the grip tab would perform. The `loaded` credit (tray at
   rest at the back stop) is produced by contact against the stop, never written.
   FORCE-FRAME GUARD: some pods rotate an applied "global" wrench by the body's
   rotation since reset; the push PROBES the frame convention at runtime and toggles
   `encode_force` mode if the tray moves the wrong way.
2. TRANSPORT (teleport): a single root-state write carries the KETCHUP bottle from
   its ground slot to free air ABOVE THE CHIMNEY MOUTH (station-local
   (-0.112, 0, 0.359): 8 mm of clearance over the mouth at 0.276 + half the bottle),
   zero velocity, upright and square to the chimney. The release point is open air
   far outside the tray's interior volume (asserted at the write), so the transport
   satisfies no rubric clause by itself.
3. DROP THROUGH THE CHIMNEY (gravity + contact, hands-off): the bottle free-falls
   ~280 mm down the chimney, through the roof hole, between the tray's wall tops,
   and lands INSIDE the tray. The `in_tray` credit (bottle at rest inside the tray)
   is produced by ballistics and contact.
4. PULL THE SHUTTLE OUT (applied force + contact): the same bang-bang force,
   station-local +x, drags the loaded shuttle forward until its front wall seats
   against the STOP BAR — the serve stop. Every success clause (bottle in tray,
   tray at serve, settled) is then a live, settled contact outcome.
5. IDENTITY: only the KETCHUP bottle is ever touched. Mustard and mayo stay in
   their ground slots.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i158.solve --headless [--seed N]
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
    from .scene import encode_force
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shuttle_hatch")().build(num_envs=args.num_envs,
                                                  device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    from isaaclab.utils.math import quat_apply

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so
    # the task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def tray_x() -> float:
        return float(scene.tray_loc()[0, 0])

    def report(tag: str) -> None:
        tl = scene.tray_loc()[0]
        kl = scene._tray_local(scene.ketchup.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | tray_loc=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):+.3f}) ketchup_tray=({float(kl[0]):+.3f},"
              f"{float(kl[1]):+.3f},{float(kl[2]):+.3f}) "
              f"loaded={bool(scene._loaded[0])} in_latch={bool(scene._in_tray[0])} "
              f"in={bool(scene.ketchup_in_tray()[0])} "
              f"at_load={bool(scene.tray_at_load()[0])} "
              f"at_serve={bool(scene.tray_at_serve()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # tray seats on the plate, bottles settle on the ground
    # custom spawn funcs ignore cfg mass schemas — assert the authored masses took
    tm = float(scene.tray.root_physx_view.get_masses().sum())
    sm = float(scene.station.root_physx_view.get_masses().sum())
    print(f"[solve] mass readback: tray={tm:.3f} kg station={sm:.1f} kg", flush=True)
    assert 0.35 < tm < 0.55, f"tray mass wrong: {tm}"
    assert 20.0 < sm < 30.0, f"station mass wrong: {sm}"
    sp = (scene.station.data.root_pos_w - scene.env_origins)[0]
    sq = scene.station.data.root_quat_w[0]
    syaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"station=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg "
          f"tray_x={tray_x():+.3f} ketchup_slot={int(scene.ketchup_slot[0])}",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.tray_at_serve()[0]), \
        f"tray must start seated in the serve zone, got x={tray_x():+.3f}"
    assert not bool(scene.ketchup_in_tray()[0]), "ketchup must start outside the tray"
    s0 = print_score("P0 reset+settle (tray at serve, bottles on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- shared bang-bang tray drive ------------------------------------------
    st_q = scene.station.data.root_quat_w
    state = {"mode": 0, "q_ref": scene.tray.data.root_quat_w.clone()}

    def drive_tray(sign: float, x_target: float, fmag0: float, fmax: float,
                   max_steps: int) -> bool:
        """Bang-bang CoM force on the tray along station-local `sign`*x until the
        tray origin passes `x_target` (in the direction of travel). Probes the
        force-frame convention every 45 steps; escalates on stall. Returns True on
        release, False on step budget exhausted. Always clears the force."""
        fdir = quat_apply(st_q, torch.tensor([sign, 0.0, 0.0],
                                             device=device).expand(n, 3))
        fmag = fmag0
        probe_s = sign * tray_x()
        released = False
        for i in range(max_steps):
            s_now = sign * tray_x()
            if s_now > sign * x_target:
                released = True
                break
            v = float(scene.tray.data.root_lin_vel_w[0].norm())
            mag = fmag if v < 0.06 else (0.4 * fmag if v < 0.12 else 0.0)
            if mag > 0.0:
                f = encode_force(state["mode"], state["q_ref"],
                                 scene.tray.data.root_quat_w,
                                 mag * fdir).view(n, 1, 3)
                scene.tray.set_external_force_and_torque(f, zero_wrench,
                                                         env_ids=all_ids,
                                                         is_global=True)
            else:
                scene.tray.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                         env_ids=all_ids)
            env.step(no_action)
            if i % 45 == 44:
                s_new = sign * tray_x()
                if s_new < probe_s - 0.004:
                    state["mode"] ^= 1  # frame drag: the push moved it the wrong way
                    state["q_ref"] = scene.tray.data.root_quat_w.clone()
                    print(f"[solve] drive moved the tray the WRONG way "
                          f"(s {probe_s:+.3f} -> {s_new:+.3f}); "
                          f"force-frame mode -> {state['mode']}", flush=True)
                elif s_new < probe_s + 0.002:
                    fmag = min(fmag + 1.0, fmax)
                    print(f"[solve] drive stalled at x={tray_x():+.3f}; "
                          f"force -> {fmag:.1f} N", flush=True)
                probe_s = s_new
        scene.tray.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)
        return released

    # ---------------- phase 1: push the shuttle in to the back (load) stop ------------------
    ok = drive_tray(-1.0, -0.105, 2.5, 8.0, 1500)
    if not ok:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (tray never reached the load stop)", flush=True)
        os._exit(1)
    step(90)  # coast into the back stop and settle, hands-off
    report("tray->load")
    assert bool(scene.tray_at_load()[0]), \
        f"tray must seat at the load stop, got x={tray_x():+.3f}"
    assert bool(scene._loaded[0]), "loaded latch must be set"
    assert not bool(scene.success()[0]), "cannot be success at the load stop"
    s1 = print_score("P1 shuttle pushed in to the load stop")
    assert s1 >= s0 - 1e-6 and s1 >= c.w_loaded - 1e-6, \
        f"P1 score {s1} (expect loaded={c.w_loaded})"

    # ---------------- phase 2: drop the ketchup bottle down the chimney ---------------------
    # TRANSPORT: write the bottle to free air 8 mm above the chimney mouth, upright,
    # square to the chimney, zero velocity. The release point is far outside the
    # tray volume (asserted) — gravity does every load-bearing part.
    st_p = scene.station.data.root_pos_w
    st_qn = scene.station.data.root_quat_w
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0] = c.x_load
    loc[:, 2] = c.chimney_top + c.bottle_size[2] / 2 + 0.008
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = st_p + quat_apply(st_qn, loc)
    st[:, 3:7] = st_qn  # cross-section square to the chimney opening
    scene.ketchup.write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    assert not bool(scene.ketchup_in_tray()[0]), \
        "release point must be OUTSIDE the tray volume"
    assert not bool(scene._in_tray[0]), "in_tray latch must not fire at the write"
    kx = float(scene._station_local(scene.ketchup.data.root_pos_w)[0, 0])
    assert c.hole_x[0] < kx < c.hole_x[1], f"release not over the chimney: x={kx:+.3f}"
    # hands-off: free-fall down the chimney, land in the tray, settle
    settled_in = False
    for j in range(600):
        env.step(no_action)
        if bool(scene.ketchup_in_tray()[0]) and bool(scene.settled()[0]):
            settled_in = True
            break
        if j % 180 == 179:
            report(f"drop-{j + 1}")
    step(60)  # margin of settle, hands-off
    report("chimney-drop")
    if not (settled_in or bool(scene.ketchup_in_tray()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (bottle did not land in the tray)", flush=True)
        os._exit(1)
    assert bool(scene._in_tray[0]), "in_tray latch must be set after the drop"
    assert not bool(scene.success()[0]), \
        "cannot be success with the tray at the load stop"
    s2 = print_score("P2 ketchup dropped down the chimney into the tray")
    assert s2 >= s1 - 1e-6 and s2 >= 0.40 - 1e-5, f"P2 score {s2} (expect 0.40)"

    # ---------------- phase 3: pull the loaded shuttle out to the serve stop ----------------
    ok = drive_tray(+1.0, 0.038, 2.5, 8.0, 1500)
    if not ok:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (tray never returned to the serve stop)", flush=True)
        os._exit(1)
    # coast into the stop bar and settle, hands-off
    got = False
    for j in range(360):
        env.step(no_action)
        if bool(scene.success()[0]):
            got = True
            break
        if j % 120 == 119:
            report(f"seat-{j + 1}")
    report("tray->serve")
    if not (got or bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the pull-out)", flush=True)
        os._exit(1)
    s3 = print_score("P3 loaded shuttle pulled out to the serve stop")
    assert s3 >= s2 - 1e-6, "score decreased across the pull-out"

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
    except Exception:  # noqa: BLE001 - fail FAST; a hung Kit burns the forge slot
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
