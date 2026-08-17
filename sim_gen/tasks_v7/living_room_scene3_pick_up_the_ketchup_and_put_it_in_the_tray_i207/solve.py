"""Teleport solution for QueueDispenserScene (sim_gen task
`living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i207`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. READ THE QUEUE (state readback standing in for the sight slot): the per-episode
   queue permutation is read back (`scene.levels`) to decide how many cartons sit
   below the ketchup — the same information an embodied agent gets by looking at
   the color order through the 16 mm sight slot.
2. REJECT PUSHES (applied force + contact): each distractor queued below the
   ketchup is ejected bottom-first by a horizontal bang-bang force at the carton's
   CoM, directed station-local +x — exactly the fingertip push through the 44 mm
   rear slot. The push only vaults the carton over the retention lip; gravity then
   slides it down the slick tongue and drops it off the tip into the REJECT PIT.
   The `rejected` credit (distractor out of the column) is produced by the vault +
   ballistics, never written. FORCE-FRAME GUARD: some pods rotate an applied
   "global" wrench by the body's rotation since reset; every drive PROBES the
   frame convention at runtime and toggles `encode_force` mode if the carton moves
   the wrong way.
3. STAGE THE TRAY (teleport = transport only): a single root-state write carries
   the tray from its random ground pose to free air ABOVE the pit-rim seat
   (station-local (0.160, 0, 0.220) — above the seat-z window, so `tray_seated`
   is FALSE at the write, asserted), zero velocity, square to the station. It
   falls ~20 mm and SEATS ON THE RIM under gravity + contact — the `staged`
   credit is a contact outcome. This is the carry-and-set-down a gripper holding
   the tray's wall would perform (slid in level under the tongue: 32 mm clearance).
4. SERVE PUSH (applied force + contact, then hands-off): the same rear-slot push
   ejects the KETCHUP; it slides off the tongue and drops INTO THE SEATED TRAY.
   Every success clause (ketchup in tray, tray seated, no distractor in tray,
   settled) is then a live, settled contact outcome. No force is ever applied to
   the tray; the tray is never touched after it seats.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it persists.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i207.solve --headless [--seed N]
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
    env = ENVS.get("simgen.queue_dispenser")().build(num_envs=args.num_envs,
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

    def loc_of(body) -> torch.Tensor:
        return scene.station_local(body.data.root_pos_w)[0]

    def report(tag: str) -> None:
        tl = scene.station_local(scene.tray.data.root_pos_w)[0]
        parts = []
        for name, body in zip(scene.ITEMS, scene.items):
            bl = loc_of(body)
            parts.append(f"{name[:4]}=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
                         f"{float(bl[2]):+.3f})")
        print(f"[solve] {tag:14s} | tray=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
              f"{float(tl[2]):+.3f}) " + " ".join(parts) +
              f" seated={bool(scene.tray_seated()[0])} "
              f"staged={bool(scene._staged[0])} "
              f"in={bool(scene.in_tray(scene.ketchup)[0])} "
              f"served={bool(scene._served[0])} "
              f"rej={scene._rejected[0].tolist()} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # queue seats on the ramp behind the lip, tray settles on the ground
    # custom spawn funcs ignore cfg mass schemas — assert the authored masses took
    tm = float(scene.tray.root_physx_view.get_masses().sum())
    km = float(scene.ketchup.root_physx_view.get_masses().sum())
    print(f"[solve] mass readback: tray={tm:.3f} kg ketchup={km:.3f} kg", flush=True)
    assert 0.18 < tm < 0.32, f"tray mass wrong: {tm}"
    assert 0.10 < km < 0.20, f"carton mass wrong: {km}"
    sp = scene._st_pos[0]
    sq = scene._st_quat[0]
    syaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    levels = scene.levels[0].tolist()  # levels[i] = queue level of item i (0 = bottom)
    order = [scene.ITEMS[levels.index(k)] for k in range(3)]  # bottom -> top
    n_need = int(scene.needed_ahead()[0].sum())
    tl0 = scene.station_local(scene.tray.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"station=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg "
          f"queue(bottom->top)={order} ketchup_level={levels[0]} "
          f"n_rejects_needed={n_need} "
          f"tray_loc=({float(tl0[0]):+.3f},{float(tl0[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    for name, body in zip(scene.ITEMS, scene.items):
        assert bool(scene.in_magazine(body)[0]), f"{name} must start in the magazine"
    assert not bool(scene.tray_seated()[0]), "tray must start off the seat"
    assert not bool(scene.in_tray(scene.ketchup)[0]), "ketchup must start outside the tray"
    s0 = print_score("P0 reset+settle (queue seated, tray loose on the ground)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- shared rear-slot dispense push ----------------------------------------
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
    fdir = quat_apply(scene._st_quat, ex)  # station-local +x, world frame
    state = {"mode": 0}

    def clear(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def push_out(body, name: str) -> bool:
        """Bang-bang CoM force, station-local +x (the rear-slot fingertip push),
        until the carton is clearly PAST the lip and onto the tongue (x > 0.095 —
        releasing at the lip itself leaves it see-sawed on the crest with the
        follower stack pinning its tail). Probes the force-frame convention every
        45 steps; escalates on stall (the caps are generous: a tight cap makes a
        limit cycle right at the tip-over torque). Always clears the force."""
        q_ref = body.data.root_quat_w.clone()
        fmag = 4.0  # ~ the tip-over threshold with the full stack; escalates on stall
        probe_x = float(loc_of(body)[0])
        out = False
        for i in range(900):
            x = float(loc_of(body)[0])
            if x > 0.095:
                out = True
                break
            v = float(body.data.root_lin_vel_w[0].norm())
            mag = fmag if v < 0.15 else (0.4 * fmag if v < 0.30 else 0.0)
            if mag > 0.0:
                f = encode_force(state["mode"], q_ref, body.data.root_quat_w,
                                 mag * fdir).view(n, 1, 3)
                body.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids,
                                                   is_global=True)
            else:
                clear(body)
            env.step(no_action)
            if i % 45 == 44:
                x_new = float(loc_of(body)[0])
                if x_new < probe_x - 0.004:
                    state["mode"] ^= 1
                    q_ref = body.data.root_quat_w.clone()
                    print(f"[solve] push moved {name} the WRONG way "
                          f"(x {probe_x:+.3f} -> {x_new:+.3f}); "
                          f"force-frame mode -> {state['mode']}", flush=True)
                elif x_new < probe_x + 0.002:
                    fmag = min(fmag + 1.5, 12.0)
                    print(f"[solve] push stalled at x={x_new:+.3f} ({name}); "
                          f"force -> {fmag:.1f} N", flush=True)
                probe_x = x_new
        clear(body)
        return out

    def dispense(body, name: str, into_tray: bool) -> bool:
        """Push the carton over the lip, then HANDS-OFF until it comes to rest in
        the pit (into_tray=False) or in the tray (into_tray=True). Retries the
        push if the carton hangs up on the tongue."""
        for attempt in range(3):
            if not push_out(body, name):
                print(f"[solve] {name}: push never crossed the lip "
                      f"(attempt {attempt + 1})", flush=True)
                continue
            for _ in range(600):
                env.step(no_action)
                slow = float(body.data.root_lin_vel_w[0].norm()) < 0.08
                if into_tray and bool(scene.in_tray(body)[0]) and slow:
                    return True
                if not into_tray and bool(scene.in_pit(body)[0]) and slow:
                    return True
            bl = loc_of(body)
            print(f"[solve] {name}: not delivered after push "
                  f"(attempt {attempt + 1}), at ({float(bl[0]):+.3f},"
                  f"{float(bl[1]):+.3f},{float(bl[2]):+.3f}) — retrying", flush=True)
        return False

    def wait_queue_reseat() -> None:
        """Hands-off until every carton still in the magazine is slow again."""
        for _ in range(360):
            env.step(no_action)
            calm = True
            for body in scene.items:
                if bool(scene.in_magazine(body)[0]) \
                        and float(body.data.root_lin_vel_w[0].norm()) > 0.08:
                    calm = False
            if calm:
                break
        step(60)

    # ---------------- phase 1: reject every distractor queued below the ketchup -------------
    for k in range(3):
        if k >= levels[0]:
            break  # ketchup is now the bottom carton
        idx = levels.index(k)
        body, name = scene.items[idx], scene.ITEMS[idx]
        print(f"[solve] rejecting {name} (queue level {k})", flush=True)
        if not dispense(body, name, into_tray=False):
            report("FAIL-state")
            print(f"SIM_GEN_SOLVE: FAIL ({name} never reached the reject pit)",
                  flush=True)
            os._exit(1)
        wait_queue_reseat()
        report(f"rejected-{name}")
        assert bool(scene._rejected[0, idx]), f"rejected latch must be set for {name}"
        assert not bool(scene.in_magazine(body)[0]), f"{name} must be out of the column"
    s1 = print_score(f"P1 rejects done ({n_need} distractor(s) into the pit)")
    assert s1 >= s0 - 1e-6, "score decreased across P1"
    if n_need > 0:
        assert s1 >= c.w_reject - 1e-4, f"P1 score {s1} (expect >= {c.w_reject})"
    assert not bool(scene.success()[0]), "cannot be success before the serve"

    # ---------------- phase 2: stage the tray on the pit rim (transport only) ---------------
    # TRANSPORT: write the tray to free air ABOVE the seat-z window, square to the
    # station, zero velocity. It falls ~20 mm and seats on the rim by contact.
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0] = c.seat_x
    loc[:, 2] = c.seat_z[1] + 0.005  # above the seat window -> not seated at the write
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.station_world(loc)
    st[:, 3:7] = scene._st_quat
    scene.tray.write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    assert not bool(scene.tray_seated()[0]), \
        "the tray write must land ABOVE the seat window (credit comes from contact)"
    assert not bool(scene._staged[0]), "staged latch must not fire at the write"
    step(180)  # fall + seat + settle, hands-off
    report("tray-staged")
    tl = scene.station_local(scene.tray.data.root_pos_w)[0]
    assert bool(scene.tray_seated()[0]), \
        f"tray must seat on the rim, got loc=({float(tl[0]):+.3f},{float(tl[1]):+.3f}," \
        f"{float(tl[2]):+.3f})"
    assert bool(scene._staged[0]), "staged latch must be set"
    assert not bool(scene.success()[0]), "cannot be success with the ketchup queued"
    s2 = print_score("P2 tray staged on the pit rim")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_staged - 1e-4, f"P2 score {s2}"

    # ---------------- phase 3: serve the ketchup into the seated tray -----------------------
    assert bool(scene.in_magazine(scene.ketchup)[0]), "ketchup must still be queued"
    print("[solve] serving the ketchup", flush=True)
    if not dispense(scene.ketchup, "ketchup", into_tray=True):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (ketchup did not land in the tray)", flush=True)
        os._exit(1)
    # hands-off: let everything settle to a live success
    got = False
    for j in range(600):
        env.step(no_action)
        if bool(scene.success()[0]):
            got = True
            break
        if j % 180 == 179:
            report(f"settle-{j + 1}")
    report("served")
    if not (got or bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no live success after the serve)", flush=True)
        os._exit(1)
    ktl = scene._tray_local(scene.ketchup.data.root_pos_w)[0]
    print(f"[solve] ketchup at rest in the tray (tray-local "
          f"({float(ktl[0]):+.3f},{float(ktl[1]):+.3f},{float(ktl[2]):+.3f}))",
          flush=True)
    s3 = print_score("P3 ketchup served into the seated tray (live success)")
    assert s3 >= s2 - 1e-6 and s3 >= 0.99, f"P3 score {s3} (expect 1.0)"

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
