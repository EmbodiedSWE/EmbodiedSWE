"""Teleport solution for WeighPressCabinetScene (sim_gen task
`libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_i323`)
— the task's legitimacy certificate.

This solve applies ZERO forces and ZERO wrenches — ever. Its ONE teleport is pure
TRANSPORT: the black bowl is lifted off its plinth slot and released (zero velocity)
just above the weigh tray on the cabinet top — exactly what a hand does when it picks
the bowl up, carries it over, and sets it down. Everything judged happens through
contact dynamics under gravity:

1. PERCEPTION: station pose/yaw, the drawer's open rest q, the plunger's flush rest
   height and the bowl's dealt slot are read back from the episode state — never
   hard-coded. The set-down point is the tray centre, mapped through the MEASURED
   plunger pose.
2. TRANSPORT (the single teleport): bowl moved from its slot to just above the tray
   floor, upright, velocity zero. The drawer has not moved — only load credit follows.
3. WEIGH PRESS (hands off, no actuation at all): the bowl's weight drives the plunger
   down its shaft, the 45 deg blade presses the 45 deg fin, and the drawer runs shut
   UP its inclined slideway to the rear stop — then the weight KEEPS holding it shut.
   The judged "drawer closes AND bowl on top" outcome is delivered ENTIRELY by the
   machine, as one causal consequence of the single placement.
4. The end state is judged LIVE: it is a standing force balance (remove the bowl and
   the drawer re-opens — smoke proves that), not a latched snapshot.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the rubric
latches), then holds HANDS-OFF >= 3 simulated seconds after success() first turns
True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_i323.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weigh_press_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

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

    def loc_of(body):
        return scene._station_local(body.data.root_pos_w)[0]

    def report(tag: str) -> None:
        bl = loc_of(scene.bowl)
        pl = loc_of(scene.plunger)
        print(f"[solve] {tag:12s} | q={float(scene.drawer_q()[0]):+.4f} "
              f"plunger_z={float(pl[2]):+.4f} "
              f"bowl_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
              f"|v_bowl|={float(scene.bowl.data.root_lin_vel_w[0].norm()):.3f} "
              f"on_tray={bool(scene.bowl_on_tray()[0])} "
              f"in_shaft={bool(scene.plunger_in_shaft()[0])} "
              f"closed={bool(scene.drawer_closed()[0])} "
              f"in_channel={bool(scene.drawer_in_channel()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: settle, layout readback, baseline ---------------------------------
    step(150)   # drawer settles OPEN on its front stops; blade settles flush on the fin
    sp0 = (scene.station.data.root_pos_w - scene.env_origins)[0]
    sq0 = scene.station.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(sq0[3]), float(sq0[0])))
    q_meas = float(scene.drawer_q()[0])
    pl = loc_of(scene.plunger)
    bl = loc_of(scene.bowl)
    slot = int(scene.bowl_slot[0])
    sx, sy = c.slots[slot]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"station=({float(sp0[0]):+.3f},{float(sp0[1]):+.3f}) yaw~{yaw:+.1f}deg "
          f"q_rest={q_meas:+.4f} (q_open={c.q_open:+.4f}) "
          f"plunger_z={float(pl[2]):+.4f} (pred {c.z_press(q_meas):+.4f}) "
          f"bowl_slot={slot}@({sx:+.2f},{sy:+.2f}) "
          f"bowl=({float(bl[0]):+.3f},{float(bl[1]):+.3f})", flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert bool(scene.drawer_in_channel()[0]), "drawer must ride its slideway"
    assert not bool(scene.drawer_closed()[0]), "drawer must rest OPEN (dead-man)"
    assert abs(q_meas - c.q_open) < 0.008, \
        f"gravity must park the drawer on its front stops (q={q_meas:+.4f})"
    assert bool(scene.plunger_in_shaft()[0]), "plunger must ride its shaft"
    assert abs(float(pl[2]) - c.z_press(q_meas)) < 0.008, \
        "plunger must rest with the blade flush on the fin"
    assert abs(float(bl[0]) - sx) < c.slot_jitter + 0.015 and \
        abs(float(bl[1]) - sy) < c.slot_jitter + 0.015, \
        "bowl must rest on its dealt plinth slot"
    assert not bool(scene.bowl_on_tray()[0]), "bowl must start off the tray"
    s_prev = print_score("P0 reset+settle (drawer parked open, tray empty)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s_prev <= 0.05, f"baseline score should be ~0, got {s_prev}"

    # ---------------- phase 1: TRANSPORT — the single teleport (pick, carry, set down) -----------
    # Bowl centre moved to just above the tray floor, upright, mapped through the
    # MEASURED plunger/station pose. Velocity zero: a hand letting go.
    drop = torch.zeros(n, 3, device=device)
    drop[:, 2] = c.tray_z1 + 0.008
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.plunger.data.root_pos_w \
        + quat_apply(scene.station.data.root_quat_w.expand(n, 4), drop)
    st[:, 3:7] = scene.station.data.root_quat_w
    scene.bowl.write_root_state_to_sim(st)
    step(12)    # ~0.1 s: the bowl has landed on the tray floor; the press just begins
    report("set-down")
    assert bool(scene.bowl_on_tray()[0]), "bowl must ride the weigh tray"
    q_now = float(scene.drawer_q()[0])
    assert q_now > c.q_closed_tol + 0.020, \
        f"drawer cannot already be closed at set-down (q={q_now:+.4f})"
    s = print_score("P1 bowl set down into the weigh tray (drawer still open)")
    assert s >= s_prev - 1e-6, "score decreased across P1"
    assert 0.28 <= s <= 0.60, f"P1 must earn load credit (+ early travel), got {s:.3f}"
    s_prev = s

    # ---------------- phase 2: WEIGH PRESS — hands off, weight closes the drawer ----------------
    closed_at = -1
    for i in range(720):    # up to 6 s — the press takes well under 2 s
        env.step(no_action)
        if closed_at < 0 and bool(scene.drawer_closed()[0]):
            closed_at = i
        if closed_at >= 0 and bool(scene.settled()[0]):
            break
    print(f"[solve] press: drawer first closed at step {closed_at}", flush=True)
    report("press")
    assert closed_at >= 0, \
        f"weigh press failed to close the drawer (q={float(scene.drawer_q()[0]):+.4f})"
    step(120)   # hands-off settle: the force balance stands
    report("settled")
    assert bool(scene.drawer_closed()[0]), \
        f"drawer did not stay seated: q={float(scene.drawer_q()[0]):+.4f}"
    assert bool(scene.bowl_on_tray()[0]), "bowl must still ride the tray"
    assert bool(scene.success()[0]), "success() must hold on the settled end state"
    s = print_score("P2 weigh press: drawer seated, bowl on top, all hands off")
    assert s >= s_prev - 1e-6, "score decreased across P2"
    assert s >= 1.0 - 1e-6, "success must score 1.0"
    s_prev = s

    # ---------------- persistence (>= 3 simulated seconds, hands-off) ----------------------------
    hold_ok = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold_ok = hold_ok and bool(scene.success()[0])
    report("persist")
    s_final = print_score("P-final persistence 3.3 s")
    ok = hold_ok and bool(scene.success()[0]) and s_final >= s_prev - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
