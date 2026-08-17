"""Teleport solution for AirlockTransferScene (sim_gen task
`living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_i236`) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. OPEN THE HATCH (applied force + contact): a horizontal overdamped-servo force
   at the gate's CoM, directed station-local +x, slides the gate along the roof
   until the roof blade fully clears hatch A (c ~ +0.065; pass threshold +0.027).
   This is exactly the push a hand on the handle post would perform. Window B is
   then sealed by blade B — the airlock invariant. FORCE-FRAME GUARD: some pods
   rotate an applied "global" wrench by the body's rotation since reset; the drive
   PROBES the frame convention at runtime and toggles `encode_force` mode if the
   gate moves the wrong way.
2. TRANSPORT (teleport): a single root-state write carries the ALPHABET-SOUP can
   from its ground slot to free air ABOVE THE OPEN HATCH (station-local
   (0, 0.140, 0.310): 8 mm of clearance over the roof top + half the can), zero
   velocity, upright. The release point is open air far outside the vault, the
   tray and the antechamber (asserted at the write), so the transport satisfies no
   rubric clause by itself.
3. DROP + RAMP SLIDE (gravity + contact, hands-off): the can free-falls ~136 mm
   through the hatch onto the slick 10-deg ramp, slides down and comes to rest
   against the CLOSED window blade — the `in_chamber` and `staged` credits are
   produced by ballistics and contact, never written.
4. CYCLE THE AIRLOCK (applied force + contact + gravity): the same bang-bang
   drive, station-local -x, slides the gate to c <= -0.055 (pass threshold
   -0.002). Blade B clears window B, hatch A re-seals, the staged can slides
   through the window, falls off the sill and lands INSIDE the tray. Every
   success clause (can in tray, in the vault, settled) is then a live, settled
   contact outcome.
5. IDENTITY: only the ALPHABET-SOUP can is ever touched. Corn and cream stay in
   their ground slots.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_i236.solve --headless [--seed N]
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
    env = ENVS.get("simgen.airlock_transfer")().build(num_envs=args.num_envs,
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

    def gate_x() -> float:
        return float(scene.gate_c()[0])

    def can_loc() -> torch.Tensor:
        return scene._station_local(scene.alphabet.data.root_pos_w)[0]

    def report(tag: str) -> None:
        kl = can_loc()
        tl = scene._tray_local(scene.alphabet.data.root_pos_w)[0]
        print(f"[solve] {tag:14s} | c={gate_x():+.3f} "
              f"can_st=({float(kl[0]):+.3f},{float(kl[1]):+.3f},{float(kl[2]):+.3f}) "
              f"can_tray=({float(tl[0]):+.3f},{float(tl[1]):+.3f},{float(tl[2]):+.3f}) "
              f"cham={bool(scene._in_chamber[0])} staged={bool(scene._staged[0])} "
              f"vault={bool(scene._in_vault[0])} in={bool(scene.alphabet_in_tray()[0])} "
              f"A={bool(scene.hatch_passes()[0])} B={bool(scene.window_passes()[0])} "
              f"settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # gate seats on the roof, tray on its seat, cans on the ground
    # custom spawn funcs ignore cfg mass schemas — assert the authored masses took
    gm = float(scene.gate.root_physx_view.get_masses().sum())
    tm = float(scene.tray.root_physx_view.get_masses().sum())
    sm = float(scene.station.root_physx_view.get_masses().sum())
    am = float(scene.alphabet.root_physx_view.get_masses().sum())
    print(f"[solve] mass readback: gate={gm:.3f} tray={tm:.3f} station={sm:.1f} "
          f"can={am:.3f} kg", flush=True)
    assert 0.45 < gm < 0.75, f"gate mass wrong: {gm}"
    assert 0.30 < tm < 0.50, f"tray mass wrong: {tm}"
    assert 20.0 < sm < 30.0, f"station mass wrong: {sm}"
    assert 0.25 < am < 0.45, f"can mass wrong: {am}"
    sp = (scene.station.data.root_pos_w - scene.env_origins)[0]
    sq = scene.station.data.root_quat_w[0]
    syaw = math.degrees(2.0 * math.atan2(float(sq[3]), float(sq[0])))
    print(f"[solve] layout readback (seed {args.seed}): "
          f"station=({float(sp[0]):+.3f},{float(sp[1]):+.3f}) yaw={syaw:+.1f}deg "
          f"gate_c={gate_x():+.3f} alphabet_slot={int(scene.alphabet_slot[0])}",
          flush=True)
    report("reset")
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert abs(gate_x()) < c.stroke + 0.005, f"gate off its stroke: c={gate_x():+.3f}"
    assert not bool(scene.alphabet_in_tray()[0]), "can must start outside the tray"
    s0 = print_score("P0 reset+settle (cans on the ground, vault sealed)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- shared bang-bang gate drive ------------------------------------------
    st_q = scene.station.data.root_quat_w
    state = {"mode": 0, "q_ref": scene.gate.data.root_quat_w.clone()}

    def drive_gate(sign: float, x_target: float, fmax: float,
                   max_steps: int) -> bool:
        """Overdamped position servo on the gate along station-local x: CoM force
        kp*clamp(err) - kd*v (terminal speed ~0.08 m/s, so the gate can never slam
        a stop and slingshot). Probes the force-frame convention ONCE (drag shows
        up immediately); a small bias escalates on stall. Releases when the gate is
        within 5 mm of `x_target` (or past it, direction `sign`) and nearly still.
        Always clears the force."""
        xdir = quat_apply(st_q, torch.tensor([1.0, 0.0, 0.0],
                                             device=device).expand(n, 3))
        kp, kd, bias = 120.0, 25.0, 0.0
        probe_s = sign * gate_x()
        frame_checked = False
        released = False
        for i in range(max_steps):
            s = gate_x()
            err = x_target - s
            v = float((scene.gate.data.root_lin_vel_w[0] * xdir[0]).sum())
            if sign * (s - x_target) > -0.005 and abs(v) < 0.02:
                released = True
                break
            mag = kp * max(-0.02, min(0.02, err)) - kd * v \
                + (math.copysign(bias, err) if bias > 0.0 else 0.0)
            mag = max(-fmax, min(fmax, mag))
            f = encode_force(state["mode"], state["q_ref"],
                             scene.gate.data.root_quat_w,
                             mag * xdir).view(n, 1, 3)
            scene.gate.set_external_force_and_torque(f, zero_wrench,
                                                     env_ids=all_ids,
                                                     is_global=True)
            env.step(no_action)
            if i % 45 == 44:
                s_new = sign * gate_x()
                if not frame_checked:
                    frame_checked = True
                    if s_new < probe_s - 0.004:
                        state["mode"] ^= 1  # frame drag: it moved the wrong way
                        state["q_ref"] = scene.gate.data.root_quat_w.clone()
                        print(f"[solve] drive moved the gate the WRONG way "
                              f"(s {probe_s:+.3f} -> {s_new:+.3f}); "
                              f"force-frame mode -> {state['mode']}", flush=True)
                elif s_new < probe_s + 0.002 \
                        and sign * (x_target - gate_x()) > 0.008:
                    bias = min(bias + 0.5, 6.0)
                    from isaaclab.utils.math import quat_inv, quat_mul
                    gl = scene._station_local(scene.gate.data.root_pos_w)[0]
                    rq = quat_mul(quat_inv(scene.station.data.root_quat_w),
                                  scene.gate.data.root_quat_w)[0]
                    gav = scene.gate.data.root_ang_vel_w[0]
                    kl = can_loc()
                    print(f"[solve] drive stalled at c={gate_x():+.3f}; "
                          f"bias -> {bias:.1f} N | gate_st=({float(gl[0]):+.4f},"
                          f"{float(gl[1]):+.4f},{float(gl[2]):+.4f}) "
                          f"relq=({float(rq[0]):+.4f},{float(rq[1]):+.4f},"
                          f"{float(rq[2]):+.4f},{float(rq[3]):+.4f}) "
                          f"gav=({float(gav[0]):+.2f},{float(gav[1]):+.2f},"
                          f"{float(gav[2]):+.2f}) "
                          f"can_st=({float(kl[0]):+.4f},{float(kl[1]):+.4f},"
                          f"{float(kl[2]):+.4f})", flush=True)
                probe_s = s_new
        scene.gate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)
        return released

    # ---------------- phase 1: slide the gate RIGHT — open hatch A, seal window B -----------
    ok = drive_gate(+1.0, 0.065, 8.0, 1200)
    if not ok:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (gate never opened the hatch)", flush=True)
        os._exit(1)
    step(60)  # coast + settle, hands-off
    report("gate->right")
    assert bool(scene.hatch_passes()[0]), f"hatch must pass, c={gate_x():+.3f}"
    assert not bool(scene.window_passes()[0]), \
        f"window must be sealed while the hatch is open, c={gate_x():+.3f}"
    s1 = print_score("P1 gate slid right: hatch open, window sealed")
    assert s1 >= s0 - 1e-6, "score decreased across P1"
    assert not bool(scene.success()[0]), "cannot be success with the can outside"

    # ---------------- phase 2: drop the can through the open hatch --------------------------
    # TRANSPORT: write the can to free air 8 mm above the roof plane over the open
    # hatch, upright, zero velocity. Outside every rubric volume (asserted) —
    # gravity and the ramp do every load-bearing part.
    st_p = scene.station.data.root_pos_w
    st_qn = scene.station.data.root_quat_w
    loc = torch.zeros(n, 3, device=device)
    loc[:, 1] = 0.140
    loc[:, 2] = c.roof_top + c.can_height / 2 + 0.008
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = st_p + quat_apply(st_qn, loc)
    st[:, 3:7] = st_qn  # upright (cylinder axis stays world-z under yaw-only rot)
    scene.alphabet.write_root_state_to_sim(st, all_ids)
    env.iscene.update(0.0)
    assert not bool(scene.alphabet_in_tray()[0]), "release must be OUTSIDE the tray"
    assert not bool(scene.in_vault(scene.alphabet.data.root_pos_w)[0]), \
        "release must be OUTSIDE the vault"
    assert not bool(scene.in_chamber(scene.alphabet.data.root_pos_w)[0]), \
        "release must be OUTSIDE the antechamber (above the roof)"
    kl = can_loc()
    assert abs(float(kl[0])) < c.hatch_x - c.can_radius - 0.005 \
        and c.hatch_y[0] + c.can_radius < float(kl[1]) < c.hatch_y[1] - c.can_radius, \
        f"release not over the open hatch: ({float(kl[0]):+.3f},{float(kl[1]):+.3f})"
    # hands-off: fall through the hatch, slide the ramp, rest against blade B
    staged = False
    for j in range(720):
        env.step(no_action)
        if bool(scene._staged[0]) \
                and float(scene.alphabet.data.root_lin_vel_w[0].norm()) < 0.05:
            staged = True
            break
        if j % 180 == 179:
            report(f"drop-{j + 1}")
    step(60)  # margin of settle, hands-off
    report("hatch-drop")
    if not staged and not bool(scene._staged[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (can never staged at the ramp bottom)", flush=True)
        os._exit(1)
    assert bool(scene._in_chamber[0]), "in_chamber latch must be set"
    assert not bool(scene._in_vault[0]), \
        "can must NOT be in the vault while the window is sealed"
    assert not bool(scene.success()[0]), "cannot be success with the can staged"
    s2 = print_score("P2 can dropped through the hatch, staged against the window blade")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_chamber + c.w_staged - 1e-5, \
        f"P2 score {s2} (expect >= {c.w_chamber + c.w_staged})"

    # ---------------- phase 3: cycle the airlock — window opens, gravity delivers -----------
    ok = drive_gate(-1.0, -0.055, 8.0, 1200)
    if not ok:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (gate never opened the window)", flush=True)
        os._exit(1)
    assert bool(scene.window_passes()[0]), f"window must pass, c={gate_x():+.3f}"
    # hands-off: the can slides through the window, falls off the sill, lands in the tray
    got = False
    for j in range(720):
        env.step(no_action)
        if bool(scene.success()[0]):
            got = True
            break
        if j % 180 == 179:
            report(f"transfer-{j + 1}")
    report("window-transfer")
    if not (got or bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (can did not land in the tray)", flush=True)
        os._exit(1)
    assert bool(scene._in_vault[0]), "in_vault latch must be set"
    s3 = print_score("P3 airlock cycled: can through the window, at rest in the tray")
    assert s3 >= s2 - 1e-6, "score decreased across the transfer"

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
