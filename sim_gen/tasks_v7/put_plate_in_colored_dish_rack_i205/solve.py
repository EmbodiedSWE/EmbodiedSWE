"""Teleport-transport solution for PlateSlotSwapScene (sim_gen task
`put_plate_in_colored_dish_rack_i205`) — the task's legitimacy certificate.

PLAN (read from scene.describe(): capacity-one pockets force a 3-move swap through
the gray transfer cradle; either plate may make the buffer move — this solve parks
the YELLOW plate first):
  P1 BUFFER  — force-lift the yellow plate straight up out of the BLUE slot until its
     rim clears the fins (a real edgewise extraction through contact/friction),
     teleport it through free air to a hover directly above the cradle pocket, then
     RELEASE: gravity threads it down the fin gap onto the pocket floor (the fins and
     end stops do the fine alignment by contact). Wait for the cradle seat latch
     (30 consecutive slow steps).
  P2 MATCH 1 — same lift / free-air transport / gravity insertion for the BLUE plate:
     out of the YELLOW slot, into the now-free BLUE slot. Wait for the match latch.
  P3 MATCH 2 — recover the yellow plate from the cradle into the YELLOW slot; wait
     for success() to hold 120 consecutive steps.
  P4 hands-off persistence >= 3.3 simulated seconds.

Teleports are TRANSPORT ONLY: both endpoints of every teleport are free-air hover
poses (velocities zeroed); every load-bearing interaction — extraction against fin
friction, threading the fin gap, seating on the pocket floor — happens through
contact dynamics (applied forces / gravity + contacts).

Pod force-frame quirk (measured across four instrumented runs): the frame in which
this pod applies an external force is NOT stable — the same world-z arg arrived
world-vertical in some runs/bodies (rotation-since-spawn drag is a pure yaw for the
edge-on-spawned plates, which preserves verticality) and along the plate's
horizontal disc axis in others (raw body-frame application), differing BETWEEN THE
TWO PLATES within a single run. The solve therefore trusts nothing and SELF-CORRECTS
per plate: every forced phase runs under an escape/regression MONITOR; on a wrong
guess the plate is recovered (re-seated in its source pocket via a free-air hover
teleport + gravity drop), the encoding for that plate is toggled
("world" passthrough <-> "body" pre-rotation by R_now^T), and the phase retries.
Insertions never apply force at all — a centered free release is encoding-proof,
so only the extractions depend on the certified encoding.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds at the end.

Run (forge): python -u -m simgen_tasks.put_plate_in_colored_dish_rack_i205.solve \
    --headless [--seed N]
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
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

BLUE_SLOT_SIGN = scene_mod.BLUE_SLOT_SIGN
GRAV = 9.81

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.plate_slot_swap")().build(num_envs=args.num_envs,
                                                    device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    weight = c.plate_m * GRAV

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc_of(name: str, frame) -> torch.Tensor:
        return scene._local(scene.plates[name], frame)[0]

    # Per-plate force-frame encoding, corrected empirically by the monitors below.
    mode = {"blue": "world", "yellow": "world"}

    def toggle(name: str) -> None:
        mode[name] = "body" if mode[name] == "world" else "world"
        print(f"[solve] force-frame mode[{name}] -> '{mode[name]}'", flush=True)

    def apply_fz(name: str, fz: float) -> None:
        """Apply a WORLD-vertical force, encoded through the plate's current mode."""
        body = scene.plates[name]
        f_world = torch.zeros(n, 3, device=device)
        f_world[0, 2] = fz
        if mode[name] == "body":
            f_arg = quat_apply_inverse(body.data.root_quat_w, f_world)
        else:
            f_arg = f_world
        body.set_external_force_and_torque(f_arg.view(n, 1, 3), zero_w,
                                           env_ids=all_ids, is_global=True)

    def clear_force(name: str) -> None:
        scene.plates[name].set_external_force_and_torque(zero_w, zero_w,
                                                         env_ids=all_ids,
                                                         is_global=True)

    def report(tag: str) -> None:
        lb = loc_of("blue", scene.rack)
        ly = loc_of("yellow", scene.rack)
        print(f"[solve] {tag:12s} | blue_rack=({float(lb[0]):+.3f},"
              f"{float(lb[1]):+.3f},{float(lb[2]):+.3f}) "
              f"yellow_rack=({float(ly[0]):+.3f},{float(ly[1]):+.3f},"
              f"{float(ly[2]):+.3f}) | "
              f"m_blue={bool(scene.matched('blue')[0])} "
              f"m_yellow={bool(scene.matched('yellow')[0])} "
              f"cradle_y={bool(scene.seated_in_cradle('yellow')[0])} "
              f"settled={bool(scene.settled()[0])} | "
              f"latches c={float(scene.cradle_latch[0]):.0f}/"
              f"m={float(scene.match_latch[0]):.0f} | "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def settle(max_steps: int = 720) -> None:
        for _ in range(max_steps // 30):
            step(30)
            if bool(scene.settled()[0]):
                break

    # ----- free-air hover teleport over a pocket (transport only) -------------------------
    def teleport_hover(name: str, frame, cx: float, tag: str) -> None:
        """Both endpoints are free air: callers only invoke this with the plate clear
        of (or escaped from) a pocket, and the hover pose (frame-local (cx, 0, 0.16))
        has its bottom rim 15 mm above the fin tops. Velocities zeroed."""
        q_f = frame.data.root_quat_w
        local = torch.tensor([cx, 0.0, 0.16], device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = frame.data.root_pos_w + quat_apply(q_f, local)
        st[:, 3:7] = scene._seat_quat(q_f)
        scene.plates[name].write_root_state_to_sim(st, all_ids)
        step(2)
        loc = loc_of(name, frame) if frame is not scene.cradle \
            else scene._local(scene.plates[name], scene.cradle)[0]
        print(f"[solve] {tag}: {name} hovering at "
              f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
              f"over cx={cx:+.3f}", flush=True)

    def reseat_drop(name: str, frame, cx: float, tag: str) -> bool:
        """Recovery: hover over the pocket and let GRAVITY drop the plate back onto
        the pocket floor (no applied force — encoding-proof), then settle."""
        teleport_hover(name, frame, cx, tag)
        for _ in range(10):
            step(15)
            loc = scene._local(scene.plates[name], frame)[0]
            if float(loc[2]) <= 0.09 and \
                    float(scene.plates[name].data.root_lin_vel_w[0].norm()) < 0.10:
                break
        settle()
        ok = bool(scene._seated(scene.plates[name], frame, cx)[0])
        print(f"[solve] {tag}: reseat -> seated={ok}", flush=True)
        return ok

    def escaped(name: str, frame, cx: float) -> bool:
        """Escape monitor: the plate left the pocket somewhere other than straight
        up (over a fin, over an end stop, or fell outside). Lateral checks only
        apply while the rim is still down among the fins (z_loc < 0.14): once the
        bottom rim clears the fin tops the plate is legitimately free and a little
        lateral drift is not an encoding failure."""
        loc = scene._local(scene.plates[name], frame)[0]
        z = float(loc[2])
        if z < 0.045:
            return True
        if z < 0.140 and (abs(float(loc[0]) - cx) > 0.055
                          or abs(float(loc[1])) > 0.115):
            return True
        return False

    # ----- vertical extraction: force-lift a plate clear of a pocket's fins ---------------
    def lift_clear(name: str, frame, cx: float, tag: str) -> bool:
        """Velocity-servoed vertical force until the plate's frame-local center z
        clears fin_h + plate_r + 0.03. Runs under the escape monitor: a wrong force
        encoding shoves the plate sideways within a few steps — abort, re-seat by
        gravity drop, toggle this plate's encoding, retry. A clean but stalled climb
        escalates the servo GAIN (friction, not encoding)."""
        clear_z = c.fin_h + c.plate_r + 0.012
        for attempt in range(4):
            t = f"{tag}-a{attempt}"
            kp, extra_cap, v_des = 6.0, 4.0, 0.12
            z0 = float(loc_of(name, frame)[2]) if frame is scene.rack \
                else float(scene._local(scene.plates[name], frame)[0, 2])
            win_i, win_z, bad, stalls = 0, z0, False, 0
            for i in range(900):
                loc = scene._local(scene.plates[name], frame)[0]
                z = float(loc[2])
                if z >= clear_z:
                    clear_force(name)
                    print(f"[solve] {t}: {name} clear at z_loc={z:.3f} after "
                          f"{i} steps (mode '{mode[name]}')", flush=True)
                    return True
                if escaped(name, frame, cx) or z < z0 - 0.012:
                    bad = True
                    print(f"[solve] {t}: {name} ESCAPED/regressed at "
                          f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                          f"{float(loc[2]):+.3f}) — encoding '{mode[name]}' wrong",
                          flush=True)
                    break
                vz = float(scene.plates[name].data.root_lin_vel_w[0, 2])
                apply_fz(name, weight
                         + max(-0.8 * weight, min(extra_cap, kp * (v_des - vz))))
                env.step(no_action)
                if i - win_i >= 90:
                    if z < win_z + 0.004:
                        stalls += 1
                        if stalls >= 3:
                            # gain maxed twice with no progress: the force is not
                            # arriving vertical — treat as a wrong encoding.
                            bad = True
                            print(f"[solve] {t}: {name} stalled HARD at "
                                  f"z_loc={z:.3f} — encoding '{mode[name]}' "
                                  f"suspect", flush=True)
                            break
                        kp = min(kp * 1.6, 25.0)
                        extra_cap = min(extra_cap + 2.0, 10.0)
                        print(f"[solve] {t}: {name} stalled at z_loc={z:.3f}; "
                              f"kp -> {kp:.1f}, cap -> {extra_cap:.1f}", flush=True)
                    win_i, win_z = i, z
            clear_force(name)
            if not bad:
                print(f"[solve] {t}: TIMED OUT climbing (mode kept)", flush=True)
            toggle(name)
            if not reseat_drop(name, frame, cx, f"{t}-reseat"):
                # one more drop try before giving up
                if not reseat_drop(name, frame, cx, f"{t}-reseat2"):
                    return False
        return False

    # ----- insertion: centered free release straight down the fin gap ---------------------
    def insert_plate(name: str, frame, cx: float, seat_fn, tag: str) -> bool:
        """Hover-teleport centered over the pocket, then release: GRAVITY threads the
        plate down the fin gap onto the pocket floor. No applied force at all — the
        insertion is encoding-proof (the pod's unstable wrench-frame semantics cannot
        touch a pure free fall), and the fin gap + end stops do the fine alignment by
        contact. Retries with fresh drops on a bad settle."""
        for attempt in range(3):
            t = f"{tag}-a{attempt}"
            if reseat_drop(name, frame, cx, t):
                if bool(seat_fn()[0]):
                    loc = scene._local(scene.plates[name], frame)[0]
                    print(f"[solve] {t}: {name} SEATED at "
                          f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                          f"{float(loc[2]):+.3f})", flush=True)
                    return True
            loc = scene._local(scene.plates[name], frame)[0]
            ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
            ax = float(quat_apply(scene.plates[name].data.root_quat_w,
                                  ez)[0, 2].abs())
            print(f"[solve] {t}: {name} drop did not seat — pose "
                  f"({float(loc[0]):+.3f},{float(loc[1]):+.3f},"
                  f"{float(loc[2]):+.3f}) axis_z={ax:.2f} — retry", flush=True)
        return False

    def wait_latch(get_val, tag: str, chunks: int = 20) -> bool:
        for _ in range(chunks):
            if float(get_val()[0]) > 0.5:
                print(f"[solve] {tag}: latch fired", flush=True)
                return True
            step(15)
        print(f"[solve] {tag}: latch NEVER fired", flush=True)
        return False

    # ---------------- phase 0: reset, settle, plan readback -------------------------------
    step(90)
    report("reset")
    assert bool(scene.seated_in_slot("blue", "yellow")[0]), \
        "blue plate must start in the YELLOW slot"
    assert bool(scene.seated_in_slot("yellow", "blue")[0]), \
        "yellow plate must start in the BLUE slot"
    assert not bool(scene.matched("blue")[0]) and not bool(scene.matched("yellow")[0])
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (swapped arrangement verified)")
    assert s0 < 0.05, f"score must start ~0, got {s0}"

    blue_cx = BLUE_SLOT_SIGN * c.slot_dx
    yellow_cx = -BLUE_SLOT_SIGN * c.slot_dx

    # ---------------- phase 1: BUFFER — yellow plate, blue slot -> cradle ------------------
    assert lift_clear("yellow", scene.rack, blue_cx, "P1-lift"), \
        "P1 extraction failed"
    assert insert_plate("yellow", scene.cradle, 0.0,
                        lambda: scene.seated_in_cradle("yellow"), "P1-insert"), \
        "P1 cradle insertion failed"
    assert wait_latch(lambda: scene.cradle_latch, "P1-latch"), \
        "cradle seat latch must fire"
    report("P1-buffer")
    s1 = print_score("P1 yellow plate parked in the transfer cradle")
    assert s1 >= s0 - 1e-6 and s1 >= 0.195, f"P1 score {s1} (expect 0.20)"

    # ---------------- phase 2: MATCH 1 — blue plate, yellow slot -> blue slot --------------
    assert lift_clear("blue", scene.rack, yellow_cx, "P2-lift"), \
        "P2 extraction failed"
    assert insert_plate("blue", scene.rack, blue_cx,
                        lambda: scene.matched("blue"), "P2-insert"), \
        "P2 blue-slot insertion failed"
    assert wait_latch(lambda: scene.match_latch, "P2-latch"), \
        "match latch must fire"
    report("P2-match1")
    s2 = print_score("P2 blue plate seated in the BLUE slot")
    assert s2 >= s1 - 1e-6 and s2 >= 0.545, f"P2 score {s2} (expect 0.55)"

    # ---------------- phase 3: MATCH 2 — yellow plate, cradle -> yellow slot ---------------
    assert lift_clear("yellow", scene.cradle, 0.0, "P3-lift"), "P3 extraction failed"
    assert insert_plate("yellow", scene.rack, yellow_cx,
                        lambda: scene.matched("yellow"), "P3-insert"), \
        "P3 yellow-slot insertion failed"
    # Ring-down: wait for success() to hold CONTINUOUSLY for 1 s.
    consec = 0
    for _ in range(2400):
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P3-match2")
    s3 = print_score("P3 both plates in their color slots (swap complete)")
    assert s3 >= s2 - 1e-6, "score decreased across P3"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after swap + ring-down)", flush=True)
        os._exit(1)
    assert s3 >= 0.99, f"P3 score {s3} (expect 1.0)"

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) ----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                print(f"[solve] persist flicker @step {i}: "
                      f"m_blue={bool(scene.matched('blue')[0])} "
                      f"m_yellow={bool(scene.matched('yellow')[0])} "
                      f"settled={bool(scene.settled()[0])}", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
