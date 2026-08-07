"""Teleport solution for ArchQuarryScene (sim_gen task `roll_ball_i81`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. EXPEL A BOULDER (contact dynamics — never teleported): a velocity-regulated
   horizontal CoM force (the fingertip-push emulation; the boulder crowns stand proud
   of the walls and the trough is open-top) drives one white boulder out of its
   wall-and-ball wedge, along the trough and UP the 20-degree crest ramp against the
   restoring slope, until it crosses the crest and falls off the fixture. Every
   millimetre — the wedge breakaway, the uphill climb, the crest crossing, the drop —
   is physics. Pod force-frame quirk: some pods rotate an applied wrench by the
   body's rotation since reset (fatal for a ROLLING body); the desired world force is
   (de/en)coded per `encode_force` and the correct mode is PROBED from actual
   progress, not assumed.
2. FREE THE RED BALL (contact dynamics — never teleported): with the arch broken the
   surviving boulder topples aside; a velocity-limited vertical force (the
   pinch-and-lift emulation — the 50 mm ball fits the 80 mm jaw) lifts the red ball
   out of the V-groove and through the OPEN TOP. If the survivor is still perched on
   the red ball, a small side nudge on the survivor (also contact) topples it first.
   Only once the readback shows the red ball risen clear of the trough is it a free
   body in open air.
3. TRANSPORT + PLACE (teleport = free-space transport only): ONE pose write carries
   the already-freed red ball to a hover just above the nest-ring centre; it FALLS
   the last few centimetres and settles on the ground inside the rim by contact.
   Nothing is bypassed: the ball entered the nest under gravity through the ring's
   open top, which is exactly how the arm would deliver it.

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
    from .scene import _qapply, encode_force  # noqa: F401  (registers the scene)
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, encode_force  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.arch_quarry")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback so
    # distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc(body) -> torch.Tensor:
        return scene._quarry_local(body)[0]

    def report(tag: str) -> None:
        rl = loc(scene.red)
        al = loc(scene.boulder_a)
        bl = loc(scene.boulder_b)
        print(f"[solve] {tag:12s} | red=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):.3f}) A=({float(al[0]):+.3f},{float(al[1]):+.3f},"
              f"{float(al[2]):.3f}) B=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
              f"{float(bl[2]):.3f}) expel={bool(scene._expelled[0])} "
              f"freed={bool(scene._freed[0])} app={float(scene._app_max[0]):.3f} "
              f"nest_d={float(scene._nest_xy_d(scene.red)[0]):.3f} "
              f"in_nest={bool(scene._red_in_nest()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)  # boulders drop onto the red ball and close the arch
    qp = (scene.quarry.data.root_pos_w - scene.env_origins)[0]
    qq = scene.quarry.data.root_quat_w[0]
    q_yaw = 2.0 * math.atan2(float(qq[3]), float(qq[0]))
    np_ = (scene.nest.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"quarry=({float(qp[0]):+.3f},{float(qp[1]):+.3f}) "
          f"yaw={math.degrees(q_yaw):+.1f}deg "
          f"nest=({float(np_[0]):+.3f},{float(np_[1]):+.3f})", flush=True)
    report("reset")
    rl = loc(scene.red)
    al, bl = loc(scene.boulder_a), loc(scene.boulder_b)
    assert float(rl[2]) < 0.05, "red ball did not seat in the groove"
    for nm, xl in (("A", al), ("B", bl)):
        assert 0.040 < float(xl[2]) < 0.085, f"boulder {nm} did not wedge aloft"
        assert 0.045 < abs(float(xl[1])) < 0.085, f"boulder {nm} not at a wall seat"
    s0 = print_score("P0 reset+settle (arch closed over the red ball)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.02, f"baseline score should be ~0, got {s0}"

    # Force-frame reference orientations (readback now) + probed modes per body.
    q_ref_boulder = scene.boulder_a.data.root_quat_w.clone()
    q_ref_red = scene.red.data.root_quat_w.clone()

    # pick the boulder on the +v side or -v side — either works; take A
    tgt = scene.boulder_a
    survivor = scene.boulder_b

    # ---------------- phase 1: EXPEL BOULDER A over the crest (contact dynamics) ------------
    # Velocity-regulated horizontal CoM force along the quarry's +u axis, with a small
    # lateral centering term. The force-frame mode is probed from measured u-progress:
    # if the boulder moves AWAY, flip the encoding (rolling body — R_now churns).
    push_u = _qapply(scene.quarry.data.root_quat_w,
                     torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))
    push_v = _qapply(scene.quarry.data.root_quat_w,
                     torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3))
    crest_u = c.crest_h / math.tan(math.radians(c.alpha_deg))
    f_push, v_des = 3.0, 0.20
    mode = 0
    win_i, win_u = 0, float(loc(tgt)[0])
    stalls = 0
    expelled = False
    for i in range(3000):
        tl = loc(tgt)
        u, v = float(tl[0]), float(tl[1])
        if not bool(scene._in_cell(tgt)[0]):
            expelled = True
            break
        vel = tgt.data.root_lin_vel_w[0]
        u_vel = float((vel * push_u[0]).sum())
        f_axis = f_push if u_vel < v_des else 0.6 * f_push
        f_lat = max(-1.0, min(1.0, -8.0 * v - 1.5 * float((vel * push_v[0]).sum())))
        f_world = push_u * f_axis + push_v * f_lat
        f_arg = encode_force(mode, q_ref_boulder, tgt.data.root_quat_w, f_world)
        tgt.set_external_force_and_torque(f_arg.view(n, 1, 3).contiguous(),
                                          zero_wrench, env_ids=all_ids,
                                          is_global=True)
        env.step(no_action)
        if i - win_i >= 40:  # progress probe: a wrong force-frame mode (fatal for a
            # ROLLING body) shows up as regression OR as a persistent jiggling stall
            if u < win_u - 0.006:
                mode = 1 - mode
                stalls = 0
                f_push = min(f_push, 5.0)
                print(f"[solve] expel: moving away (u {win_u:+.3f} -> {u:+.3f}); "
                      f"force-frame mode -> {mode}", flush=True)
            elif u < win_u + 0.004:
                stalls += 1
                if stalls >= 2:
                    mode = 1 - mode
                    stalls = 0
                    f_push = min(f_push, 5.0)
                    print(f"[solve] expel: persistent stall at u={u:+.3f}; "
                          f"force-frame mode -> {mode}", flush=True)
                else:
                    f_push = min(f_push + 0.8, 8.0)
                    print(f"[solve] expel: stalled at u={u:+.3f} -> "
                          f"f_push={f_push:.1f} N", flush=True)
            else:
                stalls = 0
            win_i, win_u = i, u
    clear(tgt)
    report("expelled")
    assert expelled, "uphill push never carried the boulder over the crest"
    step(180)  # hands-off: the expelled boulder lands and rolls out; survivor topples
    report("post-expel")
    assert bool(scene._expelled[0]), "expulsion did not latch"
    assert not bool(scene._in_cell(tgt)[0]), "boulder rolled back into the cell?!"
    s1 = print_score("P1 boulder expelled over the crest (arch broken)")
    assert s1 >= s0 - 1e-6, "score decreased across expulsion"

    # ---------------- phase 2: FREE THE RED BALL by vertical contact lift -------------------
    # Velocity-limited vertical CoM force lifts the red ball out of the groove and
    # through the open top. If the survivor still perches on it (lift stalls), nudge
    # the survivor sideways (contact) and retry.
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    grav = c.red_mass * 9.81
    mode_r = 0
    freed = False
    for attempt in range(4):
        f_lift, v_lift = grav + 1.2, 0.30
        win_i, win_z = 0, float(loc(scene.red)[2])
        stalls_r = 0
        for i in range(600):
            rl = loc(scene.red)
            if float(rl[2]) > c.freed_z + 0.03:
                freed = True
                break
            vz = float(scene.red.data.root_lin_vel_w[0, 2])
            fz = f_lift if vz < v_lift else grav
            f_arg = encode_force(mode_r, q_ref_red, scene.red.data.root_quat_w,
                                 up * fz)
            scene.red.set_external_force_and_torque(
                f_arg.view(n, 1, 3).contiguous(), zero_wrench, env_ids=all_ids,
                is_global=True)
            env.step(no_action)
            if i - win_i >= 40:
                z = float(loc(scene.red)[2])
                if z < win_z - 0.004:
                    mode_r = 1 - mode_r
                    stalls_r = 0
                    print(f"[solve] lift: sinking (z {win_z:.3f} -> {z:.3f}); "
                          f"force-frame mode -> {mode_r}", flush=True)
                elif z < win_z + 0.003:
                    stalls_r += 1
                    if stalls_r >= 2:
                        mode_r = 1 - mode_r
                        stalls_r = 0
                        print(f"[solve] lift: persistent stall at z={z:.3f}; "
                              f"force-frame mode -> {mode_r}", flush=True)
                    else:
                        f_lift = min(f_lift + 0.5, grav + 3.5)  # never lifts 1.1 kg
                        print(f"[solve] lift: stalled at z={z:.3f} -> "
                              f"f_lift={f_lift:.1f} N", flush=True)
                else:
                    stalls_r = 0
                win_i, win_z = i, z
        clear(scene.red)
        if freed:
            break
        # survivor still perched: topple it off the red ball with a small +u nudge
        print(f"[solve] lift stalled (attempt {attempt}); nudging the survivor off",
              flush=True)
        for _ in range(90):
            sv = float((survivor.data.root_lin_vel_w[0] * push_u[0]).sum())
            fs = 2.0 if sv < 0.15 else 0.5
            f_arg = encode_force(mode, q_ref_boulder, survivor.data.root_quat_w,
                                 push_u * fs)
            survivor.set_external_force_and_torque(
                f_arg.view(n, 1, 3).contiguous(), zero_wrench, env_ids=all_ids,
                is_global=True)
            env.step(no_action)
        clear(survivor)
        step(60)
    report("freed")
    assert freed, "vertical lift never carried the red ball clear of the trough"
    assert bool(scene._freed[0]), "freeing did not latch"
    s2 = print_score("P2 red ball lifted out through the open top (contact)")
    assert s2 >= s1 - 1e-6, "score decreased across the lift"

    # ---------------- phase 3: TRANSPORT to the nest (teleport, free space) -----------------
    # ONE pose write: the freed red ball to a hover just above the nest centre; it
    # falls in and settles on the ground inside the rim by contact.
    nest_p = scene.nest.data.root_pos_w
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = nest_p
    st[:, 2] += c.red_r + 0.045  # hover: rim is 12 mm; drop ~2 cm into the ring
    st[:, 3] = 1.0
    scene.red.write_root_state_to_sim(st, all_ids)
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        if bool(scene._red_in_nest()[0]) and bool(scene._still(scene.red)[0]):
            quiet += 1
            if quiet >= 30:
                break
        else:
            quiet = 0
    report("nested")
    assert bool(scene._red_in_nest()[0]), "red ball did not settle inside the nest ring"
    s3 = print_score("P3 red ball delivered into the nest ring")
    assert s3 >= s2 - 1e-6, "score decreased across delivery"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after delivery)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, no intervention) ------
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
    except BaseException as exc:  # noqa: BLE001 — Kit teardown hangs; die loudly now
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
