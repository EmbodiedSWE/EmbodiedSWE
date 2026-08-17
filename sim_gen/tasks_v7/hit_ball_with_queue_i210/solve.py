"""Teleport solution for BallCorralScene (sim_gen task `hit_ball_with_queue_i210`) —
the task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. TRANSPORT THE CAGE (teleport): one pose write stages the open-mouth cage HOVERING
   45 mm above the deck, centred over the white ball with a deliberate (6, 4) mm
   lateral offset — inside the 17.5 mm capture clearance but off the perfect centre,
   so the seating must tolerate a realistic place, not a magic one. Free-space
   transport only: the cage is NOT written into the seated/caged state.
2. SEAT THE CAGE (contact dynamics — never teleported into place): the cage is
   RELEASED and falls under gravity; its rim lands on the deck AROUND the ball
   (the walls must clear the ball's crown on the way down), rocks, and settles
   flush. The scene's own `caged()` readback (rim flush + upright + ball enclosed)
   must confirm it.
3. ESCORT THE CAGED BALL (contact dynamics): a velocity-limited horizontal force at
   the cage's CoM — which is authored AT RIM LEVEL, so this is exactly a low
   fingertip push on the wall with no tipping moment — slides the caged assembly
   across the tilted deck. The captive ball is pushed along by the cage wall,
   climbing its 4 mm cradle lip and rolling enclosed the whole way; the ball itself
   is NEVER touched. A runtime probe toggles the wrench pre-encoding if the pod's
   external-force API drags the frame. The force is CUT when the cage centre is
   within 15 mm of the goal point.
4. SETTLE AND HOLD (pure physics): everything comes to rest with the cage centred
   on the green disc and the white ball enclosed at deck level inside it —
   success() judges only this physical outcome.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ball_corral")().build(num_envs=args.num_envs,
                                                 device=device)
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

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def zone_dist() -> float:
        g = scene._local(scene.cage)[0]
        return float((g[0:2] - scene._zone[0]).norm())

    def report(tag: str) -> None:
        wl = scene._local(scene.white)[0]
        gl = scene._local(scene.cage)[0]
        print(f"[solve] {tag:12s} | white_loc=({float(wl[0]):+.3f},{float(wl[1]):+.3f},"
              f"{float(wl[2]):.3f}) cage_loc=({float(gl[0]):+.3f},"
              f"{float(gl[1]):+.3f},{float(gl[2]):+.3f}) zone_d={zone_dist():.3f} "
              f"flush={bool(scene._cage_flush()[0])} "
              f"caged={bool(scene.caged()[0])} "
              f"caged_ever={bool(scene._caged_ever[0])} "
              f"delivered_ever={bool(scene._delivered_ever[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    pp = (scene.plateau.data.root_pos_w - scene.env_origins)[0]
    pq = scene.plateau.data.root_quat_w
    up_w = quat_apply(pq, torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))[0]
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(up_w[2])))))
    yaw_deg = math.degrees(2.0 * math.atan2(float(pq[0, 3]), float(pq[0, 0])))
    wl0 = scene._local(scene.white)[0]
    dl0 = scene._local(scene.decoy)[0]
    gl0 = scene._local(scene.cage)[0]
    zn = scene._zone[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"plateau=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"yaw={yaw_deg:+.1f}deg tilt={tilt_deg:.2f}deg "
          f"zone=({float(zn[0]):+.3f},{float(zn[1]):+.3f}) "
          f"white_loc=({float(wl0[0]):+.3f},{float(wl0[1]):+.3f}) "
          f"decoy_loc=({float(dl0[0]):+.3f},{float(dl0[1]):+.3f}) "
          f"cage_loc=({float(gl0[0]):+.3f},{float(gl0[1]):+.3f})", flush=True)
    report("reset")
    assert abs(float(wl0[2]) - c.ball_rest_z) < 0.010, "white ball not resting in a cradle"
    assert not bool(scene.caged()[0]), "ball spawned caged?!"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: transport the cage (teleport, HOVER only) --------------------
    # One pose write: the cage hovers with its rim 45 mm above the deck, centred over
    # the white ball with a deliberate (6, 4) mm lateral offset, upright in the
    # plateau frame. NOT written seated/caged.
    p_pos = scene.plateau.data.root_pos_w
    p_quat = scene.plateau.data.root_quat_w
    wl = scene._local(scene.white)
    hover_loc = wl.clone()
    hover_loc[:, 0] += 0.006
    hover_loc[:, 1] += 0.004
    hover_loc[:, 2] = 0.045
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = p_pos + quat_apply(p_quat, hover_loc)
    st[:, 3:7] = p_quat  # upright w.r.t. the deck
    scene.cage.write_root_state_to_sim(st, all_ids)
    step(2)
    assert not bool(scene.caged()[0]), \
        "hover pose already reads as caged (teleport must not seat the cage)"
    s1 = print_score("P1 cage transported to hover over the white ball (not seated)")

    # ---------------- phase 2: SEAT THE CAGE (contact dynamics) -----------------------------
    # Release: the cage falls, its rim lands on the deck around the ball, settles flush.
    seated = False
    for _ in range(600):
        env.step(no_action)
        if bool(scene.caged()[0]) and bool(scene._still()[0]):
            seated = True
            break
    report("seated")
    assert seated, "cage did not settle flush around the white ball"
    assert bool(scene._caged_ever[0]), "capture did not latch"
    s2 = print_score("P2 cage dropped and seated flush around the ball")
    assert s2 >= s1 - 1e-6 and s2 >= 0.30 - 1e-6, "capture credit missing"

    # ---------------- phase 3: ESCORT (contact dynamics) ------------------------------------
    # Full-vector velocity-error servo at the cage CoM (= rim level: a low wall push,
    # no tipping moment). Three field-tested safeguards:
    #  * the pod's external-force API can DRAG the wrench frame (stale R_ref across
    #    resets) — the applied force is a rotated copy of the commanded one. An
    #    axis-projected servo cannot see that (v_along stays low, full force stays
    #    on, runaway). The error here is the full deck-plane velocity VECTOR, which
    #    is self-capping under any rotation, plus a displacement-direction probe
    #    that toggles world-frame vs per-step body-frame encoding, plus a hard
    #    runaway speed guard.
    #  * a strong CoM push on a CONTAINING slider pitch-wedges it onto its payload:
    #    the cage climbs the captive ball, its rim lifts, the ball escapes under it.
    #    Guard: whenever the cage is not flush (or its origin rides high) cut the
    #    wrench and back off ~25 steps so gravity reseats the rim.
    #  * the captive ball must climb its 4 mm cradle lip right at the start — a slow
    #    crossing phase (v_tgt = 0.03 m/s) until the ball is well clear of its
    #    cradle keeps that climb quasi-static.
    up_vec = quat_apply(p_quat, torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))
    cradle_xy = scene._local(scene.white)[0, 0:2].clone()
    encode_mode = 0  # 0: world-frame call; 1: per-step body-frame pre-encode

    def apply_push(f_w: torch.Tensor) -> None:
        if encode_mode == 0:
            scene.cage.set_external_force_and_torque(
                f_w.view(n, 1, 3).contiguous(), zero_wrench, env_ids=all_ids,
                is_global=True)
        else:
            f_b = quat_apply_inverse(scene.cage.data.root_quat_w, f_w)
            scene.cage.set_external_force_and_torque(
                f_b.view(n, 1, 3).contiguous(), zero_wrench, env_ids=all_ids)

    def cut_push() -> None:
        scene.cage.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                 env_ids=all_ids)

    # Force law: feed-forward along the course (breaks stiction; ESCALATES on
    # stall — a pure velocity-error servo saturates at k_v*v_tgt < stiction and
    # freezes) + velocity-error braking term (self-capping under any wrench-frame
    # rotation: if the applied force is a rotated copy, the resulting sideways
    # velocity feeds straight back as a counter-force).
    f_ff = 2.5    # N feed-forward — assembly weighs ~7 N, sliding friction ~2 N
    f_cap = 12.0  # N absolute clamp
    k_v = 45.0    # N per m/s of velocity error (K*dt/m ≈ 0.5 < 1: stable one-step-lag)
    best_d = start_d = prev_d = zone_dist()
    arrived = False
    backoff = 0       # zero-wrench steps left (climb guard / runaway coast)
    push_steps = 0    # active-push steps since the probe window opened
    probe_p0 = scene.cage.data.root_pos_w[0, 0:2].clone()
    probe_dir = torch.zeros(2, device=device)
    last_bump = 0
    for i in range(3000):
        d = zone_dist()
        if d < 0.015:
            arrived = True
            break
        if d > prev_d + 0.005:  # impact/explosion telemetry
            cv = scene.cage.data.root_lin_vel_w[0]
            bv = scene.white.data.root_lin_vel_w[0]
            print(f"[solve] JUMP i={i} d {prev_d:.3f}->{d:.3f} "
                  f"|v_cage|={float(cv.norm()):.2f} |v_ball|={float(bv.norm()):.2f}",
                  flush=True)
        prev_d = d
        gl = scene._local(scene.cage)
        to_goal = torch.zeros(n, 3, device=device)
        to_goal[:, 0:2] = scene._zone - gl[:, 0:2]
        to_goal = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        u_dir = quat_apply(p_quat, to_goal)
        vel = scene.cage.data.root_lin_vel_w
        speed = float(vel[0].norm())

        # hard guards: cut the wrench and let physics recover
        climbing = float(gl[0, 2]) > 0.008 or not bool(scene._cage_flush()[0])
        if speed > 0.35:  # runaway = rotated force: coast + toggle encoding
            encode_mode ^= 1
            backoff = max(backoff, 30)
            print(f"[solve] i={i} runaway |v|={speed:.2f} -> coast, encode mode "
                  f"{encode_mode}", flush=True)
        elif climbing and backoff == 0:
            backoff = 25  # cage riding up on the ball: reseat under gravity
        if backoff > 0:
            backoff -= 1
            cut_push()
            env.step(no_action)
            push_steps = 0
            probe_dir = torch.zeros(2, device=device)
            probe_p0 = scene.cage.data.root_pos_w[0, 0:2].clone()
            continue

        # displacement-direction frame probe: if the measured course deviates far
        # from the commanded one, the encoding is wrong — toggle it
        push_steps += 1
        probe_dir += u_dir[0, 0:2]
        if push_steps >= 60:
            disp = scene.cage.data.root_pos_w[0, 0:2] - probe_p0
            if float(disp.norm()) > 0.004:
                cosang = float((disp / disp.norm()
                                * (probe_dir / probe_dir.norm())).sum())
                if cosang < 0.34:  # > ~70 deg off the commanded course
                    encode_mode ^= 1
                    backoff = 20
                    print(f"[solve] i={i} course "
                          f"{math.degrees(math.acos(max(-1.0, min(1.0, cosang)))):.0f}"
                          f" deg off -> encode mode {encode_mode}", flush=True)
            push_steps = 0
            probe_dir = torch.zeros(2, device=device)
            probe_p0 = scene.cage.data.root_pos_w[0, 0:2].clone()

        # slow crossing while the captive ball climbs out of its cradle recess
        wl_now = scene._local(scene.white)[0, 0:2]
        near_cradle = float((wl_now - cradle_xy).norm()) < 0.05
        v_tgt = 0.03 if near_cradle else min(0.10, 0.4 * d + 0.02)
        v_err = v_tgt * u_dir - vel
        v_err = v_err - (v_err * up_vec).sum(dim=-1, keepdim=True) * up_vec
        f_vec = f_ff * u_dir + k_v * v_err
        fn = f_vec.norm(dim=-1, keepdim=True)
        f_vec = torch.where(fn > f_cap, f_vec * (f_cap / fn.clamp(min=1e-9)), f_vec)
        apply_push(f_vec)
        env.step(no_action)
        if i % 240 == 0:
            report(f"escort+{i}")
            # the captive ball must still be enclosed at deck level
            assert bool(scene._inside(scene.white)[0]), \
                "white ball escaped the cage during the escort"
        if d < best_d - 0.002:
            best_d, last_bump = d, i
            f_ff = max(2.5, f_ff - 0.3)  # relax an escalated push once moving
        elif i - last_bump > 120:  # stalled (stiction / cradle lip): escalate the
            last_bump = i          # FEED-FORWARD, not the clamp (gain-stall memory)
            f_ff = min(f_ff + 1.5, 10.0)
            print(f"[solve] stall at d={d:.3f} -> f_ff={f_ff:.1f} N", flush=True)
    cut_push()
    report("arrived")
    assert arrived, "escort never brought the cage to the goal point"
    assert bool(scene._inside(scene.white)[0]), \
        "white ball is not enclosed at the goal"
    s3 = print_score("P3 caged ball escorted to the goal point, force cut")
    assert s3 >= s2 - 1e-6, "score decreased across the escort"

    # ---------------- phase 4: SETTLE (pure physics) ----------------------------------------
    # The captive ball may wall-roll to a cage-corner pocket before truly parking:
    # demand a LONG success streak (90 steps = 0.75 s) so persistence cannot flicker.
    quiet = 0
    settled = False
    for i in range(900):
        env.step(no_action)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 90:
            settled = True
            break
    report("settled")
    assert settled, "assembly did not settle into the success state on the goal"
    s4 = print_score("P4 assembly settled: caged ball centred on the disc")
    assert s4 >= s3 - 1e-6, "score decreased across the settle"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, no intervention) ------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
