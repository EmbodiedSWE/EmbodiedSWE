"""Teleport solution for DieRollMatchScene (sim_gen task
`libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i369`) — the
task's legitimacy certificate.

NO transport teleports are needed at all: the die starts on the counter and every
interaction runs through contact dynamics via external wrenches, re-set every step
(world wrench rotated into the die's BODY frame each step — the frame-drag trap)
and zeroed before judging:
  - TIP-ROLLS: a horizontal push force at height `roll_push_h` (above the CoM —
    force at the CoM + the lever torque r x F with r = (0,0,+0.05)); at that height
    the tipping threshold ~0.58 mg is far below the mu_s*mg = 0.9 mg slide
    threshold, so the die pivots over a bottom edge. The push is CUT once the die
    is ~5 deg past the balance point (gravity completes the quarter-roll) so
    momentum cannot chain-roll it; the outcome is VERIFIED by top-face readback
    and re-planned, with force escalation 0.70 -> 0.86 mg on failed tips.
  - SLIDE: a velocity-regulated horizontal force at height `slide_push_h` (below
    the CoM — lever r = (0,0,-0.035) so the friction couple cannot tip it);
    static-breakaway feedforward beats mu_s*m*g ~= 8.8 N; KV*dt/m = 0.25 << 1
    (one-substep wrench-delay bound).

PLAN (read-only, from scene.describe() + the reference-die READBACK):
  P0 settle, read the reference die's up face = TARGET; assert score ~0 and the
     big die does not start target-up,
  P1 quarter-roll the die over its bottom edges until TARGET faces up. Rolling
     toward a horizontal direction u brings the face that faced -u up, so:
     target horizontal -> one roll toward -n_target; target DOWN -> two rolls the
     same way (pick the bottom-edge direction that best advances toward the mat).
     First settled non-initial face latches 0.15; target-up latches +0.25,
  P2 low push-slide the die onto the crimson mat (top face preserved — sliding
     never changes it): +0.20 progress credit as the running-max approach latch,
  P3 release, ring down to success() (still 30+ steps, flat, target-up, in-pad),
  P4 hands-off persistence >= 3.3 simulated seconds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds after the hands-off hold.

Run (forge): python -u -m
  simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i369.solve
  --headless [--seed N]
"""

from __future__ import annotations

import argparse
import math

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

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:  # older isaaclab name
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

FACE_AXES = scene_mod.FACE_AXES
FACE_NAMES = scene_mod.FACE_NAMES

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

G = 9.81
# Tip-roll push (mass 1.0, tipping threshold w/(2*h)*mg = 0.583 mg at h=0.120;
# slide threshold mu_s*mg = 0.9 mg — every escalation stays BETWEEN the two).
ROLL_F0 = 0.70          # first-attempt push, x mg
ROLL_FS = (0.70, 0.78, 0.86, 0.86)  # escalation ladder, all < mu_s = 0.9
ROLL_CUT_DOM = math.cos(math.radians(50.0))  # cut past the 45 deg balance point
ROLL_WCUT = 2.6         # rad/s — momentum guard while tilting
ROLL_PUSH_MAX = 360     # steps (3 s) per push attempt
# Slide servo (mass 1.0: KV*dt/m = 30/120 = 0.25). The friction feedforward is
# CONTINUOUS kinetic mu_d*m*g in the drive direction (a hard |v|<eps static-FF
# cutoff limit-cycles: the die surfs the threshold at ~0.02 m/s and never
# arrives); at rest FF + KV*v_des ~= 7.9 + 4.5 N beats the 8.83 N breakaway.
S_KP = 1.5
S_VCAP = 0.15
S_KV = 30.0
S_FF = 7.9   # ~= mu_d * m * g
S_FMAX = 14.0
SETTLE_LIN = 0.06
SETTLE_ANG = 0.35


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_roll_match")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    mg = c.die_mass * G
    lever_roll = c.roll_push_h - c.die_half   # +0.050: push above the CoM
    lever_slide = c.slide_push_h - c.die_half  # -0.035: push below the CoM

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def die_p0():
        p = scene.die_local()[0]
        return float(p[0]), float(p[1]), float(p[2])

    def top0() -> tuple[int, float]:
        idx, dom = scene.top_face()
        return int(idx[0]), float(dom[0])

    def face_normals0() -> list[list[float]]:
        """World normals of the 6 faces (env 0): rows of R applied to FACE_AXES."""
        q = scene.die.data.root_quat_w[0]
        qw, qx, qy, qz = (float(v) for v in q)
        ex = (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy))
        ey = (2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qw * qx))
        ez = (2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy))
        out = []
        for ax, ay, az in FACE_AXES:
            out.append([ax * ex[0] + ay * ey[0] + az * ez[0],
                        ax * ex[1] + ay * ey[1] + az * ez[1],
                        ax * ex[2] + ay * ey[2] + az * ez[2]])
        return out

    def wrench_off() -> None:
        scene.die.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def set_wrench_world(f_world: torch.Tensor, t_world: torch.Tensor) -> None:
        """Rotate a WORLD wrench into the die's body frame EVERY step (frame drag)."""
        q = scene.die.data.root_quat_w
        fb = quat_apply_inverse(q, f_world).unsqueeze(1)
        tb = quat_apply_inverse(q, t_world).unsqueeze(1)
        scene.die.set_external_force_and_torque(fb, tb, env_ids=all_ids)

    def settle(max_steps: int = 600, streak: int = 20) -> None:
        """Wrench off, wait for stillness to PERSIST (thresholds above the phantom
        velocity band), bounded — never an infinite wait."""
        wrench_off()
        ok = 0
        for _ in range(max_steps):
            env.step(no_action)
            lv = float(scene.die.data.root_lin_vel_w[0].norm())
            av = float(scene.die.data.root_ang_vel_w[0].norm())
            ok = ok + 1 if (lv < SETTLE_LIN and av < SETTLE_ANG) else 0
            if ok >= streak:
                break

    def report(tag: str) -> None:
        idx, dom = top0()
        x, y, z = die_p0()
        print(f"[solve] {tag:12s} | top={FACE_NAMES[idx]}({dom:+.3f}) "
              f"tgt={FACE_NAMES[int(scene.target[0])]} xyz=({x:+.3f},{y:+.3f},{z:+.3f}) "
              f"pad_err={float(scene.pad_err()[0]):.3f} | "
              f"latches r/o/p={float(scene.roll_latch[0]):.0f}/"
              f"{float(scene.orient_latch[0]):.0f}/{float(scene.prog_latch[0]):.2f} "
              f"settled={bool(scene.settled()[0])} | success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def push_roll(u: tuple[float, float], fscale: float, tag: str) -> None:
        """One push attempt: horizontal force F at roll_push_h toward u = CoM force
        + lever torque r x F, r = (0,0,+lever_roll). CUT past the balance point or
        on the momentum guard; gravity completes the quarter-roll."""
        orig_top, _ = top0()
        F = fscale * mg
        f_w = torch.zeros(n, 3, device=device)
        t_w = torch.zeros(n, 3, device=device)
        f_w[:, 0], f_w[:, 1] = F * u[0], F * u[1]
        # r x F with r = lever*z_hat: tau = lever * F * (z_hat x u)
        t_w[:, 0], t_w[:, 1] = -lever_roll * F * u[1], lever_roll * F * u[0]
        i = 0
        for i in range(ROLL_PUSH_MAX):
            set_wrench_world(f_w, t_w)
            env.step(no_action)
            dots = scene._axes_z(scene.die.data.root_quat_w)[0]
            od = float(dots[orig_top])
            w = float(scene.die.data.root_ang_vel_w[0].norm())
            if od < ROLL_CUT_DOM or (w > ROLL_WCUT and od < 0.94):
                break
        wrench_off()
        print(f"[solve] {tag}: push cut after {i + 1} steps "
              f"(orig_dom={float(scene._axes_z(scene.die.data.root_quat_w)[0][orig_top]):+.3f})",
              flush=True)
        settle()

    def roll_once(u: tuple[float, float], tag: str) -> bool:
        """Execute one verified quarter-roll toward world direction u: escalate the
        push if the die refuses to tip; True iff the top face CHANGED."""
        orig_top, _ = top0()
        for a, fs in enumerate(ROLL_FS):
            push_roll(u, fs, f"{tag}.a{a}(F={fs:.2f}mg)")
            now_top, dom = top0()
            if now_top != orig_top and dom > 0.9:
                exp = [i for i, nrm in enumerate(face_normals0()) if nrm[2] > 0.9]
                print(f"[solve] {tag}: rolled {FACE_NAMES[orig_top]} -> "
                      f"{FACE_NAMES[now_top]} (dom {dom:+.3f}, exp={exp})", flush=True)
                return True
            print(f"[solve] {tag}: no tip (top still {FACE_NAMES[now_top]}, "
                  f"dom {dom:+.3f}) — escalating", flush=True)
        return False

    def plan_roll_dir() -> tuple[float, float]:
        """Read the TARGET face's world normal; return the horizontal push direction
        u for the next quarter-roll (rolling toward u raises the face that faced -u).
        Target DOWN -> any bottom-edge direction works for the first of two
        same-direction rolls; pick the one best aimed at the mat (progress)."""
        tgt = int(scene.target[0])
        normals = face_normals0()
        nt = normals[tgt]
        x, y, _ = die_p0()
        to_pad = (c.pad_center[0] - x, c.pad_center[1] - y)
        norm = math.hypot(*to_pad) or 1.0
        to_pad = (to_pad[0] / norm, to_pad[1] / norm)
        if nt[2] < -0.7:  # target faces DOWN: two rolls the same way
            best, bu = -2.0, (1.0, 0.0)
            for nrm in normals:
                if abs(nrm[2]) < 0.5:  # horizontal face normal = legal roll direction
                    h = math.hypot(nrm[0], nrm[1]) or 1.0
                    cand = (nrm[0] / h, nrm[1] / h)
                    d = cand[0] * to_pad[0] + cand[1] * to_pad[1]
                    if d > best:
                        best, bu = d, cand
            print(f"[solve] plan: target {FACE_NAMES[tgt]} is DOWN "
                  f"(nz={nt[2]:+.3f}) -> roll toward ({bu[0]:+.2f},{bu[1]:+.2f}) "
                  f"twice", flush=True)
            return bu
        h = math.hypot(nt[0], nt[1]) or 1.0
        u = (-nt[0] / h, -nt[1] / h)
        print(f"[solve] plan: target {FACE_NAMES[tgt]} faces "
              f"({nt[0]:+.2f},{nt[1]:+.2f},{nt[2]:+.2f}) -> roll toward "
              f"({u[0]:+.2f},{u[1]:+.2f})", flush=True)
        return u

    # ---------------- phase 0: reset, settle, layout + target readback ---------------------
    step(90)
    tgt = int(scene.target[0])
    ref_top = int(scene.ref_top_face()[0])
    top, dom = top0()
    x, y, z = die_p0()
    print(f"[solve] readback (seed {args.seed}): die top={FACE_NAMES[top]} "
          f"(dom {dom:+.3f}) at ({x:+.3f},{y:+.3f},{z:+.3f}); reference shows "
          f"{FACE_NAMES[ref_top]} up; sampled target={FACE_NAMES[tgt]} "
          f"(init_top={FACE_NAMES[int(scene.init_top[0])]})", flush=True)
    assert ref_top == tgt, "reference die must DISPLAY the sampled target face"
    assert top == int(scene.init_top[0]), "die must rest on its sampled orientation"
    assert top != tgt, "target must not start up (null never starts solved)"
    assert dom > 0.98 and abs(z - c.die_half) < 0.01, "die must rest flat on the counter"
    report("reset")
    s0 = print_score("P0 reset+settle")
    assert s0 < 0.05, "score must start ~0"

    # ---------------- phase 1: quarter-roll until the target face is up --------------------
    rolls = 0
    first_roll_score = None
    for _ in range(10):  # a target never needs > 2 planned rolls + chain-roll repairs
        top, dom = top0()
        if top == tgt and dom > 0.9:
            break
        u = plan_roll_dir()
        assert roll_once(u, f"P1.r{rolls}"), "die refused to tip at every push level"
        rolls += 1
        if first_roll_score is None:
            report(f"P1-roll{rolls}")
            first_roll_score = print_score("P1a first quarter-roll rested")
            assert first_roll_score >= max(s0, 0.14), "first-roll credit missing"
    top, dom = top0()
    assert top == tgt and dom > 0.9, "target face must be up after rolling"
    assert float(scene.orient_latch[0]) > 0.5, "orientation latch must have fired"
    report("P1-done")
    s1 = print_score("P1 orientation solved (target face up)")
    assert s1 >= max(first_roll_score or 0.0, 0.39), "orientation credit missing"

    # ---------------- phase 2: low push-slide onto the crimson mat -------------------------
    f_w = torch.zeros(n, 3, device=device)
    t_w = torch.zeros(n, 3, device=device)
    pad = torch.tensor([c.pad_center[0], c.pad_center[1]], device=device)
    done, i = 0, 0
    for i in range(3600):
        p = scene.die_local()
        v = scene.die.data.root_lin_vel_w[:, 0:2]
        d = pad.unsqueeze(0) - p[:, 0:2]
        dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dirn = d / dist
        v_des = dirn * (S_KP * dist).clamp(max=S_VCAP)
        fh = S_KV * (v_des - v) + S_FF * dirn
        fn = fh.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        fh = fh * (fn.clamp(max=S_FMAX) / fn)
        f_w[:, 0:2] = fh
        # r x F with r = (0,0,lever_slide), lever_slide < 0: kills the tip couple
        t_w[:, 0] = -lever_slide * fh[:, 1]
        t_w[:, 1] = lever_slide * fh[:, 0]
        set_wrench_world(f_w, t_w)
        env.step(no_action)
        idx, dom = top0()
        assert dom > math.cos(math.radians(20.0)), "slide must not tip the die"
        if (i + 1) % 300 == 0:
            x, y, z = die_p0()
            print(f"[solve] P2 telemetry @{i + 1}: xy=({x:+.3f},{y:+.3f}) "
                  f"pad_err={float(scene.pad_err()[0]):.3f} dom={dom:+.3f} "
                  f"|f|={float(fn[0]):.2f}", flush=True)
        done = done + 1 if float(scene.pad_err()[0]) <= c.place_tol - 0.02 else 0
        if done >= 3:
            break
    wrench_off()
    settle()
    top, dom = top0()
    print(f"[solve] P2: slid to pad_err={float(scene.pad_err()[0]):.3f} in {i + 1} "
          f"servo steps, top={FACE_NAMES[top]} (dom {dom:+.3f}) — wrench OFF", flush=True)
    assert float(scene.pad_err()[0]) <= c.place_tol, "die must be centered on the mat"
    assert top == tgt, "sliding must preserve the top face"
    report("P2-done")
    s2 = print_score("P2 die slid onto the crimson mat")
    assert s2 >= max(s1, 0.55), "mat-approach credit missing"

    # ---------------- phase 3: ring down to success ----------------------------------------
    consec = 0
    for _ in range(2400):  # up to 20 s
        step(1)
        if bool(scene.success()[0]):
            consec += 1
            if consec >= 120:
                break
        else:
            consec = 0
    report("P3-ringdown")
    s3 = print_score("P3 settled on the mat, target face up")
    assert s3 >= s2 - 1e-6, "score decreased across P3"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after placement + ring-down)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands off) -----------
    hold, flickers = True, 0
    for i in range(400):  # 400 steps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                idx, dom = top0()
                lv = float(scene.die.data.root_lin_vel_w[0].norm())
                av = float(scene.die.data.root_ang_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: top={FACE_NAMES[idx]} "
                      f"dom={dom:+.3f} pad_err={float(scene.pad_err()[0]):.3f} "
                      f"lv={lv:.4f} av={av:.4f} settled={bool(scene.settled()[0])}",
                      flush=True)
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
