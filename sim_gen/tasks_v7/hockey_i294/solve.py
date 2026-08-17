"""Teleport solution for CageCaptureScene (sim_gen task `hockey_i294`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. STAGE (teleport = transport only): one pose write parks the cage on the far
   side of the white ball's approach corridor — mouth facing the ball, mouth
   centre 11.5 cm short of it. This is the arm's push-and-pivot repositioning of
   a receptacle it cannot lift; the approach credit gate (9 cm) sits INSIDE that
   distance, so the staging write earns nothing.
2. CAPTURE SLIDE (contact dynamics — never teleported): a horizontal external
   force at the cage CoM (the wrench emulation of the arm pushing the back wall)
   velocity-servos the 3.2 kg cage along the floor, steering on the BALL's
   position in the CAGE frame so the open mouth passes around the stationary
   ball, with a yaw-hold torque and a stall probe that escalates the force cap
   (the effectively-applied wrench on this pod can be a fraction of the
   commanded one). The push stops with the ball 5.5 cm behind the mouth plane;
   friction brakes the cage in ~1 mm. The ball is NEVER touched — the anchor
   clause would forfeit the run if it were.
3. CHOCK (teleport = transport only, then gravity): one pose write holds the
   yellow chock bar flat, centred, 4 mm above its seat just outside the mouth —
   the arm's place-from-above of a 0.18 kg bar — and gravity sets it down. The
   bar touches neither ball nor cage on the way down.
4. HANDS-OFF: nothing is touched from the chock set-down to the verdict.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds
after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if
success() still holds.

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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cage_capture")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    dt = 1.0 / 120.0

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def cage_yaw() -> float:
        q = scene.cage.data.root_quat_w[0]
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    def ball_loc() -> tuple[float, float, float]:
        loc = scene._local(scene.ball)[0]
        return float(loc[0]), float(loc[1]), float(loc[2])

    def anchor_err() -> float:
        d = scene.ball.data.root_pos_w[0, 0:2] - scene._anchor[0]
        return float(d.norm())

    def report(tag: str) -> None:
        xl, yl, zl = ball_loc()
        kl = scene._local(scene.chock)[0]
        print(f"[solve] {tag:12s} | ball_cage=({xl:+.3f},{yl:+.3f},{zl:+.3f}) "
              f"anchor_err={anchor_err() * 1000:.1f}mm yaw={math.degrees(cage_yaw()):+.1f}deg "
              f"chock_cage=({float(kl[0]):+.3f},{float(kl[1]):+.3f},{float(kl[2]):+.3f}) "
              f"appr={bool(scene._appr_ever[0])} part={bool(scene._part_ever[0])} "
              f"capt={bool(scene._capt_ever[0])} chk={bool(scene._chk_ever[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def apply_cage_wrench(fx: float, fy: float, tz: float) -> None:
        """World-frame horizontal force at the cage CoM + yaw-hold torque — the
        wrench emulation of the arm's push on the back wall."""
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 0], f[:, 0, 1] = fx, fy
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 2] = tz
        scene.cage.set_external_force_and_torque(f.contiguous(), t.contiguous(),
                                                 env_ids=all_ids, is_global=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)
    b0 = (scene.ball.data.root_pos_w - scene.env_origins)[0]
    d0 = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    g0 = (scene.cage.data.root_pos_w - scene.env_origins)[0]
    k0 = (scene.chock.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"ball=({float(b0[0]):+.3f},{float(b0[1]):+.3f}) "
          f"decoy=({float(d0[0]):+.3f},{float(d0[1]):+.3f}) "
          f"cage=({float(g0[0]):+.3f},{float(g0[1]):+.3f}) "
          f"yaw={math.degrees(cage_yaw()):+.1f}deg "
          f"chock=({float(k0[0]):+.3f},{float(k0[1]):+.3f})", flush=True)
    report("reset")
    assert anchor_err() < 0.010, "ball drifted off its anchor during the settle"
    assert not bool(scene._captured()[0]), "ball spawned inside the cage"
    assert not bool(scene._decoy_inside()[0]), "decoy spawned inside the cage"
    assert bool(scene.cage_seated()[0]), "cage did not settle seated"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: STAGE (teleport = transport only) ----------------------------
    # One pose write parks the cage 0.22 m short of the ball on the -x side, mouth
    # (local +x) facing the ball: the arm's push-and-pivot repositioning of a
    # receptacle it cannot lift. Mouth centre lands 0.115 m from the ball — OUTSIDE
    # the 0.09 m approach gate, so the stage earns nothing.
    bw = scene.ball.data.root_pos_w
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = bw[:, 0] - 0.22
    st[:, 1] = bw[:, 1]
    st[:, 2] = scene.env_origins[:, 2] + 0.002
    st[:, 3] = 1.0  # yaw 0: mouth faces +x = toward the ball
    scene.cage.write_root_state_to_sim(st, all_ids)
    step(60)
    xl, yl, _ = ball_loc()
    report("staged")
    assert abs(xl - 0.22) < 0.02 and abs(yl) < 0.02, "staging pose readback off"
    assert anchor_err() < 0.010, "staging disturbed the ball"
    s1 = print_score("P1 cage staged before the approach corridor")
    assert s1 <= 0.02, f"staging must earn nothing, got {s1}"
    assert s1 >= s0 - 1e-6, "score decreased across staging"

    # ---------------- phase 2: CAPTURE SLIDE (contact dynamics) -----------------------------
    # Velocity-servoed horizontal push at the cage CoM: v_des points down the
    # corridor, steered on the ball's cage-frame y so the mouth passes AROUND the
    # stationary ball. Friction feedforward ~14 N (mu_pair 0.425 x 3.2 kg); a
    # stall probe escalates feedforward+cap if measured progress dies (pod-side
    # wrench under-application); yaw-hold PD torque with FINITE-DIFFERENCE yaw
    # rate (ang-vel readback is phantom under external wrenches). Stop target
    # x_loc <= hx - 0.055 = 0.050: 10 mm deeper than the 0.040 capture margin,
    # 143 mm short of the back wall — the cage never touches the ball.
    speed = 0.10
    kv, kp_yaw, kd_yaw = 80.0, 2.0, 0.15
    ctl = {"ff": 14.0, "cap": 30.0, "yaw_prev": cage_yaw(), "x_ref": None, "i_ref": 0}
    stop_x = c.hx - 0.055
    done = False
    for i in range(1500):
        xl, yl, _ = ball_loc()
        if xl <= stop_x:
            done = True
            break
        yaw = cage_yaw()
        yaw_rate = (yaw - ctl["yaw_prev"]) / dt
        ctl["yaw_prev"] = yaw
        # desired planar velocity (cage frame ~ world frame at yaw~0, rotate anyway)
        ky = max(-0.5, min(0.5, 4.0 * yl))
        norm = math.hypot(1.0, ky)
        vdx = speed * (math.cos(yaw) * 1.0 - math.sin(yaw) * ky) / norm
        vdy = speed * (math.sin(yaw) * 1.0 + math.cos(yaw) * ky) / norm
        v = scene.cage.data.root_lin_vel_w[0]
        fx = ctl["ff"] * vdx / speed + kv * (vdx - float(v[0]))
        fy = ctl["ff"] * vdy / speed + kv * (vdy - float(v[1]))
        fmag = math.hypot(fx, fy)
        if fmag > ctl["cap"]:
            fx, fy = fx * ctl["cap"] / fmag, fy * ctl["cap"] / fmag
        tz = max(-1.5, min(1.5, -kp_yaw * yaw - kd_yaw * yaw_rate))
        apply_cage_wrench(fx, fy, tz)
        env.step(no_action)
        # stall probe every 0.5 s: escalate if the cage is not progressing
        if ctl["x_ref"] is None:
            ctl["x_ref"], ctl["i_ref"] = xl, i
        elif i - ctl["i_ref"] >= 60:
            if ctl["x_ref"] - xl < 0.005:
                ctl["ff"] = min(ctl["ff"] * 1.5, 45.0)
                ctl["cap"] = min(ctl["cap"] * 1.5, 60.0)
                print(f"[solve] slide stalled at x_loc={xl:+.3f} — force ff -> "
                      f"{ctl['ff']:.1f} N cap -> {ctl['cap']:.1f} N", flush=True)
            ctl["x_ref"], ctl["i_ref"] = xl, i
    apply_cage_wrench(0.0, 0.0, 0.0)
    step(90)
    report("captured")
    assert done, "capture slide never reached the stop target"
    assert bool(scene._capt_ever[0]), "capture latch did not set"
    assert anchor_err() < c.anchor_tol, "the slide displaced the ball off its anchor"
    assert bool(scene.cage_seated()[0]), "cage tipped or lifted during the slide"
    assert not bool(scene._decoy_inside()[0]), "the slide swallowed the decoy"
    s2 = print_score("P2 cage slid mouth-first around the anchored ball")
    assert s2 >= s1 - 1e-6, "score decreased across the capture slide"
    assert s2 >= 0.55, f"capture credit missing, got {s2}"

    # ---------------- phase 3: CHOCK (teleport = transport only, gravity seats it) ----------
    # One pose write holds the bar flat, centred on the mouth, aligned with it,
    # 4 mm above the floor JUST OUTSIDE the walls (bar inner face 6 mm clear of
    # the wall fronts, 31 mm clear of the ball surface) — the arm's
    # place-from-above of a 0.18 kg bar. Gravity does the set-down.
    cp = scene.cage.data.root_pos_w
    cq = scene.cage.data.root_quat_w
    seat_loc = torch.tensor([c.hx + c.chock_seat_dx, 0.0, 0.0],
                            device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = cp + quat_apply(cq, seat_loc)
    st[:, 2] = scene.env_origins[:, 2] + c.chock_side / 2 + 0.004
    st[:, 3:7] = cq  # bar long axis (local y) along the cage mouth line
    scene.chock.write_root_state_to_sim(st, all_ids)
    step(150)
    report("chocked")
    assert bool(scene._chock_seated()[0]), "chock did not seat across the mouth"
    assert bool(scene._chk_ever[0]), "chock latch did not set"
    assert anchor_err() < c.anchor_tol, "chock set-down disturbed the ball"
    s3 = print_score("P3 chock bar laid across the mouth")
    assert s3 >= s2 - 1e-6, "score decreased across the chock placement"

    # success should follow once everything rings down
    quiet = 0
    for _ in range(600):
        env.step(no_action)
        quiet = quiet + 1 if bool(scene.success()[0]) else 0
        if quiet >= 30:
            break
    report("settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)
    s3b = print_score("P3b success reached and quiet")
    assert s3b >= s3 - 1e-6, "score decreased across the ring-down"

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3b - 1e-6
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)
