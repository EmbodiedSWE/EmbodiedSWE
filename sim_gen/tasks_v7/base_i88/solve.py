"""Teleport solution for BallastLiftScene (sim_gen task `base_i88`) — the task's
legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction goes through
contact/constraint dynamics:
  1. BALLAST (transport + gravity/contact): each blue cube is teleported to a point in
     FREE SPACE just above the hopper chimney's open mouth (computed from the beam's
     CURRENT pose — the mouth moves as the beam tips), released with zero velocity,
     and DROPPED. The insertion is made by gravity and wall contact — the chimney
     guides the falling cube to the stack. Nothing is ever spawned seated.
  2. LIFT (pure dynamics): the see-saw tips because the accumulated ballast
     out-moments the ball — a real hinge-constraint moment balance. No body is
     touched while the beam swings the tray up to the raised stop.
  3. TRANSFER (pure dynamics): the red ball — NEVER teleported at any point — is
     pushed along the hinge-axis direction (level at every beam angle) by a
     velocity-regulated horizontal force: over the tray's low lip, across the 10 mm
     gap, into the shelf dock, where it is released and settles against the far wall
     by contact alone.

The force servo uses a HIGH gain with a hard clamp (f = gain*(v_des - v), clamped to
[0, push_f_max]): the stall force at v = 0 is the clamp (4 N), comfortably above the
~2.3 N the 12 mm lip demands, while the regulated cruise speed stays low — a constant
4 N jackhammer would launch the ball over the dock walls (a low-gain servo would stall
at the lip: stall force = gain*v_des must exceed the obstacle force).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.base_i88.solve --headless [--seed N]
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
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_lift")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero3 = torch.zeros(3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def wrench(body, f3: torch.Tensor) -> None:
        """Apply a WORLD force to `body`, expressed in its CURRENT link frame
        (`is_global=True` silently drops torques on this stack — transform manually,
        the house convention). Re-set every step while pushing (`control_period == 1`
        for the null robot, so each env.step is one physics substep)."""
        from isaaclab.utils.math import quat_apply_inverse

        q = body.data.root_link_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f3.view(1, 3).expand(n, 3)).unsqueeze(1),
            torch.zeros(n, 1, 3, device=device),
            env_ids=all_ids)

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        tilt = float(scene.beam_tilt()[0])
        hop = scene.cubes_in_hopper()[0]
        b = rel(scene.ball)
        print(f"[solve] {tag:16s} | tilt={tilt:+.1f}deg"
              f" in_hopper=({bool(hop[0])},{bool(hop[1])},{bool(hop[2])})"
              f" raised={bool(scene.raised()[0])}"
              f" ball=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f})"
              f" in_dock={bool(scene.ball_in_dock()[0])}"
              f" success={bool(scene.success()[0])}"
              f" score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def beam_settled(max_steps: int = 720) -> None:
        """Hands-off wait until the beam stops swinging (and the ball stops rolling
        with it)."""
        for _ in range(max_steps // 30):
            step(30)
            if float(scene.beam.data.root_ang_vel_w[0].norm()) < 0.05 \
                    and float(scene.ball.data.root_lin_vel_w[0].norm()) < 0.05:
                break

    # ---------------- contact primitives ---------------------------------------------------
    def drop_cube(i: int) -> None:
        """TRANSPORT cube `i` to free space just above the hopper mouth (computed from
        the beam's CURRENT pose — the mouth moves as the beam tips), release with zero
        velocity, and let gravity + chimney-wall contact make the insertion."""
        body = scene.cubes[i]
        hop_mid_y = (c.hop_y[0] + c.hop_y[1]) / 2  # -0.210, beam local
        local = torch.tensor([0.0, hop_mid_y, c.hop_top + 0.055], device=device)
        mouth = scene.beam.data.root_pos_w[0] + quat_apply(
            scene.beam.data.root_quat_w[0], local)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = mouth  # already world (env origin included via beam pos)
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, all_ids)
        # free fall down the chimney + stack settling (+ any beam swing it triggers)
        for _ in range(20):
            step(30)
            if float(body.data.root_lin_vel_w[0].norm()) < 0.05 \
                    and float(scene.beam.data.root_ang_vel_w[0].norm()) < 0.05:
                break
        assert bool(scene.cubes_in_hopper()[0, i]), \
            f"cube {i} missed the hopper (rel={rel(body).tolist()})"

    def push_ball() -> None:
        """Velocity-regulated +x push on the ball (hinge-axis direction — level at
        every beam angle): over the lip, across the gap, into the dock. The ball is
        never teleported; the force is re-set every physics step in the ball's
        current frame."""
        f3 = torch.zeros(3, device=device)
        x0 = float(rel(scene.ball)[0])
        for _ in range(1500):
            rx = float(rel(scene.ball)[0])
            if rx > 0.585:  # deep enough — coast the rest of the way
                break
            vx = float(scene.ball.data.root_lin_vel_w[0, 0])
            f = 60.0 * (0.11 - vx)  # stall force = clamp (4 N) > 2.3 N lip demand
            f3[0] = max(min(f, 4.0), 0.0)
            wrench(scene.ball, f3)
            env.step(no_action)
        wrench(scene.ball, zero3)
        rx = float(rel(scene.ball)[0])
        assert rx - x0 > 0.05, f"push did not move the ball (x {x0:.3f}->{rx:.3f})"
        assert rx > c.cross_x, f"ball did not cross onto the shelf (x={rx:.3f})"

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(90)  # beam settles onto its lower stop; ball drops the last mm into the tray
    b = rel(scene.ball)
    print(f"[solve] layout readback (seed {args.seed}): "
          f"tilt={float(scene.beam_tilt()[0]):+.1f}deg "
          f"ball=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):.3f}) "
          + " ".join(
              f"cube{i}=({float(rel(scene.cubes[i])[0]):+.3f},"
              f"{float(rel(scene.cubes[i])[1]):+.3f})" for i in range(c.n_cubes)),
          flush=True)
    report("reset")
    assert bool(scene.lowered()[0]), "beam must start cargo-end-down at the stop"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, "null credit at reset — rubric leak"

    # ---------------- phase 1: ballast the hopper (drop + gravity/contact) -----------------
    scores = [s0]
    for i in range(c.n_cubes):
        drop_cube(i)
        report(f"cube {i} in")
        si = print_score(f"P1{'abc'[i]} cube {i} dropped into the hopper (gravity+contact)")
        assert si >= scores[-1] - 1e-6
        scores.append(si)

    # the accumulated ballast out-moments the ball: hands-off while the see-saw tips
    beam_settled()
    report("beam tipped")
    assert bool(scene.raised()[0]), \
        f"beam did not tip with full ballast (tilt={float(scene.beam_tilt()[0]):+.1f})"
    s1 = print_score("P1d see-saw tipped by ballast moment (pure dynamics)")
    assert s1 >= scores[-1] - 1e-6

    # ---------------- phase 2: push the ball into the dock (pure dynamics) -----------------
    push_ball()
    report("ball pushed")
    s2 = print_score("P2 ball pushed over the lip into the dock (contact dynamics)")
    assert s2 >= s1 - 1e-6

    # ---------------- phase 3: settle to success -------------------------------------------
    for _ in range(16):  # up to 4 s of hands-off settling
        if bool(scene.success()[0]):
            break
        step(30)
    report("settled")
    s3 = print_score("P3 all settled")
    assert s3 >= s2 - 1e-6, "score decreased across settling"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after ballast+tip+push+settle)",
              flush=True)
        threading.Timer(10.0, lambda: os._exit(1)).start()
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) -------
    hold, flickers = True, 0
    for i in range(400):  # 400 substeps = 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:  # diagnose exactly which predicate broke
                bb = scene.ball
                print(f"[solve] persist flicker @step {i}: "
                      f"in_dock={bool(scene.ball_in_dock()[0])} "
                      f"settled={bool(scene.settled()[0])} "
                      f"lin={float(bb.data.root_lin_vel_w[0].norm()):.4f}", flush=True)
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
