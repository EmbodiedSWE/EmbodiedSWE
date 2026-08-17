"""Teleport solution for DieRollScene (sim_gen task `slide_block_to_target_i135`) —
the task's legitimacy certificate.

The task's load-bearing interaction is REORIENTATION UNDER CONTACT: a cube flat on
the floor changes its up face only by pivoting over a bottom edge through the balance
point and slapping down on the next face. How this solution executes it:

1. TRANSPORT (teleport): one pose write stages the die near the disc — orientation
   PRESERVED EXACTLY (a teleport that rotated the die would bypass the load-bearing
   reorientation, so no teleport in this file ever changes the quaternion), position
   chosen so the planned roll sequence lands the die centre on the disc, and
   verifiably OUTSIDE the scoring zone (asserted by readback).
2. QUARTER-ROLLS (contact dynamics — never teleported): each roll applies a
   velocity-capped torque about the horizontal edge axis (the applied-wrench emulation
   of the arm pushing high on the die's face), pivoting the die on its bottom edge —
   the balance point, the fall onto the next face, the landing slap and the settling
   rock are all physics. Torque is CUT at 55 deg of tilt; gravity completes the roll
   hands-off, and each roll is verified by orientation readback (the old up-face must
   point along the roll direction). Plan: blue sideways -> one roll away from blue;
   blue down -> two rolls in the same direction. Between rolls of a multi-roll plan
   the die may be re-staged (pure translation, orientation preserved, outside the
   zone) to absorb landing skid.
   Torque-frame note: the applied torque axis IS the axis the die rotates about, so
   the command is invariant under the pod-dependent rotation-since-reset wrench drag;
   a stall probe still flips to body-frame encoding if tilt never starts.
3. If the final roll lands blue-up but off-disc, recovery re-stages 4 edge lengths
   out (blue not yet up during the intermediate rolls) and rolls 4x in one direction
   (blue: up -> leading -> down -> trailing -> up), landing on the disc.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.4 simulated seconds after success()
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
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

FACE_NAMES = scene_mod.FACE_NAMES

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.die_roll")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)
    s = c.size

    # Seed AFTER build (the EnvCfg.build reseed trap); print the scene's own readouts.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def die_xy() -> torch.Tensor:
        return self_pos()[0:2]

    def self_pos() -> torch.Tensor:
        return (scene.die.data.root_pos_w - scene.env_origins)[0]

    def disc_xy() -> torch.Tensor:
        return (scene.disc.data.root_pos_w - scene.env_origins)[0, 0:2]

    def report(tag: str) -> None:
        p = self_pos()
        b = scene.blue_normal_w()[0]
        print(f"[solve] {tag:12s} | die=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) up_face={FACE_NAMES[int(scene.up_face_idx()[0])]} "
              f"blue_z={float(b[2]):+.3f} d_disc={float(scene.dist_to_disc()[0]):.3f} "
              f"in_zone={bool(scene.in_zone()[0])} blue_up={bool(scene.blue_up()[0])} "
              f"blue_ever={bool(scene._blue_ever[0])} app={float(scene._app_max[0]):.3f} "
              f"zone_ever={bool(scene._zone_ever[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    last_score = [0.0]

    def print_score(tag: str) -> float:
        sc = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {sc:.4f}", flush=True)
        assert sc >= last_score[0] - 1e-6, \
            f"score decreased across {tag}: {last_score[0]:.4f} -> {sc:.4f}"
        last_score[0] = sc
        return sc

    def settle_wait(max_steps: int = 700) -> bool:
        quiet = 0
        for _ in range(max_steps):
            env.step(no_action)
            still = (float(scene.die.data.root_lin_vel_w[0].norm()) < 0.035
                     and float(scene.die.data.root_ang_vel_w[0].norm()) < 0.40)
            quiet = quiet + 1 if still else 0
            if quiet >= 25:
                return True
        return False

    def horiz_face_dirs() -> list[torch.Tensor]:
        """The die's 4 world-horizontal face normals (valid roll directions:
        perpendicular to a bottom edge), z zeroed and renormalized."""
        q = scene.die.data.root_quat_w
        out = []
        for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            w = quat_apply(q, torch.tensor(v, dtype=torch.float32,
                                           device=device).view(1, 3))[0]
            if abs(float(w[2])) < 0.5:
                w = torch.tensor([float(w[0]), float(w[1]), 0.0], device=device)
                out.append(w / w.norm())
        return out

    def toward_disc_dir() -> torch.Tensor:
        v = disc_xy() - die_xy()
        best, best_dot = None, -2.0
        for d in horiz_face_dirs():
            dd = float(d[0] * v[0] + d[1] * v[1])
            if dd > best_dot:
                best, best_dot = d, dd
        return best

    def plan() -> list[torch.Tensor]:
        """Roll list from the CURRENT orientation: [] if blue is up; [d] if blue is
        sideways (d = away from blue: the trailing face comes up); [d, d] if blue is
        straight down (two same-direction rolls bring the bottom face up)."""
        b = scene.blue_normal_w()[0]
        if bool(scene.blue_up()[0]):
            return []
        if float(b[2]) < -0.7:
            d = toward_disc_dir()
            return [d, d]
        d = torch.tensor([-float(b[0]), -float(b[1]), 0.0], device=device)
        return [d / d.norm()]

    def stage(k_remaining: int, d_hat: torch.Tensor, tag: str) -> None:
        """Teleport (TRANSPORT ONLY): pure translation to the point from which
        `k_remaining` rolls along d_hat land the die centre on the disc centre.
        Orientation and the reorientation problem are untouched; the staging point is
        asserted OUTSIDE the scoring zone by readback."""
        q = scene.die.data.root_quat_w[0].clone()
        target = disc_xy() - d_hat[0:2] * (k_remaining * s)
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1] = float(target[0]), float(target[1])
        st[:, 2] = s / 2 + 0.008
        st[:, 3:7] = q
        st[:, 0:3] += scene.env_origins
        scene.die.write_root_state_to_sim(st, all_ids)
        settle_wait(240)
        d_now = float(scene.dist_to_disc()[0])
        print(f"[solve] staged ({tag}): d_disc={d_now:.3f} (zone_r={c.zone_r})",
              flush=True)
        assert d_now > c.zone_r + 0.004, \
            f"staging point is not outside the scoring zone (d={d_now:.3f})"
        assert not bool(scene.in_zone()[0]), "staged die already reads in-zone"

    def do_roll(d_hat: torch.Tensor, tag: str) -> bool:
        """One quarter-roll toward d_hat through contact: velocity-capped torque about
        the horizontal edge axis until 55 deg of tilt, then hands-off fall + settle.
        Verified by readback: the old up-face must end pointing along d_hat."""
        a_hat = torch.tensor([-float(d_hat[1]), float(d_hat[0]), 0.0], device=device)
        ez = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3)
        q0 = scene.die.data.root_quat_w.clone()
        n_local = quat_apply_inverse(q0, ez)[0]  # body dir currently pointing up
        tau, w_cap = 0.16, 1.2
        mode_body = False
        theta_mark, mark_i = 0.0, 0
        tipped = False
        for i in range(1400):
            q = scene.die.data.root_quat_w
            up_now = quat_apply(q, n_local.view(1, 3))[0]
            theta = math.degrees(math.acos(max(-1.0, min(1.0, float(up_now[2])))))
            if theta > 55.0:
                tipped = True
                break
            wv = scene.die.data.root_ang_vel_w[0]
            w_axis = float(wv[0] * a_hat[0] + wv[1] * a_hat[1])
            t_mag = tau if w_axis < w_cap else 0.0
            t_world = (a_hat * t_mag).view(n, 1, 3)
            if mode_body:
                t_cmd = quat_apply_inverse(q, t_world.view(n, 3)).view(n, 1, 3)
                scene.die.set_external_force_and_torque(
                    zero_w, t_cmd.contiguous(), env_ids=all_ids)
            else:
                scene.die.set_external_force_and_torque(
                    zero_w, t_world.contiguous(), env_ids=all_ids, is_global=True)
            env.step(no_action)
            if theta > theta_mark + 2.0:
                theta_mark, mark_i = theta, i
            elif i - mark_i > 240:
                if tau < 0.42:
                    tau += 0.07
                    print(f"[solve] {tag}: stall at tilt {theta:.1f} deg -> "
                          f"tau={tau:.2f} N.m", flush=True)
                elif not mode_body and theta < 8.0:
                    mode_body = True
                    print(f"[solve] {tag}: still stalled -> body-frame torque "
                          f"encoding", flush=True)
                mark_i = i
        scene.die.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        if not tipped:
            print(f"[solve] {tag}: torque never carried the die past 55 deg",
                  flush=True)
            return False
        settle_wait(600)
        rolled = quat_apply(scene.die.data.root_quat_w, n_local.view(1, 3))[0]
        fwd = float(rolled[0] * d_hat[0] + rolled[1] * d_hat[1])
        print(f"[solve] {tag}: done, old-up now along-roll dot={fwd:+.2f}, "
              f"up_face={FACE_NAMES[int(scene.up_face_idx()[0])]}", flush=True)
        return fwd > 0.7

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    p0 = self_pos()
    dxy = disc_xy()
    face0 = FACE_NAMES[int(scene.up_face_idx()[0])]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"die=({float(p0[0]):+.3f},{float(p0[1]):+.3f}) up_face={face0} "
          f"disc=({float(dxy[0]):+.3f},{float(dxy[1]):+.3f})", flush=True)
    report("reset")
    assert not bool(scene.blue_up()[0]), "die spawned blue-up (randomization broken)"
    assert int(scene.up_face_idx()[0]) != 4, "up-face readback says blue at spawn"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phases 1-2: stage (teleport) + quarter-rolls (contact) ----------------
    solved = False
    staged_once = False
    for attempt in range(3):
        rolls = plan()
        if not rolls:
            if bool(scene.in_zone()[0]):
                solved = True
                break
            # blue up but off-disc: 4-roll recovery (blue leaves "up" during rolls
            # 1-3, so intermediate re-staging stays legitimate transport)
            d = toward_disc_dir()
            rolls = [d, d, d, d]
            print("[solve] recovery: blue up but off-disc -> 4-roll cycle", flush=True)
        stage(len(rolls), rolls[0], f"attempt{attempt}, {len(rolls)} roll(s)")
        if not staged_once:
            staged_once = True
            report("staged")
            print_score("P1 die transported to the staging point (outside the zone)")
        ok = True
        for j, d in enumerate(rolls):
            if j > 0:
                # absorb landing skid: if the remaining rolls would miss the disc,
                # re-stage (pure translation, blue not yet up, outside the zone)
                rem = len(rolls) - j
                landing = die_xy() + d[0:2] * (rem * s)
                err = float((landing - disc_xy()).norm())
                if err > 0.04:
                    print(f"[solve] mid-plan correction: projected miss {err:.3f} m",
                          flush=True)
                    stage(rem, d, f"correction roll {j}")
            if not do_roll(d, f"attempt{attempt}-roll{j}"):
                ok = False
                break
        report(f"attempt{attempt}")
        if ok and bool(scene.blue_up()[0]) and bool(scene.in_zone()[0]):
            solved = True
            break
    if not solved:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (never reached blue-up on the disc)", flush=True)
        os._exit(1)

    # ---------------- phase 3: hands-off settle + success -----------------------------------
    settle_wait(600)
    report("settled")
    s2 = print_score("P2 rolled blue-up onto the disc, settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)
    print_score("P3 success verified")

    # ---------------- phase 4: persistence (>= 3.4 simulated seconds, no intervention) ------
    hold = True
    for _ in range(10):  # 10 x 42 steps = 420 substeps = 3.5 s at 120 Hz
        step(42)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.5 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s2 - 1e-6
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
    main()
