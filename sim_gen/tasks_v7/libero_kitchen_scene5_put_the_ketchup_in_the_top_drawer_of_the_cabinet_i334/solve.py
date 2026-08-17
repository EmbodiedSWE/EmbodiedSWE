"""Teleport solution for LetterboxCradleScene (sim_gen task
`libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_i334`) —
the task's legitimacy certificate.

Teleports are used for TRANSPORT ONLY: one teleport carries the standing bottle
from its sampled apron pose to a hover LYING ON ITS SIDE over the guide lane
(exactly what a pick-and-lay-down does). Everything load-bearing after that is
contact dynamics under applied forces within the declared caps:

1. PERCEPTION: the bottle's sampled staging pose is read back from the episode
   state — never hard-coded.
2. TRANSPORT (teleport): the bottle is placed at rest, lying, base toward the
   slot, hovering just above the guide lane. It drops ~15 mm and settles — that
   earns exactly the `laid` credit a real lay-down earns, nothing more
   (asserted).
3. SLIDE (real force): a velocity-capped horizontal push (|F| <= push_f_max
   = 4.5 N, a fingertip push) slides the bottle base-first through the letter
   slot onto the cradle bed until its base seats against the end plate. The
   asserted design contract guarantees this push can NEVER torque the resting
   cradle off its stop.
4. CRANK (real torque): a ramped, torque-limited servo (|tau| <= crank_tau_max
   = 1.2 N.m at the external knob) raises the cradle to ~80 deg — past the
   loaded over-centre angle (~68 deg) — then RELEASES. Gravity tips the loaded
   cradle onto its 90-deg joint stop by itself, standing the bottle upright on
   the end plate inside the vault. The goal pose is produced by the mechanism.
5. HANDS-OFF: success() turns True on the live settled state and persists
   >= 3.3 simulated seconds with zero applied wrench.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then prints `SIM_GEN_SOLVE: SUCCESS` only if success still
holds after the hold. The whole solve is then repeated on a SECOND seed (fresh
episode, no score prints — the rubric restarts at 0 on reset) to certify seed
robustness.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_i334.solve
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
    from isaaclab.utils.math import quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.letterbox_cradle_vault")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    zero13 = torch.zeros(n, 1, 3, device=device)

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def bpos() -> torch.Tensor:
        return (scene.bottle.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        p = bpos()
        print(f"[solve] {tag:10s} | b=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) upz={float(scene.bottle_up()[0, 2]):+.2f} "
              f"phi={math.degrees(float(scene.phi()[0])):+.1f}deg "
              f"laid={bool(scene._flaid[0])} inside={bool(scene._finside[0])} "
              f"ferect={float(scene._ferect[0]):.2f} riding={bool(scene.bottle_riding()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def clear_wrench() -> None:
        scene.bottle.set_external_force_and_torque(zero13, zero13)
        scene.cradle.set_external_force_and_torque(zero13, zero13)

    def episode(seed: int, announce: bool) -> bool:
        """One full episode on `seed`. SIM_GEN_SCORE is printed only when
        `announce` (the rubric restarts at 0 on reset, and the printed score
        stream must be non-decreasing)."""
        env.reset(seed=seed)
        s_prev = 0.0

        def print_score(tag: str) -> float:
            nonlocal s_prev
            s = float(scene.score()[0])
            print(f"[solve] phase boundary: {tag}", flush=True)
            if announce:
                print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
            assert s >= s_prev - 1e-6, f"score decreased: {s_prev:.4f} -> {s:.4f}"
            s_prev = s
            return s

        # ---------------- phase 0: settle, layout readback, baseline --------------------
        step(120)
        sx, sy = float(scene.stage_xy[0, 0]), float(scene.stage_xy[0, 1])
        p = bpos()
        print(f"[solve] layout readback (seed {seed}): sampled stage=({sx:+.3f},{sy:+.3f}) "
              f"settled bottle=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})",
              flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert abs(float(p[0]) - sx) < 0.01 and abs(float(p[1]) - sy) < 0.01, \
            "bottle must rest at its sampled staging pose"
        assert bool(scene.bottle_upright()[0]), "bottle must start standing"
        assert abs(math.degrees(float(scene.phi()[0]))) < 2.0, "cradle must rest on its 0-stop"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (bottle standing on the apron, cradle at rest)")
        assert s <= 0.02, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: TRANSPORT — lay the bottle in the guide lane ---------
        # A single teleport to a hover LYING over the lane (axis along x, base
        # toward the slot), 15 mm up; it drops and settles. This is the state a
        # real pick-and-lay-down produces, and it earns exactly the laid credit.
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = scene.env_origins[:, 0] + 0.14
        st[:, 1] = scene.env_origins[:, 1]
        st[:, 2] = scene.env_origins[:, 2] + c.apron_z1 + c.bottle_r + 0.015
        h = math.sqrt(0.5)  # rotate +90 deg about Y: body +z (cap) -> world +x
        st[:, 3] = h
        st[:, 5] = h
        scene.bottle.write_root_state_to_sim(st, torch.arange(n, device=device))
        step(120)
        report("laid")
        assert bool(scene.bottle_lying()[0]), "bottle must lie on its side in the lane"
        assert abs(float(bpos()[1])) < 0.02, "bottle must lie centred in the lane"
        s = print_score("P1 bottle laid on its side in the guide lane (transport + drop)")
        assert abs(s - c.w_laid) <= 0.011, f"P1 score should be the laid credit, got {s}"

        # ---------------- phase 2: SLIDE — real force pushes it through the slot --------
        # Velocity-capped horizontal push at the bottle's CoM, world -x, rotated
        # into the body frame each step (external wrenches are body-frame).
        seat_x = c.hinge_x + c.foot_x1 + c.bottle_l / 2  # centre x when seated
        v_des = -0.15
        stall, x_last = 0, float(bpos()[0])
        for i in range(900):
            x = float(bpos()[0])
            if x <= seat_x + 0.004:
                break
            if i % 30 == 29:
                stall = stall + 1 if abs(x - x_last) < 0.001 else 0
                x_last = x
                if stall >= 3:
                    break
            vx = scene.bottle.data.root_lin_vel_w[:, 0]
            fw = torch.zeros(n, 3, device=device)
            fw[:, 0] = (30.0 * (v_des - vx)).clamp(-c.push_f_max, c.push_f_max)
            fb = quat_apply_inverse(scene.bottle.data.root_quat_w, fw)
            scene.bottle.set_external_force_and_torque(fb.reshape(n, 1, 3), zero13)
            env.step(no_action)
        clear_wrench()
        step(90)
        report("seated")
        p = bpos()
        assert bool(scene.bottle_inside()[0]), \
            f"bottle must be fully through the slot (x={float(p[0]):+.3f})"
        assert bool(scene.bottle_lying()[0]), "bottle must still be lying on the bed"
        assert bool(scene.bottle_riding()[0]), "bottle must ride the cradle bed"
        assert abs(float(p[0]) - seat_x) < 0.015, \
            f"bottle base must seat at the end plate (x={float(p[0]):+.3f} vs {seat_x:+.3f})"
        assert abs(math.degrees(float(scene.phi()[0]))) < 3.0, \
            "the push must not have unseated the cradle"
        s = print_score("P2 bottle slid base-first through the slot, seated on the cradle")
        assert s >= c.w_laid + c.w_inside - 1e-6, f"P2 score {s:.3f} below laid+inside"

        # ---------------- phase 3: CRANK — torque servo past over-centre, release -------
        phi_tgt_final = math.radians(84.0)  # well past loaded over-centre (~68 deg)
        ramp_steps = 480
        for i in range(ramp_steps + 90):
            tgt = phi_tgt_final * min(1.0, (i + 1) / ramp_steps)
            phi = scene.phi()
            w = scene.cradle.data.root_ang_vel_w[:, 1]
            # phi = -theta_y: positive tau_y drives phi DOWN, so negate the servo.
            tau = -(2.0 * (tgt - phi) - 0.3 * (-w)).clamp(-c.crank_tau_max, c.crank_tau_max)
            tw = torch.zeros(n, 3, device=device)
            tw[:, 1] = tau
            scene.cradle.set_external_force_and_torque(zero13, tw.reshape(n, 1, 3))
            env.step(no_action)
        phi_rel = math.degrees(float(scene.phi()[0]))
        clear_wrench()
        report("released")
        assert phi_rel >= 72.0, f"must release past over-centre, got {phi_rel:.1f} deg"

        won_at = None
        for i in range(600):
            env.step(no_action)
            if bool(scene.success()[0]) and \
                    math.degrees(float(scene.phi()[0])) >= 86.0:
                won_at = i
                break
        report("erected")
        assert won_at is not None and bool(scene.success()[0]), \
            (f"success must arrive hands-off after release: "
             f"phi={math.degrees(float(scene.phi()[0])):+.1f}deg "
             f"upz={float(scene.bottle_up()[0, 2]):+.2f}")
        assert math.degrees(float(scene.phi()[0])) >= 86.0, "cradle must land on its 90-stop"
        assert bool(scene.bottle_upright()[0]) and bool(scene.bottle_inside()[0])
        s = print_score(f"P3 cradle tipped over-centre onto its stop; bottle erected "
                        f"({won_at} hands-off steps)")
        assert s >= 1.0 - 1e-6, "success must score 1.0"

        # ---------------- persistence (>= 3 simulated seconds, hands-off) ---------------
        hold = True
        for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
            step(40)
            hold = hold and bool(scene.success()[0])
        report("persist")
        s_final = print_score("P-final persistence 3.3 s")
        ok = hold and bool(scene.success()[0]) and s_final >= 1.0 - 1e-6
        print(f"[solve] episode seed={seed}: {'OK' if ok else 'FAILED'}", flush=True)
        return ok

    ok = episode(args.seed, announce=True)
    ok = episode(args.seed + 1, announce=False) and ok

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
