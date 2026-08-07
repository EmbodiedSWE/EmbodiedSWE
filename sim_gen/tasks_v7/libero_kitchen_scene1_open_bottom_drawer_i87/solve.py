"""Teleport solution for RockerCabinetScene (sim_gen task
`libero_kitchen_scene1_open_bottom_drawer_i87`) — the task's legitimacy
certificate.

This solve uses NO teleports at all: every interaction is applied force through
contact dynamics. (Teleports are permitted for transport only; this task needs
none — the whole solution is one honest push.)

1. PERCEPTION: the top drawer's sampled opening d0 and the distractor poses are
   read back from the episode state — never hard-coded.
2. PUSH (applied force): a PD force servo (what a fingertip pressing the top
   drawer's protruding red lip does, force-limited) drives the TOP drawer
   inward along its prismatic axis. After ~1 cm of free travel the drawer's
   rear pusher plate meets the rocker's upper pad — a real collision — and from
   there every millimetre of bottom-drawer motion is transmitted through
   plate->pad->pivot->pad->plate contact. The wrench goes through the top
   drawer only; the judged BOTTOM drawer is never forced, never teleported,
   never touched.
3. RELEASE (nobody's hands): forces off; the friction-held, mass-balanced
   mechanism keeps the bottom drawer standing open; everything settles and
   success() turns True on the live state.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.
The whole solve is then repeated on a SECOND seed (fresh episode, no score
prints — the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_bottom_drawer_i87.solve
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
    from . import scene as scene_mod  # noqa: F401
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rocker_cabinet")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply_inverse

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    zero = torch.zeros(n, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | top={float(scene.top_open()[0]):+.4f} "
              f"bot={float(scene.bot_open()[0]):+.4f} "
              f"eng={bool(scene._engage[0])} crack={bool(scene._crack[0])} "
              f"ajar={bool(scene._ajar[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def push_top(tgt: float, *, kp: float, kd: float, clamp: float, steps: int,
                 done=None, label: str = "") -> None:
        """Applied-force push: PD along the top drawer's prismatic axis toward
        opening `tgt` (what a fingertip on the protruding lip does,
        force-limited). The wrench goes through the TOP drawer only."""
        for _ in range(steps):
            d = scene.top_open()
            v = scene.top.data.root_lin_vel_w[:, 0]
            fx = (kp * (tgt - d) - kd * v).clamp(min=-clamp, max=clamp)
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = fx
            f_b = quat_apply_inverse(scene.top.data.root_quat_w, f_w)
            scene.top.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            env.step(no_action)
            if done is not None and done():
                break
        scene.top.set_external_force_and_torque(zero, zero)
        print(f"[solve] push {label}: top={float(scene.top_open()[0]):+.4f} "
              f"bot={float(scene.bot_open()[0]):+.4f} after <= {steps} steps",
              flush=True)

    def episode(seed: int, announce: bool) -> bool:
        """One full episode on `seed`. SIM_GEN_SCORE is printed only when
        `announce` (the rubric restarts at 0 on reset, and the score stream
        must be non-decreasing)."""
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
        d0 = float(scene.d0[0])
        bw = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        pl = (scene.plate.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback (seed {seed}): top opening d0={d0:+.4f} "
              f"bowl=({float(bw[0]):+.3f},{float(bw[1]):+.3f}) "
              f"plate=({float(pl[0]):+.3f},{float(pl[1]):+.3f})", flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert abs(float(scene.top_open()[0]) - d0) < 0.006, \
            "top drawer must rest at its sampled opening"
        assert float(scene.bot_open()[0]) < 0.005, "bottom drawer must start closed"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (top drawer out, bottom sealed)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: push in until the rocker engages ---------------------
        push_top(0.0, kp=400.0, kd=40.0, clamp=35.0, steps=480,
                 done=lambda: bool(scene._engage[0]), label="to-engage")
        assert bool(scene._engage[0]), "engage latch must fire during the push"
        s = print_score("P1 top drawer pushed onto the rocker (engaged)")
        assert s >= c.w_engage - 1e-6, f"P1 score {s:.3f} below engage credit"

        # ---------------- phase 2: keep pushing — bottom drawer cracks open -------------
        push_top(0.0, kp=400.0, kd=40.0, clamp=35.0, steps=480,
                 done=lambda: bool(scene._crack[0]), label="to-crack")
        assert bool(scene._crack[0]), "crack latch must fire as the rocker transmits"
        s = print_score("P2 bottom drawer cracked >= 3 cm (via the rocker)")
        assert s >= c.w_engage + c.w_crack - 1e-6, f"P2 score {s:.3f} below crack credit"

        # ---------------- phase 3: push flush — bottom drawer fully driven out ----------
        push_top(0.0, kp=400.0, kd=40.0, clamp=35.0, steps=720,
                 done=lambda: bool(scene._ajar[0])
                 and float(scene.top_open()[0]) < 0.004, label="to-flush")
        assert bool(scene._ajar[0]), "ajar latch must fire on the full push"
        assert float(scene.top_open()[0]) < 0.010, "top drawer must end ~flush"
        s = print_score("P3 top drawer flush; bottom drawer driven out")
        assert s >= 0.65 - 1e-6, f"P3 score {s:.3f} below full latch credit"

        # ---------------- phase 4: hands off — the mechanism holds; success -------------
        won_at = None
        for i in range(360):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("hands-off")
        assert won_at is not None and bool(scene.success()[0]), \
            f"success must hold hands-off: bot={float(scene.bot_open()[0]):+.4f}"
        assert float(scene.bot_open()[0]) >= c.open_goal, "bottom must stand open"
        s = print_score(f"P4 released; settled open ({won_at} hands-off steps)")
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
