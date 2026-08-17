"""Teleport solution for VaultDepositScene (sim_gen task `obstacle_i284`) — the
task's legitimacy certificate.

Teleports are used for TRANSPORT ONLY (one: carrying the red cube through free
space to above the exposed bay mouth). Every load-bearing interaction goes
through contact dynamics:

1. PERCEPTION: the housing pose (xy jitter + 0/180-flip yaw), the sampled bolt
   lock position, the sampled lid closed position and the red/blue slot
   assignment are all READ BACK from the live state — never hard-coded.
2. UNLOCK (applied force): a force-limited velocity servo (what a fingertip
   hooked on the brass knob does) pulls the BOLT along its channel, housing
   +y, until the blade has retracted clear of the lid corridor. The wrench
   goes through the bolt only.
3. OPEN (applied force): the same kind of servo pushes the LID along its rails,
   housing +x, until the bay mouth is fully exposed. Contact with the rails,
   fences and caps carries the lid the whole way.
4. DEPOSIT (teleport = transport, then gravity): the red cube is teleported to
   free air above the open mouth and RELEASED; it falls, lands on the grippy
   bay floor through real contact, and settles. Nothing is placed "into" the
   goal pose — gravity and contact finish the job.
5. CLOSE (applied force): the servo drives the lid back, housing -x, until it
   re-enters its closed window and coasts onto the closed stop.
6. RELEASE: forces off; everything settles; success() turns True on the live
   state (cube settled in the bay, lid settled closed, decoy outside).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing — the
rubric latches), holds HANDS-OFF >= 3 simulated seconds after success() first
turns True, and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds. The
whole solve is then repeated on a SECOND seed (fresh episode, no score prints —
the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.obstacle_i284.solve --headless [--seed N]
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.vault_deposit")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    zero = torch.zeros(n, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        print(f"[solve] {tag:12s} | bolt={float(scene.bolt_travel()[0]):+.4f} "
              f"lid={float(scene.lid_open()[0]):+.4f} "
              f"cube_in={bool(scene.in_bay(scene.cube)[0])} "
              f"decoy_in={bool(scene.in_bay(scene.decoy)[0])} "
              f"lid_closed={bool(scene.lid_closed()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def servo(body, axis_w: torch.Tensor, v_des: float, *, kv: float, clamp: float,
              steps: int, done, label: str) -> None:
        """Applied-force velocity servo along a world axis (what a finger hooked on
        the knob / pressed on the handle does, force-limited). The wrench goes
        through `body` only, re-set every substep, zeroed on exit."""
        for _ in range(steps):
            v = (body.data.root_lin_vel_w * axis_w).sum(dim=-1)
            f = (kv * (v_des - v)).clamp(min=-clamp, max=clamp)
            f_w = axis_w * f.unsqueeze(-1)
            f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            env.step(no_action)
            if done():
                break
        body.set_external_force_and_torque(zero, zero)
        print(f"[solve] servo {label}: bolt={float(scene.bolt_travel()[0]):+.4f} "
              f"lid={float(scene.lid_open()[0]):+.4f} after <= {steps} steps", flush=True)

    def episode(seed: int, announce: bool) -> bool:
        """One full episode on `seed`. SIM_GEN_SCORE is printed only when `announce`
        (the rubric restarts at 0 on reset, and the score stream must be
        non-decreasing)."""
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

        # ---------------- phase 0: settle, layout readback, baseline ----------------
        step(120)
        h_pos = scene.housing.data.root_pos_w.clone()
        h_quat = scene.housing.data.root_quat_w.clone()
        ex_w = quat_apply(h_quat, torch.tensor([[1.0, 0.0, 0.0]], device=device).repeat(n, 1))
        ey_w = quat_apply(h_quat, torch.tensor([[0.0, 1.0, 0.0]], device=device).repeat(n, 1))
        yaw = math.degrees(math.atan2(float(ex_w[0, 1]), float(ex_w[0, 0])))
        red_l = scene._local(scene.cube)[0]
        blue_l = scene._local(scene.decoy)[0]
        print(f"[solve] layout readback (seed {seed}): housing=({float(h_pos[0, 0]):+.3f},"
              f"{float(h_pos[0, 1]):+.3f}) yaw={yaw:+.1f}deg "
              f"bolt_ref={float(scene._bolt_ref[0]):+.4f} "
              f"lid_ref={float(scene._lid_ref[0]):+.4f} "
              f"red=({float(red_l[0]):+.3f},{float(red_l[1]):+.3f}) "
              f"blue=({float(blue_l[0]):+.3f},{float(blue_l[1]):+.3f})", flush=True)
        report("reset")
        assert torch.isfinite(scene.cube.data.root_pos_w).all(), "NaN/inf after settle"
        assert float(scene.bolt_travel()[0]) < 0.005, "bolt must start locked"
        assert float(scene.lid_open()[0]) < 0.005, "lid must start closed"
        assert bool(scene.lid_closed()[0]), "lid must start in its closed window"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (vault locked, cubes on the floor)")
        assert s <= 0.02, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: pull the bolt clear (applied force) ---------------
        servo(scene.bolt, ey_w, 0.10, kv=5.0, clamp=2.0, steps=600,
              done=lambda: float(scene.bolt_travel()[0]) >= c.bolt_clear + 0.002,
              label="bolt-retract")
        step(60)  # coast onto the far stop / settle
        bt = float(scene.bolt_travel()[0])
        assert bt >= c.bolt_clear, f"blade must be clear: travel {bt:.4f} < {c.bolt_clear}"
        s = print_score("P1 bolt retracted (blade clear of the lid corridor)")
        assert s >= c.w_bolt - 1e-6, f"P1 score {s:.3f} below bolt credit"

        # ---------------- phase 2: slide the lid open (applied force) ----------------
        servo(scene.lid, ex_w, 0.15, kv=20.0, clamp=6.0, steps=900,
              done=lambda: float(scene.lid_open()[0]) >= c.lid_open_need + 0.006,
              label="lid-open")
        step(60)
        lo = float(scene.lid_open()[0])
        assert lo >= c.lid_open_need, f"mouth must be exposed: open {lo:.4f}"
        s = print_score("P2 lid slid open (bay mouth exposed)")
        assert s >= c.w_bolt + c.w_lid - 1e-6, f"P2 score {s:.3f} below lid credit"

        # -------- phase 3: deposit (teleport = transport only, then gravity) ---------
        drop_l = torch.tensor([[0.0, 0.0, 0.22]], device=device).repeat(n, 1)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = h_pos + quat_apply(h_quat, drop_l)
        st[:, 3] = 1.0
        scene.cube.write_root_state_to_sim(st)
        step(240)  # free fall through the mouth + settle on the grippy bay floor
        assert bool(scene.in_bay(scene.cube)[0]), "red cube must rest inside the bay"
        assert bool(scene.settled(scene.cube, c.cube_settle_lin, c.cube_settle_ang)[0])
        assert not bool(scene.in_bay(scene.decoy)[0]), "decoy must stay outside"
        s = print_score("P3 red cube dropped through the mouth, settled on the bay floor")
        assert s >= c.w_bolt + c.w_lid + c.w_deposit - 1e-6, f"P3 score {s:.3f} low"

        # ---------------- phase 4: slide the lid shut (applied force) ----------------
        servo(scene.lid, ex_w, -0.12, kv=20.0, clamp=6.0, steps=900,
              done=lambda: float(scene._local(scene.lid)[0, 0]) <= 0.006,
              label="lid-close")

        # ---------------- phase 5: hands off — settle into success -------------------
        won_at = None
        for i in range(360):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("hands-off")
        assert won_at is not None and bool(scene.success()[0]), \
            f"success must hold hands-off: lid_x={float(scene._local(scene.lid)[0, 0]):+.4f}"
        s = print_score(f"P4 lid closed; settled ({won_at} hands-off steps)")
        assert s >= 1.0 - 1e-6, "success must score 1.0"

        # ------------- persistence (>= 3 simulated seconds, hands-off) ---------------
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
