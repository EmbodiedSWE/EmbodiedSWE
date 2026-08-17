"""Teleport solution for BayonetDrawerScene (sim_gen task
`libero_kitchen_scene1_open_bottom_drawer_i88`) — the task's legitimacy
certificate.

Teleports are permitted for TRANSPORT only; this solve uses exactly ONE: after
reading the key's sampled floor pose back from the live state, the key is
teleported to a staging pose hovering in front of the drawer's slot (that is
carrying the tool across the room — what a gripper transport does). EVERY
load-bearing interaction after that is applied force/torque through contact
dynamics:

1. INSERT (applied force): a 6-DOF wrench servo (what a hand holding the knob
   does, force- and torque-limited) carries the key straight through the slot —
   the crossbar passes the 16 mm opening FLAT, with mm clearances, a real
   guided-insertion contact problem.
2. TWIST (applied torque): the same servo ramps a roll about the world x-axis;
   the crossbar rotates INSIDE the drawer to vertical — the bayonet interlock.
   From here the 46 mm bar cannot pass the 16 mm slot: pure geometry.
3. PULL (applied force): the servo target retreats +x; the crossbar bears on
   the slab's inner face — a real unilateral contact — and DRAGS the drawer
   open past the 8 cm goal. The drawer is never forced directly, never
   teleported, never touched anywhere but through this coupling.
4. RELEASE (nobody's hands): wrench off; the friction-held drawer stands open
   with the key dangling in the slot; everything settles and success() turns
   True on the live state.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.
The whole solve is then repeated on a SECOND seed (fresh episode, no score
prints — the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_bottom_drawer_i88.solve
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- key wrench-servo gains (what a hand on the knob supplies, limited) ----------------------
KP_P = 150.0     # N/m position stiffness (0.1 kg key: w ~ 39 rad/s, dt*w ~ 0.32)
KD_P = 6.0       # N.s/m velocity damping (zeta ~ 0.8 with authored lin_damp 3)
F_CLAMP = 20.0   # N per-axis force clamp
KR = 0.02        # N.m/rad orientation stiffness (I_roll ~ 1e-5: w ~ 43 rad/s)
T_CLAMP = 0.06   # N.m per-axis torque clamp (authored ang_damp 3 damps the mode)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bayonet_drawer")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_conjugate, quat_mul

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    zero = torch.zeros(n, 1, 3, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        p = scene.cross_local()[0]
        print(f"[solve] {tag:12s} | open={float(scene.drawer_open()[0]):+.4f} "
              f"cross=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"tilt={float(scene.cross_tilt()[0]):.3f} "
              f"keyed={bool(scene.keyed()[0])} locked={bool(scene.locked()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def key_quat(theta: float) -> torch.Tensor:
        """Target orientation: yaw pi (key +x faces the cabinet) composed with a
        roll `theta` about the WORLD x-axis: (w,x,y,z) = (0, 0, -sin, cos)."""
        q = torch.zeros(n, 4, device=device)
        q[:, 2] = -math.sin(theta / 2.0)
        q[:, 3] = math.cos(theta / 2.0)
        return q

    def key_servo_step(p_t: torch.Tensor, q_t: torch.Tensor) -> None:
        """One control tick: PD force to local target p_t (+gravity feedforward)
        and kp-only torque to q_t, both encoded into the reset-frame convention
        (quat_apply_inverse with the LIVE quat) before set_external_force_and_torque."""
        p = scene.key.data.root_pos_w - scene.env_origins
        v = scene.key.data.root_lin_vel_w
        q = scene.key.data.root_quat_w
        f_w = KP_P * (p_t - p) - KD_P * v
        f_w[:, 2] += c.key_mass * 9.81
        f_w = f_w.clamp(min=-F_CLAMP, max=F_CLAMP)
        q_err = quat_mul(q_t, quat_conjugate(q))
        aa = axis_angle_from_quat(q_err)
        t_w = (KR * aa).clamp(min=-T_CLAMP, max=T_CLAMP)
        f_b = quat_apply_inverse(q, f_w)
        t_b = quat_apply_inverse(q, t_w)
        scene.key.set_external_force_and_torque(f_b.reshape(n, 1, 3), t_b.reshape(n, 1, 3))

    def ramp(steps: int, tail: int, p_from, p_to, th_from: float, th_to: float,
             done=None, label: str = "") -> None:
        """Servo through a linear target ramp over `steps`, then hold up to
        `tail` more steps (early-out on `done`). Forces stay ON at exit (the
        next phase keeps holding); only RELEASE clears them."""
        p0 = torch.tensor(p_from, device=device).expand(n, 3)
        p1 = torch.tensor(p_to, device=device).expand(n, 3)
        for i in range(steps + tail):
            a = min(1.0, i / max(steps, 1))
            key_servo_step(p0 + (p1 - p0) * a, key_quat(th_from + (th_to - th_from) * a))
            env.step(no_action)
            if done is not None and a >= 1.0 and done():
                break
        p = scene.cross_local()[0]
        print(f"[solve] ramp {label}: open={float(scene.drawer_open()[0]):+.4f} "
              f"cross_x={float(p[0]):+.4f} tilt={float(scene.cross_tilt()[0]):.3f} "
              f"after <= {steps}+{tail} steps", flush=True)

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
        k0 = scene.k0[0]
        kp = (scene.key.data.root_pos_w - scene.env_origins)[0]
        bw = (scene.bowl.data.root_pos_w - scene.env_origins)[0]
        pl = (scene.plate.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback (seed {seed}): key sampled "
              f"({float(k0[0]):+.3f},{float(k0[1]):+.3f},yaw {float(k0[2]):+.2f}) "
              f"settled ({float(kp[0]):+.3f},{float(kp[1]):+.3f}) "
              f"bowl=({float(bw[0]):+.3f},{float(bw[1]):+.3f}) "
              f"plate=({float(pl[0]):+.3f},{float(pl[1]):+.3f})", flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert (kp[:2] - k0[:2]).norm() < 0.03, "key must settle near its sampled pose"
        assert float(scene.drawer_open()[0]) < 0.005, "drawer must start closed"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (key on the floor, drawer sealed)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: TRANSPORT teleport to the staging pose ---------------
        # (carry the tool across the room — the one permitted teleport)
        stage_x = 0.090
        s13 = torch.zeros(n, 13, device=device)
        s13[:, 0] = scene.env_origins[:, 0] + stage_x
        s13[:, 1] = scene.env_origins[:, 1]
        s13[:, 2] = scene.env_origins[:, 2] + c.slot_z
        s13[:, 3:7] = key_quat(0.0)
        scene.key.write_root_state_to_sim(s13, torch.arange(n, device=device))
        ramp(60, 60, (stage_x, 0.0, c.slot_z), (stage_x, 0.0, c.slot_z), 0.0, 0.0,
             done=lambda: float(scene.key.data.root_lin_vel_w.norm()) < 0.02, label="stage-hold")
        kp1 = (scene.key.data.root_pos_w - scene.env_origins)[0]
        assert abs(float(kp1[0]) - stage_x) < 0.01 and abs(float(kp1[2]) - c.slot_z) < 0.008, \
            "key must hover at the staging pose"
        assert not bool(scene.keyed()[0]), "staging must not be keyed yet"
        s = print_score("P1 key transported to the staging hover (crossbar outside)")
        assert s <= 0.05, f"transport alone must not score, got {s}"

        # ---------------- phase 2: INSERT — crossbar through the slot, FLAT -------------
        ramp(300, 240, (stage_x, 0.0, c.slot_z), (c.key_ins_x, 0.0, c.slot_z), 0.0, 0.0,
             done=lambda: bool(scene.keyed()[0])
             and abs(float((scene.key.data.root_pos_w - scene.env_origins)[0, 0])
                     - c.key_ins_x) < 0.006,
             label="insert")
        report("inserted")
        assert bool(scene.keyed()[0]), "crossbar must be keyed behind the slab"
        assert float(scene.drawer_open()[0]) < 0.005, "insertion must not move the drawer"
        s = print_score("P2 crossbar inserted through the real slot (keyed)")
        assert s >= c.w_keyed - 1e-6, f"P2 score {s:.3f} below keyed credit"

        # ---------------- phase 3: TWIST — roll 90 deg, bayonet interlock ---------------
        ramp(300, 240, (c.key_ins_x, 0.0, c.slot_z), (c.key_ins_x, 0.0, c.slot_z),
             0.0, math.pi / 2,
             done=lambda: bool(scene.locked()[0]) and float(scene.cross_tilt()[0]) > 0.95,
             label="twist")
        report("locked")
        assert bool(scene.locked()[0]), "twist must engage the bayonet interlock"
        assert float(scene.cross_tilt()[0]) > 0.90, "crossbar must stand near-vertical"
        assert float(scene.drawer_open()[0]) < 0.005, "twisting must not move the drawer"
        s = print_score("P3 key twisted 90 deg (locked behind the slab)")
        assert s >= c.w_keyed + c.w_locked - 1e-6, f"P3 score {s:.3f} below locked credit"

        # ---------------- phase 4: PULL — crossbar drags the drawer open ----------------
        ramp(480, 300, (c.key_ins_x, 0.0, c.slot_z), (c.key_pull_x, 0.0, c.slot_z),
             math.pi / 2, math.pi / 2,
             done=lambda: float(scene.drawer_open()[0]) >= c.open_goal + 0.008,
             label="pull")
        report("pulled")
        assert float(scene.drawer_open()[0]) >= c.open_goal + 0.005, \
            f"pull must clear the goal: open={float(scene.drawer_open()[0]):+.4f}"
        assert bool(scene._crack_l[0]), "crack latch must fire during the pull"
        s = print_score("P4 drawer dragged open through the crossbar coupling")
        assert s >= 0.60 - 1e-4, f"P4 score {s:.3f} below full latch credit"

        # ---------------- phase 5: hands off — the drawer stands open; success ----------
        scene.key.set_external_force_and_torque(zero, zero)
        won_at = None
        for i in range(360):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("hands-off")
        assert won_at is not None and bool(scene.success()[0]), \
            f"success must hold hands-off: open={float(scene.drawer_open()[0]):+.4f}"
        assert float(scene.drawer_open()[0]) >= c.open_goal, "drawer must stand open"
        s = print_score(f"P5 released; settled open ({won_at} hands-off steps)")
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
