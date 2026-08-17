"""Teleport solution for CarouselVaultScene (sim_gen task `scene_d_i130`) — the
task's legitimacy certificate.

Teleports are permitted for TRANSPORT only; this solve uses exactly ONE: after
the red cube has been physically lifted OUT of the vault through the roof
window, it is teleported across the room to a hover just above the green pad
(what a gripper transport does). EVERY load-bearing interaction is applied
force/torque through contact dynamics:

1. OPEN (applied force): a PD force servo along +y on the shutter body — what
   a hand pinching the orange knob supplies, force-limited — slides the plate
   along its prismatic rails past `open_thresh`; the roof window is now clear.
2. INDEX (applied torque): a PD torque servo about z on the disc — what a
   fingertip dragging the exposed rim lip through the front slit supplies,
   torque-limited — rotates the turntable by the shortest path until the RED
   pocket is centred under the window, then brakes and lets it settle. The
   torque is about the rotor's own axis, so it is invariant under the
   pod-dependent wrench frame drag (memory: hinge-axis shortcut).
3. EXTRACT (applied force): a PD force servo (+ gravity feedforward) on the
   red cube lifts it straight up out of the pocket, through the open window,
   to a hover above the roof — a real aperture-passage contact problem (the
   closed shutter or an un-indexed pocket physically caps this motion; smoke
   proves both with the same kind of force).
4. TRANSPORT (the one teleport): wrench zeroed, then the airborne cube is
   written to a hover 3.5 cm above the pad and DROPPED. Nobody touches it
   again; it lands, settles, and success() turns True on the live state.

Servo gains audited against dt = 1/120 (dt*wn << 1, damped):
  shutter  m=0.35, lin_damp 4: KP=60 N/m (wn 13.1, dt*wn 0.11), KD=8, |F|<=8 N
  disc     Izz~0.024, ang_damp 4: KP=0.8 N.m/rad (wn 5.8, dt*wn 0.05),
           KD=0.25, |T|<=0.6 N.m  (terminal spin ~1.7 rad/s, rim ~0.13 m/s)
  cube     m=0.06, lin_damp 0.2: KP=8 N/m (wn 11.5, dt*wn 0.10), KD=0.8,
           |F|<=2.5 N per axis + 0.589 N gravity feedforward

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.
The whole solve is then repeated on a SECOND seed (fresh episode, no score
prints — the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.scene_d_i130.solve --headless [--seed N]
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

# ----- servo gains (what a hand supplies at each affordance, limited) ---------------------------
KP_S, KD_S, F_S = 60.0, 8.0, 8.0        # shutter knob push (N/m, N.s/m, N)
KP_D, KD_D, T_D = 0.8, 0.25, 0.6        # disc rim drag (N.m/rad, N.m.s/rad, N.m)
KP_B, KD_B, F_B = 8.0, 0.8, 2.5         # cube lift (N/m, N.s/m, N per axis)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carousel_vault")().build(num_envs=args.num_envs, device=device)
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
        p = (scene.prize.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | open={float(scene.shutter_open_amt()[0]):+.4f} "
              f"yaw={float(scene.disc_yaw()[0]):+.3f} "
              f"prize=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"opened={bool(scene.opened()[0])} indexed={bool(scene.indexed()[0])} "
              f"out={bool(scene.prize_out()[0])} on_pad={bool(scene.on_pad()[0])} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def body_force(body, f_w: torch.Tensor) -> None:
        """Encode a world force into the default (body-frame) wrench convention."""
        f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
        body.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)

    # ----- phase drivers ------------------------------------------------------------------------
    def push_shutter(y_to: float, steps: int, tail: int) -> None:
        """PD force servo along +y on the shutter (target ramped), release when
        past open_thresh AND slow (never release a servo mid-swing)."""
        y0 = float(scene.shutter_open_amt()[0])
        for i in range(steps + tail):
            a = min(1.0, i / max(steps, 1))
            y_t = y0 + (y_to - y0) * a
            y = scene.shutter_open_amt()
            vy = scene.shutter.data.root_lin_vel_w[:, 1]
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 1] = (KP_S * (y_t - y) - KD_S * vy).clamp(min=-F_S, max=F_S)
            body_force(scene.shutter, f_w)
            env.step(no_action)
            if a >= 1.0 and float(scene.shutter_open_amt()[0]) >= c.open_thresh + 0.005 \
                    and abs(float(vy[0])) < 0.05:
                break
        scene.shutter.set_external_force_and_torque(zero, zero)
        step(30)

    def spin_disc(steps: int) -> None:
        """PD torque servo about z on the disc to yaw 0 (shortest path), brake,
        zero the torque only once slow (coast overshoot ~ w/damp << index tol)."""
        for _ in range(steps):
            yaw = scene.disc_yaw()
            err = torch.atan2(torch.sin(-yaw), torch.cos(-yaw))
            wz = scene.disc.data.root_ang_vel_w[:, 2]
            t_w = torch.zeros(n, 3, device=device)
            t_w[:, 2] = (KP_D * err).clamp(min=-T_D, max=T_D) - KD_D * wz
            t_b = quat_apply_inverse(scene.disc.data.root_quat_w, t_w)
            scene.disc.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))
            env.step(no_action)
            if abs(float(err[0])) < 0.05 and abs(float(wz[0])) < 0.08:
                break
        scene.disc.set_external_force_and_torque(zero, zero)
        step(60)

    def lift_prize(z_to: float, steps: int, tail: int) -> None:
        """PD force servo (+ gravity ff) lifting the cube straight up through
        the window; xy held at the capture point (the live pocket centre)."""
        p0 = (scene.prize.data.root_pos_w - scene.env_origins).clone()
        rose = False
        for i in range(steps + tail):
            a = min(1.0, i / max(steps, 1))
            p_t = p0.clone()
            p_t[:, 2] = p0[:, 2] + (z_to - p0[:, 2]) * a
            p = scene.prize.data.root_pos_w - scene.env_origins
            v = scene.prize.data.root_lin_vel_w
            f_w = KP_B * (p_t - p) - KD_B * v
            f_w[:, 2] += c.block_mass * 9.81
            f_w = f_w.clamp(min=-F_B, max=F_B)
            body_force(scene.prize, f_w)
            env.step(no_action)
            if i == 90 and not rose:
                dz = float((scene.prize.data.root_pos_w - scene.env_origins)[0, 2] - p0[0, 2])
                assert dz > 0.005, f"lift stalled: dz={dz:+.4f} after 90 steps"
                rose = True
            if a >= 1.0 and float((scene.prize.data.root_pos_w
                                   - scene.env_origins)[0, 2]) > c.out_z + 0.03:
                break

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
        lay = scene.layout[0]
        th0 = float(lay[0])
        th0_w = math.atan2(math.sin(th0), math.cos(th0))
        yaw = float(scene.disc_yaw()[0])
        pk = scene.red_pocket_xy()[0]
        pz = (scene.prize.data.root_pos_w - scene.env_origins)[0]
        pd = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] layout readback (seed {seed}): theta0 {th0:+.3f} "
              f"(wrapped {th0_w:+.3f}, live yaw {yaw:+.3f}) "
              f"pad sampled ({float(lay[1]):+.3f},{float(lay[2]):+.3f}) "
              f"live ({float(pd[0]):+.3f},{float(pd[1]):+.3f}) "
              f"pocket ({float(pk[0]):+.3f},{float(pk[1]):+.3f}) "
              f"prize ({float(pz[0]):+.3f},{float(pz[1]):+.3f},{float(pz[2]):+.3f})",
              flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert abs(math.atan2(math.sin(yaw - th0_w), math.cos(yaw - th0_w))) < 0.06, \
            "disc must settle at its sampled yaw"
        assert (pz[:2] - pk).norm() < 0.02, "prize must ride its pocket"
        assert abs(float(pd[0]) - float(lay[1])) < 0.005 and \
            abs(float(pd[1]) - float(lay[2])) < 0.005, "pad must sit at its sampled xy"
        assert float(scene.shutter_open_amt()[0]) < 0.01, "shutter must start closed"
        assert not bool(scene.indexed()[0]), "disc must start un-indexed"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (vault sealed, pocket off-window)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: OPEN — force the shutter along its rails --------------
        push_shutter(0.162, 300, 300)
        report("opened")
        assert bool(scene.opened()[0]), \
            f"shutter must stand open: {float(scene.shutter_open_amt()[0]):+.4f}"
        assert not bool(scene.indexed()[0]), "opening must not index the disc"
        s = print_score("P1 shutter slid open by real y-force on the knob")
        assert s >= c.w_open - 1e-6, f"P1 score {s:.3f} below open credit"

        # ---------------- phase 2: INDEX — torque the disc by its exposed rim ------------
        spin_disc(1500)
        report("indexed")
        assert bool(scene.indexed()[0]), \
            f"red pocket must index under the window: yaw={float(scene.disc_yaw()[0]):+.3f}"
        pz2 = (scene.prize.data.root_pos_w - scene.env_origins)[0]
        assert float(pz2[2]) < c.roof_z0, "cube must still be inside the vault"
        s = print_score("P2 disc indexed by real z-torque (red pocket under window)")
        assert s >= c.w_open + c.w_index - 1e-6, f"P2 score {s:.3f} below index credit"

        # ---------------- phase 3: EXTRACT — lift the cube out through the window --------
        lift_prize(0.30, 240, 240)
        report("extracted")
        pz3 = (scene.prize.data.root_pos_w - scene.env_origins)[0]
        assert float(pz3[2]) > c.out_z + 0.02, \
            f"cube must hover above the roof: z={float(pz3[2]):+.4f}"
        assert bool(scene._out_l[0]), "out latch must fire during the lift"
        s = print_score("P3 red cube lifted out through the open window by real force")
        assert s >= 0.60 - 1e-4, f"P3 score {s:.3f} below full latch credit"

        # ---------------- phase 4: TRANSPORT — the one teleport, then drop ---------------
        scene.prize.set_external_force_and_torque(zero, zero)
        s13 = torch.zeros(n, 13, device=device)
        s13[:, 0] = scene.env_origins[:, 0] + scene.layout[:, 1]
        s13[:, 1] = scene.env_origins[:, 1] + scene.layout[:, 2]
        s13[:, 2] = scene.env_origins[:, 2] + c.pad_h + c.block_s / 2 + 0.035
        s13[:, 3] = 1.0
        scene.prize.write_root_state_to_sim(s13, torch.arange(n, device=device))

        # ---------------- phase 5: hands off — drop, land, settle; success ---------------
        won_at = None
        for i in range(480):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("hands-off")
        assert won_at is not None and bool(scene.success()[0]), \
            "success must hold hands-off after the drop"
        s = print_score(f"P4+P5 transported, dropped, settled on pad "
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
