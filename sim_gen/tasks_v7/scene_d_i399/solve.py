"""Teleport solution for MastLoweringScene (sim_gen task `scene_d_i399`) — the
task's legitimacy certificate.

Teleports are permitted for TRANSPORT only; this solve uses exactly ONE: the
parked cradle is teleported from its rack to a hover just above its corridor
target (what a gripper carrying it by the crest bar does). EVERY load-bearing
interaction is applied force/torque through contact dynamics:

1. DEPLOY (applied force): after the hover-drop, a real downward PRESS on the
   cradle seats it square on the court floor — what a hand pressing the crest
   bar supplies. Nothing touches it again; the deployed() latch is judged on
   the settled body.
2. FELL (applied torque): a ramped PD torque about the mast's own hinge axis
   (body-frame, along the body y axis — invariant under the wrench frame drag)
   pushes the mast off its back stop and past vertical, force-limited to what
   a hand at the shaft supplies. The torque is CUT at 18 deg — past the
   commit angle — and gravity does the rest.
3. CATCH (nothing at all): the mast swings down hands-off, passes over the
   bottle, and lands in the cradle saddle; the heavy angular damping makes the
   catch gentle. success() turns True on the live settled state: mast in the
   92..112 deg band (floor rest ~116 deg can never pass), bottle still
   standing on its pad.

Servo/press magnitudes audited against dt = 1/120:
  cradle press  8 N downward for 0.5 s (m=1.2 -> firm seat, no launch)
  mast torque   KP=2.0 N.m/rad, KD=0.3, |T|<=1.6 N.m  (I_hinge~0.123:
                wn 4.0, dt*wn 0.033; static hold at the stop is 0.31 N.m)

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.
The whole solve is then repeated on a SECOND seed (fresh episode, no score
prints — the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.scene_d_i399.solve --headless [--seed N]
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

# ----- actuation magnitudes (what a hand supplies at each affordance, limited) -------------------
F_PRESS = 8.0                     # cradle seating press (N, downward)
KP_M, KD_M, T_M = 2.0, 0.3, 1.6   # mast hinge push (N.m/rad, N.m.s/rad, N.m)
CUT_DEG = 18.0                    # release the mast here (past commit_deg = 15)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mast_lowering")().build(num_envs=args.num_envs, device=device)
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
        b = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
        cr = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | phi={math.degrees(float(scene.phi()[0])):+7.2f}deg "
              f"w={float(scene.mast_w()[0]):+.3f} "
              f"bottle=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f}) "
              f"cradle=({float(cr[0]):+.3f},{float(cr[1]):+.3f},{float(cr[2]):+.3f}) "
              f"dep={bool(scene.deployed()[0])} band={bool(scene.in_band()[0])} "
              f"bok={bool(scene.bottle_ok()[0])} settled={bool(scene.settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def press_cradle(steps: int) -> None:
        """REAL seating press: constant downward force on the cradle (body-frame
        encoded), then release and settle."""
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = -F_PRESS
        for _ in range(steps):
            f_b = quat_apply_inverse(scene.cradle.data.root_quat_w, f_w)
            scene.cradle.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            env.step(no_action)
        scene.cradle.set_external_force_and_torque(zero, zero)
        step(60)

    def push_mast(steps: int) -> None:
        """REAL fell: ramped PD torque about the hinge axis, applied in the
        MAST's body frame along its local y (the hinge axis expressed in the
        body — invariant under the wrench frame-drag quirk). Cut at CUT_DEG."""
        phi0 = float(scene.phi()[0])
        tgt = math.radians(25.0)
        for i in range(steps):
            a = min(1.0, i / 120.0)
            phi_t = phi0 + (tgt - phi0) * a
            p = scene.phi()
            w = scene.mast_w()
            tq = (KP_M * (phi_t - p) - KD_M * w).clamp(min=-T_M, max=T_M)
            t_b = torch.zeros(n, 3, device=device)
            t_b[:, 1] = tq  # body-frame: along the mast's own y = the hinge axis
            scene.mast.set_external_force_and_torque(zero, t_b.reshape(n, 1, 3))
            env.step(no_action)
            if float(scene.phi()[0]) >= math.radians(CUT_DEG):
                break
        scene.mast.set_external_force_and_torque(zero, zero)

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
        db, py = float(lay[0]), float(lay[1])
        rx, ry, ryaw = float(lay[2]), float(lay[3]), float(lay[4])
        b = (scene.bottle.data.root_pos_w - scene.env_origins)[0]
        cr = (scene.cradle.data.root_pos_w - scene.env_origins)[0]
        pd = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        phi_deg = math.degrees(float(scene.phi()[0]))
        print(f"[solve] layout readback (seed {seed}): pad sampled ({db:+.3f},{py:+.3f}) "
              f"live ({float(pd[0]):+.3f},{float(pd[1]):+.3f}) "
              f"bottle ({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f}) "
              f"rack sampled ({rx:+.3f},{ry:+.3f},yaw {ryaw:+.2f}) "
              f"cradle live ({float(cr[0]):+.3f},{float(cr[1]):+.3f}) "
              f"mast phi {phi_deg:+.2f}deg", flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert c.limit_lo_deg - 0.8 <= phi_deg <= -4.0, \
            f"mast must rest on its back stop, phi={phi_deg:+.2f}"
        assert abs(float(pd[0]) - db) < 0.005 and abs(float(pd[1]) - py) < 0.005, \
            "pad must sit at its sampled marker"
        assert abs(float(b[0]) - db) < 0.01 and abs(float(b[1]) - py) < 0.01, \
            "bottle must stand at its sampled marker"
        assert bool(scene.bottle_ok()[0]), "bottle must start upright on its pad"
        assert abs(float(cr[0]) - rx) < 0.02 and abs(float(cr[1]) - ry) < 0.02, \
            "cradle must park at its sampled rack spot"
        assert not bool(scene.deployed()[0]), "cradle must not start deployed"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (mast on back stop, cradle racked)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: DEPLOY — the one teleport, then a real press ----------
        d_star = db + c.deploy_off
        s13 = torch.zeros(n, 13, device=device)
        s13[:, 0] = scene.env_origins[:, 0] + d_star
        s13[:, 1] = scene.env_origins[:, 1]
        s13[:, 2] = scene.env_origins[:, 2] + c.court_z1 + 0.010
        s13[:, 3] = 1.0  # yaw 0: channel along the corridor
        scene.cradle.write_root_state_to_sim(s13, torch.arange(n, device=device))
        step(30)  # hover-drop
        press_cradle(60)
        report("deployed")
        assert bool(scene.deployed()[0]), "cradle must stand deployed on the corridor floor"
        assert bool(scene.bottle_ok()[0]), "deploying must not disturb the bottle"
        s = print_score("P1 cradle transported, hover-dropped, press-seated")
        assert s >= c.w_deploy - 1e-6, f"P1 score {s:.3f} below deploy credit"

        # ---------------- phase 2: FELL — real hinge torque past vertical ----------------
        push_mast(600)
        phi2 = math.degrees(float(scene.phi()[0]))
        report("committed")
        assert phi2 >= CUT_DEG - 0.5, f"mast must pass the cut angle, phi={phi2:+.2f}"
        assert bool(scene._commit_l[0]), "commit latch must fire during the push"
        s = print_score("P2 mast pushed past vertical by real hinge torque (then cut)")
        assert s >= c.w_deploy + c.w_commit - 1e-6, f"P2 score {s:.3f} below commit credit"

        # ---------------- phase 3: CATCH — hands off, gravity + cradle -------------------
        won_at = None
        for i in range(900):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("caught")
        assert won_at is not None and bool(scene.success()[0]), \
            "success must arrive hands-off after the cut"
        phi3 = math.degrees(float(scene.phi()[0]))
        exp = c.phi_rest_deg(d_star + c.plate_hx)
        assert abs(phi3 - exp) < 4.0, \
            f"caught angle {phi3:+.2f} must match the cradle geometry ({exp:+.2f})"
        assert bool(scene.bottle_ok()[0]), "the bottle must have survived the fell"
        s = print_score(f"P3 mast caught in the saddle ({won_at} hands-off steps, "
                        f"phi {phi3:+.1f} ~ {exp:+.1f} deg)")
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
