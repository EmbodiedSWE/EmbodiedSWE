"""Teleport solution for CrankFerryScene (sim_gen task `native_libero_i365`) —
the task's legitimacy certificate.

Teleports are used for TRANSPORT ONLY (one hop: the red cube from its sampled
apron pose to the window mouth, still on the apron surface). Every load-bearing
interaction is applied force/torque through contact dynamics:

1. PERCEPTION: theta0, cage_x, and both cube poses are read back from the
   episode state — never hard-coded.
2. CRANK TO ALIGN (applied torque): a velocity-servo torque on the crank disc's
   hinge axis (what a hand pushing the handle peg around its circle does,
   torque-limited) turns the yoke until the cage sits at the window dead
   center, then brakes.
3. LOAD (applied force): a force servo pushes the cube across the apron through
   the window into the cage — real sliding contact the whole way. The dead
   center holds the cage still (measured and asserted: it barely moves).
4. CRANK TO DELIVER (applied torque): the same servo cranks ON through the dead
   center; the cage plows the cube down the sealed lane through cage-wall
   contact until it tips over the hole edge and gravity drops it into the
   vault. The judged outcome is produced entirely by the mechanism.
5. RELEASE (nobody's hands): torques off; everything settles; success() turns
   True on the live state.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), then holds HANDS-OFF >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.
The whole solve is then repeated on a SECOND seed (fresh episode, no score
prints — the rubric restarts at 0 on reset) to certify seed robustness.

Run (forge): python -u -m simgen_tasks.native_libero_i365.solve --headless [--seed N]
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

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
    env = ENVS.get("simgen.crank_ferry")().build(num_envs=args.num_envs, device=device)
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
        p = scene._local(scene.cargo)[0]
        print(f"[solve] {tag:12s} | th={math.degrees(float(scene.disc_yaw()[0])):+7.1f}deg "
              f"cage={float(scene.cage_x()[0]):+.4f} "
              f"cargo=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
              f"latch(a/l/d)=({int(scene._align[0])},{int(scene._loaded[0])},"
              f"{int(scene._dropped[0])}) ferry={float(scene._ferry[0]):.2f} "
              f"settled={bool(scene.settled()[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def disc_torque(tau: torch.Tensor) -> None:
        """Apply `tau` (N,) about the disc's hinge axis, in the BODY frame (the
        disc only yaws, so body z is the hinge axis exactly)."""
        t = torch.zeros(n, 1, 3, device=device)
        t[:, 0, 2] = tau
        scene.disc.set_external_force_and_torque(zero, t)

    def crank(dir_: float, *, dist, done, steps: int, label: str,
              w_max: float = 0.9) -> None:
        """Applied-torque crank: a velocity servo about the crank hinge (what a
        hand pushing the 18 mm handle peg around its circle does), torque-
        limited, with stall escalation. `dist` (m along the rail) schedules the
        approach speed; `done` is checked every step."""
        K, taumax = 0.35, 0.8
        last_x, stall = float(scene.cage_x()[0]), 0
        for i in range(steps):
            if done():
                break
            d = max(float(dist()), 0.0)
            w_des = dir_ * (w_max if d > 0.05 else max(0.25, w_max * d / 0.05))
            w = scene.disc.data.root_ang_vel_w[:, 2]
            disc_torque((K * (w_des - w)).clamp(min=-taumax, max=taumax))
            env.step(no_action)
            if i % 120 == 119:  # stall watchdog: escalate the cap, never the goal
                x_now = float(scene.cage_x()[0])
                if abs(x_now - last_x) < 0.004:
                    stall += 1
                    taumax = min(taumax * 1.6, 2.4)
                    print(f"[solve] crank {label}: stall #{stall}, taumax={taumax:.2f}",
                          flush=True)
                last_x = x_now
        # active brake: servo the hinge rate to zero, then torque off
        for _ in range(180):
            w = scene.disc.data.root_ang_vel_w[:, 2]
            if float(w.abs()[0]) < 0.04:
                break
            disc_torque((0.6 * (0.0 - w)).clamp(min=-1.2, max=1.2))
            env.step(no_action)
        scene.disc.set_external_force_and_torque(zero, zero)
        step(30)
        print(f"[solve] crank {label}: th={math.degrees(float(scene.disc_yaw()[0])):+7.1f}deg "
              f"cage={float(scene.cage_x()[0]):+.4f} after <= {steps} steps", flush=True)

    def push_cargo(steps: int) -> None:
        """Applied-force load: a velocity servo pushes the cube in -y across the
        apron and through the window (a fingertip push, force-limited), with a
        mild x servo keeping it centered on the cage. World-frame forces are
        re-expressed in the cube's body frame every step."""
        for _ in range(steps):
            p = scene._local(scene.cargo)
            if float(p[0, 1]) < -0.005:
                break
            v = scene.cargo.data.root_lin_vel_w
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = (1.5 * (scene.cage_x() - p[:, 0]) - 0.6 * v[:, 0]).clamp(-0.6, 0.6)
            # velocity servo sized to beat ~0.4 N apron friction at rest:
            # 7.0 * 0.15 = 1.05 N standstill force; K*dt/m = 0.49 < 1 (stable)
            f_w[:, 1] = (7.0 * (-0.15 - v[:, 1])).clamp(min=-2.0, max=2.0)
            f_b = quat_apply_inverse(scene.cargo.data.root_quat_w, f_w)
            scene.cargo.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            env.step(no_action)
        scene.cargo.set_external_force_and_torque(zero, zero)
        step(60)
        p = scene._local(scene.cargo)[0]
        print(f"[solve] push load: cargo=({float(p[0]):+.4f},{float(p[1]):+.4f},"
              f"{float(p[2]):+.4f}) after <= {steps} steps", flush=True)

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
        th0 = float(scene.th0[0])
        x0 = float(scene.x0[0])
        pc = scene._local(scene.cargo)[0]
        pd = scene._local(scene.decoy)[0]
        print(f"[solve] layout readback (seed {seed}): theta0={math.degrees(th0):+.1f}deg "
              f"cage_x={float(scene.cage_x()[0]):+.4f} (implied {x0:+.4f}) "
              f"cargo=({float(pc[0]):+.3f},{float(pc[1]):+.3f}) "
              f"decoy=({float(pd[0]):+.3f},{float(pd[1]):+.3f})", flush=True)
        report("reset")
        assert bool(scene._finite()[0]), "NaN/inf after settle"
        assert abs(float(scene.cage_x()[0]) - x0) < 0.008, \
            "cage must rest where the sampled crank angle implies"
        assert c.th0_lo - 1.0 <= abs(math.degrees(th0)) <= c.th0_hi + 1.0
        assert float(pc[1]) > 0.06 and float(pd[1]) > 0.06, "cubes must start on the apron"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (cage away from the window, cubes on the apron)")
        assert s <= 0.05, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: crank the cage to the window dead center -------------
        dir_ = 1.0 if th0 > 0 else -1.0  # toward theta=+/-180: cage_x decreases
        crank(dir_, dist=lambda: abs(float(scene.cage_x()[0]) - scene.XL),
              done=lambda: abs(float(scene.cage_x()[0]) - scene.XL) <= 0.005,
              steps=900, label="to-align")
        assert bool(scene._align[0]), "align latch must fire during the crank"
        assert abs(float(scene.cage_x()[0]) - scene.XL) <= c.align_tol, \
            "cage must hold the dead center after the brake"
        s = print_score("P1 cage cranked to the window dead center (aligned)")
        assert s >= c.w_align - 1e-4, f"P1 score {s:.3f} below align credit"

        # ---------------- phase 2: teleport to the window mouth (TRANSPORT), then -------
        # ----------------          push the cube through the window (applied force) -----
        x_cage_before = float(scene.cage_x()[0])
        mouth = torch.zeros(n, 13, device=device)
        mouth[:, 0] = scene.env_origins[:, 0] + (c.win_x0 + c.win_x1) / 2
        mouth[:, 1] = scene.env_origins[:, 1] + 0.088
        mouth[:, 2] = scene.env_origins[:, 2] + c.apron_z1 + c.cargo_s / 2 + 0.003
        mouth[:, 3] = 1.0
        scene.cargo.write_root_state_to_sim(mouth, torch.arange(n, device=device))
        step(60)  # settle the set-down before any judging
        push_cargo(480)
        p = scene._local(scene.cargo)
        assert float(p[0, 1]) < 0.030, "cargo must be through the window"
        assert abs(float(p[0, 0]) - float(scene.cage_x()[0])) <= c.cage_half_in, \
            "cargo must sit inside the cage interior"
        assert bool(scene._loaded[0]), "loaded latch must fire"
        moved = abs(float(scene.cage_x()[0]) - x_cage_before)
        print(f"[solve] dead-center hold: cage moved {moved * 1000:.1f} mm during loading",
              flush=True)
        assert moved < 0.010, "the dead center must hold the cage during loading"
        s = print_score("P2 cargo pushed through the window, aboard the cage")
        assert s >= c.w_align + c.w_loaded - 1e-4, f"P2 score {s:.3f} below loaded credit"

        # ---------------- phase 3: crank on — plow to the hole, gravity discharge -------
        crank(dir_, dist=lambda: max(scene.XD - float(scene.cage_x()[0]), 0.0),
              done=lambda: bool(scene._dropped[0])
              or float(scene.cage_x()[0]) >= scene.XD - 0.006,
              steps=1100, label="to-deliver")
        step(120)
        assert bool(scene._dropped[0]), "cargo must drop through the hole into the vault"
        s = print_score("P3 cargo plowed down the lane and dropped into the vault")
        assert s >= 0.85 - 1e-4, f"P3 score {s:.3f} below full latch credit"

        # ---------------- phase 4: hands off — everything settles; success --------------
        won_at = None
        for i in range(480):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("hands-off")
        pz = float(scene._local(scene.cargo)[0, 2])
        assert won_at is not None and bool(scene.success()[0]), \
            f"success must hold hands-off: cargo z={pz:+.4f}"
        s = print_score(f"P4 released; cargo at rest in the vault ({won_at} hands-off steps)")
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
