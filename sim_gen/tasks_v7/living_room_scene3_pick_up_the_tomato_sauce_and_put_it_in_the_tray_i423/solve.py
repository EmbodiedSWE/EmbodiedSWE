"""Teleport solution for InertiaDerbyScene (sim_gen task
`living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i423`) — the
task's legitimacy certificate.

Teleports are TRANSPORT ONLY (carrying free bodies between rest poses); every
load-bearing outcome is delivered by contact dynamics under gravity:

1. PERCEPTION: the orb->cradle permutation, the tray pose and the per-orb
   inertia factors are read back from the episode state. The IDENTITY of the
   genuine orb is NOT read — it is INFERRED from the physical race, and only
   ASSERTED against the hidden truth afterwards.
2. STAGE (a carry): orbs #0 and #1 are teleported to hover over the two ramp
   lanes uphill of the start bar and dropped; gravity rolls them down against
   the resting bar, where they settle (contact).
3. RACE (the experiment): hand-proxy HOLDS (<= 2.5 N per orb: feed-forward
   cancels gravity along the slope, PD nulls drift) pin both racers at their
   staged poses while a force-limited PD lift (<= 1.5 N on the 35 g bar — a
   two-finger pinch) raises the bar out of its pockets; the bar is parked out
   of the way (transport) and the racers settle to rest under the holds. BOTH
   holds are then cut on the SAME step: the orbs start rolling from rest, from
   equal x, at the same instant — the release is symmetric by construction and
   the only differentiator is rotational inertia
   (a = g sin(theta) / (1 + I/(m r^2))). When the leader crosses the
   checkpoint, the gap is read: leader ahead by >= gap_thresh => the leader is
   the genuine solid orb; near-tie => both racers are hollow and the un-raced
   orb #2 is genuine. The verdict is ASSERTED against the hidden genuine_idx.
4. DELIVER (a carry): the identified genuine orb is teleported to hover over
   the tray and DROPPED; it lands and settles on the tray floor by gravity.
   No counterfeit ever touches the tray.
5. HANDS-OFF: success() turns True on the live settled state (genuine in the
   upright tray, no fake inside, race performed, no foul) and persists.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
rubric latches), holds HANDS-OFF >= 3 simulated seconds after success() first
turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds. The
whole solve is then repeated on a SECOND seed (fresh episode, no score prints)
to certify seed robustness.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i423.solve
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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.inertia_derby")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)

    th = math.radians(c.theta_deg)

    def surf(x: float) -> float:
        return c.bench_z + c.lip + (c.ramp_x1 - x) * math.tan(th)

    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def bx(b: int) -> float:
        return float(scene._local(scene.balls[b])[0, 0])

    def bv(b: int) -> float:
        return float(scene.balls[b].data.root_lin_vel_w[0].norm())

    def report(tag: str) -> None:
        ks = scene.ball_k()[0]
        print(f"[solve] {tag:10s} | x=({bx(0):+.3f},{bx(1):+.3f},{bx(2):+.3f}) "
              f"k=({float(ks[0]):.2f},{float(ks[1]):.2f},{float(ks[2]):.2f}) "
              f"raced={scene._raced[0].tolist()} gin={bool(scene.genuine_in()[0])} "
              f"fin={bool(scene.fake_in()[0])} foul={bool(scene._foul[0])} "
              f"del={bool(scene._delivered[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    zero3 = torch.zeros(n, 1, 3, device=device)

    def hover_drop(body, x: float, y: float, z: float) -> None:
        """Transport teleport: place at rest, slightly above a support."""
        s = torch.zeros(n, 13, device=device)
        s[:, 0] = scene.env_origins[:, 0] + x
        s[:, 1] = scene.env_origins[:, 1] + y
        s[:, 2] = scene.env_origins[:, 2] + z
        s[:, 3] = 1.0
        body.write_root_state_to_sim(s)

    def episode(seed: int, announce: bool) -> bool:
        """One full episode on `seed`. SIM_GEN_SCORE printed only when
        `announce` (the rubric restarts at 0 on reset; the announced score
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
        perm = scene.perm[0].tolist()
        ks = scene.ball_k()[0]
        tray_p = scene._local(scene.tray)[0]
        print(f"[solve] layout readback (seed {seed}): perm={perm} "
              f"k=({float(ks[0]):.3f},{float(ks[1]):.3f},{float(ks[2]):.3f}) "
              f"tray=({float(tray_p[0]):+.3f},{float(tray_p[1]):+.3f})", flush=True)
        report("reset")
        # masses equal (the anti-weighing guarantee), inertias written: one solid, two hollow
        for b in range(3):
            m_b = float(scene.balls[b].root_physx_view.get_masses().view(-1)[0])
            assert abs(m_b - c.ball_m) < 0.02 * c.ball_m, f"orb {b} mass {m_b}"
        n_solid = int(((ks - c.k_true).abs() < 0.05).sum())
        n_hollow = int(((ks - c.k_fake).abs() < 0.05).sum())
        assert n_solid == 1 and n_hollow == 2, f"inertia write-through broken: {ks.tolist()}"
        # orbs nested in their cradles, per the sampled permutation
        for b in range(3):
            p = scene._local(scene.balls[b])[0]
            assert abs(float(p[0]) - c.slots_x[perm[b]]) < 0.012 \
                and abs(float(p[1]) - c.slot_y) < 0.012, f"orb {b} not in its cradle"
            assert bv(b) < c.settle_lin, f"orb {b} not settled"
        assert not bool(scene.success()[0]), "fresh reset must not be success"
        s = print_score("P0 reset+settle (orbs cradled, bar seated, tray placed)")
        assert s <= 0.01, f"baseline score should be ~0, got {s}"

        # ---------------- phase 1: STAGE — carry orbs 0,1 to the lanes ------------------
        hz = surf(-0.345) + c.ball_r / math.cos(th) + 0.008
        hover_drop(scene.balls[0], -0.345, +c.lane_y, hz)
        hover_drop(scene.balls[1], -0.345, -c.lane_y, hz)
        step(200)  # gravity: drop, roll down, rest against the slick bar
        report("staged")
        for b in (0, 1):
            x = bx(b)
            assert -0.365 < x < c.gate_x, f"orb {b} not resting against the bar (x={x:+.3f})"
            assert bv(b) < c.settle_lin, f"orb {b} still moving on the bar"
        bar_z0 = float(scene._local(scene.bar)[0, 2])
        assert abs(bar_z0 - (surf(c.gate_x) + c.bar_gap + c.bar_lx / 2)) < 0.006, \
            "bar must still be seated in its pockets"
        print_score("P1 two orbs staged against the start bar")

        # ---------------- phase 2: RACE — hold, lift the bar, release together ----------
        # Hand-proxy holds pin both racers (feed-forward cancels the slope
        # component of gravity, PD nulls drift) while the bar is lifted clear
        # and parked. Cutting both holds on the SAME step gives a symmetric
        # release: equal x, rest, same instant — only inertia differentiates.
        from isaaclab.utils.math import quat_apply_inverse

        g_sin = c.ball_m * 9.81 * math.sin(th)
        ff_x, ff_z = -g_sin * math.cos(th), g_sin * math.sin(th)
        p_ref = {b: scene.balls[b].data.root_pos_w[0].clone() for b in (0, 1)}

        def hold_racers() -> None:
            for b in (0, 1):
                ball = scene.balls[b]
                p = ball.data.root_pos_w
                v = ball.data.root_lin_vel_w
                fw = torch.zeros(n, 3, device=device)
                fw[:, 0] = ff_x + 60.0 * (p_ref[b][0] - p[:, 0]) - 8.0 * v[:, 0]
                fw[:, 1] = 60.0 * (p_ref[b][1] - p[:, 1]) - 8.0 * v[:, 1]
                fw[:, 2] = ff_z
                fw = fw.clamp(-2.5, 2.5)
                fb = quat_apply_inverse(ball.data.root_quat_w, fw)
                ball.set_external_force_and_torque(fb.reshape(n, 1, 3), zero3)

        clear_z = surf(c.gate_x) + 2 * c.ball_r + c.bar_lx / 2 + 0.012
        lifted = None
        for i in range(360):
            hold_racers()
            z = float(scene._local(scene.bar)[0, 2])
            vz = float(scene.bar.data.root_lin_vel_w[0, 2])
            fw = torch.zeros(n, 3, device=device)
            fw[:, 2] = min(max(8.0 * (0.32 - z) - 1.2 * vz + 0.40, 0.0), 1.5)
            fb = quat_apply_inverse(scene.bar.data.root_quat_w, fw)
            scene.bar.set_external_force_and_torque(fb.reshape(n, 1, 3), zero3)
            env.step(no_action)
            if float(scene._local(scene.bar)[0, 2]) > clear_z:
                lifted = i
                break
        scene.bar.set_external_force_and_torque(zero3, zero3)
        assert lifted is not None, \
            f"bar lift failed (z={float(scene._local(scene.bar)[0, 2]):.3f})"
        hover_drop(scene.bar, *c.bar_park)  # park the tool out of the way (transport)
        for _ in range(150):  # settle to rest under the holds
            hold_racers()
            env.step(no_action)
        x0r, x1r = bx(0), bx(1)
        for b in (0, 1):
            assert bv(b) < c.settle_lin, f"racer {b} not at rest under the hold"
            assert -0.365 < bx(b) < c.gate_x, f"racer {b} drifted (x={bx(b):+.3f})"
        assert abs(x0r - x1r) < 0.006, \
            f"release not symmetric: x0={x0r:+.3f} x1={x1r:+.3f}"
        released = lifted
        # release BOTH holds on the same step: free roll from rest
        for b in (0, 1):
            scene.balls[b].set_external_force_and_torque(zero3, zero3)
        # free roll: wait for the leader to cross the checkpoint, then read the gap
        crossed = None
        for i in range(360):
            env.step(no_action)
            if max(bx(0), bx(1)) >= c.chk_x:
                crossed = i
                break
        assert crossed is not None, \
            f"no racer crossed the checkpoint (x0={bx(0):+.3f}, x1={bx(1):+.3f})"
        gap = bx(0) - bx(1)
        leader = 0 if gap > 0 else 1
        verdict = leader if abs(gap) >= c.gap_thresh else 2
        truth = int(scene.genuine_idx[0])
        print(f"[solve] race verdict: gap={gap:+.4f} m (thresh {c.gap_thresh}) "
              f"=> genuine is orb {verdict} (hidden truth: {truth})", flush=True)
        assert verdict == truth, "the physical experiment must identify the genuine orb"
        step(260)  # let the racers coast into the finish wall and settle
        report("raced")
        assert bool(scene._raced[0, 0]) and bool(scene._raced[0, 1]), \
            "both racers must earn the raced latch"
        s = print_score(f"P2 race performed and read ({released} lift steps, "
                        f"{crossed} roll steps)")
        assert s >= c.w_race - 1e-6, f"P2 score {s:.3f} below the race credit"

        # ---------------- phase 3: DELIVER — carry the genuine orb to the tray ----------
        tray_p = scene._local(scene.tray)[0]
        hover_drop(scene.balls[verdict], float(tray_p[0]), float(tray_p[1]),
                   float(tray_p[2]) + c.tray_t + c.ball_r + 0.030)
        won_at = None
        for i in range(360):
            env.step(no_action)
            if bool(scene.success()[0]):
                won_at = i
                break
        report("delivered")
        assert won_at is not None and bool(scene.success()[0]), \
            (f"success must arrive after the drop settles: gin={bool(scene.genuine_in()[0])} "
             f"fin={bool(scene.fake_in()[0])} race={bool(scene.race_done()[0])} "
             f"foul={bool(scene._foul[0])}")
        s = print_score(f"P3 genuine orb settled in the tray ({won_at} settle steps)")
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
