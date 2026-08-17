"""Teleport solution for MugTrapScene (sim_gen task `put_mug_i289`) — the task's
legitimacy certificate.

The task's load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, the only pose write on the mug): one root-state write
   carries the mug from its upright floor spawn across open air to an INVERTED
   hover directly above the ball — mouth-down, rim ~30 mm above the floor, the
   ball centred inside the rim ring with 19 mm of radial clearance. Both endpoints
   are contact-free; this is exactly what a pick-flip-carry does. The written
   state cannot satisfy success(): the mug is aloft (rest-band False), the ball is
   far from the disc, and the pose jump resets the stillness streak anyway.
2. CAPTURE (contact dynamics): gravity drops the mug the last 30 mm; the rim
   lands on the floor around the ball and the ball is genuinely trapped.
3. HERD (contact dynamics, no teleport): a horizontal external force at the mug's
   CoM (velocity-regulated bang-bang toward the disc, stall escalation) slides
   the capped mug along the floor; the interior wall pushes the CAPTIVE ball
   ahead of it — the ball's pose is never written after reset; it travels only by
   rim-ring contact — until the ball's own readback sits well inside the disc.
4. SETTLE (hands-off): forces cleared; ball and mug rattle to rest and the
   stillness streak accumulates to success(). If the coast-out parks the ball
   marginally outside the disc, the herd re-engages (still pure contact).

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.put_mug_i289.solve --headless [--seed N]
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
import traceback

import torch

import robobench
from isaaclab.utils.math import quat_apply_inverse
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)",
                                             flush=True), os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_trap")().build(num_envs=args.num_envs,
                                              device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap) and print the layout readback
    # so distinct seeds are provable from stdout.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        mp = (scene.mug.data.root_pos_w - scene.env_origins)[0]
        bp = (scene.ball.data.root_pos_w - scene.env_origins)[0]
        print(f"[solve] {tag:12s} | mug=({float(mp[0]):+.3f},{float(mp[1]):+.3f},"
              f"{float(mp[2]):.3f}) up_z={float(scene._mug_up_z()[0]):+.3f} "
              f"ball=({float(bp[0]):+.3f},{float(bp[1]):+.3f},{float(bp[2]):.3f}) "
              f"d_zone={float(scene.ball_zone_dist()[0]):.3f} "
              f"flip={bool(scene.flipped_now()[0])} "
              f"cov={bool(scene.covered_now()[0])} "
              f"zone={bool(scene.in_zone_now()[0])} still={int(scene._still[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    last_score = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= last_score[0] - 1e-6, f"score decreased at {tag}"
        last_score[0] = s
        return s

    def clear_force() -> None:
        scene.mug.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                env_ids=all_ids)

    def herd(stop_dist: float, max_steps: int) -> None:
        """Slide the capped mug toward the disc with a velocity-regulated bang-bang
        horizontal push (stall escalation); the captive ball is pushed by the
        interior wall. The push is applied LOW on the wall (fingertip height,
        ~12 mm above the floor) — force at the CoM plus the compensating torque
        r x F — so even the 6 N ledge-climb setting stays far below the rim-tip
        threshold and the trap is never pitched open.
        Breaks on the BALL's own distance-to-disc readback."""
        f_mag, v_des = 1.5, 0.05
        r_app = torch.tensor([0.0, 0.0, 0.012 - c.inv_rest_z], device=device)
        best = float(scene.ball_zone_dist()[0])
        last_bump, uncov = 0, 0
        for i in range(max_steps):
            d = float(scene.ball_zone_dist()[0])
            if d < stop_dist:
                break
            uncov = 0 if bool(scene.covered_now()[0]) else uncov + 1
            assert uncov < 60, \
                f"ball escaped the trap while herding (step {i}, d_zone={d:.3f})"
            bp = (scene.ball.data.root_pos_w - scene.env_origins)[0, :2]
            dirv = torch.zeros(3, device=device)
            dirv[:2] = scene._zone_xy[0] - bp
            dirv = dirv / dirv.norm().clamp(min=1e-9)
            v_along = float((scene.mug.data.root_lin_vel_w[0] * dirv).sum())
            fw = dirv * (f_mag if v_along < v_des else 0.0)
            tw = torch.linalg.cross(r_app, fw)  # move application point down
            # set_external_force_and_torque takes BODY-frame wrenches: convert.
            q = scene.mug.data.root_quat_w
            f_body = quat_apply_inverse(q, fw.expand(n, 3))
            t_body = quat_apply_inverse(q, tw.expand(n, 3))
            scene.mug.set_external_force_and_torque(
                f_body.view(n, 1, 3).contiguous(),
                t_body.view(n, 1, 3).contiguous(), env_ids=all_ids)
            env.step(no_action)
            if d < best - 0.004:
                best, last_bump = d, i
            elif i - last_bump > 240:  # stalled: push harder
                f_mag = min(f_mag + 1.5, 6.0)
                last_bump = i
                print(f"[solve] herd stalled at d_zone={d:.3f}, raising force to "
                      f"{f_mag:.1f} N", flush=True)
        clear_force()

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(60)
    m_mug = float(scene.mug.root_physx_view.get_masses().reshape(-1)[0])
    m_ball = float(scene.ball.root_physx_view.get_masses().reshape(-1)[0])
    assert abs(m_mug - c.mug_mass) < 0.02, f"mug mass not applied ({m_mug})"
    assert abs(m_ball - c.ball_mass) < 0.005, f"ball mass not applied ({m_ball})"
    zc = scene._zone_xy[0]
    bp = (scene.ball.data.root_pos_w - scene.env_origins)[0]
    mp = (scene.mug.data.root_pos_w - scene.env_origins)[0]
    d_bz = float(scene.ball_zone_dist()[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"zone=({float(zc[0]):+.3f},{float(zc[1]):+.3f}) "
          f"ball=({float(bp[0]):+.3f},{float(bp[1]):+.3f},{float(bp[2]):.3f}) "
          f"mug=({float(mp[0]):+.3f},{float(mp[1]):+.3f},{float(mp[2]):.3f}) "
          f"d_ball_zone={d_bz:.3f} masses mug={m_mug:.3f} ball={m_ball:.3f}",
          flush=True)
    report("reset")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert d_bz >= c.ball_zone_min_sep - 0.01, "ball spawned too near the disc"
    assert float(scene._mug_up_z()[0]) > 0.95, "mug should spawn upright"
    print_score("P0 reset+settle")

    # ---------------- phase 1: TRANSPORT (teleport across free air only) --------------------
    # One pose write: the mug INVERTED, hovering with its rim ~30 mm above the
    # floor, ball centred inside the rim ring (19 mm radial clearance — no
    # contact). Aloft + far from the disc: nothing about this state is success.
    bp = (scene.ball.data.root_pos_w - scene.env_origins)[0]
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = bp[0]
    st[:, 1] = bp[1]
    st[:, 2] = c.inv_rest_z + 0.030
    st[:, 4] = 1.0  # q = (0, 1, 0, 0): pi about x -> mouth-down
    st[:, 0:3] += scene.env_origins
    scene.mug.write_root_state_to_sim(st, all_ids)
    assert not bool(scene.success()[0]), "hovering transport pose must not judge"
    assert not bool(scene.flipped_now()[0]), "aloft mug must not count as resting"
    report("hover")
    print_score("P1 transport: mug flipped, hovering over the ball")

    # ---------------- phase 2: CAPTURE (gravity drop, contact) ------------------------------
    step(120)  # 30 mm free fall; rim lands around the ball, brief rattle, streak
    report("captured")
    assert bool(scene.flipped_now()[0]), "mug did not come to rest mouth-down"
    assert bool(scene.covered_now()[0]), "ball is not trapped under the mug"
    assert bool(scene._captured[0]), "captured latch did not fire"
    assert not bool(scene.success()[0]), "captured far from the disc is not success"
    print_score("P2 capture: rim dropped around the ball (contact)")

    # ---------------- phase 3: HERD (external force, contact) -------------------------------
    herd(stop_dist=0.025, max_steps=3600)
    step(30)  # let the coast-out damp before the readback asserts
    report("herded")
    assert bool(scene.covered_now()[0]), "trap lost after herding"
    assert float(scene.ball_zone_dist()[0]) < c.zone_r, \
        "ball did not reach the disc"
    print_score("P3 herd: captive ball pushed into the disc (contact)")

    # ---------------- phase 4: hands-off settle to success ----------------------------------
    ok_settle = False
    for chunk in range(40):  # up to 10 s
        step(30)
        if bool(scene.success()[0]):
            ok_settle = True
            break
        if chunk in (12, 24) and not bool(scene.in_zone_now()[0]):
            # coast-out parked the ball marginally outside: re-engage (contact)
            print("[solve] ball settled outside the disc — re-herding", flush=True)
            herd(stop_dist=0.020, max_steps=1200)
    report("settled")
    if not ok_settle:
        print("SIM_GEN_SOLVE: FAIL (no success after settle)", flush=True)
        os._exit(1)
    print_score("P4 settled to success (covered ball inside the disc)")

    # ---------------- phase 5: persistence (>= 3 simulated seconds, no intervention) --------
    hold = True
    for _ in range(10):  # 10 x 42 steps = 420 substeps = 3.5 s at 120 Hz
        step(42)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.5 s")
    ok = hold and bool(scene.success()[0]) and s5 >= 1.0 - 1e-6
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
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
