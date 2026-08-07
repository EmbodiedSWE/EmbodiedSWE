"""Teleport solution for RampChockScene (sim_gen task `pour_water_i7`) — the task's
legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, one body at a time): a single root-state write carries the
   chock — and later each ball — from its floor spawn to a FREE-SPACE hover pose
   8 mm above the deck (zero velocity), exactly the pose a gripper would release it
   from. The write puts the body in open air; it satisfies no rubric clause by
   itself.
2. WEDGING / PARKING (gravity + contact, hands-off): from the hover the chock FALLS
   flat onto the deck and grips by friction; each ball FALLS onto the band and
   ROLLS back down the slope until the chock arrests it — the two balls park SIDE
   BY SIDE against the chock's uphill face. The judged equilibrium — balls at rest
   on the incline, chock loaded below them — is produced entirely by gravity,
   rolling contact and friction, never written. If a ball somehow stalls short, a
   small escalating downslope force at its CoM (starting well under its own
   weight) stands in for a fingertip nudge — contact-consistent, cleared at once.
3. ORDER: chock first, then the balls — enforced by gravity itself: a ball
   released on the band before the chock rolls off the ramp (the retry logic
   would see it escape). The red decoy ball is never touched.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.pour_water_i7.solve --headless [--seed N]
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
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qmul, _qy = scene_mod._qmul, scene_mod._qy

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ramp_chock")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    th = math.radians(c.slope_deg)
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def slope_uvh(body) -> tuple[float, float, float]:
        u, v, h = scene._slope_coords(body.data.root_pos_w)
        return float(u[0]), float(v[0]), float(h[0])

    def report(tag: str) -> None:
        s = scene._status()
        cu, cv, chh = slope_uvh(scene.chock)
        parts = []
        for nm in scene.BALL_NAMES:
            u, v, h = slope_uvh(scene.balls[nm])
            parts.append(f"{nm}=(u{u:+.3f},v{v:+.3f},h{h:+.3f})")
        print(f"[solve] {tag:14s} | chock=(u{cu:+.3f},v{cv:+.3f},h{chh:+.3f}) "
              + " ".join(parts)
              + f" zone={bool(s['ch_zone'][0])} "
              f"park={s['park'][0].tolist()} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def deck_pose(u: float, v: float, h: float, deck_quat: bool) -> torch.Tensor:
        """(N,13) root state: ramp-frame slope coords (u, v, h); orientation either
        deck-matched (chock, axis cross-slope) or identity (spheres); zero velocity."""
        from isaaclab.utils.math import quat_apply

        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = u * math.cos(th) - h * math.sin(th)
        loc[:, 1] = v
        loc[:, 2] = u * math.sin(th) + h * math.cos(th)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.ramp.data.root_pos_w + quat_apply(scene.ramp.data.root_quat_w, loc)
        if deck_quat:
            st[:, 3:7] = _qmul(scene.ramp.data.root_quat_w,
                               _qy(torch.full((n,), -th, device=device)))
        else:
            st[:, 3] = 1.0
        return st

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def settle_on_deck(body, tag: str, max_steps: int = 480) -> bool:
        """Hands-off: let the body fall/roll under gravity until it settles on the
        deck. Returns False if it escapes the ramp (rolled off)."""
        for i in range(max_steps):
            env.step(no_action)
            u, v, h = slope_uvh(body)
            if h < 0.0 or u < 0.02 or abs(v) > c.deck_w / 2 + 0.05:
                print(f"[solve] {tag} escaped the ramp (u={u:+.3f} v={v:+.3f} "
                      f"h={h:+.3f})", flush=True)
                return False
            lv = float(body.data.root_lin_vel_w.norm(dim=-1)[0])
            av = float(body.data.root_ang_vel_w.norm(dim=-1)[0])
            if i > 40 and lv < 0.02 and av < 0.25:
                return True
        return True

    def nudge_downslope(body, newtons: float, steps: int) -> None:
        """Small CoM force along the downslope direction (world), then cleared."""
        from isaaclab.utils.math import quat_apply

        d = torch.zeros(n, 3, device=device)
        d[:, 0] = -math.cos(th)
        d[:, 2] = -math.sin(th)
        f = quat_apply(scene.ramp.data.root_quat_w, d).reshape(n, 1, 3) * newtons
        body.set_external_force_and_torque(f, zero_wrench, env_ids=all_ids, is_global=True)
        step(steps)
        clear_force(body)

    def place_ball(name: str, u_target: float, v_target: float) -> None:
        """TRANSPORT the ball to a free-space hover 8 mm above the band, slightly
        uphill of its rest — then hands-off: it falls and ROLLS back until the
        chock arrests it. Retries the drop if it escapes; escalating downslope
        nudge if it stalls short (spheres are analytic, so this should be idle)."""
        body = scene.balls[name]
        idx = scene.BALL_NAMES.index(name)
        for attempt in range(4):
            body.write_root_state_to_sim(
                deck_pose(u_target, v_target, c.ball_r + 0.008, deck_quat=False), all_ids)
            if settle_on_deck(body, name):
                break
            print(f"[solve] {name} retry {attempt + 1}", flush=True)
        else:
            print(f"SIM_GEN_SOLVE: FAIL ({name} kept escaping the ramp)", flush=True)
            os._exit(1)
        # stall short of contact? push gently downslope until parked or moving
        nudge = 0.15
        for _ in range(6):
            if bool(scene._status()["park"][0, idx]):
                break
            print(f"[solve] {name} stalled short; downslope nudge {nudge:.2f} N",
                  flush=True)
            nudge_downslope(body, nudge, 60)
            step(120)
            nudge = min(nudge * 1.6, 0.8)
        step(90)  # full settle, hands-off

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(120)
    rp = (scene.ramp.data.root_pos_w - scene.env_origins)[0]
    rq = scene.ramp.data.root_quat_w[0]
    ryaw = math.degrees(2.0 * math.atan2(float(rq[3]), float(rq[0])))
    spawns = []
    for nm, body in (("chock", scene.chock), ("decoy", scene.decoy),
                     ("ball_a", scene.balls["ball_a"]),
                     ("ball_b", scene.balls["ball_b"])):
        p = (body.data.root_pos_w - scene.env_origins)[0]
        spawns.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f})")
    print(f"[solve] layout readback (seed {args.seed}): "
          f"ramp=({float(rp[0]):+.3f},{float(rp[1]):+.3f}) yaw={ryaw:+.1f}deg "
          + " ".join(spawns), flush=True)
    report("reset")
    for body in (scene.chock, scene.decoy, *scene.balls.values()):
        assert torch.isfinite(body.data.root_pos_w).all(), "NaN/inf after settle"
    s0 = print_score("P0 reset+settle (everything scattered on the floor)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: wedge the chock across the band -----------------------------
    u_chock = c.band_lo + 0.005
    scene.chock.write_root_state_to_sim(
        deck_pose(u_chock, 0.0, c.chock_h / 2 + 0.008, deck_quat=True), all_ids)
    if not settle_on_deck(scene.chock, "chock"):
        print("SIM_GEN_SOLVE: FAIL (chock slid off the deck)", flush=True)
        os._exit(1)
    step(90)
    report("chock")
    assert bool(scene._chock_zone[0]), "chock zone latch did not set"
    assert not bool(scene.success()[0]), "chock alone cannot be success"
    s1 = print_score("P1 chock wedged across the band, held by friction")
    assert s1 >= s0 - 1e-6 and s1 >= 0.14, f"P1 score {s1} (expect 0.15)"

    # ---------------- phase 2: first ball rolls back against the chock ---------------------
    cu, cv, _chh = slope_uvh(scene.chock)
    u_drop = cu + c.chock_w / 2 + c.ball_r + 0.012
    place_ball("ball_a", u_drop, cv - 0.045)
    report("ball_a")
    assert bool(scene._parked[0, 0]), "ball_a park latch did not set"
    assert not bool(scene.success()[0]), "one ball cannot be success"
    s2 = print_score("P2 first ball parked against the chock")
    assert s2 >= s1 - 1e-6 and s2 >= 0.39, f"P2 score {s2} (expect 0.40)"

    # ---------------- phase 3: second ball parks beside the first --------------------------
    place_ball("ball_b", u_drop, cv + 0.045)
    report("ball_b")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the second ball)", flush=True)
        os._exit(1)
    s3 = print_score("P3 both balls parked side by side, chock loaded")
    assert s3 >= s2 - 1e-6, "score decreased across the second ball"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
