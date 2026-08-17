"""Teleport solution for CaptureArenaScene (sim_gen task `setup_chess_i302`) — the
task's legitimacy certificate.

The CAPTURE — the load-bearing interaction of the task — is executed entirely
through contact dynamics: a horizontal velocity-servoed CoM push force (the wrench
a Franka fingertip pressing the exposed shaft would transmit; F = m*K*(v_des - v)
+ friction feedforward, K*dt = 0.25, cap ~1.5x piece weight, v capped at 0.12 m/s
so nothing is ever slammed) drives the black king off the gold dais, across the
board and through the rim opening; from there GRAVITY carries it down the chute
into the capture box — no teleport ever touches the black king. The white king's
TRANSPORT is the one teleport (carrying through free space, the legal use): it is
parked 25 mm ABOVE the dais and RELEASED — the seating is a real contact
settle onto the dais under gravity, judged as a settled upright pose.

Phases (the occupancy of the dais enforces the order; the solve just follows):
  P0  settle + layout readback (side, throne, masses); baseline score ~0.
  P1  push the BLACK king from the dais through the rim opening; gravity takes it
      down the chute (l1 latches, 0.25).
  P2  hands-off: the black king slides into the capture box and settles there
      (l2, 0.55).
  P3  teleport the WHITE king to 25 mm above the now-vacant dais, release, let it
      seat itself by contact (l3 en route) -> success, 1.0.
  P4  >= 3.4 simulated seconds HANDS-OFF; print SIM_GEN_SOLVE: SUCCESS only if
      success() still holds live.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
partial credit is latched).

Run (forge): python -u -m simgen_tasks.setup_chess_i302.solve --headless [--seed N]
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
import traceback

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

_qmul, _qconj = scene_mod._qmul, scene_mod._qconj

# Global watchdog (daemon): if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.capture_arena")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
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

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        b = scene.board_local(scene.black.data.root_pos_w)
        w = scene.board_local(scene.white.data.root_pos_w)
        ch = scene.chute_local(scene.black.data.root_pos_w)
        bv = float(scene.black.data.root_lin_vel_w.norm(dim=-1)[0])
        wv = float(scene.white.data.root_lin_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:12s} | black_board=({float(b[0, 0]):+.3f},{float(b[0, 1]):+.3f},"
              f"{float(b[0, 2]):+.3f}) black_chute=({float(ch[0, 0]):+.3f},"
              f"{float(ch[0, 1]):+.3f},{float(ch[0, 2]):+.3f}) "
              f"white=({float(w[0, 0]):+.3f},{float(w[0, 1]):+.3f},{float(w[0, 2]):+.3f}) "
              f"bv={bv:.3f} wv={wv:.3f} l1={bool(scene._l1[0])} l2={bool(scene._l2[0])} "
              f"l3={bool(scene._l3[0])} captured={bool(scene.black_captured()[0])} "
              f"enthroned={bool(scene.white_enthroned()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # frame-drag-robust force encoding: command R_ref * R_now^T * f_world
    q_ref = {}

    def encode(body, name: str, f_world: torch.Tensor) -> torch.Tensor:
        return quat_apply(_qmul(q_ref[name], _qconj(body.data.root_quat_w)), f_world)

    def servo(body, name: str, loc_fn, *, tol: float = 0.012, max_steps: int = 2400,
              stop_fn=None) -> bool:
        """Push `body` toward the board-local waypoint `loc_fn()` with a horizontal
        velocity-servoed CoM force (v capped at 0.12 m/s — quasi-static, nothing is
        slammed); release when close AND slow, or when `stop_fn` fires. Escalates
        gain (bounded: K*dt <= 0.75) and cap on stall."""
        m_kg = float(c.piece_mass)
        gain, f_cap = 30.0, 2.2
        v_max = 0.12
        reached = False
        last_xy = body.data.root_pos_w[:, 0:2].clone()
        for i in range(max_steps):
            if stop_fn is not None and stop_fn():
                reached = True
                break
            tgt_w = scene.board_to_world(loc_fn())
            err = tgt_w - body.data.root_pos_w
            err[:, 2] = 0.0
            dist = err.norm(dim=-1)
            v = body.data.root_lin_vel_w.clone()
            v[:, 2] = 0.0
            if stop_fn is None and float(dist[0]) < tol \
                    and float(v.norm(dim=-1)[0]) < 0.02:
                reached = True
                break
            dirv = err / dist.clamp(min=1e-6).unsqueeze(-1)
            v_des = dirv * torch.minimum(
                torch.full_like(dist, v_max), 2.5 * dist).unsqueeze(-1)
            f_w = m_kg * gain * (v_des - v) + 0.35 * dirv  # friction feedforward
            fmag = f_w.norm(dim=-1, keepdim=True)
            f_w = f_w * (f_cap / fmag.clamp(min=1e-9)).clamp(max=1.0)
            body.set_external_force_and_torque(
                encode(body, name, f_w).unsqueeze(1), zero_wrench,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i % 240 == 239:
                moved = float((body.data.root_pos_w[:, 0:2] - last_xy).norm(dim=-1)[0])
                if moved < 0.005:
                    gain = min(gain * 1.6, 90.0)  # K*dt stays <= 0.75
                    f_cap = min(f_cap * 1.4, 5.0)
                    print(f"[solve] {name} stalled at dist={float(dist[0]):.3f}; "
                          f"gain->{gain:.0f} cap->{f_cap:.1f}", flush=True)
                last_xy = body.data.root_pos_w[:, 0:2].clone()
        clear_force(body)
        return reached

    def wp(x_fn, y_fn):
        """Waypoint factory: board-local (x, y) per env (z is ignored by the servo)."""
        def loc_fn() -> torch.Tensor:
            loc = torch.zeros(n, 3, device=device)
            loc[:, 0] = x_fn()
            loc[:, 1] = y_fn()
            loc[:, 2] = c.board_h
            return loc
        return loc_fn

    # ---------------- phase 0: reset, settle, readback -------------------------------------
    step(180)
    q_ref["black"] = scene.black.data.root_quat_w.clone()
    pp = (scene.platform.data.root_pos_w - scene.env_origins)[0]
    pq = scene.platform.data.root_quat_w[0]
    pyaw = math.degrees(2.0 * math.atan2(float(pq[3]), float(pq[0])))
    side = float(scene.side[0])
    tx, ty = float(scene.throne[0, 0]), float(scene.throne[0, 1])
    b0 = scene.board_local(scene.black.data.root_pos_w)[0]
    w0 = scene.board_local(scene.white.data.root_pos_w)[0]
    ch_l = scene.board_local(scene.chute.data.root_pos_w)[0]
    bk_l = scene.board_local(scene.blocker.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"platform=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) yaw={pyaw:+.1f}deg "
          f"gap_side={'+y' if side > 0 else '-y'} throne=({tx:+.3f},{ty:+.3f}) "
          f"chute_y={float(ch_l[1]):+.3f} blocker_y={float(bk_l[1]):+.3f} "
          f"black=({float(b0[0]):+.3f},{float(b0[1]):+.3f},{float(b0[2]):+.3f}) "
          f"white=({float(w0[0]):+.3f},{float(w0[1]):+.3f})", flush=True)
    report("reset")
    assert torch.isfinite(scene.black.data.root_state_w).all(), "NaN in black state"
    assert torch.isfinite(scene.white.data.root_state_w).all(), "NaN in white state"
    mb = float(scene.black.root_physx_view.get_masses().reshape(-1)[0])
    mw = float(scene.white.root_physx_view.get_masses().reshape(-1)[0])
    print(f"[solve] mass readback: black={mb:.3f}kg white={mw:.3f}kg", flush=True)
    assert 0.10 < mb < 0.20 and 0.10 < mw < 0.20, "authored piece mass not applied"
    # throne starts OCCUPIED by the black king, white waits on the walled side
    assert (b0[:2] - scene.throne[0]).norm() < 0.03, "black must start on the dais"
    assert float(b0[2]) > c.board_h + c.dais_h - 0.004, "black must stand ON the dais"
    assert side * float(ch_l[1]) > 0.15, "chute must sit at the active gap"
    assert -side * float(bk_l[1]) > 0.15, "blocker must plug the mirror gap"
    assert -side * float(w0[1]) > 0.10, "white must start on the walled side"
    assert bool(scene.upright(scene.black)[0]) and bool(scene.upright(scene.white)[0])
    s0 = print_score("P0 reset+settle (throne occupied by the black king)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: push the black king off the board ---------------------------
    # waypoint 1: line up with the opening (still on the board)
    ok = servo(scene.black, "black",
               wp(lambda: 0.0, lambda: scene.side * 0.15), tol=0.02)
    report("black@gap")
    if not ok:
        print("SIM_GEN_SOLVE: FAIL (black king never lined up with the opening)", flush=True)
        os._exit(1)
    # waypoint 2: through the opening — stop the instant it is ejected (l1)
    ok = servo(scene.black, "black",
               wp(lambda: 0.0, lambda: scene.side * 0.32),
               stop_fn=lambda: bool(scene._l1[0]), max_steps=1800)
    report("black-eject")
    if not (ok and bool(scene._l1[0])):
        print("SIM_GEN_SOLVE: FAIL (black king was not ejected through the opening)",
              flush=True)
        os._exit(1)
    assert not bool(scene.success()[0])
    s1 = print_score("P1 black king ejected through the rim opening")
    assert s1 >= s0 - 1e-6 and s1 >= 0.24, f"P1 score {s1} (expect l1=0.25)"

    # ---------------- phase 2: gravity delivery into the capture box (hands-off) -----------
    streak = 0
    for i in range(900):
        step(1)
        if bool(scene.black_captured()[0]) \
                and float(scene.black.data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_speed:
            streak += 1
        else:
            streak = 0
        if streak >= 30:
            break
    report("black-boxed")
    if not bool(scene.black_captured()[0]):
        print("SIM_GEN_SOLVE: FAIL (black king did not settle inside the capture box)",
              flush=True)
        os._exit(1)
    s2 = print_score("P2 black king captured (settled inside the box)")
    assert s2 >= s1 - 1e-6 and s2 >= 0.54, f"P2 score {s2} (expect l1+l2=0.55)"

    # ---------------- phase 3: white king — transport teleport, contact seating ------------
    # TRANSPORT ONLY: park the white king 25 mm above the vacant dais with zero
    # velocity, then hands-off — the seating is a pure contact settle under gravity.
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0] = scene.throne[:, 0] + 0.003
    loc[:, 1] = scene.throne[:, 1]
    loc[:, 2] = c.board_h + c.dais_h + 0.025
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.board_to_world(loc)
    st[:, 3:7] = scene.platform.data.root_quat_w
    scene.white.write_root_state_to_sim(st, all_ids)
    streak = 0
    for i in range(600):
        step(1)
        if bool(scene.white_enthroned()[0]) and bool(scene.settled()[0]):
            streak += 1
        else:
            streak = 0
        if streak >= 30:
            break
    report("white-seated")
    if not (bool(scene.white_enthroned()[0]) and bool(scene.success()[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (white king did not seat on the dais)", flush=True)
        os._exit(1)
    s3 = print_score("P3 white king enthroned on the gold dais: capture + coronation done")
    assert s3 >= s2 - 1e-6 and s3 >= 0.99, f"P3 score {s3} (expect success=1.0)"

    # ---------------- phase 4: persistence (>= 3.4 simulated seconds, hands-off) -----------
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.42 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.4 s hands-off")
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
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
