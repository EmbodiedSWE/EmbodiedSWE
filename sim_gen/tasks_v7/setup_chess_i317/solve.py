"""Teleport solution for MoveClockScene (sim_gen task `setup_chess_i317`) — the
task's legitimacy certificate.

The PUNCH — the load-bearing interaction of the task — is executed entirely
through contact/joint dynamics: a body-frame torque about the rocker's hinge axis
(0.030 N*m — exactly the wrench a Franka fingertip pressing the raised button cap
transmits: ~0.36 N of downward push at the 83 mm button lever, about 3.2x the
9.5e-3 N*m gravity holding torque) drives the beam off its start stop, through
over-center, onto the far stop, with a bang-bang speed governor (~0.8 rad/s) so
nothing is slammed. No teleport ever touches the rocker. The queen's TRANSPORT is
the one teleport (carrying through free space, the legal use): it is parked 20 mm
ABOVE the promotion-square centre and RELEASED — the seating is a real contact
settle onto the board under gravity, judged as a settled upright pose.

Phases (the declared chess rule fixes the order; the solve obeys it):
  P0  settle + layout readback (board pose, promotion square, queen spot, rocker
      start side, masses); baseline score ~0.
  P1  MOVE: teleport the queen to 20 mm above the promotion-square centre,
      release, let it seat itself by contact (l_seat latches, 0.45).
  P2  PUNCH: press the raised clock button (hinge torque) until the beam tips
      fully onto the far side; coast onto the stop, settle -> success, 1.0.
  P3  >= 3.4 simulated seconds HANDS-OFF; print SIM_GEN_SOLVE: SUCCESS only if
      success() still holds live.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's partial credit is latched).

Run (forge): python -u -m simgen_tasks.setup_chess_i317.solve --headless [--seed N]
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

_ = scene_mod  # imported for its SCENES/ENVS registration side effect

# Global watchdog (daemon): if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.move_clock")().build(num_envs=args.num_envs, device=device)
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

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        ql = scene.board_local(scene.queen.data.root_pos_w)
        ang = math.degrees(float(scene.rocker_angle()[0]))
        qv = float(scene.queen.data.root_lin_vel_w.norm(dim=-1)[0])
        rw = float(scene.rocker.data.root_ang_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:12s} | queen_board=({float(ql[0, 0]):+.3f},"
              f"{float(ql[0, 1]):+.3f},{float(ql[0, 2]):+.3f}) "
              f"rocker={ang:+.1f}deg qv={qv:.3f} rw={rw:.3f} "
              f"seat={bool(scene._l_seat[0])} flip={bool(scene._l_flip[0])} "
              f"foul={bool(scene._foul[0])} seated={bool(scene.seated()[0])} "
              f"flipped={bool(scene.flipped()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, readback -------------------------------------
    step(180)
    bp = (scene.board.data.root_pos_w - scene.env_origins)[0]
    bq = scene.board.data.root_quat_w[0]
    byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
    fx, fy = float(scene.frame_xy[0, 0]), float(scene.frame_xy[0, 1])
    fl = scene.board_local(scene.frame.data.root_pos_w)[0]
    q0 = scene.board_local(scene.queen.data.root_pos_w)[0]
    s0_side = float(scene.start_side[0])
    ang0 = float(scene.rocker_angle()[0])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"board=({float(bp[0]):+.3f},{float(bp[1]):+.3f}) yaw={byaw:+.1f}deg "
          f"frame_xy=({fx:+.3f},{fy:+.3f}) frame_body=({float(fl[0]):+.3f},"
          f"{float(fl[1]):+.3f},{float(fl[2]):+.3f}) "
          f"queen=({float(q0[0]):+.3f},{float(q0[1]):+.3f},{float(q0[2]):+.3f}) "
          f"start_side={'ivory(+y)' if s0_side > 0 else 'black(-y)'} "
          f"rocker={math.degrees(ang0):+.1f}deg", flush=True)
    report("reset")
    assert torch.isfinite(scene.queen.data.root_state_w).all(), "NaN in queen state"
    assert torch.isfinite(scene.rocker.data.root_state_w).all(), "NaN in rocker state"
    mq = float(scene.queen.root_physx_view.get_masses().reshape(-1)[0])
    mr = float(scene.rocker.root_physx_view.get_masses().reshape(-1)[0])
    print(f"[solve] mass readback: queen={mq:.3f}kg rocker={mr:.3f}kg", flush=True)
    assert 0.10 < mq < 0.20 and 0.10 < mr < 0.20, "authored masses not applied"
    # layout invariants: frame body sits at frame_xy on the board top; queen stands
    # on the OPPOSITE half; rocker is held on its start stop by gravity
    assert abs(float(fl[0]) - fx) < 0.005 and abs(float(fl[1]) - fy) < 0.005, \
        "frame body must sit at the sampled promotion-square spot"
    assert abs(float(fl[2]) - c.board_h) < 0.005, "frame must lie on the board top"
    assert fy * float(q0[1]) < 0.0, "queen must start on the opposite half"
    assert abs(float(q0[2]) - c.board_h) < 0.006, "queen must stand on the board"
    assert bool(scene.upright(scene.queen)[0]), "queen must start upright"
    assert s0_side * ang0 > math.radians(c.flip_deg), \
        "rocker must rest on its start stop"
    s0 = print_score("P0 reset+settle (move not yet made, clock not yet punched)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: MOVE — transport teleport, contact seating ------------------
    # TRANSPORT ONLY: park the queen 20 mm above the promotion-square centre with
    # zero velocity, then hands-off — the seating is a pure contact settle under
    # gravity through the frame's opening onto the board top.
    loc = torch.zeros(n, 3, device=device)
    loc[:, 0] = scene.frame_xy[:, 0]
    loc[:, 1] = scene.frame_xy[:, 1]
    loc[:, 2] = c.board_h + 0.020
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = scene.board_to_world(loc)
    st[:, 3:7] = scene.board.data.root_quat_w
    scene.queen.write_root_state_to_sim(st, all_ids)
    streak = 0
    for _ in range(600):
        step(1)
        if bool(scene.seated()[0]) \
                and float(scene.queen.data.root_lin_vel_w.norm(dim=-1)[0]) < c.settle_speed:
            streak += 1
        else:
            streak = 0
        if streak >= 30:
            break
    report("queen-seated")
    if not (bool(scene.seated()[0]) and bool(scene._l_seat[0])):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (queen did not seat in the promotion square)",
              flush=True)
        os._exit(1)
    assert not bool(scene.success()[0]), "seat alone must not be success"
    s1 = print_score("P1 queen seated in the promotion square (move made)")
    assert s1 >= s0 - 1e-6 and s1 >= 0.44, f"P1 score {s1} (expect l_seat=0.45)"

    # ---------------- phase 2: PUNCH — press the raised button through the hinge -----------
    # Body-frame torque about the hinge axis (the rocker only ever rolls about x,
    # so body x == hinge axis), sign toward the far side, bang-bang speed governor
    # so the beam crosses over at <= ~0.8 rad/s. Cut the wrench the instant
    # flipped() fires (3 deg short of the stop) and let it coast on.
    tau = 0.030   # N*m ~ 0.36 N fingertip press at the 83 mm button lever
    w_cap = 0.8   # rad/s
    torque = torch.zeros(n, 1, 3, device=device)
    tipped = False
    for i in range(1200):
        if bool(scene.flipped()[0]):
            tipped = True
            break
        w_hinge = scene.rocker.data.root_ang_vel_w[:, 0]
        prog = (-scene.start_side) * w_hinge  # rad/s toward the far side
        on = (prog < w_cap).float()
        torque[:, 0, 0] = (-scene.start_side) * tau * on
        scene.rocker.set_external_force_and_torque(zero_wrench, torque, env_ids=all_ids)
        env.step(no_action)
        if i % 240 == 239:
            print(f"[solve] pressing: rocker={math.degrees(float(scene.rocker_angle()[0])):+.1f}deg",
                  flush=True)
    clear_wrench(scene.rocker)
    report("clock-tipped")
    if not tipped:
        print("SIM_GEN_SOLVE: FAIL (rocker never tipped past the crossover)", flush=True)
        os._exit(1)
    # hands-off: coast onto the far stop and settle (gravity holds the toggle there)
    streak = 0
    for _ in range(600):
        step(1)
        if bool(scene.success()[0]):
            streak += 1
        else:
            streak = 0
        if streak >= 30:
            break
    report("clock-settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (flip did not settle into success)", flush=True)
        os._exit(1)
    s2 = print_score("P2 clock punched legally (beam on the far stop): move + punch done")
    assert s2 >= s1 - 1e-6 and s2 >= 0.99, f"P2 score {s2} (expect success=1.0)"
    assert not bool(scene._foul[0]), "legal punch must not have fouled"

    # ---------------- phase 3: persistence (>= 3.4 simulated seconds, hands-off) -----------
    hold = True
    for _ in range(10):  # 10 x 41 steps = 410 substeps = 3.42 s at 120 Hz
        step(41)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.4 s hands-off")
    ok = hold and bool(scene.success()[0]) and s3 >= s2 - 1e-6
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
