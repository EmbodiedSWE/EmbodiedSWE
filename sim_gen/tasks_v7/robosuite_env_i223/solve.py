"""Teleport-free solution for SliderGauntletScene (sim_gen task `robosuite_env_i223`) —
the task's legitimacy certificate.

Nothing is ever teleported after reset: every piece is CAPTIVE (the roof makes lifting
geometrically impossible), so the whole solve is contact dragging — a world-frame
velocity-servo force at each piece's CoM, exactly the horizontal pull a gripper holding
the mast would exert:

1. CLEAR STATION A: read the open-arm side from the scene's sampled layout (the plug's
   board-local y sign — the same bit a camera reads from the open-topped arms), then
   servo-drag blocker A along its cross corridor into the open pocket until its centre
   is >= 95 mm off-axis (clear gate: 80 mm). Release; the streak latch arms.
2. CLEAR STATION B: same for blocker B.
3. RUN THE GAUNTLET: servo-drag the runner along board +x, past both (now open)
   stations, over the exit lip; force is cut the moment it starts dropping, and
   gravity + momentum land it in the catch tray. Settle, hands off.

Servo sizing (see the module docstring in scene.py for masses/frictions):
  - stability: kp * dt / m must stay well under 1 (wrenches act one substep late) ->
    kp = 9 for the 0.15 kg blockers (0.5), kp = 12 for the 0.20 kg runner (0.5);
  - anti-stall: the stall-force kp * v_des must beat sliding friction (mu*m*g =
    0.51 N / 0.69 N) -> v_des starts at 0.10 m/s (stall force 0.9 N / 1.2 N) and
    ESCALATES on stall (raising v_des raises the stall force without touching the
    stability bound);
  - no tipping: force cap 2.5 N is far below every m*g*b/h_com tipping threshold.

If the pod's external-force frame drag defeats the raw world-frame push (R_ref stale
across resets — the board is yawed up to 90+/-30 deg from spawn), the push is retried
pre-encoded with M = R_ref @ R_now^T.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success() still holds.

Run (forge): python -u -m simgen_tasks.robosuite_env_i223.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401 — import registers the scene + env
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import matrix_from_quat, quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.slider_gauntlet")().build(num_envs=args.num_envs, device=device)
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

    def report(tag: str) -> None:
        rl = scene.runner_loc()[0]
        bl = scene.blocker_loc()[0]
        print(f"[solve] {tag:12s} | runner=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
              f"{float(rl[2]):+.3f}) "
              f"blkA=({float(bl[0, 0]):+.3f},{float(bl[0, 1]):+.3f}) "
              f"blkB=({float(bl[1, 0]):+.3f},{float(bl[1, 1]):+.3f}) "
              f"obst={scene.obstructing()[0].tolist()} clear={scene._clear[0].tolist()} "
              f"prog={float(scene._prog[0]):.3f} in_tray={bool(scene.in_tray()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def clear_force(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def servo_push(body, dir_loc, stop_fn, *, kp: float, v_des: float, fmax: float,
                   max_steps: int, mode: int, tag: str) -> bool:
        """World-frame velocity servo along the board-local direction `dir_loc`.
        mode 0 = raw global force; mode 1 = pre-encoded with M = R_ref @ R_now^T
        (the pod wrench frame-drag fallback). Returns True when stop_fn() fires;
        False on stall (no displacement over a 90-step window at full authority)."""
        d_loc = torch.tensor(dir_loc, device=device, dtype=torch.float32).expand(n, 3)
        r_ref = matrix_from_quat(body.data.root_quat_w)
        last_pos = body.data.root_pos_w[0].clone()
        stall = 0
        for i in range(max_steps):
            d_w = quat_apply(scene.board.data.root_quat_w, d_loc)
            v_along = (body.data.root_lin_vel_w * d_w).sum(dim=-1)
            f_mag = (kp * (v_des - v_along)).clamp(-fmax, fmax)
            f_des = d_w * f_mag.unsqueeze(-1)
            if mode == 1:
                m_enc = r_ref @ matrix_from_quat(body.data.root_quat_w).transpose(1, 2)
                f_des = (m_enc @ f_des.unsqueeze(-1)).squeeze(-1)
            body.set_external_force_and_torque(
                f_des.unsqueeze(1), zero_wrench, env_ids=all_ids, is_global=True)
            env.step(no_action)
            if stop_fn():
                clear_force(body)
                return True
            if i % 90 == 89:
                moved = float((body.data.root_pos_w[0] - last_pos).norm())
                last_pos = body.data.root_pos_w[0].clone()
                stall = stall + 1 if moved < 0.004 else 0
                if stall >= 2:
                    print(f"[solve] {tag}: stalled (moved {moved * 1000:.1f} mm in "
                          f"90 steps, mode={mode} v_des={v_des:.2f})", flush=True)
                    clear_force(body)
                    return False
        print(f"[solve] {tag}: max_steps hit (mode={mode} v_des={v_des:.2f})", flush=True)
        clear_force(body)
        return False

    def drag(body, dir_loc, stop_fn, *, kp: float, fmax: float = 2.5,
             tag: str) -> bool:
        """Escalating drag: raise v_des on stall (raises the anti-stall force,
        keeps kp*dt/m fixed); if raw world-frame pushes never move the piece,
        fall back to the frame-drag encoding."""
        for mode in (0, 1):
            for v_des in (0.10, 0.15, 0.22):
                if stop_fn():
                    return True
                if servo_push(body, dir_loc, stop_fn, kp=kp, v_des=v_des, fmax=fmax,
                              max_steps=1600, mode=mode, tag=tag):
                    return True
            print(f"[solve] {tag}: mode {mode} exhausted", flush=True)
        return False

    # ---------------- phase 0: reset, settle, baseline ------------------------------------------
    step(180)
    bpos = (scene.board.data.root_pos_w - scene.env_origins)[0]
    bq = scene.board.data.root_quat_w[0]
    byaw = math.degrees(2.0 * math.atan2(float(bq[3]), float(bq[0])))
    sides = scene.open_side[0].tolist()
    # perception stand-in: the open side is the sign OPPOSITE the plug's local y
    plug_y = [float(scene._board_local(scene.plugs[p].data.root_pos_w)[0, 1])
              for p in scene.PLUG_NAMES]
    rl = scene.runner_loc()[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"board=({float(bpos[0]):+.3f},{float(bpos[1]):+.3f}) yaw={byaw:+.1f}deg "
          f"open_side={sides} plug_y=({plug_y[0]:+.3f},{plug_y[1]:+.3f}) "
          f"runner_x0={float(rl[0]):+.3f}", flush=True)
    for i in range(2):
        assert plug_y[i] * sides[i] < 0, \
            f"plug {i} must fill the CLOSED arm (plug_y={plug_y[i]}, open={sides[i]})"
    masses = scene.runner.root_physx_view.get_masses()
    assert abs(float(masses.flatten()[0]) - c.runner_mass) < 0.02, \
        f"authored runner mass did not take: {masses}"
    report("reset")
    assert bool(scene.obstructing()[0].all()), "both stations must start sealed"
    assert torch.isfinite(scene.runner.data.root_pos_w).all(), "NaN after settle"
    s0 = print_score("P0 reset+settle (corridor double-blocked)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phases 1+2: shunt each blocker into its open pocket -----------------------
    scores = [s0]
    for i, bname in enumerate(scene.BLOCKER_NAMES):
        body = scene.blockers[bname]
        side = float(sides[i])

        def cleared(idx: int = i, sd: float = side):
            yb = scene.blocker_loc()[0, idx, 1]
            return float(yb) * sd >= 0.095

        ok = drag(body, (0.0, side, 0.0), cleared, kp=9.0,
                  tag=f"blocker {bname} -> side {side:+.0f}")
        if not ok:
            report(f"STALL-{bname}")
            print(f"SIM_GEN_SOLVE: FAIL (blocker {bname} never cleared)", flush=True)
            os._exit(1)
        step(60)  # release; quiet streak (20 substeps) arms the clear latch
        report(f"clear-{bname}")
        assert not bool(scene.obstructing()[0, i]), f"station {i} must be open"
        assert bool(scene._clear[0, i]), f"clear latch {i} must be armed"
        s = print_score(f"P{i + 1} station {'AB'[i]} cleared (blocker in its pocket)")
        assert s >= scores[-1] - 1e-6 and s >= c.w_clear * (i + 1) - 1e-6, \
            f"P{i + 1} score {s} (expect >= {c.w_clear * (i + 1)})"
        scores.append(s)

    # ---------------- phase 3: drag the runner down the corridor and out ------------------------
    def out_or_dropping():
        loc = scene.runner_loc()[0]
        return float(loc[0]) > 0.215 or float(loc[2]) < 0.038

    ok = drag(scene.runner, (1.0, 0.0, 0.0), out_or_dropping, kp=12.0,
              tag="runner -> exit")
    if not ok:
        report("STALL-runner")
        print("SIM_GEN_SOLVE: FAIL (runner never reached the exit)", flush=True)
        os._exit(1)
    step(300)  # the drop, the landing, the settle — all hands-off
    report("in-tray")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the run)", flush=True)
        os._exit(1)
    s3 = print_score("P3 runner dropped into the catch tray")
    assert s3 >= scores[-1] - 1e-6, "score decreased across the run"

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) ----------------
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
    try:
        main()
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — die loudly, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
