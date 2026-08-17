"""Teleport solution for BowlLetterboxScene (sim_gen task
`libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i212`)
— the task's legitimacy certificate.

The task's load-bearing interaction and how each phase is executed:

1. PICK + REORIENT (held-object transport): the bowl is lifted off the floor and
   rotated 90 deg about the cabinet's x-axis — from flat to ON EDGE like a coin —
   by per-step pose writes with path-consistent linear/angular velocities (the
   emulation of a wrist-roll while the gripper pinches the rim). Transport only:
   nothing else moves, nothing is judged on this except the latched pose milestone.
2. STAGE (transport): the on-edge bowl is carried around the front of the cabinet
   and lowered onto the APRON shelf, centred on the lower slot, then RELEASED
   (writes stop) — it stands on its own flat (a metastable but real rest).
3. POST THROUGH THE SLOT (contact dynamics — the load-bearing interaction, never
   teleported): a bounded (<= 10 N) horizontal servo force along the cabinet's +x
   axis (the applied-wrench emulation of the arm pushing the bowl's trailing edge)
   slides/rolls the standing bowl across the apron, THROUGH the 70 mm vertical slot
   (48 mm dish + ~2 cm clearance) and into the sealed bottom compartment. The drive
   is then removed; the bowl settles inside under gravity/contacts alone. The bowl's
   pose is never written after the stage release; success() is judged on the settled
   physical state.
4. PERSISTENCE: >= 3 simulated seconds hands-off after success() first holds.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched) and `SIM_GEN_SOLVE: SUCCESS` only if success persists.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True  # never keep a dead interpreter alive waiting for the timer
_wd.start()


def main() -> None:
    from isaaclab.utils.math import (
        quat_apply,
        quat_apply_inverse,
        quat_from_angle_axis,
        quat_mul,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.bowl_letterbox")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    dt = 1.0 / 120.0

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    all_ids = torch.arange(n, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def loc0() -> list:
        return [float(v) for v in scene.bowl_local()[0]]

    def axz() -> float:
        return float(scene.bowl_axis_z()[0])

    def report(tag: str) -> None:
        lx, ly, lz = loc0()
        print(f"[solve] {tag:12s} | loc=({lx:+.3f},{ly:+.3f},{lz:.3f}) axz={axz():+.2f} "
              f"near={float(scene._near_max[0]):.3f} posed={bool(scene._posed[0])} "
              f"ins={float(scene._ins_max[0]):.3f} in_cell0={bool(scene.bowl_in_cell(0)[0])} "
              f"still={bool(scene.bowl_still()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ----- cabinet frame (kinematic, fixed after reset) -----
    cab_p = scene.cabinet.data.root_pos_w[0].clone()
    cab_q = scene.cabinet.data.root_quat_w[0].clone()

    def to_world(local) -> torch.Tensor:
        lp = torch.tensor(local, device=device, dtype=torch.float32)
        return cab_p + quat_apply(cab_q.unsqueeze(0), lp.unsqueeze(0))[0]

    x_cab_w = quat_apply(cab_q.unsqueeze(0),
                         torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]

    # ----- held-bowl transport helpers (pose written per step; transport only) -----
    def hold_write(pos_w: torch.Tensor, quat: torch.Tensor,
                   lin_vel=None, ang_vel=None) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat
        if lin_vel is not None:
            st[:, 7:10] = lin_vel
        if ang_vel is not None:
            st[:, 10:13] = ang_vel
        scene.bowl.write_root_state_to_sim(st, all_ids)

    def bowl_pos_w() -> torch.Tensor:
        return scene.bowl.data.root_pos_w[0].clone()

    def carry_to(target_w: torch.Tensor, steps: int, quat: torch.Tensor) -> None:
        """Move the held bowl along a smoothstep path at a fixed orientation."""
        p0 = bowl_pos_w()
        for i in range(steps):
            s = (i + 1) / steps
            w = s * s * (3 - 2 * s)  # smoothstep
            dw = 6 * s * (1 - s) / steps  # d(w)/d(step)
            pos = p0 + (target_w - p0) * w
            vel = (target_w - p0) * dw / dt
            hold_write(pos, quat, lin_vel=vel)
            env.step(no_action)

    def reorient(center_w: torch.Tensor, q0: torch.Tensor, ang_total: float,
                 steps: int) -> torch.Tensor:
        """Rotate the held bowl in place about the cabinet's x-axis (wrist roll)."""
        q1 = q0
        for i in range(steps):
            s = (i + 1) / steps
            w = s * s * (3 - 2 * s)
            dw = 6 * s * (1 - s) / steps
            th = ang_total * w
            omega = ang_total * dw / dt
            q1 = quat_mul(
                quat_from_angle_axis(torch.tensor([th], device=device),
                                     x_cab_w.unsqueeze(0)),
                q0.unsqueeze(0))[0]
            hold_write(center_w, q1, ang_vel=x_cab_w * omega)
            env.step(no_action)
        return q1

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    lx, ly, lz = loc0()
    print(f"[solve] layout readback (seed {args.seed}): bowl_local=({lx:+.3f},{ly:+.3f}) "
          f"axis_z={axz():+.2f}", flush=True)
    report("reset")
    assert lx < -0.25, f"bowl did not spawn on the floor in front ({lx:+.3f})"
    assert abs(axz()) > 0.9, "bowl did not settle flat"
    assert not bool(scene.success()[0]), "success at reset"
    s0 = print_score("P0 reset+settle")
    assert s0 <= 0.02, f"score not ~0 at reset ({s0:.3f})"

    # ---------------- phase 1: PICK + REORIENT on edge (transport) --------------------------
    p_spawn = bowl_pos_w()
    q0 = scene.bowl.data.root_quat_w[0].clone()
    lift = p_spawn.clone()
    lift[2] = 0.30
    carry_to(lift, 150, q0)  # lift straight up
    q_edge = reorient(lift, q0, math.pi / 2, 240)  # wrist-roll 90 deg: flat -> on edge
    report("reoriented")
    assert abs(axz()) < 0.2, f"bowl not on edge after reorient (axz={axz():+.2f})"

    # ---------------- phase 2: STAGE on the apron in front of the lower slot ----------------
    stage_z = c.floor_z + c.bowl_out_r  # resting on an octagon flat
    hover = to_world((-0.115, 0.0, 0.30))
    stage = to_world((-0.115, 0.0, stage_z + 0.004))
    carry_to(hover, 300, q_edge)   # traverse around the front, above the apron
    carry_to(stage, 200, q_edge)   # lower onto the apron
    hold_write(to_world((-0.115, 0.0, stage_z + 0.002)), q_edge)
    # release: no further pose writes for the rest of the run
    step(150)
    report("staged")
    lx, ly, lz = loc0()
    assert abs(axz()) < 0.35, f"bowl fell over on the apron (axz={axz():+.2f})"
    assert abs(lx + 0.115) < 0.05, f"bowl not staged before the slot (x={lx:+.3f})"
    assert abs(ly) < 0.04, f"bowl not centred on the slot (y={ly:+.3f})"
    assert bool(scene._posed[0]), "posed latch did not fire"
    assert not bool(scene.success()[0]), "success before insertion"
    s1 = print_score("P1 bowl on edge, standing on the apron before the lower slot")
    assert s1 >= s0 - 1e-6, "score decreased across staging"

    # ---------------- phase 3: POST through the slot (contact dynamics) ---------------------
    # Bounded horizontal servo force along the cabinet's +x axis; the bowl slides or
    # rolls across the apron, through the slot, into the compartment. Gain escalates
    # on stall (never the 10 N cap); the bowl is never pose-written here.
    x_t = 0.15
    kp, kd, fmax = 40.0, 4.0, 10.0
    last_x, stall_ref, entered = None, None, False
    for i in range(1500):
        loc = scene.bowl_local()[0]
        x = float(loc[0])
        v_loc = quat_apply_inverse(cab_q.unsqueeze(0),
                                   scene.bowl.data.root_lin_vel_w[:1])[0]
        vx = float(v_loc[0])
        f = kp * (x_t - x) - kd * vx
        scene.push_drive[:] = max(0.0, min(fmax, f))
        env.step(no_action)
        if x >= 0.13:
            entered = True
            break
        if i % 200 == 199:
            if stall_ref is not None and x - stall_ref < 0.005:
                kp *= 1.6
                print(f"[solve] push stalled at x={x:+.3f}; kp -> {kp:.0f}", flush=True)
            stall_ref = x
        last_x = x
    scene.push_drive[:] = 0.0
    assert entered, f"bowl never passed the slot (x={last_x:+.3f})"
    # settle inside: wait for a stillness streak, hands off
    streak = 0
    for _ in range(600):
        env.step(no_action)
        streak = streak + 1 if bool(scene.bowl_still()[0]) else 0
        if streak >= 60:
            break
    report("posted")
    assert bool(scene.bowl_in_cell(0)[0]), "bowl did not settle fully inside the bottom cell"
    s2 = print_score("P2 bowl pushed through the slot, settled inside the bottom compartment")
    assert s2 >= s1 - 1e-6, "score decreased across insertion"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after insertion)", flush=True)
        os._exit(1)

    # ---------------- phase 4: persistence (>= 3 simulated seconds, no intervention) --------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s3 = print_score("P3 persistence 3.3 s")
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
    except BaseException:  # noqa: BLE001 - Kit threads would hang the interpreter
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)
