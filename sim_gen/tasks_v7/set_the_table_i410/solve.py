"""Teleport solution for WobblyBistroScene (sim_gen task `set_the_table_i410`) —
the task's legitimacy certificate.

The task's load-bearing interactions and how they are executed:

1. DIAGNOSE (readback): after reset+settle the table rests LEVEL on its three long
   legs; the only sign of the fault is the short leg's foot hovering ~25 mm off the
   floor. The solve reads the hover gap and the short-corner direction from the
   settled pose (free 360 deg table yaw — the corner points anywhere).
2. STAGE THE WEDGE (teleport = transport only): one pose write parks the wedge ON
   THE FLOOR outside the short corner, thin tip pointing at the hovering foot,
   35 mm short of it. It is NOT written into the shored state (asserted).
3. SHORE THE LEG (contact dynamics — the heart of the repair): a velocity-limited
   horizontal force at the wedge CoM (the fingertip-push emulation: the wedge is
   pushed at floor level, no grasp) slides it tip-first under the hovering foot.
   A per-step readback of the foot position IN THE WEDGE FRAME tracks the ramp
   height under the foot; the force is CUT when that height reaches ~1.2 mm BELOW
   the foot (inside the scene's shored band). Deliberately NOT lifting the foot:
   the shim bears the moment the first dish load rocks the corner down onto it —
   the load transfer itself is physics. A runtime probe toggles the wrench
   pre-encoding if the pod's external-force API drags the frame.
4. SET THE TABLE (teleport hover + gravity): the plate, then the cup, is teleported
   to hover 20 mm ABOVE its own marked seat (transport; asserted not-yet-seated)
   and RELEASED. Each lands on the polished top under gravity; the first landing
   rocks the short corner down onto the wedge ramp (~0.35 deg transient) and the
   shim takes the load — the top stays level and the slick top KEEPS the dishes
   (tan(level_tol) << mu). Seating, stillness and levelness are all physics.
5. PERSISTENCE: >= 3.3 simulated seconds hands-off; success() must still hold.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
stage credit is streak-latched), then `SIM_GEN_SOLVE: SUCCESS` only if success()
holds through the hands-off window.

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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.wobbly_bistro")().build(num_envs=args.num_envs,
                                                   device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def foot_in_wedge() -> torch.Tensor:
        """(N,3) short-leg foot bottom centre, wedge body frame."""
        rel = scene.leg_tip_w() - scene.wedge.data.root_pos_w
        return quat_apply_inverse(scene.wedge.data.root_quat_w, rel)

    def report(tag: str) -> None:
        fw = foot_in_wedge()[0]
        print(f"[solve] {tag:12s} | tilt={float(scene.tilt_deg()[0]):.2f}deg "
              f"foot_z={float(scene.leg_tip_w()[0, 2]):.4f} "
              f"foot_in_wedge=({float(fw[0]):+.3f},{float(fw[1]):+.3f},"
              f"{float(fw[2]):.3f}) shored={bool(scene.shored()[0])} "
              f"shored_ever={bool(scene._shored_ever[0])} "
              f"plate={bool(scene.plate_seated()[0])} "
              f"cup={bool(scene.cup_seated()[0])} "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # ---------------- phase 0: reset, settle, diagnose the fault ---------------------------
    step(150)
    t_pos = (scene.table.data.root_pos_w - scene.env_origins)[0]
    tq = scene.table.data.root_quat_w[0]
    t_yaw = 2.0 * math.atan2(float(tq[3]), float(tq[0]))
    foot = scene.leg_tip_w()[0] - scene.env_origins[0]
    wp = (scene.wedge.data.root_pos_w - scene.env_origins)[0]
    pp = (scene.plate.data.root_pos_w - scene.env_origins)[0]
    cp = (scene.cup.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"table=({float(t_pos[0]):+.3f},{float(t_pos[1]):+.3f}) "
          f"yaw={math.degrees(t_yaw):+.1f}deg "
          f"foot=({float(foot[0]):+.3f},{float(foot[1]):+.3f}) "
          f"hover_gap={float(foot[2]) * 1000:.1f}mm "
          f"wedge=({float(wp[0]):+.3f},{float(wp[1]):+.3f}) "
          f"plate=({float(pp[0]):+.3f},{float(pp[1]):+.3f}) "
          f"cup=({float(cp[0]):+.3f},{float(cp[1]):+.3f})", flush=True)
    report("reset")
    # The DIAGNOSIS: empty table level, short foot hovering ~ the stub height.
    assert float(scene.tilt_deg()[0]) < c.level_tol_deg, "empty table should rest level"
    assert 0.015 < float(foot[2]) < 0.035, \
        f"short foot should hover ~{c.stub * 1000:.0f}mm, got {float(foot[2]) * 1000:.1f}mm"
    assert not bool(scene.shored()[0]), "shored at reset?!"
    s0 = print_score("P0 reset+settle: table level, short foot hovering")
    assert s0 <= 0.02, f"reset score should be ~0, got {s0}"

    # ---------------- phase 1: stage the wedge (teleport = transport only) ------------------
    # Push line: the table-frame (+1,+1) diagonal (centre -> short corner), world xy.
    diag = torch.tensor([1.0, 1.0, 0.0], device=device) / math.sqrt(2.0)
    s_dir = quat_apply(scene.table.data.root_quat_w,
                       diag.expand(n, 3)).clone()
    s_dir[:, 2] = 0.0
    s_dir = s_dir / s_dir.norm(dim=-1, keepdim=True)
    # Wedge +x = s_dir (heel outward), so its -x THIN TIP points at the foot; parked
    # on the floor 35 mm outside the foot, centred on the push line.
    yaw_w = math.atan2(float(s_dir[0, 1]), float(s_dir[0, 0]))
    foot_w = scene.leg_tip_w()
    st = torch.zeros(n, 13, device=device)
    st[:, 0:2] = foot_w[:, 0:2] + s_dir[:, 0:2] * (c.wedge_len / 2 + 0.035)
    st[:, 2] = 0.001 + scene.env_origins[:, 2]  # foot_w xy is already world
    st[:, 3] = math.cos(yaw_w / 2)
    st[:, 6] = math.sin(yaw_w / 2)
    scene.wedge.write_root_state_to_sim(st, all_ids)
    step(10)
    report("staged")
    assert not bool(scene.shored()[0]), \
        "staging pose already reads shored (teleport must not shore the leg)"
    s1 = print_score("P1 wedge staged on the floor, tip aimed at the hovering foot")

    # ---------------- phase 2: SHORE THE LEG (contact dynamics) -----------------------------
    # Velocity-limited horizontal push at the wedge CoM, along -s_dir (tip-first,
    # inward). The 30 mm foot meets the incline at its heel-side EDGE, so the cut
    # targets the ramp height under that edge reaching stub - 2 mm: the shim then
    # sits in the shored band WITHOUT lifting the foot (lifting needs ~3.7 N
    # vertical and could skid the table); the first dish load rocks the foot the
    # last ~2 mm down onto the ramp — load transfer by physics. If the slide stalls
    # bearing against the foot edge past x=0, that IS the inserted state.
    slope = (c.wedge_hh - c.wedge_h0) / c.wedge_len
    edge_off = c.leg_w / 2 - 0.002  # foot-centre -> bearing-edge offset
    x_target = (c.stub - 0.0020 - c.wedge_h0) / slope - c.wedge_len / 2 - edge_off
    x_accept = 0.000  # a bearing stall past here already satisfies the shored band
    x_abort = c.wedge_len / 2 - 0.010
    push_dir = -s_dir
    q_ref = scene.wedge.data.root_quat_w.clone()
    encode_mode = 0  # 0: raw world force; 1: pre-rotate by R_ref * R_now^T

    def encode(f_w: torch.Tensor) -> torch.Tensor:
        if encode_mode == 0:
            return f_w
        q_now = scene.wedge.data.root_quat_w
        return quat_apply(quat_mul(q_ref, quat_conjugate(q_now)), f_w)

    f_push, v_des = 0.8, 0.05
    best_x = float(foot_in_wedge()[0, 0])
    start_x, last_bump, stalls = best_x, 0, 0
    inserted = False
    for i in range(1500):
        x = float(foot_in_wedge()[0, 0])
        if x >= x_target or x >= x_abort:
            inserted = True
            break
        vel = scene.wedge.data.root_lin_vel_w[0]
        v_along = float((vel * push_dir[0]).sum())
        f_mag = f_push if v_along < v_des else 0.0
        f_w = (push_dir * f_mag).view(n, 1, 3)
        scene.wedge.set_external_force_and_torque(encode(f_w).contiguous(),
                                                  zero_wrench, env_ids=all_ids,
                                                  is_global=True)
        env.step(no_action)
        if i % 150 == 0:
            print(f"[solve] push+{i:4d}: x={x:+.4f} foot_z="
                  f"{float(scene.leg_tip_w()[0, 2]):.4f} "
                  f"tilt={float(scene.tilt_deg()[0]):.2f} f={f_push:.1f}",
                  flush=True)
        if x > best_x + 0.002:
            best_x, last_bump, stalls = x, i, 0
        if x < start_x - 0.010 and encode_mode == 0:
            encode_mode = 1  # pushed the wrong way: the pod drags the wrench frame
            print(f"[solve] frame-drag detected (x={x:+.3f} < start) -> "
                  f"pre-encode mode 1", flush=True)
            last_bump = i
        elif i - last_bump > 150:  # stalled
            if x >= x_accept:
                # bearing stall: the ramp is up against the foot's edge — the shim
                # is in the shored band; grinding further would lift the table
                print(f"[solve] bearing stall at x={x:+.3f} -> accept inserted",
                      flush=True)
                inserted = True
                break
            stalls += 1
            last_bump = i
            if stalls >= 3:
                encode_mode ^= 1
                f_push, stalls = 0.8, 0
                print(f"[solve] stall persists at x={x:+.3f} -> toggle pre-encode "
                      f"to mode {encode_mode}", flush=True)
            else:
                f_push = min(f_push + 0.6, 2.0)  # stay below the table-skid budget
                print(f"[solve] stall at x={x:+.3f} -> f_push={f_push:.1f} N",
                      flush=True)
    scene.wedge.set_external_force_and_torque(zero_wrench, zero_wrench,
                                              env_ids=all_ids)
    report("inserted")
    assert inserted, "push never carried the wedge tip under the hovering foot"
    step(180)  # 1.5 s: wedge stops, streak latch accrues
    report("shored")
    assert bool(scene.shored()[0]), "wedge in place but shored() not satisfied"
    assert bool(scene._shored_ever[0]), "shoring did not latch"
    s2 = print_score("P2 wedge pushed under the foot — short leg shored, force cut")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_shore - 1e-6, "shoring credit missing"

    # ---------------- phase 3: SET THE PLATE (teleport hover + gravity) ---------------------
    # Transport: hover 20 mm above the red-ring seat, table frame. Release: it lands
    # on the polished top; its weight rocks the short corner down ONTO the wedge —
    # the shim takes the load and the top stays level (this is where an unshored
    # table sheds the dish instead).
    t_pos_w = scene.table.data.root_pos_w
    t_quat_w = scene.table.data.root_quat_w
    hover = torch.tensor([c.seat_plate[0], c.seat_plate[1],
                          c.top_face_z + c.plate_h / 2 + 0.020],
                         device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = t_pos_w + quat_apply(t_quat_w, hover)
    st[:, 3:7] = t_quat_w
    scene.plate.write_root_state_to_sim(st, all_ids)
    step(2)
    assert not bool(scene._plate_ever[0]), \
        "hover pose already latched plate credit (teleport must not seat the dish)"
    seated = False
    for i in range(480):
        env.step(no_action)
        if bool(scene._plate_ever[0]) and bool(scene.plate_seated()[0]):
            seated = True
            break
    report("plate")
    assert seated, "plate did not settle seated on the red ring with the top level"
    s3 = print_score("P3 plate released over the red ring — landed, table held level")
    assert s3 >= s2 - 1e-6 and s3 >= c.w_shore + c.w_plate - 1e-6, \
        "plate credit missing"

    # ---------------- phase 4: SET THE CUP (teleport hover + gravity) -----------------------
    t_pos_w = scene.table.data.root_pos_w
    t_quat_w = scene.table.data.root_quat_w
    hover = torch.tensor([c.seat_cup[0], c.seat_cup[1],
                          c.top_face_z + c.cup_h / 2 + 0.020],
                         device=device).expand(n, 3)
    st = torch.zeros(n, 13, device=device)
    st[:, 0:3] = t_pos_w + quat_apply(t_quat_w, hover)
    st[:, 3:7] = t_quat_w
    scene.cup.write_root_state_to_sim(st, all_ids)
    step(2)
    assert not bool(scene._cup_ever[0]), \
        "hover pose already latched cup credit (teleport must not seat the dish)"
    done = False
    for i in range(480):
        env.step(no_action)
        if bool(scene._cup_ever[0]) and bool(scene.success()[0]):
            done = True
            break
    report("cup")
    assert done, "cup did not settle seated on the blue disc / success not reached"
    s4 = print_score("P4 cup released over the blue disc — table set, success")
    assert s4 >= s3 - 1e-6, "score decreased across the cup placement"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after settling)", flush=True)
        os._exit(1)

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    except Exception as exc:  # noqa: BLE001 — die fast, Kit teardown hangs
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL ({exc})", flush=True)
        os._exit(1)
