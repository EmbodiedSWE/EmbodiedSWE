# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Scripted solver for Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0.

Strategy B — bimanual handoff with corner grip at LOW y.

The bimanual pin (left holds the wheel down while the right grasps, then left releases
while the right pins) is the only thing that PREVENTS the lift-fling (wheel stays low).
Its past failure was the right's forward reach dropping to y~0.30 when both arms are
forward, so it missed the wheel. FIX: keep the wheel at VERY LOW y and have the right
grip the wheel's near/body-side CORNER (low y, releases cleanly).

  1. LEFT grasps near-center (y~0.40), drags the wheel to (x~0.12, y~0.34) actively
     pulling to LOW y, then HOLDS (closed, low).
  2. Read wheel (cx,cy). RIGHT high-traverse approach, grasps the body-side corner at
     (x~cx-0.13, y~max(cy-0.10,0.30)). LEFT keeps holding (cl=True) throughout.
  3. LEFT releases (open) + lifts away while RIGHT keeps gripping (cr=True) -> right pins,
     then LEFT parks far back-and-low (-0.42,0.05,0.80) to free the shared waist for the right.
  4. RIGHT LIFTS the corner-gripped wheel to CARRY_Z~1.00 and carries it +x AT HEIGHT, then
     lowers and places it. KEY UNLOCK: the right wrist's +x reach is only ~0.23 flat on the
     table but ~0.44 when raised (z>=1.0), so carrying raised lets the hub reach x>0.40.
  5. RIGHT opens (corner grip releases cleanly), lifts, retracts the right wrist to x<0.26. Settle.

Success (task_done_pick_place, world==env frame since num_envs=1):
  object x in (0.40,0.85), y in (0.35,0.60), z<1.10, |vel|<0.20 all axes, right_wrist_x<0.26.
Object must never drop below z=0.5.
"""

import sys

if sys.platform != "win32":
    import pinocchio  # noqa: F401

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1200)
parser.add_argument("--out_dir", type=str, default="Opus/G1_Upper_body_IK")
parser.add_argument("--skip_left", action="store_true", default=False,
                    help="teleport the wheel to the handoff spot and test only the right hand")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import numpy as np
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"

# Orientations (wxyz). Probe confirmed rotY+90 makes the LEFT hand fingers point -Z (down).
Q_DOWN = [0.7071, 0.0, 0.7071, 0.0]
Q_FWD = [0.7071, 0.0, 0.0, 0.7071]   # home orientation: fingers point +Y (forward/horizontal)
HOME_L = np.array([-0.18, 0.10, 0.80])
HOME_R = np.array([0.18, 0.10, 0.80])


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    # Drive the whole scripted episode without auto-reset; we score success manually.
    for term in ("time_out", "object_dropping", "success"):
        if hasattr(env_cfg.terminations, term):
            setattr(env_cfg.terminations, term, None)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(TASK, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        os.makedirs(args_cli.out_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.out_dir,
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
            name_prefix="g1_pickplace",
        )

    e = env.unwrapped
    device = e.device
    robot = e.scene["robot"]
    obj = e.scene["object"]
    body_names = robot.data.body_names
    li = body_names.index("left_wrist_yaw_link")
    ri = body_names.index("right_wrist_yaw_link")
    origin = e.scene.env_origins[0]
    hand_names = e.action_manager.get_term("upper_body_ik")._hand_joint_names

    env.reset()
    if args_cli.video:
        e.sim.set_camera_view(eye=[2.2, -1.6, 1.7], target=[0.20, 0.45, 0.80])

    from isaaclab.utils.math import matrix_from_quat

    def opos(idx):
        return (robot.data.body_pos_w[0, idx] - origin).cpu().numpy()

    def finger_dir(idx):
        q = robot.data.body_quat_w[0, idx]
        return matrix_from_quat(q.unsqueeze(0))[0, :, 0].cpu().numpy()  # local +X in world

    def obj_pos():
        return (obj.data.root_pos_w[0] - origin).cpu().numpy()

    def obj_vel():
        return obj.data.root_vel_w[0, :3].cpu().numpy()

    def hand_vec(close_left=False, close_right=False):
        v = {n: 0.0 for n in hand_names}
        if close_left:
            v["left_hand_index_0_joint"] = -1.3
            v["left_hand_index_1_joint"] = -1.5
            v["left_hand_middle_0_joint"] = -1.3
            v["left_hand_middle_1_joint"] = -1.5
            v["left_hand_thumb_0_joint"] = 0.6
            v["left_hand_thumb_1_joint"] = 0.5
            v["left_hand_thumb_2_joint"] = 1.0
        if close_right:
            v["right_hand_index_0_joint"] = 1.3
            v["right_hand_index_1_joint"] = 1.5
            v["right_hand_middle_0_joint"] = 1.3
            v["right_hand_middle_1_joint"] = 1.5
            v["right_hand_thumb_0_joint"] = -0.6
            v["right_hand_thumb_1_joint"] = -0.5
            v["right_hand_thumb_2_joint"] = -1.0
        return np.array([v[n] for n in hand_names], dtype=np.float32)

    def make_action(lp, rp, hand, lq=Q_DOWN, rq=Q_DOWN):
        a = np.zeros(28, dtype=np.float32)
        a[0:3] = lp
        a[3:7] = lq
        a[7:10] = rp
        a[10:14] = rq
        a[14:] = hand
        return torch.tensor(a, device=device).unsqueeze(0)

    success_hits = 0

    def check_success():
        op = obj_pos()
        ov = obj_vel()
        rwx = opos(ri)[0]
        cond = {
            "x_in(0.40,0.85)": 0.40 < op[0] < 0.85,
            "y_in(0.35,0.60)": 0.35 < op[1] < 0.60,
            "z<1.10": op[2] < 1.10,
            "rwrist_x<0.26": rwx < 0.26,
            "vel<0.20": np.all(np.abs(ov) < 0.20),
        }
        return all(cond.values()), cond, op, ov, rwx

    TRANSIT_Z = 1.18                   # transit height: open fingers (~0.18 below wrist) clear wheel top (0.78)
    GRASP_Z = 0.80                     # deep grasp (IK stalls ~0.90, fingertips ~0.72): reliably DRAGS the wheel
    GY = 0.40                          # LEFT grasps near-center (y~0.40): descends with minimal shove

    def run_phase(label, lp, rp, cl, cr, n, lq=Q_DOWN, rq=Q_DOWN):
        nonlocal success_hits
        hand = hand_vec(close_left=cl, close_right=cr)
        act = make_action(np.array(lp, float), np.array(rp, float), hand, lq=lq, rq=rq)
        op = ov = None
        for _ in range(n):
            env.step(act)
            ok, cond, op, ov, rwx = check_success()
            if ok:
                success_hits += 1
        print(f"[{label:11s}] obj={np.round(op,3)} vel={np.round(ov,2)} "
              f"Lwrist={np.round(opos(li),2)} Rwrist={np.round(opos(ri),2)}")

    print("\n================ SCRIPTED PICK-PLACE ================")
    print(f"start object: {np.round(obj_pos(),3)}")

    # x,y the left drags the wheel to before handing off to the right (LOW y!)
    HANDOFF_X = 0.12
    LOW_Y = 0.34

    # ====================================================================================
    # PHASE 1: LEFT (solo) grasps near-center, drags wheel to LOW y, then HOLDS (closed, low)
    # ====================================================================================
    run_phase("L_raise",   [-0.18, 0.10, TRANSIT_Z], HOME_R, False, False, 30)
    run_phase("L_over",    [-0.35, 0.10, TRANSIT_Z], HOME_R, False, False, 35)
    run_phase("L_above",   [-0.35, GY, TRANSIT_Z],   HOME_R, False, False, 40)
    run_phase("L_descend", [-0.35, GY, GRASP_Z],     HOME_R, False, False, 45)
    run_phase("L_close",   [-0.35, GY, GRASP_Z],     HOME_R, True,  False, 45)
    # drag horizontally toward the handoff while ALSO pulling y down to LOW_Y
    run_phase("L_drag1",   [-0.12, 0.37, GRASP_Z],   HOME_R, True,  False, 50)
    run_phase("L_drag2",   [HANDOFF_X, LOW_Y, GRASP_Z], HOME_R, True, False, 65)
    # HOLD at the handoff spot (stay closed, low) — do NOT release yet
    run_phase("L_hold",    [HANDOFF_X, LOW_Y, GRASP_Z], HOME_R, True, False, 30)

    # ---- closed loop: read the wheel pose, plan the right grasp near the HUB so the wheel
    #      tracks the wrist ~1:1 (a body-corner grip just plows the wheel +y instead of moving it +x). ----
    cx, cy = obj_pos()[0], obj_pos()[1]
    gx = cx - 0.12                              # body-side (-x) corner: hub rides +0.12 in x AHEAD of the wrist
    gy = max(cy - 0.08, 0.32)                   # near corner (low y) so the right can reach & release clean
    print(f">>> handoff: wheel center=({cx:.3f},{cy:.3f}) -> right grasp x={gx:.3f}, y={gy:.3f}")

    # ====================================================================================
    # PHASE 2: RIGHT high-traverse approach + grasp body-side corner; LEFT keeps holding
    # ====================================================================================
    run_phase("R_raise",   [HANDOFF_X, LOW_Y, GRASP_Z], [0.18, 0.10, TRANSIT_Z], True, False, 30)
    run_phase("R_over",    [HANDOFF_X, LOW_Y, GRASP_Z], [gx, 0.10, TRANSIT_Z],   True, False, 35)
    run_phase("R_above",   [HANDOFF_X, LOW_Y, GRASP_Z], [gx, gy, TRANSIT_Z],     True, False, 40)
    run_phase("R_descend", [HANDOFF_X, LOW_Y, GRASP_Z], [gx, gy, GRASP_Z],       True, False, 45)
    run_phase("R_close",   [HANDOFF_X, LOW_Y, GRASP_Z], [gx, gy, GRASP_Z],       True, True,  50)

    # ====================================================================================
    # PHASE 3: LEFT disengages by sliding -x at LOW z (never lift through wheel), then parks
    #          LOW & back while RIGHT pins the wheel DOWN on the table (cr=True throughout).
    # ====================================================================================
    run_phase("L_open",    [HANDOFF_X, LOW_Y, GRASP_Z], [gx, gy, GRASP_Z],       False, True, 35)
    run_phase("L_slideoff",[HANDOFF_X - 0.22, LOW_Y, GRASP_Z], [gx, gy, GRASP_Z], False, True, 35)
    run_phase("L_lift",    [HANDOFF_X - 0.22, LOW_Y, TRANSIT_Z], [gx, gy, GRASP_Z], False, True, 30)
    # Park the LEFT far to -x and LOW: frees the shared waist to rotate RIGHT so the right
    # wrist can extend toward +x (its x-reach collapses when the left sits near center).
    run_phase("L_park",    [-0.42, 0.05, 0.80],          [gx, gy, GRASP_Z],       False, True, 35)

    # ====================================================================================
    # PHASE 4: RIGHT corner-grip, then LIFT the wheel to z~0.95 and carry +x at that height.
    #   KEY: the right wrist x-reach is ~0.23 at z=0.84 but ~0.34 at z>1.0 (the arm extends
    #   further +x when raised). Carrying the wheel raised (z<1.10) lets the hub reach x>0.40.
    #   A CORNER/rim grip releases cleanly on a straight-up lift (no fling). Keep LEFT far/low.
    # ====================================================================================
    LPARK = [-0.42, 0.05, 0.80]                # keep the left far/low for the whole right move
    CARRY_Y = 0.42                             # hold y ~constant during the +x carry (stay in (0.35,0.60))
    CARRY_Z = 1.00                             # raised carry: extends the right wrist's +x reach (wheel stays z<1.10)

    run_phase("R_lift1",   LPARK, [gx, gy, CARRY_Z],          False, True, 45)  # lift the corner-gripped wheel up
    run_phase("R_carry1",  LPARK, [gx + 0.20, CARRY_Y, CARRY_Z], False, True, 60)
    run_phase("R_carry2",  LPARK, [0.55, CARRY_Y, CARRY_Z],   False, True, 110) # push +x at height; wrist saturates near reach limit
    # lower the wheel toward the table at the place spot, still gripped
    run_phase("R_lower",   LPARK, [0.55, CARRY_Y, 0.86],      False, True, 55)
    # release corner grip: open long, then lift straight up (corner grip releases clean), retract
    run_phase("R_open",    LPARK, [0.55, CARRY_Y, 0.86],      False, False, 55)
    run_phase("R_up",      LPARK, [0.50, CARRY_Y, TRANSIT_Z], False, False, 35)
    run_phase("R_retract", HOME_L, [0.14, 0.12, 1.00],        False, False, 50)
    run_phase("settle",    HOME_L, HOME_R,                     False, False, 90)

    ok, cond, op, ov, rwx = check_success()
    print("\n================ RESULT ================")
    print(f"final object pos: {np.round(op,3)}  vel: {np.round(ov,3)}  right_wrist_x: {rwx:.3f}")
    print("success conditions:")
    for k, val in cond.items():
        print(f"   {'PASS' if val else 'FAIL'}  {k}")
    print(f"TASK SUCCESS (all conditions): {ok}   (success frames during run: {success_hits})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
