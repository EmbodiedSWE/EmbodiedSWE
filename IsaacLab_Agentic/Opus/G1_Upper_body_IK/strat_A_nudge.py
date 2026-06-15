# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Scripted solver for Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0.

Strategy (forced by the reach map: left hand reaches the object at x=-0.35 but maxes at
wrist-x~0.26; right hand reaches the place region x>0.40 but cannot cross to the object):

  LEFT  phase: top-down grasp/drag the wheel from x=-0.35 to the center (~x=0.10), release.
  RIGHT phase: top-down hook the body-side (-x) rim, push/drag the wheel into the place
               region (hub -> ~0.5), release, and retract the right wrist to x<0.26.

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
    GY = 0.43                          # grasp ~ON the hub (wheel starts y=0.45): descend at hub shoves only slightly
    # Keep the IDLE hand LOW at home: parking it high shifts the shared-waist posture and makes the
    # working arm lift the wheel instead of dragging it cleanly along the table.

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

    HANDOFF_X = 0.15                   # x the left drags the wheel to before handing off to the right
    HANDOFF_Y = 0.30                   # deliver the wheel LOW so the right can descend/grip its rim; the
                                       # diagonal drag then carries it UP into the y window as it advances +x

    # ---- LEFT (solo): grasp the hub at the wheel's natural y, drag the wheel forward in +x AND
    #      gently down to a LOW y so the right (which only reaches far +x at low y) can grasp it.
    #      Release by opening long then lifting STRAIGHT UP in many slow stages (the fling happens
    #      in the upper lift z~0.95->1.18 when the last spoke snaps free -- go slow there).
    run_phase("L_raise",   [-0.18, 0.10, TRANSIT_Z], HOME_R, False, False, 30)
    run_phase("L_over",    [-0.35, 0.10, TRANSIT_Z], HOME_R, False, False, 35)
    run_phase("L_above",   [-0.35, GY, TRANSIT_Z],   HOME_R, False, False, 40)
    run_phase("L_descend", [-0.35, GY, GRASP_Z],     HOME_R, False, False, 45)
    run_phase("L_close",   [-0.35, GY, GRASP_Z],     HOME_R, True,  False, 45)
    run_phase("L_drag1",   [-0.12, 0.41, GRASP_Z],   HOME_R, True,  False, 50)
    run_phase("L_drag2",   [HANDOFF_X, HANDOFF_Y, GRASP_Z], HOME_R, True, False, 70)
    run_phase("L_open",    [HANDOFF_X, HANDOFF_Y, GRASP_Z], HOME_R, False, False, 85)
    run_phase("L_lift1",   [HANDOFF_X, HANDOFF_Y, 0.88],  HOME_R, False, False, 40)
    run_phase("L_lift2",   [HANDOFF_X, HANDOFF_Y, 0.96],  HOME_R, False, False, 40)
    run_phase("L_lift3",   [HANDOFF_X, HANDOFF_Y, 1.05],  HOME_R, False, False, 40)
    run_phase("L_lift4",   [HANDOFF_X, HANDOFF_Y, TRANSIT_Z], HOME_R, False, False, 40)
    run_phase("L_back",    [-0.30, 0.10, TRANSIT_Z], HOME_R, False, False, 30)

    # ============================================================================================
    # RIGHT closed-loop nudging LOOP: re-read obj_pos() each iteration. OPEN fingers descending onto
    # the hub shove/lift the light wheel, so descend a CLOSED FIST just BEHIND the body-side (-x) rim
    # in EMPTY space (clean, no contact). THEN open the fingers (now straddling the rim), CLOSE to
    # GRIP the rim from behind (a rim grip drags cleanly, unlike a hub grip), and DRAG +x. Holding
    # the rim, the wrist reaches x~0.27 and the hub rides ~0.13 ahead toward x>0.40, staying LOW.
    # Open, lift a little, re-read, repeat until the hub passes the target x.
    # ============================================================================================
    TARGET_X = 0.42                    # stop the loop once the wheel hub is past this (then retract)
    GZ = 0.80                          # proven deep grasp z
    DRAG_Z = 0.79                      # drag near grasp z so the rim grip neither lifts nor digs
    RIM = 0.14                         # wheel rim radius (AABB 0.286 -> ~0.143)
    # Park the idle LEFT hand back and low during the right phase: pulling the left arm back rotates
    # the shared waist to give the RIGHT more forward/outward reach (opposite of the high-park penalty
    # in LEARNINGS). Too far back over-frees it and the right FLINGS the wheel, so keep it moderate.
    PARK_L = [-0.32, 0.04, 0.80]
    for it in range(6):
        cx, cy = obj_pos()[0], obj_pos()[1]
        print(f">>> nudge {it}: wheel center=({cx:.3f},{cy:.3f})")
        if cx > TARGET_X:
            print(f">>> wheel past target x={TARGET_X}; stop nudging")
            break
        # CLOSED-FIST QUASI-STATIC PUSH: descend the closed fist just behind the -x rim (clean, no open
        # phase -> no +y brush, no release -> no fling), then advance it in SMALL SLOW segments so the
        # wheel is pushed quasi-statically (it stops when the fist stops, instead of being batted off).
        # The left is back-parked so the wrist can extend to x~0.40; each segment commands a modest
        # forward target and dwells many steps to let the wheel settle. Diagonal toward (high x, mid y).
        gx = cx - RIM - 0.01          # fist contacts the body-side rim
        gy = min(cy, 0.30)            # descend low; right reaches this y here
        run_phase(f"R{it}_over",     PARK_L, [gx, 0.10, 1.28], False, True, 30)
        run_phase(f"R{it}_above",    PARK_L, [gx, gy, 1.28],   False, True, 34)
        run_phase(f"R{it}_descend",  PARK_L, [gx, gy, GZ],     False, True, 38)
        # slow segmented push: each step ~0.06 x, dwelling so the wheel doesn't coast.
        for k, (txx, tyy) in enumerate([(cx + 0.06, cy + 0.03), (cx + 0.14, cy + 0.07),
                                        (cx + 0.22, cy + 0.10), (cx + 0.30, cy + 0.12)]):
            run_phase(f"R{it}_p{k}", PARK_L, [min(txx, 0.46), min(tyy, 0.44), DRAG_Z], False, True, 40)
        run_phase(f"R{it}_lift",     PARK_L, [cx + 0.24, min(cy + 0.10, 0.44), 1.18], False, True, 28)

    # ---- final: retract the right wrist to x<0.26 at HIGH z (don't drag the wheel back), settle ----
    run_phase("R_retract", HOME_L, [0.12, 0.10, 1.05],  False, False, 50)
    run_phase("settle",    HOME_L, HOME_R,               False, False, 110)

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
