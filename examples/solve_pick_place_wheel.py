"""Scripted pick-and-place for Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0.

The Unitree G1 (fixed base, three-fingered trihands) picks up the steering wheel
from the table with its LEFT hand and places it in the packing-table basket on its
right, using the env's Pink-IK absolute action space:

    action[0:3]   left wrist position target  (env-local frame)
    action[3:7]   left wrist quaternion target (wxyz)
    action[7:10]  right wrist position target
    action[10:14] right wrist quaternion target
    action[14:28] hand joint position targets in the order:
        [L_idx0, L_mid0, L_th0, R_idx0, R_mid0, R_th0,
         L_idx1, L_mid1, L_th1, R_idx1, R_mid1, R_th1, L_th2, R_th2]

Success (env's task_done_pick_place): object in the basket box x(0.40,0.85)
y(0.35,0.60) z<1.10 (env-local), settled |v|<0.2 m/s per axis, and
right_wrist_yaw_link x < 0.26. The right arm is parked at its initial pose
(x=0.149) so the wrist condition holds throughout.

Grasp design (hand geometry from USD introspection: fingers extend along wrist +x,
index/middle offset along wrist +/-z, thumb extends along wrist -y with only ~6.5 cm
reach): a horizontal HOOK grasp on the wheel's near-left rim, approached radially
from outside with fingers pointing at the hub. Lessons baked in from failed runs:

- run 1: rotating the fingers downward while translating raked the wheel away ->
  approach is now radial, outside the wheel, with live retargeting per phase.
- run 2: the fingers could not finish curling around the rim tube because the
  fingertips would sweep below the table surface; the half-open hook let the wheel
  pivot, dangle vertically, swing, and get flung. Fix: the thumb is yawed aside
  during the approach (it otherwise hangs 6 cm below the wrist and fouls the table),
  and a SCOOP raises the wrist while half-curling so the curl completes mid-air
  into a closed fist; the thumb then returns to lock the tube from below.

NOTE: `import pinocchio` must precede AppLauncher (forces the pip pinocchio over
Isaac Sim's bundled copy), and `enable_pinocchio=True` must be set on the launcher
args so it patches pxr.Gf.Matrix4d for compatibility. The task is also blacklisted
from the bulk isaaclab_tasks import scan and must be imported explicitly.

Run (records video into this folder's videos/):
    env_isaaclab/bin/python -u Fable_5/G1_Upper_body_IK/pick_place_wheel.py
"""

import argparse

# Import pinocchio before AppLauncher to force the pip-installed version over the
# one bundled with Isaac Sim (boost-python type registration conflict otherwise).
import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted G1 pick-and-place (steering wheel).")
parser.add_argument("--video_name", type=str, default="g1_pick_place")
parser.add_argument("--no_video", action="store_true", default=False)
parser.add_argument("--wheel_radius", type=float, default=0.14, help="Steering wheel rim radius (m).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_pinocchio = True  # AppLauncher patches pxr.Gf.Matrix4d for pinocchio
if not args_cli.no_video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils

import isaaclab_tasks  # noqa: F401

# pick_place is blacklisted from the bulk import scan (pinocchio compat TODO);
# the task only registers when this subpackage is imported explicitly.
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.pick_place.mdp import task_done_pick_place
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0"
FABLE_DIR = os.path.dirname(os.path.abspath(__file__))

# hand joint action order (indices into action[14:28])
HAND_ORDER = [
    "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
    "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint", "right_hand_thumb_2_joint",
]
# left-hand states (right hand stays at 0 = relaxed)
# left finger joints close toward negative values; thumb_1/2 close toward positive.
L_OPEN_THUMB_ASIDE = {"index_0": 0.0, "middle_0": 0.0, "thumb_0": 1.0, "index_1": 0.0, "middle_1": 0.0,
                      "thumb_1": 0.0, "thumb_2": 0.0}
L_HALF_CURL = {"index_0": -0.8, "middle_0": -0.8, "thumb_0": 1.0, "index_1": -0.9, "middle_1": -0.9,
               "thumb_1": 0.0, "thumb_2": 0.0}
L_FIST = {"index_0": -1.25, "middle_0": -1.25, "thumb_0": 0.0, "index_1": -1.5, "middle_1": -1.5,
          "thumb_1": 0.8, "thumb_2": 1.2}
L_RELEASE = {"index_0": 0.0, "middle_0": 0.0, "thumb_0": 1.0, "index_1": 0.0, "middle_1": 0.0,
             "thumb_1": 0.0, "thumb_2": 0.0}

# hook-grasp geometry (wrist frame: fingers +x, thumb -y)
GRIP_DEPTH = 0.10  # rim tube this far along the fingers (at the finger crotch)
GRIP_RAISE = 0.024  # wrist above tube center: fingers pass ~1 cm over the tube top
RADIAL = (0.6, 0.8)  # unit direction from rim grasp point toward the wheel hub
APPROACH_BACK = 0.13  # staging distance behind the grasp point (outside the wheel)
SCOOP_RAISE = 0.07  # wrist lift during the scoop (curl completes in the air)


def hand_action_values(left_state: dict) -> list[float]:
    vals = []
    for name in HAND_ORDER:
        if name.startswith("left"):
            key = name.replace("left_hand_", "").replace("_joint", "")
            vals.append(left_state[key])
        else:
            vals.append(0.0)
    return vals


def side_grasp_quat(device, radial=RADIAL):
    """Left wrist orientation for the horizontal hook grasp.

    Wrist axes in world: x (fingers) -> radial inward (horizontal),
    y -> +z (thumb at -y points down, under the rim tube), z = x cross y.
    """
    ux, uy = radial
    x_w = torch.tensor([ux, uy, 0.0], device=device)
    y_w = torch.tensor([0.0, 0.0, 1.0], device=device)
    z_w = torch.linalg.cross(x_w, y_w)
    rot = torch.stack([x_w, y_w, z_w], dim=1)
    return math_utils.quat_from_matrix(rot)


def nlerp(q0, q1, t):
    """Normalized quaternion lerp with sign correction (fine for small arcs)."""
    if torch.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - t) * q0 + t * q1
    return q / q.norm()


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    # The env's success criterion fires as soon as the (still-grasped) object is
    # carried into the basket box at low velocity, resetting the episode mid-place.
    # Disable the success TERMINATION and instead evaluate the same function
    # manually, requiring it to hold for SUCCESS_HOLD_STEPS consecutive steps after
    # the object has actually been released into the basket.
    env_cfg.terminations.success = None
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (1.6, -0.4, 1.35)
    env_cfg.viewer.lookat = (-0.1, 0.5, 0.7)
    env_cfg.viewer.resolution = (1280, 720)

    render_mode = None if args_cli.no_video else "rgb_array"
    env = gym.make(TASK, cfg=env_cfg, render_mode=render_mode)
    if not args_cli.no_video:
        video_kwargs = {
            "video_folder": os.path.join(FABLE_DIR, "videos"),
            "name_prefix": args_cli.video_name,
            "step_trigger": lambda step: step == 0,
            "video_length": 1100,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    uenv = env.unwrapped
    env.reset()

    robot = uenv.scene["robot"]
    obj = uenv.scene["object"]
    device = uenv.device
    origin = uenv.scene.env_origins[0]
    li = robot.body_names.index("left_wrist_yaw_link")
    ri = robot.body_names.index("right_wrist_yaw_link")

    left_pos0 = (robot.data.body_pos_w[0, li] - origin).clone()
    left_quat0 = robot.data.body_quat_w[0, li].clone()
    right_pos0 = (robot.data.body_pos_w[0, ri] - origin).clone()
    right_quat0 = robot.data.body_quat_w[0, ri].clone()
    wheel0 = (obj.data.root_pos_w[0] - origin).clone()
    print(f"[CTRL] wheel at {wheel0.cpu().numpy().round(3)}, left wrist {left_pos0.cpu().numpy().round(3)}, "
          f"right wrist {right_pos0.cpu().numpy().round(3)}", flush=True)

    R = args_cli.wheel_radius
    gq = side_grasp_quat(device)
    u = torch.tensor([RADIAL[0], RADIAL[1], 0.0], device=device)  # radial inward (= wrist x)

    def wheel_pos():
        return (obj.data.root_pos_w[0] - origin).clone()

    z_tube = wheel0[2] + 0.015  # rim tube center height (wheel root + tube radius)

    def grasp_wrist_for(center):
        """Wrist position s.t. the rim tube sits at the finger crotch."""
        rim = center.clone()
        rim[0:2] = center[0:2] - R * u[0:2]
        w = rim - GRIP_DEPTH * u
        w[2] = z_tube + GRIP_RAISE
        return w

    # During the carry the wrist yaws so the fingers point deeper into +x: the wheel
    # hangs (GRIP_DEPTH + R) along the finger direction, so re-aiming the fingers
    # from RADIAL=(0.6,0.8) to RADIAL2=(0.85,0.53) shifts the wheel ~7 cm deeper
    # into the basket while the wrist stays within cross-body reach (x ~= 0.30).
    RADIAL2 = (0.849, 0.529)
    gq2 = side_grasp_quat(device, RADIAL2)
    u2 = torch.tensor([RADIAL2[0], RADIAL2[1], 0.0], device=device)
    # wheel center target ~(0.53, 0.43) inside the basket box:
    carry_wrist_xy = torch.tensor([0.53, 0.43], device=device) - (GRIP_DEPTH + R) * u2[0:2]

    def t_raise():
        return torch.tensor([-0.20, 0.25, 0.95], device=device), gq

    def t_stage():
        s = grasp_wrist_for(wheel_pos()) - APPROACH_BACK * u
        s[2] = 0.80
        return s, gq

    def t_advance():
        return grasp_wrist_for(wheel_pos()), gq

    def t_scoop():
        w = grasp_wrist_for(wheel_pos())
        w[2] += SCOOP_RAISE
        return w, gq

    def t_clamp():
        # hold position (use last commanded, i.e. scoop height at current wheel xy lock)
        w = grasp_wrist_for(wheel_pos())
        w[2] += SCOOP_RAISE
        return w, gq

    def t_lift():
        w = torch.zeros(3, device=device)
        w[0:2] = grasp_wrist_for(wheel_pos())[0:2]
        w[2] = 0.98
        return w, gq

    def t_carry1():
        return torch.tensor([0.0, 0.30, 0.98], device=device), gq

    def t_carry2():
        # translate the rest of the way while yawing fingers toward +x
        return torch.tensor([carry_wrist_xy[0], carry_wrist_xy[1], 0.98], device=device), gq2

    def t_lower():
        # wheel (hanging ~0.14 below the wrist) descends into the basket interior,
        # its near edge below the rim so it cannot pivot back out at release
        return torch.tensor([carry_wrist_xy[0], carry_wrist_xy[1], 0.79], device=device), gq2

    def t_release():
        # open while nudging forward and slightly up: the wheel slides off the
        # fingers deeper into the basket instead of dragging back over the rim
        w = torch.tensor([carry_wrist_xy[0], carry_wrist_xy[1], 0.84], device=device)
        w[0:2] += 0.04 * u2[0:2]
        return w, gq2

    def t_nudge():
        # sweep the open hand forward-down: tips the wheel off the basket's left
        # wall top so it falls flat inside (it otherwise rests leaning on the rim)
        w = torch.tensor([carry_wrist_xy[0], carry_wrist_xy[1], 0.80], device=device)
        w[0:2] += 0.13 * u2[0:2]
        return w, gq2

    def t_retreat_up():
        # rise straight up first so the retreat cannot drag the wheel back out
        w = torch.tensor([carry_wrist_xy[0] + 0.02, carry_wrist_xy[1], 0.98], device=device)
        return w, gq2

    def t_retreat():
        return torch.tensor([0.0, 0.22, 0.95], device=device), gq

    phases = [
        ("RAISE", 1.4, t_raise, L_OPEN_THUMB_ASIDE),
        ("STAGE", 1.5, t_stage, L_OPEN_THUMB_ASIDE),
        ("ADVANCE", 1.5, t_advance, L_OPEN_THUMB_ASIDE),
        ("SCOOP", 1.2, t_scoop, L_HALF_CURL),
        ("CLAMP", 1.0, t_clamp, L_FIST),
        ("LIFT", 1.2, t_lift, L_FIST),
        ("CARRY", 1.8, t_carry1, L_FIST),
        ("CARRY2", 2.2, t_carry2, L_FIST),
        ("LOWER", 1.5, t_lower, L_FIST),
        ("RELEASE", 1.2, t_release, L_RELEASE),
        ("NUDGE", 1.2, t_nudge, L_RELEASE),
        ("RETREAT_UP", 1.0, t_retreat_up, L_RELEASE),
        ("RETREAT", 1.6, t_retreat, L_RELEASE),
        ("SETTLE", 1.6, t_retreat, L_RELEASE),
    ]

    action = torch.zeros((1, 28), device=device)
    action[0, 7:10] = right_pos0
    action[0, 10:14] = right_quat0

    seg_pos = left_pos0.clone()
    seg_quat = left_quat0.clone()
    seg_hand = dict(L_OPEN_THUMB_ASIDE)

    SUCCESS_HOLD_STEPS = 25  # success must hold 0.5 s continuously after release
    success_latched = False
    success_streak = 0
    released = False
    aborted = None
    total_step = 0
    for name, duration, target_fn, end_hand in phases:
        tgt_pos, tgt_quat = target_fn()
        steps = int(duration * 50)
        if name == "RELEASE":
            released = True
        for k in range(steps):
            t = (k + 1) / steps
            pos = seg_pos + (tgt_pos - seg_pos) * t
            quat = nlerp(seg_quat, tgt_quat, t)
            hand = {key: seg_hand[key] + (end_hand[key] - seg_hand[key]) * t for key in end_hand}

            action[0, 0:3] = pos
            action[0, 3:7] = quat
            hv = hand_action_values(hand)
            for i in range(14):
                action[0, 14 + i] = hv[i]

            with torch.inference_mode():
                obs, rew, terminated, truncated, info = env.step(action)
            total_step += 1

            # same criterion as the env's success termination, evaluated manually
            # and only counted after the object has been released
            success_now = bool(task_done_pick_place(uenv, task_link_name="right_wrist_yaw_link")[0])
            success_streak = success_streak + 1 if (success_now and released) else 0
            if success_streak >= SUCCESS_HOLD_STEPS:
                success_latched = True
            if bool(terminated[0]) or bool(truncated[0]):
                dropped = bool(uenv.termination_manager.get_term("object_dropping")[0])
                aborted = "object_dropped" if dropped else "timeout"

            if total_step % 50 == 0 or success_latched or aborted:
                w = (obj.data.root_pos_w[0] - origin).cpu().numpy()
                lw = (robot.data.body_pos_w[0, li] - origin).cpu().numpy()
                rwx = (robot.data.body_pos_w[0, ri, 0] - origin[0]).item()
                vel = obj.data.root_vel_w[0, :3].abs().max().item()
                print(f"[CTRL] {name:8s} step {total_step:4d} | wheel ({w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f}) "
                      f"|v|max {vel:.2f} | Lwrist ({lw[0]:+.3f},{lw[1]:+.3f},{lw[2]:+.3f}) | Rwrist_x {rwx:+.3f} "
                      f"| success_now={success_now} streak={success_streak}", flush=True)

            if aborted:
                break
        seg_pos = tgt_pos.clone()
        seg_quat = tgt_quat.clone()
        seg_hand = dict(end_hand)
        # on success, keep executing the remaining phases (retreat) so the video
        # ends with the arm pulled back and the wheel resting in the basket
        if aborted:
            break

    verdict = "SUCCESS" if success_latched else f"FAIL ({aborted or 'sequence ended without success'})"
    print(f"[CTRL] RESULT: {verdict} after {total_step} steps ({total_step/50:.1f} s)", flush=True)
    env.close()
    if not args_cli.no_video:
        print(f"[CTRL] video saved under: {os.path.join(FABLE_DIR, 'videos')}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
