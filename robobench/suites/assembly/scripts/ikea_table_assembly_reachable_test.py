"""Reachability test for the IKEA-table assembly scene — can a fixed-base humanoid reach the legs?

Basic sanity check on a scene+robot *configuration* (base placement, surface height, leg layout): under
**Pink IK**, drive one wrist toward a chosen table leg and report how far it closes. A large residual =
that leg is outside the arm's workspace -> retune the layout (move the robot, raise/lower the bench,
bring the legs closer) before bothering with grasping.

Robot-agnostic via `--robot` (see ROBOT_SPECS): any embodiment whose `pink_ik` mode exposes the standard
two-wrist action (action = [left wrist pose, right wrist pose, hands]) can be tested by adding its env
name + wrist-link names below.
  - g1    — built + validated.
  - gr1t2 — needs its `pink_ik` controller first (USD->URDF vendoring; not built yet — see gr1t2.py).
            Running `--robot gr1t2` until then surfaces a clear NotImplementedError.

WAYPOINTS, not a lunge: the commanded wrist target starts at the current wrist pose (zero error) and
marches toward the goal at <= `--max_step` per tick, so the IK sees a small error each step and the arm
moves smoothly instead of jumping at a far goal. (Trajectory shaping is caller-side; the controller just
solves each near target.)

NOTE: `import pinocchio` is FIRST, before AppLauncher — required for the pink_ik solver (see
`robobench.controllers.pink_ik`). It's a visual check — watch live with --livestream 2.

    python -m robobench.suites.assembly.scripts.ikea_table_assembly_reachable_test --livestream 2 --leg 0
    python -m robobench.suites.assembly.scripts.ikea_table_assembly_reachable_test --robot gr1t2 --leg 3 --arm left
"""

from __future__ import annotations

import pinocchio  # noqa: E402, F401  # MUST precede AppLauncher (registers pinocchio's eigenpy converters)

import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

# Per-robot wiring: the registered pink_ik env + the USD body names of the two wrists (left, right).
# The action layout is the same across robots — composite([arm pink_ik over (left, right) frames, hands]) —
# so only these three strings change per embodiment.
ROBOT_SPECS: dict[str, dict[str, str]] = {
    "g1": {
        "env": "assembly.ikea_table.g1.pink_ik",
        "left_wrist": "left_wrist_yaw_link",
        "right_wrist": "right_wrist_yaw_link",
    },
    "gr1t2": {  # pink_ik not built yet (USD->URDF) — wired here so the test works the moment it lands
        "env": "assembly.ikea_table.gr1t2.pink_ik",
        "left_wrist": "left_hand_pitch_link",
        "right_wrist": "right_hand_pitch_link",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=sorted(ROBOT_SPECS), default="g1", help="embodiment to test")
parser.add_argument("--arm", choices=("right", "left"), default="right", help="which wrist approaches the leg")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--leg", type=int, default=0, help="which table leg to approach (0..num_legs-1)")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--hover", type=float, default=0.08, help="target height above the leg (m)")
parser.add_argument("--max_step", type=float, default=0.01, help="max wrist-target move per step (m); waypoints so the IK doesn't lunge at a far goal")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
livestream_on = args.livestream > 0

app = AppLauncher(args).app

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402


def main() -> None:
    robobench.discover()
    spec = ROBOT_SPECS[args.robot]
    env = ENVS.get(spec["env"])().build(num_envs=args.num_envs)
    render = (not args.headless) or livestream_on
    art = env.robot.articulation
    scene = env.scene
    origins = env.iscene.env_origins

    # The wrist that reaches vs. the one that holds, by --arm. find_bodies returns ([indices], [names]).
    ai = art.find_bodies(spec[f"{args.arm}_wrist"])[0][0]  # approach wrist
    hold_name = spec["left_wrist" if args.arm == "right" else "right_wrist"]
    hi = art.find_bodies(hold_name)[0][0]  # holding wrist
    hand_ids = env.robot.controller.controllers[1].joint_ids  # the hand JointController's DOFs

    env.reset()
    a0 = art.data.body_link_state_w[:, ai, :7].clone()  # approach-wrist start pose (hold orientation)
    h0 = art.data.body_link_state_w[:, hi, :7].clone()  # holding-wrist start pose (held)
    hands0 = art.data.joint_pos[:, hand_ids].clone()
    hold_pos = h0[:, :3] - origins

    # Pink-IK frames are ordered (left, right); the action is [left pose, right pose, hands].
    def build_action(approach_pos: torch.Tensor) -> torch.Tensor:
        if args.arm == "right":
            left_pos, left_quat, right_pos, right_quat = hold_pos, h0[:, 3:7], approach_pos, a0[:, 3:7]
        else:
            left_pos, left_quat, right_pos, right_quat = approach_pos, a0[:, 3:7], hold_pos, h0[:, 3:7]
        return torch.cat([left_pos, left_quat, right_pos, right_quat, hands0], dim=1)

    cur_tgt = (a0[:, :3] - origins).clone()  # start at the current wrist -> no initial jump
    for _ in range(args.steps):
        goal = scene.legs[args.leg].data.root_pos_w - origins  # leg in env frame
        goal[:, 2] += args.hover  # hover above the leg
        delta = goal - cur_tgt
        cur_tgt = cur_tgt + delta * (args.max_step / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
        env.step(build_action(cur_tgt), render=render)

    env.close()


if __name__ == "__main__":
    main()
    app.close()
