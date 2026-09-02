"""Probe the screw dynamics from the 'rested' snapshot: per-step q7 / bulb spin /
depth under different commanded yaw errors and presses."""
import os, sys, math
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch
import robobench; robobench.discover()
from robobench.core.registries import ENVS

env = ENVS.get('assembly.bulb.franka.osc')().build(num_envs=1)
sys.path.insert(0, "/workspace/solution")
import solve as S

task = S.BulbTask(env, log=print)
states = torch.load("/workspace/dev/snaps/rested.pt", weights_only=False)
env.set_states(states)
task._grip = 0.0
task.hold(None, 2)

def bulb_spin():
    bx = S._quat_apply(task.bulb_quat, torch.tensor([1.0, 0.0, 0.0], device=task.dev))
    return math.atan2(float(bx[1]), float(bx[0]))

def hand_yaw():
    hy = S._quat_apply(task.ee_quat, torch.tensor([0.0, 1.0, 0.0], device=task.dev))
    return math.atan2(float(hy[1]), float(hy[0]))

def stroke(dpsi, dz, n, grip, tag):
    print(f"--- stroke {tag}: dpsi={dpsi} dz={dz} grip={grip}")
    for i in range(n):
        dxy = (task.bulb_pos[:2] - task.ee_pos[:2]).clamp(-0.003, 0.003)
        dpos = torch.tensor([float(dxy[0]), float(dxy[1]), dz], device=task.dev)
        task.act(dpos, task._down_yaw_rot(dpsi), grip)
        if i % 2 == 0:
            print(f"  {i:3d} q7={task.q7:+.3f} yaw={hand_yaw():+.3f} spin={bulb_spin():+.3f} "
                  f"depth={task.bulb_depth():.4f} gap={task.finger_gap:.4f} "
                  f"tilt={float(task.bulb_axis[2]):.4f} xy={task.bulb_xy_err()*1000:.1f}")

# baseline: what run7 did
stroke(-0.4, -0.0008, 20, S.GRIP_CLOSED, "baseline")
# bigger commanded error
stroke(-1.5, -0.0008, 20, S.GRIP_CLOSED, "big-err")
# huge err, no press
stroke(-2.5, 0.0, 20, S.GRIP_CLOSED, "huge-err-nopress")
print("DONE-PROBE")
os._exit(0)
