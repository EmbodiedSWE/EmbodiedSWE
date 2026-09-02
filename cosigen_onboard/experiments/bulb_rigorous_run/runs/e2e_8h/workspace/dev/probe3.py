"""Probe 3: gravity self-threading — lower the aligned bulb, release, watch depth."""
import os, sys
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch
import robobench; robobench.discover()
from robobench.core.registries import ENVS

env = ENVS.get('assembly.bulb.franka.osc')().build(num_envs=1)
sys.path.insert(0, "/workspace/solution")
import solve as S

task = S.BulbTask(env, log=print)
states = torch.load("/workspace/dev/snaps/over.pt", weights_only=False)
env.set_states(states)
task._grip = 0.0
task.hold(None, 2)
print("start depth", task.bulb_depth(), "xy", task.bulb_xy_err(), "tilt", float(task.bulb_axis[2]))

# lower gently to depth ~0.048 (cap ~1cm above the mouth), tracking socket axis
dev = task.dev
for i in range(60):
    d = task.bulb_depth()
    if d <= 0.050:
        break
    off = env.scene._bulb_offsets_in_socket()[0, 0, 0]
    dxy = torch.tensor([-float(off[0]), -float(off[1])], device=dev).clamp(-0.003, 0.003)
    dpos = torch.tensor([float(dxy[0]), float(dxy[1]), -0.006], device=dev)
    task.act(dpos, task._down_yaw_rot(0.0), S.GRIP_CLOSED)
print("lowered: depth", task.bulb_depth(), "xy", task.bulb_xy_err()*1000, "tilt", float(task.bulb_axis[2]))

# release: open fully and freeze the arm
for i in range(40):
    task.act(torch.zeros(3, device=dev), task._down_yaw_rot(0.0), S.GRIP_OPEN)
    if i % 2 == 0:
        print(f"  {i:3d} depth={task.bulb_depth():.4f} xy={task.bulb_xy_err()*1000:.1f} "
              f"tilt={float(task.bulb_axis[2]):.4f} seated={task.seated()}")
print("after release:", task.bulb_depth(), "seated", task.seated())
print("DONE-PROBE3")
os._exit(0)
