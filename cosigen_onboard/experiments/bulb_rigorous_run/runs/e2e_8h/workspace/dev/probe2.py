"""Probe 2: strong press, then press+CW / press+CCW threading from 'rested'."""
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

def stroke(dpsi, dz, n, tag):
    print(f"--- {tag}: dpsi={dpsi} dz={dz}")
    for i in range(n):
        dxy = (task.bulb_pos[:2] - task.ee_pos[:2]).clamp(-0.003, 0.003)
        dpos = torch.tensor([float(dxy[0]), float(dxy[1]), dz], device=task.dev)
        task.act(dpos, task._down_yaw_rot(dpsi), S.GRIP_CLOSED)
        if i % 2 == 0:
            print(f"  {i:3d} q7={task.q7:+.3f} depth={task.bulb_depth():.4f} gap={task.finger_gap:.4f} "
                  f"tilt={float(task.bulb_axis[2]):.4f} xy={task.bulb_xy_err()*1000:.1f} "
                  f"eez={float(task.ee_pos[2]):.4f}")

stroke(0.0, -0.03, 14, "press only (~3N+)")
stroke(-0.4, -0.03, 40, "press + CW")
print("q7 now", task.q7, "depth", task.bulb_depth())
stroke(0.0, 0.0, 4, "pause")
stroke(+0.4, -0.03, 40, "press + CCW")
print("DONE-PROBE2")
os._exit(0)
