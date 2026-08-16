"""Retest the glass_away case with the reposition fallback."""
import os, sys, math
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch
import robobench; robobench.discover()
from robobench.core.registries import ENVS

env = ENVS.get('assembly.bulb.franka.osc')().build(num_envs=1)
sys.path.insert(0, "/workspace/solution")
import solve as S

def place_bulb(x, y, yaw):
    bulb = env.scene.bulbs[0]
    st = torch.zeros(1, 13, device=env.device)
    st[0, 0:3] = torch.tensor([x, y, 0.024], device=env.device)
    qx = torch.tensor([math.cos(math.pi/4), math.sin(math.pi/4), 0.0, 0.0], device=env.device)
    qz = torch.tensor([math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2)], device=env.device)
    st[0, 3:7] = S._quat_mul(qz, qx)
    bulb.write_root_state_to_sim(st, torch.tensor([0], device=env.device))

env.reset()
place_bulb(0.25, 0.12, -math.pi / 2)  # glass end toward +x (away from base)
zero = torch.zeros(1, env.robot.action_dim, device=env.device)
zero[:, 6:8] = 0.04
for _ in range(12):
    env.step(zero)
S.solve(env)
print("CASE glass_away(retry): seated=", bool(env.scene.seated().all()))
print("DONE-POSES2")
os._exit(0)
