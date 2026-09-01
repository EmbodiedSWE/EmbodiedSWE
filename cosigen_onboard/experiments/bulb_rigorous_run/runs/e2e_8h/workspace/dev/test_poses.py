"""Pick/solve generality: place the bulb at non-default poses, then solve()."""
import os, sys, math
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch
import robobench; robobench.discover()
from robobench.core.registries import ENVS

env = ENVS.get('assembly.bulb.franka.osc')().build(num_envs=1)
sys.path.insert(0, "/workspace/solution")
import importlib
import solve as S

def place_bulb(x, y, yaw):
    """Lying bulb at (x, y) with its axis yawed by `yaw` (0 = default: glass toward -y)."""
    bulb = env.scene.bulbs[0]
    st = torch.zeros(1, 13, device=env.device)
    st[0, 0:3] = torch.tensor([x, y, 0.024], device=env.device)
    # default init quat: 90 deg about x. Pre-multiply a world-z yaw.
    qx = torch.tensor([math.cos(math.pi/4), math.sin(math.pi/4), 0.0, 0.0], device=env.device)
    qz = torch.tensor([math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2)], device=env.device)
    st[0, 3:7] = S._quat_mul(qz, qx)
    bulb.write_root_state_to_sim(st, torch.tensor([0], device=env.device))

cases = [
    ("far_right", 0.30, 0.30, 0.0),
    ("glass_toward_robot", 0.28, 0.18, math.pi / 2),   # glass end toward -x (robot side)
    ("glass_away", 0.25, 0.12, -math.pi / 2),          # glass end toward +x (away)
]
results = []
for name, x, y, yaw in cases:
    env.reset()
    place_bulb(x, y, yaw)
    # settle
    zero = torch.zeros(1, env.robot.action_dim, device=env.device)
    zero[:, 6:8] = 0.04
    for _ in range(12):
        env.step(zero)
    importlib.reload(S)
    S.solve(env)
    ok = bool(env.scene.seated().all())
    print(f"CASE {name}: seated={ok}")
    results.append((name, ok))
print("RESULTS", results)
print("DONE-POSES")
os._exit(0)
