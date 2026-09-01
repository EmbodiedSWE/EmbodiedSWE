"""Grading-like test: fresh resets, then solve(env) — repeated trials."""
import os, sys, time
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch
import robobench; robobench.discover()
from robobench.core.registries import ENVS

env = ENVS.get('assembly.bulb.franka.osc')().build(num_envs=1)
sys.path.insert(0, "/workspace/solution")
import importlib
import solve as S

N = int(os.environ.get("TRIALS", "3"))
results = []
for t in range(N):
    env.reset()
    importlib.reload(S)
    t0 = time.perf_counter()
    S.solve(env)
    ok = bool(env.scene.seated().all())
    wall = time.perf_counter() - t0
    print(f"TRIAL {t}: seated={ok} wall={wall:.0f}s")
    results.append(ok)
print("RESULTS", results, f"{sum(results)}/{len(results)}")
print("DONE-TEST")
os._exit(0)
