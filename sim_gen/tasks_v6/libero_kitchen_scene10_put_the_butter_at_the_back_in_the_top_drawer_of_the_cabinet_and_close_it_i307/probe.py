"""Diagnostic probe (NOT a deliverable — deleted before the final submit)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401
except ImportError:
    import scene as scene_mod  # noqa: F401

threading.Timer(600.0, lambda: (print("PROBE: watchdog", flush=True), os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.butter_cellar")().build(num_envs=1, device=device)
    scene = env.scene
    no_action = torch.empty(0, device=device)

    env.reset(seed=0)

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/envs/env_0/Pawl")
    print("[probe] pawl prim attrs:", flush=True)
    for name in ("physxRigidBody:disableGravity", "physics:mass",
                 "physxRigidBody:sleepThreshold", "physxRigidBody:angularDamping"):
        a = prim.GetAttribute(name)
        print(f"[probe]   {name} = {a.Get() if a else '<missing>'}", flush=True)
    jp = stage.GetPrimAtPath("/World/envs/env_0/pawl_hinge")
    for name in ("physics:lowerLimit", "physics:upperLimit", "physics:axis",
                 "physics:localPos0", "physics:localRot0"):
        a = jp.GetAttribute(name)
        print(f"[probe]   hinge {name} = {a.Get() if a else '<missing>'}", flush=True)
    print(f"[probe] body disable_gravity via API: "
          f"{scene.pawl.root_physx_view.get_disable_gravities() if hasattr(scene.pawl, 'root_physx_view') else 'n/a'}",
          flush=True)

    def ang() -> float:
        return math.degrees(float(scene.pawl_angle()[0]))

    # A: normal stepping (scene post_step spring active)
    for i in range(12):
        env.step(no_action)
        if i % 3 == 0:
            print(f"[probe] A step {i}: pawl={ang():+.2f}deg "
                  f"w={float(scene.pawl.data.root_ang_vel_w[0, 1]):+.4f}", flush=True)
    for _ in range(60):
        env.step(no_action)
    print(f"[probe] A settled: pawl={ang():+.2f}deg", flush=True)

    # B: kill the scene's post_step (no spring, no platform force) — where does the
    # pawl go from its current angle? gravity ON -> falls to -85; OFF -> stays.
    scene.post_step = lambda env_ids=None: None
    zero = torch.zeros(1, 1, 3, device=device)
    scene.pawl.set_external_force_and_torque(zero.clone(), zero.clone())
    scene.platform.set_external_force_and_torque(zero.clone(), zero.clone())
    for i in range(90):
        env.step(no_action)
        if i % 15 == 0:
            print(f"[probe] B step {i}: pawl={ang():+.2f}deg", flush=True)
    print(f"[probe] B settled (no spring): pawl={ang():+.2f}deg", flush=True)
    print("PROBE: DONE", flush=True)
    threading.Timer(5.0, lambda: os._exit(0)).start()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"PROBE: FAIL ({exc})", flush=True)
        os._exit(1)
