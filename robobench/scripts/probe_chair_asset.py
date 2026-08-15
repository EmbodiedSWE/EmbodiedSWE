"""Boot-probe one chair asset in isolation — bisects which USD poisons the GPU solver.

Spawns ground + ONE asset (by --asset) under the chair scene's exact physx config, then
steps the bare sim 30 times. A clean run prints BOOT_OK; the solver crashing on step 1
reproduces the chair_assembly_smoke failure for that asset alone.

    /opt/venv/bin/python -m robobench.scripts.probe_chair_asset --asset base --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--asset", required=True,
                    choices=["none", "nut", "back", "base", "base_nostuds", "base_boxesonly", "base_novisual"])
parser.add_argument("--usd", default="", help="override the asset USD path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

ASSETS = Path(__file__).resolve().parents[1] / "suites" / "assembly" / "assets"

PHYSX = sim_utils.PhysxCfg(
    solver_type=1,
    bounce_threshold_velocity=0.2,
    friction_offset_threshold=0.01,
    friction_correlation_distance=0.00625,
    gpu_max_rigid_contact_count=2**23,
    gpu_max_rigid_patch_count=2**23,
    gpu_collision_stack_size=2**28,
    gpu_max_num_partitions=1,
)


def rigid(usd: str, z: float, iters: int = 192, disable_articulation: bool = False,
          contact_offset: float | None = None) -> RigidObjectCfg:
    spawn = sim_utils.UsdFileCfg(
        usd_path=usd,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=iters,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=5.0,
        ),
    )
    if disable_articulation:
        spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False)
    if contact_offset is not None:
        spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=contact_offset, rest_offset=0.0)
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Probe",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z)),
    )


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=device, physx=PHYSX))

    @configclass
    class ProbeSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                usd_path=str(ASSETS / "props" / "ground" / "default_ground.usd")),
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
        )

    cfg = ProbeSceneCfg(num_envs=1, env_spacing=3.0)
    if args.asset == "nut":
        cfg.probe = rigid(args.usd or str(ASSETS / "factory" / "factory_nut_m16.usd"),
                          z=0.05, disable_articulation=True)
    elif args.asset == "back":
        cfg.probe = rigid(args.usd or str(ASSETS / "chair" / "chair_back.usd"),
                          z=0.30, contact_offset=0.0005)
    elif args.asset.startswith("base"):
        name = {"base": "chair_base.usd", "base_nostuds": "chair_base_nostuds.usd",
                "base_boxesonly": "chair_base_boxesonly.usd",
                "base_novisual": "chair_base_novisual.usd"}[args.asset]
        cfg.probe = rigid(args.usd or str(ASSETS / "chair" / name),
                          z=0.02, contact_offset=0.0005)

    scene = InteractiveScene(cfg)
    sim.reset()
    print(f"[probe] {args.asset}: sim reset OK, stepping", flush=True)
    for i in range(30):
        sim.step()
        scene.update(sim.get_physics_dt())
    if "probe" in scene.keys():
        pos = scene["probe"].data.root_pos_w[0]
        print(f"[probe] final pos {[round(float(v), 4) for v in pos]}", flush=True)
    print("BOOT_OK", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)
