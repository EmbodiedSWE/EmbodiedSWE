"""Helix-consistency probe for the factory M16 SDF thread pair, gravity-aligned.

The chair base lies on its back (studs pointing UP); a factory nut is dropped onto one
stud's thread tip and driven with nut_thread's exact press + twist. Telemetry prints
axial travel vs cumulative spin every 25 steps — a real thread advances ~2 mm per 360
degrees; axial travel without matching spin is crest-skipping.

This decides whether the horizontal-chair threading failures are axis-specific (catch
geometry under lateral gravity) or fundamental to the SDF pair at this force scale.

    /opt/venv/bin/python -m robobench.scripts.probe_thread_vertical --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=3200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

ASSETS = Path(__file__).resolve().parents[1] / "suites" / "assembly" / "assets"
PRESS, TWIST, TARGET_W = -2.5, -0.15, -3.0
STUD_X, STUD_Z_LOCAL = 0.14, 0.397  # stud axis in the chair-base frame
SHANK_Y1, THREAD_TIP = 0.2946, 0.3296

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


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=device, physx=PHYSX))

    # Chair base on its back: -90 deg about x maps base +y (stud axis) to world +z.
    q = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0)

    @configclass
    class VCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                usd_path=str(ASSETS / "props" / "ground" / "default_ground.usd")),
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
        )
        base = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ChairBase",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ASSETS / "chair" / "chair_base.usd"),
                activate_contact_sensors=True,
                # dynamic-but-immovable vise: 50 kg, gravity off — holds pose under the
                # 2.5 N press without kinematic_enabled (SDF-on-kinematic stays out of
                # the experiment's variables), and needs no resting pose analysis
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.30), rot=q),
        )
        nut = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Nut",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ASSETS / "factory" / "factory_nut_m16.usd"),
                activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                    sleep_threshold=0.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.3, 0.05)),
        )

    scene = InteractiveScene(VCfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    nut, base = scene["nut"], scene["base"]
    dt = sim.get_physics_dt()

    # Stud axis is world +z; the stud line in world coords: base at (0,0,0.30) rotated by
    # q: base-local (x=-STUD_X, y, z=STUD_Z_LOCAL) -> world (x=-STUD_X, y=STUD_Z_LOCAL,
    # z=0.30 + y_local). Thread tip local y=THREAD_TIP -> world z = 0.30 + THREAD_TIP.
    sx, sy = -STUD_X, STUD_Z_LOCAL
    tip_z = 0.30 + THREAD_TIP
    print(f"[vprobe] stud line at ({sx:.3f},{sy:.3f}), tip z={tip_z:.4f}", flush=True)

    # Stage: nut 4 mm above the tip, axis up (identity quat), tiny descent velocity.
    st = torch.zeros(1, 13, device=device)
    st[:, 0] = sx + scene.env_origins[0, 0]
    st[:, 1] = sy + scene.env_origins[0, 1]
    st[:, 2] = tip_z + 0.004
    st[:, 3] = 1.0
    st[:, 9] = -0.02
    nut.write_root_state_to_sim(st, torch.tensor([0], device=device))

    f = torch.zeros(1, 1, 3, device=device)
    f[:, 0, 2] = PRESS
    cum_spin = 0.0
    depth0 = spin0 = None
    for i in range(args.steps):
        wz = nut.data.root_ang_vel_w[:, 2]
        t = torch.zeros(1, 1, 3, device=device)
        t[wz > TARGET_W, 0, 2] = TWIST
        nut.set_external_force_and_torque(f, t)
        sim.step()
        scene.update(dt)
        cum_spin += float(wz[0]) * dt
        depth = tip_z - float(nut.data.root_pos_w[0, 2])
        lat = math.hypot(float(nut.data.root_pos_w[0, 0] - scene.env_origins[0, 0]) - sx,
                         float(nut.data.root_pos_w[0, 1] - scene.env_origins[0, 1]) - sy)
        if depth0 is None and depth > 0.002:
            depth0, spin0 = depth, cum_spin
            print(f"[vprobe] ENGAGED at step {i}", flush=True)
        if i % 25 == 24:
            extra = ""
            if depth0 is not None:
                trav = (depth - depth0) * 1000
                spun = math.degrees(abs(cum_spin - spin0))
                extra = f" travel={trav:+.1f}mm spun={spun:.0f}deg (helix wants {abs(trav) * 180:.0f}deg)"
            print(f"[vprobe] step {i + 1:4d} depth={depth * 1000:+.1f}mm lat={lat * 1000:.1f}mm{extra}",
                  flush=True)
        if depth0 is not None and (depth - depth0) > 0.020:
            trav = (depth - depth0) * 1000
            spun = math.degrees(abs(cum_spin - spin0))
            print(f"[vprobe] DONE: travel={trav:+.1f}mm spun={spun:.0f}deg "
                  f"(helix wants {abs(trav) * 180:.0f}deg) "
                  f"ratio={spun / max(abs(trav) * 180, 1e-9):.2f}", flush=True)
            break
    print("VPROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    import threading as _threading

    watchdog = _threading.Timer(10.0, lambda: _os._exit(0))
    watchdog.daemon = True
    watchdog.start()
    app.close()
    _os._exit(0)
