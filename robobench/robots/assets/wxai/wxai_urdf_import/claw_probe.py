"""Decisive claw probe for the URDF-imported WXAI: do jaw collision shapes TRACK on GPU?

Method (campaign-proven): gravity-off free cube parked in the jaw gap; close the
carriages. Frozen shapes -> jaws sweep through, cube never moves, carriages hit target.
Tracking shapes -> the cube is pushed/pinched and the carriages stall on it.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import quat_apply

SP = "/tmp/claude-1000/-home-yilang-research-CoSiGen/d3f3c284-6383-4fab-8182-739421689887/scratchpad"
USD = f"{SP}/wxai_urdf_import/wxai_follower_urdf.usd"

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))

ground = sim_utils.GroundPlaneCfg()
ground.func("/World/ground", ground)

robot = Articulation(ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD, activate_contact_sensors=False),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "arm": ImplicitActuatorCfg(joint_names_expr=["joint_.*"], stiffness=600.0, damping=60.0),
        "grip": ImplicitActuatorCfg(joint_names_expr=[".*carriage_joint"], stiffness=2000.0, damping=100.0),
    },
))

cube = RigidObject(RigidObjectCfg(
    prim_path="/World/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.015, 0.015, 0.015),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 0.5)),
))

sim.reset()
print("[probe] joints:", robot.joint_names, flush=True)
print("[probe] bodies:", robot.body_names, flush=True)

gidx = [robot.joint_names.index(j) for j in ("left_carriage_joint", "right_carriage_joint")]
open_c = 0.041

def step_hold(nsteps: int, grip: float) -> None:
    tgt = robot.data.default_joint_pos.clone()
    tgt[:, gidx] = grip
    for _ in range(nsteps):
        robot.set_joint_position_target(tgt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        cube.update(sim.get_physics_dt())

# open the jaws and let everything settle
step_hold(60, open_c)

# park the cube in the jaw gap: carriage midpoint, pushed down the tool axis into the blade zone
ee = robot.body_names.index("link_6")
cl = robot.body_names.index("carriage_left")
cr = robot.body_names.index("carriage_right")
p6 = robot.data.body_pos_w[:, ee]
q6 = robot.data.body_quat_w[:, ee]
mid = 0.5 * (robot.data.body_pos_w[:, cl] + robot.data.body_pos_w[:, cr])
gap0 = (robot.data.body_pos_w[:, cl] - robot.data.body_pos_w[:, cr]).norm(dim=-1)
tip_zone = mid + quat_apply(q6, torch.tensor([[0.058, 0.0, 0.0]], device="cuda:0"))
st = torch.zeros(1, 13, device="cuda:0")
st[:, 0:3] = tip_zone
st[:, 3] = 1.0
cube.write_root_state_to_sim(st)
for _ in range(4):
    robot.set_joint_position_target(robot.data.default_joint_pos)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    cube.update(sim.get_physics_dt())
c0 = cube.data.root_pos_w.clone()
print(f"[probe] link6 z {float(p6[0, 2]):.3f} | carriage gap {float(gap0[0]) * 1e3:.1f}mm"
      f" | cube parked at {[round(float(v), 3) for v in c0[0]]}", flush=True)

# CLOSE onto the cube
for k in range(140):
    tgt = robot.data.default_joint_pos.clone()
    tgt[:, gidx] = 0.004
    robot.set_joint_position_target(tgt)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.get_physics_dt())
    cube.update(sim.get_physics_dt())
    if k % 35 == 0:
        jp = robot.data.joint_pos[0, gidx]
        dc = (cube.data.root_pos_w - c0).norm(dim=-1)
        print(f"[probe] close k{k:3d}: carriages ({float(jp[0]) * 1e3:5.1f}, {float(jp[1]) * 1e3:5.1f})mm"
              f" | cube moved {float(dc[0]) * 1e3:5.2f}mm", flush=True)

jp = robot.data.joint_pos[0, gidx]
dc = (cube.data.root_pos_w - c0).norm(dim=-1)
pair = float(jp[0] + jp[1]) * 1e3
moved = float(dc[0]) * 1e3
print(f"[probe] FINAL: carriage pair-sum {pair:.1f}mm (cube would stall ~>=15mm; empty close ~8mm)"
      f" | cube displaced {moved:.2f}mm", flush=True)
if moved > 2.0 or pair > 14.0:
    print("[probe] VERDICT: SHAPES TRACK — the URDF import fixed the claw. Real grasps possible.", flush=True)
else:
    print("[probe] VERDICT: shapes still frozen (swept through the cube).", flush=True)
app.close()
