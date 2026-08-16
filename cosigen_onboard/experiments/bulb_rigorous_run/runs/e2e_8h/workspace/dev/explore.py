"""Explore the bulb env: geometry, poses, joint info, camera frame."""
import argparse, os, sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import torch
import robobench; robobench.discover()
from robobench.core.registries import ENVS

env = ENVS.get('assembly.bulb.franka.osc')().build(num_envs=1)
print("action_dim", env.robot.action_dim, "control_period", env.robot.control_period, "dt", env.dt)

art = env.robot.articulation
print("body_names", art.body_names)
print("joint_names", art.joint_names)
print("joint_pos", art.data.joint_pos[0].tolist())
print("joint_limits", art.data.joint_pos_limits[0].tolist())

scene = env.scene
bulb = scene.bulbs[0]
sock = scene.sockets[0]
print("bulb pos", bulb.data.root_pos_w[0].tolist(), "quat", bulb.data.root_quat_w[0].tolist())
print("sock pos", sock.data.root_pos_w[0].tolist(), "quat", sock.data.root_quat_w[0].tolist())
ee_idx = art.body_names.index("panda_hand")
print("ee pos", art.data.body_pos_w[0, ee_idx].tolist(), "quat", art.data.body_quat_w[0, ee_idx].tolist())

# settle physics a bit
zero = torch.zeros(1, env.robot.action_dim, device=env.device)
zero[:, 6:8] = 0.04
for _ in range(15):
    env.step(zero)
print("after settle:")
print("bulb pos", bulb.data.root_pos_w[0].tolist(), "quat", bulb.data.root_quat_w[0].tolist())
print("ee pos", art.data.body_pos_w[0, ee_idx].tolist(), "quat", art.data.body_quat_w[0, ee_idx].tolist())
print("offsets in socket", scene._bulb_offsets_in_socket()[0].tolist())
print("axis cos", scene._bulb_axis_cos()[0].tolist())
print("seated", scene.seated()[0].tolist())

# stage description around bulb and socket
for d in env.describe_stage():
    p = d.get("path", "")
    if "Bulb" in p or "Socket" in p or "hand" in p or "finger" in p:
        print(d)

# camera frame
import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep
cam = prim_utils.create_prim(
    "/World/DebugCam", "Camera",
    translation=(1.4, 0.9, 0.8),
    attributes={"focalLength": 18.0, "clippingRange": (0.01, 20.0)},
)
# aim camera at the table centre (0.5, 0, 0)
from pxr import Gf, UsdGeom
import numpy as np
eye = Gf.Vec3d(1.4, 0.9, 0.8); target = Gf.Vec3d(0.4, 0.1, 0.0)
m = Gf.Matrix4d(); m.SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
xf = UsdGeom.Xformable(cam)
xf.ClearXformOpOrder()
op = xf.AddTransformOp()
op.Set(m.GetInverse())
rp = rep.create.render_product("/World/DebugCam", (960, 720))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(rp)
for _ in range(3):
    env.step(zero, render=True)
import omni.kit.app
for _ in range(6):
    omni.kit.app.get_app().update()
data = ann.get_data()
from PIL import Image
Image.fromarray(data[..., :3]).save("/workspace/dev/scene.png")
print("saved /workspace/dev/scene.png", data.shape)

os._exit(0)
