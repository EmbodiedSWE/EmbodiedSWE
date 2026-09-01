"""Dev driver: run solve phases with camera captures + state snapshots.

Usage: python -u driver.py [--from SNAP] [--phases pick,reorient,over,descend,screw]
Snapshots land in /workspace/dev/snaps/, frames in /workspace/dev/frames/.
"""
import argparse, os, sys

p = argparse.ArgumentParser()
p.add_argument("--load", default="", help="snapshot name to restore before running")
p.add_argument("--phases", default="pick,reorient,over,prewind,descend,screw")
p.add_argument("--video", action="store_true", help="capture a frame every step")
args = p.parse_args()

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import torch
import robobench; robobench.discover()
from robobench.core.registries import ENVS

env = ENVS.get('assembly.bulb.franka.osc')().build(num_envs=1)

sys.path.insert(0, "/workspace/solution")
import solve as S

os.makedirs("/workspace/dev/snaps", exist_ok=True)
os.makedirs("/workspace/dev/frames", exist_ok=True)

# ---- camera ---------------------------------------------------------------
import isaacsim.core.utils.prims as prim_utils
import omni.replicator.core as rep
from pxr import Gf, UsdGeom
from PIL import Image

cam = prim_utils.create_prim("/World/DebugCam", "Camera",
                             attributes={"focalLength": 24.0, "clippingRange": (0.01, 20.0)})
def aim(eye, target):
    m = Gf.Matrix4d(); m.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
    xf = UsdGeom.Xformable(cam); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(m.GetInverse())
aim((0.95, 0.55, 0.45), (0.35, 0.08, 0.06))
rp = rep.create.render_product("/World/DebugCam", (800, 600))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(rp)

frame_i = [0]
def snap_frame(tag=""):
    zero = torch.zeros(1, env.robot.action_dim, device=env.device)
    env.sim.render()
    import omni.kit.app
    for _ in range(2):
        omni.kit.app.get_app().update()
    data = ann.get_data()
    Image.fromarray(data[..., :3]).save(f"/workspace/dev/frames/{frame_i[0]:04d}_{tag}.png")
    frame_i[0] += 1

task = S.BulbTask(env, log=print)

if args.video:
    orig_act = task.act
    def act_and_frame(dpos, drot, grip=None):
        orig_act(dpos, drot, grip)
        if task.steps % 3 == 0:
            snap_frame(f"s{task.steps:05d}")
    task.act = act_and_frame

if args.load:
    states = torch.load(f"/workspace/dev/snaps/{args.load}.pt", weights_only=False)
    env.set_states(states)
    # keep the restored gripper command (don't let the default open it)
    task._grip = float(task.art.data.joint_pos_target[0, task.fin_idx].mean())
    task.hold(None, 2)
    print("restored", args.load, "grip", task._grip)

def save_snap(name):
    torch.save(env.get_states(), f"/workspace/dev/snaps/{name}.pt")
    print("snapshot", name)

phases = args.phases.split(",")
try:
    for ph in phases:
        print(f"=== phase {ph} (step {task.steps}) ===")
        if ph == "pick":
            task.hold(S.GRIP_OPEN, 8)
            snap_frame("start")
            ok = task.pick()
            snap_frame("picked")
            print("PICK", ok, "bulb", task.bulb_pos.tolist(), "gap", task.finger_gap)
            if not ok:
                break
            save_snap("picked")
        elif ph == "reorient":
            task.reorient()
            snap_frame("reoriented")
            save_snap("reoriented")
        elif ph == "over":
            task.move_over_socket()
            snap_frame("over")
            save_snap("over")
        elif ph == "prewind":
            task.prewind()
            task.move_over_socket()
            snap_frame("prewound")
            save_snap("prewound")
        elif ph == "descend":
            task.descend_to_rest()
            snap_frame("rested")
            save_snap("rested")
        elif ph == "screw":
            ok = task.screw()
            snap_frame("done")
            print("SCREW", ok)
    print("FINAL seated:", task.seated(), "depth:", task.bulb_depth(), "steps:", task.steps)
except Exception:
    import traceback
    traceback.print_exc()
    snap_frame("crash")
print("DONE-DRIVER")
os._exit(0)
