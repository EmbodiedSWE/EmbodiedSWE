"""Render a trained 3DGUT splat from an ARBITRARY camera pose.

Pose: 4x4 cam-to-world in the model's frame (raw COLMAP world for our runs),
OpenCV camera convention (x right, y down, z forward). This is the standalone
splat renderer for the composite pipeline (backgrounds at recorded cam poses).

Usage:
  python render_splat_pose.py --checkpoint ckpt.pt --dataset <colmap ws> \
      --pose pose.json --fx 890 --fy 890 --cx 640 --cy 360 --width 1280 --height 720 \
      --out out.png
pose.json: {"T_cam_to_world": [[...4x4...]]}  (or 16 flat numbers)
"""

import argparse
import json
import os
import pathlib
import sys

_STAGE = pathlib.Path(__file__).resolve().parents[1]
_VENV_PY = _STAGE / ".venv" / "bin" / "python"
if _VENV_PY.exists() and pathlib.Path(sys.executable).resolve() != _VENV_PY.resolve():
    # re-exec through activate: threedgrut's JIT needs the venv env vars
    # (slangc on PATH, CUDA_HOME, ...) that a bare interpreter swap skips
    import shlex

    cmd = f'source {shlex.quote(str(_STAGE / ".venv/bin/activate"))} && exec python ' + " ".join(
        shlex.quote(a) for a in sys.argv)
    os.execv("/bin/bash", ["bash", "-c", cmd])


import numpy as np
import torch

from threedgrut.datasets.protocols import Batch
from threedgrut.datasets.utils import pinhole_camera_rays
from threedgrut.render import Renderer
from threedgrut.utils.render import apply_background

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset", required=True, help="colmap workspace (for Renderer init)")
ap.add_argument("--pose", help="json with T_cam_to_world (single render)")
ap.add_argument("--poses", help="json with {'poses': [4x4,...]} (batch render)")
ap.add_argument("--out-dir", help="output dir for batch mode (bg_%%04d.png)")
ap.add_argument("--fx", type=float, required=True)
ap.add_argument("--fy", type=float, required=True)
ap.add_argument("--cx", type=float, required=True)
ap.add_argument("--cy", type=float, required=True)
ap.add_argument("--width", type=int, required=True)
ap.add_argument("--height", type=int, required=True)
ap.add_argument("--out")
args = ap.parse_args()

renderer = Renderer.from_checkpoint(
    checkpoint_path=args.checkpoint,
    path=args.dataset,
    out_dir="/tmp/splat_render_tmp",
    save_gt=False,
    computes_extra_metrics=False,
)
model = renderer.model
device = "cuda"

if args.poses:
    pose_list = [np.array(p, dtype=np.float32).reshape(4, 4) for p in json.load(open(args.poses))["poses"]]
    assert args.out_dir
    import os
    os.makedirs(args.out_dir, exist_ok=True)
else:
    pose_list = None
    T = np.array(json.load(open(args.pose))["T_cam_to_world"], dtype=np.float32).reshape(4, 4)

W, H = args.width, args.height
x, y = np.meshgrid(np.arange(W), np.arange(H))
_, rays_dir = pinhole_camera_rays(x, y, args.fx, args.fy, W, H, cx=args.cx, cy=args.cy)
rays_dir = torch.tensor(rays_dir.reshape(1, H, W, 3), dtype=torch.float32, device=device)
rays_ori = torch.zeros_like(rays_dir)

from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType

params = OpenCVPinholeCameraModelParameters(
    resolution=np.array([W, H], dtype=np.uint64),
    shutter_type=ShutterType.GLOBAL,
    principal_point=np.array([args.cx, args.cy], dtype=np.float32),
    focal_length=np.array([args.fx, args.fy], dtype=np.float32),
    radial_coeffs=np.zeros((6,), dtype=np.float32),
    tangential_coeffs=np.zeros((2,), dtype=np.float32),
    thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
)

import cv2


def render_one(T_pose):
    batch = Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=torch.tensor(T_pose, dtype=torch.float32, device=device).unsqueeze(0),
        intrinsics_OpenCVPinholeCameraModelParameters=params.to_dict(),
    )
    with torch.no_grad():
        outputs = model(batch)
        outputs = apply_background(model.background, outputs, batch, training=False)
        pp = getattr(renderer, "post_processing", None)
        if pp:
            from threedgrut.utils.render import apply_post_processing
            outputs = apply_post_processing(pp, outputs, batch, training=False)
    rgb = outputs["pred_features"] if "pred_features" in outputs else outputs["pred_rgb"]
    return (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


if pose_list is not None:
    for i, Tp in enumerate(pose_list):
        img = render_one(Tp)
        cv2.imwrite(f"{args.out_dir}/bg_{i:04d}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if i % 40 == 0:
            print("rendered", i, flush=True)
    print("WROTE", len(pose_list), "backgrounds to", args.out_dir)
else:
    img = render_one(T)
    cv2.imwrite(args.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("WROTE", args.out, img.shape)
