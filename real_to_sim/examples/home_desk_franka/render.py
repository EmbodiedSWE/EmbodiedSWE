#!/usr/bin/env python3
"""Step 6 concrete: render the recorded run into composited videos.

    python render.py <scene> <run_name>

Renders splat backgrounds at the recorded camera poses (one image for the
fixed external camera, per-frame for the wrist), mask-composites the Isaac
foreground over them, and writes external/wrist/both mp4s. Runs inside the
stage venv (self-bootstraps).
"""

import json
import os
import pathlib
import shlex
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve()
BG = _HERE.parents[2] / "background"
_VENV = BG / ".venv" / "bin" / "activate"
if os.environ.get("_R2S_BOOTSTRAPPED") != "1":
    os.environ["_R2S_BOOTSTRAPPED"] = "1"
    cmd = f"source {shlex.quote(str(_VENV))} && exec python " + " ".join(shlex.quote(a) for a in sys.argv)
    os.execve("/bin/bash", ["bash", "-c", cmd], os.environ)

import cv2
import numpy as np

if len(sys.argv) < 3:
    sys.exit("usage: render.py <scene> <run_name>")
SCENE, RUN = sys.argv[1], sys.argv[2]
DATA = BG / "data"
WSDIR = DATA / "colmap" / SCENE
RUND = DATA / "outputs" / RUN
REC = RUND / "franka_desk"
RB = BG / "scripts" / "render_background.py"
CLASSES = {"robot", "table", "object"}

ckpt = json.load(open(WSDIR / "scene.json"))["checkpoint"]

# --- backgrounds ------------------------------------------------------------
ei = json.load(open(REC / "ext_pose.json"))["intrinsics"]
subprocess.run(["python", str(RB), "--checkpoint", ckpt, "--dataset", str(WSDIR),
                "--pose", str(REC / "ext_pose.json"),
                "--fx", str(ei["fx"]), "--fy", str(ei["fy"]), "--cx", str(ei["cx"]),
                "--cy", str(ei["cy"]), "--width", str(ei["width"]), "--height", str(ei["height"]),
                "--out", str(RUND / "ext_bg.png")], check=True)
wi = json.load(open(REC / "wrist_poses.json"))["intrinsics"]
subprocess.run(["python", str(RB), "--checkpoint", ckpt, "--dataset", str(WSDIR),
                "--poses", str(REC / "wrist_poses.json"),
                "--fx", str(wi["fx"]), "--fy", str(wi["fy"]), "--cx", str(wi["cx"]),
                "--cy", str(wi["cy"]), "--width", str(wi["width"]), "--height", str(wi["height"]),
                "--out-dir", str(RUND / "wrist_bgs")], check=True)

# --- composite --------------------------------------------------------------
def composite_cam(cam, bg_static=None, bg_dir=None):
    rgb_dir, seg_dir = REC / cam / "rgb", REC / cam / "semantic_segmentation"
    out_dir = REC / cam / "composite"
    out_dir.mkdir(exist_ok=True)
    frames = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    for i, f in enumerate(frames):
        idx = f[len("rgb_"):-len(".png")]
        fg = cv2.imread(str(rgb_dir / f))
        seg = cv2.imread(str(seg_dir / f"semantic_segmentation_{idx}.png"), cv2.IMREAD_UNCHANGED)
        labels = json.load(open(seg_dir / f"semantic_segmentation_labels_{idx}.json"))
        ids = [int(k) for k, v in labels.items() if isinstance(v, dict) and v.get("class") in CLASSES]
        mask = np.isin(seg, ids)
        bg = bg_static if bg_static is not None else cv2.imread(str(bg_dir / f"bg_{i:04d}.png"))
        out = bg.copy()
        out[mask] = fg[mask]
        cv2.imwrite(str(out_dir / f"comp_{i:04d}.png"), out)
    print(cam, "composited", len(frames), "frames")
    return out_dir

ext_dir = composite_cam("external", bg_static=cv2.imread(str(RUND / "ext_bg.png")))
wrist_dir = composite_cam("wrist", bg_dir=RUND / "wrist_bgs")

# --- videos ------------------------------------------------------------------
for name, d in [("external", ext_dir), ("wrist", wrist_dir)]:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-i", str(d / "comp_%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(RUND / f"{name}.mp4")], check=True)
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(RUND / "external.mp4"), "-i", str(RUND / "wrist.mp4"),
                "-filter_complex", "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack=inputs=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(RUND / "both.mp4")], check=True)
print("VIDEOS:", RUND)
