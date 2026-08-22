# Converting a real background into simulation

Capture a real scene with a phone → train a Gaussian splat → render it as the
photoreal background behind Isaac's robot/objects (mask composite).
All data lives in `data/` here (gitignored).

## Step 0 — install

```bash
bash scripts/install.sh
```

Idempotent; ~20–40 min first run. Creates this stage's own env at `.venv/`
(torch+CUDA splat stack — kept separate from the isaacsim env on purpose) and
builds the pinned `3dgrut/` submodule into it. Needs: uv, NVIDIA GPU, gcc ≤ 14,
ffmpeg, wget, ~10 GB disk. New-distro quirks are handled automatically.

Verify: `source .venv/bin/activate && python -c "import torch, pycolmap, threedgrut; print(torch.cuda.is_available())"`

## Step 1 — capture (no code)

Photograph the scene: **~300 phone photos** (preferred; HEIC fine) and/or slow
4K video with locked exposure. Cover three things: where the sim cameras will
be, close-up sweeps over the work surface, and a walk-around. Tape-measure the
**surface height from the floor**. Keep lighting constant; remove objects that
will be simulated; robot out of view.

```
data/captures/<name>/
    imgs/       # photos
    videos/     # optional videos
    notes.md    # measurements + anything worth remembering
```

## Step 2 — reconstruct camera poses

```bash
python scripts/reconstruct_cam_pose.py data/captures/<scene> <scene>
```

Photos/videos → culled images → COLMAP poses (`data/colmap/<scene>/`).
Slowest step (exhaustive matching). Good capture ≈ >90% images registered.

## Step 3 — train the splat

```bash
bash scripts/train_splat.sh data/colmap/<scene> <scene>
```

~30 min on a strong GPU. Product: `data/runs/<scene>/*/ckpt_last.pt`.
Quality check: compare `ours_<iter>/renders/` in the run folder against the real images.

## Step 4 — calibrate the scene

Pin the reconstruction to meters and define the world frame; write the result
as `data/colmap/<scene>/scene.json` (scale, surface plane, world rotation,
extents, checkpoint pointer) — the scene package all consumers read. Use a
large unambiguous measured span for the scale (a surface height over the
floor, a ChArUco board) — never the reconstructed surface extents. Sanity:
camera heights should look like a human holding a phone.

Concrete tabletop implementation: `../examples/home_desk_franka/calibrate_scene.py`
(fits the dominant work-surface plane + the floor below it):

```bash
python ../examples/home_desk_franka/calibrate_scene.py <scene> --height 0.77 [--extents 0.57 0.43]
```

## Step 5 — construct the scene

Place what the scene package can't know into Isaac: an invisible collider at
the surface (z=0, your measured extents), the robot, task objects, and cameras
(real calibration when you have a rig; else a chosen capture pose). Run
physics and record per camera and frame: RGB, segmentation masks of the
sim-owned prims, and camera poses converted to the scene's COLMAP frame
(`scene.json` has the transform).

Concrete: `python ../examples/home_desk_franka/demo_franka_scene.py <scene> <run>`

## Step 6 — render

For every recorded camera pose, render the splat background
(`scripts/render_background.py` — single pose or batch, any intrinsics), then
mask-composite: sim pixels where the mask is set, splat pixels elsewhere.
Fixed cameras need ONE background image; moving cameras one per frame.

Concrete: `python ../examples/home_desk_franka/render.py <scene> <run>`

## Test data

A complete worked scene (raw captures → COLMAP workspace → trained splat →
reference output videos) lives in the private HF dataset
[`CoSiGen/real2sim-home-desk`](https://huggingface.co/datasets/CoSiGen/real2sim-home-desk)
(org members have access):

```bash
hf download CoSiGen/real2sim-home-desk --repo-type dataset --local-dir data
```

Entry points after download: run the full pipeline from `data/captures/`
(steps 2-3, slow), or skip straight to the example's calibrate/construct/render
using the included `colmap/` workspace + checkpoint. `reference_outputs/` shows
what the final videos should look like.

## Notes

- The one law: **a splat is only sharp for views resembling the capture** —
  blur = seen badly, black = never seen. Fix at capture time (step 1).
- `scripts/render_background.py`: render the calibrated splat from ANY camera
  pose/intrinsics (single or batch) — the read-side of the scene package.
- More tools in `../examples/home_desk_franka/` (e.g. `compare_runs.py`).
