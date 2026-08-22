# home_desk_franka — one scene, end to end (steps 4–6 concrete)

A real home desk, captured per step 1, then: calibrated → a Franka constructed
beside it in Isaac (external + wrist cameras) → rendered as composited videos.

Data (no raw videos needed — the processed scene suffices):
```bash
hf download CoSiGen/real2sim-home-desk --repo-type dataset \
    --include "colmap/*" "runs/*" --local-dir ../../background/data
```

```bash
python calibrate_scene.py home_desk_v2 --height 0.77 --extents 0.57 0.43   # step 4
python demo_franka_scene.py home_desk_v2 my_run                           # step 5
python render.py home_desk_v2 my_run                                      # step 6
# -> background/data/outputs/my_run/{external,wrist,both}.mp4
```

- `calibrate_scene.py` — tabletop calibration: fits the work-surface plane and
  the floor below, pins scale to the measured height, writes `scene.json`;
  also picks the demo's external camera (best capture pose framing
  surface + robot) into `ext_cam_real.json`.
- `demo_franka_scene.py` — constructs the scene in Isaac (surface collider,
  Franka at x = −0.5 m, cameras) and records RGB + masks + camera poses.
- `render.py` — splat backgrounds at the recorded poses, mask composite,
  videos.

Deliberately opinionated (this robot, these cameras) — adapt per scene rather
than generalize.
