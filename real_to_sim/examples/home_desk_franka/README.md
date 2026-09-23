# home_desk_franka — one scene, end to end (steps 4–6 concrete)

A real home desk, captured per step 1, then: calibrated → a Franka constructed
beside it in Isaac (external + wrist cameras) → rendered as composited videos.

Data (no raw videos needed — the processed scene suffices):
```bash
hf download EmbodiedSWE/real2sim-home-desk --repo-type dataset \
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

## With an object on the desk (pick demo)

Generated real objects (see `../../objects/README.md`) spawn on the desk via
`--object`; when objects are present, the arm tries to pick the first one
(RMPflow: hover → descend → close → lift). The worked bottle asset:

```bash
hf download EmbodiedSWE/real2sim-home-desk --repo-type dataset \
    --include "objects/bottle/*" --local-dir ../../objects/data
python calibrate_object.py bottle --diameter 0.08 --mass 0.35 --flip  # optional: bottle.usd ships baked
python demo_franka_scene.py home_desk_v2 my_pick --object bottle:0.02,0.0,30
python render.py home_desk_v2 my_pick
# -> background/data/outputs/my_pick/{external,wrist,both}.mp4
```

- `calibrate_object.py` — objects-stage mesh + one caliper measurement →
  rigid-body USD (visual mesh, lathe-stack colliders, friction, mass).
- Expected result (`reference_outputs/bottle_pick/` in the dataset): the arm
  grasps the cap and lifts; the bottle slips near the end of the hold. That's
  honest physics — this 8 cm bottle sits exactly at the Panda gripper's 8 cm
  maximum opening, so only the tapering cap-top is pinchable.
- `--object` is repeatable (`name[:x,y[,yaw_deg]]`); without positions the
  objects walk along preset desk spots. Without any `--object`, the demo does
  its original random task-envelope motion (`reference_outputs/*.mp4`).

Deliberately opinionated (this robot, these cameras) — adapt per scene rather
than generalize.
