# push_cube_d1 — flip the die (blue face up)

| Field | Value |
|---|---|
| Seed | `RoboVerse/roboverse_pack/tasks/maniskill/push_cube.py` (ManiSkill PushCube) |
| Mutation | Objective changed from **translation** (cube into a goal region) to **reorientation** (cube must come to rest with a specific face pointing up). Position on the table is irrelevant. |
| Strategy | Tip / roll / pick-and-reorient the cube about a horizontal axis until the blue face is up, then let it settle flat. |

## Why this is strategically different

The seed's entire plan is *planar transport*: push the object along the table until its
xy position lands in a marked region; orientation is never touched and never judged.
Here the success predicate reads **only orientation** (world z-component of the cube's
blue-face normal) plus "settled flat on the table":

- Executing the seed's strategy verbatim — sliding the cube any distance across the
  table without reorienting it — changes nothing the criterion measures: success stays
  `False` and score stays ~0 (this is negative control A).
- Near-miss variants of reorientation also fail: rotating to an adjacent face (90° off,
  control B), leaving the blue face down (control C), and holding the cube blue-face-up
  in the air (control D) all fail.
- The solver needs a qualitatively different plan: choose a rotation (which axis, how
  many quarter-turns) that maps the current orientation to blue-up, and execute it
  through contact (tipping over edges) or by lifting and reorienting — not "move toward
  a target point".

## Semantics

- **Scene**: 5 cm die (gray body, six colored face decals; blue = target face on the
  body's +z) on a large table. No goal marker — position does not matter.
- **Instance distribution**: xy ~ U[-0.15, 0.15]²; initial orientation drawn from the
  five axis-aligned rest orientations with blue NOT up (blue down / blue on each of the
  four sides), composed with a uniform random yaw. No instance starts solved.
- **Success**: blue-face world normal within `up_tol_deg` (15°) of world +z AND the cube
  rests flat on the table (height within 5 mm of the rest height, speed < 0.05 m/s).
- **Score**: `1.0` iff success, else `0.95 · clip(up_z, 0, 1)` where `up_z` is the
  blue-normal's world z-component — 0 for blue-down and blue-sideways starts (so the
  null policy and the seed strategy score ~0), monotone as the rotation progresses.

## Checks (smoke.py)

settle/no-drift (position AND orientation) · no-NaN · determinism · physics readback
(cube friction & mass — tipping depends on them) · randomization-is-real (xy varies;
≥3 distinct starting faces across seeds; no seed starts solved) · null-policy fails
with score ~0 · oracle (lift–reorient–lower–settle) succeeds on 3 seeds · rubric
monotone (start ≤ mid-rotation ≤ end = 1.0) on 3 seeds · negative controls: (A) the
seed's own strategy — a 0.25 m planar slide with no reorientation — must fail with
score ~0; (B) adjacent-face-up must fail; (C) blue-face-down must fail; (D) blue-up
hover in the air must fail · calibration sweep: cube released at tilts {10°, 30°, 60°,
80°} from upright — sub-45° tilts fall back to blue-up (success), past-45° tilts tip
over to a side face (fail), pinning the physical knee of the criterion at the cube's
45° tipping angle.
