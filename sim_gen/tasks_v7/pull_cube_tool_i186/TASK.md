# shelf_drop_dispatch (`pull_cube_tool_i186`)

Pull the loaded falsework column out from under a raised hinged shelf so the shelf
swings down into a ramp and **gravity delivers** the out-of-reach cube into a walled
catch pit near the robot; then pick the cube out of the pit and set it **centered on
top of the yellow pedestal** (position randomized per episode).

- Env id: `simgen.shelf_drop_dispatch` (NullRobot scene-level env)
- Files: `scene.py` (scene + rubric + plant), `solve.py` (teleport solution
  certificate), `smoke.py` (16-check rejection battery, recorded), this file.

## Seed provenance

| | seed | this task |
|---|---|---|
| source | `maniskill/pull_cube_tool` (RoboVerse `roboverse_pack/tasks/maniskill/pull_cube_tool.py`) | new scene, procedural geometry only |
| setup | L-shaped tool within reach; cube beyond reach on open floor | NO portable tool; cube beyond reach on an **elevated hinged shelf**; a removable support column (falsework) under the shelf midspan is the only reachable thing that matters |
| plan | grasp the tool, hook it behind the cube, **drag** the cube inward under direct quasi-static control | **extract the loaded column sideways** out of its floor channel; the shelf swings down on a damped hinge; the cube slides down the ramp **by itself**; then a terminal precision pick-and-place |
| goal | proximity disc: cube within 0.6 m of the base | cube resting settled, **centered on a pedestal top** (±4.5 cm per axis, rest height, velocity-settled) |

## Why strategically different

- **vs the seed**: the seed is reach-extension with a hand-held implement — the cube's
  whole trajectory is under direct control, and the goal is a proximity region. Here
  the arm **never touches the cube while it is far away**: the far-field transport leg
  is a stored-energy RELEASE (potential energy banked in the raised shelf + cube) with
  entirely **passive** delivery under gravity, and the terminal goal is a precision
  place on a randomized pedestal — not a proximity disc. No tool is grasped or
  ferried; the manipulated object (the blue column) is *removed from* the mechanism,
  not *applied to* the cargo.
- **vs the sibling `pull_cube_tool_i1` (carousel_ferry)**: that task is a cyclic
  anchored mechanism *driven* to ferry the cube around (continuous transport under
  active drive, cube rides the mechanism the whole way). Here the mechanism is
  **one-shot and irreversible** (falsework extraction — you cannot push the shelf back
  up), the drive interaction is an extraction *under load* (friction of a loaded
  column sliding in a channel), and transport is ballistic/passive, not carried.

## Mechanics

Plain rigid bodies + one authored USD revolute joint (kinematic post → shelf, axis Y,
limits [−15°, +0.5°]). A `post_step` plant applies viscous hinge damping
(τ = −2.0·ω, so the release swings down in ~0.5–1 s instead of slamming) and consumes
the `prop_force` buffer (bounded world force at the column CoM — the same push a
fingertip exerts on a face). Slick shelf top / channel floor (μ ≈ 0.08–0.15) so the
15° ramp beats friction ~2.2× and the loaded column slides without tipping; the pit's
9 cm walls retain the ~1 m/s arriving cube (KE 0.023 J < 0.033 J wall climb).

## Rubric (latched, anchored in solve.py)

- `released` (latch): shelf pitch ≤ −10° — physically impossible while the column is
  under the shelf.
- `delivered` (latch): after release, cube inside the pit footprint **below the wall
  height** (z < 0.05 — containment, not aperture transit).
- `success()`: `released` AND cube currently resting settled, centered on the pedestal
  top (per-axis ±4.5 cm, z at rest ±1.2 cm, |v| < 0.04, |ω| < 0.6).
- `score()` = 1.0 iff success, else 0.25·released + 0.30·delivered. Null ≈ 0; a
  knock-off after success falls back to 0.55, never 0.

**Execution order is REQUIRED and physically enforced**: the cube spawns > 0.86 m
(3D) from the base — beyond the 0.855 m Franka envelope — so no placement is possible
before the release; and `success` requires the `released` latch, so a cube placed on
the pedestal by any means before the shelf has physically dropped scores ~0
(smoke check 6).

## Randomization (verified by READBACK in smoke)

Cube spawn x ∈ [0.84, 0.96], y ± 4 cm, free yaw (on the shelf); column channel
y ± 1.5 cm; pedestal xy = (0.26, −0.34) ± 5 cm.

## Solution outline (solve.py, teleport = transport only)

0. reset(seed), settle — assert level shelf + cube on it; score ~0.
1. **EXTRACT** (contact dynamics): closed-loop y-velocity servo force
   (F = 40·(−0.10 − v), |F| ≤ 12 N) on the column CoM through `prop_force` until the
   column is fully clear of the shelf width; zero force; the shelf swings down under
   gravity; the cube slides down the ramp and settles in the pit **untouched** →
   score 0.55.
2. **PLACE**: teleport the cube from the open pit to 3 cm **above** the pedestal top
   (transport across free space only — the in-reach pick a Franka does with a pinch
   grasp), drop; it impacts and settles by contact → score 1.00.
3. **PERSIST**: ≥ 3.5 s pure simulation, success at every poll → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (0 → 0.55 → 1.0 → 1.0).

## Franka embodiment (base at (0, 0), facing +x)

- **Blue column** (6 × 9 × 20 cm, mass 0.6 kg) at x ≈ 0.66, y ± 1.5 cm: side faces at
  ~0.60–0.72 m, top at 21 cm — inside the ~0.75 m comfortable envelope. Pinch the 6 cm
  width from above or hook a face and drag sideways along the open channel (the rails
  guide y-travel; either y direction works). Extraction needs ~2–4 N — fingertip scale.
- **Red cube** (4.5 cm): at spawn it is > 0.86 m away in 3D — deliberately
  unreachable. After delivery it sits in the pit (x 0.29–0.57, mouth 16 cm wide, walls
  9 cm, open top): a top-down pinch at ~0.3–0.55 m, standard.
- **Yellow pedestal** (16 cm square, top 12 cm) at (0.26, −0.34) ± 5 cm: ~0.35–0.48 m
  from the base — release the cube a few cm above the top, as solve.py does.
- Nothing requires reaching past 0.75 m, pushing above 12 N, or squeezing into
  closed cavities.

## Checks

- **solve.py**: 4 phase gates (initial level+loaded state; extraction completed;
  passive delivery latched ≥ 0.53; success + score exactly 1.0; persistence ≥ 3.5 s
  with success at every poll), non-decreasing `SIM_GEN_SCORE` ladder. Passes on
  seeds 0 and 1.
- **smoke.py**: 16 named checks — settle, randomization readback, out-of-reach
  spawn, null-policy stability (no column creep, shelf stays up), seed-strategy end
  state rejected, teleport-onto-pedestal anti-cheat rejected (order enforcement),
  partial-pull load-bearing negative (shelf must NOT drop), oracle extraction under
  load, passive delivery ~0.55, score monotonicity, floor-beside-pedestal near-miss,
  off-center-on-top near-miss, centered-drop exactness (score == 1.0), 2 s
  persistence, knock-out latch (falls to 0.55 exactly), frames recorded
  (`frames.npz` in CWD). Verdict line: `SIM_GEN_SMOKE: ALL PASS 16/16`.
