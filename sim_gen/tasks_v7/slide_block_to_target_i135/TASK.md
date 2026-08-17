# die_roll — tumble the die over its edges until BLUE faces up, resting on the disc

**Package:** `sim_gen/tasks_v7/slide_block_to_target_i135`
**Env:** `simgen.die_roll` (scene-level, `robot="null"`)

## Seed provenance

`rlbench/slide_block_to_target`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/slide_block_to_target.py`): push a
red 5.6 cm cube across the floor until it sits on a flat target marker — a single
planar translation; success is position-only and the block's orientation never
matters.

## What changed and why it is strategically different

The target marker survives (the magenta disc), but the goal is now dominated by a
quantity the seed's plan cannot touch — the block's ORIENTATION:

- The cube becomes an 8 cm **DIE** with six distinctly colored face stickers (red +x /
  orange −x / green +y / yellow −y / **BLUE +z** / white −z in the body frame).
  Success requires the BLUE face pointing UP (within 20°) AND the die centre over the
  disc (≤ 7 cm) AND the die settled.
- Every episode spawns the die with blue NOT up — the up-face is sampled from the five
  non-blue faces (blue sideways in 4 of them, straight down in 1), plus free yaw and
  xy jitter on both die and disc. The seed's entire strategy — slide the object onto
  the marker — therefore produces a settled state the rubric REJECTS (smoke #6
  constructs exactly that end state and watches it fail).
- A rigid cube on a flat floor changes its up face **only by toppling**: pivoting over
  a bottom edge through the balance point and slapping down onto the next face — a
  quarter-roll, 90° of reorientation AND one edge-length (8 cm) of travel per roll.
  Sliding, spinning flat, or approaching the disc changes nothing the rubric wants.
  Reorientation and locomotion are **coupled**: the solver must plan a roll sequence
  (which edge, how many rolls: one if blue starts sideways, two same-direction if blue
  starts face-down) whose LAST roll lands blue-up with the centre on the disc.

A solver therefore needs a different PLAN (discrete reorientation planning over the
die's orientations, executed as edge-pivot topples with the landing cell on the disc —
not one planar push at a marker) and different CODE STRUCTURE (per-roll pivot control
with tilt readback, torque cut at the balance point, hands-off fall and settle
verification per roll — instead of a single position servo). The seed's plan, executed
here, is the smoke #6 reject state.

## Teleport-solution outline (solve.py, phases; `SIM_GEN_SCORE` at each boundary)

- **P0** reset + settle; layout readback printed (die xy + up-face, disc xy); asserts
  blue not up and score ~0.
- **P1 TRANSPORT (teleport):** one pose write stages the die at
  `disc_centre − k·s·d̂` (k = planned roll count, s = 8 cm edge) — orientation
  PRESERVED EXACTLY (a teleport that rotated the die would bypass the load-bearing
  reorientation; no teleport in solve.py ever changes the quaternion), asserted
  OUTSIDE the scoring zone by readback.
- **P2 QUARTER-ROLLS (contact dynamics — never teleported):** each roll applies a
  velocity-capped torque (0.16 N·m start, ω ≤ 1.2 rad/s; tipping threshold is
  mg·s/2 ≈ 0.098 N·m) about the horizontal edge axis — the applied-wrench emulation of
  the arm pushing high on the die's face. The torque is CUT at 55° of tilt; the
  balance point, the fall onto the next face, the landing slap and the settling rock
  are all hands-off physics; each roll is verified by orientation readback (the old
  up-face must end pointing along the roll direction). Stall handling escalates torque
  then flips to body-frame wrench encoding (pod-dependent frame drag; the torque axis
  IS the rotation axis, so world-frame encoding is drag-invariant by construction).
  Between rolls of a multi-roll plan, landing skid is absorbed by re-staging (pure
  translation, blue not yet up, outside the zone). If a plan ends blue-up but
  off-disc, recovery re-stages 4 edge lengths out and rolls 4× in one direction
  (blue: up → leading → down → trailing → up), landing on the disc.
- **P3** hands-off settle; success() verified live.
- **P4** hands-off persistence 3.5 s, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds **0, 1** (blue sideways — one-roll plans) and **28** (blue
straight down — the two-roll plan: white → yellow → blue), all
`SIM_GEN_SOLVE: SUCCESS` first run; scores monotone 0 → 0.09–0.17 → 1.0 → 1.0, die
landing 0.000 m from the disc centre on every seed.

## Embodiment argument (single Franka arm, parallel jaw, OSC)

Plausible base pose: **base at the world origin on the floor plane**; all required
contacts lie at radius 0.10–0.62 m, heights ≤ 8 cm — inside the Franka envelope.

- **The die is never grasped** — at 8 cm it is wider than the parallel jaw's maximum
  opening, which is exactly why this is a nonprehensile push-topple task. Each
  quarter-roll is a closed-fingertip push HIGH on the die's trailing face (contact at
  ~6–7.5 cm height, above the pivot edge): pushing above the edge line produces the
  same torque about the bottom edge as the solve's applied wrench, and the arm backs
  off once the die passes the balance point (~55° tilt), matching the torque cut. The
  ±20° blue tolerance and the 7 cm zone radius sit far above OSC noise; landing skid
  is absorbed by where the next push starts, exactly as solve.py re-stages.
- **The disc** is a 4 mm plate — an overlay marker, not an obstacle; the final roll
  simply lands on it. No contact with the disc is ever required.

## Execution order

NOT constrained (declared in describe()): any roll order/path that ends blue-up,
centred on the disc, settled, is accepted. Rolls may happen anywhere on the open
floor; only the final settled state is judged.

## Rubric

`score()` = 0.30·blue_ever (die ever blue-up while rotationally quiet — latched; only
a real reorientation, or a smoke probe, sets it) + 0.25·approach (latched running max
of 1 − d/0.25; the minimum spawn gap of 0.30 m makes this exactly 0 at reset) +
0.15·zone_ever (die centre ever over the disc at resting height, latched), capped at
0.70; exactly **1.0 iff `success()`**: blue face within 20° of world-up ∧ centre
within 7 cm of the disc centre at resting height ∧ linear AND angular velocity below
the settle gates — live physical state, no latches. Null policy scores ~0.

## Check list (smoke.py — rejection battery, forge: `SIM_GEN_SMOKE: ALL PASS 14/14`)

1. settle/no-NaN: die at rest on the floor, disc flat at its slot, still, blue NOT up
2. score ~0 at reset, no success
3. randomization readback: up-face varies across 8 seeded resets, NEVER blue
4. randomization readback: die xy jitter (> 8 mm), disc xy jitter (> 8 mm), spawn yaw
   spread (blue-normal azimuths differ > 10°)
5. null policy (240 steps): score ~0, no success
6. **seed strategy**: die slid onto the disc with its spawn orientation preserved →
   in-zone True but NOT success, score ≤ 0.45
7. blue-down in zone: settled on the disc blue straight down (two rolls short) →
   NOT success, score ≤ 0.45
8. blue-side in zone: settled on the disc blue sideways (one roll short) →
   NOT success, score ≤ 0.45
9. tolerance gate: in zone, zero velocity, blue 25° off vertical (> 20° tol), judged
   on the written state → in-zone AND settled read True, yet NOT success (then
   relocated before it can topple flat)
10. latched credit: teleporting the die far away after earning zone/approach credit
    leaves the latched score unchanged, still no success
11. position near-miss: settled blue-UP just outside the zone radius → NOT success,
    score ≤ 0.61
12. airborne: blue-up in free fall directly over the disc, judged mid-air → NOT
    success (resting-height gate; relocated before landing)
13. rejection audit: success() never True anywhere in the battery
14. final no-NaN

frames.npz recorded and saved in cwd by smoke.py.
