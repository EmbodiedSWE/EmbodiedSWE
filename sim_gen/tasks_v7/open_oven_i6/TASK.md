# open_oven_i6 — Dice Tumble

Tumble two oversized six-color dice onto two color-sampled floor pads, matching face up
(scene `dice_tumble`, env `simgen.dice_tumble`).

## Seed provenance

Seed: `rlbench/open_oven` — grasp the oven door's handle and rotate the door panel
about its BUILT hinge until the oven stands open. One prehensile act (grasp the
handle), one guided rotation on a fixed, pre-built axis, judged by an articulation
joint reading.

The seed's end state (a door swung open on its hinge) is **not expressible** in this
scene — there is no articulated object at all; documented N/A. The nearest naive plan
from the seed's family — transport the object to the goal region without reorienting
it — is constructed as a settled state in smoke check 6 and scores ~0.

## What the task is

Two dynamic 100 mm cubes ("dice", 350 g, hollow six-plate shells — each face IS one
color of a six-color palette) rest on the floor. Two thin 220 mm square pads lie
flush with the floor (collision-free painted markings), each painted ONE palette
color; the two pad colors are SAMPLED per episode (15 pairs), pad positions jitter
with free yaw, and each die spawns at a jittered slot with free yaw and an up-face
drawn from the four NON-pad colors — so every episode needs a fresh visual read and a
fresh tumble plan, and the null policy shows neither pad color.

Goal: for each pad, leave one die resting centered on it (5.5 cm per-axis window in
the pad's frame, at floor height) with the face matching THAT pad's color pointing
straight up (15°), both pads served at once, everything settled. A wrong face up,
off-center, stacked on the other die (height window rejects +100 mm), or still moving
counts for nothing.

## Strategic difference

- **vs. the seed**: the seed rotates a panel about a hinge that the scene BUILT for
  it, holding a handle designed to be held. Here there is no hinge, no handle, no
  appliance, and no grasp at all: the rotation the task demands must be MANUFACTURED
  by the solver, one quarter-turn at a time, by tipping a free cube over its own
  ground edge — the transient "hinge" is a contact line that exists only while the
  cube pivots on it. The goal is an SO(3) face-up predicate + placement, not a joint
  readout; the plan (how many tips, which directions) changes per episode with the
  sampled colors and orientations.
- **vs. corpus tasks read**: the door/articulation family (`close_microwave_i4/i5`,
  `close_grill_i8`, libero drawer/stove tasks) manipulates built joints; the
  pick-place family (libero bowls/pan, `coke_task_i15`, `pick_and_lift_i16`,
  `approach_grasp_spoon_i12`) transports grasped objects without any orientation
  goal; `peg_insertion_side_i1/i2` are insertions; `pick_single_egad_i3` slides
  through a covered tunnel, `pick_single_egad_i4` threads tags onto pegs
  (suspension); `setup_checkers_i2` drop-stacks; `pour_water_i7` pours. **No corpus
  task has nonprehensile in-place reorientation** — an SO(3) goal reached by
  edge-pivot quarter-turns — nor the height-keyed push dichotomy (same contact, push
  HIGH to tip / push LOW to slide) that this task is built on.
- The mechanism is physically honest and asserted in `__post_init__`: friction is
  high enough that a push at edge height tips before it slides, low enough that a
  low push slides before it tips (tip height (s/2)/μ must fall strictly inside the
  face), the die is wider than a parallel jaw, the pads are farther apart than one
  die can straddle, and the on-pad height window rejects a stacked die.

## Solution outline (solve.py — the legitimacy certificate)

Nothing is ever teleported at all — no `write_root_state` after reset; every
goal-directed change goes through contact dynamics. Per die (nearest-assignment
order; no order is required):

1. **Plan (read-only)** — read both pad colors and the die's orientation; one tip if
   the target face is on a side (push INTO it: u = −horizontal(face normal), itself a
   face normal), two same-way tips if it is underneath (walk toward the pad along the
   side face closest to the pad direction). Re-plan from readback after every tip.
2. **Tumble (contact)** — a slowly ramped horizontal push (1.8 → 2.9 N; tip threshold
   ≈ 1.81 N, static slide threshold ≈ 3.1 N) applied as force at the COM plus the
   couple of an application point 45 mm above it — exactly the wrench a fingertip
   pressed high on the face exerts. The ground edge becomes a transient hinge; the
   push is cut just past the 45° balance point (watch-face dot > 0.72) and gravity
   finishes the quarter-turn.
3. **Slide (contact)** — a velocity-regulated push (≤ 0.22 m/s, cap 4.5 N) at 20 mm
   effective height (couple shifts the application point 30 mm BELOW the COM — under
   the tip threshold even at the cap), with a stiction floor for the PD deadband, a
   tilt guard that cuts before an incipient tip passes the balance point, and a
   placement check that re-tumbles + re-slides if anything was lost.

Phases print `SIM_GEN_SCORE` at every boundary (non-decreasing, asserted), then ≥ 3.3
simulated seconds hands-off persistence before `SIM_GEN_SOLVE: SUCCESS`.
**Verified on the forge: seeds 0, 1, 2 all SUCCESS (~21 s wall each).**

## Rubric

Latched per active pad (credit never evaporates): 0.10 · some die EVER showed this
pad's color face-up (reorientation done) + 0.10 · ever face-up within 18 cm of the pad
(transported while reoriented) + 0.25 · ever face-up inside the pad window at floor
height (placed); sum 0.90; 1.0 iff `success()` — both pads simultaneously served by
settled dice. Null policy ≈ 0 (spawn up-faces exclude both pad colors). Demonstrated
solve margins: placements land ≤ 1.5 cm from pad center vs the 5.5 cm window.

## Embodiment argument (Franka, parallel-jaw)

- The die is 100 mm — wider than the ~80 mm Franka jaw opening, so the task is
  nonprehensile *by construction* (asserted); no grasp is ever needed.
- Every solve force is a fingertip push on a flat face: tip pushes 1.8–2.9 N applied
  ~95 mm up a face, slide pushes ≤ 6 N at ~20 mm — trivially within Franka's payload
  and force envelope; the closed gripper's fingertips are the natural end effector.
- Workspace: dice spawn near x ≈ 0.02, |y| ≤ 0.16; pads at x ≈ 0.30, |y| ≈ 0.17
  (± jitter). With the base at ≈ (−0.35, 0, 0) facing +x everything lies within a
  0.75 m reach at heights 0–0.10 m; push directions are horizontal and approach from
  open floor (pads are flush markings — nothing to collide with during the slide).
- Execution order: NONE required — either die may serve either pad, in either order
  (the solve's nearest-assignment order is one valid demonstration).

## Checks (smoke.py — rejection battery, 16/16 PASS on forge, frames.npz recorded)

1. settle + no-NaN, dice at rest on the floor; 2. score ~0 at reset AND neither die
spawns with either pad color up; 3–4. randomization readback over 6 seeded resets
(pad-color pair varies — 5 distinct pairs observed; pad jitter, die position and
spawn up-face vary; spawn honesty re-verified each reset); 5. null policy 2 s → score
~0; 6. naive transport (the seed family's move — carry to the goal WITHOUT
reorienting): die at pad center with spawn face up → NOT served, score ~0; 7. wrong
face (pad1's color on pad0) rejected; 8. off-center (right face, 9 cm out) rejected
by the per-axis window; 9. stacked die on a correctly served die: top die rejected by
the height window; 10. partial: ONE pad properly served — predicate ACCEPTS a correct
single placement, success still False, score < 0.9; 11. settle gate: the served die
moving at 0.43 m/s is not served; 12. latched credit survives teleporting the served
die away while `served()` correctly drops; 13. mechanism (slide): LOW push moves the
die 14 cm, up-face PRESERVED; 14. mechanism (tip): HIGH ramped push changes the
up-face by contact physics; 15. rejection audit — success() never True anywhere in
the battery; 16. final no-NaN.

Run (forge):
`python -u -m simgen_tasks.open_oven_i6.solve --headless [--seed N]`
`python -u -m simgen_tasks.open_oven_i6.smoke --headless`
