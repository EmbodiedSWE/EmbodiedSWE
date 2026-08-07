# pull_cube_i20 — Beam Scale

Load the raised pan of a balance beam with red cubes until their moment beats the
hidden counterweight's and the beam swings the other way (scene `beam_scale`,
env `simgen.beam_scale`).

## Seed provenance

Seed: `maniskill/pull_cube` — hook a cube with an L-shaped tool and drag it a few
centimeters across the plane into a painted goal region. One planar non-prehensile
act, judged by an xy-distance readout on the moved object itself.

The seed's end state (the cube resting at a floor goal spot) is **not expressible**
here — there is no floor goal region at all. The nearest naive plan from the seed's
family — move the cubes to some spot on the floor — is constructed as a settled
state in smoke check 6 and scores ~0; even parking a cube ON the beam scores nothing
unless it lands inside the correct pan (check 7).

## What the task is

A two-pan balance beam (procedural, 350 g) rests on a knife-edge pivot: a
45°-rotated square shaft seated in two captive stand pockets — a pivot made purely
of contact geometry, no USD joint. The beam can rock ±13.9° until a spine tip
touches the ground. One pan carries a dark counterweight block whose mass is
SAMPLED per episode (40 / 100 / 170 g → needs exactly 1 / 2 / 3 cubes); its side
(±x) is sampled too, the stand jitters in xy with ±30° yaw, the block jitters in
its pan, and three red 50 mm / 100 g cubes spawn on a jittered arc around the
stand. At reset the beam is level, then falls hands-off onto the counterweight
side; the target pan is the raised, empty one.

Goal: get the beam to rest tilted at least `tip_deg = 8°` toward the cube side —
with the counterweight still in its original pan, at least one cube inside the
opposite pan, and beam + all pan cargo settled. The only path to that state is
banking enough cube weight in the raised pan that the accumulated moment about the
knife edge beats the counterweight's.

## Strategic difference

- **vs. the seed**: the seed moves an object into a region and reads the moved
  object's position. Here the moved objects are not the goal at all: the predicate
  reads the TILT of a body the robot never touches. The reward-bearing rotation is
  INDIRECT — produced by where mass is banked, not by any contact with the rotating
  body — and the episode's stopping rule ("enough cubes") cannot be known in
  advance: the solver must add one cube, watch the beam, and stop when it tips.
- **vs. corpus tasks read**: no corpus task's goal is a lever / torque threshold on
  an untouched body. `open_oven_i6` (dice) reorients the pushed body itself (SO(3)
  on the manipulated object); the pick-place family (libero bowls/pan,
  `coke_task_i15`, `pick_and_lift_i16`, `approach_grasp_spoon_i12`) reads
  containment of the carried object; the articulation family (`close_microwave_i4/
  i5`, `close_grill_i8`, libero drawer/stove) rotates BUILT joints by direct grasp;
  `peg_insertion_side_i1/i2` insert, `pick_single_egad_i3/i4` thread/slide,
  `setup_checkers_i2` drop-stacks, `pour_water_i7` pours. **No corpus task converts
  placed mass into torque about a contact-geometry pivot**, and none has a
  per-episode unknown-count stopping rule.
- The mechanism is physically honest and asserted in `__post_init__`: the pans are
  slick (pair μ ≈ 0.15 < tan 13.9° = 0.247), so pan cargo deterministically slides
  to the downhill wall — the counterweight to the low pan's OUTER wall, cubes to
  the raised pan's INNER wall. With those deterministic lever arms, k cubes
  out-torque counterweight k with ≥10% margin while k−1 cubes fall short by ≥10%
  (k = 1, 2, 3) — the "one cube short" near-miss genuinely stays down (margins at
  the rest tilt: tip +72% / +38% / +22%, hold-down verified per k).

## Solution outline (solve.py — the legitimacy certificate)

Teleportation is TRANSPORT ONLY: each cube is placed at rest ~5 cm above the raised
pan's center (exactly what a pick-and-carry delivers) and RELEASED. Everything the
rubric reads happens through contact after release: the cube free-falls into the
pan, the slick floor slides it to the inner wall, its weight loads the beam, and
when the moment beats the counterweight's the beam rotates about the knife edge and
falls onto the opposite stop. The beam, stand, and counterweight are never touched
after reset; no wrench is ever applied to anything.

The demonstrated policy is the honest one from `describe()`: add ONE cube, hands
off, watch the beam; if it stays down, add the next; stop when it swings over
(1, 2 or 3 cubes depending on the sampled counterweight). `SIM_GEN_SCORE` prints at
every settled phase boundary (non-decreasing, asserted), then ≥ 3.3 simulated
seconds hands-off persistence before `SIM_GEN_SOLVE: SUCCESS`.
**Verified on the forge: seeds 0, 1, 2 all SUCCESS (~17–22 s wall each), covering
counterweights needing 3 and 1 cubes on both sides.**

## Rubric

`score()` is stateless: 0.15 · cubes-in-target-pan (cap 3) + 0.20 · tilt-progress
(gated on the counterweight staying in its pan), capped at 0.65; 1.0 iff
`success()`. Null policy ≈ 0: the beam rests fully tilted the WRONG way
(s·u = −0.242), so tilt-progress is 0 and no cube is in the target pan.
`success()` = counterweight in its original pan + ≥1 cube in the opposite pan +
beam at rest tilted ≥ 8° toward the cube side (rest stop 13.9°) + beam and all pan
cargo settled.

## Embodiment argument (Franka, parallel-jaw)

- The cubes are plain 50 mm / 100 g rigid cubes — comfortably inside the ~80 mm
  jaw (asserted) and trivial payload. The task is a repeated pick-and-drop.
- Drop tolerance is generous: the pan cavity is 10 × 17 cm for a 5 cm cube
  (≥ 4.5 cm slack, asserted), the release is a free drop from above the 35 mm
  walls, and the slick pan self-locates the cube at the correct wall — placement
  precision is NOT required.
- Workspace: the stand sits near the origin (± 3 cm, ± 30° yaw); cubes spawn on an
  arc of radius 0.34–0.40 m at world angles 210–330°, i.e. the half-plane y < 0.
  With the base at ≈ (0, −0.60, 0) facing +y, every cube pickup and both pan drops
  (pan centers at |x| = 0.20, drop height ≤ 0.17 m) lie within a 0.75 m reach.
- Execution order: cubes may be taken in ANY order; the only sequencing constraint
  is the task's own stopping rule (add cubes one at a time until the beam tips —
  the demonstrated policy). No other ordering is required.

## Checks (smoke.py — rejection battery, 15/15 PASS on forge, frames.npz recorded)

1. settle + no-NaN: hands-off from reset the beam falls onto the counterweight
side and RESTS fully tilted (s·u < −0.15, settled); 2. block in its pan, cubes flat
on the floor, score ≤ 0.02, no success; 3. randomization readback over seeds 21–32:
counterweight mass AND side vary (3 masses, 5 mass×side combos observed); 4. stand xy +
yaw and cube spawn positions vary; 5. null policy 300 steps → score ~0; 6. seed
strategy (pull cubes to a floor spot — the seed family's move): all three cubes
settled at a floor cluster → score ≤ 0.02, beam undisturbed; 7. cube parked on the
beam's SPINE (on the beam, outside both pan windows) → no credit, beam stays down;
8. wrong pan: a cube dropped into the COUNTERWEIGHT's pan → no credit, beam stays
down; 9. near-miss underload: on a seed needing k cubes, k−1 cubes are banked
properly → the beam genuinely STAYS down, score ≤ 0.15(k−1)+0.02, no success;
10. mechanism (slick pans): cargo self-locates at the deterministic walls the
torque design assumes (block at outer wall, cube at inner wall); 11. remove-CW
exploit: teleporting the block out tips the beam but success is False and the
tilt term earns nothing (gated on the block staying); 12. physics restores: a beam
POSED at the cube-side stop with the block aboard and no cubes swings BACK to the
counterweight side; 13. settle gate: a statically-correct success layout judged
with the beam spinning (3 rad/s) is not success; 14. rejection audit — success()
never True anywhere in the battery (the "accepts" evidence is solve.py itself);
15. final no-NaN.

Run (forge):
`python -u -m simgen_tasks.pull_cube_i20.solve --headless [--seed N]`
`python -u -m simgen_tasks.pull_cube_i20.smoke --headless`
