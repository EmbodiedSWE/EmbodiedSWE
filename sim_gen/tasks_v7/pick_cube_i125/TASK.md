# pick_cube_i125 — Roll the Die Blue-Side-Up (scene `die_roll`)

A 90 mm cubic DIE — too wide for a parallel-jaw gripper to grasp — rests on the floor
with its six faces painted six distinct colors; the BLUE face never starts up. Tip it
over its ground edges (discrete quarter-rolls: push high on a face, gravity slams it
onto the next face) until BLUE points up, then slide it (push low: translation that
provably cannot reorient) and leave it AT REST centered on a GREEN target mat. A
same-size dark-GRAY decoy mat jitters and randomly swaps sides with the target every
episode and pays nothing.

## Provenance

- **Seed:** `maniskill/pick_cube`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/pick_cube.py`) — "pick up the
  red cube": ONE grasp of a 4 cm cube plus one guided free-space lift, judged by a
  0.1 m position shift along z. Orientation is irrelevant, the object is chosen to be
  trivially graspable, and the whole plan is a single pick.
- **Files:** `scene.py` (cfg + scene + rubric + the shared `ExternalWrenchDriver`,
  registered as scene `die_roll`, env `simgen.die_roll`, robot `"null"`), `solve.py`
  (wrench solution — zero teleports), `smoke.py` (rejection battery), all procedural
  geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed's cube exists to be GRASPED and its goal is a pure
  POSITION change (lift 0.1 m; orientation never judged). Here the cube is sized
  90 mm — beyond the ~80 mm jaw span — precisely so it CANNOT be grasped, and the
  goal is an ORIENTATION: which painted face points up. The seed's entire skill
  (grasp + carry) is reproduced verbatim in smoke check 7 (the die teleport-carried
  to the mat center, orientation preserved) and scores ≤ 0.16 with no success.
  Reorientation is reachable ONLY through planned sequences of edge-pivots — a
  discrete rolling-cube puzzle with nontrivial group structure (each quarter-roll
  permutes the face cycle AND displaces the die one edge length), plus a second,
  physically distinct contact skill (sliding) for the position clause. Nothing in
  the seed plans, nothing in the seed reorients, nothing in the seed distinguishes
  two contact strategies on the same object by contact height.
- **vs `close_grill_i8` (read this session):** that task CREATES a multi-body
  structure by stacking free bodies. Here there is one manipulated body, no
  stacking, no structure — the challenge is the orientation group of a single cube
  and the force/torque thresholds that separate tipping from sliding.
- **vs `base_i88` (read this session):** that is a counterweight/see-saw ballast
  task — continuous statics on a mechanism. Here there is no mechanism, no joint,
  no counterweight anywhere; the dynamics are discrete (quarter-roll or not).
- **vs `roll_ball_i81` (read this session):** that rolls a SPHERE to a position —
  rolling is continuous, orientation meaningless. Here "rolling" is a sequence of
  discrete edge-pivots chosen to reach an orientation target, and pure translation
  (the sphere task's whole content) is the thing that provably CANNOT solve this
  task (smoke check 5).
- **vs `pen_holder` (exemplar read this session):** no container, no insertion, no
  filling; nothing is placed INTO anything.
- **Execution order (declared):** the rubric enforces no fixed roll/slide order —
  interleavings are legal. What is physically REQUIRED is at least one quarter-roll
  (blue never starts up: reset samples only the five non-blue faces) and that
  sliding can never substitute for rolling (smoke 5: a slide-window push drags the
  die ≥ 0.10 m without ever changing the up face). Position and orientation clauses
  must hold SIMULTANEOUSLY at the settled end state.

## Randomization (per episode, verified by readback in smoke)

Die: start face sampled from the FIVE non-blue faces, free yaw (±180°), xy jitter
(±4 cm x, ±12 cm y). Mats: independent xy jitter (±4 cm) and a random SIDE SWAP
(green/gray exchange y sides), so a memorized push direction fails and the target
must be identified by color. Discrete draws use `torch.rand` comparisons (the first
`torch.randint` after `manual_seed` is near-degenerate across seeds).

## Rubric

`success()` iff, live: BLUE axis within 12° of world-up, die center within 55 mm of
the GREEN mat center, resting on the ground (center height = e/2 ± 12 mm), settled
(|v| < 0.05 m/s, |ω| < 0.40 rad/s), finite.

`score()` (latched in `post_step`; the up-face latches additionally require the die
calm, |ω| < 0.5): `0.20 ×` rolled (the settled up face ever differs from the reset
up face) `+ 0.25 ×` blue (the die ever settles blue-up anywhere) `+ 0.15 ×` near
(die center ever within 0.15 m of the target-mat center — the spawn separation
≥ 0.21 m makes this unreachable by the null policy, asserted in `__post_init__`).
Capped at 0.60 (float32 0.60000002); exactly 1.0 iff `success()`. Doing nothing
scores ~0.

Cfg `__post_init__` asserts the honesty geometry: die spawn outside the approach
radius, decoy separation beyond the success tolerance, success tolerance inside the
mat, and μ < 1 (the slide window `μmg = 3.09 N < mg = 4.41 N` is real).

## Wrench solution (`solve.py`) — zero teleports, all contact dynamics

No teleports at all. A closed-loop planner re-reads the die pose after every
primitive: blue sideways → ONE roll about `blue × ẑ` lifts blue up; blue DOWN → two
rolls about the SAME axis `ẑ × dir(goal)` (each advances one edge length toward the
mat and the second brings blue up); blue up → SLIDE to the mat center. ROLL = bang-
bang torque about the horizontal roll axis (τ₀ = 1.25·mge/2 ≈ 0.25 N·m, FD-measured
rate capped at 1.3 rad/s, released at ~50° past which gravity completes; landing
retains ~ω/4 — 5 mJ ≪ the 82 mJ second-roll barrier, so exactly one quarter-roll per
command). SLIDE = bang-bang 3.6 N CoM force inside the slide-without-tip window,
velocity capped 0.12 m/s and tapered near the goal. The wrench frame is never
assumed: an `ExternalWrenchDriver` is calibrated once by a push probe (−x, away
from the mats) and every primitive re-verifies the measured response direction,
cycling the encode mode on disagreement. `SIM_GEN_SCORE` at every phase boundary is
non-decreasing (P0 ~0 → P1 ≥ 0.20 first roll → P2 ≥ 0.45 blue up → P3 = 1.0 at rest
on the mat → P4 = 1.0), ≥ 3.3 simulated seconds hands-off persistence, then
`SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge: seeds 0 and 1, both SUCCESS,
provably distinct layouts by stdout readback.**

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (0, 0), facing +x: die spawn 0.24–0.32 m, mats 0.52–0.68 m — inside
Franka's ~0.85 m dexterous shell, tallest judged point 90 mm. The die is
deliberately UNgraspable (90 mm > 80 mm jaw stroke), so both skills are non-
prehensile pushes with the closed fist/fingertips — exactly what the wrench solution
emulates: **tip** = push horizontally near the TOP edge of a face (lever arm e
about the ground pivot: ~2.5 N suffices, comfortably below the 3.1 N sliding
breakaway, so the die pivots instead of skidding); **slide** = push at/below CoM
height (tipping from there needs > mg = 4.4 N while sliding breaks away at 3.1 N —
push at ~3.6 N and it translates without tipping). Both are sub-5 N horizontal
pushes at ≤ 9 cm height with clear vertical approaches; the 55 mm position and 12°
orientation tolerances are far beyond arm repeatability. Mats are paint-thin
(no collider) — nothing to snag a fingertip during the final slide.

## Checks (`smoke.py` — rejection battery, 13 named checks, ALL PASS on the forge)

1. settle/no-NaN: seeded reset settles finite, die flat on a non-blue face, outside
   the approach radius; score ~0, no success.
2. randomization readback: die xy, die orientation, green-mat xy all differ across
   seeded resets.
3. coverage: ≥ 3 distinct start faces (never blue) and the green mat occupies BOTH
   y sides over 10 resets.
4. null policy: 240 idle steps → die stays put, score ~0, no success.
5. SLIDE CANNOT ORIENT (identity claim is physics): a slide-window CoM push drags
   the die ≥ 0.10 m — the up face never changes, blue never rises.
6. torque tip gate: 0.6× the edge-pivot breakaway torque held 300 steps does NOT
   tip; 1.8× tips onto a new settled face (threshold real, sub-threshold half
   non-vacuous).
7. SEED-analog carry: die teleport-carried orientation-preserved to the mat center
   → blue not up, no success, score ≤ 0.16.
8. wrong face: PURPLE-up die dead-center on the green mat → no success, ≤ 0.36.
9. near-miss position: blue-up die settled 75 mm off the mat center (> 55 mm) →
   latches pin at the 0.60 cap, success refused.
10. latch persistence: from 9, the die carried far away and settled → the latched
    0.60 survives, still no success.
11. decoy: blue-up die centered on the GRAY mat → no approach credit, no success,
    ≤ 0.46.
12. settle gate: blue-up dead-center but MOVING (0.45 m/s, 3 rad/s) is refused at
    the judged instants (die removed before it can settle into success).
13. frames.npz recorded and saved in CWD.
