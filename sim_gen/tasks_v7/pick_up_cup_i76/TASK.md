# pick_up_cup_i76 — Bend Gallery

Pick up the cup by first MAKING it liftable: thread a long-handled dipper out of a
roofed L-gallery — piano-movers style — then stand the cup on the pad (scene
`bend_gallery`, env `simgen.bend_gallery`).

## Seed provenance

Seed: `rlbench/pick_up_cup` — a free cup stands on a table; the robot grasps it and
lifts it straight up, judged by cup height. One grasp, one vertical lift, in open
space.

The seed's strategy is here **physically void, and the smoke battery proves it**: the
cup is the head of a rigid 400 mm dipper whose handle starts deep inside a roofed
dead-end tunnel, and the dipper is longer than the diagonal of the only open-top
pocket — smoke check 4 applies the seed's exact move (a sustained 1.7×-weight
vertical pull) and the rod rises 57 mm, wedges against the roof, and falls back with
score 0.

## What the task is

A KINEMATIC grey-blue GALLERY (walls 20 mm thick, 120 mm tall, roof underside at
105 mm) stands on the floor: a dead-end tunnel (240 mm, ROOFED) and an exit tunnel
(150 mm, ROOFED), both 180 mm wide, meeting at right angles through an open-top
corner PLAZA (260 mm square). Inside lies a DIPPER — a free rigid compound body,
400 mm overall: a 310 × 50 × 50 mm handle bar ending in an open-top red CUP (octagon
of eight 8 mm wall plates, ~96 mm across, 80 mm tall), 350 g — with its handle deep
in the dead-end tunnel and only the cup head exposed in the plaza. Gallery pose
(xy ± 25 mm, yaw ± 28°), handle insertion depth (70 mm range), dipper lateral/yaw
jitter and the pose of a flat grey collision-free PAD (160 mm square) in the open
YARD beyond the exit tunnel all vary per episode.

Goal: get the ENTIRE dipper out of the gallery (every point of it past the exit
line) and stand the cup UPRIGHT on the pad (cup centre within 50 mm, body +z within
10° of up), everything settled. The extraction is geometrically forced through the
corner: the rod is longer than the plaza diagonal (400 vs 368 mm) so no pose inside
the gallery is fully unroofed (it can never be simply lifted out), a straight push
out of the dead-end tunnel jams on the far plaza wall (smoke 5), and the tunnel
dead-ends behind (smoke 6). The only way out is confined planar maneuvering: slide
out of the dead-end tunnel, rotate ~90° through the corner (validated by a
config-space BFS over (x, y, θ) with the footprint inflated 16 mm — the pass exists
with real clearance), and thread head-first down the exit tunnel into the yard.

## Strategic difference

- **vs. the seed**: the seed's cup is graspable and liftable from step one; the goal
  is height. Here the object CANNOT be lifted out at all — the roof and the
  rod-longer-than-diagonal geometry make the seed's one-motion strategy impossible
  (proven physically in smoke 4), and the goal is a placement that is only reachable
  after a multi-stage confined extraction. The pick-up is EARNED by solving a
  piano-movers problem first.
- **vs. corpus tasks read**: the transport family (`pick_and_lift_i16`,
  `coke_task_i15`, `approach_grasp_spoon_i12`, libero bowls/pan/milk) moves free
  objects through open space; `pull_cube_i20` drags a cube in the open;
  `close_box_i26` slides a trap on open floor; the articulation family
  (`close_microwave_i4/i5`, `close_grill_i8`, libero drawer/stove) actuates built
  joints. **No corpus task has confined-passage kinematic threading**: a free body
  whose path to the goal is topologically constrained by a fixed narrow environment
  (dead-end + corner + roofed tunnels), where orientation and translation must be
  coordinated (the tail can only clear the first tunnel MID-ROTATION — asserted in
  `__post_init__` and observed in the solve trace) and the naive straight-line moves
  provably jam.
- The constraint is physically honest and proven, not asserted: lift blocked
  (smoke 4), straight push jams at the far wall with `cleared` never latching
  (smoke 5), dead end real (smoke 6), and the corner pass itself is demonstrated by
  the solve on three seeds.

## Solution outline (solve.py — the legitimacy certificate)

**NO teleports at all** — the entire trajectory is contact dynamics: a
velocity-regulated horizontal force (≤ 3.5 N) and z-torque (≤ 0.3 Nm) at the CoM,
world-frame direction with the pod force-frame quirk handled by `encode_force`
(mode fixed from measured behavior, with an in-flight divergence watch).

1. **P0 settle + readback** — layout printed from readback; score 0. A gentle
   stiction sanity probe (breaks at ~8 mm displacement).
2. **Gauntlet** — 14 waypoints (x, y, θ) in the gallery frame, distilled from the
   config-space BFS path: slide out of the dead-end tunnel while feeding into the
   corner rotation (~0 → −90°), thread the exit tunnel, emerge into the yard, turn
   back to θ ≈ 0 in the open, and dock the cup on the pad (stop ≤ 20 mm from
   centre). Servo: speed ~0.10 m/s, stiction floors, tilt guard, stall watch with
   one backoff-and-retry per waypoint.
3. **P4 place** — forces cleared, 1.5 s settle, assert `success`, score 1.0.
4. **P5 persistence** — ≥ 3.3 simulated seconds hands-off, success still holds,
   then `SIM_GEN_SOLVE: SUCCESS`.

Phases print `SIM_GEN_SCORE` at every latch/boundary (non-decreasing, asserted).
**Verified on the forge: seeds 0, 1, 2 all SUCCESS (58 s / 71 s / 73 s wall), score
trace 0.00 → 0.20 → 0.45 → 0.70 → 1.00 → 1.00 each.** Note the latch order on the
real path: `entered` (cup into the exit tunnel, θ ≈ −48°) fires just BEFORE
`cleared` (tail out of the dead-end tunnel, θ ≈ −61°) — both are mid-corner events
and the score is monotone either way; the phase prints reflect the observed order.

## Rubric

Latched partial credit (never evaporates): 0.25 · tail EVER cleared the dead-end
tunnel (geometrically impossible for an axis-aligned rod — asserted; it requires the
corner rotation) + 0.20 · cup EVER inside the exit tunnel + 0.25 · cup EVER emerged
into the yard; cap 0.75; exactly 1.0 iff `success()` — extracted AND cup-on-pad AND
upright AND settled AND finite, all judged LIVE. Null policy ≈ 0 (an untouched
dipper latches nothing). Demonstrated margins: the solve docks the cup ≤ 20 mm from
the pad centre vs the 50 mm window, upright to within 1° vs 10°.

## Embodiment argument (Franka, parallel-jaw)

- **Grasp**: the handle is a 50 mm square bar — a natural parallel-jaw pinch (jaws
  open ~80 mm). At EVERY insertion depth at least ~80 mm of handle plus the whole
  cup head is exposed in the open-top plaza (asserted in `__post_init__`), so a
  top-down pinch of the handle is always available at start.
- **Maneuver**: the extraction is planar at floor height: drag/slide the pinched
  dipper (0.35 kg, sliding friction ~1.2 N, demonstrated forces ≤ 3.5 N — trivially
  within Franka's envelope), regrasping as needed. Along the entire demonstrated
  path some portion of the dipper is always in unroofed space (start: cup + handle
  in the plaza; mid-turn: everything in the plaza; threading: tail in the plaza;
  emerging: cup in the yard), so a top-down grasp point always exists.
- **Workspace**: gallery frame origin at (0.36, 0.04) ± jitter; everything lives in
  x 0.08–0.65, y −0.45–0.35, z ≤ 0.13 m. With the base at (0, 0, 0) facing +x all
  grasp points lie at 0.25–0.70 m reach at near-floor heights.
- Execution order: the extraction STAGES are geometrically forced (out of the dead
  end → through the corner → down the exit tunnel → onto the pad), but judging is
  **END-STATE-ONLY** (stated in `describe()`): any maneuver achieving the final
  state is legal — e.g. pitching the rod toward vertical through the open plaza top
  during the corner is permitted, not punished.

## Checks (smoke.py — rejection battery, 13/13 PASS on forge, frames.npz recorded)

1. settle + no-NaN, dipper flat with tail deep in the tunnel, score 0; 2.
randomization readback over 3 seeds (gallery xy 20 mm + yaw 27°, insertion depth
35 mm, pad local xy 65 mm); 3. null policy 240 steps → score 0; 4. SEED STRATEGY:
1.7×-weight vertical pull → rod rises 57 mm, wedges under the roof (tail z max 82 mm
< 105 mm), never extracted, falls back, score 0; 5. MECHANISM must-rotate: gentle
regulated push straight out slides 74 mm then jams on the far plaza wall (head at
260 mm, tail −140 mm), `cleared` never latches; 6. MECHANISM dead end: push inward
slides 25 mm then jams on the cap (tail −240 mm); 7. corridor near-miss: cup upright
and centred ON the pad but the tail still inside the gallery (tail y +3 mm vs exit
line −160 mm) → extracted() False, no success (the full-extraction clause is
load-bearing); 8. latched credit: rod teleported back inside — live predicates drop,
cleared+emerged credit survives (0.50); 9. pad near-miss (85 mm vs 50 mm window) →
rejected; 10. flipped: extracted, on-pad but upside-down (up_z −0.995) → rejected;
11. settle gate: the exact success pose sliding at 0.45 m/s is refused (dismantled
before it can settle); 12. rejection audit — success() never True at ANY step of the
battery; 13. frames.npz saved (94 frames, 960×600).

Run (forge):
`python -u -m simgen_tasks.pick_up_cup_i76.solve --headless [--seed N]`
`python -u -m simgen_tasks.pick_up_cup_i76.smoke --headless`
