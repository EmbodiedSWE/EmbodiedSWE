# feed_line_cull — "Cull the reject, release the line"

## Seed provenance

Seed: `rlbench/place_shape_in_shape_sorter` — pick up a shape and insert it
through the matching cutout in the top of a shape-sorter box.

## Strategic difference

The seed's strategy is **selection by insertion**: identify the matching
piece, carry it to the box, and thread it through a hole. This task inverts
every one of those beats:

1. **Selection by REMOVAL, not insertion.** A queue of cubes sits on an
   inclined feed chute; exactly one is a red reject. The reject must end up
   *off the machine, on the ground, clear of the line* — putting it in the
   collection bin (the seed's "put the piece in the box" move) is an
   explicit failure mode (smoke check 4).
2. **Conveyance by MECHANISM + GRAVITY, not carrying.** The good (blue)
   cubes are never carried anywhere. They are held on the 16° chute by a
   knobbed release pin threaded through square holes in both channel walls.
   Extracting the pin axially (a real lateral force servo, ~0.13 m of
   travel) releases the queue; low-friction materials guarantee the cubes
   slide down, launch off the lip, and land in the walled catch bin
   entirely hands-off.
3. **No hole matching.** The pin passes through square wall holes but is a
   one-way mechanism (the knob cannot enter its hole — smoke check 8 proves
   the −y direction is blocked), not a shape-matching puzzle: there is one
   pin and one way to pull it.

Difference from other tasks_v7 packages examined:
- `put_money_in_safe_i174` (rotary airlock): its core is rotating a drum to
  ferry an object *into* an enclosure; here nothing is ferried — a captive
  linear pin is extracted and gravity does the delivery.
- `libero_pick_chocolate_pudding_i46` (sieve size-sorting): sorting by
  geometry through apertures; here the "sorting" is a color cull done by
  transport-removal, and the aperture (pin hole) is part of a latch, not a
  classifier.
- `lift_numbered_block_i153` / pick-and-place family: those end with an
  object held or placed by the agent; here the scored deliverable (blues in
  bin) is produced with zero agent contact with the blues.

## The scene

Rig (kinematic, one compound): a 16°-pitch channel chute (walls 28 mm high,
75 mm inside width) descending to a launch lip at z = 0.22, feeding a walled
catch bin (interior 290×150 mm, back wall to z = 0.30, side walls to 0.20,
and a high-friction landing mat on its floor so launched cubes stop where
they strike instead of piling into a tower at the back wall) at the rig
origin.
At station s = 0.10 up the slope, two amber brackets outboard of the walls
carry 20 mm square through-holes centered 20 mm above the chute floor. A
steel release pin (r = 8 mm shaft, 128 mm long, density 4000, with a 16 mm
knob on the +y end only) lies through both holes, crossing the channel and
blocking the queue. Behind it: 2–4 blue cubes (45 mm, randomized count,
axial gaps and lateral jitter) and one red reject at a random slot in the
queue. A top-end stop closes the uphill end of the channel.

Randomization: rig yaw ±15° and xy jitter ±5 cm; blue count 2–4; red slot
uniform among queue positions; per-cube axial gap and lateral jitter; pin
axial seat jitter.

## Success

All of, settled and finite:
- every present blue cube banked (inside the bin interior band, latched
  with 3-step persistence; the band's z ceiling 0.185 covers piles while
  the x/y bands exclude every wall-top rest),
- red reject clear: on the ground (z < 0.075) AND outside the machine
  footprint (|y| > 0.13 or x > 0.66 or x < −0.20), rig-local,
- pin extracted (max rig-local pin y latched ≥ 0.118 — fully out of both
  holes).

Score: 0.20·red_clear + 0.15·pin_frac + 0.45·banked_frac, capped 0.80;
1.0 on success. Null policy ≈ 0.

## Solve phases (all interaction physical; teleports = transport only)

- **P0** settle + audits (masses, pin seated |y| < 6 mm, queue on chute).
  Score 0.
- **P1 cull**: teleport the red reject to a rig-local hover over open
  ground (0.40, 0.32, 0.10) — asserted NOT red_clear at the write — and
  let it fall/settle. Score ≥ 0.19.
- **P2 extract**: velocity servo on the pin via external wrenches in the
  body frame (axial pull +y, capped 8 N, gravity support during the pull,
  small alignment torque), until rig-local pin y ≥ 0.175 (the −y tip then
  clears the bin side wall below); then brake and lower (support 0.90)
  until the pin is near the ground; release. All force, no teleport.
  Score ≥ 0.33.
- **P3 delivery (hands-off)**: nothing touches anything; the queue slides,
  launches, and banks. Loop until all present blues latched-banked and
  settled. Score ≥ 0.999.
- **P4 persistence**: ≥ 3 sim-seconds untouched, success still true, then
  `SIM_GEN_SOLVE: SUCCESS`.

Execution order: the intended order is cull-then-pull. The task does not
enforce a strict order via the rubric — pulling the pin first is
*recoverable* but strictly worse: the red reject rides the line into the
open-top bin and must then be lifted back out over a 200–300 mm wall before
red_clear can hold, so the cull-first order is the rational plan. No
ordering latch is needed because no wrong order destroys feasibility.

## Franka embodiment (per object)

Plausible base pose: on the +y (knob) side of the rig, base at rig-local
(0.25, 0.55, 0) facing −y; everything scored is within a 0.75 m disc of
that point at z between 0 and 0.35.

- **Red reject cube (the only carried object)**: 45 mm cube — inside the
  Franka gripper's 80 mm span. It sits in a 75 mm-wide channel with 28 mm
  walls; 17 mm of cube stands proud of the walls and there is a 15 mm
  finger slot on each side of the cube inside the channel, so a top pinch
  grasp across y is collision-free. Carrying it 0.3–0.5 m sideways and
  releasing at ground level is a nominal pick-and-place.
- **Release pin**: the 16 mm knob at z ≈ 0.27 is a natural pinch target
  (knob diameter 32 mm < 80 mm span, protruding 25 mm axially with free air
  around it). Extraction is a straight ~0.13 m lateral pull at constant
  height toward the robot base — well inside the workspace, forces ~2–8 N,
  far below the arm's payload. The solve's capped-force axial servo is
  exactly what an impedance-controlled arm pull looks like.
- **Blue cubes / bin**: never touched by the agent; no embodiment demand.

## Checks (smoke.py, 11)

1. Settle/no-NaN + queue verified pressing the pin gate (non-vacuous).
2. Randomization readback (seeds 101 vs 202 differ: yaw, xy, pin seat /
   red slot / blue count).
3. Null policy 240 steps: success never, score ≈ 0.
4. SEED-strategy rejection: everything (including red) placed in the bin,
   pin out → NOT success, score ≤ 0.80.
5. Blue left out on the ground, rest banked, red clear → NOT success.
6. Red under the chute (on ground but inside footprint) → red_clear False.
7. Pin-gate reality: 0.9 N/cube downhill shove for 240 steps — pin holds,
   nothing banks (front cube verified pressing the pin).
8. Knob one-way: −4 N axial push — pin advances only a few mm and stops
   (knob cannot pass its hole).
9. Cull skipped: pull the pin with the red still in the queue → blues bank,
   red lands IN the bin, NOT success (seed strategy fails end-to-end).
10. Top-end stop: uphill shove — queue retreats but cannot exit the top;
    score stays ≈ 0.
11. frames.npz written with > 10 frames.
