# libero_kitchen_scene9_turn_on_the_stove_i36 — Gap Ferry

Turn the crossing ON by BUILDING it: seat a loose channel plank into the rebate
ledges of two chasm-facing docks, then roll an ungraspable ball across the bridge
into the far dock (scene `gap_ferry`, env `simgen.gap_ferry`).

## Seed provenance

Seed: `libero_90/libero_kitchen_scene9_turn_on_the_stove` — an articulated stove USD
stands in the scene; the robot rotates its knob past a joint threshold
(`joint_pos > 0.5`). One actuation of a mechanism the SCENE built, judged by a joint
readout.

The seed's end state (a knob joint past a threshold) is **not expressible** in this
scene — nothing is articulated and there is no joint to read; documented N/A. The
seed family's nearest naive transfer — actuate straight toward the goal, build
nothing (push the ball at the target dock with no bridge) — is executed in smoke
check 5: the ball falls into the chasm, unrecoverable, and scores ~0.

## What the task is

Two KINEMATIC walled DOCKS (deck tops 120 mm up, 60 mm walls on three sides, open
toward each other) face each other across a fixed 140 mm CHASM; the whole assembly
jitters in xy and heading (±30 mm, ±25°) per episode. Each dock's chasm edge carries
a recessed rebate LEDGE (55 mm deep, 15 mm below the deck). The START dock (grey
deck) holds a 90 mm, 450 g ORANGE BALL — **wider than the 80 mm Franka jaw opens**
(asserted), so it cannot be carried, only rolled. The TARGET dock (blue deck) is
empty. A free 210 × 130 × 15 mm CHANNEL PLANK (two raised guide rails, 150 g) parks
on a low two-sleeper rack that spawns on a random side of the docks; the plank's
pose on the rack jitters (±30 mm lengthwise, ±20° yaw, end-for-end flip coin).

Goal: the ball resting ON the blue deck — inside its walls, past the ledge, at
deck+radius height (windows in the far dock's body frame + a height window),
settled, finite. The only route is to BRIDGE first: seated in the rebates the
plank's top is FLUSH with both decks (plank thickness == rebate drop, asserted) and
the rails form a guide channel; the backstops box the seating to ±20 mm of physical
slack. A ball pushed toward the gap without a seated bridge falls in and is
unrecoverable (smoke 5). Bridge-before-crossing is imposed by physics, not decree.

## Strategic difference

- **vs. the seed**: the seed actuates a BUILT mechanism (rotate a knob about a
  scene-provided hinge) and reads a joint. Here there is no joint and no mechanism
  until the solver MAKES one — the "stove that must be turned on" is the crossing
  itself, assembled from a free plank and two rebate ledges — and the payoff act is
  then a second, distinct interaction (nonprehensile rolling of a deliberately
  ungraspable object across the built infrastructure). Construction precedes and
  enables actuation; the seed has only the actuation.
- **vs. corpus tasks read**: the articulation family (`close_microwave_i4/i5`,
  `close_grill_i8`, libero drawer/stove) manipulates built joints. The pick-place
  family (libero bowls/pan, `coke_task_i15`, `pick_and_lift_i16`) carries grasped
  objects freely. `close_box_i26` (trap crate) is the nearest neighbour in spirit —
  a solver-constructed state — but its mechanism is containment-coupled floor
  transport (cage a block, slide the cage; lifting forbidden). Here the mechanism is
  **infrastructure construction**: the manipulated object (plank) is not the scored
  object (ball); the plank must be placed to millimetre-class flushness dictated by
  geometry, and the scored object then traverses it under nonprehensile control. No
  corpus task builds a load-bearing structure that a second, ungraspable object must
  then cross; none has an irreversible hazard (the chasm) that makes the naive
  strategy terminally scoreless.
- The mechanism is physically honest and proven, not asserted: the flush seat, the
  seating slack, the channel-clears-ball margin, the jaw bound, wall retention and
  the ledge/plank-end exclusion windows are all enforced in `__post_init__`; the
  chasm swallows an unbridged crossing (smoke 5), gravity really seats the plank
  (smoke 6, solve P1), the seated bridge really carries the rolling ball over the
  void (smoke 7), and near-miss states — diving-board plank, rails-down plank,
  ball on the plank's far end, ball on the bare ledge — are each rejected by a
  specific live clause (smoke 8–11).

## Solution outline (solve.py — the legitimacy certificate)

Teleports are TRANSPORT ONLY; every rubric-relevant fact is produced by contact:

1. **P0 settle + readback** — layout printed from readback; plank racked, ball in
   the start dock, score ~0.
2. **P1 bridge (gravity)** — one root-state write carries the plank to a free-space
   hover 22 mm ABOVE seat height over the chasm, aligned, rails up, zero velocity
   (satisfying nothing: `bridged()` demands seat height ±6 mm and calm). The plank
   FALLS into the rebates; both-ends support, seat height, level and calm are all
   made by contact. Asserts `bridged`, score 0.30.
3. **P2 ferry (applied force + rolling)** — a transport write moves the ball a few
   centimetres to the channel mouth (still on the start deck — score unchanged,
   asserted), then a velocity-regulated horizontal force at the ball's CoM
   (~0.22 m/s, clamp 3.0 N) rolls it down the channel and across the bridge, with a
   lateral steering term. The pod force-frame quirk is handled by `encode_force`
   with the mode PROBED from measured progress and — critical for a ROLLING body —
   the encoding recomputed from a fresh orientation readback every step. The push
   releases once the ball is past the far ledge; arrival and rest are gravity +
   walls + damping. Asserts `success`, score 1.0.
4. **P3 persistence** — ≥ 3.3 simulated seconds hands-off, success still holds,
   then `SIM_GEN_SOLVE: SUCCESS`.

Phases print `SIM_GEN_SCORE` at every boundary (non-decreasing, asserted).
**Verified on the forge: seeds 0, 1, 2 all SUCCESS (20 s / 25 s / 24 s wall), score
trace 0.00 → 0.30 → 1.00 → 1.00 each; the force-frame probe detected and locked
mode 1 on every run.**

## Rubric

Latched partial credit (never evaporates): 0.30 · plank EVER seated across the
chasm (both ends on the ledges, seat height, level, calm) + 0.15 · ball ever ON the
plank over the void + 0.25 · ball ever inside the far-deck windows; cap 0.70;
exactly 1.0 iff `success()` — ball in the far-deck windows AND ball+plank settled
AND finite, all judged LIVE. Null policy ≈ 0 (a racked plank never bridges; a ball
in the start dock is in no window). Demonstrated margins: the seated plank lands
within ~1 mm of seat height vs the 6 mm window; the delivered ball rests near the
back of the 100 mm x window.

## Embodiment argument (Franka, parallel-jaw)

- **Bridge**: the plank is 15 mm thick with 10 mm rail lips — a natural parallel-jaw
  pinch anywhere along an edge (jaw opens 80 mm), and the rack holds it 25 mm off
  the floor for finger clearance. It weighs 150 g (payload 3 kg). Plan: pinch an
  edge, lift, yaw to the crossing heading, hover over the chasm and lower/release —
  gravity performs the same seating the solve demonstrates, and the rebate
  backstops forgive ±20 mm of lengthwise error (the whole seating slack).
- **Ferry**: the ball (90 mm > 80 mm jaw, asserted) cannot be grasped; the natural
  strategy is exactly the demonstrated one — fingertip/knuckle rolling pushes of
  ≤ 3 N at ~0.2 m/s, with the guide rails absorbing lateral error. The dock walls
  (60 mm > ball centre height) retain a slightly brisk arrival.
- Workspace: assembly centre (0.40, 0) ± 30 mm with ±25° heading; docks extend
  ~0.19–0.62 m in x, |y| ≤ 0.31, rack at |y| ≈ 0.33; all contact heights
  0.02–0.18 m. With the base at (0, 0, 0) facing +x every required contact lies at
  0.19–0.70 m reach.
- Execution order: bridge-before-crossing is **imposed by physics** (an unbridged
  push loses the ball forever); there is no other ordering freedom to declare.

## Checks (smoke.py — rejection battery, 16/16 PASS on forge, frames.npz recorded)

1. settle + no-NaN, plank racked, ball in start dock, score 0; 2. randomization
readback (assembly xy + yaw, plank xy + yaw, ball xy differ across seeds); 3. rack
side: BOTH sides over 10 resets; 4. null policy 240 steps → score ~0; 5. SEED-family
naive (push the ball with no bridge) → ball lost to the chasm, score ~0; 6.
bridge-only → 0.30, no success; 7. MECHANISM ferry: the rolled ball is physically
supported over the void by the seated plank (probe dismantled before arrival); 8.
diving-board plank (far end unsupported) → not bridged; 9. rails-down plank → not
bridged; 10. ball parked on the plank's far end over the far ledge → x window
rejects; 11. ball on the bare far ledge → height window rejects; 12. ball on the
floor beside the target dock → rejected; 13. settle gate: the exact success position
moving at 0.45 m/s is refused (dismantled); 14. latched credit survives carrying the
plank off a made bridge; 15. rejection audit — success() never True at ANY step of
the battery; 16. frames.npz saved (53 frames, 960×600).

Run (forge):
`python -u -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_i36.solve --headless [--seed N]`
`python -u -m simgen_tasks.libero_kitchen_scene9_turn_on_the_stove_i36.smoke --headless`
