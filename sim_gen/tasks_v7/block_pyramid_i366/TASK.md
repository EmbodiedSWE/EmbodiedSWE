# block_pyramid_i366 — `jenga_quarry`

Slide the two red blocks lengthwise out of a pre-built Jenga-style tower — under the
weight of the layers riding on them — and bank them in the blue tray, while every
cream block stays seated in its original slot.

## Seed provenance

Derived from **`rlbench/block_pyramid`**
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/block_pyramid.py`): six loose green
cubes among red distractors must be stacked up into a 3-2-1 pyramid. The seed is a
purely **constructive** task — repeated free-space pick-and-place, judged on the
assembled end pose; nothing starts in contact with anything and no existing structure
constrains any move.

## What changed and why it is strategically different

Every axis of the seed's strategy is inverted, so a solver needs a different plan and
different code structure — not a re-parameterized pyramid:

1. **Construction → extraction.** The structure ALREADY EXISTS at reset: a 3-layer ×
   3-block crisscross tower (alternating x/y layers) on a plinth. Nothing is built;
   two marked blocks are quarried OUT of it.
2. **Free-space placement → slide under load.** Each red target is buried in the
   bottom or middle layer, fully roofed by the perpendicular layer above. It cannot
   be lifted out (raising it heaves the whole superstructure — physically denied and
   probe-verified); the only working move is the lengthwise Jenga slide beneath the
   upper layers' weight, in sliding contact with the roof and floor of its channel
   for the whole stroke. The seed has no in-contact manipulation at all.
3. **Assembly goal → preservation constraint.** The seed judges what you built; this
   task judges what you did NOT disturb: all seven cream blocks must stay in their
   slots (xy/z/tilt tolerances tighter than one slot pitch / one layer height).
   "Topple the tower and pick the reds from the rubble" scores zero — every
   partial-credit latch is gated on `tower_standing()` at the moment it is earned.
4. **Color selection → positional selection.** WHICH slot of layer 0 and of layer 1
   is red is randomized per episode, so a memorized extraction sequence fails.

Also different from the sibling tasks read during this session (i42 tilt-labyrinth
ball routing; i352 tunnel-relay proxy-column push: there the cargo is untouchable
and pushed indirectly by a feeder column — here the targets are manipulated
directly, but through an under-load slide subject to a preservation constraint).

## Apparatus (fully procedural, no external assets)

- Kinematic **plinth** 9.3 cm square × 6 cm, gray; high-friction surface (anchors
  the bottom layer against slide drag).
- **9 dynamic blocks** 13 × 3 × 1.8 cm, 100 g: 2 red targets, 7 cream. 3 layers × 3
  slots, 3.1 cm pitch, 1 mm side gaps; block ends protrude ~1.9 cm past the tower
  span on both sides (the grasp/push feature). Block material is slick (0.18) while
  furniture is grippy (0.60) — pair-averaged, the stationary blocks anchor to the
  plinth harder than the sliding red drags its roof.
- Kinematic blue **tray**, 16 cm square inner, 4 cm walls, open top, on the table
  clear of the extraction fall zone.

Per-episode randomization (readback-verified in smoke): rig xy ±5 cm, rig yaw ±30°,
red slot of layer 0 and of layer 1 (0–2 each), tray xy ±3 cm.

## Solution outline (solve.py)

1. **Extract red0** (bottom layer): velocity-capped (2 cm/s) horizontal force at
   the CoM along the block's own axis — the fingertip pull on its protruding end —
   slides it through its channel under the superstructure load, with a weak lateral
   centring force and a yaw-squaring torque. The force controller is jolt-free:
   slow feedforward ramp + viscous brake, with a one-shot breakaway cut (streak-
   gated velocity hysteresis) so the stick→slip transition cannot slam the roof
   into slipping and drag it along. Past the last support edge a level-hold pitch
   torque (gravity-overhang feedforward + PD) stops the emerging block seesawing
   its tail up into the roof. Direction −x, away from the tray. The block emerges
   past the span and drops off the plinth; a ground-phase nudge takes it past the
   extraction line. **Teleports are never used for extraction.**
2. **Bank red0**: one pose write (transport only) carries the free block to above
   the open tray; the descent is a gravity drop and it must physically settle on the
   tray floor.
3. **Extract red1** (middle layer, +y) and **bank** it the same way, with one
   addition: past mid-stroke the grip also carries HALF the block's weight (its
   exit pivot is a cream block's top edge, not the kinematic plinth — full weight
   there walks that cream, while a full-weight lift presses the block up into the
   roof and makes the roof block ride it; half-weight keeps both under control).
   The lift cuts once the tail clears the roof, for a clean level drop.
4. Assert live success (both reds in tray + all creams in their slots + settled),
   hold hands-off ≥3.5 simulated seconds, re-assert, print `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` prints at every phase boundary and is non-decreasing (latched
credit: 0.05 slide + 0.10 extract + 0.20 bank per red, cap 0.70; 1.0 iff live
success).

## Embodiment argument (single Franka, parallel-jaw gripper)

- Every block end protrudes ~19 mm past the tower span and past the plinth — a
  clean pinch-grasp target for parallel jaws (block cross-section 30 × 18 mm, well
  within the Franka gripper's 80 mm stroke).
- The extraction is exactly what a gripped pull does: a horizontal force of ≤ a
  few N along the block axis at grasp height. The solve's velocity-capped CoM
  force emulates a compliant pull on the gripped end; the level-hold pitch torque
  and the half-weight lift on red1 are what a rigid wrist grip on the protruding
  end provides for free (the gripper holds its end at height and level while
  pulling); all magnitudes (~few N, ~0.1 N·m) are far below the arm's capability.
- The banked drop through the tray's open top from ~10 cm emulates carry-and-release.
- One base pose works for all episodes: the tower stays within ±5 cm / ±30° of the
  origin and the tray within a few cm of a fixed spot 34 cm away; a base at
  ~(0.15, −0.55) faces both the tower's protruding ends and the tray's open top
  within a 0.65 m reach envelope. Rig yaw is capped at ±30° so a protruding end of
  every layer axis is always presented to the robot side.
- No bimanual need: the tower is self-supporting during a slow slide (verified by
  the solve's `tower_standing()` asserts after each extraction).

## Execution order

The two reds may be extracted and banked **in either order** (declared in
`describe()`); no other ordering constraint exists. solve.py does red0 then red1;
the rubric never requires that order.

## Checks (smoke.py — rejection-only battery, 16 checks)

1. Settle/no-NaN: all 9 blocks rest in their slots, still.
2. Score ~0 at reset, no success.
3. Randomization readback: rig xy/yaw + tray jitter really vary.
4. Randomization readback: red slots vary per layer (≥2 values each, ≥3 pairs).
5. Null policy 240 steps → score ~0, no success.
6. Seed-strategy analog (reds STACKED on top of the tower) → rejected.
7. Topple-and-collect (creams scattered first, then reds dropped in the tray) →
   score ~0, no success (latch gating is load-bearing).
8. Partial: one red banked → no success, score in the partial band.
9. Near-miss: beside the tray and at rim height → `in_tray` False.
10. Wrong objects: creams in the tray → rejected.
11. Airborne over the tray, judged mid-air → rejected.
12. Latched credit survives regression (banked red returned to its slot).
13. Real contact Jenga slide of red1 (as in solve) → earns slide+extract latches,
    tower standing, still no success.
14. Roof-pin denial, non-vacuous: 2 N straight up moves the buried red < 8 mm while
    the same pull lifts a free block > 40 mm.
15. Rejection audit: success() never True at any judged point.
16. Final no-NaN.
