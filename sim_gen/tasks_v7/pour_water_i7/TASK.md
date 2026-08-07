# pour_water_i7 — RampChockScene (`simgen.ramp_chock`)

Park two green balls at rest on the yellow band of a 12° ramp — a pose in which a
free ball physically CANNOT rest — by first wedging a blue chock bar across the band
(held by friction) and then releasing each ball just uphill of it so gravity rolls it
back into the chock face: a constructed static equilibrium (deck → chock → balls),
maintained by contact alone.

## Seed provenance

- **Seed task**: `pick_place/pour_water` (RoboVerse
  `roboverse_pack/tasks/pick_place/pour_water.py`) — "pour the water". Grasp a small
  block standing in for water, then follow a five-waypoint SE(3) trajectory that
  carries it over a vase while tracking prescribed ROTATIONS (the tip-over motion).
  The reward is dominated by how faithfully the gripper follows that path; the end
  state is a HELD, tilted object above a target. One prescribed trajectory, no
  mechanism, no relation between free objects, nothing left standing when the hand
  opens.

## What the task is

A kinematic gray RAMP (550 × 340 mm deck, pitched 12°, downhill edge at ground
level, high-friction surface μ≈0.85/0.75) stands on the floor; a YELLOW BAND is
painted across the deck at slope coordinate u ∈ [0.22, 0.38] m (visual only — no
proud collider). Per episode the ramp's foot jitters ±40 mm and its heading swings
±50° (the fall line rotates — it must be read, not memorized), and four loose bodies
scatter on the flat floor with free yaw (keep-out resampled: inside a 0.18–0.60 m
annulus, off the ramp footprint, pairwise separated): two GREEN BALLS (Ø60 mm,
400 g), a BLUE CHOCK BAR (50 × 160 × 34 mm, 800 g) and a smaller RED BALL (Ø40 mm,
300 g — the decoy).

Goal: both green balls at rest ON the deck inside the band, each directly
uphill-adjacent to the chock, the chock at rest flat on the deck, cross-slope, in
the band zone — everything settled and held only by gravity and contact.

## Why strategically different

- **vs. the seed**: the seed's whole skill is *tracking a prescribed pour path with
  a held object*; its end state is a gripper mid-air above a target. Here nothing is
  poured or tilted along a path and nothing is held at the end: the deliverable is a
  STATIC EQUILIBRIUM THAT DOES NOT EXIST NATURALLY. The seed's transport-and-deliver
  plan is constructed verbatim in smoke 5 — both balls carried to the band and
  released — and both roll straight off the ramp: transport alone leaves nothing
  standing, score ≤ 0.11, never success.
- **The physics is load-bearing, not rubric fiat**: PhysX spheres are analytic
  colliders with zero rolling resistance — a free ball has NO rest pose anywhere on
  the bare 12° deck (smoke 4 proves it empirically). So (1) the chock is physically
  NECESSARY, (2) the execution order (chock first, then balls) is enforced by
  GRAVITY — a ball released before the chock simply rolls out of the scene — and
  (3) the judged state is live: teleport the chock away after success and both balls
  roll off, success collapses (smoke 11).
- **The decoy is physically self-rejecting**: the red ball wedged at the chock
  station rolls off the ramp itself, taking the "chain" with it (smoke 6) — only the
  flat-faced bar can hold the slope (its no-slide and no-tip inequalities under the
  loaded chain are asserted in `__post_init__`).
- **vs. corpus tasks read**: no gravity-gate door closure (`close_microwave_i4/i5`),
  no pour-then-invert-park (`libero..._i2`), no bayonet twist (`libero..._i3`), no
  rod-ram ejection (`peg_insertion_side_i1`), no bar-seat + pin fastening
  (`peg_insertion_side_i2`), no pin-unlock + tunnel slide (`pick_single_egad_i3`),
  no thread-and-hang (`pick_single_egad_i4`), no funnel drop-stacking
  (`setup_checkers_i2`), no counterweight lever (`light_bulb_i5`), no die
  edge-tumbling onto pads (`open_oven_i6`). *Constructing a multi-body force chain
  on an incline — placing a blocker so that other bodies can rest where they
  otherwise cannot* — appears in none of them, nor in the robobench house suites
  (pen_holder inserts downward into a cup; balance_scale, combination_safe, syringe,
  packing/pouring share no mechanic).

A solver therefore needs a different PLAN each episode (read the ramp heading,
orient the chock across the fall line, wedge it first, then park each ball against
it) and different CODE STRUCTURE (a ramp-frame slope coordinate system, a per-ball
three-body support predicate, latched per-body credit) — not a waypoint tracker.

## Solution outline (solve.py = the legitimacy certificate)

1. **P0** settle 1 s; READBACK the layout (ramp xy + yaw, all four spawns); baseline
   score 0, no success.
2. **P1** (teleport = transport only): one root-state write carries the chock from
   the floor to a FREE-SPACE hover 8 mm above the deck at the band's lower edge,
   deck-matched orientation — the pose a gripper would release from. Hands-off: it
   FALLS onto the deck and grips by friction. Zone latch → score 0.15.
3. **P2** ball_a teleported to a hover 8 mm above the band just uphill of the chock
   face (offset −45 mm cross-slope); hands-off: it falls, ROLLS back down the slope
   and the chock face arrests it. Park latch → score 0.40.
4. **P3** ball_b likewise at +45 mm cross-slope: rolls back and parks SIDE BY SIDE
   with ball_a against the chock → success, score 1.0. (An escalating downslope
   CoM nudge ≤ 0.8 N stands by for stalls; analytic spheres never needed it.)
5. **P4** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

No judged clause is ever written: resting heights, the rolling approach, the arrest
and the final equilibrium are all outcomes of gravity, rolling contact and friction.
Monotone `SIM_GEN_SCORE`: 0.0000 → 0.1500 → 0.4000 → 1.0000 → 1.0000.

## Rubric

- 0 → 0.65 latched shaping (never decays): 0.15 chock ever at rest cross-slope in
  the band zone + 0.05 per ball ever on the deck + 0.20 per ball ever parked
  against the chock inside the band (cap 0.65).
- 1.00: success — live: both balls at rest on the deck (center one radius above the
  plane) inside the band, each within 85 mm along-slope of the chock center and
  within its 160 mm face span; chock at rest flat, cross-slope (±25°), in the zone;
  all three settled (velocity gates) and finite.

Unfakeable without the mechanic: a ball cannot rest on the bare deck at all
(smoke 4), and every resting place off the deck fails the on-deck height clause.

## Franka embodiment (single arm, parallel jaw 80 mm, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m). The
spawn annulus (0.18–0.60 m, all bearings) surrounds the base — joint 1 spans ±166°,
and the residual rear wedge is covered by elbow-flipped configurations; nothing
spawns closer than 0.18 m. The ramp foot sits at 0.26 ± 0.04 m; the farthest work
point (band top, u = 0.38) is ≤ 0.68 m away at 0.11 m height — comfortably inside
the envelope, all manipulation below 0.12 m.

- **Chock** (50 × 160 × 34 mm, 800 g): pinch across the 50 mm width (or 34 mm
  height) — both ≪ 80 mm jaw stroke; 160 mm length gives a long stable contact
  line. Placing it is a lay-down on the deck with a wrist roll to match the
  cross-slope heading read from the scene; the ±25° axis tolerance and the 60 mm
  zone slack are coarse by placement standards. Friction holds it the instant it is
  released (tan 12° = 0.21 ≪ μ = 0.75; margins asserted).
- **Balls** (Ø60 mm, 400 g): sphere pinch at the equator (60 ≪ 80 mm), approach
  from above at any yaw (sphere = yaw-symmetric). The release is deliberately
  forgiving: hover anywhere up to 85 mm uphill of the chock inside the band and let
  go — gravity does the fine positioning (the ball rolls back into the face). No
  precision insertion anywhere in the task.
- **Ordering** (declared): chock first, then either ball in either order — the
  cross-slope stations are independent. The order is gravity-enforced, not fiat.
- **Forces**: heaviest lift 800 g; the ramp is a kinematic fixture, so incidental
  contact cannot move the goal frame; the decoy needs never be touched.

## Validation evidence (forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0`: SUCCESS, scores 0.0000/0.1500/0.4000/1.0000/1.0000, balls at
  v = −46/+45 mm on the chock face, 27.8 s wall.
- `solve --seed 1` (ramp yaw +39.0° vs −10.1°): SUCCESS, same monotone scores,
  27.8 s.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 13/13**, frames.npz saved:
  1. settle/no-NaN: all four loose bodies settle finite on the floor; score 0
  2. randomization readback: ramp yaw, ramp xy, ball_a spawn all differ across seeds
  3. null policy: 240 idle steps, score ~0, no success
  4. NO-CHOCK counterfactual: ball released on the band rolls OFF the ramp — a
     sphere has no rest pose on the bare deck (the chock is necessary; order is
     gravity-enforced)
  5. SEED STRATEGY: both balls transported to the band and released — both roll
     off; delivery without the mechanism is worth ≤ 0.11, never success
  6. DECOY: red ball wedged at the chock station rolls off itself; the chain
     collapses — no chock credit, no park
  7. angled chock: yawed 45° off cross-slope on the band — axis clause refuses;
     no park, no success
  8. below-zone near-miss: chock wedged below the band, ball rests against it — a
     REAL settled equilibrium, rejected for place: no park, no success
  9. chock-only: score ~0.15, no success
  10. one-ball: chock + one parked ball ~0.40 < 0.9, no success
  11. REMOVE-CHOCK: constructed success is live; chock teleported away → both
      balls roll off, success COLLAPSES, latched score survives at 0.65
  12. rejection audit: success never fired during checks 4–10
  13. video frames.npz saved

## Files

- `scene.py` — RampChockScene + kinematic ramp compound spawner + rubric; registers
  `simgen.ramp_chock`.
- `solve.py` — teleport-transport + gravity wedge/roll certificate (`--seed N`).
- `smoke.py` — 13-check rejection battery + video.

Run (forge):
`python -u -m simgen_tasks.pour_water_i7.solve --headless [--seed N]`
`python -u -m simgen_tasks.pour_water_i7.smoke --headless`
