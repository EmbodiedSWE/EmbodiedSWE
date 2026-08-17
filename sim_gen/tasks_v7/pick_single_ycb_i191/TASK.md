# pick_single_ycb_i191 — Cradle & Clamp

Lay the BLUE square bar into a fixture's V-saddle (diamond orientation), then seat the
YELLOW U-clamp bracket over it so its legs bottom out in two pocket wells and its
cross-bar locks the bar captive (scene `cradle_clamp`).

## Seed provenance

Seed: `maniskill/pick_single_ycb` (RoboVerse
`roboverse_pack/tasks/maniskill/pick_single_ycb.py`) — pick one loose YCB object off
the table and raise it 7.5 cm (`PositionShiftChecker(obj_name="obj", distance=0.075,
axis="z")`). The seed is a bare pick-and-lift: one grasp + one vertical position
delta on a single free object; no placement, no second object, no ordering.

## What the task is

A gray KINEMATIC fixture stands at a per-episode jittered position with FREE yaw: a
base slab carrying a V-saddle (two pairs of 45-degree tilted plates forming one
straight saddle line) and, on the same line outboard of the saddle at y = ±64 mm, two
rectangular pocket wells (44 × 18 mm openings, 22 mm deep). On the ground nearby (each
at a jittered spawn with free yaw): a BLUE square bar (40 mm across flats, 80 mm,
150 g), an identical RED decoy bar, and a YELLOW U-clamp bracket (60 × 150 × 12 mm
cross-bar, two 36 × 10 × 66 mm legs matching the pockets, dark grasp knob on top,
120 g) standing on its legs.

Goal — a TWO-PART ORDERED ASSEMBLY, judged in the fixture's body frame:

1. The BLUE bar seated in the V-saddle: axis along the saddle line (≤ 12°), center in
   the saddle window, z inside ±7 mm of the flush DIAMOND rest height (apex +
   apothem·√2 = 59 mm). The diamond roll is load-bearing: dropped face-down the bar
   perches on its corner edges 12 mm proud of the flush seat and is rejected.
2. The bracket seated: origin in the seat window and band (legs bottomed INSIDE the
   pockets; legs standing ON the 22 mm walls read far outside the band), upright, leg
   line parallel to the pocket line.

Both at rest, simultaneously. The RED decoy in the saddle counts for nothing — the
seed's "which object" identity made load-bearing.

## Execution order — REQUIRED, enforced by geometry

Bar FIRST, bracket SECOND. Once the bracket is seated in the empty fixture its
cross-bar underside (96 mm) closes the saddle: entering over a V crest lifts the bar's
crown to ≥ 118 mm at the cross-bar's edge (> 96 + 15 mm, asserted in `__post_init__`),
and low entry wedges under the 45° plate overhangs. Smoke check 13 DRIVES the bar at
the closed saddle with a regulated push (gravity-compensated, velocity-regulated,
axial force escalating 0.8 → 2.4 N, 4 s): it advances 55 mm, jams at |x| = 75 mm
(window is 12 mm), and never enters. The bracket may be seated only after the bar is
in.

## Strategic difference

- **vs. the seed** (free-space pick-lift of one object): success here is a relational
  two-body assembly on a fixture — a re-orientation (ground face-down → axis-aligned
  diamond) + flush V-seat for one object, then a guided two-peg insertion that CAPS
  the first object, with a geometry-forced order and a wrong-object decoy. The seed's
  own end state (bar raised well off the ground, held) is constructed by smoke check 6
  with a real force and rejected: it scores only the small latched transport credit
  (≤ 0.15) and can never satisfy success().
- **vs. corpus tasks read**: `pick_single_egad_i3` (tunnel shuttle: pin-unlock +
  covered slide into a bin) and `pick_single_egad_i4` (tag hangers: thread aperture
  onto a peg, judged by hanging suspension) share no mechanic — here nothing hangs,
  nothing slides through a tunnel; the mechanics are flush V-seating with a required
  45° roll and a two-peg mortise insertion that closes over a captive part, in a
  forced order. No corpus task read has an order-enforcing occlusion assembly.
- **vs. robobench house suites**: `pen_holder` inserts pens downward into an open cup
  (containment, unordered); here the second insertion locks the first object captive
  and the order is geometrically forced; `balance_scale`, `combination_safe`,
  `syringe_dosing`, pouring/packing suites share no mechanic.

## Solution outline (the teleport solution = legitimacy certificate)

1. **Bar transport (teleport)** — the blue bar teleports from the ground to a free-air
   hover 55 mm above the saddle (fixture frame (0,0,0.114)), axis along the saddle
   line, rolled 45° to the diamond — the pose an arm reaches after grasp + re-orient.
   Non-scoring (far outside the z band); latches only transport lift/approach credit.
2. **Bar seating (contact)** — released: it falls onto the two 45° V faces, which
   center it into the flush diamond rest (readback z = 0.059 = predicted value).
3. **Bracket transport (teleport)** — hover at (0,0,0.145): upright, leg line along
   the pocket line, feet above the wall tops. Non-scoring.
4. **Bracket seating (contact)** — a gravity-assisted, velocity-regulated downward
   push with weak centering springs; the legs enter the wells under real collision and
   bottom out on the pocket floors (readback z = 0.102 = seat height); wrench cleared,
   ring-down hands-off.

Then ≥ 3.3 simulated seconds hands-off persistence before `SIM_GEN_SOLVE: SUCCESS`.
`SIM_GEN_SCORE` prints at every phase boundary and is non-decreasing (all shaping is
latched). Passed on seeds 0 and 1 with visibly different layouts (fixture at
(+0.292,+0.001) yaw 170° vs (+0.331,−0.038) yaw 62°).

## Rubric

- 0 → 0.60 latched shaping (never decays): bar ever lifted 0.06, best airborne bar
  approach to the saddle 0.14, bar ever in the saddle (loose window) 0.20, bracket
  ever lifted 0.06, best airborne bracket approach to its seat 0.14.
- 0.90: both geometric predicates hold simultaneously (bar flush in the V AND bracket
  seated in its pockets).
- 1.00: success — both hold AND both parts settled.
- Null policy ≈ 0; seed-strategy end state ≤ 0.15; either part alone ≤ 0.40/0.20.

## Embodiment argument (Franka, parallel-jaw)

- The bar is 40 mm across flats (< 80 mm jaw stroke): pinch two opposite flats with
  full pad contact; the 45° roll to diamond is a wrist roll about the grasp axis. The
  bracket is grasped by its dedicated 14 × 32 mm knob; the cross-bar and legs stay
  clear of the fingers, so neither insertion is occluded by the grasp.
- Clearances are visual-servo friendly: legs-to-pocket 8 mm in x, 8 mm in y over a
  22 mm-deep well; the V-saddle is self-centering (several cm of capture range), so
  bar placement tolerance is generous; seat/saddle heights (59 mm, 102 mm) and ground
  picks (20 mm, 72 mm) are comfortable table heights.
- Base pose: place the Franka base at ≈ (−0.35, 0, 0) facing +x. The fixture center
  sits at 0.30 ± 0.04 m and all spawn zones within |y| ≤ 0.24 m — everything inside a
  0.85 m reach envelope; the fixture's free yaw only re-orients the saddle line, all
  approach directions remain top-down.
- Execution order: REQUIRED — blue bar first, bracket second (see above; the reverse
  order is geometrically impossible and smoke-verified).

## Checks (smoke.py, rejection-only battery — 17)

1. settle + no-NaN after reset; 2. baseline score ≤ 0.02; 3–4. randomization readback
across 6 seeds (fixture x/y/yaw spread; part spawn spread); 5. null policy 240 steps
≈ 0; 6. SEED-strategy rejection: real upward force raises the bar far past 7.5 cm and
holds it — never success, latched ≤ 0.15; 7. wrong object: RED decoy physically
seated flush in the V (geometry readback) — no saddle credit, not success; 8.
face-down near-miss: no 45° roll → perches ~12 mm proud, rejected by the z band; 9.
crosswise bar (axis across the valley) rejected by the alignment clause; 10. bracket
perched ON the pocket walls (+22 mm readback) rejected by the seat band; 11. bracket
yawed 90° cannot seat; 12. out-of-order (a): bracket really seats in the empty
fixture — partial credit < 0.9; 13. out-of-order (b): a regulated push (grav-comp,
velocity-regulated, up to 2.4 N axial, 4 s) drives the bar 55 mm at the closed saddle
— it jams outside the window (min |x| = 0.075 > 0.012), never enters (order is
geometry); 14. bar seated alone < 0.9; 15. latched credit survives teleporting the
seated bar back to the ground; 16. rejection audit: success never fired during the
battery; 17. final no-NaN. frames.npz (rgb) saved from a replicator camera.

Run (forge):
`python -u -m simgen_tasks.pick_single_ycb_i191.solve --headless [--seed N]`
`python -u -m simgen_tasks.pick_single_ycb_i191.smoke --headless`
