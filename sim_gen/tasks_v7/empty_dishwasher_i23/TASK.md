# empty_dishwasher_i23 — captive sliding-tile shunt puzzle (`tile_shunt`)

## Seed provenance

Seed: `rlbench/empty_dishwasher`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/empty_dishwasher.py`) — swing the
dishwasher door down, pull the sliding rack out along its prismatic run, lift the
plate up and OUT of the machine. A trajectory-driven open-appliance **extraction**:
every interaction serves removal (create an opening, expose the payload, take it
away).

What survives from the seed: a shallow appliance-like frame holding plate-like
payloads in a rack lattice, payloads that must be moved through a constrained
mechanism, and a designated target among several identical distractors.

## Why this is strategically different

**vs the seed.** The seed's whole plan is *get the object out*. Here extraction is
**physically impossible** (60 mm plates are captive under a lid with 24 mm slots —
pulling a knob up just presses the plate against the lid; smoke check 6 proves it
with an 8 N yank) and appears nowhere in the goal. The plan is a combinatorial
**in-place rearrangement**: a 2×3 lattice of six cells holds five knobbed tiles and
exactly ONE empty cell. Physics enforces 15-puzzle move legality — the only
possible move is sliding a tile adjacent to the empty cell into it (a full line of
tiles jams against the wall; smoke check 7 proves it with the solver's own 2.5 N
force budget). Since the red tile can only ever advance into the empty cell, the
solver must first **cycle white distractors** to walk the empty cell onto the red
tile's path (a make-way plan, replanned per episode from the randomized
arrangement), then shunt the red tile into the goal corner marked by a green post
outside the frame. The seed's strategy applied here — red tile teleported out of
the frame — earns nothing (progress credit is gated on captivity; smoke check 2).

**vs the tasks_v7 corpus.** No other task is a combinatorial rearrangement with a
shared resource and physics-enforced move legality:

- *Pick/place & lift* (`approach_grasp_i29/_knife_i25/_spoon_i12`,
  `pick_and_lift_i16`, `pick_single_egad_i3/i4`, `coke_task_i15`,
  `libero_pick_tomato_sauce_i28`, the four libero put-bowl/pan tasks,
  `living_room_..._i18`): free objects, lift-and-carry. Here nothing can be
  lifted, ever.
- *Articulated open/close* (`close_box_i26`, `close_grill_i8`,
  `close_microwave_i4/i5`, `open_oven_i6`, `libero..open_top_drawer_i14`,
  `libero..top_drawer_..._i9`, `libero..turn_off_the_stove_i13`): drive one joint
  to a limit. Here there are no articulations at all — six rigid bodies and a
  kinematic frame.
- *Insertion / mechanism* (`peg_insertion_side_i1/i2`,
  `plug_charger_in_power_supply_i21`): single payload, single fixed channel;
  i21's slide is an unlock step *before extraction*. Here there is no extraction,
  five interacting payloads, and the move sequence (5–9 moves) is coupled through
  the single empty cell — order is decided by search, not by a fixed mechanism.
- *Push/track* (`pull_cube_i20`, `obstacle_i17`, `track_bowl_i27`,
  `track_ceramic_teapot_i22`, `pour_water_i7`): continuous guidance of one body.
  Here progress is discrete (cell hops) and mostly made by moving *distractors*.
- *Board setup* (`setup_checkers_i2`): closest neighbor — but its pieces are free
  bodies placed independently by pick/place; any order works. Here tiles are
  captive, only one legal move exists at any instant, and moves are strictly
  sequential through the shared empty cell.

## Scene

Procedural only (`UsdGeom.Cube` + `UsdGeom.Cylinder`, 1 mm contact offsets):

- **frame** (kinematic compound, ~230×160×43 mm): floor slab, four walls to
  lid-top height, and a 12-patch lid leaving a slot ladder open along the three
  column lines and two row lines of the lattice (slot width 24 mm);
- **tiles** (dynamic compounds ×5): 60×60×22 mm plate + Ø16×58 mm knob that rides
  the slots and stands ~45 mm proud of the lid. Tile 0 RED, four WHITE;
- **marker**: kinematic green post diagonally outside the goal corner.

Geometry honesty is asserted in `TileShuntSceneCfg.__post_init__`: captivity
(60 > 24 + 20), cell-to-cell slide clearance, knob-slot clearance, ≤ 8 mm headroom
(no climbing over a neighbor), knob graspability, and goal tolerance (18 mm) that
cleanly separates adjacent cells (70 mm pitch).

**Randomization** (readback-verified in smoke): frame xy jitter ± 40 mm + free yaw
(±180° — the whole lattice rotates; the solver must read it), the arrangement
(a random permutation of 5 tiles into 6 cells — which cell is empty and where the
red tile starts both vary), and the goal corner (never the red start cell).

**Rubric.** `success()` iff, settled: red plate center within 18 mm of the goal
corner cell center, in the captive under-lid band, and *all five* tiles captive in
the frame. `score()` = 0.15·(make-way latched) + 0.55·(best red progress latched,
gated on captivity) + 0.15·(red-at-goal latched), capped at 0.85; exactly 1.0 iff
success. Latches update every physics substep (credit never evaporates). Null
policy scores ~0.

## Teleport solution (solve.py)

Zero pose writes after reset — **every** interaction is contact dynamics:

- **P0** settle + readback: frame pose, tile occupancy (nearest-cell), goal corner
  derived from the *marker position* and cross-checked against the rubric buffer.
- **Plan**: BFS over the (red-cell, empty-cell) abstraction (30 states; white
  distractors are interchangeable) → shortest legal move sequence, usually
  starting with white-tile moves that walk the empty cell onto the red path.
- **P1..Pk**: each move is one constrained slide executed by a floating-hand force
  controller on that tile: position-PD toward the target cell center +
  friction-breaking bias (0.5→2.0 N stall-escalated), yaw-steadying torque, total
  force clamped to 2.5 N. Occupancy is re-read after every move; on a mismatch the
  solver replans (≤4 times). `SIM_GEN_SCORE` printed at every phase boundary with
  a non-decrease assert.
- **Final**: settle, success check, then a 10×41-step hands-off persistence loop
  (≥3.4 sim-seconds) before `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seed 0 (7 moves, 25.8 s) and seed 1 (8 moves, 26.8 s), both
`SIM_GEN_SOLVE: SUCCESS` with monotone score 0.00 → 0.15 → 0.43 → 1.00.

## Embodiment argument (Franka, base at ~(0, −0.45) from frame center)

Every interaction is a knob interaction in a shallow, fully exposed workspace:

- **Tiles**: the Ø16 mm knob stands 45 mm proud of the lid with knob tops at
  z ≈ 88 mm — a natural pinch for the Franka gripper, or a single-finger side
  push. Executing one move = push/drag the knob 70 mm along a straight slot with
  ~1–2.5 N lateral force — far below Franka payload, and the slot itself guides
  the motion (the controller's steadying torque stands in for the compliance a
  finger pair provides for free).
- **Frame**: kinematic; the robot never needs to touch it. All six cell centers
  lie in a 230×160 mm footprint — comfortably inside the Franka workspace from a
  base 0.45 m away, with no reach-around or regrasp: knobs are approached from
  straight above regardless of frame yaw.
- **Tolerances**: 18 mm goal tolerance and 10 mm slide clearances are an order of
  magnitude above Franka repeatability; the 24 mm slot vs Ø16 mm knob leaves 4 mm
  a side, and the puzzle *jams safely* (blocked lines just stop) rather than
  breaking on an over-push.
- **Marker**: read-only landmark, nothing to manipulate.

## Execution order

Physically forced, not rubric-declared: the red tile can only enter the goal cell
after the empty cell has been walked there, which requires the white-tile moves
first; each slide is only possible into the current empty cell (blocked lines jam
— smoke check 7). The rubric mirrors this with make-way → progress → at-goal
latches, but the *ordering itself* is enforced by contact dynamics.

## Checks (smoke.py — rejection battery, 16 checks, recorded to frames.npz)

1. settle/no-NaN — reset readback: all five tiles at their sampled cells (<6 mm);
2. SEED strategy — red tile placed OUT of the frame (the seed's end state):
   score ~0, no success;
3. randomization — frame xy + yaw vary (readback over 8 seeded resets);
4. randomization — red start / empty cell / goal corner all vary; marker-derived
   goal always matches the rubric buffer; red never starts on the goal;
5. null policy — 240 idle steps: score ~0, no success;
6. captivity yank — 8 N straight-up pull for 2 s: plate bears on the lid, origin
   never rises above the lid underside (lift-out physically impossible);
7. blocked line — 2.5 N (the solver's whole budget) on the middle tile of the
   full row for 2 s: the line jams on the wall, occupancy unchanged;
8. near-miss — red centered in a cell ADJACENT to the goal: not success;
9. near-miss — red 28 mm off the goal center (> 18 mm tol): not success;
10. lid percher — red on TOP of the lid above the goal cell: correct xy, above
    the captive band, not success;
11. wrong object — a WHITE tile in the goal cell, red untouched: not success,
    score ≤ 0.20;
12. constraint — red AT the goal but a white tile removed from the frame:
    red-at-goal true yet NOT success (captivity cuts both ways);
13. latched credit — a one-cell advance keeps its credit after sliding back;
14. monotonicity — a deeper advance latches strictly more progress;
15. rejection audit — success() never fired anywhere in the battery;
16. final no-NaN.

## Files

- `scene.py` — cfg + scene + rubric, registers `simgen.tile_shunt` (robot="null").
- `solve.py` — planner + force-controller solution (`SIM_GEN_SOLVE: SUCCESS`).
- `smoke.py` — the 16-check rejection battery (`SIM_GEN_SMOKE: ALL PASS 16/16`).
