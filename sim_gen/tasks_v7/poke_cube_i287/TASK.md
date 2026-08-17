# poke_cube_i287 — Shunt Tray (captive 2x2 sliding-tile puzzle)

Env id: `simgen.shunt_tray` (scene-level, `robot="null"`).

## Seed provenance

Seed task: **`maniskill/poke_cube`** (RoboVerse
`roboverse_pack/tasks/maniskill/poke_cube.py`): grasp a peg tool and poke a red
cube a few centimeters across an open plane into a painted goal region. One
object, one planar stroke, judged by the poked object's own xy distance to the
goal marker.

## What changed

- The open plane becomes a walled TRAY: a 2x2 grid of cells (70 mm pitch)
  spanned by a flat COVER at wall height. Three tiles (one RED, two GRAY,
  56x56x36 mm bodies with a 16 mm knob on top) ride UNDER the cover; only the
  knobs poke up through a square-ring SLOT cut in the cover. The fourth cell is
  empty. A blue plate sunk into the floor of one cell marks the goal.
- The goal cell ALWAYS starts covered by a gray tile (sampling guarantees it),
  and two 56 mm tiles cannot share a 70 mm cell — so no single straight push of
  the red tile can ever score.
- Captivity is geometric and asserted in `ShuntTraySceneCfg.__post_init__`: the
  cover leaves 5 mm headroom over a tile, its rim + central island overhang
  every reachable footprint, and the slot (35 mm) is far narrower than a tile
  (56 mm) — no lift-out, tip-out, or squeeze-through exists. The island also
  blocks any diagonal knob path across the tray center.
- Per-episode randomization (readback-verified in smoke): the (goal, red,
  empty) cell permutation — uniform over all 24 assignments — plus tray xy
  jitter (±30 mm) and free tray yaw (±180°).

## Strategic difference (vs the seed and the corpus tasks read)

- **Seed (`poke_cube`)**: one open-plane planar stroke on one object. Here the
  seed's entire plan is a *constructed rejection case* (smoke check 7): pushing
  red straight at the blue cell jams it against the covering gray tile. The
  solver must instead solve a COMBINATORIAL rearrangement: route the vacancy
  around the 2x2 ring so the occupier vacates the goal, then bring red in — a
  seed-dependent 2–6 move BFS plan with per-move occupancy readback and
  replanning, a different plan structure and different code structure.
- **`poke_cube_i83` (same seed)**: armed-latch hands-off cascade (domino /
  hammer / ball chain, judged after a hands-off window). Here there is no
  stored energy and no cascade — every joule enters through direct quasi-static
  pushes; the difficulty is ordering, not dynamics.
- **`slide_block_to_target_i135` / die-roll family**: reorientation (topple to
  a target face) of a single body. Here orientation is only a validity gate;
  the challenge is multi-object ordered displacement.
- **`push_cube_i30`**: tool-mediated scooping of granular material. No tool, no
  granularity here.
- **`pen_holder` (packing)**: insertion/filling with free pick-and-place. Here
  pick-and-place is geometrically impossible (captive tiles) — transport is
  restricted to grid slides.

## Teleport-solution outline (`solve.py`)

Zero transport teleports: every object already starts in the tray and no tile
pose is ever written after reset — the whole solution is contact dynamics.

1. Read occupancy from physical tile poses (nearest cell center, tray frame).
2. BFS over the 2x2 vacancy puzzle → shortest legal shunt sequence ending with
   red on the goal cell.
3. Execute one move at a time: a velocity-capped horizontal force at the moving
   tile's CoM (the applied-wrench emulation of a fingertip on the knob), a
   lateral P-servo + small yaw-correcting torque to keep the knob centered in
   its slot corridor, force cut near the target center, coast + friction
   settle, occupancy-readback verification, and replanning after any imperfect
   move (stalls escalate the drive force).
4. Final settle → `success()` → ≥ 3.5 s hands-off persistence →
   `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing
(the scene latches partial credit).

## Embodiment argument (Franka, base at the origin)

- The tray anchor is (0.40, 0.0) with ±30 mm jitter; every knob stays within
  horizontal reach 0.26–0.55 m of the base, knob tops ~85 mm above the ground
  — comfortable Franka workspace, no reach-around needed (the tray is open on
  all sides and the knobs stand proud of the cover by 13 mm).
- Contact strategy per object: close the parallel jaw to a 16 mm pinch on a
  knob (or simply press a fingertip flat against a knob face) and drag
  horizontally along the slot — the 25 mm success tolerance, the wall stops,
  and the slot itself forgive lateral error; the cover prevents any accidental
  lift-out. The solve's CoM forces (1.6–5 N, ≤ 0.12 m/s) are well inside
  fingertip-contact capability.
- Nothing requires simultaneous two-object actuation: moves are strictly
  sequential single-tile slides.

## Execution-order declaration

Ordering is REQUIRED and geometrically forced: the gray tile covering the blue
cell must vacate it (into the empty cell, possibly after other tiles rotate the
vacancy around the ring) BEFORE the red tile can enter — two tiles cannot share
a cell, diagonal shortcuts jam the knob on the cover island, and tiles cannot
leave the tray. Smoke check 7 constructs the out-of-order end state (red pushed
at the still-covered goal) and asserts rejection.

## Rubric

- 0.25 — `vacated_ever` (latched): the covering gray ever left the goal cell.
- 0.15 — `red_moved_ever` (latched): red ever left its start cell.
- 0.30 — `app_max` (latched running max of 1 − d/d0, red to goal center).
- Non-success cap 0.70; **1.0 iff `success()`**: red settled ON the blue cell —
  center within 25 mm, at floor height (±8 mm), upright (≤ 15°), all tiles
  still. Live physical outcome only.

## Smoke battery (15 checks, all rejection/audit)

1. settle/layout readback (occupancy matches sampled cells, marker at goal,
   rest heights, still, finite)
2. reset score ≤ 0.02, no success
3. randomization: ≥ 3 distinct (goal, red, empty) triples; goal always covered
4. randomization: tray xy jitter > 8 mm, free yaw > 10°
5. null policy: 240 idle steps → score ≤ 0.02, no success
6. captivity probe: 6 N up + 2.5 N center-ward yank lifts the tile (non-vacuous)
   but the cover keeps it in the tray
7. captivity probe earns no success and stays under the non-success cap
8. seed strategy: straight push jams on the covering gray — rejected
9. position near-miss: 33 mm > 25 mm tolerance — rejected
10. wrong tile: gray centered on the blue cell — rejected
11. stacked: red atop the covering gray, judged on write — floor-height gate
12. tilt: red at the blue center pitched 25° — upright gate
13. latched credit unchanged after full regression; never success
14. rejection audit: success() never True at any judged point
15. final no-NaN

Verification: forge runs of `solve.py` (seeds 0 and 3) to
`SIM_GEN_SOLVE: SUCCESS` and `smoke.py` to `SIM_GEN_SMOKE: ALL PASS`.
