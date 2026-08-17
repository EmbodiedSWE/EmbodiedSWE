# hand_trajectory_i374 — Sliding-Tile Keystone Shuffle (`simgen.tile_shuffle`)

## Seed provenance

Derived from **pick_place/hand_trajectory**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/hand_trajectory.py`): a humanoid
(Vega) GRASPS a small box and carries it through FREE SPACE, visiting five floating
waypoint markers in order; reward is dense distance-to-marker tracking of the HAND with
auto-closing fingers. The seed's whole strategy is *grasp, then fly the hand along a
memorized list of aerial waypoints*; the object touches nothing, the route is a fixed
list, and the episode ends hovering at the last marker.

## What changed, and why it is strategically different

Kept only the barest skeleton (an object must end up at a marked goal location). Every
element of the seed's strategy is inverted:

- **Nothing is carried — nothing CAN be.** The playfield is a walled 3×2 cell lattice
  covered by a slotted ROOF 5 mm above the tile tops. Each of the five tiles carries a
  knob that pokes up through the roof's slot network; the 84 mm slab cannot pass the
  30 mm slots (54 mm interference, asserted), so every tile is **topologically captive**.
  Smoke check 5 *constructs* the seed's strategy — flies the gold tile over every cell in
  turn, ending directly over the target — and latches exactly nothing (z gate); check 6
  then releases it and it lands ON the roof: the goal cell physically cannot be entered
  from above.
- **There is no memorized route — the route is combinatorial.** Five tiles fill six
  cells, so exactly ONE cell is vacant, and a tile can only slide into the CURRENT
  vacancy. Which moves exist depends on the per-episode random layout (gold cell, target
  cell, vacancy, frame pose); most of the work is moving the *other* tiles to route the
  vacancy around the gold tile. The solver must read the configuration and re-plan a
  puzzle (BFS over the 30 (gold, blank) states), not track a waypoint list.
- **The move law is physical, not scripted.** The roof slots run only along the two row
  lines and three column lines, so a knobbed tile cannot move diagonally or free-roam;
  walls and neighbouring tiles block everything except the vacancy. Smoke checks 10/11
  prove it with forces, not assertions: the same 0.8 N push that walks a free tile a
  full cell into the vacancy advances a blocked tile < 15 mm into an occupied chain —
  and an off-centre tile is even refused *lateral* slot entry (the first run of check 11
  failed precisely because the knob jammed on the column-slot mouth from 14 mm off-line).
- **The goal is settled containment, not a hover.** success() requires the gold tile
  *seated* in the marked cell (22 mm tolerance, z-gated) with all five tiles settled.
- **Versus the read corpus (tasks_v4–v7):** no existing task is a multi-body
  rearrangement puzzle. Sibling `hand_trajectory_i200` pushes ONE puck along a fixed
  prev-gated route; `fifo_rack` orders insertions in a no-passing channel; `tile`-less
  push tasks move one object to one place. The vacancy-routing sliding puzzle — where
  the obstacle set IS the manipulandum set and the move order is layout-dependent —
  appears nowhere.

## Apparatus (fully procedural, no meshes)

11 kinematic pieces re-pinned per reset (floor slab 37×28 cm; four walls 45 mm tall
enclosing the 27×18 cm interior; a 7-piece slotted roof — two row bands + four mid
segments — leaving 30 mm slots along the row/column centre lines, roof underside 45 mm
up = 5 mm over the tile tops), 4 green kinematic marker pads (visual-only, no collider —
a proud collider edge would wall the sliding knobs) pinned at the target cell's roof
corners, and 5 dynamic tiles (84×84×40 mm slab + ⌀20×40 mm knob, 150 g, damped,
solver 16/4): one GOLD, four grey. Config honesty is asserted in `__post_init__`: slot
interference 54 mm, roof headroom 5 mm (traps the tile), knob-slot clearance ≥ 1 mm with
contact offsets, knob protrudes 27 mm above the roof, cell play 6 mm (two tiles never
share a pitch), a resting tile passes the z gate while any above-roof pose fails it, and
the target is always ≥ 2 lattice moves from the gold start (null policy scores 0).

**Randomization (readback-verified):** frame xy ±3 cm + yaw ±12°; layout sampled per
episode — gold cell uniform, target uniform among cells ≥ 2 moves away, vacancy uniform
among the rest, greys fill in; tile spawn jitter ±1.5 mm. Smoke reads back the physical
wall pose, the gold tile's cell, and the pad centroid every seed (8 seeds, ≥ 3 distinct
layouts observed — 7/8 distinct in the logged run).

## Rubric

- `success()` = gold tile seated in the TARGET cell (nearest-cell < 22 mm, root below
  the 35 mm z gate) ∧ all five tiles settled (< 0.08 m/s).
- `score()` (latched, monotonic): 0.08 once any tile has made a real cell move (55 mm,
  z-gated) + 0.52 × best latched lattice progress of the gold tile, (d0 − d_min)/d0,
  updated only while *seated* in a cell; exactly 1.0 iff success(). Max non-success
  score 0.60; null policy scores 0 (d0 ≥ 2).

## Solution outline (solve.py — the legitimacy certificate)

**Teleports for nothing** — there is nothing to transport; the tiles spawn in the tray
and every millimetre of every move is contact dynamics. Before EVERY move the solver
reads the physical occupancy (nearest-cell readback of all five tiles), re-plans with
BFS over (gold_cell, blank_cell), and executes only the first move of the plan — so it
solves any layout, including both d0 = 2 and d0 = 3, with one code path. A move is an
axis-decomposed velocity-servoed WORLD-frame force at the tile's CoM (the applied-wrench
emulation of a fingertip push on the knob): the cross-axis error is servoed to < 3 mm
*before* the along-axis drive engages (an off-line knob jams the slot mouth — the
physical move law), with a yaw-keeper torque (±0.02 N·m) and slow-mode gain sized above
sliding friction (12 N/(m/s) × 0.04 m/s = 0.48 N > 0.37 N; K·dt/m = 0.67). Stalls
trigger a back-off-and-re-centre recovery with escalating cap 1.5 → 2.5 N. Phases: P0
settle (0.000) → one phase per executed move (score after the first gold seat-in
0.340 = 0.08 + 0.52·½) → final seat in the target (1.000) → 3.5 s hands-off persistence
→ `SIM_GEN_SOLVE: SUCCESS`. Verified on the forge for seeds **0** (gold 0→target 3,
24.0 s), **1** (24.1 s) and **2** (gold 2→target 1, mirrored frame yaw −7.6°, 24.0 s),
scores non-decreasing 0 → 0.340 → 1.000 every time, 4 moves each.

## Embodiment argument (Franka, one base pose)

Base at the origin facing +x; tray centre at (0.45, 0) ± 3 cm, yaw ±12°. The farthest
knob (an outer cell centre) sits ≤ 0.61 m from the base with its push band at world
z ≈ 0.093–0.100 m — comfortably inside the 0.855 m reach envelope with the standard
top-down wrist. Per-object contact strategy:

- **Tile knobs (the only things touched):** every knob protrudes 27 mm above the roof
  top and stands alone on its slot line, so a closed-gripper fingertip can contact its
  side from above anywhere on the lattice and push at ≤ 0.5 N (µmg ≈ 0.40 N to slide).
  The roof doubles as an anti-tip guard: with 5 mm headroom the tile cannot pitch, so
  even an imperfect high push only slides it. Direction changes are re-approaches
  around a ⌀20 mm knob with 5 mm of clear slot on either side. **No grasp is required
  anywhere**, and no part of the gripper ever needs to enter the covered volume.
- **Tray, walls, roof, pads:** furniture; no contact required.

## Execution-order declaration

There is NO fixed global order — and that is the point. The only physical law is *a
tile can move only into the current vacancy* (walls + neighbours + slot network enforce
it; smoke 10/11 prove it with matched forces). The actual move sequence emerges
per-episode from the layout; the rubric imposes nothing beyond latched best progress, so
any legal shuffle that seats the gold tile in the target and settles scores 1.0. The
physics closes the loopholes: no insertion from above (roof — smoke 5/6), no diagonal
or passing moves (slots + 54 mm interference), no credit for straddles, wrong cells, or
grey imposters (smoke 7/8/9).

## Checks (smoke.py — rejection battery, `SIM_GEN_SMOKE: ALL PASS 15/15` on forge)

1. Settle/no-NaN: all five tiles seated in their assigned cells (gold in gold_start,
   vacancy empty), settled, score 0.
2. Randomization readback: the frame physically moves across seeds (wall-piece pose +
   yaw spread).
3. Layout diversity: ≥ 3 distinct (gold, target, blank) triples across 8 seeds (7/8 in
   the logged run); physical gold pose and pad centroid match the claimed layout every
   seed; every layout honours min_dist ≥ 2.
4. Null policy (240 steps): score 0, no latches, no success.
5. **Seed-strategy analog (aerial carry):** gold flown over every cell centre, ending
   over the target, altitude verified 65 mm above the z gate → nothing latches, score 0.
6. Captivity: the released tile lands ON the slotted roof — the target cell cannot be
   entered from above; score 0.
7. Straddle: gold at the midpoint between target and neighbour (45 mm from both
   centres) is seated nowhere — score = the 0.08 moved latch only.
8. Wrong cell: gold seated one cell from the target — no success, score < 0.45.
9. Imposter: a GREY tile seated in the target cell earns nothing.
10. Occupancy (neg): 0.8 N push into a fully occupied chain advances < 30 mm (14.3 mm
    measured), tile never leaves its cell, stays below the z gate.
11. Occupancy (pos): the SAME force toward the vacancy walks the tile a full cell
    (93 mm) and seats it (positive control; requires re-squaring on the slot line —
    itself a demonstration of the move law).
12. Settle gate: gold written INTO the target sliding at 0.36 m/s is refused (progress
    latches the 0.60 cap, success stays False).
13. Latched credit: gold yanked back to its start cell keeps the 0.60; no success.
14. Audit: success() never True at any judged point; max score 0.60.
15. Final no-NaN; frames.npz saved.
