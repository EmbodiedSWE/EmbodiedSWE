# setup_checkers_i2 — CheckerSiloScene (`simgen.checker_silo`)

Load the silo through its yellow funnel so the settled stack reads, bottom to top,
RED / WHITE / RED — exactly three checkers in, the spare white left outside. The
channel is one checker wide and cannot be unloaded, so the drop order IS the stack.

## Seed provenance

- **Seed task**: `rlbench/setup_checkers` (RoboVerse
  `roboverse_pack/tasks/rlbench/setup_checkers.py`) — "Place the checkers on the
  board in the starting arrangement." A chessboard USD plus 24 identical cylinder
  checkers (12 white, 12 red) on a table; the plan is repetitive, unordered,
  tolerance-loose flat placement of pieces onto marked squares of an open horizontal
  surface, judged per-piece by position.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| destination | open horizontal board, every square directly reachable | enclosed vertical SILO: a 32 × 52 mm channel, 150 mm tall, walls on all sides (24 mm viewing slit), entered ONLY through a flared funnel mouth on top |
| placement act | lay each piece FLAT onto a square | release the piece on edge over an aperture; **gravity and the funnel do the seating** — the gripper never reaches the resting pose |
| ordering | none — 24 independent placements, any order | **total order, physically encoded**: the one-disc channel stacks in drop order and cannot be unloaded, so red→white→red is forced and a wrong drop is a permanent failure |
| count | place all pieces | **exactly three of four**: the spare white is a decoy that must be withheld |
| per-piece precision | pose tolerance on a surface | replaced by release-over-aperture + sequencing + restraint |
| which piece | pieces interchangeable within a color; fixed spawn | rack slot **permutation randomized** — which slot holds which color must be perceived |
| assets | RLBench USD board + primitive checkers | 100 % procedural USD (compound spawners): kinematic silo + funnel, kinematic grooved rack, 4 dynamic discs |
| judging | per-piece bbox/position | latched approach/prefix-1/prefix-2 credit + live success: exact count, contiguity from the floor, color order, on-edge alignment, everything at rest |

Code structure shares nothing with the seed: `@SCENES.register` BaseScene with two
kinematic compound spawners, silo-frame stack readout (sort inside discs by height,
gather colors), prefix predicates, latches in `post_step`,
`register_env(..., robot="null")`.

## Why strategically different

The seed's entire skill is *pick a piece, lay it flat on the right square* — repeat,
in any order, onto a surface that accepts corrections. Here that plan gains nothing:
no piece is ever laid flat anywhere (smoke check 6 constructs exactly that end state
— R/W/R flat in a neat row — and the rubric scores it 0), the goal region cannot be
reached by the gripper at all (fingers do not fit the 32 mm bore or the 24 mm slit),
and the last centimeters of every placement are performed by **gravity through a
funnel**, not by the arm. What the solver must bring instead is (1) **perception of a
randomized color permutation** on the rack, (2) a **strict, irreversible execution
order** (the bottom-up pattern is the drop order — the seed has zero ordering
constraints), and (3) **restraint** — the fourth checker must NOT be placed, inverting
the seed's "place everything" objective. Wrong order or a fourth drop is latched into
the physics itself: the channel cannot be unloaded (smoke 5 shows a 3×-weight shove
cannot push a seated checker back out), so [W,R,R] (smoke 8) and [R,W,R,W] (smoke 10)
are permanent failures, not recoverable states.

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.5 s; readback layout (silo xy/yaw ±30°, rack xy/yaw ±25°, slot
   permutation e.g. `[WRWR]`); baseline score 0, no success.
2. **P1** (teleport = transport only): carry `red_0` to the free-space release point
   above the funnel mouth — on edge, axis along the slit axis, zero velocity — and
   let it FALL: funnel → throat → channel → seated on the floor (silo-frame z 0.052)
   → prefix-1 latch (score 0.45).
3. **P2**: same release for `white_0`; it lands ON the red (z 0.090) → prefix-2
   latch (score 0.75).
4. **P3**: same release for `red_1` (z 0.128) → stack reads [R,W,R], exact count,
   aligned, at rest → success (score 1.0). `white_1` is never touched.
5. **P4** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints: 0.0000 → 0.45 → 0.75 → 1.0 → 1.0. Every insertion
is pure gravity + contact (a small escalating downward CoM nudge is armed for funnel
wedging but **never fired on any tested seed**).

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(0.00, −0.05, 0.00), facing +x** (nominal reach 0.855 m).
Silo foot ≈ 0.43 m away (mouth at 0.22 m height, release ≈ 0.24 m), rack ≈ 0.35 m:
everything inside a comfortable dexterous shell.

- **Checkers in the rack** (discs Ø40 × 18 mm, 30 g): they already stand ON EDGE in
  the groove — rails 28 mm apart, rail tops at 30 mm, so the upper ~20 mm of each
  disc is exposed. A top-down parallel-jaw pinch across the two flat 18 mm faces
  (max opening 80 mm ≫ 18 mm) needs no reorientation: grasp, lift straight up out
  of the groove, carry upright.
- **Release**: hover over the yellow funnel mouth (110 × 130 mm — a generous target)
  at ~0.25 m and open the gripper. The funnel and channel do all fine positioning;
  no insertion force, no precision placement, no reach into the silo (impossible and
  unnecessary — the 32 mm bore and 24 mm slit admit no finger).
- **Pull force / payload**: 30 g discs, trivial for the Franka; the only contact-rich
  moment is the pinch in the groove, with 20 mm of clear disc above the rails and
  6 mm nubs preventing neighbor collisions from rolling.
- **Spare white**: never touched — restraint, not manipulation.
- Silo and rack are kinematic (20 kg / 3 kg equivalents): incidental contact cannot
  move the goal frame.

## Execution order (declared)

`drop red → drop white → drop red, withhold the spare white`. The order is enforced
by physics, not rubric fiat: the channel stacks bottom-up in drop order and smoke 5
shows a seated checker cannot be pushed back out (3×-weight shove, max excursion
7 mm vs the 12 mm slit half-width), so any wrong drop is permanent (smoke 8, 10).

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (silo yaw −6.1°, rack `[WRWR]`): SUCCESS, scores
  0.0000/0.45/0.75/1.0/1.0.
- `solve --seed 1` (different silo/rack poses and slot permutation): SUCCESS.
- `solve --seed 2`: SUCCESS — three seeds, no funnel wedging, no nudge ever fired.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 11/11**, frames.npz (62 × 600 × 960) saved:
  1. settle/no-NaN; four checkers on edge in the rack, none inside; score 0
  2. randomization readback differs (silo yaw Δ24.2°, silo xy Δ49 mm, rack Δ48 mm)
  3. slot permutation: 5 distinct rack patterns over 10 resets, slot 0 both colors
  4. null policy: 240 idle steps, score 0, no success
  5. SLIT INTERLOCK: 3×-weight shove toward the slit, checker never leaves (7 mm max)
  6. SEED STRATEGY: R/W/R laid flat in a row on the ground → score 0, no success
  7. right order, wrong place: R/W/R pancake pile on the ground → score 0, no success
  8. out-of-order: channel stack [W,R,R] → p1/p2 unlatched, score 0.15, no success
  9. near-miss: [R,W] correct + second red against the OUTSIDE wall → no success
  10. exactly-three: all four in, [R,W,R,W] → count gate refuses, no success
  11. video frames.npz saved

## Files

- `scene.py` — CheckerSiloScene + silo/rack spawners + rubric; registers
  `simgen.checker_silo`.
- `solve.py` — teleport-transport + gravity-drop certificate (`--seed N`).
- `smoke.py` — 11-check rejection battery + video.
