# serving_shelf (open_grill_i260)

Deploy the grill cart's drop-leaf serving shelf on the marked side, then serve the
platter onto it.

## Seed provenance

Seed task: `rlbench/open_grill` — lift the barbecue grill's hinged lid to its open
pose. One hinged body, one motion (raise past a threshold angle), goal judged on the
lid angle alone.

## What changed / strategic difference

The seed is a single-hinge "raise the lid" task. This task keeps the outdoor-cooking
cart context but is built around a **fold-out drop-leaf mechanism with a forced
three-step order and a placement payload**, none of which the seed (or the corpus
tasks read this session) exercises in this combination:

1. **Raise is a sub-goal, not the goal.** The leaf must go UP past vertical onto its
   over-center keeper stop (+95°, where gravity presses it onto the joint stop, a
   stable hands-off rest) — but the *goal* state has the leaf back DOWN at level.
   A seed-style strategy ("raise the hinged thing and stop") is explicitly scored as
   a rejected near-miss (smoke check 5).
2. **Physically forced order.** The stowed brace bar sits inboard of the hinge line;
   its outward sweep passes through the hanging leaf's plane. With the leaf hanging,
   a brace push at the plant's torque cap (0.10 N·m) jams on the leaf tip at ~7–8°
   (smoke check 7); once the leaf is up on the keeper the same torque swings the
   brace freely past 75° (check 8). Lowering the leaf without the brace deployed
   drops it straight back to the hang stop — there is nothing to rest on (check 6).
3. **Contact-seated goal.** "Level" is not a commanded pose: the leaf is *lowered
   onto* the deployed brace and rests on it by contact (bar top 0.293 vs leaf
   underside at level 0.294; seats at ≈ −0.6°).
4. **Placement payload + distractor symmetry.** The platter must end up ON the
   deployed shelf (leaf-frame window), not on the cart top where it starts
   (check 9) and not past the shelf edge (check 13). The cart has mirror-image leaf
   + brace assemblies on both sides; a green tile marks the serve side, grey the
   decoy. Deploying the wrong/both sides revokes success (checks 11, 12).

Corpus overlap check: no task read this session combines an over-center hinge
keeper, an order-forcing swing-out brace under a torque cap, a contact seat, and a
frame-relative placement on the deployed structure. The nearest neighbours
(open_washing_machine_i252: single door + detent; close_box_i26: single lid) are
one-hinge single-motion tasks.

## Randomization (per seed, reset())

- Serve side: ±y (green/grey tiles swap physically with it).
- Cart base: x, y ∈ ±5 cm, yaw ∈ ±0.45 rad; the whole 5-body linkage (cart, two
  leaves, two braces, tile pair) is written as one composed SE(3) so joints stay
  consistent.
- Platter start on the cart top: x ∈ range, small yaw.
- Leaves written at −78° and settle onto the −80° hang stop (avoids flush-start
  depenetration).

Smoke check 3 verifies over 8 seeds: both sides drawn, ≥5 distinct cart yaws and
platter x's, cart/platter/tile positions readback-verified from sim state.

## Rubric

`success()` (all simultaneously, settled + finite):
- target leaf level: |θ| < 5°;
- target brace deployed: φ > 75°;
- platter on the shelf: leaf-frame window |x| < 0.08, y ∈ (0.045, 0.165),
  z ∈ (0.006, 0.032);
- decoy leaf hanging (θ < −60°) and decoy brace stowed (φ < 15°);
- settled: platter/leaf/cart lin vel < 0.05, FD hinge rates < 0.5.

`score()`: 1.0 iff success; else partial credit from slow-gated latches
(0.15 up-latch + 0.15 brace-latch + 0.25 seat-latch [level while brace deployed]
+ 0.20 platter-latch [while seated]) — non-decreasing, capped at 0.75 without
success. The seed-strategy end state (leaf parked up) scores ≈ 0.15.

## Teleport-solution phases (solve.py, transport-only teleports)

1. **Raise**: gravity-feedforward + velocity-cascade hinge torque (body-frame, along
   the leaf's own axis — drag-invariant) to +95° keeper; release; assert it rests
   hands-off > 88°. `SIM_GEN_SCORE 0.15`.
2. **Brace out**: same servo structure on the brace hinge to 90°, torque ≤ 0.10 N·m
   cap. `SIM_GEN_SCORE 0.30`.
3. **Lower + seat**: servo the leaf down to −2° target; it seats on the brace by
   contact at ≈ −0.6°; release, assert level + seat latch. `SIM_GEN_SCORE 0.55`.
4. **Serve**: teleport the platter (transport) to a free-air hover 5 cm above the
   shelf in the leaf frame, zero velocity, pure gravity drop (encoding-proof), latch.
   `SIM_GEN_SCORE 1.0`.
5. Hands-off persistence ≥ 3.5 sim-seconds with all drives asserted zero, then
   `SIM_GEN_SOLVE: SUCCESS`.

Forge results: seeds 0 (side −y) and 1 (side +y) both SUCCESS, rc=0.

## Embodiment argument (single Franka, one base pose)

Base pose: on the serve side of the cart, ~0.55 m from the cart face, centred on the
hinge line (cart top 0.40 m, hinge at 0.30 m, shelf at ~0.30 m — all inside the
0.855 m Franka envelope at comfortable height).

- **Leaf** (0.24×0.18 m plate, 0.35 kg): grasp bar (2.2 cm square section) along the
  outer edge is a parallel-jaw target from above/outside; the raise arc (−80°→+95°,
  radius ≈ 0.19 m from the hinge at 0.30 m) stays within reach; the gripper can
  release at the keeper and re-grasp for the lowering stroke. Torque needed at the
  bar ≈ 0.4 N·m / 0.19 m ≈ 2.1 N — trivial for the arm.
- **Brace** (0.14 m bar, 0.06 kg, ≤ 0.10 N·m): finger-push on the down-tab at the
  free tip (force ≈ 0.7 N horizontal sweep at z ≈ 0.29 m) — a one-finger sweep, no
  grasp needed.
- **Platter** (0.14×0.10 m board, 0.15 kg, 2.4 cm knob): knob is a top-grasp handle;
  pick from the cart top (0.41 m) and place 0.11 m outward onto the shelf — a short
  free-space pick-and-place at constant height.

All interactions happen on one side of the cart within a ~0.5 m³ workspace; no
regrasp behind the cart, no bimanual need (the keeper stop holds the leaf while the
brace is deployed, replacing a second hand).

## Execution order

Order is physically forced and rubric-enforced: brace-before-raise jams on the
hanging leaf (torque cap); lower-before-brace has no support and falls; platter
placed before the shelf exists cannot satisfy the leaf-frame window. Smoke checks 6,
7, 9 verify each out-of-order route fails.

## Smoke battery (14 checks)

1. settle sanity (hang stops, stowed braces, platter readback, score < 0.05)
2. side tiles physically placed (green = serve side)
3. randomization real + readback (8 seeds, both sides, distinct yaws/platter x,
   green tile follows side)
4. null policy 4 s: nothing moves, score < 0.05, never success
5. seed-strategy near miss: raise-and-park on keeper → holds hands-off, no success,
   0.10 < score < 0.20
6. leaf-level-without-brace: falls back to hang, seat latch stays 0
7. brace blocked by hanging leaf at torque cap (φ_max < 60°)
8. same torque clears once leaf is on keeper (φ > 75°)
9. platter on cart top with structure deployed → not success
10. exact solution → success, score == 1.0, persists 1 s
11. both sides deployed → success revoked, latched credit only
12. platter on decoy shelf → score < 0.05
13. near-miss drop past the shelf edge → not on shelf
14. ≥ 20 camera frames saved (frames.npz)
