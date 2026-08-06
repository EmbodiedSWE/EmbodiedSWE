# board_lean — prop the board leaning against the display wall

## Seed provenance

- Seed id: `maniskill/pick_single_ycb`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/pick_single_ycb.py`
  ("pick up the single YCB object": grasp the one free object on the table and raise it —
  the checker is literally a +7.5 cm position shift on the object along z; the dense-RL
  variant holds it grasped at a goal point with the arm static. The object's final
  support is the GRIPPER, and any grasp that lifts wins.)

## What changed, and why it is strategically different

The seed is a **prehensile transport/hoist**: the entire plan is "grasp the free object,
raise it, keep holding it" — success is a displacement of a held body, and the episode
ends mid-air. This task keeps the seed's minimal object world (ONE free body, one
fixture, a ground plane) but replaces the hoist with a goal the gripper cannot provide
and holding can never satisfy: a **metastable two-point friction equilibrium**.

- The free object is a board (420×120×30 mm) lying flat; the fixture is a kinematic
  display wall with a **slick face** (physics material μ≈0.1, combine-mode `min`) and a
  marked floor lane at its foot, re-posed (xy + yaw) every episode.
- Goal: the board must end **leaning against the wall** — bottom edge on the floor
  inside the lane, top edge against the wall face, tilted 45–80° — **released and
  settled**, held up by nothing but gravity, floor friction and the wall.
- The judged quantity is the final **unpowered equilibrium**, not displacement: success
  is honest *by construction* — a settled board with its bottom on the floor, ≥40 mm of
  horizontal top-over-bottom overhang toward the wall and its top end at the wall plane
  has its CoM outside the floor-contact support, so no such pose is static without the
  wall actually carrying load.
- The physics itself grades placement quality: below the slip angle
  (analytic `tan θc = (1 − μ_f μ_w)/(2 μ_f)` ≈ 38° for floor μ 0.6 / wall μ 0.1) the
  bottom edge slides out and the board falls flat on its own. The smoke measures this
  cliff (25° slips, 55°/70° hold).

Plan skeleton required of a solver: grasp the flat board, **reorient it to ~60° while
translating**, rest the top edge on the wall with the bottom staying grounded in the
lane, then **release and verify the equilibrium holds**. None of these steps exist in
the seed's plan, and the seed's plan (lift + hold) scores ~0 here.

Differentiation from sibling tasks (strategy-ledger check): not stacking-to-height
(i11), not bridging/two-post spanning (i31), not righting-a-fallen-object (i3), not
flip-into-rack-slot geometry-held placement (i16 — here nothing holds the board but
friction), not perch-on-convex/gentleness (i10), not topple (i25), not
rotate-in-place/anti-transport (i34). The claimed axes are: **leaning / propped
friction-equilibrium placement**, **release-to-judge (unpowered final state)**, and a
**physics-graded angle band with a real slip cliff**.

## Tier / stages

- Declared tier: **easy** — 1 stage, single skill (reorient-and-prop placement).
- Execution order: none required (single-stage; the 0.2 staging credit is a subset of
  the natural approach, not an ordered stage).

## Rubric

- 0.0 — nothing / board held aloft / board outside the lane.
- 0.2 — `staged` latch: either board end entered the lane on the floor (any pose).
- 0.6 — `propped` latch: a full valid settled lean was achieved at some point.
- 1.0 — iff `success()` NOW: valid lean (in-lane, top at wall plane, 45–80°, ≥40 mm
  overhang, bottom on floor, top raised) AND settled.

## Check list (smoke battery, 16 checks)

1. settle/no-NaN: reset layout finite, board flat, at rest, score 0.
2. randomization-is-real: rack xy+yaw and board spawn pose move across seeded resets
   (READBACK); board never spawns staged (out-of-lane by construction, 23 mm margin at
   worst-case jitter).
3. null-policy-fails: 240 idle steps → score ~0, no success.
4–6. oracle ×3 seeds: teleport-place a 62° lean with 2 mm standoff, release, real
   physics settles it into wall contact → success, score 1.0.
7–8. rubric monotonicity: ladder 0 → 0.2 (flat staged) → 0.6 (leaned then knocked back
   flat; propped latch persists) → 1.0 (re-leaned, current success); strictly increasing.
9–10. negative control A (the seed's own strategy): board hoisted 25 cm and HELD
   (gravity-compensated kinematic hold) scores ~0 with no latch; releasing it drops a
   flat board — still no success.
11. negative control B (pick-and-place instinct): board laid FLAT at the target, end
   touching the wall base, settled → staging credit only (0.2), no success.
12. negative control C (tolerance near-miss): near-vertical park at 86° — physically
   stable but overhang 29 mm < 40 mm and θ > 80° → rejected, no propped latch.
13. near-miss (physics): 25° lean is below the slip angle — board slides out and falls
   flat on its own, no success.
14–15. calibration probe: lean-angle sweep 25/40/55/70° → hold/slip table printed;
   25° slips, 55°/70° hold as success; hold-set monotone in angle (40° straddles the
   analytic 38° cliff and is allowed either way — its `success` is false below the 45°
   band edge regardless).
16. state roundtrip: get_state/set_state restores a success lean after the board was
   teleported away.

Physics honesty: the oracle is teleport-only for placement, but every judged state is a
settled, unpowered configuration produced by real physics after release; success() and
score() read only simulated poses/velocities.
