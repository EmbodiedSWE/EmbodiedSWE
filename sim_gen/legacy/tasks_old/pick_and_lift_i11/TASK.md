# scaffold_lift — build a block scaffold so the red cube RESTS at the marked height

## Seed provenance

- **Seed:** `rlbench/pick_and_lift`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/pick_and_lift.py`) — a red target
  cube among colored distractor cubes (`stack_blocks_distractor0/1`) and a floating
  `success_visual` sphere; the plan is: identify the red cube, grasp it, and **lift it
  to the floating target position** — a prehensile carry to a point in the air, judged
  while the cube is held. The distractors are pure clutter.
- **This task:** `sim_gen.scaffold_lift` (scene `scaffold_lift`, robot `null`), fully
  procedural geometry (built-in cuboid/cylinder spawners + one custom visual-only ring
  spawner; no asset files).

## What changed, and why it is strategically different

The same cast — a red payload cube, blocks, a floating height marker — is rearranged so
the seed's plan is *physically incapable* of succeeding, not just miscalibrated:

1. **The goal is a free-standing outcome, not a held pose.** Success requires the red
   cube's center to REST inside a ±10 mm band at the ring's height, above a plinth,
   settled, and *persistently* so (30 consecutive settled sub-steps). A cube carried to
   the marker never satisfies it: while held its post-step speed is g·dt ≈ 0.08 m/s
   (above the 0.05 m/s settle gate, so the persistence streak never accumulates), and
   the moment it is released it falls. This is asserted as a negative control — the
   seed's own strategy, executed verbatim, scores ~0.
2. **The distractors are promoted to essential material.** Nothing in the scene is tall
   enough to reach the band (lowest band 195 mm; tallest single block + payload tops
   out at 165 mm). The only route up is **construction**: stack scaffold blocks on the
   plinth and perch the payload on top. The seed's clutter becomes the load-bearing
   solution.
3. **A discrete selection problem replaces color matching.** Block heights are 60 / 90 /
   120 mm and the ring height is re-sampled each episode among the three *pair* sums
   (+ payload half-height): 195 / 225 / 255 mm. Exactly one pair reaches the band; every
   other subset (singles, the wrong pairs, the triple) misses by ≥ 20 mm by
   construction. The solver must read the marker height and *choose* the matching pair,
   not just recognize "the red one".

A solver therefore needs a different plan — *measure → select materials → build a
stable tower bottom-up → perch the payload* — instead of *pick the red cube → hoist to
the marker → hold*.

## Difficulty tier / stages

**medium — 4 stages, execution order REQUIRED** (bottom-up, forced by gravity: support
must exist before anything can rest on it; the payload cannot be inserted underneath):

1. read the ring height and select the correct pair of scaffold blocks;
2. place the first (larger) block on the plinth (latched, 0.20);
3. stack the second block on it (latched, 0.45);
4. set the red cube on top and let it settle inside the band (0.70 latched for any
   elevated rest; 1.0 iff success).

Within the scaffold pair the physical order (larger-first) is the stable choice but is
not asserted; support-before-payload IS forced.

## Rubric (graded score in [0, 1])

- `0.00` — nothing / blocks untouched (null policy; everything spawns on open ground).
- `0.20` — latched: a scaffold block settled ON the plinth top (its resting height
  there is 20 mm above on-ground rest — unambiguous by construction).
- `0.45` — latched: a two-block scaffold (a block settled ≥ 45 mm above its own
  on-plinth height, over the plinth).
- `0.70` — latched: the red cube settled elevated on structure over the plinth (any
  height — a wrong-pair tower earns exactly this, never more).
- `1.00` — **iff `success()`**: red center inside the band (|z − band| < 10 mm), over
  the plinth axis (xy < 70 mm), payload AND all blocks settled (|v| < 0.05 m/s), held
  for 30 consecutive sub-steps. Latches run in `post_step` at sim rate, so transient
  achievements are kept; success itself is a persistent physical present-state outcome.

Physics honesty: the plinth is kinematic furniture and the ring is a visual-only marker
(no colliders), both re-posed per episode with readback-verified randomization; all four
cubes are fully dynamic. The oracle carries kinematically but every placement ends in a
genuine 3 mm release-drop-and-settle, and success is judged only on settled poses.

## Randomization (per episode, verified by readback in the smoke)

- plinth xy: ±60 mm jitter;
- required pair → ring/band height: sampled from 3 discrete values (structural
  randomization — the solution's block subset changes);
- block + payload scatter: ±20 mm xy jitter and free yaw per body, on collision-safe
  staggered slots.

## Smoke checks (18)

1. settle/no-NaN: reset layout settles finite, clean slate (score 0, no success)
2. randomization-is-real: plinth/block readback differs across resets
3. randomization-is-real: band height (required pair) varies across resets
4. null-policy-fails: 240 idle steps, score ~0
5–7. oracle reaches success() on seeds 0 / 1 / 2
8. rubric: base block on plinth scores 0.20
9. rubric: two-block scaffold scores 0.45
10. rubric: red elevated scores ≥ 0.70
11. rubric: monotone 0 → 0.20 → 0.45 → 0.70 → 1.0
12. negative (SEED'S OWN STRATEGY, expressible here): red carried to the ring and held
    — success never fires while held
13. negative (seed strategy, cont.): the released cube falls — score ~0, no success
14. negative (near-miss): wrong-pair tower misses the band by ≥ 15 mm, no success
15. negative (near-miss, cont.): wrong-pair tower keeps partial credit 0.70, never 1.0
16. negative (tolerance/place): the correct tower built beside the plinth scores ~0
17. calibration: COM-inside top-face offsets seat ≥ 8/9 (analytic support-polygon cliff)
18. calibration: COM-outside offsets (≥ 10 mm past the edge) seat ≤ 1/6

`frames.npz` (show + oracle seed 0 + seed-strategy control) is saved to the CWD.
