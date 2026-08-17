# approach_grasp_knife_i292 — key-turn anchor spreader

Scene `keyturn_spreader` · env `simgen.keyturn_spreader` · NullRobot, force/torque-driven.

## Seed provenance

Seed task: `pick_place/approach_grasp_knife` (RoboVerse
`roboverse_pack/tasks/pick_place/approach_grasp_knife.py`): a Franka approaches a knife
on a table, closes the gripper on it (gripper-object distance check, 5-frame grasp
persistence), and lifts it. The manipulated object IS the judged object, and picking it
up IS the task.

## What changed and why it is strategically different

Every load-bearing element of the seed's plan is inverted:

| | seed | this task |
|---|---|---|
| judged object | the grasped knife itself | two anchor blocks that can NEVER be touched (sealed in a roofed tunnel) |
| role of the knife-shaped object | the goal | a TOOL (a paddle-footed key); lifting/carrying it scores 0 (smoke check 5) |
| decisive interaction | close a jaw + lift | a QUARTER-TURN UNDER LOAD: the paddle cams both anchors apart through contact |
| approach direction | free-space approach to an exposed object | work through a 30 × 98 mm roof slot that admits the paddle in exactly one orientation |
| verification | gripper distance + lift height | armed-run latch over the anchor gap, live covered/settled conjunction |

The plan differs (insert-then-rotate a tool through an aperture vs. approach-grasp-lift),
and the code structure differs (kinematic 9-piece rig re-pinned per episode, compound
pxr-authored key spawner, rig-local frame rubric with an armed-run state machine —
nothing shared with the seed's gripper-distance checker). It is also different from its
sibling `approach_grasp_knife_i25` (support swap via insert-before-remove with a spoil
latch: different mechanism, different rubric machinery, different failure modes).

## The scene

A 340 mm roofed tunnel (interior 100 × 120 mm) with 56 mm letterbox mouths and a
30 × 98 mm roof slot sits at a randomized position/yaw. Inside, a crimson and a blue
anchor block (60 × 90 × 50 mm) stand at a sampled 24–34 mm gap under the slot. A key —
20 × 88 × 48 mm paddle foot under a 210 mm shank — lies on the ground at a randomized
pose beside the tunnel. Goal: gap ≥ 80 mm with both anchors still under the covered
span, spread ONLY during an armed covered run (armed at gap ≤ 42 mm, broken the moment
an anchor leaves |x| ≤ 110 mm).

Geometry honesty is asserted in `__post_init__`: slot admits the paddle broadside
(+8 mm) and refuses it turned (−58 mm); the swept diagonal clears the walls; the roof
underside is 70 mm above the block tops (a poked finger dangles); the letterbox opening
exceeds the block tops by only 6 mm < half the paddle thickness (no over-the-top lever);
the cammed anchors stay ≥ 30 mm inside the covered span; an uncovered-but-inside band
exists; anchors slide before tipping.

Rubric: `score()` = 0.10 key-through-slot + 0.15 paddle-seated-in-gap + 0.45 · latched
armed-run gap progress + 0.15 split latch; exactly 1.0 iff `success()` = split latched
∧ both covered now ∧ live gap ≥ 75 mm ∧ both anchors settled. Max non-success score
0.85; null policy 0.

Known bound (disclosed): `success()` does not name the key — it judges the anchors'
trajectory. Any covered spread requires camming through the slot because every other
route is barred by asserted geometry (mouth pushes only close or evict — smoke check 7;
turned paddle refused — check 6; rearranging after a cover break never latches —
checks 8/9).

## Teleport solution (solve.py)

Teleport = transport only (one hop, free air, never again). All load-bearing physics is
applied wrenches on the key:

- **P0** reset, settle, rig-frame readback → `SIM_GEN_SCORE 0.000`
- **P1** teleport the key to a hover 20 mm above the roof slot, upright, paddle aligned
  with the slot's long axis → 0.000
- **P2** wrench-servo descent: gravity-feedforward vertical velocity servo + world-frame
  xy PD to the slot centre + upright/yaw-hold PD torques; pause under the roof to
  re-centre; seat the paddle on the floor between the anchors (stall-detect + lift-retry
  if a block-top edge is caught) → 0.250
- **P3** quarter-turn cam: feedforward +z torque (0.18 N·m, escalating ×1.7 to 1.2 on
  stall) with a bang-bang speed governor (ω_z < 0.9 rad/s), small down-force keeps the
  cam seated; the 88 mm paddle spreads the anchors to ~85–90 mm; cut at gap ≥ 85 mm →
  0.850 (prints 1.000 when success is already live)
- **P4** all wrenches cut, anchors settle hands-off → 1.000
- **P5** ≥ 3.5 simulated seconds hands-off; success must still hold →
  `SIM_GEN_SOLVE: SUCCESS`

All gains respect the one-substep wrench delay (K·dt/m ≈ 0.05–0.15; √(Kp/I)·dt ≤ 0.11).
Verified on the forge: seeds 0 and 3, both `SIM_GEN_SOLVE: SUCCESS` (rc=0, ~20 s each),
scores monotone 0.000 → 0.250 → 1.000.

## Embodiment argument (Franka)

Base pose: on the key-spawn side of the tunnel, ~0.45 m from the rig centre — the slot
(≤ 0.5 m away, 130 mm high) and the key spawn (±3 cm around 30 cm from the tunnel) are
inside the 0.85 m reach envelope. Per-object contact strategy:

- **Key**: the 20 mm square shank is a canonical parallel-jaw grasp (Franka opening
  80 mm); grasp the shank high, lift, hold vertical.
- **Slot insertion**: with the paddle on the floor the grip point sits 178–258 mm up the
  shank, i.e. 48–128 mm ABOVE the roof — the hand never has to enter the slot.
- **Turn**: Franka joint 7 rotates ±166°, ample for 90° + coast; wrist torque limit
  (~12 N·m) dwarfs the 0.2–1.2 N·m the cam needs. Re-grasping halfway is possible but
  not required.
- **Anchors**: intentionally untouchable — 30 mm slot vs 90 mm hand span, 70 mm of
  clearance under the roof vs ~50 mm finger reach, letterbox lintels 6 mm above the
  block tops.

## Execution order (declared)

1. `scene.py` written with the minimal goal predicate (armed-run gap latch + covered ∧
   live-gap ∧ settled conjunction) and geometry asserts.
2. `solve.py` iterated on the forge until physically solved — passed on the first
   forge run for seed 0, then seed 3.
3. Rubric finalized (unchanged after solve: latches + weights confirmed), then
   `smoke.py` written.
4. `smoke.py` run on the forge → `SIM_GEN_SMOKE: ALL PASS 16/16` (first run), frames.npz
   recorded (212 frames, 960×600).

## Checks (smoke.py, 16)

1. settle/no-NaN + anchors physically at sampled g0 + key far + still
2. score 0, no latches, run armed (arming-gate positive control)
3. randomization readback across 8 seeds (rig xy/yaw, g0, key xy/yaw; anchors at g0)
4. null policy: 240 steps → 0
5. seed strategy: lift-and-carry the key (incl. hover exactly above the slot, altitude
   verified) → nothing latches
6. turned key physically refused by the slot (upright root never below the roof plane)
7. mouth ram (3 N inward, the only push a mouth allows) slides the PAIR, breaks cover,
   kills the run, zero progress (displacement verified)
8. rearrange cheat: covered spread constructed AFTER a cover break → no split, no success
9. re-arm only from near-closed (readback), still score 0
10. key seated untwisted → exactly 0.25, no success
11. near-miss 70 mm spread → 0.25 + 0.45·prog, no split, no success
12. latched credit survives regressing the gap
13. settle gate: covered spread moving at 0.35 m/s refused (split latches, success False)
14. cover break after split: uncovered / re-closed states never succeed, 0.60 stays 0.60
15. audit: success() never True anywhere in the battery, max score ≤ 0.61
16. final no-NaN (+ frames.npz saved)
