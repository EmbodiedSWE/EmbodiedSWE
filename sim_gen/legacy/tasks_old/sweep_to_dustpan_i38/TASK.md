# sowing_tray — de-aggregate a hopper of pellets, exactly one per planter cell

| | |
|---|---|
| Task id | `sweep_to_dustpan_i38` (env `simgen.sowing_tray`) |
| Seed | `rlbench/sweep_to_dustpan` (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/sweep_to_dustpan.py`) |
| Tier | **hard** — long horizon: 4–6 repeated extract→carry→precision-drop cycles (≈ 8–12 primitive stages), 18 mm per-cell funnel, count-keeping across the episode |
| Execution order | **free** — cells may be sown in any order; the only ordering structure is that each cycle needs an *empty* cell (over-filled cells are voided until fixed) |
| Robot | `null` (scene-level task; robot bindings are a later stage) |

## The seed, and what changed

The seed is **tool-mediated aggregation**: five loose dirt cubes are swept with a broom
across the table plane into one dustpan. Its plan has three load-bearing pillars:
(1) the material is an *undifferentiated pile* — nothing is counted, any dirt anywhere in
the pan counts; (2) transport is *many-at-once* — one broom stroke moves the whole pile;
(3) the transport channel is the *plane itself* — everything slides at table level into a
container whose mouth opens onto that plane.

This task keeps the seed's vocabulary (a scatter of small granular objects, a container,
a tabletop) and inverts all three pillars:

- The pellets start **already aggregated** — clumped inside a deep open hopper cup. The
  dustpan-analog is the *source*, not the goal (container role inversion).
- The goal is **dispersal with exact counting**: a raised planting tray (3×2 grid of
  64 mm walled cells) must end with **exactly one pellet resting on each used cell's
  floor** — every present pellet planted, no cell double-seeded. A cell holding ≥ 2
  pellets is *voided* by the rubric until the extras are removed.
- The transport channel is **vertical**: the bed floor is 55 mm above the ground and
  ringed by walls, so nothing at ground level is inside a cell, and the hopper is deep
  enough that pellets must be lifted out over its rim.

## Why strategically different (a different PLAN, not different numbers)

- **Aggregation → de-aggregation.** The seed's entire skill — gather everything into one
  place — scores ~0 here and is a tested failing control (negative A: the whole clump
  dumped into one cell is over-occupied and voided). A solver must instead *separate* the
  clump: one pellet at a time, distinct destination each time, keeping count of which
  cells are used. Bulk moves are not merely inefficient; exact-occupancy judging makes
  them worthless.
- **The seed's transport channel is physically disconnected from the goal.** Sweeping /
  planar pushing — the seed's one motor primitive — can bring pellets flush against the
  planter's base and still score exactly 0 (negative B): the raised walled bed is
  unreachable from the plane. Every scoring trajectory requires repeated prehensile
  lift-out (a deep 55 mm-radius cup), airborne carry, and a precision release into an
  18 mm radial funnel.
- **One indiscriminate stage → 4–6 counted stages.** The seed is a single sweep with a
  binary outcome; here the horizon is 4–6 extract→carry→drop cycles whose intermediate
  states matter (a mis-drop into an occupied cell *subtracts* the credit that cell was
  worth until repaired), with the pellet count and the tray's full-yaw cell frame
  re-sampled every episode.

Sibling differentiation (checked against all 50 in-flight/complete sibling TASK.md files
and the claimed-axes memory): `registry_i28` is occupied-goal *permutation with a forced
buffer* (rearranging distinct identities among one-slot cups); this task has no
identities, no occupied homes and no buffer — its axis is *even dispersal of
indistinguishable material with exact-occupancy counting*, which no sibling claims.
`slot_deposit_i14` (orientation-keyed aperture), `empty_the_bowl_i5` (pour contents out),
and `vault_unstack_i17` (demolition-for-access) share containers but not the plan.

## Scene & rubric

Fully procedural: a dynamic octagonal hopper cup (compound spawner), a **kinematic**
raised tray (bed slab + grid walls, one body, re-posed with xy jitter + full yaw per
reset), six dynamic 28 mm sphere pellets (count 4–6 sampled per episode; absent pellets
parked off-camera and masked).

- `good_cells()`: a cell counts iff **exactly one** present pellet occupies its airspace
  and that pellet rests settled on the cell floor. Honest by construction: the per-axis
  28 mm gate exceeds the max physical on-floor offset (18 mm) but rejects wall-perches
  (36 mm); the z gate rejects perches and pellets riding on top of others.
- `success()`: number of good cells == number of present pellets (cells are disjoint, so
  this holds iff every pellet is the sole floor occupant of its own cell).
- `score()`: 0 for doing nothing; +0.10 latched once any pellet is genuinely raised above
  the hopper's rim height (extraction, latched in `post_step`); +0.70 × fraction of
  pellets planted-alone; exactly 1.0 iff success. Max non-success ≈ 0.68.

## Smoke battery (20 checks)

1–2 reset sanity (finite, clump genuinely inside the hopper by readback, depot parking,
score 0) · 3–4 randomization by readback (count, full tray yaw, tray + hopper jitter) ·
5 null-policy fails · 6–8 teleport-oracle success ×3 seeds (real 38 mm drops, score 1.0)
· 9–11 rubric ladder (extraction 0.10 → +0.70/k per planted pellet, strictly increasing,
< 1.0 until the last → 1.0) · 12 negative A: the seed's aggregation plan (dump the clump
into one cell) → voided, ~0 · 13 negative B: the seed's planar-sweep channel (pellets
flush against the planter base) → exactly 0, no latch · 14–15 near-miss tolerance
controls (wall-perch counts nowhere; in-airspace-but-off-floor occupies-yet-earns-nothing,
then becomes good after its real 26 mm fall) · 16–17 exact-occupancy void + recovery
(two-in-one-cell voided; splitting them makes both cells good) · 18 reset clears latches
· 19–20 calibration drop sweep (0/6/10 mm plant 3/3; 45 mm — past the wall — never ends
in the target cell; 14–32 mm published).

Physics honesty: the oracle only teleports pellets *through the air* and releases them
6 mm above the rim — capture, settling, piling, voiding and every judged predicate are
real physics on settled poses.
