# track_knife_i42 — Butter Station (charge the knife, frost the tart, serve, tidy up)

**Env name:** `simgen.butter_station` (scene `butter_station`, robot `null`)
**Tier: hard — ~8 stage-instances across 4 distinct stage types**, with a REQUIRED
partial execution order (see below).

## Seed provenance

- Seed: `pick_place/track_knife`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_knife.py`)
- Seed semantics: stage-3 **trajectory tracking**. The knife starts *already grasped*
  (initial states loaded from a pkl of grasp states), five XFORM markers prescribe the
  exact free-space path, randomization is explicitly zeroed, the gripper is forced
  closed every step, and the episode *terminates if the knife leaves the gripper*.
  Reward = waypoint approach/progress + rotation tracking. The whole skill is carrying
  a held object along a given aerial curve, touching nothing.

## What changed, and why it is strategically different

Every load-bearing pillar of the seed is inverted; a solver needs a different **plan**,
not different numbers:

1. **The knife stops being the payload and becomes a finite-capacity TOOL.** Nothing
   starts in hand: the knife lies in a cradle. The goal is not to move the knife
   somewhere — at the end it must be back where it started (net tool transport = zero).
2. **Free-space transit is replaced by contact-band surface COVERAGE.** The tart's top
   is a 4x3 cell grid; a cell frosts only while the blade tip sweeps a *slow,
   continuous, blade-flat stroke* within a [-3, +5] mm band of the top face — judged in
   the **tart's body frame**. There is no prescribed path and no markers: any stroke
   pattern that covers the cells works, so nothing can be memorized as a trajectory.
3. **A consumable resource forces recharge/apply ALTERNATION.** One dip of the blade
   into the butter dish (a real dwell inside the aperture, below the rim) loads butter
   for exactly `capacity = 4` cells; at zero charge, strokes accrue nothing. 12 cells
   force **at least 3 dip trips interleaved with strokes** — a scheduling structure the
   seed has no concept of. The blade recolors yellow while charged; frosted cells
   recolor cream (state is observable, not hidden).
4. **Randomization is maximal where the seed's was explicitly zeroed:** dish / plate /
   cradle bearings, cradle yaw, and jig position + yaw (the cell grid rides the tart's
   frame) are re-sampled per episode.
5. **The one transport that remains (tart -> plate) is a gated FINAL stage**, worthless
   without full coverage — so the seed's plan cannot even earn the transport credit.

The seed's own strategy is a tested failing control: **carrying the knife along a
smooth aerial waypoint arc to a goal station and setting it down scores ~0** (negative
A), and transporting the *payload* instead (tart physically served on the plate,
settled, upright) also scores ~0 because serve credit is gated on full frosting
(negative A2).

Differentiation from sibling generated tasks (checked against their TASK.md claims):
no sibling claims consumable-budget recharge/alternation or tool-tip surface-coverage
axes. Nearest neighbours: `draw_triangle_i8`'s *seed* traces a prescribed outline (its
generated task moved to rigid-part construction) — here nothing prescribes a path and
coverage is budget-constrained, area-based, and judged in a movable workpiece's frame;
`track_screwdriver_i12` (same seed family) went to size-keyed hanging, untouched here.

## Stages (declared: hard, ~8 stage-instances, 4 types)

1. dip #1 (charge the knife in the dish — dwell, below the rim)
2. stroke row 1 (4 cells, charge exhausted)
3. dip #2
4. stroke row 2
5. dip #3
6. stroke row 3 (full coverage latched)
7. serve: lift the tart out of the jig, set it settled + upright on the plate
8. park: lay the knife back in its cradle channel, settled

**Execution order: partially REQUIRED.** Each stroke block strictly requires a
preceding dip (physically enforced by the charge counter — tested: an uncharged stroke
frosts nothing). Full coverage strictly precedes *credited* serving (rubric gate,
tested), and park credit is gated on served (tested: parking early holds the score at
0.70). Row order and dip timing beyond the alternation are free.

## Rubric (score in [0, 1])

- +0.10 `loaded_ever` latch (first successful dip)
- +0.50 x frosted fraction (12 latched cells, ~0.042/cell)
- +0.10 all cells frosted
- +0.15 tart served (settled + upright on the plate) — **gated on full frosting**
- +0.15 knife parked in the cradle — **gated on served + frosted**
- exactly 1.0 iff `success()` = all frosted & tart settled on plate & knife parked.
- 0 for doing nothing (the knife *starts* parked; that credit is gated — tested by the
  null-policy check).

Anti-teleport: dip dwell and frosting accrue only in an **entry-latched continuous
stroke** — the working zone must be *entered* with per-substep blade-tip travel below
6 mm; a knife teleported into a perfect in-band pose (or into the dish) never accrues
until it leaves and re-enters continuously (tested). Physical outcomes (tart on plate,
knife in cradle) are judged on settled poses under real gravity/contact.

## Smoke checks (23 — forge result: `SIM_GEN_SMOKE: ALL PASS 23/23`)

1. settle/no-NaN: reset layout settles finite; knife lies in its cradle; tart still
2. reset score ~0, no success (the knife STARTS parked — that credit is gated)
3. randomization-is-real (READBACK: dish/plate xy, cradle yaw, tart yaw all move)
4. null-policy fails (240 idle steps -> score ~0)
5-7. oracle reaches success() + score 1.0 on seeds 0/1/2
8. milestone: first dip -> charge=capacity, score ~0.10
9. milestone: row 1 frosts 4 cells and EXHAUSTS the charge, score ~0.267
10. capacity: an uncharged stroke frosts nothing until a re-dip
11. milestone: re-dip unlocks row 2 (8 cells, score ~0.433)
12. milestone: full coverage -> score ~0.70
13. order gate: parking BEFORE serving stays ~0.70
14. milestone: serving the frosted tart -> success, exactly 1.0
15. monotonicity: milestone ladder [0.10, 0.267, 0.433, 0.70, 1.0] strictly increases
16. negative A (seed strategy): aerial waypoint-carry of the knife to the plate ~0
17. negative A2: unfrosted tart physically served on the plate ~0 (gate proof)
18. negative B: never-dipped strokes frost nothing
19. negative D: teleport-hopping a charged knife across in-band poses frosts nothing
20. negative D recovery: the same row then frosts via an honest continuous stroke
21. calibration: in-band stroke heights (+2/+4 mm) frost the full row
22. near-miss: out-of-band heights (+8/+12 mm) frost nothing (measured cliff matches
    the [-3, +5] mm band with the ~0.7 mm one-substep gravity sag, as dry-computed)
23. flat gate: a 45-deg (nose-down pitch) tilted in-band stroke frosts nothing

Assets: fully procedural (compound spawners; primitives only). Oracle: continuous
kinematic dragging through scene handles (NullRobot); all judged outcomes settled
physics. Video frames recorded via the RTX annotator -> `frames.npz` in CWD.
