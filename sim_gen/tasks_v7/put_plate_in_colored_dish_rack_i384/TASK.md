# fold_rack — the drying rack starts FOLDED FLAT; erect the bistable fin comb to CREATE the color slots, then insert the plate into the BLUE gap

Package: `sim_gen/tasks_v7/put_plate_in_colored_dish_rack_i384`
Env: `simgen.fold_rack` (scene-level, `robot="null"`)

## Seed provenance

Seed task: `rlbench/put_plate_in_colored_dish_rack`. In the seed, the robot lifts THE
plate off its stand and lowers it into the OPEN slot of a static dish rack whose slots
are color-named — one prehensile transport into a fixed, pre-existing, sky-open
target, driven by recorded waypoints, with the color binding given for free by the
variation index.

## What changed, and why it is strategically different

1. **The target slot does not exist at reset — the rack itself must be
   reconfigured.** The rack is a fold-flat rack whose hinged FIN COMB starts folded
   DOWN over the deck like a closed lid: a full-width top cover plate roofs the fin
   gaps, so while folded there is no slot anywhere in the world. The seed's entire
   strategy — carry the plate to its color slot and lower it in — has no target to
   aim at, and smoke check 6 proves it physically: the plate dropped ON EDGE exactly
   where the blue slot WILL be lands on the closed lid and is denied. Two
   `__post_init__` asserts make the denial checkable geometry (the disc chord over
   the cover pushes every in-band on-edge stand out of the seat bands, front and
   rear).

2. **Phase 1 is mechanism actuation with a gravity-bistable hand-off, not a pick.**
   The comb is a real articulated body on a spawn-authored revolute hinge with
   deliberate over-center geometry (CoM crosses the hinge vertical at ~74°, computed
   from part volumes in `__post_init__`, authored into the spawner): gravity holds it
   CLOSED at the 0° stop, and once pushed past over-center gravity carries it the
   rest of the way and parks it OPEN at the 98° stop. Both end states are
   self-holding — the solver drives the red handle through ~84° of travel and then
   must LET GO and let gravity finish (smoke checks 8/9: released at 60° it falls
   back closed, at 85° it parks open). Erecting the comb stands the three fins up as
   posts and CREATES the two 34 mm color gaps.

3. **Phase 2 is an edgewise thread, twice pose-dependent.** The plate starts FLAT on
   a pedestal stand and must end ON EDGE between fin posts — a 90° reorientation the
   seed never does — threaded into a gap that passes ONE plate with 7 mm per-side
   slack. The chassis takes xy jitter + FULL ±180° yaw and the stand/plate jitter
   independently, so the handle, the hinge direction, and the blue-vs-yellow binding
   must all be read from the scene (fin colors + deck tabs) every episode.

Unlike sibling `put_plate_in_colored_dish_rack_i86` (a carousel mechanism that
ROTATES pre-existing slots, with a non-prehensile shuffleboard insert) and sibling
`put_plate_in_colored_dish_rack_i205` (no mechanism at all — an occupancy-swap puzzle
through a capacity-one buffer), i384's difficulty is CONSTRUCTING the receptacle:
drive a bistable fold-out mechanism through its over-center hand-off first, then
insert. No corpus task read this session shares this plan shape.

## Solution outline (teleports = free-air TRANSPORT only — verified on forge)

`solve.py` uses one teleport, to carry the plate already hanging clear in free air to
a hover directly over the blue gap (velocities zeroed); every load-bearing
interaction happens through contact dynamics:

- **P0** settle + readbacks; assert folded (hinge readback ~0°), score 0.
- **P1 DEPLOY** — hinge-axis torque plant on the comb (gravity feedforward
  mgd·cos(φ0+θ) + velocity servo, FD hinge rate, torque encoded body-frame along the
  comb-local hinge axis so the wrench-frame instability cannot tilt it); torque CUT
  at over-center + 10° (~84°) — gravity carries the comb the rest of the way and
  parks it on the 98° stop. Deploy latch (24 slow steps) fires. Score → 0.35.
- **P2 INSERT** — velocity-servoed vertical FORCE lift extracts the plate from its
  stand to free air (readback z ≥ 0.30, escape/regression monitor with per-body
  encoding self-correction); free-air teleport to a hover centered over the BLUE gap
  in the CHASSIS frame, on-edge seat quat; RELEASE — gravity threads the plate down
  the fin gap onto the deck (fins, spine and lip funnel it by contact). Seat latch
  fires. Score → 1.00 once stillness persists.
- **P3/P4** ring-down + hands-off ≥ 3.3 simulated seconds; assert success persists;
  print `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (latched credit).
**Verified on forge seeds 0 and 1 — both `SIM_GEN_SOLVE: SUCCESS`, scores monotone
0.00 → 0.35 → 1.00, ~19 s wall clock each, zero servo retries (deploy parked at
98.0° in mode 'enc'; the insert drop seated first attempt in both runs).**

## Embodiment argument (single Franka + parallel-jaw gripper)

Base at (0.42, 0.12, 0), between the rack (nominal (0, 0.06)) and the stand (nominal
(0, 0.38)): every task point is within 0.65 m (asserted ≤ 0.72) — inside Franka's
~0.85 m envelope at extreme jitter/yaw. **Deploy:** the red handle tab (12×14×36 mm)
stands ~36 mm proud of the folded cover — pinch it or push it with a closed gripper;
the drive torque is ≤ 0.15 N·m about the hinge ≈ 1.5 N at the handle's ~96 mm arm,
and the stroke is an ~84° arc of that radius, after which the arm simply lets go
(gravity finishes and holds — demonstrated by the torque-cut in the verified solve;
no holding force ever needed). **Plate pickup:** the plate lies flat with its rim
overhanging the pedestal all around (r 85 mm vs stand r 45 mm, asserted ≥ 25 mm
overhang); rim thickness 20 mm vs the 80 mm jaw span — a top rim pinch, then a wrist
rotation to vertical. **Insertion:** center the on-edge plate over the blue gap
(±7 mm lateral slack, generous ±50 mm along the slot) and lower until the rim enters,
then release — the verified solve's release-and-drop shows the fins funnel the last
~150 mm by contact alone, so gripper accuracy of a few mm suffices; the seated top
rim stands ≥ 47 mm proud of the fin posts (asserted), so the release pose is never
inside the rack. **Perception:** the fin colors and matching deck tabs name the
gaps; the handle is a saturated red tab on a wood-tone lid. Everything is always
upright-accessible from the sky.

## Execution order

The two phases are REQUIRED and their order is enforced by PHYSICS, not the rubric:
while the comb is folded its top cover roofs the gap volume and no in-band on-edge
stand exists anywhere (asserted in `__post_init__`, exercised by smoke check 6), and
the seat predicate is additionally gated on `deployed()` (smoke check 7 shows the
geometric bands alone would hold in a constructed pose — the gate is load-bearing).
Within that order: the handle may be pushed or pulled, approach direction is free,
and pausing between phases is harmless — gravity holds both comb end states. All of
this is stated in `describe()`.

## Rubric

- +0.35 once the comb has EVER been OPEN (hinge ≥ 92°) for 24 consecutive slow steps
  (it starts folded and gravity-held, so a null policy can never earn it), latched.
- +0.20 once the plate has EVER been seated in EITHER color gap for 24 consecutive
  slow steps (no gap exists at reset), latched.
- Partial credit capped at 0.55; `score = 1.0` iff `success()` (plate seated in the
  BLUE gap ∧ comb open ∧ persistently still, live). A seat = chassis-frame x within
  the blue band ∧ y inside the spine-to-lip roll range ∧ center z in the on-edge
  rest window ∧ plate ON EDGE (|axis·ẑ| ≤ 0.30) ∧ `deployed()`. `settled()` is a
  stillness counter-latch (30 consecutive steps under the velocity gates), so a
  flying/jolted arrangement judged mid-motion never counts. Null policy scores ~0.

`__post_init__` honesty asserts: the gap passes one plate with margin over both
contact offsets yet two plate thicknesses exceed it; the comb CoM is COMPUTED from
part volumes and the over-center angle sits well inside the travel with margin on
both sides; the swept spine corners clear the deck; the folded comb covers the
deposit area, nothing passes underneath, and (cover asserts) no in-band on-edge
stand exists front or rear of the folded lid; the erected posts are tall enough to
arrest tipping and reach low enough to catch the rim; the x band covers seated leans
while the two gap bands stay disjoint; the z window accepts max-lean and upright
rests but rejects folded-cover, fin-top and lip perches; the y band covers the full
spine-to-lip roll range; the seated top rim stands proud for a rim pinch; reset
jitter keeps the stand and rack collision-free.

## Checks (smoke: `SIM_GEN_SMOKE: ALL PASS 14/14`)

1. Reset settles: states finite, comb FOLDED by hinge readback, plate FLAT on the
   stand, everything settled.
2. Fresh reset: score ~0, no success.
3. Randomization: chassis yaw varies across 8 seeded resets (wide spread, both
   signs), live quat agrees with the cached sample, xy jitter real (readback).
4. Randomization: comb start angle varies inside its range (cache + live
   agreement); stand xy and plate on-stand xy + free yaw jitter are real.
5. Null policy 240 steps: score ~0, no success, comb stays closed, plate stays on
   its stand.
6. Order forced (seed-analog): the plate dropped ON EDGE exactly where the blue
   slot WILL be, while the comb is FOLDED, lands on the closed lid and is denied —
   no seat, no geometric seat band, comb still folded, score ~0.
7. Deployed gate: comb held at 85° (geometrically erect, below `deploy_min`) +
   plate written into a perfect blue seat pose — the geometric bands HOLD yet
   `seated()` is False: the gate is load-bearing, no credit.
8. Bistable closed side: comb released at 60° (below over-center) falls back
   CLOSED; no deploy credit.
9. Bistable open side: comb released at 85° (past over-center) gravity-parks OPEN
   at the stop → deploy latch fires EXACTLY +0.35, not success.
10. Yellow gap: plate gravity-seated in the YELLOW gap latches the seat credit →
    score capped at 0.55, NOT success (the goal is the BLUE gap).
11. Latched credit: removing the seated plate to the floor keeps the latched 0.55
    while the live seat drops — score never decreases.
12. Settle gate: plate written into a perfect BLUE seat with the comb open and
    judged immediately — seated LIVE but stillness has not persisted → NOT success;
    dismantled before the stillness latch can complete.
13. Rejection audit: `success()` never fired at any judged point.
14. Final state no-NaN.

Smoke also records `frames.npz` (600 × 960 × 3 frames) from a perspective camera.
