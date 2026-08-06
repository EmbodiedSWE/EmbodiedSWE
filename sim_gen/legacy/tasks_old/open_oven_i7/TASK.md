# oven_dials — set the oven's two control dials to their indicated settings

**Registered as:** `SCENES["oven_dials"]`, env `simgen.oven_dials` (robot="null", scene-level).
**Tier: easy — 2 stages** (one detented dial per stage, **execution order NOT required**).

## Seed provenance

- Seed: `rlbench/open_oven`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/open_oven.py`) — a Franka pulls the
  hinged oven door open by its handle (USD oven asset + recorded trajectory, binary
  fixed goal, no checker).

## What changed, and why it is strategically different

The seed's plan skeleton is: *approach the one handle → grasp → pull the hinged panel
through a large free arc until "open"*. A gross-motion, fixed-goal, binary pull on the
oven's single moving part.

This task keeps the oven's control fascia and **removes the door entirely** (the fascia
is kinematic — nothing on this oven opens). What remains is the oven's *other*
affordance: two spring-**detented control knobs** on revolute spindles. Each episode
samples, per knob, a random start setting and a random target setting (≥ 2 detents
apart); the target is shown physically by an amber lamp parked at that setting's tick
mark. The solver must:

1. **read the sampled goal from the scene** (which tick glows, per knob) — the seed has
   no goal-conditioning at all;
2. perform **precision rotary positioning**: turn each knob about its own axis and
   *stop* inside an 8° tolerance backed by a 20° detent capture basin — a
   stop-at-target skill, where the seed's pull runs to a hard limit;
3. do it **twice, on independently-goaled dials** (order free).

A different PLAN, not different parameters: no grasp-and-pull arc, no door, no binary
"open" predicate; instead goal reading, axis-aligned turning, tolerance stopping. The
seed's own strategy is **expressible and measured as negative control A**: a sustained
25 N outward pull plus a hinge-style torque on a knob (exactly the wrench that opens
the seed's door) moves nothing and scores ~0.

## Scene / physics honesty

Procedural only: kinematic fascia (Cuboid), knobs (Cylinder) on authored USD revolute
joints, symmetric grip bars (fixed joints; CoM on the spindle so there is no gravity
pendulum), visual tick/cap decorations, amber lamp bodies teleported to the target
tick each reset. Knob "feel" (viscous friction + nearest-setting detent spring,
overdamped) is applied in `post_step` — the combination_safe pattern. The oracle
teleports knob+bar as one rigid assembly, but every judgment is physical: `at_target`
requires the settled pointer inside tolerance under the live detent plant, and the
calibration probe measures the real capture basin (≤ 15° snaps in, ≥ 25° falls to the
neighbour detent).

## Rubric (graded, 0..1)

- per knob: latched best progress `1 − err/err0` (forced to 1.0 once the pointer ever
  enters the target zone) × 0.35, plus 0.15 if *currently* resting on target;
- `score == 1.0` **iff** `success()` (both dials settled on their targets);
- ~0 for doing nothing (start jitter ±4° vs authored `err0 ≥ 80°`);
- transient achievements latch: a dial knocked off after success leaves 0.85.

## Smoke check list (15)

1. settle: clean reset (finite, pointers on start settings, score ~0)
2. randomization is real (start/target draws differ across seeds, by readback)
3. goal indicator: lamps sit at the sampled target marks (readback)
4. null policy: score < 0.05 and no success
5. rubric monotonicity: latched score never decreases along the approach
6. partial credit: knob 0 on target → score ~0.5, no success
7. success: both dials set → success() and score == 1.0
8. achievement latch: knock-off revokes success, score 0.85
9. oracle reaches success on 3/3 fresh seeds
10. negative A (seed strategy): pulling/prying the "door" moves nothing, score ~0
11. negative B (near miss): adjacent setting rejected by the stop tolerance
12. negative C (swap): dials on each other's targets rejected
13. calibration: offsets ≤ 15° captured back to the target detent
14. calibration: offsets ≥ 25° fall to the neighbour detent
15. video frames recorded (frames.npz in CWD)
