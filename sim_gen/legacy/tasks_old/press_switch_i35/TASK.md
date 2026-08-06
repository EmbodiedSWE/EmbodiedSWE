# press_switch_i35 — bayonet_switch (seat a captive knob through an L-slot interlock)

**Registered as:** `SCENES["bayonet_switch"]`, env `sim_gen.bayonet_switch` (robot="null",
scene-level).
**Tier: easy — 2 stages** (press the knob the full depth of the press leg; sweep it along
the cross leg and release so the spring seats it in the locking notch).
**Execution order: REQUIRED — and enforced mechanically.** The cross leg is only reachable
at full press depth; the notch is only reachable from the far end of the cross leg. The
walls themselves impose the order; no rubric bookkeeping is needed to sequence the stages.

## Seed provenance

- Seed: `rlbench/press_switch`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/press_switch.py`) — a Franka reaches a
  wall-mounted articulated switch and flicks it with a fingertip: one momentary,
  monotonic, single-direction poke on the fixture's one moving part (USD assets + recorded
  trajectory, no checker).

## What changed, and why it is strategically different

The seed's plan skeleton is *approach → touch → push through a small arc → done*: any
transient contact in the right direction actuates the switch, and nothing about the
mechanism can reject a poke.

Here the switch is rebuilt as a **child-safety / bayonet interlock**, and the seed's plan
is precisely the thing the mechanism rejects:

1. **A straight press accomplishes nothing.** The knob is spring-returned along the press
   axis; pushing it the full depth of the press leg and letting go (even after a patient
   hold) springs it straight back to the OFF stop. This is smoke negative-control A — the
   seed's own strategy, expressed and tested, fails with the score pinned at the
   press-latch (~0.25).
2. **The solver must execute a compound, multi-directional stroke**: press to full depth,
   then sweep 90° sideways along a cross leg, then *release deliberately* so the spring
   itself pulls the knob into a locking notch cut back toward the OFF end. The final
   actuation is done by the mechanism — the solver's job is to put the knob where the
   spring's return path is captured by the notch shelf instead of the home leg.
3. **The goal state is held by geometry, not by code**: the seated knob rests against the
   notch end-stop, pressed there by the live return spring, with the notch shelf blocking
   the return path. success() is a settled, sustained, physical predicate (position in the
   notch + low speed + persistence); remove nothing and it persists forever, exactly like
   a real bayonet lock.
4. **Plan-affecting randomization**: the cross leg's handedness (left/right of the press
   axis) is sampled per episode, along with the press-leg length, console position and
   yaw — the stroke's shape, not just its coordinates, changes between episodes. The smoke
   oracle is verified on both handednesses.

A solver therefore needs a different *plan* (guided compound stroke through a shaped
channel, terminated by a deliberate release into a mechanical latch), not different
parameters of the seed's poke.

### Differentiation from sibling tasks (strategy-axis check)

- `push_button_i6` (pressure plate): weight-activated spring plate — no weights, no
  loading here; the spring here is the *adversary of the seed plan*, not a scale.
- `open_oven_i7` (oven_dials): detented rotary dials set to commanded settings — no
  rotation, no commanded set-point here.
- `push_buttons_i31` (circuit_bridge): elevated spanning/ordered construction — nothing is
  built or spanned here.
- `living_room_scene2_i14` (slot_deposit): a *payload* reoriented to pass through an
  aperture — here nothing is transported or inserted; a captive fixture element is guided
  along a constrained path (fixture actuation, not object transport).
- `libero_kitchen_scene1_i15` (drawer_fetch_restore): open-then-extract-then-restore on a
  container — no containment, no restore stage here.

Claimed axes: **bayonet/maze-slot actuation (compound multi-directional stroke through a
shaped guide)** + **spring-return makes naive actuation self-undoing (partial progress is
physically transient)** + **mechanically latched goal state (geometry holds it against the
live spring)**.

## Mechanism / physics honesty

- All procedural primitives: 8 kinematic wall boxes + base slab + 2 indicator tiles form
  the channel; each is re-posed per reset from the sampled parameters (fixed sizes,
  sliding centers, tails always outside the channel — the jointless-guide-wall pattern).
- The knob is a free dynamic **cylinder** (rotation-proof footprint — it cannot wedge in
  the channel the way a yawed box could). The return spring is a post_step external force
  along the console's canonical press axis (`f = k(home − x) − c·v`), plus a small lateral
  damper; low-friction materials guarantee released partial strokes glide back.
- The oracle drives the knob kinematically along the slot at riding height
  (gravity-compensated +g·dt hold), but success()/score() judge only the settled physical
  outcome; the final seating is performed by the spring, in free physics, after release.
- **Anti-cheat**: the knob is nominally captive (a real bayonet knob has a flange); the
  channel top is open only so the task reads on camera. A permanent `escaped` latch trips
  if the knob ever rises above the wall tops — lift-it-out-and-drop-it-in lands seated
  geometrically but is voided (success false, score capped at 0.30). Tested as negative
  control C.

## Rubric

- 0 for doing nothing (knob preloaded at the OFF stop).
- 0→0.25: latched deepest press-leg excursion (fraction of the press leg, 5% deadband).
- 0.25→0.55: latched farthest cross-leg sweep (geometrically requires full depth).
- 0.70: latched once the knob has entered the notch.
- 1.0 iff success(): knob settled in the notch, held by the live spring, sustained 30
  consecutive substeps, never escaped.
- Escaped episodes capped at 0.30.

## Smoke checks (15)

1. settle/no-NaN (knob preloaded at OFF home, score 0)
2. randomization is real (readback across seeded resets)
3. sampled press-leg length physically realized (independent wall readback)
4. both cross-leg handednesses occur across seeds
5. null policy scores ~0
6.–8. oracle × 3 seeds covering BOTH mirror handednesses (success + score 1.0)
9. negative A — the seed's strategy: straight press, held, released → never succeeds
10. spring return: the released knob glides back to the OFF home on its own
11. negative B — near-miss: full press + half sweep, released → pinned short of the
    notch, no success
12. negative C — lift-in cheat: knob dropped into the notch from the air → `escaped`
    voids it
13. rubric monotonicity: 0 < half press < full press < mid sweep < notch latch < 1.0
14. calibration sweep: releases well short of the shelf corner never seat
15. calibration sweep: releases at the far end always seat (measured corner-funnel
    boundary published)
