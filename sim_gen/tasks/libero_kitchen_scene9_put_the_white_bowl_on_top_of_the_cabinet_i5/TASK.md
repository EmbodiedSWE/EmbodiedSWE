# counterweight_shelf — level the tipping shelf, THEN put the white bowl on top

**Env name:** `simgen.counterweight_shelf` (scene `counterweight_shelf`, robot `null`;
`solve.py` builds `scene="counterweight_shelf", robot="franka", control_mode="osc"`).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet.py`)
- Seed plan: one pick-and-place — grasp the white bowl, set it down inside a bbox on the
  STATIC top of a cabinet/shelf. The support surface is passive and always ready.

## What changed, and why it is strategically different

Kept: the white bowl as the manipulated object, an elevated "cabinet top" as the goal
surface, a single-arm kitchen-scale layout.

Changed (the plan, not the numbers):

1. **The goal surface is a mechanism, not a fixture.** The pedestal's top is a TIPPING
   SHELF: a tray on a revolute hinge along the pedestal's front top edge, front-heavy by
   construction, resting tipped 38° like a ramp against its lower joint stop.
2. **The seed's plan is an executable failing control.** Grasping the bowl and setting
   it down on top — executed perfectly — dumps the bowl on the floor: it slides off the
   ramp (smoke negative A, measured). A solver that treats the top as a static surface
   scores only the 0.15 lift latch.
3. **The real task is to CONFIGURE the mechanism first.** The shelf carries a walled
   socket BEHIND its hinge (lever 0.105 m); the 0.62 kg counterweight cube seated there
   swings the shelf up against its level stop and pins it — a binary gravity latch, no
   fine balancing. Measured torque margin: rear ≥ 0.0558 kg·m (block against the
   downhill socket wall) vs ≤ 0.0377 kg·m front (bowl at the worst-case platform-band
   edge + tray's own front-heaviness) ≈ 1.48×; the shelf sits at 0.0±0.1° in every
   passing run. Only after seating does the seed's move work.
4. **The ordering is enforced by physics, not by code.** No latch forbids bowl-first;
   the tipped shelf simply cannot hold the bowl, and nothing else in the scene can level
   it: the bowl itself set over the socket slides off the tilted rim (smoke negative B),
   and an arm pressing the shelf level cannot satisfy the settled, self-supporting
   success state (success persistence is checked over 240 extra steps).

The seed judges *where the bowl is* on an always-ready surface; this task judges the
same final geometry on a surface that must first be made to exist — a two-object,
mechanism-first plan (counterweight → seat → then place), strategically disjoint from
the seed's single pick-and-place and from its parameter variations.

Claimed axes (vs. sibling tasks from this seed family): mechanism-configuration before
placement / counterweight-torque gravity latch. (The older i5 variant `empty_the_bowl`
judged the bowl's CONTENTS via pouring; nothing here involves contents or pouring.)

## Difficulty / stages / order

- **Tier: medium — 2 manipulation stages + 1 mechanism wait**: pick+drop the cube into
  the tipped socket (the socket rides on the 38°-tilted tray, so the drop point is the
  midpoint of the projected inner-rim opening computed from the LIVE tray pose); wait
  out the damped swing to level; rim-pinch the bowl and place it on the platform band.
- **Execution order:** REQUIRED, physics-enforced (counterweight seated before the bowl
  can rest). Not code-enforced; a bowl placed early merely ends up on the floor and can
  be re-picked — the order gate is a state requirement, not an irreversible hazard.

## Solution outline (solve.py — the feasibility certificate)

- Franka base pose: `(-0.25, 0.0, 0.0)`, identity rotation (facing +x), OSC with the
  known-good gains (kp 220/600, rot_scale 0.15, `nullspace_dof_pos=()`), gripper
  effort 120 N / stiffness 4000.
- PHASE 1: top-down pinch of the 45 mm cube (ramped close to 36 mm; lift verdict), carry
  it over the live-computed socket drop point WITHOUT re-orienting the wrist (retargeting
  the jaw azimuth or stepping the grip command mid-carry ejects the cube — both
  measured), low release; the shelf swings level in <1 s (`SIM_GEN_SCORE` 0.00 → 0.30).
- PHASE 2: cage-then-squeeze rim pinch of the bowl (fingers pre-closed to 26 mm around
  the 12 mm rim wall of the tracked octagon segment, slow squeeze to 8 mm; empty-close
  width check), slow straight lift (0.45 after the lift latch), gentle closed-loop carry
  on the bowl center (8 mm correction steps, xy_boost 1.2, drop abort), lower to 4 mm
  above the platform, slow release, retreat, settle → `success()`
  (`SIM_GEN_SCORE` 1.0, `SIM_GEN_SOLVE: SUCCESS`).
- Arm-only: task-object state is never written; no external forces. Verified on the
  forge: `--seed 0` (default invocation, 53 s) and `--seed 1` (59 s, including a clean
  retry after one missed block grasp).

## Rubric (score in [0,1], latched via post_step)

- 0.20 — counterweight ever seated in the socket (geometric, tray frame);
- 0.10 — shelf ever level WITH the counterweight seated (the honest level latch);
- 0.15 — bowl ever lifted above 0.12 m;
- 0.45 — bowl ever resting upright on the platform band of the level shelf (tray frame);
- 1.0 iff `success()`: bowl upright inside the platform band, shelf level, everything
  settled. Pre-success total capped at 0.95.
- Null policy ≈ 0. Seed strategy ≤ 0.15. Latched credit never evaporates (the
  `SIM_GEN_SCORE` prints along the real solve trajectory are monotone:
  0.00 → 0.30 → 0.45 → 1.00).

## Check list (smoke.py — 16 checks, ALL PASS on the forge; teleported probes are
instrumentation only, not a solution)

1-2. settle/no-NaN: reset settles finite, tray on its tipped stop, score ~0, no latches.
3. randomization by READBACK across 6 seeds: bowl/block xy + yaw move, the flank sides
   swap, tray always starts tipped.
4. null policy: 240 idle steps → score ~0, no success.
5-7. oracle ×3 seeds: seat cube → shelf levels; bowl on platform → success, score 1.0,
   persists 240 further steps (no flicker).
8-9. monotonicity ladder: 0 < 0.30 (seated+level) < 0.45 (+lift) < 1.0 (success);
   partials < 1.0.
10. negative A — the SEED's strategy: bowl set on the tipped shelf top → slides to the
   floor, shelf stays tipped, no success, score ≤ 0.15.
11. negative B — wrong place: bowl over the rear socket → no success (slides off the
   tilted rim; cannot level the shelf).
12. negative C — wrong object placement: cube on the front platform → slides off, no
   seat/level credit, score ~0.
13. negative D — near-miss y: bowl resting 25 mm outside the lateral tolerance on the
   level shelf → no success.
14. negative E — near-miss x: bowl resting over the hinge, outside the platform band →
   no success.
15. negative F — bowl upside-down on the level platform → no success.
16. calibration: bowl rest-x sweep (−70/−100 in-band placed; −135 past the front band
   and −5 over the hinge rejected).

Out-of-order end state: N/A as a *settled end state* (physics erases it — the bowl
cannot remain on the tipped shelf; negative A is exactly the bowl-first attempt).
