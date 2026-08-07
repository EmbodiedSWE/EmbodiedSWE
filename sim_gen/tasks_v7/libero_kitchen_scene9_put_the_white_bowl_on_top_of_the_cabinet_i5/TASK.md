# counterweight_shelf — level the tipping shelf, THEN put the white bowl on top

**Env name:** `simgen.counterweight_shelf` (scene `counterweight_shelf`, robot `null`;
both `solve.py` and `smoke.py` build the same scene-level env).

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
   success state (success persistence is checked over 240 extra steps in smoke and a
   3.5 s hands-off window in solve).

The seed judges *where the bowl is* on an always-ready surface; this task judges the
same final geometry on a surface that must first be made to exist — a two-object,
mechanism-first plan (counterweight → seat → then place), strategically disjoint from
the seed's single pick-and-place and from its parameter variations.

Claimed axes (vs. sibling tasks from this seed family): mechanism-configuration before
placement / counterweight-torque gravity latch. (The i4 sibling from the same LIBERO
scene is a slide-under-a-low-shelf extraction; an older i5 variant `empty_the_bowl`
judged the bowl's CONTENTS via pouring; nothing here involves contents, pouring, or
sliding under an overhang.)

## Difficulty / stages / order

- **Tier: medium — 2 manipulation stages + 1 mechanism wait**: deliver the cube into
  the tipped rear socket (the socket rides on the 38°-tilted tray, so the drop point
  must be computed from the LIVE tray pose); wait out the damped swing to level;
  place the bowl upright on the front platform band.
- **Execution order:** REQUIRED, physics-enforced (counterweight seated before the bowl
  can rest). Not code-enforced; a bowl placed early merely ends up on the floor and can
  be re-picked — the order gate is a state requirement, not an irreversible hazard.

## Teleport-solution outline (solve.py — the task's legitimacy certificate)

Scene-level env, robot `null`. Teleports do TRANSPORT only; both release points are
deliberately chosen OUTSIDE every scoring band (asserted in code), so no credit can
latch at a teleport instant — every latch fires only after contact dynamics:

- **PHASE 1** — teleport the counterweight cube from the floor to ~12 mm above the RIM
  of the tipped rear socket (tray-local z ≈ 0.063 > seated-band z_hi 0.055; pose
  computed from the live tray quaternion, tilt-matched, zero velocity). It falls in,
  seats against the downhill socket wall by contact, and its rear torque swings the
  shelf up against the level stop (ang-damped, <1 s). `SIM_GEN_SCORE 0.000 → 0.300`.
- **PHASE 2** — teleport the bowl from the floor to ~38 mm above the now-level front
  platform (tray-local z ≈ 0.063 > placed-band z_hi 0.060, identity attitude, zero
  velocity). It free-falls the last gap, lands on its flat bottom, and RESTS by
  contact; `success()` additionally requires the shelf level and the whole scene
  settled. `SIM_GEN_SCORE → 1.000`.
- **PHASE 3** — hands off for 420 physics steps (3.5 simulated seconds); `success()`
  must still hold (rejects fly-through / precarious states) before
  `SIM_GEN_SOLVE: SUCCESS` is printed. Watchdog Timer + `os._exit` guard teardown.

Nothing is pinned, velocity-clamped, or spawned inside a goal region; the printed
`SIM_GEN_SCORE` sequence is monotone (asserted in code). Verified on the forge on
seeds 0 and 1.

## Embodiment argument (single Franka + parallel jaw, OSC)

This exact plan was previously EXECUTED by a real Franka arm in this scene (an earlier
arm-driven run of the same task package): base at `(-0.25, 0.0, 0.0)` facing +x, OSC
(kp 220/600), SUCCESS on 2 seeds end-to-end, ~55 s per run. Per-object contact
strategy:

- **Counterweight cube (0.62 kg, 45 mm)**: top-down pinch across two opposite faces —
  fits the 80 mm jaw with room; carried at grip 18 mm / effort 120 N (0.62 kg is inside
  the measured 0.70 kg slip ceiling); released ~10 mm above the projected socket-rim
  midpoint computed from the live tray pose. The socket inner opening (75×70 mm) vs the
  45 mm cube leaves ±15 mm of drop tolerance — far above OSC noise.
- **White bowl (0.26 kg, 114 mm across, 12 mm-thick rim)**: cage-then-squeeze rim
  pinch — fingers pre-closed to 26 mm straddle the 12 mm wall, slow squeeze to 8 mm;
  the 90 mm-diameter inner opening admits the finger with clearance. Placement band on
  the platform is 80 mm long × 100 mm wide for a 114 mm bowl root tolerance of ±40 mm
  laterally — comfortably above control noise.
- **Reach**: pedestal front face at x = 0.30, socket at x ≈ 0.405, all contact heights
  0.02–0.31 m — inside the Franka 0.45–0.71 m comfortable envelope from the stated
  base pose.
- No contact is required near the floor under an overhang, through an aperture, or at
  sub-centimeter tolerance.

## Rubric (score in [0,1], latched via post_step)

- 0.20 — counterweight ever seated in the socket (geometric, tray frame);
- 0.10 — shelf ever level WITH the counterweight seated (the honest level latch);
- 0.15 — bowl ever lifted above 0.12 m;
- 0.45 — bowl ever resting upright on the platform band of the level shelf (tray frame);
- 1.0 iff `success()`: bowl upright inside the platform band, shelf level, everything
  settled. Pre-success total capped at 0.95.
- Null policy ≈ 0. Seed strategy ≤ 0.15. Latched credit never evaporates (the
  `SIM_GEN_SCORE` prints along the solve trajectory are monotone:
  0.000 → 0.300 → 1.000, asserted in code).

## Check list (smoke.py — 16 checks; teleported probes are instrumentation only, not a
solution)

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
Video: smoke saves `frames.npz` in the current working directory.
