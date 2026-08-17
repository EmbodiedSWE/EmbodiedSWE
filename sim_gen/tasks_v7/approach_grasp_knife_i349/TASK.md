# timer_flip_dock (approach_grasp_knife_i349)

Flip a marble-bearing two-chamber TIMER capsule fully upside-down and dock it foot-first
into a snug keyed well until its collar flange seats on the rim — with the captive marble
fallen through the internal waist as the physical proof of the flip.

## Seed provenance

- Seed: `pick_place/approach_grasp_knife` — servo the gripper to a knife on a table,
  close the jaw, lift; success is a few frames of stable grasp. The carried object's
  pose never matters, nothing is ever placed, and there is no mechanism.

## What changed (and why it is strategically different)

The seed's entire strategy — approach, grasp, lift — is worth 0.10 here and nothing
more. The task is built so that everything the rubric actually pays for is what the
seed never asks:

1. **Reorientation IS the task.** The capsule spawns standing collar-end DOWN and only
   docks fully INVERTED (up-axis flipped past 150 deg). Carrying it upright — the
   seed's whole plan — parks the run at 0.10 forever (smoke check 5 pins the upright
   carry directly over the well and asserts exactly 0.10).
2. **A one-way key by construction.** The 62 mm collar cannot enter the 50 mm well
   (wrong-end insertion perches 60 mm too high on the rim — smoke check 6), and the
   40 mm square tube cannot enter turned 45 deg (57 mm diagonal jams on the mouth —
   smoke check 7). Inserted foot-first the collar is the seating STOP: full depth is a
   hard, repeatable pose (the foot cap hovers 10 mm above the well floor).
3. **A captive internal payload proves the flip through physics.** A 12 mm marble is
   sealed in the collar-end chamber; only a real inversion (while the capsule is aloft
   or docked upright-down) makes it fall through the 22 mm waist between the internal
   ledges into the foot-end chamber. success() checks the marble LIVE in the foot
   chamber: teleport-faking the capsule pose without an honest flip + settle leaves the
   marble on the wrong side (smoke check 9: the no-marble capsule physically seats yet
   caps at 0.70).
4. **No pose-goal overlap with siblings.** i292 (keyturn_spreader) is
   tool-through-slot + cam twist + anchor spread; the seed is grasp-and-hold. Neither
   involves reorientation of the goal object, a keyed insertion, or a captive payload.

## Execution order (declared)

FLIP strictly BEFORE INSERT. The well only admits the inverted capsule (the collar
blocks the wrong end; the diagonal blocks unaligned entry), and the marble can only
cross the waist along the tube axis — i.e. while the capsule is near-vertical, which an
in-well capsule already is only AFTER an aloft flip. The rubric's insertion latches
(`tip_in`, `depth_max`) gate on the up-axis being flipped past 120 deg, so no insertion
credit exists before the flip.

## Rubric

`score() = 0.10 lifted + 0.15 inverted-aloft + 0.15 foot-tip-in-well + 0.30 * latched
max insertion depth (rim -> seat, gated on tip-in-well while flipped) + 0.15
marble-crossed`; exactly 1.0 iff `success()` = live seated pose (up-axis <= -0.90,
centred within 10 mm, collar on the rim within 8 mm) AND marble live in the foot-chamber
band AND capsule + marble settled. Latched credit never evaporates; null policy = 0;
max non-success total = 0.85.

Geometry honesty is asserted in `__post_init__` (18 asserts): jaw fit; collar > well;
well > tube > well/√2; collar stops before the foot bottoms out; perch and seat 60 mm
apart (> 4x seat_tol); waist passes the marble with 10 mm margin while the ledges can
never shelf it; the collar ring never intrudes into the marble's path; the marble bands
bracket the physical rest poses; center_tol covers the well play; a >= 40 mm bare-tube
band stands proud of the seated rim; the spawn ring never overlaps the dock.

## Randomization (per episode, verified by readback in smoke)

Dock centre xy +/- 30 mm and yaw +/- 25 deg; capsule on a free-azimuth ring of radius
160–240 mm around the well with free yaw; marble xy jitter inside its chamber.

## Solution outline (solve.py)

- P0 settle + dock readback (score 0.000).
- P1 ONE transport hop: capsule to an inverted hover 15 mm above the rim, faces aligned
  to the well mod 90 deg; the marble is re-written at its unchanged body-frame offset in
  the same write (the hop moves the assembly coherently, it does not move the marble
  relative to the capsule). A gravity-feedforward hover servo (z position PD + world xy
  PD + flipped-upright PD torque + mod-90 yaw hold) holds the capsule while GRAVITY
  drops the marble ~125 mm through the waist into the foot chamber (score 0.400).
- P2 wrench-servo descent (vertical velocity servo + xy PD + attitude hold) lowers the
  foot into the well until the collar lands on the rim — the geometric stop — plus a
  brief gentle press to snug it; stall-detect + lift-and-retry up to 4 attempts
  (score 0.850).
- P3 wrenches cut; the capsule rests collar-on-rim; success() goes live (score 1.000).
- P4 >= 3.5 s hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.

Teleport = TRANSPORT ONLY (the single P1 hop through free air); every load-bearing
interaction (marble crossing, insertion, seating) is contact dynamics under applied
wrenches or gravity. Verified on the forge: seeds 0 and 1 both SUCCESS, scores
0.000 -> 0.400 -> 1.000, single-attempt descents, distinct randomization readbacks.

## Embodiment argument (Franka, parallel jaw)

- **Grasp**: the capsule is a 40 mm square tube — comfortably inside the ~80 mm jaw
  span. Standing, a ~100 mm band of bare tube sits ABOVE the collar flange (collar top
  at 40 mm from the ground, tube top at 152 mm), so a top-down or side grasp on the
  upper tube is unobstructed. Seated, >= 40 mm of bare tube still stands proud of the
  rim for the release/regrasp.
- **Flip**: a wrist roll (joint 7 or a 180 deg wrist reorientation) with the capsule in
  the jaw performs the inversion in free air; the marble transfer needs only that the
  capsule be held near-vertical for ~1 s, which a stationary inverted hold provides.
- **Insertion**: 5 mm per-side lateral clearance (50 mm well vs 40 mm tube) and mod-90
  yaw symmetry are far above Franka repeatability (~0.1 mm); the collar stop makes the
  final seat compliant — press down until contact, no precision force control needed.
- **Base pose**: base at the origin; the dock at (0.42, 0.0) +/- 30 mm and the spawn
  ring at <= 0.69 m worst-case are inside the ~0.85 m Franka workspace with the
  standard tabletop mount; all manipulation heights (0–210 mm) are in the dexterous
  zone.
- Tolerances: seat 8 mm, centring 10 mm, marble band 14 mm — all far above control
  noise.

## Files

- `scene.py` — cfg (tunables + info + 18 honesty asserts), custom compound spawner
  (capsule: 2 caps + 4 walls + 2 waist ledges + 4-box collar RING — the ring never
  seals the interior), 9 kinematic dock pieces, latches, success/score, register_env.
- `solve.py` — transport hop + wrench-servo dock, phase scores, watchdog, verdict.
- `smoke.py` — 14-check rejection battery (below), frames.npz, watchdog, verdict.

## Smoke checks (14)

1. Reset settles finite: capsule standing on the sampled ring (radius readback), marble
   at its chamber home, still.
2. Reset: score 0, no latches, no success.
3. Randomization readback across 8 seeds (dock xy/yaw, ring radius, capsule world xy,
   spawn yaw all vary; physical placement verified every seed).
4. Null policy 240 steps -> score 0.
5. Seed strategy (upright grasp-lift-carry, incl. hovering exactly over the well,
   altitude verified) -> only `lifted`, score exactly 0.10.
6. Wrong-end drop: collar catches on the rim, perches at perch_z (readback), insertion
   latches never fire -> 0.10.
7. 45-deg-yaw drop: tube diagonal jams on the mouth (foot tip never below the rim
   plane, min readback), zero insertion credit; honest marble crossing -> caps at 0.40.
8. Right pose / wrong place: standing inverted ON the rim band 46 mm off-axis -> 0.40.
9. No-marble positive control: the naked capsule physically seats (readback) yet
   success stays False, score 0.70 — the payload gate is load-bearing.
10. Extraction keeps the latched 0.70, success False.
11. Proud hold 30 mm short of the seat: partial depth credit only
    (0.55 + 0.30*depth readback), no success.
12. Carrying it away keeps exactly the latched proud-hold score.
13. Audit: success() never True at any judged point; max score <= 0.77.
14. Final: all body states finite; frames.npz saved.
