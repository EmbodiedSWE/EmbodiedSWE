# Task `libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i244` — Egg Under the Upturned Pan

Scene: `pan_dome_egg` (env `simgen.pan_dome_egg`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf`
— "put the frying pan under the cabinet shelf": grasp the chefmate frypan, carry it, and
the task terminates the instant the pan's position enters the shelf's `bottom_region`
bbox. One cargo object, one fixture region, membership test.

Kept from the seed: the same cast — a frying pan, a shelf with a sheltered under-region,
a white bowl — and the surface story of "the pan ends up under the shelf".

## Strategic difference

**vs the seed:** the seed's pan is CARGO and its bbox membership IS the goal. Here the
pan is a TOOL, and its final pose is only a means: a fragile PALE-YELLOW egg must first
be fetched out of the bowl and seated inside the lip ring of the GREEN pad under the
shelf, and then the pan must be REORIENTED 180 deg (flipped upside-down) and set over
the egg so its rim rests on the floor all around — a protective dome. Success judges a
**relation between two objects** (egg physically inside the inverted pan's cavity,
still seated on the correct pad, everything at rest), plus an orientation predicate
(pan flipped past 15 deg of straight-down) and a height predicate (rim-down rest, not a
tilted perch on the egg) — not a point-in-bbox test. A solver needs a different plan
(two objects in a forced order; a flip regrasp) and different code (relational
enclosure + inversion + rest-height predicates, streak-latched credit), not the seed's
region check. Decoys punish memorization: a RED tomato ball of identical size sits next
to the egg in the bowl, and a GRAY decoy pad mirrors the green one — which side each
pad is on re-randomizes every episode (coin-flip pi yaw of the pads body).

**vs the corpus read this session:** the sibling i4 re-uses this seed as a non-prehensile
STUFF-THE-PAN-UNDER-A-48-mm-ROOF slide (pan still the judged object; clearance makes
grasping impossible); i32 is a gear-train machine-repair task; the packing exemplar
(pen_holder) is a fill-the-container task. None judges an object ENCLOSED UNDER a
second, deliberately inverted object: here the goal state is "A covered by upturned B
while A stays seated on the right marker", i.e. the container itself is manipulated
into an upside-down shelter over a previously-placed object. No corpus task read this
session has (a) tool-use by reorientation (flip the container upside-down), (b) a
cover/shelter relation as the goal predicate, or (c) the covered object's *retention on
a marked seat* as a simultaneous constraint.

## Scene summary

All procedural, one rigid body per asset (compound spawners; children of one body never
self-collide). A KINEMATIC wooden alcove (interior 34 x 24 cm, roof underside 0.20 m,
front open at local -y) stands at the back, xy-jittered (±3 cm) and yawed (±8 deg) per
episode. On the floor under it, a KINEMATIC two-pad body at the shelf pose plus a
coin-flip pi of yaw: GREEN target pad and GRAY decoy (discs r 30 mm, t 6 mm, each with
an 8-box lip ring, mid r 21 mm, 5 mm tall) at local x = ±70 mm — WHICH SIDE IS GREEN
re-randomizes. On the open table: the dark PAN (base disc r 75.5 mm, t 8 mm; 12-box
dodecagon wall h 45 mm, inner apothem 68 mm; straight handle 110 x 24 x 12 mm at
mid-wall height so the inverted pan rests rim-down with the handle 16 mm clear of the
floor), spawned flat right-side-up with ±5 cm jitter and free yaw; the white BOWL
(r 65 mm, wall h 28 mm — balls peek over the rim), ±3 cm jitter; inside it the
PALE-YELLOW egg and RED tomato balls (r 15 mm each), sides swapped by a second coin.
Sleep/stabilization thresholds zeroed on dynamic bodies; contact offsets 2 mm (5 mm
lips drown in the default). `describe()` gives the full layout and recipe.

Honesty geometry asserted in `__post_init__`: egg fits under the dome with >= 5 mm
headroom; every egg physically inside the wall is within `enclose_tol` (apothem 68 −
egg r 15 = 53 mm max in-dome offset < 58 mm tol < 75 mm wall outer; a pinched-outside
egg is >= 88 mm off-axis); the dome over either pad clears the alcove walls by
>= 15 mm; the roof leaves >= 10 cm above the rim-down rest for a hover-and-lower.

## Rubric (latched, streak-gated, non-decreasing)

- 0.10 · egg approach (running max of 1 − d/d0, d0 = spawn-to-green-pad distance)
- 0.35 · egg seated latch: within 12 mm xy of the green pad centre AND resting at seat
  height 21 ± 6 mm (the z window rejects an egg balanced ON TOP of the dome, z ≈ 68 mm,
  and an egg lying inside the right-side-up pan, z ≈ 29 mm), speed-gated, 20-substep
  streak (a ball flying across the pad never scores)
- 0.10 · pan prepped latch: flipped past 30 deg of inverted within 0.25 m of the pad
- 0.25 · covered latch: pan inverted within 15 deg AND body height within 8 mm of the
  49 mm rim-down rest (a pan perched tilted on the egg sits >= 14 mm high — rejected)
  AND egg within 58 mm of the pan axis AND egg still seated, both slow, 20-substep streak
- Base capped at 0.80; score 1.0 iff success() = egg seated ∧ covered ∧ settled.
  Null policy ≈ 0 (approach normalized by the episode's own spawn distance).

Cheat paths closed: covering the TOMATO (egg approach/seat/cover all reference the egg
only), using the GRAY pad (green side re-randomizes; seat is green-pad-relative),
leaving the pan right-side-up with the egg inside it (egg z ≈ 29 mm fails the 21 ± 6 mm
window; pan up_z = +1 fails inversion), balancing the egg on top of the dome (z ≈ 68 mm),
propping the pan tilted on the egg (rim height window), dragging the egg under a pan
parked elsewhere (seat is pad-relative), transit fly-throughs (streak gates + latches).

## Teleport solution (`solve.py`) — legitimacy certificate

- **P0** settle + layout readback (shelf xy/yaw, pads yaw, green side, ball/pan/bowl
  spawns, d0) — proves seed-dependent randomization from stdout.
- **P1 egg, transport only:** teleported to FREE AIR 15 mm above its seat, centred on
  the green pad, and released — seating is the free fall into the lip ring, retained by
  real contact (re-dropped from hover if a bounce leaves it unseated). Expected score
  0.45 (approach 0.10 + seat 0.35).
- **P2 pan, transport + regulated contact lowering:** one pose write puts the pan
  UPSIDE-DOWN in free air 50 mm above its rim-down rest (quat qz(shelf_yaw − pi/2) ·
  qx(pi): cavity down, handle out the open front), centred over the pad. The descent is
  a velocity-regulated world-z support force (gravity feedforward + P on vertical
  speed, clamped, cut when the rim reaches the floor) — never a pose write; the rim
  lands on real contact and the pan settles hands-off. Expected score 0.80.
- **P3** judge; **P4** hands-off >= 3.3 simulated seconds; `SIM_GEN_SOLVE: SUCCESS`
  only if success() persists and the score never decreased. Verified on seeds 0 and 1.

## Embodiment argument (Franka, single arm)

- Base at ~(0, −0.60), facing the alcove's open front; both the bowl (~(0.26, −0.12))
  and pan spawn band (~(−0.22, −0.16) ± 5 cm) are within ~0.45 m reach, on open table.
- Egg: a 30 mm sphere whose top stands 36 mm proud of the bowl floor while the bowl rim
  is only 32 mm — a top-down pinch (80 mm jaw) clears the rim. Seating = hover over the
  green pad and release from a few mm; the lip ring centres the ball. The pad sits at
  alcove mid-depth but the roof is 0.20 m up — a horizontal wrist reach-in with the
  fingers pointing down has ~15 cm of headroom; no part of the arm needs to touch the
  shelf.
- Pan: grasped across the 24 mm handle (jaw 80 mm) at 16–65 mm height; the wrist flips
  it 180 deg in free air (0.40 kg, well under payload). With the pan centred over a
  pad, the handle points out the open front and its tip stays ~60 mm OUTSIDE the
  alcove's front plane — the gripper holds the handle from outside and lowers; the
  fingers never enter under the roof. Rim-down landing needs only ~5 mm of vertical
  precision (rim z tol 8 mm), and the 15 deg tilt tolerance is generous for a held
  lower-and-release.

## Ordering

Egg-then-pan is physically forced: once the pan is a rim-down dome on the pad, the
only openings are the 16 mm handle slit — the egg cannot be inserted afterwards, and
the cover latch itself requires the egg ALREADY seated while the pan settles. The
rubric imposes nothing beyond what the geometry does.

## Checks (`smoke.py`, 15)

1. settle/no-NaN + readback (balls in bowl, pan flat, pads at shelf); 2. baseline
score ≤ 0.02; 3. randomization by readback across 6 seeds (pan/bowl/ball xy, pan yaw);
4. shelf xy/yaw vary AND the green side flips across seeds; 5. null policy 240 steps
≈ 0; 6. SEED-STRATEGY end state (pan carried flat right-side-up under the shelf, egg
untouched): ≤ 0.15, not success; 7. wrong object (TOMATO seated + covered on green):
≤ 0.25, not success; 8. wrong place (egg seated + covered on the GRAY pad): ≤ 0.30,
not success; 9. near-miss uncovered (egg seated, inverted pan parked beside the pad):
score in [0.40, 0.60], not success; 10. tilted rim-perch (pan lowered onto an
off-centre egg, rim on the egg): rim-height window rejects; 11. egg lying inside the
right-side-up pan on the pad: z window + inversion reject; 12. latched credit survives
the egg being removed back to the bowl (non-decreasing latches); 13. egg balanced ON
TOP of the upturned dome: z window rejects; 14. rejection audit (no rejection check
ever saw success()); 15. final no-NaN. Frames recorded to `frames.npz`.
