# stack_wine_i354 — Clamp the Bottle into the Spring-Loaded Bay (scene `bottle_spring_bay`)

A low kinematic rack carries TWO identical open-top bays. Each bay is an 80 mm
channel between low walls with, at its far end, an end wall wearing an inward
overhanging LIP, and at its near end a DYNAMIC spring-loaded PLUNGER (X prismatic
joint + post_step return spring) carrying a square socket CUP that faces the wall.
The relaxed cup-to-wall gap is 240 mm — SHORTER than the 250 mm wine bottle
standing on the floor. An amber beacon tile marks the TARGET bay. Success = the
bottle clamped horizontally in the target bay like a battery in a compartment:
neck sprung into the cup with the spring at its SEAT compression (~10 mm, band
4–16 mm), base face pinned against the end wall UNDER the lip, bottle at lying
height, everything settled and RELEASED. Because the bottle is longer than the
relaxed gap, the only way in is the compression cycle — enter pitched (a level
bottle does not fit), press the spring well past the seat point (~28 mm) so the
raised base can drop past the lip, lower the base to the deck, then release and
let the spring shove the base under the lip onto the wall.

## Provenance

- **Seed:** `rlbench/stack_wine`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/stack_wine.py`) — grasp THE
  wine bottle and LAY it on THE rack: one pick, one support-from-below set-down,
  no mechanism between grasp and goal.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene
  `bottle_spring_bay`, env `simgen.bottle_spring_bay`, robot `"null"`),
  `solve.py` (teleport-transport + wrench-servo capture cycle), `smoke.py`
  (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed's goal contact is "object rests ON TOP of the
  fixture" and its plan is approach-grasp-carry-set — the rack never pushes
  back. Here the fixture is an ACTIVE mechanism that opposes the insertion the
  whole way: the goal state is a spring-retained CLAMP whose geometry
  (G0 = L − seat_c) makes the seed's set-down literally unconstructible, and the
  plan is a forced four-stage cycle — pitched entry (level does not fit), press
  PAST the goal compression (seat is 10 mm but insertion transits ~28 mm — you
  must overshoot the goal to reach it), lower behind the lip, then RELEASE so
  the spring itself finishes the job. The rubric's seat band [4, 16] mm rejects
  both endpoints the manipulator can hold: uncompressed rest and full press.
  Support-from-below versus press-past-and-release; the seed's strategy
  transplanted verbatim (bottle laid on top of the bay walls) is physically
  built by smoke check 6 and scores 0.
- **vs `stack_wine_i48` (`cask_weight_sort`):** same seed, but i48's
  load-bearing problem is EPISTEMIC — hidden shuffled masses must be probed and
  sorted; its motor act is a plain drop into an open cradle. Here everything is
  visible and the load-bearing problem is a compliant MECHANISM: a spring that
  must be driven through a compression overshoot and then handed the final
  seating motion. Measure-then-place versus press-lower-release.
- **vs `stack_wine_i202` (`bottle_hang_rail`):** same seed, and both are
  geometry-gated insertions — but i202's keyhole rail is entirely PASSIVE (rigid
  apertures; the object simply cannot take the wrong path) and its goal is a
  gravity hang below the fixture. Here the gate is an ACTIVE energy-storing
  element: the blocking constraint (gap shorter than bottle) is created by a
  live spring, the insertion stores elastic energy that the rubric requires to
  be RE-EXPENDED into the final seating (release is load-bearing — a held press
  at ~28 mm is explicitly rejected, smoke check 9), and success is judged on the
  spring's own compression readback, a mechanism state, not just the object
  pose. Thread-through-and-hang versus compress-overshoot-and-release; i202 has
  no state the robot can hold that looks almost right, while this task's
  central rejection is exactly that held state.
- **Execution order (declared):** within the cycle, pitch-enter -> press ->
  lower -> release is mechanically forced by the geometry and the spring (the
  base cannot pass the lip below ~28 mm compression; the seat band cannot be
  reached while held), not by the rubric. There is one bottle and two bays; the
  beacon names the target, and no other ordering constraint exists.

## Randomization (per episode, verified by READBACK in smoke)

Target bay: a fair coin (after burning draws — the first post-seed draw is
degenerate on this stack) teleports the amber beacon in front of bay 0 or bay 1;
the rubric follows the beacon. Bottle spawn: x in [−0.42, −0.32], y in
[−0.12, 0.12], free yaw. The rack/plunger JOINTED pairs are deliberately fixed
at their authored pose (jointed pairs cannot be teleported per episode on this
PhysX stack — the joint frame stays anchored), so only the free bodies and the
target assignment randomize. Smoke check 3 reads back 8 seeded resets: both bays
appear, the beacon matches the target every time, spawn xy spreads > 20 mm, yaw
varies.

## Rubric

`success()` iff, judged live on physical poses (fixture frame == env frame):

- bottle axis within `align_deg` = 10° of the bay axis, neck toward the cup;
- neck tip y within 10 mm of the bay centreline and base centre y within 15 mm
  (the cup and bay walls physically bound both — every true clamp passes by
  construction, asserted in `__post_init__`);
- bottle CoM z within 8 mm of the lying axis height (rejects on-top at +45 mm,
  standing, and every perch);
- neck tip at the LIVE cup back plate (−4 to +6 mm of its readback x);
- the TARGET plunger's spring compression in the SEAT band [4, 16] mm — the
  released clamp rests at 10 mm; an uncompressed rest reads 0, a held press
  ~28–31 mm, a base-first insertion ~30 mm (all outside, asserted);
- base face x at the end wall (≥ x_wall − 6 mm);
- settled: bottle |v| < 0.10 m/s (above the GPU phantom band; the tight
  position/compression windows do the work), |ω| < 0.60 rad/s, target plunger
  |v| < 0.08 m/s.

`score()` (latched every physics substep in `post_step`): 0.10 ever lifted above
150 mm + 0.20 neck ever inside the target cup + 0.20 spring ever pressed past
24 mm with the neck cupped + 0.10 ever in the full seat clause for 24
consecutive substeps, capped at 0.60; exactly 1.0 iff `success()`. Doing nothing
scores ~0; the seed strategy scores 0; the held press caps at 0.40–0.50.

## Teleport solution (`solve.py`) — transport only, every load-bearing act by contact

**T** teleport to a hover in free air above the bay, pitched 45° base-up, neck
toward the cup, velocities zeroed (never inside the bay, never touching
anything); **A** a CoM-force + axis-torque PD holds the hover and doubles as the
wrench-frame divergence probe (forge pods drag world wrenches by body rotation;
commands are pre-encoded into the current body frame — and this bottle is held
40°+ from identity, so the encode is exercised hard before anything
load-bearing); **B** slew the neck tip straight down into the open-top bay at
45° (the pitch is what makes entry geometrically possible); **C** one coupled
sweep θ 45°→0° with tip_x(θ) = X_KEEP − r·sinθ − L·cosθ and a spring
feed-forward −k·c_ref(θ): the neck enters the cup, presses the plunger to
~30 mm, and the base swings down past the lip onto the deck (X_KEEP keeps every
body point clear of the lip's x band while descending); **D** once slow, all
wrenches zeroed — the SPRING, not the servo, shoves the bottle +x until the base
slides under the lip and seats on the end wall at exactly the seat compression.
Gains audited in the docstring (all discrete-stability ratios ≪ 1); release only
below 0.05 m/s (wrenches act one substep late). `SIM_GEN_SCORE` at every phase
boundary is non-decreasing (0.00 → 1.00 at seating), ≥ 3.3 simulated seconds
hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge:
seeds 0, 1 and 2, all SUCCESS first attempt, provably distinct by stdout
readback** — spawns (−0.418, +0.069) / (−0.395, +0.062) / bay flip: seeds 0–1
target bay 0, seed 2 target bay 1 (press verdict base y = +0.110); every run
presses to 30.0 mm and releases to a 10.0 mm seat with base_x = +0.1400 (the
wall).

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (0.35, 0.0) behind the end walls, or (0, −0.55) beside the rack:
spawn band (x ≈ −0.37, |y| ≤ 0.12), hover, and both bays (y = ±0.11, z ≤ 0.10)
sit within a ~0.75 m reach envelope at comfortable heights. Per-object contact
strategy: a top-down side grasp of the 60 mm body with the 80 mm jaw (20 mm
margin; 0.45 kg ≪ 3 kg payload), carry pitched ~45°, lower into the 80 mm
open-top channel — the bay is 20 mm wider than the body, the side walls are only
45 mm tall, and the hand rides ABOVE the open top the whole time (the walls stay
below the bottle's upper surface, asserted), so the fingers never enter a
confined space. The press is a guarded ~30 mm horizontal push (3.6 N ≪ Franka's
capability) with the cup walls funnelling the 22 mm neck into the 34 mm opening
(6 mm slack per side — compliance, not precision); the lower is a ~20 mm wrist
arc; the release is just opening the jaw and withdrawing vertically — the spring
finishes the seating without the hand. The rack is kinematic and cannot be
knocked over; the wrong bay is 220 mm away from the right one, trivially
distinguished by the beacon.

## Checks (`smoke.py` — rejection battery, 15 named checks, ALL PASS on the forge)

1. settle: states finite; bottle standing on the floor at CoM height (readback),
   both plungers at their extended homes (comp ≈ 0), settled.
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization readback: 8 seeded resets — target bay takes BOTH values,
   beacon tile matches it every time, spawn xy and yaw vary.
4. spring principle is physical: PhysX mass readback matches the authored bottle
   and plunger masses, and a plunger teleported to 28 mm compression RETURNS to
   its home under the post_step spring (residual < 4 mm) — live physics, not
   bookkeeping.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: the bottle laid ON TOP of the bay walls (readback z ≈ 0.085 vs
   seat height 0.040), settled — NOT success, score 0 (support-from-below is
   exactly what this task rejects).
7. wrong-bay clamp: the FULL spring-retained clamp constructed in the non-target
   bay stays physically clamped through 2 s of contact (compression readback in
   the seat band, base on the wall — retention is the spring) yet NOT success.
8. base-first insertion: the reversed bottle (60 mm body cannot enter the 34 mm
   cup) rests pressed at ~30 mm compression, far outside the seat band — NOT
   success.
9. held-press: the mid-cycle state held at full press by a constant axial force
   (comp readback ~31 mm ≥ press_c, neck in the cup) earns stage credit only
   (0.40) and is NOT success — release is load-bearing. The bottle is removed
   before the force is (a bare release would seat it — that is the solution,
   demonstrated by solve.py, not this battery).
10. dropped-at-the-mouth perch: a bottle dropped pitched at the cup mouth lets
    gravity drive the neck in — compression can even reach the seat band — but
    the base stays perched on the lip/wall top; the seat-plane and alignment
    clauses reject the spoof, NOT success.
11. standing-in-bay: the bottle standing upright on the deck inside the target
    bay, xy aligned — NOT success.
12. partial credit: a lift above `lifted_z` earns exactly the lifted stage
    credit (score ≈ 0.10), NOT success.
13. latched credit: parking the bottle on the far floor leaves the latched
    credit unchanged, success stays gone.
14. rejection audit: success() never True at any judged point in the battery.
15. final no-NaN. Plus `frames.npz` (~270 frames) recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task
honest: the neck passes the cup opening but the body does not, the cup catches a
lifted neck within millimetres, the relaxed gap is genuinely shorter than the
bottle (level set-down impossible), the insertion compression c_need fits the
joint travel yet sits far outside the seat band (held press rejected), the
spring at seat compression out-pulls deck friction ×1.8 (release genuinely
seats), a 25°-pitched bottle fits the relaxed gap (insertion feasible), a
base-first insertion rests far outside the band but inside the travel, the side
walls corral the lying bottle yet leave the top open, an on-top rest sits ≥
30 mm above the seat band, fingers fit beside the body, the bottle is trivially
graspable and liftable, and the y tolerances admit every physically cupped/bayed
pose.
