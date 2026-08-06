# serve_ingredient — pour the ball out of its cup onto the dish (i2)

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate.py`)
- Seed plan: among three identical black bowls, select the MIDDLE one by position, grasp the
  container, transport it, set it down ON the plate. Success = bowl xy within 6 cm of the
  plate and 0–3 cm above it — i.e. *container on target*.

## What changed, and why it is strategically different

Same object vocabulary (three identical containers + one flat target vessel), inverted plan:

1. **The deliverable is the CONTENTS, not the container.** One of three identical cups holds
   an ingredient ball. Success = the *ball alone* rests on the dish floor inside its rim. The
   solver must EXTRACT the contents — tip/pour the cup over the dish, or pluck the 30 mm ball
   out of the cup — a skill family (controlled reorientation / emptying, or precision
   small-object grasp inside a cavity) absent from the seed.
2. **The seed's plan is an explicit failure mode.** Carrying the loaded cup onto the dish
   fails twice over: the ball rides ~8 mm above the dish floor (fails the on-floor z gate),
   and any cup over the dish footprint violates the `dish_clear` clause (the dish must hold
   the ball alone). Negative control A executes exactly this strategy and must score ~0.
3. **Selection is by content, not by position.** The seed names bowl_2 ("the middle one") in
   the goal; here WHICH cup is loaded is resampled every episode, so the solver must perceive
   the contents (ball visible from above only) rather than execute a memorized index.

A solver that transfers the seed policy (grasp container → place on target) gets ~0; the
correct plan is: find the loaded cup → extract the ball onto the dish (pour or pluck) → keep /
put the cup back away from the dish.

## Difficulty tier / stages

- **Tier: easy.** Declared **2 stages**: (1) extract the ball from its cup onto the dish;
  (2) set the (possibly carried) cup down clear of the dish. A pluck-style solver that never
  brings the cup over the dish collapses this to a single stage.
- **Execution order: NOT required.** Any order/plan that ends with the ball settled on the
  dish floor and all cups clear of the dish succeeds; the rubric judges physical end states
  (and one latched transient: "the ball has left its cup").

## Rubric (graded, latched)

| score | state |
|-------|-------|
| 0.00  | nothing done (ball still in its source cup) |
| 0.25  | ball has left its source cup at least once (LATCHED in `post_step`) |
| 0.60  | ball currently on the dish floor inside the rim, dish upright (not settled) |
| 0.85  | ball served + settled, but some cup still over/on the dish |
| 1.00  | success(): ball settled on dish floor, dish upright, every cup clear |

Honesty: judged in the dish body frame; `on_xy_tol` = 66 mm is just above the deepest
physical corner-nestle of the ball inside the octagonal rim (~65 mm), so anything physically
on the dish floor counts; a ball balanced ON the rim (~79 mm, +z) or riding inside a cup on
the dish (+8 mm z) is rejected. The oracle may teleport (teleport-oracle), but every judgment
is on settled physical states.

## Scene / assets

Fully procedural, one rigid body per object: a shared octagonal-vessel compound spawner
(bottom disc + 8 wall boxes; explicit MassAPI, small contact offsets, 0.5 m/s depenetration
cap) builds both the 3 deep cups (inner r 30 mm, h 58 mm) and the wide low dish (inner
r 75 mm, rim 20 mm above a 12 mm anti-tunneling floor); the ball is a 15 mm sphere with zero
restitution + damping. Randomized per episode: dish pose (xy + yaw), each cup pose (xy +
yaw), source-cup index, ball jitter.

## Smoke check list (19)

1. settle: reset states finite (no NaN)
2. rubric clean at reset (score 0)
3. null policy: score ~0, no success after 200 idle steps
4. randomization: dish + cup layouts differ across resets (readback)
5. randomization: >=2 distinct source cups sampled (readback)
6. randomization: ball sits in the sampled source cup (readback)
7–9. oracle seeds 0/1/2: kinematic-carry POUR reaches success() (ball genuinely rolls out of
   the tilted mouth and falls onto the dish — never pose-set on the happy path)
10. oracle milestone: escape latched at >=0.25 (partial credit)
11. oracle milestone: ball served, cup overhead -> 0.85, not success
12. rubric monotonicity along the oracle (0 < escape < 0.85 < 1.0)
13. negative A (SEED STRATEGY): loaded cup carried onto the dish -> no success
14. negative A: score pinned ~0 (ball never left the cup — the plan itself is wrong)
15. negative B (near miss): ball beside the dish -> exactly 0.25, no success
16. negative C (dish not cleared): empty cup parked in the dish caps score at 0.85
17. tolerance twin: removing the cup completes to success (served ball 35 mm off-axis counts)
18. calibration sweep: within-funnel (<=50 mm) drop-capture rate >= 70% (funnel = 60 mm)
19. calibration sweep: some offset fully reliable

Measured on the forge (final passing run): capture 2/2 at every offset 0–70 mm (the rim
wall funnels near-misses inward), within-funnel rate 100%; oracle 3/3 seeds with 0 pour
retries; milestone ladder read exactly 0 → 0.25 → 0.85 → 1.0.

`smoke.py` exports `oracle_solution(env)`, records video and saves `frames.npz` in the CWD,
prints `SIM_GEN_SMOKE: ALL PASS n/n` on success, and hard-exits behind a 10 s watchdog.
