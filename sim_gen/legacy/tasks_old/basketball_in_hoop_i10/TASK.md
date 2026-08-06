# tee_up — un-basket a bouncy ball and perch it on a narrow tee

## Provenance

- **Seed:** `rlbench/basketball_in_hoop`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/basketball_in_hoop.py`) — grasp a free
  ball, hold it anywhere above an elevated hoop, release; a relative-bbox detector under
  the hoop fires when the ball passes through.
- **This task:** `sim_gen.tee_up` (scene `tee_up`, robot `null`), fully procedural
  geometry (compound box/cylinder spawners, no asset files).

## What changed, and why it is strategically different

The seed rewards **ballistic containment**: the target is a forgiving concave funnel,
gravity does the aiming, and any release above the hoop succeeds — the plan is
"hover anywhere above, let go". This task inverts the seed on three axes at once, so the
seed's plan is not just insufficient but **actively fails**:

1. **Start state is the seed's goal state.** The ball begins *contained* (resting at the
   bottom of a low open basket) and must first be **extracted** — putting the ball into a
   container scores nothing here (checked by a negative control).
2. **The goal support is convex, not concave.** The target is a shallow cradle (10 mm
   rim, 40 mm aperture < 60 mm ball) atop a slender post: the ball can only **perch** on
   the rim circle, mostly exposed. There is no funnel to catch a toss.
3. **The ball is bouncy** (restitution ≈ 0.6, combine-mode max). The seed's own strategy
   — dump from height above the target — physically **bounces/deflects off the tee**
   (asserted as a negative control at 0.25 m, and measured as a release-height sweep).
   Only a low-energy (≲ 3 cm), laterally centered (≲ 1–2 cm) release seats the ball.

A solver therefore needs a different plan: *extract from container → transport →
precision gentle placement on a perch*, instead of *pick free ball → hover → release*.
Same-strategy-different-numbers cannot pass: executing the seed's toss on this scene is
one of the smoke's negative controls.

## Difficulty tier / stages

**easy — 2 stages, execution order REQUIRED** (forced by geometry: the ball cannot be on
the tee while still in the basket):

1. extract the ball from the basket (latched, 0.20);
2. seat it on the tee cradle (success).

## Rubric (graded score in [0, 1])

- `0.00` — nothing / ball still in the basket (null policy).
- `0.20` — latched: ball extracted (clear of the basket footprint or lifted clear of the
  rim). Latched in `post_step` at sim rate, so transient achievements are kept.
- `0.45` — latched: ball brought aligned above the cradle.
- `1.00` — **iff `success()`**: ball *currently* seated (center within 12 mm of the tee
  axis, height within 10 mm of the geometric perch height
  `rim_top + sqrt(ball_r² − cradle_r²)`) **and** settled (|v| < 0.05 m/s). Success is a
  physical present-state outcome: a bounced-off or rolled-away ball keeps only the
  latched partial credit.

Physics honesty: the basket and tee are kinematic furniture (the seed's hoop was a fixed
XFORM too) but are re-posed with real randomization each reset (verified by readback);
the ball is fully dynamic, and the oracle's final seat is a genuine release-and-settle
(drop from 6 mm), not a pose freeze.

## Randomization

Basket xy (± 6 cm) + yaw, tee xy (± 6 cm) + yaw, ball offset anywhere on the basket
floor. Tee height and ball bounciness are build-time tunables (`tee_h`, `restitution`).

## Smoke check list (15 checks)

1. settle/no-NaN — reset settles finite, ball contained, score 0.
2. randomization-is-real — basket/tee/ball readback deltas across seeded resets.
3. null-policy-fails — 240 idle steps, score ≤ 0.05.
4–6. oracle reaches `success()` on seeds 0, 1, 2 (teleport-carry + real release).
7. rubric: extraction stage scores 0.20.
8. rubric: hover stage scores 0.45.
9. rubric monotone 0 → 0.20 → 0.45 → 1.0.
10. **negative (seed strategy)** — 0.25 m dump above the tee bounces off, no success.
11. negative (near-miss) — ball resting at the tee base (xy-near, z-low) rejected.
12. negative (re-contain) — ball returned into the basket scores ≤ 0.20.
13. negative (tolerance) — gentle release 35 mm off-axis rolls off, rejected.
14. calibration — gentle releases (≤ 30 mm) seat ≥ 5/6 (3 seeds × 2 heights).
15. calibration — high dumps (≥ 200 mm) seat ≤ 2/6.

The seed's own strategy **is expressible** here (hover above the target and release) and
is check 10; the calibration sweep (release height ↔ seat rate) maps the gentleness
envelope that separates it from the intended plan.

## Measured result (forge, 2026-07-23): SIM_GEN_SMOKE: ALL PASS 15/15

Calibration sweep (release height above the perch @ 12 mm offset, 3 seeds each):
10 mm 3/3 · 30 mm 3/3 · 60 mm 0/3 · 120 mm 0/3 · 200 mm 0/3 · 280 mm 0/3 — a sharp
gentleness cliff between 30 and 60 mm. The 0.25 m seed-style dump left the ball ~0.9 m
from the tee.
