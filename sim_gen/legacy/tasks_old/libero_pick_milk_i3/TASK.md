# milk_upright — stand the fallen milk carton back up

## Seed provenance

- Seed id: `libero/libero_pick_milk`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/libero/libero_pick_milk.py`
  ("Pick the milk and place it in the basket": grasp the milk carton among 5
  distractors, transport it, drop it inside a basket; success = a relative-bbox
  containment check against the basket).

## What changed, and why it is strategically different

The seed is a **prehensile transport-into-container** task: identify target → grasp →
lift → carry → release inside a container. This task keeps the LIBERO tabletop flavor
(milk carton, clutter, a container in the scene) but **inverts the required plan into a
reorientation task with zero transport and zero containment**:

- The milk carton starts **fallen on its side** (on a random one of its 4 side faces,
  random yaw). The goal is to **pivot it back upright onto its base, directly on the
  work surface**, and leave it standing still. Nothing is carried anywhere; there is no
  destination container or zone — the solve is a ~90° reorientation roughly in place
  (pivot about a base edge, or grasp–rotate–set-down).
- The container from the seed is present as **seed-bait**: an open crate with a
  deliberately **raised inner floor (35 mm, > 2× the 15 mm base-height tolerance)**.
  Executing the seed's plan — put the carton in the container — yields "upright but not
  on the table" and is **rejected by the base-height gate** (negative control below).
- Orientation semantics the seed never needed: the carton has a distinct dark **base**
  and a **gabled roof** (two slanted panels meeting in a ridge). Standing it on the
  wrong end is **physically impossible** — upside-down it balances on a knife edge and
  topples under real physics — so the "base down" clause is honest by construction.
- A scene-integrity clause: a **standing juice bottle** nearby must still be upright at
  judgment time (flailing the carton around and wrecking the scene is not a solve, and
  caps the score at 0.4 while the bottle is down).

A solver that replays the seed's strategy (pick up, carry to the container, drop in)
tops out at 0.75 partial credit and never reaches success; the winning plan —
edge-pivot / reorient and stabilize — does not appear in the seed at all.

## Difficulty tier / stages

- **Tier: easy. Declared stage count: 1** (single skill: reorient the fallen carton to
  a stable upright stance on the surface).
- **Execution order: not required** — there is only one stage; no ordering constraints.

## Judging (physical outcomes only)

`success()` (all four, live): roof axis within 12° of world-up · bottom face within
15 mm of the surface (rejects in/on the crate, on the butter box) · carton settled
(|v| < 0.05 m/s, |ω| < 0.5 rad/s) · juice bottle still standing (< 30° tilt).

`score()` (graded, latched transients, 1.0 iff success): 0.25 once the carton's long
axis was raised past 30°, 0.50 past 60°, 0.75 once it was upright anywhere (e.g. held
in the air); capped at 0.4 while the bottle is knocked over; 0 for doing nothing.

Randomization: carton resting face (4-way) + yaw ±180° + xy ±5 cm; crate, bottle and
butter box xy/yaw jitter. All assets procedural (compound box spawners / primitives).

## Smoke check list (16 checks — final forge run: `SIM_GEN_SMOKE: ALL PASS 16/16`)

1. settle/no-NaN — reset layout settles, all root states finite, untouched score = 0.
2. randomization-is-real — READBACK: 3 seeded resets move the carton (pos + heading)
   and the crate.
3. null-policy-fails — 150 idle steps: score stays ~0, no success.
4–6. oracle reaches success() on seeds 0 / 1 / 2 (teleport lift → pivot → lower →
   release; success judged by real settled physics). Counts as 3 checks; seed 0 also
   records the per-step score trace.
7. rubric monotonicity — seed-0 trace: milestones first-hit in order
   0.25 → 0.50 → 0.75 → 1.00, non-decreasing up to the first 1.0.
8. negative (intact-scene) — knocking the bottle over after a solve kills success and
   caps the score at 0.4.
9. negative (SEED STRATEGY, expressible) — carton placed perfectly upright INSIDE the
   crate: upright gate passes, base-height gate rejects → no success, score ≤ 0.75.
10. negative (near-miss) — 16° lean rejected by the 12° upright gate (judged at the
   authored pose, zero velocity, so the verdict pins on tilt alone).
11. calibration — 8° lean accepted by the same gate (tolerance bracketed).
12. negative (upside-down) — inverted carton rejected by the gate AND toppled by real
   physics off the roof ridge (measured: up_z −1.00 → +0.00); score pinned at 0.
13–15. calibration sweep (3 checks) — lean-release 5–35°: 5–20° physically rock back
   upright to success, 25–35° topple flat, measured recovery boundary = 20°
   ∈ [10°, 30°] (rubric gate 12° < physical topple basin 20°: the tolerance is
   physically reachable and strictly inside the recoverable region).
16. video frames captured → `frames.npz` written to the CWD (212 frames, 960×600).

Every check is named PASS/FAIL in stdout; the printed marker is authoritative.

## Iteration notes (honesty fixes found on the forge)

- Runs 1–2 exposed a fake-physics trap: `settle_until` judged freshly-authored
  zero-velocity poses without a single physics step (a written pose trivially passes a
  velocity settle gate). The sweep now forces 60 real steps before judging.
- The roof ridge originally had a 4 mm panel overlap — a blunt strip an inverted
  carton could balance on. Reduced to 1 mm (true knife edge) and the inverted control
  is given a gentle 0.3 rad/s symmetry-breaking nudge; it now genuinely topples.
- The carton's PhysX sleep/stabilization thresholds are zeroed so unstable authored
  poses can never freeze mid-lean.
