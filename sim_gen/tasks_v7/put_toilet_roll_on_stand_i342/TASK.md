# put_toilet_roll_on_stand_i342 — slide the captive ring around the bent rail onto the stand

Scene `rail_ring`, env `simgen.rail_ring` (robot="null").

## Seed provenance

Derived from **rlbench/put_toilet_roll_on_stand**: a free toilet roll and a
wall-mounted stand with a fixed horizontal peg (mesh visuals, franka, proximity
checker); the implied strategy is **free-space transport** — pick the roll up,
carry it through the air, and slide it sideways onto the peg. The roll starts
unconstrained and the whole solution is one grasp-carry-place.

## Strategic difference

The ring never leaves the rail — free-space transport is **impossible by
construction**. The blue octagonal ring (bore corner radius ≈ 26 mm) is spawned
threaded on a bent guide rail (radius 8 mm) whose free end is sealed by a red
ball cap (radius 32 mm > bore corner): the ring is **topologically captive**.
The rail runs from the capped end through two 90° yaw corners, over a downturn
crest, and down a vertical end post onto a grippy green landing plate. The goal
is the seed's end state (ring encircling a vertical post, resting flat at its
base) but the ONLY way to reach it is to **negotiate the entire course**:

- The seed's plan (grab, carry, slide on) scores ~0 here: you cannot remove the
  ring from the rail (smoke check 10 pulls it against the cap with a regulated
  force and asserts it blocks), and any pose that "looks delivered" off the rail
  is physically unreachable.
- The skill is **constrained pushing**, not transport: drive a loose annulus
  along a captive path, re-orienting it around corners where the free DOF is
  the ring's attitude, then let gravity finish the last drop down the post.
- The rubric is **arc-length progress** along the course (latched running-max,
  gated on staying within 45 mm of the rail centerline), capped at 0.85; 1.0
  iff the ring rests flat, settled, encircling the post base on the plate.
- vs sibling **i188** (spindle_roll): i188 is a two-body assembly (thread a free
  rod through a free roll, then suspend the pair); here there is ONE free body,
  zero assembly, and the constraint is topological captivity + path traversal.
- vs sibling **i304** (roofed magazine) and **handover_i339** (escrow clamp):
  no magazine, no dispensing, no clamp/handover — different mechanism class.

A solver needs a different plan and different code: no pick-and-place at all,
but a tangent-following push controller with corner negotiation and a
hands-off gravity descent.

## Solution outline (solve.py — NO teleports at all)

Transport teleports are permitted by the task rules but **unused**: the ring is
captive, so every displacement is contact dynamics (external wrenches).

- **P0** settle + layout readback; assert on-course (dist < gate), score ≤ 0.02,
  not delivered (rubric-leak guard).
- **P1–P3 TRAVERSAL (contact dynamics)**: a pursuit controller pushes the ring
  along the course — force along the local path tangent (looked ahead 20 mm),
  velocity-regulated to 0.12 m/s (f = 3.0·(v_des − v_along), clamped
  [−0.5, 0.8] N; K·dt/m ≈ 0.42, inside the wrench-delay bound), plus an
  attitude PD (τ = 0.006·(â_eff × t̂) − 0.002·ω_perp, clamped 8 mN·m; ω_perp
  only — damping axial spin through the tiny axial inertia would bang-bang).
  A stall detector (no station gain for 400 steps) fires a 60-step reverse
  pulse with a z-torque wiggle, then retries (≤8). SIM_GEN_SCORE printed as
  the ring passes the first corner (P1), the second corner (P2), and the
  downturn crest (P3) — then the wrench is CUT.
- **P4 DESCENT (gravity + hands off)**: the ring free-falls down the vertical
  end post. If it wedges (calm, slow, high), pursuit is re-engaged for 120
  steps at 0.10 m/s then cut again (≤6 re-engagements; the tangent on the post
  is straight down and the attitude PD flattens the ring). Chunked settle with
  velocity telemetry until success(), + 240 extra hands-off steps. Score 1.0.
- **P5 persistence**: ≥ 3.3 more simulated seconds hands-off with success()
  checked every substep (flicker diagnostics), then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (seed 0 on the
forge: 0 → 0.151 → 0.233 → 0.552 → 1.0 → 1.0). Passes on seeds 0, 1, 2 on the
forge — the stall detector and descent re-engage guard never had to fire.

## Rubric

`score()` = 0.85 · prog_latch; 1.0 iff `success()`.

- **prog_latch**: latched running max of normalized arc-length progress
  clamp((s − s0 − 15 mm deadband)/(S_total − s0 − deadband), 0, 1), counted
  ONLY while the ring center is within 45 mm of the course polyline (the gate
  makes off-rail states — unreachable in legal rollouts anyway — non-scoring).
  s0 is the per-episode start station, so credit is normalized to the actual
  distance the episode requires. Null policy ≈ 0 (deadband > settle drift).
- **success() = delivered ∧ settled**:
  - **delivered**: ring center within 24 mm (xy) of the post axis, |z −
    LAND_Z| < 10 mm (resting ON the plate, not hanging above), axis within 25°
    of vertical (flat, not cocked on the post).
  - **settled**: stillness sustained 30 consecutive substeps (counter in
    post_step; instantaneous thresholds fire at swing turning points).
- Anti-seed clauses are geometric, verified by ~14 `__post_init__` audits: the
  cap seals the bore (32 > 26 + 5 mm); a ring against the post side reads
  xy ≈ 42 mm > 24 mm tol; flat on the ground reads |z| error 16 mm > 10 mm;
  side-lying on the plate reads tilt ≈ 90° > 25°; parked on top of the post
  reads z error > tol; corners are negotiable (8 + 14·tan 22.5° = 13.8 mm <
  24 − 6 mm bore margin) but the chord bound keeps the ring from cutting them.

## Embodiment argument (Franka)

The ring affords a Franka grasp across opposite outer flats (68 mm < 80 mm
stroke) and its working band sits at z ≈ 0.13–0.19 m in free air along the
rail — graspable from above or the side with no fixture interference except
at the corners. From a base at the origin facing +x (rail fixture at nominal
(0.40, −0.08) ± 5 cm, ±30° yaw — the whole course, cap at local x ≈ −0.26 to
plate at local y ≈ 0.245, stays inside Franka's ~0.85 m envelope): the arm
grips the ring and slides it along the straight runs (the rail's low-friction
material takes ~0.1 N of drag, negligible vs payload limits); at each 90°
corner it releases, regrasps on the adjacent flat pair, and feeds the ring
around (the bore-to-rail clearance of ~16 mm forgives large alignment error);
at the crest it pushes the ring over the downturn and releases — gravity and
the post guide the final drop, and the 25° delivery cone plus the plate's
funnel-free flat make the landing tolerant. The hard parts are perception
(rail pose, ring attitude) and corner regrasps, not force or dexterity.

## Execution order

1. `scene.py` written first (geometry + course rubric); all captivity /
   anti-seed margins asserted numerically in `__post_init__` before any run.
2. `solve.py`: **SUCCESS first-try on seeds 0, 1, 2** (18 s each) — the corner
   chamfers, crest handover, and post descent all worked as designed; the stall
   and wedge recovery paths went unused.
3. Rubric confirmed against the demonstrated traversal (delivered readbacks:
   dpost 6–10 mm, tilt_cos 1.000, z = LAND_Z across seeds).
4. `smoke.py`: one iteration. The captivity check first asserted absolute
   score ≤ 0.02, but `hang_at(0.160)` legitimately latches ~0.06 of forward
   credit when the seed's start station is behind 0.160 — fixed by asserting
   the score is UNCHANGED by the backward pull (pre-pull latch recorded, pull
   earns nothing), not by weakening the gate. Then **ALL PASS 14/14**,
   frames.npz saved.

## Smoke checks (14)

1. settle/no-NaN + layout sanity (on-course, start band, hang height, low
tilt); 2. score ≤ 0.02, no success at reset; 3–4. randomization readback over
seeds 21–26 (rail x/y spread, yaw spread, start-station spread, all layouts
sane); 5. null policy ≈ 0 after 240 idle steps; 6. mid-course hang: partial
progress in (0.02, 0.50), on-rail, not delivered; 7. near-crest hang: high
partial credit ≤ cap, still short of the post, not delivered; 8. seed-strategy
analog — ring flat on the landing plate BESIDE the post (not encircling it):
not delivered; 9. ring leaning against the post at 40°: not delivered; 10.
captivity physical probe — regulated pull toward the capped end travels
> 50 mm (non-vacuous) then BLOCKS at the cap: the ring cannot leave the rail,
and the backward pull earns nothing (score unchanged from the pre-pull latch,
and small); 11. settle gate — the exact goal pose with an injected 4 rad/s
spin is delivered-but-not-settled, no success; 12. latched credit survives
teleport-away (progress keeps its running max, delivered False, no success);
13. rejection audit (success never True in the battery); 14. final no-NaN.
