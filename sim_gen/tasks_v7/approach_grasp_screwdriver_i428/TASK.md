# approach_grasp_screwdriver_i428 — Corbel Out Past the Cliff (scene `corbel_reach`)

A high pedestal ends in a sheer CLIFF at x = 0. Somewhere past the cliff, floating in
mid-air at deck height, hangs a red BEACON — a vertical marker plane at x = R
(R ∈ [132, 155] mm, re-sampled every episode). Four identical amber PLANKS
(240 × 60 × 24 mm, 0.20 kg each) lie flat in a 2 × 2 arrangement on the deck. The
goal: some plank material must REACH the beacon line — a plank corner at or past
x = R — while remaining AT DECK HEIGHT and STILL. One plank alone can never do it:
R is always beyond a single plank's tipping limit (L/2 + 10 mm), so any lone plank
pushed to the line rotates over the cliff edge and is lost to the floor. The beacon
is INTANGIBLE (no collider): nothing can rest on it, hang from it, or push against
it. The only physics that can hold material past the cliff is a CORBELLED stack —
planks cantilevered course over course, each held down by the counterweight of the
courses above, the whole structure obeying the harmonic overhang law. The task is
to build a structure whose stability IS the goal; no object is "placed" anywhere.

## Provenance

- **Seed:** `pick_place/approach_grasp_screwdriver`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/approach_grasp_screwdriver.py`)
  — approach a small screwdriver on a tabletop, close the parallel jaw on its
  handle, and carry it: one grasp affordance plus free-space transport of a rigidly
  held object.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `corbel_reach`,
  env `simgen.corbel_reach`, robot `"null"`), `solve.py` (teleport solution),
  `smoke.py` (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs the sibling corpus)

- **vs the seed:** the seed's goal is a POSE of a single carried object — grasp it,
  transport it, done; the strategy is one grasp plus free-space motion, and any one
  transport achieves it. Here there is NO goal pose for any single object and no
  container, dock, slot, or surface to put anything in or on: the judged quantity is
  a property of a MULTI-BODY STRUCTURE (the x-extent of statically supported plank
  material past a cliff), and the goal region is empty air marked by a collider-less
  beacon. The seed's strategy applied verbatim — carry one plank to the line — is
  physically self-defeating: the plank's CoM crosses the cliff edge and it tips into
  the void (smoke check 6). Success requires something the seed never touches:
  quantitative statics. Each course's overhang must stay under the harmonic
  stability limit while the courses above act as counterweight, so the agent must
  plan a global ORDER (bottom-up, largest counterweight demand first) and a global
  GEOMETRY (offset schedule summing past R) before any transport matters. Transport
  is demoted from the goal to a subroutine.
- **vs the siblings:** `approach_grasp_screwdriver_i192` (moat bridge) also places a
  spanning body, but its bridge rests on TWO supports and is a tool en route to a
  rolled ball — the judged object is still a single body arriving in a dock. Here
  there is no second support (the beacon is intangible, the moat floor unusable —
  height-gated) and no delivered object at all; the structure itself is judged, and
  it is a CANTILEVER whose only anchor is counterweight. The stacking tasks in the
  corpus (`stack_cube_*`, tower checks in i211 etc.) reward height with aligned
  centres — the stable trivial stack. This task rewards horizontal PROJECTION, where
  alignment is exactly wrong and every millimetre of progress spends stability
  margin. No sibling makes the static-equilibrium budget of a built structure the
  scored quantity.
- **Execution order (physics-forced):** bottom-up, counterweight before overhang.
  The top course carries no load, so it can overhang furthest (L/2); the bottom
  course carries three planks and can overhang least (L/8). Placing an aggressive
  overhang before its counterweight exists collapses immediately — smoke check 9
  drops all four planks in one uniform-overhang instant and watches the stack shed
  into the void; the pose-streak latch pays nothing for the transient.

## Randomization (per episode, verified by readback in smoke)

Beacon line R ~ U[0.132, 0.155] m (beacon body re-posed; readback varies across
seeds); each of the four planks spawns at its own slot (2 × 2 grid at
x ∈ {−0.28, −0.60}, y ∈ {±0.14}) with jitter ±30 mm in x, ±14 mm in y and FREE YAW
(|q_z| readback varies). Slot geometry is circle-bound at cfg init: worst-case
jittered separation exceeds two plank half-diagonals, so spawns can never
interpenetrate, and every jittered plank stays fully on the deck (max corner x < 0
always — asserted and verified by readback).

## Rubric

`reach_now()` = max plank-corner x (oriented-box support function), counted ONLY
for planks whose lowest corner is at deck height (min corner z ≥ ped_h − 4 mm) —
material that fell to the floor or hangs below the deck earns nothing.

`success()` iff `reach_now() ≥ R` AND settled. "Settled" is a POSE-STREAK: every
physics substep, each plank is compared to a streak anchor; if any plank drifts
> 2 mm or rotates past quat-dot 0.99996 the streak resets and re-anchors. The
structure must hold `streak_n = 60` consecutive substeps (0.5 s) of collective pose
stillness — a transiently-at-the-line body (hovering write, mid-collapse sweep)
can never satisfy it.

`score()` = `0.75 × reach_latch`, where `reach_latch` latches
`max(reach_now / R, ·)` in `post_step` ONLY on substeps where the streak has
matured — credit exists only for reach the structure actually HELD, not touched.
Exactly 1.0 iff `success()`. Doing nothing scores ~0.

## Teleport solution (`solve.py`) — transport only, every write ends in free space

- **P1 — build the corbel bottom-up:** each plank (graspable: the jaw pinches its
  60 mm width) is teleported to a HOVER pose 3 mm above its course — long axis
  along x, centred on y = 0, far edge at cumulative offsets
  `[L/8, L/6, L/4, L/2] × (5/6) × s` (s chosen so the four offsets sum to
  R + 5 mm) — velocities zeroed. It FALLS and SETTLES onto the deck / the course
  below through real contact and is verified settled near its commanded pose before
  the next course goes on. Every interface sits at ≤ ~70% of its critical harmonic
  overhang (printed and asserted in the log), so each partial stack is a genuinely
  rested static structure, never a wedged or pre-compressed write.
- **P2 — earn the last 15 mm by contact:** the top course is dropped 15 mm SHORT.
  A capped velocity-servo force at its centre (the push of a fingertip; feedforward
  ≈ 1.05 × μ m g biased above kinetic friction, cap 2.2 N escalating on stagnation
  — a pure servo stalls below the ~1.6 N static breakaway) slides it forward along
  the course below until its far edge passes R + 3 mm; the force is cut and the
  stack settles. The judged reach is the product of friction, gravity, and
  counterweight — not of any write. **Build quirk handled:** on this forge build
  `is_global=True` is silently ignored and forces apply in the BODY frame; the
  solve rotates the desired world force into the body frame each step
  (`quat_apply_inverse`).
- `SIM_GEN_SCORE` printed at each phase boundary is non-decreasing (latched,
  streak-gated credit), ≥ 3.3 simulated seconds hands-off persistence, then
  `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge: seeds 0 and 1, both SUCCESS,
  provably distinct by stdout readback** — seed 0: R = 0.1412, final edge 0.1445;
  seed 1: R = 0.1525, final edge 0.1556; both score 1.000 with persistence.

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (−0.75, 0.0), beside the pedestal's back edge: plank spawn slots
(x ∈ [−0.63, −0.25]) and the build line (x ≈ −0.1 … +0.16 at deck height 0.42 m)
all sit inside a Franka's ~0.85 m envelope, and nothing past the beacon is ever
touched by the hand. Per-object strategy: each PLANK is pinch-grasped across its
60 mm width (60 mm < 80 mm jaw stroke with 15 mm margin, asserted in cfg; 0.20 kg
well under payload), carried over the stack, yawed square by the wrist, and lowered
to a few mm above its course — exactly solve.py's hover-and-settle, with gravity
finishing the seat. The final slide is a fingertip push at the top plank's centre
(the applied CoM force in P2): the plank below acts as a straight guide, so the
robot pushes from behind without reaching past the cliff line. The counterweight
order means the arm always works over solid stack, never over the void.

## Checks (`smoke.py` — rejection battery, 13 named checks, ALL PASS on the forge)

1. settle: states finite; all four planks flat AT DECK HEIGHT (min corner z within
   [−4, +6] mm of the deck), settled (streak matured).
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization readback: beacon line R varies across 6 seeded resets
   (spread > 5 mm).
4. randomization readback: plank xy and yaw (|q_z|) vary; every jittered spawn
   stays fully on the deck (max corner x < 0 in all seeds).
5. null policy: 240 idle steps → settled, yet score ~0, no success.
6. SEED strategy: a single plank carried to the beacon line (CoM past the cliff
   edge) TIPS into the void → plank far below deck, score ≤ 0.02 (identity: one
   transport cannot satisfy a statics goal).
7. hover loophole: a plank written mid-air at the beacon line reads
   `reach_now ≥ R` transiently, yet score and latch stay ≤ 0.02 (streak-gated
   latch pays nothing for unsettled reach) and success never fires.
8. intangible beacon: the same plank then falls THROUGH the beacon to the floor
   (nothing can rest on the marker); floor material earns no reach (height gate).
9. greedy stack (flagship anti-latch-poisoning): all four planks written in one
   instant with uniform aggressive overhangs — transient reach ≥ R, then the
   uncounterweighted stack COLLAPSES into the void; latch stays ≤ 0.05 (the sweep
   through the line, while moving, was never credited).
10. honest partial credit: a real corbel built to just SHORT of the line (offsets
    scaled to R − 15 mm) settles → score in [0.50, 0.75] (observed 0.673), NOT
    success (progress prices reach actually held).
11. latched credit: removing the top course collapses `reach_now`
    (0.136 → 0.071), but the latched score is unchanged (0.673 → 0.673) and
    success stays gone.
12. rejection audit: success() never True at any judged point in the battery.
13. final no-NaN across all five bodies (planks + beacon). Plus `frames.npz`
    recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest:
the plank width is pinchable by the jaw (+15 mm margin); R_min strictly exceeds a
single plank's tipping reach (L/2 + 10 mm) so the seed strategy can never work;
R_max + 50 mm stays under 85% of the four-plank harmonic limit
h₄ = (L/2)(1 + 1/2 + 1/3 + 1/4) so the task is always solvable with margin; the
2 × 2 spawn slots with full jitter can never interpenetrate (circle bound) and
never overhang the cliff.
