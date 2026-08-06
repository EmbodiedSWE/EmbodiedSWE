# oven_dials — set the oven's two control dials to their indicated settings

**Registered as:** `SCENES["oven_dials"]`, env `simgen.oven_dials` (robot="null",
scene-level; `solve.py` builds its own Franka env).
**Tier: medium — 2 goal-conditioned stages** (one detented dial per stage;
**execution order NOT required** — either knob first).

## Seed provenance

- Seed: `rlbench/open_oven`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/open_oven.py`) — a Franka pulls the
  hinged oven door open by its handle (USD oven asset + recorded trajectory, binary
  fixed goal, no checker).

## What changed, and why it is strategically different

The seed's plan skeleton is: *approach the one handle → grasp → pull the hinged panel
through a large free arc until "open"*. A gross-motion, fixed-goal, binary pull on the
oven's single moving part.

This task keeps the oven's control surface and **removes the door entirely** (the deck
is kinematic — nothing on this oven opens). What remains is the oven's *other*
affordance: two spring-**detented control knobs** on vertical revolute spindles rising
from the top console. Each episode samples, per knob, a random start setting and a
random target setting (≥ 2 detents apart); the target is shown physically by an amber
lamp parked at that setting's tick mark. The solver must:

1. **read the sampled goal from the scene** (which tick glows, per knob) — the seed has
   no goal-conditioning at all;
2. perform **precision rotary positioning**: turn each knob about its vertical spindle
   and *stop* inside an 8° tolerance backed by a ±20° detent capture basin — a
   stop-at-target skill, where the seed's pull runs to a hard limit;
3. do it **twice, on independently-goaled dials** (order free).

A different PLAN, not different parameters: no grasp-and-pull arc, no door, no binary
"open" predicate; instead goal reading, spindle-axis turning, tolerance stopping. The
seed's own strategy is **expressible and measured as a negative control**: a sustained
25 N pull plus a 2.5 N·m hinge-style prying torque on a knob (exactly the wrench that
opens the seed's door) moves nothing and scores ~0 (smoke check 5).

## Scene / physics honesty

Procedural only: kinematic deck (Cuboid), knobs (Cylinder) on authored USD Z-axis
revolute joints (limits ±128°), symmetric grip bars (fixed joints; CoM on the spindle),
visual-only tick/cap decorations, amber lamp bodies (kinematic) teleported to the
target tick each reset. Knob "feel" (viscous friction + nearest-setting detent spring,
overdamped: ζ≈2.4, capture measured 15° in / 25° out) is applied in `post_step` — the
scene owns the knobs' external-wrench slot; the `knob_drive`/`knob_force`/
`knob_torque_ext` buffers exist for smoke instrumentation only and are never touched by
the robot solution. The detent spring (0.25 N·m/rad → ~2.6 N peak at the bar) is sized
so a fingertip-scale push can dominate it while the capture snap stays crisp.

## The real-robot solution (solve.py — the feasibility certificate)

Franka, OSC, **base at (0, 0, 0) on the ground, facing +x** (identity rotation); the
deck's near edge is 0.16 m ahead, knobs 0.48 m ahead at ~0.22 m height — the proven
top-down envelope. Per knob (right knob first, order free):

1. READ the pointer angle (knob yaw) and the target setting;
2. choose the grip END of the symmetric bar whose swept arc stays farthest from the
   base direction (dragging toward the base is the arm's worst-conditioned pull — the
   force leaks into a parasitic vertical push and the pads climb off the bar);
3. APPROACH above that end, open jaw perpendicular to the bar, wrist-roll branch
   chosen to centre the chunk's rotation on the home azimuth;
4. DESCEND straddling the bar's lower half; PINCH (12 mm command on the 16 mm bar);
5. SWEEP: drag the pinched end along the spindle-centred arc, wrist tracking the bar,
   with a 14–26° servo lead (the stalled position error IS the drive force; a stall
   booster adds up to +18°), chunked ≤ 100° per pinch;
6. RELEASE inside the stop window; the detent seats the pointer exactly; VERIFY with
   the scene's own `at_target()`.

Verified SIM_GEN_SOLVE: SUCCESS on seeds 0, 1, 2, 3 (distinct sampled instances,
including a 240° traverse in two chunks and both turn directions); `SIM_GEN_SCORE`
prints are non-decreasing along every run (latched credit never evaporates).

## Rubric (graded 0..1)

- per knob: latched best progress `1 − err/err0` (forced to 1.0 once the pointer ever
  enters the target zone) × 0.35, plus 0.15 if *currently* resting on target;
- `score == 1.0` **iff** `success()` (both dials settled on their targets);
- ~0 for doing nothing (start jitter ±4° vs authored `err0 ≥ 80°`);
- transient achievements latch: a dial knocked off after success leaves 0.85.

## Smoke check list (14 — rejection battery + sanity; smoke never solves)

1. settle: clean reset (finite, pointers on start settings, score ~0)
2. randomization is real (start/target draws differ across seeds, by readback)
3. goal indicator: lamps sit at the sampled target marks (readback)
4. null policy: score < 0.05 and no success
5. negative (SEED strategy): pulling/prying a knob opens nothing, score ~0
6. negative (near miss): adjacent setting (40° off) rejected by the 8° stop tolerance
7. negative (swap): dials on each other's targets rejected — goal assignment matters
8. rubric monotonicity: latched score never decreases along the approach
9. partial credit: knob 0 on target → score ~0.5, no success
10. exactness: both dials set → success() and score == 1.0
11. achievement latch: knock-off revokes success, score 0.85
12. calibration: offsets ≤ 15° captured back to the target detent
13. calibration: offsets ≥ 25° fall to the neighbour detent (rejected)
14. video frames recorded (frames.npz in CWD)

Both `smoke` (ALL PASS 14/14) and `solve` (SUCCESS, seeds 0–3) verified on the forge
from the submitted package.
