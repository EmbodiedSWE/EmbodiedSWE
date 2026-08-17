# libero_pick_orange_juice_i256 — Shrouded-carousel indexed extraction

**Scene:** `simgen.juice_carousel` (`JuiceCarouselScene`, robot="null")
**Seed:** `libero/libero_pick_orange_juice`
(`RoboVerse/roboverse_pack/tasks/libero/libero_pick_orange_juice.py`)

## Seed provenance and what changed

The LIBERO seed is the plainest pick-and-place there is: spot the orange-juice
carton among grocery distractors sitting in the open, grasp it, carry it through
free space, release it over an always-reachable static basket. One prehensile
transport; every object is exposed and graspable from step zero.

This task keeps the surface goal (get the juice carton into the receptacle, among
distractor cartons) but replaces the **plan skeleton**:

1. **The target starts inaccessible.** All three cartons ride a turntable inside
   a fixed cylindrical shroud (ring wall r_i = 0.175 m spanning z 0.085–0.260,
   with a roof) whose only opening is one 64° **window**. On reset the juice's
   slot is randomized 45–175° *behind the wall*: there is no line of approach,
   and the roof forbids lifting anything out from above (smoke check 9 shows the
   solve's own extraction drag moving the carton but the wall stopping it inside).
2. **Access is created by servoing a mechanism DOF, not by walking around it.**
   The turntable is a free vertical revolute bearing (no drive, no limits, angular
   damping its only resistance) with a crank arm and a 30 mm knob orbiting *above*
   the roof. The solver must rotate the carousel until the juice's slot faces the
   window — a position-servo on a real dynamic DOF carrying all three cartons.
   Overshoot matters: the rubric's success is live state, so a carousel left
   spinning (or a distractor slung off the disc) revokes it (smoke checks 14, 16).
3. **Extraction is a low through-the-window slide, then the seed's transport is
   the trivial tail.** With the window aligned, a low radial drag (≤ 1.7 N,
   velocity-regulated ~0.10 m/s) slides the carton off the disc rim onto a flush
   porch by friction contact the whole way — the porch underlaps the disc rim so
   there is no ledge. Only the *last* step is the seed's whole task: carry the
   extracted carton and gravity-drop it into the open floor basket.

So the plan skeleton changes from *"grasp exposed target, place into open
receptacle"* to *"index a shrouded rotary magazine to its aperture, extract the
payload low through the window without disturbing the other riders, then deliver"*
— mechanism indexing gates every later step.

## Why strategically different from every examined sibling

- **The seed** (`libero_pick_orange_juice`): all objects exposed, one
  grasp-carry-release; distractors are purely visual. Here the target is
  physically unreachable until a rotary DOF is servoed, and the distractors are
  *dynamic cargo* on the same mechanism that must survive the indexing.
- ***libero_pick_orange_juice_i6*** (same seed): non-prehensile ledge push and
  ballistic drop into a pre-staged mobile basket — the receptacle staging is the
  trick, access is never gated. Here the receptacle is trivial and static; the
  *access* is the task, via a mechanism DOF i6 does not have.
- ***libero_pick_orange_juice_i129*** (same seed): counterweight-tool loading to
  hold a self-closing vault lid open — a tool load/unload cycle. Here there is no
  tool and no self-closing member: the gate is a free bearing that must be
  *position-servoed* to a target angle and parked, with cargo riding on it.
- ***change_channel_i249*** (nearest captive-mechanism neighbour): a captive
  prismatic slider is the goal object itself. Here the DOF is rotary, is never
  the goal, and the payload must *leave* the mechanism through its aperture.
- ***change_clock_i181***: card exchange at a clock face — objects exchanged at
  exposed stations; no shroud, no aperture gating, no ride-along distractors.
- ***living_room…i178***: capsize recovery (re-orientation of the receptacle);
  nothing shrouded, no mechanism.

Same-strategy-different-numbers this is not: the seed and both same-seed siblings
have no shroud, no rotary indexing, no aperture timing, and no
keep-the-distractors-riding clause.

## Solution outline (solve.py, teleport = transport only)

Teleportation is used exactly once, for the free-space carry a Franka performs.
Alignment and extraction run through applied-wrench + friction-contact dynamics.

- **Phase 0 (reset):** settle; juice readback riding its slot at r ≈ 0.105,
  45–175° behind the wall. `SIM_GEN_SCORE` 0.000.
- **Phase 1 (rotate):** torque servo on the turntable about its axle (outer loop
  ω_des = clamp(−1.5·az, ±0.60 rad/s), inner τ = clamp(0.50·(ω_des−ω), ±0.20 N·m)
  — the crank-knob push, ≤ 1.6 N tangential at the 0.125 m orbit). The disc, the
  bearing damping, and the three riding cartons are all live dynamics; exits on a
  60-step streak of |az| < 6° with the wheel still, then torque off — the parked
  carousel HOLDS by damping (null-policy drift < 2° over 400 steps, smoke 8).
  `SIM_GEN_SCORE` ≈ 0.350 (align progress + align latch).
- **Phase 2 (extract):** bang-bang horizontal drag on the carton (≤ 1.7 N at CoM,
  regulated to ~0.10 m/s, slide-not-tip margin asserted in the cfg) along base +x:
  the carton slides across the disc, through the window, onto the flush porch —
  friction contact end to end; cut at r > 0.24. `SIM_GEN_SCORE` ≈ 0.650.
- **Phase 3 (deliver):** the only teleport — the porch carton is carried to a
  hover 3 cm above the basket rim, zero velocity, released; gravity seats it.
  `SIM_GEN_SCORE` 1.000.
- **Phase 4 (persistence):** ≥ 3.5 simulated seconds hands-off; `success()` is
  live state (a spinning carousel or a slumping carton would revert it). Only
  then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0 (initial window error +166.5°), 1 (+77.4°) and
2 (−65.0°) all print the monotone sequence 0.000 → ~0.350 → ~0.650 → 1.000 →
1.000 and `SIM_GEN_SOLVE: SUCCESS` (~20 s wall each).

## Rubric (score 0..1, latched; score == 1.0 iff success())

- Align progress (0.20): scaled running-max reduction of the window error |az|.
- `align_latch` (0.15): the juice's slot has ever settled inside the ±10° band.
- `extract_latch` (0.30): the juice has ever been calm at base-frame r > 0.22
  **and z < 0.30** — low, through the window; a carton held high outside the
  shroud does not count (smoke 11).
- Success (→ 1.0): juice settled inside the basket (below rim, basket frame),
  **both distractors still riding the disc**, carousel at rest, everything
  settled and finite — all live state. Non-success capped at 0.65. Null ≈ 0.

## Embodiment argument (single Franka, parallel jaw)

Base placed ≈ 0.6 m from the carousel axis, offset ~25° from the window azimuth
toward the basket side: the knob orbit (r = 0.125 m), the porch (r 0.153–0.300),
and the basket (r = 0.48 ± 0.06 m, jittered on an arc around the window side) all
fall inside a 0.80 m reach disc, with contact heights 0.15–0.39 m.

- **Crank knob (30 mm square × 70 mm):** orbits at z 0.309–0.388, entirely above
  the roof (top 0.274) — exposed from above at every angle. A 30 mm pinch fits
  the standard Franka jaw; pushing tangentially with ≤ 1.6 N reproduces the
  servo's ≤ 0.20 N·m cap exactly. At that torque the cartons cannot slip on the
  disc (μ·g/r margin asserted in the cfg), so indexing is jerk-safe.
- **Juice carton through the window (6×6×11 cm, 350 g):** the window is ≈ 0.19 m
  wide and 0.157 m tall above the riding surface — room for the gripper to reach
  in and hook or pinch the carton's near face and pull low along the radial with
  ≤ 1.7 N, the exact force the solve's drag applies at CoM. The carton never
  needs to be lifted inside the shroud (the roof forbids it anyway).
- **Delivery:** side grasp on the 60 mm width (within the 80 mm jaw span) from
  the open porch, carry, hover 3 cm over the rim, release — a ~10 cm gravity
  drop, the same transport the teleport stands in for.
- **Execution order (REQUIRED and physically enforced):** rotate → extract →
  deliver. Extract-before-rotate is stopped by the wall (smoke 9: the identical
  drag moves the carton but it stays inside); lifting out through the roof is
  impossible (central hole passes only the axle); delivery requires a carton
  that has left the shroud. And the indexing must be *controlled*: dumping a
  distractor off the disc or leaving the wheel spinning forfeits success.

## Checks (smoke.py — rejection battery, 19 checks)

1. Clean reset: finite state, juice riding its slot misaligned 45–175° behind
   the wall, both distractors riding, everything settled, score < 0.05.
2. Masses: PhysX readback matches authored masses (base 25 / wheel 1.8 /
   cartons 0.35 / basket 1.2 kg — guards the custom-spawner mass trap).
3. Baseline: fresh-reset score ~0, no success.
4. Randomization: base xy/yaw vary across seeds (readback).
5. Randomization: the juice's initial window error varies AND takes both signs.
6. Randomization: the juice's slot assignment varies (all three slots seen).
7. Randomization: basket floor position varies.
8. Null policy: 400 idle steps — score ~0, no success, and the carousel HOLDS
   its angle (drift < 2°): alignment cannot happen by itself.
9. Wall blocks: the misaligned carton under the solve's OWN drag MOVES
   (non-vacuous) but is stopped inside (r_max < 0.22), never leaves low, no
   latch, score stays < 0.30.
10. Window admits: the carousel SERVOED into alignment by crank torque (real
    dynamics), then the IDENTICAL drag takes the carton out onto the porch —
    both latches set, distractors still riding, no success, score ≤ 0.65.
11. Transient guard: the carton held HIGH outside the shroud (r ≥ 0.22,
    z = 0.50) does NOT arm the extract latch — only low-through-the-window.
12. Mid-fall guard: the carton in flight above the basket mouth is not
    in-basket and not success.
13. Acceptance + exactness: gravity-seated carton in the basket, distractors
    riding, all still → success TRUE and |score − 1.0| < 1e−3.
14. Distractor clause: the soda knocked off the disc flips success FALSE while
    the juice is still delivered; score falls to the latched base ≤ 0.65.
15. Restore: the soda stood back on its slot → success TRUE again (14's
    rejection was the distractor clause and nothing else).
16. Liveness: the carousel left SPINNING in the accepted state flips success
    FALSE (score falls to the latched value, not 1.0); once damping parks it,
    success returns — success is live state.
17. Rejection audit: success() was never True at any judged point except the
    constructed acceptance probes.
18. Final: all task-object states finite.
19. Camera: ≥ 20 rgb frames captured across the checks → `frames.npz`.
