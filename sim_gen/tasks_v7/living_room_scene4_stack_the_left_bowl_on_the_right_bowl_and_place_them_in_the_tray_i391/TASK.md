# Hitch-and-tow: pin-coupled freight delivery (`simgen.hitch_tow`)

## Seed provenance

Derived from
`libero_90/living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray`.
The seed is a two-stage *stack-then-place* task: pick the left bowl, stack it on the
right bowl, then move the stacked pair into a tray — a nesting/containment plan in
which the two payloads become one composite by gravity alone and the goal region is
an open container.

## What changed, and why this is strategically different

The seed's plan skeleton is **grasp A → rest A on B → carry (A on B) → set down in
open tray**; every joint between objects is an unsecured gravity stack, every object
is free and reachable, and order barely matters (you could put the right bowl in the
tray first and stack inside it).

This task replaces every one of those elements:

1. **Composite by mechanical fastener, not gravity.** The two carts become one unit
   only when a shear pin threads two aligned bores. A gravity "stack" of the plates
   transmits no tow force — pulling the tractor without the pin just separates the
   carts (smoke check 7 proves it, and the rubric's `coupled` predicate reads the
   pin's live pose inside both bore volumes, not proximity of the carts).
2. **1-DOF rail kinematics instead of free 6-DOF payloads.** Both carts ride
   spawn-authored D6 joints locked to translation-x. Nothing is carried through the
   air; the payload is *dragged along a constrained path* by a force-bearing
   linkage. The seed has no joints at all.
3. **Reachability asymmetry forces the plan.** The trailer body spawns fully under
   a low tunnel roof (roof z 0.20–0.22, trailer top below it): it cannot be grasped,
   lifted, or pushed directly. Only its tongue plate protrudes. The *only* way to
   move it is to mate the tractor's coupler plate onto the tongue, pin them, and tow.
   In the seed everything is in the open.
4. **Goal region is a flat dock pad, not a container**, and the goal predicate is
   about *how* the trailer got there: a tow odometer (`tow_travel`, per-step-clamped
   forward progress counted only while `coupled_now`) must exceed 0.40 m. A trailer
   that arrives any other way — teleported, shoved unhitched, carried — scores
   nothing (smoke checks 5, 10, 13).
5. **Order is physically forced, not conventional.** Mate before hitch: the pin can
   only drop through *aligned* bores, and alignment only exists at the stop-block
   hard stop. Hitch before tow: without the pin the tractor departs alone. And the
   hitch must *survive* the tow and the final settle — yanking the pin after arrival
   fails (`coupled` is judged at the settled end state).

It is also unlike the other tasks authored in this campaign that I have seen: i194
(“Pagoda Relay”, chained see-saws), i367 (umbrella bayonet twist-lock drawer), and
the pen-holder exemplar — none involve a rail-constrained vehicle pair, a
shear-loaded fastener, or a travel-odometer rubric.

## Scene summary

- **bed** (kinematic): rail slab, x ∈ [−0.70, 0.80].
- **yard** (kinematic): tunnel (roof z [0.20, 0.22], walls |y| ∈ [0.075, 0.095],
  x ∈ [−0.62, −0.26]) over the trailer parking strip; green dock pad x ∈ [0.12, 0.42];
  wooden pin pedestal at (0.10, −0.20), socket at z = 0.120.
- **tractor** (blue, 0.8 kg, D6 transX ∈ [−0.14, 0.64]): body + yellow grasp mast +
  rear overhanging grey coupler plate (z 0.084–0.096) with a square bore at
  x_rel −0.08.
- **trailer** (crimson, 0.6 kg, D6 transX ∈ [−0.42, 0.34]): roofed crate body +
  protruding tan tongue plate (z 0.069–0.081) with a matching bore at x_rel +0.22
  and a crimson stop block at the tongue root. At the stop block the two bores are
  coaxial (`mate_dx = 0.30`).
- **pin** (free rigid, 0.05 kg): steel shank r 0.007 × 0.040 + red square cap
  0.036; spawns upright in the pedestal socket. Seated: cap on coupler plate,
  shank threading both bores (pin z = 0.076).

Randomization (per seed): trailer spawn x ∈ [−0.41, −0.37], tractor spawn
x ∈ [0.02, 0.14], pin cap yaw free. Verified by readback spread in smoke check 3.

## Rubric

- `mated` (latched, settled): coupler plate over the tongue at the stop,
  |rel − mate_dx| ≤ 10 mm → 0.25.
- `coupled` (latched while true now): pin shank inside BOTH bores (xy ≤ 9 mm,
  z window ≤ 8 mm, tilt ≤ 20°) → +0.35 (score capped 0.60 before success).
- `success` = `docked` (trailer x ≥ 0.20) ∧ `coupled` (live, at the end) ∧
  `towed` (odometer ≥ 0.40 m accumulated only while coupled, per-step clamp
  4 mm — a teleport credits at most one step) ∧ `settled` (pose-stillness
  counter-latch, 30 steps). Score 1.0 iff success, else latch sum.

## Solution outline (solve.py, teleport-for-transport only)

- **P0** settle, spawn-band readback asserts, score < 0.05.
- **P1 MATE**: velocity-capped force servo (±3 N on the tractor body) pushes the
  tractor −x until the coupler plate rides over the tongue and `mated_now` holds a
  40-step streak; then a slow shove parks the mated pair on the trailer's −x joint
  stop (bores stationary for the drop) and holds a 0.4 N press. Score 0.25.
- **P2 HITCH**: pin teleported (transport only) to a hover 15 mm above the coupler
  plate over the live bore center, upright, zero-vel; then pure contact physics: xy
  PD onto the live bore + downward velocity servo presses it through both bores
  (4 mm slop) until `coupled_now` streaks 30 steps; stall-detect + re-lift retry
  (≤3 attempts). Wrenches off, verify the latch survives hands-off. Score 0.60.
- **P3 TOW**: +x velocity-capped force servo on the tractor (≤6 N, vcap ramps to
  0.10 m/s); the pin, loaded in shear, drags the trailer out of the tunnel to the
  pad (tow_travel ≈ 0.69 m ≥ 0.40); brake and release.
- **P4/P5**: ring-down to settled success, then ≥3.3 s hands-off persistence.
  `SIM_GEN_SOLVE: SUCCESS` only if success still holds. Passes seeds 0 and 3.

## Embodiment argument (single Franka, OSC, base ≈ (0.15, −0.55, 0))

Every solve interaction maps to a standard Franka contact skill inside a
~0.85 m reach envelope centered on the yard:

- **Mate / tow** = planar push/pull of the tractor via its yellow grasp mast
  (a 0.03 m square post at z ≈ 0.16–0.26, in the open, graspable from −y with a
  top-down or side pinch). The servo forces used (≤6 N) are far under Franka's
  payload; the rail constrains the cart so only x-force fidelity matters.
- **Hitch** = pick-and-insert: pinch the 36 mm red cap (pedestal top at z 0.10,
  unobstructed), transport ~0.3 m, press straight down with 4 mm total slop and
  a stop (cap lands on the plate) — a peg-in-hole easier than most NIST board
  insertions. The teleport in solve.py replaces exactly this transport.
- **No interaction requires reaching under the tunnel roof** (the trailer is towed
  out, never touched), and nothing requires two hands or regrasp-in-air.

## Declared execution order (physically forced)

MATE → HITCH → TOW. Hitch-before-mate is impossible (bores only align at the hard
stop; elsewhere the pin lands on solid plate/tongue — smoke check 9). Tow-before-
hitch moves nothing (check 7). Unhitch-then-arrive fails `coupled` (check 13);
arrive-without-travelling-coupled fails `towed` (checks 5, 10).

## Checks (smoke.py — 15)

1. settle / no-NaN / spawn + socket readback bands
2. fresh-reset score ≤ 0.02
3. randomization by readback across seeds 21–26 (trailer, tractor spreads; pin yaw)
4. null policy 240 steps → score ≤ 0.02
5. free-transport cheat: both carts teleported to goal poses, settled → not towed,
   score ≤ 0.02 (kills the seed's "just move things to the goal" strategy)
6. mate mechanism honest-positive: servo mate → score ≈ 0.25 (moved-guard asserted)
7. tow-without-pin: tractor departs ≥ 0.15, trailer stays (≤ 0.02), tow ≤ 0.005
8. honest hitch → 0.58 ≤ score ≤ 0.62, not success
9. pin-perch near-miss: pin standing on the plate 30 mm off-bore → not coupled
10. teleport-the-rig cheat: hitched rig moved as a unit to the dock → docked ∧
    coupled ∧ settled but towed=False, score ≤ 0.62
11. honest tow stopped short of the pad (0.15 < 0.20) → not docked
12. settle gate: judged while still rolling → not success
13. arrive unhitched: pin yanked mid-tow, trailer coasts onto the pad → not coupled
14. never-success audit across all constructed states
15. final no-NaN

`SIM_GEN_SMOKE: ALL PASS 15/15`; frames.npz recorded (960×600, up to 500 frames).
