# ferry_door_cabinet — the sliding door is the cargo ram: ferry two cups into a sealed roofed bay with the door's own closing strokes, then leave it closed

Package: `sim_gen/tasks_v7/slide_cabinet_open_and_place_cups_i432`
Env: `simgen.ferry_door_cabinet` (scene-level, `robot="null"`)

## Seed provenance

Seed task: `rlbench/slide_cabinet_open_and_place_cups`. In the seed, the robot slides
one cabinet door OPEN — a single articulation act whose only purpose is to get the
panel out of the way — and then reaches into the revealed volume to place
interchangeable cups, judged against a recorded waypoint trajectory (the checker is a
TODO). The door's terminal state is irrelevant, and every placement is a direct
hand-to-goal reach.

## What changed, and why it is strategically different

1. **The panel's role is INVERTED: it is the transport tool, not the obstacle.** The
   cabinet is a low, fully roofed gallery whose storage bay lies at the far end of a
   roofed corridor. The only cup-sized opening in the shell is a roof PORT over the
   corridor, 3–10 cm short of the bay; the bay itself is NEVER hand-accessible (no
   state of the mechanism exposes it — unlike the seed, where opening the door is
   precisely what exposes the goal volume). A cup dropped through the port lands in
   the path of the sliding door — a captive piston filling the corridor cross-section
   to within 2–4 mm (asserted: no cup-sized gap ever exists around it) — and the
   door's own CLOSING stroke is the only way to move it: the leading face rams the
   cup down the roofed corridor into the bay. Cargo moves exclusively while the door
   closes.

2. **The door must be CYCLED, not one-shot displaced.** The port and the bay are at
   opposite ends of the door's duty: staging a cup needs the door retracted clear of
   the port, delivering needs a full closing stroke. So the solver must schedule
   open → drop → close → re-open → drop → close (or a partial-stroke variant that
   stages both cups before one final ram). The seed's "open once, then place
   everything" plan is impossible here, and unlike sibling `i112` (bypass shutter:
   coverage multiplexing of two top openings) or `i195` (a gravity-returned gate
   propped by a wedge tool), the articulated panel here does positive WORK on the
   cargo — every centimetre of cup travel inside the shell is door-driven contact
   dynamics.

3. **The door's final pose is part of the goal.** Success requires the door parked at
   its fully CLOSED stop (leading face 18 mm past the bay line), sealing the
   corridor — the very stroke that delivers the last cup also completes the goal
   state. The door has no spring or detent (smoke check 9: parked mid-track it
   drifts < 2 mm over 240 steps): openness is free to maintain, but worthless until
   the ram plan uses it.

4. **Physics is the judge, not proximity to waypoints:** cabinet-frame membership
   windows whose z band accepts only floor-rest height (rejecting cups on the door
   top at +0.106, on the roof at +0.120, and stacked at +0.100 vs rest +0.040), an
   uprightness cone (a lying cup PASSES the z band — uprightness is load-bearing,
   asserted in `__post_init__`), a joint-readback closed-stop predicate, and a
   stillness counter-latch (30 consecutive quiet steps).

## Teleport-solution outline (transport only — verified on forge)

Teleportation is used ONLY to release cups (zero velocity, upright) ~3.5 cm above the
roof port — gravity carries them through the port onto the corridor floor; the far,
roofed bay is unreachable by any drop. Every load-bearing interaction is real contact
dynamics: the door is driven by a velocity-regulated external force along its
spawn-authored prismatic track (outer position loop → v_des, inner velocity loop →
force, plus a constant feedforward bias toward the target that defeats the P-servo
stall asymptote at the cups' static-friction breakaway; KV·dt/m ≈ 0.29 respects the
one-substep wrench delay; the wrench is zeroed at every park, so stops are true
hard-stop rests). The cups are never pushed by anything but the door's face.

- **P0** settle + readbacks (door start position, cup slots); assert score < 0.05.
- **P1** drive the door to the +stop (port exposed).
- **P2** release cup_a above the port → lands staged on the corridor floor.
- **P3** closing stroke: the door face rams cup_a ~11 cm down the corridor into the
  bay (center −0.046); assert `seated` on readback.
- **P4** re-open to the +stop; assert cup_a stays put in the bay.
- **P5** release cup_b through the re-exposed port.
- **P6** final closing stroke: the door rams cup_b, which shunts cup_a deeper
  (chain push: cup_a → −0.106, cup_b → −0.043) and parks fully CLOSED.
- **P7** ring-down until `success()` holds 120 consecutive steps.
- **P8** hands-off ≥ 3.3 simulated seconds; assert success persists; print
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (latched credit).
**Verified on forge seeds 0, 1 and 2 — all `SIM_GEN_SOLVE: SUCCESS`, scores monotone
0.00 → 0.10 → 0.30 → 0.40 → 1.00, zero persistence flickers.**

## Embodiment argument (single Franka + parallel-jaw gripper)

Base at ~(0.45, 0.15, 0) beside the cabinet (cabinet footprint 42 × 9 cm, 9 cm tall,
at the origin; cup staging row at x ≈ 0.16). Reachability: every manipulated pose lies
0.20–0.55 m from the base at heights 0.03–0.16 m — comfortably inside Franka's
~0.85 m envelope. Graspability: the cups are 56 mm cylinders (< 80 mm jaw opening,
asserted), side-pinched from open ground positions and released upright directly over
the 68 × 72 mm roof port from ~4 cm — an ordinary top-down place; the arm never needs
to enter the shell (nothing fits under the 7 cm roof, by design). The door is driven
by its yellow HANDLE post (16 × 24 mm — pinchable) rising through the roof slot;
every door stroke is a straight horizontal drag of that handle at fixed height, the
easiest primitive a manipulator has. Forces: the worst stroke (final chain ram)
pushes two 60 g cups against floor friction ~0.6 N plus door damping — single
newtons, trivial for the arm. Staging and ramming are strictly sequential single-arm
acts; no "hold this while doing that" conflict ever arises.

## Execution order

`describe()` states the mechanically-forced constraints: a cup can only enter through
the port while the door is retracted clear of it (a drop onto the covered port lands
on the door top), and a cup can only reach the bay by a closing stroke. Free choices:
which cup goes first, one-cup-per-stroke vs. staging both (with a partial pre-push to
clear the port) and ramming once, and how the door is scheduled between strokes. The
solve's order (open → A → close → open → B → close) is one convenient schedule.

## Rubric

- +0.10 per cup that has EVER been staged upright at floor rest in the corridor
  (through the port), latched.
- +0.20 per cup that has EVER been upright at floor rest inside the bay membership
  window (|x| ≤ 0.020, y ∈ [−0.148, −0.040], |z − 0.040| ≤ 0.015, uprightness
  within 30°), latched.
- +0.10 once both cups have been in the bay simultaneously (latched).
- +0.05 once both-in coincides with the door closed (gated on both-in — a door that
  merely starts or is merely driven closed over an empty cabinet earns nothing).
- Partial credit capped at 0.75; `score = 1.0` iff `success()` = both seated ∧ door
  at the closed stop band (d ≤ −0.058) ∧ settled (stillness counter-latch, 30
  consecutive quiet steps).
- Null policy scores ~0.

`__post_init__` honesty asserts (~24): no cup-sized gap around the door (side/top/
bottom slits ≤ 6 mm ≪ cup 56 mm); the port passes a cup while the handle slot never
does (and passes the handle); the open door fully clears the port; the closed stroke
geometrically delivers both cups inside the bay band (rammed −0.046, chained −0.102,
back-wall slack); the z band accepts floor rest AND a lying cup (uprightness
load-bearing) while rejecting door-top/roof-top/stacked heights; a ram push slides
rather than tips a cup (μ·h ≤ r); staged/bay band separation; jaw fit; reset never
writes the door into a stop; ground slots never overlap and clear the footprint.

## Checks (smoke: `SIM_GEN_SMOKE: ALL PASS 18/18`)

1. Reset settles: states finite, cups upright on the ground, door on its track.
2. Fresh reset: score ~0, no success.
3. Randomization: door start position readback spread 80 mm across 6 seeds, always
   on the track.
4. Randomization: cup slot permutation varies (4 distinct / 6), slots never shared,
   x jitter > 10 mm.
5. Null policy 240 steps: score ~0.
6. Sealed port: the solve's own transport over a COVERED port lands the cup ON the
   door top (z +0.106 readback) — not staged, not in the bay, score ~0.
7. Roof denial: a cup rested on the roof directly over the bay's y window is
   rejected by the z band (the bay is never top-loadable).
8. Hard stops are real: a regulated push aimed 36 mm beyond the +stop drives the
   door 116 mm and the stop clamps it (never past stroke + 4 mm).
9. No spring: the door parked mid-track drifts < 2 mm over 240 hands-off steps.
10. Seed-analog ("open the door, place the cups in the revealed volume, walk
    away"): both cups on the corridor floor under the open port → staged credit
    only (0.20), nothing in the bay, NOT success.
11. Door-open near-miss: both cups correctly in the bay but the door 28 mm short of
    the closed band → NOT success, latched 0.50.
12. Latched credit: removing a bay cup keeps the score while both_seated() drops.
13. Doorway loiterer: a cup at rest between the bay band and the staged band
    (y −0.030) is in neither window.
14. Lying cup in the bay: z band PASSES (readback +0.038) and in_bay is True, yet
    uprightness rejects seating.
15. Stacked under the port: the top cup reads +0.100 → z band rejects it while the
    base cup stays staged.
16. Settle gate: the completed arrangement judged the instant door_closed first
    reads True (door still moving at 0.40 m/s) is NOT success; a cup is removed
    before ring-down (battery never succeeds).
17. Rejection audit: `success()` never fired at any judged point.
18. Final state no-NaN.

Smoke also records `frames.npz` (423 × 600 × 960 × 3) from a perspective camera.
