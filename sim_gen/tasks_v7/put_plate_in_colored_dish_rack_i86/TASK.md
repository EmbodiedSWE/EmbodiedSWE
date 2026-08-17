# carousel_dish_rack — align the blue bay, feed the plate through the one gate, stow it

Package: `sim_gen/tasks_v7/put_plate_in_colored_dish_rack_i86`
Env: `simgen.carousel_dish_rack` (scene-level, `robot="null"`)

## Seed provenance

Seed task: `rlbench/put_plate_in_colored_dish_rack`. In the seed, the robot lifts THE
plate off its stand and lowers it into the open slot of a static dish rack whose slots
are color-named — one prehensile transport into a fixed, sky-open target, driven by
recorded waypoints, with the color binding given for free by the variation index.

## What changed, and why it is strategically different

1. **The rack is a mechanism the solver must drive, not a static target.** The four
   color-coded wedge bays live on a free-spinning covered carousel (spawn-authored
   revolute joint, kinematic housing). Only the bay currently facing the single side
   GATE is reachable, and the blue bay starts a random signed 55–180° away. The goal
   location must be actively *positioned* (rotate by the compass-rose knobs) before it
   can be used, and *re-positioned* after (stow ≥ 45° away) — the target pose changes
   twice during the episode by the solver's own actions.

2. **The plate is never lifted.** The housing is closed from above by a solid roof
   (the roof-to-fin gap, 10 mm, is thinner than the 12 mm plate — asserted in
   `__post_init__` — so no top drop-in and no bay-hopping), and the feed shelf top
   stands 1 mm above the turntable. The insert is a planar shuffleboard slide through
   the gate aperture: a non-prehensile push, where the seed is a pick-and-place. The
   seed's entire strategy (approach from the sky) is physically blocked, and smoke
   check 11 proves it (the dropped plate lands ON the roof and is rejected).

3. **A forced three-stage order, enforced by physics rather than the rubric.**
   Align → insert → stow: the gate is the only way in, only the aligned bay is
   reachable, and stowing means rotating the loaded bay away — the turntable carrying
   the plate by floor friction (a real transported contact, verified by readback:
   carousel-frame drift < 2 mm over a 90° spin). The seed has a single stage.

No corpus task read this session shares this plan shape: `place_cups_i82` is a
mass-partition statics puzzle, `empty_dishwasher_i23` a captive-slide force servo,
`stack_wine_i48` hidden-mass probing; none has a rotate-position-then-feed-through-
an-aperture-then-rotate-away mechanism.

## Solution outline (NO teleports — verified on forge)

`solve.py` uses zero teleports: the plate starts on the shelf and every phase is
applied forces/torques through contact dynamics.

- **P0** settle + readbacks (initial offset, plate pose); assert misaligned, score ~0.
- **P1 ALIGN** — z-torque velocity servo on the carousel (τ = clamp(0.6·(ω_des − ω)),
  ω_des = clamp(3·err, ±1.5)) until the blue bay faces the gate (|offset| ≤ 6°). Pure
  z-torque on a yaw-only body is frame-drag invariant; gain (not cap) escalates on
  stall. Score → 0.15.
- **P2 INSERT** — shuffleboard push: horizontal CoM force on the plate, velocity-
  regulated at 0.10 m/s toward the bay, lateral PD centering on the gate line, while a
  gentle torque servo holds the carousel aligned. The pod force-frame mode is PROBED
  from measured progress (moving away toggles the encoding). The push stops at
  carousel-frame r = 0.100 — short of hub contact at 0.085, so the mechanism is never
  pressed. Seat latch (24 consecutive slow in-bay steps) fires. Score → 0.55.
- **P3 STOW** — the same torque servo rotates the loaded carousel to ±90° (continuing
  the direction it came from); the platform carries the plate by friction
  (centripetal demand ~0.06 m/s² vs μg ~5 m/s²). Wait for `success()` to hold 120
  consecutive steps. Score → 1.00.
- **P4** hands-off ≥ 3.3 simulated seconds; assert success persists; print
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (latched credit).
**Verified on forge seeds 0, 1 and 2 — all `SIM_GEN_SOLVE: SUCCESS`, initial offsets
+104.9°, +166.3°, −108.2° (both directions exercised), scores monotone
0.00 → 0.15 → 0.55 → 1.00, zero persistence flickers, no frame-drag toggles, no
servo stalls, ~20–25 s wall clock each.**

## Embodiment argument (single Franka + parallel-jaw gripper)

Base at ~(0.55, 0, 0) facing −x, in front of the shelf. Reachability: the farthest
task point (a pointer knob on the far side of the axle) is ≤ 0.65 m away; the plate's
shelf start is 0.28–0.33 m — all inside Franka's ~0.85 m envelope. **Rotating:** the
compass rose stands ABOVE the roof in free air (knobs at 0.24–0.28 m height, ≥ 70 mm
of clearance over the roof plane); each knob is a 22 mm-diameter × 40 mm cylinder — a
comfortable pinch for the 80 mm jaw, or a closed-jaw fingertip side-push, re-grasping
per quarter turn. Its own arm color names the bay, so perception is direct.
**Inserting:** a fingertip push on the plate rim slides it flat along the shelf
(shelf top 1 mm ABOVE the platform — the plate steps down, never climbs). The rubric
accepts the plate up to carousel-frame r = 0.120, where its rear edge sits exactly at
the wall line: the fingertip never needs to pass the gate plane — pushing flush is
enough (the demonstrated slide to r = 0.100 needs only ~20 mm of fingertip reach
through the 13 cm-wide, fully open gate). **Stowing** is the same knob action again.
The gate aperture is generous for the plate (126 mm chord vs 90 mm plate, ±18 mm of
lateral slack), and the demonstrated 0.10 m/s slide is well within a gentle arm push.

## Execution order

REQUIRED, and enforced by physics (not the rubric): insert is impossible while
misaligned (the gate frames a wall or a wrong bay), and stow is only meaningful after
insert. Within that: the align rotation may go either way (shortest or long way
around), the stow rotation may go either way, and pausing between stages is harmless.
All of this is stated in `describe()`.

## Rubric

- +0.10 once the plate has EVER rested fully inside the carousel (any bay:
  carousel-frame r ≤ 0.125, z window [0, 0.030] over the floor), latched.
- +0.15 once the blue bay has EVER been aligned with the gate (|offset| ≤ 15°;
  reset guarantees the episode starts ≥ 55° away, so a null policy can never earn
  it), latched.
- +0.30 once the plate has rested in the BLUE wedge (radial band [0.055, 0.120] +
  angular half-window 30° + floor z window) for 24 CONSECUTIVE slow steps — a
  fly-through never latches — latched.
- Partial credit capped at 0.55; `score = 1.0` iff `success()` (in_blue ∧ stowed
  |offset| ≥ 45° ∧ settled, live). `settled()` is a stillness counter-latch (24
  consecutive steps under the velocity gates), so a spinning carousel judged
  mid-rotation never counts. Null policy scores ~0 (the blue bay starts stowed —
  stow alone earns NOTHING).

Scene `__post_init__` carries honesty asserts: gate chord passes the plate with
margin; roof-to-fin gap thinner than the plate (no top entry, no bay hopping); the z
window accepts floor rest and rejects roof/fin perches; the radial band rejects a
threshold-straddling plate; the sampled initial offset can never start aligned;
align/stow hysteresis ≥ 15°.

## Checks (smoke: `SIM_GEN_SMOKE: ALL PASS 16/16`)

1. Reset settles: states finite, plate flat on the shelf, blue bay starts misaligned
   (readback ≥ 40°), settled.
2. Fresh reset: score ~0, no success.
3. Randomization: initial offset varies across 8 seeded resets (spread > 60°, both
   signs), always inside the configured [55°, 180°] band, view yaw agrees with the
   cached sample.
4. Randomization: plate shelf xy jitter and spawn yaw vary (readback).
5. Null policy 240 steps: score ~0 (the blue bay starts stowed — stow alone earns
   nothing).
6. Seed-analog (insert without aligning): plate seated in the bay at the gate (not
   blue) → entered yet NOT in_blue, not success, score ≤ 0.11.
7. Mechanism reality: a 0.12 N·m z-torque (bang-bang speed-capped) spins the loaded
   carousel ~90° by yaw readback and the platform CARRIES the plate (carousel-frame
   drift < 20 mm, world motion > 60 mm) — a working lazy susan.
8. Wrong bay stowed and settled → still not success, score ≤ 0.11 (color binding).
9. Stopped halfway: blue bay parked at the gate, plate on the shelf → aligned credit
   only (score ≤ 0.16), not success.
10. Jammed in the gate: plate straddling the threshold at r ≈ 0.135 → not entered,
    not in_blue, not success.
11. Covered top blocks the seed's strategy: plate dropped from above the blue bay
    lands on the roof/compass rose (z readback 0.230), rejected by the z window, not
    success.
12. Out of order: plate on the blue bay floor but the bay still at the gate → not
    success, score capped at 0.55.
13. Latched credit: removing that plate keeps the latched 0.55 while live in_blue
    drops — score never decreases.
14. Settle gate: plate in the blue bay, bay stowed, carousel still spinning (ω
    readback above the gate) → not success until ring-down; probe dismantled first.
15. Rejection audit: `success()` never fired at any judged point.
16. Final state no-NaN.

Smoke also records `frames.npz` (600 × 960 × 3 frames) from a perspective camera.
