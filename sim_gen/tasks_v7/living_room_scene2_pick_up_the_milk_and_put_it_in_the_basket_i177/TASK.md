# living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket_i177 — Dump Hopper (scene `dump_hopper`)

Get the WHITE milk carton into the basket — but the carton is captive inside a roofed,
caged tipping tray on a pedestal and can never be grasped. The delivery is inverted
logistics: carry the EMPTY BASKET to the green catch mark under the tray's discharge
spout, press the YELLOW lever paddle down and hold it so the tray tips about its hinge
and gravity slides the carton out through its low slot into the basket, then let go —
the back-heavy tray returns level on its own. Finish with the milk at rest INSIDE the
upright, on-floor basket and the RED juice decoy out of it.

**Scene** `dump_hopper` / env `simgen.dump_hopper` (NullRobot, scene-level physics).
**Seed** `libero_90/living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket`.

## Provenance and strategic difference

The seed is a **prehensile pick-and-place of the judged item**: a Franka approaches the
free-standing milk carton on a living-room table, closes its jaw around it, carries it,
and releases it over the basket. Every element of that plan is removed or reversed here:

- **The judged item can never be touched.** The carton lies inside a steel cage on the
  tipping tray: 12 mm roof at 150 mm (a carton stood up inside would need 120 + 35 mm —
  the roof is lower, asserted), full side and back walls, and a single 100 mm-tall ×
  200 mm-wide discharge slot. A LYING carton (65 mm) passes the slot with ≥30 mm margin;
  a STANDING one (120 mm) cannot (asserted both ways). Smoke check 7 hoists it with 15 N
  of lift — it rises 93 mm, the roof caps it at 126 mm (< 140 mm), and it settles back
  caged. There is no approach, no grasp, no carry of the milk.
- **The transported object is the CONTAINER, not the item.** The seed moves the item to
  the container; this task moves the container to the item's forced landing point. The
  basket (300 mm square, 150 mm walls, 8 mm rim — parallel-jaw graspable) must stand ON
  the floor on the catch mark before the dump; a basket merely held aloft over the mark
  earns nothing (smoke 6), and dumping first just strews the milk on the floor for a
  capped 0.45 (smoke 8) — an ordering constraint the seed does not have.
- **The release is a held mechanism actuation, not a jaw opening.** The tray is a real
  `UsdPhysics.RevoluteJoint` (hinge 0.360 m up, travel −0.5°…35° hard stop) with
  authored CoM (x = −55 mm) and diagonal inertia; 1.2 kg gives a 0.647 N·m gravity-return
  bias, so the lever must be pressed AND HELD (≈ 3.5 N at the 270 mm paddle — comfortably
  a one-finger Franka press) through the whole discharge, and letting go re-levels the
  tray on its own (smoke 9 reads −0.51° after the torque clears). The slide is honest
  physics: slick plate μ 0.10 vs carton μ 0.35 combine to ≈ 0.225, and tan 35° = 0.70
  beats that friction angle 2.5× (asserted).

"Seed end state rejected" is N/A by design: the seed's end state IS this task's goal.
What is strategically different is that the seed's MEANS is physically unavailable
(probes 7 and 8 demonstrate it), and the required plan — stage the catch, actuate a
gravity discharge, withdraw — shares no step with the seed's approach-grasp-carry-drop.
Also distinct from the other packages read while building this: `i38 spring_bay`
(elastic compress-then-seat of a held object) and the pen-holder tip-up insertion — no
elastic preload here, no insertion; the core is an ordered container-staging + captive
gravity discharge.

## Mechanism numbers (station frame; station yaw randomized ±180°)

- Pedestal slab half-x 130 mm; hinge axis = tray body-y at height 0.360 m; joint limits
  −0.5°…35°; joint pair collision-filtered; station is a heavy DYNAMIC fixture (30 kg,
  never kinematic, so the hinge anchor follows reset teleports).
- Tray: 300 × 200 mm slick plate (12 mm), 10 mm walls, cage roof at 150 mm, lintel
  leaving the 100 mm slot; YELLOW paddle outboard at x = 270 mm (asserted outside the
  carton's discharge path); authored CoM (−0.055, 0, 0.02), inertia (0.008, 0.013,
  0.014) kg·m².
- At full depression the paddle sits at z = 0.360 − 0.270·sin 35° ≈ 0.205 m and the
  discharge edge at ≈ 0.274 m — both clear the 0.162 m basket rim (asserted ≥ +30 /
  +50 mm), so the basket never fouls the mechanism.
- Catch zone: x ∈ [0.290, 0.400], |y| ≤ 0.080 (green mark at x = 0.345); a basket
  anywhere in the zone clears the slab (asserted). Basket spawn arcs (bearing 30–70°,
  radius 0.42–0.50 m, opposite side from the juice) start well outside the zone
  (asserted), so the null policy scores 0.
- Randomization: station xy ± 50 mm and yaw ± 180°; milk pose in the cage (x ∈ [−75,
  −20] mm, |y| ≤ 45 mm, yaw ± 15°); basket and juice on their arcs. Smoke 3–4 read all
  of these back across 8 seeds.

## Rubric

`success()` = milk **in the basket** (basket-local |xy| ≤ 115 mm, z ∈ [10, 120] mm —
below the rim, asserted) AND **basket upright on the floor** (up-axis within 10°, base
z ∈ [4, 32] mm) AND **milk and basket settled** (lin ≤ 0.05 m/s, ang ≤ 1.0 rad/s) AND
**decoy out** (no part of the juice in or over the basket interior).

`score()` (monotone ladder, `max` of): 0.15 basket staged in the catch zone · 0.45
milk released from the cage · 0.70 released AND in the basket · 1.0 iff `success()`.

## Solution outline (`solve.py`, transport-only teleports)

- **P0** settle + layout readback (tray level within 2°, milk caged, score ≤ 0.03).
- **P1** carry the empty basket: teleport (free transport of a graspable object) to the
  catch mark at the station's heading; assert `basket_in_zone`; score 0.15.
- **P2** press the lever: pure hinge-axis torque on the tray in the **tray body frame**
  (body-y IS the hinge axis at every tilt), bang-bang rate-capped plant — FD tilt rate
  per step, τ = 0.25 N·m when the rate exceeds 0.33°/step else a ramp to τ_max; retry
  ladder (2.2 body → 2.2 world → 3.2 body → 4.5 world) with a stall probe at step 479
  and a missed-landing re-aim + `set_state` rollback. The carton slides out and drops
  in; assert released ∧ in_basket; score 0.70. (Body frame matters: the pod's
  world-frame wrench drag reference is captured at the FIRST application per body and
  goes stale across resets — a world-frame torque loses its hinge component after a
  new station yaw.)
- **P3** let go: clear the wrench, the back-heavy tray gravity-returns below 3°; wait
  for a 60-consecutive-step success streak (a single instantaneous reading can land at
  a velocity turning point while the carton still creeps), then score 1.0.
- **Persistence**: 400 hands-off steps (3.3 s) with success re-checked every step, then
  `SIM_GEN_SOLVE: SUCCESS`. Watchdog `threading.Timer` + `os._exit` hard exit.

The milk is never teleported and never touched by anything but the tray, gravity, and
the basket.

## Embodiment (single Franka, parallel jaw, one base pose)

Base ≈ 0.75 m out along the discharge axis reaches everything: the basket rim (8 mm
wall < jaw stroke; carry ≤ 0.9 m), the paddle (a ≈ 3.5–7 N downward press at 0.21–0.36 m
height, well inside the arm's wrench envelope), and the juice if it ever needed moving.
Reaching IN for the milk is barred by geometry, not fiat: the carton's front face sits
≥ 100 mm behind the lintel while a Franka finger is ~50 mm long, and the roof denies
any top-down approach.

## Execution order

`scene.py` (geometry + honesty asserts in `__post_init__`) → `solve.py` iterated on the
forge until the press plant was robust → rubric finalized (streak/settle gates) →
`smoke.py` rejection battery. No check was ever weakened to make a run pass; the two
live failures (world-frame torque stalling after a reset; a 1-step persistence flicker)
were fixed in the solver (body-frame torque; P3 success streak gate).

## Smoke battery (19 checks, rejection-driven)

1. reset settles: finite, tray level, milk caged · 2. baseline score ≈ 0 · 3. station
xy + yaw randomization readback (8 seeds) · 4. milk/basket/juice poses vary · 5. null
policy: 300 idle steps, score ≈ 0 · 6. basket POSED aloft over the mark → no zone
credit · 7. captivity: 15 N lift × 150 steps hoists the carton (non-vacuous, +93 mm)
but the roof caps it (< 140 mm) and it re-settles caged · 8. out-of-order dump with no
basket → carton on the FLOOR, score 0.45, no success · 9. gravity-return readback
(< 3° after the torque clears) · 10. carton leaning on the basket's OUTER wall → 0.45
only · 11. basket on its SIDE with the carton in its mouth → upright gate rejects ·
12. airborne basket judged immediately → on-floor gate rejects · 13. acceptance:
basket on the mark + lever press → success TRUE · 14. juice posed over the rim →
success flips FALSE (decoy clause) · 15. juice removed → success returns TRUE ·
16. settle gate: delivered carton kicked and judged immediately → rejected ·
17. wrong object: JUICE in the basket, milk caged → ≤ 0.16 · 18. audit: success() was
never True at any judged point except the two constructed acceptance probes ·
19. final no-NaN. Frames (294, 600×960) → `frames.npz`.

## Verification record (forge, 2026-08-08)

- `solve --seed 0`: rc=0, `SIM_GEN_SOLVE: SUCCESS`, 17.2 s. Scores 0.000 → 0.150 →
  0.700 → 1.000 → 1.000 (non-decreasing). Layout yaw +170°; press held +35.00°;
  landing at station (+0.372, +0.081); tray returned to −0.50°.
- `solve --seed 1`: rc=0, SUCCESS, 17.1 s (landing +0.377, +0.010; return −0.50°).
- `solve --seed 2`: rc=0, SUCCESS, 17.0 s (landing +0.400, −0.051; return −0.50°).
- `smoke`: rc=0, `SIM_GEN_SMOKE: ALL PASS 19/19`, 41.8 s, 294 frames saved.
