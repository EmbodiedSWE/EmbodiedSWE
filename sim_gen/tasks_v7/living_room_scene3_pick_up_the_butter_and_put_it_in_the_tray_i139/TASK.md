# Beam-balance tray: weigh the butter, then level the scale

Task id: `living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray_i139`
Env id: `simgen.beam_balance_tray` (scene `simgen.beam_balance_tray`, robot `"null"`)

## Seed provenance

Seed: `roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray.py`
(LIBERO-90 `living_room_scene3`, "pick up the butter and put it in the tray").
The seed is a passive pick-and-place: a butter stick and four distractor cans sit on a table
next to a static `wooden_tray`; `_terminated` fires the moment the butter's position enters the
tray's `contain_region` bounding box. The tray has no dynamics of its own and placement alone is
the entire task.

## What changed and why it is strategically different

| Axis | Seed | This task |
| --- | --- | --- |
| Tray | Static open box; any pose inside the bbox terminates | The "tray" is one **pan of a two-pan beam balance** on a revolute pivot; loading it tips the beam to its +18° stop |
| Success | Geometric containment, instantaneous | Butter at rest **in the tray pan** AND beam **level within ±12°** AND everything settled — the beam's own damped pendulum dynamics must be brought back to equilibrium |
| Objects | 1 butter + irrelevant can distractors | 1 of **3 butter size classes** (150/300/600 g, equal density — size is the visible mass cue) is present per episode, plus **3 candidate counterweight cubes** (150/300/600 g), only one of which matches |
| Cognition | None (any placement works) | **Measurement/selection**: read the butter's size class, select the mass-matched cube, load the opposite (ballast) pan |
| Seed's plan | Sufficient | **Demonstrably insufficient**: butter-in-tray with no counterweight leaves the beam pinned at its stop (17–18° > the 12° gate) — constructed and rejected in smoke check 5, and asserted non-success mid-solve (P1) |

The same-seed-family sibling `..._pick_up_the_cream_cheese_..._i33` (sealed pantry, one-way
flap, push-through) shares no mechanism with this task: that one is about forcing entry through
a compliant barrier; this one is about a two-sided equilibrium the agent must actively restore
by choosing the right second object. No other constructed task in the corpus uses a mass
measurement / beam balance.

Honest torque budget (L = 0.30 m arms, restoring k = M·g·d ≈ 1.20 N·m per sin(tilt),
stops at ±18°): the smallest butter alone (150 g at worst-case inward arm 0.24 m) demands
sin θ = 0.294 → 17.1° > the 12° gate; the smallest wrong-mass pairing (Δm = 150 g at 0.30 m)
demands sin θ = 0.368 → 21.6°, i.e. past the stop. The correct pairing with realistic ≤15 mm
placement offsets settles at ≤8.5° < 12°. High-friction pan floors (μ_s = 0.75 > tan 18°) keep
loads seated on tilted pans, so credit cannot leak via sliding.

## Solution outline (solve.py, teleport = transport only)

- **P0** settle 180 steps; assert the empty beam self-centres |tilt| < 3°, no success,
  score ≈ 0. `SIM_GEN_SCORE 0.0000`.
- **P1** teleport the present butter to free air **above the tray pan's scoring volume**
  (pan floor + 95 mm, i.e. above the z gate at floor + 90 mm) in the beam's current frame,
  zero velocity; gravity loads it (~1.4 m/s at gate entry, faster than the 0.10 m/s at-rest
  latch). The beam swings to its +18° stop. Assert in_tray, tilt > 12°, **success is False**
  (the seed's plan fails here). `SIM_GEN_SCORE 0.2000`.
- **P2** select the counterweight by **physical mass match** (cube index whose mass equals the
  butter's — visually, the equal-density size cue); teleport it above the ballast pan's scoring
  volume **in the tilted beam frame**; hands-off while contact + pivot dynamics level the beam.
  `SIM_GEN_SCORE 1.0000`.
- **P3** hands-off persistence ≥ 3.3 simulated s; success must hold. `SIM_GEN_SCORE 1.0000`,
  then `SIM_GEN_SOLVE: SUCCESS`.

Scores are non-decreasing (in_tray / on_pan partial credit is latched at 0.20 each, cap 0.40;
1.0 iff live success). No forces are ever applied; no write ever places an object inside a
rubric volume.

## Franka feasibility

A single Franka based near the world origin facing +x reaches everything: the item table slots
sit at base-local (0.0, 0.24) and (−0.16..0.16, 0.34..0.42) on the ground plane, and the two
pans at (0.50, ±0.30) rotated by the episode yaw, at pan-floor height ≈ 0.29 m — all inside a
0.85 m reach envelope. Grasps: the butter bars are 75–115 mm long but only 38–57 mm wide and
30–45 mm tall, so a top pinch across the width fits the 80 mm parallel jaw for every class; the
cubes are 38–61 mm across, also within jaw span. Placement is a low-precision drop into a
130 mm-square walled pan with ≥ 35 mm clearance over the walls. No force control is needed —
both load-bearing interactions are "release above the pan and let it seat", exactly what the
teleport solution demonstrates. Required order: butter first or cube first both physically work,
but success requires BOTH pans loaded correctly; the solve does butter → cube. **No particular
order is required by the rubric** (only the settled end state is scored).

## Randomization

Per episode: scale-base yaw 90° ± 25° plus ±4 cm base translation; butter class uniform over
{0,1,2} (absent classes parked in a far depot); the 3 cubes permuted over 3 table slots with
±2.5 cm jitter; butter slot jittered ±2.5 cm. Smoke check 3 verifies ≥2 butter classes and ≥2
cube-slot assignments across 12 resets; check 2 verifies pose deltas across seeds.

## Smoke battery (11 rejection-only checks)

1. Settle / no-NaN; empty beam level; score ≈ 0.
2. Randomization readback: base yaw/pos and butter pos differ across seeds.
3. Class + slot-permutation coverage over 12 resets.
4. Null policy 240 steps → score ≤ 0.01, no success.
5. **Seed strategy**: butter dropped into the tray pan, nothing else → beam at stop
   (tilt > 12°), in_tray credit only (score ≈ 0.20), NOT success.
6. Wrong counterweight (too light): 600 g butter vs 300 g cube → still past the gate.
7. Wrong counterweight (too heavy): 150 g butter vs 300 g cube → beam tips the OTHER way.
8. Mirrored pans: butter on the ballast pan + matched cube on the tray pan → beam level but
   NOT success (butter not in the tray).
9. Cube-only: matched cube on the ballast pan, butter untouched → on_pan credit only, no success.
10. Overload: butter in tray + ALL three cubes on ballast → past the stop the other way,
    score capped ≤ 0.40.
11. `frames.npz` rendered and saved to CWD (> 10 frames).

Prints `SIM_GEN_SMOKE: ALL PASS 11/11`.

## Execution order declaration

Success is a settled end-state predicate; the rubric imposes **no step ordering**. The solve's
butter-then-cube order is one valid order; cube-then-butter also reaches the same end state.
