# cellar_roll_stow — unbar the low wine locker, roll the bottle in, re-bar the slot

`sim_gen` task `libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_i250`
· env id `simgen.cellar_roll_stow` · robot `null` (scene-level task)

## Seed provenance

Derived from RoboVerse `libero_90/kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet`
(`roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet.py`):
pick an asset-loaded wine bottle off the table and lower it **upright** into the open
**bottom drawer** of an articulated white cabinet; success is a static bbox test on the
bottle centre in drawer-local coordinates.

What is kept from the seed: a wine bottle must end up **contained inside a low
receptacle compartment**, and a movable part of the receptacle governs access.

## What changed and why it is strategically different

| axis | seed | this task |
|---|---|---|
| receptacle | articulated cabinet drawer (joint) | rigid twin-bay **wine locker**, no articulation anywhere |
| access mechanism | drawer slides open passively | loose **red bar** seated in U-notches across the target ramp; must be lifted out AND **seated back after** stowing |
| insertion motion | carry upright, lower in from above | **impossible here**: bay interior ≤ 93 mm < 175 mm bottle, roof blocks top insertion; the only way in is a ~84 mm letterbox slot — reorient the bottle to **horizontal** and **gravity-roll** it down the ramp through the slot |
| target identification | named fixture | by **mechanism state**: the barred bay is the target; the identical open bay is a decoy |
| judging | static bbox containment only | latched staged rubric (unbar → gated approach → stowed-lying → rebar-while-stowed) + settled end-state success with bar-restored clause |

Also different from its sibling `..._i237` (tilt-out bistable bin from the same seed):
i237's mechanism is a torque-driven articulated door with bistable rest states; here
there is **no articulation and no applied force at all** — the "mechanism" is a free
rigid body (the bar) plus gravity, the delivery is a rolling transit rather than a
vertical drop, and the mechanism must be **restored** (bar re-seated) after use, which
i237 does with a door-close torque, and this task does with a placement + drop-seat.

Metric interlocks that force the strategy (all readback-verified in geometry):
- bay interior height 88 mm (floor→roof) and slot 84 mm < bottle Ø60 × 175 mm length →
  the bottle fits **only lying down**, entering **only through the slot**;
- seated bar leaves 20 mm below and 40 mm above itself, both < the 60 mm body →
  **no roll past the seated bar**, in either gap; the bar top sits 10 mm above the
  rolling bottle's centre, so the bottle cannot vault it (energy check ~2× margin);
- bay floor pitched 3° inward → a bottle that clears the sill **parks itself** at the
  back wall (no flat-sill parking);
- prong gap 34 mm vs 20 mm bar → the drop-seat has ±7 mm lateral tolerance.

## Execution order (declared)

Mechanism-forced, in this order:
1. **Unbar** — lift the red bar out of the target bay's notches, set it aside (the
   approach/stow path is physically blocked until this happens; approach credit is
   gated on it).
2. **Reorient + deliver** — lay the bottle on its side on the target ramp, axis across
   the ramp, release; gravity rolls it through the slot and parks it at the back wall.
3. **Re-bar** — return the bar and drop-seat it back into the same notches (rebar
   credit only accrues while the bottle is currently stowed, so re-barring first earns
   nothing and re-blocks the slot).

## Teleport-solution outline (solve.py)

Teleports are transport only; every load-bearing interaction is contact dynamics:
- P0 reset + settle 90 steps: bar seated (assert), bottle standing outside, score ~0.
- P1 UNBAR (transport): one pose write parks the bar on the open floor at
  (0.08, ±0.35). Legitimate: the bar rests freely in upward-open notches — lifting it
  out is unobstructed carrying.
- P2 LAY (transport): one pose write puts the bottle lying (axis ∥ y) 4 mm above the
  target ramp at x = 0.315 — asserted OUTSIDE the containment box; only gated
  approach credit can result.
- P3 ROLL-IN (contact dynamics): release → gravity carries it down the 7.4° ramp
  (the body carries a slightly-proud exact-capsule contact band with a slick
  min-combine material — the GPU contact solver caps a driven roll-from-rest at a
  few mm/s, so the lying bottle glides on the band instead of fighting that
  artifact), through the slot, onto the 3° inward floor, parks near the back wall;
  streak-gated settle; no pose write from release to verdict.
- P4 RE-BAR (transport + contact dynamics): pose write to a level hover in the notch
  gap at z = 0.070, ABOVE the seat window (asserted not seated) → release → gravity
  drops it ~15 mm onto the post tops where it seats; settle. success() first True here.
- P5 persistence: ≥ 3.3 simulated seconds hands-off, success must hold →
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary; monotonicity asserted
(observed ≈ 0 → 0.19–0.22 → 0.28 → 0.60 → 1.0; all partial terms latched).

## Embodiment argument (single Franka, parallel-jaw gripper, OSC)

Base at the world origin, facing +x. All interactions lie at radius 0.24–0.55 m,
heights 0–0.16 m — inside a Franka's comfortable dexterous shell.
- **Bar** (Ø-equivalent 20 mm square, 320 mm long, 0.15 kg): top-down pinch anywhere
  along its free middle span (the notches grip only the last ~30 mm at each end; the
  span between posts is fully exposed, 20 mm wide — an easy parallel-jaw target).
  Lift straight up 40 mm (clears the 35 mm prongs), carry aside, set down. Re-seat:
  hover the bar level over the notch line and lower/release — the ±7 mm x-tolerance
  between prongs and the ±50 mm y window make the drop-seat forgiving.
- **Bottle** (0.30 kg): grasp the standing bottle around its 24 mm neck (ideal jaw
  width), lift, reorient wrist to horizontal (a pure wrist-roll — the same regrasp
  every pour uses), lay it on the upper ramp between the side rails, release. The
  task then completes itself by gravity: no in-slot manipulation is ever needed, which
  is exactly why the letterbox (too low for any wrist) is fair.
- Nothing requires two arms, force beyond ~3 N, or reaching inside the bay (the roof
  overhang region x > 0.49 never needs to be entered by the gripper).

## Rubric

`score() = 0.15·unbarred + 0.20·approach_max (gated on unbarred) + 0.25·stowed +
0.25·rebar_max (counted only while stowed)`, all latched, clamp 0.85; exactly 1.0 iff
`success()`: bottle lying at rest inside the TARGET bay ∧ bar seated back in the
target notches (pose + level-alignment window) ∧ everything settled. Null policy ~0.

## Checks (smoke.py — rejection battery, recorded to frames.npz)

1. settle/no-NaN: reset finite, bar seated, bottle standing outside, still
2. reset score ~0, no success
3. randomization readback: barred side flips; bar really sits on the target side
4. randomization readback: bottle spawn jitter > 4 mm; bar seat-y jitter real
5. null policy: 240 idle steps → score ~0, no success
6. bar blocks the roll: released bottle is STOPPED by the seated bar (mechanism is
   load-bearing physics), bar stays seated, score ~0
7. wrong-bay stow: bottle rolled into the open DECOY settles inside it → no credit
8. on-roof cheat: bottle lying on the roof → no containment credit
9. no re-bar: bottle genuinely rolled in + bar parked aside → NOT success, ≤ 0.85
10. wrong notches: bar drop-seated in the DECOY notches → NOT success
11. near-seat miss: bar resting short of the posts (outside seat x-window) → NOT success
12. latched credit survives regression (bottle teleported back out; score unchanged)
13. rejection audit: success() never True anywhere in the battery
14. final no-NaN

Verified on the forge: `solve` seeds 0 and 1 → `SIM_GEN_SOLVE: SUCCESS`;
`smoke` → `SIM_GEN_SMOKE: ALL PASS 14/14`.
