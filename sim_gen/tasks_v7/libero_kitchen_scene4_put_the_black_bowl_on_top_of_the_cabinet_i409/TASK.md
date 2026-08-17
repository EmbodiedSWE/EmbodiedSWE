# switchback_ramp — push the ungraspable basin up a two-flight switchback gallery onto the cabinet roof

**Task id:** `libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet_i409`
**Env:** `simgen.switchback_ramp` (scene-level, `robot="null"`)
**Files:** `scene.py` (scene + rubric), `solve.py` (zero-teleport force solution), `smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet`
(RoboVerse `roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet.py`).
The seed is a single pick-and-place: grasp the akita black bowl off the table and set it inside a
bbox above a white cabinet. Success is a static pose test on the bowl alone; nothing else moves.

What is kept from the seed: a black bowl-shaped cargo (here an octagonal basin), a white kitchen
cabinet as the destination fixture, and the goal of getting the cargo on top of the cabinet's roof.

## What the task is

A kinematic white **cabinet** (48 × 28 × 23 cm) stands on the floor with slick guard rails around
its roof, except for one **entry gap** on the gallery side. Bolted to the cabinet's +y face is a
**switchback ramp gallery**: a ground-level porch, a shallow outer **flight A** (11°) rising away
from the entry, a **turning pad** at the far end where the direction of travel REVERSES, and a
steeper inner **flight B** (21.5°) climbing back the other way to a **top pad** level with the
roof, right at the rail gap. All lanes are open-top channels between slick guard walls; every seam
in the travel direction steps DOWN 2 mm, so the route is one-way friendly but gravity opposes it
the whole way. The black **basin** (Ø 14.5 cm, 600 g, grippy base) starts on the porch. The goal:

> "Push the basin up the switchback ramp — out along the lower flight, around the turn, back up
> the steeper flight — and through the rail gap onto the cabinet roof."

`success()` = basin upright at rest on the roof plateau ∧ everything settled ∧ the basin actually
TRAVERSED flight A, the turning pad, and flight B in that order (chained order-gated latches). A
maroon bottle on the floor is a distractor. There is no other route: the roof is fenced by rails
everywhere except the gap at the top pad, and the walls of the gallery are too tall to lift over
(and the basin cannot be lifted at all — see embodiment).

## Strategic difference — vs the seed and vs every corpus neighbor

- **vs the seed:** the seed is one grasp and one free-space carry. Here the basin is *impossible to
  grasp* (Ø 14.5 cm > 8 cm jaw span, no handles, walls taller than finger reach from the rim), so
  the cargo can never be carried — the ENTIRE altitude gain (0 → 23 cm) is earned by pushing the
  cargo along a constrained inclined route with a mandatory direction reversal. The bowl never
  leaves contact support for a single step.
- **vs i308 wedge_lift (same seed):** there the elevation is manufactured by a force-multiplying
  machine — a pushed wedge ram jacks a third body (the elevator platform) vertically, and the push
  direction never changes. Here there is no mechanism at all: the basin itself is pushed the whole
  way, altitude is gained continuously along inclines, and the defining constraint is *route
  geometry* — a switchback whose middle leg reverses the direction of travel, enforced by an
  order-gated latch chain (A → turn → B).
- **vs i61 cart_ferry:** the cargo rides a repositioned vehicle; here nothing carries the basin —
  it is the pushed body throughout, and the route is vertical-gaining, not planar.
- **vs i3 tunnel_shuttle / i19 moat_causeway:** those push cargo along an essentially horizontal
  route; no sustained climb, no direction reversal, no order-gated multi-leg traversal.
- **vs i306 hatch_shelf / i5 counterweight_shelf:** those gate ACCESS to a shelf; the placement
  itself is still a transport step. Here the traversal IS the task — success is unreachable
  without all three legs firing in order, and smoke proves that even a legitimate flight-B climb
  earns zero if flight A was skipped.
- **vs robobench suite (balance_scale, combination_safe, syringe, pen_holder):** no articulated
  joint anywhere; the task is pure contact dynamics on static inclined geometry.

No task in the read corpus routes the cargo up a multi-flight inclined gallery with a direction
reversal, and none makes *sustained uphill pushing of the goal object itself* the sole source of
the required elevation.

## Rubric (latched credit, `score()`)

| latch | condition (fixture frame, every physics substep) | weight |
|---|---|---|
| `_flightA_ever` | basin in the flight-A band (outer lane, above porch level) | 0.15 |
| `_pad_ever` | `_flightA_ever` ∧ basin on the turning pad, crossed into the inner row (**order-gated**) | 0.20 |
| `_flightB_ever` | `_pad_ever` ∧ basin in the flight-B band (inner lane, high) (**order-gated**) | 0.25 |

`score = Σ weights`, capped at 0.60 unless `success()`, which returns exactly 1.0. Success itself
requires `_flightB_ever`, so the seed's strategy (bowl placed straight onto the roof) scores **0**
— proven by smoke negatives A/A2: even entering the roof from a LEGITIMATE flight-B climb earns
nothing if the earlier legs did not fire in order.

## Solution (`solve.py`) — ZERO teleports

All interaction is pulsed external force + contact dynamics (the climb is the task, so teleporting
would skip it). Pulsed push = force on only while the basin's speed along the drive axis is below
a cap; forces sized from F = mg(sinθ + μcosθ)/(cosθ − μsinθ) with ≥ 50% margin:

1. **FLIGHT A** — 9 N along fixture +x (need ≈ 5.7 N at 11°), porch → turning pad. → 0.150
2. **THE TURN** — 7 N along −y across the pad into the inner-lane row. → 0.350
3. **FLIGHT B** — 13 N along −x (need ≈ 8.3 N at 21.5°), pad → top pad. → 0.600
4. **ENTRY** — 7 N along −y, off the top pad (2 mm step DOWN) through the rail gap onto the
   roof; release, settle. → 1.000
5. **PERSISTENCE** — ≥ 3.5 simulated seconds fully hands-off; `success()` must still hold before
   `SIM_GEN_SOLVE: SUCCESS` is printed.

On these pods the default external-force call applies the wrench in the body's CURRENT frame, so
`drive()` pre-encodes every command per step with `quat_apply_inverse(q_now, f_world)` (correct
for the basin's free spawn yaw) and keeps a raw-world fallback behind a measured-progress stall
probe. The basin rests on either incline with the force off (μ_s 0.70 > tan 21.5° = 0.39), so the
latched credit survives any pause.

Verified on the forge: seeds 0, 1, 2 all print 0.000 → 0.150 → 0.350 → 0.600 → 1.000 and
`SIM_GEN_SOLVE: SUCCESS`, rc = 0.

## Embodiment argument (single Franka, 8 cm parallel jaw, OSC)

The same plan executes with one arm, and **no stage admits a grasp-and-carry shortcut**:

- **Basin cannot be grasped or lifted.** Outer diameter 14.5 cm > 8 cm jaw span — no side or rim
  straddle anywhere. The walls are 7.5 cm tall with a grippy 600 g base; a top-down rim pinch has
  nothing to close on (wall thickness 10 mm < a stable pinch on a smooth vertical wall under
  0.6 kg load, and the rim circle is far wider than the jaw). The only way the basin moves is
  pushing on its wall.
- **Every push is a fingertip push into an open-top channel.** All lanes are roofless; the guard
  walls flanking each lane are low enough to reach over (north wall 16 cm, divider 28 cm, east and
  south walls 23.5 cm, west wall 29 cm — all well under the arm's overhead reach) but tall enough
  that the basin (7.5 cm) cannot be tipped or lifted over them. Pushing low on the basin wall from
  above needs at most ≈ 8.3 N; the fingertip force to tip the basin exceeds ≈ 12 N, so a low push
  slides rather than topples it.
- **One plausible base pose:** base at fixture ≈ (x = 0.0 m, y = +0.85 m), facing the gallery.
  From there the porch (x ≈ −0.44, y ≈ 0.45), the far turning pad (x ≈ 0.41, y ≈ 0.34), the top
  pad (x ≈ −0.14, y ≈ 0.24) and the roof entry are all within ~0.9 m reach, and the push
  directions (+x, −y, −x, −y) are all executable as horizontal fingertip drags from above the
  open lanes without the forearm fouling the walls.

## Execution order (declared)

Strictly sequential; each leg gates the next physically AND in the rubric:

1. Flight A (the porch is the only ground-level entry to the gallery; the turn pad is 12 cm up).
2. The turn (the divider wall separates the lanes; the only crossing is the turning pad).
3. Flight B (the only way to roof height; the rail gap is only reachable from the top pad).
4. Roof entry + hands-off persistence (the terminal state is fully passive on the static roof).

Skipping any leg is unrewarded by construction: `_pad_ever` requires `_flightA_ever` first and
`_flightB_ever` requires `_pad_ever` first, and `success()` requires `_flightB_ever` (smoke
negatives A and A2).

## Checks (`smoke.py`) — 16

1. reset settles finite (basin upright on the porch in its spawn band, decoy standing); 2. no
latch / score ~0 at reset; 3. randomization READBACK (fixture xy/yaw, basin spawn x, basin yaw
spreads across 6 seeds); 4. null policy ≈ 0; 5–7. force-driven oracle passes on seeds 0/1/2
(score 1.0, persists 240 steps); 8. incline rest — basin released mid-flight-A stays put
(latched 0.15 credit survives, no roll-back); 9. monotone ladder 0 < 0.15 < 0.35 < 0.60 (on the
top pad, no success) < 1.0; 10. partials < 1.0; 11. negative A: basin teleported straight onto
the roof (seed strategy) → on_roof geometrically true, score 0; 12. negative A2: basin teleported
to the turning pad, then a LEGITIMATE flight-B climb and roof entry → still score 0 (order gate);
13. negative B: after a legit climb, basin astride the rail-gap seam (not fully on the roof) →
score stays 0.60; 14. negative C: inverted basin on the roof rejected; 15. negative D: DECOY
bottle on the roof scores nothing; 16. landing/drop calibration sweep (in/in/out/out). Frames
recorded to `frames.npz` (600 × 960 × 3).
