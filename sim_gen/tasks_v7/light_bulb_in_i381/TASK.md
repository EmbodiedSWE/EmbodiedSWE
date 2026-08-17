# light_bulb_in_i381 — Stand the fallen bulb on its pad and cage it under the keyed storm-guard

**Scene:** `simgen.bulb_guard_cap` (`GuardCapScene`, robot="null")
**Seed:** `rlbench/light_bulb_in`
(`RoboVerse/roboverse_pack/tasks/rlbench/light_bulb_in.py`)

## Seed provenance and what changed

The RLBench seed is a socket insertion: a small bulb stands in a holder, the
plan is **grasp the bulb, carry it over the lamp, lower it INTO a recessed
socket and screw it home with a wrist rotation** — the moved object is the
bulb, the goal fixture is a passive receptacle, the terminal act is a driven
rotation under load, and a decoy bulb must not be taken.

This task keeps the goal family ("the correct bulb ends up properly installed
at the lamp station, distractor excluded") but inverts the plan skeleton:

1. **The bulb is never inserted into anything.** It stands EXPOSED on a flat
   brass contact pad (Ø34 mm, 5 mm proud) on an open plinth deck. The bulb
   manipulation is an UPRIGHTING: it spawns knocked over, lying on the floor
   off the plinth, and must be stood upright on the pad — re-orientation +
   placement with nothing wrapping the base (no socket, no recess).
2. **The enclosure moves, not the bulb.** The load-bearing terminal act is
   performed with a SECOND free body: a rigid storm-guard cage (octagonal
   walls, closed lid, grasp knob, 250 g) that must be lowered OVER the
   standing bulb until its rim rests flush on the deck inside a curb ring.
   The seed moves the bulb to the fixture; here the fixture-like part is the
   payload and the bulb is the fragile obstacle the cage must swallow
   without touching.
3. **A passive yaw key replaces the screw.** Two red key bridges stand 8 mm
   proud on the deck under the cage's rim annulus (radial coverage asserted:
   bridge annulus [0.044, 0.060] spans the wall band [0.047, 0.053]). Only
   when the cage's two rim NOTCHES (bottom raised to 11 mm) line up over the
   bridges (yaw key, mod 180°, slack ±14.8°) can the rim descend past them
   to flush (~1–2 mm); at any other yaw the rim rests ON the bridges, 8 mm
   proud — outside the 3.5 mm capped z-window (smoke check 7: a cage
   released concentric but rotated 45° settles proud and earns nothing).
   The seed's screw is a driven rotation under load; the key is passive
   admission geometry — align, then a straight vertical descent.
4. **Execution order is physically forced.** The cage lid is CLOSED (solid
   disc + knob) and the seated cage leaves no bulb-passing aperture (largest
   under-rim gap is the 11 mm notch clearance vs the 30 mm bulb base —
   asserted; smoke check 6 drops the bulb from directly above a pre-capped
   pad and it bounces off the lid, never stands). Stand-then-cap is the only
   order.
5. **The decoy is excluded by rubric identity, not geometry alone.** An
   identically shaped burnt-out (dark) bulb lies on the opposite side;
   standing IT on the pad and capping it earns ~0 (smoke check 9).

So the plan skeleton changes from *"grasp bulb, carry, insert, screw"* to
*"upright the fallen bulb on an exposed pad, then yaw-align a keyed rigid
enclosure and lower it over the bulb to a flush geometric seat"* — the
terminal contact-rich act happens on a different object than the
goal-defining one, guarded by a passive alignment key instead of a driven
screw.

## Why strategically different from every examined sibling

- **Seed (`rlbench/light_bulb_in`):** see above — no insertion, no recess,
  no driven screw; the bulb's act is uprighting, the terminal act moves the
  enclosure, and admission is a passive yaw key.
- **`light_bulb_in_i219` (shelter roll-and-shutter):** there the goal object
  (an ungraspable 90 mm globe) is rolled non-prehensilely through a
  floor-level doorway over a force-threshold barrier, and a free-sliding
  slab seals the portal; the enclosure is STATIC. Here both handled objects
  are graspable, the goal object is uprighted and placed (prehensile
  schema), and the ENCLOSURE ITSELF is the moved payload, keyed by yaw.
  Disjoint machinery (upright + keyed vertical descent vs. planar roll +
  portal + detent), disjoint forced structure (stand-then-cap via closed
  lid vs. enter-then-seal via doorway).
- **`light_bulb_in_i120` (turnstile delivery):** there a jointed carrier
  mechanism is operated out–load–in and the bulb rides it through a wall.
  Here there are no joints anywhere and no carrier: the two free bodies are
  placed/lowered directly, and the challenge is orientation (uprighting) +
  a passive yaw key, not mechanism operation.
- **`light_bulb_out_i204` (rear-port tool ejection):** there a tool is
  threaded through a service port to eject the bulb and gravity delivers
  it. Here there is no tool, delivery is inward, and the terminal act is a
  guarded descent of a second body over the goal object.
- **`lamp_off_i101` (twistlock unplug):** there identification is by
  cord-tracing among decoys and the act is a grasped key-twist-pull on a
  jointed plug. Here identification is visual (lit vs. dark globe), there
  are no joints, and the yaw key is passive rim geometry on a free body.

## Solution outline (solve.py — teleport is TRANSPORT ONLY)

External-wrench plant recipe: `enable_external_forces_every_iteration: True`
in the scene's physx cfg; authored CoM + diagonal inertias so the cage servo
is auditable and discretely stable — kd·dt/m = 8/(120·0.25) = 0.27 < 1,
kdw·dt/I = 0.015/(120·6e-4) = 0.21 < 1. Wrenches are encoded body-frame via
`quat_apply_inverse`. Nothing is ever written into a goal pose: every
load-bearing seat is reached by gravity or by the wrench servo through
contact.

- **Phase 0 (reset):** settle 0.5 s, read back layout, assert score ≤ 0.02.
  `SIM_GEN_SCORE` 0.000.
- **Phase 1 (stand, gravity):** the bulb is transported to a hover 14–26 mm
  above the pad, upright, and RELEASED; gravity seats the base on the pad
  (90 settle steps). Escalating hover heights across attempts. Assert
  standing + stand latch. `SIM_GEN_SCORE` 0.250.
- **Phase 2 (cap, dynamics):** the cage is transported to a key-aligned
  hover 105 mm above the plinth (well above the standing bulb, rim above
  the globe top) and lowered by a 6-DOF wrench servo (force cap 8 N, torque
  cap 0.05 N·m; z-ref ramps down at 0.05 m/s; lateral + attitude hold to
  the plinth axis). The rim funnels into the curb (±11 mm), the notches
  pass the bridges, and the servo releases inside the flush window when
  slow; 90 settle steps. Three attempts. Assert capped + standing.
  `SIM_GEN_SCORE` 0.450 → success → 1.000.
- **Phase 3 (settle):** all bodies at rest, success live. 1.000.
- **Phase 4 (persistence):** 420 substeps (3.5 s) hands-off; `success()` is
  live state every step. Only then `SIM_GEN_SOLVE: SUCCESS`.

**Execution order is REQUIRED and physically forced:** the closed lid means
a capped cage passes no bulb (asserted + smoke 6), so stand-then-cap is the
only order; the key bridges reject every misaligned descent (asserted +
smoke 7), so yaw alignment is mandatory; the dead bulb earns nothing
(smoke 9). The rubric's latches mirror exactly this chain.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `stand_latch` (0.25): fresh bulb ever standing on the pad (plinth-frame
  |xy| ≤ 10 mm, root z ∈ [0.0035, 0.009] — on-pad rest ~0.006,
  deck-standing ~0.001 and lying ≥ 0.015 both rejected — tilt ≤ 10°).
- `cap_latch` (0.20): cage ever concentric (|xy| ≤ 12 mm) with rim below
  40 mm WHILE the fresh bulb is standing (a lateral swipe at rim height
  topples the bulb before the cage can read concentric — first wall–globe
  contact is at cage-local x ≈ −0.073, far outside the window; smoke 12).
- success (to 1.0): bulb standing AND cage capped (concentric, rim flush
  z ∈ [−0.002, 0.0035], upright ≤ 8°) AND both settled.
- Latches are transient-achievement credit (`torch.maximum`); cap 0.45
  without success (float32 boundary honoured with +eps in smoke). Null
  policy ≈ 0 (both bulbs spawn lying on the open floor, cage parked on the
  ground beside the plinth).

## Embodiment argument (single Franka, parallel jaw 80 mm, OSC)

Base at (0, 0) on the floor, facing +x: the plinth (x ≈ 0.42–0.50,
|y| ≤ 0.05, deck top at 0.05), both bulb spawn bands (plinth-frame x ≈
−0.32..−0.23, i.e. world x ≈ 0.10–0.27) and the cage park (world reach ≤
~0.66 m) all sit inside a 0.75 m reach disc at floor-to-0.18 m heights.

- **Uprighting the bulb:** the brass base is Ø30 mm and the lying bulb's
  highest graspable feature is at ≥ 15 mm — a standard side pinch within
  the 80 mm jaws; stand-up is a wrist re-orientation + place on a pad that
  is proud, visually distinct (brass) and fully supports the centred base
  (pad_r > base_r asserted).
- **Carrying the cage:** the cage body (circumradius 115 mm) exceeds the
  jaw span — asserted — so the Ø25 mm top knob is the handle; a knob pinch
  gives a vertical-axis carry exactly matching the keyed descent.
- **The descent:** curb funnel ±11 mm lateral and key slack ±14.8° yaw are
  generous alignment budgets for OSC; the seat is a hard flush stop
  (contact-terminated guarded move), no fine force control needed.
- **No blind reach:** pad, bridges (red), curb, notches, both bulbs (lit
  vs. dark globes) and the knob are all visible from outside; the flush
  seat is confirmed by the felt hard stop.

## Checks (smoke.py — rejection battery, 18 checks)

1. Clean reset: states finite; both bulbs lying in their floor bands on
   opposite sides, cage parked on the ground off the plinth, pad clear.
2. Score ~0 / no success at rest.
3. Randomization (8 seeds): plinth xy + yaw vary by READBACK.
4. Randomization: bulb spawn spreads; dead bulb changes sides; cage park
   spreads.
5. Null policy: 240 idle steps, score ≤ 0.02.
6. **Seed-strategy family (drop-in denial):** bulb dropped from directly
   above the CAPPED pad — bounces off the closed lid, never stands, ~0.
7. **Yaw key:** cage released concentric but rotated 45° — rests ON the
   bridges, proud of the capped window, no credit.
8. Capped-empty: cage seated flush with no bulb — no stand credit, no
   success.
9. Wrong object: the DEAD bulb stood on the pad + cage seated — ~0.
10. Off-pad stand: fresh bulb standing on the DECK beside the pad — no
    stand credit.
11. Lying-on-pad: fresh bulb lying across the pad — rejected by the z
    window and tilt.
12. Side-swipe: cage dragged laterally at rim height ≥ 100 mm — topples
    the bulb, cap latch never fires, score stays at the stand latch.
13. Settle gate: bulb standing + cage written into the flush window but
    rising at 0.25 m/s — capped+standing read True, not settled, not
    success.
14. Cap: both latches constructed, then the bulb yanked away → score ==
    0.45 cap (+eps), NOT success.
15. Latched credit: 40 further steps, score unchanged.
16. Rejection audit: success() never True at any judged point.
17. Final no-NaN.
18. Camera: ≥ 20 rgb frames captured → `frames.npz`.
