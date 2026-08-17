# shutter_vault — slide the interlocked shutters open, then serve the FRONT bowl down the well (i227)

**Env name:** `simgen.shutter_vault` (scene `shutter_vault`, registered with robot `null`;
solve.py and smoke.py build the same scene-level env).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate.py`)
- Seed plan: among three identical black bowls in a row, select the FRONT one by table
  position, grasp it, carry it across free space, set it down on a flat, openly reachable
  plate. Success = one geometric relation, reached by a single unconstrained
  grasp-carry-place.

## What changed, and why it is strategically different

The "front black bowl of three, onto the plate" premise is kept (three identical bowls in
a row, target = the front slot, identity permuted per episode); what changed is that the
**plate is sunk in a shuttered well and the shutters form a captive sliding-tile
interlock**:

1. **The goal region is sealed by a MECHANISM the robot must reconfigure.** The plate sits
   at the bottom of a 120 mm-square well recessed into a vault block. The well mouth is
   covered by the RED shutter tile; the only path to the plate is straight down through
   the aperture after the RED tile has been slid clear. The plate itself never moves and
   never needs to (4 mm radial slack inside the well — and extracting it is a REJECTED
   outcome, smoke 14).
2. **The mechanism is CAPTIVE — tiles slide, nothing lifts.** Both 140 mm shutter flanges
   run under continuous overhanging rails with 4 mm of headroom in an L-shaped 3-cell
   channel A–B–C (pitch 150 mm). At every reachable position both opposite flange edges
   remain under rails (max attainable tilt ~2°), so the seed's primitive — pick the
   obstruction up — is geometrically impossible; tiles admit exactly one DoF each
   (horizontal sliding), driven by their Ø22 mm knob posts.
3. **A two-move ordered sub-puzzle gates the goal.** At reset RED covers the well (cell A)
   and BLUE occupies the only cell RED can slide into (B); the free cell C is reachable
   only from B. So the ONLY opening sequence is BLUE B→C, then RED A→B — order enforced by
   occupancy geometry, verified by a calibrated blocked/free push pair (smoke 9–10).
4. **The final placement is a blind vertical insertion, not a set-down.** The bowl
   (~91 mm over corners) must be dropped through the 120 mm aperture (≈14 mm radial
   clearance) and seat upright on the sunken plate 45 mm below the deck.
5. **A solver needs different code structure**: two closed-loop horizontal slide servos on
   the tiles (push knob, regulate to cell centres, respect the interlock order) bracketing
   one aperture-threading place, instead of the seed's single place-on-target primitive.

Corpus differentiation (tasks read this batch): sibling i189 (same seed) claims **hidden
physical property identification** — a balance-beam measurement with an
information-gating rubric; here nothing is hidden, no measurement exists, and the
challenge is mechanism reconfiguration. i78 claims **mechanism-mediated ferrying** — a
carousel CARRIES the payload both ways; here the mechanism never carries anything: the
tiles are obstructions to be re-arranged, the payload travels by hand. i56 claims a
rule-governed Hanoi relay on static pegs (order from the rubric); here order is enforced
by occupancy geometry, and the rubric is orderless. robobench's `pen_holder` is container
filling with open access. This task claims **captive sliding-shutter interlock over a
sunken goal (slide-to-unblock, then thread the aperture)**: nothing here is weighed,
ferried, rule-sequenced, stacked, or de-stacked.

## Embodiment argument (single Franka arm + parallel jaw, OSC)

Intended base pose: **(-0.05, 0.0, 0.20)** — mounted on the counter slab (the sibling
tasks' verified on-counter mount).

- **Shutter knobs (the drive handles):** Ø22 mm × 45 mm posts centred on each flange, tops
  127 mm above the counter — open top-down pinch (22 mm << 80 mm jaw span, ≥30 mm finger
  engagement above the rail plane; the rails stop 10 mm below the flange-top plane, so
  fingers never enter the rail gap). Cell centres lie 0.37–0.51 m from the base — inside
  the proven 0.35–0.65 m comfort band. Slide resistance is the 0.35 kg tile's friction,
  ~1.7 N (the reference drive uses 2.5 N) — trivial for the arm; strokes are straight
  150 mm horizontal translations guided by the channel walls (10 mm lateral slack).
- **Front bowl (the payload):** bowls stand at 0.21–0.53 m from the base in an open row.
  Octagonal, ~91 mm over corners (too wide to span) with a 6 mm rim wall 40 mm tall: the
  standard mug-style rim pinch (6 mm << 80 mm span, up to 40 mm engagement).
- **The insertion:** hold the bowl over the opened well and lower/release ~28 mm above the
  plate. Radial clearance bowl-to-aperture is ≈14 mm and the serve tolerance is 35 mm to
  the plate axis — both far above closed-loop OSC placement noise (1–3 mm in prior
  tasks); the drop is a pure gravity seat, encoding- and controller-proof. The gripper
  itself never needs to enter the well: release happens above the deck plane and the
  45 mm-deep well funnels the last stretch.
- **Plate:** intentionally never grasped and never moved.

All interactions happen 200–330 mm above the arm's base plane on open top-down
approaches; no contact near the ground or through any side aperture.

## Execution order

**Fully ordered, enforced by GEOMETRY, not by the rubric:** the bowl cannot reach the
plate while RED covers the well (a bowl set down over the well rests on the closed
shutter, 70 mm too high — smoke 8), so OPEN must precede SERVE; RED cannot slide into B
while BLUE occupies it (blocked push advances only the ~10 mm slack — smoke 9), and C is
the only other cell, so UNLOCK (BLUE B→C) must precede OPEN (RED A→B). success() is a
pure settled-state predicate with no order clause; the three latches are additive and
monotone along the only feasible order.

## Rubric (graded, latched in post_step, additive)

| score | state |
|-------|-------|
| 0.00  | nothing done (null policy; also the seed-strategy end state) |
| +0.25 | UNLOCKED: BLUE tile parked within 30 mm of cell C centre, tile slow |
| +0.25 | OPENED: RED tile parked within 30 mm of cell B centre, tile slow |
| +0.20 | SERVED: the front bowl within 45 mm of the well axis with its base below deck level (vault-local z < 0.050), bowl slow |
| 1.00  | success(): front bowl upright & centred on the plate (35 mm xy, 10 mm z band, 12° tilt) + plate seated in the well (20 mm xy, 8 mm z, 10° tilt) + everything settled |

Honesty notes. The SERVED latch cannot fire over a closed shutter: on the shutter a bowl's
base rests at vault-local z ≈ 0.082 > 0.050, and the knob post occupies the well axis
(smoke 8 asserts both). Tile latches are velocity-gated (0.05 m/s): a tile written through
its cell at 0.40 m/s latches NOTHING; the same pose at rest latches (smoke 15–16). Latched
credit is history ("has been open"), deliberately kept when the shutters are closed again
(smoke 11) — but full credit needs the settled goal state. Wrong-bowl (smoke 12),
inverted-bowl (smoke 13) and plate-extracted (smoke 14 — the literal seed goal geometry,
bowl-on-plate-on-counter) end states are all rejected. The served bowl stands 8 mm proud
of the deck plane, so "close the shutter over the served bowl" is unconstructible by
geometry.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move the free bowl through free air ONLY; every load-bearing interaction runs
through contact dynamics:

- Phase 0: `env.reset(seed=N)`, settle, layout readback (vault pose, front-bowl identity,
  slot permutation) → `SIM_GEN_SCORE 0.000`.
- Phase 1 UNLOCK: capped-speed bang-bang force servo on the BLUE tile (2.5 N vs the
  ~1.7 N friction budget, 0.08 m/s cap, stall escalation to 6 N, braking tail, force cut)
  B→C along the channel; hard-FAIL if the latch is cold → `SIM_GEN_SCORE 0.250`.
- Phase 2 OPEN: same servo, RED A→B → `SIM_GEN_SCORE 0.500`.
- Phase 3 SERVE: front bowl — raise to 0.42 m, hop over the well axis, lower down the
  aperture, release ~28 mm above the plate; gravity + the well funnel seat it (retries
  with small lateral offsets) → `SIM_GEN_SCORE 1.000`.
- Phase 4: ≥3.3 simulated seconds hands-off; success() must hold → `SIM_GEN_SOLVE:
  SUCCESS`. Any score decrease prints FATAL/FAIL. Hard exit behind watchdog Timers.
- Force-frame robustness: the pod may apply external wrenches in the body frame (a
  yaw-ψ tile pushed with a raw world vector moves along 2ψ — seed 7's −20° yaw
  wedge-jammed the RED tile). The drive pre-encodes the intended world force with
  `quat_apply_inverse` per step and carries a progress probe that toggles the encoding if
  full force buys no progress. The smoke's push probes calibrate the same way against the
  free BLUE tile before the interlock checks.

## Check list (smoke.py — rejection battery; solve.py is the acceptance proof)

1. settle/no-NaN: authored layout settles finite; RED covers the well, BLUE in B, plate
   seated, bowls in their slots
2. rubric clean at reset (score 0, no success)
3. determinism: same seed → identical layout readback
4. randomization: front-slot bowl identity varies across seeds (readback)
5. randomization: vault pose varies across seeds (xy and yaw readback)
6. randomization: bowl positions vary across seeds (permutation + jitter readback)
7. null policy: score ~0, no success after 240 idle steps
8. SEED STRATEGY (front bowl set down over the well on the CLOSED shutter): rests 70 mm
   too high beside the knob, SERVED never latches, score 0
9. interlock: RED pushed toward B while BLUE occupies it advances only the slack (<30 mm)
10. interlock twin: the SAME push recipe moves the unobstructed BLUE tile full travel into
    C (>100 mm) and fires UNLOCK — anti-vacuity for check 9
11. latched credit: a constructed OPEN state latches 0.50; closing the tiles again keeps
    0.50 (history, not state)
12. wrong bowl: a REAR bowl dropped through the open aperture is on the plate but success
    stays False (score 0.50)
13. inverted bowl: the FRONT bowl upside-down in the open well → upright gate rejects
14. plate extracted: the literal seed goal geometry (plate on the counter, front bowl on
    it) → plate_in_well False, success False, score 0
15. velocity gate: BLUE written into C at 0.40 m/s latches NOTHING
16. velocity gate twin: the same pose at rest DOES latch (0.25)
17. finite: all states finite at the end; frames.npz saved

Teleports in the smoke are instrumentation (constructed settled states, real physics
steps before judging); no probe constructs full success.

## Scene / assets

Fully procedural, one rigid body per object, explicit MassAPI on dynamics: kinematic
vault block (well floor + walls, two 150 mm decks, six channel walls forming the L-shaped
A–B–C corridor, six overhanging rails at 4 mm flange clearance) rooted at the cell-A
centre; two dynamic shutter tiles (140 × 140 × 12 mm flange + Ø22 × 45 mm knob, 0.35 kg,
RED at A over the well, BLUE at B); white plate (Ø112 × 10 mm, 0.20 kg, friction
0.7/0.6) seated at the well bottom; three identical black octagonal bowls (~91 mm over
corners × 48 mm, 0.15 kg, 6 mm rim wall) in a row, FRONT = smallest x; counter =
kinematic slab (top at 0.20 m). Randomized per episode: vault xy (±25 mm) + yaw (±20°),
which bowl body occupies the front slot (permutation), per-bowl xy jitter (±15 mm).
Contact offsets 1.5 mm (2×1.5 mm < the 4 mm rail gap and 4 mm plate slack).
