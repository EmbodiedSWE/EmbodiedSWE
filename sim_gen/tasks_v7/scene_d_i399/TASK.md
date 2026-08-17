# scene_d_i399 — mast lowering (controlled fell into a placed catch cradle, over a protected bottle)

## Seed provenance

Seed: `calvin/scene_D` (tabletop manipulation scene: articulated furniture — a hinged/sliding
element that must be actuated — plus small free props on a flat work surface).

What was kept from the seed:
- A single-DOF hinged element on a fixed base that the agent must actuate through a large
  angle (CALVIN's switch/door lever → the pylon-hinged mast).
- Small free props standing on the work surface that share the workspace with the articulated
  element (CALVIN's blocks → the green bottle on its marker pad).
- A flat table-like work surface with everything in reach of a single arm.

What was changed (all geometry procedural, no assets reused):
- The hinged element is now a 0.62 m mast on a revolute joint (−7°…+128°) whose actuation is
  **destructive by default**: released past vertical it falls through the corridor where the
  bottle stands, and the floor rest pose (~116°) is *outside* the scored band.
- A free **catch cradle** (guide plates + crest bar) must first be transported and stood at a
  spot that *depends on the sampled bottle position* (deploy_off past the pad), so the falling
  mast is arrested at φ_rest ≈ 96–98°, inside the scored band [phi_lo, phi_hi], spanning
  *over* the still-standing bottle.
- The prop is no longer something to move: it is a **protected bystander**. Success requires
  the bottle upright on its marker at the end; the task is judged on the felled-and-caught
  mast *and* the undisturbed bottle simultaneously.

## Strategic difference

vs the seed (calvin/scene_D): CALVIN tasks actuate the articulated element *as the goal*
(open drawer / flip switch). Here actuating the hinge is trivially easy and *ruinous* — the
skill is preparing the environment (cradle placement keyed to the sampled bottle spot) so
that an irreversible gravity-driven event ends in the scored band. The prop goes from
"object to move" to "object to protect".

vs tasks already in the corpus (read set):
- `scene_d_i130` (carousel vault): transport-by-riding a free rotor. Here nothing rides
  anything; the dynamic event is a fall that must be *arrested*, not a delivery.
- `scene_b_i305` (marble router): routing a ball via a bistable switch. No falling
  long-body, no protected bystander, no fixture-placement-keyed-to-randomization.
- `scene_c_i171` (beam balance): static force balance verdicts. Here the physics is a
  transient (fall → catch), judged live with settle gates.
- `pen_holder`, `trestle_service`: insertion/support tasks where placement is the goal
  itself. Here placement (cradle) is only *instrumental*; the goal predicate is about the
  mast angle band + bottle integrity, and a perfectly placed cradle scores only 0.20.

Distinct axes: (a) **required destructive actuation** — you must knock the mast over;
(b) **protected bystander** — a fragile prop in the fall corridor judged for
non-disturbance; (c) **continuous coupling** between randomization and required fixture
pose (deploy spot = f(sampled pad_x)); (d) **physics-forced ordering** with no order
latch (see below).

## Execution-order declaration

Order is **physically encouraged, not latch-enforced**. There is no "deploy before fell"
latch. Instead the geometry guarantees (asserted in `__post_init__`) that a cradle-less
fell can never end in band, on either physical branch:

- **Swept-aside branch**: if the mast knocks the bottle away it reaches the floor rest
  pose φ_floor ≈ 116° > phi_hi, with its face below the cap height
  (`_face_z(pad_x[0], phi_floor) ≤ bottle_top − 15 mm`).
- **Propped branch**: the heavily damped mast can instead land face-down on the bottle's
  flat cap and *prop* there (observed on the forge: a near-centered cap contact is a
  stable vertical support). The cfg asserts `_face_z(db, phi_hi + 2°) ≥ bottle_top + 4 mm`
  at both pad extremes, so cap contact — and any bottle-propped rest (~106.5–112°) —
  happens strictly beyond the band's upper edge (phi_hi = 103°; the cradle-caught rest
  is 96.2–97.6°, > 5° inside).
- In both branches the mast decelerates only *past* the band, so the fast mid-sweep pass
  through the band can never satisfy the arrest gate (|ω| < settle_ang). Recovery
  requires re-erecting the mast against gravity — the episode is effectively spent.
  Smoke check 5 verifies the fell-first outcome (out of band, commit credit only,
  score == w_commit, no arrest/deploy latches, success never observed).

## Teleport-solution outline (solve.py)

Teleports are used for **transport only** (the single cradle move); every load-bearing
interaction is contact dynamics via applied wrenches.

- **P0 settle** (120 steps): readback asserts — mast on the back stop (φ ∈
  [limit_lo−0.8°, −4°]), pad/bottle within 5/10 mm of the sampled layout, cradle at the
  rack, not deployed, score ≤ 0.05.
- **P1 deploy**: the ONE teleport — cradle written to (pad_x + deploy_off, 0) at hover
  height, 30-step drop, then a real −8 N press for 60 steps (contact seating). Assert
  `deployed()` and `bottle_ok()`; score ≥ w_deploy. Print SIM_GEN_SCORE.
- **P2 fell**: ramped body-frame PD torque about the hinge axis (kp 2.0, kd 0.3, cap
  1.6 N·m) toward +25°, torque cut at φ ≥ 18° (past vertical + commit threshold);
  gravity takes over. Assert commit latch; score ≥ 0.40. Print SIM_GEN_SCORE.
- **P3 hands-off catch** (≤900 steps): mast falls at damping-limited ~1.4 rad/s, shaft
  face lands on the cradle crest bar spanning over the bottle. Assert caught φ within 4°
  of the predicted φ_rest, bottle_ok, success, score 1.0. Print SIM_GEN_SCORE.
- **Persistence**: 10×40 further steps (≥3.33 sim s) fully hands-off; success must hold
  every probe; only then print `SIM_GEN_SOLVE: SUCCESS`.
- Run on `--seed` and `--seed+1`; monotone SIM_GEN_SCORE asserted; watchdog + hard exit.

## Embodiment argument (single Franka, parallel jaw)

- **Cradle**: carried by its crest bar (0.04 m wide × 0.016 m tall — inside the ~0.08 m
  jaw span), mass 1.2 kg, well within payload. Set-down is a hover-drop plus a gentle
  press — exactly what an arm does after releasing.
- **Mast**: pushed at the shaft around z ≈ 0.7 m lever height; passing vertical from the
  −6° stop needs < 1 N of fingertip push, and the solve's 1.6 N·m torque cap corresponds
  to ~2.3 N at that height. Single-finger side push, no grasp needed.
- **Bottle**: never touched by the solution.
- **Base pose**: Franka base at ≈ (0.25, −0.55, 0) beside the fall corridor: rack
  (x 0.18–0.34, y ±0.24–0.30), deploy spot (x ≈ 0.42–0.53, y 0), and the mast shaft
  (x ≈ 0, z ≤ 0.7) are all within ~0.65 m reach, and the base is outside the corridor so
  the falling mast cannot hit the robot.

## Randomization (verified by readback in smoke)

Per-seed: pad_x ∈ (0.28, 0.38), pad_y ∈ ±0.02 (bottle + marker together); cradle rack
x ∈ (0.18, 0.34), y = ±(0.24…0.30) with random side sign and free yaw. The required
deploy x moves with pad_x, so no fixed-pose script solves all seeds.

## Scoring

Latched partials (0.20 each): deployed cradle · commit (φ past commit_deg with cradle
deployed & bottle ok) · arrested (in band, slow, bottle ok). Capped at 0.60 unless
`success()`: mast in [phi_lo, phi_hi] ∧ bottle upright on its marker ∧ settled ∧ finite
→ 1.0. Success is judged live (state, not memory); latches only feed partial credit.

## Smoke checks (13)

1. Settle & no-NaN: spawn state on stops/markers, score ≈ 0.
2. Randomization A (readback): pad/bottle span across 8 seeds, bottle tracks pad.
3. Randomization B (readback): rack x/yaw span, cradle tracks rack.
4. Null policy 240 steps: no success, score stays ≈ 0.
5. Fell-first (REAL torque, cradle racked): mast ends OUT of band (propped on the
   bottle cap or on the floor), no arrest latch, score == w_commit exactly — wrong
   order is not recoverable credit.
6. Bottle lying on its side at the pad (constructed before the caught mast): band +
   still but success rejected (tilt gate).
7. Bottle standing but 7 cm off the marker: rejected (xy gate).
8. Bottle perched on the caught beam top: rejected (rest-height gate).
9. Deploy-only: score == w_deploy exactly; mast still on the back stop.
10. Written "success" pose with mast ω = 0.40 rad/s: rejected pre-step (settle gate);
    state broken up before stepping so the audit can't be poisoned.
11. Rejection audit: success never fired during any rejection construct.
12. Final no-NaN.
13. Video: frames.npz written, > 10 frames.

## Files

- `scene.py` — MastLoweringScene (`SCENES.register("mast_lowering")`, env
  `simgen.mast_lowering`, robot="null"), procedural geometry, per-env USD revolute joint.
- `solve.py` — teleport solution as outlined; 2 seeds; hard exit.
- `smoke.py` — 13-check rejection battery; frames.npz; hard exit.
