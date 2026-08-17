# gate_hopper — tilt the sealed hopper over the bin so its gravity gate drops the ball in

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet`
— pull open the top drawer of a fixed kitchen cabinet; success is a prismatic
joint position crossing a threshold.

## What changed, and why it is strategically different

The seed's plan is: approach a **fixed, anchored** cabinet, grasp the drawer's
**handle**, and pull the **sliding member directly** until a joint reading
crosses a threshold. The slide IS the goal, and one pulling contact on the
slide achieves it.

This task keeps a sliding member but makes the seed's entire plan impossible
and demotes the slide from goal to *means*:

- **The slide cannot be actuated directly.** The "drawer" is the floor GATE of
  a free-standing, fully sealed hopper: a slick captive plate riding in
  internal rails whose only exposed face is FLUSH with the front wall — no
  handle, no lip, nothing to grasp or pull. Smoke check 4 constructs the
  seed's end state (gate slid fully open where the box stands) and shows it
  earns ~0.
- **The actuation model is inverted: orientation of the housing is the only
  control input.** The gate is opened by *gravity* — lift the whole hopper by
  its roof handle and pitch it nose-down ~15–30°; the plate slides out its
  slot until an internal stop-tab catches. The robot never touches the moving
  part; it reorients the mechanism's frame.
- **The goal is delivery, not opening.** The hopper imprisons an orange ball
  (four walls + roof + gate; the floor opening is the only exit). Success is
  the ball resting INSIDE a randomized blue catch bin, with the hopper set
  back down upright and clear of the bin (≥ `clear_min` = 0.18 m,
  center-to-center; the no-touch distance is 0.15 m). Opening the gate in the
  wrong place (on the ground, or aloft over bare ground) strands the ball
  irrecoverably outside the bin.
- **Judging is a compound spatial predicate** (ball inside the bin's inner box
  in the bin's frame, hopper upright + grounded + clear, everything settled
  and finite), not a joint readout.

Strategically the solver needs: a carry grasp on the housing, an *aiming*
judgment (put the floor opening over the bin mouth BEFORE the floor lets go),
gravity-actuation of a captive mechanism through housing orientation, and a
set-down. None of these exist in the seed; the seed's one skill (pull the
slide) is useless here. Against the surveyed sibling from the same seed
(`..._i74` fold_collar: transport + fold an articulated chain to enclose a
column), this task shares no mechanism (captive prismatic gate vs revolute
wings), no actuation mode (gravity-through-reorientation vs applied fold
torques), and no goal topology (payload delivery into a container + set-down
vs enclosure of a fixed column).

## Teleport solution (solve.py phases)

Teleport = transport only; the gate is opened by gravity through the tilt, the
ball's delivery is free fall + contact, the final park is a release.

- **P0 — settle & readback** (score ~0): 150 settle steps; readback of hopper
  pose/yaw, bin pose, ball's in-cavity spot, and the authored masses
  (0.30 / 0.035 / 0.030 kg — custom-spawner MassAPI verification); assert
  gate flush, ball sealed, not success, score ≤ 0.03.
- **P1 — carry (teleport)** (score 0.15): one rigid write per body carries
  hopper + plate + ball, relative poses preserved, to a LEVEL hover with the
  base plane at 0.135 m, hopper center backed off 30 mm from the bin center,
  nose pointing at the bin; 0.5 s kinematic hold. Lift latch arms; gate still
  shut (asserted < 10 mm).
- **P2 — tilt (gravity actuation)** (score 0.70): per-step kinematic hold of
  the HOPPER ONLY, pitch ramped 0→30° over 2.5 s about its own axis. The
  plate slides out (measured: creeps from ~18°, runs to the 93 mm stop at
  30°), the ball rolls to the front sill and falls through the vacated
  opening into the bin (measured landing ≈ 31 mm from bin center, bound
  47.5 mm). Plate and ball are never written. Gate + drop latches arm.
- **P3 — park (teleport + release)** (score 1.0): hopper + extended plate
  teleported rigidly to open ground 0.30 m from the bin (> `clear_min`),
  nose away from the bin, 2 mm hover, then released; 2.5 s settle. success()
  turns True live.
- **P4 — persistence**: 3.33 sim-seconds fully hands-off; success must still
  hold; then `SIM_GEN_SOLVE: SUCCESS`, watchdog + `os._exit`.

Verified on the forge: seeds 0 and 1, both `SIM_GEN_SOLVE: SUCCESS`, score
trajectory 0.00 → 0.15 → 0.70 → 1.00 (monotone).

## Execution-order declaration

**Lift before tilt, aim before the floor opens — enforced physically, not
scripted.** The gate responds only to pitch, and pitching the hopper is only
useful aloft: with the hopper on the ground the gate can still be opened (by
the seed-style direct slide, or by tilting the box in place), but then the
ball simply drops ~10 mm onto the ground UNDER the hopper — outside the bin,
unrecoverable through the gate route (the rubric latches `gate` credit only
while aloft, and `drop` only inside the bin; smoke checks 4 and 6 verify both
wrong orders cap at ~0 and 0.40). The park must come last: `success()` judges
the live final state, so parking before delivering leaves `ball_in_bin` False.

## Embodiment argument (single Franka, parallel-jaw gripper)

- **Hopper carry**: the roof carries a dark 12 × 72 × 12 mm handle bar on two
  posts with 24 mm of daylight beneath it — a canonical top-down parallel-jaw
  grasp. Total carried mass ≈ 0.37 kg (hopper 0.30 + plate 0.035 + ball
  0.030), far under Franka payload; the carry height (base plane 0.135 m,
  handle ≈ 0.25 m) is mid-workspace.
- **Tilt**: pitching the wrist 30° about the grasp axis is a single joint-7 /
  wrist reorientation, well inside joint limits; the gate begins moving at
  ~18° and completes at 30°, so ±5° of orientation error still delivers.
- **Aim tolerance**: the bin mouth is 115 mm square for a 48 mm ball; the
  measured landing scatter across seeds (≤ ~35 mm from center) leaves ≥ 12 mm
  of margin — generous vs Franka repeatability.
- **Set-down**: lowering the ~0.34 kg emptied hopper onto flat ground with a
  10° uprightness tolerance and 20 mm height tolerance is a standard place.
- **Ball / plate / bin are never contacted** — the ball is sealed until it is
  delivered, the plate is flush (ungraspable) by design, the bin is fixed.
- **Base pose**: everything happens between the hopper spawn (~(0.0, 0.0)) and
  the bin (~(0.55, 0.0)); a Franka based at ~(0.28, −0.40) reaches the spawn,
  the hover over the bin, and the park spot within a ~0.75 m radius.

## Rubric

Latched partial credit (transients survive; success is judged live):
- 0.15 `w_lift` — hopper aloft (base > 0.06 m) with the ball still inside it.
- +0.25 `w_gate` — gate extension > 50 mm while the hopper is aloft
  (ground-level gate opening earns nothing).
- +0.30 `w_drop` — ball inside the bin (bin frame, below the rim). Partial
  cap 0.70.
- 1.0 iff `success()`: ball inside the bin ∧ hopper upright (≤10°) on the
  ground (|z| < 20 mm) ∧ hopper ≥ 0.18 m from the bin ∧ ball and hopper
  settled ∧ all poses finite.

Honesty asserts in `__post_init__`: ball fits the floor opening (64 mm vs
48 mm) but not the slot (17.5 mm); stop-tab travel (93 mm) exceeds the
drop-vacating travel (75 mm) with margin; the under-plate tab clears both the
rails and the ground; `clear_min` exceeds the no-touch distance; the spawn
layout parks the hopper far beyond `clear_min` with the ball inside (null ~0).

## Smoke rejection battery (12 checks)

1. Settle / no-NaN: gate flush, ball sealed, hopper upright; score ≤ 0.01.
2. Randomization READBACK, seeds 101 vs 202: hopper xy + yaw, bin xy + yaw,
   ball in-cavity xy all differ measurably.
3. Null policy, 240 idle steps: score ≤ 0.05, no success.
4. **Seed-plan negative**: plate teleported to full extension with the hopper
   ON THE GROUND — ball drops to the ground under the hopper, nothing aloft,
   no latch arms, score ≤ 0.01.
5. Mechanism (level carry): hopper held aloft LEVEL 2 s — gate stays shut
   (< 20 mm), ball stays sealed; score ≤ 0.15.
6. Wrong place: the full tilt maneuver over BARE GROUND — gate opens aloft
   (latch fires), ball lands on the ground; score ≤ 0.40, no success.
7. Near-miss: ball resting beside the bin's outer wall — no credit.
8. Near-miss: ball IN the bin but hopper toppled on its side — score ≤ 0.30.
9. Near-miss: ball IN the bin, hopper upright but centered 0.15 m away
   (< `clear_min`, no contact) — clearance clause refuses; score ≤ 0.30.
10. Rejection audit: success() never True at any step of checks 1–9.
11. Constructed-success sanity (audit closed): ball centered in the bin,
    hopper parked at its spawn → success True, score 1.0.
12. Video frames captured to `frames.npz` (> 10 frames).
