# carafe_pour — pour the alphabet-soup can out of a grasp-captive carafe into the tray

`living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_i316` · scene `carafe_pour` · env `simgen.carafe_pour`

## Seed provenance

Seed task: `roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.py`
(RoboVerse / LIBERO-90). The seed is a pick-and-place: lift the alphabet-soup can
off a table shared with look-alike grocery cans and lower it into an open tray;
`_terminated` is bounding-box containment on the tray's `contain_region`.

Kept from the seed: the protagonist (a small soup can, here 48 mm x 85 mm), the
"identify the red can among a look-alike" framing (corn yellow, assignment shuffled
per episode), the open tray, and the goal readout "the alphabet-soup can inside the
tray".

Changed: the can is never a valid grasp target — it starts at the bottom of a tall,
narrow octagonal CARAFE whose annular clearance (6 mm flats / 8.5 mm corners) is
below any parallel-jaw finger and whose rim stands 65 mm above the can's top. The
manipulated object becomes the CONTAINER: the seed's grasp-carry-release of the can
becomes grasp-the-carafe, carry, then a controlled past-horizontal POUR that lets
gravity discharge the can into the tray — followed by righting and setting the
emptied carafe back down on the ground. The containment readout stayed, but the
rubric now also demands the can be free of every carafe, no carafe in the tray, the
corn can NOT in the tray, and both carafes back at ground level.

## Strategic difference argument

- **vs the seed:** the seed's whole strategy is direct prehension of the goal
  object. Here that strategy is geometrically void (nothing can grip the can), and
  its transport-only ghost — dump the whole carafe into the tray with the can still
  inside — is constructed in smoke check 5 and REJECTED: the can's position is
  inside the tray volume, yet it earns nothing because it is still captive. The
  load-bearing skill moves from position-space (place a grasped object) to
  ORIENTATION-space (roll a held container through ~100 degrees so gravity does the
  delivery), plus a second obligation the seed never had: restore the tool (carafe
  righted, grounded, off the tray).
- **vs sibling i236 (airlock) and i158 (shuttle kiosk):** those are FIXTURE
  protocols — the agent actuates a mechanism (slider, drawer) that is part of a
  static station, and the protagonist travels alone through apertures. Here there
  is no mechanism at all: the "aperture" is the carafe's own mouth, the actuated
  object is a free rigid body held in the hand, and the delivery happens while the
  agent is holding it.
- **vs i139 / i253 (balance tasks):** those are continuous equilibrium tasks — a
  scalar (tilt) must be regulated to a band. The pour is a threshold traversal:
  nothing happens below ~97 degrees (smoke check 7 holds the carafe at 60 degrees
  and the can provably stays), everything happens once past it.
- **vs i33 (flap chute), i166 (roll chute), i207 (queue):** those route a free can
  through passive static geometry by drops and pushes. Here the can is never free
  until the final instant — it rides inside the held object, and its release angle,
  release height and landing point are all consequences of how the agent poses the
  carafe.

## The geometry (numbers)

- Carafe (0.25 kg, CoM z 0.050, authored inertia diag(0.004, 0.004, 0.002)):
  octagonal cup, floor 60 x 60 x 8 mm, eight 5 mm walls z 0.008..0.158 at
  wall-centre inradius 0.0325 -> bore inner inradius 0.030, outer flats 0.070.
  Interior (floor + walls) slick 0.06/0.05: pair friction with the can ~0.125,
  so the bore discharges once its axis passes ~97 deg from vertical.
- Captivity: can r 0.024, h 0.085 -> 6 mm annular gap at the flats (8.5 mm at
  corners), can top 65 mm below the rim. No parallel-jaw finger fits the gap; the
  can leaves the bore only by pouring.
- Tray (1.2 kg, free): floor 260 x 200 x 10 mm, walls 10 x 60 mm -> interior
  240 x 180 mm, rim z 0.070.
- Layout: tray nominal (0.42, 0), yaw +/-25 deg, xy +/-0.03; carafe slots
  (0.26, -0.28) and (0.26, +0.28), xy +/-0.02, free yaw, slot assignment AND
  which carafe holds the red can both shuffled per episode.

## Rubric

`success()` (all clauses live): red can inside the tray interior AND in no carafe
AND no carafe in the tray volume AND the corn can NOT in the tray AND both carafes
grounded (origin z < 0.060) AND everything settled AND finite.

`score()`: latched partial credit 0.30 decanted (red can at rest, low, outside both
carafes) + 0.30 delivered (red can at rest in the tray, in no carafe), capped at
0.60; exactly 1.0 iff success() live. Null policy ~0 (the can starts captive in a
bore, outside every credit state).

**Ordering declaration:** the rubric imposes NO step ordering — the captive bore
does. The can cannot be in the tray (or anywhere) before a pour has happened, and
no prefix of the solve satisfies the goal (transport grants nothing, the pour alone
leaves the carafe aloft/ungrounded).

## Solution (solve.py) — teleport for transport only

- P0 settle; mass + layout readback asserts (carafe/tray/can masses, tray pose,
  red-jar assignment, both cans captive). Score 0.
- P1 TRANSPORT: ONE rigid write carries the red-can carafe — can still seated
  inside, relative pose preserved, zero velocity — to a tray-side hover (origin at
  tray-local (-0.070, 0, 0.230)); asserted: can still captive, outside the tray
  volume, score unchanged. A 6-DOF wrench servo (force kp 50 / kd 10, cap 12 N,
  live gravity feedforward for carafe + resident can; torque kq 0.8 / kdw 0.08,
  cap 1.0 N m, plus the can-offset gravity moment) engages and PROBES the pod's
  force-frame convention: a 35 deg tilt probe at the hover (far below the 97 deg
  discharge angle), plus a 0.10 m tracking-divergence guard during the pour —
  the frame drag is invisible while upright and only bites mid-roll; on
  divergence the mode toggles and the pour restarts from the hover.
- P2 POUR: mouth-anchored trajectory — the mouth glides from the hover to
  tray-local (0, 0, 0.135) while the roll angle ramps to 105 deg (~26 deg/s),
  escalating to 120/135/150 deg until the can actually exits (outcome-driven,
  sag-tolerant). The can slides out, falls ~70 mm, lands in the tray; decant +
  deliver latch to 0.60; asserted NOT success (carafe aloft).
- P3 RETRACT: reverse the pour to upright, translate back over free ground, lower
  onto the carafe's own vacated spawn slot, release wrench, settle -> success
  live, score 1.0.
- P4 >= 3 simulated seconds hands-off persistence (10 x 40 steps, success asserted
  every block), then `SIM_GEN_SOLVE: SUCCESS`.

Passes on >= 2 seeds; `SIM_GEN_SCORE` printed at each phase boundary is
non-decreasing (0 -> 0 -> 0.60 -> 1.0 -> 1.0).

## Franka embodiment argument

- **The carafe** is the only manipulated object: outer flats 70 mm across (within
  the 80 mm jaw span), 158 mm tall with clear air on all sides at its ground slot
  — a side grasp at mid-height (z ~0.08–0.12) with no clutter contact. Carafe +
  can weigh 0.55 kg; the solve's wrench caps (12 N, 1.0 N m) are what a wrist
  applies through such a grasp, well inside Franka payload limits.
- **The pour** is the classic Franka decant: hold the carafe over the tray at
  z ~0.23–0.30 and roll the wrist ~105 deg. The mouth-anchored path keeps the
  carafe body over/beside the tray, never demanding reach beyond ~0.55 m; wrist
  roll range (>±150 deg on joint 7) covers the tilt with margin.
- **The cans** are never grasped — by construction they cannot be; no in-bore
  reach is ever needed (the geometry forbids it, and the solve never does it).
- **Base pose:** at the origin facing +x. The carafe slots (~0.38 m), the tray
  centre (~0.42 m), the hover/pour poses (~0.35–0.45 m at z 0.13–0.30) and the
  set-down all lie inside the ~0.85 m reach envelope at comfortable elbow-up
  postures.

## Smoke battery (12 checks, `SIM_GEN_SMOKE: ALL PASS 12/12`)

1. settle/no-NaN — seeded reset settles finite; both cans captive in their
   carafes, tray grounded; score ~0, no success.
2. randomization readback — tray xy + yaw, carafe xy, red-can xy all differ
   (3 seeds, max-pairwise).
3. red-jar shuffle — over 10 resets the red can starts in BOTH carafes.
4. null policy — 240 idle steps: nothing moves, score ~0.
5. JAR DUMP REJECTED — the whole carafe laid into the tray with the can still
   inside: the can's POSITION is inside the tray volume, yet no credit and no
   success (the transport-only ghost of the seed strategy).
6. captive bore — a real, velocity-capped 5 N lateral push scoots the carafe
   >= 20 mm along the ground; the can never leaves the bore, no credit.
7. partial tilt — the carafe HELD at 60 deg for 2 s: the can stays in the bore —
   only a past-horizontal pour discharges it.
8. decant near-miss — the red can settled on open ground: exactly the 0.30 decant
   latch, no success.
9. goal + live judge — the goal state constructed -> success TRUE (positive
   control); the can then removed to the ground -> success back to False while
   the latched 0.60 survives.
10. wrong object — CORN settled in the tray: zero credit; with the red can ALSO
    in the tray, success is STILL False (corn-exclusion clause).
11. carafe in tray — red can delivered but the red carafe parked standing inside
    the tray: no success.
12. frames.npz — video captured to CWD.
