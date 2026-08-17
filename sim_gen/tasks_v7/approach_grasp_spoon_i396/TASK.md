# idler_gearbox (`approach_grasp_spoon_i396`)

Repair a broken powertrain by dropping the missing idler gear onto its bearing peg,
then crank the completed gear train so its rack pushes the untouchable payload cube
out of a roofed tunnel into the walled catch pocket.

## Seed provenance

- **Seed:** `rlbench/approach_grasp_spoon`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/approach_grasp_spoon.py`) —
  approach and GRASP a spoon lying among clutter and carry it toward a container.
- **Seed strategy:** prehensile pick-carry-release of the payload itself; the whole
  task is one grasp-transport of the goal object.

## What changed and why it is strategically different

The seed's core verb — *grasp the payload and carry it to the goal* — is made
physically void, and the grasp is redirected at a machine part that must then be
actively DRIVEN:

1. **The payload can never be grasped, carried, or even touched.** The orange 40 mm
   cube sits inside a fully roofed tunnel: rear port 24 × 24 mm (≪ cube, asserted),
   roof closed, exit hanging over the pocket. The ONLY mover that reaches it is the
   machine's own rack nose sliding through the port. smoke check 6 executes the
   seed's plan as the strongest possible carry — a *teleport* of the cube into the
   pocket, settled — and shows the rubric rejects it (transmission latch never
   fired): even a magic carry scores ~0.
2. **The grasped object is infrastructure with a selection problem:** two brass
   wheels with identical collar hubs and identical bores lie on the ground — the
   toothed IDLER and a smooth-rimmed BLANK whose rim reaches *no* mesh circle
   (asserted). Both seat perfectly on the peg; only the idler transmits. smoke
   check 8 physically installs the blank and cranks: the crank spins freely, the
   rack never moves.
3. **Installation alone delivers nothing — the machine must be powered.** After the
   idler seats, the robot must turn the red crank counter-clockwise through a
   sustained ~250°: crank gear (8T) → idler (10T) → rack, converting rotation into
   ~13 cm of rack travel. This is a continuous force-closure interaction, not a
   place-and-watch: null policy after seating scores 0.25 max and never succeeds.
4. **Execution order is physics-forced:** cranking before the install spins a
   disconnected gear (smoke check 7); wrong-direction cranking locks the rack
   against its retracted stop (smoke check 10). Only install-then-crank-CCW works.
5. **The rubric is teleport/wrench-proof:** transmission credit accumulates SIGNED
   per-step rack increments taken WHILE the idler is seated — forward steps capped
   at 2 mm, backward steps subtracted in full, total floored at 0. A teleport jump
   credits one capped step and the depenetration oscillation it excites cannot
   ratchet (forward creep is wiped by the full-value backward strokes), so only
   sustained NET forward geared motion earns credit; success() requires that latch.
   smoke check 11 teleports the rack to full stroke in nine jumps with the idler
   seated — the cube is physically shoved forward, yet the credited advance stays
   below trans_min and success is rejected.

Distinct from the corpus siblings read this session:

- **hit_ball_with_queue_i77** (seat a bridge, then a *passive gravity ride*): there
  the install is the whole job and physics finishes alone. Here seating the idler
  delivers nothing — the robot must then continuously drive the mechanism.
- **close_drawer_i386** (push a jointed drawer shut): single direct push on the
  jointed body. Here the driven joint (crank) is two gear meshes away from the
  payload, and the payload is not jointed at all.
- **draw_triangle-style direct pushes / tipping siblings:** the payload here is
  untouchable by construction; all force reaches it through a rotary-to-linear
  transmission the robot must first complete.

Loophole audit: the port (24 mm) refuses fingers and the cube both ways; the rack
teeth stay behind the pusher nose so nothing but the flat nose ever touches the
cube (asserted); the rack is captive behind a wall + roof strips and its west end
never leaves the guide, and even a wrenched/teleported rack fails the transmission
latch (smoke 11); the blank transmits nothing (rim < every mesh circle, asserted);
backlash (10 mm) exceeds twice the bore slop (asserted) so a worst-case-seated
idler still meshes.

## Teleport solution outline (solve.py)

1. **P0** reset + settle; layout readback printed (seed-provable); rack at q=0 and
   score ~0 asserted.
2. **P1 (teleport = transport only):** one pose write stages the idler HOVERING
   62 mm above its seat, bore roughly over the peg with a deliberate 3 mm lateral
   offset — asserted NOT seated.
3. **P2 (contact dynamics):** release → gravity drops the bore over the stepped peg
   taper onto the boss. A wheel hanging on the peg (bore wedge / tooth-top rest)
   gets an escalating physical unjam — 5 N press + 0.05 N·m spin on the wheel with
   a ±0.06 N·m crank wiggle, the wiggle a hand doing this install would make; a
   failed drop earns a re-hover with rotated tooth phase (≤ 4 attempts). The
   scene's `idler_seated()` readback confirms → `SIM_GEN_SCORE 0.25`.
4. **P3 (contact dynamics — the heart):** velocity-regulated torque (0.15 N·m,
   escalating to 0.6 on stall; ω ≤ 2 rad/s) about the crank's OWN body z axis (the
   hinge axis — immune to wrench-frame drag) turns it counter-clockwise; every
   newton of rack thrust flows through both gear meshes. A one-shot direction probe
   flips the sign if the rack does not advance. Rack driven to q > 126 mm; the nose
   pushes the cube through the tunnel and off the lip → transmission + eject
   latches → `SIM_GEN_SCORE 0.65`.
5. **P4 (hands-off):** torque cut; the cube falls off the lip and settles inside
   the walled pocket on gravity alone → `SIM_GEN_SCORE 1.0`.
6. **P5:** ≥ 3.3 further simulated seconds hands-off; success must persist →
   `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge on **seeds 0 and 1** (rc=0; distinct layouts in the
readback).

## Embodiment argument (single Franka, OSC, parallel jaw)

- **Base pose (frame-local):** (x, y) = (−0.05, −0.28), facing +y — every contact
  below is within ~0.45 m reach: wheel ground slots at y = −0.22, the peg at
  (0, 0), the crank knob orbit centred (−0.049, −0.049) radius 60 mm.
- **Idler / blank:** pinch the raised collar hub — a 44 mm octagonal boss standing
  18 mm proud of the gear face (44 ≪ 80 mm jaw span, asserted with 20 mm margin);
  wheels lie flat on open ground with the collar up, so a top-down pinch is
  unobstructed. Lift, carry ~0.3 m, hold the bore over the peg and open the jaw —
  the stepped peg (12 → 11 → 8.5 mm radius taper) funnels the 15 mm bore the last
  centimetre, exactly what the solve's hover-and-release (plus press-and-twist
  wiggle) emulates.
- **Crank:** the red knob is a 16 mm post standing 32 mm tall at the end of the
  overhead arm, 158 mm above the floor — graspable or push-cuffable by the closed
  fist; driving it is a horizontal circular sweep of 60 mm radius at constant
  height (a canonical OSC circle), force ≈ 0.15–0.6 N·m / 0.06 m ≈ 2.5–10 N
  tangential. The arm clears the guide roof (136 mm > 62 mm) all the way around.
- **Cube:** untouchable by design — never grasped, never pushed by the robot.

## Execution order (declared)

REQUIRED order: (1) seat the toothed idler on the peg, (2) crank counter-clockwise
until the rack ejects the cube. The order is enforced by physics (cranking first
moves nothing — smoke 7), and the rubric's transmission credit additionally
accrues only while the idler is seated.

## Rubric

- `0.25 seated` (latched) — the IDLER (toothed wheel specifically) ever seated on
  the boss: centred on the peg, in the height band, upright.
- `0.20 transmission` — NET geared rack advance (signed per-step increments:
  forward capped at 2 mm, backward subtracted in full, floored at 0) accumulated
  WHILE seated reaches 60 mm (≥ 30 honest forward steps; an honest full-stroke
  drive earns ~120 mm while a full-stroke teleport barrage peaks near ~31 mm of
  depenetration creep — 2× margin both ways).
- `0.20 ejected` (latched) — cube past the tunnel lip AFTER transmission.
- `1.0 iff success()` — cube settled inside the pocket AND the transmission latch
  earned. Non-success cap 0.65. Null policy exactly 0.

## Checks (smoke.py — rejection-only battery, recorded to frames.npz)

1. settle/no-NaN: cube in tunnel, wheels on ground, rack at q=0, all still
2. reset score ~0, no success
3. randomization readback: machine yaw + xy + crank start angle real
4. randomization readback: wheel slot swap flips; per-wheel jitter + cube depth real
5. null policy (240 steps) → score ~0
6. SEED STRATEGY: cube magic-carried (teleported) into the pocket, settled →
   in_pocket true but success REJECTED (no transmission), score ~0
7. crank without idler: crank demonstrably spins, rack never moves → score ~0
8. wrong object: BLANK physically seats on the peg (readback) + cranked → rack
   unmoved, no seat credit, score ~0
9. off-peg: idler dropped on the well floor → NOT seated, no credit
10. wrong direction: seated idler + clockwise cranking → rack pinned at q=0 stop,
    no transmission, ≤ seat credit
11. anti-cheat: rack teleported to full stroke (idler seated) → credited advance
    ≪ trans_min, eject never latches, success REJECTED
12. near-miss: cube settled just short of the lip → not in pocket
13. beside-pocket: cube on the ground outside the pocket wall → y-band rejects
14. latched credit: probe-seated idler earns exactly 0.25, survives removal, never
    success
15. rejection audit: success() never True anywhere in the battery
16. final no-NaN

`SIM_GEN_SMOKE: ALL PASS 16/16` expected; solve.py separately proves acceptance.
