# light_bulb_out_i204 — Eject the dead bulb through the floodlight's rear service port

**Scene:** `simgen.floodlight_eject` (`FloodlightEjectScene`, robot="null")
**Seed:** `rlbench/light_bulb_out`
(`RoboVerse/roboverse_pack/tasks/rlbench/light_bulb_out.py`)

## Seed provenance and what changed

The RLBench seed is a direct removal: a bulb sits in the lamp's open socket and
the plan is **grasp the bulb, unscrew it with a wrist rotation, and carry it to
a holder** — one grasp ON the goal object, one continuous wrist rotation, one
free-space transfer to an open receptacle.

This task keeps the goal predicate ("the dead bulb ends up out of the lamp and
in a receptacle") but replaces the entire plan skeleton:

1. **The bulb is untouchable.** It rests in a seat pocket 128 mm deep inside a
   floodlight shroud whose 66 mm mouth cannot pass the 48 mm bulb plus two jaw
   fingers, far deeper than a 54 mm finger — asserted in `__post_init__`. A
   thin tool through the mouth can only PRESS the bulb deeper into its seat
   (smoke check 6 pushes with 3× its weight and it stalls on the seat back).
   The seed's grasp-and-unscrew schema transfers zero.
2. **A tool, threaded through a rear access, does the pushing.** The only
   interface is a rod-sized service port through the BACK of the housing: pick
   the 300 mm push rod off the floor, thread it through the stepped countersink,
   guide tube and port hole, and push the bulb forward by real contact.
3. **A real retention threshold, then gravity delivers.** The bulb sits behind
   a low amber ridge — a genuine quasi-static force threshold (~0.43 N for the
   50 g sphere; smoke check 7 holds a sub-threshold 0.25 N push for 2.5 s and
   the bulb stays put). Past the ridge the shroud floor is a downhill ramp:
   the bulb rolls out of the mouth on its own and falls ballistically into the
   green disposal bin below. The fixture's geometry — not the arm — performs
   the delivery; the bulb is NEVER grasped, NEVER screwed, NEVER carried.
4. **Live physics, not scripted state.** There are no joints and no flags at
   all: four rigid bodies (kinematic housing + bin, free bulb + rod), authored
   friction/restitution/inertia, and every predicate (rod tip through the port,
   bulb past the ridge, transit through the mouth, resting in the bin) is read
   from live poses.

So the plan skeleton changes from *"grasp, unscrew, carry, place"* to
*"acquire a tool, thread it through a rear access, push the goal object over a
retention detent, let gravity deliver it"* — tool-mediated indirect extraction
with a ballistic hand-off, instead of direct manipulation of the goal object.

## Why strategically different from every examined sibling

- **Seed (`rlbench/light_bulb_out`):** see above — the direct
  grasp/unscrew/carry strategy is physically rejected (mouth denial asserted +
  force-probed); here the goal object is only ever touched by the rod tip.
- **`light_bulb_in_i120` (turnstile delivery):** there a *carrier mechanism*
  (free revolute turnstile) is operated out–load–in to deliver a bulb INTO a
  sealed cabinet; the arm places the bulb onto the carrier. Here there is no
  mechanism DOF anywhere (no joints at all), nothing is delivered inward, and
  the object is never placed: an inserted TOOL pushes it over a passive detent
  OUT of the fixture and gravity does the transfer. Opposite goal direction,
  disjoint machinery (tool-through-port + ridge + ballistics vs. jointed
  carrier), different forced structure (insert-push-release vs. out–load–in).
- **`lamp_off_i101` (twistlock unplug):** there the plug is identified by
  cord-trace among decoys, GRASPED, key-twisted and pulled — rotation gates a
  translation of the held object. Here nothing is identified (one bulb, one
  rod), the extracted object is never grasped, and there is no keying or
  interlock: the challenge is threading a long tool through a narrow port
  chain and crossing a force threshold.
- **`robobench/packing/pen_holder` (exemplar, not corpus):** direct drop of
  free objects into an open receptacle — exactly the schema removed here (the
  bulb cannot be picked up at all; it reaches the bin only by ejection).

## Solution outline (solve.py — teleport = TRANSPORT ONLY)

External-wrench plant recipe: `enable_external_forces_every_iteration: True`
in the scene's physx cfg; rod inertia authored diagonal (It = 4.5e-4) so the
6-DOF wrench servo (the virtual grasp) is auditable and discretely stable:
kd·dt/m = 0.33 < 1, ka·dt/It = 0.074 < 1; angular damping is TRANSVERSE-ONLY
(the 6 mm rod's axial inertia is ~1e-6 — damping that axis would explode).

- **Phase 0 (reset):** settle 0.5 s, read back layout, assert score ≤ 0.02.
  `SIM_GEN_SCORE` ≈ 0.000.
- **Phase 1 (acquire + align, transport; insert, dynamics):** the rod is
  teleported from the floor to free air behind the port (tip at housing
  x = −0.29, just outside the countersink entry), axis on the port axis — pure
  transport. The wrench servo (position spring + damping + gravity
  feedforward, alignment torque; force cap 5 N, lead clamp 50 mm) then walks
  the tip through countersink → tube → port hole to mid-tube by real contact.
- **Phase 2 (through the port):** tip advances past the back-wall plane into
  the probe window. `SIM_GEN_SCORE` 0.150.
- **Phase 3 (push over the ridge, dynamics):** the tip meets the bulb's rear
  pole and pushes; the lead clamp bounds the contact force (~1.2 N max vs. the
  0.43 N ridge threshold, 5 N cap). The bulb pivots over the ridge.
  `SIM_GEN_SCORE` ≥ 0.400.
- **Phase 4 (gravity ejection):** the rod HOLDS; the bulb rolls down the ramp,
  flies through the mouth transit window and lands in the bin — nothing
  touches it. `SIM_GEN_SCORE` 0.650.
- **Phase 5 (release, dynamics):** wrench zeroed; the rod comes to rest
  supported inside the guide tube (CoM within the supported span). The bulb
  settles in the bin → success. `SIM_GEN_SCORE` 1.0000.
- **Phase 6 (persistence):** ≥ 3.5 simulated seconds hands-off; `success()`
  is live state. Only then `SIM_GEN_SOLVE: SUCCESS`.

**Execution order is REQUIRED and physically forced:** front-door extraction
is impossible (mouth denial asserted; smoke 6 proves pressing only seats the
bulb deeper), the ridge cannot be crossed without a super-threshold push that
only the port-threaded rod can deliver (smoke 7), and the bulb reaches the bin
only along the ramp-and-fall path (it cannot pass the port; the housing is
kinematic). Insert → push → gravity is the only causal chain; the rubric's
latches mirror exactly this order.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `probe_latch` (0.15): a rod tip ever through the service port (narrow
  on-axis window inside the back wall, before bulb contact).
- `unseat_latch` (0.25): bulb ever past the ridge inside the channel band.
- `eject_latch` (0.25): bulb ever inside the mouth transit window (a narrow
  box just outside the mouth on the ballistic exit path — teleported
  constructions elsewhere never touch it).
- success (to 1.0): bulb resting INSIDE the bin (bin-frame containment window
  that accepts every physically-in-bin resting pose and rejects beside-bin and
  rim-perch — asserted) AND settled.
- Latches are transient-achievement credit (torch.maximum); cap 0.65 without
  success (float32 boundary honoured with +eps in smoke). Null policy ≈ 0
  (bulb spawns seated behind the ridge, rod on the floor).

## Embodiment argument (single Franka, parallel jaw 80 mm, OSC)

Base at (0, 0) on the floor, facing +x: the rod spawn zones (x ≈ 0.24–0.36,
y ≈ ±0.16–0.28), the port entry (housing x 0.68–0.76 minus 0.285 ≈
world 0.40–0.48 at z = 0.244) and the whole insertion stroke (hand on the rod's
rear half, world x ≈ 0.30–0.48, z ≈ 0.24) sit inside a 0.75 m reach disc at
comfortable height; the arm never needs to reach the mouth, the shroud
interior, or the bin.

- **Rod pick:** a 6 mm cylinder lying on open floor is a textbook top-down
  pinch for the 80 mm jaw (48 mm free above the floor plane at grasp depth);
  regrasp to an axial hold near the rear end is standard.
- **Insertion:** the stepped countersink (52 → 36 → 24 mm apertures funneling
  to the 22 mm hole) forgives ±15 mm of tip placement error; the servo's 5 N
  cap and 30 mm/s feed are gentle jaw-held pushing. The Franka wrist holds the
  rod horizontal at z = 0.244 — mid-workspace.
- **Push:** the ridge needs 0.43 N through the rod — trivial for the arm; the
  lead-clamped advance is exactly a compliant guarded move.
- **No blind reach:** rod, port collar (blue), amber ridge line and bin are
  all visible from outside; the bulb's seat depth only matters through the
  felt contact, which the guarded push provides.

## Checks (smoke.py — rejection battery, 17 checks)

1. Clean reset: states finite; bulb seated behind the ridge, rod on the floor,
   bin empty.
2. Score ~0 / no success at rest.
3. Randomization (8 seeds): housing xy + yaw vary; seated bulb + bin TRACK the
   housing frame.
4. Randomization: rod spawn side / xy vary.
5. Null policy: 240 idle steps, score ≤ 0.02.
6. **Seed-strategy family (front door):** regulated press (≤ 1.5 N ~ 3× bulb
   weight) through the mouth moves the bulb ≥ 2 mm (non-vacuous) and stalls it
   on the seat back — it never advances toward the mouth, ~0.
7. **Ridge retention:** quasi-static creep to ridge contact, then a constant
   0.25 N push (< 0.43 N threshold) held 2.5 s — never crosses, settles back
   pocketed, no unseat credit, ~0.
8. Ejected-but-missed: bulb settled on the floor beside the bin — rejected by
   the bin-frame window, no transit credit.
9. Rim perch: bulb balanced on the bin rim reads above the height window —
   rejected, removed before it topples.
10. Settle gate: bulb inside the bin but moving — not success; removed.
11. Wrong object: the ROD dumped into the bin, bulb still seated — ~0.
12. Rod-in-port (held by state writes): probe latch fires, score == 0.15 + eps
    ONLY — mere insertion earns nothing more.
13. Cap: all three latches constructed, bulb yanked away mid-flight → score ==
    0.65 cap (+eps), NOT success.
14. Latched credit: 40 further steps, score unchanged, in_bin stays False.
15. Rejection audit: success() never True at any judged point.
16. Final no-NaN.
17. Camera: ≥ 20 rgb frames captured → `frames.npz`.
