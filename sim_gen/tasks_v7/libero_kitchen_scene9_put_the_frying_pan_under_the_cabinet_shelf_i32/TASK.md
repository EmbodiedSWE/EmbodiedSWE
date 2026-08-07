# Task `libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf_i32` — Gear-Train Dial Repair

Scene: `gear_train_dial` (env `simgen.gear_train_dial`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf`
— "put the frying pan under the cabinet shelf": pick one loose object off a surface and set
it down inside a fixture's designated region; the moment the object's position enters the
region bbox, the task terminates.

Kept from the seed: one loose target object that must be picked off the floor and installed
at a specific spot on a fixed fixture, a look-alike distractor object nearby (identity
control), and a fixture whose geometry defines where the object belongs.

## Strategic difference

**vs the seed:** In the seed, placing the object IS the task. Here, placing the blue toothed
gear over the empty middle axle is demoted to a *worthless-alone middle step* (0.20 credit,
no success). The judged outcome is a MACHINE CONFIGURATION: a three-wheel spur-gear train is
missing its middle wheel; the agent must (1) diagnose the drivetrain gap, (2) select the
TOOTHED part over an identically-bored SMOOTH decoy disc, (3) seat it so its teeth
interleave with both neighbours, and (4) crank the green drive pin so torque transmitted
THROUGH TOOTH CONTACT rotates a caged, hand-inaccessible output wheel until its red marker
bar points at the magenta stripe (±12°) — direction and amount re-randomized every episode.
The seed's "object in region" check becomes a seat predicate plus a **gated angular-error
account** that only credits output rotation produced while the train is physically complete
and the per-substep motion is plausible. A solver needs a different plan (repair → operate
to a dial target) and different code (assembly predicate, transmission accounting,
configuration goal), not a bbox test.

**vs the rest of the corpus read this session (31 tasks):** the corpus has pick-and-place,
stacking, drawers/doors (articulated open/close), insertion, balance beams/see-saws,
chutes/ramps, hooks, herding, checkers arrangement, bulb unscrew, box-close, and my
carousel airlock (i9: payload swept by a pre-installed rotor the agent spins). No corpus
task has (a) tooth-mesh POWER TRANSMISSION as the load-bearing interaction, (b) mechanism
COMPLETION by part selection (functional part vs geometrically-seatable decoy), or (c) a
pure CONFIGURATION goal (dial azimuth) where the goal body is enclosed and can only be
moved through the mechanism. i9 moves an object through a machine; here NOTHING is
delivered anywhere — the machine's own state is the goal. The drawer task (i11) and other
fixture tasks judge object positions; none judge a driven internal degree of freedom.

## Scene summary

A kinematic FRAME (randomized xy + free yaw): base plate 58×34 cm with three stub axles
(r 11 mm, 58 mm) at local x = −15/0/+15 cm. On the −x axle: the ORANGE 8-tooth driver gear
with a hollow riser and radial arm carrying the GREEN crank pin (r 8 mm, top ~17 cm).
On the +x axle: the pale output wheel with a RED marker bar, enclosed by a 20-segment cage
(inner r 98 mm, 175 mm tall) whose only wall opening is the MESHING OPENING facing the
middle axle (sized to the idler's tooth sweep — the idler drops in from above and fills it
once seated), capped by an annular roof leaving a viewing hole over the marker; one cage
segment + roof piece at output-local
azimuth +90° are MAGENTA — the target mark. The middle axle is EMPTY. Loose on the floor
(random sides/jitter/yaw): the blue TOOTHED idler gear and the blue SMOOTH decoy disc,
identical bores. Wheels are low-friction (μ 0.10/0.08) with 5.5 mm bore slop and 20 mm of
tooth overlap (~31° backlash); the decoy's rim clears both neighbours' teeth by 7 mm.

Per-episode randomization (readback-verified): frame xy + free yaw, driver yaw, SIGNED
initial dial error U[100°,160°] (crank direction changes!), idler/decoy side swap + jitter
+ yaw. `describe()` gives the full layout and recipe; it alone suffices to solve the task.

## Rubric (latched, non-decreasing)

- 0.10 · approach latch (idler toward the middle axle, normalized by spawn distance d0)
- 0.20 · seat latch (idler seated: bore on the axle within 10 mm, hub flush on the plate,
  upright)
- 0.45 · alignment latch — **gated account**: every substep, the change in marker error is
  credited only while idler AND driver are seated and |Δψ_output| < 0.12 rad (a teleported
  yaw write is a radians-scale jump — discarded); error increases are always charged.
  Progress = (err0 − err_acc)/(err0 − tol).
- Success (score 1.0): train complete ∧ marker within 12° of the stripe ∧ err_acc < tol
  (the alignment was net-driven through the physical train) ∧ everything settled. Base
  credit capped at 0.85 without success. Null policy ≈ 0 (initial error ≥ 100° ≫ 12°).

Cheat paths closed: teleported dial (account keeps the debt), trainless cranking or direct
output rotation (gate closed — this, not the cage wall, is the load-bearing guard),
smooth-decoy substitution (no transmission by construction, asserted in `__post_init__`),
working the output directly (caged behind a tall wall + roof; the only opening faces the
middle axle and is filled by the seated idler). Back-driving from the output side is
geometrically impractical (caged) and would still be honest mesh physics if attempted.

## Teleport solution (`solve.py`) — legitimacy certificate

- **P0** settle + layout readback (frame pose, spawns, signed initial dial error, d0) —
  proves seed-dependent randomization from stdout.
- **P1 transport only:** the toothed idler is teleported to a hover pose in FREE SPACE,
  bore centred over the empty middle axle, 10 mm above the axle tip. Installation is the
  free fall (bore captures the axle) plus, if it lands tooth-on-tooth, a speed-governed
  wiggle (±2 rad/s alternating z-torque ≤ 0.06 N·m, 1.5 N press — a hand seating a part)
  until `idler_seated()` reads True. Re-drop retries at other yaws. The decoy is never
  touched; the output is never written after reset.
- **P2 mechanism drive:** torque governor on the driver — τ_z = clamp(2.0·(ω_tgt − ω_z),
  ±0.20 N·m), ω_tgt = ±1.2 rad/s steady (slower speeds chatter inside the ~31° backlash
  without net transmission; a per-step stop check + brake lands within ~1° of the
  threshold), direction = shorter way
  (two meshes ⇒ output co-rotates 1:1). Tooth contact carries driver → idler → caged
  output. Cut + brake, settle; re-crank passes if the stop overshot. All wrenches are pure
  world-z (immune to the pod-dependent external-force frame-drag quirk).
- **P3 persistence:** hands-off ≥ 3.3 simulated seconds; `SIM_GEN_SOLVE: SUCCESS` only if
  success still holds and the score never decreased.

`SIM_GEN_SCORE` printed at every phase boundary; asserted non-decreasing. Verified on
seeds 0 and 7.

## Embodiment argument (Franka, single arm)

- The idler is a 17 cm gear, 3.2 cm hub, mass 0.25 kg on open floor: graspable across the
  hub or across one tooth (18 mm wide < 80 mm jaw), unobstructed from above.
- Seating: hold the gear over the middle axle (axle top 78 mm up — open sky above it; the
  cage is 5 cm away laterally) and lower/release; 5.5 mm bore slop over an 11 mm-radius
  axle is a forgiving drop, and the tooth-on-tooth case is resolved by the small wrist
  wiggle every human uses — exactly P1's governed jiggle.
- Cranking: the GREEN pin (16 mm dia < jaw span) rides a 46 mm crank at z ≈ 108–170 mm,
  the tallest thing on the driver; a Franka can cage it between closed fingers and drive
  circles, or push it tangentially. Sustaining 1.2 rad/s needs ~3.3 N at r = 46 mm
  (τ ≈ 0.15 N·m) — far below Franka payload; the crank circle (r 5.5 cm) fits easily in
  the wrist workspace with the base ~0.55 m from the plate centre, which also reaches both
  floor spawn bands (~0.32 m from the frame on either side).
- Nothing requires reaching into the cage — that is the point: the output is only
  operable through the train.

## Ordering

Seat-before-crank is physically inherent (no mesh, no transmission) and the account's gate
makes it the only scoring order: crank rotation before the idler is seated moves nothing
that earns credit, and any output motion while the train is incomplete earns nothing (or
charges, if it increases the error). No arbitrary ordering constraints beyond what the
mechanism imposes.

## Checks (`smoke.py`, 14)

1. settle/no-NaN + seated-fixture readback; 2. baseline score ≤ 0.03; 3–4. randomization
by readback across 6 seeds (frame xy/yaw, driver yaw, SIGNED dial error, idler/decoy xy,
d0); 5. null policy ≈ 0; 6. SEED-STRATEGY end state (part carried onto the plate beside
the axle and left): NOT seated, ≤ 0.15; 7. teleported dial with an honestly-completed
train (every instantaneous predicate passes): err_acc keeps the debt, rejected; 8.
trainless rotation (output slow-torqued onto the stripe, physically plausible motion, gate
closed): rejected; 9. decoy substitution (smooth disc seated, driver verified spinning
> 0.8 rad/s): no transmission, dial moves < 8°, rejected; 10. near-miss seat 35 mm off
the axle: rejected; 11. latched credit survives idler removal; 12. approach monotonicity;
13. rejection audit (no check ever saw success); 14. final no-NaN. Frames recorded to
`frames.npz`.
