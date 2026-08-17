# meat_off_grill_i355 — Serve the patty: scoop the un-graspable meat off the walled griddle

- **Scene**: `griddle_spatula` — env `simgen.griddle_spatula` (robot=`null`)
- **Seed task**: `rlbench/meat_off_grill` (RoboVerse `roboverse_pack/tasks/rlbench/meat_off_grill.py`)

## Provenance

The seed puts two free rigid meat pieces (chicken, steak) resting on a grill; the demo
trajectory grasps each free body, lifts it off the grill, and sets it beside. The
whole task is direct prehension of unconstrained bodies: the hand touches the MEAT,
nothing restricts the contact, and no tool exists.

This task keeps the theme — cooked meat must come off the heat — and rebuilds the
physics so the seed's plan is impossible and a tool-mediated plan is mandatory.

## Strategic difference

**Vs the seed** (`rlbench/meat_off_grill`):

1. *The meat cannot be grasped.* The patty is a 90 x 90 x 14 mm slab lying flat:
   90 mm exceeds the Franka jaw's 80 mm opening in EVERY horizontal direction, and
   the 14 mm edge cannot be pinched because one finger would have to pass below the
   resting plane. Direct prehension — the seed's entire plan — is dead on arrival.
2. *The meat cannot be slid off either.* The griddle top is enclosed by a 30 mm rim
   wall on all four sides (more than twice the patty height). Smoke's rim probe
   shoves the patty quasi-statically into the rim at ~10x the friction need and it
   stays on the griddle: there is no push/plow exit.
3. *A tool is the only way.* The provided free spatula (thin blade, slick tapered
   leading ramp, jaw-sized vertical grip post) must be wedged UNDER the patty. The
   far rim wall — the very thing that closes the slide-off exit — is the backstop
   that makes wedging work: the patty pins against it and rides up the taper onto
   the blade. Then: lift the loaded blade over the rim, carry the patty as a passive
   friction-held rider, tilt nose-down over the dish so it slides off and lands
   FLAT, and finally park the tool ≥ 20 cm from the dish (success is gated on the
   hand having put the tool DOWN and away — no hover-and-hold finish).

So the solver needs a different plan (grasp the TOOL not the meat; press-and-drive a
wedge against a backstop; transport a rider; tilt-pour; park the tool) and a
different code structure (6-DOF tool pose/force control through a
wedge/lift/carry/discharge/park sequence instead of grasp-lift-place).

**Vs every other corpus task read** (survey of all tasks_v7 TASK.md files, plus full
reads of `meat_off_grill_i64`, `hockey_i325`, `close_grill_i8`):

- `meat_off_grill_i64` (same seed): captive rings threaded on a spit rail, metered
  fingertip push off the tip — no tool, no grasping at all, topological captivity.
  Here nothing is captive: the closure is metric (jaw span, rim height), the
  manipulated object is a free TOOL grasped conventionally, and the payload rides
  on the tool through five distinct contact phases.
- `close_grill_i8` (same seed family): free-body grasp-and-place of coals; here the
  goal body is un-graspable and never touched by the hand.
- `hockey_i325`, `scoop_with_spatula`-adjacent silos in the corpus: hockey strikes a
  projectile with a stick (impulsive, ballistic); no corpus task wedges a blade
  UNDER a goal object against a backstop, carries it as a friction rider, and
  discharges by tilt-pour. The `held_container pour` family pours FROM a container
  the payload starts in; here the payload must first be ACQUIRED onto the carrier
  through the wedge move.
- Insertion/extraction silos (`peg_insertion`, `plug_charger`, `pen_holder`): those
  move the goal body directly or eject it with a ram; none interpose a carried flat
  tool between hand and goal for the entire task.
- Tool-away parking as a success clause (the task is not done while the tool lies
  at the dish) appears in no other corpus task.

No corpus task combines: un-graspable-by-metric goal object + walled-in fixture
closing the push exit + wedge-acquire onto a carried tool + tilt-pour delivery +
tool-parked-away success gate.

## Scene

Procedural compound spawners only (no external assets):

- **griddle** (kinematic): dark hot-plate slab, top 0.10 m up, inner 0.40 x 0.24 m,
  30 mm rim wall on all four edges (μ 0.05), ember-glow stripe (visual only — a
  proud collider would park slides), top μ 0.50.
- **dish** (kinematic): shallow white square dish (160 mm inner, 10 mm walls) on the
  ground on a RANDOM side (+y or −y of the griddle), μ 0.40.
- **patty** (dynamic): 90 x 90 x 14 mm brown slab, sear stripes visual-only, mass
  0.12 kg (explicit MassAPI), μ 0.70, high angular damping (soft meat). The seared
  edge has curled up off the heat: the underside is an inset 42 mm pad, leaving a
  4 mm-high, 24 mm-deep rim undercut around the patty — the wedge's admission
  geometry. Its open mouth (12 mm) admits the 2.4 mm blade tip with clearance;
  its inner half is a slick greasy chamfer (μ 0.10 — the curled edge itself)
  bridging the pad's top edge down to just above the bottom plane, so the
  advancing tip only ever meets INCLINED faces — the wedge is continuous, with
  no flat-on-flat jam anywhere in the path. The 4 mm gap is far below finger
  scale, so the undercut creates no grasp affordance.
- **spatula** (dynamic): blade 90 x 100 x 6 mm (μ 0.10), slick tapered leading ramp
  (40 mm at 10°, μ 0.02 — the edge the patty climbs), red vertical grip post
  (25 mm square, 110 mm tall — jaw-sized, and the rear backstop for the rider),
  mass 0.30 kg with authored CoM + diagonal inertia (wrench-servo stability).

**Geometry honesty** (asserted in `__post_init__`): patty > 82 mm (un-graspable
flat); grip < 70 mm (graspable); rim overtops the patty by > 10 mm; the rim
undercut admits the ramp's blunt leading end with ≥ 1.5 mm clearance yet stays
≤ 6 mm (no finger); the chamfer shields the pad's vertical face (its outer edge
gap is smaller than the tip thickness) and stays ≤ 25° (wedgeable); patty spawn
band clears the rims and leaves wedge room; the tool-entry pose behind the patty
clears the −x rim; the dish clears the griddle; the patty's diagonal fits the dish;
the park spot satisfies the tool-away clause for every dish sample.

**Randomization** (readback-verified in smoke): griddle xy ±3 cm + yaw ±10°; dish
side FLIPS (±y) with dx ∈ ±5 cm; patty griddle-local x ∈ [2,7] cm, y ∈ ±6 cm, yaw
±12°; spatula spawn xy ±3 cm + yaw ±20° on the ground opposite the dish.

## Rubric

Latched in `post_step` (transient achievements keep credit; removing the patty
afterwards does not reduce the score — smoke check 12):

- 0.15 × engage — running max of the patty's ON-GRIDDLE travel (ramp 4 cm; ~0 for
  idling; a teleported exit earns nothing)
- 0.30 × scooped — patty ever SUPPORTED BY THE BLADE (rides in the spatula's body
  frame; latched)
- 0.25 × transit — patty ever carried on the blade OUTSIDE the griddle footprint
  above griddle height (latched)
- cap 0.70 without success; **1.0 iff `success()`**: patty settled FLAT (≤ 15° tilt)
  on the dish floor fully inside the walls (tight z ceiling: a patty still riding
  the 6 mm blade sits ≥ 8 mm too high and does NOT count — smoke check 11) ∧ still
  ∧ spatula set down ≥ 20 cm from the dish, low (≤ 8 cm) and still ∧ all states
  finite.

## Teleport solution (solve.py)

**One teleport, transport only**: the free spatula is moved from its spawn spot to a
hover pose on the griddle behind the patty (a pose the arm reaches by carrying the
tool). Everything load-bearing is contact dynamics via a 6-DOF external-wrench PD
servo on the spatula — the bounded wrench a hand holding the grip post applies:

- **P0** settle 60; layout readback (fixture pose/yaw, patty local xy, dish side,
  MassAPI masses — seed-dependence provable from stdout); score ≤ 0.02.
- **P1** wedge: enter SQUARE to the patty's sampled yaw (a corner-first meeting of
  the undercut mouth jams), press the blade down onto the plate (bounded z-press)
  and drive along the fixture's +x at 4.5 cm/s with a lead-clamped carrot; the
  orientation reference tracks the patty's live yaw (rate-limited) while the rim
  squares it, with stall escalation (lead up to 0.22, then a 3° yaw wiggle — how a
  human works a spatula under a stuck patty); the patty pins against the far rim
  and climbs the slick taper onto the blade; break on a 40-step scooped streak.
  `SIM_GEN_SCORE` (~0.45).
- **P2** lift + transit: pitch −6° (grip post = rear backstop), rise to 0.20 m over
  the rim, carry to the dish at 6 cm/s with payload feedforward (gated on a
  physical rider-support test); assert scooped and transit latched.
  `SIM_GEN_SCORE` (0.70).
- **P3** discharge + park: descend over the dish's near third, then PRESSED-CONTACT
  pour — the z-reference drops deep so the lead-clamped PD holds a bounded ~1.8 N
  press pinning the blade nose to the dish floor (the payload feedforward cannot
  track the patty-to-floor weight handover; the press absorbs that error in floor
  reaction instead of blade height, so there is no room to tumble): tilt to +28°,
  steepen to +40° if the patty bridges (leading edge anchored on the μ 0.55 floor),
  and finally pull the sheet out — withdraw the still-pressed blade backward so the
  floor friction strips the patty off flat; then rise, carry the tool to the park
  spot (griddle-local (−0.34, 0), ≥ 38 cm from every dish sample), set it down, cut
  all forces.
- **P4** hands-off success wait, `SIM_GEN_SCORE` (1.0), then ≥ 3.3 simulated s
  persistence with no forces, success still true → `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka + OSC)

Base at the world origin; the griddle at (0.45, 0), the dish at ±0.26 m to the
side, the spatula spawn at (0.20, ∓0.29), the park spot at ~(0.11, 0): every
interaction point sits at radius 0.25–0.60 m, and the highest carry pose (0.20 m,
grip post top ~0.31 m) is comfortably inside the Franka workspace.

Per-object contact strategy: the ONLY object the hand ever touches is the spatula's
25 mm grip post — a standard top-down parallel-jaw grasp. The solve's 6-DOF wrench
servo (|F| ≤ 15 N, |τ| ≤ 0.8 N·m, few-cm/s reference rates) is exactly the wrench a
wrist applies through that grasp: press-and-slide the blade (the wedge, with a small
yaw wiggle), lift/carry a 0.42 kg loaded tool, tilt the wrist ~40° to pour against a
light pressed contact, draw the blade back out, set the tool down. The patty is
never grasped, never directly pushed by the hand, and never teleported.

## Execution-order declaration

No rubric-declared order. The physical structure forces the order (wedge must
precede carry; carry must precede discharge), and the rubric only judges outcomes:
patty flat in the dish, tool parked away. The tool-away clause is stateless — it is
judged live at the end state, not as an event sequence.

## Checks (smoke.py — rejection battery, 14 checks)

1. settle: finite, patty flat on the griddle, spatula on the ground, all still
2. settle: score ~0, no success
3. randomization: patty xy and fixture yaw spreads real (readback)
4. randomization: dish side flips across seeds (readback)
5. null policy: 240 idle steps → score ~0, no success
6. rim retention: quasi-static velocity-limited shove into the +x rim → patty stays
   on the griddle (no slide-off exit — the tool is mandatory), score ≤ 0.16
7. seed strategy: patty "lifted off and set beside" on the ground → score ~0
8. near-miss: patty on the ground beside the dish → rejected
9. near-miss: patty straddling the dish wall → rejected
10. tool-near: patty perfectly served but the spatula lies 10 cm from the dish →
    tool-away clause alone rejects (constructed tool-first: no prefix succeeds)
11. not released: loaded spatula resting ON the dish, patty still on the blade →
    rejected
12. latched credit: constructed scoop fires the latch (~0.30 < 0.70 cap); removing
    the patty leaves the score unchanged, no success
13. rejection audit: success() never true anywhere in the battery
14. final: all states finite

Verdict lines: `SIM_GEN_SOLVE: SUCCESS` (seeds 0 and 1), `SIM_GEN_SMOKE: ALL PASS 14/14`.
