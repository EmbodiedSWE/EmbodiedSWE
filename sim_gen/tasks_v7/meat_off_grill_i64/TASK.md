# meat_off_grill_i64 — Serve the cooked kebab: meter the brown pieces off the spit tip

- **Scene**: `kebab_spit` — env `simgen.kebab_spit` (robot=`null`)
- **Seed task**: `rlbench/meat_off_grill` (RoboVerse `roboverse_pack/tasks/rlbench/meat_off_grill.py`)

## Provenance

The seed puts two free rigid meat pieces (chicken, steak) resting on a grill; the demo
trajectory grasps each free body, lifts it off the grill, and sets it beside. The whole
task is an unordered pick-and-place of unconstrained bodies: nothing restricts how a
piece may leave the grill, there is no wrong object to avoid, and no stopping decision.

This task keeps the theme — cooked meat must come off the heat — and rebuilds the
physics so the seed's plan is impossible and a different plan is mandatory.

## Strategic difference

**Vs the seed** (`rlbench/meat_off_grill`):

1. *No free bodies, no carrying.* Every piece is a ring (42 mm cube with a 26 mm
   square through-channel) threaded captive on a 16 mm square rail. It cannot be
   lifted off, tipped off, or pulled sideways off — smoke's captivity probe hauls a
   threaded piece straight up at ~6× its weight and it stays threaded. The seed's
   grasp-and-lift plan is dead on arrival.
2. *The exit is topological, not spatial.* The ONLY way off is axial travel past the
   rail's free tip, which overhangs a serving tray. The judged transport is sliding +
   free fall, never a carried pose.
3. *A metered stop condition.* The train is FIFO (pieces cannot pass each other);
   cooked (brown) pieces sit outboard, raw (pink) inboard. Pushing the whole train
   from the butt end overruns — raw pieces get served too, which is failure. The
   correct plan is to find the cooked/raw color boundary, reach a fingertip into the
   inter-piece gap there, and push only the cooked group. WHERE you push *is* the
   stop condition. The seed has no analogue of any of this.

So the solver needs a different plan (identify a boundary in a captive train, push at
the boundary, convey off an overhang, let gravity capture, leave the raw group
untouched) and a different code structure (velocity-limited axial push control with a
built-in stop, instead of grasp-lift-place).

**Vs every other corpus task read** (survey of all tasks_v7 TASK.md files, plus full
reads of `hockey_i325`, `close_grill_i8`):

- `close_grill_i8` (same seed family): free-body stacking of coals into a crib — free
  grasp-and-place; here nothing is free and nothing is placed by hand.
- `setup_checkers_i2` / `pen_holder`-style insertion silos: those INSERT free bodies
  into guides; here the judged bodies EXIT a guide they start captive on, and the
  hard part is stopping the exit at the right count.
- `peg_insertion_side_i1` (ramrod eject): pushes a tool through a hole to eject a
  separate payload; here the pushed bodies ARE the payload, the push point (the
  boundary gap) encodes the stop, and part of the train must explicitly stay.
- `pick_single_egad_i3` / `plug_charger_i21` (unhook/unlock then withdraw): those
  need an unlock or re-orientation step before extraction; the ring/rail interlock
  here has no unlock — captivity is permanent everywhere except past the tip.
- `hockey_i325` (this format's template): projectile striking into a goal; no
  captivity, no ordering, no partial-unload metering.

No corpus task combines: captive-train FIFO ordering + partial group unload with a
"leave the rest" constraint + an overhang drop as the only exit.

## Scene

Procedural compound spawners only (no external assets):

- **spit** (kinematic): mast at local x=−0.335, square rail (16 mm, axis z=0.16)
  cantilevered to the free tip at x=+0.24, dark grill bed with an ember-glow strip
  under the span, white serving tray (inner x∈[0.21,0.41], half-width 0.096, walls
  0.082) on the ground under the overhanging tip.
- **pieces** (dynamic, 5 spawned: cooked_0..2 brown, raw_0..1 pink): rings — 4-box
  tube, 42 mm outer, 26 mm channel (5 mm/side clearance around the rail, >2 mm
  contact offset each side), mass 0.05 kg, μ 0.35 on a μ 0.20 rail, explicit MassAPI
  mass so the CoM stays on the channel axis. Absent pieces park in a ground depot.

**Geometry honesty** (asserted in `__post_init__`): worst-case train + gaps +
standoff fits the free span; the tip overhangs the tray inner region by 30 mm with
fall room; the minimum inter-piece gap (40 mm) admits a Franka fingertip; max escape
tilt while any channel length overlaps the rail is ~13°, so captivity holds.

**Randomization** (readback-verified in smoke): fixture xy jitter ±3 cm + yaw
90°±12°; cooked count ∈{1..3}, raw count ∈{1..2}; tip standoff ∈[5,9] cm and every
inter-piece gap ∈[4.0,5.8] cm resampled each episode — the boundary gap the solver
must find moves every episode.

## Rubric

Latched in `post_step` (transient achievements keep credit; teleporting a served
piece back out does not reduce the score — smoke check 12):

- 0.15 × advance — running max of cooked +x conveyance while threaded on the rail
  (ramp 8 cm; ~0 for idling; off-rail displacement earns nothing)
- 0.35 × off-tip — fraction of present cooked pieces ever past the tip
- 0.25 × in-tray — fraction of present cooked pieces ever settled inside the tray
- cap 0.75 without success; **1.0 iff `success()`**: every present cooked piece
  settled INSIDE the tray ∧ every present raw piece still THREADED and still ∧ no
  absent piece in the tray ∧ all states finite.

## Teleport solution (solve.py)

**Zero teleports.** Every judged body starts threaded; the whole task is the
load-bearing interaction, executed through contact dynamics:

- **P0** settle 60 steps; layout readback (counts, fixture pose, per-piece x —
  seed-dependence provable from stdout); assert all threaded; score ≤ 0.02.
- **P1** metered axial push at the boundary: the innermost present cooked piece
  (index nc−1) — exactly where a fingertip reaches into the boundary gap — gets a
  velocity-limited force (1.2 N, v_des = 0.12 m/s, stall escalation to 6 N) along
  the fixture's +x. It conveys the cooked group through ring-on-rail sliding and
  ring-on-ring pushing; each piece tips over the free tip and falls. The raw pieces
  behind the push point are never touched. Force cut when the boundary CoM passes
  the tip. `SIM_GEN_SCORE` at first-off and at all-past-tip.
- **P2** hands-off: fall, land, settle in the tray (≤1200 steps, 30-step quiet
  window); assert cooked in tray, raw still threaded.
- **P3** persistence: ≥3.3 simulated s hands-off, success still true, score
  non-decreasing → `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka + OSC)

Base at the world origin; the fixture's nominal pose (0.42, 0.02) with yaw 90° lays
the rail roughly along world +y, so every interaction point — the boundary gap, the
span, the tray — sits at radius 0.35–0.55 m, inside the Franka's comfortable
workspace, with the rail axis at z=0.16 reachable from above.

Per-object contact strategy: the only manipulated objects are the cooked rings, and
the strategy is a *fingertip push into the boundary gap* — close the gripper, lower
the fingertip (~18 mm wide < 40 mm minimum gap) into the gap between the innermost
brown and outermost pink piece from above, and servo along the rail axis at a few
cm/s pressing on the brown face. The solve's velocity-limited CoM force is the
wrench this fingertip contact applies (bounded force, bounded speed); no grasp, no
carry, no wrench a fingertip could not produce. The raw pieces are behind the finger
and physically cannot be pushed by it.

## Execution-order declaration

No rubric-declared order. The FIFO serving order is physically inherent (pieces
cannot pass each other on the rail), and the cooked-before-raw structure is enforced
by geometry (cooked group outboard) — the rubric only judges outcomes: cooked in
tray, raw threaded.

## Checks (smoke.py — rejection battery, 15 checks)

1. settle: finite, every present piece threaded, all still
2. settle: score ~0, no success
3. randomization: cooked/raw counts vary across seeded resets (readback)
4. randomization: yaw spread > 4° and standoff jitter > 6 mm (readback)
5. null policy: 240 idle steps → score ~0, no success
6. captivity probe: threaded piece hauled up at ~6× weight then sideways stays
   threaded (no lift-off exit — the seed's plan is impossible)
7. seed strategy: cooked pieces "set beside the grill" on the ground → score ~0
8. overshoot: cooked group + a raw piece in the tray (butt-end push outcome) →
   raw-on-rail clause alone rejects
9. near-miss: one cooked piece on the ground beside the tray → rejected
10. near-miss: cooked group conveyed short of the tip, still on rail → ≤ 0.20
11. partial: one of two cooked served → partial credit in [0.10, 0.75], no success
12. latched credit: served piece teleported back out → score unchanged
13. wrong object: an ABSENT piece constructed into the tray, present all correct →
    absent-piece clause alone rejects
14. rejection audit: success() never true anywhere in the battery
15. final: all states finite

Verdict lines: `SIM_GEN_SOLVE: SUCCESS` (seeds 0 and 1), `SIM_GEN_SMOKE: ALL PASS 15/15`.
