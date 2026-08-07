# libero_kitchen_scene1_open_top_drawer_i14 — Route the Balls Through the Switch Chute (scene `chute_switch`)

Sort three cargo balls into color-matched covered bins by ROUTING, not by placing: an
inclined twin-channel roofed chute ends above a GREEN and a RED bin, and a sliding
SWITCH GATE — a prismatic bar riding through slots in the chute walls between two end
stops — always blocks exactly one channel. Balls can only enter the system at the open
inlet at the TOP of the chute; gravity rolls them down the open channel, off the
outlet, through the bin mouth. Both colors' channels share the ONE gate, so the solver
must set the switch for one color, feed those balls, slide the switch to the other
stop, and feed the rest — at least one mid-task reconfiguration is structurally
forced. A wrong routing is irreversible (bins are roofed; a sill blocks the mouth
outward), and a BLUE decoy ball belongs in neither bin.

Env: `simgen.chute_switch` (scene `chute_switch`, robot `"null"`).
Files: `scene.py`, `solve.py`, `smoke.py`, `__init__.py`, `TASK.md`. Procedural
geometry only — no external assets.

## Seed provenance

Seed: `libero_90/libero_kitchen_scene1_open_top_drawer`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene1_open_top_drawer.py`):
"open the top drawer of the wooden cabinet" — grasp the drawer handle and pull one
prismatic joint to a threshold; the checker is `JointPosChecker` on that joint. The
seed's whole task IS the prismatic slide.

## Strategic difference vs the seed

The seed's skill is kept but stripped of all judged value; the judgment moves to what
the slide is FOR:

- **The prismatic slide is present and worth exactly zero.** The switch gate is the
  drawer's motion — a bar sliding in a straight travel between stops, pushed by its
  exposed tail. But `score()` awards it nothing: smoke check 5 executes the seed's
  entire strategy for real (force-flip to the far stop and back, parks verified) and
  asserts score ~0, no success.
- **The slide is re-purposed from goal to ROUTING DECISION.** In the seed, moving the
  joint terminates the task. Here the gate's position is a persistent BINARY STATE
  that determines where *other* objects end up; it must be READ (which side is
  parked?), matched against the episode's bin arrangement, and CHANGED mid-task.
- **The judged outcome is transport the solver never performs directly.** Success is
  "cargo inside the correct covered bins" — reachable only by releasing balls at the
  inlet and letting gravity + the switch's parked position decide the path. The seed
  has no objects in transit at all.
- **Different checker structure.** Seed: one joint position threshold. Here:
  body-frame containment in two bins for four balls with color identity, a
  bins-seated gate, a settled gate, latched per-delivery credit, and a routed-past-
  the-switch latch — none reducible to a joint readout.

## Strategic difference vs every corpus task read this session

No corpus task contains a state-dependent ROUTER: a shared binary switch whose parked
state redirects gravity transport of multiple objects and must be reconfigured
between deliveries. Closest neighbors first:

- `pick_single_egad_i3` (tunnel_shuttle): nearest relative — remove a gate-pin so a
  shuttle can slide through a tunnel. But its gate is a REMOVE-ONCE obstacle on a
  single fixed path for a single object; nothing is ever routed BETWEEN destinations,
  and the pin never comes back. Here the gate is a two-position selector that must be
  set, used, and RE-set (both directions exercised), with color→destination matching
  and irreversible wrong outcomes.
- `libero_..._i9` (carousel_airlock): the mechanism CARRIES the cargo (rotor sweeps
  the ball around); driven kinematic rotation, one object, one path. Here the
  mechanism never touches the cargo — it only selects which of two static paths
  exists, and the cargo moves by free rolling.
- `libero_..._i11` (matchbox_drawer): prismatic slide as the GOAL enclosure
  (open-load-close around the object). Here the slide is judged nowhere and encloses
  nothing; it selects between two branches.
- `plug_charger_..._i21` (keyhole unplug/stow): prismatic keyhole slide frees the
  judged object itself for an ordered extraction. No branching, no routing, no decoy.
- `setup_checkers_i2` (checker_silo): ordered loading down ONE fixed funnel —
  sequence identity in a single path. Here there are two paths, a switch choosing
  between them, and no imposed order.
- `obstacle_i17` (skittle_gallery): ballistic aiming through a floor tunnel —
  precision dynamics. Here transport is quasi-static release; all precision is in the
  SWITCH STATE, not the throw.
- `pour_water_i7` (ramp_chock): statics — hold balls STILL on a ramp. Here the ramp
  is the transport medium and balls must GO.
- `living_room_..._i18` (wedge_hopper): tip a hopper by jacking to drain cargo out.
  No routing choice, no switch, single destination.
- `pick_and_lift_i16` (ballast_lever_lift) / `pull_cube_i20` (beam_scale) /
  `coke_task_i15` (two-pan balance): the torque family — accumulate mass to tip or
  level a lever. Nothing tips here; the mechanism is a slider and outcomes are
  containment, not equilibrium.
- `peg_insertion_side_i1` (ramrod eject) / `i2` (hasp pin): push a rod/pin through a
  bore — insertion. The gate never inserts into anything; it blocks a channel.
- `close_microwave_i5` / `libero_..._i3` (bayonet twist-locks): align-insert-twist.
  No lugs, no twist.
- `close_microwave_i4` (drop_gate_oven): remove a prop, gravity closes a gate — the
  gate motion IS the goal. Here gravity moves the CARGO and the gate is hand-set.
- `close_grill_i8` (fire_crib): build a stable stack. Nothing is stacked.
- `open_oven_i6` (dice_tumble): orientation outcome. Orientation is irrelevant here.
- `approach_grasp_spoon_i12` (die_tip_pad): non-prehensile tipping of the judged
  object. The judged balls are never pushed; the only pushed body scores zero.
- `pick_single_egad_i4` (tag_hangers) / `track_ceramic_teapot_i22` (mug rack, WIP):
  hang objects on matched pegs/hooks — suspension placement. Matching exists here
  too, but it is executed by CONFIGURING A MACHINE, not by placing onto the target.
- `libero_..._i19` (moat_causeway): build a bridge then push across — structure
  building. Here the structure exists; only its state is manipulated.
- `libero_..._i2` (bowl_decant): pour and park. No pouring vessel; discrete cargo.
- `libero_..._i13` (burner_snuff): two placements with an occupancy precondition.
  Ordering there is spatial (clear first); here ordering is INFORMATIONAL (switch
  state must match the color being fed).
- `empty_dishwasher_i23`: empty directory — no task to collide with.

## The teleport solution (solve.py) — transport only, physics does the work

- P0: settle 120 steps, layout readback (router pose/yaw, bins_swapped, gate side,
  slot permutation), baseline asserts (score ~0, not success).
- SWITCHING (real actuation, never posed): a PD force along the bar's own axis
  (≤ 5 N, what a fingertip on the exposed tail applies), converted into the gate's
  body frame each step; the end-stop post arrests it and static friction holds it.
  Ramming at constant force wedges the tip into the post — the braking law is part of
  the contract that the switch is operated, not written.
- TRANSPORT (teleport): each cargo ball is written once from its ring slot to a
  free-space hover 50 mm above the OPEN inlet of the chosen channel (live router
  body frame), zero velocity. The write satisfies no rubric clause: the ball is
  airborne at the top of the chute, two channel-lengths from any bin.
- ROUTING (gravity + contact, hands-off): the ball falls onto the inlet floor and
  rolls ~0.5 m down the channel, under the guide rails, past the open switch
  station, off the outlet edge, through the bin mouth (sill-to-lintel window sized
  around the arrival height), onto the bin floor. Every centimetre is decided by the
  chute geometry and the gate's parked side.
- P1–P2 greens, P3 MANDATORY flip then red (`assert flips >= 1`), P4 ≥ 3.3 simulated
  seconds hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`. `SIM_GEN_SCORE` is
  printed at each phase boundary and is non-decreasing (delivery credit latches).
- Verified on the forge on seeds 0 and 1 (different yaws, bin arrangements, slot
  permutations; seed 0 flips once at P3, seed 1 flips at P1 AND back at P3 — both
  travel directions exercised; rc=0 both).

## Embodiment argument (single Franka, 80 mm parallel jaw)

- **Balls (the only objects the plan carries):** 40 mm spheres, 60 g — a trivial
  pinch. Each is released above the open inlet run (drop height ~0.24 m world) — a
  pure hover-and-drop, exactly what the teleport emulates.
- **The switch:** the 144 mm bar protrudes ~36 mm beyond the outer chute wall at
  either park, at ~0.15 m height, below the rail line — an exposed tail for a
  fingertip push of ≤ 5 N along a straight 0.14 m travel. This is the seed's drawer
  gesture, unchanged.
- **One plausible base pose per episode:** all four ball slots are constrained to the
  UPHILL half-ring (angles 115–265 ±12° from the downhill axis, radii 0.50–0.62 m),
  so a base ~0.62 m from the chute centre on the uphill bisector reaches every ball
  (worst case ~0.83 m, nearest ~0.14 m), the inlet drop point (~0.36 m), and both
  gate-tail positions (~0.65 m) — no reach over or through the chute; the bins face
  away downhill and are never approached.
- **Forces:** ≤ 5 N lateral push, 60 g payloads dropped ~5 cm. Nothing bimanual, no
  press-fits, no regrasp-in-flight.

## Execution order

Declared: color-block order is FREE (greens-then-red or red-then-greens both
succeed; within a color, ball order and even interleaving with gate flips are free),
but each feed requires the gate parked OPPOSITE the target channel first, and because
the two colors' channels share the one gate, at least ONE mid-task flip is forced.
The rubric imposes no order beyond that physical constraint (latches fire whenever a
correct delivery settles).

## Rubric

`success()` (live, all): both greens inside the green bin ∧ red inside the red bin ∧
blue in NEITHER bin ∧ both bins still seated at their outlets ∧ everything settled ∧
finite. `score()`: 0.10 once any cargo ball is ever routed past the switch + 0.20
per correct delivery (latched: green×2, red×1), capped at 0.70; exactly 1.0 iff
success. Null policy ~0; sliding the gate alone ~0; wrong-bin deliveries latch
nothing; the blue ball latches nothing anywhere.

## Checks (smoke.py — rejection-only battery; success() must never fire)

1. Settle/no-NaN: gate parked at a stop, bins mated at the outlets, score ~0.
2. Randomization A: router yaw spans > 90° and xy jitters across resets.
3. Randomization B: bin arrangement, gate initial side, slot permutation all vary.
4. Null policy: 240 idle steps → score ~0, no success.
5. SEED strategy: the seed's whole skill (slide the prismatic bar) executed for
   real, twice — parks verified, score ~0, no success.
6. Roof holds: a green ball dropped from above the GREEN bin lands on the roof,
   never inside — no latch.
7. Mouth gap holds: a ball laid at the chute/bin seam cannot enter (sill + sub-ball
   gap).
8. Blocked channel: a ball fed into the gated channel jams UPSTREAM of the switch —
   never routed, never binned.
9. Wrong routing (constructed, settled): greens in RED, red in GREEN → no latch, no
   success, score ~0.
10. Blue decoy: correct greens+red PLUS blue inside a bin → success refused, score
    capped at 0.70.
11. Bin displaced: full correct arrangement but the green bin dragged off its
    outlet → bins_seated refuses success.
12. Missing delivery: one green still on the ground → partial score only (~0.40).
13. Latched credit: red delivered then removed → the 0.20 latch survives, success
    does not.
14. Rejection audit: success() observed False at every step of the battery.
15. Final no-NaN; frames.npz (960×600 RGB) written to the CWD.
