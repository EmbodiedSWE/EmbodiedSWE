# libero_kitchen_scene1_open_top_drawer_i325 — Push the Cargo Up the Ratchet Ramp (scene `ratchet_ramp`)

Drive two BLUE cargo balls UPHILL: from open ground, in through the ground-level
mouth of an enclosed 12° one-lane ramp channel, THROUGH two passive one-way flap
gates (pet-door ratchet pawls that the ball itself shoves open and that fall shut
behind it), over the crest, and down into a roofed CATCH BASIN sunk one ball-radius
below the crest, where a retaining step keeps them. A ball released mid-climb rolls
back only to the last flap it passed and is HELD there — progress is checkpointed by
the ratchet, never lost to the bottom. The RED decoy ball must be left OUT of the
basin; the basin holds at most two balls and nothing can be pushed back out, so a
delivered decoy is a permanent dead end. The flaps are judged nowhere: waving them
open by hand earns nothing.

Env: `simgen.ratchet_ramp` (scene `ratchet_ramp`, robot `"null"`).
Files: `scene.py`, `solve.py`, `smoke.py`, `__init__.py`, `TASK.md`. Procedural
geometry only — no external assets.

## Seed provenance

Seed: `libero_90/libero_kitchen_scene1_open_top_drawer`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene1_open_top_drawer.py`):
"open the top drawer of the wooden cabinet" — grasp the drawer handle and pull one
prismatic joint to a threshold; the checker is `JointPosChecker` on that joint. The
seed's whole task IS one guided articulation act.

## Strategic difference vs the seed

- **The articulated parts are demoted from goal to passive checkpoint hardware.**
  The seed judges a joint readout; here the two hinged flaps have joint readouts that
  are judged NOWHERE. Smoke check 4 executes the seed's whole strategy for real —
  torques a flap open past 45°, lets it fall shut — and asserts score ~0, no success.
- **Nobody's hand operates the mechanism.** In the seed the gripper drives the joint.
  Here the flaps are operated BY THE CARGO in passing: the ball shoves each flap open
  and gravity closes it behind the ball (solve.py never touches a flap). The
  manipulation is displaced from the mechanism to the transported object.
- **The work is sustained transport against gravity, not one threshold crossing.**
  The seed's pull is quasi-horizontal and ends the moment a coordinate is crossed.
  Here the judged outcome is containment at elevation (~10 cm climb) with the
  mechanism RE-CLOSED — the end state has the articulation back where it started.
- **Different checker structure.** Seed: one joint position threshold. Here: basin
  containment for two identified balls + decoy exclusion + flaps-hanging-closed +
  settled + finite, with latched per-checkpoint partial credit.

## Strategic difference vs every corpus task read this session

No corpus task makes UPHILL transport through PASSIVE ONE-WAY GATES the judged
outcome. Nearest relatives first:

- `libero_..._open_top_drawer_i14` (chute_switch — same seed): gravity does the
  transport DOWNHILL and the solver's real act is setting a binary switch; the
  mechanism is actively reconfigured, the cargo just falls. Here the mechanism is
  never touched, gravity is the ADVERSARY, and the solver's real act is the climb
  itself.
- `sweep_to_dustpan_i156` (letterbox_bin): the corpus's other one-way flap — a single
  flap over an elevated slot, each cube pressed horizontally through it once, done.
  No ramp, no climb, and one gate crossed once; here two gates in series checkpoint a
  continuous climb, and their ratchet function (holding a ball that rolls BACK) is
  load-bearing and separately smoke-verified.
- `open_window_i283` (sash_vent): a one-way pawl props the judged sash — the ratchet
  serves an articulated GOAL part. Here the ratchet serves the CARGO and the
  articulated parts must end closed.
- `roll_ball_i81` (boulder arch): has an uphill boulder push, but as a sacrificial
  unlock with a deliberately RESTORATIVE incline ("an abandoned boulder always slides
  back and re-locks — not a ratchet", its own docstring). Here the incline is
  ratcheted: partial progress is banked at each flap, and the delivered state is the
  goal, not a key to it.
- The gravity-chute family — `...i214` (sluice_hopper_catch), `...i166`
  (sauce_chute), `put_rubbish_in_bin_i147` (rubbish_chute), `scene_b_i305`
  (marble_router), `place_shape_in_shape_sorter_i180`: all release cargo at the TOP
  and let gravity deliver it down through gates/routing. Exactly inverted here: entry
  is only at the BOTTOM (the roof seals everything else), and every centimetre is
  won against gravity.
- `obstacle_i17` (skittle_gallery): ballistic precision through a tunnel — one aimed
  launch. Here precision is unnecessary (one straight lane) and the budgeted
  quantity is sustained work against the slope, with checkpoints making impulses
  safe.
- `pour_water_i7` (ramp_chock): statics — hold balls STILL on a ramp. Here balls
  must CLIMB the ramp.
- `pick_single_egad_i3` (tunnel_shuttle): pull a pin, slide a shuttle OUT of a
  tunnel — removal of a blocking gate, level travel. Here nothing blocking is
  removed; the gates yield one way and re-arm themselves.
- `hit_ball_with_queue_i77` / `close_drawer_i58` / `put_books_on_bookshelf_i198`:
  slopes appear as delivery/settling surfaces, not as an adversary to be climbed
  through one-way hardware.
- The remaining corpus families (balances/weighbridges, bayonet twists, stacking,
  carousels, presses, drawers/cabinets, pick-and-place variants) share neither the
  uphill-transport objective nor the passive-ratchet mechanism.

## The teleport solution (solve.py) — transport only, physics does the work

- P0: settle 120 steps; layout readback (bay permutation, ball xy, flap angles,
  masses); baseline asserts (flaps closed, score ~0, not success).
- TRANSPORT (teleport): each cargo ball is written ONCE from its ground bay to a
  ground-level staging spot just downhill of the mouth (x = −0.070), zero velocity.
  The write satisfies no rubric clause: the whole climb is still ahead.
- CLIMB (applied force + contact): a capped horizontal force (≤ 2.5 N — a fingertip
  on a 150 g ball) drives the ball through the mouth and up the ramp under a
  velocity servo (0.22 m/s) with gravity feedforward and a small lateral centering
  term, converted into the rolling ball's body frame every step. Both flaps are
  opened BY THE BALL in passing; each falls shut behind it (asserted after every
  delivery).
- DELIVERY (gravity + contact, hands-off): the force is cut at the crest lip; the
  ball rolls over, drops one ball-radius into the basin, and the step + end wall
  settle it inside.
- P1 first cargo (score latches 0.35), P2 second cargo (success), P3 ≥ 3.3 simulated
  seconds hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`. `SIM_GEN_SCORE` is
  printed at each phase boundary and is non-decreasing (checkpoint credit latches).
- The decoy is never touched. Verified on the forge on seeds 0 and 1 (different bay
  permutations; scores 0 → 0.35 → 1.0; rc=0 both).

## Embodiment argument (single Franka, 80 mm parallel jaw)

- **Objects:** 60 mm, 150 g balls — a comfortable pinch (60 < 80 mm) for staging,
  and a trivial fingertip-push load. No grasp is ever required by the plan; every
  load-bearing interaction is a non-prehensile push at ground level.
- **The climb without entering the channel:** the first 135 mm of the channel are
  open-topped and the rest is roofed, but the fingertip never needs to follow the
  ball in. A 2.5 N stroke along the open runway (net ~2.2 N over ~0.135 m after the
  slope tax) launches the ball at ~2.0 m/s = 0.30 J, nearly 3× the 0.10 J needed to
  crest plus flap work — one firm carrom-style stroke from the mouth delivers a
  ball. And the ratchet makes conservative strokes SAFE: an undershot ball parks
  against the last flap it passed instead of returning; a follow-up ball struck up
  the one-lane channel transfers its momentum head-on to a parked one. The
  applied-force servo in solve.py is the low-variance proxy for exactly this
  fingertip stroke (same contact, same force cap).
- **One plausible base pose:** bays (x = −0.17, y = ±0.10) and the mouth runway
  (x ∈ [−0.02, 0.115]) all lie in a ~0.3 m disc at ground level; a base at
  ~(−0.30, −0.35), 0.4–0.6 m from every working point, reaches all of it well inside
  the 0.855 m envelope — no reach over the walls or into the roofed section, which
  the plan never requires.
- **Forces:** ≤ 2.5 N pushes, 150 g payloads. Nothing bimanual, no press-fits, no
  tool use.

## Execution order

Declared: cargo ball order is FREE (either blue ball first; the two deliveries are
independent and independently checkpointed), and the decoy never needs to be
touched. Within any single ball's climb, flap 1 before flap 2 before the crest is
physically forced (one sequential lane); the rubric imposes no ordering beyond that
physics (latches fire whenever a checkpoint is genuinely reached).

## Rubric

`success()` (live, all): both cargo balls inside the basin ∧ decoy NOT in the basin
∧ both flaps hanging closed at their hinges ∧ everything settled ∧ finite.
`score()`: per cargo ball, latched: 0.10 once past flap 1 + 0.15 once past flap 2 +
0.10 once settled in the basin, capped at 0.70; exactly 1.0 iff success. Null policy
~0; swinging a flap ~0; the decoy latches nothing anywhere.

## Checks (smoke.py — rejection-only battery; success() must never fire)

1. Settle/no-NaN: flaps hang closed at their hinges, balls in their bays, score ~0.
2. Randomization: bay permutation varies (≥3 distinct in 8 resets) and ball xy
   jitters (readback).
3. Null policy: 240 idle steps → score ~0, no success.
4. SEED strategy: flap 1 torqued open past 45° for real (flaps_closed goes False at
   the peak — non-vacuous), falls shut on release — latches nothing, score ~0.
5. Ratchet retention: a ball placed between the flaps rolls back downhill and is
   HELD by flap 1 (the pawl works) — only the p1 latch fires (0.10), no success.
6. Top-entry holds: a ball dropped through the open mouth stretch lands upstream of
   flap 1 and rolls back out — no latch, no credit.
7. Roof holds: a ball dropped from above the basin lands ON the roof, never
   inside — no bin latch.
8. Decoy poisons: the decoy settled in the basin (with one cargo) — the decoy
   latches nothing (score counts only the cargo), success refused.
9. Missing delivery: one cargo in the basin, the other in its bay → partial 0.35
   only, no success.
10. Latched credit: the delivered cargo removed to open ground → the 0.35 latch
    survives, success does not.
11. Extraction holds: a sustained full-strength (2.5 N cap, quasi-static) downhill
    press pins a basin ball against the retaining step (it travels to the step —
    non-vacuous) but cannot lift it out.
12. Rejection audit: success() observed False at every step of the battery.
13. Final no-NaN.
14. Video: >10 frames recorded; frames.npz (960×600 RGB) written to the CWD.
