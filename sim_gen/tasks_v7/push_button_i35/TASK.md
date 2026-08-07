# push_button_i35 — Turn the Staircase Under the Button (scene `cam_press`)

Fully depress a 70 mm-travel push button WITHOUT ever pressing it: the button is a
captive cam follower whose ball foot rests on the TOP STEP of a spiral staircase cut
into a heavy CAM WHEEL. Pressing the cap just loads the wheel against its pedestal —
the follower's load path goes staircase → wheel → pedestal → ground, so the seed's
gesture is physically dead. The only way down is to ROTATE the wheel by its blue rim
pegs in the staircase's descending direction: each 34° of rotation carries the next
step under the foot and the button drops one 10 mm riser, eight steps, ~240° of
rotation in total, until an underside pin arrests on the frame's stop post with the
foot on the lowest step. The machine is self-locking (release the wheel anywhere and
the current riser holds the button), one-way (a tall headwall plus the pin on the
OTHER stop post jam reverse rotation within a few degrees), and fully captive (a cap
on the centre post stops the wheel lifting; ball foot and head cap are both wider
than the sleeve bore). Staircase CHIRALITY — which mirror wheel is mounted — flips
the required rotation direction 50/50 per episode and must be read from the scene.

Env: `simgen.cam_press` (scene `cam_press`, robot `"null"`).
Files: `scene.py`, `solve.py`, `smoke.py`, `__init__.py`, `TASK.md`. Procedural
geometry only — no external assets.

## Seed provenance

Seed: `rlbench/push_button`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/push_button.py`): "push the
button" — move the end-effector onto a small button and press it down a few
millimetres; the checker is essentially a downward displacement of the button head.
The seed's whole task IS the press.

## Strategic difference vs the seed

The seed's skill is kept visible but stripped of all value; the judgment moves to
the MACHINE that stands between the hand and the outcome:

- **The press is present and worth exactly zero.** The red cap sticks up in plain
  view and can be pushed for real — smoke check 5 presses it with 30 N (far beyond
  any fingertip) for two seconds and asserts depth ~0, score ~0, no success. The
  follower is grounded through the staircase; pressing is a no-op by STATICS, not by
  a rubric veto.
- **Displacement becomes transmission.** In the seed the judged displacement is
  produced directly by the actor. Here every millimetre of the button's 70 mm travel
  is produced by a DIFFERENT body's rotation carrying steps out from under the foot;
  the solver never touches the judged object at all (the nominal plan applies torque
  only to the wheel).
- **One binary press becomes a long, quantized, one-way descent.** The seed is a
  threshold on a few mm of travel. Here the travel is 7 discrete riser drops spread
  over ~240° of rotation, physically latched (the foot cannot climb a riser), with a
  read-then-act chirality decision (mirrored staircases) and a geometric arrest (pin
  on stop post) defining "fully down".
- **Different checker structure.** Seed: one displacement threshold. Here: guarded
  depth latches (wheel seated ∧ follower in sleeve ∧ slow), a wrap-aware wheel park
  window, a live conjunction (full depth ∧ wheel at low stop ∧ seated ∧ in sleeve ∧
  settled ∧ finite) — none reducible to "did the button go down".

## Strategic difference vs every corpus task read this session

No corpus task makes rotary-to-linear TRANSMISSION the whole task: a rotating cam
staircase that converts ~240° of one-way, self-locking, chirality-randomized wheel
rotation into a quantized linear descent of a captive follower the solver never
touches. Closest neighbors first:

- `libero_..._i9` (carousel_airlock): nearest relative — rotate a mechanism to move
  something else. But its rotor CARRIES loose cargo through a doorway (transport in
  a pocket, direction free, stop where you like). Here nothing rides the wheel: the
  staircase is a CAM SURFACE whose rotation is geared 34°-per-10 mm to a captive
  follower's axis, direction is forced by chirality, progress is quantized by risers
  and terminated by a hard pin/post arrest.
- `close_microwave_i5` / `libero_..._i3` (bayonet twist-locks): rotate-until-stop
  exists there too, but the rotation acts on the judged object itself (align, insert,
  twist to lock ~60–90°). Here rotation is ~240° on a body that is NOT judged, whose
  only purpose is to transmit motion to the judged follower; there is no insertion
  and no lug engagement.
- `open_oven_i7` / `oven_dials` family: turn a dial to a target angle — the angle IS
  the goal. Here the wheel angle is worth nothing by itself (smoke: wheel arrested
  with the follower absent scores 0); only the follower's transmitted depth counts.
- `pick_and_lift_i16` (ballast_lever_lift) / `pull_cube_i20` (beam_scale) /
  `coke_task_i15` (two-pan balance): the torque family — pile mass on a lever until
  it tips or levels. Continuous equilibrium outcomes, gravity does the actuation.
  Here the actuation is a driven rotation against friction with NO equilibrium to
  find — the outcome is a discrete arrested configuration.
- `libero_kitchen_..._i14` (chute_switch): a machine redirects other bodies, but its
  slide is a binary ROUTER for free-rolling cargo. Here the wheel is a TRANSMISSION
  in permanent contact with the follower — no cargo, no branching, no routing
  choice; the information read is chirality (which way), not state matching.
- `pick_single_egad_i3` (tunnel_shuttle): remove-once gate pin, then slide a shuttle
  along a fixed path. Nothing is transmitted; the freed object is pushed directly.
- `libero_..._i11` (matchbox_drawer) / `plug_charger_..._i21` (keyhole unplug):
  prismatic slides on/around the judged object. Here the prismatic motion (the
  button) is the OUTPUT of the machine, never actuated directly.
- `peg_insertion_side_i1` (ramrod eject) / `i2` (hasp pin): push a rod through a
  bore. The follower's stem rides a bore but is never pushed — it descends because
  its support rotates away beneath it.
- `close_microwave_i4` (drop_gate_oven): remove a prop and gravity closes a gate —
  one support removal, one fall. Here support is removed SEVEN times, 34° apart, by
  the same continuous rotation, and the "fall" is 10 mm per riser with self-locking
  holds in between.
- `pour_water_i7` (ramp_chock): statics — keep balls still. Here statics is the
  ADVERSARY the machine defeats step by step.
- `obstacle_i17` (skittle_gallery): ballistic aiming. Nothing ballistic here;
  quasi-static torque throughout.
- `setup_checkers_i2` (checker_silo): ordered loading down a funnel. No loading, no
  ordering beyond the physically forced rotation direction.
- `living_room_..._i18` (wedge_hopper): jack a hopper to tip cargo out. No cargo, no
  tipping — the wheel stays seated and level the whole time (it's a judged clause).
- `close_grill_i8` (fire_crib): build a stack. Nothing is stacked.
- `open_oven_i6` (dice_tumble): orientation outcome of a tumbled die. Orientation of
  the follower is irrelevant; the wheel's orientation is constrained, not free.
- `approach_grasp_spoon_i12` (die_tip_pad): tip the judged object non-prehensilely.
  The judged object here is never touched.
- `pick_single_egad_i4` (tag_hangers) / `track_ceramic_teapot_i22` (mug rack): hang
  matched objects on pegs. The pegs here are HANDLES on the machine, not targets.
- `libero_..._i19` (moat_causeway): build a bridge then push across. The machine
  pre-exists; only its configuration is driven.
- `libero_..._i2` (bowl_decant) / `libero_..._i13` (burner_snuff): pour/park and
  placement-with-precondition. No pouring, no placement.
- `empty_dishwasher_i23`: empty directory — no task to collide with.

## The teleport solution (solve.py) — no teleport at all

Nothing needs transporting: the entire solution is one real actuation of the
machine's only degree of freedom.

- P0: settle 120 steps; layout readback (tower pose/yaw, CHIRALITY, jittered park
  angle — all read live, never assumed); baseline asserts (seated, in sleeve, depth
  ~0, score ~0, not success).
- ROTATE (applied torque + contact): a pure z-axis torque on the wheel, capped at
  1.2 N·m — what a fingertip pushing ~8 N tangentially on a rim peg at r = 0.148 m
  applies — under a velocity-limited PD law (≤1.2 rad/s: spinning the wheel up and
  slamming the staircase would be nothing like a hand on a peg). Unwrapped-angle
  tracking (the readout crosses ±180° mid-journey; a wrapped PD would push the
  blocked direction). The BUTTON IS NEVER TOUCHED.
- SELF-LOCKING CHECKPOINTS: at each rubric depth (25/45 mm and the arrest) the
  torque is RELEASED and the machine holds by itself — every `SIM_GEN_SCORE` print
  is taken hands-off.
- ARREST: the PD aims 3° PAST the pin/post arrest and presses gently into the post;
  `parked()` requires the stall to PERSIST 30 consecutive steps under torque, so a
  momentary brake from the last riser drop cannot fake it — geometry, not the
  controller, chooses the final park. Travel assert ≥ 225° (measured 236–244°).
- P4: ≥ 3.3 simulated seconds hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.
  `SIM_GEN_SCORE` is non-decreasing (latched depth credit; risers make regression
  physically impossible).
- Verified on the forge on seeds 0, 1, 2 — BOTH chiralities and both rotation
  directions demonstrated, different yaws and park angles; rc=0 on all three.

## Embodiment argument (single Franka, 80 mm parallel jaw)

- **The pegs (the only things the plan pushes):** four Ø18 mm × 50 mm blue posts on
  the wheel's deck at r = 0.148 m, spaced 90° — at any wheel angle at least one peg
  is on the near side at a graspable/pushable height (deck top 0.090 m, peg top
  0.140 m). The nominal actuation is a fingertip pushing a peg tangentially at ≤8 N
  (= the 1.2 N·m torque cap); the ~240° journey is 3–4 push-reposition strokes of
  ≤90° each, exactly like winding a large winch handle. No grasping is required
  (pushing suffices); no regrasp-in-flight.
- **Nothing else is manipulated:** the button (Ø45 mm ball, 50 mm cap) is never
  touched; the wheel is never lifted (captive under the cap).
- **One plausible base pose per episode:** the machine is one compact tower (base
  0.36 m square) with jitter ±5 cm; a base ~0.55–0.65 m from the tower centre
  reaches the full peg circle (max reach ~0.80 m) from one side; the wheel brings
  every peg past every azimuth during the journey, so even a fixed-side robot sees a
  reachable peg at least once per 90°.
- **Forces:** ≤8 N tangential fingertip pushes; 0.8 kg wheel driven at ≤1.2 rad/s.
  Nothing bimanual, no press-fits.

## Execution order

Declared: the rotation is ONE continuous monotone motion whose direction is
physically forced by the mounted chirality (readable from the staircase); where to
pause is FREE (the machine self-locks at every angle), which peg to push and when to
re-stroke is FREE. The rubric imposes no order beyond the physics: depth latches
fire whenever their guarded depths are reached, necessarily in 25→45→66 mm order
because the staircase is one-way.

## Rubric

`success()` (live, all): depth ≥ 66 mm ∧ wheel parked in the low-stop window
(wrap-aware, chirality-signed) ∧ wheel seated on the pedestal ∧ follower in the
sleeve ∧ settled ∧ finite. `score()`: 0.20/0.20/0.30 latched at 25/45/66 mm of
travel, each latch guarded by (wheel seated ∧ follower in sleeve ∧ slow ∧ finite),
capped at 0.70; exactly 1.0 iff success. Null policy ~0; pressing the button (the
seed's whole skill) ~0; reverse rotation ~0; wheel removed latches nothing at any
depth (a merely displaced wheel is physically recaptured by the centre post).

## Checks (smoke.py — rejection-only battery; success() must never fire)

1. Settle/no-NaN: wheel seated, follower in sleeve on step 0, depth ~0, score ~0.
2. Randomization A: tower yaw spans > 90° and xy jitters across resets (readback).
3. Randomization B: BOTH chiralities drawn and the park angle jitters (readback).
4. Null policy: 240 idle steps → depth ~0, score ~0, no success.
5. SEED strategy: 30 N pressed straight down on the cap for 2 s — depth ~0,
   score ~0, no success (the press path is statically dead).
6. Reverse rotation: full-strength torque the wrong way jams within a few degrees —
   no depth, no credit.
7. Pry the button: 25 N straight up — the ball foot jams under the sleeve, never
   extracted, re-seats with no credit.
8. Pry the wheel: 25 N straight up — the centre-post cap holds the hub inside the
   seat tolerance; disc_seated never breaks.
9. Off-mount cheat: wheel written to the depot — the follower free-falls to a depth
   DEEPER than d3 (head catches on the sleeve top), and the disc_seated guard
   refuses every latch.
10. Sideways dismount: the wheel written 30 mm off the pedestal axis is RECAPTURED
    by the centre post (hub ring collides with it and slides back inside the seat
    tolerance, still at the park angle) — the wheel cannot be dismounted sideways,
    nothing gained.
11. Near miss: machine constructed at step 5 (50 mm, past d2) — partial credit is
    exactly 0.40; l3 and success refuse.
12. Latched credit: machine constructed back to the start park — the 0.40 latch
    survives, success does not.
13. Rejection audit: success() observed False at every step of the battery.
14. Final no-NaN.
15. Video: frames.npz (960×600 RGB) written to the CWD.

Forge: `SIM_GEN_SMOKE: ALL PASS 15/15` (rc=0); solve `SIM_GEN_SOLVE: SUCCESS` on
seeds 0, 1 (chirality −1) and 2 (chirality +1), rc=0.
