# pick_and_lift_i16 — Ballast the Lever (scene `ballast_lever_lift`)

Raise the caged RED cube by COUNTERWEIGHT: the cube rides in a roofed cage on one end
of a seesaw beam and can never be grasped, so the only way to lift it is to fetch
heavy BLUE steel blocks from the floor and drop at least three of them into the open
YELLOW basket on the other end of the beam, until real ballast torque tips the lever
onto its raised stop and hoists the cage ~12 cm. A same-size WHITE foam block is a
mass decoy that contributes nothing.

Env: `simgen.ballast_lever_lift` (scene `ballast_lever_lift`, robot `"null"`).
Files: `scene.py`, `solve.py`, `smoke.py`, `__init__.py`, `TASK.md`. Procedural
geometry only — no external assets.

## Seed provenance

Seed: `rlbench/pick_and_lift`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/pick_and_lift.py`): "pick up the red
block and lift it up to the target" — grasp the red 5 cm cube among color-distractor
cubes and carry it up to a marked height. It is the purest prehensile plan there is:
the object is sized for the jaw, and the lift IS the grasp-and-raise.

## Strategic difference vs the seed

The judged outcome is exactly the seed's — the RED cube ends up HIGH — but the seed's
entire strategy is physically removed and replaced with mechanism operation:

- **The red cube cannot be grasped or lifted.** It sits inside a walled, roofed cage
  whose every opening (30 mm roof gaps, 30 mm side slots) is smaller than the 60 mm
  cube; asserted in `scene.py::__post_init__` and physically verified in smoke by a
  1.5x-weight upward pull that the cage holds.
- **The lift is indirect — weight transfer through a lever.** The solver never touches
  the judged object. It must move OTHER objects (steel blocks) to a place from which
  gravity does the lifting: ballast in the basket tips the beam about its axle.
- **The distractors differ by MASS, not color-of-target.** The seed's distractors are
  wrong-colored cubes to be ignored visually. Here the white foam block (25 g vs
  260 g) looks like the steel blocks but is mechanically useless: the torque window is
  tuned (and asserted) so 3 steel ALWAYS tip the beam and 2 steel + foam NEVER do.
- **Different checker structure.** The seed needs `object.z >= target_z`. This rubric
  needs torque-driven mechanism state: beam-frame containment in a MOVING basket,
  count-latched ballast credit, a raise predicate GATED on ballast (so pressing or
  pulling the beam by hand farms nothing), plus seated-on-cradle and settled gates.

## Strategic difference vs every corpus task read this session

No existing task uses a lever/counterweight, and none achieves its goal by loading
ballast so gravity lifts a different object:

- `peg_insertion_side_i2` (hasp_pin_link): ordered fastening — pin through aligned
  holes. Here: no insertion; torque balance on a free pivot.
- `peg_insertion_side_i1` (ramrod ball eject): push a rod through a bore. No rod, no
  bore here.
- `approach_grasp_spoon_i12` (die_tip_pad): non-prehensile edge-tipping of one big
  die. Here the moved objects ARE graspable; the judged object is untouchable, and
  the tipping body is a two-sided lever driven by transferred weight, not by pushing.
- `close_grill_i8` (fire_crib): stack a stable structure. Here nothing is stacked —
  blocks are dropped loose into a container; the outcome is a rotation.
- `close_microwave_i4` (drop_gate_oven): remove a prop, gravity drops a gate. Related
  in "gravity finishes the job", but opposite mechanism: there a support is removed;
  here weight is ADDED to overbalance a balance — and the gate motion is the goal,
  while here the rotation is only a means to hoist the cargo.
- `close_microwave_i5` (bayonet_canister) / `..._i3` (lid_bayonet): align-insert-twist
  locking. No lugs, no twist here.
- `..._i9` (carousel_airlock): rotate a carousel through an airlock sequence — driven
  rotation of a kinematic mechanism; here the rotating body is fully dynamic and
  rotates only because of mass loaded onto it.
- `..._i2` (bowl_decant): pour and park. No pouring; discrete rigid ballast.
- `..._i13` (burner_snuff): swap-and-cap with color identification. Color identifies
  the TARGET there; here color marks mass classes and the discrimination is dynamic.
- `..._i11` (matchbox_drawer): sliding drawer containment. No prismatic anything here.
- `open_oven_i6` (dice_tumble): tumble dice to a face. Orientation of the moved object
  is irrelevant here; only accumulated weight matters.
- `pick_single_egad_i3` (tunnel_shuttle): gate-pin + tunnel slide. No pin, no channel.
- `pick_single_egad_i4` (tag_hangers): hang tags on matching pegs. Placement-on-peg vs
  drop-into-moving-container.
- `pour_water_i7` (ramp_chock): chock balls on a ramp — statics of holding things
  STILL against gravity; here gravity is harnessed to MOVE the mechanism.
- `setup_checkers_i2` (checker_silo): ordered funnel loading — sequence identity in a
  fixed container. Here the container MOVES (it is the lever end), order is free, and
  the count matters only through real torque.
- `obstacle_i17` (skittle_gallery): knock over one pin by shooting a ball through a
  floor tunnel — projectile selectivity. Nothing is thrown here; quasi-static loading.
- `living_room_..._i18` (wedge_hopper): the nearest neighbor — it also tips a body to
  move an untouchable contained object. But there the tip is produced by JACKING (a
  wedge inserted under the hopper edge — a tool/insertion plan) and the goal object
  DRAINS OUT downward into a basin. Here the tip is produced by WEIGHT TRANSFER onto
  the opposite end of a free two-sided balance (no tool ever touches the beam), with
  a mass-decoy discrimination and a count threshold enforced by an asserted torque
  window, and the judged object goes UP while STAYING contained. Different plan
  (fetch-and-load repeated vs insert-once), different rubric machinery (count-latched
  ballast credit + ballast-gated raise vs drain detection).

## The teleport solution (solve.py) — transport only, physics does the work

- P0: settle, layout readback (machine pose + yaw, permuted block slots, d0).
- P1–P3: for steel blocks 0–2, teleport the block from the floor to a FREE-SPACE
  hover ~15 mm above the open basket top, computed in the live beam body frame (one
  y-lane each at -42/0/+42 mm across the 130 mm basket), orientation matched to the
  beam, zero velocity — then step. The block FALLS the last ~5 cm under gravity onto
  the tilted basket floor (friction holds it: tan 12.4 deg << mu 0.7). After the third
  drop the accumulated ballast torque — real weights at a real lever arm on real
  contact — swings the beam through ~25 deg onto its raised stop, hoisting the cage.
  4 s hands-off settle for the swing; scores are latched and non-decreasing.
- P4: >= 3.3 simulated seconds hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.

No load-bearing teleports: no block is ever written inside the basket, the beam is
never posed, pushed, or held, and the red cargo is never touched by anything but the
cage it rides in. Verified on the forge on seeds 0 and 1 (different machine poses,
yaws, and slot permutations; rc=0 both).

## Embodiment argument (single Franka, 80 mm parallel jaw)

- **Steel blocks (the only objects the plan touches):** 40 mm cubes, 260 g — a
  comfortable top-down or side pinch for an 80 mm jaw, well under payload. Each is
  carried and RELEASED above the basket's open top (60 x 130 mm throat, rim at
  ~0.13–0.25 m height depending on beam angle) — a pure hover-and-drop, exactly what
  the teleport emulates; a 40 mm block through a 60 mm throat leaves 10 mm per side.
- **One plausible base pose:** the cradle spawns near (0.10, 0.00) with the basket end
  at roughly (-0.16, 0.00) +- machine yaw swing, and the blocks scatter on an arc at
  x in [-0.68, -0.44]. A base at ~(-0.40, 0.00) puts both the whole block field
  (0.10–0.35 m reach) and the basket rim (0.25–0.45 m reach, top-down access from
  above the open basket) inside the Franka's sweet band with no reach through or over
  the machine; the cage end faces away.
- **Forces:** dropping a 260 g block from ~5 cm is impact the mechanism tolerates by
  design (verified in every forge run); no press-fits, no bimanual holds, no
  regrasp-in-flight.

## Execution order

Declared: NO required order. Any 3 of the 4 steel blocks, loaded in any order and any
basket lanes, succeed; the foam block may be moved anywhere harmlessly. The rubric
imposes order nowhere (k-latch counts settled blocks whenever they arrive).

## Rubric

`success()` (all, settled): beam raised (cargo-end pitch sin >= sin 10 deg, stop at
~13.4 deg), >= 3 steel blocks inside the basket (beam-frame containment), cargo still
inside its cage, beam still seated on its cradle. `score()`: 0.10 x latched best
block approach (normalized per-block by spawn distance) + 0.15 per steel block ever
settled in the basket (max 3) + 0.15 x latched raise fraction gated on >= 3 steel in
the basket at that instant; capped at 0.85; exactly 1.0 iff success. Null policy ~0.

## Checks (smoke.py — rejection-only battery, 15/15 on forge, frames.npz recorded)

1. Settle/no-NaN: beam seated cargo-end down on its stop, cargo caged, blocks flat.
2. Score ~0 at reset, no success.
3. Randomization readback: machine xy + yaw vary.
4. Randomization readback: slot permutation (steel_0 + foam positions), d0, cargo
   in-cage jitter vary.
5. Null policy: 240 idle steps -> score ~0, no success.
6. SEED strategy: 1.5x-weight grasp-pull on the cargo — cage HOLDS the cube, machine
   stays seated, beam raised by hand yet k=0 -> NOT success, score <= 0.02.
7. Nothing lasting: pull released -> beam falls back cargo-down, still ~0.
8. Foam decoy: 2 steel + foam verified IN the basket -> beam must NOT tip (mass, not
   looks), score <= 0.45.
9. Beside the basket: 3 steel on the ground under the basket end -> k=0, no credit.
10. On the beam but outside the basket: weight on the plate -> k=0, no tip, no credit.
11. Off-cradle tableau: full success geometry off the cradle -> beam_seated False ->
    NOT success.
12. Latched credit: loading then removing 2 blocks leaves the latched score unchanged.
13. Monotonicity: closer hover latches strictly more approach credit.
14. Rejection audit: success() never True anywhere in the battery.
15. Final no-NaN.
