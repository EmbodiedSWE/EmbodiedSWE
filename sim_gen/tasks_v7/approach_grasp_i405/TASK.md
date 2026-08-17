# flywheel_interlock (`approach_grasp_i405`)

Stop a RUNNING bladed flywheel, jog its single open sector over the sealed vault's only
mouth, post the red parcel through it, then RESTART the machine and leave it coasting.
NullRobot scene-level env: `simgen.flywheel_interlock`.

## Seed provenance

Derived from `pick_place/approach_grasp` (RoboVerse `roboverse_pack/tasks/pick_place/
approach_grasp.py`): a Franka approaches a STATIC red cube resting in a quiet scene,
closes its jaw around it, and lifts. The kept DNA: a single red cube as the payload, a
tabletop-scale reach-and-transport interaction, and "move the red cube to where it must
go" as the visible object transaction.

## Why it is strategically different

- **From the seed**: `approach_grasp` is the canonical static-acquisition primitive —
  every phase acts on a world at rest and the goal is a held object. Here the world is
  ALREADY IN MOTION at t=0 (the flywheel coasts at 1.8-3.0 rad/s, random sign, and
  never stops on its own: bearing time-constant ~167 s), and the goal state is ALSO
  kinetic: success requires the machine verifiably RUNNING again (live |ω| ≥ 1.0 rad/s)
  with the parcel sealed inside. The cube transaction is bracketed between two
  deliberate machine-state changes (arrest → … → restart); nothing is grasped-and-held
  as a goal and there is no proximity target.
- **From the ~377-task corpus**: grep-audited before design. (a) The corpus has 16
  `carousel_*` scenes plus `carousel_airlock` — in every one the rotor is STATIC at
  reset and is used as a **vehicle/conveyor** for the payload; here the rotor starts
  LIVE and is never a vehicle — it is a **gatekeeper hazard** that must be stopped,
  indexed, and restarted. (b) No corpus task starts with the world in motion (all
  "arrest" hits are passive end-stops), and none has a kinetic goal state (every
  success predicate is a settled rest state; here a stopped machine with the parcel
  inside is explicitly NOT success — smoke check 10). (c) The declared safety
  interlock (parcel in the blade sweep while |ω| > 0.25 rad/s ⇒ permanent foul,
  physically a blade strike) makes the ORDER physical law, not rubric fiat: the
  running wheel's blade ring also literally walls the mouth off (smoke check 9).

## The four phases (teleport solution = legitimacy certificate)

`solve.py` — teleport for TRANSPORT ONLY (one deck-to-deck staging hop, both poses far
outside the machine); every load-bearing interaction is real physics through the
scene's sanctioned plant (`wheel_drive`, a bounded ±0.30 N·m hand torque about the
axle) or a bounded fingertip force on the parcel:

1. **ARREST** (→ 0.15): brake servo `τ = −0.20·ω` until the FD rate stays < 0.25 rad/s
   for 0.5 s (`calmed` latch).
2. **ALIGN** (→ 0.15): park servo jogs the stopped wheel until the 90° open sector
   faces the mouth at six o'clock (|yaw| < 0.06 rad, held 30 substeps).
3. **STAGE + POST** (→ 0.50): one transport teleport to the deck centerline
   (y = −0.075, still 47 mm outside the sweep slab), then a regulated push force
   (≤ 0.55 N, velocity servo at 0.08 m/s) slides the parcel across the deck, through
   the parked open sector and the 50 mm mouth, over the sill — gravity drops it 35 mm
   onto the vault floor (`deposited` latch: continuity-gated + transit credential).
4. **RESTART** (→ 1.00): spin-up servo to 1.6 rad/s; `respun` latches after 0.75 s
   sustained ≥ 1.2 rad/s; drive is zeroed and the wheel COASTS. success() goes live;
   3.5 s hands-off persistence is verified at every poll before
   `SIM_GEN_SOLVE: SUCCESS`.

Score = 0.15·calmed + 0.35·deposited + 0.20·respun + 0.30·success — exactly 1.0 iff
success; null ≈ 0 (the wheel never self-calms in-episode); fouled episodes cap at 0.15
forever. Anti-teleport: `deposited` needs per-substep continuity (< 20 mm) AND a
persistent transit credential that any discontinuity disarms and that only re-arms
while the parcel is back outside the wall on the deck side — a parcel teleported into
(or dropped above) the vault earns nothing (smoke checks 4, 5, 8).

## Embodiment argument (single Franka + parallel jaw, OSC)

Documented base at (0.0, −0.50); farthest workspace point 0.64 m < 0.75 m envelope.

- **ARREST / ALIGN / RESTART** = the plant torque: the hand drags the spinning rim
  (150 mm radius, 12 mm-thick plate edges) or hooks the 45 mm yellow handle peg that
  protrudes toward the robot; ±0.30 N·m about the axle is ~2 N tangential at the rim —
  fingertip-scale. Jogging to a park angle is slow peg-pushing on the gravity-neutral
  wheel (CoM on the axis: it stays where left).
- **POST** = a fingertip push or a grasp-slide of the 36 mm cube across the deck
  (deck top 107 mm, well inside the jaw's approach cone); the mouth is 50 mm square so
  a pushed cube self-clears with 7 mm per side; the robot never needs to reach inside
  the vault (gravity finishes the deposit).
- The interlock is an ordering constraint, not a dexterity trap: the deck holds the
  parcel ≥ 50 mm clear of the sweep slab, so a policy that simply does phases in order
  never risks the foul.

## Execution order (declared)

1. `scene.py` designed first with minimal success(); asserts hand-derived.
2. `solve.py` iterated on the forge until the goal was physically reached
   (2 runs: fix = push cutoff moved past the wall back face). SUCCESS on seeds 0, 1.
3. Rubric finalized from the demonstrated trajectory (latched thresholds anchored to
   observed speeds/poses; transit-credential hole found while designing smoke and
   closed BEFORE smoke ran).
4. `smoke.py` (2 runs: fix = closed-gate park moved 180°→90° so the handle peg —
   which blocked the probe after 4 mm — swings clear and the BLADE ring does the
   refusing). `SIM_GEN_SMOKE: ALL PASS 15/15`.
5. Final clean re-runs of solve (seeds 0 and 1) on the final scene.

## Check list (smoke.py, 15)

1. settle: clean reset, parcel on deck, wheel LIVE at the drawn rate, score ~0.
2. randomization real: spin sign flips; magnitude/yaw/spawn vary; rate AND yaw
   readback match the draw.
3. null 3 s: wheel still coasting fast, score ~0, no success.
4. seed-strategy negative: parcel teleported into the vault while the machine runs →
   rejected, score ~0.
5. transit credential: honest calm THEN teleport into the vault → deposited never
   latches, score stays 0.15.
6. declared interlock: real push into the sweep while running → permanent foul,
   cap 0.15.
7. irreversibility: post-foul real brake earns `calmed`, score stays capped.
8. roof denial: drop from above lands ON the roof — the mouth is the only way in.
9. order-forcing: calmed wheel parked open-sector-away physically refuses the same
   push (parcel really moved, no foul, no deposit).
10. kinetic goal: real post with the machine stopped = 0.50 but NOT success.
11. oracle: real restart → success, score exactly 1.0.
12. monotonicity along the whole oracle trace (ladder 0 → 0.15 → 0.50 → 1.00).
13. persistence: 2 s hands-off, no flicker.
14. live RUNNING clause: re-brake revokes success (0.70 floor holds); real re-spin
    restores it.
15. video frames recorded (frames.npz).
