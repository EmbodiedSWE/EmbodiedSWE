# put_mug_i228 — MugDispenserScene (`simgen.mug_dispenser`)

Stage the mug under an elevated ball silo, pull the silo's sliding gate open through
its channel, and catch all three dropping balls IN the mug — finish with the balls
settled inside the upright, resting mug.

## Seed provenance

- **Seed task**: `roboverse_pack/tasks/embodiedgen/put_mug.py` (`embodiedgen/put_mug`,
  RoboVerse). A Franka picks up a mug and sets it down inside a marked target region;
  success is bbox containment of the mug (`DetectedChecker` + `RelativeBboxDetector`,
  ±0.1 m box, orientation ignored). The episode is over the moment the mug stands in
  the box.

## What changed and why it is strategically different

- **The seed's whole goal is demoted to an unmarked sub-step.** Here, carrying the mug
  and setting it down earns *nothing by itself* (smoke check 6). The mug must be placed
  *functionally*: upright on the dispenser's base plate, centred under the silo bore —
  no marking defines the spot, only the mechanism above it (a spatial-reasoning
  placement, not a "reach the marked box" placement).
- **A mechanism the seed does not have.** The three balls start sealed in a deep,
  narrow silo (bore 32 mm, top at 38 cm — no gripper reaches them). The only way to
  move them is the captive sliding gate under the silo: a real slide-in-channel
  interaction (the plate rides between rails and guide strips with ~1.5 mm play,
  loaded by the ball column's weight) pulled outward by its tab.
- **The payload is never manipulated directly.** The balls travel exclusively by
  gravity: they ride the retracting gate, drop through the bore, fall ~7 cm and are
  caught by the mug. The seed moves its payload (the mug) entirely in-hand.
- **Different from sibling `put_mug_i173`** (same seed): i173 turns the mug into a
  *hung* object — thread the handle window over a specific peg, a suspension topology
  goal with a decoy peg. Here the mug is a *receptacle* for other bodies, the goal is
  containment-of-many, and the load-bearing interaction is a prismatic gate pull, not
  handle threading. No shared mechanism, goal predicate, or interaction type.
- **Different from the `pen_holder` exemplar**: no hand-insertion of items into a
  container — the robot never touches what ends up inside.

## Task description (what the scene builds)

Fully procedural (custom compound spawners, one rigid body each):

- **rig** (kinematic, repositionable per reset): dark base plate on four legs, a slide
  channel at 16 cm (two lower rails + two upper guide strips; the rear leg pair is the
  open-end stop, a stop block closes the other end), and a steel-blue square silo tube
  above the channel (bore 32×32 mm, z 0.178–0.380, open top).
- **gate** (dynamic, 40 g): blue plate (72×52×6 mm) captive in the channel, sealing the
  silo bottom, with an upright 60 mm pull-tab blade on the outward side.
- **balls** (dynamic ×3, 22 mm, 18 g, orange): stacked in the bore, resting on the gate.
- **mug** (dynamic, 250 g): cream hollow cup Ø80×92 mm (floor disc + 8-box octagonal
  wall) with a loop handle; spawns upright on the ground in a random side band.

Randomization (verified by readback in smoke): rig xy ±3 cm and yaw 180°±45°; mug x
∈ [0.03, 0.20], |y| ∈ [0.10, 0.28] with a 50/50 side flip, free yaw ±180°.

**Success** (`success()`): all three ball centres inside the mug interior *in the mug's
body frame* (radial < 26 mm, z band covering floor-to-rim rest positions), mug upright
(≤10°) at resting height, and a 45-step stillness streak of mug+balls with no per-step
pose jump > 2 cm (anti-fly-through: teleported states judge False until they really
settle there). **Score**: latched 0.15·staged + 0.25·opened + 0.15·per-ball-contained,
capped at 0.85; exactly 1.0 iff success() live.

## Teleport solution (solve.py)

- **P0** — reset, settle 60 steps, mass/gate-pose readbacks, score 0.
- **P1 TRANSPORT (the only teleport)** — one root-state write carries the mug from its
  ground spawn to the staged pose: upright on the base plate, centred under the bore,
  handle turned to the rig's clear side. Both endpoints are contact-free rests — this
  is exactly what a carry does. The written state cannot satisfy success(): the balls
  are still sealed, and the jump resets the stillness streak anyway. Score 0.15.
- **P2 PULL THE GATE (contact dynamics)** — a horizontal external force at the gate's
  CoM (velocity-regulated bang-bang along the rig's channel direction, 1.5→8 N stall
  escalation, v_des 0.05 m/s) drags the captive plate outward through its channel
  under the ball column until the bore is uncovered (gate-x ≥ 0.058). The balls ride
  the plate, drop through the bore and are caught by the mug — gravity + contact all
  the way; nothing ever writes a ball pose after reset. Score 0.40.
- **P3 SETTLE (hands-off)** — forces cleared; balls rattle to rest in the mug, streak
  accumulates to success(). Score 1.0.
- **P4 PERSISTENCE** — 3.5 more simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS`
  only if success() still holds.

`SIM_GEN_SCORE` is printed at each boundary and asserted non-decreasing.

## Embodiment argument (Franka, base ≈ (−0.20, 0, 0) facing +x)

- **Mug carry**: body Ø80 mm with a 6 mm wall — rim pinch from above — or the loop
  handle (bar cross-section 10×8 mm) fits the parallel gripper. Ground spawn at
  x 0.03–0.20 and the rig at x ≈ 0.42 are both inside a comfortable reach envelope.
- **Staging**: tolerance is a 20 mm-radius disc — an order of magnitude above
  closed-loop arm precision; the mug passes under the channel with 52 mm of clearance
  (rail underside 164 mm vs mug rim 112 mm) and the staged spot is approached from the
  open flank between the leg pairs.
- **Gate pull**: the tab is a vertical 20 mm-wide, 60 mm-tall blade — a natural pinch
  target; the pull is a straight horizontal 5 cm stroke along the channel, at yaw
  180°±45° roughly *toward* the robot, and ≤8 N ≪ Franka payload. The tab clears the
  silo wall by the full plate travel, so the grasp never collides with the rig.
- **No step needs the balls touched**: the silo seals them until the gate opens; after
  that, gravity does the transport into the already-staged mug.

## Execution order

No hard order is enforced — only the final physical state is judged. The intended
order (stage, then pull) is the easy path; opening the gate first spills the balls
onto the base/ground, and recovering every ball into the mug by hand remains
physically possible and is honestly judged (per-ball latches + the same containment
predicate). `describe()` states this explicitly.

## Verification (forge, RTX 4090)

- **solve**: `SIM_GEN_SOLVE: SUCCESS` on **seed 0** and **seed 1** (distinct layouts by
  readback: rig yaw 170.9° vs 215.1°, different rig/mug xy). Scores 0 → 0.15 → 0.40 →
  1.0 → 1.0, non-decreasing; ~27 s wall-clock each.
- **smoke**: `SIM_GEN_SMOKE: ALL PASS 16/16`, 185 frames recorded (frames.npz):
  1. settle/no-NaN + layout readback (rig in jitter box, gate closed, balls in bore)
  2. score ~0 at reset, no success
  3. randomization readback: rig xy + yaw vary (6 seeds)
  4. randomization readback: mug xy + yaw vary, side flips, never near the rig
  5. null policy 240 steps → score ~0
  6. seed strategy (carry & set down on open ground) → earns nothing
  7. staged-only → only the 0.15 latch, balls stay sealed
  8. partial open (readback: gate really moved, gap < ball Ø) → no latch, no drop
  9. spill (gate fully opened, mug elsewhere) → 0.25 only, every ball outside the mug
  10. near miss (2 in / 1 out, genuinely settled) → not success, ~0.30
  11. tipped mug with all balls in the cavity (balls_in true) → not success, ≤0.46
  12. fly-through: success geometry teleported in, judged cold → rejected (jump guard)
  13. fly-through B: real frames then a ball yanked out → success never fired, latch ~0.45
  14. staged latch survives the mug's return to the ground (~0.15)
  15. rejection audit: success() never True anywhere in the battery
  16. final no-NaN
