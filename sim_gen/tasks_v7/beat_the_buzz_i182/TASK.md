# beat_the_buzz_i182 — Beat the buzz by weight substitution (pressure-plate heist)

**Scene:** `simgen.pressure_plate_heist` (`PressurePlateHeistScene`, robot="null")
**Seed:** `rlbench/beat_the_buzz`
(`RoboVerse/roboverse_pack/tasks/rlbench/beat_the_buzz.py`)

## Seed provenance and what changed

The RLBench seed is the buzz-wire game: grasp a wand and guide its loop along a
bent wire from one end to the other without the loop ever touching the wire. Its
whole plan is a **keep-out traversal** — one grasped object, one continuous
guarded sweep, and the "don't trigger the buzzer" constraint is *geometric
proximity along a path*. Skill = clearance control while tracking a curve.

This task keeps the seed's one recognizable predicate — *an alarm that must
never fire while you complete the goal* — and replaces the entire plan skeleton:

1. **Avoidance is worthless.** The alarm here is not a proximity sensor along a
   path; it is a spring-loaded PRESSURE PLATE under the golden idol. There is no
   trajectory of the idol, however slow, careful, or clearance-maximizing, that
   avoids it: the moment the plate is unloaded it rises 15 mm and the alarm
   latches (smoke check 5 executes exactly the "gentle careful lift" — the
   solve's own velocity servo — and it fires). The seed's entire skill transfers
   zero.
2. **The constraint is a load invariant, not a clearance invariant.** What must
   be maintained is *force on the plate ≥ threshold at every instant*, which
   forces a **strict execution order**: a sufficient counterweight must be ON
   the plate *before* the idol leaves it.
3. **The counterweight must be selected by inferred mass, not appearance.** Two
   visually similar 5.5 cm cubes sit on the table: dark granite (700 g) and
   white foam (60 g). Only the granite exceeds the spring's threshold — the foam
   *physically seats on the plate* and the plate still rises when the idol is
   lifted (smoke check 7: occupancy is not mass). Which side of the table each
   block is on is sampled per episode.
4. **Failure is permanent.** The alarm is a latch on the physical plate
   extension. Smoke check 6 rebuilds the *complete goal geometry through the
   plant* after a wrong-order attempt — granite dropped onto the risen plate
   pushes it back down, idol delivered upright to the pad — and it is still
   worth exactly 0. "Undo and redo" is not a strategy; the order itself is the
   content of the task.

So the plan skeleton changes from *"guide one object along a curve, maximizing
clearance"* to *"infer which of two objects is heavy enough, install it as a
counterweight FIRST, then swap the target out and deliver it upright"* —
ordered weight substitution under a latched load invariant, instead of a
guarded traversal.

## Why strategically different from the examined sibling

Sibling examined in tasks_v7: **lamp_off_i101** (twist-lock unplug). Both keep a
"negative constraint" flavour, but the structures are disjoint: lamp_off's core
is *identification by cord-tracing plus a rotation-gated extraction* on a D6
mechanism — its constraint (keep the fan powered) is about *which* object you
act on. Here there is no mechanism to unlock and no key motion: the plate's D6
travel is not something the agent manipulates directly at all — it is the
*alarm sensor*. The decisive content is a **temporal order** (counterweight
before removal) enforced by a **mass threshold** (weight, not identity, of the
substitute), and the deliverable is a place-upright goal on a separate pad.
No twist, no tracing, no extraction axis; conversely lamp_off has no ordering
constraint driven by a load invariant and no mass discrimination.

## Solution outline (solve.py — teleports are TRANSPORT ONLY)

All load-bearing interaction flows through contact dynamics: bodies are lifted
and set down by vertical velocity-servo forces through their CoM written into
the scene's `drive_f` buffers (the stand-in for the Franka's pinch-grasp lift,
below), applied by `post_step`, which also runs the spring/alarm plant every
step of every phase — a solution that unloaded the plate at any instant would
zero its own score. Teleports only carry an already-lifted body across free
space (zero velocity, gravity-held for the transition step); every set-down is
a release from 8 mm and a real contact settle. Wrenches act one substep late,
so gains obey K·dt/m « 1 (idol 0.167, granite 0.143 at dt = 1/120 s).

- **Phase 0 (reset):** settle 0.5 s, read the sampled granite side / pad pose /
  idol jitter, mass readback (0.700/0.060/0.500 kg), drive buffers asserted
  zero. `SIM_GEN_SCORE` 0.000.
- **Phase 1 (counterweight):** servo-lift the granite off the table
  (f = m·g + KV·(v_des − v), v_des = clamp(3·(z_des − z), ±0.25 m/s), f
  clamped to [−3, 16] N), carry it to 8 mm above the plate on the far side from
  the idol (offset 62 mm: fully inside the on-plate band, ≥ 2.7 mm contact gap
  to the idol), release, streak-gated settle. Readback asserts granite on
  plate, plate down, no alarm. `SIM_GEN_SCORE` 0.350.
- **Phase 2 (steal + deliver):** servo-lift the idol straight up — the plate's
  extension is monitored the whole lift and stays at 0.0 mm (granite pins it) —
  carry to 8 mm above the sampled pad, release, settle upright by contact; wait
  for `success()`. `SIM_GEN_SCORE` 1.000.
- **Phase 3 (persistence):** drive buffers asserted zero, 3.5 simulated seconds
  hands-off; `success()` is live state. Only then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0 (granite side +y), 1 (side −y) and 2 (side +y)
all print the monotone sequence 0.000 → 0.350 → 1.000 → 1.000 and
`SIM_GEN_SOLVE: SUCCESS`.

**Declared abstraction:** the plate rides an authored USD D6 joint
(pedestal → plate) freeing exactly vertical translation in [0, 35 mm]; the
spring is a constant 3.5 N upward force applied in `post_step` — chosen
strictly between (plate+foam)·g ≈ 2.6 N and (plate+idol)·g ≈ 6.9 N ≤
(plate+granite)·g ≈ 8.8 N, so the interlock is real rigid-body statics, not a
scripted flag. The alarm is a latch on the *physical* plate extension readback.

## Required execution order (declared)

Counterweight BEFORE removal: the granite must be resting on the plate before
the idol's weight comes off it. The reverse order — or a foam substitute, or a
counterweight anywhere but on the plate — unloads the plate below the spring
threshold and permanently fails the episode. This order is forced by the load
invariant, not by convention; smoke checks 5–8 construct each violation
physically and assert rejection.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `hold_latch` (0.35): the granite has rested fully on the depressed plate,
  settled, alarm unfired.
- `clear_latch` (0.15): the idol has left the pedestal region (or been lifted
  high) while the plate stayed down and the alarm stayed unfired.
- `pad_latch` (0.15): the idol has stood upright, settled, on the delivery pad.
- current success (→ 1.0): idol upright + settled on the pad AND plate
  currently held down AND alarm never fired AND everything still.
- The whole score is multiplied by (1 − alarm): tripping the alarm zeroes it
  forever — credit only evaporates under incorrect behavior. Latches are
  NaN-safe transient-achievement credit (torch.maximum); null policy ≈ 0.

## Embodiment argument (single Franka, parallel jaw, OSC)

Base at ≈ (0.42, 0.0) on the table top (z = 0.40), facing −x: the pedestal
plate centre is 0.60 m away at z ≈ 0.52, the blocks are at 0.25–0.50 m, the
pad at 0.22–0.35 m — everything inside a 0.75 m reach disc at elbow height.

- **Mass identification:** the blocks are visually distinct (dark granite vs
  white foam) and, if ambiguity remained, a single grasp-and-hoist of either
  block resolves it by wrist-torque readback — no extra dexterity.
- **Granite (5.5 cm cube, 700 g):** textbook top-down parallel-jaw pinch for
  the 80 mm jaw, well under payload; set-down on the plate is a place-from-8 mm
  at the far side from the idol — the 19 cm plate leaves a 3+ cm corridor.
- **Idol (4.5 cm square column, 11 cm tall, 500 g):** waist pinch grasp with
  vertical wrist axis; the lift is a straight +z Cartesian retreat at
  ≤ 0.25 m/s — exactly the vertical CoM servo the drive buffers stand in for —
  then a free-space carry to the pad and an upright place from 8 mm.
- **Order constraint:** satisfied by sequencing alone; no simultaneous
  two-hand action is ever needed (the plate is held down by the *granite*, not
  by the robot, while the idol moves).

## Checks (smoke.py — rejection battery, 14 checks)

1. Clean reset: finite state, plate seated at bottom, idol standing on it,
   score ~0.
2. Readback: physical masses order the interlock (foam < spring < idol <
   granite, via `get_masses()`); blocks on opposite sampled sides; pad and
   blocks inside their cfg bands.
3. Randomization is real (8 seeds): both granite sides drawn; granite, pad and
   idol-jitter positions all vary (readback).
4. Null policy: 2 s of no action — no alarm, no latch, score < 0.05.
5. **Seed-strategy family (careful lift):** the idol lifted first with the
   solve's own gentle servo — the plate physically rises past the 15 mm
   trigger (readback) and the alarm latches.
6. **Permanence:** the goal geometry then fully rebuilt through the plant
   (granite pushes the risen plate back down; idol upright on the pad) — every
   geometric predicate True, success False, score == 0.
7. Wrong counterweight: the foam seats on the plate (readback) but the idol
   lift still fires the alarm — mass threshold, not occupancy.
8. Wrong place: granite parked on the table right beside the pedestal (moved,
   near, not on the plate) — idol lift still fires the alarm.
9. Near miss: correct order but the idol lands ~13 cm off the pad — 0.50
   partial credit, no success.
10. Toppled delivery: idol released lying down over the pad — rests on the pad
    but not upright: no pad credit, no success (0.50).
11. Exactness: full correct strategy → success() and |score − 1.0| < 1e−3.
12. Achievement latch: an 8 N shove pushes the delivered idol off the pad —
    success revoked (live state), latched 0.65 remains.
13. Alarm latch: lifting the granite back off frees the spring — plate rises,
    alarm fires, all credit evaporates to 0.
14. Camera: ≥ 20 rgb frames captured across the checks → `frames.npz`.
