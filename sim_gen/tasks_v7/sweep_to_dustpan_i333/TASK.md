# sweep_to_dustpan_i333 — RiddleTrayScene (`simgen.riddle_tray`)

SORT a mixed load that sits on a hinged RIDDLE TRAY 140 mm up on a dock: pick the
orange CUBES (32 mm, 1–3 sampled per episode) off the bed into an open wooden crate on
the ground, then press the tray's red PADDLE and hold the bed tilted (up to 12°) so
every steel MARBLE (22 mm, 2–4 sampled per episode) rolls out through the tray's
28 mm end-slot into a SEALED HOPPER, then release so the tray falls back onto its
rest stop. The slot is a size riddle: marbles pass it, cubes geometrically cannot
(and high friction keeps them put at full tilt). The hopper's only mouth faces the
slot; every other gap around it (end gap 16 mm, sky slot 20 mm, flank slits 6 mm)
is narrower than a marble, so nothing can be posted in by hand or dropped in from
above. At rest the tray is RECLINED (−2.5°) so its bed drains AWAY from the slot —
the null tray is strictly retentive. Success: all present marbles settled inside the
hopper, all present cubes settled in the crate, tray back at rest on its stop.

## Seed provenance

- **Seed task**: `rlbench/sweep_to_dustpan` (RoboVerse
  `roboverse_pack/tasks/rlbench/sweep_to_dustpan.py`) — "sweep dirt to dustpan":
  grasp a broom (USD asset), sweep five identical 10 mm dirt cubes across the open
  table into a wide, ground-level dustpan mouth. Robot=franka, trajectory-driven.
  One tool grasp, undifferentiated debris, ground-plane transport toward a
  receptacle whose mouth is at ground level and approachable from anywhere.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| where the debris is | on the open ground | ON A MACHINE: the bed of a hinged riddle tray 140 mm up; nothing useful happens at ground level |
| debris identity | five identical dirt cubes, all wanted, one destination | MIXED load, two species with opposite fates: marbles → sealed hopper (through the slot), cubes → crate (over the fences); both counts sampled per episode |
| the receptacle | wide dustpan MOUTH at ground level, open from the whole half-plane | SEALED hopper: the only mouth hides behind the tray's 28 mm end-slot, 85 mm up; end gap 16 mm / sky slot 20 mm / flank slits 6 mm all refuse a 22 mm marble |
| transport actuator | a grasped broom dragging debris along the ground | the TRAY ITSELF: a sustained paddle press (hinge torque against tray weight + payload) tilts the bed and gravity rolls the marbles out; cubes go by per-piece pick-and-place |
| selectivity | none | the slot passes marbles and refuses cubes (28 vs 32 mm); wrong destinations (cube in hopper, marble in crate) earn nothing |
| null behaviour | dirt stays where left | rest RECLINE (−2.5°, the lower joint stop) drains the bed AWAY from the slot: an untouched or released tray retains every marble |
| judging | scripted trajectory, no checker | live geometric success (marbles below the hopper mouth sill, cubes in the crate box, tray back on its stop) + latched credit (crate 0.20·fraction, loaded-tilt 0.15, bin 0.45·fraction, non-success cap 0.80) |
| assets | broom + dustpan USDs | 100 % procedural (kinematic dock/hopper + crate compounds, dynamic tray with authored MassAPI + spawn-authored revolute joint, primitive spheres/cubes) |

Code shares nothing with the seed: `@SCENES.register` BaseScene, three custom
compound spawners with explicitly authored physics materials, per-env spawn-authored
hinge (`_author_hinge`, limits [−2.5°, +12°]), dock/tray/crate-frame predicates,
latches in `post_step`, `register_env(..., robot="null")`.

## Why strategically different

The seed's skill is *grasp a long tool and sweep undifferentiated debris across the
ground into a wide ground-level mouth* — planar pushing, zero selectivity, target
approachable from anywhere, no mechanism. Here that plan earns nothing: smoke check 5
CONSTRUCTS it (a marble dragged quasi-statically along the ground at 3× its weight
straight at the hopper) — it travels freely under the tray, genuinely reaches the
hopper's ground-level blank front wall (peak approach at the wall face, 74 mm below
the mouth sill), is refused, and even rolls itself back off the wall on its own
spin: never inside at any step, no credit. What the solver must bring instead: (1) **mechanism operation as transport**
— the marbles are moved by TILTING THE TRAY (a sustained press on a handle against a
gravity return), not by pushing them; the seed has no articulated element at all;
(2) **species-selective routing** — one mixed pile, two receptacles, two mechanics:
cubes are lifted out per piece over the fences (the slot refuses them), marbles are
released through the slot (fences and height refuse hand-carrying them in — every
hopper gap is sub-marble); (3) **count uncertainty** — both species' counts are
sampled per episode and must be read off the scene; (4) **a live end-state
constraint** — success requires the tray RELEASED back onto its stop, so "hold the
mechanism forever" cannot finish the episode (rejected by smoke 11).

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.5 s; layout readback (present patterns, kb/kc, tray angle, tray
   mass via physx view, piece + crate poses); asserts: finite, tray reclined on its
   stop (−2.53°), nothing binned/crated, all present marbles on the bed, score 0.0,
   no success.
2. **P1 crate the cubes** (teleport = TRANSPORT ONLY; settling is contact physics):
   each present cube is carried to a hover 100 mm above the open crate mouth
   (crate-frame slot pattern, yaw-aware) and dropped; verify `cubes_in_crate` +
   settled, crate latch fired, score monotone (≤ 4 attempts; retries never fired).
3. **P2 paddle-press discharge**: hinge-torque servo about the world-y axis =
   gravity feedforward from the MEASURED payload + PD (kp 12, kd 0.5), capped at
   3.5 N·m ≈ a 19 N fingertip press at the 0.18 m paddle. Ramp to 11.5° in 0.4 s
   (the tilt latch arms while marbles still ride), hold until every present marble's
   hopper latch fires; a rock cycle (drop to 4°, re-tilt) handles stragglers — never
   fired on any seed. Asserts: max tilt > 9°, tilt latch fired, all binned.
4. **P3 release**: ramp the torque out over 1 s, clear it, settle 1.5 s; asserts:
   `tray_at_rest`, cubes still crated, success() live, score 1.0; hands-off
   persistence 10 × 40 steps = 3.33 s; success still holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints, e.g. seed 0 (kb=4, kc=2): 0.0000 → 0.1000 →
0.2000 → 0.8000 → 1.0000 → 1.0000.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(−0.15, 0.00, 0.00), facing +x** (nominal reach 0.855 m).
The tray bed spawn band lies at world x ∈ [−0.12, −0.06], z ≈ 0.15 (0.10–0.30 m
from the base), the crate at (0.02 ± 0.05, 0.40 ± 0.05) ≈ 0.45 m, and the paddle at
world ≈ (0.48, 0, 0.26) ≈ 0.68 m — all inside the dexterous shell. The arm never
needs to reach into the hopper or through the slot.

- **Cubes** (32 mm, 80 g): standard top pinch (32 ≪ 80 mm jaw span) from the bed —
  the spawn band is asserted BEHIND the canopy, so every cube is open to a straight
  top-down grasp over the 50 mm fences at comfortable height; carry to the crate is
  a plain pick-and-place. Color/count read trivially off a wrist camera.
- **Marbles** (22 mm, 30 g): NEVER touched. They are transported by the tray; the
  agent only has to see when the bed is empty (open top over the spawn band).
- **Paddle** (40 × 120 mm red plate on the beam): a sustained fingertip/knuckle
  press straight down at ≈ 0.68 m reach, ~4–8 N for the whole hold (worst-case
  hold torque ≈ 1.4 N·m at the 0.18 m lever = 8 N) — well inside Franka's
  continuous capability; releasing is just lifting the finger, and the tray falls
  back on its own (gravity return, no re-grasp).
- **Tray fences / slot / hopper**: never grasped; the dock and crate are kinematic,
  so incidental contact cannot move the goal frames.

## Execution order (declared)

**Cubes before tilt is the natural order but is NOT hard-required by the rubric** —
the rubric is order-blind (latched per-piece credit both ways). The physical
couplings are: a cube left aboard during the tilt stays put (friction + the slot
refuses it, smoke 6) so tilting first merely leaves the cube pick for later; and
each marble must be discharged THROUGH the slot by a genuine loaded tilt (tilt
credit requires a marble aboard — empty tilting stays dark, smoke 8). The only hard
per-piece constraints: identity → destination (wrong destinations earn nothing,
smoke 9) and the tray must end RELEASED (smoke 11). solve.py crates cubes first
purely because a clear bed discharges cleanest.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (kb=4, kc=2): SUCCESS, 0.0 → 0.1 → 0.2 → 0.8 → 1.0 → 1.0.
- `solve --seed 1` (kb=2, kc=1): SUCCESS, same monotone ladder.
- `solve --seed 2` (kb=2, kc=3): SUCCESS — three seeds, three distinct present
  patterns; the cube re-drop, straggler rock cycle, and marble retries never fired.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 12/12**, frames.npz saved:
  1. settle/no-NaN: tray reclined on its stop, all present marbles on the bed,
     nothing binned/crated, score 0, no success
  2. randomization readback: piece xy / cube yaw / crate pose all differ across seeds
  3. count coverage: over 25 resets kb covers {2,3,4}, kc covers {1,2,3}; every
     piece index present and parked at least once; depot readback agrees with mask
  4. null policy: 400 idle steps — no tray drift, nothing enters, score 0
  5. SEED STRATEGY: a quasi-static 3×-weight ground drag straight at the hopper
     travels freely and genuinely reaches the ground-level blank front wall (peak
     approach tracked AT the wall face, 74 mm below the mouth sill), is refused,
     and rolls back off it — never inside at any step (latch dark), score ~0
  6. SLOT SELECTIVITY: at full tilt a pushed cube (1.2× weight) visibly slides and
     is refused by the 28 mm header (stays on the bed) while a marble released at
     the same station rolls through into the hopper
  7. RECLINE RETENTION: a marble placed just behind the slot on the resting tray
     rolls BACKWARD to the back wall — the null tray never discharges
  8. EMPTY-TILT GATE: full tilt with no marble aboard leaves the tilt latch dark
  9. WRONG DESTINATIONS: cube constructed in the hopper + marble settled in the
     crate → no latch, score ~0, no success
  10. near-misses: a marble dropped over the 20 mm sky slot never enters; a cube on
     the ground beside the crate is not crated
  11. TRAY-HELD-UP GATE: everything binned+crated but the tray still pressed at
     full tilt → success False, score capped 0.80; released onto the stop, the rest
     conjunct flips it True
  12. video frames.npz saved

## Files

- `scene.py` — RiddleTrayScene + dock/tray/crate compound spawners + per-env hinge
  authoring + rubric; registers `simgen.riddle_tray`.
- `solve.py` — crate-drop + torque-servo tilt-discharge certificate (`--seed N`).
- `smoke.py` — 12-check rejection battery + video.
