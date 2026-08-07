# coke_task_i15 — Counterweigh the beam balance

## Seed provenance

Seed: `simpler_env/coke_task` — grasp a coke can and lift it. The seed's plan is
"acquire one known object and change its pose"; its rubric reads the manipulated
object's own state (can height).

## What changed, and why it is strategically different

This task keeps the coke-can *objects* and discards the seed's *plan* entirely.

The scene is a knife-edge **two-pan beam balance** (pendulum-stable: beam CoM
50 mm below the pivot) whose left pan is preloaded at reset with a random
non-empty subset of three steel plates (100/200/400 g — a 3-bit binary code,
7 possible subsets, plate thickness encodes mass). Three cans stand on the
floor: red 100 g, blue 200 g, green 400 g (height encodes mass). The goal:
**place exactly the matching can subset on the empty pan so the beam settles
level (|tilt| ≤ 4°)**, with all spare plates untouched on their pan and all
cans either on the right pan or clear of the instrument.

Strategic differences from the seed and from every other tasks_v7 task:

1. **The rubric never reads the manipulated objects' target pose.** Success is
   the settled equilibrium angle of a *mechanism* — a physical null
   measurement. Gravity, not a pose check, decides whether the answer is
   right. No mass is ever summed by the rubric; the beam performs the
   arithmetic. No other tasks_v7 task judges a mechanism's equilibrium.
2. **The plan changes per episode.** The preloaded subset is one of 7; the
   solver must *read* the scene (plate presence by readback/describe) and
   select a different can combination each time. The seed and the parameter
   variants of it have a fixed plan with jittered coordinates.
3. **Wrong answers fail physically, not by tolerance.** A ±100 g error rests
   the beam on its stop at ≥13.6° — more than 3× the 4° gate — so the
   discrimination margin is built into the statics, and the dead band is <1°.

## Scene (engineered numbers, asserted in `__post_init__`)

- Kinematic stand: base plate, two posts, V-notch cradle (walls at ±22.5°),
  stop platforms (top z 0.0535) that catch a tipped beam at **11.0°**.
- Dynamic beam (0.8 kg): axle r 5 mm riding the V-notch (geometric bearing, no
  USD joints), CoM 50 mm below the axle (restoring pendulum), angular damping
  6.0 (velocity-only — every judged state is pure statics), arm half-length
  0.17 m, pans: floor disc r 75 mm + 12 rim posts (ring r 71 mm, h 30 mm).
- Anti-cheat geometry: the drop-post slot is **45.5 mm < 56 mm** can diameter
  (a can cannot bridge/prop the beam center); the arm-over-stop gap is
  **33.5 mm** (no can fits under an arm to prop it).
- Randomization (verified by smoke readback): stand xy ±3 cm around (0.28, 0),
  yaw ±35°; preload subset `randint(1, 8)`; cans spawn on the floor in a
  0.20–0.50 m keep-out annulus with random headings.
- Physics notes that matter: loose bodies get damping (lin 0.3 / ang 0.8),
  32/4 solver iterations, and the sim sets
  `enable_external_forces_every_iteration: True` — without these, a light 8 mm
  plate atop a 3-deep stack chatters ~0.35 rad/s forever (sleeping is
  disabled) and no settle gate is honest. Damping acts on velocity only, so
  equilibria are unchanged.

## Rubric (`success` / `score`)

`success` requires ALL of, simultaneously and settled:
- `seated` — beam axle in the V-notch (dz < 15 mm), else nothing counts
  (laying the beam level on the floor scores 0);
- `level` — |tilt| ≤ 4° AND the beam is *free of its stops*;
- `preload_ok` — every preloaded plate still rides its pan;
- `spares_ok` — no spare plate moved onto a pan (blocks "top up with a spare
  plate" — physically level, still rejected);
- `cans_ok` — every required can on the right pan, every non-required can
  clear of the instrument (blocks shuffles like "wrong can on the empty pan,
  compensating can on the loaded pan" — also physically level, still
  rejected).

`score` is monotone via latches: 0.10 first correct can on pan + 0.15 each
correct can + 0.10 near-level, capped 0.65 until full success → 1.0. Removing
a can after success collapses the live clauses; the latched partial remains.

## Teleport solution (`solve.py`) — transport only

- **P0** — settle, read the preload by readback, print the layout, assert the
  beam is seated, tipped hard (|tilt| > 2× gate), score ≤ 0.03.
- **P1..k** — for each required can: teleport it to an 8 mm hover over the
  empty pan (beam-local frame, live beam quat) and release; the drop, the
  contact, and the beam's response are pure dynamics. Drop offsets are
  zero-net-moment layouts (k=2: x=0, y=±31 mm; k=3: equilateral triangle,
  circumradius 35.5 mm, rotated so Σmᵢxᵢ ≈ −2e-5 kg·m ≈ 0.02°) with ≥5.5 mm
  hull clearance (> the 3 mm summed contact offsets). Up to 4 drop retries
  per can; escaped cans re-dropped.
- **Trim loop** — if the settled tilt is outside 0.9× the gate, re-drop the
  heaviest can nudged along the arm (closed loop on the *instrument's*
  reading); if inside the gate but a settle gate flickers, wait instead.
- **P-final** — assert success and score ≥ 0.999, then 3.3 sim-seconds fully
  hands-off with success sampled 10×, all samples required, before
  `SIM_GEN_SOLVE: SUCCESS`.

Teleports move cans through free space only; every load-bearing interaction
(can–pan contact, beam rotation, plate stacks) is contact dynamics.

## Embodiment argument

A Franka based at ~(−0.15, 0, 0) facing +x covers the workspace: the stand
nominal (0.28, 0) ± 3 cm and the 0.20–0.50 m can annulus are inside its
~0.85 m reach. Per-object strategy: each can is an upright cylinder d 56 mm —
under the 80 mm parallel-jaw opening — at 100–400 g, well under payload; a
side pinch at mid-height, lift over the 30 mm pan rim, and release from a low
hover reproduces the solution's drop. Required placement accuracy is ±3 cm
(pan_fit_r 62 mm vs rim ring 71 mm), coarse for a Franka. The plates are
never manipulated. The stop platforms bound the beam's excursion to 11°, so
no fast moving parts threaten the arm.

## Execution order (declaration)

1. Designed the mechanism and a *minimal* `success()` first.
2. Wrote `solve.py` and iterated it on the forge until the goal was reached
   robustly — **22/22 seeds (0–21) SUCCESS**, covering all 7 preload subsets,
   with zero trims needed on final code.
3. Only then finalized the full rubric (latches, spares_ok/cans_ok clauses,
   score shape) and the `__post_init__` statics asserts.
4. Wrote `smoke.py` last, as an adversarial battery against the final rubric.
5. Re-ran both clean on the forge on the final submitted code.

## Checks

`solve.py` (per run): P0 asserts (seated, preload settled, tipped > 2× gate,
score ≤ 0.03), per-can drop verification, final success + score ≥ 0.999,
strict all-samples 3.3 s persistence. Passed on seeds 0–21.

`smoke.py` — `SIM_GEN_SMOKE: ALL PASS 13/13`:
1. Settle / no-NaN; beam seated and resting on a stop; score 0.
2. Randomization readback over 6 resets (yaw spread 36.5°, stand xy 56 mm,
   can_red 782 mm, ≥3 distinct preload subsets).
3. Null policy 300 steps → no success, score ~0.
4. Unload-the-preload: plates removed → beam levels (−0.12°) but score 0.
5. Seed-strategy end state: cans lifted/delivered beside the balance → 0.
6. Wrong subset (±100 g): rests on stop at −11.0°, rejected, score ≤ 0.65.
7. Spare-plate cheat: level within 0.30° — rejected by `spares_ok`.
8. Shuffle cheat (wrong can + compensator on loaded pan): level within
   0.27° — rejected by `cans_ok`.
9. Near-miss: correct can on the stand base — rejected.
10. Off-cradle: beam laid level on the ground, correct cans placed — every
    clause but `seated` holds; rejected.
11. Remove-a-can after constructed success → score collapses to the latched
    partial (0.350 observed), success false.
12. Rejection audit: success never fired during any wrong-state check.
13. frames.npz recorded (297 × 600 × 960 × 3).
