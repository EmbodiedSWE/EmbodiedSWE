# open_oven_i213 — Batch Scale

Weigh out the sampled target load on a baker's spring scale: place brass/iron
counterweights in the hanging tray until the orange pointer rests level with the green
target tab (scene `batch_scale`, env `simgen.batch_scale`).

## Seed provenance

Seed: `rlbench/open_oven` — grasp the oven door's handle and pull the hinged panel
through a large free arc until the oven stands open. One prehensile act on the
appliance's single moving part, judged by a binary articulation reading; direction of
travel is the only decision, and "more open" is never worse.

The seed's end state (a door panel swung open on its hinge) is **not expressible** in
this scene — nothing here has a door, lid or panel; documented N/A. The seed family's
naive plan — grab the appliance's moving part and haul on it — is constructed as an
actual force probe in smoke check 5 (a sustained 25 N pull + 2.5 N·m pry on the tray):
the tray rides its 4 mm rail to the stop, opens nothing, springs straight back, and
scores ~0.

## What the task is

A spring SCALE stands on the floor: a kinematic pedestal and tick mast, and a dynamic
open-top TRAY (20 cm square, 3.2 cm colliding rim) hanging on an authored vertical
prismatic joint, suspended by a spring/damper plant applied every substep in
`post_step` (k = 78.48 N/m, c = 14 N·s/m — discrete-stability audited at 120 Hz).
Loading the tray sinks it: **one 120 g unit = exactly one 15 mm tick** on the mast
(white tick = empty rest, 7 black ticks below). An orange pointer fixed to the tray
reads against the ticks.

Five loose counterweights lie scattered on the floor: three small brass cubes (40 mm,
120 g = 1 unit) and two large iron cubes (50 mm, 240 g = 2 units). Every episode
samples a target load u* ∈ {2..6} units and physically parks a green tab beside the
target tick (kinematic teleport, readback-verified); the weights land on a permuted
slot row with xy jitter and free yaw.

Goal: place counterweights INSIDE the tray until the pointer rests level with the
green tab — deflection within band_tol = 6 mm (±0.4 units; the band admits exactly ONE
integer load), the deflection **accounted for** by the mass actually resting in the
tray, and that state held for 24 consecutive substeps. Too little AND too much both
fail; a wrong load must be exchanged, not nudged.

## Strategic difference

- **vs. the seed**: the seed's whole plan is one grasp and one guided pull through a
  pre-built hinge arc — a binary, monotone goal ("more open" never hurts) on a part
  the appliance provides. Here the manipulands are not part of the appliance at all,
  no configuration of the scale itself is the goal, and hauling on the scale's moving
  part (the seed's move) is an explicit constructed failure. The task is
  **goal-conditioned discrete mass selection with an indirect indicator**: the pointer
  cannot be "moved" anywhere by manipulation — only the resting load determines where
  it settles — and the target changes every episode, so the solver must READ the green
  tab, decompose u* into the two available denominations (2 → [2], 3 → [2,1],
  4 → [2,2], 5 → [2,2,1], 6 → [2,2,1,1]), and transfer exactly that subset. The failure
  direction is two-sided (under- and over-load), unlike the seed's one-sided arc.
- **vs. corpus tasks read**: the door/articulation family (`close_microwave_i4/i5`,
  `close_grill_i8`, libero drawer/stove) drives built joints to a fixed side of a
  threshold; `open_oven_i7` (oven dials) is a set-point stop on a detented rotor — the
  robot's own drive parks the indicator, and any single dial reaches every setting.
  Here NO drive can park the indicator: it is back-computed by the plant from what
  rests in the tray, so the only control channel is which objects you commit. The
  pick-place family (libero bowls/pan, `coke_task_i15`, `pick_and_lift_i16`,
  `pen_holder`-style fill tasks) transports a FIXED set of objects to regions —
  "transfer everything" always works; here transferring everything (7 units) FAILS
  for every sampled target, and the correct subset is different every episode.
  `open_oven_i6` reorients dice nonprehensilely; nothing here is reoriented. **No
  corpus task selects a variable-size subset of objects to satisfy a sampled physical
  quantity with both-direction failure.**
- The mechanism is physically honest and asserted in `__post_init__`: the band
  excludes the neighbouring integer loads, the consistency tolerance resolves one
  unit, the joint travel covers the deepest sampled load with margin, the loaded tray
  never touches the pedestal, both cube sizes fit an 80 mm parallel jaw, and the tray
  opening admits the biggest cube with 10 cm to spare.

## Solution outline (solve.py — the legitimacy certificate)

Teleport = TRANSPORT ONLY: each selected weight's pose is written once, to a free
hover 10 mm above the current tray floor (recomputed from readback — the tray is lower
after every drop), zero velocity. Everything load-bearing is contact dynamics: the
cube falls in, the spring plant sinks one tick per unit, rings once (ζ ≈ 0.7–1.3 by
design) and settles. The tray, marker and unused weights are never written; the probe
buffers stay zero (asserted at every phase boundary).

1. **Read (readback, not privilege)** — read the green tab's z off the mast, count
   ticks down from the white zero; asserted equal to the scene's sampled u*.
2. **Combo** — greedy big-first exact decomposition (see table above); the resting
   load only ever grows toward u*, so the printed score sequence is monotone by
   construction.
3. **Drop loop** — per weight: hover over an unoccupied tray quadrant (84 mm spacing >
   50 mm cube), release, hands off until the scene's own accounted streak confirms the
   new resting load; `SIM_GEN_SCORE` at each boundary (non-decreasing, asserted).
4. **Verify + persist** — settle until `success()`, then ≥ 3.5 simulated seconds
   hands-off before `SIM_GEN_SOLVE: SUCCESS`.

**Verified on the forge: seeds 0, 1, 2 (u* = 3, 6, 4) all SUCCESS, ~18 s wall each.**

## Rubric

score = 0.15·stage1 (latched: some weight ever rested in the tray, accounted) +
0.45·best_prog (latched: max over settled+accounted frames of 1 − |units_on − u*|/u*)
+ 0.40 iff `success()` now. All credit is gated by the **accounted** clause
(|deflection − resting_load·g/k| ≤ 7 mm) and a 10-substep settle streak: pressing the
tray into the band (in_band with no load) and holding a weight inside the tray (load
with no deflection) both earn ~0 — constructed and rejected in smoke. score == 1.0 iff
success; null policy ~0; a correct load knocked out afterwards leaves the latched
0.60. Demonstrated solve margins: settled deflections land ≤ 0.9 mm from the tick vs
the 6 mm band.

## Embodiment argument (Franka, parallel-jaw)

- Both cube sizes (40/50 mm) sit well inside the ~80 mm jaw; masses 120/240 g are
  trivial payload. Every transfer is a top-down pinch off open floor, a carry at
  ~0.35 m, and a release over the open 20 cm tray (rim only 3.2 cm — drop-in from
  above, nothing to thread).
- Workspace: weights spawn on a row at x ≈ 0.27–0.63, y ≈ 0.16; the tray centre is at
  (0.50, −0.20), floor ≈ 0.21 m up. With the base at the origin facing +x everything
  lies within 0.30–0.68 m reach at heights 0–0.35 m; the mast stands behind the tray
  (+x side), leaving the approach corridor from the robot side clear.
- Reading the goal is visual (count ticks from the white zero to the green tab), the
  same readback the solve performs; correcting an overshoot is the same pick, in
  reverse (the rim is below jaw reach-over height).
- Execution order: NONE required — any order of drops, and ANY exact decomposition of
  u*, succeeds (smoke check 16 accepts the small-first alternative combo).

## Checks (smoke.py — rejection battery, 18/18 PASS on forge, frames.npz recorded)

1. settle + no-NaN, tray resting at the zero tick, weights at rest; 2. randomization
readback across 6 seeds (u* draws [6,2,4,2,3,3], layouts all distinct); 3. green tab
parked at the sampled tick (readback err 0.0 mm); 4. null policy 2 s → score ~0;
5. seed-strategy negative (25 N pull + 2.5 N·m pry): tray rides its rail to the stop,
springs back, no credit; 6. press cheat: constant force parks the pointer EXACTLY in
the band — in_band true (sampled during the press) but unaccounted, streak 0, no
credit; 7. hover cheat: a statically held weight registers as load (on_tray TRUE — the
trap is armed) but is unaccounted → no credit; 8. under-load u*−1 resting+accounted →
rejected; 9. over-load u*+1 → rejected (two-sided failure); 10. correct total mass
settled on the FLOOR → ~0; 11. anti fly-through: all five dumped from 15 cm — the
pointer sweeps THROUGH the band (observed) but success never fires; 12. monotone
score climb along the loading sequence; 13. partial credit after the first weight, no
success; 14. exactness: exact load → success and score == 1.0; 15. knock-off: success
revoked, latched 0.60 remains; 16. equivalence: a different exact decomposition also
succeeds; 17. final no-NaN; 18. frames recorded (280 × 500 × 800).

Run (forge):
`python -u -m simgen_tasks.open_oven_i213.solve --headless [--seed N]`
`python -u -m simgen_tasks.open_oven_i213.smoke --headless`
