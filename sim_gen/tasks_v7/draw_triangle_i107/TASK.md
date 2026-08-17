# draw_triangle_i107 — `beam_balance`

A two-pan beam balance stands on the bench with an unknown slate stone in one
tray. Weigh it: load the labeled brass unit weights into the trays until the
beam floats level, sustained-still and hands-off — the exact combination and
nothing else survives the physics.

- Scene id: `beam_balance` · Env id: `simgen.beam_balance` · Robot: `null`
- Files: `scene.py` (procedural geometry only), `solve.py`, `smoke.py`, this file.

## Seed provenance and why this is strategically different

Seed: **maniskill/draw_triangle** — the Franka holds a rigid stylus and traces a
triangle outline on a passive canvas; one long continuous guarded tool-tip sweep
along a PRESCRIBED curve, judged on the coverage of the trace the arm itself
generated. Nothing about the goal is hidden and the scene never pushes back.

This task inverts that on every axis that matters for a solver:

| | seed (draw_triangle) | this task (beam_balance) |
|---|---|---|
| goal knowledge | prescribed curve, fully known | **hidden scalar** — the stone's mass (1–7 units) is unreadable from its fixed 46 mm size |
| scene | passive canvas | **articulated mechanism** — keeled beam on a revolute pivot with hard stops, two pendant tray pans on their own hinges |
| robot's product | a trace of its own tip positions | **a physical equilibrium** the moments must genuinely cancel to produce |
| plan structure | waypoint tracking | **closed-loop measurement**: place a weight, let the beam settle, read which way it tips, revise — greedy binary search over discrete masses |
| error signal | path deviation | the mechanism itself: a single-unit error drives the equilibrium tilt PAST the ±12° stops (asserted with margin), so every wrong subset rests pinned on a stop |

Against the tasks_v7 corpus (surveyed before design: aperture posting, frame
construction, containment/stacking/pouring transport, articulated open/close,
chute routing, pendulum arrest...): no existing task hides a scalar the solver
must MEASURE through the scene's static response, and none judges a balanced
equilibrium of an articulated mechanism.

## Success criteria (all judged live, simultaneously)

- **A. LEVEL** — beam axis within `level_tol_deg = 4°` of horizontal.
- **B. STILL (sustained)** — beam angular speed < 0.10 rad/s AND every
  pan/stone/weight linear speed < 0.09 m/s for `still_steps = 60` consecutive
  sim steps (0.5 s). A beam swinging *through* level is fast at level; the
  consecutive-step latch rejects fly-through readings (verified in smoke
  check 6: a kicked beam crossing the ±4° window never opens the gate).
- **C. STONE HOME** — the active stone still seated in the tray it spawned in
  (tray-frame test). Rejects "remove the stone and let the empty balance level
  itself".
- **D. CLEAN** — every brass weight either seated in a tray or ≥
  `exclusion_r = 0.42 m` from the stand; all six look-alike spare stones ≥
  `exclusion_r` too. Rejects weights parked on the beam/stand and spare-stone
  swaps.
- **E. STAND HOME** — the stand within 4 cm / 10° of its spawn pose. Rejects
  tipping or dragging the scale to fake level.

Given C/D/E and the geometry (the tray floors hang 0.238 m above the bench —
higher than a pile of every movable in the scene plus margin, asserted), the
only hands-off way to hold A+B is a genuine unit-exact moment balance. The
pendant trays stay level as the beam tilts, so a load's moment is set by the
hang point, not by where in the tray it sits — placement cannot fake or spoil
a balance.

**Score (latched, non-decreasing):** 0.15 once at least one brass weight has
ever been seated (settled in a tray, stone home, stand home) + 0.10 once two
have ever been seated simultaneously, capped at 0.25; exactly 1.0 iff
`success()` holds now. Null policy ≈ 0: the stone parks the beam on a stop and
no weight is ever seated (smoke check 5).

## Per-seed randomization (readback-verifiable via `describe()`)

Stone mass k ∈ {1..7} units (realized by seven identical-looking pre-spawned
stones of masses 1u..7u — the sampled one goes in the tray, the six spares wait
in a floor depot outside the exclusion radius; per-episode mass randomization
without runtime mass writes), stone side (PanP/PanN), stand xy (±3 cm) + yaw
(±10°) jitter, weight slot permutation + xy jitter (±2 cm) + free yaw. All
discrete draws use `torch.rand` comparisons (first-`randint` degeneracy on this
stack).

## Teleport-solution outline (`solve.py`)

Teleportation is transport only; every load-bearing interaction is contact
dynamics, and the solve never reads the hidden mass to pick the answer (the
oracle indices are read only to cross-check the measurement in the log):

1. Settle; read the tilt sign — the beam is pinned on a stop, tipping toward
   the stone. That observation picks the counter tray.
2. Greedy binary weighing, largest first (4u, 2u, 1u): teleport the weight to a
   hover 12 mm above its spot in the LIVE pan frame (zero velocity,
   pan-aligned) and let it drop the last millimetres; settle; if the beam now
   tips *toward* the brass side past 7°, the trial is OVER — teleport the
   weight back to an empty bench parking spot; otherwise keep it seated.
3. After the three trials the seated total equals the stone's units for every
   k ∈ 1..7 (classic balance-scale binary search). The exact load floats the
   beam level; wait for `success()`.
4. Hold ≥ 3.3 simulated seconds fully hands-off, printing `SIM_GEN_SCORE` at
   each phase boundary (non-decreasing, latched) and `SIM_GEN_SOLVE: SUCCESS`
   only if success still holds at the end.

## Franka embodiment argument

Every manipulation in the plan is Franka-feasible:

- The brass weights are square blocks 22/31/44 mm wide × 35 mm tall, 60–240 g —
  all under the 80 mm parallel-jaw opening (asserted in `__post_init__`) and
  far below payload; top-down pinch grasps on free-standing blocks staged on
  the open bench front.
- Placing into a tray is a coarse drop: the tray inner half-width (47.5 mm)
  leaves ±few-mm slack around even the 44 mm weight (asserted), the pendant
  tray stays level while loaded, and the fork posts leave a clear vertical
  approach corridor above each tray (clearance asserted).
- Reading the balance needs vision only: a 0.60 m beam whose tilt saturates at
  ±12° — a coarse, high-contrast angular signal.
- Everything actionable sits on a 1.10 × 1.00 m bench (top at 0.10 m); the
  tray floors hang at world z ≈ 0.40 m and the staging slots at y ≈ −0.44 —
  inside Franka's 0.85 m reach from a front-edge base.

**One plausible base pose:** base at the bench front edge, env-local
`(0.0, −0.62, 0.10)`, facing +y — the three staging slots (±0.20/0.0, ≈−0.44),
both trays (±0.26 from the stand centre at (0.0, 0.08)) and the parking
corners all fall in a 0.2–0.75 m fan in front of the base.

## Execution order

**No execution order is required.** The weights may be seated in any order and
in either tray (counter-loading and differential loading both balance — the
arithmetic, not the order or side, is judged). `solve.py` happens to run
largest-first single-tray greedy weighing; the rubric never inspects order.

## Verification

- `smoke.py`: **14 checks**, rejection-only battery — (1) premise: finite
  state, beam pinned past 9° with the tilt sign pointing at the stone side
  (readback), stone home, weights clean, stand home; (2) reset: score ≈ 0, not
  level, no success; (3) stone mass/side randomization across 10 seeds by
  readback (≥3 distinct k, both sides); (4) stand/slot/yaw randomization by
  readback + invariants; (5) null policy: still pinned, score 0 after 3 s;
  (6) fly-through: a kicked beam sweeps through the level window but the
  sustained-still gate never opens, no success, falls back pinned; (7)
  under-load (k−1 seated) → still pinned toward the stone; (8) over-load (k+1)
  → pinned on the opposite stop; (9) exact load but a leftover weight parked
  inside the exclusion radius → level+still+home yet success refused (clause
  D), score capped at 0.25; (10) stone removed to the floor → empty balance
  levels but success refused (clause C); (11) whole balance dragged 10 cm with
  exact load → success refused (clause E); (12) score latch: seating credit
  persists after the weight is parked again, never 1.0 without success; (13)
  success never observed during the battery; (14) no NaN/Inf and frames.npz
  recorded. Prints `SIM_GEN_SMOKE: ALL PASS 14/14`.
- `solve.py` on the forge, seeds 0/1/2: `SIM_GEN_SOLVE: SUCCESS` on all three
  (k = 3, 7, 3; both stone sides exercised), score trajectory
  0 → 0.15 → 0.25 → 1.0 (k-dependent middle), non-decreasing, ≥ 3.3 s
  hands-off persistence.

## Design notes (physics traps avoided)

- Keel CoM authored explicitly (`CreateCenterOfMassAttr`) below the pivot —
  MassAPI-only mass on the compound beam would leave the CoM AT the pivot and
  the balance would be neutral instead of restoring.
- The stand is a heavy (26 kg) **dynamic** body: a joint anchored to a
  teleported kinematic body0 stays world-fixed at spawn pose on this stack.
- Single-unit-error rejection is *asserted*, not hoped for: the moment of one
  unit at the hang point vs. the keel's restoring moment at the stop angle
  carries a 1.25× margin, so every wrong subset rests on a stop at 12° while
  the exact one floats within 4°.
- A loaded pendant pan + cargo sits in a persistent contact-driven velocity
  limit cycle (~0.05–0.06 m/s, forge-measured, undamped by design damping):
  `obj_still_speed = 0.09` sits above that artifact floor while fly-through
  remains rejected by the beam angular gate alone (crossing the ±4° window
  within the 0.5 s latch requires ≥ 0.28 rad/s ≫ the 0.10 gate).
- Stillness is a consecutive-step counter latched in `post_step`, never an
  instantaneous velocity test (turning-point trap).
- All discrete randomization uses `torch.rand` comparisons — the first
  `torch.randint` after `manual_seed` is near-constant across seeds here.
