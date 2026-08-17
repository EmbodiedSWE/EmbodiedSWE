# put_groceries_in_cupboard_i258 — Balance-Shelf Cupboard

## Seed provenance

Seed task: `rlbench/put_groceries_in_cupboard` — pick grocery items off a table
and set them down inside a cupboard. The seed's load-bearing content is
pick-and-place onto a *static* shelf: any set-down inside the cupboard volume
succeeds, order and placement geometry are irrelevant.

## What changed, and why it is strategically different

The cupboard's only shelf is now a **free-pivoting beam balance**: a two-pocket
tray on a revolute joint (axis X, hard stops at ±12°) with a weighted keel bob
below the pivot that gives a small self-leveling restoring torque
(~1.24 N·m/sinθ). At reset a **heavy blue decoy can (~0.62 kg)** sits in one
pocket and pins the beam against a stop. Two **red grocery cans (~0.30 kg
each)** start on the ground in front of the cupboard.

Goal: the shelf must end **level, loaded with exactly the two red cans, one per
pocket, and the decoy off the beam** — everything settled.

Strategic difference from the seed and from every sibling examined:

- vs **seed** (`put_groceries_in_cupboard`): success there is a passive
  set-down into a static container. Here the support itself is a dynamic
  mechanism; a set-down only counts if the *system's equilibrium* comes out
  right. The load-bearing decisions are eviction and symmetric loading, not
  transport.
- vs **i169** (carousel cupboard): no crank/indexing mechanism; the mechanism
  here is a passive balance the solver must reason about statically, not drive.
- vs **i222** (FIFO gravity rack): no queue/ordering-by-container; ordering
  here is *forced by physics*, not by geometry of a lane.
- vs **i329** (spring pusher lane): no spring press or force-threshold
  insertion; the only "actuator" is gravity acting on what the solver chooses
  to place where.

Physics-forced ordering (no rubric timestamps anywhere):

1. While the decoy is aboard, the beam **cannot** be level (decoy drive torque
   ~2.4× the keel's restoring capacity) — so eviction must come first.
2. One red can alone pins the beam at a stop (drive 0.42 vs 0.26 N·m
   restoring) — so being level with cans aboard **requires** the symmetric
   1-per-pocket load. The torque budget is verified numerically in
   `SceneCfg.__post_init__` asserts (red pins ≥1.3×, decoy out-torques red
   head-to-head ≥1.1×, worst-case placement-slack tilt < half the level
   tolerance).

## Teleport-solution phases (solve.py)

Teleports are transport-only (free-air pickup after force-based extraction,
free-air release); every load-bearing interaction is contact dynamics under
applied forces/torques.

- **P0 — settle & audit** (300 steps): readback asserts — masses (cupboard
  >30 kg, beam 0.9–1.6 kg, reds 0.25–0.36 kg, decoy 0.50–0.75 kg), beam pinned
  at the decoy-side stop, decoy seated, score ≤ 0.01.
- **P1 — force-evict the decoy**: PD force-carry (gravity feedforward +
  righting torque, clamped) lifts the decoy out of its pocket through contact
  with the beam until it clears the fences; then free-air transport to the
  ground beside the cupboard; hands-off wait until the keel self-levels the
  empty beam (`unloaded` latch, |tilt| < 2°). Score ≥ 0.15.
- **P2 — load red A**: free-air release 35 mm above the near pocket of the
  *level* beam; the can seats and its weight pins the beam at that side's
  stop. Score ≥ 0.40.
- **P3 — load red B**: free-air release 8 mm above the *raised* pocket,
  aligned to the beam's current (tilted) quaternion; the counterweight brings
  the beam back level; both-seated + level + settled ⇒ live success.
  Score ≥ 0.999.
- **P4 — persistence**: ≥3.3 sim-seconds hands-off, success must hold every
  probe, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at each phase boundary and never decreases.

## Embodiment argument (single Franka, parallel-jaw)

- **Red cans** (r=32 mm ⇒ 64 mm diameter, h=90 mm): top grasp across the
  diameter, 64 mm < 80 mm jaw span. Ground pickup at ~0.45 m from the cupboard
  face; release 8–35 mm above pocket rims at ~0.19–0.25 m height — well inside
  the Franka workspace.
- **Blue decoy** (d=68 mm, h=115 mm): top grasp, 68 mm < 80 mm. The cupboard
  front is fully open (posts 0.18 m apart flank the opening, roof at 0.44 m),
  so a straight-in approach reaches the pocket; the solve's force-carry uses
  bounded forces (≤12 N) and small righting torques — comfortably within
  Franka payload (3 kg) and wrench limits.
- **One plausible base pose**: base at the world origin facing +x; the
  cupboard front face is ~0.35–0.55 m away (cupboard at x≈0.5, yaw 180°±15°),
  cans on the ground 0.40–0.52 m from the cupboard center. All grasp and
  release poses are 0.25–0.65 m from the base at heights 0.05–0.30 m.

## Execution-order declaration

The rubric contains **no timestamps and no phase counters**. Ordering is
enforced purely by mechanics: (a) `level()` is impossible while the decoy is
aboard (torque budget), so the 0.15 `unloaded` latch and everything after it
require eviction first; (b) `level()` with reds aboard is impossible unless
both pockets are loaded, so the final state requires the full symmetric load.
Latches (0.15 unloaded / 0.25 first-seated / 0.30 both-matched, cap 0.70) each
require 3-step persistence with slow-motion gates; score 1.0 iff the *live*
success predicate holds: matched ∧ level ∧ decoy-off-beam ∧ settled ∧ finite.

## Smoke rubric battery (11 checks, rejection-only)

1. Settle / no-NaN: seed 100, 300 steps — beam pinned at the decoy side,
   decoy seated, reds upright on the ground, score ≤ 0.01.
2. Randomization by readback: seeds 101 vs 202 — cupboard yaw differs >1.5°,
   cupboard xy >3 mm, both can spots >30 mm apart.
3. Decoy-side shuffle: seeds 300–309 produce both sides; beam-local seat
   readback and pinned-tilt sign agree with the sampled side.
4. Null policy: 240 steps hands-off — still pinned, score ≤ 0.01.
5. Seed strategy fails: set both reds down inside the cupboard the
   RLBench way (floor + roof) — nothing seated, score ≤ 0.01.
6. Wrong object / decoy still aboard: seating a red opposite the decoy leaves
   the beam pinned — level False, score ≤ 0.25 + ε.
7. Single can: after eviction and leveling, one red pins the beam — score
   ≤ 0.40 + ε (unloaded + first latches only).
8. Same-side stack: both reds stacked in one pocket — matched False,
   score ≤ 0.40 + ε.
9. Toppled bridger: a red lying on its side across the fences does not count
   as seated — matched False, score ≤ 0.40 + ε.
10. Latch regression + live veto: with both reds matched but the decoy dropped
    back aboard, latched score stays 0.70 but success is False; removing a red
    afterwards cannot lower the latched 0.70 (no un-earning) yet success stays
    False.
11. Video: frames.npz written with >10 frames.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` on success.

## Check list

- [x] `describe()` sufficient for a solver: geometry, masses, torque logic,
  goal, and tolerances all stated.
- [x] `instruction()` short VLA form (<200 tokens).
- [x] Randomization is real and verified by readback (smoke #2, #3).
- [x] Teleport = transport only; all load-bearing interaction via forces.
- [x] Score monotone during solve; `SIM_GEN_SCORE` at phase boundaries.
- [x] ≥3 sim-seconds hands-off persistence after first success.
- [x] Watchdog + hard exit in both solve and smoke.
- [x] No rubric timestamps; ordering physics-forced.
- [x] Solve passes on ≥2 seeds (forge).
- [x] Smoke rejection battery incl. seed-strategy failure, near-misses,
  wrong object, null policy, settle/no-NaN, video.
