# oven_dials (open_oven_i7) — set the oven's two control dials to their indicated settings

**Env name:** `simgen.oven_dials` (scene `oven_dials`, robot `null`; both `solve.py`
and `smoke.py` build the same scene-level env).
**Tier:** medium — 2 goal-conditioned stages (one detented dial per stage).
**Execution order:** NOT required — either knob first; declared in `describe()`.

## Seed provenance

Seed: `rlbench/open_oven`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/open_oven.py`) — a Franka pulls the
hinged oven door open by its handle (USD oven asset + recorded trajectory, binary
fixed goal, no checker).

## What changed, and why it is strategically different

The seed's plan skeleton is: *approach the one handle → grasp → pull the hinged panel
through a large free arc until "open"*. A gross-motion, fixed-goal, binary pull on the
oven's single moving part.

This task keeps the oven's control surface and **removes the door entirely** (the deck
is kinematic — nothing on this oven opens). What remains is the oven's *other*
affordance: two spring-**detented control knobs** on vertical revolute spindles rising
from the top console. Each episode samples, per knob, a random start setting and a
random target setting (≥ 2 detents apart); the target is shown physically by an amber
lamp parked at that setting's tick mark. The solver must:

1. **read the sampled goal from the scene** (which tick glows, per knob) — the seed has
   no goal-conditioning at all;
2. perform **precision rotary positioning**: turn each knob about its vertical spindle
   and *stop* inside an 8° tolerance backed by a ±20° detent capture basin — a
   stop-at-target skill, where the seed's pull runs to a hard limit;
3. do it **twice, on independently-goaled dials** (order free).

A different PLAN, not different parameters: no grasp-and-pull arc, no door, no binary
"open" predicate; instead goal reading, spindle-axis turning, tolerance stopping. The
seed's own strategy is **expressible and measured as a negative control**: a sustained
25 N pull plus a 2.5 N·m hinge-style prying torque on a knob (exactly the wrench that
opens the seed's door) moves nothing and scores ~0 (smoke check 5).

## Scene / physics honesty

Procedural only: kinematic deck (Cuboid), knobs (Cylinder) on authored USD Z-axis
revolute joints (limits ±128°), symmetric grip bars (fixed joints; CoM on the spindle),
visual-only tick/cap decorations, amber lamp bodies (kinematic) teleported to the
target tick each reset. Knob "feel" (viscous friction 0.05 N·m·s/rad + nearest-setting
detent spring 0.25 N·m/rad, overdamped ζ≈2.4; capture basin measured 15° in / 25° out)
is applied in `post_step` — the scene owns the knobs' external-wrench slot; the
`knob_drive`/`knob_force`/`knob_torque_ext` buffers are written only by solve.py's
fingertip-scale drive and smoke's rubric probes. success()/score() judge only the
physical pointer state: yaw read back from the knob root quat, settled spindle rate.

## Teleport-solution outline (solve.py — the task's legitimacy certificate)

Scene-level env, robot `null`. **This task has no transport component, so the teleport
budget goes unused: solve.py never writes a root pose of any task object.** The
load-bearing interaction — turning each detented knob and stopping on its sampled
target — runs entirely through the live dynamics, via the scene's `knob_drive` spindle
torque clamped to **TAU_MAX = 0.16 N·m ≈ 3.8 N tangential at the grip bar's 42.5 mm
half-length** — the same authority the real Franka pinch-turn measured on this scene
(3–5 N of OSC servo force). The drive exceeds the detent's 0.087 N·m peak resist (so
basins can be crossed at all) but by < 2×, so the detent plant, not the drive, owns
the motion. Nothing is pinned, no velocity is written, no rubric state is touched.

- **PHASE 0** — reset(seed), settle 90 steps. `SIM_GEN_SCORE ~0.000`.
- **PHASE 1 (knob 0)** — clamped PD (KP·err − KD·ω, |τ| ≤ 0.16 N·m) slews the knob at
  ~30°/s toward the target, crossing intermediate detents through their springs; the
  drive is **zeroed at 9° from target** (inside the measured 15° capture basin,
  asserted) and the **detent spring alone seats the pointer** hands-off — the arm's
  release-in-basin endgame. Seating confirmed by the scene's own `at_target()`.
  `SIM_GEN_SCORE ~0.500`.
- **PHASE 2 (knob 1)** — same skill on the other spindle; success() = both dials
  settled on target. `SIM_GEN_SCORE 1.000`.
- **PHASE 3** — 420 physics steps (3.5 simulated seconds) with all drive buffers zero
  (asserted); success() is live state (a dial knocked off would revert it), so
  persistence rejects fly-through outcomes. Only then `SIM_GEN_SOLVE: SUCCESS`.
  Watchdog Timer + `os._exit` guard teardown.

The printed `SIM_GEN_SCORE` sequence is monotone (asserted in code). Verified on the
forge on seeds 0 and 1 (distinct sampled instances, both turn directions).

## Embodiment argument (single Franka + parallel jaw, OSC)

This exact plan was previously EXECUTED end-to-end by a real Franka arm on this exact
scene (the earlier arm-driven run of the same task package): **base at (0, 0, 0),
identity rotation (facing +x)**; deck near edge 0.16 m ahead, knobs 0.48 m ahead at
~0.22 m height — inside the 0.45–0.71 m comfortable top-down envelope.
`SIM_GEN_SOLVE: SUCCESS` on seeds 0–3, including a 240° traverse in two chunks and
both turn directions. Per-object contact strategy:

- **Grip bars (85 × 16 × 32 mm brass bars, the only objects the arm must move)**:
  straddle one end of the bar with the open jaw perpendicular to it, pinch (12 mm
  command on the 16 mm bar, gripping the bar's lower half with a −2 mm down-bias so
  the wedge-tipped pads don't extrude upward), then drag the pinched end along the
  spindle-centred arc with wrist-roll tracking, ≤ 100° per pinch, with a 14–26° servo
  lead (the stalled position error is the drive force — measured 3–5 N, which is what
  TAU_MAX in solve.py is calibrated to). Choose the grip END whose swept arc stays
  farthest from the base direction (a base-directed drag leaks the force into a
  parasitic vertical push). Release inside the ±20° detent basin; the detent seats
  the pointer exactly.
- **Clearances**: bar tops sit ~0.24 m above ground on a bare kinematic deck — no
  overhangs, no apertures; the 8° stop tolerance is backed by the detent's snap
  (anything released within 15° self-seats), so required precision is a detent basin,
  not a control-noise lottery.
- **Lamps, ticks, deck**: never touched.

## Randomization (per episode)

Per knob: start setting index, target setting index (independent, ≥ 2 detents apart →
authored error ≥ 80°), ±4° start jitter, and the amber lamp physically re-parks at the
target tick (verified by READBACK in smoke checks 2–3). The deck/knob mechanism stays
at its authored pose (jointed-pair teleport is unreliable on this stack); dial ANGLES
re-pose about the unchanged spindle — the proven follower-only teleport.

## Rubric (score in [0, 1], partial progress latched)

- per knob: latched best progress `1 − err/err0` (forced to 1.0 once the pointer ever
  enters the target zone) × 0.35, plus 0.15 if *currently* resting on target
  (≤ 8° AND spindle rate < 0.15 rad/s);
- `score == 1.0` **iff** `success()` (both dials settled on their targets);
- null policy ≈ 0 (start jitter ±4° vs authored err0 ≥ 80°); seed strategy ≈ 0;
- transient achievements latch: a dial knocked off after success leaves 0.85; the
  `SIM_GEN_SCORE` prints along the solve trajectory are monotone (asserted in code).

## Check list (smoke.py — 14 checks; rejection tests only — solve.py is the acceptance
evidence; teleported dial states are rubric instrumentation, not a solution)

1. settle: clean reset (finite, pointers on start settings, score ~0)
2. randomization is real (start/target draws differ across seeds, by readback)
3. goal indicator: lamps sit at the sampled target marks (readback)
4. null policy: score < 0.05 and no success
5. **negative (SEED strategy)**: pulling/prying a knob opens nothing, score ~0
6. negative (near miss): adjacent setting (40° off) rejected by the 8° stop tolerance
7. negative (swap): dials on each other's targets rejected — goal assignment matters
8. rubric monotonicity: latched score never decreases along the approach
9. partial credit: knob 0 on target → score ~0.5, no success
10. exactness: both dials set → success() and score == 1.0
11. achievement latch: knock-off revokes success, score 0.85
12. calibration: offsets ≤ 15° captured back to the target detent
13. calibration: offsets ≥ 25° fall to the neighbour detent (rejected)
14. video frames recorded (frames.npz in CWD)

Out-of-order end state: N/A — no execution order is declared (order-free dials; the
"swap" check 7 covers the goal-assignment failure mode instead).

Plus the solve gate: `SIM_GEN_SOLVE: SUCCESS` on ≥ 2 seeds with non-decreasing
`SIM_GEN_SCORE` prints and success persisting under 3.5 s of extra hands-off
simulation.
