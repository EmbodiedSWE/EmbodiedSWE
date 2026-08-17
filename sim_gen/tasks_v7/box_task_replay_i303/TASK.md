# carton_flip_pack (`box_task_replay_i303`)

The cardboard carton lies OVERTURNED (mouth-down) on the floor with the two products
standing loose beside it. Right the carton by a nonprehensile 180-degree roll — two
chained edge-pivots driven by a fingertip-scale push near its top edge, above the
tipping threshold — then drop the ORANGE can and the GREEN candle in through the
mouth and leave everything settled.

- Env id: `simgen.carton_flip_pack` (NullRobot scene-level env)
- Files: `scene.py` (scene + rubric + roll plant), `solve.py` (teleport solution
  certificate), `smoke.py` (15-check rejection battery, recorded), this file.

## Seed provenance

| | seed | this task |
|---|---|---|
| source | `box_task/box_task_replay` (RoboVerse `roboverse_pack/tasks/box_task/box_task_replay.py`) | new scene, procedural geometry only |
| setup | open cardboard box READY on a table, soda can + scented candle beside it, bimanual openarm robot | the carton itself is the obstacle: it starts MOUTH-DOWN on the floor (nothing can enter it), items on the floor to its ridge sides |
| plan | replayed bimanual prehensile pick-and-place of both items into the waiting box | first a large-body NONPREHENSILE reorientation (roll the carton 180 deg over two floor edges, tip-over dynamics + momentum management, the carton walks ~24 cm), THEN the drop-in loading |
| goal | items in the box (box passive scenery throughout) | carton upright + settled AND both items contained below its rim; entering items before righting is physically impossible |

## Why strategically different

- **vs the seed**: the seed's container is passive, pre-oriented scenery and every
  action is a prehensile item transport (replayed, bimanual). Here the container is
  the *mechanism*: the load-bearing skill is reorienting a large, ungraspable-by-form
  hollow body through two edge-pivots — a stored-geometry problem (tipping thresholds,
  balance angles, landing impacts, rock-to-rest) with the item transport reduced to a
  terminal drop-in. The seed's plan literally executed on this scene (put items
  "onto/into the box" as found) scores ~0 (smoke check 5).
- **vs the corpus**: no examined task rights an overturned container by rolling it
  over floor edges. `pull_cube_tool_i186` releases stored energy via falsework
  extraction; `close_box`-style tasks articulate lids on joints; stacking/nesting
  tasks never reorient the receptacle. The two-pivot roll with a torque threshold and
  a walk of the container across the floor is a distinct plan skeleton.

## Mechanics

Plain rigid bodies, no joints. The carton is a dynamic COMPOUND body (base plate +
4 walls; custom spawner authors mass 0.30 kg, CoM 11.5 mm below centre, hollow-box
diagonal inertia, mu 0.85/0.75, restitution 0 — custom spawners apply no cfg
schemas). A `post_step` plant consumes a `roll_tau` buffer: torque about the carton's
BODY-X (ridge) axis, clamped to `tau_max` = 0.45 N m (~a 4.5 N fingertip at the
0.10 m top edge) — body-x IS the roll axis at every roll angle, so the command is
immune to wrench-frame convention quirks. Tipping thresholds are real: pivot 1
(mouth-down) needs 0.206 N m, pivot 2 (on side) 0.113 N m; 88% of threshold does
nothing (smoke check 7). Landing on the flat base kills essentially all rotational
energy (wide-block rocking ratio ~0.005), so the roll cannot overshoot past upright.
`enable_external_forces_every_iteration` is set — without it the wrench is integrated
only on the first solver iteration and the pivot plant degenerates into a
rock-and-scoot limit cycle (first forge iteration).

## Rubric (latched, anchored in solve.py)

- `righted` (latch, 30-substep streak): carton upright (mouth-up within 10 deg) at
  upright rest height, settled.
- `packed_can` / `packed_candle` (latch, 30-substep streak): item inside the upright
  carton in the carton BODY frame (|x| < 7 cm, |y| < 5 cm, centre in
  z in (-5.5, +2.8) cm — below the rim by > 2.2 cm: containment below the aperture),
  carton + item settled.
- `success()`: CURRENT state — carton upright + settled, BOTH items contained,
  items settled.
- `score()` = 1.0 iff success, else 0.30·righted + 0.25·packed_can
  + 0.25·packed_candle. Null ~0; a knock-out after success falls back to 0.80.

**Execution order is REQUIRED and physically enforced**: while the carton is
mouth-down its mouth is sealed against the floor — an item can only be ON TOP of it
(rejected: not contained, carton not upright) or TRAPPED UNDER it in the cavity
(rejected: upright gate). Every containment path runs through the righting
(smoke checks 5, 6).

## Randomization (verified by READBACK in smoke)

Carton xy ± 4 cm + free yaw; each item's polar offset from the carton (radius
0.20–0.27 m, angle within ±45 deg of the ±ridge directions — both roll corridors
stay clear by construction) and which item is on which side.

## Solution outline (solve.py, teleport = transport only)

0. reset(seed), settle — assert mouth-down + items standing; score ~0.
1. **RIGHT** (contact dynamics): pick the roll direction whose corridor is farthest
   from both items; smooth one-sided rate servo on the ridge-axis torque
   (τ = clamp(0.20 + 0.25·(1.0 − s·ω), 0, 0.42) N m through `roll_tau`; gravity is
   the brake), cut past the second balance point (up_z ≥ 0.70); the carton falls
   onto its base and rocks to rest → score 0.30.
2. **PACK can**: teleport the can from the open floor to 2 cm above the rim at
   body-x +4 cm (transport across free space only — the in-reach pick a Franka does
   with a pinch grasp), release; it falls in through the mouth and settles → 0.55.
3. **PACK candle**: same at body-x −4 cm → success, score 1.00.
4. **PERSIST**: ≥ 3.5 s pure simulation, success at every poll →
   `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` at every phase boundary, non-decreasing (0 → 0.30 → 0.55 → 1.0 → 1.0).
Passes on seeds 0 and 1 (opposite roll directions chosen).

## Franka embodiment (base at (0.42, 0), facing the carton)

- **Carton** (18 × 14 × 10 cm, 0.30 kg) at (0,0) ± 4 cm: 0.36–0.48 m from the base —
  well inside the ~0.75 m comfortable envelope. To roll it: push down-and-forward
  near the top edge of the upturned base (a 10 cm-high box — fingertip or knuckle
  contact, ~2–4.5 N, which is what `tau_max` encodes); repeat over the second edge.
  Either roll direction works — the robot picks the corridor away from the items,
  exactly as solve.py does. After the ~24 cm walk the carton is at most ~0.70 m away.
- **Items** (can Ø5.2 × 8.5 cm, 0.15 kg; candle Ø5.6 × 5 cm, 0.12 kg): standing
  upright on open floor 0.20–0.27 m from the carton — at most ~0.73 m from the base;
  top-down pinch grasps on free-standing cylinders, standard.
- **Loading**: release each item a couple of cm above the 16.4 × 12.4 cm open mouth
  (rim at 10 cm height) — drop offsets ±4 cm give ≥ 14 mm wall clearance and a 26 mm
  inter-item gap; the can (8.5 cm) fits standing under the rim (interior depth
  9.2 cm).
- Nothing requires reaching past ~0.75 m, more than ~4.5 N, bimanual coordination,
  or entering a closed cavity.

## Checks

- **solve.py**: 4 phase gates (initial mouth-down + items standing; roll past the
  second balance point + upright rest + `righted` latch ≥ 0.30; each pack latched,
  success + score exactly 1.0; persistence ≥ 3.5 s with success at every poll),
  non-decreasing `SIM_GEN_SCORE` ladder. Passes on seeds 0 and 1.
- **smoke.py**: 15 named checks — settle, randomization readback (5 seeds, both item
  sides seen), reach + roll-corridor clearance, null-policy stability, seed-strategy
  negative (items on top of the overturned carton), trapped-under-the-carton
  negative, sub-threshold torque negative (88% of tipping moment, 1.5 s), oracle roll
  (same buffer; travel bounds prove pivoting, not scooting), score monotonicity,
  leaning-outside near-miss, stacked-above-the-aperture near-miss, exactness
  (ladder 0 → 0.30 → 0.55 → 1.00, score == 1.0), 2 s persistence, knock-out latch
  (falls to 0.80 exactly), frames recorded (`frames.npz` in CWD). Verdict:
  `SIM_GEN_SMOKE: ALL PASS 15/15`.
