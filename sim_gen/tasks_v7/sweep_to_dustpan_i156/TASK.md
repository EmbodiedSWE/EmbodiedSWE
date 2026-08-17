# sweep_to_dustpan_i156 — LetterboxBinScene (`simgen.letterbox_bin`)

Post every RED waste cube through a gravity-closed ONE-WAY LETTERBOX FLAP into a
sealed collection bin, and leave the two BLUE keep cubes outside. The bin's only
opening is an elevated slot (80 × 56 mm, sill 120 mm up) in its front face, covered
from inside by a yellow flap hinged along the slot's top edge: press a cube
horizontally through the slot, the flap swings inward, the cube tips over the sill
and falls in, and the flap falls shut behind it. Entry is irreversible (the sill
stands 4 cube-half-heights above the bin floor and the flap's closed hang is its
joint's hard upper limit), so posting a blue cube permanently fails the episode.
The red present-count is sampled per episode (1–3): the agent must count what it
sees, select by color — the cubes are identical except color — and post each piece
one at a time.

## Seed provenance

- **Seed task**: `rlbench/sweep_to_dustpan` (RoboVerse
  `roboverse_pack/tasks/rlbench/sweep_to_dustpan.py`) — "sweep dirt to dustpan":
  grasp a broom (USD asset), sweep five identical 10 mm dirt cubes across the open
  table into a wide, ground-level dustpan mouth. Robot=franka, trajectory-driven.
  One tool grasp, aggregate (undifferentiated) debris, a receptacle whose mouth is
  at ground level and approachable from anywhere, no selectivity, nothing
  irreversible.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| receptacle | a wide dustpan MOUTH at ground level, open from the whole half-plane | a SEALED BIN: solid roof, walls, blank lower face; the only entry is an elevated letterbox slot behind a gravity-closed ONE-WAY flap (spawn-authored revolute joint, limits [-85°, 0°], closed hang = upper limit) |
| how debris enters | swept along the ground into the mouth | staged on a small ledge at sill height, then PRESSED horizontally through the flap; the cube tips over the sill and falls in — per-piece posting, not aggregate transport |
| the tool | a graspable broom is the point of the task | no tool at all; the mechanism (flap) is part of the receptacle and is operated by the pushed cube itself |
| selectivity | all five dirt cubes identical, all wanted | 30 mm cubes identical EXCEPT color: 1–3 red wanted (count sampled per episode, readback-verified), 2 blue must stay out — posting a blue is IRREVERSIBLE failure (latched cap) |
| reversibility | dirt can be swept out and back freely | one-way: sill 120 mm above the floor, flap cannot swing outward (joint stop); smoke shoves an inside cube at the slot at 3× weight — it stays in |
| judging | scripted trajectory, no checker | live geometric success (every present red settled BELOW the sill inside, no blue ever in, flap hanging closed and still) + latched credit (approach 0.15, insert 0.45·fraction, contamination cap 0.30, non-success cap 0.60) |
| assets | broom + dustpan USDs | 100 % procedural (kinematic slotted bin + ledge compound, dynamic flap with authored MassAPI, primitive cubes) |

Code shares nothing with the seed: `@SCENES.register` BaseScene, two custom compound
spawners, per-env spawn-authored hinge (`_author_hinge`), bin-frame predicates,
latches in `post_step`, `register_env(..., robot="null")`.

## Why strategically different

The seed's skill is *grasp a long tool and sweep undifferentiated debris across an
open plane into a wide ground-level mouth* — one grasp, planar pushing, zero
selectivity, target approachable from anywhere, every mistake recoverable. Here
that plan earns nothing: smoke check 5 CONSTRUCTS the seed's strategy (a red cube
dragged quasi-statically along the ground straight at the slot at 3× its weight) —
it travels 415 mm freely and ends at ground level (z 15 mm) pressed against the
blank lower face, 105 mm below the sill, score 0.000, approach not even latched.
What the solver must bring instead: (1) **per-piece, elevated insertion** — each
cube must be carried to sill height and pushed through an 80 mm aperture (the seed
never leaves the ground plane); (2) **mechanism interaction** — the entry is
guarded by a hinged one-way flap that must be pressed open THROUGH the object and
falls shut behind it (the seed has no articulated element at all); (3) **identity
selection under count uncertainty** — red vs blue is the only difference between
otherwise identical cubes and the red count changes per episode; (4)
**irreversibility management** — a blue posted through the flap can never be
retrieved (one-way physics + latched cap + latched success clause), so a single
selection error is terminal, where the seed's every state is recoverable.

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.5 s; layout readback (present pattern, k, flap angle, all cube
   positions); asserts: finite, flap hangs closed < 3°, nothing inside, baseline
   score 0.0, no success.
2. **Per present red cube** (teleport = TRANSPORT ONLY; entry is contact physics):
   - **stage**: teleport to a hover 20 mm above the outside LEDGE, drop, verify it
     rests on the runway (≤ 4 attempts, restage on fall-off — never fired);
   - **press** (horizontal force at the CoM, the push a closed gripper produces):
     velocity servo `f = 6·(v_des − v)`, v_des 0.08 → 0.05 m/s near the wall, force
     cap 1.8 N; stiction floor 0.5 → 1.6 N on measured stall (never fired);
     force-frame mode PROBED from measured progress every 45 steps and toggled on
     8 mm regression (never fired). The cube slides down the runway, presses the
     flap inward, tips over the sill, and falls onto the bin floor; the flap swings
     shut behind it (settled readback +0.0° every time);
   - asserts after each: `max_flap > 15°` during the press (the flap was genuinely
     pushed open — no bypass), insert latch fired, `SIM_GEN_SCORE` monotone.
     Measured: cubes stack neatly at the front wall (z 0.027/0.057/0.087 for k=3).
3. **Final**: flap closed < 8°, success() live, score 1.0; hands-off persistence
   10 × 40 steps = 3.33 s; success still holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints, e.g. seed 1 (k=3): 0.0000 → 0.3000 → 0.4500 →
1.0000 → 1.0000 → 1.0000.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(−0.25, 0.00, 0.00), facing +x** (nominal reach 0.855 m).
The scatter slots lie 0.35–0.50 m from the base, the ledge at 0.50–0.58 m, and the
slot mouth (the farthest working point, x = 0.33, z = 0.15) at 0.60 m — all inside
the dexterous shell. The arm NEVER needs to enter the bin or the slot.

- **Red/blue cubes** (30 mm, 50 g): comfortably graspable (30 ≪ 80 mm jaw span) —
  a standard top pinch from the ground; carrying one to the ledge and setting it
  down is a pick-and-place at ordinary heights. Color selection is trivially in
  reach of any wrist camera; the count (1–3) is resolved by looking.
- **The press**: with the cube on the ledge, a closed-gripper fingertip push at
  sill height (z ≈ 0.135) along ~10 cm of runway, force ~1–2 N. The slot is 80 mm
  wide vs the 30 mm cube (± 25 mm lateral slack), and the cube tips over the sill
  as soon as its centre crosses the inner edge — at that moment the cube's back
  face (where the fingertip presses) is still 5 mm OUTSIDE the wall's outer plane,
  so the fingers never have to enter the slot; the flap and gravity finish the job.
- **Flap** (76 × 58 × 6 mm, 30 g): never grasped and never needs to be — it is
  operated through the pushed cube; its gravity hang re-closes it. The ~0.3 N it
  takes to hold it open is well under the arm's capability had a strategy needed it.
- **Bin / ledge**: kinematic; incidental contact cannot move the goal frames.

## Execution order (declared)

**No required order among the red cubes** — any permutation of posting them
succeeds, and the rubric is order-blind (per-cube latched insert credit scaled by
the present count). The only hard sequencing constraints are per-piece and
physical: a cube must be staged at sill height *before* it can be pressed (the
ground path is geometrically refused — smoke 5), and identity must be decided
*before* posting, because entry is one-way (smoke 8/9): a posted blue can never be
retrieved and latches the failure. solve.py posts the present reds in index order
purely for convenience.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (k=2: red_1, red_2): SUCCESS, scores 0.0 → 0.375 → 1.0 → 1.0.
- `solve --seed 1` (k=3): SUCCESS, 0.0 → 0.30 → 0.45 → 1.0 → 1.0 (three-cube stack
  z 0.027/0.057/0.087, all below the 0.110 inside bound).
- `solve --seed 2` (k=2: red_0, red_2): SUCCESS — three seeds, distinct present
  patterns; the stiction escalation, restage, and force-frame toggle never fired.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 12/12**, frames.npz (253 × 600 × 960) saved:
  1. settle/no-NaN: flap +0.00°, nothing inside, score 0, no success
  2. randomization readback: max cube xy Δ2056 mm (permutation + jitter), max yaw
     Δ51.3°, present patterns differ
  3. red-count coverage: k ∈ {1,2,3} over 15 resets; every red index present and
     parked at least once; depot world-readback agrees with the mask
  4. null policy: 300 idle steps, flap drift 0.00°, score 0, no success
  5. SEED STRATEGY: quasi-static 3×-weight ground drag straight at the slot —
     travels 415 mm freely, ends at z 15 mm against the blank lower face (sill
     120 mm); never inside, approach never latches, score 0.000
  6. drop-in: red dropped from above lands ON the roof (z_loc 285 mm) and stays out
  7. FLAP ONE-WAY: the same 1.0 N force swings the flap to +85.2° inward but moves
     it 0.0° outward (joint stop); released, it falls shut
  8. INTERIOR RETENTION: an inside red shoved at the slot at 3× weight moves
     115 mm (to the front wall) yet stays inside; k=3 episode → no success
  9. CONTAMINATION: 2 reds + 1 blue constructed inside → score capped 0.300
     (< the 0.45 earned insert credit), no success; blue teleport-extracted →
     latch holds, still 0.300, still no success (irreversible)
  10. doorway near-miss: red at rest ON the sill in the slot mouth, 2 mm from the
     flap — approach credit only (0.150), not inside (z 135 > 110 mm bound)
  11. flap-held-open: at +85.0° `flap_closed` is False (success conjunct dead);
     released, gravity re-closes to +0.0°
  12. video frames.npz saved

## Files

- `scene.py` — LetterboxBinScene + bin/flap compound spawners + per-env hinge
  authoring + rubric; registers `simgen.letterbox_bin`.
- `solve.py` — stage-on-ledge + press-through-flap certificate (`--seed N`).
- `smoke.py` — 12-check rejection battery + video.
