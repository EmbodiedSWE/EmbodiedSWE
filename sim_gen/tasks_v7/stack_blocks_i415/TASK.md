# stack_blocks_i415 — GradeBeacon (`simgen.grade_beacon`)

Build a support pillar to a **measured grade** and set the beacon level with the
collar.

## Seed provenance

- **Seed**: `rlbench/stack_blocks`
  (`RoboVerse/roboverse_pack/tasks/rlbench/stack_blocks.py`): four identical red
  50 mm target cubes + four yellow distractors + a target plane; the strategy is
  a repetitive color-selection pick-and-place loop — pick each RED cube out of
  the distractors and pile it on the marked plane. The **stack itself** is the
  goal; its height never matters and every episode needs the same fixed count of
  identical blocks.

## What changed, and why it is strategically different

| | seed (`stack_blocks`) | this task (`grade_beacon`) |
|---|---|---|
| goal object | the stack of 4 red cubes | ONE red beacon cube; **no block is a goal object** |
| what is judged | which cubes ended on the plane | the beacon's **elevation vs a measured spec** (collar band on a mast), + on-pad column + resting on a **built** support |
| episode variation | poses only | poses **and the spec itself**: 5 collar grades; the correct block subset — and even the **number** of blocks — changes per episode |
| plan | fixed loop: select red, place on plane, repeat ×4 | **read the grade, solve an exact subset-sum over four distinct block heights (30/45/60/75 mm), build tallest-first, crown with the beacon** |
| code structure | count/containment-on-plane test | grade-alignment band + pad-column + block-top-supports-beacon-bottom predicate with a sustained hold window |

The seed's own strategy is expressly rejected by the rubric: stacking **all** the
blocks (the fixed-loop plan) overshoots every grade and earns nothing beyond the
latched partial build credit (smoke check 6). No single block — upright (30, 45,
60, 75 mm) or lying (55 mm) — lands within the ±10 mm band of any grade
(asserted in `__post_init__`), so a multi-block pillar must be *composed*;
adjacent grades are 15 mm apart, outside each other's band, so the grade must be
*read*, not guessed (smoke check 8).

Distinct from the corpus tasks I inspected: `trestle_service` / the bridge tasks
span **fixed** supports at a fixed height; `sorting_tower` is a keyed shaft;
`pile_driver` is impact flushness; the ballast / beam-balance / weighbridge
families measure **weight**, not a randomized build-to-elevation spec;
`counterpoise_rack` does lever arithmetic. No corpus task requires composing a
support to a per-episode **measured height** and judging a crown's elevation
against a visual grade marker.

## Scene (procedural geometry only)

- Kinematic **mast** (r 16 mm, h 420 mm) with a **visual-only amber collar** band
  (r 19 mm, h 24 mm, **no collider** — nothing can perch on it), re-posed every
  reset to the sampled grade: collar center = pad_t + grade + beacon_s/2 (the
  required **beacon-center** height).
- Kinematic amber **build pad** (90 × 90 × 6 mm), 115 mm from the mast, bearing =
  the (free) rig yaw.
- Four free **gray blocks**, 55 × 55 mm cross-section, heights {30, 45, 60, 75} mm
  (lighter gray = shorter), and the free red **beacon** cube (45 mm), scattered on
  a 0.34 m ring around the rig (5-slot permutation + jitter + free yaw).
- Randomization (readback-verified): grade ∈ {90, 105, 120, 135, 150} mm (each an
  exact ≥2-block upright subset sum), rig xy jitter ± free rig yaw, scatter
  permutation/jitter/yaw.

`success()`: beacon center within 48 mm (xy) of the pad center, within ±10 mm of
the collar center height, **supported** — some in-column block's top face at the
beacon's bottom face (±8 mm; a beacon merely held in the air at grade fails) —
and held still ≥ 20 sustained substeps (isolated <3-substep GPU resting-contact
noise spikes stall, not reset, the counter; a real kick resets it within 3
substeps).

`score()`: 0.4 × latched pillar progress (top/grade, earned only while the
column is settled and **not overshot**) + 0.4 × latched first-success; exactly
1.0 iff `success()` now. Monotone at phase boundaries; null ~0.

## Teleport solution (solve.py) — honesty

Teleports do **transport only**; every load-bearing interaction is contact
dynamics:

1. **P0** reset + settle; read the grade from the spec; solve the exact
   subset-sum (integer mm); score 0.
2. **Per block (tallest first)** — CARRY: teleport to a hover pose 18 mm ABOVE
   its rest height over the pad (never into contact). SEAT: gravity-feedforward
   + PD velocity-regulated force (CoM only, no torque, no pose pinning) lowers
   it onto the pad/pillar with an xy centering PD; the wrench is dropped at seat
   contact; the pillar carries its own weight. Score rises (latched build
   credit).
3. **Crown**: the beacon is hover-teleported above the finished pillar and
   force-lowered the same way, coming to rest with its center level with the
   collar band. Success; score 1.0.
4. **Persistence**: ≥ 3.3 simulated seconds hands-off; success holds; then
   `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0 (grade 90 = 60+30), 1 (grade 150 = 75+45+30), 2
(grade 120 = 75+45) — three different grades, three different plans, all
`SUCCESS` with monotone scores and flicker-free persistence.

## Embodiment argument (single Franka, 80 mm jaw)

Blocks are 55 mm across and the beacon 45 mm — comfortable side grasps for an
80 mm jaw. Everything to grasp starts on an open floor ring (r 0.34 m) with
≥ 0.40 m between slots; placement targets are on/above a 90 mm pad with the
highest grasp-release at ~195 mm + approach — well inside Franka's envelope. The
mast is 115 mm away from the build column, leaving a clear top-down approach.
Tolerances are cm-scale (±10 mm band, 48 mm column radius). A plausible base
pose: on the floor ~0.45 m from the mast, opposite the pad bearing, so the arm
reaches both the scatter ring and the pad. No execution order is declared beyond
what physics forces (support before crown); block choice/order within the subset
is free.

## Checks (smoke.py — rejection-only, recorded, 15/15 on the forge)

1. settle/no-NaN + layout sane (scatter outside the column; collar readback on
   the grade ladder; pad `mast_gap` from the mast)
2. score ~0 at reset
3. randomization: the GRADE varies across seeds (≥3 rungs seen, readback)
4. randomization: rig yaw (pad bearing) + scatter permutation vary; every reset
   sane
5. null policy: 240 idle steps → score ~0
6. seed-strategy: ALL four blocks stacked + beacon (210 mm tower) → overshoot
   rejected; score ≤ build credit
7. single tallest block + beacon → ≥15 mm below every band → rejected
8. neighbor-grade exact pillar → |dz| = 15 mm > band → rejected
9. correct-height pillar + beacon on bare floor OFF the pad → in band but out of
   column → rejected
10. beacon HELD in the air at grade in-column (40 substeps, velocity re-zeroed)
    → `supported()` stays False, success never fires; released, it falls
11. sustain gate: correct crown judged 6 substeps after contact → not yet
    success
12. latched credit: beacon removed → score == build credit exactly, place credit
    never earned
13. no invisible shelf: beacon beside the visual-only collar falls to the floor
14. rejection audit: success() never True anywhere in the battery
15. final no-NaN

`frames.npz` (236 × 600 × 960 × 3) recorded on the forge.
