# approach_grasp_d1 — ordered tower build on a pedestal

| Field | Value |
|---|---|
| Seed | `RoboVerse/roboverse_pack/tasks/pick_place/approach_grasp.py` (`pick_place.approach_grasp_simple`) |
| Tier | middle |
| Stages | 3 (place large → stack medium → stack small; strictly bottom-up) |
| Ordered? | **Yes** — the tower must be built bottom-up in size order; wrong-order execution is a negative control and fails. |

## What changed vs. the seed

The seed is a single-object *approach → grasp → lift* task: success is entering (and
keeping) a stable grasp of one cube, i.e. the goal state is "object held in the air".
Here there are **three blocks of different sizes and a raised pedestal**, and the goal
is a **settled structure, not a held object**: the large block must rest centered on
the pedestal, the medium block on the large one, and the small block on top, each pair
aligned within a per-axis tolerance and everything at rest.

## Why it is strategically different

- The seed's entire plan — approach one object, close the gripper, lift, hold —
  achieves *nothing* here: a block held in the air is at the wrong height, unsettled,
  and part of no structure. The smoke executes exactly this plan (grasp-lift-hold a
  block, including holding it directly above the pedestal) and asserts failure.
- A solver needs a genuinely different plan: sequence three pick-and-place operations,
  bottom-up (each placement needs the previous block as its base), with placement
  precision (align within tolerance) and stability (release and let it settle — the
  success check requires near-zero velocities, so "hover it in place" cannot pass).
- Order is enforced by geometry + identity, not by convention: success names *which*
  block is at each level, so a small-first tower fails regardless of how neatly it is
  stacked (and is physically fragile anyway — a larger block overhangs a smaller one).

## Intended strategy

1. Carry the large block up, over the pedestal, and set it down centered; release and
   let it settle.
2. Stack the medium block on the large one (aligned to the *actual* large-block pose).
3. Stack the small block on the medium one; release; the whole tower must be at rest.

## Rubric (score)

Latched stage credits + latched approach shaping, capped at 0.95 unless success():
`0.15·approach(large→pedestal, best-so-far) + 0.30·[large placed] + 0.30·[medium
stacked] + 0.20·[full tower observed]`. Latches are cleared in `reset_instance` and
appended to `get_state()`/`set_state()` so snapshot/restore round-trips keep earned
credit. Null policy: all latches false, approach 0 → score ≈ 0. `score() == 1.0` iff
`success()`.

## Checks (smoke.py)

settle/no-drift · no-NaN · determinism · randomization-is-real (blocks *and* pedestal)
· physical-property readback (masses ordered l>m>s, table friction) · null policy
fails (success False, score ~0) · oracle succeeds on 3 seeds · rubric monotone across
stage boundaries on 3 seeds · success stable over +300 steps · **seed-strategy control**
(grasp-lift-hold a block → fail; also held directly above the pedestal → fail) ·
**wrong-order control** (small→medium→large tower → fail) · wrong-location control
(correct-order tower on the table, off the pedestal → fail) · latch persistence under
state save/restore and under continued correct behavior · tolerance calibration sweep
(top-block xy offset; knee at `align_tol`).
