# setup_chess_i121 — the Castling Gallery

Slide the captive **rook** off the one-piece-wide main gallery into the side pocket marked by
the green **beacon** (the mirrored pocket is a decoy), then slide the captive **king** past
the junction to the **gold** home cell. Env id: `simgen.castling_gallery` (scene-level,
`robot="null"`).

## Provenance

- **Seed task:** `rlbench/setup_chess` — "set up the chess board": 32 chess pieces are
  picked off the table one by one and placed on their marked board squares
  (`RoboVerse/roboverse_pack/tasks/rlbench/setup_chess.py`; USD board + free-standing
  pieces, pure pick-and-place, no ordering, no checker in the ported config).
- **Kept from the seed:** the chess dressing (a king and a rook, a "board" fixture, the goal
  of putting pieces onto their proper squares) and the placement-tolerance flavor of the
  success predicate.
- **Replaced:** everything about how placement happens (see below).

## Why it is strategically different

| axis | seed `setup_chess` | this task |
|---|---|---|
| transport | lift piece into free space, carry, lower | pieces are **captive**: flanged bases trapped under overhanging lip rails; they physically cannot be lifted (smoke-proved) — only **slid** along channels |
| placement | release above a marked square | **push to a hard stop**: every goal pose is produced by contact against an end wall, never by a release |
| ordering | none — 32 independent moves | **geometry-enforced strict order**: the gallery is one piece wide and the rook starts between the king and home, so the rook *must* be garaged first (smoke-proved: a pushed king shoves the rook train into the gold stop and tops out at x ≈ 0.156, short of the 0.181 home band) |
| perception | square markings only | **beacon disambiguation**: two mirrored pockets; only the green-tile side counts, randomized per episode — a rook parked in the decoy pocket scores ~0 |
| failure modes | drop/misplace | wrong pocket, wrong order, wrong piece identity, resting *on top of* the rails instead of *in* the channel |

Also differs from the sibling tasks read while working: `setup_checkers_i2` (gravity-drop
funnel silo, drop-order stacking), `setup_checkers_i39` (lever-pry crate),
`lift_peg_upright_i116` (uprighting). Nothing here is dropped, stacked, pried, or uprighted;
the manipulation is planar captive-slide routing through a channel network with a decoy.

## Scene (fully procedural)

- **Plinth** (kinematic compound): 0.60×0.44×0.12 m slate block; on top a channel network —
  main gallery (interior x −0.24..+0.24), junction at x_j = −0.02 opening into two mirrored
  pockets (out to |y| = 0.16). Flange-level walls (16 mm tall, 64 mm gap) under overhanging
  lip rails (z 16–30 mm, 38 mm slot). The +x end wall is gold and taller (home landmark).
  Captivity is real everywhere: flange Ø56 mm > 38 mm slot, and even the junction's
  cross-shaped lip opening inscribes only Ø53.7 mm (= 2·√2·19 mm) < Ø56 mm.
- **King** (dynamic): flanged base Ø56×12 mm, ivory shaft Ø30×110 mm, gold ball crown;
  m = 0.20 kg, authored base-weighted CoM (z = 10 mm) + diagonal inertia.
- **Rook** (dynamic): same base, short dark-red shaft (60 mm), wide flat turret drum.
- **Beacon**: kinematic green tile re-posed each reset beside the target pocket's end.
- Friction materials bound per-collider (μ_s 0.25 / μ_d 0.20); solver pos/vel iters 16/4;
  dt = 1/120.

**Randomization (readback-verified):** plinth yaw ±25° + xy jitter ±5 cm; king start
x ∈ [−0.20, −0.14]; rook start = king + sep, sep ∈ [0.09, 0.24] clamped ≤ 0.13 (always
between king and home); target pocket side ±y (beacon follows).

## Rubric (latched partial credit; success judged live)

- **0.20** `l1` — rook garaged: flange fully clear of the gallery, on the *beacon* side.
- **0.25** `l2` — rook seated at the target pocket's end cell (±25 mm, captive, upright).
- **0.30** `l3` — with `l2` set, king past the junction (x > x_j + 0.07, captive).
- **1.0 iff `success()`** (else capped at 0.75): rook seated at the target pocket end AND
  king at the gold home cell (±25 mm), both upright (≤15°), both captive in the channels
  (z-gate rejects on-the-lips poses at +30 mm), both settled (<0.03 m/s), states finite.

Tolerances are honest by construction: walls centre a captive flange to ±4 mm and the end
stops bound the along-axis coordinate to 132 mm (pocket) / 212 mm (home) — both inside their
25 mm bands — so pushing to the stop always passes, while the blocked-order king (≤156 mm),
an on-the-lips piece (+30 mm z), and the decoy pocket can never pass.

## Solution outline (`solve.py`, demonstrated on forge)

No transport teleports at all — both pieces are driven exclusively by a horizontal
velocity-servoed CoM force (F = m·K·(v_des − v) + friction feedforward, K·dt = 0.25, cap
2 N ≈ 1× piece weight, released only when close *and* slow; world forces pre-encoded by
R_ref·R_now^T against external-force frame drag):

P0 settle + layout/beacon/side readback (score 0) → P1 rook pushed along the gallery to the
junction (5 mm alignment), then sideways into the beacon pocket (`l1`, 0.20) → P2 on to the
pocket's end stop (`l2`, 0.45) → P3 king pushed past the junction (`l3`, 0.75) and up to the
gold stop → success, 1.0 → P4 ≥3.3 s hands-off persistence → `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (Franka, one base pose)

Base at ≈ (0, −0.60, 0), facing +y: every channel cell lies within a 0.30–0.75 m reach disc.
Per-object contact strategy: both pieces expose a Ø30 mm vertical shaft standing 18–110 mm
*above* the lip rails (plinth top at 0.12 m, comfortable working height), so a Franka
two-finger gripper side-grasps or fingertip-pushes the shaft and translates horizontally —
exactly the horizontal CoM-level wrench the solve applies (the base-weighted CoM keeps a
shaft push from tipping: ≤15° upright margin never approached). No lifting is ever needed —
or possible. The plinth and beacon are never manipulated. Force scale: ≤2 N servo push and
the smoke's 5.9 N probe are both far inside Franka payload.

## Declared execution order

1. Garage the rook into the *beacon* pocket (through the junction, to the pocket stop).
2. Slide the king past the junction to the gold home cell.

The order is physical, not narrated: smoke check 10 pushes the king first and proves the
rook train caps it at x ≈ 0.156 < 0.181 (home band edge).

## Validation evidence (all on the forge, Isaac Lab / RTX 4090)

- `solve` seeds 0, 1 (−y pocket) and 2 (+y pocket): `SIM_GEN_SOLVE: SUCCESS`, scores
  monotone 0 → 0.20 → 0.45 → 0.75 → 1.0, ~20–25 s each (seeds 0 & 2 re-run after the final
  junction-geometry change; seed 1 on the prior, looser junction).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 11/11`, frames.npz (122, 600, 960, 3) saved:
  1. settle/no-NaN (captive, upright, score 0);
  2. randomization readback (Δyaw 20.2°, Δxy 49 mm, Δpiece-x 15/66 mm);
  3. target-side coverage over 10 resets + beacon-follows-side readback;
  4. null policy ~0;
  5. **lip captivity** with proven actuator: shove slides the rook 246 mm, then a
     3×-weight upward pull *at the junction opening* rises only 4.0 mm (= head room), stays captive;
  6. **seed strategy rejected**: pieces placed from above ON TOP of the rails at perfect
     goal xy → not in channel, score 0;
  7. near-miss (king 6 cm short) → success False, score = 0.75 cap;
  8. **decoy pocket** tableau (rook wrong side, king home) → score 0;
  9. swapped identities (king in pocket, rook at home) → score 0;
  10. **order is physics**: king pushed 5 s with rook ungaraged advances 31 cm to
      x = 0.156 and is denied (band starts 0.181);
  11. video saved.
