# push_buttons_i40 — counterweigh the balance scale (`counterweight_scale`)

Count the red reference cubes in the red pan of a two-pan balance scale (k ∈ 1..7),
then place the unique subset of three blue weight blocks (1 u / 2 u / 4 u — a binary
weight set) into the blue pan so both pans carry equal weight and the beam settles
level. References must stay untouched in the red pan; unused blocks must rest far
from the scale. Judged only at sustained rest.

## Seed provenance

- Seed: `rlbench/push_buttons`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/push_buttons.py`).
- Seed plan: a Franka presses 3 colored buttons (articulated push-buttons) in a
  prescribed color order; success = each button's joint reaches its pressed position.

## What changed, and why it is strategically different

| | seed `push_buttons` | this task |
|---|---|---|
| core skill | ordered pressing of named colored targets | **count k by sight, solve a subset-sum over a binary weight set {1,2,4}, verify through a physical mechanism** |
| mechanism | per-button prismatic joints | a passive revolute **balance beam** with a pendulum bob and two freely hanging pans |
| evidence of success | joint positions | the beam's settled equilibrium: level within 4° **only if** the loaded mass truly equals k units |
| failure modes | wrong order / wrong button | wrong count, wrong subset, disturbing the references, propping/parking cheats |

Nothing is pressed, no ordering exists, and the judged quantity (equilibrium tilt of
a 1-DoF passive mechanism under load) does not appear in the seed at all. The plan —
*perceive a count, decompose it in binary, prove the decomposition by counter-
weighing* — is a different strategy, not a re-parameterization.

Against the tasks_v7 corpus (surveyed before design): no existing task uses a
balance/counterweight mechanism or counting/subset-sum mass reasoning.
`push_button_i35` (the other descendant of this seed family) is lever-gated ordered
*pressing* — this task presses nothing. `play_jenga_i31` spins captive dial arrows;
`living_room_..._i33` transports one item into a tray; `pen_holder`-style packing
tasks judge containment only. Here containment (clauses C/D) is only a guard; the
load-bearing predicate is the mechanism's equilibrium.

## Scene

- Balance scale, all procedural: 30 kg dynamic base (a heavy fixture so the spawn-
  authored revolute joints stay anchored after teleports), amber beam on a revolute
  pivot (±12° limits) with a below-pivot bob (restoring moment ≈ 0.061 kg·m), and
  two 130 mm pans hanging from revolute hangers at ±0.16 m — pans self-level, so
  only pan *load*, never load *position*, biases the beam.
- k ∈ U{1..7} red reference cubes (32 mm, 100 g) seated in a grid on the red pan
  floor; the scale pose is jittered/yawed and randomly flipped 180° (which side is
  red swaps); three blue blocks (32/40/51 mm = 1/2/4 u) shuffle over three staging
  slots outside the 0.28 m exclusion radius.
- Physics honesty (asserted in `__post_init__`): a 1-unit imbalance rests beyond the
  ±12° joint limits (≈15° free equilibrium) while the 4° level tolerance would need
  a <1/3-unit error to fool; the exclusion radius covers the scale's whole reach, so
  a propping block can never satisfy clause D; every block fits its pan and the
  Franka jaw.

## Rubric

`success() = level & still & refs_home & cand_ok`, all physical:

- **level** — |beam tilt| ≤ 4°.
- **still** — SUSTAINED rest: beam/pans/blocks below velocity thresholds for 60
  consecutive steps (0.5 s). A beam swinging through level dips below the velocity
  threshold for an instant at its turning point; the sustained gate is what makes
  "judged only at rest" real (regression-tested in smoke check 6).
- **refs_home** (clause C) — every present reference cube inside the red pan
  (rejects emptying the red pan or moving references).
- **cand_ok** (clause D) — every candidate inside the blue pan OR ≥ 0.28 m from the
  scale axis (rejects propping the beam, parking on the scale, and red-pan
  compensation).

Given C ∧ D, statics guarantees level-at-rest ⇔ blue load = k units.
`score() = clamp(0.55·latched-load-progress + 0.35·ever-success, ≤0.90)`, exactly
1.0 iff success now; latches only advance (credit survives regressions).

## Teleport solution (`solve.py`)

Teleport = TRANSPORT ONLY; everything load-bearing happens through contact:

- P0 — hands-off settle: the beam falls onto the red-side limit; read k.
- Decompose k in binary over {1,2,4} (each k in 1..7 has exactly one subset).
- P1..Pn — for each selected block, largest first: teleport it to a hover point 10 mm
  above the blue pan's rim (pan-local, mapped through the pan's live pose), release;
  it free-falls into the pan, the pan floor carries its weight into the hanger, the
  beam responds. Hands-off adaptive settle; containment readback with re-drop retry.
- P9 — hands-off wait for the balanced equilibrium (success at sustained rest).
- P10 — persistence: ≥3.3 simulated seconds hands-off, success at every checkpoint.

Verified on the forge: seeds 0 (k=6 → medium+large), 1 (k=2 → medium), 2 (k=5 →
large+small) all print `SIM_GEN_SOLVE: SUCCESS` with non-decreasing scores.

## Execution order

**No fixed order is required or declared.** Blocks may enter the blue pan in any
order (the instruction says so); only the settled end state is judged. The solver's
largest-first order is a convenience (heaviest impact lands on the emptiest pan),
not a contract.

## Seed-strategy smoke check: N/A

The seed's strategy ("press each colored button") is inexpressible here: the scene
contains no buttons and nothing press-able — pressing down on either pan merely
tilts the beam until the presser releases it, leaving no state behind. Documented
N/A; in its place the battery attacks this task's own cheat surface (clauses C/D,
under/over-load, mid-swing judging).

## Franka embodiment (single arm, parallel jaw)

One plausible base pose: on the bench at **(0.0, −0.45, bench top), facing +y**
(toward the scale) — staging row (y = −0.22 ± jitter) 0.23–0.29 m away, pan centers
≈ 0.53 m away at z ≈ 0.16–0.19 m: inside a Franka's ≈0.85 m reach with top-down
grasps.

- Weight blocks (32/40/51 mm cubes, 100–400 g): all narrower than the ≈80 mm jaw
  opening; free-standing on the bench with ≥90 mm between staging slots — clean
  top-down side-face grasps.
- Release into the pan: the pan is 130 mm wide and self-levels; a block released a
  centimeter above the rim (what the solver's drop emulates) lands inside — no
  precision insertion needed. The pans swing at ±0.16 m from the axis, so the arm
  never needs to reach over the beam.
- Counting k: the red cubes sit in a single-layer grid in an open-top pan, sizes and
  colors are high-contrast — countable from a wrist or overhead camera.
- Forces: nothing must be pushed or held; the heaviest object is 0.4 kg.

## Checks (smoke battery, rejection-only — 15)

1. settle/no-NaN: refs seated, beam on the red-side limit, sustained rest.
2. score ~0 at reset, no success.
3. randomization readback: k ≥ 3 distinct values, scale xy moves.
4. randomization: red-pan side/heading + staging permutation vary; invariants (k
   refs in red pan, candidates beyond exclusion).
5. null policy: 360 idle steps → score ~0, no success.
6. mid-swing: correct load, beam seen level() while swinging — success never fires,
   no ever-success latch (sustained-stillness regression test).
7. under-load by 1 unit → beam stays on the RED side, not success.
8. over-load by 1 unit → beam tips past level to the BLUE side, not success.
9. clause C: red pan emptied — scale balances level+still, rejected.
10. clause C: reference moved into the blue pan (1 v 1 level), rejected.
11. clause D: correct subset + unused block parked inside the exclusion radius —
    level+still+refs_home, rejected.
12. clause D: k=1 red-pan compensation (2 v 2 level), rejected.
13. latched credit: 4/6 → 5/6 strictly increases, survives removal, never success.
14. rejection audit: success() never True anywhere in the battery.
15. final no-NaN.

Plus acceptance evidence: `solve.py` SUCCESS on 3 seeds (forge, IsaacSim 5.1).
