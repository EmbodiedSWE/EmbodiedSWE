# put_money_in_safe_i51 — "vault_chute" (dam the return chute, then bank the cash)

Registered env: `simgen.vault_chute` (robot="null", scene-level task).

## Seed provenance

- Seed: `rlbench/put_money_in_safe`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/put_money_in_safe.py`)
- Seed strategy: grasp a dollar stack, carry it to an open safe, set it down on a shelf
  inside. One carry-and-place into a passive, welcoming container.

## What changed and WHY it is strategically different

The container is redesigned to be **adversarial to placement itself**: the vault's only
"shelf" is a polished chute tilted 18° toward the permanently open doorway, running out
to the floor. The cash bundles are slick (bound material µ≈0.1, combine="min";
tan 18° ≈ 0.32 ≥ 2×µ, asserted in the cfg), so the seed's entire plan — set the money
down inside — is **physically futile**: a bundle laid on the chute accelerates back out
of the vault and ends up on the floor outside (tested, with the exit time asserted
inside the analytic µ/tilt window). A high-friction rubber chock bar lies on the floor
(combine="multiply"; its grip on the chute is asserted ≥ 1.5× the summed gravity load of
both stacks). The required plan is different in kind, not in numbers:

1. **Fashion a retainer** — lay the chock across the chute inside the doorway; friction
   makes it a dam (the same surface that rejects the cash holds the rubber).
2. **Bank the cash against the dam** — rest each present bundle uphill against the
   chock, where *normal contact*, not friction, holds it.

A solver must therefore reason about material/friction contrast and build a supporting
fixture before any placement can stick — placement order is enforced by physics, not by
a code latch (cash placed first has left the vault long before the chock arrives).
Success judges the physical outcome only: every present stack at rest fully inside the
vault, with a 30-substep persistence latch so kinematic teleport-flashes never count.

Differentiation from sibling generated tasks: no spring/prop-with-undo mechanism (i22),
no obstruction clearing (i39), no stack-to-height support tower / part selection (i11),
no dynamic substrate pull (i57), no aperture keying (i14), no transport/stray latches
(i34) — the claimed axis here is **"the goal surface actively rejects the naive
placement; convert a provided material into a retaining dam via friction contrast."**

## Difficulty tier / stages / order

- Declared tier: **easy** — 2 stages (install dam → lay cash against it; 1–2 bundles).
- Execution order: **required**, enforced physically (dam must exist before the cash can
  rest; simultaneity is unrealistic since undammed cash exits in ~0.3 s).

## Randomization (per episode)

- Vault pose: xy jitter ±5 cm + **full yaw** (doorway direction must be read from the
  scene).
- Cash and chock scattered on the floor in front of the doorway (bearing + radius bands,
  free yaw).
- Cash-count subset sampling (1 or 2 bundles present; judged on the sampled subset;
  absent stack parks in an off-camera depot).

## Rubric

- `score()` = 0.25·(chock currently deployed as a dam on the chute) + 0.10·(fraction of
  present cash that ever entered the vault — latched) + 0.55·(fraction currently
  secured); exactly **1.0 iff success()**; doing nothing = 0.0; the seed's own strategy
  measures 0.05.
- `success()` = every present stack: `secured` persistence latch (30 consecutive
  inside+settled substeps) AND currently inside (safe-body-frame containment box, fully
  behind the doorway plane, within a height band of the chute surface) AND settled.

## Smoke check list (19 checks — forge result: `SIM_GEN_SMOKE: ALL PASS 19/19`)

1. settle/no-NaN: reset settles finite, at rest (1); score exactly 0, no success (2)
2. randomization READBACK: vault yaw/pose + cash scatter move (3); subset sampling
   yields both 1 and 2 stacks (4)
3. null-policy fails: 240 idle substeps → score ~0 (5)
4. **negative A = the seed's own strategy** (expressible here): cash laid on the bare
   chute slides out of the vault, score ≤ 0.15 (6); exit time inside the analytic
   [15, 150]-substep window — slide calibration (7)
5. calibration B: rubber chock grips the same chute (< 8 mm drift in 2 s), exactly the
   0.25 dam credit (8)
6. oracle on 3 seeds, subset ON: dam + lay present cash → success, score 1.0 (9–11)
7. two-stack chain, subset OFF → success (12)
8. rubric monotonicity: 0 → 0.25 (dam) → ~0.575 (one secured) → 1.0, strictly
   increasing (13); partials < 0.9, final exact 1.0 + success (14)
9. **negative B / near-miss**: dam laid just OUTSIDE the doorway catches the cash
   straddling the threshold → no dam credit, no success, score ≤ 0.15 (15); re-damming
   inside recovers success (16)
10. negative C: cash parked on the vault roof — never entered, score ~0 (17)
11. anti-flash: a perfect in-vault pose teleported without a dam is not success at the
    written state — persistence latch gates (18); physics has evicted it 0.75 s
    later, still no success (19)

Video: `frames.npz` (viewport rgb annotator) written to the working directory.
