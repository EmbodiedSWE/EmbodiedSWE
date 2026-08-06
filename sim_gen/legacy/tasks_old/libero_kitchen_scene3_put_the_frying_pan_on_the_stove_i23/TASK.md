# seesaw_stove — level the tilting cooktop (i23)

**Scene:** `sim_gen.seesaw_stove` · **Tier: medium (3 stages)** · robot="null" (scene-level)

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene3_put_the_frying_pan_on_the_stove`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene3_put_the_frying_pan_on_the_stove.py`)
- Seed plan: grasp the chefmate frying pan, lift, set it on the RIGID flat stove; the moka
  pot is a pure distractor that is never touched. Success = pan within 8 cm xy / 3 cm z of
  the burner site. One pick-and-place, no interaction between objects.

## What changed, and why it is strategically different

Same object vocabulary (frying pan, moka pot, a stove), inverted mechanics: the stove top
is a **see-saw cooktop** — a 72 cm board on a central sprung hinge (weak centering spring,
±12° travel). Placing the ~360 g pan on a burner puts ~0.4–0.8 N·m on the hinge, far above
what the 2 N·m/rad spring can hold: **the seed's entire plan ends with a capsized cooktop**
(the board rests on its tilt stop, 7–12° off level) and the rubric pins at the placement
stage. This is smoke negative control A — the seed's strategy, executed verbatim, fails.

A solver needs a different plan built on **static-equilibrium reasoning**:

1. **Read the commanded goal** — one of three burners is commanded per episode, marked by
   an orange indicator post standing beside the board (goal selection, randomized).
2. **Place the pan** flat on the commanded burner (judged in the board's body frame — a
   tilted board judges its riders consistently).
3. **Counterweight and level** — park the moka pot on the counterweight rail on the
   opposite half at the lever arm that cancels the pan's torque
   (`arm = burner_x · m_pan/m_pot`, i.e. 132/198/264 mm for the three burners, against a
   weak spring ⇒ a ±41 mm placement band). The **distractor becomes a required
   counterweight, and the decision variable is CONTINUOUS (where along the rail), not
   discrete**.

Success is a current physical state: pan seated on the commanded burner (≤3 cm), pot on
the rail, board level within 3.5°, everything settled. Balancing around the WRONG burner
leaves the board perfectly level and still scores ≤0.15 (negative control B): the goal is
commanded, not merely "level". Propping the board level from below is impossible by
construction (board floats 165 mm up; the 110 mm pot would need to sit ≥0.9 m from the
hinge — off the board) and is measured as negative control D.

Differentiation from sibling batch tasks (axes already claimed): no structure is built or
stacked to a height (i11), no weight is *chosen* among distractors (i6 — here both objects
are essential and the variable is WHERE, not WHICH), no confinement/sliding (i4/i9), no
pouring or contents-inversion (i5/i2), no goal-orientation reorientation (i16), and the
stove has no knobs or switches (i7 / sibling i21's turn-on-the-stove axis untouched).

## Execution order

Partially free, declared: pan and pot may be placed in either order (the rubric latches
both independently); the fine leveling necessarily comes last because the equilibrium
depends on both placements. Declared stage count: 3 (read goal + place pan / place
counterweight / level within tolerance).

## Physics honesty

The oracle teleports (teleport-oracle), but every judgment is a settled physical outcome:
the hinge is a real revolute joint with ±12° limits, the spring/damper is a body-local
post_step torque (`τ = −kθ − cω`, the i6/microwave-proven pattern), the pan and pot load
the board through real contact, and the judged tilt is what the sprung board settles to.
The calibration sweep measures the settled tilt as a function of pot arm and must be
monotone through zero with the predicted ~84°/m slope — the balance band is physically
real, not a rubric fiction.

## Check list (smoke battery, 15 checks)

1. settle/no-NaN: reset settles, board level, score 0
2. randomization is real (readback: pedestal pose, indicator, pan/pot spawns)
3. commanded burner varies — all 3 indices sampled; indicator tracks it (readback)
4. null policy: score ~0, no success
5–7. oracle solves each of the 3 commanded burners: success + score 1.0 (≥3 episodes)
8. rubric monotonicity: 0 → 0.15 (pan on board) → 0.40 (pan on commanded burner,
   capsized) → ~0.73 (pot at wrong arm, partial level credit) → 1.0
9. negative A (the seed's strategy): pan on burner, no counterweight → tips >7°, score
   pinned 0.40, never success
10. negative B: wrong burner balanced → board LEVEL yet score ≤0.2, never success
11. negative C (near-miss): pot ≥72 mm short of the exact arm → 4–11.5° off level,
    partial score, never success
12. negative D (prop cheat): pot stood under the board tip can't reach — still ≥4.5° off
13. mechanism: unloaded board springs back level
14. calibration: settled tilt monotone in pot arm, ~0 at the exact arm
15. calibration: level inside ±30 mm of the exact arm, off level beyond ±60 mm
