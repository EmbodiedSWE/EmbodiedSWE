# drawer_fetch_restore (i15)

**Env id:** `sim_gen.drawer_fetch_restore` · **Tier: medium (4 stages)** · robot="null"
(scene-level; robot bindings are a later stage)

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene1_open_top_drawer`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene1_open_top_drawer.py`)
- Seed task: pull the top drawer of a wooden cabinet open past a joint-position
  threshold (`JointPosChecker` on the drawer joint) and you are done. One stage, and the
  drawer's final state is OPEN.

## What changed

Fully procedural re-build (no external assets, no articulation joint): a KINEMATIC
cabinet shell (side walls + roof + back wall, open front; the roof overhangs the mouth)
stands on the ground, and the drawer is a FREE rigid open-top tray that slides on the
ground between the shell's guide walls — "open"/"closed" are judged from the tray's
pose in the shell's body frame (honest physical state, not a joint readout). A red item
cube starts inside the covered tray; a green delivery pad lies elsewhere on the table.

The solver must: (1) pull the drawer out far enough that the item clears the roof —
geometry makes this mandatory: with the drawer shut, the roof underside (62 mm) is below
tray-wall-top + item-size (80 mm), so the item cannot clear the tray wall anywhere under
the roof; (2) lift the item out of the tray; (3) set it down centred on the pad;
(4) push the drawer fully SHUT again.

## Why strategically different (not same-strategy-different-numbers)

- **The seed's entire plan is stage 1 of 4, and it is a tested failing control.** A
  solver that opens the drawer and stops earns exactly the 0.15 stage credit and can
  never succeed (smoke negative A).
- **The goal state of the drawer is the INVERSE of the seed's.** The seed judges
  joint-open; here success REQUIRES the drawer closed — opening is only a transient
  means, and the score latches record it as history, not as the goal.
- **Different plan structure:** open → extract → deliver → restore, with prehensile
  transport of a second object and a terminal fixture-restoration step. The seed has no
  object transport at all (its bowl/plate are inert distractors).
- **Different judged quantity:** seed = one articulation coordinate; here = a
  conjunction of a placed free object (pad containment, current-state) and a restored
  fixture pose, gated by anti-teleport physical-history latches.
- Sibling-collision note: i9 claims "retrieve out of low CLEARANCE then place" (pan slid
  out of an open cubby — clearance blocks LIFTING the pan itself); here confinement is an
  OPERABLE drawer that must be actuated open before extraction is even geometrically
  possible, and — unlike any sibling — the fixture must be actively RETURNED to its
  initial state as a terminal stage. No sibling requires undoing stage 1 at the end.

## Execution order

Partially ordered, and order is physically enforced where declared: open ≺ extract
(roof geometry + latch gating), extract ≺ {place, close-with-item-out}. The final two
achievements (item on pad, drawer shut) may be completed in either order; success needs
both simultaneously.

## Rubric (score in [0,1])

- 0.15 — `opened` latched: opening crossed 60 mm with per-step delta < 50 mm
  (anti-teleport transit latch);
- +0.15 — `extracted` latched: item crossed inside-tray → outside with per-step
  displacement < 50 mm, gated on `opened`;
- +0.10 · latched best carry progress toward the pad (gated on `extracted`);
- 0.70 — item resting ON the pad (current-state: 45 mm xy tol, on-top z, settled),
  gated on both latches so a teleported item earns nothing;
- 1.00 — iff success: item on pad AND drawer closed (< 18 mm opening, seated, settled)
  AND both latches. Doing nothing scores 0.

## Smoke checks (15)

1. settle: reset clean (finite, still, closed, item in tray, score 0)
2. randomization is real (item/pad/yaw/gap READBACK across seeds)
3. null policy: score stays 0
4-6. oracle succeeds on 3 seeds (incremental kinematic pull/lift/place/close)
7. rubric ladder strictly increasing (0 < 0.15 < ~0.32 < 0.70 < 1.0)
8. rubric partials strictly interior
9. negative (SEED'S OWN STRATEGY): open-only pins at ~0.15, no success
10. latch: opened credit persists after re-closing (transient achievement latched)
11. negative (teleport cheat): tray jumped open + item jumped onto pad → score 0
12. near-miss (tolerance): item 85 mm off the pad centre → mid score, no success
13. near-miss (restore): delivered but drawer left open → 0.70, no success
14. calibration: 1.8 m/s upward kick capped by the roof at 30 mm opening
15. calibration: same kick flies free at the extraction opening
