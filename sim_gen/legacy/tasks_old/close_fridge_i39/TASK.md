# close_fridge_i39 — `simgen.fridge_clearway`

**Seed:** `rlbench/close_fridge`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/close_fridge.py`) — a franka pushes a
hinged fridge door shut; the entire task is one push on an articulated door.

**Tier: HARD** — 5 declared stages, partial order required (see below).

## What changed

The seed's whole goal — push the revolute door closed — survives only as the *trivial
final stage*, and executing it first is made physically self-defeating. The door's
closing sweep is littered with grocery clutter (a heavy high-friction crate, a tall
carton, a jug lying on its side, 1–3 present per episode). A door pushed with a bounded,
realistic torque PLOWS the clutter along its arc until an item wedges between the door
face and the cabinet front plane, and the door stalls far from closed (measured: 2.2 N·m
sustained for 6 s stalls at −60…−77° across seeds, while the same push closes the
empty-arc door in under a second — the paired calibration A / negative A checks). The
work is therefore all in the *obstacles*, plus one precision element: the last few
degrees are a magnetic-gasket capture zone (released at −4° the door snaps shut, at
−12° it stays — a measured capture cliff).

### Declared stages (partial order: stow before close; stow order free)

1. Stow the crate on the raised stow shelf beside the fridge (resting fully on the
   shelf top, inside the margin, settled).
2. Stow the carton likewise.
3. Reorient the jug: it lies on its side and only counts stowed standing **upright**
   (axis within 15° of vertical).
4. Place the upright jug on the shelf (the shelf is small — three footprints require
   deliberate arrangement).
5. Swing the now-unobstructed door through its ~100–112° arc into the ≤8° magnetic
   capture zone and let the gasket seat it against the closed stop.

With the sampled subset < 3 the stage count shrinks accordingly (count-what-you-see:
success is judged on the sampled subset).

## Why strategically different (not same-strategy-different-numbers)

- The seed's plan is a **tested failing control**: applying the seed's one action
  (push the door) as the opening move wedges the clutter and permanently occupies the
  arc — the solver needs a different *plan* (reorganize the environment first), not
  different push parameters.
- The dominant work is **transport + reorientation + placement** of free bodies onto a
  raised shelf — object manipulation the seed has none of — with the door reduced to a
  gated terminal actuation.
- **Anti-shortcut clause:** sweeping the clutter *into* the open fridge cavity also
  clears the arc and lets the door close, but success requires every present item
  resting ON the shelf — the stash-inside cheat is a tested negative control.
- Sibling differentiation: no spring-return mechanism or prop/undo (i22), no
  open-then-restore of the same fixture (i15), no lid/cargo role reversal (i13 — here
  nothing is loaded into the container; the cavity must effectively stay empty), no
  dwell timing (i18/i21), no ordered code-latch interlock (i31 — the ordering here is
  enforced purely by wedge physics).

## Honesty notes

- The fridge cabinet + door hinge are authored once and never teleported (jointed-pair
  teleports are unreliable on this stack); randomization lives in the door's initial
  angle (−96…−112°), the item subset (1–3) and sector poses, and the shelf position.
- Oracle is teleport-style (kinematic carries, pin-drag on the door) but every judged
  outcome is physical: settled rests on the shelf, real wedge jams, magnet capture from
  a genuine release, settled closed door. Success = current-state physical predicate;
  only partial credit uses latched (per-item ever-stowed) milestones.
- All geometry procedural (primitives + one compound door spawner); no external assets.

## Smoke check list (15)

1. settle — clean reset, finite, door wide open, score ~0
2. randomization by readback — door angle / subset / poses / shelf differ across seeds
3. null policy — score < 0.05, no success
4. calibration A — bounded push closes the door once the arc is CLEAR
5. calibration B — magnet capture cliff (−4° captures, −12° doesn't)
6. negative A — seed strategy (push first) wedges and stalls; score ≤ 0.25
7. rubric monotonicity — staged oracle scores strictly increase
8. partial credit — all stowed, door open: 0.5 ≤ score < 1, no success
9. oracle A — success, score exactly 1.0
10. oracle B — fresh seed
11. oracle C — subset seed (success judged on sampled subset)
12. negative B — stash-inside cheat refused (door closes, no success, score ≤ 0.15)
13. near miss A — jug stowed lying is refused; no success with door closed
14. near miss B / calibration C — stow-tolerance sweep (past-margin edge rest refused)
15. video frames recorded (frames.npz in CWD)
