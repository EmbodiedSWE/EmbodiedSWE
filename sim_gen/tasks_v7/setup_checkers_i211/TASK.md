# setup_checkers_i211 — TiltLabyrinthScene (`simgen.tilt_labyrinth`)

Route the SEALED red checker into the green-marked corner pocket by TILTING the
whole gimballed case; the piece itself can never be touched.

## Seed provenance

- **Seed**: `RoboVerse/roboverse_pack/tasks/rlbench/setup_checkers.py`
  (`rlbench/setup_checkers`): a chessboard USD lies flat on a table with 24 free
  primitive checkers (r ≈ 18 mm cylinders) scattered around it; the task is
  "place the checkers on the board in the starting arrangement" — an unordered,
  repetitive sequence of direct pick-and-place actions onto marked squares of an
  open horizontal surface.

## What was kept / changed

Kept from the seed: a red checker (48 mm × 12 mm slick cylinder), the notion of a
marked destination square on a bounded rectangular play field, and "checker
settled on its marked spot" as the goal predicate.

Changed (everything load-bearing):

| Seed | This task |
|---|---|
| 24 identical free checkers | ONE checker, SEALED under a slotted grille roof |
| Piece is grasped and carried | Piece is physically untouchable; the ENVIRONMENT (a 2-axis gimballed case) is the manipuland |
| Board is a passive open surface | The case is a spring-centred, ±12° limited gimbal actuated by pressing its rim |
| Direct placement onto the square | Indirect gravity routing: funnel → baffle gap → far wall → sunken corner pocket |
| Any order, mistakes recoverable | A mirrored DECOY pocket retains the checker permanently — a wrong drop is unrecoverable |
| Destination marked on the board | Target marked by a green beacon post on the PEDESTAL, side randomized per episode |
| Done when placed | Done only after the rim is RELEASED and the springs re-level the case (release is judged) |

## Why strategically different

- **From the seed**: the seed's entire skill — grasp a free checker, lay it flat
  on a marked square — is inexpressible here. There is no grasp (smoke
  force-proves the seal: a 50× weight pull cannot extract the piece), no
  placement surface (the board interior is reachable only through tilt-commanded
  gravity), and the seed's nearest end-state analog (checker laid on TOP of the
  case over the right pocket) is explicitly smoke-rejected by the play-volume
  z-band. A solver must instead: perceive the beacon side, plan a two-leg tilt
  route through a maze, and control a spring-loaded 2-DOF mechanism with
  release-to-level as the final act.
- **From sibling derivatives** (read for differentiation): `setup_checkers_i2`
  (gravity-drop silo stacking — pieces are grasped and dropped), `setup_checkers_i39`
  (lever-pry a crate lid — tool use on a lid, pieces then handled directly),
  `setup_chess_i121` (captive slide routing — the piece is pushed directly along
  slots). None seals the piece away from all contact; none actuates the
  environment's attitude as the sole transport mechanism; none has an
  irreversible decoy trap.

## Scene (fully procedural, compound spawners)

- **Stand**: heavy dynamic pedestal (40 kg, damped) — dynamic, not kinematic, so
  the whole jointed linkage can be teleported together at reset (kinematic-body0
  joint anchors stay world-fixed after teleport).
- **Frame**: blue gimbal ring, spawn-authored X-axis RevoluteJoint to the stand,
  ±12°, spring-return angular drive to 0 (drive units are per-DEGREE: 0.06 →
  ≈ 3.4 N·m/rad, confirmed by the ~0.9° residual tilt readback under the
  off-centre checker).
- **Tray**: the case — floor with two pocket holes, 7 mm sunken pocket floors,
  press rim (16 mm wide), 45° funnel guide walls, gapped full-height baffle,
  slotted grille roof (12 mm slots ≪ 48 mm disc; 22 mm clear height < disc
  diameter so it can never flip upright). Y-axis RevoluteJoint to the frame,
  same limits/drive. Slick physics material (μ 0.12/0.10) authored and bound in
  the spawn func (custom spawners apply no cfg schemas).
- **Disc**: the checker, 30 g authored root mass, velocity iters 4 (GPU cylinder
  phantom-creep armor).
- **Beacon**: kinematic green post, re-posed to the target side each reset.
- Geometry honesty is asserted in `__post_init__` (gap passable, grille seals,
  pockets admit + retain, funnel feeds the gap, pocket z-threshold separation,
  press force ≤ 8 N, tilted tray clears the column at the combined-tilt worst
  corner).

**Randomization** (readback-verified in smoke): stand yaw ±20° + xy jitter
±5 cm, checker start cell in the start chamber, target-pocket side (torch.rand
comparison — first-randint-after-seed is degenerate).

**Rubric** (latched stage credit anchored in the demonstrated solve):
0.25 `gap_latch` (past the baffle INSIDE the play volume, z-banded, 3-step
streak) + 0.35 `pocket_latch` (inside the target pocket volume, 3-step streak),
non-success capped at 0.60; 1.0 iff `success()` = in target pocket AND case
level (≤3°) AND settled AND finite, judged live.

## Solution outline (solve.py — passes seeds 0, 1, 2; both target sides exercised)

1. **P0** settle + readback (layout, masses, spring-residual tilt ~0.9°, score ~0).
2. **P1** tilt-servo leg 1: PD attitude servo + spring feedforward (the
   feedforward kills the proportional droop measured on the forge) applies a
   ≤1.5 N·m torque on the tray — the wrench of a hand pressing the rim — tilting
   downhill +x ≈ 10.9°; gravity slides the checker through funnel + gap to the
   far wall. SCORE 0.25.
3. **P2** leg 2a: pure ±y tilt parks the checker against the beacon-side wall (a
   diagonal is friction-infeasible — measured stall); leg 2b: pure +x tilt
   slides it along the wall (no side-press → no wall friction) until it drops
   into the target pocket. SCORE 0.60.
4. **P3** release: servo to level, wrenches zeroed; springs hold the case level,
   the pocket ledge holds the checker → success. SCORE 1.0.
5. **P4** ≥3.3 s hands-off persistence → `SIM_GEN_SOLVE: SUCCESS`.

Teleport is used for NOTHING in the solve — there is no free object to
transport; every rubric bit is produced by hinge + contact dynamics.

## Embodiment argument (Franka, plausible base pose)

Base ~0.55 m from the pedestal centre, facing the case broadside; the press rim
is a 16 mm flat band at 0.29 m height around a 300×220 mm case — comfortably
inside the workspace at half extension.

- **Tilting**: close the gripper and press DOWN with the fingertip pad on the
  rim mid-edge. Holding a hinge on its stop needs ≈ 0.72 N·m / 0.15 m ≈ 4.8 N
  (asserted < 8 N in cfg); the solve's 1.5 N·m torque clamp ≈ 10 N transient at
  the same lever — trivial for a Franka (continuous payload 3 kg). Steering =
  choosing which rim edge to press and how hard; releasing = lifting the finger.
  Two-edge combinations (e.g. +x then ±y) are sequential single presses — no
  bimanual contact needed.
- **The checker**: requires NO contact strategy — that is the point of the task.
  The grille slots (12 mm) are narrower than any Franka fingertip pad and 4×
  narrower than the checker; smoke proves even a 50× weight pull cannot extract
  it.
- **The beacon**: perception only, never touched.
- **Camera**: the grille is open enough (12 mm slots at 20 mm pitch) that the
  red checker is visible from an over-shoulder view at all times.

## Execution order (geometry-forced partial order)

gap BEFORE pocket: the full-height baffle admits the checker into the pocket
half of the case only through the central gap, and the pockets are recessed
behind it — `pocket_latch` is unreachable without `entered_far` having been
true. Within the far chamber the wall leg and pocket drop can interleave freely
(any trajectory that ends in the beacon pocket is fine); the release must come
LAST because success requires level + settled while the checker is pocketed.
A premature correct-pocket drop is simply success once released — there is no
timing trap on the honest route; the only irreversible mistake is the decoy.

## Checks

- solve: `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1 (+y target) and 2 (−y target);
  scores 0.00 → 0.25 → 0.60 → 1.00, non-decreasing at every boundary.
- smoke battery (10 checks): settle/no-NaN; randomization readback (yaw / xy /
  start cell); target-side swap + beacon consistency; null-policy ~0; SEAL
  reality (anti-vacuous 1.5 N pull arrested by the grille); seed end-state
  analog on the roof rejected by the z-band; decoy permanence under full-limit
  escape presses; far-wall near-miss capped at 0.25; held-tilt-never-success +
  release-positive-control + pocket retention; frames.npz video.
