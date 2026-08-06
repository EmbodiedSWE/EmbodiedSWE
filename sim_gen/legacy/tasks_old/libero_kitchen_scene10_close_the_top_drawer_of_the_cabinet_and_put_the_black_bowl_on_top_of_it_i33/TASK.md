# unjam_drawer — clear the carton jamming the self-closing drawer (i33)

**Registered as:** `SCENES["unjam_drawer"]`, env `simgen.unjam_drawer` (robot="null",
scene-level). **Tier: easy — 1 manipulation stage** (a single pick-and-place; the second
goal completes autonomously). **Execution order: trivially forced by causality** — the
drawer *cannot* close while the carton stands in its doorway (measured, not scripted),
so removal necessarily precedes closure; the solver itself executes only one stage.

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.py`)
- Seed plan: two independent motor acts by the solver — (1) push the cabinet's top drawer
  shut (joint position > -0.01), (2) pick up the akita black bowl and place it on the
  cabinet's top surface (bbox test on the `top_side` site). Nothing resists either act.

## What changed, and why it is strategically different

The seed's two goals are kept recognizable — the drawer ends CLOSED and the black bowl
ends ON TOP — but **who performs them, and what the solver must therefore plan, is
inverted**:

1. **The drawer closes ITSELF.** It is spring-loaded (constant closing force + axial
   damping on a bind-time prismatic joint). The seed's headline motor act — pushing the
   drawer home — is performed by the environment, not the solver.
2. **…but it is physically vetoed by a jam.** A tall carton stands in the drawer's
   doorway, pinched between the drawer's face panel and the cabinet fascia by the
   spring's own preload. The carton (130 mm) overtops the 90 mm lintel, so the doorway
   can never swallow it; the panel (78 mm) passes under with 12 mm to spare. The pinch is
   pure contact statics and self-locks (horizontal normals — no squeeze-out; rotating the
   pinched carton would have to push the drawer back open against the spring *and* lift
   the carton's CoM).
3. **The seed's plan is the tested failing control, at both ends.** Shoving the jammed
   drawer with a sustained 3x-spring force never closes it (negative A). Parking the
   carton "on top of the cabinet" — the seed's placement move — leaves the job unfinished
   (negative B).
4. **The seed's second goal is pre-satisfied and must merely be PRESERVED.** The black
   bowl already sits on the cabinet top at reset; knocking it off voids success
   (negative C). There is nothing to place on top — the "put on top" affordance is a trap.
5. **The solver's one real skill is different in kind:** diagnose the veto, lift the
   carton OUT of the doorway (a vertical extraction of a free body — never touching the
   drawer), and tidy it away onto a marked floor mat. Closure then happens on its own.

A solver bringing the seed's plan ("push drawer, place object on top") scores ~0. The
required plan is *remove-the-obstruction-and-let-the-mechanism-act* — a different plan
skeleton, not different parameters.

**Sibling-axis differentiation** (parallel batch): unlike `i19 barred_drawer` (remove a
drop-bar *so the solver can then open/shut the drawer itself* — 4 manual stages), here the
solver NEVER actuates the drawer: actuation is the environment's job and manual pushing is
a measured-futile control; the obstruction is itself the goal object, and the jam is an
in-path pinch held by the mechanism's own preload, not a bar in cradles. Unlike `i6
pressure_plate` (weight holds a spring DOF depressed), nothing is loaded onto anything —
the spring completes, rather than opposes, the goal once the geometric veto is removed.
Unlike `i15 drawer_fetch_restore`, no pull-open/push-shut manual articulation exists at
all. No low-clearance sliding (i4/i9), no dwell timing (i18/i21), no aperture keying (i14).

## Scene / physics honesty

- Fully procedural primitives: kinematic housing compound (side walls, back wall, roof =
  "cabinet top", fascia above the 90 mm lintel), a JOINTLESS dynamic one-piece tray
  sliding on the ground between the housing's guide walls (5 mm/side play; the closed
  stop is the tray back wall meeting the housing back wall — physical, not scripted),
  plain cuboid carton, black cylinder bowl, kinematic mat. The tray wears a slick
  physics material (`friction_combine_mode="min"`, μ≈0.04) so the spring dominates
  ground drag deterministically.
- The closing spring is a `post_step` external force computed entirely in WORLD
  coordinates from the housing's live heading (applied `is_global=True`, damping on the
  full velocity vector), and the tray's sleep threshold is zeroed — a sleeping body
  ignores applied wrenches and a kinematic removal of the carton raises no wake event.
- All judged quantities are live body-frame readbacks: drawer gap = tray root x in the
  housing frame (flush stop is physical); carton-on-mat in the pad frame at ground
  height; bowl-on-roof in the housing frame. The oracle teleports only the carton
  (kinematic hold with g·dt compensation); the drawer's closure is always real
  spring-driven dynamics — never written.
- Rubric: 0.2 latched once the carton has ever left the tray volume (post_step latch),
  +0.25 drawer closed (live), +0.35 carton settled on the mat (live), capped 0.8;
  1.0 iff success (all of it + bowl still on top). Null policy = exactly 0 (the jam is
  static — its 240-step stability is itself a check).

## Smoke check list (18)

1. reset settles finite, score 0, jam engaged (drawer held open by the carton)
2. randomization is real: housing xy + yaw and mat xy vary by readback (3 seeds)
3. initial drawer opening varies across 8 resets (slack readback)
4. null policy: 240 steps, drawer never closes, score exactly 0
5-7. oracle reaches success() on seeds 0 / 1 / 2
8. extraction latch pays exactly 0.2 the moment the carton leaves the tray (seed 0)
9. drawer-closed milestone reads exactly 0.45 with the carton still in hand (seed 0)
10. rubric strictly increases through extract → closed → delivered on every seed
11. negative A (seed skill 1): sustained 3x-spring shove never closes the jammed drawer
12. negative B (seed skill 2): carton parked on the cabinet top → closes, but no success
13. negative C sanity: extraction to the mat completes to success / score 1.0
14. negative C: knocking the bowl off the top voids success (score falls to 0.8)
15. near-miss: carton on the floor just beside the mat does not count
16. near-miss: 20 mm short of flush is not closed; authored flush is closed
17. calibration: jam gap = carton + panel thickness (band) on every oracle seed
18. calibration: autonomous closure completes ≤ 400 steps, ending under closed_tol
