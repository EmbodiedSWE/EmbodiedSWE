# drawer_stash (i22) — prop the self-closing drawer, stash the bottle, let it shut

**Registered as:** `SCENES["drawer_stash"]`, env `simgen.drawer_stash` (robot `"null"`, scene-level).
**Tier: medium — 4 stages** (open ≺ prop ≺ load ≺ shut). **Execution order REQUIRED**
(the prop can only go in while the drawer is held open; the bottle can only latch as
loaded through an open drawer; closing only counts after loading).

## Seed provenance

Seed: `libero_90/libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/..._bottom_drawer_of_the_cabinet.py`) —
a Franka opens a passive cabinet drawer, picks the wine bottle off the table, and places
it into the STANDING-OPEN drawer. Success = bottle centre inside the open drawer's bbox;
the drawer's articulation is inert (stays wherever it is put) and the task ends with the
drawer open.

## What changed, and why it is strategically different

The drawer is no longer passive furniture — it is an **adversarial mechanism**. The
drawer rides a per-env authored prismatic slide with a **return spring** (`f = −k·s −
c·ṡ` along the drawer's body-x, applied every substep): the moment nothing holds it, it
glides shut. The seed's plan skeleton — *open, then go fetch the bottle, then place it
into the waiting open drawer* — is *expressible and fails* here (negative control A):
by the time the bottle arrives, the drawer has sprung closed and the bottle lands on the
cabinet face.

A solver needs a different PLAN, not different parameters:

1. **Open** the drawer against the spring — and realize the opening is *transient*.
2. **Create a temporary fixture**: a yellow stop bar fits the floor gap between the
   drawer's oversized face panel and the cabinet plinth — the classic doorstop move.
   This is tool use whose purpose is to *hold state*, absent from the seed entirely.
3. **Load** the bottle **lying flat** — the cavity lintel (165 mm) is far below the
   bottle's standing height (185 mm + basin floor), so the seed's "set the bottle in"
   any-orientation placement is replaced by forced reorientation; an upright bottle is
   rejected by below-rim containment and physically jams the closing drawer on the
   lintel.
4. **Undo the fixture**: remove the bar and let the spring shut the drawer with the
   bottle riding inside.

The **terminal state is inverted** versus the seed: the seed leaves the bottle in an
*open* drawer (that exact state here scores 0.45 partial credit, never success); success
here requires the bottle **enclosed in the CLOSED drawer** — a goal state in which the
bottle is hidden from view.

Sibling-axis note: unlike i4/i9 (low-clearance forcing *sliding* along a surface), the
clearance here only forces *orientation*; the claimed axes are **transient-access
mechanism**, **temporary-fixture (prop) tool use with a required undo step**, and
**closed-containment terminal state**.

## Physics honesty

Procedural only (primitives + two compound spawners). The cabinet is 6 kinematic slabs;
the drawer is one dynamic compound body on a real USD prismatic joint (symmetric limits,
pair collision disabled, 5 mm real slide clearances with 3 mm contact offsets); spring
and damping are body-frame post_step forces.
The oracle drags/pins kinematically and teleport-places, but every verdict is physical:
the spring's self-close is measured (calibration a), the prop bar holds by real
compression between panel and plinth (calibration b), the bottle load is a real 26 mm
drop, and success reads settled poses plus the `loaded` transit latch — which only sets
while the bottle is inside an OPEN drawer without a same-substep teleport, so writing
the bottle straight into the closed drawer (negative control B) never succeeds.

## Randomization (per episode, verified by readback)

- Flank swap: the bottle spawns on the cabinet's left or right flank (sampled), the
  prop bar on the opposite flank — the approach geometry mirrors per episode.
- Bottle and prop bar: ±6 cm xy jitter + free yaw, clear of the drawer's sweep.
- The jointed cabinet+drawer pair itself stays at its authored pose: PhysX does not
  reliably re-anchor an authored joint's kinematic body0 frame after per-episode
  teleports (measured in probes — teleported-pair resets intermittently left the
  drawer squeeze-wedged against the cavity walls), so the mechanism is fixed and the
  world randomizes around it.

## Rubric (score in [0,1], transient achievements latched)

- 0.15 — latched once the drawer is ever pulled past 10 cm open.
- 0.45 — latched once the bottle is ever physically inside the open (≥ 6 cm) drawer.
- 0.45–0.80 — live while loaded, rising as the loaded drawer approaches closed.
- 1.0 — **iff** `success()`: loaded latch ∧ bottle fully below the basin rim (both
  endpoints, drawer body frame) ∧ drawer closed (< 2 cm) ∧ bottle + drawer settled.
- ~0 for doing nothing (nothing latches at reset; null-policy checked).

## Smoke check list (16)

1. settle: clean reset (finite, drawer closed on its spring home, score ~0)
2. randomization is real (cabinet yaw + bottle/prop poses, by readback)
3. null policy: score < 0.05, no success
4. calibration a: released empty drawer self-closes from every pull depth (times published)
5. calibration b: stop bar props the drawer open at 3 lateral positions (rest opens published)
6. opened latch: 0.15 latches on the pull and persists
7. rubric monotonicity across oracle stages
8. loaded partial credit: bottle in propped drawer → 0.45 ≤ score < 1, no success
9. near miss: prop left in → drawer cannot close → success rejected
10. oracle A (staged): success() with score exactly 1.0
11. oracle B: fresh seed success
12. oracle C: fresh seed success
13. negative A (seed strategy): open, release, deliver → drawer already shut, bottle
    outside, score ≤ 0.2, no success
14. negative B (anti-teleport): bottle written inside the closed drawer → no success, score < 0.1
15. upright bottle rejected by below-rim containment (jam aftermath published)
16. video frames recorded (frames.npz in CWD)
