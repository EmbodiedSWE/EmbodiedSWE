# put_umbrella_in_umbrella_stand_i72 — hang the umbrella on the rail by its crook

Scene `umbrella_rail`, env `simgen.umbrella_rail` (robot="null").

## Seed provenance

Derived from **rlbench/put_umbrella_in_umbrella_stand**: an umbrella and a floor
stand (both mesh visuals, franka, no checker); the implied strategy is transport +
**vertical insertion** — grab the umbrella, point its tip DOWN, and plunge it into
the open tube of the stand so it ends up standing upright in a receptacle.

## Strategic difference

No receptacle and no insertion anywhere. The fixture is a horizontal **RAIL** (a
steel bar between two posts) and the umbrella carries a **J-shaped CROOK handle** —
real open-arc geometry, not a scripted attachment. The goal state is a **suspension
equilibrium**: the crook hooked over the bar, the umbrella released and hanging
FREELY, carried only by crook-on-bar contact, its tip clear of the floor.

A solver needs a different plan and different code, not new constants:

- orient the **handle UP** (the seed points the tip down);
- approach the bar **laterally from above** and lower the open hook over it (the
  seed plunges into an aperture);
- **release and let go** — the last stage is a hands-off pendulum settling to a
  ~14° tilted equilibrium, which insertion never has;
- the rubric is a hook-containment predicate (crook arc center vs bar segment) plus
  a **ground-clearance** test — the exact opposite of the seed's success geometry
  (the seed WANTS the umbrella resting on the floor in a stand; here an umbrella
  standing on the floor is the canonical FAILURE).

A decoy CANE (straight stick + ball knob, no hook) lies on the other side of the
workspace; success and score are judged on the umbrella by identity.

## Solution outline (solve.py — teleports for transport only)

- **P0** settle + layout readback; assert score ≈ 0 (rubric-leak guard).
- **P1 CARRY (transport)**: teleport the umbrella from the ground to a hover pose —
  crook UP, hook plane across the bar, arc mouth 50 mm ABOVE the bar; asserted
  `engage_dist > engage_tol` (the bar starts OUTSIDE the hook; nothing touches).
  Ten regulated zero-velocity hold steps (gravity feed-forward + PD at the CoM).
- **P2 HOOK (contact dynamics)**: a velocity-regulated vertical force (target
  −0.10 m/s, force at the CoM only — no torque, no orientation pinning) lowers the
  umbrella so the bar passes through the open mouth of the crook; the wrench is
  DROPPED the instant the bar reads inside the arc. The catch is pure contact.
- **P3 HANG (hands off)**: gravity swings the umbrella to its pendulum equilibrium
  (~14° shaft tilt); damping settles it; +2 s extra hands-off margin.
- **P4 persistence**: ≥ 3.3 more simulated seconds hands-off with success() checked
  every substep, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing (0 → 0.25 → 0.60 →
1.0 → 1.0). Passes on seeds 0, 1, 2 on the forge.

## Rubric

`success()` = engaged ∧ crook_up ∧ suspended ∧ settled, judged on the umbrella:

- **engaged**: crook arc center within `engage_tol` = 30 mm of the bar SEGMENT. A
  real hang reads 23–25 mm; every outside-the-arc perch reads ≥ 49 mm (post-top
  perch, shaft drape ~150 mm) — margins asserted in `cfg.__post_init__`.
- **crook_up**: shaft within 30° of vertical (free hang ≈ 14°, margin asserted).
- **suspended**: canopy tip > `clear_min` = 50 mm above the floor (the anti-seed
  clause: a standing/leaning umbrella reads ~0 and is rejected). A true hang reads
  ~120 mm (margin asserted).
- **settled**: stillness SUSTAINED for 30 consecutive substeps (counter in
  `post_step`) — an instantaneous velocity threshold fires at every pendulum
  turning point; the counter latch rejects a swing (observed live, then fixed).

`score()`: latched 0.25 (ever raised to rail height) + 0.35 (bar ever inside the
arc), cap 0.60; 1.0 iff success(). Null policy ≈ 0 (asserted: the lift latch cannot
fire for an umbrella standing on the ground).

## Embodiment argument (Franka)

The umbrella exposes ~28 cm of bare 20 mm-diameter shaft between canopy and crook —
a natural power grasp anywhere along it. From a base at the origin facing the rack
(nominal (0.42, 0), bar at 0.62 m — inside Franka's envelope), the arm: grasps the
lying shaft, lifts and reorients crook-up (a wrist rotation), positions the hook
above the bar, lowers ~8 cm, and opens the gripper. The release-and-hang stage
needs no dexterity at all — the task's difficulty is perception (bar pose, which
object has the hook) and the hook-over-bar alignment, not force control.

## Execution order

1. `scene.py` written first (geometry + minimal rubric).
2. `solve.py` iterated on the forge: run 1 exposed the turning-point settle bug
   (success fired mid-swing, persistence failed); fixed with the sustained-stillness
   counter + stronger canopy damping + 2 s extra hands-off settle. SUCCESS on seeds
   0, 1, 2.
3. Rubric finalized against the demonstrated hang (23 mm engage, 120 mm tip
   clearance, 14° tilt).
4. `smoke.py` rejection battery: **ALL PASS 13/13** on the forge, frames.npz saved.

## Smoke checks (13)

1. settle/no-NaN + layout sanity; 2. score ≈ 0 at reset; 3. rack pose randomization
readback; 4. item poses + side-swap readback; 5. null policy ≈ 0; 6. seed-strategy
analog (umbrella stood/leaning at the rack, tip on ground) rejected by suspension;
7. horizontal drape across the bar rejected (eng_d ≈ 150 mm, not crook-up); 8.
post-top perch rejected by engagement (eng_d ≈ 49 mm > 30 mm despite hanging
crook-up); 9. cane draped on the rail counts for nothing (identity); 10. settle
gate — hooked but swinging is not success (and eng_d ≈ 23 mm anchors the tolerance);
11. latched credit survives teleport-away (0.60 kept, engaged drops); 12. rejection
audit (success never True in the battery); 13. final no-NaN.
