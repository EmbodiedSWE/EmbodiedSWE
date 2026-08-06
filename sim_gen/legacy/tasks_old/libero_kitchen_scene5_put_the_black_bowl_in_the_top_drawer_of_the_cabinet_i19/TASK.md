# barred_drawer — unbar the locked drawer, stow the black bowl, shut it away (i19)

**Registered as:** `SCENES["barred_drawer"]`, env `sim_gen.barred_drawer` (robot="null",
scene-level). **Tier: medium — 4 stages, execution order REQUIRED** (mechanically
enforced prefix: unbar → open → stow; shut must come last).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.py`)
- Seed plan: slide the cabinet's top drawer open (nothing resists the pull), pick up the
  black bowl, drop it inside. The seed's checker passes with the drawer **open** — its
  success bbox rides the drawer joint, so "bowl sitting in the pulled-out drawer" is the
  terminal state, and the drawer is freely openable from step one.

## What changed, and why it is strategically different

Same object vocabulary (a cabinet with one sliding drawer + a black bowl on top), but the
seed's plan is broken at BOTH ends and a longer, order-constrained plan is required:

1. **The drawer is mechanically locked by a removable security drop-bar.** A loose red
   batten rests in two open-top cradles bolted to the cabinet face, spanning the drawer
   front. The block is pure contact physics — the drawer face presses the free bar into
   the cradle front walls and stops after a *measured* ~34 mm (asserted < 60 mm in the
   smoke). The solver must first perform a **vertical tool-extraction** (lift the bar
   ~5-8 cm straight up out of its cradles) and park it clear — a prerequisite
   *unlock-the-mechanism* stage with no counterpart in the seed, executed on an object
   that is neither the goal item nor the container.
2. **The goal state is inverted at the far end: success requires the drawer SHUT again
   with the bowl inside.** The seed's own terminal state — bowl resting in the OPEN
   drawer — is an explicit, tested negative control here (score pinned at 0.60, never
   success). The added closing stage is real manipulation-with-payload: the drawer is
   pushed home with the bowl riding inside, retained by contact/friction dynamics, and
   the stowed bowl ends hidden under the counter ("put away", not "dropped in").

So the solver's plan is *unlock → open → stow → shut* instead of *open → stow*: a
different plan skeleton (tool removal + articulated-mechanism closure with payload), not
different parameters. Sibling-strategy axes deliberately avoided: no contents-vs-container
inversion (i2/i5), no low-clearance slide-vs-lift (i4/i9), no stack-to-height (i11), no
goal-conditioned rotary positioning (i7), no spring/weight mechanism (i6), no transit
tunnel (i1).

## Stage count / execution order

- **4 stages (medium):** (1) lift the bar clear of the cradles; (2) slide the drawer open
  past `open_min`; (3) place the bowl upright into the basin; (4) push the drawer fully
  shut with the payload inside.
- **Order REQUIRED and mechanically enforced:** the drawer cannot open while barred
  (contact block, measured); the basin is unreachable while shut (covered by the counter,
  faced by the plate — verified gaps are smaller than the bowl in every direction); the
  shut can only meaningfully happen after the stow.

## Scene / physics honesty

- Fully procedural compound spawners (pen_holder pattern): kinematic carcass (panels,
  counter, two bar cradles), dynamic one-piece drawer on a bind-time PrismaticJoint
  (pair collision ENABLED — the closed stop is the face plate meeting the panel fronts;
  the open stop is the joint limit = the rail; symmetric limits = the GPU sign hedge),
  free red bar, black octagonal bowl.
- All judged quantities are physical readbacks: drawer displacement from body poses in
  the carcass frame; bowl containment geometric in the drawer body frame (tolerance
  honest by construction: max physically-inside centre offset 54 mm < 60 mm gate; a bowl
  on the rim or counter is far outside); success additionally needs upright + shut +
  everything settled. The oracle opens/closes the drawer with honest 2-3 N body-frame
  forces (never pose-writes through the bar).
- Rubric (score in [0,1]): 0.15 latched bar-cleared, 0.35 latched drawer-opened, 0.60
  live bowl-in-basin, 0.85 stowed upright + shut (settling), 1.0 iff success; ~0 for
  doing nothing.
- Randomization: whole cabinet assembly (carcass + drawer + bar under one planar
  transform, so the joint is undisturbed) gets xy jitter + / - 5 cm and yaw + / - 30 deg;
  the bowl spawns independently on the counter with its own jitter + free yaw.

## Smoke checks (15)

1. settle/no-NaN (drawer shut, bar barred, score 0)
2. randomization-is-real (readback across seeded resets)
3. null policy scores ~0
4-6. oracle x 3 seeds: full unbar-open-stow-shut -> success + score 1.0
7. rubric monotone milestones 0 -> 0.15 -> 0.35 -> 0.60 -> 1.0
8. negative A — the seed's opening move: pulling the BARRED drawer (oracle's own force)
   moves it < 60 mm and scores ~0
9. negative A2 — same pull with the bar removed opens the drawer (the bar is the only
   blocker; validates the force convention)
10. negative B — the seed's goal state: bowl in the OPEN drawer is NOT success (0.60)
11. negative C — upside-down bowl in the shut drawer: in_basin but never success
12. near-miss D — drawer ajar 70 mm (> 30 mm tolerance) with bowl stowed upright: not
    success
13. near-miss E — bowl on the counter directly above the shut basin: no credit
14. calibration — bar held at +20 mm lift still bars the drawer
15. calibration — bar held at +80 mm lift clears it (+50 mm midpoint published)
