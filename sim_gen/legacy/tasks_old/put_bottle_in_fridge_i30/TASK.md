# decant_and_chill — the bottle can't fit: repackage its contents for the fridge (i30)

## Seed provenance

- Seed id: `rlbench/put_bottle_in_fridge`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/put_bottle_in_fridge.py`
  (RLBench asset port: a rigid bottle + an articulated fridge; the demo trajectory
  grasps the bottle, carries it, and releases it inside the fridge — a prehensile
  transport-into-container task where the bottle itself is the goal object).

## What changed, and why it is strategically different

The seed's entire plan is **move the named object into the container**. Here that plan
is **measured-impossible by construction**: the bottle is a 260 mm open tube while the
fridge chamber's interior space diagonal is ~246 mm (`assert`ed in the scene cfg), so no
orientation of the bottle can ever be contained — sliding it in through the 120 x 110 mm
doorway (it does enter lying down; the doorway is not a wall) jams it with ~110 mm
protruding. A solver must discover a **repackaging plan** the seed never needs:

1. **Decant** — transfer the 3-5 loose balls out of the bottle into the squat amber jar
   that *does* pass the doorway (pour, or move them one by one — both admissible);
2. **Stow** — carry the loaded jar THROUGH the doorway and leave it standing upright,
   settled, fully inside (an anti-teleport **transit latch** requires a physical
   crossing of the doorway plane with small per-step motion: writes through the wall
   earn no stow credit and can never succeed);
3. **Recycle** — lay the now-useless bottle on the recycling pad (both ends over the
   pad, low, settled, outside the fridge).

The seed's goal object is demoted from "the thing to transport" to "a source container
to empty and discard"; the actual cargo (the contents) never appears in the seed at all.

Differentiation from sibling batch tasks (claimed-axes registry): i5 `empty_the_bowl`
and i2 `serve_ingredient` pour contents OUT as the terminal goal (extraction onto/into
an open target). Here the transfer is a *means*: the claimed axes are **capacity-driven
goal-object substitution** (the nominal object provably cannot achieve the goal in any
pose), **nested containment** (balls ∈ jar ∈ fridge), and **disposal of the original
container as a scored terminal stage**. No sibling claims the fit-impossibility →
repackage axis. The doorway admits the jar in any yaw (12.4 mm/side margin) — this is
capacity exclusion, NOT i14's orientation-selective aperture.

## Difficulty tier and stages

**Medium — 3 stages** (decant → stow → recycle), plus per-ball breadth inside stage 1
(3-5 balls, subset-sampled).

**Execution order: PARTIALLY required.** Decanting MUST precede stowing: the chamber
roof (interior 125 mm, doorway 110 mm) blocks pouring into a jar already inside, and the
bottle cannot enter tilted. Bottle-to-pad may interleave freely with the other stages.

## Rubric

- 0.05 — latched once the bottle is ever lifted (transient, `post_step`);
- +0.45 × fraction of present balls delivered (settled in the jar, not in the bottle —
  the in-bottle clause defeats "stand the full bottle inside the jar");
- +0.30 × delivered fraction once the jar is stowed (an EMPTY stowed jar pays zero);
- 0.95 — every present ball delivered AND jar stowed;
- 1.0 — iff `success()`: all delivered + jar stowed (transit-latched) + bottle on pad.
- Doing nothing scores exactly 0.

## Check list (17, smoke.py)

1. reset settles finite with score 0
2. randomization is real (fridge pose+yaw, bottle, jar — by READBACK)
3. ball-count subset sampling varies (8 resets)
4. null policy scores ~0, never succeeds
5. lift latch pays 0.05 before any delivery
6-8. teleport-oracle (kinematic pour + doorway slide) reaches success() on seeds 0/1/2
9. rubric strictly increases ball-by-ball (alternative no-pour plan)
10. all-delivered + stowed with bottle untouched reads exactly 0.95
11. alternative plan reaches 1.0 + success once the bottle rests on the pad
12. negative A — the SEED's own strategy: bottle slid into the doorway jams protruding
    (>20 mm past the plane), in_fridge false, score ≤ 0.06
13. negative B — stowing the EMPTY jar pays nothing (latch verified live, score ≤ 0.02)
14. negative C — anti-teleport: loaded jar written through the wall is geometrically
    inside but never transited → no stow credit, no success
15. near-miss — loaded jar left at the doorway threshold is not stowed
16. calibration — first-departure pour tilt within (91°, 168°) on every oracle seed
17. calibration — pour completes by 170° on ≥ 2/3 oracle seeds

## Physics honesty notes

- The oracle is a teleport/kinematic-hold oracle (gravity-compensated +g·dt re-pin), but
  every judged quantity is a settled physical outcome: balls really roll down the tube
  and drop into the jar; the jar really passes the doorway (13 mm head clearance,
  12.4 mm side margins, 2 mm contact offsets); the bottle really jams half-in.
- All assets procedural: octagonal compound-body vessels (pen_holder spawner pattern),
  kinematic compound fridge chamber + pad slab, plain spheres. No external files.
- Scene registered as `simgen.decant_and_chill`, robot="null" (scene-level task).
