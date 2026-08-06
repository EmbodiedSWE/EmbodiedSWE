# bowl_quarantine — sort the two bowls into a divided tray; they must NEVER touch

**Env name:** `simgen.bowl_quarantine` (scene `bowl_quarantine`, robot `null`)

## Seed provenance

- Seed id: `libero_90/living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.py`
- Seed plan: stack akita_black_bowl_1 ON akita_black_bowl_2 (xy < 6 cm, 0 < dz < 5 cm),
  then place the nested pair inside the wooden tray's contain region. The seed's whole
  strategy is **combine the two objects, then transport them together in one trip**;
  stacking is both a required subgoal and the transport optimization.

## What changed, and why it is strategically different

Kept: two bowls and a tray as the only task objects; the same *surface* goal — the
episode ends with both bowls inside the tray.

Changed — the PLAN, not the numbers:

1. **The seed's core verb becomes the forbidden hazard.** The bowls are now a red
   "raw" bowl and a blue "clean" bowl under a mutual-exclusion constraint: the moment
   their keep-apart envelopes intersect (nesting, rim-stacking, side contact — the
   envelope is a padded bounding cylinder, ~12.7 cm center distance at similar heights),
   a permanent `touched` contamination latch caps the score at 0.05 *forever*.
   Separating them and finishing the job perfectly afterwards does not restore credit
   (tested). The seed's own strategy — stack, then carry the stack to the tray — is the
   explicit failing control (checks A1–A3).
2. **One trip becomes two mandatory independent deliveries.** The tray is split by a
   center divider into two color-keyed compartments (red pad at tray-local +x, blue at
   −x). Success = each bowl settled upright on the floor of its OWN compartment, with
   the latch never fired. Geometry backs the rule up so the rubric is honest by
   construction: two bowls inside one 160 mm compartment can be at most ~52 mm apart —
   always inside the ~127 mm contact envelope — while bowls in adjacent compartments
   are at least ~153 mm apart, always outside it (dry-asserted in check 17). There is
   no legal single-trip plan.
3. **Sorting replaces stacking.** Bowl→compartment correspondence is color-keyed, the
   tray's yaw (which carries which side is red) is free-sampled, and the bowl→spawn-slot
   assignment is shuffled per episode, so the solver must read the scene; a memorized
   trajectory pair mis-sorts (mis-sorting scores only a 0.10/bowl consolation, tested).

A solver that transfers the seed's policy (bring the bowls together, move them as a
unit) scores ≤ 0.05 here; a solver of this task must plan two separated transports with
an explicit spatial-exclusion constraint — a different plan, not different parameters.

## Difficulty tier / stages

**easy** — 2 stages (deliver red bowl, deliver blue bowl), **no required execution
order** (either bowl may go first; the constraint is spatial, not sequential). Single
skill: pick-and-place into a walled compartment, made non-trivial only by the standing
keep-apart rule.

## Rubric

- 0.05 per bowl ever lifted above 10 cm (latched transient achievement);
- 0.35 per bowl **currently** settled upright on its own compartment floor (judged in
  the tray's body frame — yaw-robust);
- 0.10 consolation per bowl settled in the wrong compartment;
- exactly 1.0 iff success (both delivered + latch never fired);
- capped at 0.05 forever once `touched` latches; doing nothing scores exactly 0.

Physics are honest: the oracle places kinematically (paced pin-writes), but success and
score judge only physically settled poses (settle-gated, upright-gated, floor-height
gated), and the contamination latch is evaluated every physics substep on real positions.

## Randomization

Tray xy jitter (±4 cm) + free yaw (±180°, carries the red-side heading); bowl→slot
assignment shuffled; per-bowl xy jitter (±3 cm) + free yaw. Spawn slots are ≥ 24 cm
apart worst-case (readback-asserted > envelope + 2 cm every seed), so no episode starts
contaminated.

## Check list (smoke battery, 17 checks)

1. settle/no-NaN, score ~0 and no success at reset;
2. randomization is real (READBACK: tray pose/yaw spread, slot assignment flips both
   ways, spawn separation always safe, score 0 at every reset);
3. null policy fails (240 idle steps → score ~0, no latch);
4–6. teleport-oracle reaches success() with score 1.0 on 3 seeds (two separate paced
   deliveries);
7–8. rubric monotonicity (lift red → deliver red → lift blue → deliver blue strictly
   increases to exactly 1.0; partials < 1.0);
9. negative A1 — the SEED's strategy step 1: rim-stacking the bowls trips the latch;
10. negative A2 — the seed's terminal state: the carried stack parked inside the tray
   stays capped ≤ 0.05, no success;
11. negative A3 — latch permanence: bowls afterwards separated into BOTH correct
   compartments (geometric goal state fully achieved) still capped ≤ 0.05, no success;
12. negative B — both bowls mis-sorted without touching → no success, consolation band;
13–14. near-miss — red bowl upside-down in its correct compartment rejected by the
   upright gate; righted in place → delivered;
15. calibration A — kinematic approach sweep: contamination cliff measured at the
   configured ~127 mm envelope;
16. calibration B — placement tolerance: floor offsets 0/12/18 mm all deliver, ground
   rest beside the tray does not, sweep stays untouched;
17. geometry honesty (dry, from cfg): one compartment provably cannot hold two
   untouched bowls; adjacent compartments provably can; placement slack ≥ 12 mm.

## Differentiation from sibling generated tasks (claimed-axes check)

Checked `simgen-batch-task-strategies` memory + sibling TASK.md files. The claimed axes
here — **mutual-exclusion / keep-apart constraint between the two graded objects (the
seed's combine-and-carry verb becomes the permanent hazard)** and **same surface goal,
forbidden seed plan (forced plan decomposition into separated deliveries, keyed by
color)** — are claimed by no sibling. Nearest neighbours: i32 `shell_cover` claims a
*hands-off single target* (one object must not MOVE; here both objects must move and
the constraint is pairwise proximity, not immobility); i25 claims anti-carry judging
for toppling; i34 claims an anti-transport stray latch (object must stay put; here
transport is required, only co-location is fatal); i28 claims occupancy/buffer move
sequencing (no keep-apart rule, and its conflict is about goal slots, not object
pairs); i43/i16/i2 (bowl/plate family) claim rack-loading role-swap and
contents-vs-container axes. Several sibling packages were in-flight with empty TASK.md
at authoring time (i38, i40, i44–i47, i49–i51) and could not be cross-checked; the
axis claim above is recorded in the shared strategy memory to prevent collisions.
