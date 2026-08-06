# empty_the_bowl — pour the white bowl's contents into the sink basin

**Env name:** `simgen.empty_the_bowl` (scene `empty_the_bowl`, robot `null`)

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet.py`)
- Seed plan: grasp the (empty) white bowl, carry it LEVEL, set it down on top of an
  elevated cabinet. Success = the bowl's own pose inside a bbox on the cabinet top;
  the bowl's contents never matter.

## What changed, and why it is strategically different

Kept: the white bowl as the manipulated object, a kitchen-counter layout with an
elevated cabinet-like fixture, easy single-skill scope.

Changed (the plan inversion, not the numbers):

1. **The bowl starts FULL** (2–4 loose balls, count subset-sampled per episode) and the
   goal is judged on the **contents**: every present ball must end up settled inside a
   separate walled sink basin, and the bowl must end **back on the counter**, upright,
   empty, outside the basin.
2. **The core skill is controlled reorientation (tilt/pour)** over the basin — precisely
   the motion the seed's plan forbids (tilting the carried bowl is how you *fail* the
   seed). A solver that reuses the seed's plan — carry the level bowl to an elevated
   surface and set it down — executes flawlessly and scores ~0.1: the balls are still in
   the bowl. That exact strategy is present as a fixture ("cabinet ledge") and as
   negative control A.
3. **Containment honesty:** a ball counts only if it is inside the basin AND outside the
   bowl volume, so parking the loaded bowl inside the basin (the container cheat, i.e.
   the seed's place-the-bowl move aimed at the target region) delivers nothing
   (negative control B). A ball perched on the basin rim or dropped beside the tub does
   not count (near-miss controls).
4. Judging is final-state and strategy-agnostic: ball-by-ball transfer without ever
   lifting the bowl is an admissible alternative plan and reaches 1.0 (verified).

The seed judges *where the container is*; this task judges *what the container no
longer contains*. Same object, opposite plan.

## Difficulty / stages / order

- **Tier: easy — 2 stages**: (1) pour the balls into the basin; (2) set the empty bowl
  back down upright on the counter.
- **Execution order:** only causally constrained (the pour necessarily precedes the
  final set-down of an *empty* bowl); no order is enforced beyond the final settled
  state, and the ball-by-ball alternative ordering is equally valid.

## Rubric (score in [0,1])

- 0.1 latched once the bowl has ever been lifted (transient-achievement latch,
  updated in `post_step`)
- +0.6 × fraction of present balls delivered (settled in basin, out of the bowl,
  below the rim)
- 0.9 when ALL present balls are delivered (e.g. bowl still held)
- 1.0 **iff** `success()`: all delivered + bowl standing upright on the surface outside
  the basin, settled. Null policy scores 0.

## Randomization

Bowl xy+yaw, basin xy+yaw, ledge xy+yaw, and present-ball subset (2–4) every reset;
verified by state readback in the smoke.

## Smoke checks (17 — verified ALL PASS on the forge)

1. reset settles finite with score 0
2. randomization is real (bowl+basin move on readback)
3. ball-count subset sampling varies
4. null policy scores ~0 and no success
5. lift latch pays 0.1 before any delivery (oracle seed 0)
6. all-delivered-but-still-held reads exactly 0.9 (oracle seed 0)
7. oracle solve reaches success() on seed 0
8. oracle solve reaches success() on seed 1
9. oracle solve reaches success() on seed 2
10. rubric strictly increases ball-by-ball (alternative plan)
11. ball-by-ball alternative plan reaches 1.0 + success
12. negative A: the seed's own strategy (carry loaded bowl onto the cabinet ledge) fails
13. negative B: loaded bowl parked in the basin delivers nothing (container cheat)
14. near-miss: ball perched on the basin rim is not delivered
15. near-miss: ball on the table beside the basin is not delivered
16. calibration: first ball departs between 15° and 120° of tilt on every oracle seed
17. calibration: pour completes by 140° of tilt on ≥2/3 oracle seeds

Measured pour curve (forge run): first departure at 90–100°, pour complete at 110°
on all three seeds, zero recoveries — the bowl walls genuinely hold the payload until
a committed tilt, so the reorientation skill is load-bearing, not decorative.

Oracle: gravity-compensated kinematic hold of the bowl (teleport-oracle); all delivery
physics (balls rolling out, falling into the basin, settling) is real. Video recorded
to `frames.npz` in the working directory.
