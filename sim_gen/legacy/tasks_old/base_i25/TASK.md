# base_i25 — `simgen.topple_order` ("ordered demolition")

## Seed provenance

- Seed id: `pick_place/base`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/base.py`
- Seed plan: grasp a single free cube, **carry it along a prescribed 5-waypoint
  free-space trajectory** (dense tracking rewards, per-waypoint bonuses), and hold it at
  the final waypoint. One object, one prehensile transport, success = trajectory
  completed while grasping.

## What changed, and why it is strategically different

Every load-bearing element of the seed is inverted — a solver needs a different **plan**,
not different parameters:

| | seed (pick_place/base) | this task (topple_order) |
|---|---|---|
| verb | grasp + carry (prehensile transport) | push past the tipping point (aimed instability) |
| what moves | the object translates along a path | **nothing translates** — every judged object must stay on its own footprint and change only standing → lying |
| gravity | the adversary (fight it the whole carry) | the actuator (the solver only crosses the ~10.9° critical angle; gravity does the work) |
| trajectory | prescribed waypoints for the carried object | no path at all — what is prescribed is a **fall heading** (outward sector) per pillar |
| structure | one object, one stage | three pillars, **strictly ordered** stages + a standing constraint (the vase) that must hold throughout |
| failure | drop the cube (recoverable) | knock the vase (a **permanent** loss latch) |

The seed's own strategy is expressible here and is a tested failing control: kinematically
carrying the red pillar off its base and laying it down in a *geometrically flawless* end
pose (lying, in sector, on its footprint, settled) earns **zero** — the anti-carry latch
(`uprooted`: translated/lifted while upright) plus the crossing-time base-home clause
reject transport, so only in-place falls count. Same-strategy-different-numbers cannot
pass: there is no grasp-and-track plan that scores here.

Axes not reused from siblings (checked against `simgen-batch-task-strategies` memory +
sibling TASK.md files): no low-clearance/confinement (i4, i9), no stacking/support
building (i11, i17), no size-keyed matching or hanging (i12), no container role reversal
(i13), no interception deadline (i20), no dwell-time window (i18, i21), no pouring (i5),
no bounce-gentleness cliff (i10). The claimed axes here are **ordered directional
toppling** (aimed instability, work *with* gravity), **nothing-is-transported judging**
(anti-carry latch), and a **fragile-bystander constraint** (permanent collateral-damage
latch). The vase-loss latch is mechanically similar to i20's "lost can" latch but is a
side constraint, not the task's axis.

## Task

A demolition range: three colored pillars (red "first", amber "second", blue "third" —
50×50×260 mm boxes, tipping angle atan(25/130) ≈ 10.9°) stand on an arc; a slender
porcelain vase stands inside the arc within falling reach of one sampled slot. Fell the
pillars **in color order**, each falling **outward** (heading within ±45° of its own
outward radial), each landing on its own base spot (centre within 0.30 m), all settled,
vase still standing.

- **Tier: medium — 3 ordered stages + 1 standing constraint.**
- **Execution order: REQUIRED** (red → amber → blue, enforced by a down-crossing latch
  that only credits the next expected pillar).
- Randomization: color→slot permutation (memorized motion sequences fail), slot azimuth
  ±8° / radius ±2 cm jitter, free pillar yaw, sampled guarded slot + vase jitter.
- Rubric: 0.30 per credited pillar (latched `felled_ok` × live lying/sector/footprint/
  settled), 1.0 iff success, 0 for doing nothing, pinned 0 forever if the vase breaks.
- Physics honesty: the oracle only teleport-tips a pillar 25° (just past critical, base
  kept on its pedestal, 1 mm release drop); the judged outcome — the fall, the lying
  pose, the heading, the vase's survival — is produced by gravity and contacts.

## Check list (smoke battery, 17 checks)

1. reset settles finite, score 0, all standing, vase intact
2. randomization is real by readback (pillar homes + target azimuths move)
3. color→slot permutation and guarded slot vary across resets
4. null policy: 240 idle steps → score 0, no latches, no success
5–7. oracle reaches `success()` with the vase alive on seeds 0/1/2
8. oracle milestones strictly increase (0.30 / 0.60 / 1.00) on every seed
9. negative A (**seed's own strategy**): carry + lay down with a perfect end pose →
   anti-carry latch voids it, score 0
10. negative B: out-of-turn amber earns nothing (even lying in its sector); red still
    earns its 0.30 afterwards
11. negative B cont.: blue after the broken sequence earns nothing — score pinned 0.30,
    no success
12. negative C: felling the guarded pillar inward breaks the vase → score 0
13. negative C cont.: the loss latch is terminal
14. near-miss: red felled in order but 90° off-sector — mechanics latch sets, credit 0
15. tolerance twin: 35° off-target (sector 45°) still earns the 0.30
16. calibration, tipping cliff: initial tilts ≤8° recover upright
17. calibration, tipping cliff: initial tilts ≥14° fall (together bracketing the 10.9°
    critical angle); 10°/12° published

`frames.npz` (whole battery, viewport rgb) is saved to the CWD.
