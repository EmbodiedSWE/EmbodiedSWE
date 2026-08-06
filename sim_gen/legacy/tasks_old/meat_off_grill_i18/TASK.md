# sear_and_serve (meat_off_grill_i18) — cook each patty in a time window, then serve it

**Env name:** `simgen.sear_and_serve` (scene `sear_and_serve`, robot `null`)
**Tier:** medium — 3 stages per patty (load the sear pad → timed dwell inside the cook
window → serve to the plate), with up to 2 patties processed **sequentially** through a
one-slot pad: 3–6 stage executions per episode.
**Execution order:** partially ordered and REQUIRED — each patty must cook before it may be
served, the pad physically holds only one patty at a time (so with two patties the schedule
is cook A → serve A → cook B → serve B, either patty first), and every patty carries a hard
deadline: once its accumulated pad time reaches `burn_time` it is spoiled forever.

## Seed provenance

- Seed: `rlbench/meat_off_grill`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/meat_off_grill.py`)
- Seed plan: the chicken/steak already sits ON the grill; a Franka grasps it, lifts it off,
  and puts it down beside the grill. One free pick-and-place, judged (in RLBench) by a
  final position detector. Time plays no role; the grill is only a start pedestal.

## What changed, and why it is strategically different

Kept: the object vocabulary (a grill, a steak, a chicken piece) and the "meat comes OFF the
grill" moment as one step of the plan.

Changed — the required plan, not the parameters:

1. **The judged quantity is a contact-time integral, which the seed never had.** Patties
   start RAW on a prep board (not on the grill). Success requires each present patty to
   accumulate ≥ `cook_min` = 1.5 s of real, settled contact on the grill's sear pad — and
   to be REMOVED before its accumulated pad time reaches `burn_time` = 4.0 s, a permanent
   spoil latch. Transport (the seed's whole task) is the trivial part here; the skill being
   tested is *scheduling*: when to put meat on, when to take it off.
2. **The seed's own plan is an explicit failing control, twice.** (a) Perfect transport of
   the raw meat straight to the serving plate — exactly the seed's grasp-carry-place —
   scores **exactly 0** (raw meat earns nothing; smoke negative control A). (b) The seed's
   literal move, "grab the meat off the grill promptly", is the near-miss control: a patty
   pulled off at 0.6 × `cook_min` is not cooked and earns nothing beyond the pad latch.
3. **Overshooting fails too.** Unlike any pick-and-place, patience is punished: past
   `burn_time` the patty is spoiled *permanently* — plating a burned patty is worthless and
   the episode can no longer reach success (smoke negative control B, incl. a terminality
   probe). The rubric is deliberately non-monotone only on this branch; the oracle path
   never burns and walks strictly increasing milestones.
4. **The one-slot pad forces true sequencing.** The sear pad (130 mm square) is sized so
   two non-overlapping patties can never both be fully inside it (max both-inside centre
   separation 63.6 mm < touching distance 85 mm — asserted at config time and probed
   physically in the smoke). With two patties present, a solver must interleave two cook
   windows through one shared resource instead of batching.

Axes deliberately NOT reused from sibling tasks: no low-clearance/confinement (i4, i9), no
contents-vs-container pour (i2, i5), no impulse/momentum shove (i1), no reorientation-
upright (i3), no bounciness/perch (i10), no construction (i8), no weight-on-plate hold
(i6 — its hold is passive and force-based; here the clock is active, bounded on BOTH sides,
and per-object with a shared one-slot resource).

## Physics honesty

The oracle places patties kinematically (teleport-oracle), but everything judged is
physical: the cook clock accumulates per physics substep only while the patty's live,
settled pose rests flat on the pad (position, height and uprightness read back from sim);
"served" is a current-state predicate (settled flat inside the plate disc, cooked, not
burned); success is the conjunction over the sampled patty subset. Randomization is real
and verified by readback: board/grill/plate xy-jitter + yaw, per-patty slot jitter, and
1-vs-2 patty subset sampling (the absent patty parks in an off-camera depot).

## Rubric (graded, latched)

Per present patty: 0.20 once it has ever rested on the pad (latched), 0.55 once cooked
(latched via the cook-time integral), 1.00 while served; burned caps the patty at 0.05
forever. Score = 0.9 × mean over present patties; forced to 1.0 iff success (all present
patties served); 0.0 for doing nothing and for the seed's raw-delivery plan.

## Smoke check list (17)

1. reset settles finite with score 0
2. randomization is real (grill+plate move on readback, yaw changes)
3. patty-count subset sampling varies (both 1 and 2 seen)
4. null policy scores ~0, cooks nothing, no success
5–7. oracle reaches success() on seeds 0/1/2 (final score exactly 1.0, nothing burned)
8. oracle milestone scores strictly increase on every seed
9. oracle always unloads the grill ≥ 1 s before burn_time
10. guaranteed two-patty episode: sequential schedule walks 6 milestones to 1.0
11. negative A: seed strategy (raw meat straight to the plate, settled — sanity-asserted
    via geometric on_plate) scores 0
12. negative B: overcooked patty is spoiled — plating it earns ~nothing
13. negative B: the burn latch is terminal
14. near-miss: patty pulled off the grill too early is not cooked, no serve credit
15. one-slot pad: two patties can never both be on the pad (dry math + stacking probe)
16. calibration: cook clock tracks real pad time (rate ~1 s/s)
17. calibration: window boundaries behave (raw at 0.5×cook_min / cooked at 1.2×cook_min /
    burned past burn_time)
