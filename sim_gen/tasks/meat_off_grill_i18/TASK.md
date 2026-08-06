# sear_and_serve (meat_off_grill_i18) — cook each patty in a time window, then serve it

**Env name:** `simgen.sear_and_serve` (scene `sear_and_serve`, registered robot `null`;
solve.py builds its own Franka env)
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
   probe). The rubric is deliberately non-monotone only on this branch; a correct schedule
   never burns and walks strictly increasing milestones (demonstrated by solve.py).
4. **The one-slot pad forces true sequencing.** The sear pad (91 mm square) is sized so
   two non-overlapping patties can never both be fully inside it (max both-inside centre
   separation 48.1 mm vs touching distance 57 mm: both-fully-inside would require ≥ 8.9 mm
   of interpenetration — asserted at config time and probed physically in the smoke). With
   two patties present, a solver must interleave two cook windows through one shared
   resource instead of batching.

## Embodiment feasibility (designed in, then demonstrated)

- Patty diameters 60 / 54 mm — the proven Franka parallel-jaw pinch sizes (80 mm max
  aperture); squat cylinders never roll, so board/plate placements persist.
- Base-polar layout around `stage_anchor` = (-0.55, 0): board 0.52 m at bearing −61°,
  grill 0.55 m at 0°, plate 0.52 m at +61°. Worst-case (jitter + slot + yaw) grasp/place
  radii stay in 0.40–0.64 m, inside the measured ground-level envelope; fixture
  separations exceed circumradius sums + worst-case jitter, so fixtures never overlap.
- The cook dwell needs no re-grasp: the solver holds the patty against the pad through
  the window (the clock is a geometric on-pad clause), so the burn deadline is met by
  construction. Grill top at 108 mm keeps fingertips 11 mm clear of the pad at hold.

## Solution outline (solve.py — the feasibility certificate)

Franka on the ground at base (-0.55, 0, 0), OSC, franka_session-lineage kernel (kp 220/600,
nullspace_dof_pos=(), ramped 1.2 s close, width-band + rise lift verdict, dewind).
Per present patty (steak then chicken, sequentially — the pad is one-slot):

1. top-down pinch of the disc on the board (grip 26/22 mm, width gates 50–70 / 44–64 mm);
2. closed-loop carry on the PATTY xy to the live pad centre at 0.20 m transit height;
3. lower until the scene's own `on_pad()` holds, then HOLD against the pad until the real
   cook clock crosses `cook_min` + 0.6 s (≈ 2.1 s ≪ `burn_time` 4.0 s), lift straight off;
4. carry to a free plate slot, lower to rest height, slow release, retreat; wait for
   `served()` (settled + flat + cooked + unburned + on the plate).

`SIM_GEN_SCORE` printed at start / on-pad / cooked / served per patty / final — all
latched or persistent, monotone along this trajectory. Passed on seeds 0 and 1 (and the
subset variation they sample) in this session's forge runs.

## Physics honesty

solve.py commands ONLY arm + gripper joints; it never writes task-object state. Everything
judged is physical: the cook clock accumulates per physics substep only while the patty's
live pose rests flat on the pad (position, height and uprightness read back from sim);
"served" is a current-state predicate (settled flat inside the plate disc, cooked, not
burned); success is the conjunction over the sampled patty subset. smoke.py's teleports are
probe CONSTRUCTORS (instrumentation for rejection tests) — no smoke probe reaches
success(). Randomization is real and verified by readback: board/grill/plate xy-jitter +
yaw, per-patty slot jitter, and 1-vs-2 patty subset sampling.

## Rubric (graded, latched)

Per present patty: 0.20 once it has ever rested on the pad (latched), 0.55 once cooked
(latched via the cook-time integral), 1.00 while served; burned caps the patty at 0.05
forever. Score = 0.9 × mean over present patties; forced to 1.0 iff success (all present
patties served); 0.0 for doing nothing and for the seed's raw-delivery plan.

## Smoke check list (16 — rejection-only; solve.py is the acceptance proof)

1. reset settles finite with score 0
2. randomization is real (grill+plate move on readback, yaw changes)
3. patty-count subset sampling varies (both 1 and 2 seen)
4. null policy scores ~0, cooks nothing, no success
5. negative A: seed strategy (raw meat straight to the plate, settled — sanity-asserted
   via geometric on_plate) scores 0
6. near-miss: patty pulled off the grill at 0.6×cook_min is not cooked, no serve credit
7. negative B: overcooked patty is spoiled — plating it earns ~nothing
8. negative B: the burn latch is terminal
9. near-miss: patty resting 25 mm off the pad centre (outside fully-inside slack) never
   starts the cook clock
10. wrong surface: patty on the grill body beside the pad accrues nothing
11. wrong pose: a COOKED patty lying on its side on the plate is never served
12. wrong place: a COOKED patty delivered back to the prep board is not served
13. one-slot pad: two patties can never both be on the pad (dry math + stacking probe)
14. calibration: cook clock tracks real pad time (rate ~1 s/s)
15. calibration: window boundaries behave (raw at 0.5×cook_min / cooked at 1.2×cook_min /
    burned past burn_time)
16. video frames captured and saved to frames.npz
