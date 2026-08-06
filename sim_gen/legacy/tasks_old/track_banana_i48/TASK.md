# sweeper_gauntlet — time three crates through two patrolling sweepers

**Env name:** `simgen.sweeper_gauntlet` (scene `sweeper_gauntlet`, robot `null`)
**Tier: HARD — 9 sub-stages** (3 crates x [timed crossing 1 -> timed crossing 2 ->
basket deposit]). **Execution order:** per crate the two corridor crossings are ordered
by geometry AND by the rubric (the corridor-2 transit latch is gated on the corridor-1
latch); order ACROSS crates is free. Declared as: partial order required.

## Seed provenance

- Seed: `pick_place/track_banana`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_banana.py`)
- Seed semantics: stage-3 **trajectory tracking**. The banana starts *already grasped*
  (initial states loaded from a pkl of grasp states), five XFORM waypoint markers show a
  prescribed aerial curve (waypoints rising to z 0.36-0.47) toward a basket,
  randomization is explicitly zeroed, the gripper is forced closed every step, and the
  episode terminates if the object leaves the gripper. Reward = waypoint
  approach/progress + rotation tracking. The whole skill is **smooth, continuous
  path-following with a held object in a completely STATIC world** — time does not
  matter, only geometric progress along the given curve.

## What changed, and why it is strategically different

The surface situation is kept (fruit crates must end up in a basket across the room);
every load-bearing pillar of the seed is inverted, so a solver needs a different PLAN,
not different parameters:

1. **Static world -> adversarial clockwork.** The scene itself drives two kinematic
   sweeper blocks that patrol two corridor strips between staging and basket, every
   physics substep (triangle wave; period AND phase sampled per episode, independent per
   corridor). Nothing in the seed moves unless the robot moves it; here the central
   difficulty IS the autonomous environment motion, which the solver cannot influence.
2. **Path-following -> synchronization.** There is no prescribed path and no waypoint
   markers. The required skill is *timing*: observe a patrol rhythm, wait for a safety
   window, then dash the crate low across the strip; then do it again for a second,
   differently-phased strip; three times over. The solve is a sequence of discretely
   timed wait/dash decisions — the opposite of the seed's continuous constant-progress
   tracking. The median strip between corridors is an explicit safe harbor, so the
   plan has stage structure (cross 1, re-synchronize, cross 2).
3. **The seed's own strategies are the tested failing controls.**
   - *Negative A:* a phase-blind constant-speed ground carry straight across (the
     seed's tracking pace, no timing logic) is **guaranteed** to meet the sweeper — its
     ~4.1 s corridor exposure exceeds half the slowest patrol period (2.3 s), so a
     sweeper always arrives (dry-computed over 160 phase/period combos; measured on the
     forge). The permanent per-crate `struck` latch caps that crate's credit at 0.02
     forever — even after a later PERFECT basket placement, and even after a clean
     window-timed re-run of the same crate (latch permanence is asserted).
   - *Negative B:* the seed's aerial waypoint arc (carry high over the strips — its
     waypoints rise to z 0.36+) trips the permanent `flew` latch (crate above 0.12 m
     while over a corridor strip): the lofted shortcut that trivially defeats
     ground-level hazards is refused despite a perfect end pose. The same latch kills
     riding across on a sweeper's roof.
4. **Nothing is pre-grasped, and the goal is judged physically.** Crates start settled
   on the floor; `success()` requires all three resting settled inside the basket
   interior, each with per-substep transit latches (`t1`,`t2`) proving it physically ran
   both corridors low and unstruck — a teleport into the basket earns nothing
   (negative C).

Strategy axes claimed (vs. sibling generated tasks): **synchronize-with-autonomous
periodic environment motion / timing-window crossing**; **hazard latches make timing
errors irreversible**; **repeated re-synchronization** (two independent rhythms x three
crates). Distinct from i20 (one-shot interception deadline of a moving OBJECT), i18/i21
(dwell-duration windows on a static station), i36 (solver-actuated indexing mechanism),
i37 (carrier-mediated crossing latch): here the environment moves BY ITSELF, forever,
periodically, and the solver's only lever is WHEN to act.

## Rubric (graded score, 1.0 iff success)

Per crate: 0.10 once corridor 1 is transited (latched), 0.18 once corridor 2 is
transited (latched, gated on t1), 0.30 while delivered (in basket, settled, both
transits, no hazard); `struck`/`flew` permanently cap that crate at 0.02. Sum + 0.10
success bonus; exactly 1.0 iff `success()`, exactly 0 for doing nothing. Milestone
sequence dry-computed strictly increasing: 0.10, 0.18, 0.30, 0.40, 0.48, 0.60, 0.70,
0.78, 1.00.

## Randomization (verified by readback)

Sweeper patrol periods (3.2-4.6 s each) and phases, crate spawn jitter + yaw, basket
position (x band + lateral side/offset). A memorized fixed schedule fails: the safety
windows move per episode.

## Check list (smoke battery, 17 checks)

1. settle/no-NaN + score 0 at reset; 2. sweeper drive real + matches analytic patrol
(readback); 3. randomization readback (scatter, basket, phases); 4. null policy ~0;
5-7. oracle success + score 1.0 on 3 seeds; 8. milestone monotonicity (9 milestones);
9. negative A: phase-blind constant-speed carry struck; 10. negative A: perfect basket
pose refused; 11. negative A: latch permanence after a clean re-run; 12. negative B:
seed's aerial arc -> `flew`, refused; 13. negative C: teleport-to-basket not delivered
(anti-teleport transit latches); 14. near-miss: clean gauntlet, set down beside the
basket -> t2 credit only; 15. near-miss recovery -> delivery credit; 16. calibration:
mistimed dash (sweeper approaching) struck 3/3; 17. calibration: window-timed dash
clean 3/3.

## Physics/honesty notes

All assets procedural (primitives + one compound basket spawner). Sweepers and walls
kinematic; crates dynamic (max_depenetration 0.5, light damping). The hazard judgment is
geometric (footprint overlap + 8 mm face pad), so it fires identically on free and
kinematically-held crates and never depends on solver pop. Timing feasibility is
guaranteed by construction and asserted in cfg `__post_init__`: at the fastest period a
0.72 m/s dash's danger exposure (0.52 s) fits inside the worst post-pass safe run
(0.86 s); at the slowest period a 0.06 m/s carry cannot fit between passes. The fly cap
(0.12 m) leaves 70+ mm of headroom over a legal ground dash and sits 48 mm below a
crate riding a sweeper.
