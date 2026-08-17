# pour_from_cup_to_cup_i100 — TipDumpScene (`simgen.tip_dump_station`)

Drain every ball from a trunnion-mounted tipping hopper into the green catch basin by
**pressing and holding the rocker pedal on the basin's side**, then release so the
return spring swings the hopper back upright.

## Seed provenance

Seed: `rlbench/pour_from_cup_to_cup`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/pour_from_cup_to_cup.py`) — "pour the
liquid (balls) from the source cup into the target cup", with look-alike distractor
cups. Its plan is **grasp → carry → wrist-tilt pour → set down**: transport plus
reorientation of a *held free-standing vessel*; the only decision is which cup is the
target.

## What changed and why it is strategically different

The theme kept: gravity pours loose contents from a source vessel into a target
vessel. **The manipulation model is replaced wholesale:**

| | seed | this task |
|---|---|---|
| source vessel | free cup, grasped and carried | hopper trunnion-mounted to a stand — cannot be grasped, lifted, or carried |
| pour actuation | wrist reorientation of the held cup | **press-and-hold a spring-loaded rocker pedal** against a return spring, past a geometric ~57° drain angle |
| temporal profile | one smooth pour arc | **sustained hold** (let go early → the spring slams the hopper shut with the balls inside), then a **release endgame** (success requires the hopper back upright at rest) |
| decision | pick the right target cup among look-alikes | **direction commitment**: two pedals; the wrong one dumps the balls onto the bare floor — an irrecoverable ~0, not a recoverable mis-aim |
| contents transfer | cup held over the target | **ballistic**: balls roll over a low spout lip and fly a real arc into the basin; the solver never touches a ball |

A solver needs a different plan (mechanism actuation with hold + release phases and a
side choice) and different code (spring-hinge plant monitoring, drain detection), not
different numbers on the seed's grasp-carry-tilt plan.

Also distinct from the sibling tasks viewed this session: `close_grill_i8` (structure
*building* — stacking a crossed crib), `open_oven_i7` (precision rotary *positioning*
onto detented targets — here there is no goal angle to park at; the skill is a
sustained hold above a threshold plus a content-transfer wait and a release),
`block_pyramid_i42` (stacking), `pen_holder` (pick-and-insert).

## Scene

Fully procedural (compound spawners, authored MassAPI CoM, friction materials bound to
every collider). Kinematic stand (pillars + bearings, pivot at z = 0.30); dynamic
hopper on an authored Y-axis revolute joint (±68° stops) with a return spring +
damping plant in `post_step` (`tau = press − 1.2·θ − 0.30·ω`, discrete-stability
audited at 120 Hz); low ±x spout lips (9 mm vs 20 mm ball radius → drain angle
`atan(√(r²−(r−h)²)/(r−h)) ≈ 57°`, asserted in `__post_init__`); rocker crossbar with
two yellow pedals at x = ±0.155 on the near side; kinematic deep-walled basin
teleported per episode to the +x or −x side; 2–4 dynamic balls (velocity iters 4
against the GPU sphere-creep artifact; restitution 0).

**Randomization (readback-verified in smoke):** basin side ±x, basin xy jitter,
present-ball count 2–4 + slot permutation + jitter, hopper start tilt ±3°. The stand
is world-fixed because the trunnion joint anchors to a kinematic body (anchors stay
world-fixed after teleports — known IsaacLab quirk), so episode diversity lives in the
basin/balls/tilt, all jointless or pure joint-coordinate re-poses.

`__post_init__` asserts every geometric claim: drain-angle band, latch < drain <
hold < limit, pedal-vs-basin-rim clearance at the joint stop, ballistic capture window
(zero-speed dribble clears the near wall; fastest far-side roller lands inside the far
wall), lip hop-out speed, spring holdability within the press bound.

## Rubric (anchored in the demonstrated solve)

- `0.15` — **tip latch** (latched): hinge ever passed 55° **toward the basin side**
  (signed — a wrong-side dump never arms it);
- `0.60 · frac` — live fraction of present balls inside the basin;
- `1.0` **iff `success()`**: every present ball settled inside the basin AND the
  hopper released back upright at rest (all live physical clauses); non-success caps
  at 0.75.

Null policy ≈ 0 (spring holds the hopper upright; nothing drains — verified in smoke).
Latched credit cannot evaporate; the frac term is live but balls in the deep basin
stay there (walls retain the landing slam — rim-corner vault geometry avoided).

## Teleport-contract solution (solve.py)

**The teleport budget goes unused** — no root pose of any task object is ever written.
The only interface is the scene's own `press_tau` hinge-torque input, self-clamped to
2.2 N·m (< the scene's 2.5 N·m physical bound; ≈ 14 N at the 155 mm pedal — a
one-finger press, only ~1.5× the ~1.45 N·m the spring demands at the hold angle).

- **P0 settle** — 1.5 s hands-off; layout readback; assert score ≈ 0, no success.
- **P1 tip** — feedforward+PD press (target slewed at 40°/s) on the basin-side pedal
  until the hinge passes 58° (past the 55° latch and the ~57° drain). `SIM_GEN_SCORE ≥ 0.15`.
- **P2 hold** — hold at 63°; if stragglers remain after 3 s, rock the tilt ±4° at
  1.2 Hz (hinge-axis modulation — the axis that changes the roll-out force) until every
  present ball is in the basin. `SIM_GEN_SCORE ≥ 0.75`; success asserted **False** while held.
- **P3 release** — zero the press; the overdamped spring returns the hopper upright
  hands-off; wait for the scene's own `success()`. `SIM_GEN_SCORE = 1.0`.
- **P4 persist** — ≥ 3.5 more simulated seconds with the press buffer asserted zero;
  `SIM_GEN_SOLVE: SUCCESS` only if `success()` still holds live.

Printed score sequence is asserted non-decreasing.

## Embodiment argument (single Franka)

Base pose: **(0.0, −0.55, 0)**, facing the station. Both pedals sit at
(±0.155, −0.115, ≈ 0.30) — 0.46–0.58 m reach, comfortably inside the Franka envelope,
on the robot's side of the station (the crossbar is on the −y face).

- **Pedal press**: close the gripper and press the 50 mm yellow pedal down with the
  fingertip side; the pedal needs ~9–14 N and sweeps a 155 mm-radius arc dropping
  ~13 cm over 63° — a shallow arc the wrist tracks while the elbow stays clear of the
  basin (asserted: the pedal never enters the basin walls even at the ±68° stop).
- **Hold**: sustained downward force well inside Franka payload; no regrasp needed.
- **Release**: retract vertically; the spring does the rest.
- No ball is touched by the intended strategy. (A hand-transfer of balls into the
  basin would also be judged fairly by the rubric — proportional frac credit, success
  on the same settled end state — but is strictly harder than the pedal.)

**Execution order**: only the causal chain press → hold-past-drain → wait-for-transfer
→ release is required; no arbitrary ordering constraints are imposed. Either pedal
side may be required depending on the episode's basin side.

## Checks (smoke.py — 10)

1. settle/no-NaN (present balls in hopper, absent parked, hinge vertical, score ~0);
2. randomization readback differs across seeds (basin, ball_0, start tilt);
3. coverage: both basin sides ≥ 2× and ≥ 2 distinct ball counts over 10 resets;
4. null policy: 240 idle steps → score ~0, no success;
5. **sub-drain tip** (verified ≥ 40°, quasi-static 8°/s approach — the ~57° drain
   angle is a static threshold; a jerked tilt can slosh a ball over the lip earlier,
   as with a real hopper): nothing spills, score ~0; spring returns it;
6. **wrong-side dump** (verified ≥ 55° away): balls on the bare floor, zero in basin,
   latch never arms, score ~0 — the direction commitment is irrecoverable;
7. **held-open near-miss**: all balls drained but pedal still held → success refuses,
   score ≤ 0.75 (release endgame is load-bearing);
8. **overshot spill**: balls settled just past the basin's far wall → score ~0;
9. **partial drain**: k−1 of k balls in basin → proportional 0.60·(k−1)/k, no success;
10. frames.npz video.

**Seed-strategy end state: N/A** — the seed's naive strategy (grasp the source vessel,
carry it over the target, pour) has no analog: there is no carryable vessel; the
hopper is joint-mounted to the stand. The nearest wrong-outcome analogs (wrong-side
dump, overshot spill, held-open) are checks 6–8.

## Verification

- `solve.py` on the forge, seeds 0, 1 and 2 (both basin sides, a 4-ball episode):
  `SIM_GEN_SOLVE: SUCCESS`, monotone scores.
- `smoke.py` on the forge: `SIM_GEN_SMOKE: ALL PASS 10/10`, frames.npz saved.
