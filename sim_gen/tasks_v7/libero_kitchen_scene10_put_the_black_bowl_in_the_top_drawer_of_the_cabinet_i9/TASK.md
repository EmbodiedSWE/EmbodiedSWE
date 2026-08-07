# Task `libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i9` — Carousel Airlock Feeder

Scene: `carousel_airlock` (env `simgen.carousel_airlock`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet`
— "put the black bowl in the top drawer of the cabinet": pick a black object off a surface
and deposit it inside an enclosed compartment of a fixture.

Kept from the seed: a BLACK target object that starts loose on the floor, a fixed fixture
with an enclosed interior compartment reachable only through a designed aperture, and the
core act of *putting the black object into the compartment*. A red distractor object plays
the role of the scene's other bowls (identity control).

## Strategic difference

**vs the seed:** In the seed, depositing the object into the compartment IS the task. Here,
depositing the black ball through the roof's loading window into the covered annulus is
demoted to a *worthless-alone first step* (0.20 credit, no success). The actual objective is
to operate a MACHINE: after loading, the agent must spin a rotor (by pushing its yellow
spokes — the only part of the mechanism that protrudes above the roof) so that an angled
orange plow vane sweeps the ball ~180° around the covered annulus, expels it through a side
discharge gap, and lets gravity carry it down a hooded chute into a separate green basin.
The seed's "place inside" verb becomes "feed a part through a rotary airlock" — a
transport-through-mechanism task with an internal state (rotor azimuth, swept angle) that
the agent cannot directly touch once the ball is inside.

**vs the rest of the corpus read this session:**
- `close_microwave_i4/i5` (articulated door closing) — no articulation here; the moving part
  is a free rigid rotor driven by external contact torque, and the object travels *through*
  the fixture.
- `libero_scene7_i2/i3` (tabletop pick-and-place / stacking) — nothing is stacked or placed
  at a pose; success is delivery through a covered mechanism.
- `peg_insertion_side_i1/i2` (tight-tolerance lateral insertion) — the only "insertion" is a
  55 mm ball dropped through a ~70° open roof sector; the challenge is downstream mechanism
  drive, not tolerance.
- `pick_single_egad_i3/i4` (grasp-centric object lifting) — grasping is trivial (single
  smooth sphere); credit comes from the machine, not the grasp.
- `setup_checkers_i2` (multi-object arrangement) — one target object, one distractor, no
  arrangement pattern.
- `light_bulb_out_i5` (unscrew/extract from a socket) — rotation here is of a *tool the
  agent drives*, not of the object; the object is never rotated about its own axis on
  purpose.
- `open_oven_i6` (articulated door opening) — again no articulated joint; and the goal is
  object delivery, not fixture reconfiguration.

No other corpus task couples "load through an aperture" with "drive a rotary mechanism to
convey the object somewhere else". That coupling — and the sweep-gated rubric that scores
the *pathway*, not just the end pose — is the strategic identity of this task.

## Scene summary

A kinematic HOUSING (randomized xy + free yaw each episode): circular floor + 24-segment
ring wall + 24-segment roof forming a covered annular well; the roof has a ~±35° open
LOADING WINDOW sector (housing +x), the ring wall has a ~±30° DISCHARGE GAP (housing −x)
under an overhanging hood, feeding a 6°-pitched walled CHUTE that ends over a GREEN walled
BASIN. A free dynamic ROTOR sits in the well: bottom disc (the floor the ball rides on),
angled ORANGE plow vane, central axle with a bearing collar riding in the roof's center
hole (3–8 mm slop — the rotor can only spin in place), and YELLOW crossed spokes above the
roof — the drive handle. A BLACK ball (55 mm, the target) and a RED decoy ball spawn loose
on the floor away from the machine.

`describe()` states the housing pose, window/gap azimuths in world terms, rotor/vane
azimuth, ball/decoy positions, the required direction of rotation (counter-clockwise seen
from above — the vane is angled to plow outward only in CCW rotation), and the full recipe;
it is sufficient to solve the task from alone.

## Rubric (latched, non-decreasing)

- 0.10 · approach latch (ball approaches the loading window; denominator d0 from spawn)
- 0.20 · load latch (ball inside the covered annulus — on the disc, under the roof)
- 0.30 · sweep latch (max swept azimuth |az|/π while in the annulus)
- 0.25 · transit latch — **gated**: only latches while the ball is in the discharge/chute
  corridor AND sweep_latch ≥ 0.75. A ball inserted into the corridor from outside (never
  swept) can never earn it, so "carry the ball around the outside and stuff it up the
  chute" scores ≤ ~0.15 and never succeeds.
- Success (score 1.0): ball settled in the green basin ∧ load latch ∧ transit latch ∧ decoy
  NOT in the basin. Base credit capped at 0.85 without success.

## Teleport solution (`solve.py`) — legitimacy certificate

- **P0** settle + layout readback (housing pose, vane azimuth, spawns, d0) — proves
  seed-dependent randomization from stdout.
- **P1 transport only:** the black ball is teleported to a hover pose 150 mm up in FREE
  SPACE above the loading window, at an azimuth inside the window chosen away from the
  parked vane (read from the scene). It is released; loading is the free fall through the
  window onto the rotor disc under gravity and real contact — the same release an arm
  performs. One retry at another in-window azimuth if the ball perches on the vane.
- **P2 mechanism drive:** a torque governor applies a bounded external torque about
  vertical to the rotor — τ = clamp(1.6·(1.2 − ω_z), ±0.35 N·m), i.e. the moment of a hand
  pushing a spoke tangentially with ~2.3 N at r ≈ 0.15 m. The vane sweeps the ball around
  the covered annulus, out the gap, and gravity delivers it down the chute into the basin.
  Torque is cut; everything settles on real contact. The ball is never teleported into the
  well, the corridor, or the basin.
- **P3 persistence:** hands-off ≥ 3.3 simulated seconds; `SIM_GEN_SOLVE: SUCCESS` only if
  success still holds and the score never decreased.

`SIM_GEN_SCORE` printed at every phase boundary; asserted non-decreasing. Verified on
seeds 0 and 7.

## Embodiment argument (Franka, single arm)

- The ball is a smooth 55 mm sphere on open floor — a standard top grasp (Franka max jaw
  opening 80 mm) at the spawn, unobstructed.
- Loading: hold the ball above the roof's open window sector (roof top at z ≈ 0.112, an
  annulus of open sky ~±35° wide at radius 0.05–0.157) and open the gripper — exactly the
  P1 release. No insertion tolerance: the window chord is ≥ 3× the ball diameter.
- Driving: the yellow spokes are at z ≈ 0.16 (world ≈ 0.16–0.19 with housing on the
  ground), 22 mm square bars extending to r = 0.15 — pushable tangentially with the closed
  gripper or graspable side-on. Sustaining ~1.2 rad/s needs ~2.3 N tangential at
  r ≈ 0.15; with 4 spoke ends, re-engaging every ≤ 90° of rotation keeps a push stroke
  ≤ 0.24 m of arc — comfortably inside a Franka workspace with the base ~0.55 m from the
  well center, which also reaches the ball/decoy spawn band (0.31–0.49 m from origin) and
  the window hover pose.
- Nothing requires exceeding ~5 N or reaching under the roof: the covered annulus is
  intentionally unreachable, which is the point of the mechanism.

## Ordering

Load-before-sweep is physically inherent (the vane can only sweep a ball that is in the
annulus), and the rubric's transit gate (sweep ≥ 0.75) makes the pathway order —
approach → load → sweep → transit → basin — the only scoring order. No arbitrary
ordering constraints beyond what the mechanism itself imposes.

Disclosure: an agent could in principle "bowl" the ball (launch it fast through the window
so it coasts around the annulus without driving the rotor). The parked vane blocks the
path at a random azimuth in most episodes, ball angular damping (0.30) kills coasting, and
such a strategy still exercises honest contact physics through the same pathway latches —
it is a hard alternate physics route, not a rubric bypass.

## Checks (`smoke.py`, 15)

1. settle/no-NaN + height readback; 2. baseline score ≤ 0.02; 3–4. randomization by
readback across 6 seeds (housing xy/yaw, rotor yaw, ball/decoy xy, d0); 5. null policy
≈ 0; 6. SEED-STRATEGY end state (honest load through the window, then stop): load latch
set but NOT success, score ≤ 0.45 — the seed task's outcome is worth less than half
credit here; 7. teleport-direct-to-basin settled: rejected (≤ 0.15); 8. up-the-chute
cheat (ball stuffed into the corridor from outside, verified in-corridor, rolls to
basin): transit latch stays 0, rejected; 9. wrong object (red decoy in basin): rejected;
10. near-miss outside the far basin wall: rejected; 11. near-miss parked ON the roof:
not loaded, rejected; 12. latched credit persists after ball removal; 13. approach
monotonicity (closer hover ⇒ strictly more credit); 14. rejection audit (no check ever
saw success); 15. final no-NaN. Frames recorded to `frames.npz`.
