# libero_kitchen_scene3_turn_on_the_stove_i280 — turn on the gas stove by LOADING its counterweighted safety weigh-beam: both steel ingots into the weigh basket, wood decoys are dead weight that isn't enough

## Provenance

Seed: `libero_90/libero_kitchen_scene3_turn_on_the_stove` — rotate the flat
stove's knob past a joint-angle threshold
(`states.objects["flat_stove"].joint_pos[:, 0] > 0.5`); moka pot and frypan
sit around as untouched distractors.

## Strategic difference from the seed

The seed is a *single direct fixture actuation judged from a fixture joint the
robot grasps*. This task keeps the "turn on the stove" story but the robot
NEVER touches the valve mechanism:

- **No knob, no fixture-joint readout anywhere in the rubric.** The gas valve
  is held shut by a counterweighted WEIGH-BEAM (a steelyard on a real
  revolute pivot atop the valve post): inner arm carries a cast
  counterweight, outer arm an open-top weigh basket. "Turn on the stove" =
  make the basket end sink past the ON angle **by loading mass** — a
  cumulative, statics-held weighing outcome judged from the beam's FREE-BODY
  root pose plus the ingots' beam-frame positions. No articulation joint is
  ever read.
- **Mass discrimination, not shape.** All four cubes are 40 mm; the two dark
  steel ingots weigh 400 g, the two pale wood decoys 30 g. The counterweight
  is sized (cfg-asserted with cube-at-wall worst-case lever arms) so BOTH
  steels out-torque it by ≥ 15 % at any in-basket placement while ONE steel
  plus BOTH woods falls ≥ 15 % short at the most favourable placement. The
  torque margins are angle-invariant by construction: the basket floor is
  sunk so a resting cube's CoM sits AT hinge height, so both lever arms
  scale with cos(angle) together.
- **Pressing the beam is not success — a live anti-actuation clause.** The
  seed's whole verb (push the fixture past a threshold) scores 0 here:
  success requires the known load *physically counted inside the basket* at
  the same time as the angle, and the empty/under-loaded beam pressed past
  θ_on reads False and swings back closed on release (smoke checks 5 and 6
  prove both, with an external hinge torque as the "hand").
- **The naive seed plan scores ~0**: there is no knob, and no single grasp
  or turn opens anything.

Distinct from the corpus tasks I inspected: i193 (same seed family) feeds
briquettes through a one-way flap into a closed firebox — containment via
mechanism traversal, with a contamination clause; here nothing is enclosed,
nothing is one-way, and the physics that satisfies the goal is *statics on a
lever*, with an anti-press clause instead of an exclusion clause. i159 (same
family) is a tool-mediated momentary piezo press plus placement. i141 weighs
UNKNOWN hidden masses on a symmetric two-pan balance to *identify* an odd
object; here the masses are known and color-coded, there is one basket, the
beam itself IS the valve, and the challenge is that the goal state must be
*reached and held* against a counterweight with decoys that can't do it.
i36 bridges a gap for a rolling ball, i13 carries a kettle to a trivet, i75
hangs a pan on a wall hook, i148 extracts rings along an L-rail, i101
unplugs a twist-lock plug — none judges loading N known masses onto a
counterweighted lever until statics swings and holds it open.

## Scene

Fully procedural (native PhysX box colliders; the beam rides a spawn-authored
per-env revolute Y joint whose LIMITS [−16°, +16°] are the hard stops; the
kinematic stove is the joint anchor and never moves):

- **counter** (kinematic): 800 × 640 × 140 mm bench.
- **stove fixture** (kinematic, FIXED at (−0.05, +0.10)): burner box
  (160 × 160 × 40 mm) with a black plate and a VISUAL flame ring that
  recolors when the valve is held open; brass supply pipe (visual); 44 mm
  valve post carrying the hinge 130 mm above the deck.
- **beam** (dynamic, 1.3 kg, authored CoM −74.5 mm on the inner arm,
  authored diagonal inertia): spine + counterweight + red flag + open-top
  basket at +160 mm (interior 64 × 96 mm, 45 mm walls, floor sunk so a
  resting cube's CoM is at hinge height). Empty closing torque 0.95 N·m.
- **2 steel ingots** (dynamic, 40 mm, 400 g, dark) and **2 wood decoys**
  (dynamic, 40 mm, 30 g, pale) — identical size, different mass.

Randomization (readback-verified in smoke): the 4 cubes are **permuted over
4 apron slots** (`argsort(torch.rand)` — never a first `randint`) at
x ∈ {−0.27, −0.09, 0.09, 0.27}, y = −0.20, ±12 mm xy jitter, free yaw.

~20 honesty asserts in `ValveBeamStoveSceneCfg.__post_init__` pin the claims:
2-steel worst-case torque ≥ 1.15 × closing; 1-steel + all-woods best case
≤ closing / 1.15; basket admits a cube with ≥ 10 mm clearance and walls
retain it; the judged in-window covers a seated and a stacked cube but stays
inside the walls; θ_on strictly between the stops with margin; the beam
sweep clears the deck, burner, and post at both stops; slots clear each
other, the stove, and the counter edge at any jitter/yaw.

## Rubric

Free-body pose + beam-frame containment; no joint readout, no click latch:

| credit | clause |
|---|---|
| 0.25 × k (k ≤ 2) | k steels counted in the basket: **latched in-and-calm AND currently in** |
| 1.00 | `success()` live |

`success()` = count ≥ 2 AND beam angle ≥ θ_on (+12°) AND beam calm AND every
counted steel calm AND all bodies finite. `score()` = 1.0 **iff**
`success()`. Monotone along the solve (deposits only add load; the loaded
beam only sinks). The count cannot over-read: counting requires *currently
inside*, so a bounced-out or removed ingot loses its credit (smoke check 8),
and a pressed beam with an under-loaded basket scores only its count.

## Solution outline (solve.py — teleport-transport contract)

All load-bearing interaction is contact dynamics; nothing is ever written
inside the judged basket window:

- **P0** settle 120 steps; mass readbacks (custom spawners!); beam ON the
  closed stop (−16°); basket empty; cubes on the apron row; score ≈ 0.
- **P1/P2 DEPOSIT × 2**: teleport steel k (transport only) to a RELEASE POSE
  in the air above the basket mouth — beam-frame (basket_x, ∓22 mm,
  in_z_hi + 12 mm) with the beam's own tilt, strictly OUTSIDE the judged
  window (asserted per drop: not in-basket, not counted, score unchanged) —
  then it free-falls in, strikes real walls/floor, settles. Side-by-side y
  offsets so the second never lands ON the first. After P1 the beam must
  STAY closed (one steel is not enough — asserted); after P2 the load
  out-torques the counterweight and the beam swings to the open stop under
  gravity alone — **no force is ever applied to the beam by the solve**.
  3-attempt re-drop retry per ingot (a retry, not a cheat).
- **P3** ≥ 3.3 s fully hands-off persistence (10 × 40 steps, success at
  every checkpoint) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed non-decreasing at every phase boundary:
0.00 → 0.25 → 1.00 → 1.00. Verified on the forge for seeds 0 and 1
(`--args "--headless --seed 1"`), with visibly different layouts.

## Embodiment argument (Franka)

- **Pick**: 40 mm cubes on an open apron row — worst-case edge-to-edge
  spacing ≥ 116 mm (180 mm slot pitch, ±12 mm jitter) — canonical top-down
  pinch with the 80 mm jaw; nothing overhangs the row.
- **Steel vs wood**: visually unambiguous (dark blue-grey vs pale tan), and
  a wrong pick is recoverable — wood in the basket costs nothing (no
  exclusion clause; it simply can't open the valve).
- **Drop-in**: the basket mouth is 64 × 96 mm — a 40 mm cube passes with
  ≥ 12 mm side clearance; release a few cm above the rim (~0.34 m up at the
  closed stop, over the basket at ~(+0.11, +0.10)) and gravity does the
  rest, exactly as the solve demonstrates. No insertion, no push, no
  precision beyond the 64 mm mouth.
- **The beam is never touched.**
- All contacts lie 0.14–0.35 m high on an 80 × 64 cm bench — comfortable for
  a Franka based at ~(+0.05, −0.50) facing +y: apron row at reach
  0.30–0.44 m, basket drop point ~0.60 m, both inside the 0.855 m envelope
  with height to spare.

## Execution order

No mandatory order: the two steels in either order, with pauses, and
dumping ALL FOUR cubes in also succeeds (the woods just ride along — the
clause set is order-free and superset-closed). Declared module order:

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the drop-in
   trajectory and the monotone score trace 0.00 → 0.25 → 1.00.
2. `smoke.py` — rejection battery, 10 checks:
   1. settle/no-NaN (beam on the closed stop, basket empty, 4 cubes on the
      apron row, score 0);
   2. randomization readback (steel0 xy + yaw, wood0 xy differ across
      seeds);
   3. permutation: steel0 occupies ≥ 3 distinct slots over 8 resets;
   4. null policy 240 steps (score ~0, beam does not creep);
   5. SEED-NAIVE press: the beam provably driven past θ_on by an external
      hinge torque with an EMPTY basket — success/score refused at every
      substep; released, gravity re-closes it;
   6. DECOY load: one steel + BOTH woods constructed in the basket — beam
      stays closed, count 1, score 0.25; pressing the under-loaded beam is
      STILL refused and it re-closes;
   7. wrong place: steels ON the burner plate and ON the counterweight arm
      → count 0, score 0;
   8. latch honesty: a seated (counted, 0.25) steel lifted back OUT loses
      its credit → count 0, score ~0;
   9. audit: success() was never True at any smoke step;
   10. frames.npz video.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS 10/10`) and hard-exit; a top-level try/except
prints a FAIL verdict on any exception and a daemon watchdog Timer kills
the process on a hang, so the forge never waits out the timeout.
