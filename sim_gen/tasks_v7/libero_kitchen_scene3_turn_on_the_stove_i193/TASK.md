# libero_kitchen_scene3_turn_on_the_stove_i193 — turn on the pellet stove by FUELLING it: feed 3 briquettes through the one-way stoker flap, keep the butter out

## Provenance

Seed: `libero_90/libero_kitchen_scene3_turn_on_the_stove` — rotate the flat
stove's knob past a joint-angle threshold
(`states.objects["flat_stove"].joint_pos[:, 0] > 0.5`); moka pot and frypan
sit around as untouched distractors.

## Strategic difference from the seed

The seed is a *single fixture actuation read directly from a fixture joint*.
This task keeps the "turn on the stove" story but replaces what "on" *means*,
how it is achieved, and how it is judged:

- **No knob, no fixture joint angle anywhere in the rubric.** The stove is
  cold because its **firebox is empty**: "turn it on" = **fuel it** — at
  least 3 of the 4 briquettes must physically end up INSIDE the closed
  firebox. Success is a **cumulative multi-object containment outcome**
  (positions of free bodies), never a threshold on any articulation.
- **The only way in is a one-way mechanism traversal.** The firebox's top is
  a grate whose 16 mm gaps deny any drop-in (briquettes are 32 mm); the
  stoker port is covered by a gravity flap hinged at the top whose revolute
  LIMITS make it strictly one-way (in: yes, out: denied by the closed stop).
  Fuel must be pushed *along a chute, through the flap, over a 26 mm sill* —
  a sustained guided push, not a grasp-and-turn. Feeding is **irreversible**,
  which is also what makes the latched count honest and monotone.
- **A live exclusion clause.** The butter stick *fits through the port*
  (asserted in cfg), so "keep the butter out" is a real constraint the
  policy could violate, not decoration: butter inside multiplies the score
  by 0.2 and blocks success.
- **The naive seed plan scores ~0:** there is nothing to rotate, and piling
  fuel ON the stove (the surface-placement reading) leaves everything on the
  grate — rejected by the inside window (smoke check 5).

Distinct from the corpus tasks I inspected: i159 (this seed family) is a
tool-mediated momentary *press* (piezo striker down a guard shaft) plus a
placement — no containment, no one-way mechanism, single payload. i141
weighs hidden masses on a two-pan balance; i75 hangs a pan on a wall hook
(suspension); i148 extracts threaded rings along an L-rail into a bin
(starts constrained, constrained *extraction*); i101 unplugs a twist-lock
plug (cord tracing, rotation-then-pull). None judges *feeding N-of-M objects
through a one-way flap into a closed vessel with a same-port-sized
contaminant to exclude*.

## Scene

Fully procedural (native PhysX box colliders; the flap rides a spawn-authored
per-env revolute X joint whose LIMITS [−0.5°, +88°] are the one-way stops; the
kinematic stove is the joint anchor and never moves):

- **counter** (kinematic): 800 × 640 × 140 mm bench.
- **stove/firebox** (kinematic, FIXED at (−0.08, +0.10)): 200 × 180 × 170 mm
  box, 8 mm walls; 46 × 46 mm stoker port in the front wall (sill 34 mm up,
  26 mm above the interior floor); 5-bar top grate (24 mm bars, **16 mm
  gaps**); loading **chute** running 146 mm south from the port — channel
  exactly port-wide (46 mm) between two **14 mm** lips (low enough to set a
  gripped briquette down inside), floor level with the sill; lamp + orange
  trim above the port (visual only; lamp recolors when properly fuelled).
- **flap** (dynamic, 20 g): 56 × 5 × 52 mm plate hanging from a hinge at
  z 84 mm, 1 mm inside the front wall; covers the port past the sill;
  holding it open costs < 0.1 N (≪ a briquette's push).
- **4 fuel briquettes** (dynamic): 32 mm charcoal cubes, 50 g.
- **butter stick** (dynamic): 50 × 32 × 30 mm pale-yellow block, 40 g —
  passes the port with clearance (cfg-asserted).

Randomization (readback-verified in smoke): the 5 items are **permuted over
5 apron slots** (`argsort(torch.rand)` — never a first `randint`) at
x ∈ {−0.28, −0.14, 0, 0.14, 0.28}, y = −0.20, with ±12 mm xy jitter and free
yaw each.

~25 honesty asserts in `StokerStoveSceneCfg.__post_init__` pin the claims:
port/chute pass a briquette with ≥ 10 mm clearance; the butter passes too
(exclusion is live); the flap overhangs the port, reaches past the sill,
hangs clear of the wall, opens far enough for a briquette, and is featherweight
to hold; grate gaps deny briquette AND butter with margin, grate spans the
whole cavity; sill ≥ 20 mm above the interior floor (escape denial) and the
closed stop blocks outward swing; the judged inside window sits strictly
inside the cavity (above the floor, below the grate, wall margin smaller than
a half-briquette); success needs ≤ n−1 briquettes; slots clear the chute,
each other and the counter edge at any jitter/yaw; stove and chute fit the
counter.

## Rubric

Physical containment, no software click latch:

| credit | clause |
|---|---|
| 0.25 × k (k ≤ 3) | k briquettes counted inside: **latched inside-and-calm AND currently inside** |
| × 0.2 | while the butter is inside the firebox (spoilage) |
| 1.00 | `success()` live |

`success()` = count ≥ 3 AND butter NOT inside AND every counted briquette
calm (|v| < 0.05) AND all bodies finite. `score()` = 1.0 **iff** `success()`;
monotone along the solve because fed briquettes physically cannot come back
out (sill + one-way flap), and the count cannot over-read (counting requires
*currently inside*, the latch only adds the calm-history requirement).

## Solution outline (solve.py — teleport-transport contract)

All load-bearing interaction is contact dynamics; nothing is ever written
inside the firebox:

- **P0** settle; assert mass readbacks (custom spawners!), flap shut, firebox
  empty, items on the apron row, score ≈ 0.
- **P1–P3 FEED × 3**: teleport briquette k (transport only) flat into the
  chute channel 100 mm short of the port (asserted: not inside, satisfies
  nothing). Then a **velocity-servo push** (external force, kv·dt/m = 0.42,
  cap 0.55 N escalating to 1.5 N only on stalls; lateral centering ±0.15 N;
  anti-spin torque kw·dt/I = 0.39) slides it through the flap — the
  briquette's nose rotates the flap about its real hinge — over the sill,
  where gravity topples it in. Wrench ZEROED, 90 hands-off settle steps.
  Asserted per feed: flap max angle > 15° (mechanism non-vacuous), count
  increments, butter still out, score non-decreasing (0.25 → 0.50 →
  0.75/1.0). 3-attempt re-transport retry per briquette.
- **P4** ≥ 3.3 s fully hands-off persistence (10 × 40 steps, success at every
  checkpoint) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed non-decreasing at every phase boundary. Verified on
the forge for seeds 0 and 1 (`--args "--headless --seed 1"`).

## Embodiment argument (Franka)

- **Briquette pick**: 32 mm cube on an open apron row (≥ 116 mm to any
  neighbour at worst jitter) — canonical top-down pinch with the 80 mm jaw,
  48 mm of free height above the 14 mm chute lips.
- **Set-down into the chute**: the channel is 46 mm wide but its lips are
  only 14 mm tall, so cube + two 10 mm fingers descend from above without
  side-wall interference; release at the chute start (±7 mm lateral slack).
- **Feed push**: close the jaw and push the cube along the channel with the
  fingertips — the hand stays outside the port (the cube's trailing face
  needs to reach only ~40 mm past the outer wall face; fingertips extend
  ~50 mm below the hand). The channel self-guides; the flap needs < 0.1 N.
  No precision beyond the 46 mm channel entry; forces ≈ 0.5 N.
- **Butter**: never touched.
- All contacts lie 140–200 mm high on an 80 × 64 cm bench — comfortable for
  a Franka based at ~(−0.08, −0.55) facing +y: apron row at reach ~0.35 m,
  chute run 0.25→0.11 m ahead of the base.

## Execution order

No mandatory order: any 3 of the 4 briquettes, in any sequence, with pauses —
the count is cumulative and irreversible, and the butter clause is
order-free (it must simply never be in the firebox at success). Declared
module order:

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the feed trajectory
   and the monotone score trace 0.00 → 0.25 → 0.50 → 1.00.
2. `smoke.py` — rejection battery, 13 checks:
   1. settle/no-NaN (flap shut, firebox empty, 5 items on the apron row,
      score 0);
   2. randomization readback (pellet0 xy + yaw, butter xy differ across
      seeds);
   3. permutation: butter occupies ≥ 3 distinct slots over 8 resets;
   4. null policy 240 steps (score ≤ 0.01, flap does not creep);
   5. SEED-NAIVE: 3 briquettes set ON the stove top rest on the grate —
      none inside, score ~0;
   6. grate drop-in denial: a dropped briquette is never inside at any
      substep and rests ON the bars (height readback);
   7. near-miss: 2 in + 1 settled at the port mouth → count 2, score 0.50,
      no success;
   8. CONTAMINANT: butter in FIRST (no prefix satisfies the goal), then 3
      briquettes in → spoilage refuses, score ≤ 0.16;
   9. butter-only inside → score ~0;
   10. ONE-WAY flap: an inside briquette force-pushed outward at port height
       (gravity cancelled so the sill wall can't shadow the flap) provably
       reaches the flap, the closed stop holds, it never exits, still
       counted after release;
   11. MECHANISM: one briquette servo-fed the solve's way — flap swings
       > 15° (actuator provably moved), count 1, score 0.25;
   12. audit: success was never True at any smoke step;
   13. frames.npz video.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS <n>/<n>`) and hard-exit; a top-level try/except
prints a FAIL verdict on any exception and a watchdog Timer kills the
process on a hang, so the forge never waits out the timeout.
