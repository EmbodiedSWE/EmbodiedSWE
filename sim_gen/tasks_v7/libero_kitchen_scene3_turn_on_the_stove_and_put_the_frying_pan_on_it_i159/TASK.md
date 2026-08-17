# libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i159 — click the recessed piezo igniter with the pan's own handle, then seat the pan on the plate

## Provenance

Seed: `libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it`
— rotate the stove knob past a joint-angle threshold (`joint_pos > 0.5`) and set
the chefmate frypan down on the flat, always-available burner (xy within
0.08 m, height 0–0.03 m over the cook region).

## Strategic difference from the seed

The seed is *fixture actuation read from a joint and held past a threshold* +
*free placement onto a static surface*. This task keeps the "turn on the
stove, then place the pan" story but replaces the actuation mechanism, the
actuation *reading*, and the role structure of the objects:

- **No knob, no joint-angle clause anywhere in the rubric.** "Turning it on"
  is a **momentary tool-mediated press**: the burner's piezo igniter is a
  spring-return striker sunk **86 mm** down a **48 mm** square guard shaft.
  Pressing it ≥ 8 mm for 3 substeps latches the burner LIT (the plinth lamp
  flips dark-red → orange and stays on); the spring pops the striker back and
  **nothing is held at success time** — ignition is a latched mechanism event
  produced by real striker displacement, not a fixture pose.
- **The payload IS the tool.** Nothing in the scene reaches the striker except
  the frying pan's own 150 mm stick handle (held handle-down, the pan's weight
  drives the press). The pan must be used *as an instrument first, then as the
  payload* — a role flip the seed (and the corpus) never asks for.
- **The distractor is provably useless, by geometry pinned in cfg asserts:**
  the kettle's 70 mm body cannot enter the 48 mm shaft, and its 24 mm lid knob
  dangles ~62 mm short of the striker (proven physically in smoke).
- **The naive seed plan scores ~0.15, never 1.0:** a pan seated perfectly on
  the cold plate satisfies every placement clause and still fails (no lit).

Distinct from the corpus tasks I inspected: i150 (same seed family, scene9)
turns on a fire-pit stove by a *two-body rail shunt with a self-aligning end
stop* — rails, trains, and a moved placement surface; here there are no rails,
no shunt, the cook plate is static from reset, and the mechanism is a
spring-return *press* reached only through a reach-denial shaft. i75 (scene3
sibling) *hangs* the pan on a wall hook and judges retention — no ignition
mechanism at all. No other task uses a tool-role flip of the payload object or
a momentary latched press as the "turn on".

## Scene

Fully procedural (native PhysX box colliders; the striker rides a
spawn-authored per-env prismatic Z joint whose LIMITS are the hard stops; the
kinematic igniter tower is the joint anchor and never moves):

- **counter** (kinematic): 780 × 620 × 150 mm bench.
- **igniter tower** (kinematic, FIXED at (0.26, 0.04)): 90 mm column, 150 mm
  tall, with a 48 × 48 mm guard shaft sunk to a chamber floor 110 mm down;
  orange mouth trim (visual).
- **striker** (dynamic, on the Z joint): 40 mm cap, 30 g, 12 mm travel; a
  scene-owned post_step spring (preload 0.8 N, k = 80 N/m, damped) holds it at
  the top stop. Cap top rests **86 mm** below the mouth (fingertip reach
  ~58 mm — denied). Click = depth > 8 mm for 3 substeps → lit latch.
- **burner** (kinematic, randomized xy ±35 mm about (−0.13, 0.08)): 160 mm
  plinth + 140 mm cook plate (top at z 188 mm) + south-face lamp (visual;
  recolored on lit).
- **pan** (dynamic, free): 110 mm square dish + 150 mm × 20 × 14 mm stick
  handle, 420 g; spawns on a random slot, handle-south yaw cone (−90° ± 40°).
- **kettle** (dynamic, free distractor): 70 mm body + 24 mm knob, 500 g;
  spawns on the other slot, free yaw.

Randomization (readback-verified in smoke): burner xy; pan/kettle **slot-swap
coin** (`torch.rand`) at (±0.150, −0.170); per-object xy jitter; pan yaw cone;
kettle free yaw.

~19 honesty asserts in `PiezoStoveSceneCfg.__post_init__` pin the claims:
kettle wider than the shaft; knob far short of the striker; handle enters with
≥10 mm clearance and reaches full press depth with 30 mm margin while the dish
(wider than the whole tower) stays above the mouth; striker recess beyond
fingertip reach; 0.7·pan-weight beats the full spring stack; spring preload
beats 1.5× striker weight (true return); click depth inside the stroke; plate
covers the judged window; the lift latch is unreachable by on-fixture
placements; spawn slots clear the tower, the burner and each other (the yaw
cone provably keeps the handle off the kettle); everything fits the counter.

## Rubric

Latched partial credit, capped at 0.55 unless success holds live:

| credit | clause |
|---|---|
| 0.15 | pan ever carried clearly off the bench — origin > 120 mm above deck (latched) |
| 0.25 | igniter ever clicked — striker depth > 8 mm for 3 substeps (latched; lamp shows it) |
| 0.15 | pan ever seated on the plate, calm (latched) — radial xy ≤ 0.040, bottom within 10 mm of plate top, upright < 12° |
| 1.00 | `success()` live |

`success()` = burner lit (latched mechanism state — the click physically
happened) AND pan seated on the plate (live geometry) AND pan settled
(|v| < 0.05 m/s) AND finite.

## Solution outline (solve.py — teleport-transport contract)

All load-bearing interaction is contact dynamics; the striker is NEVER
written by the solve:

- **P0** settle; assert masses readback, striker at top, burner off, pan
  upright on its slot, score ≈ 0.
- **P1 IGNITION** (gravity, hands-off): teleport the pan (transport only) to
  the carry pose handle-down over the shaft — `qy(+90°)`, handle axis on the
  shaft axis, tip 15 mm above the cap — and release. The pan falls and its
  own weight drives the striker past the click (3-attempt retry). Carry the
  pan away; assert the spring returned the striker (depth ~0) while lit stays
  latched → score 0.40.
- **P2** teleport the pan (transport only) to a hover 20 mm above the plate,
  handle south, and let it **drop and settle**; 3-attempt re-drop retry →
  success → score 1.0.
- **P3** ≥ 3.3 s fully hands-off persistence (10 × 40 steps, success at every
  checkpoint) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed non-decreasing at every phase boundary. Verified on
the forge for seeds 0 and 1 (`--args "--headless --seed 1"`).

## Embodiment argument (Franka)

- **Ignition press**: pinch the handle (20 × 14 mm section, a canonical 80 mm
  jaw grasp), invert the pan wrist-down, align the handle over the 48 mm
  mouth at z 0.30 (insertion tolerance ±14 mm — generous), lower ~110 mm and
  let the arm's compliance rest the pan's weight (~4 N) on the striker; the
  shaft self-guides the tip, and the wide dish physically cannot follow the
  handle in. No force beyond the pan's own weight; no precision beyond the
  ±14 mm mouth entry.
- **Placement**: same pinch grasp; carry ~45 cm and lower 2 cm onto the
  140 mm plate; the 40 mm radial window on a 140 mm plate gives real slack.
- All contacts lie 150–300 mm high inside a 78 × 62 cm bench — comfortable
  for a Franka based at ~(0.05, −0.55) facing +y, reach 0.20–0.45 m.

## Execution order

Press-before-place is the natural embodied order (the pan in hand does the
press, then goes to the plate) but is NOT decreed: the rubric accepts either
order — a pan seated first on the cold plate simply forces a re-grasp to go
click the igniter. Declared module order:

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the press-then-place
   trajectory and the monotone score trace 0.00 → 0.40 → 1.00.
2. `smoke.py` — rejection battery, 14 checks:
   1. settle/no-NaN (burner off, striker at top, pan upright on a slot,
      kettle on the other, score 0);
   2. randomization readback (burner xy, pan xy, pan yaw, kettle xy differ
      across seeds);
   3. pan slot coin takes both sides over 10 resets;
   4. null policy 240 steps (score ≤ 0.01, not lit, no striker creep);
   5. SEED-NAIVE: pan seated perfectly on the COLD plate → seat latch only,
      ≤ 0.151;
   6. FAT-OBJECT: kettle offered knob-down over the shaft → perches on the
      mouth (height readback), striker depth audited every substep stays
      ~0, not lit;
   7. shallow press: striker released 5 mm down (< 8 mm click) → never
      ignites AND the spring returns it to the top;
   8. wrong-object: burner lit (physically, by the pan), KETTLE seated on the
      plate → refused, ≤ 0.401;
   9. near-miss: lit, pan resting 55 mm off the plate centre → radial window
      refuses, ≤ 0.401;
   10. inverted pan: lit, pan upside-down on the plate → upright/z refuse,
       ≤ 0.401;
   11. settle gate: lit + exact seat pose written with 0.40 m/s velocity →
       geometric predicates true, success refused while moving (dismantled);
   12. MECHANISM: from a fresh reset, the released pan's weight clicks the
       striker by contact (max depth readback > click), spring returns, lit
       latches → score 0.40, no success;
   13. audit: success was never True at any smoke step;
   14. frames.npz video.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS <n>/<n>`) and hard-exit; a top-level try/except
prints a FAIL verdict on any exception and a watchdog Timer kills the
process on a hang, so the forge never waits out the timeout.
