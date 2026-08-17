# libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it_i278 — park the cast-iron weight on the dead-man's gas pedal (the foam decoy is too light), then seat the pan on the plate

## Provenance

Seed: `libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it`
— rotate the stove knob past a joint-angle threshold (`joint_pos > 0.5`) and set
the chefmate frypan down on the flat, always-available burner (xy within
0.08 m, height 0–0.03 m over the cook region).

## Strategic difference from the seed

The seed is *direct fixture actuation read from a held joint angle* + *free
placement*. This task keeps the "turn on the stove, then place the pan" story
but inverts the actuation semantics and adds a selection problem:

- **Nothing latches — the gas state is LIVE.** The burner is fed through a
  DEAD-MAN'S PEDAL: a safety-yellow platform on a sprung 22 mm prismatic
  slide. The burner is lit only WHILE the pedal is held ≥ 15 mm down,
  sustained for 30 consecutive substeps (streak-gated); the moment the load
  comes off, the spring shuts the valve and the flame DIES. "Turning it on"
  is therefore not an event to cause but a **force-closure to arrange**: park
  the 1.2 kg cast-iron ballast on the pedal so its weight holds the gas open
  hands-free. Pressing the pedal and letting go achieves nothing (smoke
  proves the spring beats the 30-substep streak by an order of magnitude) —
  the exact opposite of the seed's held knob angle and of any latched press.
- **Selection by MASS.** A foam decoy with the *identical* body and pinch bar
  (20× lighter) sits among the movables; the spring's preload alone provably
  beats it (cfg-asserted, smoke-audited every substep). Shape and appearance
  cannot solve the task — only the mass ordering can, and the rubric never
  reads object identity at the pedal: whatever really holds the pedal down
  works (the smoke's cross-arrangement probe parks the PAN on the pedal and
  the gas genuinely opens — but then the pan isn't on the burner).
- **The naive seed plan scores ≤ 0.15, never 1.0:** a pan seated perfectly on
  the cold plate satisfies every placement clause and still fails (gas shut).

Distinct from the corpus tasks I inspected: i159 (same seed) turns the stove
on by a *momentary tool-mediated press* — a recessed piezo striker clicked by
the pan's own handle, with the lit state **latched** by a mechanism event and
the pan role-flipped into a tool; here the state is the anti-latch (live,
sustained, dies on release), the pan is a pure payload, nothing is recessed
or reach-denied, and the core competences are mass-based object selection
plus arranging a lasting force-closure. i150 (scene9 family) moves a
rail-shunt; i75 hangs the pan on a wall hook. No corpus task I saw uses a
dead-man (hold-to-run) mechanism or a mass-twin decoy.

## Scene

Fully procedural (native PhysX box colliders; the pedal rides a spawn-authored
per-env prismatic Z joint whose LIMITS are the hard stops; the kinematic
housing is the joint anchor and never moves):

- **counter** (kinematic): 800 × 660 × 150 mm bench.
- **pedal housing** (kinematic, FIXED at (0.24, 0.10)): 140 mm base slab
  (12 mm, collider) + four corner guide posts (VISUAL only — the joint is the
  real guide, so no post can prop a sloppy placement).
- **pedal** (dynamic, on the Z joint): 110 mm safety-yellow platform with a
  hazard stripe, 60 g, 22 mm travel; a scene-owned post_step spring (preload
  2.0 N, k = 160 N/m, damped) holds it at the top stop. Gas OPEN while depth
  ≥ 15 mm sustained 30 substeps (0.25 s); the streak zeroes instantly.
- **burner** (kinematic, randomized xy ±35 mm about (−0.18, 0.10)): 160 mm
  plinth + 140 mm cook plate (top at z 188 mm) + south-face lamp (visual;
  orange while lit, dark red otherwise — recolored LIVE).
- **pan** (dynamic payload): 100 mm square dish + 140 mm stick handle, 500 g.
- **ballast** (dynamic): 70 mm cast-iron block, raised 18 mm pinch bar,
  1.2 kg — the intended pedal weight.
- **decoy** (dynamic): the SAME geometry in pale foam, 60 g.

Randomization (readback-verified in smoke): burner xy; a **random
permutation** deals pan/ballast/decoy onto three front slots
(x ∈ {−0.24, 0, +0.24}, y = −0.20; `torch.rand(m,3).argsort` — never the
first-randint trap); per-object xy jitter; pan yaw cone (−90° ± 35°);
blocks free yaw.

~18 honesty asserts in `DeadmanStoveSceneCfg.__post_init__` pin the claims:
preload holds the empty pedal up; preload beats 1.5×(pedal+decoy) (the decoy
can never crack it); 0.6×(pedal+ballast)·g beats the full spring stack (the
ballast bottoms it out); (pedal+pan)·g clears the threshold force (the
cross-arrangement probe is physically real — mass, not identity, is the
mechanism); the threshold sits inside the stroke; the full-travel stop is the
JOINT limit, above slab contact; the pedal covers a sloppy block placement;
the posts sit outside the pedal; the plate covers the judged window; the
carry latch is unreachable from any on-fixture placement; slots clear the
burner, the housing and each other at any jitter (incl. the pan handle's yaw
cone); everything fits the counter; the pinch bar is a real jaw feature.

## Rubric

Latched partial credit, capped at 0.55 unless success holds live:

| credit | clause |
|---|---|
| 0.15 | ballast ever carried clearly off the bench — origin > 120 mm above deck (latched) |
| 0.25 | gas ever held open — pedal ≥ 15 mm for 30 consecutive substeps (latched credit; the LIVE state is what success needs) |
| 0.15 | pan ever seated on the plate, calm (latched) — radial xy ≤ 0.045, bottom within 12 mm of plate top, upright < 12° |
| 1.00 | `success()` live |

`success()` = gas open LIVE (the streak-gated dead-man state — a real
sustained load rests on the pedal at judge time) AND pan seated on the plate
(live geometry) AND pan settled (|v| < 0.05 m/s) AND finite. Nothing in
success() reads a latch.

## Solution outline (solve.py — teleport-transport contract)

All load-bearing interaction is contact dynamics; the pedal is NEVER written
by the solve:

- **P0** settle; assert masses readback (MassAPI won over density), pedal at
  the top, gas shut, each object on its dealt slot per the permutation
  readback, score ≈ 0.
- **P1 GAS** (gravity, hands-off): teleport the ballast (transport only)
  through a high waypoint (latches carry) to a level hover 10 mm above the
  pedal top and release. Its weight bottoms the pedal (readback ~22 mm >
  15 mm threshold) and HOLDS it; the streak completes and the burner lights
  (3-attempt retry). 60 hands-off steps prove a stable rest → score 0.40.
- **P2** teleport the pan (transport only) to a hover 20 mm above the plate
  top (asserted: the hover satisfies nothing), handle south, and let it
  **drop and settle**; 3-attempt re-drop retry → success → score 1.0. The
  ballast is untouched — the gas stays open because it keeps resting there.
- **P3** ≥ 3.3 s fully hands-off persistence (10 × 40 steps, success at every
  checkpoint — the dead-man hold never lapses) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed non-decreasing at every phase boundary. Verified
on the forge for seeds 0 and 1 (`--args "--headless --seed 1"`).

## Embodiment argument (Franka)

- **Ballast to pedal**: pinch the raised bar (18 mm wide, 16 mm tall on
  14 mm feet — finger clearance under the bar; a canonical 80 mm parallel-jaw
  grasp), lift the 1.2 kg block (well inside Franka's 3 kg payload), carry
  ~35 cm and set it down centred on the 110 mm pedal — the pedal covers the
  70 mm block with 20 mm margin per side, so ±20 mm placement slop is fine.
  The press force is the block's own weight; the arm never pushes.
- **Pan to plate**: pinch the 20 × 14 mm stick handle, carry ~45 cm and
  lower 2 cm onto the 140 mm plate; the 45 mm radial window gives real slack.
- **Selection**: mass is observable by lifting (wrist force) and by the
  visual convention (dark cast iron vs pale foam, stated in the
  description/instruction).
- All contacts lie 150–230 mm high inside an 80 × 66 cm bench — comfortable
  for a Franka based at ~(0.0, −0.55) facing +y, reach 0.25–0.45 m.

## Execution order

Weight-first is the natural embodied order (the flame then simply stays on
while the pan is fetched) but is NOT decreed: the rubric accepts either
order — a pan seated first on the cold plate latches the seat credit and
simply waits for the ballast to open the gas. Declared module order:

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the park-then-place
   trajectory and the monotone score trace 0.00 → 0.40 → 1.00.
2. `smoke.py` — rejection battery, 16 checks:
   1. settle/no-NaN (gas shut, pedal at top, each object on its dealt slot,
      pan upright, score 0);
   2. randomization readback (burner xy, pan xy+yaw, ballast xy, decoy xy
      differ across seeds);
   3. slot permutation: pan AND ballast each visit ≥ 2 distinct slots over
      10 resets, poses match the `obj_slot` flags;
   4. null policy 240 steps (score ≤ 0.01, not lit, no pedal creep);
   5. SEED-NAIVE: pan seated perfectly on the COLD plate → seat latch only,
      ≤ 0.151;
   6. DECOY-IS-TOO-LIGHT: decoy parked squarely on the pedal (z readback
      proves the rest) + pan on the plate → pedal audited every substep
      never reaches the threshold, never lit, ≤ 0.151;
   7. DEAD-MAN (signature): ballast parked (lit), then REMOVED → lit dies
      within a second, streak zeroed, spring returns; the gas CREDIT stays;
      pan seated afterwards is STILL no success, ≤ 0.5502;
   8. PAN-AS-BALLAST cross-arrangement: the pan on the pedal really opens
      the gas (depth readback) but is then not on the burner → no success;
      removing it shuts the valve again;
   9. near-miss: lit, pan resting 65 mm off the plate centre → radial window
      refuses, ≤ 0.401;
   10. inverted pan: lit, pan upside-down on the plate → upright/z refuse,
       ≤ 0.401;
   11. settle gate: lit + exact seat pose written with 0.40 m/s velocity →
       geometric predicates true, success refused while moving (dismantled);
   12. shallow dip: pedal released 8 mm down (< 15 mm) → never lit AND the
       spring returns it to the top;
   13. PRESS-AND-RELEASE exploit: pedal written to FULL travel with nothing
       on it → the spring pops back in ~2 substeps, max streak 1 ≪ 30,
       never lit;
   14. MECHANISM: fresh reset, the released ballast's weight presses the
       pedal past the threshold by contact (readback) and HOLDS it → lit
       stable, score 0.40, no success;
   15. audit: success was never True at any smoke step;
   16. frames.npz video.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS 16/16`) and hard-exit; a top-level try/except prints
a FAIL verdict on any exception and a watchdog Timer kills the process on a
hang, so the forge never waits out the timeout.
