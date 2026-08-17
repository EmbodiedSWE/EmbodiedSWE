# libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it_i150 — shunt the grate over the fire pit, then seat the pan on it

## Provenance

Seed: `libero_90/libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it`
— rotate the stove knob past a joint-angle threshold (`joint_pos > 0.5`) and set
the frypan down on the flat, always-available cook plate (xy within 0.08 m,
height 0–0.03 m over the cook region).

## Strategic difference from the seed

The seed is *fixture actuation read from a joint* + *free placement onto a
static surface*. This task keeps the "turn on the stove, then cook-place the
pan" story but removes both mechanisms:

- **No knob, no joint readout anywhere in the rubric.** The stove is a lit
  fire pit sunk into the bench, sealed by a sliding safety **cover**. "Turning
  it on" means *uncovering* it — and the only way to do that usefully is a
  **cascaded shunt**: push the free **grate** along the shared one-axis rail so
  it collides with the cover and drives the two-body train until the *cover's*
  joint-limit end stop halts it. The stop position is built so that exactly
  there the grate sits centred over the well (stop = cover half-length + grate
  half-length; asserted in cfg). The robot cannot and need not fine-position
  the grate — the mechanism self-aligns, which is a planning insight, not a
  dexterity test.
- **The placement surface does not exist at reset.** The seed's cook plate is
  static; here the cook surface is the *moved body itself* — the grate spawns
  parked in a per-episode random side bay (east or west) and must be shunted
  into place before any placement is possible.
- **The naive seed plan is terminal.** A pan set over the open well without
  the grate is narrower than the well mouth (half-diagonal 59 mm vs 62 mm
  mouth) and tips into the fire pit — a real physics-imposed hazard, proven in
  smoke. A pan parked on the closed cover scores ~0.

Distinct from the rest of the corpus: i87 (lever→drawer articulation) actuates
a fixture *joint* to open a container; i36 builds a bridge from free parts.
No other task uses a two-body same-rail shunt with a self-aligning end stop,
nor makes the placement target a body the robot had to move first.

## Scene

Fully procedural (native PhysX box colliders; sliders ride spawn-authored
per-env prismatic joints whose LIMITS are the hard stops; the kinematic bench
is the joint anchor and never moves):

- **bench** (kinematic compound): 780×510×160 mm counter, square 130 mm
  fire-well shaft sunk through the middle, glowing fire bed at the pit floor
  (collider ON) + orange flame flicker (visual only, entirely inside the pit).
  South apron (y ∈ [−300, −65] mm) is the pan spawn zone.
- **cover** (dynamic, on rail): 220×200×12 mm plate + red knob, 0.40 kg,
  spawned sealing the well (x jitter ±20 mm). Joint limits ±200 mm — the +stop
  IS the self-aligning stop (110 + 90 mm).
- **grate** (dynamic, on rail): 180×200×12 mm plate, 0.50 kg, with two low rim
  bars (y retainers, 15 mm tall, window 154 mm) and two full-width 47 mm push
  **tabs** at the x ends (robot push feature, x retainers window 156 mm, and
  the shunt contact face). Spawns in a random side bay at |x| = 240±10 mm.
- **pan** (dynamic, free): 84 mm square dish (12 mm base, 30 mm walls) with a
  150 mm × 20 mm stick handle, 0.300 kg; spawns on the apron (x ±100 mm,
  y −210±30 mm, yaw −90°±75° so the handle stays south), clear of the rail
  sweep (asserted).

Randomization (all readback-verified in smoke): cover x jitter, grate bay side
(coin via `torch.rand`) + x jitter, pan xy jitter + free yaw.

13 honesty asserts in `SceneCfg.__post_init__` pin the geometric claims:
self-align stop ⟹ cover clear of well; cover seals at any spawn; bay spawn
touches nothing; ungrated pan fits into the well mouth at any yaw; retainer
windows box a seated pan inside the judged xy tolerance; pan drops *between*
the retainers; a bar-perched pan reads outside the z window; spawn zones and
deck extents are consistent; the seated pan's handle clears the rim bars.

## Rubric

Latched calm-gated partial credit, capped at 0.55 unless success holds live:

| credit | clause |
|---|---|
| 0.20 | cover ever clear of the well, calm (latched) — \|x_cover\| ≥ 0.180 |
| 0.20 | grate ever centred over the well, calm (latched) — \|x_grate\| ≤ 0.015 |
| 0.15 | pan ever seated on the grate (grate-relative xy ≤ 0.045, bottom within 10 mm of the grate top, upright < 12°), pan AND grate calm (latched) |
| 1.00 | `success()` live |

`success()` = grate centred over the well AND pan seated on it AND all three
movable bodies settled (|v| < 0.05 m/s) AND finite. Because grate-in-tolerance
geometrically implies the cover is clear (non-penetration keeps the pair a full
stop-length apart), success certifies the whole cascade.

## Solution outline (solve.py — teleport-transport contract)

All load-bearing interaction is contact dynamics or applied forces:

- **P0** settle; assert cover sealing, grate in bay, pan upright on apron,
  score ≈ 0.
- **P1 SHUNT** (force, not teleport): velocity-regulated horizontal force on
  the grate (`f = clamp(m·20·(v_des − v), ±8 N)`, v_des = 0.15 m/s, wrench
  zeroed afterwards) pushes it along the rail; it collides with the cover and
  drives the train to the cover's end stop; release, settle → both latches →
  score 0.40. The prismatic grate never rotates, so the pod wrench-frame quirk
  is moot.
- **P2** teleport the pan (transport only) to a hover 20 mm above the grate,
  handle south, and let it **drop and settle** between the retainers; 3-attempt
  re-drop retry → success → score 1.0.
- **P3** ≥ 3.3 s fully hands-off persistence (10 × 40 steps, success at every
  checkpoint) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed non-decreasing at every phase boundary. Verified on
the forge for seeds 0 and 1 (`--args "--headless --seed 1"`).

## Embodiment argument (Franka)

- **Grate shunt**: the 47 mm-tall full-width push tab at 160–210 mm height is
  a canonical drawer-push feature — flat palm or closed-gripper push, ≤ 8 N
  horizontal, straight-line ~240 mm stroke ending on a hard stop (no precision
  needed; the stop self-aligns). Identical contact class to opening/closing a
  drawer.
- **Pan**: the 150 mm stick handle (20 mm square section) is a canonical pinch
  grasp (80 mm jaw); carry ~25 cm and lower 2 cm onto the grate; the retainer
  windows give ±35 mm placement slack.
- All contacts lie 160–210 mm high inside a 78×51 cm footprint — comfortable
  for a Franka based at ~(0, −0.55) facing +y, reach 0.13–0.45 m.

## Execution order

Shunt-before-seat is imposed by physics (no cook surface exists until the
grate arrives; an early pan drop falls into the fire), not decreed by the
rubric. Declared module order:

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the cascade and the
   monotone score trace 0.00 → 0.40 → 1.00.
2. `smoke.py` — rejection battery, 13 checks:
   1. settle/no-NaN (cover seals, grate parked, pan on apron, score 0);
   2. randomization readback (cover x, grate x, pan xy, pan yaw differ across
      seeds);
   3. grate bay side takes both signs over 10 resets;
   4. null policy 240 steps (score ≤ 0.01, no slider creep);
   5. SEED-NAIVE: pan placed on the *closed cover* → score ≤ 0.01;
   6. HAZARD: cover slid open, pan dropped over the ungrated well → pan sinks
      into / tips toward the fire pit, never seated, score ≤ 0.201;
   7. pan seated on the grate *in the bay* → seat latch only, ≤ 0.151;
   8. cover open + grate short of the well + pan seated → not-at-well,
      ≤ 0.351;
   9. rim-bar perch (pan offset onto a bar) → xy/z windows refuse, ≤ 0.401;
   10. settle gate: exact seat pose written with 0.40 m/s velocity → geometric
       predicates true, success refused while moving;
   11. MECHANISM: the servo push from a fresh reset actually shunts the train
       (grate moved > 0.15 m by *contact*, cover clear, grate in tolerance,
       score 0.40, no success);
   12. audit: success was never True at any smoke step;
   13. frames.npz video.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS <n>/<n>`) and hard-exit; a top-level try/except prints
a FAIL verdict on any exception so the forge watchdog never hangs.
