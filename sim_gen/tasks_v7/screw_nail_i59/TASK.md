# screw_nail_i59 — `latch_vault`

Unlock the double-latched case, remove the trapped lid, and move the green block into
the dish.

## Provenance

- **Seed task**: `rlbench/screw_nail`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/screw_nail.py`) — a Franka grasps a
  screwdriver, mates its tip to a nail on a block, and drives the fastener IN by
  continuous rotation about a vertical axis. One tool grasp, one long twisting motion,
  judged by fastener depth. (The seed's end state is the nail seated in the block —
  there is no meaningful "reversed end state" corpus entry to avoid; N/A.)
- **Scene**: `latch_vault`, registered as `simgen.latch_vault` with `robot="null"`.

## Strategic difference (vs the seed and the corpus read this session)

The seed is **tool-mediated rotary assembly**: acquire a tool, maintain a tip-on-nail
constraint, rotate continuously, drive a part INTO another. This task is an **ordered
mechanical DISASSEMBLY chain with no tool and no rotation**:

1. slide TWO latch knobs outward along horizontal prismatic rails (real joints with
   limit stops);
2. lift the freed lid out of its pocket and set it aside;
3. extract the revealed green block and place it inside a dish (with a same-size
   orange decoy that never counts).

A solver needs a different **plan** (read the lock state; three heterogeneous
sub-goals in a mechanically forced order — linear slides, a vertical extraction, a
pick-and-place — instead of one continuous tool rotation) and a different **code
structure** (joint-gated stage latches plus an in-dish containment predicate, not a
fastener-depth readout). It is also unlike the other corpus tasks read this session:
`pen_holder` (insert pens into a cup), `open_oven_i6`/dice_tumble (nonprehensile
tumbling to color pads), `push_button_i6` (press a plate). No other task read is an
unlock-uncover-retrieve chain.

## The lock is real geometry (mechanically enforced ordering)

- The lid sits in a pocket: lip walls (±y) and corner stubs (±x) stand 22 mm above
  the seated lid top and block every horizontal escape; only lift remains.
- Each engaged latch tongue overlaps the lid edge by ≥ 32 mm with a 4 mm vertical
  gap, capping lid lift at 4 mm flat — far below the 22 mm pocket walls. The tilt
  escape is closed too: the tongue caps lid pitch at `tongue_gap/overlap`, so even a
  maximally tilted lid's bottom edge stays below the pocket walls (asserted in
  `LatchVaultSceneCfg.__post_init__`, force-probed in smoke checks 7–9: the same
  velocity-regulated ≤ 6 N lift — solve's own P2 controller — that pops the unlocked
  lid clear moves the fully locked lid 4 mm and the one-latch lid 13 mm, and both
  locked probes leave the engaged latches engaged).
- The green block spawns on the cavity floor UNDER the lid; the cavity walls enclose
  it. Order is therefore physical: **latches → lid → block**.
- Latch rails are horizontal (gravity-neutral DOF) with high linear damping, so a
  latch stays where it is left — no hidden spring, motor, or scripted flag. The
  lock probes use the regulated lift (not a constant-force hammer) because a
  sustained ~50 m/s² constant pull vibration-walks any springless slide latch —
  that is a jackhammer attack, not the manipulation this task grades.

## Solution outline (solve.py — teleport = transport only)

1. **P1 unlock (dynamics)**: velocity-regulated horizontal force (≤ 6 N) slides each
   latch body outward; the prismatic joint constrains the path and its limit is the
   stop. Nothing teleported.
2. **P2 lid off (dynamics + transport)**: vertical velocity-regulated force (≤ 6 N)
   lifts the lid dynamically out of the pocket through the now-open gap (asserts
   z > 0.15 — exactly what a still-engaged latch would prevent); only the free lid is
   then teleported across open space to a parking spot and settled.
3. **P3 retrieve (dynamics + transport + contact)**: vertical force lifts the block
   dynamically out of the open cavity (asserts clearance, proving the mouth is open),
   then the block is teleported to 10 cm above the dish, released with zero velocity,
   and seated by **gravity + contact** — never spawned in place.
4. **P4/P5**: hands-off settling to `success()`, then 3.3 s persistence
   (400 substeps) with `SIM_GEN_SCORE` non-decreasing at every phase boundary.

Passes seeds 0 and 1 on the forge (`SIM_GEN_SOLVE: SUCCESS`).

## Rubric

- `success()`: the GREEN block rests inside the upright dish — dish-frame containment
  window that by construction accepts every physically-in-dish resting pose and
  rejects rim/outside poses (asserted) — with block and dish settled. Judged by
  identity: the decoy can never substitute.
- `score()` (latched, never evaporates): 0.10 per latch ever fully retracted + 0.20
  lid ever clear + 0.20 block ever out of the cavity + 0.15 block ever near the dish,
  capped at 0.75; 1.0 iff `success()`. Null policy scores ~0 (everything spawns
  locked and covered).
- Per-episode randomization (readback-verified in smoke): block position in the
  cavity, latch engagement depth, dish slot (4) + jitter + free yaw, decoy slot (2) +
  jitter.

## Embodiment argument (single Franka arm, OSC)

Base at the env origin (0, 0) on the floor; everything lies within ~0.72 m reach
(case center at 0.45 m, farthest dish slot at ~0.68 m):

- **Latches**: each knob is an upright 16 mm post rising well above its rail
  (top ~0.13 m) at the case ends — graspable from above or simply hooked and pushed
  outward with the closed gripper; the 5.6 cm stroke is a straight horizontal slide.
- **Lid**: 16 mm-wide yellow handle block centered on the lid, protruding 24 mm above
  it — a standard top grasp; the lid then lifts straight up with 4 mm/side pocket
  clearance and is set aside anywhere on the open floor.
- **Block**: 28 mm cube in an open 16 × 10 cm cavity; block top at 0.040 m, cavity
  walls at 0.055 m, pocket walls at 0.089 m — the Franka gripper (fingers ~50 mm
  long) reaches it with a vertical top grasp without touching the walls.
- **Dish drop**: 9.4 cm inner opening for a 2.8 cm cube — release from above.

No tool use, no regrasping under clutter, no bimanual requirement.

## Files

- `scene.py` — cfg + scene, procedural compound-box spawners, spawn-authored
  prismatic latch joints, predicates, latched score. Registers `latch_vault` and
  `simgen.latch_vault`.
- `solve.py` — teleport solution (above); watchdog + hard exit.
- `smoke.py` — 16-check rejection battery (records `frames.npz`): reset/layout
  sanity, randomization readback over 6 seeds, null policy, seed-strategy analog
  (press+twist achieves nothing), the three-way lock force probe
  (locked/one-latch/unlocked under the identical velocity-regulated ≤ 6 N lift —
  solve's own P2 controller; a constant-force pull is a jackhammer, not a lift, and
  can vibration-walk any springless slide latch), near-miss beside the
  dish, rim perch, decoy-in-dish identity, settle gate, latched-credit persistence,
  success-never-True audit, no-NaN. `SIM_GEN_SMOKE: ALL PASS 16/16`.

## Check list

- [x] Strategically different from the seed (disassembly chain vs rotary tool
      assembly) and from every task read this session.
- [x] Teleport = transport only; every load-bearing interaction (latch slide, lid
      lift, block seating) through contact/joint dynamics.
- [x] `SIM_GEN_SCORE` non-decreasing at phase boundaries; ≥ 3 s persistence after
      success; hard exit with watchdog.
- [x] Solve passes on ≥ 2 seeds (0 and 1) on the forge.
- [x] Rubric rejection battery passes on the forge; success never True in it.
- [x] describe()/instruction() state goal, mechanism, randomization, and what does
      not count; all claims physically true (interlock asserted + force-probed).
- [x] Fully procedural geometry; no external assets; files only in the task dir.
