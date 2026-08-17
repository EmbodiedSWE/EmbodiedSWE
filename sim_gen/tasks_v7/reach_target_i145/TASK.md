# reach_target_i145 — ColorCarouselScene

Rotate a covered four-cell color carousel until the cue-card's color cell sits under
the machine's loading chute, then drop the white token through the chute so it lands
inside that cell and settles there.

## Seed provenance

Derived from **rlbench/reach_target**: a wand must be reached to the ONE colored
sphere (out of three) named by the instruction — a pure goal-selection + free-space
reaching task with distractor goals and per-episode color assignment.

## What changed and why it is strategically different

The seed's core skill is *selecting the right colored goal and reaching it through
free space*. This task keeps the color-selection DNA (four colored cells, a cue card
naming the target color, per-seed shuffled geometry) but the selected goal is **not
reachable**: every cell is covered by a fixed roof with a single off-center loading
chute. Reaching the target color therefore becomes a **mechanism-alignment problem**:

1. **Cue reading, not label reading**: the target color is shown physically — a small
   colored card sitting in a tray on the machine plate (the other three cards are
   parked out of the workspace). The instruction never names the color.
2. **Rotate-to-select instead of reach-to-touch**: the carousel is a free revolute
   rotor. The agent must *turn the mechanism* (via the three orange drive pegs on its
   rim) until the target-color cell sits under the chute, within a ±7° window, and
   then let it come to rest there — a regulated, contact-driven manipulation of a
   damped rotating body, with an anti-drive-by latch (the window must be occupied
   at low angular speed for consecutive substeps).
3. **Gated insertion instead of touch**: success requires delivering the white token
   *through* the chute collar into the aligned cell. The roof blocks every direct
   placement (roof-to-wall gap 10 mm ≪ 30 mm token); the only way in is the chute,
   and the chute only feeds the cell that is currently aligned — a misaligned drop
   lands on the covered roof or in the wrong cell for zero credit (both smoke-proven).

Strategic delta vs. the other tasks built in this campaign: no vault/latch door
mechanism (screw_nail_i59), no receptacle-insertion-by-pose (pen_holder) — the
signature skill here is *closed-loop rotary alignment of a free damped rotor against
a fixed aperture*, followed by a gravity-fed drop that certifies the alignment.

## Scene summary

- Fixed **machine** (kinematic): base plate, center hub, roof at z 0.105–0.117 with a
  single rectangular aperture on the −x side (azimuth π, radial band r 0.075–0.125,
  half-width 0.0325), a chute collar rising to z 0.167 above the aperture, and a cue
  tray at machine-rel (−0.26, −0.26).
- **Carousel** (dynamic, mass 1.5 kg, ang. damping 4.0): free continuous revolute
  joint about the hub axis; solid eight-slab deck; four colored open-top cells
  (red/green/blue/yellow at 0/90/180/270°, interior 0.090², walls to z 0.095 —
  10 mm below the roof underside); three rim tabs with orange drive pegs at
  r 0.245, 120° apart (one is always reachable at any rotor angle).
- **Token**: white 30 mm cube on the floor at one of four randomized slots.
- **Cards**: four 50 mm colored cards; the target-color card spawns in the cue tray,
  the rest are parked far outside the workspace.
- **Randomization** (readback-verified in smoke): target color (torch.rand-based),
  initial rotor offset uniform in ±[25°, 335°] away from alignment, token slot +
  jitter + yaw, card jitter.

## Rubric (anchored in the demonstrated solve)

Latched, monotone score, cap 0.70 before success:

- **0.25 align latch**: target cell inside the ±7° window AND rotor slow for 10
  consecutive substeps (drive-by sweeps through the window earn nothing — smoke #10).
- **0.45 drop latch**: token at rest inside the *target* cell's containment window
  (cell-local |u|,|v| ≤ 0.040, z ∈ [0.010, 0.033]) while aligned. The window
  provably accepts every interior rest pose and rejects wall-top/roof/deck rests
  (`__post_init__` asserts the geometry; smoke #7/#8/#11 prove it dynamically).
- **success()**: token in target cell AND everything settled → score 1.0.

## Teleport solution (solve.py) — transport only

1. **P0 settle**: 60 substeps, assert score ≤ 0.02 (null credit).
2. **P1 ALIGN (pure dynamics, nothing teleported)**: a velocity-regulated torque
   about the pivot axis (τz = clamp(1.2·(ω_des − ω), ±0.25), ω_des = clamp(−2.5·err,
   ±0.45) — the same wrench a peg push produces) turns the rotor into the window;
   the torque is then RELEASED and the rotor must hold the window on its own damping
   for the latch to fire.
3. **P2 FEED (transport + dynamics)**: the token is teleported across free space to
   hover inside the collar mouth (env (0.32, 0, 0.185)) and RELEASED with zero
   velocity. Gravity and contact do the insertion: it falls through the chute, past
   the roof plane, into the aligned cell, and settles on the deck. It is never
   spawned inside the cell; the identical drop with the wrong cell (or roof) under
   the chute scores nothing (smoke #7/#8).
4. **P3 settle → success**, **P4 persistence**: ≥ 3.3 simulated seconds hands-off
   with success() held, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seed 0 (target green, initial err −33.6°) and seed 1 (target
yellow, err −126.7°) both SUCCESS with monotone scores 0 → 0.25 → 1.0.

## Embodiment argument (single Franka arm, parallel-jaw gripper, OSC)

Base pose: env (−0.10, 0, 0), facing +x; the hub is at env (0.42, 0).

- **Read the cue**: the card tray is at env (0.16, −0.26), card top z 0.028 — inside
  the wrist-camera view from a standard overhead scan pose; no contact needed.
- **Turn the carousel**: the three orange pegs (20 mm square, height to z ~0.125,
  above the roof plane at their radius — outside the roof half-extent 0.16) sit at
  r 0.245 with 120° spacing, so at ANY rotor angle the nearest peg lies within
  x ∈ [0.175, 0.30], |y| ≤ 0.21 of the base — comfortably inside the Franka's
  ~0.855 m reach at working height. The gripper caresses/pushes a peg tangentially
  (or grasps it — 20 mm fits the 80 mm jaw span) to apply exactly the pivot torque
  the solve's servo applies; the rotor's damping holds the window after release.
  Multiple strokes with re-grasps handle large offsets; the ±7° window plus latch
  requires only ~0.5 cm tangential precision at the peg radius.
- **Feed the token**: 30 mm cube, top grasp from the floor slot (all slots are in
  x ∈ [0.04, 0.18], |y| ≤ 0.38 — reachable, clear of the machine), transported to
  above the collar mouth at env (0.32, 0, ~0.19) and released — exactly the solve's
  release state. The collar mouth (65 × 50 mm inner) leaves ≥ 15 mm clearance
  around the token at the release pose, tolerant to cm-level placement error.
- No bimanual need, no force beyond the 0.25 N·m-scale peg pushes, no in-hand
  re-orientation (the cube may land in any yaw; the rubric window is yaw-agnostic).

## Execution order (as actually followed)

1. Wrote `scene.py` with minimal success()/score skeleton; submitted.
2. Iterated `solve.py` on the forge until the goal was physically reached
   (seed 0 SUCCESS, then seed 1 SUCCESS — first submitted version passed).
3. Finalized the rubric anchored in the demonstrated solution (latched 0.25/0.45,
   cap 0.70, honesty asserts in `__post_init__`).
4. Wrote `smoke.py`; forge run: `SIM_GEN_SMOKE: ALL PASS 15/15`.
5. Final clean re-runs of both modules on the submitted package.

## Smoke battery (15 checks, all PASS on the forge)

1. settle + finite state + layout sanity (readback)
2. null score ~0 at rest
3. randomization: ≥ 2 target colors, err spread > 40°, min |err| > 17° (seeds 21–26)
4. token slot spread + sanity across seeds
5. null policy 240 substeps → no credit, nothing moves
6. seed-analog nudge (small wrench) → moves > 5 mm, still no credit
7. roof covers misaligned cells: drop above covered target cell → rests on roof /
   off-cell, no containment
8. wrong-cell chute drop → lands in the WRONG cell, drop latch 0, score ≤ 0.02
9. alignment window honesty: 12° → no latch; 2° → latch fires, score 0.25
10. drive-by sweep through the window at speed → latch 0, score ≤ 0.02
11. deck-rider: token on the deck between cells → in NO cell
12. settle gate: token moving at 0.35 m/s inside the cell is not success
13. latched credit persists (0.70 stays ≥ 0.69 after regression)
14. audit: success() never fired spuriously during the battery
15. final no-NaN / finite state

Run (forge):
`python -u -m simgen_tasks.reach_target_i145.solve --headless [--seed N]`
`python -u -m simgen_tasks.reach_target_i145.smoke --headless`
