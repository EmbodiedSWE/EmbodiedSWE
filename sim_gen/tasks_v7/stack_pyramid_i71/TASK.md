# stack_pyramid_i71 — Falsework Tent

Scene: `falsework_tent` · Env: `simgen.falsework_tent` · Robot slot: `null` (scene-level task)

## Seed provenance

Seed: `maniskill/stack_pyramid` (`RoboVerse/roboverse_pack/tasks/maniskill/stack_pyramid.py`):
three 0.04 m cubes; "pick up the red cube, place it next to the green cube, stack the blue
cube on top of both". Success is a `DetectedChecker` with two relative-bbox detectors — the
blue cube on top of red AND green. The entire task is repeated free-space pick-and-place;
every intermediate state is independently stable, and the two base cubes are passive.

## What changed and why it is strategically different

The seed's structural principle is inverted. In the seed, the goal configuration can be
built by placing one object at a time: each part sits stably wherever it is released, and
the third object rests on the other two only at the very end, passively. Here the GOAL
configuration is a **mutually-supporting pair** — a card tent of two thin panels (14 mm)
leaning against each other — where **neither part is stable alone**: a lone panel at any
in-band lean (8–55° off vertical) has its CoM outside its footprint and falls (smoke #7
demonstrates the collapse). With a single gripper you cannot hold one panel while placing
the other, so *direct sequential placement — the seed's whole strategy — cannot produce the
goal*. The task forces the civil-engineering answer: **falsework**. A green pillar stands at
the build site; each panel CAN rest stably against a pillar face (those intermediates are
stable), and lifting the pillar straight up out of the pair releases the panels sequentially
so they fall inward onto each other and settle into the self-supporting tent. Success
additionally demands the falsework be *removed* (≥ 15 cm from both panels — the tent must
demonstrably carry itself) and *parked* on the magenta pad.

Distinct from the corpus read this session: no lever/pry (setup_checkers), no tilt-maze
gimbal (block_pyramid), no compass-heading dials (play_jenga), no hidden-mass sorting
(stack_wine), no captive-gate extraction/conveyance (hockey), and none of the noted corpus
strategies (bridging, counterweights, insertion, twist-lock, pouring, toppling, silo
drop-order, carousel). The signature skill — *build with temporary support, then remove the
support without destroying what it supported* — appears nowhere else in the corpus.

The seed's own strategy is a settled reject state: panels stacked flat on the build pad
score ~0 (smoke #5 — flat is 90°, outside the tilt band).

## Teleport-solution outline (solve.py)

1. **P0** settle + layout readback; assert score ≈ 0, no success.
2. **P1 lean A (contact statics)**: teleport panel_a to the release pose — 25° tilt
   (near the final tent angle, so the fall arc is short), base 3 mm above the pad, face
   2 mm off the pillar's +y face — and release. Gravity closes both gaps; the panel tips
   onto the column (CoM inside the base→face support interval) and the settled lean
   latches `lean` (0.25).
3. **P2 lean B**: mirror on the −y face → opposed pair latches `pair` (0.30).
4. **P3 extraction (applied force)**: vertical force servo at the pillar CoM —
   `F_z = m(g + 8(v_des − v_z))` capped [0, 2.5 mg], soft xy position hold plus an
   uprighting-torque orientation hold (the rigid grip on the handle: an airborne body
   held by a point force has no rotational stiffness, and the asymmetric panel normals
   would spin the falsework into the leans). Vertical pull is yaw-invariant, so the
   pod's wrench-frame drag has nothing to bite on. The pull is two-speed: slow
   (0.08 m/s) while the panels bear on the faces, then fast (0.55 m/s) through the
   release band — a released panel tips inward and its top edge RISES, so the column
   bottom must outrun the rising tops or they strike it and the pair scatters. The
   panels then collapse inward onto each other into the tent (combine-"min" 0.12
   friction on the pillar faces makes the slide nearly free).
   At root z > 0.30 the force is cleared and the held pillar is carried
   (transport teleport) to 3 cm above the magenta pad and released; everything settles
   → `free` latches (0.30), success. All teleports are transport-only; every rubric fact
   (leans, tent, resting pillar) is produced by contact/gravity.
5. **P4** hands-off persistence 10×40 steps (3.33 s at 120 Hz); `SIM_GEN_SOLVE: SUCCESS`
   only if success still holds. `SIM_GEN_SCORE` printed at every boundary, non-decreasing
   (latched credit).

## Embodiment argument (Franka, base near origin)

Everything sits within ~0.6 m of a base at the origin (build pad 0.45 m out, park pad
(0.15, −0.38), scatter slots ~0.35 m). Per-object contact strategy:

- **Panels** (14 mm thick, 80/70 g): parallel-jaw pinch across the thickness — 14 mm is
  well inside the ~80 mm Franka jaw span. Pick a flat panel by its raised long edge (the
  2 mm authoring gap plus pad edge give finger clearance), reorient in-hand to ~14°, and
  release at the lean pose against the pillar face — exactly the write the solve performs.
- **Pillar** (0.7 kg, thin 24 mm column): grasped by the yellow T-handle bar; its ends overhang ±80 mm in x,
  beyond the panels' ±50 mm half-width, so the fingers close on free bar with no panel
  contact. Extraction is a straight vertical lift — one DoF, no reorientation — matching
  the solve's vertical force servo; then a free-space carry to the magenta pad.
- No forces beyond 2.5·(0.7 kg)·g ≈ 17 N (well under Franka payload), no bimanual holds
  (the pillar removes the need — that is the point of the task).

## Execution order declared

Built in this order: (1) minimal goal predicate + scene, (2) working solve, (3) final
rubric (latched weights 0.25/0.30/0.30, cap 0.85), (4) smoke battery.

## Checks (smoke.py)

13 checks: settle/baseline · randomization readback (build xy+yaw, park xy, panel xy) ·
slot-swap coverage · null-policy · SEED-strategy stack-flat reject · falsework-in-place
reject (pillar_clear) · LONE-LEAN-FALLS physics certificate (+ latch debounce honesty) ·
half-collapse tilt-band reject · out-of-zone tent reject · tent-without-park 0.85 cap ·
pillar-adjacent clear_r reject · parallel-lean opposed reject · frames.npz saved.
