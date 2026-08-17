# hand_trajectory_i200 — S-Channel Puck Run (`simgen.channel_run`)

## Seed provenance

Derived from **pick_place/hand_trajectory**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/hand_trajectory.py`): a humanoid
(Vega) GRASPS a small box and carries it through FREE SPACE, visiting five floating
waypoint markers in order; reward is dense distance-to-marker tracking of the HAND, with
auto-closing fingers. The seed's whole strategy is *grasp, then fly the hand along a
memorized list of aerial waypoints*; the object never touches anything, order is imposed
by markers, and the episode ends hovering at the last marker.

## What changed, and why it is strategically different

Kept only the skeleton (an object must visit an ordered sequence of stations). Every
element of the seed's strategy is inverted:

- **The route is physical, not virtual.** The five stations are not floating markers but
  points along the floor of a walled S-channel (two 90-degree corners) milled into a
  track slab. Nothing is memorizable: the track is jittered, yawed, AND randomly
  **mirrored** (left- vs right-handed S) per episode — a memorized waypoint list is
  wrong half the time; the solver must read the geometry.
- **The object is never carried.** Checkpoints latch ONLY while the puck's centre rides
  the channel-floor band (z gate `cp_zmax`): a carried puck misses the gate by ≥ 25 mm
  (asserted). Smoke check 6 *constructs* the seed's strategy — pins the puck at every
  checkpoint xy in perfect route order, 65+ mm up — and latches exactly nothing.
- **Progress is order-gated in the physics frame.** Each checkpoint latches only with
  its predecessor already latched (sequential prev-gating), so skipping legs, running
  the route backwards, or starting from the goal end scores 0.
- **The goal is a gravity event under an occluder, not a hover.** The last leg dead-ends
  under an amber ROOF slab that covers a 25 mm recessed pocket in the channel floor. The
  pocket cannot be entered from the air (smoke-verified); the only way in is to push the
  puck under the roof lip until its CoM passes the pocket edge (roof face is 12 mm shy
  of the lip = 16 mm CoM overhang at flush push, asserted) and it TIPS AND DROPS in by
  gravity — the terminal event is a blind push + an uncontrolled fall, exactly when the
  seed would be holding tightest.
- **Versus the read corpus:** no existing task is a floor-bound ordered-route push.
  `track_banana` (i79) releases clamped objects into a crate, `push_button` (i6) is a
  weighbridge press, `slide_block_to_target` (i135) rolls a die by tipping,
  `pen_holder`-style tasks are free-space insertions. The walled-maze traversal with
  prev-gated floor checkpoints, per-episode chirality flip, and roof-occluded
  gravity-drop goal appear nowhere in tasks_v4–v7.

## Apparatus (fully procedural, no meshes)

12 kinematic pieces re-pinned per reset (48 × 40 cm base slab; three raised channel
floors 25 mm thick — the pocket is the *missing* floor rectangle in the north leg; four
boundary walls + a centre island forming a 100 mm-wide S-channel with walls 42 mm above
the channel floor; two amber flank pillars carrying an amber roof whose underside sits
80 mm above the channel floor, east face at local x = −0.123, covering the pocket whose
lip is at x = −0.135) and one dynamic red puck (⌀56 × 60 mm cylinder, 150 g,
`solver_velocity_iteration_count=4` — the cylinder-creep fix). Config honesty is
asserted in `__post_init__`: flush push drops the CoM 16 mm past the lip, the roof
clears the standing puck by 20 mm, the channel clears the puck by 44 mm, a riding puck
passes the checkpoint z gate while a carried one fails it by ≥ 25 mm, the pocket admits
the puck standing or lying while an on-floor rest fails the well gate by 10 mm, and the
spawn band clears the closed end of the start leg.

**Randomization (readback-verified):** track xy ±3 cm + yaw ±15° + 50/50 mirror (the
pieces physically flip sides — asserted by un-rotated piece readback), puck spawn x over
a 75 mm band + lateral jitter + free spin.

## Rubric

- `success()` = all five checkpoints latched (in route order, on the channel floor)
  ∧ the puck is inside the roofed pocket NOW (inset rectangle + z gate) ∧ settled.
- `score()` (monotonic, latched): 0.09 per ordered checkpoint (max 0.45) + 0.25 once
  delivered (ever in the pocket with the full route already latched); exactly 1.0 iff
  success(). Max non-success score 0.70; null policy scores 0 (first checkpoint is
  100+ mm from every spawn).

## Solution outline (solve.py — the legitimacy certificate)

**Zero teleports.** The puck spawns in the start bay and every millimetre is contact
dynamics: a velocity-servoed WORLD-frame horizontal force at the puck's CoM (the
applied-wrench emulation of a fingertip push; `is_global=True` defeats body-frame wrench
drag) drives it at ~0.12 m/s along the channel floor through both corners. Tip safety:
the force cap (1.10 N, stall-escalated ≤ 1.30 N) stays below the 1.37 N quasi-static
tipping threshold, so the puck can only SLIDE; gain K = 6 N/(m/s) keeps K·dt/m = 0.33
(one-substep wrench-delay stable). Waypoints are read from the scene's randomized track
frame (mirror-aware), so one code path solves both chiralities. Phases: P0 settle (0) →
P1 south leg (0.18) → P2 east leg (0.36) → P3 north-leg run-up (0.45) → P4 press under
the roof at 0.08 m/s, force CUT at local x = −0.142 (CoM 7 mm past the lip), gravity
tips the puck into the pocket, delivered + live success (1.0) → P5 hands-off persistence
3.5 s → `SIM_GEN_SOLVE: SUCCESS`. Verified on the forge for seeds **0** (23.6 s,
mirror −1), **1** (23.2 s, mirror −1) and **2** (24.1 s, **mirror +1**), scores
non-decreasing 0 → 0.18 → 0.36 → 0.45 → 1.0 every time.

## Embodiment argument (Franka, one base pose)

Base at the origin facing +x; track centre at (0.42, 0) ± 3 cm. The farthest work
points (outer corners of the channel, local r ≈ 0.21 m from the track centre) sit
≤ 0.67 m from the base at 5–11 cm height — inside the 0.855 m reach envelope with the
standard top-down wrist. Per-object contact strategy:

- **Puck (the only thing touched):** it protrudes 18 mm above the wall tops, and the
  100 mm channel leaves 22 mm per side around it, so a closed-gripper fingertip can
  contact its upper band from above anywhere on the route and push at ≤ 0.5 N
  (µmg ≈ 0.44 N to slide). Direction changes at the corners are re-approaches around
  the standing puck in an open-top channel. For the final press the fingertip pushes
  the puck's trailing face: at the force-cut pose the trailing face is at local
  x = −0.114, still 9 mm east of the roof face (−0.123), and the puck top (world
  0.105 m) passes 20 mm under the roof — **no part of the gripper ever needs to enter
  the roofed volume**; gravity finishes the job.
- **Track, walls, roof:** furniture; no contact required. No grasp is required
  anywhere in the task.

## Execution-order declaration

The rubric enforces the only order that matters: checkpoints latch strictly in route
sequence (prev-gated), and delivery latches only with the full route already latched, so
the drop must come LAST. The physics closes the loopholes: the pocket is roofed (no
aerial insertion — smoke 7), carried transit latches nothing (z gate — smoke 6), and a
puck parked at the roof face without the press scores 0.45 with no success (smoke 11).
Within a leg the servo may wander; between legs the order is fixed.

## Checks (smoke.py — rejection battery, `SIM_GEN_SMOKE: ALL PASS 16/16` on forge)

1. Settle/no-NaN: puck standing still in the start bay, finite state.
2. Score 0 at reset, nothing latched.
3. Randomization readback: track pose and puck spawn vary across seeded resets.
4. Mirror readback: both hands appear across 8 seeds AND the pieces physically flip
   sides (un-rotated piece-position readback).
5. Null policy (240 steps): score 0, nothing latched.
6. **Seed-strategy analog (aerial carry):** the puck pinned at every checkpoint xy in
   perfect route order, 65+ mm above the gate (altitude verified by the probe itself)
   → zero latches, score 0.
7. Air-drop over the pocket: the roof physically blocks entry — never in_well, score 0.
8. Pocket shortcut: puck placed inside the pocket with no route latched — in_well True
   (positive control of the geometric gate) but no delivery, score 0.
9. Order enforcement: cp3/cp5 visits latch nothing without predecessors; cp1 alone
   latches exactly one (0.09); cp3 still refuses.
10. Partial shortcut: pocket entry with only cp1 latched delivers nothing (0.09).
11. Near-miss: full route latched, puck parked at the roof face, never dropped — 0.45,
    no success.
12. Latched credit survives pushing the puck back to the start bay (0.45).
13. Settle gate: the delivered configuration moving at 0.35 m/s is refused (delivered
    latches 0.70; success stays False).
14. No live success after yanking the puck back out: score capped at 0.70.
15. Rejection audit: success() never True at any judged point; max score 0.70.
16. Final no-NaN; frames.npz saved.
