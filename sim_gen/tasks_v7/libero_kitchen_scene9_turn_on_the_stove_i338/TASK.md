# turn_on_the_stove_i338 — the three-dial gas interlock

**Env:** `simgen.stove_dial_interlock` (scene `stove_dial_interlock`, robot `"null"`)
**Files:** `scene.py` (scene + rubric), `solve.py` (demonstration), `smoke.py` (rejection
battery, recorded), this file.

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene9_turn_on_the_stove` (RoboVerse
`roboverse_pack/tasks/libero_90/...`). There a Franka stands at a flat stove and turns
its single knob; the checker is one fixture joint angle crossing a threshold
(`flat_stove.joint_pos[:, 0] > 0.5`). Kept from the seed: the kitchen-console setting,
the "rotate a stove dial" motion vocabulary, and the goal reading "the stove is now on".

## Why this is strategically different

- **One actuation → a three-way conjunction.** The seed's whole plan is a single
  grasp-and-twist of one DOF. Here no single actuation can turn the stove on: three
  free-spinning dial drums must EACH be turned until the single notch in its raised rim
  wall sits under its fence foot, and only the simultaneous alignment of all three
  releases the gas. Different seeds need different turn directions and amounts per dial
  (three continuous randomized DOF), not one canned twist.
- **No fixture joint angle is ever the goal.** The seed's rubric reads a commanded joint
  coordinate. Here success is the state of a PASSIVE FOLLOWER: the silver gas fence — a
  prismatic slider the robot never needs to touch — physically DROPS ~22 mm into the
  three notches when (and only when) they all line up. The rubric reads a drop that only
  gravity + contact can produce: a physical AND-gate over the three dial angles.
- **Perception is decoupled from actuation.** Each dial is turned by its colored crank
  peg, but the peg's bearing is deliberately decorrelated from the notch (per-dial fixed
  offsets 150°/270°/30°): the robot must LOOK at the amber notch marker to know when to
  stop, unlike the seed's fixed knob throw. Under-turning and over-turning both leave a
  wall under the foot.
- **Different failure surface.** The seed fails only by not-twisting-enough. Here the
  interlock rejects: turning just one dial (the seed's strategy — check 6), pressing the
  fence down by force (checks 5, 8), stopping one dial outside the ±22.8° physical
  window (check 8), and faking the drop by teleport (check 9 — the rubric refuses it
  and the buried feet pin the dials outside the hard window).
- Unlike sibling `..._i36` (same seed: plank bridge + rolled ball) this task has no
  transported cargo at all — the only motions are rotations of mounted dials; the moving
  goal object (the fence) is part of the appliance and is never handled.

## The scene

A kinematic console deck on a table carries three dynamic dial drums (Ø116 mm) on
spawn-authored **RevoluteJoints** (axis Z, NO limits — free continuous dials, floated
4 mm above the deck so they never rub). Each drum's rim carries a 22 mm-tall wall over
288° of arc; the 72° gap is the notch, marked amber on the drum top (visual-only marker
— a flush collider would shave the drop). A silver fence (beam + three 14 mm square
feet) rides on the wall tops via a **PrismaticJoint** (axis Z; upper stop keeps it
captive, lower stop sits BELOW the full drop so the landing is drum-top contact, never
the joint stop). Wall tops and feet are slick (PhysX pair-averages friction) so drums
spin freely under the resting feet. Physical drop window per dial:
±(36° − asin(foot/2 ⊕ slack / ring_r)) ≈ **±22.8°** — derived and margin-asserted in
`cfg.__post_init__` against every rubric threshold.

**Randomization** (readback-verified in smoke): the three dial start yaws, uniform over
the circle minus a ±45° exclusion band around aligned (null policy scores ~0, fence can
never start dropped).

## Rubric

- `success()` = fence physically dropped > 15 mm below ride **and** all three dials
  inside a generous ±30° hard window (sanity conjunct containing the physical window —
  it only rejects solver-glitch tunnels) **and** everything settled **and** finite.
- `score()` = 1.0 iff success; else 0.2 per dial currently within ±15° (live credit: a
  released dial holds its angle, so correct behavior never loses it; knocking a dial
  back away honestly loses it). Null policy ~0 by the spawn exclusion band.
- Threshold bracketing asserted at import: `align_tol (15°) ≤ phys_window − 5°` and
  `align_hard (30°) ≥ phys_window + 5°`.

## Solution outline (solve.py)

**No teleports at all.** Phases (SIM_GEN_SCORE at each boundary, asserted
non-decreasing): (0) reset + settle, score ~0; (1..3) for each dial L, M, R: a
velocity-cascade torque servo on the dial's bearing (K_A 4/s → W_CAP 1.2 rad/s → KW
0.08 N·m·s/rad, τ ≤ 0.3 N·m; discrete-stability KW·dt/Iz ≈ 0.61 < 1) turns the notch to
bearing 0, released only close AND slow — score 0.2/0.4 after the first two; the fence
is asserted still riding until the third aligns; (drop) the fence falls ~22 mm by
gravity alone → success, score 1.0; (4) ≥3.5 simulated seconds hands-off with all drive
buffers asserted zero, then `SIM_GEN_SOLVE: SUCCESS`. Passing on forge: seeds 0, 1, 2.

## Embodiment argument (single Franka, parallel-jaw gripper)

Base pose: on the user side of the table, facing the console (drums at
x≈0.05, y∈{−0.15, 0, +0.15}, tops at z≈0.49 — comfortably inside a table-mounted
Franka's dexterous workspace; the fence line is on the far side of the drums, out of
the approach path).

- **Dials:** each carries a vertical crank peg Ø16 mm × 70 mm at a 22 mm orbit — sized
  for a parallel-jaw pinch (opening ≤ 80 mm) from above. Strategy per dial: pinch the
  peg, orbit the wrist about the drum axis (or make repeated ~90° cranks with re-grasps
  — the dial has no limits and holds its angle between pushes; even fingertip nudges on
  the peg work since 0.3 N·m at 22 mm is ~14 N, well under Franka payload). The
  ±15° tolerance at the 48 mm ring is ±13 mm of arc — coarse motor precision.
  The solve's torque servo on the bearing is the stand-in for exactly this
  peg-crank; its τ cap (0.3 N·m) and rate cap (1.2 rad/s ≈ 69°/s) are hand-scale.
- **Stopping criterion is visual:** the amber notch strip and the fixed silver foot
  above the rim are both visible from the front-top; align notch under foot.
- **Fence:** never needs to be touched (it drops by itself). Its beam sits behind the
  peg orbit (asserted clearance), so cranking never collides with it.
- **Order-free:** any dial order and either turn direction work — no long-horizon
  constraint beyond "all three aligned".

## Execution order declaration

Interactions may happen in ANY dial order and either rotation direction; the ONLY
ordering fact is the conjunction itself — the fence drops exactly when the last of the
three dials enters the window (asserted in solve: not dropped after 1st and 2nd
alignments). Success is judged on settled state and survives hands-off persistence
(3.5 s in solve; adversarial re-crank in smoke check 11 — the dropped fence pins the
dials inside the hard window).

## Smoke battery (12 checks, rejection-only, recorded)

1. clean reset: finite, fence riding, dials at their sampled yaws (readback), score ~0;
2. randomization real across 8 seeds (all spawns ≥40° off, per-dial variety, no
   repeated triple, readback);
3. null policy 2 s: score < 0.05, no success;
4. captive fence: +5 N pull lifts to the ~12 mm joint stop (probe force path proven
   live), returns to ride;
5. 25 N press on the unaligned interlock: peak drop < 8 mm, no success;
6. seed-strategy analog: ONE dial aligned → score ~0.2, fence riding, no success;
7. two dials aligned → score ~0.4, one wall still carries the fence, no success;
8. near miss: third dial at ~33° (outside the ±22.8° physical window) + 25 N press:
   fence never drops, no success;
9. cheat: fence teleported to the dropped pose with dials unaligned — the feet bury
   in the rim walls, the rubric refuses the fake (no success, no credit — the
   hard-window conjunct), and the illegal state cannot be legalized (a dial cranked
   toward alignment jams on the buried foot far outside the hard window);
10. exactness: all three aligned → gravity drop → success ∧ score 1.0, stable 1 s
    hands-off;
11. pinned: adversarial crank cannot turn a dial out from under the dropped fence
    (jams inside the hard window, moved ≥5°) — success survives;
12. frames.npz recorded in CWD.

Verdict line: `SIM_GEN_SMOKE: ALL PASS 12/12`.
