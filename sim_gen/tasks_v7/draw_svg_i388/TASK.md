# draw_svg_i388 — `gantry_stamp`

Operate a 2-axis Cartesian gantry plotter: slide the bridge (X) and carriage
(Y) by their handles to place a spring-returned stylus tip over each of three
RED pads and press the stylus head down until the tip dwells on the pad face —
stamping all three — while NEVER pressing over either GRAY decoy pad (one full
press on gray is a permanent foul). Finish with the stylus spring-parked and
the machine at rest. Env id: `simgen.gantry_stamp`.

## Seed provenance

Seed: `maniskill/draw_svg`
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/draw_svg.py`) — a Franka
holds a free red marker cube and drags it along a prescribed SVG path on a
table; the reward is continuous path-tracking of a held object.

## Strategic difference

- **Captive tool through machine kinematics vs. a held free object.** The seed
  manipulates a free marker directly; here the marking tool is captive on a
  prismatic-joint chain (frame → bridge → carriage → stylus) and targets can
  only be reached *through* the machine: pushing the green mast moves X,
  pushing the yellow knob moves Y, pressing the orange head plunges Z against
  a return spring.
- **Discrete decomposed positioning vs. continuous path tracing.** There is no
  path. The solver must read the randomized pad layout, decompose each target
  into machine coordinates (bridge = pad_x − tip offset, carriage = pad_y),
  settle each axis, and execute discrete full-depth press strokes. The smoke
  battery transplants the seed's strategy (sweep the tool across every target)
  and shows it scores 0.
- **A forbidden-action clause.** Two gray decoys make one wrong press
  permanently unrecoverable — the seed has no irreversible failure mode.
- Distinct from the surveyed sibling corpus: no domino relay (draw_svg_i271),
  no pendulum arrest (draw_svg_i49), no aperture sorter (draw_triangle_i1), no
  beam balance (draw_triangle_i107), no bar construction (draw_triangle_i8),
  no impact pile-driving (libero_pick_milk_i51). Nothing else surveyed is an
  "operate a captive multi-axis machine with a decoy foul" task.

## Teleport-solution outline (solve.py)

Nothing is transportable in this scene (all bodies are captive in the machine
or kinematic), so the solution uses **no teleports at all** after reset —
every interaction is an applied force equivalent to a hand push:

1. **Read** — settle, read back frame pose, pad centres (frame-local), machine
   configuration; assert score 0.
2. **Aim** (per red pad, in bridge-coordinate order) — dual-axis PD force
   servo: bounded horizontal force on the bridge along the frame x-axis
   (≤ 8 N) and on the carriage along the frame y-axis (≤ 5 N) until both
   coordinates are within 4 mm (stamp tolerance is 18 mm) and slow.
3. **Press** — hold the axes and push the stylus head straight down at 6 N
   (full-stroke spring+weight load is ~3.5 N) until the scene's
   sustained-press latch fires (8 consecutive steps at stamp depth), dwell,
   release; the spring retracts the tip. `SIM_GEN_SCORE` printed per stamp
   (0 → 0.25 → 0.50 → 0.75).
4. **Finish** — zero all forces, hands off, success() turns True (score 1.0);
   hold hands-off 3.5 simulated seconds, then `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (Franka)

One base pose suffices: place the base ≈ 0.52 m from the frame centre along
frame-local +x (facing the board's short side). All three handles stay below
0.40 m and within a ~0.66–0.74 m worst-case reach:

- **Bridge (X):** the green mast (22 mm square post, top at 0.395 m) is a
  parallel-jaw grasp or open-jaw side-push; sliding the bridge takes ≤ 8 N
  horizontally — well within Franka payload.
- **Carriage (Y):** the yellow knob (18 mm square post, top at 0.330 m) is a
  pinch grasp; sliding takes ≤ 5 N.
- **Stylus (Z):** the orange head (56 mm disc) is pressed straight down with
  closed fingertips at ~6 N over a 27 mm stroke — a standard press primitive.
- The pads are 55 mm squares with 18 mm stamp tolerance; the axis handles give
  the arm mechanical guidance (each slide is a 1-DOF constraint), so no
  free-space precision beyond the tolerance is required.

## Execution order

**No order is declared or enforced.** The three red pads may be stamped in any
order; only the gray-pad foul (never press over gray) and the final state
(stylus parked, machine still) are constrained.

## Seed-strategy end state note

The seed's end state ("marker traced along the path") has no free-object
analogue here — there is no free object. The smoke battery instead transplants
the seed's *strategy*: the tip is swept across all five pads at traverse
height (check 7) and scores 0, and partial/misplaced presses (checks 9–10)
are rejected.

## Checks (smoke.py: 21 — items 1–2 and the two flip-pairs 16/17 each emit two checks)

1. settle/no-NaN; spring holds the stylus parked ~19 mm above the pads
2. fresh reset: score ~0, no success
3. randomization readback: frame xy + yaw vary
4. randomization readback: pad layout varies
5. randomization readback: initial bridge/carriage configuration varies
6. null policy 400 steps: score ~0, tip does not drift
7. seed strategy fails: tip swept over all five pads at traverse height → 0
8. spring is real: 6 N press over the empty slot reaches the bare board
   (through stamp depth, no latch) and retracts on release
9. near miss (xy): full press 30 mm off a red centre (on the pad edge,
   outside stamp_r) → no latch
10. near miss (depth): 1.2 N partial press dead-centre → stalls above stamp
    depth → no latch
11. real stamp: 6 N press latches → score 0.25, no success
12. latches persist: machine parked far away, score kept
13. wrong object: full press on a gray pad latches the foul
14. foul irreversible: remaining reds stamped honestly → still rejected,
    score capped ≤ 0.75
15. acceptance: fresh seed, three honest presses → success True, score 1.0
16. retract clause: stylus held pressed → False; released → True
17. settle gate: bridge kicked → False; re-settled → True
18. rejection audit: no unexpected success at any judged point
19. final no-NaN
