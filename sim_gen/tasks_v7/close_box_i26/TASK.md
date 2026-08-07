# close_box_i26 — Trap Crate

Close the box by MAKING it: invert a free open-top crate over the red block, then
slide the capped trap into the pen (scene `trap_crate`, env `simgen.trap_crate`).

## Seed provenance

Seed: `rlbench/close_box` — an articulated box USD stands with its lid open; the
robot pushes the lid shut about its BUILT hinge, judged by the lid joint angle
(`JointPosChecker` on `box_joint`). One nonprehensile push on a panel the scene
provides, ending in a joint readout.

The seed's end state (a lid rotated shut on its hinge) is **not expressible** in this
scene — nothing is articulated and there is no lid; documented N/A. The seed family's
nearest naive plan — get the contents into the box / the box to the goal without any
closure — is constructed as a settled state in smoke check 5 and scores ~0.

## What the task is

A free, upright, open-top CRATE (150 × 150 × 100 mm, 8 mm walls, 450 g, a procedural
compound body) stands on the floor with two loose 40 mm cubes, one RED and one BLUE;
which scatter slot holds which cube is shuffled per episode, and crate (xy + yaw),
pen (xy + yaw) and blocks (xy + yaw) all jitter. A flat grey 260 mm PEN square
(collision-free painted marking) is the goal region.

Goal: shut the RED cube inside the crate and deliver it — the crate has no lid, so
the only closure is to turn the crate UPSIDE-DOWN over the red cube (walls descend
around it, rim seats on the floor: crate within 12° of inverted at floor height,
cube inside the wall box) and end with the capped crate centred on the pen (55 mm →
50 mm per-axis window in the pen frame) with the cube still inside, the BLUE cube
off the pen and out from under the crate, everything settled. The trap only holds on
the floor: **lifting the overturned crate frees the cube** (smoke check 13 proves a
carried crate delivers nothing), so transport must be a floor SLIDE under
containment — the walls drag the trapped cube along inside (smoke check 12 proves
the mechanism).

## Strategic difference

- **vs. the seed**: the seed pushes a panel the scene built, about a hinge the scene
  built, and reads a joint. Here there is no joint to read and no lid to push — the
  "closed box" is a STATE THE SOLVER CREATES by capturing a free block under an
  inverted free container, and that closed state must then be TRANSPORTED under a
  constraint the seed never has: the closure is held by gravity + the floor, so the
  natural pick-and-carry is self-defeating. Closure is manufactured, not actuated.
- **vs. corpus tasks read**: the articulation family (`close_microwave_i4/i5`,
  `close_grill_i8`, libero drawer/stove) manipulates built joints — i4 releases a
  prop so gravity closes a drop-gate, i5 twists a bayonet lock, i8 builds a ramp of
  free bodies, i6 reorients a die by edge-pivots (SO(3) goal, no containment). The
  pick-place family (libero bowls/pan, `coke_task_i15`, `pick_and_lift_i16`,
  `approach_grasp_spoon_i12`) transports grasped objects freely. **No corpus task
  has containment-coupled transport**: moving object A only by moving object B that
  cages it, under a lift-forbidden constraint where the obvious carry strategy
  destroys the goal condition — nor a goal state (a "closed box") that the solver
  assembles out of two free bodies.
- The mechanism is physically honest and proven, not asserted: the capture window is
  enforced by the walls themselves (a contained block passes it, an outside block
  fails it by >40 mm — asserted in `__post_init__` and measured in smoke 7), the
  height window rejects a roof-parked block (smoke 9), a 3.5 N push slides the trap
  100 mm with the block dragged along inside (smoke 12), and a perfect carry leaves
  the block behind (smoke 13).

## Solution outline (solve.py — the legitimacy certificate)

Teleports are TRANSPORT ONLY; every rubric-relevant fact is produced by contact:

1. **P0 settle + readback** — layout printed from readback; crate asserted upright,
   score ~0.
2. **P1 capture (gravity)** — one root-state write carries the crate to a free-space
   hover 18 mm above the red block, already inverted, zero velocity (satisfying
   nothing: `trapped()` demands a floor-seated crate). The crate FALLS ~58 mm; the
   walls descend around the block and the rim seats by contact. Asserts `trapped`,
   score 0.30.
3. **P2 haul (applied force + friction)** — a velocity-regulated horizontal force at
   the crate CoM (~0.12 m/s, clamp 5.0 N — under the ~0.33 Nm tipping margin; 2.6 N
   stiction floor; tilt guard; containment watch), world-frame direction with the
   pod force-frame quirk handled by `encode_force` mode PROBING from measured
   progress. Detours around the blue block if the straight path grazes it. The
   trapped block is dragged along inside by wall contact — never touched. Stops at
   20 mm per-axis in the pen frame, settles hands-off. Asserts `success`, score 1.0.
4. **P3 persistence** — ≥ 3.3 simulated seconds hands-off, success still holds, then
   `SIM_GEN_SOLVE: SUCCESS`.

Phases print `SIM_GEN_SCORE` at every boundary (non-decreasing, asserted).
**Verified on the forge: seeds 0, 1, 2 all SUCCESS (49 s / 38 s / 40 s wall),
score trace 0.00 → 0.30 → 1.00 → 1.00 each.**

## Rubric

Latched partial credit (never evaporates): 0.30 · red block EVER trapped under the
capped, calm crate + 0.20 · ever trapped within 15 cm of the pen + 0.25 · ever
trapped inside the pen window; cap 0.75; exactly 1.0 iff `success()` — trapped AND
in-pen AND blue clear AND settled AND finite, all judged LIVE. Null policy ≈ 0 (an
upright crate never latches anything). Demonstrated margins: the solve stops ≤ 2 cm
from the pen centre vs the 5 cm window; capture lands the block within ~5 mm of the
crate axis vs the 55 mm window.

## Embodiment argument (Franka, parallel-jaw)

- **Cap**: the crate walls are 8 mm thick — a natural parallel-jaw pinch (jaw opens
  ~80 mm). The crate is 450 g (payload 3 kg). Plan: pinch a wall, lift, invert with
  the wrist (the ±2.9 rad wrist roll covers a 180° flip), hold the inverted crate
  ~2 cm over the red cube and release — gravity performs the same capture the solve
  demonstrates. No precision beyond the 55 mm capture window is needed.
- **Haul**: the slide is a fingertip push low on a wall face: ≤ 5 N horizontal at
  ~0.12 m/s — trivially within Franka's force envelope; the closed gripper's
  fingertips are the natural end effector. Pads are flush markings, nothing to
  collide with.
- Workspace: crate spawns near (0.50, −0.06), blocks at (0.30, 0.08)/(0.47, 0.21),
  pen at (0.30, −0.22) (± jitter): everything within x 0.26–0.55, |y| ≤ 0.27, at
  heights 0–0.15 m. With the base at (0, 0, 0) facing +x all points lie at
  0.28–0.62 m reach.
- Execution order: **NONE required** — cap-then-slide (the demonstrated plan) and
  move-the-red-cube-onto-the-pen-then-cap-it-there are both valid; only the final
  state is judged (stated in `describe()`).

## Checks (smoke.py — rejection battery, 18/18 PASS on forge, frames.npz recorded)

1. settle + no-NaN, crate upright, score 0; 2. randomization readback (crate
xy 34 mm + yaw 166°, pen xy 32 mm + yaw 176°, red xy 202 mm between two seeds);
3. slot swap: red block seen at BOTH slots over 10 resets; 4. null policy 240 steps
→ score ~0; 5. seed-family naive (red block inside the UPRIGHT crate centred on the
pen) → score ~0; 6. wrong object (blue trapped and delivered) → rejected; 7.
near-miss capture (block 107 mm off-axis, just outside a wall) → not trapped; 8.
rim-perch (rim dropped ON the block: z 69 mm, tilt 16°) → not capped; 9. roof-park
(block at 120 mm on the inverted roof vs 65 mm window) → rejected; 10. restraint:
correct trap delivered but blue on the pen → no success, score capped 0.75; 11.
near-miss delivery (85 mm vs 50 mm window) → rejected; 12. MECHANISM slide: 3.5 N
push moves the trap 103 mm, block dragged along inside; 13. MECHANISM lift: a
perfect carry (250 mm) leaves the block behind (0 mm moved) — carrying delivers
nothing; 14. latched credit survives the carry while live `trapped()` drops; 15.
crate on its side → not capped; 16. settle gate: the exact success pose sliding at
0.45 m/s is refused; 17. rejection audit — success() never True at ANY step of the
battery; 18. frames.npz saved (92 frames, 960×600).

Run (forge):
`python -u -m simgen_tasks.close_box_i26.solve --headless [--seed N]`
`python -u -m simgen_tasks.close_box_i26.smoke --headless`
