# reach_and_drag_i298 — DropCourierScene (`simgen.drop_courier`)

Deliver a red cargo cube INTO a mobile open-top TRAY and park the loaded tray at the
delivery bay: slide the tray along its ground channel until it waits directly under the
cube's spot on a raised ledge's DROP EDGE, push the cube off the edge so GRAVITY drops
it inside the tray, then push the loaded tray along the channel to the end marked by
the GREEN POST. The cube is 90 mm — wider than the 80 mm parallel jaw — so it can never
be picked up, and the tray's walls (62 mm) are taller than the cube's centre (45 mm),
so a cube on the ground can only shove the tray, never enter it: a misdrop is
unrecoverable, which physically forces align-before-drop.

## Seed provenance

- **Seed task**: `rlbench/reach_and_drag` (RoboVerse
  `roboverse_pack/tasks/rlbench/reach_and_drag.py`) — "reach and drag": grasp a
  free stick, use it as a tool to drag a cube across the open table onto the one
  colored target square. Robot=franka, USD assets, judged by cube-on-target.
  One graspable tool, open-plane dragging, a flat unmissable target, no container,
  no ordering constraint, no gravity hand-off.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| what moves toward what | the CARGO is dragged to a fixed target | the plan is inverted: the RECEIVER moves — an open-top tray shuttles along a 1-D ground channel; the cargo only ever moves a few cm on its ledge and then FALLS |
| the "tool" | a free stick you grasp and wield | none — nothing graspable exists (cube 90 mm > 80 mm jaw; tray is pushed by its walls); the mobile CONTAINER is the means of transport |
| target | flat colored square, approachable from anywhere at any time | the INSIDE of a container whose walls (62 mm) are above the cube's centre (45 mm): entry exists ONLY from above, via the drop edge — and the guide wall (55 mm) refuses ground entry into the channel from outside |
| delivery | planar dragging all the way | a vertical GRAVITY hand-off (220 mm drop) that nothing steers after release, then transport BY CONTAINER (the cube rides inside the shuttling tray) |
| ordering | none — one monotone drag | FORCED two-stage order: align the tray under the drop line FIRST (a misdrop lands on the channel floor and is unrecoverable), then drop, then shuttle |
| perception | fixed target square | the delivery end swaps sides per episode (green post readback); rig xy/yaw, cargo slot along the edge, and tray start all randomized (readback-verified) |
| assets | RLBench USDs | 100 % procedural (kinematic ledge+wall+stops rig, dynamic compound tray with authored floor-bottom CoM, dynamic cube, kinematic marker post) |
| judging | cube on square | live geometric success (cargo contained in the TRAY BODY FRAME + tray parked at the bay + upright + settled) + latched stage credit (aligned 0.25, loaded 0.45, cap 0.70) |

Code shares nothing with the seed: `@SCENES.register` BaseScene, three custom compound
spawners, tray-body-frame containment, rig-frame alignment/parking predicates, latches
in `post_step`, `register_env(..., robot="null")`.

## Why strategically different

The seed's skill is *grasp a stick and sweep a cube across an open plane onto a flat
square* — cargo-centric planar transport with a hand-held tool, no ordering, target
approachable from anywhere. Here that plan earns nothing: smoke check 5 CONSTRUCTS the
seed's strategy (cube on the open ground, quasi-static 3x-weight drag aimed straight at
the tray) and it is geometrically refused — the drag travels 176 mm freely and ends
against the guide wall, 45 mm short of the channel, score 0. What the solver must bring
instead: (1) **receiver positioning** — the goal is not a place but a MOBILE CONTAINER
that must be staged under the drop line before delivery (the seed never moves its
target), (2) **a gravity hand-off** — delivery happens by pushing the cargo off a
ledge and letting it fall 220 mm into the waiting tray, with nothing steering the
fall (the seed's transport is entirely quasi-static and planar), (3) **a forced,
irreversible order** — dropping before aligning strands the cube on the channel floor
forever (90 mm > 80 mm jaw and walls above its centre: smoke checks 6 and 7 prove the
ground state is dead), (4) **transport by container** — the final leg moves the cargo
by moving the tray it sits in, judged in the tray's body frame, and (5) **perception**
of a per-episode side swap of the delivery bay (green post). A drive-by under the drop
line at speed earns nothing (`tray_still` gate, smoke check 8).

## Solution outline (as demonstrated by solve.py on the forge; ZERO teleports)

1. **P0** settle 1.5 s; layout readback (rig xy/yaw, cargo slot, tray start, bay sign
   via green post); tray guaranteed >= 100 mm off the drop line and >= 160 mm off the
   bay; baseline score 0.0.
2. **P1 align**: CoM-level horizontal force on the tray (the shove a fingertip on its
   end wall produces) under a velocity servo — |v| capped 0.12 m/s (0.05 near target),
   gain 25, force cap 9 N, stiction/creep floor 2.8→6.0 N that engages below
   max(0.04, v_des/2) m/s and escalates on measured slow progress (a pure P velocity
   servo otherwise settles into a ~0.01 m/s creep on the loaded tray — the one solve
   iteration this task needed), force-frame probe (mode toggles if the tray regresses —
   never fired). Ground friction brakes and holds. Exit at |align_err| < 8 mm, settle
   → `aligned` latch (score 0.25).
3. **P2 dispense**: velocity-regulated push on the cube toward the edge (force capped
   2.2 N < the m·g = 2.45 N tipping bound), accelerated to ~0.3 m/s for the exit so
   the CoM carries past the tray's back wall, RELEASED at the edge — gravity delivers
   it into the tray → `loaded` latch (score 0.70).
4. **P3 shuttle**: same tray servo to the green-post end; the cube rides inside
   (containment is tray-body-frame). Tray rests in the bay band → success (score 1.0).
5. **P4** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints: 0.0000 → 0.2500 → 0.7000 → 1.0000 → 1.0000.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m). The rig
centre is at (0.45, 0) ± 30 mm; the guide wall at ~0.35 m, the drop edge at ~0.55 m,
the ledge top at 0.22 m, the channel ends at ~0.62 m lateral reach — all inside the
dexterous shell, and the arm never reaches INTO the tray or over the ledge back.

- **Tray** (170 mm square, 12 mm-thick walls, 0.5 kg): slide it by pushing an end
  wall with a fingertip/closed gripper at CoM height (~30 mm), or pinch a wall top
  (12 ≪ 80 mm jaw) and drag. The channel constrains it to 1-D, so coarse pushes plus
  closed-loop nudges reach the ±25 mm align window; friction holds it when released.
- **Cargo cube** (90 mm, 0.25 kg): deliberately UNGRASPABLE (90 > 80 mm jaw). The
  dispense is a fingertip drag/push of ~2 N across the ledge top at z ≈ 0.27 m —
  toward the robot (−x), the natural pull direction. No precision anywhere: the tray
  mouth (146 mm) forgives ±28 mm of drop error, and the ±25 mm align window on the
  slow tray is comfortable for visual servoing.
- **Rig/post**: kinematic — incidental contact cannot move the goal frames. The green
  post stands outside the channel and is never touched.

## Execution order (declared)

`slide the tray under the drop line (to a stop) → push the cargo off the edge →
shuttle the loaded tray to the green bay`. REQUIRED and physically forced: dropping
first lands the cube on the channel floor where it is unrecoverable (ungraspable, and
the tray walls are above its centre — smoke 6/7), and parking first means nothing
falls into the tray. The rubric mirrors the physics: `aligned` demands the cargo still
on the ledge and the tray near-still; `loaded` is reachable only over the walls;
success additionally demands the loaded tray parked at the bay.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (bay at −y, cargo_y −24 mm, tray start +235 mm): SUCCESS,
  scores 0.0/0.25/0.70/1.0/1.0.
- `solve --seed 1` (bay at +y, cargo_y −163 mm, tray start +59 mm): SUCCESS.
- `solve --seed 2` (bay at +y, cargo_y +216 mm): SUCCESS — three seeds, both bay
  sides, monotone scores; the force-frame mode toggle never fired.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 11/11**, frames.npz (190 × 600 × 960) saved:
  1. settle/no-NaN: cargo on the ledge, tray in the channel, score 0, no success
  2. randomization readback differs across 3 seeds (max pairwise: rig xy 42 mm,
     rig yaw > 1°, cargo slot 259 mm, tray start 481 mm)
  3. bay-end shuffle: the green post stands at BOTH ends over 14 resets
     (world-pose readback, consistent with `bay_sign`)
  4. null policy: 300 idle steps, tray drift 0.0 mm, cargo on the ledge, score 0
  5. SEED STRATEGY: quasi-static 3x-weight ground drag aimed at the tray travels
     176 mm freely, ends refused at the guide wall (x −144 mm vs wall face −99 mm),
     never contained, score 0
  6. GROUND LOAD SHOVE: cube on the channel floor shoved at the tray 3x-weight —
     it moves 97 mm and shoves the TRAY 82 mm instead of entering (walls above its
     centre); never contained
  7. WRONG ORDER: tray parked at the bay first, cargo sent off the edge — lands on
     the channel floor (z 45 mm), not contained, score 0
  8. FLYBY: tray swept under the drop line at 0.32 m/s (min|err| 0.2 mm) —
     `aligned` does NOT latch (`tray_still` gate)
  9. near-miss park: loaded tray 60 mm short of the bay band — contained but not
     parked, no success, score 0.450 <= 0.4501
  10. flipped tray: cargo resting ON the upside-down tray (z 107 mm, supported by
     the tray) — not contained (body frame), not upright, no credit
  11. video frames.npz saved

## Files

- `scene.py` — DropCourierScene + rig/tray/post compound spawners + rubric;
  registers `simgen.drop_courier`.
- `solve.py` — tray velocity-servo align/shuttle + capped cube dispense certificate
  (`--seed N`).
- `smoke.py` — 11-check rejection battery + video.
