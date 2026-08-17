# port_dropbox — post the red sauce can end-on through the drop-box's side port

`simgen.port_dropbox` · scene-level (robot="null") · package
`living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i203`

## Provenance

Seed: `libero_90/living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket`
(RoboVerse `roboverse_pack/tasks/libero_90/...`). The seed is a one-stage free
pick-and-place: grasp a tomato-sauce can standing in the open, carry it over an
OPEN-TOPPED basket, release, and pass a bounding-box containment check. Objects kept:
a red sauce-can-sized cylinder (the target) and a container. Everything else is new,
procedural geometry.

## What changed and why it is strategically different

**vs the seed** — the container is SEALED on top by a full roof; its only entry is an
80 mm square PORT partway up one side wall, fronted by an exterior loading tray with
guide rails. The seed's entire plan (carry above, let go) is a constructed negative
here: a can released over the box settles ON the roof (smoke check 4). The can is
64 mm across but 110 mm long, so the port admits it END-ON only — upright or sideways
presentations jam geometrically (smoke check 5). The winning plan needs stages the
seed never has: **reorient** the upright can to horizontal, **stage** it on the tray
between the rails aligned with the port axis, **push it axially** through the port
tunnel until its CoM crosses the inner wall face, where it tips and gravity finishes
the job irreversibly. A beige distractor can of identical shape adds a wrong-object
clause (`success()` requires the beige can NOT inside). Code structure differs too:
kinematic compound fixture authored as child colliders, wrench-driven plant in
`post_step`, latched two-stage rubric — nothing like the seed's `_terminated` bbox.

**vs the sibling `..._i8` (sliding-lid hamper)** — that task's crux is a MECHANISM:
open a captive sliding lid, insert through the top, close the lid. This task has NO
moving parts at all; the crux is object REORIENTATION plus a threaded, tolerance-bound
axial insertion through a fixed aperture. Different plan (reorient/stage/push vs
open/drop/close), different rubric anchors, different failure modes.

## Scene

- **Drop-box** (one kinematic compound body; base-slab root + authored child
  colliders): 28 cm square footprint, 32 cm tall with roof. Front wall carries the
  80 × 80 mm port (sill at z = 0.14 m, lintel 0.22 m), outlined by a white visual
  frame. Exterior gray loading tray (17 cm long, top at 0.141 m = 1 mm ABOVE the sill,
  a step down into the port so nothing catches) with two dark guide rails 82 mm apart.
- **Red can** (target): dynamic cylinder r = 32 mm, l = 110 mm, 350 g.
- **Beige can** (distractor): identical shape.
- **Randomization (readback-verified in smoke check 2)**: box yaw ±15° + 2.5 cm xy
  jitter; both cans' ground poses jittered and the red/beige SIDES SWAP 50/50, so "the
  can on the left" is not memorizable.

## Rubric (anchored in the demonstrated solve trajectory)

- `0.25 · staged_latch` — the red can has RESTED lying horizontal on the tray between
  the rails, aligned with the port axis (slow-gated: only sets below 0.10 m/s).
- `0.35 · insert_latch` — latched running max of gated axial CoM progress from the
  tunnel mouth to the commitment line 2 mm past the inner wall face (gated on lying,
  aligned, in the channel, below 0.6 m/s — no ballistic credit).
- `1.0` iff `success()`: red can physically inside (bin-frame footprint, centre below
  sill − 3 cm — roof/tray/straddle rests never count), beige can not inside, red can
  settled (a position-delta stillness streak, ~0.4 s under 0.3 mm/substep — velocity
  readback is phantom under the wrench plant, positions are authoritative). Partial
  credit capped at 0.60; ~0 for null; latches never evaporate.

## Teleport solution (solve.py)

Teleports do TRANSPORT ONLY; all load-bearing interaction is contact dynamics.

- **P0** reset, settle, layout readback. `SIM_GEN_SCORE` ~0.
- **P1 TRANSPORT** (the only pose write on the red can): from its ground spawn to a
  free rest pose LYING on the tray, axis along the port, nose 30 mm outside the outer
  wall face, 3 mm above the tray → free drop, rest, slow-gated staged latch. ~0.25.
- **P2 INSERT**: a velocity-regulated horizontal push force at the can's CoM along the
  port axis (bang-bang: F while axial speed < 0.12 m/s; 3.5 N escalating +1.5 N per
  1.5 s of stall up to 9 N — fingertip-scale on a 350 g can) slides it across the
  tray, over the sill, through the tunnel with rails/jambs/sill/lintel acting the
  whole way. Force cuts once the CoM is 12 mm past the inner wall face (10 mm beyond
  the commitment line, covering the rubric's one-step sampling lag); the can tips and
  falls to the box floor UNAIDED. 1.0.
- **P3 persistence**: ≥ 3.5 more simulated seconds hands-off (push buffer asserted
  zero), success must still hold → `SIM_GEN_SOLVE: SUCCESS`.

Scores asserted non-decreasing across phases. Passes required on seeds 0 and 1.

## Embodiment: single Franka arm, OSC, one base pose

Base at roughly (−0.38, 0), facing the tray mouth; everything actuated lives in a
~55 cm-radius workspace at heights 0–25 cm.

- **Red can, reorient + place**: side pinch on the upright can (64 mm diameter fits
  the 80 mm jaw with margin), lift, a 90° wrist rotation lays it horizontal, set down
  on the tray between the rails. The 82 mm rail channel vs the 64 mm can gives ±9 mm
  lateral tolerance and the rails self-align the axis — well above OSC noise.
- **Red can, push**: fingertip (closed gripper) on the can's rear end face at
  ~0.17 m height, pushing straight along the rail channel — a planar quasi-static
  push, the rails and jambs absorb lateral error; required force ≤ 9 N is trivially
  within Franka payload. The commitment line is ~2 cm inside the outer face, so the
  fingertip never needs to enter the tunnel more than a fingertip's depth.
- **Beige can / box**: never touched.

## Execution order

Staging-before-push is geometrically forced (a can not lying aligned in the channel
cannot enter the port), not rubric-imposed; there is no additional order constraint.
The insert latch's gate (lying + aligned + in-channel) simply mirrors the geometry.

## Smoke battery (smoke.py) — 13 checks, rejection-only, recorded

1. clean reset: finite, cans upright AT their sampled spots (readback), score ~0;
2. randomization real over 8 seeds: yaw both signs (≥5 distinct), red/beige sides
   swap, positions vary, cans ≥ 9 cm apart, clear of the tray — READBACK;
3. null policy 2 s: score < 0.05, no success;
4. seed's strategy (release above the container): settles ON THE ROOF, never inside;
5. sideways at the port, pushed with solve-scale force: ADVANCES then JAMS outside
   (110 mm cannot pass 80 mm), no credit;
6. staged-only: exactly the staged share, no success;
7. near miss straddling the sill (CoM outside): settled, NOT inside, partial only;
8. latch: pulling the straddler back to open ground leaves credit unchanged;
9. wrong object: beige constructed inside, red out — no success, ~0;
10. beige-outside clause: red inside TOO, still no success;
11. audit: success() never fired at ANY step of the battery;
12. finite: no NaN anywhere;
13. frames ≥ 20 recorded → `frames.npz` in CWD.

Verdict line: `SIM_GEN_SMOKE: ALL PASS <n>/<n>`.

## Files

- `scene.py` — cfg + scene + guarded registration (`port_dropbox`,
  `simgen.port_dropbox`).
- `solve.py` — teleport solution (phases above).
- `smoke.py` — the 13-check battery, camera-recorded.
