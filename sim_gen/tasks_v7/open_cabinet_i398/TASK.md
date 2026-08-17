# open_cabinet_i398 — Nooked cabinet: drag the whole cabinet away from the wall so the door CAN swing, then open it, extract the cube, deliver to the mat

Env: `simgen.nook_cabinet` (scene `nook_cabinet`, robot `"null"`).

## Seed provenance

Seed task: `mujoco_playground/open_cabinet`
(`sim_gen/RoboVerse/roboverse_pack/tasks/mujoco_playground/open_cabinet.py`) — a
Panda reaches a WALL-MOUNTED cabinet (`fix_base_link=True`), grasps the handle
and swings the hinged door open. The cabinet base cannot move by construction,
the door is free to swing from step zero, and the door's opened pose IS the
goal.

## What changed and why it is strategically different

| | seed | this task |
|---|---|---|
| cabinet base | fixed to the world (`fix_base_link=True`) | a free-standing 1.8 kg dynamic body parked in a NOOK: its door-side face only 18–28 mm from a fixed wall |
| can the door open at t=0? | yes — just pull | no — the knob strikes the wall after < 18 deg of swing; the blocker is a SPATIAL RELATION (cabinet-vs-wall pose), not any joint or lock, so no manipulation of the door itself can fix it |
| how the blocker is cleared | n/a | relocate the WHOLE cabinet (drag it by the roof carry-bar ≥ 250 mm of face-to-wall daylight) while keeping it upright — the seed's immovable base becomes the object you must move |
| role of the articulation | its end pose IS the goal | instrumental GATE only: the door must pass 45 deg so the sealed cube can leave, but the door angle is never the scored outcome |
| payload | none — empty cabinet | a 45 mm cube sealed behind the shut door must be extracted through the mouth and delivered to a mat on the OTHER side of the room |
| goal predicate | door joint angle | cube resting on the mat AND door open past 45 deg AND the cabinet still upright at rest |

So the inversion is: the seed rewards swinging an articulation on an immovable
base; here the base's mobility is the whole puzzle — the articulation is
unopenable until the room-scale spatial relation is repaired by transporting
its parent body, and the articulation's angle is only an instrumental gate for
a payload extraction. It also differs from every other corpus package built
around blocked articulations: `open_washing_machine_i393` gates its door with a
mass-keyed pay-lever (a mechanism INSIDE the scene solved by depositing a
payload), `open_washing_machine_i252` cycles a drawer that must RETURN, and no
corpus task makes "move the parent furniture out of its nook" the enabling
step for an articulation.

## Scene

- Wall: kinematic slab, inner plane at world x = 0.70 (0.06 thick, 1.6 long,
  0.40 high). Never moves; the fixed half of the spatial relation.
- Cabinet: ONE dynamic compound shell (interior 150 x 200 x 150 mm, 12 mm
  panels: floor, roof, back, both sides + a roof carry-bar spanning 90 mm at
  30 mm clearance — an obvious handle for dragging), mass 1.8 kg, mu 0.25 on
  ground mu 0.30 (slides under ~6 N, far below tipping).
- Door: dynamic compound (blue panel 212 x 150 x 12 mm + knob bar protruding
  30 mm) on a spawn-authored `UsdPhysics.RevoluteJoint` (cabinet -> door,
  axis z at cabinet-local (0.083, 0.106), limits [-2 deg, +150 deg]); angular
  damping 12/s makes it a STIFF hinge — it parks where released and a cabinet
  drag cannot ratchet it open by inertia; deliberate opening takes a sustained
  ~0.03 N m (a fingertip's ~0.2 N at the knob).
- Cube: red, 45 mm, 60 g, sealed in the interior behind the shut door.
- Mat: kinematic green pad 150 mm square across the room (x in [0.10, 0.25],
  |y| in [0.33, 0.40], side sampled per episode).
- `post_step` owns three wrench slots (`cab_f`, `door_tau`, `cube_f`), applied
  each substep with world->body re-encode and bang-bang velocity caps
  (0.30 m/s cab, 2.0 rad/s door, 0.25 m/s cube); progress latches only engage
  through calm-speed gates.
- cfg `__post_init__` asserts honesty by construction: at the WIDEST sampled
  nose gap + yaw slack the knob strikes the wall below 0.4x open_min (~18 deg
  << 45 deg), and clear_min daylight passes open_min with >= 30 mm margin.

Per-episode randomization (readback-verified, 3-seed max-pairwise): cabinet
slot along the wall (+/-0.12 m), nose gap 18–28 mm, spawn yaw +/-2 deg, cube
pose inside the interior, mat position AND side of the room.

## Rubric (anchored in the demonstrated solve trajectory: 0 -> 0.30 -> 0.50 -> 0.75 -> 1.0)

- `success()` = cube resting centred on the mat (xy within 55 mm per axis,
  z window rejects perching on the cabinet or stacking) AND door past 45 deg
  AND cabinet upright at floor rest height (tilt < 10 deg — you cannot tip it
  over to shake the cube out) AND everything settled.
- `score()` = 1.0 iff `success()`; else latched partial credit
  `0.10*moved` (cabinet displaced >= 50 mm) `+ 0.20*clear` (face-to-wall
  daylight >= 250 mm) `+ 0.20*open` (door past 45 deg, calm) `+ 0.25*out`
  (cube at rest outside the shell footprint), capped at 0.75. Null policy ~0;
  latches only engage below calm-speed gates so flings and crashes do not
  ratchet credit.

## Solution outline (solve.py, robot="null", teleport = transport only)

1. SETTLE + readback — assert the nooked layout (score <= 0.03, cube inside,
   door shut, clearance < 0.10).
2. DRAG — 6 N force servo on the cabinet body along its own -x (away from the
   wall) until face-to-wall daylight >= 0.32 m; release, let it park. The door
   must NOT fly open here (asserted < open_min - 10 deg) and the score must be
   exactly the moved+clear 0.30 -> 0.30.
3. OPEN — PD torque servo on the hinge (kp 0.30, kd 0.05, cap 0.08 N m,
   target 110 deg), exit past 95 deg calm -> 0.50.
4. EXTRACT — 0.6 N velocity-capped push on the cube along the cabinet's +x,
   through the mouth and out past the shell footprint; released calm -> 0.75.
5. DELIVER — the cube (now in the open) is teleported to free air 30 mm above
   the mat (point-to-footprint-box clearance asserted against cabinet, door
   and wall), falls, settles -> 1.0.
6. PERSIST — 400 hands-off steps (wrench slots asserted zero), success
   re-checked, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0 and 1, both `SIM_GEN_SOLVE: SUCCESS` with the
monotone score trace 0.000 / 0.300 / 0.500 / 0.750 / 1.000 / 1.000.

## Embodiment argument (a real Franka could do this)

Base on the floor plate at ~(0.15, 0, 0), facing +x toward the nook (~0.45 m
to the cabinet — mid-workspace; the mat spawns within 0.45 m on either side).
Per phase:

- DRAG: the roof carry-bar is a 90 mm free-spanning rod at 30 mm clearance —
  the canonical two-finger wrap. Pulling 6 N horizontally against a 1.8 kg
  mu-0.25 cabinet is an easy one-hand drag; the velocity-capped force servo is
  exactly a compliant arm pull. Total travel ~0.3 m stays in the envelope.
- OPEN: the knob bar protrudes 30 mm from the panel — two-finger wrap, swing
  along the arc; 0.03–0.08 N m at the 180 mm knob radius is under 0.5 N of
  fingertip force. The PD-with-cap servo mirrors a compliant wrist.
- EXTRACT: the roof forbids top grasps while inside; the cube is reached
  THROUGH the 150 x 200 mm mouth (opened door swings clear past 95 deg) and
  slid out over the flush floor lip with fingertips — the 0.6 N capped push
  is a fingertip rake. Once out on the open floor it affords a standard
  45 mm top pinch.
- DELIVER: carry ~0.5 m and release 30 mm above the 150 mm mat — generous.

No phase needs more than one hand, exotic wrenches, or poses outside the
reach envelope; every contact surface is a plain bar or face >= 12 mm across.

## Execution order — declared and physically enforced

Declared order: DRAG before OPEN before EXTRACT (DELIVER last). Enforced by
geometry, not rubric fiat: the wall stops the knob after < 18 deg of swing at
any sampled gap (cfg assert + smoke check 4), and shoving the door harder just
presses the knob into the wall, so OPEN physically requires DRAG; the shell is
sealed on five sides and the shut door covers the mouth (the 45 mm cube
cannot pass the 12 mm ground clearance — smoke check 5), so EXTRACT requires
OPEN; and stopping after any prefix (dragged + opened but cube inside —
near-miss checks 8/9) is not success. The rubric additionally demands the
enabling state STAND at the end: door still past 45 deg, cabinet still
upright, cube still on the mat (revocation — check 11).

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 12/12` on the forge)

1. settle — clean reset; nooked layout by READBACK (clearance < 0.09, door
   shut, cube inside, upright); score ~0.
2. random — 3 seeds: nose gap, wall slot, cube and mat poses all vary
   (max-pairwise deltas); both mat sides appear over 8 resets.
3. null — 240 steps of nothing: score <= 0.05, no success.
4. nook blocks the door — the solve's own 0.08 N m on the hinge, then 3x
   escalation to 0.24 N m: door peaks > 0.5 deg (hinge live, not fused) but
   < 15 deg, cabinet undisplaced, no open credit — the spatial blocker holds.
5. shut door seals — the solve's own 0.6 N extraction push on the cube for
   240 steps: the cube stays inside, zero out credit.
6. freed door opens — the SAME 0.08 N m after the cabinet is placed clear of
   the wall: door passes 50 deg — the mechanism differential (it was only
   ever the wall).
7. delivery bypass — cube constructed resting on the mat, cabinet still
   nooked and shut (the "just deal with the cube" breach): no success,
   score <= 0.30.
8. near-miss door — dragged clear but door parked at 35 deg < open_min:
   no open credit, score <= 0.56.
9. near-miss mat — full chain but cube parked 12 cm off the mat centre:
   no success, score <= 0.76.
10. toppled — cabinet on its back (mouth up), door hanging past 45 deg, cube
    on the mat: upright() False -> no success, score <= 0.76 — you cannot
    dump the cabinet over as a shortcut.
11. exactness + revocation — full correct end state: success and
    score >= 0.999, stable 120 steps; removing the cube from the mat revokes
    success, latched credit remains.
12. frames — rgb frames recorded and saved as frames.npz.
