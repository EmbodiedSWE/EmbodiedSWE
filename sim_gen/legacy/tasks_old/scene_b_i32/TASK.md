# scene_b_i32 — shell_cover (hide the marked cube under an upturned cup, hands off the cube)

**Registered as:** `SCENES["shell_cover"]`, env `sim_gen.shell_cover` (robot="null", scene-level).
**Tier: easy — 2 stages** (flip the cup mouth-down, cap it over the marked cube).
**Execution order:** the flip must precede the final set-down (inherent — a mouth-up cup
covers nothing); no other ordering exists.

## Seed provenance

- Seed: `calvin/scene_B`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/calvin/scene_B.py`) — the CALVIN table B:
  an articulated table (button / switch / slider / drawer) plus three colored blocks
  (pink middle, blue big, red small). Every seed task **grasps and transports a block**
  (lift / place / stack / put-in-drawer) or pulls an articulated handle; the block is
  always the payload, judged by where the robot carried it.

## What changed

The three seed blocks are kept (procedural stand-ins, same identities: pink 40 mm /
blue 50 mm / red 30 mm) but become **untouchable**. New assets: an open octagonal cup
(inner inradius 55 mm, 90 mm tall, mouth-UP at spawn) and a kinematic beacon post whose
flag recolors per episode to mark ONE cube. Goal: **conceal the marked cube** — flip
the cup upside down and set it over the cube so the rim rests flush on the floor and
the cube is fully inside the aperture, while the cube itself is never lifted or slid.
A per-cube `disturbed` latch (xy drift > 30 mm or lift > 25 mm) fires permanently; a
disturbed marked cube caps the score at 0.2 and blocks success forever.

Per-episode randomization: marked-cube identity (beacon-flag displayColor rewrite),
cube-to-slot permutation on the scatter arc, per-body xy jitter + free yaw, cup pose
jitter + yaw.

## Why strategically different (a different PLAN, not different numbers)

- **The graded object is never manipulated.** CALVIN's plan skeleton is
  grasp-block → transport → release-at-goal. Here any grasp/lift/slide of the marked
  block is *the failure mode* (irreversible latch). The solver manipulates the OTHER
  body — the container — and the block's pose must be exactly preserved.
- **Inverted containment relation.** The seed (and its drawer variant) puts the payload
  *into* a receptacle. Here the receptacle is carried onto the payload: cover, not
  insert. Success geometry is judged cup-around-cube with the cube untouched, plus a
  180° reorientation (mouth-up → mouth-down) the seed never needs.
- **Goal is read from the scene.** The seed's goal is fixed per task; here the beacon
  flag samples which cube matters, so a memorized fixed-target policy fails (covering
  the wrong cube is a tested control).
- Tested seed-strategy control: dropping the marked block INTO the mouth-up cup —
  genuine, physically verified containment, exactly the seed's transport plan — trips
  the disturb latch and scores ≤ 0.2 with no success.

Sibling-axis note (parallel batch): differs from `close_box_i13` (lid seated onto a
box RIM, cargo actively unloaded) and `living_room_scene2_i14` (payload inserted
through an aperture) — here nothing is inserted and the covered object is a hands-off
bystander; differs from `base_i25`'s anti-carry latch in that i25's payloads must be
*toppled in place* while here the scored object must remain exactly at rest and a
second body is transported over it.

## Rubric

- 0.15 latched: cup left its spawn pose (moved > 60 mm or lifted > 30 mm).
- 0.40 latched: cup seen mouth-down (up-axis z < −0.5).
- 0.70 latched: cup mouth-down + rim-flush within 55 mm of the marked cube.
- 1.00 iff success: cup settled mouth-down, rim within 8 mm of the floor, marked cube's
  footprint inside the inner octagon (cup-axis distance < 55 mm − cube xy circumradius:
  34/27/20 mm tolerance for red/pink/blue), cube top under the interior ceiling,
  everything settled, marked cube never disturbed.
- Disturbed marked cube: score clamped to 0.2, success impossible (anti-seed-plan and
  anti-capture-drag: plowing the cube somewhere convenient before capping also fails).

Physics honesty: the oracle teleports (kinematic pick + flip), but every judged
predicate is a settled physical outcome — the rim genuinely drops 30+ mm around the
cube and concealment/flushness/stillness are read back from sim.

## Check list (smoke.py, 17 checks — forge result: `SIM_GEN_SMOKE: ALL PASS 17/17`)

1. settle/no-NaN (cup mouth-up, score 0)
2. randomization is real (pose readback differs across seeded resets)
3. beacon command varies (marked-cube index differs across resets)
4. null policy scores ~0
5-7. oracle × 3 seeds (success, score 1.0, milestones latched, cube undisturbed;
   local draws cover all three cube sizes as targets)
8. rubric monotonicity (0 → 0.15 → 0.40 → 1.0 strictly increasing)
9-10. seed-strategy control, 2 assertions (block INTO cup: containment verified real,
   then score ≤ 0.2 / no success)
11. near-miss cap 40 mm off (tolerance control; not concealed, no success)
12. cap beside the cube (flipped-only 0.4, no success)
13. wrong-cube perfect cap (semantic control; ≤ 0.45, no success)
14-15. anti-capture-drag, 2 assertions (drag latches disturbed; then perfect cap:
    covered geometrically, still ≤ 0.2, no success — latch irreversibility)
16-17. calibration sweep (red target, conceal bound 33.8 mm: 0/10 mm conceal 6/6,
    45/60 mm conceal 0/6; measured 30 mm 3/3, 45 mm 0/3 — cliff lands on the
    33.8 mm geometric bound)
