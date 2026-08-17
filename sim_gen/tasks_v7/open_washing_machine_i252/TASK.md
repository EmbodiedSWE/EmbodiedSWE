# open_washing_machine_i252 — Dose the washer: open the detergent drawer, drop the pod in the blue-marked compartment, slide it shut

Env: `simgen.detergent_drawer` (scene `detergent_drawer`, robot `"null"`).

## Seed provenance

Seed task: `rlbench/open_washing_machine`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/open_washing_machine.py`) — a Franka
grabs the washing machine door handle and swings the hinged door open. The entire
plan is ONE pull on ONE articulation, and the articulation's end state (door open)
IS the goal.

## What changed and why it is strategically different

| | seed | this task |
|---|---|---|
| articulation | revolute door | prismatic detergent drawer (spawn-authored joint, damping parks it — no spring) |
| role of the articulation | its end pose IS the goal | a GATE that must end where it started (shut) |
| plan | one pull | three ordered phases: pull open -> selective deposit -> push shut |
| discrimination | none (any opening works) | two identical cells; the correct one is marked by a BLUE tile whose side is re-sampled every episode (tiles physically swapped); the white cell fails |
| ordering | trivial | enforced by GEOMETRY, not rubric fiat: shut, the hood leaves a ~6 mm slit over the cells — a 22 mm pod bounces off (smoke check 7); a correct deposit left in an open drawer is also not success |

So the strategy is inverted end-to-end: the seed opens an articulation and stops;
here opening is merely instrumental, the scored object is a free body that must end
up sealed INSIDE the articulation, and the articulation must be returned to its
initial (shut) state with the payload riding inside. It also differs from the other
packages inspected while building it (`press_switch_i161`: rotary setpoint
regulation on free-spinning dials; `lamp_off_i247`: rotor toggle;
`robobench/suites/packing/pen_holder`: multi-object insertion into a static cup) —
none of them contain an open-deposit-close cycle through a payload-carrying
articulation.

## Scene

- Washer console on a table: ONE kinematic compound (bottom block, two pillars, hood,
  rear wall) forming a rectangular mouth. Never teleported, so it is a safe joint
  anchor.
- Detergent drawer: ONE dynamic compound (base plate, four walls, center divider ->
  two open-top cells, front handle plate on a neck), hung on a spawn-authored
  `UsdPhysics.PrismaticJoint` (console -> drawer, local +x, limits `[0, 0.13]`),
  linear damping 5/s parks it where released; mass 0.30 kg.
- Green pod: 22 mm dynamic cube on the table (20 g).
- Two kinematic marker tiles on the console face above the drawer, physically
  swapped at reset: BLUE over the sampled main-wash cell, WHITE over the softener
  cell.
- `post_step` owns the wrench slots: `drive_f` (drawer pull axis) and `pod_f`
  buffers; it also latches rubric progress.

Per-episode randomization (readback-verified in smoke check 1/3): target side
(left/right, tiles swapped), pod start pose (x, y incl. side, yaw), drawer initial
crack 0–12 mm (always << the 90 mm open latch and too narrow to admit the pod).

## Rubric (anchored in the demonstrated solve trajectory: 0 -> 0.20 -> 0.65 -> 1.0)

- `success()` = drawer SHUT (`opening < 6 mm`) AND pod resting inside the target
  cell, judged in the DRAWER'S BODY FRAME (so containment rides the closing push)
  with a z-window that rejects rim/divider perching, AND drawer+pod settled.
- `score()` = 1.0 iff `success()`; else `0.20*opened_latch` (geometric, latched —
  correct behavior must UNDO the opening) `+ 0.25*deposit_latch` (pod seen at rest
  in the target cell; latched through a slow gate, so a pod flying through the cell
  volume earns nothing) `+ 0.20*in_cell` (live containment). Null policy ~0; latched
  credit never evaporates under correct behavior.

## Solution outline (solve.py, robot="null", teleport = transport only)

1. OPEN — velocity-cascade force servo on the drawer body (KV·dt/m ≈ 0.42 < 1
   against the one-substep wrench delay; F ≤ 6 N) pulls the live joint to 8 mm short
   of the stroke; released close AND slow. Latches `opened` -> score 0.20.
2. DEPOSIT — the pod is teleported ONLY to free air ~20 mm above the exposed target
   cell (assert: clear of the hood footprint), then FALLS in and settles by contact
   on the tray floor. Slow-gated `deposit` latch -> score 0.65.
3. SHUT — the same servo pushes the drawer home (gentler rate cap: end stop ahead)
   with the pod riding inside; released at the lower stop -> success, score 1.0.
4. PERSIST — 3.5 simulated seconds hands-off (drive buffers asserted zero), success
   re-checked, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0 (target −y, crack 11.7 mm) and 1 (target +y, crack
8.1 mm), both `SIM_GEN_SOLVE: SUCCESS` with the monotone score trace
0.000 / 0.200 / 0.650 / 1.000 / 1.000.

## Embodiment argument (a real Franka could do this)

Base on the table at ~(0.55, 0.0, 0.40), facing −x toward the console (~0.46 m to
the console face — mid-workspace). Per object:

- Drawer OPEN: the handle is an upright 56×32×8 mm grip plate on a 20 mm neck
  standing proud of the front wall — a standard two-finger side pinch (8 mm plate
  between the fingertips, 20 mm of neck clearance behind it); pull straight −x…+x
  along the rail, ≤ 6 N against a 0.3 kg drawer with 5/s damping (an easy one-hand
  pull). The solve's capped-force, capped-velocity servo is exactly a compliant
  gripper pull.
- Pod PICK+PLACE: a 22 mm cube on an open table — canonical top grasp; carry over
  the pulled-out drawer and release ~20 mm above the blue-side cell (the exposed
  open-top cell is ~120×85 mm — generous release tolerance); the solve's free-air
  release + gravity settle mirrors this.
- Drawer SHUT: push on the grip/front face along +…−x until flush; the gentle rate
  cap (0.10 m/s) is a natural guarded move into the end stop.

No phase needs more than one hand, exotic wrenches, or poses outside the Franka's
reach envelope; every contact surface is a plain box face ≥ 8 mm across.

## Execution order — declared and physically enforced

Declared order: OPEN before DEPOSIT before SHUT. It is enforced by geometry, not by
the rubric reading a phase variable: shut, the hood-to-rim slit is ~6 mm, so the
22 mm pod cannot enter (smoke check 7 drops it and it lands ON the hood — deposit
latch stays 0); and success additionally requires the drawer back at shut with the
pod inside, so stopping after any prefix (open only — check 6; open+deposit but
ajar — check 9) is not success.

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 12/12` on the forge)

1. settle — clean reset; drawer AT its sampled crack and pod AT its sampled pose by
   READBACK; score ~0.
2. markers — blue/white tiles physically posed over the sampled sides (readback vs
   the cfg formula, < 2 mm).
3. random — 8 seeds: both sides drawn, crack + pod pose vary, all readback-verified.
4. null — 2 s of nothing: score < 0.05, no success.
5. end stop — servo commanded 50 mm past the stroke parks at the ~130 mm joint stop.
6. seed strategy — drawer opened and LEFT OPEN: opened latch only (~0.20), no success.
7. ordering is geometry — pod dropped over the SHUT drawer lands on the hood: never
   in the cell, deposit latch 0, score ~0.
8. wrong compartment — pod rests in the WHITE cell, drawer shut: no deposit latch,
   ~0.20, no success.
9. near miss — correct deposit, drawer parked ~30 mm ajar: ~0.65, no success.
10. exactness — full correct sequence: success() and score == 1.0, stable 1 s
    hands-off.
11. achievement latch — reopening revokes success; latched credit (~0.65) remains.
12. frames — 280 rgb frames recorded and saved as frames.npz.
