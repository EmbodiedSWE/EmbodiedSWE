# reach_and_drag_i111 — CarouselDispatchScene (`simgen.cargo_carousel`)

Deliver a red cargo cube INTO the red garage by operating a free-spinning CAROUSEL:
rotate the platter (push its rim or the yellow handle peg) until the cargo bay's one
open side faces the red garage's mouth, then push the cube radially outward so it
slides out of the bay, across the gap, through the mouth, and comes to rest fully
inside. The cube is 90 mm wide — wider than the 80 mm parallel jaw — so it can never
be picked up; the mechanism (a joint-free pivot bearing with dry-friction braking)
is the tool, and aiming it is most of the task.

## Seed provenance

- **Seed task**: `rlbench/reach_and_drag` (RoboVerse
  `roboverse_pack/tasks/rlbench/reach_and_drag.py`) — "reach and drag": grasp a
  free stick, use it as a tool to drag a cube across the open table onto the one
  colored target square. Robot=franka, USD assets, judged by cube-on-target.
  One graspable tool, open-plane dragging, a flat unmissable target, no mechanism,
  no orientation reasoning, no delivery aperture.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| the "tool" | a free stick you grasp and wield | a constrained MECHANISM — a platter on a joint-free pivot bearing (round stub in a square well, dry friction brakes and centres it); you operate it, you cannot carry it |
| what dragging achieves | moves the cube across an open plane | rotation only AIMS the bay; a separate radial push performs the transfer — two distinct sub-skills in forced sequence |
| target | flat colored square, approachable from anywhere | a covered GARAGE with one mouth: floor at deck height (blank face below — ground approach is geometrically impossible, the deck-to-plinth gap is 10 mm vs a 90 mm cube), roof blocks drop-ins; entry exists ONLY along the aligned bay axis |
| perception | fixed target square | three garages (red/green/blue) shuffled over three slots every episode; pedestal xy/yaw and platter start angle randomized (readback-verified) |
| grasping | the stick is the graspable object | NOTHING load-bearing is graspable: the cube exceeds the jaw span, the platter is a 320 mm disc; the whole task is non-prehensile pushing |
| dynamics constraint | none | rotate GENTLY: an aligned flyby at speed does not count (`spin_still` gate) and fast spinning risks flinging the cargo; the bay's rails+backstop retain it against a 3x-weight shove |
| assets | RLBench USDs | 100 % procedural (compound kinematic pedestal, compound dynamic platter with bay + handle, three compound kinematic garages) |
| judging | cube on square | live geometric success (cargo fully past the mouth plane, on the garage floor, settled, platter still) + latched stage credit (aligned 0.25, delivered 0.35, cap 0.60) |

Code shares nothing with the seed: `@SCENES.register` BaseScene, three custom
compound spawners, dock-frame predicates, latches in `post_step`,
`register_env(..., robot="null")`.

## Why strategically different

The seed's skill is *grasp a stick and sweep a cube across an open plane onto a
flat square* — one free tool, translation-only, target approachable from any
direction at any speed. Here that plan earns nothing: smoke check 6 CONSTRUCTS the
seed's strategy (cube on the open ground, dragged quasi-statically straight at the
red garage's mouth) and it is geometrically refused — the mouth floor sits at deck
height with a blank face below, so the ground path ends against the plinth with
score ~0. What the solver must bring instead: (1) **mechanism operation** — spin a
platter on a friction pivot bearing and STOP it inside an 8° window (a
rate-limited, braked rotation; the seed has no rotational reasoning at all),
(2) **perception** of a per-episode color shuffle over three identical garages,
(3) **non-prehensile manipulation throughout** — the cube physically cannot be
grasped (90 mm > 80 mm jaw) and the platter cannot be carried; the seed's central
act (grasping the tool) is impossible here, (4) **a forced two-stage order**
(aim, then eject — ejecting misaligned throws the cargo away from every mouth),
and (5) **speed discipline**: the aligned-credit gate requires the platter
near-still, and smoke proves a 1.9 rad/s flyby through 0.2° of error latches
nothing.

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.5 s; layout readback (pedestal xy/yaw, red slot, platter
   centred < 12 mm on its bearing, start error 30–150°); baseline score 0.0.
2. **P1 aim** (pure z-torque on the platter — yaw-invariant, immune to the pod's
   external-wrench frame drag): bang-bang torque servo tracking
   `w_des = clamp(1.8·err, ±0.6)` with a 2.5° deadband; dry bearing friction is
   the brake. Stall watchdog escalates torque 0.8→3.5 N·m (never fired past
   0.8 on any tested seed). Exit at |err| < 2.5° and |ω| < 0.08 → `aligned`
   latch (score 0.25).
3. **P2 eject** (horizontal force at the cargo CoM, capped 2.2 N < the m·g=2.45 N
   tipping bound): velocity servo toward the red mouth, speed-capped 0.15 m/s
   (0.07 near the mouth), stiction floor 1.4→2.1 N on measured stalls (fires at
   the deck-edge/gap crossing on every seed); force frame verified by a live
   progress probe (mode toggles if the cube regresses — never needed). The cube
   slides out of the bay, across the 10 mm gap, through the mouth, onto the
   garage floor → success (score 1.0).
4. **P3** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints: 0.0000 → 0.2500 → 1.0000 → 1.0000.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m). The
pedestal centre is at (0.40, 0) ± 25 mm; the platter rim at ~0.56 m; the garage
mouths at ~0.57 m and garage backs at ~0.72 m — all inside the dexterous shell,
and the arm never needs to reach INTO a garage (the cube is pushed in from
outside).

- **Handle peg** (12 mm thick, 80 mm tall, on the platter rim): pinch it
  (12 ≪ 80 mm jaw) or simply push it tangentially with a fingertip and walk the
  platter around; friction stops the platter when the arm stops. Fine aiming =
  small pushes near the stop — the 8° window on a 160 mm radius is ±22 mm of rim
  arc, comfortable for closed-loop nudging.
- **Cargo cube** (90 mm, 0.25 kg): deliberately UNGRASPABLE (90 > 80 mm jaw).
  Ejection is a flat-palm/closed-gripper radial push from behind the backstop
  side: the bay is open only outward, the rails guide the slide, and the garage
  mouth (150 mm wide vs the 90 mm cube) forgives ±30 mm of lateral error. Push
  force ~2 N at deck height — no precision grasp anywhere in the task.
- **Platter/pedestal/garages**: rim pushes on the platter are equivalent to the
  handle; the pedestal and garages are kinematic, so incidental contact cannot
  move the goal frames.

## Execution order (declared)

`rotate-to-align (gently, to a stop) → push the cargo radially out`. REQUIRED and
physically forced: ejecting before alignment throws the cube onto the open ground
facing no mouth (and ground re-entry is geometrically impossible — smoke 6), while
rotating after ejection moves only the empty bay. The rubric mirrors the physics:
`aligned` credit demands the cargo still aboard and the platter near-still, and
`delivered` is reachable only through the mouth plane.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (start err +84.0°): SUCCESS, scores 0.0/0.25/1.0/1.0.
- `solve --seed 1` (start err −49.2°): SUCCESS, same trace.
- `solve --seed 2` (start err −143.8°): SUCCESS — three seeds; identical stiction
  escalations at the deck-edge crossing; the frame-drag mode toggle and the
  torque escalation never fired.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 11/11**, frames.npz (120 × 600 × 960) saved:
  1. settle/no-NaN: platter z 0.0620, centred 0.4 mm, cargo in bay, score ≤ 0.01
  2. randomization readback differs (base xy Δ24.4 mm, yaw Δ9.7°, start err
     Δ140.1°)
  3. dock shuffle: the RED garage occupies all 3 slots over 15 resets (world-pose
     readback)
  4. null policy: 300 idle steps, platter drift 0.00°, cargo aboard, score ≤ 0.05
  5. BAY RETENTION: misaligned 3x-weight shove toward the red garage — the bay
     drags the whole platter 12.2° rather than surrendering the cube (33 mm
     travel, still aboard, no garage entered)
  6. SEED STRATEGY: cube on open ground, quasi-static drag (speed-capped
     0.35 m/s) straight at the red mouth — travels 1511 mm freely but ends at
     ground level (z_loc 45 mm vs floor rest 111 mm), deflected along the blank
     faces below floor height; never inside, score ~0
  7. FLYBY: platter spun through perfect alignment (min|err| 0.2°) at 1.90 rad/s
     — `aligned` does NOT latch (spin gate)
  8. wrong garage: cube settled inside the GREEN garage → no success, score ≤ 0.01
  9. near-miss half-in: cube 15 mm short of the full-inside plane → refused
  10. drop-in: cube dropped from above the red garage lands ON the roof (z_loc
     235 mm) → refused
  11. video frames.npz saved

## Files

- `scene.py` — CarouselDispatchScene + pedestal/platter/garage compound spawners +
  rubric; registers `simgen.cargo_carousel`.
- `solve.py` — torque-servo aim + force-servo ejection certificate (`--seed N`).
- `smoke.py` — 11-check rejection battery + video.
