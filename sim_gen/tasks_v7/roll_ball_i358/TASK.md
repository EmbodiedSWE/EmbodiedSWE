# roll_ball_i358 — Windmill Toll-Gate

**Env name:** `simgen.windmill_tollgate` (robot `"null"`)
**Seed task:** `maniskill/roll_ball`
**Files:** `scene.py`, `solve.py`, `smoke.py`, `TASK.md`

## 1. Seed provenance and what changed

The ManiSkill seed `roll_ball` is: one dynamic ball on a flat open table, push/roll it
laterally across the tabletop into a flat circular goal region. One object, one contact,
open workspace, translation-only strategy, success = ball xy inside a painted disc.

`windmill_tollgate` keeps exactly one seed atom — *the target ball reaches the goal
region by rolling, launched with a push* — and demotes it to the harmless last mile.
The load-bearing work is defeating an **active mechanism** the seed has no concept of:

| Aspect            | Seed `roll_ball`               | `windmill_tollgate` (this task)                          |
|-------------------|--------------------------------|----------------------------------------------------------|
| Workspace         | flat open table                | walled, **ROOFED** courtyard; only entrance is a doorway; tilted interior floor |
| Obstacle          | none                           | a **motor-driven two-blade windmill** on a free vertical axle, torque-capped velocity servo that NEVER stops on its own; its sweep covers the doorway vestibule and the delivery corridor |
| Objects           | 1 ball                         | ball (60 mm) + steel **brake beam** (26 × 26 × 200 mm) + jointed rotor |
| Required strategy | push ball to goal              | **silence the mill first**: thread the beam from OUTSIDE through a 30 mm wall bore at blade height until its tip crosses the sweep circle; the blade slams into it and the motor stalls against it. Only then bowl the ball up the apron, through the doorway; gravity on the tilted floor finishes the delivery hands-off |
| Goal region       | flat painted disc              | green pad in the downhill corner, **outside** the blade sweep circle |
| Success           | ball in disc                   | ball at rest on the pad **AND** the mill held stalled *now* (pose-window verdict, motor still driving) — a live conjunction |

The direct seed strategy is provably worthless here: a ball delivered to the pad earns
nothing while the mill runs (smoke check 2), and it *cannot* be what stops the mill —
the pad lies outside the sweep circle by construction (asserted in `__post_init__`),
so the parked ball is untouchable by the blade. Conversely the beam cannot smuggle the
ball in (60 mm ball vs 30 mm bore, smoke check 4) and a beam lying on the interior
floor passes *under* the blade band (asserted), so there is no trivial floor-drop
stall. The two sub-goals need two different objects, two different entry ports, and
two different skills (peg-in-bore insertion at 105 mm height vs a ramp bowl).

## 2. Why it is different from the rest of the corpus

Nearest corpus neighbours checked this session and the discriminating feature:

- **arch_quarry_i81** (same seed family): obstacle is a *passive, emergent* friction
  arch of free bodies, destroyed by pushing. Here the obstacle is an *actively driven
  jointed mechanism* that must be **held** defeated against a live motor — release the
  brake and it spins back up (smoke check 10 demonstrates exactly this reversibility).
- **carousel family / key-turn cam / ram-dispenser**: driven rotors that must be *used*
  (ride, pump, rotate to a target angle). Here the rotor is an adversary to be
  *arrested*; no corpus task scores "mechanism stalled while its motor still drives".
- **close_door_i162 / shift-park grill / flip-board**: hinged fixtures moved *to a pose*.
  This rotor has no target pose at all — any angle is fine so long as a 1 s pose window
  shows < 2° motion under drive.
- **tilt_labyrinth_i42 / crater_run_i128**: gravity-routed ball delivery through static
  geometry. The delivery leg here is similar in spirit but is gated by the mechanism
  arrest, and the corridor is defined by the *stalled blade's own body* — the obstacle
  becomes part of the track.
- **skyway_bridge_i77 / plank-bridge / gauge-adapter**: insert/place a tool to *create
  passage*. The beam here creates no path — it is a **brake**; the passage (doorway)
  was always open, just swept.
- **drain_plug_i196 / falsework_tent_i71**: removal/support-withdrawal tasks; nothing
  is removed here and nothing collapses.
- **pen_holder exemplar**: insertion for *containment* of the inserted object; here
  insertion depth is instrumental (tip past the sweep circle) and the inserted object
  is never the scored one.

No corpus task has (a) a tireless motor-driven sweeper as the obstacle, (b) success
conditioned on a *live stall* of that motor (not a latch, not a pose), or (c) the
seed's own goal object rendered incapable of the blocking sub-goal by aperture *and*
sweep-circle geometry simultaneously.

## 3. Scene summary

Kinematic **courtyard** fixed at world (0.38, 0.06): interior 340 × 340 mm, walls
160 mm tall, full roof (top-down delivery denied), one 110 mm doorway in the front
wall, interior floor slab tilted 2.3° downhill toward the back-right corner where a
90 × 90 mm green pad (visual-only, no collider) marks the goal. An exterior apron ramp
with rails climbs from the ground through the doorway (seam proud by ≥ 0.5 mm — no
step-up). Through the right wall at z 90–120 mm runs a square 30 mm guide **bore**
aimed at the axle, extended by an interior sleeve and a 34 mm exterior mouth sleeve
(8 mm hand slack). The **mill** — orange bar 224 × 24 × 72 mm, density 400 — sits on a
spawn-authored free `RevoluteJoint` (vertical axis, court-local (−0.04, −0.04)); a
`post_step` torque-capped velocity servo (2.5 rad/s target, 0.05 N·m cap, body-z
torque) drives it every step, direction randomized. The **beam** (26 mm square steel,
~1.05 kg) and **ball** (60 mm, 0.25 kg) rest on the open ground outside. The court is
never teleported (kinematic-body0 joint anchors are world-fixed); the rotor origin
lies on the axle axis so its random yaw is a joint-consistent pose write.

~27 numeric asserts in `__post_init__` certify the gate geometry: doorway passes the
ball, bore does not; blade clears every wall, the roof, a floor-lying beam and the
sleeves, yet strikes both the rolling ball and the inserted beam; corridor around a
stalled blade (flank ≥ 118 mm, tip gap 98 mm) passes the ball; pad band outside the
sweep circle; stall window cannot alias with free spin.

Randomised per seed (verified by readback, smoke check 6): mill angle (±180°), motor
direction (coin, drawn last), ball spawn (x ∈ [−0.12, 0.01], y ∈ [−0.46, −0.38]),
beam spawn (x ∈ [0.315, 0.38], y ∈ [0.04, 0.15], yaw ±0.25).

**Rubric (latched, cleared on reset, null ≈ 0):** 0.25 *engaged* (beam tip ever in the
bore lane past engagement depth) + 0.25 *silenced* (mill ever pose-window-stalled while
engaged) + 0.25 *entered* (ball inside while stalled), capped at 0.75; exactly 1.0 iff
`success()` holds **live**: mill stalled now ∧ ball at rest on the pad ∧ finite.

## 4. Teleport-solution phase outline (solve.py)

Teleports are used **only** for transport of grasped/free bodies through free space
outside the courtyard; every load-bearing interaction (insertion, blade slam, stall,
ramp climb, doorway transit, corridor ride, pad settle) is contact physics.

- **P0 settle** (1 s): layout readback (ball/beam/mill/spin sign — per-seed proof on
  stdout); assert the mill actually turns (> 0.5 rad per 0.5 s) with the drawn sign,
  score ≤ 0.02, not success. `SIM_GEN_SCORE 0.0000`.
- **P1 thread the brake** : one pose write carries the beam to a hover just outside the
  exterior mouth, axis on the lane (emulates carrying the grasped 26 mm bar). Then a
  **held wrench** — gravity compensation + y/z PD onto the lane + slow axial velocity
  servo (−0.06 m/s, 5→9 N escalation; a passing blade may block the tip for a beat) +
  small-angle orientation PD — threads it through mouth → bore → sleeve until the tip
  crosses the engagement line. Force-frame mode probed from tip-x progress (40-step
  windows). Release; the next blade pass slams the flank and the motor stalls; wait for
  the scene's pose-window verdict + silenced latch. `SIM_GEN_SCORE 0.5000`.
- **P2 bowl the ball**: pose write to the apron approach on open ground; regulated +y
  CoM push (0.55 m/s servo, x-centering PD) **aimed through the axle line** (x =
  axle_x, so the doorway impact has ~zero lever arm on the stalled rotor) and **cut at
  y = −0.22**, outside the wall plane. From there everything is hands-off: climb,
  doorway, deflection off the stalled blade's flank, creep along the flank around the
  tip, downhill into the corner pad. Under/over-shoots re-stage (teleport back to the
  same outside start) and re-bowl at 0.65/0.50/0.78/0.90 m/s — the delivery that
  counts is the final, entirely hands-off run. In practice attempt 0 delivered on all
  three seeds. `SIM_GEN_SCORE 1.0000`.
- **P3 persistence**: 3.33 s (10 × 40 steps) fully hands-off with the motor still
  driving against the beam; success re-checked each block, then
  `SIM_GEN_SOLVE: SUCCESS`. Watchdog `threading.Timer(1350) → os._exit`; any exception
  prints traceback + `SIM_GEN_SOLVE: FAIL` + hard exit.

**Execution-order declaration:** success is an order-free conjunction, but brake-first
is the only *practical* order and the solve declares and uses it: while the mill runs,
the blade sweeps the doorway vestibule and the delivery corridor, batting any entrant
indefinitely (and the `entered` credit is explicitly gated on the stall). Ball-first
cannot terminate: the pad is outside the sweep circle, so no delivered ball ever stalls
the mill (smoke check 2). The rubric latches mirror this: engaged → silenced → entered.

## 5. Embodiment argument (Franka, single arm, parallel jaw ≤ 80 mm, OSC)

Plausible base pose: base at the world origin facing +x; the courtyard centre is
0.38 m away, the bore mouth at ~(0.63, 0.02, 0.105), the apron start at ~(0.34,
−0.35) — everything inside a 0.75 m reach envelope, and both manipulated objects stay
**outside** the roofed courtyard at all times (the arm never needs to reach inside).

- **Beam threading (P1)**: the beam is a 26 mm square bar — side grasp anywhere along
  its 200 mm length with the 80 mm jaw (54 mm margin). Carry to the exterior mouth
  (34 mm opening = 8 mm insertion slack, a standard peg-in-hole tolerance), align
  axis with the bore (the exterior sleeve funnels the final 4 mm), and push axially
  at ≤ 0.06 m/s with ≤ 9 N — well inside Franka payload (~3 kg) and force limits.
  The mouth is at 105 mm height on the outside of the right wall, approached
  horizontally with the wrist clear of the roof edge (roof overhangs 10 mm; the
  mouth sleeve extends 50 mm proud of the wall). Regrip is allowed but not needed:
  200 mm bar − 170 mm final insertion depth beyond the mouth leaves the last 30 mm +
  the initial 15 mm standoff, and pushing the tail flush with the mouth face reaches
  engagement depth (tail ends 8 mm proud of the mouth in the final pose).
- **Ball bowl (P2)**: 60 mm ball in the 80 mm jaw (20 mm margin) — but no grasp is
  even needed: the solve emulates a non-prehensile *bowl*: fingertip push behind the
  ball, accelerating it to ~0.55 m/s over ~15 cm of open ground and apron, releasing
  35 mm before the doorway plane. OSC straight-line push at ground height with a
  1.5 N lateral centering budget. After release the arm retracts; gravity delivers.
- **Per-object contact summary:** beam → side *pinch grasp* + guided axial insertion
  (tool-in-bore); ball → non-prehensile *push/bowl* (the seed's own skill); nothing
  inside the courtyard is ever touched by the arm.

## 6. Smoke battery (smoke.py) — 13 checks

1. Settle & no-NaN: both free bodies at rest outside, mill demonstrably turning,
   score ~0.
2. **Seed-strategy rejection**: ball constructed ON the pad while the mill runs — it
   stays (pad outside the sweep: the blade cannot reach it, so it also can never be
   the staller), score stays ~0, no clause fires.
3. **Shallow beam**: beam resting in the bore with tip short of the sweep circle —
   mill turns right past it (+427°/3 s), engagement never latches, score ~0.
4. **Bore vs ball**: 60 mm ball pushed at the bore side travels 19 cm freely (probe
   not vacuous) then presses the wall and never gets inside.
5. Null policy 2.5 s → mill still turning, score ~0, never success.
6. Randomization readback, 8 seeds: ball/beam xy spans > 2 cm, mill angle span
   > 40°, both motor directions observed.
7. Determinism: same seed twice → readback Δ < 1e-4.
8. **Conjunction**: real stall constructed (beam at full depth → blade slams → motor
   stalls) with the ball still outside → exactly 0.50, not success.
9. **Wrong level**: ball settled on the ROOF directly above the pad (pad-band x,y) →
   z clause rejects.
10. **Reversible stall + latched credit**: beam withdrawn → motor spins the mill back
    up (stall is a live state, not a latch) while latched credit stays exactly 0.50.
11. **Live-stall gate**: ball on the pad with engaged+silenced latched but the mill
    running again → success refuses (it demands the stall *now*).
12. Rejection audit: success() never True at any judged point.
13. Final finite-state check; frames.npz written (persp camera, 960 × 600).

Exact final line: `SIM_GEN_SMOKE: ALL PASS 13/13`, then hard exit.

## 7. Validation evidence (forge, RTX-4090 pod)

- `solve --seed 0`: SUCCESS, 21.5 s, scores 0.00 → 0.25 → 0.50 → 1.00 → 1.00
  (spin −, stall at yaw +193°; bowl attempt 0 delivered, ball settles (0.140, 0.140)).
- `solve --seed 1`: SUCCESS, 22.3 s (spin +, stall from the other side at +166°;
  bowl attempt 0 delivered).
- `solve --seed 2`: SUCCESS, 21.8 s (distinct layout, bowl attempt 0 delivered).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 13/13`, 61.6 s, frames.npz (410 × 600 × 960 × 3)
  saved.
- One scene iteration total: the initial slab-tilt quaternion was inverted (ball
  parked in the wrong corner); flipped the rotation axis sign — all other geometry
  (insertion, jam, stall, corridor) worked on the first forge run.
