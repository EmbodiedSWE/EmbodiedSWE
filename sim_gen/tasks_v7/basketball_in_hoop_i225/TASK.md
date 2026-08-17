# basketball_in_hoop_i225 — "tilt_dispenser"

Dispense a sealed, ungraspable ball out of a pivoting cage by pressing and
holding the correct paddle of a see-saw mechanism, then let go and let physics
finish the job.

## Seed provenance

Seed task: `rlbench/basketball_in_hoop` — pick up a basketball and drop/dunk it
through a hoop. One free graspable ball, one static goal, strategy =
grasp -> carry -> release over the hoop.

## Strategic differences

**vs the seed (`rlbench/basketball_in_hoop`):**

1. **The ball can never be grasped or even touched.** It is Ø88 mm (larger than
   the Franka's 80 mm jaw span — asserted in the scene cfg) *and* it is sealed
   inside a roofed, walled cage whose port openings admit the ball but whose
   wall slots (0.061 m) and roof recess are too small for any useful hand
   entry. The seed's entire strategy (grasp, carry, drop through the hoop) is
   physically impossible here; `smoke.py` proves the two blocking facts
   numerically (jaw < ball, slot < ball).
2. **The manipulation target is a mechanism, not the ball.** The robot operates
   a keel-stabilised see-saw cage through its yellow paddles: press and HOLD
   one paddle down (~11 N sustained at the 0.195 m arm), and gravity — via the
   cage's tilted V-dish floor — conveys the ball out through the port and into
   the catch bin below. The ball is only ever touched by the cage, gravity and
   the bin.
3. **A binary perception/decision step the seed never has.** Each episode a red
   kinematic shutter seals ONE of the two mirrored ports (uniform coin flip per
   reset). Pressing the wrong paddle tilts the cage toward the shutter and the
   ball just parks against the panel (recoverably — the smoke test proves
   both the retention and the recovery). The agent must read the scene and
   choose the correct side before acting.
4. **Success requires release.** The cage must return level on its own (keel
   pendulum, Mgd·sin(dish) ≈ 0.70 N·m restoring at 8°, > 1.3x any ball-induced
   torque — asserted) and the ball must come to rest in the bin hands-off.
   A robot still holding the paddle at the moment of judging cannot be scored:
   `success()` demands the settled, level, at-rest final state.

**vs sibling `basketball_in_hoop_i128` ("crater_run", same seed):**

- i128 has the robot *push the ball itself* along a course (sustained contact
  conveyance of the ball). Here the robot **never contacts the ball at all** —
  the only interaction is a press-and-hold on a hinged mechanism, plus the
  binary open-side decision. Different manipulated object (mechanism vs ball),
  different verb (press-hold-release vs push), different perception content
  (which port is open vs where the crater is).

## The scene

- **Stand** (dynamic root, 28 kg, on the ground): base plate, two pillars, two
  hinge stubs, and two mirrored catch bins (floor, near wall, tall backboard,
  side walls) hanging off both ±x ends at bin-floor-top z = 0.05.
- **Cage** (dynamic, ~4.5 kg, hinged to the stand at z = 0.34 by a
  spawn-authored USD RevoluteJoint, axis local y, travel ±15°): an 8° V-dish
  floor (self-centres the ball at the valley), side walls with a 0.061 m slot,
  a roof, four posts framing two open ±x ports, two yellow 0.09 m paddles at
  x = ±0.195, and a dense keel (ρ 5200) that makes the cage a stable pendulum
  (CoM ≈ 0.113 m below the hinge). Angular damping 4.0 gives an overdamped-ish
  return (overshoot ≈ 5.6° < the 8° dish angle).
- **Shutter** (kinematic): red panel + pillars covering the canonical −x port;
  its yaw flips per episode so the OPEN port is on the `side` (±1) drawn at
  reset.
- **Ball**: orange Ø88 mm, 0.25 kg, spawns in the cage with x/y jitter and
  rolls to the dish valley.

Randomization (all verified by readback in smoke): open side (coin flip),
stand yaw ±10°, stand xy ±0.03 m, ball start x ±0.05 m / y ±0.010 m.

## Rubric

Canonical frame = stand-local with x and y folded by `side`, so the open port
is always canonical +x.

- `tilt_latch` (0.2): cage has been tilted ≥ 10° toward the open side.
- `exit_latch` (0.2): ball has been inside the open-port exit window
  (canonical x in [0.16, 0.60], |y| ≤ 0.13, z ≤ 0.27).
- `bin_latch` (0.2): ball has been inside the open-side bin window
  (canonical x in [0.17, 0.41], |y| ≤ 0.08, z in [0.07, 0.13]).
- Progress score = 0.2·(tilt + exit + bin), capped at 0.60.
- `success()` (score 1.0): ball settled (|v| < 0.18, |ω| < 4.5 — gates sit just
  above the GPU phantom-velocity readback artifact; the tight position window
  plus 3.5 s persistence carry the at-rest semantics) inside the open-side bin
  window. Latches are monotone; score never decreases.

## Teleport solution (solve.py) — the legitimacy certificate

No teleports at all. One honest mechanism interaction:

- **P0 READ**: settle 90 steps; read the open side from the scene state
  (baseline score asserted ≈ 0).
- **P1 PRESS**: constant hinge torque τ = side·2.2 N·m on the cage body
  (body-frame y — the hinge axis is body-fixed), the exact equivalent of
  ~11 N pressed down on the paddle at its 0.195 m arm (11 N < the asserted
  25 N sustained-press budget). Cage swings to the 15° stop. Assert
  score ≥ 0.20.
- **P2 HOLD**: keep the torque on while gravity rolls the ball down the
  now-downhill dish, out the open port, off the dribble step and into the bin.
  Assert score ≥ 0.60.
- **P3 RELEASE**: clear the wrench BEFORE success is ever true; the keel swings
  the cage back level and the ball settles in the bin hands-off.
- **P4 PERSIST**: 3.5 simulated seconds hands-off; success must hold every
  step. Only then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at each phase boundary and asserted non-decreasing.

## Franka embodiment argument

Single Franka base pose: on the ground at canonical (0, −0.75, 0), facing the
stand (+y toward it).

- **Paddles** (the only thing the robot must touch): centres at world height
  ≈ 0.34 + 0.071 = 0.41 m, radius ≈ 0.20 m from the stand axis; both paddles
  lie within ~0.55–0.75 m of the base — comfortably inside the Franka's
  0.855 m reach envelope from the one pose. The press is a downward fingertip/
  knuckle push on a 0.09 m × 0.09 m plate: ~11 N sustained, far below the
  arm's ~25 N (asserted budget) and below the wrist's continuous rating. Hold
  at the 15° stop requires the same wrench; OSC with a z-force target does it.
- **The ball**: needs no grasp (impossible anyway: Ø88 > 80 mm jaw). Never
  touched.
- **The shutter**: never touched; it is read visually (a 0.21 m-wide red panel).
- No bimanual need, no regrasp, no tool: read side, press one paddle, hold,
  release. All within one base pose and OSC end-effector control.

## Execution order

`scene.py` (registers `simgen.tilt_dispenser`) → `solve.py` (forge, seeds 0/1)
→ `smoke.py` (forge). Both entry points are standalone AppLauncher scripts:

```
python -u -m simgen_tasks.basketball_in_hoop_i225.solve --headless [--seed N]
python -u -m simgen_tasks.basketball_in_hoop_i225.smoke --headless
```

## Smoke battery (18 checks, rejection-first)

1. Initial settle: finite state, sane layout (level cage, shutter on −x,
   ball at the dish valley).
2. Reset score ≈ 0 (null credit).
3. Randomization by READBACK over 8 seeds: both sides drawn, stand yaw/xy
   spread, shutter tracks the side, layout sane every seed.
4. Ball start-x jitter readback (instant) + self-centring to the valley.
5. Null policy 240 steps: score stays ≈ 0.
6. Seed strategy impossible: ball > jaw span AND wall slot ≪ ball (numeric).
7. Wrong-paddle press: cage reaches the −15° stop, shutter retains the ball
   (never past the panel face), zero credit.
8. Recovery: release after the wrong press — cage returns level, ball
   re-centres, still zero credit.
9. Ball nudge toward the open port with the cage LEVEL: dish retains it
   (rolls back to valley), zero credit — tilting is load-bearing.
10. Ball placed settled in the WRONG-side bin: rejected.
11. Ball settled on the ground beside the stand: rejected.
12. Ball perched on a bin side wall: rejected.
13. Settle gate: ball moving/spinning inside the correct bin is NOT success.
14. Honest latch construction (ordered so no prefix is ever success): ball
    parked outside first, real press to the stop (tilt latch), release,
    exit-window fly-through, bin-window pass — latches (1,1,1), score capped
    at 0.60, still not success.
15. Idle steps: capped score unchanged (no drift).
16. Audit: success was never true at any point during the battery.
17. Final no-NaN.
18. ≥ 20 camera frames recorded → `frames.npz`.

Verdict line: `SIM_GEN_SMOKE: ALL PASS 18/18`.
