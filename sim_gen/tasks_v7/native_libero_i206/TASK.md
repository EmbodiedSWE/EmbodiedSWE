# native_libero_i206 — `tilt_maze`

Steer a captive ball through a switchback maze by tilting a hinged tray (press the blue
tab, then the red tab), until the ball drops into a sunken green pocket; release and let
the keel bring the tray back level. Leave the white spare ball in its cradle.

Env: `simgen.tilt_maze` · Scene: `TiltMazeScene` (`scene.py`) · No robot slot (NullRobot).

## Provenance

Derived from seed `libero/native_libero` (vendored LIBERO MJCF scenes: an OSC arm
grasps ONE object and carries it to a region / articulates a fixture; a static BDDL
`on`/`in` predicate judges the end pose).

## Strategic difference

- **vs the seed:** nothing is ever grasped or carried, and no end-pose can be produced
  by transport: the goal object is a 26 mm ball SEALED under a slatted cage roof (all
  gaps ≤ 16 mm — asserted). The only interface the world offers is the tray's ONE
  degree of freedom (a revolute hinge on a yoke stand), driven by pressing paddle tabs.
  The seed's strategy executed here at its physical best — setting the ball on top of
  the roof above the pocket — scores ~0 (smoke check 4: it rests 50 mm above the sink
  plane, then rolls off the tray).
- **The plan is closed-loop and two-stroke, with a physically forced order:** tilt
  blue-end-down so the ball runs the north lane east and the diagonal baffle deflects
  it around the divider into the south lane; then tilt red-end-down so it runs the
  south lane west and sinks into the recessed pocket; then release. Red-first parks the
  ball harmlessly at the start wall; the divider makes the pocket unreachable except
  through the bay, and the roof denies any approach from above.
- **vs previously built tasks:** not a lid/containment closure (close_box_i26), not an
  insertion (pen_holder), not a chain-reaction trigger (native_libero_i157 domino
  relay), not a balance/counterweight problem (base_i88 see-saw). The mechanism here is
  *gravity steering through a 1-DoF orientation channel*: the agent never touches the
  goal object at all, only the fixture that reorients its supporting surface — and the
  fixture must then be RELEASED and return level on its own (authored keel CoM), so
  "hold it tilted forever" cannot succeed.

## Solution outline (solve.py — teleport = transport only)

P0 settle + layout asserts → P1 body-frame hinge-torque servo (gravity feedforward +
PD, sign/stall probes) holds ~+8° until the ball rounds the baffle into the south lane
(score 0.50) → P2 servo ~−6° then −2.5° coast until the ball sinks in the pocket
(score 0.75) → P3 zero wrench, keel returns the tray level, 60-step success streak
(score 1.00) → P4 ≥3.3 s hands-off persistence → `SIM_GEN_SOLVE: SUCCESS`.
Passes seeds 0 and 1 on the forge (rc=0, ~18 s each, monotone score trace).

## Franka embodiment (single arm, parallel-jaw gripper; base at world (0, 0, 0))

- **Paddle tabs (the only things touched):** flat 50×50 mm tabs at deck height
  (~0.135 m) outside each end wall. A closed-fingertip press works; the required
  fingertip force is τ/lever ≈ 0.40 N·m / 0.217 m ≈ 1.9 N — trivial for the arm.
  With assembly jitter ±5 cm and free yaw, the far tab centre is ≤ 0.72 m from the
  base — inside the ~0.85 m reach envelope at that height, approached from above.
- **Yellow ball:** never grasped, by design — the cage roof makes it impossible
  (asserted), so no grasp feasibility argument is needed.
- **White spare ball:** 26 mm — graspable by an 80 mm parallel jaw, but the task is to
  LEAVE it; it exists to punish "grab any ball and put it somewhere".
- **Stand/tray/cradle:** fixtures; never need to be grasped (stand is 8 kg and grippy).

## Execution order

Declared AND physically forced: blue-end press strictly before red-end press (the south
lane is reachable only via the bay + baffle). The rubric encodes it as ordered latches
(bay 0.30 → crossed 0.20 → returned 0.25) capped at 0.75, with 1.0 iff live success.

## Checks

- `solve.py` on forge: **seed 0 PASS, seed 1 PASS** (score 0 → 0.50 → 0.75 → 1.00,
  ≥3.3 s persistence).
- `smoke.py` on forge: **`SIM_GEN_SMOKE: ALL PASS 13/13`** — settle/no-NaN + authored
  tray mass/CoM readback; randomization readback; null policy ~0; seed-strategy
  (roof placement) rejected; tilted-pocket (press actuator real, lip retains at the
  stop, `level` refuses); keel return; wrong (white) ball rejected; distractor clause
  load-bearing; bay latch credit; regression keeps latched credit, no more; settle
  gate; success-never-True audit; frames.npz saved.
