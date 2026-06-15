# Fable_5/G1_Upper_body_IK — Scripted pick-and-place for Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0

A Unitree G1 humanoid (fixed base, three-fingered trihands, Pink-IK upper-body
action space) picks up the steering wheel from the packing table with its LEFT
hand and places it in the basket on its right. Result: **SUCCESS** — the env's own
`task_done_pick_place` criterion holds continuously for the last 3.3 s of the
episode with the wheel resting flat inside the basket and both arms retracted.

## Files

- `pick_place_wheel.py` — main script: waypoint state machine + success checking
  + video recording.
- `g1_pick_place_success.mp4` — final video of the successful trajectory
  (1280x720 @ 50 fps, one frame per policy step, ~20 s).
- `smoke_test.py` — env bring-up + introspection (action layout validation via a
  hold-pose stepping check, hand joint limits, finger geometry, etc.).
- `run*.log` — logs of the debugging iterations.

## Run

```bash
# from the repo root
env_isaaclab/bin/python -u Fable_5/G1_Upper_body_IK/pick_place_wheel.py
# video lands in Fable_5/G1_Upper_body_IK/videos/; --no_video to skip recording
```

The env has no randomization events (reset-to-default only), so the run is
deterministic and reproduces exactly.

## Action space (28-dim, absolute)

`[0:3]` left wrist pos (env-local), `[3:7]` left wrist quat (wxyz), `[7:10]`/`[10:14]`
right wrist pose, `[14:28]` 14 raw hand-joint targets (see HAND_ORDER in the script).
Wrist targets are tracked by a differential Pink-IK QP at 200 Hz (gain 0.5); waist
yaw/pitch/roll are IK-resolved automatically and extend reach via torso lean.

## Controller sequence

RAISE -> STAGE -> ADVANCE (radial approach from outside the wheel) -> SCOOP
(half-curl while lifting) -> CLAMP (fist) -> LIFT -> CARRY -> CARRY2 (yaw the
wrist mid-carry) -> LOWER -> RELEASE -> NUDGE -> RETREAT_UP -> RETREAT -> SETTLE.

Success is evaluated manually with the env's exact `task_done_pick_place` function
(object in box x(0.40,0.85), y(0.35,0.60), z<1.10, |v|<0.2 m/s per axis, right
wrist x<0.26), latched after 25 consecutive true steps post-release. The right arm
is parked at its initial pose (x=0.149 < 0.26) the whole time.

## Hard-won lessons (each cost a debugging run)

1. **Setup**: `import pinocchio` must precede `AppLauncher` (else Isaac Sim's bundled
   pinocchio wins -> boost-python `std::vector<string>` TypeError in pink); set
   `enable_pinocchio=True` on launcher args (patches `pxr.Gf.Matrix4d`); import
   `isaaclab_tasks.manager_based.locomanipulation.pick_place` explicitly (the
   subpackage is blacklisted from the bulk import scan).
2. **Don't rotate the hand while translating past the object** — the 16 cm fingers
   raked the wheel 22 cm across the table. Approach radially from outside,
   retarget from the object's live position at each phase entry.
3. **Trihand thumb is short** (~6.5 cm vs 16 cm fingers): top-down pinches at finger
   depth can't be opposed by the thumb. Use a horizontal hook grasp instead
   (fingers over the rim tube, thumb beneath).
4. **The table blocks finger curls** on objects lying flat: fingertips must sweep
   below the rim tube to close. Fix: yaw the thumb aside on approach and SCOOP —
   lift the wrist while half-curling so the curl completes mid-air, then clamp.
5. **A rim-hooked wheel pivots and dangles vertically**; with a loose hook it slides
   to the fingertips and gets flung by the stored finger-PD energy. The closed
   fist (curl to -1.25/-1.5) survives the dangle.
6. **The env's success criterion has no "not grasped" check** — it fires the moment
   the carried object enters the basket box at low velocity, and the success
   TERMINATION then resets the episode mid-place. For a full place-and-retreat
   video, disable `terminations.success` and evaluate the same function manually.
7. **Cross-body reach limit**: the left wrist saturates near x ~= 0.30 — but the
   wheel hangs 0.24 m along the finger direction, so yawing the wrist mid-carry
   (RADIAL (0.6,0.8) -> (0.849,0.529)) shifts the payload ~7 cm deeper into the
   basket at no reach cost.
8. **Released-on-the-rim objects slide back** when the fingers retract: the wheel
   twice settled 2-4 mm outside the success box leaning on the basket wall. The
   NUDGE phase (forward sweep with the open hand) tips it fully inside; retreat
   upward before retreating back.
