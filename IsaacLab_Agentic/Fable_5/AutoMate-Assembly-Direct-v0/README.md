# Fable_5/AutoMate-Assembly-Direct-v0 — Scripted plug insertion for Isaac-AutoMate-Assembly-Direct-v0

Inserts the held plug (AutoMate asset 00015: 8 mm peg, ~0.11 mm hole clearance) into
its socket with the Franka robot, using a scripted vectorized controller instead of a
trained policy. Result: **4/4 parallel envs succeed** under the eval-style hard start
(`if_sbc=False`: plug starts fully outside the socket with ±1 cm XY noise), verified
by the env's own success criterion (`automate_algo.check_plug_inserted_in_socket`,
also latched in `env.ep_succeeded`).

## Files

- `insert_plug.py` — main script: controller + success checking + video recording.
- `plug_insertion_success.mp4` — final video of the successful trajectory
  (env 0 close-up, 1280x720 @ 15 fps, one frame per policy step).
- `smoke_test.py` — minimal env bring-up check.
- `probe_reset.py` — debugging probe that instruments the env reset pipeline
  (IK convergence, gripper close, plug retention).
- `run*.log`, `probe*.log` — logs from the debugging iterations.

## Run

```bash
# from the repo root (uses the uv env that has isaacsim + isaaclab installed)
env_isaaclab/bin/python -u Fable_5/AutoMate-Assembly-Direct-v0/insert_plug.py --num_envs 4
# video lands in Fable_5/AutoMate-Assembly-Direct-v0/videos/; --no_video to skip recording
```

## Controller

State machine on top of the env's task-space impedance controller, driving the
6-DoF delta-pose action space:

1. **ALIGN** — hover the plug 5 mm above the socket mouth, null XY error to <1.5 mm.
2. **INSERT** — rate-limited descent (~1.5 cm/s) down the socket axis;
   spiral search if descent stalls with small XY error (rim catch);
   retry from ALIGN if it stalls with large XY error (grasp slip).
3. **SEAT** — press to 2 mm above the socket origin.
4. **HOLD** — keep position once the env reports success.

Since the plug is rigidly grasped, the fingertip is commanded by the plug's position
error (privileged state), making the loop independent of the grasp offset.

## Gotchas found while debugging

- **Never change `env_cfg.sim.render_interval`** for this env. It sets `rendering_dt`,
  and the reset's `step_sim_no_action()` calls `sim.step(render=True)`, which advances
  `rendering_dt` worth of physics per call — the scripted grasp sequence then runs 8x
  too long and ejects the plug from the gripper (fingers close fully on nothing and
  the plugs scatter meters away).
- **Outer-loop gain must be < 1.** The action space is already a lagged position servo
  (EMA 0.2 + impedance). A P gain of 2.0 on plug position error produced a divergent
  limit cycle (XY error oscillating 5 -> 45 mm, plug slipping inside the fingers);
  0.5 converges smoothly.
- The controller loop must stay under `max_episode_length` (150 steps at
  `episode_length_s=10`) or the env auto-resets mid-recording.
