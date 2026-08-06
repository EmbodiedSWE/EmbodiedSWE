---
name: cosigen-rl-subpolicy
description: Designs, trains, evaluates, and deploys a small local PPO policy for a dexterous CoSiGen subtask. Use when scripted control reaches the right local state but repeatedly fails because of contact, alignment, insertion, grasping, or screwing dynamics.
---

# CoSiGen RL Sub-policy

Use RL only for a short local skill, not the whole long-horizon task.

A useful way to decide where the boundary lies: your code is usually the better tool
wherever the same commands reproduce the same outcome (free-space motion, alignment,
waypoints, loops, retries, phase sequencing) -- structure and long horizons are cheap in
code. A learned policy earns its cost where outcomes vary run to run under contact
(slip, friction, threading, insertion), because it reacts to that variation at every
control step, which open-loop commands cannot.

This favors composing the two: train a short primitive (roughly a second or two of
control -- one press-twist stroke, one re-grip, one insertion) and call it from your own
loop with `rl_run`, once per cycle, with your code doing the sequencing, checks and
retries around it. A single policy asked to learn a long multi-phase maneuver (e.g. a
full multi-cycle ratchet) usually needs far more horizon and exploration than the same
skill decomposed as loop-plus-primitive.

0. BEFORE training a grasp/hold policy, run a RETENTION PROBE: use `rl_train`'s
   `randomize={'obj': {'z': 0.03}}` start-jitter with a trivial reward and a `probe_fn`
   logging object height, 1-2 iterations only. If the object cannot stay elevated even when
   PLACED in the closed gripper (probe height fraction ~0), the gripper does not
   geometrically retain it -- NO amount of RL on wrist/hand DOFs will fix that. Choose a
   different strategy instead: use the ENVIRONMENT as a fixture (pin against a wall/corner,
   wedge up a step), push/drag instead of carry, use both arms as a cradle, or exploit the
   task mechanic directly (e.g. press+twist seating). RL then owns the contact-rich part of
   THAT strategy.

1. Script the robot to a reproducible start state immediately before the difficult contact.
2. Save that state with a meaningful checkpoint.
3. Define:
   - `obs_fn(view)`: compact privileged state needed for the skill; include relative poses and velocities.
   - `reward_fn(view)`: dense progress reward plus a large true-success reward.
   - `done_fn(view)`: the real local success condition.
   - `action_spec`: only the controller components the skill needs.

   Two useful reward-design heuristics:
   - Consider deriving the dense reward from the SUCCESS PREDICATE's own margins rather
     than intuition alone. The success check is code you can read (`import inspect;
     inspect.getsource(type(env.scene))`): typically a conjunction of thresholded
     continuous quantities (a depth, a distance, an angle). Rewarding progress on the
     margin that is still violated often works well, because intuition-based shaping
     (e.g. "be low and vertical") can sometimes be maximized without any task progress.
   - A MECHANISM PROBE before shaping is often worth the steps: script one crude
     open-loop attempt of the motion you believe moves that margin and check whether it
     moves (log it via probe_fn); measuring the coupling (e.g. mm of descent per
     revolution when threading) helps calibrate reward scale and horizon. A margin that
     does not move even under scripted forcing usually points at the strategy, not at
     insufficient training time.

   MDP-DESIGN RULES (violating these makes the curve flat and the policy junk):
   - `done_fn` MUST ALSO terminate irrecoverably-ruined episodes (object knocked off the
     bench / out of the workspace), and `reward_fn` MUST penalize reaching them. Exploration
     noise ruins episodes constantly; without termination+penalty most of every training
     batch is garbage experience where nothing you do changes the reward, so nothing is
     learned. (Measured: an MDP without a drop penalty spent 78% of all samples with the
     object on the floor.)
   - Do not rely on a big bonus for a state exploration will rarely reach (e.g. "+100 when
     lifted 13cm"): a jackpot that is never observed teaches nothing. Reward the NEXT
     reachable milestone densely (contact, caging, closing on the object, each mm of lift).
   - Start the episode AT the moment of difficulty (already at contact height), not
     hovering above it; every step spent approaching is a step exploration can use to ruin
     the scene.
   - Keep `wrist_pos` residuals small near objects (~0.006); prefer a larger `hand` scale
     (>=0.2) so closing is achievable within a few steps.
   - Instrument before you trust: reward_fn is a closure, so append diagnostics to a list
     you defined (e.g. object height, hand-object distance, hand closure) and print the
     distribution after training. If a milestone fraction is ~0, that reward term is dead.
4. TRAIN IN SHORT MONITORED BURSTS, never one long blind run. These are small MLPs: if an
   MDP is learnable, the curve moves within ~2 minutes. Protocol:
   - first burst: `budget_s=150, plateau_patience=8`, default `hidden_sizes=(256,256)`;
   - pass `probe_fn` logging each reward term's ingredient; check `result['probe']`:
     a term whose `max` never exceeds 0 was NEVER sampled -- that reward channel is dead
     and more training time will not help; redesign (closer start state, denser milestone);
   - if `result['improved']` is true and the behavior is on-track: CONTINUE the same policy
     with `init_from=res['pid']` for another burst;
   - if flat: change the MDP (start state / reward / obs), not the budget.
5. Observations: prefer `v.rich_obs(objects=[...])` (~40 dims: relative positions,
   orientations, velocities, hand joints). Thin hand-rolled obs that omit state the skill
   needs is a top silent failure cause. For two-hand coordination use
   `action_spec={'arm':'both', ...}`.
6. Inspect the reward curve and training success. NOTE `success` in the curve = fraction of
   envs whose `done_fn` fired -- if `done_fn` also terminates ruined episodes (it should),
   this is NOT a success rate; log real success inside your diagnostics instead.
7. Restore the exact deployment start state and call `rl_run`. API notes that cost real
   turns before: `stop_when` takes ONE argument (the env-0 view), and `rl_train`'s returned
   `curve` is a list of dicts (`{"iter","ep_reward","success","t"}`) -- don't `np.round` it.
   A policy re-created by `rl_load` has no stored `obs_fn`; pass `obs_fn=` to `rl_run`.
8. Verify the real scene predicate and rollout images. Training reward alone is not success.
9. If deployment fails, diagnose whether the problem is:
   - reward hacking,
   - train/deploy state mismatch,
   - insufficient observation,
   - action scale,
   - or physical infeasibility.
   Revise the MDP and retrain.

The policy output is a residual on robot controller targets; it does not write simulator state directly.
