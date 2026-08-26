# vla/eval — closed-loop eval sims for the VLA pipeline

Load the same world a dataset was baked from, drive it with policy-convention
actions, score with the grader. Two workflows on one loader:

- **Open-loop replay** (`replay_actions.py`) — re-drive recorded episodes'
  actions through the executor; certify the sim/executor before any policy is
  judged. Recipe below.
- **Closed-loop eval** (`serve.py` + the lerobot plugin) — a trained policy
  drives the live sim through the standard `lerobot-eval` CLI. Recipe below.

    sim.py            load_sim() + SimSpec + the EvalSim facade
    specs/            named eval setups (register_sim; one module per scene family)
    check_load.py     stage-0 loader check: build, warmup, hold, snapshot obs
    replay_actions.py re-drive recorded episodes' actions (executor certification)
    serve.py          the sim behind a socket — the closed-loop eval's Isaac side
    protocol.py       wire format shared by both venvs (stdlib+numpy only)
    lerobot_env_cosigen/  lerobot plugin (pip install -e into the lerobot venv)
    _out/             default output dir (debug scratch, delete freely)

## load_sim — one call, three sources

    # caller owns AppLauncher (--enable_cameras); module imports app-free
    load_sim(source, *, num_envs=1, device="cuda:0", **overrides) -> EvalSim

1. **registered sim name** — `load_sim("bulb_jointpd_60hz")`: a named setup
   from `specs/` (bake-free by construction: preset + control law spelled out).
2. **robobench preset name** — `load_sim("assembly.bulb.franka.osc")`: the env
   as registered, cameras attached, preset controller; warns "not calibrated
   to any dataset". For probing.
3. **bake path** — `load_sim(".../datasets/<id>/meta/bake.json")`: ALL settings
   from the one stamp. The default when evaluating a trained policy.

All three fill one `SimSpec`:

    preset            world = the ENVS name
    control_space     raw_cmd | joint_pos | joint_vel; None = preset controller
    control_freq_hz   latch rate for the joint conventions
    finger_drives     (stiffness, damping) written onto the finger joints
    tracker_gains     arm PD for joint conventions; None -> stamped arm drives
                      (joint-collected data) -> the .joint preset's PD
    grip_margin       m of finger closure commanded BEYOND the closedness label
                      when gripping — achieved-width labels carry no squeeze
                      force (convert README caveat 1); executor-side restoration
    physical_params   a PHYSICAL_PARAMS draw to re-apply; None/{} = nominal
    cams, size, warmup, stamp (full bake dict when bake-derived)

`**overrides` win; overrides of bake-derived values are echoed and recorded.

## Build path + guarantees

1. Resolve preset; `joint_pos`/`joint_vel` build the **`.joint` sibling**
   (position-PD arm actuators — torque modes zero their stiffness).
2. Inject the declared training cameras before the build (replay.py's
   `resolve_views`/`_camera_cfg`/assets-monkeypatch; nominal poses, 640x480;
   `env_spacing=50` when `num_envs > 1` keeps neighbors out of frame).
3. `ENVS.get(name)().build(...)`; physical params applied at both moments
   (onto the scene cfg pre-build, `apply_physical_params` post-build).
4. Install the control law:
   - joint conventions: composite position `JointController` at the stamped
     rate; finger drives + arm tracker gains written to the articulation.
   - `raw_cmd`: keep the preset controller, re-apply the stamped law — cfg
     scalars, the `_kp`/`_kd` instance tensors, latch `control_period`, joint
     drives. Class mismatch or an unplaceable stamped value is a HARD ERROR,
     never a silent skip (a silent skip is exactly how we shipped a 20x-weak
     wrist once; see Findings).
5. Validate vs the stamp: joint order, state/action layout, rate divisibility.

What a bake-load matches exactly: world, control space + rate, controller law,
joint drives (incl. the solve-written gripper stiffness), state/action layout,
cameras, task sentence. What it can't match: per-episode PHYSICAL_PARAMS draws
(episode meta; replay_actions re-applies them per env slot by default, so
mixed-draw batches replay in one call; a policy eval from a bake alone runs
nominal), executor knobs the data never had (tracker_gains,
grip_margin — free but provenance-recorded), and bit-level residuals (env-grid
origins, warmup prelude, GPU nondeterminism) — matched in distribution only.

## EvalSim

    task, views, rate_hz, spec, state_names
    reset(seed)                          scene randomization + warmup
    init_from_episode(dirs, t0=0)        traj.npz row t0 via set_states, origin-
                                         shifted, per-slot dirs and t0; mid-
                                         episode starts are exact (full state
                                         is recorded every tick)
    step(action)                         policy-convention action; the math
                                         lives here (joint_vel: q_live + v/rate;
                                         closedness -> finger metres - margin)
    step_targets(q_arm, closedness)      the layer under step()
    hold_action()                        identity action (margin-compensated)

    Obs = {images: {view: (E,H,W,3) u8}, state: (E,D) f32, success: (E,) bool}
    # state = [arm q…, closedness] (vla/convert's exact _split_gripper math);
    # one step() = one latch, rendered; cam.update(force_recompute) per view

Success is grader-defined (isolated from the scene by design): the suite's
`GRADERS` entry, fresh per episode init, `check_success()` -> `obs.success`;
suites without a grader fall back to `scene.success()`.

## How to: open-loop replay (replay_actions)

Feeds each episode's own actions back through the executor from its recorded
state; if the demos' own actions can't re-succeed, no policy trained on them
will. Everything runs in the Isaac venv; no lerobot involved.

1. **Pick the sim source** — a registered spec (`bulb_jointpd_60hz`,
   `bulb_osc_60hz`), the dataset's `meta/bake.json`, or `--matched-controller`
   (law from the episodes' own stamped metas).
2. **Pick episodes and start points** — `--episodes <dirs>` / `--batch <dir>`;
   `--t0` in sim time (a video timestamp works verbatim), one value or one per
   episode. Start from scratch AND from segments: a late start is the sanity
   anchor, segment sweeps localize where a run diverges.
3. **Run and read** `report.json`: per-episode success vs recorded,
   first-success tick, grader progress peak/final, tracking err; watch the
   slot-0 mp4s before trusting any number.

    # joint-PD condition, from scratch and from a mid-episode segment
    .venv/bin/python vla/eval/replay_actions.py bulb_jointpd_60hz --headless \
        --episodes <ep_dirs> [--batch <dir>] [--num_envs 4] \
        [--t0 1:20 1:40 | --t0-frac 0.85] [--grip-margin 0.005]

    # matched-controller condition (the exact controller the demos ran under)
    .venv/bin/python vla/eval/replay_actions.py bulb_osc_60hz --headless \
        --episodes <ep_dirs>          # or: <preset> --matched-controller

Semantics:

- Labels are regenerated from `traj.npz` with the bake's convention math
  (never read from parquet): `joint_pos` = achieved `[q[t+1], closed[t+1]]`;
  `joint_vel` = the finite difference (integrated from LIVE q; `--integrate
  dataset` = the pure tracker test); `--matched-controller` = the verbatim
  `action` column under the episodes' stamped controller law (the sanity
  anchor: the exact controller the demos ran under).
- **Batch sync**: one global tick clock, per-env action rows; episodes chunk
  into num_envs groups (sorted by length); a finished slot freezes on a hold
  action, its metrics stop. Per-episode PHYSICAL_PARAMS draws re-applied per
  env slot every chunk (`EvalSim.apply_episode_physics`, the same post-build
  `scene.apply_physical_params` path generation used) — mixed-draw batches
  replay in one call; mixed-law batches are still refused.
- **`--t0` is sim time** ('130', '250.5', clock '4:10.10'), one value or one
  per episode — dataset videos run one frame per tick, so a video timestamp
  is directly a start point. Segmented starts separate compounding divergence
  from genuinely unexecutable phases.
- Output: `report.json` (per-episode + aggregate: success vs recorded,
  first-success tick, per-joint tracking err, gripper err) + mp4s for the
  first `--video-slots` slots.

Interpretation guide: per-tick q error is the metric for `joint_pos`
(absolute targets); for `--matched-controller` it reads huge (~3 rad) from
phase lag on the screw strokes — there the verdict is end-state success only.

## Findings (bulb, stamp_check eps, 2026-08-20)

All artifacts under `vla/example_data/ab_test/`. Task anatomy from the traj:
pick+insert ~20 s, then ~150 s of screwing advancing ~0.3 mm / 10 s; the
grader's seat threshold (27 mm) is crossed at ~1:40 / ~2:05 — before the
recording ends.

| cell | verdict | reading |
|---|---|---|
| jointpd scratch | 0/2 | dies at the PICK: achieved-width closedness = zero squeeze force, bulb slips on lift (convert caveat 1, confirmed on video) |
| jointpd from 2:10 / preseat | 2/2 (trivial) | started past the seat threshold: certifies restore + grader + hold-without-unscrewing |
| jointpd from 1:20/1:40 | **2/2, real** | first-success 31 s / 38 s INTO the replay: joint-PD **completes the threading** — contact flattening does not block thread progress |
| jointpd from 0:36 (full threading) | 0/2 | thread advances (glow on video) but stalls short of seat over the long haul |
| jointpd from 0:36 + 5 mm grip_margin | 0/2 | margin does NOT fix long-haul threading -> the deficit is the PRESS, not grip: joint_pos labels flatten press intent (convert caveat 1) |
| jointpd scratch + 5 mm grip_margin | 0/2 | margin alone doesn't rescue the full run |
| matched scratch (v1) | 0/2 | OUR BUG, fixed: stamped `_kp/_kd` silently skipped -> rot stiffness 30 not 600 (the franka.py stall signature, visible on video); apply is now hard-error-or-applied |
| matched preseat (1:50/2:20) | 2/2 (trivial) | starts were already seated: certifies restore + hold under the OSC, not threading |
| matched scratch (v2, fixed gains) | 0/2 | THE anchor result: even the original controller + verbatim commands can't reproduce 3 min of contact open-loop from t=0 (chaotic divergence — env origins, warmup, GPU float) -> long-episode certification is segmented + statistics, full-length from scratch belongs to the closed-loop policy eval |
| matched from 0:36 | **1/2** | the decisive cell: ep_0000 seats ON the recording's schedule (first-success ~98.9 s vs the recorded ~100 s crossing) where jointpd went 0/2 on the same segment; ep_0001 stalls at progress 0.594 — divergence hits the original controller too |

Standing conclusions: certification of long episodes is **segmented replay +
statistics** (bit-exact open-loop reproduction is not achievable or needed);
the closed-loop policy eval remains the real from-scratch test; `grip_margin`
belongs in the executor because a trained policy will also emit achieved-like
closedness. The from-0:36 triple (jointpd 0/2, jointpd+margin 0/2, matched
1/2 with the pass ON the recording's schedule) supports BOTH effects: long-
segment divergence degrades every executor, and the joint_pos projection
loses strictly more — consistent with the PRESS deficit (sustained downward
intent that achieved-q labels flatten). n=2 is thin; the definitive table is
this sweep over the full 414-episode dataset. Recovery paths, in order —
eval this task in raw_cmd (`bulb_osc_60hz`), or relabel gripper + press from
the `raw_command` column at the next bake.

## Known boundaries (all loud, never silent)

- Scenes with no `CAMERAS` declaration refuse to load (declare a view first).
- New grippers need a `FINGER_TRAVEL` entry (vla/convert/episode.py).
- Custom controller classes: `raw_cmd` requires the mode registered in
  robobench; joint conventions don't care what collected the data.
- Bimanual/multi-gripper: not yet (the closedness scalar is single-hand;
  extend convert + loader together).
- Stride replay (bake rate below the recorded row rate) not implemented —
  replay at the native rate.
- The loader builds the SUITE scene: a campaign cell's hand-modified scene.py
  is deliberately not reproduced.

## How to: closed-loop lerobot eval

Two processes (lerobot needs py>=3.12, Isaac is 3.11), one contract:

    # once: install the plugin into the lerobot venv (auto-discovered by name)
    uv pip install -e vla/eval/lerobot_env_cosigen \
        --python ~/Documents/Research/lerobot/.venv/bin/python

    # terminal 1 (Isaac venv) — pins the eval condition; stays warm across runs
    .venv/bin/python vla/eval/serve.py bulb_jointpd_60hz --headless
        # --init dataset --init-batch <…/data/<batch>>  = start from recorded
        #   states (episode = seed % n); default = scene randomization

    # terminal 2 (lerobot venv)
    lerobot-eval --policy.path=<ckpt> --env.type=cosigen \
        --eval.n_episodes=20 --eval.batch_size=1 --eval.use_async_envs=false

serve.py is a shim over load_sim/EvalSim (all semantics live in sim.py); the
plugin's CosigenEnv follows lerobot's classic obs route ({"pixels": {cam:
HWC u8}, "agent_pos"} -> observation.images.<cam> / observation.state), the
handshake hard-validates config dims vs the served sim, and the condition is
pinned server-side — a result can never half-override the sim it ran on.

The reward channel is the suite grader's weighted rubric progress (0..1;
success-as-float where a suite ships no grader), so eval_info.json carries
individual scores: per-episode `max_rewards` = peak progress (the same scale
as generation's `score`), `sum_rewards` = area under the progress curve, and
each step's info exposes `progress` alongside `is_success`.

Timing semantics: sim time freezes while the policy thinks (blocking socket)
= instant inference; chunking is the policy's own select_action queue
(n_action_steps = the re-plan horizon, a legitimate eval axis). Simulated
latency/RTC would be a client-side wrapper executing stale-chunk ticks at
chunk boundaries — documented, not built.

Verified 2026-08-20: full `lerobot-eval` run (random-weight ACT, 1 episode,
120 ticks) against the live sim — plugin auto-discovery, processor pipeline,
rollout, eval_info.json + episode mp4, exit 0, ~4 ticks/s. A meaningful
success rate needs a checkpoint fine-tuned on a dataset whose bake matches
the served sim (a policy trained under another convention or action width
cannot drive it — the dims fail loudly at the seams).
