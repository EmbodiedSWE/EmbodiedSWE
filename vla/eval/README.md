# vla/eval — closed-loop eval sims for the VLA pipeline

Load the same world a dataset was baked from, drive it with policy-convention
actions, score with the grader. Two tools today; the lerobot eval server plugs
into the same loader next.

    sim.py            load_sim() + SimSpec + the EvalSim facade
    specs/            named eval setups (register_sim; one module per scene family)
    check_load.py     stage-0 loader check: build, warmup, hold, snapshot obs
    replay_actions.py re-drive recorded episodes' actions (executor certification)
    _out/             default output dir (git-ignored scratch)

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
(episode meta, matched by replay_actions by default; a policy eval from a bake
alone runs nominal), executor knobs the data never had (tracker_gains,
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

## replay_actions — executor certification

    .venv/bin/python vla/eval/replay_actions.py bulb_jointpd_60hz --headless \
        --episodes <ep_dirs> [--batch <dir>] [--num_envs 4] \
        [--t0 1:20 1:40 | --t0-frac 0.85] [--grip-margin 0.005]

    .venv/bin/python vla/eval/replay_actions.py assembly.bulb.franka.osc \
        --matched-controller --headless --batch <dir>

Feeds each episode's own actions back through the executor from its recorded
state; if the demos' own actions can't re-succeed, no policy trained on them
will. Semantics:

- Labels are regenerated from `traj.npz` with the bake's convention math
  (never read from parquet): `joint_pos` = achieved `[q[t+1], closed[t+1]]`;
  `joint_vel` = the finite difference (integrated from LIVE q; `--integrate
  dataset` = the pure tracker test); `--matched-controller` = the verbatim
  `action` column under the episodes' stamped controller law (the sanity
  anchor: the exact controller the demos ran under).
- **Batch sync**: one global tick clock, per-env action rows; episodes chunk
  into num_envs groups (sorted by length); a finished slot freezes on a hold
  action, its metrics stop. Per-episode PHYSICAL_PARAMS draws matched by
  default; mixed-draw / mixed-law batches refused.
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
| matched scratch (v1) | 0/2 | OUR BUG, fixed: stamped `_kp/_kd` silently skipped -> rot stiffness 30 not 600 (the franka.py stall signature, visible on video); apply is now hard-error-or-applied |
| matched scratch (v2) | pending | the true sanity anchor |
| jointpd scratch + 5 mm grip_margin | pending | does fixing the pick make the full open-loop run pass? |

Standing conclusions: certification of long episodes is **segmented replay +
statistics** (bit-exact open-loop reproduction is not achievable or needed);
the closed-loop policy eval remains the real from-scratch test; `grip_margin`
belongs in the executor because a trained policy will also emit achieved-like
closedness.

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

## Next

The lerobot eval server: a socket shim over this same loader (lerobot's venv
is py>=3.12, Isaac's is 3.11 — two processes), with a `lerobot_env_cosigen`
plugin so `lerobot-eval --env.type=cosigen --policy.path=...` drives EvalSim
closed-loop. The bake stays the single source of truth end to end.
