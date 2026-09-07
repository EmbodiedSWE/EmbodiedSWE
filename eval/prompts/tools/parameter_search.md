== Parameter search (the parameter_search tool) ==
Eligibility gate: use this only after a viable, stable maneuver already exists and the remaining
uncertainty is numeric — an offset, depth, angle, gain, timing, or threshold. If you are choosing
different control flow, grasp side, phase ordering, or recovery policy, use `sweep` first when it
is granted; otherwise compare those strategies explicitly before tuning. Do not ask one numeric
distribution to invent a strategy. The search is PARALLEL: every env carries one candidate, a
generation evaluates the population in passes of `num_envs`, and CMA-ES adapts between
generations. Width is purely speed: a wide env does a generation in one batched rollout; a 1-env
session runs the same search one candidate at a time.

Use the guarded `tune()` interface. It ranks validity first, goal satisfaction second, and
continuous quality last; a catastrophic candidate therefore cannot buy a better geometric
error by dropping or ejecting the object.

    from parameter_search import tune

    def rollout(env, p):                           # drives ALL envs at once
        # p[NAME] is a tensor on env.device, shape (num_envs,) — one value per env, so it
        # drops straight into GPU math (a scalar count is int(p[NAME][0]))
        for t in range(240):
            act = batched_action(env, dz=p["PRESS_DZ"], yaw=p["YAW_STEP_DEG"])
            env.step(act)                          # act: (num_envs, action_dim)

    out = tune(
        PRESET, rollout,
        goal="seated",                             # public scene bool method, or callable
        quality=lambda env: goal_error(env),       # lower is better; goal status ranks first
        invalid=lambda env: dropped_or_ruined(env),# True candidates can never win
        space={"PRESS_DZ": (0.001, 0.02), "YAW_STEP_DEG": (2, 25)},
        seed_values={"PRESS_DZ": 0.006, "YAW_STEP_DEG": 8},
        start_state="/workspace/.checkpoints/n3.pt",
        generations=8, repeats=3, validation_instances=3,
    )
    # -> best + replay evidence + verdict ADOPT/REJECT/INCONCLUSIVE/WIDEN
    if out["adoptable"]:
        candidate = out["best"]                   # still replay after integration
    # or pass an env you built yourself — your env and width are then used as-is

The tool-built env is 512 wide by default; setup time is printed. Each generation replicates
the anchor state into every env (re-offset to each env's grid cell, velocities zeroed),
gives env i candidate i's values through `p` (each entry a (num_envs,) tensor on env.device),
runs your batched rollout, and scores every end state.
`tune()` requires `seed_values`, smoke-tests them on 2–3 instances before CMA-ES, and replays
the seed and winner from the same held-out anchor afterwards. Treat only `verdict="ADOPT"` as
eligible for integration. `WIDEN` means a better replayed winner sits on a declared bound;
`INCONCLUSIVE` means the evidence is insufficient; `REJECT` means it regressed, was invalid, or
failed paired replay. The tool never edits your solution. The world is restored afterwards.

  * `space`: `{'NAME': (lo, hi)}` float; add `'int'`/`'log'` as a third element; a fixed
    set is e.g. `{'NAME': ('choices', (45, 90, 135))}`.
    A `choices` axis is for small categorical CONSTANTS — not for switching strategies:
    all branches share ONE sampling distribution over the other constants, so branches
    wanting different values sabotage each other. To compare strategies, use the sweep
    tool when granted; otherwise run an explicit controlled comparison before searching.
  * `goal`: a public boolean scene method name such as `"seated"`/`"success"`, or a callable
    returning one boolean per env. Do not pass a 0–100 score as a goal.
  * `quality`: one lower-is-better callable, or an ordered mapping of named callables. Quality
    breaks ties only after invalid and goal status, so no penalty weights can trade safety away.
  * `invalid`: an optional per-env catastrophe mask (dropped, unrecoverably tilted, escaped).
    Generic non-finite, far-out-of-workspace, below-floor, and extreme-velocity checks are also
    applied automatically and reported as fractions.
  * `repeats=k`: k envs per candidate, scores averaged — use at least 2–3 for noisy contacts.
  * `randomize`: jitter named objects' start poses per env, e.g.
    `{"part_0": {"pos": 0.005, "yaw": 0.2}}` — the values are SIGMAS (meters / radians);
    size them from your task's real start variation. The winning constants then survive
    start variation instead of overfitting one pose. Jitter needs `repeats` > 1: each
    candidate must be AVERAGED over several different starts — with one env per candidate,
    every candidate is scored on its own single random start and you are ranking noise,
    not parameters. More repeats, steadier comparison.
  * If you build a wide env YOURSELF, raise PhysX's GPU collision stack or contacts get
    silently DROPPED on contact-rich scenes (the log fills with "collisionStackSize buffer
    overflow" and every score after that is garbage):
    `cfg.build(num_envs=512, sim_overrides={"physx": {"gpu_collision_stack_size": 2**29}})`.
    The tool-built env (pass the preset name) already does this.
  * NEVER run a search in the foreground — searches run long, and a foreground command
    that outlives its timeout dies with nothing to show. Put your search in its own script
    file and let the tool own the process. The script runs as a FRESH process, so it must
    boot the app itself before any Isaac/robobench import:

        # top of /workspace/search_press.py
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True)
        # ... then the imports and the tune() call

        from parameter_search import launch, status
        launch("/workspace/search_press.py")   # detached, pinned to the freest GPU
        status()                               # instant: gens done, best so far, result

    `status()` reads a machine-written progress file — no log parsing, safe to call as
    often as you like between other work. It prints the final verdict first, reports stalled
    progress without killing anything, and names the complete log for failures.
    `launch()` refuses duplicate active content and, by default, a second search on the same GPU.
    A generation can take time at full width; the log prints a heartbeat while one is
    evaluating, so a wait between generation lines is normal — let it run.

The low-level `search()` API remains for compatibility, but it returns an unvalidated numerical
optimum. New work should use `tune()`: state the whole goal, put unrecoverable outcomes in the
`invalid` mask, and use bounded continuous quality only to rank candidates with equal validity
and goal status.

It must also DISCRIMINATE across the whole range where candidates actually land. A score
built around a success threshold (bonus if achieved, penalty if not) ranks candidates only
while some of them fail; once all of them clear the bar — or none do — there is nothing left
to rank, and the search optimizes residual noise while appearing to make progress. Prefer
continuous measures of quality (error margins, distances, times, force margins) over
threshold checks, so better-versus-worse stays defined at both ends. The result reports
`sensitivity` per parameter and, with `repeats>=2`, warns when population spread is inside
the noise. `population_spread` is not improvement; only `improvement_vs_seed` and the paired
replay verdict compare against the incumbent. Near-zero sensitivity means the searched
constants do not move the outcome, so change what you measure or what you search.

After an `ADOPT` verdict, write the candidate into the real maneuver and replay the integrated
path. Record failed candidates and regressions. Search replay is still not the final verdict:
the complete solution must pass the queued harness verification from a fresh reset.
