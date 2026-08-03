== Parameter search (the parameter_search tool) ==
When a maneuver depends on numbers you would otherwise guess — an offset, a depth, an angle,
a timing, a threshold — search them. The search is PARALLEL: every env carries one candidate,
a generation evaluates the population in passes of `num_envs`, and CMA-ES adapts between
generations. Width is purely speed: a wide env does a generation in one batched rollout; a
1-env session runs the same search one candidate at a time.

E.g.,

    from parameter_search import search

    def rollout(env, p):                           # drives ALL envs at once
        # p[NAME] is a tensor on env.device, shape (num_envs,) — one value per env, so it
        # drops straight into GPU math (a scalar count is int(p[NAME][0]))
        for t in range(240):
            act = batched_action(env, dz=p["PRESS_DZ"], yaw=p["YAW_STEP_DEG"])
            env.step(act)                          # act: (num_envs, action_dim)

    def objective(env):                            # per-env scores, LOWER IS BETTER
        return goal_error(env)                     # shape (num_envs,)

    out = search(PRESET,                           # your task's preset (task.md names it);
                 rollout, objective,               # the tool builds the search env itself:
                                                   # 512 PARALLEL ENVS BY DEFAULT
                 space={"PRESS_DZ": (0.001, 0.02), "YAW_STEP_DEG": (2, 25)},
                 seed_values={"PRESS_DZ": 0.006, "YAW_STEP_DEG": 8},
                 start_state="/workspace/.checkpoints/n3.pt",  # anchor: a saved tree node
                 generations=8, repeats=1)
    # -> {"best": {...}, "best_score": ..., "history": [...], "evaluations": N}
    # or pass an env you built yourself — your env and width are then used as-is

The tool-built env is 512 wide by default; setup time is printed. Each generation replicates
the anchor state into every env (re-offset to each env's grid cell, velocities zeroed),
gives env i candidate i's values through `p` (each entry a (num_envs,) tensor on env.device),
runs your batched rollout, and scores every end state.
`seed_values` — the constants you use now — run verbatim as one candidate of the first
generation, so the result is never worse than the incumbent. The world is restored to its
pre-search state afterwards.

  * `space`: `{'NAME': (lo, hi)}` float; add `'int'`/`'log'` as a third element; a fixed
    set is e.g. `{'NAME': ('choices', (45, 90, 135))}`.
    A `choices` axis is for small categorical CONSTANTS — not for switching strategies:
    all branches share ONE sampling distribution over the other constants, so branches
    wanting different values sabotage each other. To compare strategies, use the sweep
    tool: it runs the exact candidates you name simultaneously and ranks them.
  * `repeats=k`: k envs per candidate, scores averaged — use for noisy contact outcomes.
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
    file and let the tool own the process:

        from parameter_search import launch, status
        launch("/workspace/search_press.py")   # detached, pinned to the freest GPU
        status()                               # instant: gens done, best so far, result

    `status()` reads a machine-written progress file — no log parsing, safe to call as
    often as you like between other work. A search that stopped without a result crashed;
    `status()` names the log to read.
    A generation can take time at full width; the log prints a heartbeat while one is
    evaluating, so a wait between generation lines is normal — let it run.

Writing the objective decides whether this helps or misleads: score the WHOLE goal of the
maneuver per env (a lift that must stay upright and keep hold scores all three), cap credit
at the physically achievable value, penalise ruined outcomes, and put inf/nan in an env's
slot when that candidate must never be chosen.

It must also DISCRIMINATE across the whole range where candidates actually land. A score
built around a success threshold (bonus if achieved, penalty if not) ranks candidates only
while some of them fail; once all of them clear the bar — or none do — there is nothing left
to rank, and the search optimizes residual noise while appearing to make progress. Prefer
continuous measures of quality (error margins, distances, times, force margins) over
threshold checks, so better-versus-worse stays defined at both ends. The result reports
`sensitivity` per parameter and, with `repeats>=2`, warns outright when the gain is inside
the noise — read both: near-zero sensitivity means that under THIS objective the searched
constants do not move the outcome, so change what you measure or what you search.
