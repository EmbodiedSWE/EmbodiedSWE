== Sweep (the sweep tool) ==
When you are choosing BETWEEN approaches — not tuning numbers — run the candidates you
name SIMULTANEOUSLY, each from the same saved state, and rank them. A candidate is
(function, params): your function, with its own control flow — the same function with
different parameters, different functions, any mix — all stepping one simulator together,
each on its own slice of the parallel envs. Everything finishes in the wall-clock of the
longest candidate, each scored over its whole group: a distribution per approach, not one
noisy episode per idea.

The protocol — your candidate is a generator `def my_approach(env, ids, **params)`:

  * `ids` is the tensor of env indices assigned to you (your group — your repeats).
  * Write your normal control loop; wherever you would call `env.step(act)`, instead
    `yield act` — action rows for YOUR envs only, shape (len(ids), action_dim), on
    `env.device`. Every yield is validated; a wrong shape fails loudly, naming you.
  * Each `yield` advances the WHOLE world one step; on resume, read fresh state sliced
    by your group (e.g. `scene.part.data.root_pos_w[ids]`).
  * Returning ends your group: its envs hold at zero action until all candidates finish.
  * NEVER: call `env.step`/`env.reset`/`env.set_states` (the tool owns stepping and
    state), touch envs outside `ids`, or set SIM-GLOBAL state (controller gains, torque
    limits, physics settings) — a global write hits every candidate's envs instantly and
    silently corrupts the comparison. To compare global settings, run separate sweeps.

E.g. (an example — the functions, names and numbers are all yours; `*_rows` are action
builders you write, each returning (len(ids), action_dim) rows for your group):

    from sweep import sweep

    def one_shot(env, ids, gain):                # one phase, early exit
        for k in range(200):
            yield drive_rows(env, ids, gain)     # world steps once per yield
            if done_mask(env)[ids].all():
                break

    def two_stage(env, ids, gain):               # structurally different: two phases
        for k in range(80):
            yield align_rows(env, ids)
        while not settled(env)[ids].all():       # wait, then finish
            yield hold_rows(env, ids)
        for k in range(150):
            yield finish_rows(env, ids, gain)

    def objective(env):                          # per-env scores, LOWER IS BETTER
        return goal_error(env)                   # shape (num_envs,)

    out = sweep(PRESET,                          # your task's preset (task.md names it)
                candidates=[(one_shot,  {"gain": 0.5}),
                            (one_shot,  {"gain": 2.0}),
                            (two_stage, {"gain": 0.5})],
                objective=objective,
                start_state="/workspace/.checkpoints/n2.pt",   # anchor: a saved tree node
                randomize={"part_0": {"pos": 0.005}})          # optional start jitter
    # -> {"ranking": [best candidate's index first, ...], "best": (name, params),
    #     "scores": [...], "stds": [...]} — scores stay in your candidate order

  * Settle WHICH approach wins here, then hand the winner's constants to
    `parameter_search.search` to tune: sweep runs what you name, search invents values.
  * Run it like a search: from a script via `parameter_search.launch()` — never in the
    foreground; the log shows a running heartbeat, each candidate finishing, and the
    final ranking. Long candidates take time — let it run.
