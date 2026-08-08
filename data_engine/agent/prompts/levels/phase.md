# This level: phase

A **phase** is an entry point into strategy {base}'s solve: instead of always
starting from the scene's reset, an episode can begin mid-task — from a
prepared entry state — and still run to task-done, where the grader judges it
as usual. Phases buy state coverage the from-scratch solve rarely visits, and
they make the hard, rare segments of the task cheap to farm.

## Deliverables

Create, explicitly:

1. `phases/phase_N/` — one cell PER proposal of how to divide/enter the solve
   (via `create_cell`), self-contained and declared entirely in code:
   - `solve_by_phase.py` — THIS cell's port: `solve(env, entry=<phase>)` runs
     from any phase of the division to task-done; `ENTRIES` documents the
     phases and their preconditions. Each `phase_N` is a different division
     strategy of the solve — in most cases `phase_0` is all you need.
   - `reset/` — one file per phase, named exactly as the phase (one per
     `ENTRIES` key). Each round sweeps ALL the files, one rollout per file: a
     file chooses the entry and builds its state via `reset_0(env)`,
     `reset_1(env)`, … — all applied, the rollout's envs divided evenly among
     them.

At generation time the runner builds the entry state from the rollout's reset
FIRST, then calls your port. Entry states are the reset builders' job — your
port only steps (`reset`/`set_states` are not blocked, but teleported state
becomes part of the recorded demonstration). The phases are not given to you:
**you determine them by studying `solve.py`** — the points that can safely
accept a mid-task start.

## Instructions: solve_by_phase.py

`solve.py` usually separates into several phases. The port makes each phase an
entry point: `solve(env, entry="<phase>")` jumps directly there and runs to
task-done. A solution typically looks like:

    def solve(env):
        ...
        st = fresh_state()               # {"phase": "hover", "grip_w": None, …}
        for each control tick:
            if st["phase"] == "hover":
                ...
            elif st["phase"] == "grasp":
                ...
                st["grip_w"] = fpos          # measured HERE
                st["phase"] = "lift"
            elif st["phase"] == "insert":
                grip = st["grip_w"] - N / KP  # read LATER

The port is a copy of `solve.py` with four edits — everything else identical:

1. `def solve(env, entry=None)`; an unknown entry raises with the valid names.
2. at module top:

       ENTRIES = {"<phase>": "<one-line precondition on the world>", ...}

   (the runner never reads it — each rollout's entry is the NAME of its reset
   file; keep one reset file per ENTRIES key)
3. `fresh_state()` starts at `st["phase"] = entry or "<start>"`.
4. `entry_calibrate(...)` on the first tick: fill what the skipped phases
   would have measured, from observation — e.g. entering at "insert" above,
   `st["grip_w"]` = the observed finger joint position.

**Which phases qualify as entries**: those whose every read value can be
re-derived at entry from observation (grip widths ← finger joints; offsets ←
observed poses; heights ← known geometry) or safely defaulted (counters ← 0).
If a value cannot be reconstructed (e.g. something integrated over motion),
that phase is not an entry — do not force it; note the rejection and why in
the port's docstring.

Tips: phases that re-measure everything on entry (a regrasp) are free entries
— find them first. `solve.py` is never touched. A procedural solve (a chain
of step calls instead of an FSM) ports the same way: wrap the steps as
functions, dispatch from entry, re-derive the earlier locals.

If you cannot decide how to divide, one phase is fine: keep
`solve_by_phase.py` with a single entry — the solve's natural start — and
propose different reset builders for it (varied initial conditions instead of
mid-task entries).

## Instructions: reset/ (entry-state builders)

One file per phase, named exactly as the phase (one per `ENTRIES` key). Each
holds one or more builders `reset_0(env)`, `reset_1(env)`, … — a builder shapes
the whole batch's worlds into that phase's entry states, one draw per env, and
must satisfy the phase's precondition as documented in `ENTRIES`.

### Where entry states come from

- **Built by hand**: when you know what the entry state looks like
  geometrically, write it directly instead of restoring a recording. Often
  three lines: set the part's pose at a known relation to the observed
  fixture; mind that a written state includes velocities — set them
  deliberately (zeros for at rest; stale copied values send the part flying);
  then step the sim briefly with a hold action so the part comes to genuine
  resting contact — a computed pose is never contact-perfect. How to build depends
  on the nature of the phase: entering mid-carry (part in hand) the state is
  object-centric — place the object first, derive the arm pose from the grasp
  relation (via IK); at a hand-free phase, object and arm are independent —
  set the arm freely, or just leave it where the scene reset put it.

- **The pool** (`/workspace/data/`), when batches exist there — and your own
  test batches add to it: every episode's `traj.npz` stores full restorable
  sim states, `env.set_states(...)` any recorded step. Successes give mid-task
  boundary states (e.g. "part placed, hand free"), perturbed with jitter for
  coverage; failures give recovery starts that from-scratch data cannot
  contain. Recoverable means some phase of your division can redo the work
  from there — a part dropped beside the goal is a valid entry to the pick
  phase, with the dropped pose as the initial condition. An unrecoverable
  mess is not an entry state.

### Practice

- randomize PER ENV: each of the `env.num_envs` worlds gets its own draw
  (poses, sources, jitters) — a builder that sets every env identically wastes
  the batch.
- locate the campaign relative to YOUR OWN FILE, never a mount path: builders
  also run outside the container. From `reset/<phase>.py` the pool is
  `Path(__file__).resolve().parents[7] / "data"` — `/workspace/...` breaks on
  the host.
- the arm: the scene's own reset already ran — leaving the arm where it is
  often IS the precondition ("hand free and clear"). Only pose the arm if the
  phase genuinely needs it.
- different builders in one file = different state families for the same phase
  (e.g. `reset_0` built-by-hand nominal, `reset_1` restored-from-failures);
  each rollout runs them all, its envs divided evenly among them.
- as MANY anchor poses as possible, each with small jitter: no need for one
  generic builder over the whole state space — cover the precondition's
  feasible region with many anchors (approach side, position on the
  workspace, orientation), recorded or hardcoded, and perturb a little around
  each. One anchor + jitter replays the same episode over and over; many
  anchors is what makes the data diverse.
- check your own precondition: settling can knock a part out of the intended
  state in some envs. After the settle, verify the precondition per env (poses
  are observable) and re-draw just the envs that missed — a dead entry state
  burns a full episode at generation time.

A builder typically looks like (this one is real, for the bulb scene — your
`scene.py` declares the asset names; everything is batched, one call covers
all `env.num_envs` worlds; `set_states` is the same API used to restore
recorded pool states):

    import torch

    def reset_0(env) -> None:
        """Bulb resting in the socket bore, jittered on the axis."""
        E = env.num_envs
        st = env.get_states()                        # {"scene": {...}, "robot": {...}}
        bulb = st["scene"]["bulbs"]                  # (E, 1, 13): pos + quat + vel
        sp = env.scene.sockets[0].data.root_pos_w    # (E, 3)
        bulb[:, 0, 0:2] = sp[:, 0:2] + (torch.rand(E, 2, device=sp.device) - 0.5) * 0.004
        bulb[:, 0, 2] = sp[:, 2] + 0.034             # just above free-rest
        bulb[:, 0, 3:7] = torch.tensor([1.0, 0, 0, 0], device=sp.device)  # upright
        bulb[:, 0, 7:13] = 0.0                       # at rest
        # the arm: hardcode a pose when the phase needs one (set the targets
        # too, or the controller pulls it back to the old ones)
        ARM = torch.tensor([0.0, -0.4, 0.0, -2.1, 0.0, 1.9, 0.8, 0.04, 0.04])
        st["robot"]["joint_pos"][:] = ARM            # (E, 9): 7 arm + 2 fingers
        st["robot"]["joint_vel"][:] = 0.0
        st["robot"]["joint_pos_target"][:] = ARM
        env.set_states(st)
        hold = torch.zeros(E, env.robot.action_dim, device=env.device)
        hold[:, 6:8] = 0.04                          # fingers open, arm holds
        for _ in range(72):                          # settle into contact
            env.step(hold)

## Verification

    generate --headless /workspace --scene <scene> --strategy {base} \
        --phase phase_N --num_envs 64 --seed 0

Each round sweeps ALL your reset files, one rollout of `--num_envs` episodes
per file, the envs divided evenly among the file's builders — a single round
already exercises every entry and every builder; each episode's meta records
its (file, builder) lineage.
This generates one batch of data using your proposed `phase_N` and its initial
conditions, under `/workspace/data/<batch>/`: one `ep_NNNN/` folder per
episode, success/fail in each episode's `meta.json`, and the batch summary
(yield) in `data/<batch>/meta.json`. Judge by success, not score — a mid-phase entry gets partial score for free (the entry state already satisfies part of the rubric).
