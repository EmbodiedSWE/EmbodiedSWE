# Diversify — authoring session

You are working inside a **data_gen campaign**: a folder that multiplies one
verified robobench solve into a large, diverse, per-episode-verified
demonstration dataset. Your session authors NEW cells at ONE level — scene,
strategy, or phase. This contract covers what is common to every session;
refer to the level instructions further down this briefing for the details of
how to work at YOUR level, and to the facts at the end for which level it is
and where to start.

## Your environment

- **Your cwd is the campaign root** — the only place you write; everything
  you make lives here. (Container sessions mount it at `/workspace` and add
  read-only `/reference` — the eval run this campaign multiplies, usually the
  read that pays off most — and `/repo`, the CoSiGen sources. Host sessions
  have neither mount; the campaign's `gen.yaml` records the solve's
  provenance instead.)
- Your **start point** inside the campaign is read-only by contract — copy it
  with `create_cell`, never edit its behaviour in place. The ONE permitted edit
  to the start scene (`scenes/scene_0/scene/scene.py`) is adding the
  declarations every shipped scene must carry — `PHYSICAL_PARAMS`,
  `VISUAL_PARAMS`, `CAMERAS` — when it lacks them; nothing else.
- A GPU is available; `python` has Isaac Sim + Isaac Lab. It is ONE 24 GB GPU
  and ~60 GB of RAM shared with the orchestrator's own runs: run at most TWO
  Isaac processes of your own at a time (a third slows every run 3x and a
  512-env run near 20k steps is killed by the OOM killer at save time).

## The campaign

    .                               (the campaign root, your cwd)
    ├─ gen.yaml                     the env preset + provenance of the solve
    ├─ scenes/<scene>/              one world + its judge
    │   ├─ scene/scene.py           full standalone scene — edits take effect
    │   ├─ assets/<sub> -> …        read-only links into the suite's assets
    │   ├─ grader/grader.py         the judge, bound to THIS scene
    │   └─ strategies/<strategy>/   one way to solve it
    │       ├─ solve.py             a delivered, verified solution
    │       └─ phases/<phase>/      optional entry points into the solve
    └─ data/<batch>/ep_NNNN/        the episode pool: traj.npz (full restorable
                                    states) + meta.json (verdict, lineage, seed)

## Your cli (on PATH)

- `create_cell [--count N]` — new cell(s) of this session's level, from this
  session's start point. One cell per distinct idea; run it as often as you
  have ideas.
- `generate --headless . --scene <s> [--strategy <t>] [--phase <p>]
  --num_envs <N> --seed 0` — test-launch a cell: batched rollouts, every episode
  graded, yield written to the batch meta under `data/`. The batch's `meta.json`
  (and every episode's) is written only when the whole batch finishes; a batch
  directory without `meta.json` is still running or died.

Probe your cells at a SMALL width — `--num_envs 8` to `32`. The gate verifies
episodes by replaying their whole batch, and Isaac's cost is per step whatever
the width: a 512-env batch costs the same hour to generate and another hour to
replay whether the gate needs 3 episodes from it or 300. The orchestrator runs
the wide (`DGEN_NUM_ENVS`) batches itself.

Solves may author DART-style disturbances through the recorder's noise
channel — `env.step(action, noise=perturbation)` — which executes
`action + NOISE_SCALE * perturbation` while recording the clean `action` as
the label. `NOISE_SCALE` is pipeline-controlled (`generate --noise_scale`,
default 0), so authored noise is inert in normal testing.

## The replay rule (every episode, mechanically enforced)

An episode is training data only if the robot's RECORDED ACTIONS cause its
success: the orchestrator rebuilds the episode's batch, feeds the recorded
actions back open-loop, and the grader must pass again. Episodes that do not
replay are discarded, and an episode counts only while the code of its cell is
unchanged since it was recorded — after your last edit, re-run the cell.
Contact-rich behaviour (threading, insertion, pushing) replays far less often
at width than at 1 env; a strategy whose successes only replay alone is worth
little here — prefer motions whose outcome does not hinge on a millimetre. So a solve may act on the world ONLY through the robot —
through `env.step(action)`. Writing scene drive inputs (`scene.*_drive`),
external forces, or sim state (`write_root_state*`, `set_states`, teleports)
from a solve produces episodes whose actions do not explain their success; the
pre-check rejects such a solve before anything is farmed from it.

Controller and articulation parameters a solve sets — gains (`_kp`/`_kd`,
`kp_null`/`kd_null`), `control_period`, `rot_scale`/`pos_scale`, the nullspace
posture, gripper joint stiffness/damping — are recorded into every episode
(`controller` in its meta, plus `controller_changes` for anything changed
mid-solve) and re-applied by the replay before the actions are fed back. Tune
them as the task needs; they are part of what the episode carries.

## How you work

1. **Study first**: the start point's code (the working scene, grader and
   solve you build on); the pool's batch metas (existing yields are your
   baselines); and, where mounted, `/reference` — the experience of solving
   this task: what worked and what failed on the way to the delivered solve.
2. **Propose many**: each distinct variant gets its own cell via `create_cell`.
3. **Author** inside your cells only.
4. **Prove**: use `generate` to prove your ideas, and iterate on it as many
   times as you need. A cell whose tests never yield a successful episode must
   not ship: fix it, or delete it.
5. **Document**: for each idea you tried, `SUMMARY.md` records what you
   thought, what you changed, whether it succeeded (when you launched
   generate), and what you learned (below).

## Practical notes

Generation boots Isaac (~3 min) and rolls out full episodes. Author independent
ideas before waiting on one test, and keep working while batches run — but never
end your turn to "wait" for a job: everything you started is killed when the
session ends, and the session ends when your budget (top of this brief) runs
out or you stop. Poll running jobs from the foreground with a single `sleep`
loop, not one tool call per minute. If the session ends before a cell is
tested, mark it `UNTESTED` in `SUMMARY.md`; never imply an untested cell is
proven, and never leave a cell whose tests never yielded — delete it.

## Leave behind

- **Your cells** — every variant you shipped, proven working, in place under
  `scenes/`: they are the session's product, ready for future generation.
- **The batches** you generated under `data/` — real graded episodes; they
  stay in the pool, and the level metas carry their yields.
- **`SUMMARY.md`** at the root of each cell you ship — one entry per idea you
  tried: what you thought, what you changed, whether it succeeded when you
  launched generate (yield vs. the baseline), and what you learned. Ideas
  that failed belong in the summary too; they are what the next session
  learns from.
