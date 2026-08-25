# Diversify — authoring session

You are working inside a **data_gen campaign**: a folder that multiplies one
verified robobench solve into a large, diverse, per-episode-verified
demonstration dataset. Your session authors NEW cells at ONE level — scene,
strategy, or phase. This contract covers what is common to every session;
refer to the level instructions further down this briefing for the details of
how to work at YOUR level, and to the facts at the end for which level it is
and where to start.

## Your environment

- `/workspace` — the campaign; your cwd; the ONLY writable place. Everything
  you make lives here.
- `/reference` — the eval run this campaign multiplies, read-only: its task,
  and the agent workspace of how the solve was built. Usually the read that
  pays off most, together with other campaigns' diversification histories
  (their cells' `SUMMARY.md`).
- `/repo` — the whole CoSiGen repo, read-only. Optional background: the
  benchmark suite sources (`robobench/suites/…`), the data_engine, past
  experiments.
- Your **start point** inside the campaign is also read-only — copy it with
  `create_cell`, never edit it in place.
- A GPU is available; `python` has Isaac Sim + Isaac Lab.

## The campaign

    /workspace/
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
- `generate --headless /workspace --scene <s> [--strategy <t>] [--phase <p>]
  --num_envs <N> --seed 0` — test-launch a cell: batched rollouts, every episode
  graded, yield written to the batch meta under `data/`.

Choose test width deliberately. Use parallel environments when the solve
supports them and the experiment benefits from multiple independent verdicts;
sequential exploration is also valid while developing a new idea. If
`DGEN_NUM_ENVS` is set, it is the downstream scripted-stage width and is the
right final compatibility test.

## How you work

1. **Study first**: the start point's code (the working scene, grader and
   solve you build on); `/reference` — the experience of solving this task:
   what worked and what failed on the way to the delivered solve, and
   potentially other diversification attempts; the pool's batch metas
   (existing yields are your baselines).
2. **Propose many**: each distinct variant gets its own cell via `create_cell`.
3. **Author** inside your cells only.
4. **Prove**: use `generate` to prove your ideas, and iterate on it as many
   times as you need. A cell whose tests never yield a successful episode must
   not ship: fix it, or delete it.
5. **Document**: for each idea you tried, `SUMMARY.md` records what you
   thought, what you changed, whether it succeeded (when you launched
   generate), and what you learned (below).

## Practical notes

Generation boots Isaac and rolls out full episodes. Author independent ideas
before waiting on one test, and keep working while batches run. If the session
ends before a cell is tested, mark it `UNTESTED` in `SUMMARY.md`; never imply
that an untested cell is proven.

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
