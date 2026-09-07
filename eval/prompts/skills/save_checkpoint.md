# Skill: checkpoint every completed stage

A checkpoint is a stage boundary: the world as it stands when one stage of your plan is
done. Save one at the end of every completed stage, and nothing in between. Every new script
starts from the task's fresh initial condition unless it explicitly restores a node; declare
each experiment as `origin=fresh` or `origin=checkpoint:<id>`.

- **Plan, then bind the plan to the tree.** `tree.plan([...])` records your stages (one is
  fine). Each stage is a module `solution/stages/stage_<k>.py` with `run(env)` and
  `check(env)`; `tree.run_stage(k)` develops it from the previous boundary and saves this
  stage's boundary when your check passes.
- **Before the boundary, settle and verify.** Let `run(env)` hold still long enough that the
  world stops moving and your `check(env)` measures the subgoal it claims. The health block
  printed at every save (motion, robot joint-limit margin, scene success flag) tells you
  whether the state is a good place to continue from; a failed check saves nothing and is
  recorded as an attempt.
- **Continue from boundaries, do not re-derive.** Develop stage k+1 from a stage-k boundary
  (`run_stage` does this; `goto_stage(k)` for anything else). Several boundary nodes may exist
  for one stage — pick the one you want to build on.
- **Go back rather than push on.** When two attempts at a stage fail the same way, the branch
  is wrong, not under-tuned: return to an earlier boundary and change approach, or re-plan.
  `tried_from()` lists what already failed from a node so you do not repeat it.
- **Record failures without promoting them.** `run_stage` records a failed stage as an
  attempt; use `record_attempt()` or `with tree.attempt(...)` for anything else you tried.
  Attempts keep their evidence but are never restorable.
- **Read a node before you land on it.** Its snapshot, log and program tell you what it holds.

The deliverable is still one integrated program — the stages composed in `solve.py` — and it
must pass the final queued verification from a fresh reset; boundaries do not replace that.
