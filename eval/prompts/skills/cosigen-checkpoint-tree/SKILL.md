---
name: cosigen-checkpoint-tree
description: Uses the checkpoint_tree tool to bind the stage plan to saved states — one boundary node per completed stage, developed with run_stage, continued from with goto_stage, backtracked to when a stage keeps failing. Use when planning stages, deciding what to checkpoint, and choosing where to continue from.
---

# Checkpoint tree

The tree is separate from your code and your notes. Restoring a node (`goto`, `goto_stage`,
`run_stage`) changes only the simulator world; your knowledge and workspace files remain. It
is disk-backed (`/workspace/.checkpoints`), so a state reached in one process can be restored
in the next. Every script nevertheless starts from the fresh reset `n0` unless it restores
explicitly: label its evidence `origin=fresh` or `origin=checkpoint:<id>`.

The protocol:

1. `tree.plan([...])` — one entry per stage, each naming the STATE that completes it. Keep it
   to a few real seams (2–4 is typical); one stage is a valid plan for a task with no clean
   seam.
2. One module per stage at `/workspace/solution/stages/stage_<k>.py`:
   `run(env)` drives from the previous boundary to this stage's goal; `check(env) -> bool` is
   your own verification of that goal. `solve.py` composes the stages in order.
3. `tree.run_stage(k)` — restores the latest stage-(k−1) boundary (or `from_node=`), runs your
   stage, runs your check, and saves the boundary node when the check passes. A failed check
   saves nothing and records the attempt with its log; you are told so.
4. `tree.goto_stage(k)` / `tree.goto(cid)` to continue from a boundary; `tree.show()` for the
   plan's progress (✓/○, boundary nodes) and the whole tree; `tree.tried_from(cid)` before
   retrying anything from a node.

Rules of thumb:

- One boundary per completed stage; nothing saved in between. Diagnostic probes and hopeful
  intermediate poses are not stages.
- Settle before the boundary: hold still until the world stops moving and your check can
  measure. The health block at every save (motion, joint-limit margin, scene success) tells you
  whether the state is a good place to continue from.
- Several boundary nodes for one stage are normal (different ways to the same subgoal); pick
  the one to build on with `goto_stage(k, cid=...)`.
- Two attempts from the same boundary failing the same way mean the approach is wrong: go back
  to an earlier boundary and change approach, or re-plan.
- Read a node before landing on it: its snapshot, `get_log(cid)` and `<cid>.code.py`.
- With a viewer attached, annotate what visibly changed: `tree.annotate(cid, scene_diff=...)`.

Boundaries accelerate development; they do not prove the integrated solution. The composed
`solve.py` must still pass the queued fresh-reset verifier.
