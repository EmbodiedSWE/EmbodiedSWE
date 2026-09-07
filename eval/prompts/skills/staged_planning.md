== Staged planning and assessment ==
Plan the task before driving it, code it up stage by stage, and judge every run that moved
the world.

  1. First make an explicit plan of stages, each a meaningful subgoal named by the STATE that
     completes it ("one leg seated", "card seated in its slot", "all three eggs in pockets").
     Keep it as a written checklist (your todo tool if you have one, else a plan file): one
     item per stage, exactly one in progress at a time. Keep the plan short — a few stages
     (say 2–4), each a real seam in the task. A single stage is a valid plan when the task has
     no clean seam or is easier to code and debug as a whole; split only where a later stage
     will be developed many times and the earlier ones are expensive to redo.
     Start every program with comments marking its stage and origin:
         # plan: [stage 2/3] <what this stage does>
         # origin=fresh
     Every program starts from a fresh environment. When a checkpoint tool is granted, only an
     explicit `goto()` / `run_stage()` changes that; a restored run must instead say
     `origin=checkpoint:<id>`.
  2. When a checkpoint tool is granted, the plan and the checkpoints are ONE thing. Record the
     plan in the tool (`tree.plan([...])`), write each stage as its own module
     (`solution/stages/stage_<k>.py` with `run(env)` and `check(env)`), and develop stage k
     with `tree.run_stage(k)`: it restores the previous stage's boundary, runs your stage,
     runs your check, and saves this stage's boundary node when the check passes. Every
     completed stage gets a boundary; between boundaries nothing is saved. Continue a later
     stage from a boundary (`goto_stage`) instead of re-deriving the world from scratch, and
     when a stage keeps failing, step back to an earlier boundary and change approach. The
     deliverable `solve.py` is the stages composed in order.
  3. One program should complete one full stage — typically several hundred sim steps, with
     internal retries and printed verification — not one micro-motion. Runs have real
     overhead, so batch exploration into one run.
  4. End every run that moves the world with evidence of where it ended: a snapshot of the
     final state (the scene_view tool, when granted) and the measurements it printed. A path
     is not an image: transport/open the actual pixels before making a visual claim. Read
     both, then write the review down (the assessment tool, when granted), whether the run
     succeeded or failed:
         from assessment import assess
         assess(scene=..., log=..., failure_modes=...)
     Say what the image shows, what the log shows, and which failure modes are visible in
     them. Naming the failure mode out loud is what stops the next run from repeating it —
     and `history()` is how a later script recalls what was already concluded.
  5. If a stage keeps failing the same way, change the approach rather than repeating it.
  6. Compare genuinely different strategies with sweep when granted (from a stage boundary,
     so every strategy starts from the same state). When parameter search is granted, tune
     numeric constants only after one stable viable maneuver exists; a seeded in-search score
     is not a no-regression guarantee until replay.
  7. After choosing a candidate, run a 2–3-instance holdout, integrate the winner into
     `solution/`, and queue the required full end-to-end verifier from a fresh reset.
     Development evidence and stage boundaries do not replace that final check.
