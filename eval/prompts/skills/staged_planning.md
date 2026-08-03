== Staged planning and assessment ==
Plan the task before driving it, and judge every run that moved the world.

  1. First make an explicit plan of stages, each a meaningful subgoal, and keep it
     as a written checklist (your todo tool if you have one, else a plan file):
     one item per stage, exactly one in progress at a time. Also start every
     program with a comment marking its stage:
         # plan: [stage 3/7] <what this stage does>
  2. One program should complete one full stage — typically several hundred sim
     steps, with internal retries and printed verification — not one micro-motion.
     Runs have real overhead, so batch exploration into one run. Splitting work
     into smaller stages is still worth it when you want progress recorded more
     often.
  3. End every run that moves the world with evidence of where it ended: a snapshot
     of the final state (the scene_view tool, when granted) and the measurements it
     printed. Read both, then write the review down (the assessment tool, when
     granted):
         from assessment import assess
         assess(scene=..., log=..., failure_modes=...)
     Say what the image shows, what the log shows, and which failure modes are
     visible in them. Assessing is not a formality: naming the failure mode out
     loud is what stops the next run from repeating it — and `history()` is how a
     later script recalls what was already concluded.
  4. If a stage keeps failing the same way, change the approach rather than
     repeating it.
