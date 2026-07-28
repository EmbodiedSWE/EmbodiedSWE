---
name: cosigen-staged-planning
description: Plans and executes long-horizon CoSiGen manipulation tasks in meaningful stages with a TodoWrite checklist and visual verification. Use for every CoSiGen task before acting and when revising a failed stage.
---

# CoSiGen staged planning

1. Inspect the initial scene numerically, and run something small that moves the world if
   you want to see it — every such run returns an image of the state it reached.
2. Write the task plan as a TodoWrite checklist: one todo per stage, each a meaningful
   physical milestone (a grasp, a placement, an insertion), not one small motion.
   Keep exactly one stage in_progress at a time.
3. Work on the current stage as a program file in your workspace: run it with
   execute(path=...), read the printed measurements and the end-state image, edit the
   file, run it again. Start the program with `# plan: [stage i/N] ...` so its purpose is
   recorded.
<!-- if:checkpoint -->
4. Save your progress often with checkpoint (a label and the path); checkpointing lets
   you build your work off of promising states and backtrack to earlier states when you
   need to. You can consider splitting your work and program into smaller pieces /
   stages, and checkpoint at stages you want, so that you could save progress more
   frequently. When a stage's program is checkpointed, mark that todo completed and
   move the next one to in_progress.
<!-- endif -->
<!-- if:opt -->
5. Use the optimize tool to find optimal values of parameter settings in your programs:
   whenever a program has parameters that you need to decide with trial and error
   (offsets, depths, angles, timings, thresholds), call optimize on that program file
   instead of guessing new values run after run.
<!-- endif -->
<!-- if:checkpoint -->
6. If a stage keeps failing in its approach, do not repeat small variations
   indefinitely: inspect the checkpoint tree, look at the best earlier node with goto
   (it shows the node before moving there, and you confirm), update the todo list to
   reflect the new plan, and change the strategy substantially.
<!-- else -->
6. If a stage keeps failing in its approach, do not repeat small variations
   indefinitely: update the todo list to reflect a new plan and change the strategy
   substantially.
<!-- endif -->
7. After every run that moved the world, assess it: what the image shows, what the log
   shows, the failure modes you can read off them, and whether to keep the state. Use
   view(frames=k) when the end state alone does not explain what happened.

Only report completion after the simulator success predicate is true.
