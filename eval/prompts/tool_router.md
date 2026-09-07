# Granted-tool router

Granted in this run: {granted}.
Use a tool when its gate below matches. Open `/task/tools.md` for the full API of
a tool when you choose that branch; do not reread the whole manual on every iteration.

1. **Inspect.** Read the task, binding rules, benchmark source, and existing workspace
   evidence before acting.
   Start later scripts by reviewing `assessment.history()` and record successful and
   failed world-moving attempts so an already-rejected approach is not repeated.
   Use `scene_view` when geometry or contact cannot be judged numerically, but a saved
   path is not visual evidence: transport/open the actual image pixels before making
   a visual claim.
2. **Establish a fresh baseline.** Run the smallest end-to-end attempt from a fresh build
   and capture measured outcomes.
   Every ordinary script starts fresh unless it explicitly restores a parent with
   `goto()`.
3. **Checkpoint every completed stage — and only those.**
   Record your stage plan in the tool (`tree.plan([...])`; one stage is a valid plan),
   write each stage as `solution/stages/stage_<k>.py` with `run(env)` and `check(env)`,
   and develop it with `tree.run_stage(k)`: it restores the previous boundary, runs the
   stage, runs your check, and saves this stage's boundary node when the check passes.
   Settle and verify before the boundary; do not save probes, partial motions or hopeful
   poses in between. Continue later stages from boundaries (`goto_stage`) instead of
   re-deriving the world; step back to an earlier boundary when a stage keeps failing.
4. **Declare the experiment origin.** Mark each experiment and its log as `origin=fresh`
   or `origin=checkpoint:<id>`; for the latter, call `goto(<id>)` explicitly after
   reset. Never assume a new process inherited the previous script's simulator state.
5. **Choose one optimization branch only when eligible.**
   - Strategy uncertainty (different control flow, grasp side, phase ordering, or
     recovery policy): use `sweep` to compare named strategies from the same origin.
   - Stable viable maneuver with genuinely numeric uncertainty (offset, depth, angle,
     gain, timing): use `parameter_search` to tune constants only after a viable seed
     exists. A seeded search protects only its reported in-search comparison; replay
     is still required before claiming the winner is no worse.
   If neither gate matches, keep diagnosing; tool availability alone is not a reason to
   launch optimization.
6. **Hold out 2–3 instances.** After a candidate works at its development origin, test it
   on two or three fresh/task-realistic instances not used to choose the strategy or
   constants. Do not tune on this holdout.
7. **Integrate.** Put the winning logic and constants into `/workspace/solution/`, rerun
   the integrated path, and preserve full logs. Record rejected branches as well as the
   winner.
8. **Queue fresh-reset verification.** The final end-to-end fresh-reset verifier remains
   required. Return control after integration; the harness runs it between legs once
   detached GPU/search work is clear. Checkpoints, sweeps, searches, and holdouts do not
   replace that official verification.
