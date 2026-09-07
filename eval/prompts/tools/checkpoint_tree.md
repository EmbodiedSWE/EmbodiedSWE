== Checkpoint tree (the checkpoint_tree tool) ==
You have a tool for saving a world state and coming back to it — including from a LATER
script, in a fresh process. Every script boots from the task's initial condition (node `n0`);
restoration happens only when that script explicitly calls `goto()`, `goto_stage()` or
`run_stage()`. Declare each experiment as `origin=fresh` or `origin=checkpoint:<id>`.

The tool is built around your stage plan. Work in this shape:

    from checkpoint_tree import CheckpointTree
    tree = CheckpointTree(env)                 # immediately after env.reset(); reconciles n0
    tree.attach_viewer(viewer)                 # optional: every save also captures a snapshot

    # 1. plan — one entry per stage, naming the STATE that completes it. Keep it to a few
    #    real seams (2–4 is typical); ONE stage is a valid plan when the task has no clean
    #    seam or is easier to debug as a whole.
    tree.plan(["card seated in its slot", "first RAM stick seated", "both RAM sticks seated"])

    # 2. one module per stage: /workspace/solution/stages/stage_<k>.py with
    #        def run(env): ...      # from the previous boundary state to this stage's goal
    #        def check(env): ...    # your own verification of that goal -> bool
    #    solve.py is their composition: for k in 1..N: stage_k.run(env)

    # 3. develop a stage from the previous boundary; its boundary is saved when check passes
    r = tree.run_stage(2)                      # goto(latest stage-1 boundary) -> run -> check -> save
    r = tree.run_stage(2, from_node="n4")      # or continue from a specific stage-1 node
    r = tree.run_stage(1, run=my_fn, check=my_check, program=__file__)   # callables instead of a module

    # 4. navigate the plan
    tree.goto_stage(1)                         # restore the latest boundary of stage 1
    tree.goto("n4")                            # or any node; returns what was already tried from it
    print(tree.show())                         # plan progress (✓/○ per stage, boundary nodes) + the tree
    print(tree.tried_from("n4"))               # every branch tried from n4 and how each ended

    # bookkeeping. save() is ONLY for a stage boundary reached outside run_stage (pass stage=k);
    # a state that completes no stage is never saved — record it as an attempt instead.
    tree.save("card seated in its slot", stage=1, action="pressed 6 mm after alignment", program=__file__, log=out)
    tree.record_attempt("press 5 mm deeper", "failed", note="part tipped at t=90", program=__file__, log=out)
    tree.annotate("n2", scene_diff="card now flush with the slot; gripper clear")

How to work with it:

  * A stage boundary is saved at the END OF EVERY COMPLETED STAGE — that is the rule, not
    an exception. Before `run(env)` returns, hold still long enough for the world to settle
    and for your `check(env)` to measure the subgoal; then `run_stage` saves the boundary.
    If `check` returns False nothing is saved: the run is recorded as an attempt and you are
    told why. A stage may end up with several boundary nodes (different ways of reaching the
    same subgoal); `goto_stage(k, cid=...)` picks one.
  * Between boundaries, do not save: diagnostic probes, partial motions and hopeful poses
    are not stages. Record failed stage attempts (`run_stage` does it for you; `record_attempt`
    or `with tree.attempt(...)` for anything else) so `tried_from()` stops you from repeating
    them.
  * Every save prints a health block (world still moving? robot joint at a limit? scene
    success flag). It is information: a moving or limit-pinned state is a fragile place to
    continue from. `save(..., require="reject")` refuses an UNSAFE state.
  * Read `tried_from()` before re-attempting a stage from the same boundary. Two attempts that
    fail with the same state change mean the approach is wrong, not under-tuned: go back to
    the previous boundary and change approach, or re-plan (`tree.plan([...])` again — existing
    boundaries keep their stage numbers).
  * Pass `program=` (run_stage does) so the node records exactly the code that reached it;
    `/workspace/.checkpoints/<cid>.code.py` and `.log.txt` hold that program and its full output.
  * With a viewer attached, look at the parent's and the node's snapshots after a save and
    write down what changed: `tree.annotate(cid, scene_diff='...')`.

Checkpoints accelerate development; they do not prove the integrated solution. The final
`solve.py` — the stages composed, run from a fresh reset — must still pass the queued verifier.
