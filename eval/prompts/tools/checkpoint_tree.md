== Checkpoint tree (the checkpoint_tree tool) ==
You have a tool for saving a world state and coming back to it — including from a LATER
script, in a fresh process. Use it: every script boots the simulator from the task's initial
condition, so without it a state that took twenty minutes to reach must be re-derived.

    from checkpoint_tree import CheckpointTree

    tree = CheckpointTree(env)            # after env.reset(); reopens an existing tree
    tree.attach_viewer(viewer)            # optional: every save also captures a snapshot
    tree.save("part_0 secured in its mount",   # -> 'n3'; label = the STATE reached
              action="approached from +x, contact-first, then re-gripped",  # what you attempted
              note="goal error 2.1mm; grip width 8.2mm",                    # measurements
              program=__file__,           # the script that produced this state
              log=my_captured_output)     # its printed output, kept in full
    tree.goto("n1")                       # restore that state and make it current
    print(tree.show())                    # the whole annotated tree, current node marked
    print(tree.tried_from("n1"))          # every branch tried from n1 and how each ended

Each node carries three annotations. `action` (yours) says what was attempted; `state_diff`
is COMPUTED automatically — object movements and joint deltas versus the parent, so what a
branch actually changed is never missing; `scene_diff` is yours to fill after LOOKING at the
parent's and the node's snapshots:

    tree.annotate("n3", scene_diff="part_0 now flush with its mount; gripper clear")

How to work with it:

  * Save whenever a stage lands — a grasp that finally holds, an aligned part, a completed
    sub-goal. Label the STATE reached in your task's own terms ("part_0 secured", "cloth
    folded over the crease", "container half filled"), not the action attempted.
  * Pass `program=` so the node records exactly the code that reached it — a later you (or a
    later script) can reread the winning program instead of reconstructing it.
  * Branch instead of gambling: before a change that could ruin a good state, save; if the
    variation is worse, `goto()` the parent and branch again. Nothing earned is lost.
  * Read `tried_from()` before re-attempting anything. Two branches failing with the same
    state_diff means the approach is wrong, not under-tuned.
  * Nodes persist under /workspace/.checkpoints across all your scripts.

State is whatever `env.get_states()` returns, restored with `env.set_states()`. This
restores state for YOUR exploration; it is not a way to place objects into a goal
configuration — the task must still be solved by physical manipulation, and the graded run
blocks state writes entirely.
