---
name: cosigen-checkpoint-tree
description: Uses the checkpoint_tree tool to save working stages as annotated nodes, inspect prior attempts, branch from good states, and recover from failed stages. Use when deciding what to checkpoint and where to continue from.
---

# Checkpoint tree

The tree is separate from your code and your notes. `tree.goto(cid)` changes only the
simulator world; your knowledge and workspace files remain available. It is disk-backed
(`/workspace/.checkpoints`), so it survives every one of your scripts — a state reached in
one process can be restored in the next.

- Checkpoint a working stage when you want to build future work off it:
  `tree.save('<state reached>', action='<what you attempted and how it ended>',
  program=__file__, log=<its printed output>)`. Passing `program=` makes the node's edge
  exactly the code that reached it; a later script can reread the winning program instead of
  reconstructing it.
- Label the STATE in your task's own terms ("component upright, regrasped top-down", "cloth
  edge aligned"), not the action — the label is what you will read in the tree later.
- `print(tree.show())` to inspect the tree and locate yourself. Each node shows its action,
  an automatically computed `state diff` versus its parent (what actually moved, by how
  much), any `scene diff` you recorded, and its snapshot path.
- Before going somewhere, look first: read the node's entry in `show()`, open its snapshot
  PNG, and `tree.get_log(cid)` for its full printed log. Captions are one line — this is how
  you find out whether the node actually holds the state you remember. Then `tree.goto(cid)`.
- `tree.goto()` returns `tried_from()` — every branch already attempted from that node with
  its action, state diff and code. Read it: two branches failing with the same state diff
  means the approach is wrong, not under-tuned.
- With a `scene_view.Viewer` attached (`tree.attach_viewer(viewer)`), every save captures a
  snapshot; after looking at the parent's and the node's snapshots, record what changed
  visually: `tree.annotate(cid, scene_diff='...')`.
- Improving a stage is the same move as backtracking: read a node's code
  (`/workspace/.checkpoints/<cid>.code.py`), go to its parent, and save a better version as
  a sibling.

Save states you would hate to re-derive. A crash costs you nothing that was checkpointed.
