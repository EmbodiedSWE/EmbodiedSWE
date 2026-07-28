---
name: cosigen-checkpoint-tree
description: Uses CoSiGen's checkpoint tree to save working stages as nodes, inspect prior attempts, branch from good states, and recover from failed stages. Use when deciding what to checkpoint and where to continue from.
---

# CoSiGen checkpoint tree

The tree is separate from your conversation and variables. `goto(cid)` changes only the
simulator world; your knowledge and workspace files remain available.

- Every program you run starts from the node you are on, and the world returns there
  when it finishes. Trying things never spoils the state you work from.
- Checkpoint a working stage when you want to save progress so that you can build your
  future work off from there: `checkpoint(label='<short description of the reached
  state>', path='/workspace/<your_program>.py')`. The program runs from your current
  node and its end state becomes a new node whose code is that program. A program that
  raises is not checkpointed.
- Use `list_checkpoints()` to inspect the tree (each node shows its parent) and locate
  yourself.
- Going back is a two-step tool call: `goto(node='n3')` shows you that node's image, its
  log and the program that reached it, and the world only moves when you call again with
  `confirm=True`. Captions are one line, so this is where you find out whether the node
  actually holds the state you remember.
- Inside a program you can still read a node without going there:
  `get_checkpoint_scene(cid)` for the oracle state (scene text + poses),
  `get_checkpoint_log(cid)` for the full printed log, `get_checkpoint_code(cid)` for the
  program itself.
- Improving a stage is the same move as backtracking: read a node's code, go to its
  parent, and checkpoint a better version as a sibling.

Simulator step count is monotonic and is not refunded by `goto`.
