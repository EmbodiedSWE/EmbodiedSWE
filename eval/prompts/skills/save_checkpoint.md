# Skill: save checkpoints as you go

Saving a reached state costs one call and buys back every minute that produced it. Use it.

- **Checkpoint whenever a stage lands.** A grasp that finally holds, an aligned part, a first
  successful insertion — save it. The state you do not save is the state you re-derive.
- **Branch instead of gambling.** Before trying something that could ruin a good state, you are
  already safe: a program runs from the current node and the world returns there afterwards. So
  try the risky version, and only keep it if it is better.
- **Go back rather than push on.** When two runs in a row fail the same way, the branch is wrong,
  not under-tuned. Return to the last good node and change approach.
- **Read a node before you land on it.** Its image, log and program tell you what it holds, and
  the "previously tried from here" digest tells you what already failed from there.

A long task is a sequence of kept states, not one heroic program.
