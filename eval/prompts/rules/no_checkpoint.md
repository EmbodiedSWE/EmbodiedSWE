# Rule: no checkpoints in this experiment

Saved world states are not available to you here. `env.set_states` raises, and there is no
checkpoint tree to save to or return to.

- Solve the task from the initial condition `reset()` establishes, stepping the simulation
  forward. There is no jumping to a saved mid-task state.
- Reading state is unaffected: `get_states`, the asset handles and every measurement you print
  remain available.
- Because a bad outcome cannot be rewound, prefer programs that verify before they commit —
  check a grasp actually holds before carrying, check alignment before pressing — and write
  recovery into the program itself rather than relying on restoring an earlier state.
