# Rule: autonomous operation

You are running autonomously. No one reads your messages or answers questions
during this session — your final message ends the session, and the container
exits with it.

- Stopping — replying without taking further actions — ends the session for
  good, not pauses it. Never stop to "wait" for a watcher, a background job,
  or a reply that will never come.
- Poll long-running commands to completion before you stop; background
  processes die with the session, and anything unfinished is lost.
- `submit` does NOT end the session — it snapshots your solution and returns
  immediately. Run it at every measurable improvement: partial progress earns
  partial credit, and unsubmitted progress earns nothing.
- Before your final message, leave the best solution you have in `solution/`
  and make sure it at least runs — an imperfect solution that makes real
  progress scores; one that crashes or was never submitted does not.
