# Rule: autonomous operation

You are running autonomously. No one reads your messages or answers questions
during this session — your final message ends the session, and so does the
session budget stated at the top of this brief (a hard kill at the stated
time; the orchestrator then grades whatever is on disk).

- Stopping — replying without taking further actions — ends the session for
  good, not pauses it. Never stop to "wait" for a watcher, a background job,
  or a reply that will never come.
- Poll long-running work (a `generate` batch) to completion before you stop;
  background processes die with the session, and anything unfinished is lost.
- When something fails, diagnose and retry yourself — there is no one to ask.
  If a direction is truly stuck, write what you tried and why it failed into
  the summary, and move to your next idea.
- Budget your time: verification runs are the slow part. Author while batches
  run; leave untested cells clearly marked UNTESTED rather than half-done.
