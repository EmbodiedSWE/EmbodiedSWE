# Rule: autonomous operation

You are running autonomously. No one reads your messages or answers questions
during this session — your final message ends the session, and the container
exits with it.

- Ending your turn is submitting, not pausing. Never stop to "wait" for a
  watcher, a background job, or a reply that will never come.
- Poll long-running commands to completion within your turn; background
  processes die with the session, and anything unfinished is lost.
- Before your final message, make sure the deliverable is complete and
  actually works — a half-finished solution scores as a failure.
