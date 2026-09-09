# Repair session — make the delivered solve pass its own grader

You are inside a **data_gen campaign** that multiplies one verified robobench
solve into a large demonstration dataset. Before diversifying, the pipeline runs
the delivered solve nominally (unmodified scene, no sampling) and checks it with
the task's own grader, then REPLAYS the successful episode: its recorded robot
actions are fed back open-loop into a rebuilt world and must reproduce the
success (an episode whose actions do not explain its success is not training
data). That check is currently FAILING, and your session's one job is to fix
the solve so a nominal run succeeds AND replays.

**This is a hard gate. The campaign does not move on until it passes.** If
your session ends with the check still failing, the next repair session starts
from your edits, until the pre-check clock ({hours_left} h left) is exhausted —
then the whole campaign ends as PRECHECK_FAILED. Do not stop at a diagnosis:
diagnose, change the solve, re-run the check, repeat.

Provenance of the delivered solve: the eval run it came from ended with status
`{source_status}` ("completed" = it solved the task there; "timeout" /
"unreachable" = it never did, and you may be finishing an unfinished solve
rather than fixing a regression — read it as such). Alternate solves from the
same run, if any, are here for reference (read, do not copy blindly; some are
earlier partial stages):
{candidates}

If a `REPAIR_NOTES.md` sits next to the solve, read its top section first — the
previous session's current state, what it tried, and what it planned next. Keep
that file SHORT: rewrite the top section ("State / Tried / Next", under 60
lines) each session instead of appending; details go below a `---` line.

{replay_note}

## Facts

- campaign root (your cwd, writable): `{gen}`
- the solve to repair (edit IN PLACE — this is the base every future cell
  copies): `{solve}`
- the judge (read it to understand what success measures; do NOT weaken it to
  make the check pass — a judge that lies produces worthless data): `{grader}`
- a GPU is available; `python` has Isaac Sim + Isaac Lab.

## Failing-run evidence

The COMPLETE failing-probe log ({fail_log_size}) is at:

    {fail_log_path}

Read it with your shell (grep/tail/sed — it is the full Isaac run, so search it
rather than dumping it whole).

## How to work

1. Read the grader first: know exactly what predicate must hold at episode end.
2. Read the solve and the failing log. Diagnose the first divergence
   from the grader's required final state; do not infer a fix from the final
   score alone.
3. Edit `solve.py` in place. Prefer the smallest behavioral correction that
   fixes the diagnosed cause. Never touch sim/PhysX settings, and never edit
   the grader to be more permissive.
4. Test with the real check:

       generate --headless {gen} --scene scene_0 --strategy strategy_0 --nominal --num_envs 1 --seed 0

   Each run writes a batch under `{gen}/data/` — its `meta.json` records
   success/fail per episode. The orchestrator probes seeds {nominal_seeds} and
   accepts a solution once one probe passes. Test all listed seeds when time
   allows and report the full yield; a multi-seed result is stronger evidence
   than a single pass.
5. Do not spend this session on `--num_envs {num_envs}` runs: the vectorize
   gate that follows enforces width with its own sessions; here the nominal
   run and its replay are the whole job.

You are running autonomously: no one answers questions; your final message ends
the session. Diagnose and retry yourself, and keep going until the check
passes or your session budget is spent. Leave `REPAIR_NOTES.md` next
to the solve with a rewritten top section: current state, what was broken, what
you changed and measured, and — if the check still fails — exactly where the run
diverges and what to try next, so the following session does not start from zero.
