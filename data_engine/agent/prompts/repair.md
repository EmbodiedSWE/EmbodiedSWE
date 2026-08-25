# Repair session — make the delivered solve pass its own grader

You are inside a **data_gen campaign** that multiplies one verified robobench
solve into a large demonstration dataset. Before multiplying, the pipeline runs
the delivered solve nominally (unmodified scene, no sampling) and checks it with
the task's own grader. That check is currently FAILING, and your session's one
job is to fix the solve so a nominal run succeeds.

## Facts

- campaign root (your cwd, writable): `{gen}`
- the solve to repair (edit IN PLACE — this is the base every future cell
  copies): `{solve}`
- the judge (read it to understand what success measures; do NOT weaken it to
  make the check pass — a judge that lies produces worthless data): `{grader}`
- a GPU is available; `python` has Isaac Sim + Isaac Lab.

## Failing-run evidence

```
{fail_log}
```

## How to work

1. Read the grader first: know exactly what predicate must hold at episode end.
2. Read the solve and the complete failing log. Diagnose the first divergence
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
5. If you have spare time after nominal passes, ALSO verify it survives
   parallel envs (`--num_envs {num_envs}`); the dedicated vectorize stage will
   enforce this before scripted farming and compounding.

You are running autonomously: no one answers questions; your final message ends
the session. Diagnose and retry yourself. Leave a short `REPAIR_NOTES.md` next
to the solve: what was broken, what you changed, what you measured.
