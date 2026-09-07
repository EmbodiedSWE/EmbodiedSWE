== Assessing runs (the assessment tool) ==
After every run that moved the world — successful or failed — write down what actually happened
BEFORE editing code:

    from assessment import assess, history

    assess(scene="part_0 in place; part_1 tipped over next to its goal",
           log="goal metric 3.1mm of 5mm; gripper lost contact at t=210",
           failure_modes="grasp slips mid-transfer — approach angle too steep",
           keep=True, label="part_0 in place", tree=tree,
           action="transfer part_0", outcome="partial", program=__file__)

  * `scene` is what transported/opened image pixels show (use scene_view when granted), `log`
    is what the numbers say, and `failure_modes` is your diagnosis — all three in words; empty
    prose is refused. A filepath alone is not visual evidence; if pixels were not actually
    transported to an image-capable viewer, say that visual assessment is unavailable.
  * `keep=True` marks an end state worth building on. When checkpoint_tree is granted, pass its
    tree only after the state is settled, portable, and checkpoint-eligible; it is then saved in
    the same call, labeled with `label`. If no tree object is available, pass `env=env`; assessment
    stores the state as an independent root rather than inventing a parent. With neither, a kept
    assessment raises instead of silently losing the state.
  * Pass `outcome="failed"` or `"crashed"` and the live `tree`/`env` for unsuccessful runs.
    Assessment records a non-reusable attempt with complete `program` and `log`, so
    `tried_from()` remembers it without treating the failed end state as a checkpoint.
  * When the outcome rests on constants you picked by hand and parameter_search is granted, set
    `constants_plan` to 'searching' only when a viable stable maneuver exists and you are
    committed to handing it those constants. Otherwise record the concrete reason you are not,
    including that numeric search is unavailable when it was not granted.

Records persist in `/workspace/.assessments/reviews.jsonl` across all your scripts. START each
new script with `print(history())` — it is your memory of what was already tried, what worked,
and what you already diagnosed. Re-attempting a failure without reading it is how the same
mistake gets made three times.

Assessment is memory, not verification. After integrating the winning logic, the complete
solution must still pass the queued harness verifier from a fresh reset.
