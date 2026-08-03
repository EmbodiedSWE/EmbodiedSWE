== Assessing runs (the assessment tool) ==
After every run that moved the world, write down what actually happened BEFORE editing code:

    from assessment import assess, history

    assess(scene="part_0 in place; part_1 tipped over next to its goal",
           log="goal metric 3.1mm of 5mm; gripper lost contact at t=210",
           failure_modes="grasp slips mid-transfer — approach angle too steep",
           keep=True, label="part_0 in place", tree=tree)

  * `scene` is what the images show (use scene_view), `log` is what the numbers say,
    `failure_modes` is your diagnosis — all three in words; empty prose is refused.
  * `keep=True` marks the end state worth building on; pass your CheckpointTree as `tree` and
    the state is saved in the same call, labeled with `label`.
  * When the outcome rests on constants you picked by hand, set `constants_plan` to
    'searching' (and hand them to parameter_search) or to the reason you are not.

Records persist in `/workspace/.assessments/reviews.jsonl` across all your scripts. START each
new script with `print(history())` — it is your memory of what was already tried, what worked,
and what you already diagnosed. Re-attempting a failure without reading it is how the same
mistake gets made three times.
