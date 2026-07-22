"""Construction-agent prompt builder (stage 2 of PIPELINE.md)."""

from __future__ import annotations

from pathlib import Path

SIM_GEN_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE = """\
You are constructing ONE new MuJoCo simulation task for a robotics post-training \
benchmark, derived from a seed task but STRATEGICALLY DIFFERENT from it.

## The seed (read-only context — do not import or modify it)

Seed id: {seed_id}
Seed source file: {seed_path}

## Difficulty tier: {tier}

{tier_text}

## Your deliverable

A task package at {task_dir}/ with exactly three files, implemented against the
sim_gen.core framework (read core/scene.py, core/recorder.py, core/checks.py first):

1. scene.py — a registered BaseScene subclass with a SceneCfg dataclass. Implement:
   xml() (full MJCF), reset_instance(rng) (REAL randomization — different seeds must
   give different instances), success() (bool), score() (the RUBRIC, see below),
   describe() (the task statement a solving agent sees).

## The rubric (score())

Write a rubric that rewards partial progress toward the task in ways you judge
reasonable — the structure (event credits, stage prefixes, continuous shaping,
count fractions, combinations...) is your call. Hard invariants only:
- score in [0, 1], and score == 1.0 exactly when success() is True;
- the null policy (doing nothing) earns ~0;
- credit must not evaporate under correct behavior (latch transient achievements —
  e.g. a "picked up the part" credit shouldn't vanish when the part is correctly put
  down; remember latches in reset and state save/restore).
Also declare in TASK.md the rough number of stages of the task (used only as a
difficulty label).
2. smoke.py — the teleport-oracle solution (no robot) plus the test-case battery, using
   sim_gen.core.Checks and sim_gen.core.Recorder. Mirror the reference smoke's
   structure. The oracle MUST be exported as a module-level callable
   `oracle_solution(scene)` (the validator re-runs it under its own instrumentation).
   Required checks: settle/no-drift, no-NaN, determinism, randomization-is-real,
   null-policy-fails (success False AND score ~0), oracle-succeeds on >= 3 seeds,
   rubric monotonicity, and NEGATIVE CONTROLS (deliberately wrong executions that MUST
   fail), of which two are mandatory:
   - the SEED'S OWN STRATEGY executed in your scene must fail (if it is not even
     expressible in your scene, document the N/A in TASK.md);
   - your task may or may not require a specific execution order — both are fine
     designs; state which in TASK.md. Only if it does require an order: add a
     wrong-order execution as a negative control (it must fail);
   plus at least one more task-specific control, and a tolerance calibration probe.
   The smoke must print the Checks verdict (ALL PASS) and save a video (H.264, via
   Recorder.save). Success must be stable: once it triggers it must keep holding
   (the validator re-checks over a 300-step hold window).
3. TASK.md — task card: seed provenance, what you changed, WHY it is strategically
   different (see below), the intended strategy, and the check list.

## The reference example

{ref_dir}/ is a complete example of the contract (scene.py + smoke.py + TASK.md).
It is the seed's own port — your task must NOT be that.

## Hard requirements

- STRATEGICALLY DIFFERENT from the seed: a solver must need a different PLAN, not
  different parameters. The seed's solution strategy (e.g. "push the object toward a
  target region") must not solve your task, and near-miss variants of it must fail
  your success criterion. You choose the mutation freely — change the objective, the
  objects, the mechanism, add constraints, or combine — but same-strategy-different-
  numbers is REJECTED.
- Physics must be honest: objects interact through contacts; no teleporting in the
  scene logic (the oracle solution may use scene.carry(), which is the sanctioned
  kinematic-carry primitive). If your task depends on a physical property (mass,
  friction), VERIFY it applied by reading it back in the smoke.
- No robot embodiment at this stage. The oracle solution manipulates objects with
  scene.carry() / scene.set_body_pose() and settles with scene.settle().
- Self-contained: only mujoco + numpy + sim_gen.core. No new dependencies, no assets
  from disk (procedural geometry only).
- Iterate until the smoke prints "SIM_GEN_SMOKE: ALL PASS". Run it with:
  cd {cosigen_root} && MUJOCO_GL=osmesa {python} -m sim_gen.tasks.{task_name}.smoke \
      --video sim_gen/artifacts/{task_name}_smoke.mp4
  Watch out: if a check fails, fix the ROOT CAUSE (scene or check), never delete or
  weaken a check to pass.
- After the smoke passes, ALSO run the validator and fix root causes until it PASSes:
  cd {cosigen_root} && MUJOCO_GL=osmesa {python} sim_gen/pipeline/validate.py \
      --task {task_name}
  You own this problem until it is accepted — do not stop at the first green smoke.

## Working style

- Read sim_gen/core/*.py and the reference task before writing anything.
- Design first: write TASK.md (the what/why) before scene.py.
- Keep the code clean and self-documenting; comments explain non-obvious physics
  decisions only. This repo will be open-sourced as-is.
- Do not touch anything outside {task_dir}/ (and do not modify core/, the reference
  task, RoboVerse/, or other tasks).

When you are done, print a one-paragraph summary: what the task is, how it differs
strategically from the seed, and the final check count.
"""


TIER_TEXT = {
    "easy": "Target roughly 1-2 stages: a short, single-skill task. Rough guide, not "
            "a straitjacket — difficulty can also come from other axes.",
    "middle": "Target roughly 3-4 stages / a multi-step task with real sequencing or "
              "breadth. Rough guide, not a straitjacket.",
    "hard": "Target a long-horizon task (roughly 5+ stages, or comparably demanding "
            "on other axes: precision, hidden state, irreversibility). Rough guide, "
            "not a straitjacket.",
}


def build_prompt(seed_id: str, seed_path: Path, task_name: str, python: str,
                 tier: str = "easy") -> str:
    return TEMPLATE.format(
        seed_id=seed_id,
        tier=tier,
        tier_text=TIER_TEXT[tier],
        seed_path=seed_path,
        task_name=task_name,
        task_dir=f"sim_gen/tasks/{task_name}",
        ref_dir="sim_gen/tasks/push_cube_ref",
        cosigen_root=SIM_GEN_ROOT.parent,
        python=python,
    )
