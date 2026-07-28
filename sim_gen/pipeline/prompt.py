"""Construction-agent prompt builder (Isaac/robobench format; stage 2 of PIPELINE.md)."""

from __future__ import annotations

from pathlib import Path

SIM_GEN_ROOT = Path(__file__).resolve().parent.parent

TIER_TEXT = {
    "easy": "Target roughly 1-2 stages: a short, single-skill task. Rough guide, not "
            "a straitjacket — difficulty can also come from other axes.",
    "medium": "Target roughly 3-4 stages / a multi-step task with real sequencing or "
              "breadth. Rough guide, not a straitjacket.",
    "hard": "Target a long-horizon task (roughly 5+ stages, or comparably demanding "
            "on other axes: precision, hidden state, irreversibility). Rough guide, "
            "not a straitjacket.",
}


ISAAC_TEMPLATE = """\
You are constructing ONE new Isaac Lab simulation task for a robotics post-training
benchmark, derived from a seed task but STRATEGICALLY DIFFERENT from it. The task is
written in the robobench format (the house framework in {cosigen_root}/robobench) and
is tested remotely on your dedicated GPU "forge" server.

## The seed (read-only context — do not import or modify it)

Seed id: {seed_id}
Seed source file: {seed_path}

## Difficulty tier: {tier}

{tier_text}

## Your deliverable

A task package at {task_dir}/ with exactly three files:

1. scene.py — a robobench-format scene module (STUDY THE EXEMPLAR FIRST:
   {cosigen_root}/robobench/suites/packing/scenes/pen_holder.py — your scene must be
   the same KIND of code, but may be lighter):
   - a `SceneCfg` dataclass holding every tunable init parameter (sizes, counts,
     spawn regions, tolerances) with meaningful randomization;
   - a `BaseScene` subclass (from robobench.core) implementing assets() /
     reset(env_ids) / get_state / set_state / describe(), plus success() and a graded
     score() (rubric: partial progress rewarded reasonably; 1.0 iff success; ~0 for
     doing nothing; latch transient achievements);
   - register with `SCENES.register("<name>")` and `register_env` with robot="null"
     (scene-level task; robot bindings are a later stage).
2. smoke.py — the teleport-oracle test battery, runnable STANDALONE on the forge
   (mimic {cosigen_root}/robobench/suites/packing/smokes/pen_holder_smoke.py's
   structure: AppLauncher --headless boot, enable_cameras + the RTX driver-check
   override kit_arg, build the env, drive objects through scene handles):
   - export `oracle_solution(scene_or_env)`; oracle must reach success() on >= 3 seeds;
   - named checks (print PASS/FAIL per check): settle/no-NaN, randomization-is-real,
     null-policy-fails (score ~0 when nothing acts), rubric monotonicity, >= 2 negative
     controls (the seed's own strategy must fail if expressible — else document N/A in
     TASK.md; plus a near-miss/tolerance control), and a calibration probe;
   - record video frames and save `frames.npz` in the CURRENT WORKING DIRECTORY
     (np.savez_compressed("frames.npz", frames=...)) so the pipeline can fetch it;
   - print exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes (this
     marker is the acceptance signal), and end with a hard exit (os._exit(0) after a
     10s watchdog Timer) — Kit teardown hangs otherwise.
3. TASK.md — task card: seed provenance, what you changed, WHY strategically
   different, the declared stage count (difficulty label), whether execution order is
   required, and the check list.

## Testing on your forge (your ONLY way to run Isaac; there is no local GPU)

  # upload the package (repeat after every edit), then run the smoke:
  python {cosigen_root}/sim_gen/pipeline/forge_client.py --forge-url {forge_url} \
      submit --task {task_name} --dir {task_dir}
  python {cosigen_root}/sim_gen/pipeline/forge_client.py --forge-url {forge_url} \
      run --task {task_name} --module smoke --timeout 1500

Each run boots Isaac fresh (~3-4 min) then runs your smoke — budget roughly 10
iterations, so make each one count: read the full stdout tail, fix ROOT CAUSES, never
delete or weaken a check to pass. Common Isaac traps (all solved in the exemplar and
{cosigen_root}/.cursor/rules/cosigen-suite-conventions.mdc): duplicate xformOp on
cloned prims (author decorations idempotently), contact offsets eating your clearance,
verifying randomization by READBACK, settle before judging.

## Hard requirements

- STRATEGICALLY DIFFERENT from the seed: a solver must need a different PLAN, not
  different parameters. Same-strategy-different-numbers is REJECTED (an LLM judge
  compares your TASK.md + code against the seed source).
- Tasks may require a specific execution order or not — both fine; declare in TASK.md.
- Self-contained: procedural geometry only (primitives/compound spawners like the
  exemplar); no external asset files.
- Physics must be honest: the oracle may place objects kinematically (teleport-oracle),
  but success()/score() must judge PHYSICAL outcomes (settled poses, real containment).
- Write ONLY inside {task_dir}. Do not modify robobench/, other tasks, or the pipeline.

When done (smoke ALL PASS on the forge), print a one-paragraph summary: the task, why
it is strategically different, the declared tier/stages, and the final check count.
"""


def build_isaac_prompt(seed_id: str, seed_path: Path, task_name: str, task_dir: str,
                       forge_url: str, tier: str = "easy") -> str:
    from pathlib import Path as _P
    return ISAAC_TEMPLATE.format(
        seed_id=seed_id, seed_path=seed_path, task_name=task_name, task_dir=task_dir,
        forge_url=forge_url, tier=tier, tier_text=TIER_TEXT[tier],
        cosigen_root=_P(__file__).resolve().parent.parent.parent)
