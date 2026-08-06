"""Construction-agent prompt builder (Isaac/robobench format; stage 2 of PIPELINE.md)."""

from __future__ import annotations

from pathlib import Path

SIM_GEN_ROOT = Path(__file__).resolve().parent.parent


ISAAC_TEMPLATE = """\
You are constructing ONE new Isaac Lab simulation task for a robotics post-training
benchmark, derived from a seed task but STRATEGICALLY DIFFERENT from it — and you must
write a TELEPORT SOLUTION that demonstrably solves it in this same session. A task
without a working teleport solution is not accepted. The task is written in the
robobench format (the house framework in {cosigen_root}/robobench) and is tested
remotely on your dedicated GPU "forge" server. When testing, read the full stdout
tail, fix root causes, and never delete or weaken a check to pass.

## The seed (read-only context — do not import or modify it)

Seed id: {seed_id}
Seed source file: {seed_path}

## The embodiment (design for it from the start)

Your teleport solution certifies the task's interactions are physically achievable in
the scene — but the task will ultimately be solved by a SINGLE FRANKA ARM with a
parallel-jaw gripper, driven through the robobench OSC controller. Designing something
the arm cannot do makes the task worthless, so think the embodiment through while
designing, and record the intended arm strategy in TASK.md:
- for every object the robot must move, know the intended contact strategy up front:
  a graspable feature that actually fits the jaw with room for the hand to approach,
  or a face it can push;
- keep required precision within what closed-loop arm control can reliably hit;
  tolerances near the arm's control noise turn a sound design into a lottery;
- watch clearances: contacts very near the ground, under low overhangs, or through
  apertures barely larger than the object are where solutions die;
- keep the action within comfortable reach of one plausible base pose (state it in
  TASK.md).
Mechanisms (interlocks, counterweights, ordered fixtures) are welcome — the
requirement is that every contact the task REQUIRES is one the arm can actually make.

## Order of work (the rubric comes AFTER the solution)

1. Design the scene; give it a MINIMAL goal predicate (success()) so you can iterate.
2. Write solve.py — the teleport solution — and iterate on the forge until the goal
   state is physically reached: teleport handles TRANSPORT only, every load-bearing
   interaction goes through contact dynamics (e.g. to thread a nut onto a bolt:
   teleport the nut to just above the bolt, then press and twist it down the thread
   with applied forces until the scene reports success — never spawn it seated).
3. Only then write the final rubric: success() plus a graded score() anchored in your
   demonstrated solution — latch the stages the solution actually passes through,
   score ~0 for the null policy, 1.0 iff success(), credit that does not evaporate
   under correct behavior.
4. Write smoke.py (rubric battery) and re-run both modules clean on the forge.

## Your deliverable (four files at {task_dir}/)

1. scene.py — robobench scene module (exemplar of the format:
   {cosigen_root}/robobench/suites/packing/scenes/pen_holder.py — same KIND of code;
   your content should NOT mirror the exemplar's design):
   - a SceneCfg dataclass holding every tunable init parameter, with real
     randomization (different seeds -> different instances);
   - a BaseScene subclass implementing assets() / reset(env_ids) / get_state /
     set_state / describe(), plus success() and score();
   - describe() is the statement a solving agent receives: goal state, how every
     target is identified visually, and any ordering constraints — complete enough
     that a competent solver could do the task from describe() alone;
   - instruction() is the SHORT form of the same task for VLA training: one or two
     imperative sentences, under 200 tokens, stating the goal and any constraint
     whose violation fails the task — nothing else;
   - register with SCENES.register("<name>") and register_env with robot="null"
     (scene-level; solve.py and smoke.py build the same scene-level env).
2. solve.py — the TELEPORT SOLUTION, and the task's legitimacy certificate:
   - standalone, AppLauncher-style entry point (must run as
     `python -m simgen_tasks.<task>.solve --headless` — the orchestrator re-runs it
     exactly that way);
   - builds the scene-level env (robot="null" — no arm; the arm strategy lives in
     TASK.md as the embodiment argument);
   - teleportation handles TRANSPORT ONLY: setting an object's pose to move it across
     free space is fine; every LOAD-BEARING interaction the task requires (insertion,
     threading, pressing, latching, sliding under contact...) must be executed through
     the simulator's contact dynamics — applied forces/torques are the tool there.
     Never teleport an object into a state that bypasses the interaction (a nut is
     teleported to just above the bolt, then pressed and twisted down the thread with
     applied forces until the scene reports success — never spawned seated);
   - reaches the goal state with everything settled, prints the scene's own readouts,
     prints `SIM_GEN_SCORE <score()>` at each phase boundary (acceptance checks these
     never decrease — latched credit must not evaporate along the solution
     trajectory); after success() first turns True, KEEP SIMULATING for at least 3
     more simulated seconds with no further intervention and only if success() still
     holds print exactly `SIM_GEN_SOLVE: SUCCESS` (this marker is the acceptance
     signal; the persistence window is what rejects fly-through successes); hard exit
     after the verdict (os._exit after a watchdog Timer — Kit teardown hangs
     otherwise);
   - must pass on at least 2 seeds in your own testing before you finish.
3. smoke.py — REJECTION TESTS for your rubric (rejection only: your solve.py already
   proves the rubric ACCEPTS a correct outcome; smoke.py proves it REJECTS wrong
   ones). For every outcome your rubric claims to reject, CONSTRUCT that outcome as a
   settled state (teleport objects, settle, evaluate) and assert rejection, as named
   PASS/FAIL checks:
   - the end state the SEED's strategy would produce -> success() False (if
     expressible in your scene; else document N/A in TASK.md);
   - a settled near-miss just outside each load-bearing tolerance -> success() False;
   - if you declare a required execution order: an out-of-order end state -> fails;
   - every other wrong outcome your TASK.md claims is rejected (wrong object, wrong
     place, ...);
   plus generic sanity: null policy (nothing acts) scores ~0; randomization differs
   across seeds, verified by READBACK; settle/no-NaN;
   - record video frames and save frames.npz in the CURRENT WORKING DIRECTORY;
   - print exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard
     exit as above.
4. TASK.md — task card: seed provenance, what you changed, WHY strategically
   different, the teleport-solution outline (phases), the EMBODIMENT ARGUMENT (per
   manipulated object: the intended Franka contact strategy; plus one plausible base
   pose), whether execution order is required, and the check list.

## Testing on your forge (your ONLY way to run Isaac; there is no local GPU)

  # upload the package (repeat after every edit):
  python {cosigen_root}/sim_gen/pipeline/forge_client.py --forge-url {forge_url} \\
      submit --task {task_name} --dir {task_dir}
  # run a module (smoke or solve):
  python {cosigen_root}/sim_gen/pipeline/forge_client.py --forge-url {forge_url} \\
      run --task {task_name} --module solve --timeout 1500

Common Isaac traps (all solved in the exemplar and
{cosigen_root}/.cursor/rules/cosigen-suite-conventions.mdc): duplicate xformOp on
cloned prims (author decorations idempotently), contact offsets eating your clearance,
verifying randomization by readback, settle before judging.

## Hard requirements

- STRATEGICALLY DIFFERENT from the seed: a solver must need a different PLAN and a
  different code structure, not different parameters or minor changes.
  Same-strategy-different-numbers is REJECTED (an LLM judge compares your TASK.md +
  code against the seed source).
- Honest teleport solution: teleports move objects, they never do the task. The
  load-bearing interactions must go through contact dynamics, and success()/score()
  judge PHYSICAL outcomes (settled poses, real containment) — a solution that reaches
  success() by teleporting past the required interaction, pinning objects against
  physics, or exploiting a rubric loophole is rejected. An LLM judge reviews your
  solve.py for exactly this.
- describe() is judged for clarity: a competent solver must be able to do the task
  from describe() alone.
- Tasks may require a specific execution order or not — both fine; declare in TASK.md.
- Self-contained: procedural geometry only; no external asset files.
- Write ONLY inside {task_dir}. Do not modify robobench/, other tasks, or the pipeline.

When done (smoke ALL PASS and solve SUCCESS on the forge), print a one-paragraph
summary: the task, why it is strategically different, the solution outline, and the
final check count.
"""


def build_isaac_prompt(seed_id: str, seed_path: Path, task_name: str, task_dir: str,
                       forge_url: str) -> str:
    from pathlib import Path as _P
    return ISAAC_TEMPLATE.format(
        seed_id=seed_id, seed_path=seed_path, task_name=task_name, task_dir=task_dir,
        forge_url=forge_url,
        cosigen_root=_P(__file__).resolve().parent.parent.parent)
