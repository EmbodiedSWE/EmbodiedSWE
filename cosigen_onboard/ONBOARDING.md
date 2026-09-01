# CoSiGen — Onboarding

Welcome! This is a quick map of the repo so you know where everything lives. For install/setup, follow the root `README.md` (uv venv + pip-installed Isaac Lab 5.1, no source clone needed).

## Repo map

```
robobench/            the core benchmark package
  core/               base abstractions: BaseEnv, BaseScene, BaseRobot, controller/grader
                      contracts, registries (env = scene + robot, wired by injection)
  suites/             task suites (assembly, articulated, folding, packing, pouring,
                      puzzle, tool_use, ...) — each has scenes/, assets/, grader/, smokes/
  robots/             robot embodiments (franka, xarm7, g1, piper, wxai, jaco2, ...)
  controllers/        pluggable controllers (joint, task_space/OSC, pink_ik, composite)
  scripts/smoke.py    generic entry point: build + step any registered env
eval/                 the rigorous evaluation harness (dockerized) — see eval/README.html
data_engine/          the data-generation engine — see data_engine/README.html
experiments/          history of solve attempts/sessions — good reference material
scripts/record_video.py   wrap any script headless and record an mp4
```

## The one equation

```
env = scene + robot (+ controller chosen by the robot's control_mode)
```

Runnable envs are **registered by name** in `configs/envs.py` under each suite
(e.g. `robobench/suites/assembly/configs/envs.py`, via `register_env`). Names follow
`suite.scene[.robot[.mode]]`, e.g. `assembly.bulb.franka.osc`.

```bash
python -m robobench.scripts.smoke --list                     # list registered envs (no sim launch)
python -m robobench.scripts.smoke --env assembly.bulb --livestream 2
python -m robobench.scripts.smoke --scene bulb --robot xarm7 --mode osc --headless   # ad-hoc combo
```

`robobench/README.md` is the design doc — read it once, it explains the philosophy behind all of this.

## Quick start (interactive)

The easy way in: open the repo in **Claude Code**, point it at `robobench/README.md` and a suite,
and ask it to solve one of the registered envs — e.g. "solve `assembly.bulb.franka.osc`" (the bulb
task is the most reliable one, a good first target). That's the core workflow of the project:
coding agents writing solver scripts against these envs. For phrasing ideas, the prompt library the
eval harness uses lives in `eval/prompts/` (task instructions contract, rules, hints, skills, and a
worked solver example under `examples/`).

**Example: `experiments/bulb_quick_run/`** (in the repo: `experiments/bulb_franka_osc_fable/`) — a
complete quick run of exactly this. It has
the final `solve.py`, a README/SUMMARY of the approach, notes and build logs from the iteration,
the full agent session transcript, and result videos.

## Rigorous run (eval harness)

The `eval/` harness runs the same thing under controlled conditions: the env is built once into an
experiment folder, the agent runs in a disposable docker container against read-only mounts under a
time budget, and grading happens post-hoc in fresh containers on independently randomized envs.

```bash
python eval/scripts/build_env.py --name bulb_e2e --stage bulb:franka   # build the world once
python eval/scripts/run_agent.py experiments/bulb_e2e --agent claude   # one agent run in docker
python eval/scripts/run_grade.py experiments/bulb_e2e --run <run>      # grade the delivery
```

Prompt conditions (hints, rules, blocked features) are authored yaml files in `eval/configs/` and
`eval/prompts/`, never ad-hoc flags. Details: `eval/README.html` and `eval/docker/README.md`.

**Example: `experiments/bulb_rigorous_run/`** (in the repo: `experiments/bulb_e2e/`) — a complete
rigorous run of the bulb task: staged manifest,
built stage under `stages/`, and full run + grading artifacts under `runs/e2e_8h/`.

## Creating a new task (worked against the G1 humanoid)

A "task" in this repo is not one file — it is a **scene + assets + env registrations + an
oracle smoke + a real robot solution**, each with its own conventions. This section lists
everything you must build, in the order that works, using `robobench/robots/g1.py` as the
target embodiment. (The same steps apply to any robot; only step 4's binding details change.)

### 0. Assets first — and make them beautiful

Before writing any code, pick the scene's objects. **Download real scanned/authored models
from the internet** — e.g. [Synthesis](https://synthesis.extwin.com/#/home), or any source of
high-quality scans — and select them deliberately: consistent scale, real textures and normal
maps, objects that read instantly on camera. **Do not build the task out of minimal primitive
boxes/cylinders just to get something running** — the benchmark's deliverables are watched as
videos, and the visual quality of a task is part of its quality bar (compare the
pc_motherboard or ikea_table scenes: real vendored products, not placeholder geometry).

Mechanics of vendoring:
- Raw downloads land outside the repo; a prep script (`scripts/vendor_<task>_assets.py` in
  cap-x, or your equivalent) converts them into task-ready USDs under
  `robobench/suites/<suite>/assets/<task>/`: visual meshes carry NO collision; colliders are
  authored separately (convex hulls for simple parts, SDF for threads/thin features), extents
  and materials fixed, texture paths relocalized (absolute paths render black on pods).
- Assets ARE tracked in git (the assembly suite carries ~230 MB); keep every single file
  under GitHub's 100 MB limit.
- Physics dials that made the asset work (friction materials, collision approximations,
  baked scales) belong in the vendor script, so the asset is reproducible.

### 1. The scene — `robobench/suites/<suite>/scenes/<task>.py`

Subclass `BaseScene` (`robobench/core/scene.py`), register it with `@SCENES.register("<task>")`,
and give it a `<Task>SceneCfg` dataclass whose fields are `tunable(...)` (placement/difficulty
dials an agent or a binding may adjust) or `info(...)` (structural, build-time). You must
implement:

- `assets() -> {name: asset_cfg}` — the objects to spawn (no robot). Load your vendored USDs
  relative to the scene file (`Path(__file__).parents[1] / "assets" / ...`) so the suite stays
  relocatable.
- `reset(env_ids)` — re-place objects (defaults + cfg randomization).
- `get_state(env_ids)` / `set_state(state, env_ids)` — EVERY episode-relevant tensor, so a
  restored snapshot behaves exactly like the moment it was taken (checkpointing and grading
  depend on this).
- `describe()` — natural-language scene + GOAL (there is no separate task layer; the goal
  lives here and is what the agent reads).
- `post_step()` — step-coupled mechanics (dose ledgers, auto-welds, button state machines);
  runs every physics substep, not for the agent to call.
- `sim_cfg()` — override for contact-rich scenes (SDF threads want small dt / big buffers).
- **Predicates**: a binary `success()` per env is mandatory — derive it from what the scene
  already computes, matching the surface the other scenes use (e.g.
  `return self.seated().all(dim=1)`). Stage predicates / `score()` follow the suite's
  existing pattern. Intermediate-credit rubrics are built downstream — do not invent
  thresholds the scene does not state.

Import the scene in `scenes/__init__.py` and list it in the suite `__init__.py` docstring.

### 2. Env registration — `robobench/suites/<suite>/configs/envs.py`

Register the runnable names with `register_env` (they derive as `suite.scene[.robot[.mode]]`):

- **Scene-physics-only first**: `register_env(SUITE, lambda: EnvCfg(scene="<task>",
  robot="null", env_spacing=3))` — the NullRobot preset your oracle smoke runs against.
- **Then the G1 binding.** The G1 (`robots/g1.py`) is a fixed-base upper-body manipulator:
  29-DOF with three-finger hands, pelvis welded to the world, control modes `joint` (31 joint
  position targets: 17 arm+waist + 14 hand) and `pink_ik` (two wrist poses + 14 hand). The
  suite conventions for its placement:
  - bench-height work: `surface_z=0.7` in the scene cfg, base at roughly
    `G1RobotCfg(base_pos=(0.0, -0.50, 0.75))` facing +y — and mind the bench face: any
    closer spawns the shins inside the bench and the contact solver kicks the robot over;
  - short arms (~0.55 m reach): pull the whole layout toward the bench front; a scene-cfg
    variant function per embodiment (`_<task>_g1_cfg()`) is the established pattern;
  - register both modes: `for _mode in ("joint", "pink_ik"): register_env(...)`.
- Placements start as guesses — **verify reach before trusting them**:
  `python -m robobench.scripts.robot_binding_smoke --env <suite>.<task>.g1.pink_ik
  --reach_body <some scene body> --headless`.

### 3. The oracle smoke — `robobench/suites/<suite>/smokes/<task>_smoke.py`

The NullRobot smoke proves the scene's physics and predicates before any robot touches it.
Conventions (copy an existing smoke, e.g. the syringe or pen_holder one):

- Drives the task with the scene's own surfaces (kinematic carries, scene dials) through the
  full happy path, `check(...)`-asserting every stage predicate and final `success()`.
- Includes a **negative control** (an episode that must NOT succeed — proves the predicates
  can fail).
- **ALWAYS records video** (viewport rgb annotator, H.264 + yuv420p) — an unrecorded run
  cannot be judged. A `--demo` flag records exactly one clean successful run for the
  deliverable video, separate from the stress smoke.
- Hard-exit teardown at the end: kit regularly hangs inside `env.close()`/`app.close()` —
  never call `env.close()` inside `main()`; use the watchdog `_hard_exit_teardown` pattern.

### 3b. WATCH the video — the footage is the verdict, not the printout

A printed `success()=True` is necessary but not sufficient: **the task counts as solved only
if the rendered video shows it solved**. Actually watch every deliverable video end to end
and treat anything that looks wrong as a bug to fix, not a footnote:

- **Penetration** — parts visually interpenetrating (drawer through a door, an item sunk
  into a tray floor) even when the predicates pass. Root causes found so far: colliders that
  don't match the visual mesh (re-measure and re-author them; SDF for thin/bowed features),
  and translucent materials making a legal gap read as overlap (nothing requires glass to be
  transparent — make it opaque). For contact-rich tasks, add a **measured separation gate**
  to the smoke (visual-mesh point clouds vs. volume models of every other part, asserted on
  every recorded frame — see the coffee/tool_packing smokes) so penetration fails the run
  instead of relying on eyeballs.
- **Rendering artifacts** — ghost shadows of moved objects (RTX accumulation: flush with
  extra `env.sim.render()` passes after teleports), black frames on L20 pods (the
  `--/rtx/verifyDriverVersion/enabled=false` kit arg), invisible meshes (missing extents,
  single-sided geometry, unresolved texture paths).
- **Framing** — the camera must show the WHOLE workspace: the full robot and every
  task-relevant object, not a partial arm with the targets off-screen. Audit the framing
  once with a still before recording the run.
- **Visual quality** — if the footage looks like a toy (placeholder geometry, flat
  materials, floating objects), go back to step 0 and pick better assets. The demo video is
  the task's face; "it technically passes" is not the bar.

### 4. The real robot solution (G1)

Solve the registered `.g1.joint` or `.g1.pink_ik` preset with real actuation — no teleports,
no state writes, every goal recomputed from live measured state (the corpus's core lesson:
stale targets miss). For the G1 specifically: `pink_ik` gives you two wrist poses as the
action surface (the natural one for bimanual work — one hand fixtures, the other operates);
the 14 hand DOFs are always direct joint targets, so grasping is a finger-close ramp you
author. Reference material: `experiments/` sessions, `eval/prompts/skills/` (the
FrankaSession substrate shows the servo/phase-kernel shape to mimic), and
`eval/prompts/examples/` + `examples/` for finished solves.

The deliverable form is the eval contract (`eval/prompts/_contract.md`): a `solve(env)` that
only steps the env it is handed — no `env.reset()`, no `set_states`, import-safe at module
level, with a `__main__` block that builds the registered preset for standalone runs.

### 5. Ship it (two repos)

- **CoSiGen (public)** gets the task: scene, assets, env registrations, oracle smoke. Robot
  solutions must NOT go here — they are the held-out oracles the benchmark grades against.
- **CoSiGen_Solutions (private)** gets the solution: one folder per preset —
  `<suite>/<task>/g1/<mode>/` with `solve.py` (contract form), `MANIFEST` (preset,
  `cosigen_rev` pinned to the commit the solution was VERIFIED at — never pin a rev you did
  not re-verify on —, solved_by, run times, description), and `solve.mp4` (< 100 MB; the
  re-encode recipe is in that repo's README). Run `python3 check.py` there before every
  commit.

### Checklist

- [ ] Assets downloaded from real sources (e.g. [Synthesis](https://synthesis.extwin.com/#/home)),
      vendored under `suites/<suite>/assets/<task>/`, beautiful on camera — not minimal primitives
- [ ] Scene class: cfg dials, `assets/reset/get_state/set_state/describe/post_step`,
      `success()` + stage predicates, `sim_cfg` if contact-rich
- [ ] `scenes/__init__.py` import + suite docstring entry
- [ ] `configs/envs.py`: null preset + `g1.{joint,pink_ik}` bindings, reach-verified
      with `robot_binding_smoke`
- [ ] Oracle smoke: full happy path + negative control + video + `--demo` + hard exit,
      ALL PASS on a clean boot (not just on your warm dev session)
- [ ] Videos WATCHED end to end: task visibly solved in the footage, no penetration or
      rendering artifacts, whole workspace framed, visuals worth presenting (separation
      gate in the smoke for contact-rich tasks)
- [ ] G1 solution in contract form, verified standalone against the registered preset
      at the exact rev you pin
- [ ] Task → CoSiGen PR; solution + video → CoSiGen_Solutions (`check.py` green)

## Going deeper

- **Data engine** (scaling solved tasks into demo data): `data_engine/` — start with `data_engine/README.html`.
