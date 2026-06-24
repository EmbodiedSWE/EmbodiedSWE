# RoboBench

A benchmark for **code-generating agents**. It tests whether an agent can *few-shot* solve
**long-horizon, complex tasks** — multi-step problems where the agent has to write code that plans,
manipulates, and recovers across a long sequence of actions, not just emit a single move.

RoboBench is organized as a growing family of **suites**, each a different flavor of hard task
(robot furniture assembly is the first; other flavors — e.g. liquid handling / latte-art benches —
can slot in the same way). The simulator backend is deliberately *not* baked into the design:
suites compose the same small set of building blocks regardless of what runs underneath, so new
backends and new task families are additive, not rewrites.

This README is the living statement of the **building blocks and the design philosophy**.

---

## Design philosophy

**One equation.** This is the heart of the design, not a diagram bolted on after:

```
env  =  scene  +  robot          (+ an optional, hidden verifier)
```

That's the entire composition. Everything else is detail under those three nouns.

- **`BaseScene`** — the object world (the things to manipulate). Embodiment-agnostic; knows nothing
  about the robot. **It also carries the goal** (see "No task layer").
- **`BaseRobot`** — the actor: the embodiment; its actions flow through a **pluggable controller**
  chosen by its `control_mode` (see below).
- **`BaseEnv`** — the open shell that holds the simulation and wires a scene and a robot together.
  Composition is by **injection**: build a scene and a robot, hand them to the env. Swapping
  embodiment = a new env with a different robot.
- **`BaseVerifier`** — a privileged, *optional*, *hidden* scorer. Not part of the env's contract
  with the agent; the env runs fine without one.

All four are import-light — they import without spinning up a simulator, so you can list registries,
read contracts, and run tests cheaply. Only *building* a `BaseEnv` touches the backend (imported
lazily).

**Registration is import-time, and cheap — because heavy work is deferred.** Scenes, robots,
controllers, and env-configs register themselves into name→factory registries *at import time* (a
`@register` decorator or a `register_env(...)` call runs when the module is imported). So
`robobench.discover()` — which imports every robot, controller, and suite to populate the registries —
is all it takes to make everything resolvable by name. That stays **~milliseconds and never launches
the simulator** only because each module keeps its **top level import-light**: the expensive work
(importing the backend, loading USD, building the scene/robot/sim) is **deferred into methods**
(`assets()`, `build()`, `to_isaaclab()`), never run at module scope. The discipline this demands:
when you add a scene/robot/controller, **keep heavy imports inside methods, not at module top level** —
otherwise discovery, and every "just list what's available", slows to a simulator boot.

**The env is open, not a sealed box.** The agent reaches the simulation, the scene, the robot, the
raw asset handles, and the underlying state. It may patch an instance, subclass-and-adopt, or
rebuild. Enforcement of "don't cheat" is by *prompt*, not by access control — ablation dials that
restrict access live in the harness (`agent/`), never in the env.

**No task layer — a scene *is* a task.** There is deliberately no `Task` class / `TASKS` registry.
Each scene is built for one task: it carries its own goal as natural language in `scene.describe()`,
and its success criteria live in an optional `BaseVerifier`. A task class would just re-wrap the
scene's goal and a hidden reward — so it's absorbed into the scene + verifier. (One scene → one
task; a variant task is a new scene.)

**The verifier is optional, hidden, and outside the agent's loop.** While solving, the agent reads
the open state and designs and trusts *its own* success metric — the env hands it no reward, so the
task is "solve the goal," not "maximize this number." The `BaseVerifier` is a separate, privileged
scorer that exists for everything *around* the solving loop: a human (or an automated harness)
running **large-scale, apples-to-apples evaluation** across agents, and producing a **reward signal
for post-hoc RL / fine-tuning** of an agent. Keeping it optional and hidden lets the *same* scene
serve both open-ended exploration and rigorous, comparable scoring — and a scene can ship with none
at all (pure sandbox).

**Control is a separate, shared, pluggable layer — selected by the robot.** A `BaseRobot` declares
the **control modes** it supports (`control_modes`, e.g. `("joint", "ee_pose", "osc_impedance")`)
and which is active (`control_mode`); the active mode maps the action vector to joint targets, so
`action_dim` follows the mode. But the *controller itself* is a **shared, reusable component** — its
own module, not baked into each robot: one implementation (joint, end-effector IK,
operational-space / impedance, a whole-body solver, or a **frozen learned policy**) serves many
embodiments. A robot just declares its modes and wires each to a controller with its own joints /
end-effector / gains. This keeps "swap the controller on the same robot + task" a clean, first-class
operation. The env dispatches through `self.robot` every step (late binding), so a patched/swapped
controller — or a custom one via an overridden `apply_action` — takes effect immediately.

**Distinguish a controller from a skill, and the env from the agent.** A **controller** is the
per-step action→joint map (the control mode). A **skill** is a higher-level primitive the agent
*invokes* ("grasp this", "walk to here") that runs many steps *through* a controller — skills live
in the **agent layer**, not the env. The guiding rule across the whole framework: the env provides
what is **shared and part of the stable contract** (scenes, robots, controllers, the verifier); the
agent owns what **is the research** (its policies, training loops, skill orchestration, curriculum).
Provided layers are baselines + seams, never locked — an agent may use a controller, extend it, or
replace it.

This is the **System 1 / System 2** split now standard in robot foundation models — a fast, reactive
low-level policy beneath a slow, deliberative high-level one (Figure's **Helix**: a 200 Hz reactive
S1 under a 7–9 Hz VLM S2; NVIDIA **GR00T N1**: a high-rate diffusion-policy S1 under a VLM S2,
explicitly "inspired by human cognition") — i.e. the **cerebellum** (fast motor coordination) under
the **cerebrum** (deliberation/planning). robobench splits along the same line: the **controller is
System 1 / cerebellum** — env-owned, fast, per-step — and the **skill + agent are System 2 /
cerebrum** — agent-owned, slow, deliberative. The env provides the fast substrate; the agent owns
the slow brain.

> **Controllers — roadmap (coming, not built yet).** The control layer is grown incrementally.
> First a basic set (joint, then an **IK solver**); then, as needed: operational-space / impedance,
> a whole-body solver, **composite** controllers (different controllers over different DOF groups —
> e.g. arms by IK + legs by a locomotion policy), a **residual-RL** wrapper (a learned correction on
> top of a base controller), and **frozen / pure-RL policy** controllers. Crucially, *training* those
> policies (residual or pure locomotion RL) is the **agent's** job — done through the env's reward +
> rollout + snapshot seams; the env only provides the controller slot they plug into.

**The base classes are strict contracts.** Everything a child must provide is an `@abstractmethod`,
so a subclass that forgets one *cannot be instantiated*. `bind()` is the one concrete shared method
(caches the env back-ref; children extend via `super().bind(env)`).

**Curriculum is for the agent to drive.** The abstractions exist so an agent can cheaply *change
the problem to debug it* — relax physics, disable collisions, or swap the embodiment to make a hard
step tractable, then tighten back. And because `get_states` / `set_states` can *set* the env to any
state, the agent can **jump straight into a not-yet-learned state** to debug that phase directly —
instead of replaying the whole task from the start to reach it. Making those switches easy to
explore is a first-class goal.

**Skills transfer across embodiments and across tasks.** Because the scene is embodiment-agnostic,
the *same* task can run with a *different* body — so "same scene, different robot" (and "same robot,
new scene") is a cheap swap, not a rewrite. This is deliberate: an agent should be able to carry a
skill from one embodiment to another, or from one task to a related one, and few-shot the new case
by adapting what it already knows instead of relearning from scratch — and even to *characterize*
what actually differs between two bodies on a simple task.

**Batched and RL-ready — so the agent can build its own debug loop.** The env runs many copies in
parallel and exposes full state capture/restore alongside a plain step loop. When an agent gets
stuck on one phase, it can stand up a *small RL environment on the spot* — treat the (otherwise
hidden) success check as a reward, restart from a saved snapshot of the hard phase, and drill just
that segment — without leaving the framework or rewriting the env. Writing a controller, scripting a
plan, or training a small policy to get past a phase are all just things you can do against the same
open object.

---

## Contracts at a glance

| Class | Children must implement (`@abstractmethod`) | Concrete on the base |
|---|---|---|
| `BaseScene` | `assets · reset · get_state · set_state · describe` | `bind` |
| `BaseRobot` | `assets · reset · get_state · set_state · describe` (+ hooks `on_bind` / `build_controller`) | `bind · apply_action · action_dim · set_controller · actuator_sink · actuator_limits`; declares `control_modes` / `control_mode` |
| `BaseController` | `action_dim · compute · _resolve_joints` | `bind · apply · reset`; `command_type` set at construction |
| `BaseVerifier` | `verify` → `Any` *(structured result TBD)* | — |
| `BaseEnv` | *(not abstract — the shell)* | `get_states · set_states · reset · step · describe · describe_stage · verify · close` |

- **`describe()`** → curated **natural-language** string (scene's objects + goal, then the robot).
  Prompt-ready.
- **`describe_stage(root=None, raw=False)`** → the low-level counterpart: structured
  `list[{path, type, pos, size}]`, or `raw=True` for the raw scene-graph text.
- **`get_states()` / `set_states()`** → full state = `{"scene": ..., "robot": ...}`; restore to
  *any* saved state (snapshots, replay of hard phases).
- **`step()` / `reset()`** return `None` — observations are a wrapper concern, not the raw env's.

---

## Using it — recipes

> Examples assume `AppLauncher` is running and the registries are populated
> (`import robobench; robobench.discover()`), plus an `env` built as in the first recipe. The env is an
> **open object**: patch or swap parts on the live instance — the step loop dispatches through
> `self.robot` / `self.scene`, so changes take effect on the next `step()`.

**Build & run an env.** Load a registered binding by name, or wire an ad-hoc combo, then step it:

```python
import robobench; robobench.discover()        # populate registries: robots, controllers, suites
from robobench.core import ENVS, EnvCfg

env = ENVS.get("assembly.ikea_table")().build(num_envs=4)          # a registered config, by name
# ...or ad-hoc:  EnvCfg(scene="ikea_table", robot="g1", control_mode="joint").build()
env.reset()
print(env.describe())                                              # NL scene + goal + robot (prompt-ready)
for _ in range(200):
    env.step(action)                                               # action: (num_envs, env.robot.action_dim)
```

(Or smoke any combo from the shell: `python -m robobench.scripts.smoke --list` / `--env <name>`.)

**Switch the control mode.** Same robot + scene, different actuation — `action_dim` follows:

```python
EnvCfg(scene="ikea_table", robot="g1", control_mode="pink_ik").build()   # vs "joint"
# or override at build time:  cfg.build(control_mode="pink_ik")
```

**Write your own controller and plug it in.** A controller maps an action to a joint command for the
DOFs it owns and writes it itself. Subclass `BaseController`, then drop it onto the live robot with
`set_controller` — it binds and takes effect next `step()`, and need **not** be in the robot's
`control_modes` menu:

```python
from robobench.core import BaseController

class ElbowWiggle(BaseController):
    def __init__(self, cfg=None):
        super().__init__(cfg, command_type="position")        # what it writes: position / velocity / effort
    def _resolve_joints(self, robot):                         # which DOFs it drives
        return robot.articulation.find_joints([".*_elbow_joint"])[0]
    @property
    def action_dim(self): return len(self.joint_ids)
    def compute(self, action):                                # PURE: action -> joint targets
        return action                                         # self.robot / self.limits available

env.robot.set_controller(ElbowWiggle())                       # live; bypasses control_modes
```

`compute` is pure math; the inherited `apply` writes its output through the **sink** the robot handed
the controller at bind (matching `command_type`). For a fully custom path, override `robot.apply_action`
instead. This is the seam for the agent's own controllers — the env owns the slot, the agent owns the policy.

**Combine controllers.** Two composable axes:

- *Different joints* → `CompositeController` (the action is split across groups; groups may even use
  **different** `command_type`s — e.g. effort arms + position hands, each writing its own):

  ```python
  from robobench.controllers import CompositeController, JointController, JointControllerCfg
  env.robot.set_controller(CompositeController([
      MyArmController(),                                                          # your arm controller
      JointController(JointControllerCfg((".*_hand.*",)), command_type="position"),   # hands, direct
  ]))
  ```

- *Same joints* → a **residual** wrapper (`base.compute(action) + scale · correction`), itself a
  controller, so it nests inside a composite leaf. *(Planned; the correction net is agent-side.)*

**Tune the build (gains, placement, sim).** It's all config — override on the `EnvCfg`, its
`robot_cfg`, or per-build:

```python
from robobench.robots import G1RobotCfg
EnvCfg(scene="ikea_table", robot="g1", control_mode="joint",
       robot_cfg=G1RobotCfg(arm_stiffness=1500.0, base_pos=(0.0, -0.6, 0.75)),   # softer arms, moved back
       sim_overrides={"dt": 1 / 200}).build(num_envs=16)
```

**Snapshot & restore (debug / RL).** Capture full state and jump back to it — replay a hard phase, or
stand up a small RL loop from a saved snapshot rather than from the episode start:

```python
s = env.get_states()        # {"scene": ..., "robot": ...}
# ... drive / train ...
env.set_states(s)           # restore to ANY saved state
```

---

## Layout

```
robobench/
  core/      shared machinery — BaseScene · BaseRobot · BaseEnv · BaseVerifier,
             scene introspection (describe_stage), registries (SCENES · ROBOTS)
  robots/    reusable embodiments — where concrete control pipelines / gains / IK live
  suites/    the task families (one folder per flavor; see each suite's own docs)
```

> A per-suite section (what each flavor contains, how to add one) will be added once the first
> suite is built.
