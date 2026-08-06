"""RETIRED null-embodiment (no-robot) session surface — moved out of the live agent
loop 2026-07-26 (user decision).

History: NullSceneApi was written for the sim_gen constructed-problem pipeline, on the
theory that agentic sessions might run against constructed tasks with no robot bound
(manipulating via kinematic staging + physics settling). In practice NO session ever
used it: sim_gen construction runs task modules standalone through the forge (its
oracle solutions are direct set_state scripts — see sim_gen/pipeline/prompt.py, the
"teleport-oracle test battery"), and evaluation runs are robot-bound (Franka) by user
mandate. The checkpoint-tree surface documented below therefore never had a consumer
on a null pod.

Kept importable so a legacy null pod (env registered as '<suite>.<scene>' with no
robot segment) still resolves: cosigen_session.resolve_api loads this file by path.
"""
from __future__ import annotations

import torch

from cosigen_apis import GenericSceneApi
from cosigen_prompts import DOC_HEAD, _compose_doc

# ---- null-embodiment (sim_gen constructed problems) prompt parts ---------------------
# No robot: staging + physics settling modality, compact checkpoint wording, and its
# own planning directive (no optimizer/tuning surface on these pods).
NULL_DOC_RAW_ACCESS = """
== RAW SIMULATOR ACCESS ==
  You may write Isaac Lab-flavored code directly: `env` (the live robobench env:
  env.scene, env.iscene, env.sim, env.step(actions)), `api` (this control object),
  `torch`, and any `import isaaclab...` (the app is booted). Scene asset handles
  (env.iscene.rigid_objects[name]) support write_root_state_to_sim /
  set_external_force_and_torque for kinematic holds and force pokes.
"""

NULL_DOC_MODALITY = """
== TASK MODALITY (READ CAREFULLY) ==
This task has NO robot. You manipulate by STAGING bodies and letting PHYSICS act
-- exactly how the task's reference solution works: place/release an object in a
sensible pre-state (e.g. slightly above a target), then step() and let it drop,
slide, settle. The rubric judges SETTLED physics states (settle gates, geometry
gates, velocity gates): teleporting an object directly into the goal pose will
NOT score. Progressive kinematic motion (e.g. a slow carried trajectory of
set_object_state calls with small displacements, stepping between writes) is a
legitimate 'hold'; instantaneous jumps into scored states are not.
"""

NULL_DOC_CONTROL = """
== Control ==
  set_object_state(name, pos, quat=None, lin_vel=(0,0,0), ang_vel=(0,0,0))
      # stage a body (env-frame pos, wxyz quat), then step() to let physics act
  step(n=1)                        # advance the sim n steps
"""

NULL_DOC_READ_ONLY = """
== Read-only queries (return copies; cannot affect the world) ==
  get_object_pose(name) -> (pos_xyz, quat_wxyz)    # objects: <OBJECTS>
  get_state(name) -> (13,) [pos3,quat4,linvel3,angvel3]
  list_objects() / describe_scene(structured=False)
  score() -> float                 # the task's partial-credit rubric (0..100)
  look(view='default') -> str      # RENDER the scene now from 'default','front',
      'left','right','top','close' (or custom eye=/target= env-frame). The image is
      attached to your NEXT feedback message -- look from several angles, then plan.
  review_rollout(k=4)              # sample k frames from the LAST executed turn's
      video and attach them to your next feedback -- watch what your code just did
"""

NULL_DOC_CHECKPOINT_TREE = """
== Checkpoint tree: save your progress, return to it any time ==
Your work is organized as a tree of saved world states (nodes). You are always
at some node. Every program you run starts from that node's state, and when it
finishes the world returns there — so a trial costs nothing you cannot redo,
and nothing you try can spoil the state you are working from.
You save progress with checkpoint(label=..., path=...): it runs that program
from your current node and saves its end state as a new node, with the program
as the node's code. You return to any saved node any time with goto(cid) —
branch from an earlier state, compare strategies, and continue from whichever
node is strongest. A program that raises an exception is not checkpointed; you
get the traceback and stay where you were.
The workflow:
  1. Keep each stage's program as a file in your workspace; run it with
     execute(path=...), read the printed measurements and images, edit the
     file, run it again.
  2. Save your progress often; tasks are difficult and long-horizon. By checkpointing
     your progress, you can save time by building your work off of promising states,
     and backtrack to earlier states when you need to.
  3. Your Python variables persist across programs and are unaffected by which
     node you are on.
  execute(path='/workspace/<your_program>.py')   # run from your current node
  checkpoint(label='<short description of the reached state>',
             path='/workspace/<your_program>.py') # save its end state as a new node
  goto(cid) -> str                 # jump to any saved node and continue from there
  list_checkpoints() -> str        # the tree: every node, its parent, where you are
  render_checkpoint(cid)           # see a node's state (image on your next feedback)
  get_checkpoint_scene(cid)        # object poses and scene text at a node
  get_checkpoint_log(cid)          # the full printed log of the program that made it
  get_checkpoint_code(cid)         # that program's code
Before moving, inspect the candidates (render_checkpoint, get_checkpoint_scene),
then goto() the one you want. Going back is also how you improve a stage: read a
node's code, goto() its parent, and checkpoint a better version as a sibling.
"""

NULL_DOC_GRADING_PROTOCOL = """
Grading: scene_summary (returned after every turn) includes success= and the
task telemetry; score() is the partial-credit rubric. The task is complete when
success=True.

Session protocol: each of your replies runs one Python program in the
simulator. Variables you define persist across programs. After each run you
receive your stdout, the scene summary, and where you are in the checkpoint tree.
"""

NULL_CODE_DIRECTIVE = """
Code directive (planning discipline):
1. First make an explicit plan of stages, each a meaningful subgoal, and keep it as a TodoWrite checklist: one todo per stage, exactly one in_progress at a time, marked completed when the stage's program is checkpointed. Also start every program with a comment marking its stage: # plan: [stage i/N] ...
2. One program should complete ONE full stage (typically several hundred sim steps, with internal retries and printed verification) -- not one micro-motion. Runs have real overhead; prefer fewer, bigger, self-contained programs.
3. checkpoint(label=..., path=...) a working stage when you want to save progress so that you can build your future work off from there.
4. When you goto() a node, read the 'previously tried from here' digest and do not repeat a branch that already failed the same way -- change approach.

Do not give up: there is no turn limit. Only reply with the single word DONE once the task is actually complete.
Write ONE Python code block per turn using ONLY the functions listed above (already in scope) plus numpy (`np`). Wrap the code in ONE ```python ... ``` fenced block."""


class NullSceneApi(GenericSceneApi):
    """Scene-only control API for null-embodiment tasks (sim_gen constructed problems).
    There is no robot: the agent manipulates through the SAME modality the tasks' oracle
    solutions use — kinematic staging of scene bodies (set_object_state / raw scene
    handles) followed by REAL physics settling. Every constructed rubric judges settled
    physics states (settle gates, geometry gates), so pose-teleporting an object straight
    into a goal state does not score. No arms/grippers, no RL surface. RETIRED to
    legacy 2026-07-26: no session ever ran on it (construction is standalone scripts,
    evals are robot-bound)."""

    IS_NULL_EMBODIMENT = True

    VIEWS = {
        "default": ((1.5, -1.5, 1.1), (0.0, 0.0, 0.15)),
        "top": ((0.05, 0.0, 2.2), (0.0, 0.0, 0.0)),
        "front": ((1.9, 0.0, 0.7), (0.0, 0.0, 0.15)),
        "left": ((0.0, 1.9, 0.7), (0.0, 0.0, 0.15)),
        "right": ((0.0, -1.9, 0.7), (0.0, 0.0, 0.15)),
        "close": ((0.9, -0.9, 0.6), (0.0, 0.0, 0.1)),
    }
    _TELEMETRY = GenericSceneApi._TELEMETRY + ("score",)

    CODE_DIRECTIVE = NULL_CODE_DIRECTIVE

    def __init__(self, env, max_steps: int = 3000):
        self._init_common(env, max_steps)  # NullRobot: articulation is None; never touched
        self._steps_used = 0
        try:
            self._set_view(*self.VIEWS["default"])
        except Exception:
            pass

    # ----- no-robot control surface -----
    def reset_state(self) -> None:
        self._steps_used = 0

    def _action(self) -> torch.Tensor:
        return torch.empty(self.n, 0, device=self.device)

    def _api_state_env0(self) -> dict:
        return {}

    def _restore_api_state(self, st) -> None:
        pass

    def set_object_state(self, name: str, pos, quat=None, lin_vel=(0.0, 0.0, 0.0),
                         ang_vel=(0.0, 0.0, 0.0)) -> None:
        """Stage a scene body: write its root state (env-frame position, wxyz quat), then
        let PHYSICS take over (follow with step()). This is the oracle solutions' staging
        modality: e.g. release an object slightly above a target and let it drop/settle.
        Rubrics judge SETTLED states -- teleporting into the goal pose does not score."""
        reg = getattr(self._env.iscene, "rigid_objects", {}) or {}
        if name not in reg:
            raise ValueError(f"unknown object {name!r}; use one of {self.list_objects()}")
        st = reg[name].data.root_state_w.clone()
        ids = torch.arange(self.n, device=self.device)
        st[:, 0:3] = self.origin + torch.tensor(list(map(float, pos)), device=self.device)
        if quat is not None:
            q = torch.tensor(list(map(float, quat)), device=self.device)
            st[:, 3:7] = q / q.norm().clamp_min(1e-8)
        st[:, 7:10] = torch.tensor(list(map(float, lin_vel)), device=self.device)
        st[:, 10:13] = torch.tensor(list(map(float, ang_vel)), device=self.device)
        reg[name].write_root_state_to_sim(st, ids)

    def score(self) -> float:
        fn = getattr(self.scene, "score", None)
        if not callable(fn):
            return 0.0
        v = fn()
        return float(v[0] if torch.is_tensor(v) and v.dim() > 0 else v)

    def functions(self) -> dict:
        return {
            # control (staging + physics)
            "set_object_state": self.set_object_state,
            "step": self.step,
            # read-only introspection
            "list_objects": self.list_objects,
            "get_object_pose": self.get_object_pose,
            "get_state": self.get_state,
            "describe_scene": self.describe_scene,
            "score": self.score,
            "look": self.look,
            "review_rollout": self.review_rollout,
            # checkpoint tree (backtracking)
            "checkpoint": self.checkpoint,
            "goto": self.goto,
            "list_checkpoints": self.list_checkpoints,
            "get_checkpoint_code": self.get_checkpoint_code,
            "get_checkpoint_scene": self.get_checkpoint_scene,
            "get_checkpoint_log": self.get_checkpoint_log,
            "render_checkpoint": self.render_checkpoint,
        }

    def api_doc(self) -> str:
        objs = self.list_objects()
        shown = ",".join(f"'{o}'" for o in objs[:12]) + (",..." if len(objs) > 12 else "")
        return _compose_doc(
            DOC_HEAD, NULL_DOC_RAW_ACCESS, NULL_DOC_MODALITY, NULL_DOC_CONTROL,
            NULL_DOC_READ_ONLY, NULL_DOC_CHECKPOINT_TREE, NULL_DOC_GRADING_PROTOCOL,
            OBJECTS=shown,
        )
