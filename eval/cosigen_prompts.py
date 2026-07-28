"""CoSiGen agent-facing prompt sections, defined once as editable triple-quoted
blocks and composed per embodiment in cosigen_apis (api_doc) / cosigen_session
(make_prompt). <ANGLE_CAPS> tokens are substituted by _compose_doc — plain replace,
so the text can hold literal {...} freely."""
from __future__ import annotations

def _strip_doc_section(doc: str, start: str, next_markers: list[str]) -> str:
    """Remove the doc section starting at `start` up to the first following marker."""
    i = doc.find(start)
    if i < 0:
        return doc
    ends = [e for e in (doc.find(m, i + 1) for m in next_markers) if e > i]
    return doc[:i] + doc[min(ends):] if ends else doc[:i]


# =====================================================================================
# PROMPT SECTIONS — every agent-facing prompt is composed from the named parts below.
# Each part is defined exactly once as an editable triple-quoted block; edit the text
# here and every embodiment's prompt picks it up. <ANGLE_CAPS> tokens are substituted
# at composition time by _compose_doc (plain replace — no f-string brace escaping, so
# the text can contain literal {...} freely). A part starts with the blank separator
# line and ends with a newline unless noted.
# =====================================================================================

def _compose_doc(*parts: str, **tokens: str) -> str:
    doc = "".join(parts)
    for key, value in tokens.items():
        doc = doc.replace(f"<{key}>", value)
    return doc


DOC_HEAD = """Available functions (already in scope; numpy as np):
"""

DOC_RAW_ACCESS = """
== RAW SIMULATOR ACCESS ==
  You may write Isaac Lab-flavored code directly: `env` (the live robobench env:
  env.scene, env.robot, env.sim, env.step(actions)), `api` (this control object),
  `torch`, and any `import isaaclab...` / `import pxr...` (the app is booted).
  The toolkit below stays available and is RECOMMENDED for control/vision/RL/
  backtracking: it keeps the step budget, video recording and checkpoint tree
  consistent. Do not teleport task objects into goal states — success must be
  achieved by physical manipulation (graded on the physics).
"""

DOC_CONTROL_TOOLKIT = """
== Control toolkit ==
  move_to(arm, position, quaternion_wxyz=None, max_steps=300, pos_tol=0.02) -> final_dist
                                                   # waypoint-move a wrist to a WORLD position
  open_gripper(arm=0) / close_gripper(arm=0)       # three-finger hand open/close
  set_hand_joints(arm, {suffix: frac}, steps=20)   # PER-JOINT hand control, frac in [0,1]
      of each joint's range; suffixes: index_0/1, middle_0/1, thumb_0/1/2. Needed for
      asymmetric grasps. PROVEN G1 tube-grasp recipe (HOOK+SCOOP): a top-down pinch
      CANNOT hold a tube (thumb too short to oppose) -- instead approach HORIZONTALLY
      (fingers level, pointing across the tube, thumb yawed aside thumb_0~1.0), hook the
      fingers over/past the tube, then SCOOP: raise the wrist ~7cm WHILE half-curling
      (fingers ~0.5) so the curl finishes mid-air, then clamp to a fist and bring the
      thumb under. Approach radially, retarget from the object's LIVE pose each phase.
  step(n=1)                                        # advance sim n steps holding targets
"""

# <OBJECTS> = the object-name listing shown on the get_object_pose line;
# <SEATED_LINE> = the complete get_seated doc line (embodiments word it differently).
DOC_READ_ONLY = """
== Read-only queries (return copies; cannot affect the world) ==
  get_object_pose(name) -> (pos_xyz, quat_wxyz)    # <OBJECTS>
  get_state(name) -> (13,) [pos3,quat4,linvel3,angvel3]
  get_eef_pose(arm) -> (pos_xyz, quat_wxyz)        # current wrist; arm 0=left,1=right
<SEATED_LINE>
  get_robot_state() -> dict                        # joints, wrist poses/vels/targets, hand state
  get_link_positions(pattern='.*hand.*') -> {name: pos3}   # LIVE robot link positions (regex).
      Use for finger/palm geometry; describe_scene positions do NOT track live physics.
  list_objects() / describe_scene(structured=False)

Seeing the scene takes no call of its own: every program that moves the world comes back
with an image of where it left things, alongside the printed numbers. The view tool shows
more of that run when you want it (frames=k samples k frames across the run, so you can
watch how it got there).
"""

DOC_CHECKPOINT_TREE = """
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
     execute(path=...), read the printed measurements and the end-state image,
     edit the file, run it again.
  2. Every run that moves the world is followed by assess: say what the image and
     the log show, what failure modes you can see in them, and whether the state
     is worth keeping. Keeping it saves the program as a node, so later work
     starts from there and you can return to it. Tasks here are long, so keep the
     states you would not want to re-derive.
  3. Your Python variables persist across programs and are unaffected by which
     node you are on.
  4. Splitting a stage into smaller programs lets you save progress more often.

== Checkpoint tree commands ==
  execute(path='/workspace/<your_program>.py')   # run from your current node
  assess(scene=..., log=..., failure_modes=..., keep=...)   # after a run that moved
             # the world; keep=True also needs label= and path= and saves the node
  checkpoint(label='<short description of the reached state>',
             path='/workspace/<your_program>.py') # save a program's end state as a node
  goto(node='n3')                  # shows that node's image, log and program;
             # call again with confirm=True to move the world there
  list_checkpoints() -> str        # the tree: every node, its parent, where you are
  get_checkpoint_scene(cid) / get_checkpoint_log(cid) / get_checkpoint_code(cid)
Going back is also how you improve a stage: read a node's code, go to its parent,
and checkpoint a better version as a sibling.
"""

# <N_ENVS> = number of parallel envs on this pod (api.n).
DOC_TUNING = """
== Tuning the numbers in your program (the optimize tool) ==
You need to use the optimize tool to find optimal values of parameter settings
in your program: whenever a program has parameters that you need to decide with
trial and error — an offset, a depth, an angle, a timing, a threshold — use
this tool. It runs <N_ENVS> copies of your program in parallel on a second
simulator, each from your current state with a different setting of the
constants you name, scores every end state with your objective, and adapts the
search over up to 8 rounds (generations=) within a time budget (budget_s=,
default 3600 — set it freely).
The call returns as soon as the search starts, so carry on with your next piece
of work while it runs. Each round's best setting is appended to your following
tool results, and /workspace/optimize_status.json always holds the latest
progress if you want to look.
  optimize(program='/workspace/<your_program>.py',
           space={'<CONST_NAME>': (low, high)},
           objective='/workspace/<your_objective>.py')
      -> {started, program_version, status_file}   # results stream in later
Every result is tagged with the program version it tuned, since you may edit the
file while the search runs. The best found within the budget is not necessarily
the optimum: the search always seeds from the values in your file, so you can
apply a result and call optimize again to keep refining from there.
  program: the file you are already iterating on, unchanged. The names in
    space must be top-level constants of the file; each copy runs with its
    own value substituted. If a value lives inside a function, lift it to a
    top-level constant first — e.g. to tune the 0.006 in
        def descend(arm):
            move_to(arm, [x, y, z - 0.006])
    rewrite it as
        PRESS_DZ = 0.006
        def descend(arm):
            move_to(arm, [x, y, z - PRESS_DZ])
    and search space={'PRESS_DZ': (0.001, 0.02)}. One requirement: the program
    must act through the toolkit functions above — raw env/articulation
    handles are global to the simulator and cannot be copied per candidate,
    so they are unavailable during a search.
  setup: optional path to a second file, run once on the search simulator
    before the copies start, where raw env/api handles ARE available. The
    search runs on a separate simulator that loads your current checkpoint,
    and a checkpoint carries world state only — so any simulator configuration
    you set up yourself with raw handles (controller gains are the common one)
    is not there. Put that configuration in a setup file and the copies run on
    the robot you have been working with:
        optimize(program='/workspace/<stage>.py', setup='/workspace/gains.py',
                 space={'PRESS_DZ': (0.001, 0.02)},
                 objective='/workspace/objective.py')
    So a program that boosts gains and then moves splits in two: the raw-handle
    part goes to setup, the motion stays in the program with its constants.
  space: {'<CONST_NAME>': (low, high)} searches a float in [low, high];
    append 'int' or 'log' as a third element for integers / log-scaled search.
    A fixed set of values searches as {'<CONST_NAME>': ('choices', (45, 90, 135))}.
    The values already in your file always run as one of the candidates and the
    search starts around them, so the result is never worse than what you have.
  objective: a small file defining
      def objective(v) -> float    # lower is better; scored on each end state
    v is that copy's oracle view: v.object_pose(name) -> (pos, quat_wxyz),
    v.object_vel(name), v.eef_pose(arm), v.seated(), v.hand_frac(arm).
    Score the measurement you already print to judge success — usually a
    distance or angle between objects. The search finds exactly what you score
    and nothing more, so make the objective reflect the whole goal of the
    maneuver: if a lift must also keep the object upright and stay gripped,
    score all three, or the best "lift" found will be a tilted one. Cap credit
    at the physically achievable value and penalize ruined outcomes (dropped,
    tilted past recovery), or the search will find those instead of the task.
Minimal example — one constant, defaults everywhere. /workspace/objective.py:
    def objective(v):
        obj_p, _ = v.object_pose('<object>')
        tgt_p, _ = v.object_pose('<target>')
        return float(((obj_p - tgt_p) ** 2).sum() ** 0.5)  # end distance to target
  optimize(program='/workspace/<stage>.py', space={'PRESS_DZ': (0.001, 0.02)},
           objective='/workspace/objective.py')
  Keep working; a later result reads
      {'best': {'PRESS_DZ': 0.0063}, 'best_score': 0.004, ...}
  Then set PRESS_DZ = 0.0063 in <stage>.py, verify with execute, and checkpoint.
Fuller example — several constants, robust to start-state variation:
  optimize(program='/workspace/<stage>.py',
           space={'PRESS_DZ': (0.001, 0.02), 'YAW_STEP_DEG': (2, 25),
                  'SETTLE_STEPS': (5, 60, 'int'),
                  'APPROACH_DEG': ('choices', (45, 90, 135))},
           objective='/workspace/objective.py',
           generations=10, budget_s=3600, repeats=2,
           randomize={'<object>': {'pos': 0.01, 'yaw': 0.2}})
  repeats=2 scores each candidate on 2 jittered starts (randomize) and
  averages, for numbers that must hold up under start variation rather than
  fit one exact pose.
"""

# <EMBODIMENT> = the embodiment-specific opening sentence(s); no trailing newline.
DOC_TAIL = """
<EMBODIMENT> Direct sim-state writes are NOT available and NOT allowed;
affect the world only through control functions. Write fault-tolerant code.
"""

# ---- embodiment lines for DOC_TAIL --------------------------------------------------
EMB_G1_ASSEMBLY = """The robot base is FIXED (upper-body only). Lowering a leg onto its stud and pressing/holding
seats it (auto-welds when settled)."""
EMB_FRANKA_SINGLE = ("The robot is a FIXED-BASE Franka arm with a 2-finger parallel gripper "
                     "(single arm: use arm=0 everywhere).")
EMB_KUKA_ALLEGRO = ("The robot is a Kuka iiwa7 arm with a 16-DOF ALLEGRO four-finger hand "
                    "(single arm: use arm=0 everywhere). set_hand_joints(0, {'index': 0.8, "
                    "'thumb_joint_1': 0.6}) gives per-finger control; get_hand_joints() reads "
                    "back live positions. Grasps are POSTURES you design (wrap the glass, "
                    "tripod the cap), not a width.")
EMB_FRANKA_BIMANUAL = ("The robot is TWO fixed-base Franka arms (arm 0 = left at x<0, arm 1 = "
                       "right at x>0), each with a 2-finger parallel gripper; all control "
                       "functions take the arm index.")
EMB_IKEA_BIMANUAL = """TWO fixed-base Franka arms with parallel grippers (arm 0 = left/slab side, arm 1 = right/leg-row side; pass arm= to every control call). Lowering a leg onto its stud and pressing/holding
seats it (auto-welds when settled)."""


def _emb_screw(robot: str, ln: str, fn: str) -> str:
    """Append the screwing hint (nut/bolt tabletop suites) to an embodiment line."""
    return (f"{robot} Screwing requires PRESSING DOWN while ROTATING the {ln} about the "
            f"{fn}'s axis (use quaternion_wxyz in move_to for wrist rotation; thread "
            "friction holds progress).")


# ---- read-only line variants ---------------------------------------------------------
SEATED_LINE_ASSEMBLY = ("  get_seated() -> [bool x4]                        "
                        "# per-leg seated; task done when all True")
SEATED_LINE_GENERIC = ("  get_seated() -> []                               "
                       "# unused in this suite; grade via scene_summary")

# ---- session protocol / planning discipline (the prompt's closing directive) --------
DIRECTIVE_SESSION_PROTOCOL = """
Session protocol: each of your replies runs one program (execute, inline code or a workspace file by path). Python variables persist across programs. After each run you receive your stdout and the scene summary.
"""

DIRECTIVE_PLANNING = """
Planning discipline:
1. First make an explicit plan of stages, each a meaningful subgoal, and keep it as a
TodoWrite checklist: one todo per stage, exactly one in_progress at a time, marked
completed when the stage's program is checkpointed. Also start every program with a
comment marking its stage: # plan: [stage 3/7] <what this stage does>
2. One program should complete one full stage (typically several hundred sim steps, with internal retries and printed verification) -- not one micro-motion. Runs have real overhead; batch exploration into one run. You can also consider splitting your work and program into smaller pieces / stages, and checkpoint at stages you want, so that you could save progress more frequently.
3. Every run that moves the world comes back with an image of where it ended and the log it printed; read both and assess them before the next run. Keep the states you would not want to re-derive: a kept state becomes a node your later work starts from and can return to.
4. Use the optimize tool to find optimal values of parameter settings in your programs: whenever a program has parameters that you need to decide with trial and error (offsets, depths, angles, timings, thresholds), call optimize on that program instead of hand-guessing values run after run.
5. Before going back to a node, look at what it holds: the goto tool shows that node's image, log and program, and asks you to confirm. Read the 'previously tried from here' digest when you land, and change approach rather than repeating a branch that already failed the same way.
"""

DIRECTIVE_CLOSING = """
Do not give up: there is no turn limit. Only reply with the single word DONE once the task is actually complete.
Write one Python code block per turn using only the functions listed above (already in scope) plus numpy (`np`). Wrap the code in one ```python ... ``` fenced block."""

CODE_DIRECTIVE = DIRECTIVE_SESSION_PROTOCOL + DIRECTIVE_PLANNING + DIRECTIVE_CLOSING

# The null-embodiment (no-robot) prompt parts moved to legacy/cosigen_null.py
# (2026-07-26, user decision): no session ever ran on the null surface — sim_gen
# construction is standalone teleport-oracle scripts, and evals are robot-bound.


