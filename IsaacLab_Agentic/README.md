# IsaacLab_Agentic

Two Isaac Lab manipulation tasks, each solved **from scratch by two different
coding agents** — **Fable 5** and **Opus** — using hand-scripted controllers
(no RL training, no teleoperation). Both agents were given the same task briefs
(see [`Fable_5/prompt.txt`](Fable_5/prompt.txt)) and produced their own
executable controller plus a rendered video of the successful trajectory.

The point of this repo is a **side-by-side comparison**: the same two tasks,
two independent agentic solutions, so you can diff the strategies, the code, and
the debugging notes each agent left behind.

| Agent | Subfolder | Notes |
|---|---|---|
| Fable 5 | [`Fable_5/`](Fable_5/) | Per-task READMEs with design + lessons learned |
| Opus | [`Opus/`](Opus/) | G1 solution ships an empirical [`LEARNINGS.md`](Opus/G1_Upper_body_IK/LEARNINGS.md) and probe/alternate-strategy scripts |

## The tasks

| Task | Robot | Goal |
|---|---|---|
| `Isaac-AutoMate-Assembly-Direct-v0` | Franka | Insert an 8 mm plug into its socket (~0.11 mm clearance) |
| `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0` | Unitree G1 humanoid (fixed base, trihands) | Pick up a steering wheel and place it in a basket |

Both agents solve both tasks. Each solution is verified with the environment's
own success criterion (`check_plug_inserted_in_socket` for AutoMate,
`task_done_pick_place` for G1) and ships a success video.

## Layout

```
IsaacLab_Agentic/
├── Fable_5/
│   ├── README.md                          # Fable 5's top-level write-up
│   ├── prompt.txt                         # the original task briefs given to both agents
│   ├── AutoMate-Assembly-Direct-v0/
│   │   ├── insert_plug.py                  # main controller
│   │   ├── plug_insertion_success.mp4      # success video
│   │   ├── README.md, smoke_test.py, *.log, videos/
│   └── G1_Upper_body_IK/
│       ├── pick_place_wheel.py             # main controller
│       ├── g1_pick_place_success.mp4       # success video
│       └── README.md, smoke_test.py, *.log, videos/
└── Opus/
    ├── AutoMate-Assembly-Direct-v0/
    │   ├── insert_plug.py                  # main controller
    │   └── successful_insertion.mp4        # success video
    └── G1_Upper_body_IK/
        ├── solve_pickplace.py              # main controller (Strategy B: bimanual handoff)
        ├── strat_A_nudge.py                # alternate strategy (left-drag → right-push)
        ├── probe.py                        # reachability / frame probe
        ├── LEARNINGS.md                    # hard-won empirical findings
        └── successful_pickplace.mp4        # success video
```

## Prerequisites

- **Isaac Lab** (developed against the 2026-06 `main`, isaaclab 0.54.x /
  Isaac Sim 5.1) with a working Python environment.
- Linux + NVIDIA RTX GPU (tested on an RTX 5090, headless).
- Internet access on first run: the envs download their assets (AutoMate
  grasp/disassembly JSONs and OBJ meshes, G1 kinematics URDF, scene USDs) from
  the NVIDIA Nucleus servers into the **current working directory**.
- All Python dependencies (including `pin-pink` / `pinocchio` for the G1 task)
  are part of Isaac Lab's standard install — no extra packages needed.

## Setup

Place this folder under your Isaac Lab repository root:

```bash
cd /path/to/IsaacLab
git clone git@github.com:YilangLiu/IsaacLab_Agentic.git IsaacLab_Agentic
```

## Running

A few conventions for every command below:

- **Run from your Isaac Lab repo root** (the parent of this `IsaacLab_Agentic/`
  folder), because the envs download assets into the current working directory.
- `<python>` is the Python of your Isaac Lab environment. Either use the repo
  launcher `./isaaclab.sh -p <script>`, or call the interpreter directly. On the
  reference machine that is:

  ```bash
  VIRTUAL_ENV=/path/to/IsaacLab/env_isaaclab PYTHONUNBUFFERED=1 \
    /path/to/IsaacLab/env_isaaclab/bin/python -u <script> [args]
  ```

  `-u` / `PYTHONUNBUFFERED=1` matters: Isaac Sim's hard exit drops
  block-buffered stdout, so without it you lose the `RESULT:` line when
  redirecting to a file.

### Fable 5 solutions

Fable 5's scripts set headless mode internally and **record a video by default**
(pass `--no_video` to skip it). Videos land in each task's `videos/` subfolder.

```bash
# Task 1 — AutoMate plug insertion (Franka)
<python> -u IsaacLab_Agentic/Fable_5/AutoMate-Assembly-Direct-v0/insert_plug.py --num_envs 4
#   → "RESULT: 4/4 envs inserted successfully"
#   → video in IsaacLab_Agentic/Fable_5/AutoMate-Assembly-Direct-v0/videos/

# Task 2 — G1 humanoid pick-and-place (wheel → basket)
<python> -u IsaacLab_Agentic/Fable_5/G1_Upper_body_IK/pick_place_wheel.py
#   → "RESULT: SUCCESS"
#   → video in IsaacLab_Agentic/Fable_5/G1_Upper_body_IK/videos/
```

Approach: Task 1 is a vectorized `ALIGN → INSERT → SEAT → HOLD` state machine
with a compliant, rate-limited descent (spiral-search fallback for rim jams);
Task 2 is a 14-phase Pink-IK waypoint controller (hook-grasp + scoop, cross-body
carry, lower-release-retreat). See
[`Fable_5/README.md`](Fable_5/README.md) and the per-task READMEs for the full
design and gotchas.

### Opus solutions

Opus's scripts require `--headless` and `--video` to be passed **explicitly**
(video is off by default). The AutoMate script writes its video next to itself;
the G1 script's `--out_dir` defaults to a cwd-relative `Opus/G1_Upper_body_IK`,
so pass an explicit path when running from the repo root.

```bash
# Task 1 — AutoMate plug insertion (Franka)
<python> -u IsaacLab_Agentic/Opus/AutoMate-Assembly-Direct-v0/insert_plug.py \
    --num_envs 4 --headless --video
#   → "[RESULT] Insertion success: N/N envs ..."
#   → video in IsaacLab_Agentic/Opus/AutoMate-Assembly-Direct-v0/

# Task 2 — G1 humanoid pick-and-place (wheel → basket)
<python> -u IsaacLab_Agentic/Opus/G1_Upper_body_IK/solve_pickplace.py \
    --headless --video --out_dir IsaacLab_Agentic/Opus/G1_Upper_body_IK
#   → "================ RESULT ================" with SUCCESS
#   → video in IsaacLab_Agentic/Opus/G1_Upper_body_IK/
```

Approach: Task 1 is an analytic two-phase task-space controller that servos the
fingertip to the env's own `gripper_goal_pos` (align above, then descend);
Task 2 (Strategy B) is a bimanual handoff — the left hand drags the wheel to a
low `y`, the right hand corner-grips and carries it *raised* into the place
region for a clean release. `strat_A_nudge.py` is an alternate single-arm relay,
`probe.py` measures reachability, and
[`Opus/G1_Upper_body_IK/LEARNINGS.md`](Opus/G1_Upper_body_IK/LEARNINGS.md)
records what worked and what flung the wheel.

## Why Fable 5's G1 pick-and-place beats Opus's

Both controllers satisfy the env's `task_done_pick_place` check, but the
**rendered trajectories are not equal**: Fable 5 performs a clean pick-and-place,
while Opus shoves the wheel into the target region. The gap comes down to how each
agent handled the task's two hard constraints — the trihand can't lift a
center-gripped wheel without flinging it, and no single arm can reach both the
wheel *and* the basket.

- **A real pick vs. a table drag.** Fable 5 grasps the wheel with the **left hand
  only**, scoops it (curls the fingers mid-air so the table doesn't block the
  curl), clamps to a closed fist, and **lifts and carries it** into the basket —
  the right arm never moves the whole episode. Opus could not lift cleanly (per its
  own [`LEARNINGS.md`](Opus/G1_Upper_body_IK/LEARNINGS.md), a straight lift
  *flings* the wheel), so its left hand instead **drags the wheel flat across the
  table** to a handoff spot before anything is ever picked up.

- **One arm vs. a fragile two-arm handoff.** Because Opus can't finish with one
  arm, it runs a multi-step **bimanual handoff** (left pins the wheel down → right
  corner-grips it → left releases and parks far back to free the shared-waist
  null-space so the right wrist can extend). Two grasp/release cycles and an
  in-contact handoff add many more points where the wheel can slip or get knocked.
  Fable 5's single grasp/release cycle is far more robust.

- **Margin vs. knife-edge.** Opus only reaches the place region by exploiting that
  the right wrist extends ~10 cm further in +x when raised, carrying the wheel at
  the edge of reach saturation, and leans on a corner grip that "releases cleanly"
  — which its own notes flag as the *unsolved crux*. Fable 5 sets the wheel flat in
  the basket, then adds a corrective **NUDGE** to push it fully inside the success
  box (fixing the slide-back-off-the-rim failure), and holds the success criterion
  continuously for ~3.3 s with both arms retracted.

Net: Fable 5 *solved* the underlying physics (scoop to complete the curl mid-air,
a closed fist to survive the dangle), whereas Opus engineered *around* it (drag +
handoff + raised carry). The result is a busier, more contact-heavy, less
pick-and-place-looking motion — exactly what the video shows.

## Gotchas

- **AutoMate:** do **not** change `sim.render_interval` for this env — it breaks
  the env's scripted reset grasp. The first run downloads `plug_grasps.json`,
  `disassembly_dist.json`, `disassemble_traj.json`, `plug.obj`, `socket.obj`
  into the working directory, so run from the repo root.
- **G1:** the task needs `import pinocchio` *before* `AppLauncher`, the
  `enable_pinocchio` launcher flag, and an explicit import of the blacklisted
  `locomanipulation.pick_place` subpackage — all of which the scripts handle
  internally. The env has no randomization, so a run is deterministic on a given
  machine; a different GPU/driver can shift contact physics enough to change the
  wheel's final resting position by ~1 cm (see each agent's notes for the knob
  to retune if it lands short).
