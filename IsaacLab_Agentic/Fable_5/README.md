# Fable_5 — Scripted solutions for two Isaac Lab manipulation tasks

Two Isaac Lab tasks solved with hand-scripted controllers (no RL training, no
teleop), each with executable code and a rendered video of the successful
trajectory:

| Task | Robot | Goal | Result | Video |
|---|---|---|---|---|
| [`Isaac-AutoMate-Assembly-Direct-v0`](AutoMate-Assembly-Direct-v0/) | Franka | Insert an 8 mm plug into its socket (~0.11 mm clearance) | 4/4 envs succeed | [`plug_insertion_success.mp4`](AutoMate-Assembly-Direct-v0/plug_insertion_success.mp4) |
| [`Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0`](G1_Upper_body_IK/) | Unitree G1 humanoid (fixed base, trihands) | Pick up a steering wheel and place it in a basket | SUCCESS (env criterion holds through episode end) | [`g1_pick_place_success.mp4`](G1_Upper_body_IK/g1_pick_place_success.mp4) |

Each task folder has its own README with the controller design and the
debugging lessons learned along the way.

## Prerequisites

- **Isaac Lab** (this code was developed against the 2026-06 `main`,
  isaaclab 0.54.x / Isaac Sim 5.1) with a working Python environment —
  e.g. installed via `./isaaclab.sh --install` into a uv/conda env.
- Linux + NVIDIA RTX GPU (tested on an RTX 5090, headless).
- Internet access on first run: the envs download their assets (AutoMate
  grasp/disassembly JSONs and OBJ meshes, G1 kinematics URDF, scene USDs)
  from the NVIDIA Nucleus servers.
- No display needed — both scripts run headless and record video offscreen.

All Python dependencies (including `pin-pink`/`pinocchio` for the G1 task) are
part of Isaac Lab's standard install.

## Setup

Place this folder under your Isaac Lab repository root:

```bash
cd /path/to/IsaacLab
git clone git@github.com:YilangLiu/Fable_5.git Fable_5
```

Below, `<python>` is the Python of your Isaac Lab environment (for a uv env
created by `./isaaclab.sh -u`: `env_isaaclab/bin/python`; alternatively use
`./isaaclab.sh -p` as the launcher).

## Task 1 — AutoMate plug insertion (Franka)

```bash
cd /path/to/IsaacLab   # run from the repo root (assets download into the cwd)
<python> -u Fable_5/AutoMate-Assembly-Direct-v0/insert_plug.py --num_envs 4
```

A vectorized ALIGN → INSERT → SEAT → HOLD state machine servos the grasped plug
onto the socket axis and performs a compliant rate-limited descent (spiral-search
fallback for rim jams). Success is checked with the env's own
`check_plug_inserted_in_socket`. Expected output: `RESULT: 4/4 envs inserted
successfully`, with the video written to
`Fable_5/AutoMate-Assembly-Direct-v0/videos/`.

Notes:
- The first run downloads `plug_grasps.json`, `disassembly_dist.json`,
  `disassemble_traj.json`, `plug.obj`, `socket.obj` into the current working
  directory — run from the Isaac Lab repo root.
- Do **not** change `sim.render_interval` for this env (see the task README:
  it breaks the env's scripted reset grasp).

## Task 2 — G1 humanoid pick-and-place (steering wheel into basket)

```bash
cd /path/to/IsaacLab
<python> -u Fable_5/G1_Upper_body_IK/pick_place_wheel.py
```

A 14-phase waypoint controller over the env's 28-dim Pink-IK absolute action
space: radial approach, hook grasp of the wheel rim with a scoop maneuver
(lift-while-curl), cross-body carry with a mid-carry wrist yaw, lower into the
basket, release + nudge, retreat. Success is evaluated with the env's exact
`task_done_pick_place` criterion (the env's success *termination* is disabled so
the full place-and-retreat completes before the episode ends — see the task
README for why). Expected output: `RESULT: SUCCESS`, with the video written to
`Fable_5/G1_Upper_body_IK/videos/`.

Notes:
- The script internally handles three setup quirks this task needs
  (`import pinocchio` before `AppLauncher`, the `enable_pinocchio` launcher
  flag, and an explicit import of the blacklisted
  `locomanipulation.pick_place` subpackage) — no extra flags required.
- The env has no randomization, so the run is deterministic on a given
  machine. The wheel's final resting position clears the success box by ~1 cm;
  if a different GPU/driver shifts contact physics enough to land it short,
  deepen the wheel-center target in `t_carry2` (`carry_wrist_xy`, currently
  aimed at `(0.53, 0.43)`).

## Repository layout

```
Fable_5/
├── AutoMate-Assembly-Direct-v0/
│   ├── insert_plug.py               # main controller
│   ├── plug_insertion_success.mp4   # success video
│   ├── smoke_test.py, probe_reset.py, *.log, videos/
│   └── README.md                    # design + gotchas
└── G1_Upper_body_IK/
    ├── pick_place_wheel.py          # main controller
    ├── g1_pick_place_success.mp4    # success video
    ├── smoke_test.py, *.log, videos/
    └── README.md                    # design + 8 debugging lessons
```
