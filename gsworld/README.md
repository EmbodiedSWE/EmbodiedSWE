# gsworld — Gaussian-splat rendering for robobench envs

Photoreal RGB for any robobench env, GSWorld style: the sim keeps physics, contacts and
proprioception; the camera image is replaced by a 3D-Gaussian-Splatting render in which the robot's
gaussians are re-posed link by link every step from the articulation's live poses, and static
splats (a scanned table, scene, objects) sit in the same metric frame.

**The unit is an object, one PLY each** (as in GSWorld's model list): a robot scan with a
per-gaussian `semantics` label (= link index), a table, a room. Nothing is stored per link.

```
robot.ply + _poses.json ──►  SplatModel (link-local on load)  ──►  posed from sim link poses ─┐
table.ply + _poses.json, cup.ply + _poses.json  ──►  SplatModel  ──►  posed from sim bodies ───┼──►  gsplat  ──►  rgb, alpha
PinholeCamera (Kit viewport / a real camera)  ───────────────────────────────────────────────┘        │
sim render with the splat-covered links hidden  ──────────────────────────────────────────►  alpha-over  ──►  composite
```

**Optional by design.** Nothing in `robobench.core`, the robots or the suites imports this package;
envs run identically without it (`splat.table.franka.joint` needs no splat at all). Enable it with
`pip install -e .[gs]` (gsplat + scipy + imageio). Everything heavy (`torch`, `gsplat`,
Isaac) is imported inside functions, so `import gsworld` is app-free and a missing `gsplat` fails
only at render time with an install hint.

## Quick start

```bash
pip install -e .[gs]                                   # gsplat + scipy + imageio; robobench unchanged

# photoreal video of any robobench target: wrap it with the recorder and add --splat
python scripts/record_video.py <target module or script> --splat auto --splat-side-by-side \
    --video output/task.mp4 --env splat.table.franka_robotiq.joint
#   left half of the video = sim, right half = photoreal
```

`--splat auto` renders the env's robot from its shipped model and keeps everything else
raytraced; `--splat path/to/scene.json` composes a full gsworld scene (robot + table/scene
models) instead.

Programmatic use:

```python
from gsworld import assets
from gsworld.config import SplatSceneCfg

senv = SplatSceneCfg.load(assets.scene("franka_robotiq_table")).build()   # env + models + camera
out = senv.step(action)                 # out["rgb"] photoreal composite, out["sim_rgb"], out["alpha"]

# or decorate an env you built yourself
from gsworld.model import SplatModel
from gsworld.wrapper import SplatEnv
senv = SplatEnv(env, robot_splat=SplatModel(*assets.robot_model("franka_robotiq")),
                object_splats={"table": SplatModel(*assets.model_files("table"))})
senv.attach_viewport_camera(eye=(1.9, -1.6, 1.1), target=(0.45, 0.05, 0.15))
```

## Scene JSON — one file per env

`gsworld.config` is GSWorld's config idea extended to the whole env; `SplatSceneCfg.load(path).build()`
returns a ready `SplatEnv`. The shipped `assets/scenes/franka_robotiq_table/scene.json`:

```json
{
  "env":     {"scene": "table", "robot": "franka_robotiq", "control_mode": "joint",
              "scene_cfg": {"layout": "gsworld_table"}, "env_spacing": 3.0},
  "robot":   {"ply": "../../models/franka_robotiq/franka_robotiq.ply",
              "poses": "../../models/franka_robotiq/franka_robotiq_poses.json"},
  "static":  [],
  "objects": [{"name": "table", "ply": "../../models/table/table.ply",
               "poses": "../../models/table/table_poses.json"}],
  "camera":  {"eye": [1.9, -1.6, 1.1], "target": [0.45, 0.05, 0.15], "size": [960, 600]}
}
```

- `env` — any registered robobench scene / robot / control mode; `scene_cfg` / `robot_cfg` are the
  cfg dataclass fields as plain JSON (lists become tuples).
- `robot` — the robot model; `objects` — models riding a sim body (`env.iscene[name]`), re-posed
  every step (GSWorld's per-object transforms — a table is one too, even if it never moves);
  `static` — plain PLYs fixed in the sim frame (backgrounds); `camera` — the viewport view.
- Models live once in `gsworld/assets/models/<object>/` (`gsworld.assets.models()`); a scene JSON only
  references them. Adding an env = adding `gsworld/assets/scenes/<name>/scene.json`
  (`gsworld.assets.scenes()`). The `env` block is optional when the JSON only decorates an env built
  elsewhere (`SplatEnv.from_config(env, path)`, `scripts/record_video.py --splat path`).

## Models

A model is **one object = one PLY + one `_poses.json`** (`gsworld/model.py`):

- `<name>.ply` — the scan in the metric sim frame. A per-gaussian float property `semantics`
  gives the link index (robots); absent = a single body (tables, cups, room pieces).
- `<name>_poses.json` — where every sim body the model rides on *was* at scan time:
  `body_names`, `pos`, `quat` (wxyz), `link_names` (label index → body name), `qpos`, `source`.

`SplatModel(ply, poses)` moves each link's gaussians into that link's frame on load and re-poses
them from the live sim (`T_link_now @ local`) — GSWorld's `sim2gs @ link_now @ inv(link_scan)` with
the alignment already applied. A robot has many links (its articulation bodies); a table has one
(its kinematic rigid body), so it is manipulable the same way even if it never moves. Links the sim
lacks are dropped (partial model). `GaussianSet.to_ply(path, semantics=...)` writes the format.

## Model preparation is external

`gsworld` only renders. Producing a model — scanning, segmenting the robot per link, aligning a
scan to the sim frame, dumping the sim's body poses at the scan configuration — is done outside the
repo (GSWorld's real2sim tooling). What a model must look like to be loadable is fully specified in
`gsworld/model.py`: one PLY (optional `semantics` per gaussian) + one `_poses.json`.

## Layout

```
gsworld/
├── splat.py        GaussianSet: PLY I/O (3DGS/2DGS, semantics), similarity/rigid transforms, crops
├── model.py        SplatModel: any object (PLY + _poses.json) -> link-local -> re-posed from sim bodies
├── camera.py       PinholeCamera from a USD camera prim (Kit fit: fy = fx) or an Isaac Lab Camera
├── renderer.py     gsplat rasterization + alpha compositing
├── wrapper.py      SplatEnv: env wrapper returning splat RGB per step
├── config.py       scene JSON: env + robot model + object models + static PLYs + camera
├── assets.py       lookup of shipped models/scenes (assets/models/<object>/, assets/scenes/<workcell>/)
└── assets/         models/<object>/ (the PLY library: franka_robotiq, table, …); scenes/<workcell>/scene.json
```

CPU-only checks live in the repo's `tests/test_gsworld.py` (`pytest tests/` or `python tests/test_gsworld.py`).

## Conventions & gotchas

- Frames: sim = robot base at origin, meters, z-up; splat frames are arbitrary similarity frames.
  Quaternions are wxyz everywhere; cameras are OpenCV (+Z forward, +Y down) — USD/Kit cameras are
  converted, and Kit fits the *horizontal* aperture so `fy == fx` (using the vertical aperture
  squashes the render by the aspect ratio).
- 2DGS PLYs (two scales) load as flat 3D gaussians; fine for rendering, not for training.
- A scan is only clean near its capture viewpoints — far-off cameras see the dark "shell";
  crop static splats to the useful volume (`crop` in the scene JSON).
- The composite plate hides only the splat-covered links, so uncovered links, the hand and scene
  objects stay raytraced (mask-composite); `hide_robot_in_plate=False` for full-scene
  splats that already cover everything.

## Future

- Wrist-camera rendering (`PinholeCamera.from_isaaclab_camera` on a mounted `Camera` sensor).
- Dynamic object models bound to rigid objects (`GaussianSet.posed` per step).
