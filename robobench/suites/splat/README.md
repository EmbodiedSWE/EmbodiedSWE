# splat suite

Sim proxies of Gaussian-splat captures: a kinematic table laid out like a scanned workcell, the
robot base at the origin, rendered photoreal through `gsworld` (robot link splats re-posed
from the sim every step, scanned table/scene splats static in the same frame). The default
layout reproduces the workcell the shipped Franka + Robotiq splat model was captured in, so it
drops in unchanged.

## Run

Registered envs: `splat.table`, `splat.table.franka_robotiq.joint`, `splat.table.franka.joint`.
The suite ships no smoke of its own — build an env like any other robobench env, or render a
rollout photoreal by wrapping any target with the recorder:

```bash
python scripts/record_video.py <target module> --splat auto --env splat.table.franka_robotiq.joint
```

## Layout

```
splat/
├── scenes/table.py        TableScene(+Cfg): layouts (top_z, top_xy, height), pedestal, CAMERAS
└── configs/envs.py        splat.table[.robot.mode] registrations
```

## Assets

None of its own: the robot USD lives in `robobench/robots/assets/franka_robotiq/`; the splat models (`assets/models/`) and
the demo workcell config (`assets/scenes/franka_robotiq_table/`) ship with `gsworld/assets/`;
real-scene splats are produced offline (see `gsworld/README.md`, "Model preparation is external").

## Future

- Task objects (RAM sticks, cups) as rigid bodies with their own models in the scene JSON.
- Layout presets per captured workcell (add to `TableSceneCfg.LAYOUTS`).
