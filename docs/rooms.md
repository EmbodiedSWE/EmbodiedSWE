# Task rooms and asset downloads

Every built-in registered task configuration declares its default room in its suite's
`robobench/suites/<suite>/configs/envs.py`. This includes robot-free configurations, alternate
robots, food variants, and control modes. Building the configuration automatically loads its room.

| Tasks | Default room |
| --- | --- |
| All five PC assembly tasks; T-shirt folding; shoe knot tying | Simple Room |
| Bulb, bolt, and nut assembly; box transport; wheel carry; fruit delivery | Factory hall |
| Table and robot-arm assembly; tool packing; pen holder; shape pushing; object classification; block stacking | Factory corridor |
| Food slicing and dicing; clearing produce; fruits on a plate; egg carton; spatula; coffee; latte; dumpling | Kitchen |
| Syringe dosing | Chemistry lab |

The project has three distinct locations:

- **Task settings:** suite `configs/envs.py` files hold room assignments, placement, camera,
  visibility, and robot-specific adjustments. Related tasks share settings within their suite.
- **Shared loading code:** `robobench/core/rooms.py` fetches, validates, and spawns room geometry.
  It contains no task-to-room registry or task-specific placement rules.
- **Downloaded assets:** `robobench/assets/rooms/<room>/` holds room models, materials, and textures.
  This directory is ignored by Git and excluded from wheels. There is no `robobench/backdrops/` package.

```python
# After AppLauncher and robobench.discover():
cfg = ENVS.get("assembly.pc_gpu.franka.osc")()
env = cfg.build()                      # task's default room and camera
# In a separate run:
env = cfg.build(room=None)             # explicitly disable scenery
```

The smoke launcher uses the task's room by default; `--room none` disables it. Ad-hoc `EnvCfg`
instances start with `room=None`: use a registered configuration to get its task settings, or
supply your own room specification. A room can be a dictionary or a factory called with
`(env_cfg, scene, robot)` to align the room with the actual scene and robot configuration.
Factories return fresh dictionaries, so adjustments do not leak into other environments.

The viewport camera remains overridable through `env.sim.set_camera_view(...)`. For offscreen
previews, call `robobench.core.rooms.wait_for_room(env)` before reading the first image.
This renders loading frames without stepping physics.

Rooms contain no Physics/Physx schemas and attach after the task's simulation model is initialized.
Hiding task furniture changes visibility only; its collision geometry stays active. Each environment
gets a room at its own origin. Batched environments use enough spacing to prevent rooms overlapping.
T-shirt, knot, latte, and dumpling require Newton. The existing single-environment restrictions for
MPM tasks remain in force. Task settings determine whether the room's furniture or the original
workbench is visible; task collision geometry is preserved in either case.

## Downloads and cache layout

The first build fetches the task suite's asset group, shared tables, its robot models, and its room
from `EmbodiedSWE/robobench-assets`. Composite robots fetch the robot bundle to include sibling arm and
gripper dependencies. Room groups are approximately 7 MB (corridor), 23 MB (hall), 111 MB (Simple
Room), 323 MB (lab), or 1.46 GB (Kitchen).

`robobench/assets_manifest.json` records local paths, sizes, SHA-256 checksums, bundle checksums,
and a pinned Hugging Face revision. Its `path_mappings` translate local room paths to the existing
published paths. The downloader handles both individual files and legacy tar member names, writing
only the new local layout. Existing Hugging Face files and bundles do not need to be duplicated.

Downloads verify bytes before replacing files and use a process lock for concurrent first loads.
Verified local files are reused without contacting Hugging Face. Missing or corrupt offline assets
produce an actionable error. Registry discovery and listing never download assets.

The default destination is the checkout. For an installed wheel or another writable destination,
set the root before importing robobench:

```bash
export COSIGEN_ASSET_DIR="$HOME/.cache/cosigen/assets"
python -m robobench.scripts.smoke --env assembly.bulb.franka.osc --headless
HF_HUB_OFFLINE=1 python -m robobench.scripts.smoke --env assembly.bulb.franka.osc --headless
```

Rooms then live at `$COSIGEN_ASSET_DIR/robobench/assets/rooms/`. Source text dependencies shipped
with the package are mirrored alongside downloaded task files. Private datasets require
`hf auth login` or `HF_TOKEN`; public assets do not require authentication.

```bash
python -m robobench.scripts.fetch_assets                 # prefetch every group
python -m robobench.scripts.fetch_assets --prefix robobench/assets/rooms/factory
python -m robobench.scripts.fetch_assets --check          # verify the whole local collection
```

## Preparing and maintaining assets

Run preparation with USD Python bindings available. It copies collected rooms into
`robobench/assets/rooms/`, repairs local references, clears known missing metadata/texture
references, and authors a visual layer without physics schemas. The source collection is read only.

```bash
python -m robobench.scripts.prepare_rooms --source /path/to/source/CoSiGen
python -m robobench.scripts.repair_task_assets
python -m robobench.scripts.fetch_assets --update-manifest
python -m robobench.scripts.fetch_assets --check
python -m robobench.scripts.fetch_assets --upload
```

Uploads respect the manifest's local-to-remote mappings and retain published archive member names.
Matching remote bundles are reused even if their original tar is not staged locally. A successful
upload pins the resulting Hugging Face revision without changing the dataset's visibility. Verify a
clean-cache download before publishing code that refers to new assets.

`repair_task_assets` packages the chopping board's five textures with the cutting assets and replaces
unavailable shelf shaders with portable USD PBR materials. Authored data outside visual material
scopes is preserved. Use `--texture-source` for a separate collected Kitchen directory.
