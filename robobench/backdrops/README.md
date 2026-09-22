# Backdrops and asset downloads

This package manages shared room scenery across task suites. `__init__.py` selects and places
rooms, while `presets.json` stores room assignments, transforms, cameras, and visibility settings.
Keeping this shared code here lets multiple tasks reuse a room without duplicating scene logic.

The local `assets/` directory holds downloaded room models, materials, and textures. It is ignored
by Git and excluded from wheels; these files come from Hugging Face. With `COSIGEN_ASSET_DIR`,
they instead live under that cache root at `robobench/backdrops/assets/`.

`EnvCfg.build()` selects default room scenery for these scene/robot pairs (all their control modes):

| Scene | Robot | Room |
| --- | --- | --- |
| `tshirt` | Franka | Simple Room |
| `pc_motherboard_gpu_ram`, `pc_gpu`, `pc_gpu_ram`, `pc_motherboard`, `pc_ram` | Franka | Simple Room |
| `ikea_table`, `so101` | Bimanual Franka | Factory corridor |
| `tool_packing` | Franka | Factory corridor |
| `bulb` | Franka | Factory hall |
| `box_to_bin`, `wheel_carry` | G1 | Factory hall |
| `slice` (including banana variants) | Franka | Kitchen |
| `spatula` | Franka | Kitchen |
| `egg_carton` | G1 | Kitchen |
| `latte` | Bimanual Franka | Kitchen |
| `syringe` | Franka, bimanual Franka | Chemistry lab |

Other scene/robot pairs keep their existing presentation. Explicit presets use the keys in
`presets.json`; each placement is aligned to its corresponding task and robot layout.
T-shirt and latte still require the Newton environment. Latte remains single-environment only,
as required by its existing MPM solver.

```python
# After AppLauncher and robobench.discover():
cfg = ENVS.get("assembly.bulb.franka.osc")()
env = cfg.build()                      # default room and viewport camera
# In a separate run:
env = cfg.build(backdrop=None)         # bare task
```

The smoke launcher accepts `--backdrop auto`, `--backdrop none`, or an explicit preset name.
The viewport camera remains overridable through `env.sim.set_camera_view(...)`.
For offscreen previews, call `wait_for_backdrop(env)` before reading the first image. This renders
loading frames without stepping physics.

Rooms are prepared without Physics/Physx schemas and attached after the task's simulation model is
initialized. Hiding an overlapping task table changes visibility only; its collision geometry stays
active. Each environment gets a room at its own origin. Multiple environments use spacing large
enough for the rooms, so their world origins differ from the bare-scene layout.

## Downloading and caching

The first build fetches the scene suite's asset group, shared table assets, its robot models, and its
selected room from `CoSiGen/robobench-assets`. Attached composite robots fetch the robot bundle to
include their sibling arm/gripper dependencies. Room groups are approximately 7 MB (corridor),
23 MB (hall), 111 MB (Simple Room), 323 MB (lab), or 1.46 GB (kitchen).

`robobench/assets_manifest.json` records paths, sizes, SHA-256 checksums, bundle checksums and the
published Hugging Face revision. A successful maintainer upload pins that revision. Downloads use
that revision and verify bytes before replacing each local file. A file lock coordinates concurrent
first loads. Once materialized and verified, assets are reused without contacting Hugging Face.

The default destination is the checkout, preserving the existing relative asset layout. For an
installed wheel or another writable destination, set the root before importing robobench:

```bash
export COSIGEN_ASSET_DIR="$HOME/.cache/cosigen/assets"
python -m robobench.scripts.smoke --env assembly.bulb.franka.osc --headless
# Reuse the materialized cache without network access:
HF_HUB_OFFLINE=1 python -m robobench.scripts.smoke --env assembly.bulb.franka.osc --headless
```

Source text layers shipped in the package are mirrored into this root alongside downloaded files.
Missing or corrupt offline assets produce an actionable error. Authentication is required only if
the dataset is private (`hf auth login` or `HF_TOKEN`).

Manual prefetch remains available:

```bash
python -m robobench.scripts.fetch_assets                               # all groups
python -m robobench.scripts.fetch_assets --prefix robobench/backdrops/assets/factory
python -m robobench.scripts.fetch_assets --check                       # verify all
```

## Preparing and publishing assets

Run preparation with USD Python bindings available. It copies collected room
assets into the destination, repairs external references where a local equivalent exists, clears
the known missing metadata/texture references, and authors a layer that removes all physics schemas.
The source checkout is read only. The complete dependency tree is audited before bundling.

```bash
python -m robobench.scripts.prepare_backdrops --source /path/to/original/CoSiGen
# For assets predating the portable shelf/board repairs (safe to rerun):
python -m robobench.scripts.repair_task_assets
python -m robobench.scripts.fetch_assets --update-manifest
python -m robobench.scripts.fetch_assets --check
# Publication: uploads files and bundles into the existing dataset, then pins the manifest.
python -m robobench.scripts.fetch_assets --upload
```

Publish assets and verify a clean-cache download before publishing the code/manifest that refers
to them. Prepared room files and bundles are ignored by Git and excluded from wheels; Git contains
the loader, presets, preparation script, and manifest. Preserve the existing dataset visibility.

`repair_task_assets` packages the five collected chopping-board textures into the cutting group
and replaces the shelf's unavailable legacy MDL shaders with portable USD PBR shaders. It verifies
that authored data outside the visual material scopes is unchanged. Use `--texture-source` to point
at an original collected Kitchen directory if the prepared room is unavailable. These are maintainer
steps; end users download the already repaired assets.

The single-arm syringe binding mounts its Franka on the existing north side table at `(0, .42, .71)`,
facing south over the medical cart. This placement is used with and without a backdrop. The lab's
visual pedestal follows that mount. The bimanual binding keeps its original poses.
