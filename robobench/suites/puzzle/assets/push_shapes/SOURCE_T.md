# push_shapes T-block provenance

The T piece of `push_shapes` is the original T block of RoboDojo's `push_T` task (the X and L
pieces, the three pads and the recessed board are authored here by
`scripts/author_push_shapes_assets.py`). It comes from the official
[`RoboDojo-Benchmark/RoboDojo`](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)
dataset, whose dataset card declares the assets under Apache-2.0.

Task provenance: upstream
[`push_T.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/tasks/push_T.py)
and
[`push_T.yml`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/config/push_T.yml)
at RoboDojo revision `2184bf8844ea9d205382c4aefa3a694311418251`. `push_shapes` keeps that
task's per-piece rubric (7 mm / 7 degrees / no lift) verbatim.

The raw USDZ package is not checked in. Run `scripts/vendor_push_shapes_t_asset.py` against a
directory containing this exact download:

| Local filename | Dataset path | SHA-256 |
| --- | --- | --- |
| `t.usdz` | `Assets/Object/RoboDojo/Rigid/t/00000/object.usdz` | `98bae9ae3dddf67e58004695930c09f3d36f715dfa2bf5d65d1ed2b7179fa493` |

The tracked `block_t/source.usdc` is the package's unmodified `main.usdc` member. The small
`block_t/main.usda` overlay preserves the source visual while enabling its authored collision
mesh as a convex decomposition; the upstream file disables collision and requests dynamic SDF,
which is unsuitable for this small GPU rigid body. No visual geometry was recreated.
