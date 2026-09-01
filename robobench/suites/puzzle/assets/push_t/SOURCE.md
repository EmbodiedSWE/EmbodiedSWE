# Push-T asset provenance

These are the original T block and gray T target used by RoboDojo's `push_T` task. They
come from the official
[`RoboDojo-Benchmark/RoboDojo`](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)
dataset, whose dataset card declares the assets under Apache-2.0.

Task provenance: upstream
[`push_T.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/tasks/push_T.py)
and
[`push_T.yml`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/config/push_T.yml)
at RoboDojo revision `2184bf8844ea9d205382c4aefa3a694311418251`.

Raw USDZ packages are not checked in. Run `scripts/vendor_push_t_assets.py` against a
directory containing these exact downloads:

| Local filename | Dataset path | SHA-256 |
| --- | --- | --- |
| `t.usdz` | `Assets/Object/RoboDojo/Rigid/t/00000/object.usdz` | `98bae9ae3dddf67e58004695930c09f3d36f715dfa2bf5d65d1ed2b7179fa493` |
| `target_t.usdz` | `Assets/Object/RoboDojo/Geometry/t_cushion/00000/object.usdz` | `4823fa3cfaf1a41cd26dd29deca0f5549da20f1211284d5d944618701a2115bf` |

The tracked `source.usdc` files are the packages' unmodified `main.usdc` members. The
small `block/main.usda` overlay preserves the source visual while enabling its authored
collision mesh as a convex decomposition; the upstream file disables collision and requests
dynamic SDF, which is unsuitable for this small GPU rigid body. No visual geometry was
recreated.
