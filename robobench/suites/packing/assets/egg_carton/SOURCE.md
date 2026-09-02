# Egg-carton asset provenance

These are the original authored/scanned assets used by RoboDojo's
`fill_egg_holder` task. They come from the official
[`RoboDojo-Benchmark/RoboDojo`](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)
dataset, whose dataset card declares the assets under Apache-2.0. CoSiGen itself also ships
under Apache-2.0; see the repository-root `LICENSE`.

Task provenance: upstream
[`fill_egg_holder.py`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/tasks/fill_egg_holder.py)
and
[`fill_egg_holder.yml`](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/config/fill_egg_holder.yml).

Raw USDZ packages are not checked in. Run `scripts/vendor_egg_carton_assets.py` against a
directory containing these exact downloads:

| Local filename | Dataset path | SHA-256 |
| --- | --- | --- |
| `egg_holder.usdz` | `Assets/Object/RoboDojo/Articulation/egg_holder/00000/object.usdz` | `417f7b067e13f5371c84164daf2d597dd29b7d96f7af8c3952758695e400f830` |
| `egg.usdz` | `Assets/Object/RoboDojo/Rigid/egg/00000/object.usdz` | `867733884c64ee06b16374c7228e328824a16538d780f2ed2f61b889a8fc872d` |
| `egg_basket.usdz` | `Assets/Object/RoboDojo/Geometry/egg_basket/00000/object.usdz` | `76e5e07252e70f39b6b07798369a5f29a470ed1d87caec323a5300f43e112b77` |

The vendor script only verifies and extracts the packages. Extraction preserves the source
USD hierarchy and relative material/texture paths and ensures every tracked file stays below
GitHub's 100 MiB single-file limit.
