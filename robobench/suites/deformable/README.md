# Deformable suite (Newton backend)

Install + per-scene notes for the `deformable` suite (`tshirt`, `latte`, `knot`, `dumpling`). The rest of the
benchmark runs on the PhysX venv described in the root README; this suite needs its own `env_newton` venv.

The `deformable` suite (`robobench/suites/deformable/` — the four scenes `tshirt`, `latte`, `knot`,
`dumpling`, formerly the separate `folding` / `pouring` / `shoe_tying` / `dough` suites; the old
preset names still resolve as aliases) runs on IsaacLab **develop**'s Newton physics
backend (cloth, liquids, and rods do not exist on the PhysX stack). That branch is not on PyPI,
so this suite gets its own project-local venv, **`env_newton`** (Python 3.12, isaacsim 6.0,
torch cu130), with the isaaclab packages installed *editable* from an IsaacLab **develop**
checkout and one shared Newton engine pin. The assembly suite keeps using `.venv` (isaaclab
2.3.2 / PhysX); the two venvs coexist — only the interpreter you launch with differs.

Extra prerequisite: the torch cu130 wheels need an NVIDIA driver ≥ r580 (CUDA 13).

## 1. Clone IsaacLab (develop)

Clone anywhere you like — it is only consumed as an editable source tree (do **not** run
IsaacLab's own installer / `isaaclab.sh`):

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
git -C ~/IsaacLab checkout d7d004217c60b4790f721565bf5d40243addcb0e   # tested commit (develop, 2026-06-17)
```

Newer `develop` may work, but this commit is what the suite is tested against — develop moves
fast and breaks conventions vs 2.x (e.g. quaternions are **xyzw** there, not wxyz).

## 2. Build env_newton (one-time)

From the CoSiGen repo root, with `SRC` pointing at *your* checkout's `source/` dir:

```bash
SRC=~/IsaacLab/source                 # <-- adjust to your IsaacLab checkout
PY=env_newton/bin/python
# isaacsim deps span pypi.org + pypi.nvidia.com at different versions, and isaacsim pins some
# pre-release deps, so its installs take these extra flags. Keep NV an ARRAY expanded as
# "${NV[@]}" (works in bash and zsh) — a scalar NV="..." breaks in zsh, which does not
# word-split unquoted $NV and passes the whole string as one argument.
NV=(--extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow)

uv venv env_newton --python 3.12 --prompt env_newton
uv pip install --python "$PY" torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$PY" "${NV[@]}" "isaacsim[all,extscache]==6.0.0.1"   # large first download
uv pip install --python "$PY" "${NV[@]}" \
  -e "$SRC/isaaclab_newton[all]" -e "$SRC/isaaclab_physx[newton]" \
  -e "$SRC/isaaclab_ovphysx" -e "$SRC/isaaclab_visualizers[kit]" \
  -e "$SRC/isaaclab_contrib" -e "$SRC/isaaclab_assets" -e "$SRC/isaaclab"
uv pip install --python "$PY" imageio imageio-ffmpeg   # for record_video
uv pip install --python "$PY" -e .                     # robobench itself (declares no other deps)

# Newton engine — the pin ALL deformable scenes (tshirt, latte, knot, dumpling) run and are tested
# against. It is newer than the commit isaaclab_newton pulls transitively, so install it last:
uv pip install --python "$PY" \
  "newton[sim] @ git+https://github.com/newton-physics/newton.git@f420998186ec70bc39323ccc374bcb6c2be1d14f" \
  "warp-lang>=1.16,<1.17" "newton-usd-schemas>=0.4.1"
# -> newton 1.6.0.dev0, warp 1.16, mujoco + mujoco-warp 3.11, newton-usd-schemas 0.5

# and apply the small vendored compat patch to the IsaacLab checkout (newton 1.5 renamed a few
# APIs the pinned develop commit still uses):
git -C ~/IsaacLab apply scripts/isaaclab_newton16_compat.patch
```

Notes:

- All seven `-e` packages are required: `isaaclab_ovphysx`/`isaaclab_physx` are hard imports of
  isaaclab's app launcher, and `isaaclab_visualizers[kit]` drives rendering (the tshirt smoke
  defaults to the kit visualizer).
- Optional — only to run IsaacLab's in-tree reference tasks (e.g. `Isaac-Lift-Cloth-Franka-v0`),
  not needed by the suites:
  `uv pip install --python "$PY" "${NV[@]}" -e "$SRC/isaaclab_tasks" -e "$SRC/isaaclab_rl" -e "$SRC/isaaclab_ov"`

## 3. tshirt (cloth folding)

The `tshirt` scene (`robobench/suites/deformable/scenes/tshirt.py`) folds a T-shirt (VBD cloth) on the coupled
MJWarp+VBD substrate. The in-tree smoke is a simulation CAPABILITY CHECK, not a solution: on
the benchmark env the Franka pinches the shirt with its real fingers and lifts it clear of the
table (cloth-rise verdict). Any solution for it stays out of the benchmark tree, in the
gitignored `experiments/` workspace.

```bash
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python -m robobench.suites.deformable.smokes.tshirt_fold_smoke --headless
```

`OMNI_KIT_ACCEPT_EULA=YES` skips isaacsim 6's first-run EULA prompt in headless runs; on
isaaclab develop a run is headless unless a kit visualizer is requested (pass `--viz kit`; do
NOT combine with `--headless`, which force-disables visualizers).

## 4. latte (liquid pouring, same venv)

The `latte` scene (`robobench/suites/deformable/scenes/latte.py`) runs particle liquids (implicit **MPM**)
coupled with MJWarp rigid dynamics: two dynamic Frankas grasp both vessels and pour milk into
coffee. ONE registered env on this scene: `deformable.latte.bimanual_franka.joint` (the
benchmark: dynamic arms + dynamic vessels + auto-weld grasp contract + 1.5-way liquid
feedback). The in-tree smoke is a simulation CAPABILITY CHECK, not a solution: both Frankas
grasp the vessels through the scene's auto-weld contract and lift them (rise/upright/spill
verdicts). Any solution for it stays out of the benchmark tree, in the gitignored
`experiments/` workspace.

```bash
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python \
  -m robobench.suites.deformable.smokes.latte_pour_smoke --headless
```

Note: do **not** record COUPLED-substrate pouring runs with `scripts/record_video.py` live —
live rendering corrupts the coupled MPM physics on this stack. Record via `--dump_states`
(poses + particles to an `.npz`) plus offline replay (a replay renderer last exists at
`f8c101d`: `scripts/replay_render.py`).

## 5. knot (shoelace tying, same venv)

The `knot` scene (`robobench/suites/deformable/scenes/shoe_knot.py`) TIES a half knot from two initially
separate shoelaces — Newton *rods* (capsule chains + cable joints, standalone VBD/AVBD) rooted
at a sneaker's top eyelets — by moving their free ends through the classic four beats: cross
into a mid-air X (pinched by 20 N spring-finger pins), thread under the junction, cross again,
pull apart and seat on the tongue. Verdict, slack and pin-free: winding >= 140 deg on the knot
sections, >= 6 cross-lace contacts, knot z < 155 mm. Rods have no IsaacLab asset type, so the
scene injects them into the Newton `ModelBuilder` through the manager's per-world builder hooks
(the in-tree MPM asset's mechanism). ONE registered env on this scene:
`deformable.knot` (robot-less; roots anchored, the free ends are kinematic handles driven per
solver substep with closed-loop planning off the measured crossing):

```bash
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python \
  -m robobench.suites.deformable.smokes.knot_smoke --headless
```

Rendering is Kit **RTX**: the smoke spawns a textured visual shoe USD and syncs one visual
capsule prim per rod segment from `body_q` (the physics rod is prim-less). Record through the
standard harness:

```bash
env_newton/bin/python scripts/record_video.py \
  robobench.suites.deformable.smokes.knot_smoke \
  --video robobench/suites/deformable/videos/knot_smoke.mp4 \
  --eye 0.33 -0.31 0.40 --target-at 0.0 0.03 0.10
```

See `robobench/suites/deformable/docs/shoe_knot.md` for the full recipe and pass criteria.

## 6. dumpling (dough rolling, same venv)

The `dumpling` scene (`robobench/suites/deformable/scenes/dumpling.py`) runs **elastoplastic dough** (implicit MPM with
finite stiffness + von-Mises yield + full cohesion — Newton's "mud" recipe stiffened for shape
retention) coupled with MJWarp rigid dynamics. The task: ROLL THE DOUGH OUT — grasp the rolling
pin through the scene's auto-weld contract and flatten the ball into a thin, wide wrapper with
low sliding passes (`scene.success()` gates the rolled sheet plus conservation guards; pushing
is the MPM colliders' one verified dough transport — see `docs/dumpling.md`'s physics findings).
ONE registered env on this scene: `deformable.dumpling` (robot-less material tuning:
pure-MPM substrate, kinematic pin). The suite ships the task only — robot bindings and
solutions live in the gitignored `experiments/` workspace, which builds its own
`EnvCfg(scene="dumpling", robot=...)` on the coupled substrate.

Same coupled-substrate recording rule as pouring: never record live — dump states during the
run and replay offline via `scripts/replay_render.py` (local-only, last in-tree at `f8c101d`;
its latte-specific particle-key mapping needs one generalization for dough dumps: map each MPM
object's prim leaf, lowercased, to the same-named dump key — `dough`).

See `robobench/suites/deformable/docs/dumpling.md` for the scene, material notes, and pass criteria.
