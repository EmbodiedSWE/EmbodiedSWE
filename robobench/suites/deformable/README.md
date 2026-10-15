# Deformable suite (Newton backend)

Install + per-scene notes for the `deformable` suite (`tshirt`, `latte`, `knot`, `dumpling`).

**On this branch (`isaacsim-6`, Isaac Sim 6.2 / Isaac Lab 3.0) the suite runs in the SAME `.venv`
as every other suite**: Isaac Lab 3.0 ships the PhysX backend (`isaaclab_physx`) and the Newton
backend (`isaaclab_newton`, newton 1.6.1) side by side, so the separate `env_newton` interpreter that
`main` (Isaac Sim 5.1 / Isaac Lab 2.3.2) needs does not exist here. Build the venv with
`./scripts/bootstrap_isaaclab_6.sh` (root README) and run the smokes below with `.venv/bin/python`.
No IsaacLab compat patch is needed: the newton 1.6 API renames `main` patches in are upstream in
Isaac Lab 3.0.

What moved versus the develop snapshot `main`'s `env_newton` was built on: VBD (`VBDSolverCfg`,
`NewtonVBDManager`) lives in `isaaclab_newton.physics`; the MJWarp+VBD coupling in
`isaaclab_contrib.custom_coupling`; the old `NewtonModelCfg` soft-contact / shape-material knobs are
`NewtonCfg.soft_contact_cfg` + `default_shape_cfg` (see `newton/cloth_sim.py`); the Featherstone+VBD
coupling was dropped upstream (`rigid_solver="mjwarp"` only). Quaternions handed to Isaac Lab are
xyzw on 3.0 — robobench cfgs stay wxyz and convert at the boundary (`robobench.core.isaaclab_compat`),
so the robot `base_rot` values in `configs/envs.py` are plain wxyz again.

## 3. tshirt (cloth folding)

The `tshirt` scene (`robobench/suites/deformable/scenes/tshirt.py`) folds a T-shirt (VBD cloth) on the coupled
MJWarp+VBD substrate. The in-tree smoke is a simulation CAPABILITY CHECK, not a solution: on
the benchmark env the Franka pinches the shirt with its real fingers and lifts it clear of the
table (cloth-rise verdict). Any solution for it stays out of the benchmark tree, in the
gitignored `experiments/` workspace.

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -m robobench.suites.deformable.smokes.tshirt_fold_smoke --headless
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
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python \
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
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python \
  -m robobench.suites.deformable.smokes.knot_smoke --headless
```

Rendering is Kit **RTX**: the smoke spawns a textured visual shoe USD and syncs one visual
capsule prim per rod segment from `body_q` (the physics rod is prim-less). Record through the
standard harness:

```bash
.venv/bin/python scripts/record_video.py \
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
