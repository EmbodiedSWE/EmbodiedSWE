# pouring suite — MPM liquids + bimanual Franka latte

Liquid-manipulation benchmark environments on IsaacLab develop's **Newton** backend: implicit-MPM
particle liquids (coffee + milk) poured between vessels by two Franka arms. The end goal is an
**agent-evaluation benchmark**, which drives every physics decision below: contacts must be real
(or explicitly documented as idealized) for every visible object pair, and metrics must not sit on
simulation artifacts.

> **Agent handoff note** — if you are a Claude Code agent picking this suite up on a fresh
> machine: this file is the canonical development status. Read it fully, then skim
> `coupled_manager.py`, `newton_sim.py`, `scenes/latte.py`, and the two bimanual smokes before
> changing anything. The "Landmine digest" section will save you days.

## Status

| Phase | What | Status | Verified |
|---|---|---|---|
| 1 | Scene-only scripted pour (kinematic vessels, MPM liquids) | DONE (`597b050`…`9409de4`) | `LATTE-POUR PASS` transfer 1.000 / retention 1.000 / spilled 0.000 |
| 2a | Bimanual **kinematic** Frankas (DiffIK writes joint state; arms are MPM colliders only, no rigid solver) | DONE (`1adb9fd`…`9409de4`) | `LATTE-BIMANUAL PASS` 0.355 / 0.645 / 1.000 / 0.000 |
| 2b | **Coupled MJWarp+MPM substrate — dynamic arms** (real gravity, actuator PD, MuJoCo rigid contacts; one-way rigid→fluid) | DONE (`3d18828`) | `LATTE-BIMANUAL-DYN PASS` ×4 runs: 0.376–0.400 transferred / 0.597–0.624 kept / 1.000 retention / ≤0.002 spilled; pour trigger reproducibly at ~91.6°; hand tracking 0.0–0.8 cm; combined MuJoCo+MPM CUDA graph captures cleanly |
| 2c | Real vessel dynamics + grasp contact + liquid feedback | **NEXT** — roadmap below | — |

## Architecture (Phase 2b)

Two contact systems, never talking directly, coupled one-way through the shared Newton state:

- **Rigid↔rigid**: MuJoCo-internal contacts inside `SolverMuJoCo` (3 substeps at 1/600 s per
  tick). Participants: Franka links (self-collision on), table, floor, sunken ground plane.
  **Kinematic vessels are ghosted** — `coupled_manager.py::_prepare_builder_for_finalize` clears
  `COLLIDE_SHAPES` on kinematic bodies' shapes (masses KEPT: MuJoCo needs inertia on their auto
  free joints; immovability comes from the KINEMATIC flag's 1e10 armature).
- **Particle↔rigid**: `SolverImplicitMPM` (one step at 1/200 s per tick, reading post-rigid
  `body_q`) rasterizes every `COLLIDE_PARTICLES` shape as SDF boundary conditions with
  complementarity contact + Coulomb friction, collider velocities from backward finite
  difference. One-way enforced by `setup_collider(body_mass=zeros)` after construction.
- **Grasps are frame attachments** (Phase 2a pattern): vessel pose = hand_ACTUAL ∘ grasp_offset,
  written post-step. Handles are visual-only meshes; fingers touch nothing on the vessels.

Key files:

- `coupled_manager.py` — `NewtonCoupledMJWarpMPMManager` + `MJWarpMPMSolverCfg` (manager selection
  via `class_type` on the solver cfg; see file docstring for every contract detail).
- `newton_sim.py` — `MpmSimCfg`: `coupled=False` → MPM-only manager (Phases 1/2a);
  `coupled=True` → the coupled manager. `_MJWARP_DEFAULTS`: njmax 600 / nconmax 300 (two arms).
- `scenes/latte.py` — scene + metrics; `latte` (MPM-only) and `latte_dyn` (coupled) registrations.
- `configs/envs.py` — `pouring.latte.bimanual_franka.joint` (2a) and
  `pouring.latte_dyn.bimanual_franka.joint` (2b: gravcomp 1.0, arm effort 300, gripper 500).
- `scripts/latte_bimanual_smoke.py` (2a) / `scripts/latte_bimanual_dyn_smoke.py` (2b).

## How to run

Everything needs the **Newton venv** (`env_newton/`, uv, py3.12) backed by an **editable IsaacLab
develop checkout** (`~/research/IsaacLab-v6`, tested @ d7d0042) and a CUDA GPU (validated on an
RTX 5090; MPM is GPU-only). `OMNI_KIT_ACCEPT_EULA=YES` always.

```bash
# Phase 2b smoke, headless (~15–20 min wall; four consecutive PASSes on record)
HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python \
    -m robobench.suites.pouring.scripts.latte_bimanual_dyn_smoke

# quick bring-up / debugging (readable stack traces, capped steps)
... latte_bimanual_dyn_smoke --sim use_cuda_graph=0 --max_steps 800

# record a video (videos/ is GITIGNORED — local artifacts only)
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python scripts/record_video.py \
    robobench.suites.pouring.scripts.latte_bimanual_dyn_smoke \
    --video robobench/suites/pouring/videos/latte_bimanual_dyn.mp4 \
    --eye 0.35 0.85 0.75 --target-at 0.05 -0.02 0.12
```

Expected verdict line: `LATTE-BIMANUAL-DYN PASS | milk transferred ≈0.38 | kept ≈0.62 |
retention 1.000 | spilled ≤0.002`.

## Phase 2c roadmap (the TODO)

Goal: **contact-complete for benchmark use** — every meaningful contact between visible objects
physically modeled at the MuJoCo+MPM fidelity class. Build order chosen so each step is
independently verifiable:

- [ ] **2c-a: dynamic vessels + weld-at-grasp + concave rigid proxies.**
  - Vessels become dynamic (free joint, real mass); attach to the flange with a fixed-joint /
    equality weld at grasp time (FluidLab-style) — real carried-mass dynamics before grasp risk.
  - Add **concave outer collision proxies** (a single convex primitive would fill the open mouth
    and break the pour, which deliberately dips the pitcher lip inside the mug's mouth): a ring
    of 8–12 boxes/capsules tracing each vessel's outer wall + a floor slab + a **handle capsule**
    (doubles as the future grasp target). Size proxies to the **visual** surfaces (visuals run a
    few mm fatter than the physics shells: pitcher visual ≈36–40 mm vs 35 mm collider).
  - Flag routing in `_prepare_builder_for_finalize` via `builder.shape_label` (prim paths):
    interior trimesh → `COLLIDE_PARTICLES` only (today's rule); proxy prims (e.g. spawn under
    `…/rigidproxy/`) → `COLLIDE_SHAPES` only, exempt from the kinematic ghosting; handle capsule
    → BOTH flags (so poured milk stops passing through the visual handle).
  - Bump the early-pour clearance margin in `cup_target` (`0.005` → ~0.015–0.020): measured
    collider-collider clearance dips to ~0.1 mm at 13.7° tilt with the current trajectory — a
    visible graze on video, and a real contact once proxies exist.
  - Rework reset choreography: `BaseEnv.reset` runs scene-then-robot; with dynamic vessels held
    by welds this order interpenetrates for a step. Also call
    `NewtonCoupledMJWarpMPMManager.resync_collider_history()` after every teleporting reset
    (backward-FD collider velocity otherwise kicks the liquid at (jump/dt)).
  - Watch `nefc overflow, increase njmax` — ring-vs-ring + vessel-table resting contacts add
    rows; 600 has headroom but verify.
- [ ] **2c-b: force closure replaces the welds.** Grip the handle capsules with real friction +
  gripper effort (500 N budget already configured). Expect a tuning pass on
  impratio / friction cone / contact softness for a stable thin-bar pinch; slip, re-grasp, and
  drops become physically possible (that's the point, for a benchmark).
- [ ] **2c-c: 1.5-way liquid→rigid feedback.** Apply `collect_collider_impulses` into `body_f`
  each tick — the exact recipe is Newton's `examples/mpm/example_mpm_twoway_coupling.py`
  (force = impulse / MPM dt held across rigid substeps, and SUBTRACT the previously-applied
  force before the MPM step or contact impulses double-count). Then the pitcher weighs what it
  holds, empties as it pours, and sloshing perturbs the wrist. Known risk: stability with light
  vessels (mobility ∝ cell_volume/body_mass) — tune or clamp.
- [ ] **Benchmark plumbing:** expose contact reporting (the manager's `_contacts` buffer +
  contact-sensor path) as evaluation observables (vessel clangs, table scrapes); keep every
  metric on **particle counts, never fill height** (implicit MPM settles ~2× denser than seeded
  and keeps compacting — surface height is a lying observable an agent will exploit).

Explicitly out of scope at this fidelity class (document in the benchmark card, don't chase):
surface tension / wetting / adhesion, air phase, soft finger pads, solref-scale contact
penetration.

## Landmine digest (hard-won; verify before "fixing")

- **Quats are xyzw** everywhere on isaaclab develop (cfg `init_state.rot`, data layer, math
  utils). A wxyz identity mounts a robot upside-down and root-frame IK still looks correct.
- **Ground plane at z=0 breaks MJWarp** (phantom kN·m joint forces). The scene sinks it to
  −1.05 with a static floor box carrying the z=0 surface. Do not "simplify" this away.
- **Vessel ghosting is deliberate** (see Architecture). Un-ghosting without concave proxies
  re-creates convex-hull grinding during the pour.
- **Servo law**: never clamp the commanded joint target to the ACTUAL joints — that caps
  sustained speed at kp·Δ/kd = 0.2 rad/s and the recover untilt needs ~1 rad/s. The smoke uses a
  bounded-lead reference (slew `max_dq`/tick, lead ≤ 0.3 rad → ~1.5 rad/s at ≤120 N·m).
- **`FrankaRobot.JOINT_CONTROL_DT` is pinned to 1/200 in the dyn smoke** so `env.step` == one
  physics tick. Phase 2a ran control_period=4, so its labeled durations were 4× stretched —
  the dyn smoke's `--time_scale 4.0` default reproduces that validated cadence. Videos therefore
  play in wall time, and the pour-trigger clock jump leaves a still tail at the end.
- **Backward collider velocities + teleports**: any reset that jumps poses must be followed by
  `resync_collider_history()` or the liquid takes a one-tick (jump/dt) kick.
- **MPM liquid facts** (probe-verified): settles ~2× denser than seeded, level never rises;
  viscosity 0.1 avalanches out of deep vessels in ~1 s at high tilt (uncuttable), 3.0 is a
  controllable ooze (scene default); keep particles-per-cell at 2.0; walls need ≥2 grid voxels
  (5 mm wall + 1 mm margin at 3 mm voxels) or particles tunnel.
- **The mug/pitcher USD assets are visual-only** (physics APIs removed at spawn): their baked
  convex-decomposition collision LEAKS particles. Physics lives on the invisible watertight
  trimesh shells, `simplify_meshes=False`, `mesh_approximation="none"`.
- **Rendering**: develop pumps rendering through visualizers — recording needs
  `--enable_cameras` (the smokes flip defaults to headless+kit); Kit particle visuals need the
  usdrt/Fabric push path (`setup_particle_visuals`/`push_particle_visuals`) or liquids render
  frozen; the cubric monkey-patch in every smoke works around an isaacsim 6.0.0.1 IAdapter
  drift. Camera for the bimanual scene: eye (0.35, 0.85, 0.75), target (0.05, −0.02, 0.12) —
  front-high on the +y side; the older (0.22, −0.28, 0.65) camera sits inside the right arm's
  reach envelope.
- **Registry names have no variant slot** (`suite.scene.robot.mode`) — new substrates need a new
  scene name (`latte_dyn` pattern), and `sim_overrides` replaces dict fields wholesale (merge
  defaults inside `to_isaaclab`, as `_MJWARP_DEFAULTS` does).
