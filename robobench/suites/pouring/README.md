# pouring suite — MPM liquids + bimanual Franka latte

Liquid-manipulation benchmark environments on IsaacLab develop's **Newton** backend: implicit-MPM
particle liquids (coffee + milk) poured between vessels by two Franka arms. The end goal is an
**agent-evaluation benchmark**, which drives every physics decision below: contacts must be real
(or explicitly documented as idealized) for every visible object pair, and metrics must not sit on
simulation artifacts.

> **Agent handoff note** — if you are a Claude Code agent picking this suite up on a fresh
> machine: this file is the canonical development status. Read it fully, then skim
> `coupled_manager.py`, `newton_sim.py`, `scenes/latte.py`, and the three bimanual smokes before
> changing anything. The "Landmine digest" section will save you days.

## Status

| Phase | What | Status | Verified |
|---|---|---|---|
| 1 | Scene-only scripted pour (kinematic vessels, MPM liquids) | DONE (`597b050`…`9409de4`) | `LATTE-POUR PASS` transfer 1.000 / retention 1.000 / spilled 0.000 |
| 2a | Bimanual **kinematic** Frankas (DiffIK writes joint state; arms are MPM colliders only, no rigid solver) | DONE (`1adb9fd`…`9409de4`) | `LATTE-BIMANUAL PASS` 0.355 / 0.645 / 1.000 / 0.000 |
| 2b | **Coupled MJWarp+MPM substrate — dynamic arms** (real gravity, actuator PD, MuJoCo rigid contacts; one-way rigid→fluid) | DONE (`3d18828`) | `LATTE-BIMANUAL-DYN PASS` ×6 runs (incl. a post-2c-a no-regression rerun): 0.376–0.400 transferred / 0.597–0.624 kept / 1.000 retention / ≤0.002 spilled; pour trigger reproducibly at ~91.6°; hand tracking 0.0–0.8 cm; combined MuJoCo+MPM CUDA graph captures cleanly |
| 2c-a | **Dynamic vessels + weld-at-grasp + concave rigid proxies** (free-joint vessels with authored mass, ring/slab/handle proxy shells as live MuJoCo geometry, MuJoCo equality welds engaged at the measured grasp pose) | DONE | `LATTE-BIMANUAL-WELD PASS` ×2 consecutive on the final (re-aimed) pour geometry: 0.473–0.500 transferred / 0.500–0.506 kept / 1.000 retention / ≤0.021 spilled; triggers 92.0–92.3°; weld tracks the script within ~0.3° through the whole 92° pour (real-tilt instrumented); welds engage at hand err 0.0 cm, vessels released upright (≤0.6°); njmax 600 holds. Before the re-aim, 2 of 4 full runs failed on CHAOTIC STREAM LANDINGS — see the landmine |
| 2c-b | **Force closure (ATTEMPTED — blocked by the substrate)**: scene `latte_grip` + pinch-grade gripper env + `latte_bimanual_grip_smoke` (slip observable, drop guard, contact probes) all live and honest — but mjwarp @ newton `811968b` cannot hold a static pinch: its CCD single-point contacts creep tangentially under load (measured: a 107–150 N/finger, μ=1, 3.7 mm-deep two-pad pinch lets a 3 N vessel slide out at ~15 mm/s — a ~100× Coulomb violation, invariant to kp 8k→20k, impratio 1→10, cone, bar shape/width, grasp depth/orientation, mesh vs analytic-box pads). Full dossier in the landmine digest. | 12 instrumented bring-ups; every layer root-caused |
| 2c-c | 1.5-way liquid→rigid feedback | after 2c-b unblocks — roadmap below | — |

## Architecture (Phase 2b + 2c-a)

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

### Phase 2c-a delta (scene `latte_weld`)

- **Vessels are FREE dynamic bodies** (authored mass: mug 0.30 kg / pitcher 0.25 kg via
  UsdPhysics MassAPI — the Newton importer keeps authored mass and scales computed inertia).
  They rest on the table and are carried by physics; nothing scripts their poses.
- **Concave rigid proxies** under `<vessel>/rigidproxy/`: a 10-box ring tracing the VISUAL outer
  wall (mug r 0.058, pitcher r 0.0425 — measured from the zup USD point clouds; the visuals run
  ~7 mm fatter than the interior collider shells), a cylinder floor slab (base–table contact),
  and the handle's outer vertical bar as a capsule (the grasp target). Geometry constants:
  `scenes/latte.py::MUG_PROXY/PITCHER_PROXY`.
- **Label-routed shape flags** (`coupled_manager._prepare_builder_for_finalize`): `/rigidproxy/`
  shapes → rigid-only; `handle` capsules → BOTH (milk must not pass the visual handle); every
  other shape on a proxy-carrying body (the concave interior trimesh) → MPM-only. The Phase 2b
  kinematic ghosting rule is untouched, so `latte_dyn` behaves exactly as before.
- **Weld-at-grasp**: disabled MuJoCo equality welds (hand ↔ vessel) are created at build from
  `MJWarpMPMSolverCfg.weld_specs`; `scene.weld_vessel()` → `manager.set_weld()` measures the
  CURRENT relative pose from `body_q`, writes it + enabled into the Newton model's
  `mujoco.equality_constraint_*` arrays and queues a `CONSTRAINT_PROPERTIES` notification — the
  solver refreshes `eq_data`/`eq_active` outside the captured graph, so engagement never snaps
  and works with `use_cuda_graph=1`.
- **Metrics track ACTUAL vessel poses** (`LatteWeldScene.post_step`), so drops and topples score
  honestly; the cylinder-mask metrics and tilt readouts are yaw-invariant by construction.

Key files:

- `coupled_manager.py` — `NewtonCoupledMJWarpMPMManager` + `MJWarpMPMSolverCfg` (manager selection
  via `class_type` on the solver cfg; see file docstring for every contract detail).
- `newton_sim.py` — `MpmSimCfg`: `coupled=False` → MPM-only manager (Phases 1/2a);
  `coupled=True` → the coupled manager. `_MJWARP_DEFAULTS`: njmax 600 / nconmax 300 (two arms).
- `scenes/latte.py` — scene + metrics; `latte` (MPM-only), `latte_dyn` (coupled, kinematic
  vessels) and `latte_weld` (coupled, dynamic vessels + welds) registrations; `MUG_PROXY` /
  `PITCHER_PROXY` measured proxy geometry.
- `configs/envs.py` — `pouring.latte.bimanual_franka.joint` (2a),
  `pouring.latte_dyn.bimanual_franka.joint` (2b) and
  `pouring.latte_weld.bimanual_franka.joint` (2c-a) — the dyn/weld rigs share the same knobs
  (gravcomp 1.0, arm effort 300, gripper 500).
- `scripts/latte_bimanual_smoke.py` (2a) / `scripts/latte_bimanual_dyn_smoke.py` (2b) /
  `scripts/latte_bimanual_weld_smoke.py` (2c-a) / `scripts/latte_bimanual_grip_smoke.py`
  (2c-b, blocked on the mjwarp pinch defect — scene `latte_grip`, env
  `pouring.latte_grip.bimanual_franka.joint`).

## How to run

Everything needs the **Newton venv** (`env_newton/`, uv, py3.12) backed by an **editable IsaacLab
develop checkout** (clone anywhere, tested @ d7d0042 — build recipe in the root README's
"Newton env" section) and a CUDA GPU (validated on an RTX 5090; MPM is GPU-only).
`OMNI_KIT_ACCEPT_EULA=YES` always.

```bash
# Phase 2c-a smoke (dynamic vessels + welds), headless (~15–20 min wall)
HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python \
    -m robobench.suites.pouring.scripts.latte_bimanual_weld_smoke

# Phase 2b smoke, headless (~15–20 min wall; five consecutive PASSes on record)
HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python \
    -m robobench.suites.pouring.scripts.latte_bimanual_dyn_smoke

# quick bring-up / debugging (readable stack traces, capped steps)
... latte_bimanual_dyn_smoke --sim use_cuda_graph=0 --max_steps 800

# record a video of the 2c-a scene — TWO STAGES (videos/ is GITIGNORED — local artifacts only).
# Live recording (scripts/record_video.py) CORRUPTS the coupled MPM physics on this stack: 5/5
# live attempts failed while headless passed (see the landmine). Stage 1: a headless PASS run
# dumps states; stage 2: replay-render them with NO live physics.
HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python \
    -m robobench.suites.pouring.scripts.latte_bimanual_weld_smoke \
    --dump_states robobench/suites/pouring/videos/weld_run_states.npz
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python scripts/replay_render.py \
    robobench/suites/pouring/videos/weld_run_states.npz \
    --video robobench/suites/pouring/videos/latte_bimanual_weld.mp4 \
    --eye 0.35 0.85 0.75 --target-at 0.05 -0.02 0.12

# live recording (2b-era recipe) — historically fine for latte / latte_dyn, DO NOT use for
# latte_weld (see landmine):
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python scripts/record_video.py \
    robobench.suites.pouring.scripts.latte_bimanual_dyn_smoke \
    --video robobench/suites/pouring/videos/latte_bimanual_dyn.mp4 \
    --eye 0.35 0.85 0.75 --target-at 0.05 -0.02 0.12
```

Expected verdict lines: `LATTE-BIMANUAL-WELD PASS | milk transferred ≈0.47–0.50 | kept ≈0.50 |
retention 1.000 | spilled ≤0.021 | vessels upright` (2c-a) and `LATTE-BIMANUAL-DYN PASS | milk
transferred ≈0.38 | kept ≈0.62 | retention 1.000 | spilled ≤0.002` (2b).

## Phase 2c roadmap (the TODO)

Goal: **contact-complete for benchmark use** — every meaningful contact between visible objects
physically modeled at the MuJoCo+MPM fidelity class. Build order chosen so each step is
independently verifiable:

- [x] **2c-a: dynamic vessels + weld-at-grasp + concave rigid proxies — DONE** (scene
  `latte_weld`; verified `LATTE-BIMANUAL-WELD PASS`, see Status). As-built deltas vs this plan:
  - Welds are MuJoCo **equality welds** created DISABLED at build (`weld_specs` on the solver
    cfg) and engaged at runtime with the measured relpose (`set_weld`) — no fixed joints, no
    kinematic-tree edits, CUDA-graph-safe.
  - Proxy geometry measured from the USD point clouds (mug visual outer 0.0577 vs 0.051
    collider; pitcher 0.0421 vs 0.035) — see `MUG_PROXY`/`PITCHER_PROXY`.
  - Clearance margin landed as `--pour_margin 0.018`; pour trigger moved 91.6° → 92.2°, verdict
    numbers stayed inside the 2b band.
  - Reset rework: welds deactivate FIRST in `LatteWeldScene.reset`; `BaseEnv.reset`'s
    scene-then-robot order is then harmless (no physics step runs between the writes), and the
    caller still runs `resync_collider_history()` after the full reset.
  - njmax 600 / nconmax 300 verified sufficient (zero nefc overflows across full runs).
  - Grippers close to bar-radius + 4 mm, NOT onto the bars — see the landmine below.
  - The pour was RE-AIMED for dynamic vessels after chaotic-landing failures (2 of 4 first
    full runs): lip anchor 1 cm deeper past the rim (−0.030), `--lip_clear` 0.032 → 0.026,
    recover untilt slowed to ~23°/s. Verified ×2 consecutive PASSes after. See the landmine.
- [ ] **2c-b: force closure replaces the welds — ATTEMPTED, blocked by the substrate** (see the
  Status row and the "mjwarp cannot hold a pinch" landmine; 12 instrumented bring-ups peeled
  back and fixed FIVE real layers first: grasp height in vessel-local vs world coords, the dead
  mesh×capsule pair, pad-edge bar escape, the soft-contact force cap, point-vs-line pad
  alignment). What EXISTS and works: `latte_grip` scene (no weld rows), pinch-grade env
  (gripper kp 20000 / damping 200 / effort 500, elliptic cone + impratio 10 via the smoke),
  wide stiff grip bars (`grip_w` + `mjc:solref` per-prim), analytic fingertip pad boxes
  (`finger_pads` → `_add_finger_pad_boxes`, AABB-derived, meshes stay MPM-only), and the grip
  smoke with slip/drop observability and `--contact_probe`. Two unblock paths, both substrate
  surgery: (a) rebuild the coupled manager on `use_mujoco_contacts=False` + Newton's own
  CollisionPipeline (proper contact manifolds; the manager currently forbids it — the MPM
  double-drive concern needs a redesign, not a flag); (b) bump the newton/mjwarp pin once its
  contact friction matures, then re-run `latte_bimanual_grip_smoke` as-is.
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
- **`root_link_quat_w` of a free trimesh body is YAWED vs its USD prim frame** on this pin (the
  importer's inertial-principal frame; measured 0.84 rad mug / 2.68 rad pitcher). The cylinder
  metrics and tilt readouts are yaw-invariant, so reads are safe for scoring — but NEVER mix
  read-back vessel poses with prim-frame scripted targets: the weld smoke anchors its ride-along
  offsets to scripted home poses for exactly this reason (a read-anchored offset threw the hand
  target 15–33 cm at weld engagement).
- **Fingers vs the real handle capsules (2c-a)**: never command pad interference while a vessel
  is free-standing — bar-radius + 1 mm plus ~1 cm tracking error pressed the bar and yawed the
  mug ~50° on its slab (yaw spills nothing, so it is INVISIBLE in the metrics until the carry
  fights it). The weld smoke closes to bar-radius + 4 mm; real pinches are Phase 2c-b's problem.
- **`set_weld` recipe** (runtime equality toggling under CUDA graph): write
  `model.mujoco.equality_constraint_relpose` (measured `inv(X_hand) ∘ X_vessel` from `body_q`,
  wp.transform, xyzw) + `_enabled`, then queue `SolverNotifyFlags.CONSTRAINT_PROPERTIES` — the
  solver's `_update_eq_properties` refreshes `eq_data`/`eq_active` outside the captured graph.
  Welds must exist (disabled) in the builder BEFORE finalize; they cannot be added later.
- **The surge-pour landing is CHAOTIC — treat single runs as anecdotes.** GPU physics is not
  bitwise deterministic (MPM atomics), and near the ~92° avalanche knee the outflow is a narrow
  fast stream whose free-fall drift (~2–8 cm) rivals the 9 cm mouth: touchdowns are
  ALL-OR-NOTHING. Observed across identical-code runs: clean PASS; full miss with 0.36 spilled;
  and a proxy-ring contact-lodge that held the welded pitcher at a sustained 26° equality
  violation (early avalanche at scripted 82°). Mitigations that made it robust: aim the lip
  DEEPER past the rim (−0.030, run-out room inside the mouth), cut the fall (`--lip_clear`
  0.026), slow the recover whip (23°/s — in-flight milk is ~20% of the pour and lands during
  recover). The smoke prints REAL vessel tilts (`real tilt M/P`) — scripted tilt is not truth;
  and any change to pour geometry needs ≥2 consecutive PASSes to count as verified.
- **Never rest a dynamic body on a CYLINDER proxy** (mjwarp): cylinder-box contacts route
  through CCD, whose contact points regenerate asymmetrically on a rotationally-symmetric
  penetrating face — the vessels crept across the table at a constant ~8 mm/s (5–8 cm of wander
  before the grasp, direction biased by the handle's CoM offset) and sank 2.5 mm, with NO
  external force and independent of friction (MuJoCo pair friction is the element-wise max, so
  the slab was never actually slippery — that first theory was wrong). Ruled out by
  discriminators: liquid (near-zero-fill run identical), arm coupling (1/25-speed arms
  identical). Fix: the floor slab is a BOX (inscribed square) — box-box gets the analytic
  4-corner manifold; drift fell from 5.5 cm to 40 µm and the sink vanished.
- **Grasp choreography on real handle capsules is TWO-STAGE**: pads stop at bar + 3 mm while
  the vessel is free (a +1 mm close pressed the bar through tracking error and yawed the free
  mug ~50°), then finish closing to bar + 0.5 mm during the first quarter of the lift — AFTER
  the weld engages, when contact can no longer displace the vessel relative to the hand. The
  grasp reads real on video; the weld still carries the load until 2c-b.
- **mjwarp @ newton `811968b` cannot hold a static PINCH — do not burn time tuning grasps on
  this pin.** Its collision is CCD-based across the board (`NATIVECCD/MULTICCD`), yielding
  SINGLE-POINT contact manifolds whose tangential friction CREEPS (viscous, never static) under
  articulated load: a verified 107–150 N-per-finger, μ=1 pair, 3.7 mm-deep two-pad pinch let a
  0.3 kg vessel slide out at ~15 mm/s against a 3 N load — ~100× beyond the Coulomb limit —
  INVARIANT to gripper kp (8k→20k), impratio (1→10), cone (pyramidal/elliptic), bar shape
  (capsule/box) and width (16→22 mm), grasp depth and orientation (side/top-down), and pad type
  (Franka mesh hulls vs analytic AABB boxes). Single-point manifolds also mean near-zero pivot
  resistance (vessels rotate to droop equilibrium in the pinch). Resting/leaning friction
  (slab-on-table) is fine — the defect bites actuated pinches. Probes that settle it fast next
  time: the grip smoke's `--contact_probe` (finger joint block, `qfrc_actuator`, live contact
  pairs + depths). Related: the slab-cylinder ratchet (above) is the same contact family
  misbehaving at rest. SHARPENING MEASUREMENT: segmenting each handle bar into 4 stacked boxes
  (one CCD contact point PER SEGMENT → a genuine 4-point planar manifold per pad, now the scene
  default) slowed the creep ~3× but did not stop it — the creep is PER-CONTACT and its rate
  scales ~1/contact-count, i.e. a friction-constraint defect, not a manifold-shape problem.
  Brute-forcing more segments trades perf for a slower loss; the fix is upstream.
- **LIVE video recording corrupts the coupled physics on `latte_weld`** (isaacsim 6.0.0.1): 5/5
  `record_video.py` attempts failed while identical headless runs passed — with graph ON the
  fill trigger shifted +3.5° off a 0.1°-tight headless baseline (a scheduling-sensitive race:
  RTX + graph replay), with graph OFF the trigger matched but landings still broke, and one run
  had the milk tunnel out of the world entirely. Record via REPLAY instead: the smoke's
  `--dump_states` (body_q + particle positions per frame, ~1.3 GB npz) + `scripts/replay_render.py`
  (writes states into the booted-but-never-stepped scene, `sync_transforms_to_usd()` + Fabric
  particle push per frame). The video then shows a verified PASS run bit-for-bit. 2b-era live
  recordings of `latte`/`latte_dyn` predate this finding and worked; scope unknown there.
- **Boot is nondeterministically flaky** (3 failures in ~12 boots, one underlying native-memory
  bug with three faces): (a) hang before solver init — last line is the cloner shape-color
  FutureWarning, ONE thread spinning ~100% CPU + ~140 threads in `futex_do_wait`, GPU never
  engaged; (b) same but fully idle; (c) `malloc(): unaligned tcache chunk detected` SIGABRT
  during the Newton->MJC conversion. No correlation found with flags or scene (headless and
  recording paths both affected). Remedy: kill the python (`pgrep -f <script>` — a shell wrapper
  hides it) and relaunch; babysit boots with a "no 'Initialize solver took' within 10 min" 
  watchdog. `sudo py-spy dump --pid <pid>` to finally catch the spin site.
