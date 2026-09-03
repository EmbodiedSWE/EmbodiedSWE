# shoe_tying suite

Rod-based shoelace manipulation on IsaacLab develop's **Newton** backend — the first robobench
suite running Newton *rods* (capsule-chain bodies joined by cable joints, solved by VBD/AVBD).
Ported from the standalone-newton `shoe_tying_knot.py`, keeping geometry, materials,
choreography, and pass criteria unchanged.

Scene: **`knot`** — TIE a half knot from two initially separate laces. The 30 cm sneaker sits
on a gray box table (ground sunk to z=-1.05, table top at `surface_z=0.2`, folding's ambient
layout). Both laces begin laid out on the shoe and the table, one per side, not intertwined
(the only crossing is the lacing criss-cross at the eyelet roots, continued visually by two
static bridge segments). Both roots stay anchored at the top eyelets; the two free ends are
kinematic handles that perform the classic four beats, with closed-loop planning (the mid-air
X is measured from the live strands and the thread/cinch are planned from that measurement):

1. **CROSS** — lift both ends taut (~96 % of each lace from its root) into a touching mid-air
   X, lace 1's strand over lace 2's, pinched by two 20 N spring-finger pins;
2. **UNDER** — lace 1's end threads the tunnel beneath the junction;
3. **CROSS AGAIN** — the end rises just past junction level so the strands cross a second time;
4. **PULL APART + SEAT** — antiparallel pulls to the flanks jam the crossing while the pins
   carry the knot onto the tongue pad, release under the held tension, and the ends slacken.

Registered envs:

- `deformable.knot` (legacy alias `shoe_tying.knot`) — robot `null`: the handles are driven
  straight through the Newton manager, per solver substep (standalone VBD manager).
- `deformable.knot.aloha.joint` (alias `shoe_tying.knot.aloha.joint`) — bimanual WidowX 250 6DOF (the classic Interbotix ALOHA
  arms; the OFFICIAL mujoco_menagerie `trossen_wx250s` model converted to USD — see
  `robobench/robots/wx250s.py`) flanking the shoe on the nut-thread task's lab-table
  workbench (cross-suite asset reference, see `configs/envs.py`; bases at x = ±0.42), arms
  by direct joint position targets, on the suite's PROXY-COUPLED substrate
  (`newton/lace_coupled_manager.py`): SolverMuJoCo owns the arms, SolverVBD the rods, and the finger
  bodies are proxied into the rod solve (newton's `example_franka_cable_ik_pick_place`
  recipe), so finger-lace contact is real contact. The scene keeps only the eyelet roots
  anchored (`kinematic_ends=False`); both fingers are actuated with the right mirroring the
  negated left, and the `wx250s_newton.usda` overlay carries the Newton-side asset fixes
  (drive gains, finger travel, pad colliders).

## Run

```bash
# full smoke: settle -> lift -> x-form -> tuck -> pull -> slack hold (headless physics + verdict)
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python -m robobench.suites.deformable.smokes.knot_smoke --headless

# ALOHA bimanual capability smoke: both arms lift the lace free ends off the table
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python -m robobench.suites.deformable.smokes.aloha_lift_smoke --headless

# author + validate the lace curves only (self/inter-lace clearance, shoe SDF), no sim
... knot_smoke --headless --validate-only

# RTX video (records the Kit RTX viewport through scripts/record_video.py)
env_newton/bin/python scripts/record_video.py \
    robobench.suites.deformable.smokes.knot_smoke \
    --video robobench/suites/deformable/videos/knot_smoke.mp4 \
    --eye 0.33 -0.31 0.40 --target-at 0.0 0.03 0.10
```

Runs under `env_newton` (root README, "Newton env" — the same newton pin as the folding and
pouring suites).

## Pass criteria (printed at run end)

Verdict sampled through the slack-and-pin-free check window (13.8–14.8 s), knot sections only
(past the root criss-cross):

- settled cross-lace contacts < 3 (the laces start NOT intertwined);
- **winding >= 140 deg** of lace 1 around lace 2's local axis (failures unwind to ~0 once the
  pins release — the topological proof);
- **>= 6 cross-lace contact pairs** while slack (snugness);
- **knot z < 155 mm** (seated on the tongue; the mid-air junction sits at 165–190);
- positions/velocities finite throughout, no lace sinks through the table top.

Reference runs (RTX 5090, 2026-08-17): winding 337–430 deg, 23–29 contacts, seat z 97–118 mm —
`videos/knot_smoke.mp4` is an RTX-recorded PASS.

`aloha_lift_smoke` gates instead on both lace end sections still >= 60 mm above the table at
the END of the hold, laces intact, states finite. Reference runs (RTX 5090, 2026-08-31, WidowX
250 on the lab-table layout, 3/3): end heights +125..149 / +136..138 mm —
`videos/aloha_lift_smoke.mp4` is an RTX-recorded PASS.

## Layout

Physics and rendering are split (rods have no IsaacLab asset type):

- `scenes/shoe_knot.py` — the scene: lace curve authoring + validation, knot metrics, and the
  per-world builder hook that injects the physics (table twin, invisible shoe collision
  trimesh, bridge capsules, both lace rods) into the Newton `ModelBuilder`.
- `smokes/knot_smoke.py` — the tying choreography (`KnotDriver`, ported 1:1 from the
  standalone script). The RTX visual layer (`LaceVisuals`: one USD capsule per rod segment,
  synced from `body_q`) lives in the scene module — both smokes use it.
- `smokes/aloha_lift_smoke.py` — the bimanual capability check: each arm lifts its lace's
  free end clear of the table with its real fingers (PASS: both end sections still >= 60 mm
  above the table at the END of the hold, laces intact).
- `lace_manager.py` — VBD manager specialization (per-substep control, rod contact recipe).
- `newton/lace_coupled_manager.py` — the robot substrate: SolverMuJoCo (arms) + SolverVBD (rods) under
  newton's `SolverCoupledProxy`, gripper bodies proxied into the rod solve.
- `newton_sim.py` — `RodSimCfg`, the sim substrate (60 fps x 12 substeps; `coupled=True`
  selects the proxy-coupled manager).
- `scripts/prepare_shoe_visual.py` — bakes `assets/shoe_right_visual.usda`, the textured render
  shoe the scene spawns.

## Assets

- `assets/shoes/scene.usdc` (+ `0/*.jpg` textures) — Sketchfab sneaker pair (Y-up, cm); the
  scene splits out the right shoe for collision, `prepare_shoe_visual.py` bakes the textured
  render mesh. **TODO before publishing: confirm the Sketchfab license and add the required
  attribution line here.**
- `assets/shoe_right_visual.usda` — generated; regenerate with
  `env_newton/bin/python -m robobench.suites.deformable.scripts.prepare_shoe_visual`.

## Future

- A robot knot-tying task on the coupled substrate (the knot choreography itself is still
  handle-driven in `knot_smoke.py`).
