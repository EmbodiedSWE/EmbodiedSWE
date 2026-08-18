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

Registered env: `shoe_tying.knot` (robot `null` — the handles are driven straight through the
Newton manager, per solver substep).

## Run

```bash
# full smoke: settle -> lift -> x-form -> tuck -> pull -> slack hold (headless physics + verdict)
OMNI_KIT_ACCEPT_EULA=YES env_newton/bin/python -m robobench.suites.shoe_tying.smokes.knot_smoke --headless

# author + validate the lace curves only (self/inter-lace clearance, shoe SDF), no sim
... knot_smoke --headless --validate-only

# RTX video (records the Kit RTX viewport through scripts/record_video.py)
env_newton/bin/python scripts/record_video.py \
    robobench.suites.shoe_tying.smokes.knot_smoke \
    --video robobench/suites/shoe_tying/videos/knot_smoke.mp4 \
    --eye 0.33 -0.31 0.40 --target-at 0.0 0.03 0.10
```

Requires `env_newton` with the newton upgrade applied (root README, "Upgrading newton" — the
rod APIs postdate the original isaaclab_newton pin).

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

## Layout

Physics and rendering are split (rods have no IsaacLab asset type):

- `scenes/shoe_knot.py` — the scene: lace curve authoring + validation, knot metrics, and the
  per-world builder hook that injects the physics (table twin, invisible shoe collision
  trimesh, bridge capsules, both lace rods) into the Newton `ModelBuilder`.
- `smokes/knot_smoke.py` — the tying choreography (`KnotDriver`, ported 1:1 from the
  standalone script) + the RTX visual layer (`LaceVisuals`: one USD capsule per rod segment,
  synced from `body_q`).
- `lace_manager.py` — VBD manager specialization (per-substep control, rod contact recipe).
- `newton_sim.py` — `RodSimCfg`, the sim substrate (60 fps x 12 substeps).
- `scripts/prepare_shoe_visual.py` — bakes `assets/shoe_right_visual.usda`, the textured render
  shoe the scene spawns.

## Assets

- `assets/shoes/scene.usdc` (+ `0/*.jpg` textures) — Sketchfab sneaker pair (Y-up, cm); the
  scene splits out the right shoe for collision, `prepare_shoe_visual.py` bakes the textured
  render mesh. **TODO before publishing: confirm the Sketchfab license and add the required
  attribution line here.**
- `assets/shoe_right_visual.usda` — generated; regenerate with
  `env_newton/bin/python -m robobench.suites.shoe_tying.scripts.prepare_shoe_visual`.

## Future

- Robot in the loop: needs a coupled MJWarp-arm + VBD-rod manager (pouring's
  `coupled_manager` precedent). Upstream reference at the pinned newton:
  `newton/examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py` and
  `example_franka_cable_ik_pick_place.py`.
