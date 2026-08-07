# libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i11 — Matchbox Drawer

Env: `simgen.matchbox_drawer` (scene `matchbox_drawer`, robot `null`).

## Seed provenance

Derived from `libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf`:
pick the chefmate frying pan and set it on the open TOP region of a two-layer wooden
shelf. The seed's checker is pure position containment of the pan in the shelf-top
bounding box — one grasp-and-set-down transport ends the task, the target region is open
from above at all times, and the fixture never moves.

## What changed, and why it is strategically different

The target volume here is **sealed at reset**: the goal is the cavity of a matchbox-style
sliding TRAY that starts fully retracted inside a roofed sleeve. No free placement can
reach it — a cube set on top of the unit (the seed's whole strategy) or dumped on the
porch scores ~0. The plan is a three-act mechanism interaction absent from the seed and
from every other constructed task in this corpus:

1. **Open** — slide the tray out along its channel by the black knob: a constrained
   prismatic actuation against rail friction (sustained horizontal force, ~14.5 cm of
   travel onto the porch).
2. **Load through the transient aperture** — drop the RED cube into the cavity that the
   opening exposed (gravity + real containment), while a same-size BLUE decoy must stay
   out.
3. **Re-close** — slide the tray back shut with the cube riding inside on friction,
   passing under the sleeve roof, ending ENCLOSED.

Versus the corpus: no other task **opens a container, loads it, and re-closes it**. The
nearest neighbours differ in plan structure: `close_microwave_i4` *unloads* via a
prop-removal gravity gate and closes a *hinged* door that starts open; the bayonet tasks
(`close_microwave_i5`, `scene7_i3`) twist-lock a lid onto a fixed vessel with nothing
loaded through an aperture; `scene10_i9` drives a rotor to convey a ball; the drop/funnel
tasks (`setup_checkers_i2`, `peg_insertion` variants) place through permanently open
openings. The rubric structure (an opening latch, an in-cavity-**while-open** pathway
latch, a closing-progress latch gated on the load) is likewise new.

Randomization (verified by readback in smoke): cabinet xy + **free yaw** (the pull
direction must be read from the scene), tray seating dither, payload and decoy each on an
independently random side of the porch with xy jitter (so they also swap sides).

## Teleport solution (solve.py)

Teleportation = TRANSPORT ONLY (one write, ending in free space): the RED cube is carried
to a hover pose ~100 mm above the OPEN tray's exposed cavity (open air, forward of the
roof) and released; loading happens by free fall + real contact. Everything else is
driven:

- **P1 open**: bounded-force governor `f = clamp(25·(0.10 − v_along), ±4 N)` on the tray
  along the cabinet's pull axis (read from the randomized cabinet yaw) until the opening
  reaches 145 mm. Rail friction is real (~0.7 N); the force is what a hand pulling the
  18 mm knob applies.
- **P2 load**: hover-and-release above the cavity; gravity + contact seat the cube on the
  tray floor (retry once off-centre if it perches on a wall).
- **P3 close**: the governor reversed (−0.08 m/s, ±4 N); the cube rides the tray on
  friction under the roof; a brief 1.2 N press seats the tray; forces cut; settle.
- **P4 persistence**: ≥ 3.3 simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS` only if
  `success()` still holds. `SIM_GEN_SCORE` printed at every phase boundary,
  non-decreasing (latched credit), asserted in-script.

Verified on the forge: seeds 0 and 1 both end `SIM_GEN_SOLVE: SUCCESS` with score 1.0
(layouts provably different from the stdout readback: yaw +170.0° vs +61.9°).

## Rubric

`success()` iff: tray fully SHUT (opening < 12 mm) and seated in-channel; RED cube in the
tray cavity (tray body frame) AND enclosed under the sleeve roof (cabinet body frame);
the **load pathway latch** earned (the cube was physically in the cavity while the drawer
was open ≥ 80 mm — a cube written into the shut drawer earns nothing); BLUE decoy not in
the cavity; everything settled. `score()` = 0.25·open_latch + 0.30·load_latch +
0.30·shut_latch (closing progress after the load), capped at 0.85; exactly 1.0 iff
success.

## Embodiment argument (Franka)

- **Tray / knob**: the knob is an 18 mm-thick bar, 35 mm tall, protruding 32 mm from the
  tray's front face at ~28 cm height — a standard parallel-jaw pinch (80 mm jaw span)
  from the front. Opening is a straight 14.5 cm horizontal pull at constant height along
  the porch; closing is the mirrored push (fingertips or closed-jaw palm on the knob
  face). Governor forces (≤ 4 N) are well inside Franka payload/wrench limits.
- **RED cube (and decoy)**: 40 mm cubes on an open counter — top grasp with the 80 mm
  jaw, lift over the 55 mm tray wall, release ~6 cm above the cavity floor: exactly the
  hover-release the teleport stands in for.
- **Base pose (one plausible)**: Franka base on the floor, ~0.55 m in front of the porch
  edge, centred on the pull axis and facing the opening. From there the knob (~0.28 m
  up, 0.2–0.35 m reach), the full 14.5 cm slide, and both cube spawn bands (±14–22 cm
  beside the porch) are inside the ~0.85 m reach envelope with the wrist above the
  counter plane.

## Execution order (declared)

OPEN → LOAD → CLOSE, geometry-enforced: the cavity is unreachable until the tray is out
(roofed sleeve + no usable gap when shut, asserted in cfg `__post_init__`), and the
`load_latch` only arms while opening > 80 mm; `shut_latch` only accrues after the load.
Closing before loading, or loading before opening, cannot reach success.

## Checks (smoke.py — rejection battery, 15 checks)

1. settle: finite states, tray shut+seated, cube resting heights;
2. settle: score ~0, no success;
3. randomization readback: cabinet xy + free yaw + tray dither vary;
4. randomization readback: item xy vary, payload occurs on BOTH sides;
5. null policy 240 steps: score ~0, no success;
6. SEED strategy (cube set on top of the unit's roof): rejected, score ~0;
7. cube WRITTEN into the shut drawer (consistent pose): in-cavity/enclosed/shut all
   verified True, load pathway latch 0 → rejected, score ~0;
8. honestly loaded but drawer LEFT OPEN: rejected, score ≤ 0.60;
9. almost-closed (30 mm opening, cube inside): rejected, score ≤ 0.85;
10. wrong object (BLUE decoy loaded and shut inside): rejected;
11. cube dumped on the porch: rejected, score ~0;
12. latched credit survives removing the loaded cube (success gone);
13. opening credit monotone in opening;
14. rejection audit: success never True anywhere in the battery;
15. final no-NaN.

Records `frames.npz` (viewport rgb) in the working directory.
