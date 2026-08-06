# shape_sorter (`draw_triangle_i1`)

**Seed:** `maniskill/draw_triangle`
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/draw_triangle.py`) — a Franka holding
a rigid stick **draws** a triangle: it traces the goal outline on a canvas with the tool
tip, judged on the coverage of the traced path. The core skill is continuous,
contact-maintained **path-following** of a prescribed curve with a held tool.

**Scene:** `simgen.shape_sorter` (robot="null", scene-level; solve.py and smoke.py build
this same env).

## What changed

| | seed | this task |
|---|---|---|
| goal artifact | a transient trace history on a canvas | three pieces physically CONTAINED in the correct compartments of a sealed sorter box |
| mechanism | one long guarded tool-tip sweep along a given outline | perception of a per-episode color->compartment permutation, then three discrete grasp / orient / post-through-aperture operations |
| what the solver must derive | nothing (the outline is given) | WHICH lid opening is which (the colored rims permute every episode — no memorized positions), and a passing attitude per piece (the card only passes its slot within a ~11° yaw window) |
| verification | traced-path coverage | settled physical containment in the box frame read back from sim: centre inside the assigned compartment, exact per-shape top below the lid underside, settled |

Geometry: a fully procedural "post box" — 19 kinematic cuboid panels (floor, walls, two
dividers, and four colored rim strips per compartment forming the three lid openings:
red 58 mm square, green 54 mm square, blue 92×34 mm slot) — plus three primitive pieces
(red cube 38 mm, green cylinder Ø40×50 mm, blue card 80×18×42 mm). Randomized per
episode (verified by READBACK in smoke check 3): box centre ±3 cm + yaw ±15° (all panels
re-posed coherently), the color→compartment permutation (rim strips physically move),
and piece staging rows (permutation + ±3 cm jitter + free yaw). The lid seals everything
except the openings, so containment is reachable only through an aperture passage.

## Why strategically different

- **Perceptual matching + discrete posting vs continuous tracing.** The seed's solution
  is one path-following sweep with a held tool over a given outline. Here the solver
  must first *read* the episode (which colored rim sits over which compartment — it
  permutes every reset), then execute three independent transport-orient-release
  operations, each gated by a different aperture geometry. There is no path to follow,
  no tool to hold, and no curve coverage to accumulate — a different plan and a
  different code structure, not different numbers.
- **The seed's plan produces nothing here.** Tracing a triangle (or any outline) on the
  lid or the floor leaves no persistent state — smoke check 13 constructs exactly the
  seed's end state (pieces arranged as a triangle outline on open floor) and the rubric
  scores it ~0. Conversely the seed's canvas has nothing to contain, so this task's
  plan does not transfer back.
- **The rubric judges settled physical containment**, not a trace history — with
  anti-cheese gates: exact per-shape topmost point below the lid underside (rejects
  on-lid rests and pieces wedged in an opening), centre inside the *assigned*
  compartment (rejects wrong-color posts that are genuinely contained — smoke check 8
  proves the physical containment and the rubric rejection simultaneously), and a
  settle gate.

## Solution (solve.py — the teleport-solution legitimacy certificate)

Teleports are TRANSPORT ONLY; the load-bearing interaction — the aperture passage into
containment — runs through contact dynamics:

1. **READ:** box pose from the floor-panel readback; assignment from the scene;
   per-seed layout printed (proves seed-specificity).
2. **POST ×3 (transport teleport + dynamics):** each piece is carried across free space
   to a hover pose fully OUTSIDE the box, bottom ~22 mm above the lid top, centred on
   its matching aperture in the aperture-aligned attitude (cube flat, cylinder
   axis-up, card upright with its long axis along the slot), zero velocity, and
   released — never written into a contained pose. It falls, threads the opening,
   impacts the compartment floor, and settles inside a sealed volume. `SIM_GEN_SCORE`
   climbs 0 → 0.25 → 0.50 → 1.00 (non-decreasing, self-policed).
3. **VERIFY (+ bounded repair, never triggered in the acceptance runs):** extra settle,
   then re-hover + re-drop any piece whose compartment readback is wrong (≤ 2 rounds);
   require success() and score == 1.0 exactly.
4. **PERSIST:** ≥ 3.5 s of pure simulation with zero intervention, success re-verified
   at every poll → `SIM_GEN_SOLVE: SUCCESS`.

Passes seeds 0, 1 AND 2 on the forge, each first try (assignments [1,2,0], [1,0,2],
[1,0,2]; box poses and scatters all differ — see acceptance logs).

## Embodiment argument (single Franka, parallel jaw, OSC)

Plausible base pose: **(−0.28, 0, 0)** on the floor facing +x. The staging rows put
every piece 0.33–0.55 m from this base; the three apertures sit 0.55–0.69 m away at
lid height (~0.10 m) over all randomization extremes — inside the comfortable envelope,
with the whole action above open floor (posting happens from ABOVE the lid; the hand
never enters an aperture or a compartment).

- **Red cube (38 mm)** — top or side pinch across two faces (38 mm ≪ the 80 mm jaw),
  identical to the proven ground-level pinches of the earlier bar/frame task on this
  rig. Post: hover over the red-rimmed hole, release. The 58 mm hole passes the cube at
  ANY yaw (its 54 mm diagonal fits), so the required precision is a ±10 mm centring —
  an order of magnitude above measured OSC noise (1–3 mm on this rig).
- **Green cylinder (Ø40×50)** — side pinch at mid-height (opening 40 mm) or top pinch;
  it spawns standing and cannot roll while staged. Post: hover axis-up over the
  green-rimmed 54 mm hole (±7 mm centring, yaw-free), release.
- **Blue card (80×18×42, lying flat, 18 mm tall)** — ground-level pinch across its
  42 mm width near one end (fingers on the two long side faces, the bar-pinch pattern
  measured on this rig), then a wrist reorientation to present it upright, long axis
  along the slot. Post: hover over the blue slot and release; tolerance ±8 mm laterally
  and ~±11° in yaw — generous against OSC noise, and the physically-verified failure
  mode (bridging the slot) is recoverable by re-grasping from the lid.
- **Clearances:** apertures are 7–10 mm per side laterally; nothing happens under an
  overhang; the only near-ground contacts are the staged-piece pinches on open floor.

## Execution order: NOT required

Pieces may be posted in any order; the rubric is symmetric across pieces (no
out-of-order negative control applies — documented per the brief). Difficulty comes
from reading the permuted rims and from per-piece attitude control at the openings.

## Rubric

- `success()`: every piece settled (< 0.05 m/s) with its centre inside its ASSIGNED
  compartment's interior footprint (2 mm inset, box frame read back from the floor
  panel), its exact per-shape topmost point ≥ 2 mm below the lid underside, and its
  centre above the box floor — all three simultaneously.
- `score()` = 0.25 per counted piece; exactly **1.0 iff success() now**. Containment
  credit is physically persistent (sealed compartments), so correct behavior's credit
  cannot evaporate (verified by smoke check 11); the null policy scores 0 by
  construction (pieces spawn on open floor, clear of the box).

## Checks (smoke.py — rubric rejection battery, 14 named checks; never constructs success)

1. settle/no-NaN: fresh reset settles finite, pieces at rest
2. reset-zero: score ~0, nothing contained
3. randomization by READBACK: box xy + yaw, ≥ 2 distinct color permutations across
   seeds, rim strips physically track the assignment, piece poses move, pieces clear
   of the box every seed
4. null policy 240 steps → score ~0, no success
5. `set_state(get_state)` roundtrip restores poses
6. on-lid near-miss: cube settled on the lid beside its own aperture → rejected
7. attitude near-miss: card lying flat over its own slot bridges it (42 across 34) → rejected
8. wrong compartment: cube posted through the green aperture — physically contained
   (bin readback proves it) yet rejected by the matching clause
9. all-in-one: all three pieces through the RED aperture → only the cube counts (0.25)
10. calibration: cylinder through its OWN aperture counts (0.25), success False
11. persistence: 240 further idle steps keep the posted piece counted
12. beside-box: pieces on the floor against the box wall at their assigned rows → ~0
13. seed-strategy: pieces arranged as a triangle outline on open floor → ~0
14. video frames recorded and saved to frames.npz (282 frames)

Forge acceptance: `SIM_GEN_SMOKE: ALL PASS 14/14`; `SIM_GEN_SOLVE: SUCCESS` on seeds
0, 1, 2 — all first run.
