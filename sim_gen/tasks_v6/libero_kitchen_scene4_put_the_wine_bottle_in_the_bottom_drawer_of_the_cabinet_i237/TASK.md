# tilt_bin_stow — stow the wine bottle in the cabinet's tilt-out bin and shut it

**Package:** `sim_gen/tasks_v6/libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_i237`
**Env:** `simgen.tilt_bin_stow` (scene-level, `robot="null"`)

## Seed provenance

`libero_90/libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/...py`): pick the wine bottle off the
table and place it inside the bottom drawer of the white cabinet — a single
pick-and-place into a passively available prismatic receptacle, judged by a
bottle-in-drawer-bbox test.

## What changed and why it is strategically different

The receptacle is replaced by a **bistable mechanism the solver must actuate twice**,
and the goal predicate moved from "bottle in receptacle" to "bottle in the **shut**
receptacle, wrong bottle excluded":

- The cabinet houses a **tilt-out bin** (revolute joint along its bottom-front edge,
  limits [0, 48°], like a laundry-hamper cabinet) instead of a sliding drawer. It is
  **bistable under gravity** (CoM crosses the hinge vertical at ~35°): it rests shut or
  rests tilted-out on its stop — each throw is a deliberate actuation, not a held state.
- A **metric interlock** forbids the seed's plan: shut, the bin's mouth is capped by the
  cabinet top panel and the remaining front slot (52 mm) is narrower than either
  bottle's body (60 / 56 mm). "Carry bottle to receptacle and lower it in" is
  geometrically impossible until the bin is tilted open.
- Success requires **closure with the payload riding inside** — the seed's terminal
  relation (bottle resting in an *open* receptacle) is explicitly rejected (smoke #6).
- A same-shape **white decoy bottle** (slot-swapped each episode) adds a color-grounded
  identification clause; stowing it — alone or additionally — fails (smoke #10, #11).

A solver therefore needs a different plan (actuate-open → deposit through the transient
aperture → actuate-shut, with object identification), and the code is structurally
different: procedural compound bodies + a bind-time `UsdPhysics.RevoluteJoint`, a
post_step-owned hinge plant (drive + viscous damping), and a mechanism-state rubric —
nothing shared with the seed's asset-loading + static-bbox test.

## Teleport-solution outline (solve.py, phases; `SIM_GEN_SCORE` at each boundary)

- **P0** reset + settle; the randomly-ajar bin falls shut; layout readback printed.
- **P1 OPEN (contact dynamics):** bounded torque about the real hinge (≤2 N·m; the
  applied-wrench emulation of pulling the red handle) swings the bin past crossover;
  drive removed; the bin coasts onto its 48° stop where gravity alone holds it. The bin
  pose is never written after reset.
- **P2 TRANSPORT (teleport):** one pose write carries the green bottle to a hover 35 mm
  above the open mouth plane — asserted **outside** the containment volume; axis along
  the mouth's long axis.
- **P3 DEPOSIT (contact dynamics):** pure release — free-fall through the mouth, impact
  on the tilted floor, slide into the floor/front-wall V, settle inside. No writes.
- **P4 CLOSE LOADED (contact dynamics):** reverse hinge torque pushes the loaded bin
  past crossover; drive removed; gravity + viscous hinge damping seat it shut on the
  lower limit with the bottle riding inside; wait for genuine stillness. success() first
  turns True here.
- **P5** hands-off persistence ≥3.3 s, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds **0 and 1** (both `SIM_GEN_SOLVE: SUCCESS`, scores
monotone 0.00 → 0.20 → 0.33 → 0.59 → 1.00).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

Plausible base pose: **base at the world origin on the floor plane**; all interactions
lie at radius 0.39–0.63 m, heights 0.06–0.34 m — comfortably inside the Franka
envelope.

- **Tilt-out bin (open):** pinch or hook the protruding red handle bar (16 mm
  cross-section, 30 mm proud of the face, at height 0.20 m — free approach from the
  front) and pull forward/down along the hinge arc. Past ~35° the bin falls onto its
  stop by itself, so mid-arc grip fidelity is not precision-critical.
- **Green wine bottle:** grasp the 26 mm neck (fits the 80 mm jaw with wide margin;
  bottle stands free on open floor, top-down or side approach unobstructed). Carry over
  the open mouth (a 120 × 200 mm aperture — the drop window for the 60 mm body is
  ±30 mm, far above OSC noise) and release above the mouth plane; the tilted floor
  funnels it in. Release-above-the-scoring-band: hovering satisfies no rubric clause.
- **Tilt-out bin (close):** push the handle bar or upper front face backward with
  closed fingertips until past ~35°; gravity + hinge damping finish seating it. No
  grasp needed.
- **Decoy:** requires no contact at all — it must merely be left alone.

## Execution order

Required and mechanism-forced (declared in describe()): open before deposit (mouth
capped + 52 mm slot < bottle body when shut), close after deposit (closing an empty bin
earns nothing; success needs containment + closure simultaneously). Declared in
describe()/instruction().

## Rubric

`score()` = 0.15·opened(≥35°, latched) + 0.20·mouth-approach (gated on opened, latched
max) + 0.25·deposited (latched) + 0.25·closing-progress (counted only while the wine is
currently inside), capped at 0.85; exactly 1.0 iff `success()`:
wine inside the bin ∧ bin ≤8° of shut ∧ bin and bottle at rest ∧ decoy **not** inside.
Null policy scores ~0 (bin starts shut; every term is gated or latched off).

## Check list (smoke.py — rejection battery, forge: `SIM_GEN_SMOKE: ALL PASS 14/14`)

1. settle/no-NaN: bin falls shut, bottles stand at slots, still, finite
2. score ~0 at reset, no success
3. randomization readback: wine/decoy slot assignment flips; always opposite slots
4. randomization readback: per-slot xy jitter real
5. null policy (240 steps): score ~0, no success
6. **seed strategy**: wine settled inside the **open** bin (the seed's end state) →
   NOT success, score < 0.9
7. on-top cheat: wine parked on the cabinet top → score ~0, no success
8. against-face: wine standing at the closest shut-bin approach → score ~0, no success
9. near-miss: wine inside, bin held ajar at 15° (tol 8°) → angle gate rejects
10. wrong object: decoy inside the shut bin, wine outside → score ~0, no success
11. decoy exclusion: wine AND decoy inside the shut settled bin — every other gate
    passes, decoy clause alone rejects → NOT success, score ≤ 0.85
12. latched credit: regressing the wine out leaves the latched score unchanged
13. rejection audit: success() never True anywhere in the battery
14. final no-NaN

frames.npz (170 × 600 × 960 × 3) recorded and saved in cwd by smoke.py.
