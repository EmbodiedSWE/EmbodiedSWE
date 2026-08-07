# approach_grasp_spoon_i12 — Tip the Big Die (scene `die_tip_pad`)

Get an OVERSIZED die — a 100 mm gray cube with six colored face decals — resting on a
white floor disk with its BLUE face pointing straight up. The die is wider than the
Franka jaw (80 mm), so it can never be grasped or lifted: the only way to move it is
non-prehensile floor work, and the only way to change which face is up is to TIP it
end-over-end about a bottom edge (each 90° tip also advances the die one face length,
coupling position and orientation). A small graspable RED cube nearby is a decoy: the
seed's whole strategy applied here ("grasp the graspable thing, set it on the target")
earns exactly nothing.

## Provenance

- **Seed:** `pick_place/approach_grasp_spoon`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/approach_grasp_spoon.py`) —
  approach a small spoon lying in tabletop clutter, close the parallel jaw on it, and
  carry it along marked waypoints toward a basket: one grasp affordance plus free-space
  transport of a rigidly held object.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `die_tip_pad`, env
  `simgen.die_tip_pad`, robot `"null"`), `solve.py` (teleport solution), `smoke.py`
  (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed is prehensile through and through — the object is chosen
  to fit the jaw, the plan is grasp → carry → release, and the spoon's own orientation
  is never load-bearing. Here grasping the judged object is PHYSICALLY IMPOSSIBLE
  (100 mm cube vs 80 mm jaw, asserted in cfg `__post_init__`), the object's
  orientation IS the goal (blue face up, and blue never spawns up so at least one real
  tip is always required), and progress happens through an edge-pivot contact mode the
  seed never touches: push HIGH on a face and the die tips over its leading bottom
  edge (h > a/(2·μ_eff) ≈ 53 mm); push LOW and it slides flat. Because every tip
  advances the die exactly one face length, the solver must plan a coupled
  position+orientation tip sequence, not a carry. The seed's strategy transplanted
  verbatim — pick the graspable object, place it on the target — is exactly the
  decoy-on-disk end state and is rejected outright (smoke check 6).
- **vs the 12 corpus tasks read this session:** none is non-prehensile floor
  reorientation. The nearest neighbours: `pour_water_i7` (ramp + chock equilibrium)
  manipulates graspable bodies to reach a statics equilibrium, never reorients an
  ungraspable one; `pick_single_egad_i3` (gate-pin tunnel slide) slides a body through
  a passage but its orientation is free; `libero_i2` (pour + invert park) inverts a
  grasped mug in-hand — prehensile. No corpus task uses tipping/rolling as its load-
  bearing mechanic, and none has a "too big to grasp" embodiment constraint as the
  core premise.
- **Execution order (declared):** orient-while-approaching — the tip sequence IS the
  approach (each tip advances 100 mm), so the natural plan stages the die a whole
  number of face-lengths short of the disk and tips it home; blue-down states need two
  tips along one bearing, blue-horizontal states need one tip against the blue normal.
  There is no hidden second ordering: slides (low pushes) may precede or follow tips,
  but at least one genuine floor tip is always required and latched as such.

## Randomization (per episode, verified by readback in smoke)

Disk centre xy jitter (±80 mm); die spawn on a random bearing/radius ring
(0.34–0.50 m) around the disk with a random UP FACE drawn from the five non-blue faces
composed with free yaw (the required tip sequence differs per episode); decoy on its
own ring (0.32–0.55 m) with batched keep-out resampling against the die. The approach
baseline `d0` is captured per episode.

## Rubric

`success()` iff, all judged live on physical settled poses:

- die centre within 80 mm (xy) of the disk centre;
- BLUE face normal within 10° of world-up;
- resting FLAT on the floor/disk: some face down within 5° AND centre height within
  8 mm of the 50 mm half-side (kills stacked-on-decoy and held-aloft states);
- settled (|v| < 0.05 m/s, |ω| < 0.40 rad/s).

`score()` (latched every physics substep in `post_step`): `0.15 ×` best approach
toward the disk (normalized by the episode's own spawn distance) `+ 0.20 ×` TIPPED
(the reset down-face rotated ≥ 60° from down while the die stayed under `low_z` =
120 mm — a genuine floor tip: a carry exceeds the gate, a yaw spin keeps the down face
down) `+ 0.15 ×` blue-up seen while low (airborne reorientation earns nothing)
`+ 0.30 ×` boarded (flat on the disk at floor height, slow). Base capped at 0.80;
exactly 1.0 iff `success()`. Doing nothing scores ~0.

## Teleport solution (`solve.py`) — transport only, ONE write, ending in free space

- **P1 — staging transport:** after settling and reading the physical pose, the die is
  teleported to the boarding line — `n_tips × 100 mm` short of the disk centre along
  the planned tip bearing — 1 mm above the open floor, ORIENTATION PRESERVED EXACTLY,
  velocities zeroed. Blue-down draws two tips along one bearing (chosen to keep the
  corridor clear of the decoy); blue-horizontal draws one tip against the blue normal.
- **P2 — contact tips:** each tip is a pure torque about the horizontal world axis
  ẑ × d̂ (the couple of a fingertip pushing HIGH on a face), ω-regulated with a
  0.55 N·m cap (> mga/2 = 0.245 N·m), cut at 55° — past the 45° balance point — so
  gravity and real floor contact finish every 90° tip; between tips the solver
  re-reads the pose and re-plans, and residual placement error is corrected by LOW
  push-slides (COM force + counter-torque of a push at 30 mm, below the 53 mm tip
  threshold). The die is never teleported onto the disk and never teleported into a
  new orientation; every orientation change goes through an edge-pivot floor contact.
- `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing, ≥3.3 simulated
  seconds hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`. **Verified on the
  forge: seeds 0 and 1, both SUCCESS, provably distinct by stdout readback** — seed 0:
  GREEN up (blue horizontal), ONE tip, d_pad 0.000 m; seed 1: ORANGE up (blue DOWN),
  TWO tips along one bearing, d_pad 0.011 m — both planner branches exercised
  (0.000 → 0.106 → 1.000 → 1.000 and 0.000 → 0.071 → 1.000 → 1.000).

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (0.65, 0.0) — or (−0.55, 0.0); the disk jitters within ±80 mm of
(0.06, 0.02) and the die ring reaches 0.58 m from it, all within a mobile-base-free
Franka's ~0.85 m envelope from either side. Per-object contact strategy: the DIE is
worked with the closed fist / fingertip — a HIGH push (fingertip at 70–90 mm on a
100 mm face, well above the 53 mm tip threshold; 2.7–4.4 N at 90 mm) tips it about the
leading bottom edge, and OSC can track the face as it rotates or simply push through
the tip start and retreat, letting gravity finish (exactly what solve.py's torque-cut
does); a LOW push (fingertip at ≤30 mm) slides it flat for centering. No grasp is ever
needed on the die — which is the point: the jaw physically cannot cage a 100 mm cube.
The DECOY is trivially graspable (45 mm < 80 mm stroke) but touching it is worthless
by rubric. The disk is flush (1 mm proud paint zone), so no step blocks a push
corridor, and the die ring (≥0.34 m from the disk centre) leaves the whole boarding
line clear of clutter by construction (decoy keep-out 0.16 m).

## Checks (`smoke.py` — rejection battery, 14 named checks, ALL PASS on the forge)

1. settle: states finite; die flat on the floor at half-height, decoy on the floor
   (readback heights), settled.
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization readback: disk xy and die xy vary across 6 seeded resets.
4. randomization readback: die up-face varies and is NEVER blue; full orientation
   quat, decoy xy and d0 vary.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: graspable RED decoy settled at the disk centre (verified on-disk by
   readback), die untouched → NOT success, score ≤ 0.02 (identity: only the die counts).
7. wrong face: die flat and settled AT the disk centre but GREEN up → on_pad yet NOT
   success, score ≤ 0.70.
8. position near-miss: die BLUE UP, flat, settled, 40 mm outside `pad_margin` → NOT
   on_pad, NOT success, score ≤ 0.55.
9. stacked loophole: die BLUE UP and xy-centred over the disk but resting ON TOP of
   the decoy (centre ~95 mm high) → floor-rest clause rejects, NOT success.
10. carried loophole: die BLUE UP in the AIR over the disk centre (transient judged
    probe) → NOT success, and the blue-up-while-LOW latch stays 0 above `low_z`.
11. latched credit: removing the boarded die leaves the latched score unchanged
    (0.650 → 0.650), success stays gone.
12. monotonicity: closer die placement latches strictly more approach credit.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN. Plus `frames.npz` (134 × 600 × 960 × 3) recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest: the
die is strictly wider than the jaw (+15 mm margin), the decoy is trivially graspable,
the friction budget puts the tip threshold in the reachable band (high pushes tip, low
pushes slide), the spawn ring keeps the die off the disk (approach credit is real),
a success die reads as ON the disk, and a mid-tip die stays under the `low_z` gate.
