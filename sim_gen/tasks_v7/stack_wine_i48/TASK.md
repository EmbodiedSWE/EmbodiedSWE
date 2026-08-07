# stack_wine_i48 — Sort the Kegs by Weight (scene `cask_weight_sort`)

Three VISUALLY IDENTICAL oak kegs (native-capsule bodies, 52 mm across — trivially
graspable) lie on the floor in front of a row of three color-coded rail cradles
(RED / YELLOW / GREEN, order shuffled per episode). The kegs' MASSES (0.3 / 0.9 /
2.7 kg) are re-shuffled across the three bodies every episode by runtime PhysX
mass+inertia writes with readback verification — there is NO visual cue to identity.
Success = every keg wedged in the cradle of its MASS RANK (heaviest→RED,
middle→YELLOW, lightest→GREEN) and settled. A solver that will not first PROBE the
kegs — give each the same gentle nudge and compare how far they roll — cannot beat a
1-in-6 permutation guess, no matter how perfect its pick-and-place is.

## Provenance

- **Seed:** `rlbench/stack_wine`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/stack_wine.py`) — grasp THE wine
  bottle (unique, visually given) and lay it on THE rack: a pure prehensile
  pick-and-place where nothing ever has to be discovered.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene
  `cask_weight_sort`, env `simgen.cask_weight_sort`, robot `"null"`), `solve.py`
  (teleport solution), `smoke.py` (rejection battery), all procedural geometry — no
  external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed's problem is motor (approach, grasp, carry, lay down);
  its goal binding is VISUAL (the one bottle, the one rack) and given for free. Here
  the motor act is deliberately easy (kegs fit the jaw with 28 mm margin, cradles are
  open troughs) and the load-bearing problem is EPISTEMIC: goal binding
  (which keg → which cradle) is decided by an invisible physical property that must
  be MEASURED by interacting — the same push rolls the light keg ~3× farther than the
  middle one and ~9× farther than the heavy one (displacement ∝ 1/m under identical
  impulse + mass-independent damping). The seed's strategy transplanted verbatim
  ("put each object on a rack") is exactly the identity-blind derangement state that
  smoke check 6 physically constructs and the rubric rejects (score ≤ 0.20, never
  success). Measure → rank → place, versus place.
- **vs the corpus tasks read this session:** no corpus task makes hidden-state
  DISCOVERY the core mechanic. The nearest cousins: `pick_cube_i40` counts visible
  markers to pick a balance load — the state is visual, not probed; the torque-beam
  tasks (i15/i16/i20) exploit known masses in statics, never identify unknown ones;
  `pour_water_i7` reaches a statics equilibrium with fully visible bodies;
  `approach_grasp_spoon_i12` moves an ungraspable body — motor-hard, epistemically
  trivial. None requires an information-gathering action whose OUTPUT selects the
  goal assignment; here every episode's correct assignment is unknowable without the
  probes (and unguessable at 5/6 probability).
- **Execution order (declared):** probe-then-place is the natural order but is NOT
  hard-coded in the rubric — any interleaving that ends with the correct permutation
  seated wins; the latched identity credit (0.20/keg) simply cannot be earned before
  a keg reaches its CORRECT cradle, and which cradle is correct is what probing
  reveals. There is no hidden second ordering constraint.

## Randomization (per episode, verified by READBACK in smoke)

Mass permutation across the three keg bodies (PhysX `set_masses`/`set_inertias`,
inertia scaled ∝ mass, CoM at origin, READBACK asserted at reset and re-verified in
smoke check 4); cradle color arrangement across the three row slots (+ per-cradle x
jitter ±8 mm, shared row-y jitter ±25 mm); keg spawn slots permuted (+ xy jitter
±10/±35 mm, yaw jitter ±8°).

## Rubric

`success()` iff, all judged live on physical settled poses, for EVERY keg:

- centre within 50 mm (x, along-trough) / 12 mm (y, across-trough) of ITS OWN
  cradle's centre — the cradle whose color matches the keg's mass rank;
- centre height within ±7 mm of the V-seat height (48.8 mm — cradled on the two 45°
  V faces: rejects floor (26 mm), on-the-plate (38 mm), perched-on-a-rail-edge
  (~69 mm), and aloft; face-tangent contacts, chosen because a keg chocked on two
  sharp box EDGES excites a PhysX phantom-velocity limit cycle that would break every
  settle gate);
- axis within 15° of the trough direction (rejects across-the-rails);
- settled (|v| < 0.05 m/s, |ω| < 0.50 rad/s).

`score()` (latched every physics substep in `post_step`): `0.05 ×` each keg ever
displaced ≥ 30 mm from spawn while low (manipulation onset; a null policy never
produces it) `+ 0.20 ×` each keg ever seated in its CORRECT cradle continuously for
24 substeps while slow (identity credit — the probe's payoff), capped at 0.75;
exactly 1.0 iff `success()`. Doing nothing scores ~0; a perfect identity-blind
racking (seed strategy) is capped at 0.15–0.35 unless it luckily hits the
permutation.

## Teleport solution (`solve.py`) — transport only, ending in free space

- **P1 — probes (pure contact dynamics, the measurement):** each keg in turn gets
  the SAME 0.8 N × 10-substep world −y pulse (`set_external_force_and_torque`) from
  rest, then coasts 2.5 s; kegs are RANKED BY MEASURED xy displacement (smallest =
  heaviest). The ground truth `rank_of_cask` is printed for audit but NEVER used for
  control. Ambiguous adjacent ratios (< 1.5) trigger a stronger re-probe battery
  (never needed on the forge: measured ratios 3.00/2.99 on seed 0, 2.99/2.99 on
  seed 1, MATCH with ground truth on both).
- **P2 — placement:** each keg is teleported to a hover 60 mm ABOVE its
  measured-rank cradle (free space above the rails), orientation preserved,
  velocities zeroed, then DROPS under gravity and settles into the self-centring
  45° V-groove through real contact; seating verified by readback, with re-centred
  retries (none needed). The keg is never teleported into the groove itself.
- `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing (0.00 → 0.15 →
  0.35 → 0.55 → 1.00), ≥ 3.3 simulated seconds hands-off persistence, then
  `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge: seeds 0 and 1, both SUCCESS,
  provably distinct by stdout readback** — seed 0: bays GREEN|YELLOW|RED,
  rank_of_cask=[2,0,1]; seed 1: bays YELLOW|RED|GREEN, rank_of_cask=[1,0,2]; probe
  displacements 0.215/0.072/0.024 m — the clean 3×/9× signature.

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (0.0, −0.55): the whole workspace — keg row at y ≈ −0.12 (x ∈
±0.235), probe roll-out down to y ≈ −0.34, cradle row at y ≈ 0.20 (x ∈ ±0.21) — is
within a ~0.85 m reach envelope. Per-object contact strategy: PROBING is a fingertip
side-push at the keg's belly (0.8 N for ~0.08 s, or simply a slow 5 cm push-and-
release; only the COMPARISON across kegs matters, and ±20 % force repeatability
leaves the 3× ratios unambiguous — alternatively the arm can heft each keg after
grasping and compare payloads, which the rubric equally permits since probing is
instrumental, not judged). PLACEMENT is a trivial top grasp of the 52 mm body with
the 80 mm jaw (28 mm margin), a carry (heaviest keg 2.7 kg < 3.0 kg payload), and a
release just above the trough: the V is open and shallow (rail top edges at 43 mm),
so the jaw opens above the seat without entering the groove, and the keg drops the
last centimetres into the self-centring V — exactly what solve.py's hover-drop does.
The cradles are kinematic and
cannot be disturbed; the keg rows are 0.32 m apart, leaving a clear carry corridor.

## Checks (`smoke.py` — rejection battery, 14 named checks, ALL PASS on the forge)

1. settle: states finite; all three kegs on the floor at capsule-radius height
   (readback z), settled.
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization readback A: the cradle color arrangement (bay x order) varies
   across 6 seeded resets and keg spawn xy varies.
4. randomization readback B: PhysX MASS readback equals the nominal
   {2.7, 0.9, 0.3} table under the scene's rank bookkeeping (the hidden state is
   real, not cosmetic) and the permutation varies across seeds.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: all three kegs PHYSICALLY WEDGED in cradles (geometric seating
   verified by readback), settled — but in a derangement → NOT success,
   score ≤ 0.20 (identity is the task).
7. one-correct swap: heavy keg in RED, other two swapped → NOT success, exactly one
   identity credit (score ≤ 0.40).
8. position near-miss: the right keg on the floor 60 mm beside its cradle → not
   seated, NOT success.
9. across-rails: the right keg centred on its cradle but lying ACROSS the trough →
   axis clause rejects, NOT success.
10. two-kegs-one-bay: the right keg correctly seated, then a SECOND keg dropped onto
    the same bay — the second keg (on top at ~seat+2r, or rolled off) is never
    seated in that bay, NOT success.
11. hover loophole: the right keg held in the air over its cradle (transient judged
    probe) → NOT success, and the 24-substep seat latch stays 0.
12. latched credit: removing the one correctly seated keg leaves the latched score
    unchanged, success stays gone.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN. Plus `frames.npz` recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest:
the V-seat contacts land ON the tilted faces (not the rail edges), a seated keg
hangs clear of the plate, the seat band excludes floor / on-the-plate /
perched-on-a-rail-edge rests, kegs are trivially graspable and the heaviest
is liftable, mass steps are ≥ 2.5× apart (probe separability), and slot spacing
keeps spawns and cradles from ever overlapping.
