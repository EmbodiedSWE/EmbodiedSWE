# counterweight_vault — weight the pan to hold the gravity gate open, deliver the cube through the opened vault mouth (i146)

**Env name:** `simgen.counterweight_vault` (scene `counterweight_vault`, registered with
robot `null`; solve.py and smoke.py build the same scene-level env).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl.py`)
- Seed plan: ONE unconstrained grasp-carry-place — pick up one loose black bowl, set it
  down on top of another bowl. Success = a single proximity relation between two bowl
  origins. One object moves; one primitive suffices.

## What changed, and why it is strategically different

The black bowl is kept, but it is never a payload to be stacked — its **WEIGHT** is what
the task consumes:

1. **The centre of the scene is a self-closing GRAVITY GATE**: a rigid rotor on a
   revolute joint whose axis is tilted 6° from vertical. A steel flap (r 125 mm, arm
   190 mm) parks at the LOW point of its orbit, covering the mouth of a walled vault; a
   brass pan (arm 130 mm, at 155° from the flap) rides the short side. Empty, the rotor
   always falls back to closed (empty gravity rest ≈ 13° open — the flap still covers
   the whole mouth; the largest crescent without a counterweight is ~3 mm, a 26 mm cube
   cannot pass).
2. **The black bowl is a COUNTERWEIGHT, not a stackable**: set into the pan
   (mass U[0.32, 0.48] kg, re-randomized every episode via `set_masses` + readback), it
   out-torques the flap side over the whole travel, so the gate swings ~100° open BY
   GRAVITY and stays open **only while the bowl stays in the pan**. The bowl ends up
   hanging off the rotor, nowhere near the other bowl — the seed's end state (bowl
   stacked on the goal region) is an explicitly tested zero-score outcome (smoke 10).
3. **The delivery is physically gated on the counterweight step** — an ordered two-stage
   goal the seed does not have: only through the opened mouth can the red 26 mm cube be
   dropped into the white bowl seated inside the vault. The smoke proves both directions:
   a cube dropped on the closed gate rests ON the flap above the rim (8), and a cube
   teleported through the closed flap never latches `delivered` because the latch demands
   gate-open at the same instant — the exact aperture condition physics enforces (9).
4. **Success is a HELD state, not an event**: the end state is load-bearing (remove the
   bowl and the gate closes — smoke 15 shows the emptied gate re-closing by itself). A
   solver needs a different plan and different code structure: load a counterweight,
   verify the mechanism responded, then thread a drop through the opened aperture — vs.
   the seed's single place-on-target primitive.

Claimed strategy axes: **weight-actuated mechanism (counterweight as a tool) +
physics-enforced two-stage order + load-bearing (self-reverting) goal state.** No other
corpus task has a weight-actuated self-closing gate: mechanism tasks in the corpus are
driven by direct contact on the mechanism (push/pull/turn); here the mechanism is never
touched — an object's mass operates it.

## Embodiment argument (single Franka arm + parallel jaw, OSC)

Intended base pose: **(-0.42, 0.0, 0.20)** — mounted on the counter slab (the corpus'
verified on-counter mount). Worst-case action sites: black-bowl park (-0.10, ±0.24) →
0.40 m; pan centre (loaded pose orbits near (0.15, 0.02)) → ~0.57 m; vault mouth
(0.10, -0.05) → 0.52 m; cube park (0.32, ±0.15) → 0.76 m at the far corner of the jitter
band, ≤0.74 m nominal — inside the demonstrated 0.35–0.71 m envelope except the extreme
cube-park corner, which the ±15 mm jitter rarely reaches and which a base yaw toward +x
covers.

Per manipulated object (both moves are carry-and-drop-from-above; the rotor is NEVER
touched):

- **Black bowl (must move):** the 96–103 mm body is too wide to palm, but the bowl is an
  OPEN octagon with a 7 mm rim wall, 52 mm deep — a rim PINCH (7 mm wall ≪ 80 mm jaw,
  fingers straddle the wall from above) is the designed grasp. Carry over the pan and
  release from a few cm up: solve.py shows an 18 mm free drop seats it (pan inner
  ~122 mm across vs 103 mm bowl → ~9 mm radial play; the pan-frame scoring band,
  xy ≤ 20 mm, is wider than the drop scatter). The pan floor sits 42 mm below the hinge
  (68 mm above the counter at rest) — an open top-down approach, nothing overhead.
- **Cube (must move):** top pinch of the 26 mm cube, carry over the opened vault, release
  above the rim (rim 100 mm above the counter; solve.py drops from 35 mm above it). The
  open mouth is a 130 mm-apothem octagon with the flap swung ~100° clear; the white
  bowl's inner clear radius is 40 mm vs a 30 mm xy scoring tolerance — solve.py lands it
  first drop on all seeds.

No required contact is near the ground (lowest is the cube park pinch at 13 mm cube
centre — top pinch from above), under an overhang, or through a closed aperture; the
delivery drop happens only after the solver itself has opened the mouth.

## Execution order

Required order **counterweight → delivery**, enforced by physics itself: with the pan
unloaded the flap covers the vault mouth, so the cube cannot reach the white bowl (smoke
8 constructs the attempt: the cube rests on the flap ABOVE the rim). The rubric's
latches are additive and `delivered` requires gate-open at the latch instant, so
pre-dropping the cube earns nothing (smoke 9). The reverse order is impossible, not just
unrewarded.

## Teleport-solution outline (solve.py — teleports are TRANSPORT ONLY; both load-bearing interactions run through contact dynamics and gravity)

0. `reset(seed)`, settle to the gravity-closed rest, layout readback (FATAL if the gate
   is not closed at rest) → score 0.000.
1. **Counterweight:** the black bowl is teleported to a hover 18 mm above the brass pan
   floor (outside the 12 mm pan-seat z-band, below the pan wall tops) with zero velocity
   and RELEASED. It falls into the pan under gravity; the loaded pan out-torques the
   flap and the rotor swings ~100° open entirely on its own — the rotor is never
   teleported or forced. Failed drops re-park the bowl, wait for the emptied gate to
   re-close by itself, and retry (4-attempt ladder) → 0.500.
2. **Delivery:** the cube is released 35 mm above the vault rim (far outside its 8 mm
   scoring z-band) over the white bowl's actual centre and must fall through the
   physically opened mouth and come to rest inside the white bowl by itself → 0.800.
3. Settle to full success → 1.000.
4. Persistence: ≥ 3.3 simulated seconds hands-off, success() must still hold →
   `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing.
**Verified on the forge: seeds 0, 1, 2 — all `SIM_GEN_SOLVE: SUCCESS`** (~18 s each;
masses 0.360/0.456/0.420 kg, park sides (−,−)/(−,−)/(+,+); every seed seats the bowl and
lands the cube on the first drop; score trace 0.000 → 0.500 → 0.800 → 1.000 monotone).

## Rubric

- `success()`: black bowl rests IN the pan (pan-frame xy ≤ 20 mm, z band, upright ≤ 18°
  — the open pan tilts ~9°) ∧ gate open (swing ≥ 90°; hard stop 100°) ∧ cube at rest
  INSIDE the white bowl (bowl-frame xy ≤ 30 mm, 8 mm z-band, tilt ≤ 30°) ∧ white bowl
  still seated in its recess ∧ everything settled.
- `settled()` is a REST STREAK — still (all |v| < 0.05 m/s, rotor |ω| < 0.08 rad/s) for
  20 CONSECUTIVE steps: a decaying rotor rock reads instantaneously-still at every
  velocity zero-crossing, and a single-poll judge at a turning point calls a moving
  scene settled (observed on the seed-1 forge run; the streak was added then and the
  rubric re-verified on all three seeds).
- `score()` (latched in post_step, velocity-gated, additive): +0.20 `loaded` (bowl ever
  rested in the pan), +0.30 `held` (bowl in pan AND gate open AND rotor at rest,
  simultaneously), +0.30 `delivered` (cube at rest in the home white bowl WHILE the gate
  is open); 1.0 iff success(). Null policy scores 0; latched credit never evaporates.

## Execution-order declaration (how this package was built)

Goal predicate first → solve.py passing on the forge (two geometry iterations: the pan
support originally crossed the pan mouth and the dropped bowl perched on it — rerouted
to hug the pan's outer wall; then the rubric was hardened with the rest-streak after the
seed-1 settled-flicker) → rubric finalized → smoke battery written against the final
rubric and passed first run.

Declared limitation (same for every jointed-fixture task in this corpus): the gate
fixture is FIXED — its revolute joint is authored against the kinematic frame at build
time and PhysX keeps kinematic-body joint anchors world-fixed, so the fixture cannot be
pose-randomized per episode. Randomization instead covers black-bowl mass (the torque
budget!), park sides/jitter/yaw for bowl and cube, and white-bowl yaw + recess play —
all verified by readback in the smoke.

## Check list (smoke.py — rejection battery, forge: `SIM_GEN_SMOKE: ALL PASS 16/16`, first run)

1. settle/no-NaN: authored layout settles finite at the gravity-closed rest (~13° <
   closed band 25°), white bowl home, nothing loaded/held/delivered
2. rubric clean at reset (score 0, no success)
3. determinism: same seed → identical layout readback
4. randomization readback: black-bowl MASS spreads across seeds
5. randomization readback: park sides both values, park centres jitter
6. null policy: 240 idle steps → gate stays closed, score 0, no success
7. **teleported-open EMPTY gate** (rotor written to 95° on the joint manifold): `held`
   never latches and gravity re-closes the gate by itself → score 0
8. **closed-gate drop**: cube released over the vault rests ON the flap ABOVE the rim —
   the mouth is physically blocked; delivered False, score 0
9. **teleport-through**: cube written INSIDE the white bowl under the closed flap is
   geometrically in, but `delivered` (gate-open clause) never latches → score 0
10. **seed strategy**: the black bowl carried onto the goal (rests on the closed
    flap/vault) latches nothing, score 0
11. wrong counterweight: the 15 g cube dropped INTO the pan (readback: it landed) moves
    the gate only a few degrees — never opens, no credit
12. real stage 1: bowl hover-released into the pan swings the gate open and HOLDS it →
    loaded+held latch, score 0.50, no success
13. displaced target: white bowl moved out of its recess, cube dropped into it
    12 mm off-centre → cube_in True (tolerance twin) but vbowl_home False →
    `delivered` does NOT latch, score pinned at 0.50
14. near miss: cube settled on the counter beside that bowl → not cube_in
15. unload: bowl re-parked → the emptied gate re-closes BY ITSELF and the latched 0.50
    survives (credit never evaporates); no success
16. finite states at the end
(+ frames.npz video recorded, 140 frames)

Assets: fully procedural (kinematic frame = hinge post + octagon vault + recess ring;
ONE dynamic rotor compound — flap disc + arms + octagon pan basket — on an authored
tilted-axis revolute joint, mass/CoM/diagonal inertia authored AND re-enforced through
the PhysX view at bind; octagon-walled bowls; plain cube). No external asset files.
