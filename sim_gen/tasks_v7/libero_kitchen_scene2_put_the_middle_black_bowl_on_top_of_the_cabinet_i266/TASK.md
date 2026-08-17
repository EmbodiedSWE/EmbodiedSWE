# scoop_lift_court — load the middle bowl into the swing-lift's scoop and crank it through the side window

**Task id:** `libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i266`
**Env:** `simgen.scoop_lift_court` (scene-level, `robot="null"`)
**Files:** `scene.py` (scene + rubric), `solve.py` (torque-crank solution),
`smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet`
(RoboVerse `roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet.py`).
The seed places three identical akita black bowls in a row and a plate beside a wooden
cabinet; the goal is one vertical pick-and-place — grasp the MIDDLE bowl and set it in a
bbox above the cabinet's always-free flat top. Success is a static pose test on that bowl.

Kept from the seed: three identical black bowls in a row with the MIDDLE one as the target
(identity discrimination), a distractor plate, a wooden cabinet, and the goal of the middle
bowl at rest on the cabinet's top level.

## What the task is

The cabinet's top is a walled **COURT** sealed under a full **ROOF** — there is **no
vertical entry at all**; the seed's release-from-above is physically impossible here
(smoke check 12: a bowl released over the cabinet parks on the roof and earns nothing).
The only way in is a side **WINDOW** in the +x wall (20 cm wide, z 0.45–0.63) that opens
onto empty air ~0.4 m above the floor — no shelf, no ramp, nothing to slide along.

Beside that wall stands a **SWING-LIFT**: a heavy free-standing pylon carrying a revolute
arm. One end of the arm is an open-topped **SCOOP pocket** (slick inside), the other a
**lever paddle** the hand can press. The arm is gravity-**bistable**: its CoM is on the
scoop side, so unpowered it rests either on the LOAD stop (θ = −20°, scoop low and level
beside the staging pad) or on the DUMP stop (θ = +130°, scoop mouth pointing 40° below
horizontal just outside the window). The joint limits are the stops.

Three identical black bowls sit in a row on a staging pad; which body occupies which slot
is a fresh random permutation each episode (readback-verified). The goal:

> "Put the MIDDLE black bowl of the three into the swing-lift's scoop, then crank the
> lever so the arm swings up and over and flings the bowl through the wall window so it
> comes to rest on the seat inside the court. Do not put the other bowls or the plate in
> the court."

`success()` = target bowl seated at rest in the court seat band ∧ the full **latch chain**
fired for that bowl (loaded → lifted → delivered → windowed → courted, in order) ∧ no
decoy bowl and no plate in the court ∧ everything settled.

## Strategic difference — vs the seed and vs every corpus neighbor

- **vs the seed:** the seed is one grasp + one vertical release onto an open top. Here the
  top is sealed — the seed's entire plan is impossible, not merely insufficient. The
  target is handled directly only once (a short put into the scoop at pad height); all
  delivery work is done by **operating a machine**: cranking a bistable lever arm through
  a 150° powered swing whose end-of-travel slam launches the bowl through the window.
- **vs sibling i140 ramp_hutch:** i140's target travels a passive ramp/doorway path under
  pushes — the hand does the transport work the whole way. Here the hand never touches the
  bowl after the load; the mechanism does the lift, the throw, and the entry, and the
  physics that matters is the arm-frame pour dynamics (centrifugal pinning, tangential
  launch), not sliding friction along a path.
- **vs sibling i208 letterbox_cabinet:** i208 keeps the seed's vertical drop as the finale
  and inverts the work onto debris eviction. Here there is no vertical entry to preserve
  — no port, no debris; the strategy is mediated delivery through an articulated machine.
- **vs robobench suite mechanisms (balance_scale, combination_safe, syringe, pen_holder):**
  those manipulate mechanisms as goals in themselves (set a state). Here the mechanism is
  a **tool**: its swing is the transport channel for a separate payload, and the rubric
  gates on the payload's ride history, not the mechanism pose.

No task in the read corpus makes "load a payload into a bistable lever machine and crank
it so the machine itself throws the payload through the only opening" the core strategy.

## Rubric (latched credit, `score()`)

Latches are per-bowl `(N, 3)` tensors evaluated in `post_step` every substep; each stage
is **gated on the previous** so out-of-order states earn nothing:

| latch | condition | weight (target bowl) |
|---|---|---|
| `_loaded_ever` | bowl in the scoop band (arm frame) while θ < 10° | 0.15 |
| `_lifted_ever` | loaded ∧ still aboard ∧ θ > 60° | 0.15 |
| `_delivered_ever` | lifted ∧ still aboard ∧ θ > 95° | 0.20 |
| `_windowed_ever` | delivered ∧ bowl inside the window band | 0.20 |
| `_courted_ever` | windowed ∧ bowl inside the court | 0.25 |

`score = Σ` (target's latches only), capped at 0.95 unless `success()`, which returns
exactly 1.0. A bowl teleported onto the seat, dropped through nothing, or pushed in
through the window without having ridden the arm earns 0 (smoke checks 13–15). Null
policy ≈ 0. The printed `SIM_GEN_SCORE` sequence is non-decreasing by construction.

## Solution (`solve.py`) — teleport = transport only

All arm motion is driven by an external **hinge torque** (body-frame wrench about the
hinge axis) with a PD + gravity-feedforward servo (kp 1.8, kd 0.6, cap 4 N·m, 60 deg/s
slew, stall-escalation on gain); the load is delivered by free fall; the pour by the
machine's own dynamics.

1. **LEVEL** — servo the arm from the load stop (−20°) to θ = 0 and hold. No credit yet
   (asserted). → 0.000
2. **LOAD** — teleport the MIDDLE bowl (scene.target_idx) from the staging row to 10 cm
   **above the open scoop pocket** (arm frame), zero velocity — pure transport through
   free air; the solve asserts the score is unchanged by the write. Hands off: the bowl
   free-falls into the pocket; `_loaded_ever` fires on contact. → 0.150
3. **LIFT** — servo to 70°: bowl rides the pocket (arm-frame gravity presses it onto the
   floor/hinge-side wall). → 0.300
4. **DELIVER** — servo to 103°, just below the slick pocket's creep-exit angle (~104°,
   where tan(θ−90°) exceeds the pair μ ≈ 0.25). → 0.500
5. **POUR** — rate-servo the final quarter-swing at ~180 deg/s: centrifugal force pins the
   bowl in the pocket until the joint-limit slam at the 130° dump stop launches it
   tangentially (~0.7 m/s) through the window onto the grippy seat (μ 0.75), then press
   the stop with 1.2 N·m until `_courted_ever` fires. → 0.950
6. **RELEASE + SETTLE** — cut all torque; the bistable arm stays on the dump stop
   (asserted θ > 100° unpowered); the bowl settles seated → success. → 1.000
7. **PERSISTENCE** — ≥ 3.5 simulated seconds fully hands-off; `success()` must still hold
   before `SIM_GEN_SOLVE: SUCCESS` is printed.

## Embodiment argument (single Franka, 8 cm parallel jaw, OSC)

The same plan executes with one arm:

- **Load the scoop:** the bowls are 6.2 cm-wide cups — within the 8 cm jaw for an outside
  pinch (and the open top admits a rim pinch regardless). Read the row for the middle
  identity, pick it off the pad (z ≈ 0.13), and set it into the open-topped scoop pocket
  while the arm rests level; the pocket mouth (8.5 cm) exceeds bowl + 1.5 cm clearance by
  design assert. A short, low, uncluttered put.
- **Crank the lever:** the paddle is a flat plate on the lever end, faced away from the
  machine, sized for a fingertip/knuckle press. The servo torques cap at 4 N·m about the
  hinge; at the paddle's r ≈ 0.20 m that is ≤ 20 N of fingertip force, well inside Franka
  payload. The paddle's arc stays outboard of the pylon and below z ≈ 0.75, always
  approachable from the open side; the fast final quarter-swing is a 0.25 s wrist flick
  through ~30° of paddle arc — a push-through, not a tracked trajectory, because the
  dump stop absorbs the end of travel.
- **One plausible base pose:** base at fixture-frame ≈ (0.55, −0.50), facing the cabinet
  flank. The staging pad (x ≈ −0.4), the level scoop (x ≈ 0.65, z ≈ 0.36), and the paddle
  arc (x ≈ 0.4–0.6, z ≤ 0.75) are all within the 0.855 m reach; the hand never needs to
  enter the window or the court.

No stage admits a shortcut: the roof forbids any drop; the window opens onto air 0.4 m up
(nothing to slide a bowl along, and window credit is gated on riding the arm anyway);
grabbing an arbitrary bowl fails the decoy vetoes, so the row must be read for identity.

## Execution order

Load-before-lift-before-window is enforced twice: physically (the scoop is the only thing
that reaches the window; a bowl at the window without the arm is in mid-air) and by the
rubric latch chain (each stage requires the previous latch for that same body; an empty
crank then a top-fed bowl at dump angle earns nothing — smoke check 16).

## Checks (`smoke.py`) — 16

1. reset settles, states finite, arm on the load stop, three bowls in a row on the pad,
target strictly the middle, court empty; 2. score ≤ 0.005, no latches at reset;
3. randomization READBACK across 6 seeds (fixture xy/yaw, TARGET BODY INDEX varies,
lift-base jitter, row span, bowl yaw); 4. null policy 240 steps ≈ 0; 5–7. oracle on seeds
0/1/2 (torque-crank replication of the solve: 0.15 → 0.30 → 0.50 → 0.95 → 1.0, persists
240 steps hands-off); 8–9. monotone score ladder strictly increasing, every pre-pour
partial < 0.951; 10. decoy-in-court forfeit (success flips off, score falls to latch sum);
11. latch permanence (decoy removed → success recovers; plate in court → off again);
12. seed-strategy negative (bowl released over the cabinet parks ON THE ROOF, readback z,
score 0); 13. teleport-bypass negative (bowl placed straight on the seat: live seated TRUE
but every latch refuses — score 0); 14. window-drop negative (bowl dropped at the sill:
live window predicate TRUE mid-fall but the windowed latch refuses without the ride
history); 15. force-push negative (bowl pushed from the sill into the court: live courted
+ seated TRUE, displacement verified, all latches refuse — score 0); 16. out-of-order
negative (empty crank to the dump stop earns nothing; a bowl then fed into the raised
pocket slides out and still earns nothing). Frames recorded to `frames.npz`.

## Forge verification

- `solve --headless --seed {0,1,2}`: all three print the ladder
  `SIM_GEN_SCORE 0.000 → 0.150 → 0.300 → 0.500 → 0.950 → 1.000`, hold success through
  the 3.5 s hands-off window, and end `SIM_GEN_SOLVE: SUCCESS`, rc = 0.
- `smoke --headless`: `SIM_GEN_SMOKE: ALL PASS 16/16`, rc = 0, `frames.npz` saved.
