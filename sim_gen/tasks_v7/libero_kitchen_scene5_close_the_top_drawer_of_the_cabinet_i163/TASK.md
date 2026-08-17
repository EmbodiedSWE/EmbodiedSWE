# Task `libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i163` — Counterweight Bell-Crank Press

Scene: `weight_crank_cabinet` (env `simgen.weight_crank_cabinet`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet` —
"close the top drawer of the cabinet": the Franka pushes the drawer front along its
prismatic travel until the joint reads closed. One guided translation OF the judged
part; the robot's hand supplies both the guidance and the energy.

Kept from the seed: a kitchen cabinet with a sliding drawer at chest height and the
identical judged OUTCOME — the drawer seated shut, judged on the drawer's terminal
position. The drawer rides a real prismatic joint whose limits are the hard stops,
exactly the seed's articulation.

## Strategic difference

**vs the seed:** in the seed the robot pushes the judged part and its arm is the
energy source. Here the robot never has to touch the drawer or drive any mechanism
through its travel: the intended skill is a PICK-AND-PLACE OF A FREE PAYLOAD plus
MASS DISCRIMINATION. A gravity-biased BELL CRANK hangs on a revolute pivot in a
gallows frame before the cabinet: a horizontal pan arm ends in a deep amber receiving
POCKET, a hanging press arm ends in a red roller nose facing the drawer's handleless
front plate. Dropping the heavy BRASS cube (0.9 kg) into the pocket makes the
payload's own weight overpower the crank's return bias: the crank swings down its
30° arc, the roller cam-presses the plate, and the drawer translates shut. The
ENERGY that closes the drawer is the payload's gravity — the robot only relocates a
0.9 kg cube. A visually similar FOAM cube (45 g) can never reach half the bias
torque: choosing the right payload by mass is part of the task. The seed's exact
skill (hand-push the drawer shut), executed for real by smoke with an 8 N PD push,
seats the drawer but leaves the pan empty and the crank up — drawer credit only
(~0.35), never success.

**vs the corpus surveyed this session:**
- `close_the_top_drawer_i98` (cam-gate): there the robot actively DRIVES the
  transmission's input link (a hinged gate) through a ~70° arc — the hand is still
  the energy source, metered through a cam. Here the robot never drives any link:
  smoke's hand-crank probe demonstrates that driving the crank by hand closes the
  drawer but can never succeed (the bias returns the crank up the moment the hand
  lets go; the pan is empty). The input to my machine is a RELEASED MASS, not a
  guided sweep — and i98 has no wrong-object discrimination at all.
- `open_bottom_drawer_i87` (rocker): robot presses/rides a rocker to open a drawer —
  again a driven input link, energy from the arm, and an OPENING goal. Opposite
  energy source here, and the goal state includes the machine loaded (weight riding
  the pan at the down-stop), not just the drawer coordinate.
- `close_drawer_i58` (incline): remove two obstructions and stored gravitational
  energy (an inclined runway) closes the tray — "clear the path, hands off". Here
  there is NO stored energy against the goal (the crank's bias points AWAY from it,
  the slide is horizontal): the robot must ADD the energy by placing mass into the
  machine. i58 subtracts objects; i163 inserts the correct one of two.
- `pen_holder` (packing exemplar): pick-and-insert into a passive container —
  insertion IS the goal. Here the pocket is not a goal container but a POWER INPUT:
  the insertion is judged only through what the payload's weight then does to a
  two-link transmission, and an identical-looking wrong payload inserts just as
  easily and achieves nothing.

## The machine (honesty by construction)

`scene.py` asserts the full transmission contract in `__post_init__`:

- Cam relation: nose surface plane `x(θ) = pivot_x − hang_len·sinθ − nose_r`, so the
  drawer is pressed to `q(θ) = x(θ) − lip_t`; at the 30° down-stop q_press ≈ 2 mm —
  inside the 10 mm success tolerance — while the drawer's own front stop sits deeper
  (0), so the terminal state is a static force balance: weight-on-pan vs
  crank-on-stop + nose-on-plate. At rest (0°) the nose clears the widest sampled
  opening by ≥ 20 mm and even the full stroke by ≥ 8 mm; the nose stays on the front
  plate (z-span checked with radius margin) through the whole arc.
- Torque contract: bias τ = crank_mass·g·com_x (authored CoM offset — no springs, no
  stored energy against the goal). The brass cube at the innermost pocket radius and
  full tilt still overpowers bias 3× (+0.10 N·m margin); the foam decoy at the pan's
  outermost point never reaches HALF the bias. Mass discrimination is a theorem of
  the config, not a tuning accident.
- The nose↔plate cam pair is bound to a slick material (μ 0.06/0.05, min combine):
  with PhysX's default ~0.5 friction the nose would drag the plate instead of
  sliding — the material is load-bearing.
- Pocket captivity (cube top below the wall top with margin, ≥30/20 mm lateral
  clearance), drop-corridor/beam/post/apron clearances, cube spawn bands clear of the
  drawer travel and the nose swing, and the no-overlap side split are all asserted.
- Drawer and crank are authored in place and JOINTED to the kinematic station at
  bind time (limits = hard stops), so "stolen drawer"/"off-pivot crank" fakes are
  structurally impossible; `drawer_in_channel`/`crank_on_pivot` remain as defensive
  guards. The crank's CoM sits near the pivot, so `settled()` judges the crank on
  ANGULAR velocity.

## Teleport solution (`solve.py`) — one transport teleport, zero wrenches

- P0 — settle 120 steps; read back q0 and both cubes' sampled poses (never
  hard-coded); crank on its up-stop; baseline score ~0 (asserted).
- P1 — TRANSPORT (the pick-and-carry): ONE teleport places the brass weight at rest
  just ABOVE the rubric's crank-local load box over the pocket mouth, computed from
  the MEASURED crank pose. Score unchanged (asserted ≤ 0.05).
- P2 — RELEASE: gravity carries the cube into the deep pocket (~8 steps); its torque
  overpowers the bias; the crank swings down; the roller cam-presses the drawer shut
  (~30 hands-off steps); crank lands on its down-stop. The drawer is never touched,
  wrenched or teleported. Score 0.30 at landing, 1.0 at success — non-decreasing
  `SIM_GEN_SCORE` at every boundary.
- Persistence — ≥ 3.3 simulated seconds hands-off with success() still true, then
  the whole episode repeats on a second seed. Passes on forge, seeds 0 and 1
  (q0 = 0.100/0.118, sides swapped between seeds), 20 s wall clock.

## Embodiment argument (Franka, single arm)

Base ~0.72 m out on the +x side, facing the cabinet: everything the robot must touch
lies in a 0.10–0.50 m x-band at z 0.40–0.93 — comfortably inside Franka's workspace.

- BRASS CUBE: the only thing the robot must touch. A 5 cm cube (≪ 8 cm jaw span)
  standing free on the apron bench top (z ≈ 0.43, x 0.03–0.27, |y| 0.02–0.11) with
  open air above — a standard top or side grasp at 0.9 kg (well inside payload).
  Carry: lift ~0.5 m and hover over the pocket mouth (pocket centre x ≈ 0.45,
  z ≈ 0.87 at the crank's rest pose), then open the gripper. The drop corridor above
  the pocket is asserted clear: the crossbeam sits ≥ 10 cm behind the pocket's inner
  edge and the pocket walls capture the cube even released a few cm high — release
  accuracy of ±2 cm suffices (inner well 10 × 8 cm).
- FOAM CUBE: never touched in the intended plan (grasping it is harmless and
  useless).
- DRAWER / CRANK: never touched. The drawer front is a smooth plate with no handle;
  a hand CAN push it (smoke does, force-limited) — it seats the drawer and earns no
  success. The pan arm CAN be pressed down by hand (smoke does, torque-limited) —
  the bias undoes it on release.

## Ordering

The rubric imposes no order — success is judged live on one settled terminal state:
weight riding the pan, crank at its down-stop, drawer seated. The intended plan is
one action (load the weight); the machine does the rest in a single causal cascade.
Hand-pushing the drawer shut FIRST and then loading the brass weight is an honest
alternate route and legitimately succeeds — it is strictly more work and still
requires the novel skill (selecting and loading the heavy payload); the crank then
swings to its stop unopposed and holds the already-seated drawer. What can NEVER
succeed: any terminal state with the pan empty (bias holds the crank up; load clause
refuses constructed cranks-down instantly) or with the foam decoy as the payload
(torque theorem). The three partial-credit latches (`_floaded`, `_fcrank`,
`_fdrawer`) only anchor demonstrated progress and are capped at 0.90; success itself
is latch-free and live.

## Rubric

- 0.30 — brass weight ever riding the pan assembly (crank-local load box,
  on-pivot guarded; latched).
- 0.25 × latched max crank closure fraction θ/θ_max (on-pivot guarded).
- 0.35 × latched max drawer closure fraction (q0 − q)/q0 (channel guarded).
- Cap 0.90 without success; exactly 1.0 iff `success()`: weight on pan AND crank at
  its down-stop (≥ 27°, on pivot) AND drawer seated (≤ 10 mm, in channel), all
  settled (crank judged on angular velocity) and finite — live.
- Null policy ~0; seed strategy (hand-push) ~0.35, never success; hand-cranking the
  press ~0.60 latched, never success.

## Checks (`smoke.py`, rejection-only, 12 checks)

settle/no-NaN (crank on up-stop, drawer at q0, cubes staged, score ~0);
randomization A (q0 span > 12 mm, settled readback tracks every reset);
randomization B (cube x span, side split seen both ways, decoy always opposite,
cubes settle on the apron); null policy ~0; SEED strategy (real 8 N PD push seats
the drawer — pan empty, crank up, ~0.35, no success); latch regression (drawer
pulled back open with a real force — `_fdrawer` survives, live clause reads open);
DECOY drop (foam cube really dropped into the pocket — lands and stays, crank and
drawer unmoved, score ~0: mass discrimination); hand-crank drive (real force-limited
torque servo to the down-stop — the drawer cam-closes, proving the transmission,
then the bias returns the crank up on release; ~0.60, never success); stolen-weight
fake (crank constructed down + drawer seated + weight far away → load clause refuses
instantly, bias swings the crank back up); rejection audit (success never observed
anywhere in the battery); final no-NaN; frames.npz video recorded.
