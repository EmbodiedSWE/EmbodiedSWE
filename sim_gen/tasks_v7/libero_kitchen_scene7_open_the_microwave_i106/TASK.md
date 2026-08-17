# libero_kitchen_scene7_open_the_microwave_i106 — `microwave_carousel`

Rotate the microwave's loaded turntable — cargo riding on friction — until the RED
cup faces the wide-open door, without dragging, tipping, or shedding either vessel.

## Provenance

- **Seed task**: `libero_90/libero_kitchen_scene7_open_the_microwave`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene7_open_the_microwave.py`)
  — a Franka grasps the microwave door handle and pulls the hinged door past a fixed
  joint-angle threshold (`joint_pos < -1.5`). One grasp, one unidirectional pull,
  judged by a single articulation joint angle. The seed's END state (door wide open,
  contents untouched) is exactly this task's START state, so the reversed/completed
  seed strategy earns ~0 by construction (smoke check 5).
- **Scene**: `microwave_carousel`, registered as `simgen.microwave_carousel` with
  `robot="null"`.

## Strategic difference (vs the seed and the corpus read this session)

The seed is **articulation actuation**: pull ONE hinged panel one way past a
threshold; any wide-open pose counts and nothing else is judged. Here the door
opening is pre-completed static scenery (the door is authored mid-swing as part of
the kinematic shell) and the task lives on the microwave's OTHER degree of freedom:

- a **free-spinning platter** (spawn-authored frictionless-limit revolute, no stops)
  carries a red cup and a blue bottle at fixed spots;
- the goal is **bidirectional precision rotation**: bring the CUP's world bearing
  into a ±20° window facing the doorway — the short way around flips per seed;
- the cargo is coupled to the platter by **friction only**, so the real constraint is
  care: both vessels must arrive upright ON their original platter-frame spots
  (3-D slot invariance, 25 mm), not dragged across the platter, tipped, or shed;
- a same-ring **distractor** (the bottle, ≥ 100° away) makes identity load-bearing.

A solver needs a different **plan** (read a randomized angular layout, choose a
rotation direction, transport cargo indirectly through a mechanism it never touches,
stop inside a window instead of pulling past a threshold) and a different **code
structure** (platter-frame slot-invariance + world-azimuth window + uprightness, not
a joint-angle readout). Unlike the corpus read this session: `screw_nail_i59`
(ordered latch-unlock/uncover/retrieve chain), `pen_holder` (insertion),
`draw_triangle` (drawing) — no other task read is an indirect-transport carousel.

## The mechanism is real (numbers)

- The platter is a plain dynamic disc (0.40 kg, r = 0.15 m, explicit diagonal
  inertia) on a spawn-authored vertical revolute joint against the kinematic shell;
  it floats 4 mm above the cavity floor (asserted) so nothing rubs.
- Torque cap 0.012 N·m (fingertip-scale rim push) → rim acceleration ≈ 0.15 m/s²
  and centripetal ≤ 0.06 m/s², both ~100× under the friction budget μg ≈ 8 m/s² —
  gentle drive genuinely transports the cargo; a slam would slide/topple it.
- Anti-cheat geometry asserted in `__post_init__`: milestone ladder leaves ≥ 25° of
  null margin; the ≥ 100° cargo separation makes cup-at-front and bottle-at-front
  mutually exclusive; the minimum drag chord from any spawn to the doorway (97 mm)
  is > 3× the slot tolerance, so a drag shortcut cannot read as riding. Milestones
  latch ONLY while both vessels are riding intact at that instant, so the drag cheat
  latches nothing (smoke check 7). An equally-hard alternative route — lift the cup
  and re-place it EXACTLY on its moving platter slot at the front — is accepted by
  design: it requires strictly finer precision than rotating the platter.

## Solution outline (solve.py — NOTHING is teleported)

1. **P0**: reset, 0.5 s settle, layout readback; assert score ≤ 0.02 and
   |cup bearing| ≥ 90°.
2. **P1 rotate (contact dynamics)**: PD torque servo on the platter body
   (K = 0.03 N·m/rad on the cup's SIGNED wrapped bearing error, D = 0.04, cap
   0.012 N·m — the proxy for a fingertip pushing the rim/pegs sideways). The wrapped
   error makes the short way the descent direction on every seed; friction carries
   the cargo; ζ ≈ 1.9 (overdamped). `SIM_GEN_SCORE` printed as each riding milestone
   (65°, 40°) latches.
3. **P2**: torque cut inside 6° at < 0.08 rad/s (coast ≈ ω/3 rad, damping 3.0 stops
   the platter); hands-off settle to `success()`.
4. **P3**: 3.3 s (400-substep) hands-off persistence → `SIM_GEN_SOLVE: SUCCESS`.

Passes seeds 0 and 1 on the forge (`SIM_GEN_SOLVE: SUCCESS`).

## Rubric

- `success()`: RED cup's bearing from the cavity axis within 20° of the doorway
  direction AND both vessels upright (≤ 15°) ON their reset platter-frame slots
  (≤ 25 mm, 3-D) AND platter + cargo settled.
- `score()` (latched in `post_step`, never evaporates): 0.25 when the RIDING cup
  first comes within 65° of the doorway + 0.20 more within 40° (cap 0.45); 1.0 iff
  `success()`. Milestones gate on instantaneous cargo integrity. Null policy ~0
  (cup spawns ≥ 95° off-door).
- Per-episode randomization (readback-verified in smoke): platter yaw (full circle),
  cup start bearing 95–175° off-door on EITHER side (direction choice flips), bottle
  separation 100–160° on either side, both vessels' own yaws.

## Embodiment argument (single Franka arm, OSC, one base pose)

Base at the env origin (0, 0) on the floor, facing +x; the cavity front plane is
0.34 m away, the platter center 0.55 m, the far rim ≤ 0.70 m — inside Franka reach.
The entire front of the cavity is open (opening y ∈ [−0.19, 0.19],
z ∈ [0.32, 0.58]; the open door is parked outside on the +y side): the gripper
enters horizontally through the doorway. **Contact strategy per object**: the
PLATTER is driven with the closed gripper's fingertip pushing tangentially on a
white peg (Ø 18 mm, tops at z ≈ 0.394, the near peg passes ≈ 0.42 m from the base)
or on the disc rim — a sequence of short lateral nudges in the chosen direction,
re-approaching as the pegs precess; the CUP and BOTTLE are never touched (they ride).
Stopping inside the ±20° window needs only visual servoing on the red cup with
gentle final nudges (each capped push moves the platter smoothly; coast after a
nudge is ≈ ω/3 rad, i.e. a few degrees). **Execution order: none mandated** — the
task is a single mechanism interaction; the only "ordering" is physics (rotate
until aligned, then stop pushing).

## Files

- `scene.py` — cfg + scene: procedural compound spawners (kinematic shell with the
  door authored mid-swing, platter with pegs, cup, bottle), spawn-authored free
  revolute joint, slot-invariance predicates, latched score. Registers
  `microwave_carousel` and `simgen.microwave_carousel`.
- `solve.py` — torque-servo solution (above); watchdog + hard exit.
- `smoke.py` — 16-check rejection battery (records `frames.npz`): reset/layout
  sanity, randomization readback over 6 seeds, null policy (= the seed's end
  state), non-vacuous torque probe (the capped rim torque really turns the loaded
  platter AND the cargo rides it — pointed away from the doorway so it earns
  nothing), drag cheat, wrong object at front, tipped cup at front, knocked bottle
  with a perfect cup, cup-on-floor at front, near-miss at 28°, settle gate,
  latched-credit persistence, success-never-True audit, no-NaN.
  `SIM_GEN_SMOKE: ALL PASS 16/16`.

## Check list

- [x] Strategically different from the seed (bidirectional friction-transport
      rotation with a care constraint vs a one-way door pull past a threshold) and
      from every task read this session.
- [x] Solution is pure contact dynamics (bounded torque on the mechanism); nothing
      is teleported; cargo transported only by friction.
- [x] `SIM_GEN_SCORE` non-decreasing at phase boundaries; ≥ 3 s persistence after
      success; hard exit with watchdog.
- [x] Solve passes on ≥ 2 seeds (0 and 1) on the forge.
- [x] Rubric rejection battery passes on the forge; success never True in it.
- [x] describe()/instruction() state goal, mechanism, randomization, and what does
      not count; all claims physically true (geometry asserted, mechanism
      force-probed in smoke).
- [x] Fully procedural geometry; no external assets; files only in the task dir.
