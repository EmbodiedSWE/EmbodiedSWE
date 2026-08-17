# drawer_bead_pour — pull the drawer open, pour the beads in, park the bowl

`sim_gen` task `libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i134`
— env `simgen.drawer_bead_pour`, scene `drawer_bead_pour`, robot slot `null` (scene-level task).

## Seed provenance

Derived from RoboVerse `libero_90/kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet`
(`roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.py`):
pick the akita black bowl off the table and place it inside the bottom drawer of the
white cabinet; `_terminated` is a single bowl-centre-inside-the-drawer OBB test (the
drawer bbox displaced by the drawer joint), with a wine bottle and wine rack as inert
distractors.

## What changed, and why it is strategically different

The scene keeps the seed's cast — a floor cabinet with a sliding bottom drawer, and the
black bowl — but **inverts the bowl's role**: it is no longer the cargo, it is the
*pitcher*. The bowl starts on the floor holding 2–4 loose yellow beads; the goal is
that **every bead ends up resting loose on the drawer's floor, out of the bowl**, and
the emptied bowl is set back down upright on the open floor. The seed's entire terminal
relation — bowl inside drawer — is an explicit **failure state** here (`bowl_parked`
requires `~bowl_in_drawer`, and a bead sitting in the bowl is "not deposited" even when
the bowl is inside the drawer; the smoke battery constructs exactly this state and the
rubric rejects it).

Structural differences from the seed (and from every task I read):

- **Judged relation**: contents-transfer per item (N beads loose in the drawer, bowl
  excluded) vs the seed's single-object containment test. The rubric is a latched
  4-term curriculum (open / gated carry / deposit fraction / park) instead of one bbox
  check.
- **Mechanism**: the drawer here is a *live, springless prismatic slide* the solver
  must actually pull open and that stays where it is left; opening is order-forced by
  a metric interlock — the shut drawer clears the cabinet top by an 8 mm slit, far
  under the 24 mm bead diameter, and its front face is solid, so no content transfer
  is possible until the drawer is out. The seed ships the drawer state in the demo
  trajectory and only reads its joint in the checker.
- **Tool use + tool return**: the task ends with a "put the tool back" clause — the
  bowl must be re-parked upright at floor height, still, outside the drawer.
- **Decoy receptacle**: an always-open top-shelf compartment sits above the drawer;
  beads left there (zero effort, no interlock) score nothing.
- Not a variant of the sibling tasks either: no bistable hinged bin / closure goal /
  decoy object (i237 `tilt_bin_stow`), no freed drawer being re-installed into a bay
  (i102 `drawer_refit`), and the container is poured *out of*, not filled (pen-holder
  exemplar).

All assets are procedural compound spawners (no external files): kinematic shell
(plinth, bay, overhanging top panel, shelf walls), dynamic drawer (open-top box +
protruding blue handle bar) bound to the shell by a bind-time `UsdPhysics.PrismaticJoint`
(limits [−0.16, 0] m, joint-pair collision off, heavy linear damping = slide friction),
dynamic octagonal black bowl (8 wall boxes + floor disc), 4 dynamic yellow beads
(a random 2–4 subset is present per episode; absent beads park in a far ground depot).

**Randomization** (readback-verified in smoke): present bead count 2–4, Bernoulli
left/right bowl slot swap, ±3 cm slot jitter, random bead ring phase in the bowl.

## Teleport-solution outline (solve.py)

Teleports are transport-only; every load-bearing interaction is contact dynamics:

1. **Open** — a bounded (≤10 N) horizontal servo force on the drawer body (the
   applied-wrench emulation of pulling the blue handle) drives the drawer out along
   its real slide; the drive is then removed and the springless slide holds the pose.
   The drawer is never pose-written after reset.
2. **Carry** — the bowl is held (per-step root-state writes with path-consistent
   velocities, smoothstep paths) and carried: lift, traverse, hover 25 cm above the
   exposed drawer mouth. The beads are **never written**: they ride in the bowl
   through real contacts, and the solve asserts none spilled.
3. **Pour** — the held bowl tilts to ~115° over the mouth; beads roll over the rim,
   fall through the mouth under gravity, land and settle on the drawer floor through
   real impacts (translational shake fallback for stragglers). Nothing writes the
   beads or the drawer.
4. **Park** — the emptied bowl is carried back and released just above the floor;
   it settles upright under gravity. `success()` first turns True on settled state,
   then the solve holds hands-off ≥3.3 sim-seconds before `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` prints at every phase boundary and is asserted non-decreasing
(latched credit). Verified on the forge: seeds 0 and 1, `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka, base at the world origin)

Every load-bearing interaction is a standard Franka contact strategy, all within a
0.34–0.63 m reach annulus of one base pose at the origin:

- **Drawer pull**: the blue handle bar (30×90×14 mm, 135 mm off the floor, on the
  front face at x ≈ 0.47–0.50) is a parallel-jaw pinch target; the pull is a straight
  −x Cartesian drag of ≤10 N over 16 cm — well inside Franka payload and workspace.
  The solve's bounded horizontal servo force is exactly this interaction's wrench.
- **Bowl carry/pour**: the bowl's 8 mm octagonal rim is a jaw-width pinch grasp
  (opening ≥ 8 mm + clearance); carrying at ≤0.25 m height and wrist-rolling to 115°
  over the drawer mouth is a single wrist-joint rotation with the arm nearly static —
  the classic pouring primitive. Bowl + beads mass ≤ 0.23 kg.
- **Park**: lowering and releasing the bowl on the open floor at (≈0.24–0.31, ±0.3)
  is a plain place. No interaction requires more than one gripper or exotic contact.

## Execution order

1. `scene.py` — scene, spawners, slide joint, rubric (written first).
2. `solve.py` — iterated on the forge until SUCCESS (seeds 0, 1).
3. `smoke.py` — rejection battery, iterated on the forge until ALL PASS.
4. `TASK.md` — this file.

## Check list (smoke.py, 14 checks)

1. Settle: states finite, drawer shut, every present bead in the bowl, all still.
2. Score ~0 at reset, no success.
3. Randomization readback: bowl side flips AND present bead count varies across seeds.
4. Randomization readback: per-slot xy jitter > 4 mm.
5. Null policy (240 idle steps): score ~0, no success.
6. **Seed strategy**: bowl + beads settled *inside the open drawer* — zero deposit
   credit, NOT success, score ≤ 0.5 (role inversion is load-bearing).
7. Shelf decoy: all beads in the top shelf, bowl parked — score ~0, no success.
8. Floor scatter: beads loose on the floor, bowl parked — score ~0, no success.
9. Partial near-miss: all present beads but one deposited — strictly partial credit,
   NOT success, score ≤ 0.85.
10. Bowl on its side after full delivery — park upright cone rejects, NOT success.
11. Bowl upright but standing in the shelf compartment — floor-height clause rejects.
12. Latched credit: teleporting deposited beads back out leaves the score unchanged.
13. Rejection audit: success() never True at any judged point in the battery.
14. Final no-NaN on all task objects.
