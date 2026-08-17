# Task `libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i319` — Ballast-Press Cabinet

Scene: `ballast_press_cabinet` (env `simgen.ballast_press_cabinet`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet` —
"close the top drawer of the cabinet": the Franka pushes the drawer front along its
prismatic travel until the cabinet joint reads closed. One guided translation OF the
judged part; the drawer's position IS the goal.

Kept from the seed: a cabinet with a sliding drawer and the identical judged OUTCOME —
the drawer's front face flush with the cabinet face, "drawer closed", judged on the
drawer's terminal position. The drawer is a free rigid body captured in a channel
(floor slab, side walls, rear hard stop, roof — contact physics, not an articulation),
reproducing the seed's drawer by mechanism rather than by joint; its rear hard stop IS
the seated pose (q_stop = +3 mm, inside the 12 mm closed tolerance).

## Strategic difference

**vs the seed:** in the seed the robot pushes the judged part directly — the hand
supplies the closing force and the closing displacement. Here the robot never has to
touch the drawer, and — decisive — the hand CANNOT be the energy source of the judged
state: the drawer is closed by a WEIGHT-POWERED PRESS. A lever pivots above the cabinet
mouth: long green arm ending in an open-top three-cell BALLAST HOPPER, short arm a
solid steel COUNTERWEIGHT, and a steel PRESS BLADE hanging from the pivot in front of
the drawer's face. The robot's entire job is MASS REDISTRIBUTION: pick 40 mm steel
blocks off the deck and set at least TWO of them into the hopper cells. The task is a
live THRESHOLD demonstration — one block is provably below the tipping point (the lever
does not move under it; asserted in cfg with >= 0.40 N m margin and demonstrated
hands-off in the solve), two blocks provably overcome the counterweight (>= 0.60 N m
lift-off margin, >= 2.5x friction drive at the seat) and the lever heels over, its
blade sweeping the drawer to its rear stop. Success is the settled LIVE conjunction:
drawer seated AND lever heeled past 28 deg AND >= 2 blocks riding inside the cells. The
counterweight makes the conjunction honest in both directions: a drawer pushed shut by
hand leaves the lever up and the hopper empty (~0.55, never success — the seed's own
end state is sub-success by construction), and a hand-pressed lever swings back up the
moment it is released (the press persists only while the ballast stays aboard —
demonstrated in smoke).

**vs the corpus surveyed this session:**
- `close_the_top_drawer_i276` (same-seed sibling, read in full): a gravity RAM — the
  robot releases a ball down a chute and one ballistic IMPACT closes the drawer;
  impulsive momentum transfer, the energy bought by the drop height of a single
  projectile. Here nothing is ballistic and nothing is thrown: the blocks are set into
  cells essentially at rest, the closing energy is the quasi-static descent of resident
  ballast, and the machine has a mass THRESHOLD (1 vs 2 blocks) that the ram task has
  no analogue of. The ram succeeds at the instant of impact; the press holds only while
  the ballast keeps riding in the cells — remove it and the press lets go.
- `close_the_top_drawer_i98` (same-seed sibling, read in full): the robot continuously
  torque-servos a hinged gate through ~70 deg and a cam transmission converts the
  regulated rotation into the drawer's translation — sustained guided actuation of a
  transmission input. Here the robot never actuates the mechanism at all: it only
  loads passive weights; the lever's whole swing is hands-off statics.
- `close_drawer_i58` (read in full): the drawer closes itself by stored gravity on an
  inclined runway once obstructions are removed — the robot SUBTRACTS constraints.
  Here the runway is flat and nothing is stored: the robot must ADD mass to the
  machine; doing nothing leaves the drawer open forever.
- `turn_on_the_stove_i280` (counterweighted steelyard, read in full): the closest
  mechanism relative — there cubes dropped in a basket tip a beam and the judged
  outcome IS the beam's own angle (the valve state). Here the heeled lever is only the
  transmission: the judged outcome is delivered to a DIFFERENT captured body (the
  drawer pressed home by the blade), the rubric demands the drawer clause on top of
  the lever clause, and the 1-vs-2-block threshold is itself part of the demonstrated
  contract (one ballast credit with the lever provably unmoved).

## The machine (honesty by construction)

`scene.py` asserts the press contract in `__post_init__` from the SAME parts table the
spawner authors (`_lever_parts`): the torque budget integrates every lever part
(structure at 900 kg/m3, counterweight and blade steel at 7800) into Sx, Sz and
requires — at the WORST-CASE block positions — that ONE block anywhere in the cells
leaves >= 0.40 N m restoring torque (the lever rests up) while TWO blocks leave
>= 0.60 N m of heel-ward torque at lift-off and >= 2.5x the drawer's friction torque at
the seated angle; that the hopper floor is sunk so a resting block's CoM sits AT hinge
height (the margins become angle-invariant: both lever arms scale with cos(theta)
together); that the seated press angle theta_seat (32.1 deg, printed by the solve at
32.14) lies strictly inside the joint travel and above press_min_deg even at the
closed-tolerance edge; that the raised nose clears the widest sampled drawer; that the
cells admit a block with margin while a block riding a cell stays below the wall top
(the in-cell window cannot be satisfied by perching); and that the dock strips clear
the hopper footprint and the slab. The lever hangs on a spawn-authored generic D6
joint (rotY free between a raised stop at -0.5 deg and 36 deg, all other axes locked)
anchored to a heavy DYNAMIC base so the whole jointed pair teleports consistently at
reset. The lever carries a hinge DASHPOT (angular damping 15/s): a dropped block's
impulsive jolt dies within ~3 deg of swing — far short of the ~8 deg the nose needs to
reach the drawer — while the sustained two-block press still heels over in ~1 s. All
sliding surfaces share a slick material (mu 0.10/0.08, min combine, restitution 0).

## Teleport solution (`solve.py`) — TWO transport teleports, ZERO forces

The solve applies no force and no wrench, ever. Its two teleports are pure transport —
one block at a time lifted off the deck, lowered into an open hopper cell and released
at rest 3 mm above the cell floor (a hand setting a block down, not hurling it; the
gentle release keeps the one-block phase surgical — a block dropped from above the
wall was measured on the forge to jolt the lever into a transient 15-27 mm tap on the
drawer before the counterweight recovered):

- P0 — settle; read back base pose/yaw, q0 (drawer readback matches the sample),
  lever at its raised stop, all three dock positions; score ~0.
- P1 — TRANSPORT A: block A released in the centre cell. The threshold is
  demonstrated live: the lever does NOT move (theta stays at -0.5 deg), the drawer
  does NOT move (< 5 mm), exactly one ballast credit (~0.15).
- P2 — TRANSPORT B: block B released in the +y cell; hands off from the release. Two
  blocks out-torque the counterweight: the lever heels, the blade sweeps the drawer
  to its rear stop (~0.8 s), everything settles at theta = 32.14 deg with the drawer
  at q_stop = 3 mm; success() live; score 1.0.
- Persistence — >= 3.3 simulated seconds hands-off, then `SIM_GEN_SOLVE: SUCCESS`.

Passes on forge on seeds 0, 1, 2 (yaws +170 / spread / -153 deg, q0 127/100/96 mm,
distinct dock layouts). `SIM_GEN_SCORE` printed at each boundary, non-decreasing
(0.00 -> 0.15 -> 1.00 -> 1.00). The drawer and the lever are never teleported and
never wrenched: the judged closing translation is delivered entirely by the ballast's
weight through the machine.

## Embodiment argument (Franka, single arm)

Base ~0.55 m out on the +x (deck) side of the station; everything touched lives at
world z 0.30-0.63 within a ~0.5 m disc — inside Franka's workspace at any episode yaw
(the +/-5 cm jitter and free yaw only rotate the approach; heights never change).

- STEEL BLOCKS: the only things the robot must touch. 40 mm cubes (well under the
  80 mm jaw span), 0.60 kg each (far under the 3 kg payload), resting in the open on
  flat deck strips with nothing above them — a standard top grasp. Carry ~25 cm up and
  over to the hopper mouth (an open-top 56 mm square cell per block, open air above)
  and release over a cell. A grasped block lowered part-way in and released, or simply
  dropped from just above the mouth, both end riding the cell — the walls funnel it;
  the dashpot means a from-the-mouth drop still converges to the same judged end state
  (the solve's extra-gentle 3 mm release is for the clean one-block threshold
  demonstration, not a success requirement). No re-grasp, no bimanual step, no
  precision beyond "over the chosen cell".
- The DRAWER never needs to be touched: the machine closes it. Pushing it by hand is
  physically possible and honestly scored (~0.55) but can never satisfy the success
  conjunction — the lever clause and the in-cell clause require resident ballast.
- The LEVER never needs to be touched either; hand-pressing it is demonstrated
  (smoke) to be non-persistent — the counterweight returns it the moment the hand
  lets go.

## Ordering

The rubric imposes no order — success is a live conjunction on the settled end state.
ANY two of the three blocks in ANY two cells, loaded in any order, succeed; the third
block is a spare. The only inherent ordering is physical: the press fires when the
second block boards. Partial-credit latches (per-block in-cell residency, channel-
guarded max closure fraction) only anchor demonstrated progress; success itself is
latch-free and judged live.

## Rubric

- 0.15 per latched ballast block (residency-streak latch, max two counted).
- 0.55 x latched max closure fraction of the drawer's initial opening
  (channel-guarded: a drawer stolen out of its channel earns nothing).
- Base capped at 0.85; exactly 1.0 iff `success()`: drawer seated (front face within
  12 mm, riding its channel) AND lever heeled >= 28 deg AND >= 2 blocks riding inside
  the hopper cells, all settled, finite and LIVE.
- Null policy ~0; the seed's end state (drawer pushed shut, hopper empty) ~0.55,
  never success.

## Checks (`smoke.py`, rejection-only, 13 checks)

settle/no-NaN (drawer at q0 in channel, lever on its stop, blocks docked, score ~0);
randomization A (yaw span > 90 deg, xy jitter); randomization B (q0 span AND dock
layout vary, settled readback tracks both every reset); null policy ~0; SEED strategy
(oracle 2.5 N push seats the drawer for real — hopper empty, lever up, ~0.55, no
success); one-block threshold (one block in a cell: lever provably stays up, drawer
unmoved, ~0.15, no success); hand-press (oracle torque heels the lever past the press
angle and seats the drawer, but on release the counterweight swings it back up —
~0.55, no success); combined near miss (drawer seated + one block riding: 2 of 3
clauses, ~0.70, no success); channel guard (drawer stolen onto the cabinet roof with
its face x reading "closed" — no drawer credit, no success); latch survival (block
latched in a cell then returned to the deck: latch survives, live count 0, no
success); rejection audit (success never observed anywhere in the battery); final
no-NaN; frames.npz video recorded. Result: `SIM_GEN_SMOKE: ALL PASS 13/13`.
