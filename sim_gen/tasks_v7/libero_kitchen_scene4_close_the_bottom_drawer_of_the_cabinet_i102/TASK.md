# Task `libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_i102` — Drawer Refit

Scene: `drawer_refit` (env `simgen.drawer_refit`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet` —
"close the bottom drawer of the cabinet": one guided push on a drawer that already
rides its cabinet's prismatic joint until the joint reads closed.

Kept from the seed: a cabinet with drawer bays, a drawer, and the identical judged
OUTCOME — the drawer's face flush with the cabinet face, "drawer closed", judged
on the drawer's terminal pose. Also kept: the seed's cabinet has TWO drawers (top
and bottom) and only a designated one counts — here that designation survives as
the blue tag chip marking the target bay.

## Strategic difference

**vs the seed:** in the seed the drawer is IN the cabinet, riding an articulated
prismatic joint — the fixture carries all six minus one DOF and the whole skill is
one push along the remaining one. Here THERE IS NO JOINT AND THE DRAWER IS NOT IN
THE CABINET: it stands on the ground as a free 6-DOF rigid body next to a green
payload block, its bay is empty, and "close the drawer" becomes RE-INSTALLATION —
(1) LOAD the block into the drawer's open-top basin while it is still accessible,
(2) carry the loaded drawer up to the sideboard, READ which of the two identical
bays carries the blue tag chip (resampled every episode), align all six DOF with
that bay's face opening, and slide the box home through the opening until the
oversized face plate lands flush on the jamb fronts. The seed's plan is one 1-DOF
push on a constrained body; this plan is perception (which bay?) + payload
containment + a guided 6-DOF insertion whose lateral/vertical guidance the agent
must supply itself until the jambs and bay floor take over. The seed's skill
executed for real is provably useless — smoke pushes the grounded drawer with a
real 8 N PD hand-push for 3 s: it slides into the plinth and stops on the ground,
score ~0 (the bays are elevated; insertion credit is gated on being inside the
target bay).

**vs the corpus surveyed this session:**
- `close_drawer_i58` (Return Dock, the sibling drawing on the same "drawer
  closes" outcome): there the robot NEVER touches the slide — the task is
  obstruction removal and gravity closes the tray by itself. Here the inverse:
  the robot manipulates the judged body through its ENTIRE trajectory, and the
  new axis is that the "drawer" must first be re-mated with its fixture (plus a
  payload constraint with a load-before-close order). No stored-energy mechanism,
  no interlocks, nothing self-acts.
- `..._close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_i34`
  (Gumball Meter): a force-driven prismatic shuttle the robot drives REPEATEDLY,
  judged on an exact cycle count plus restraint. Here there is no articulation at
  all, nothing is counted, and the drawer is driven exactly once — the difficulty
  is 6-DOF mating, not metering.
- `pen_holder` (robobench packing exemplar): repeated vertical drop-ins into a
  passive container that never moves. Here the container itself is the
  manipulated, judged body: it must be loaded AND then become the thing inserted,
  through a side-facing opening with a flush stop, into a fixture that constrains
  it — and which receptacle counts is episode-dependent (the tag chip).

## The machine (honesty by construction)

`scene.py` asserts the refit contract in `__post_init__`: the box passes the
opening with honest clearance (>= 10 mm per side, >= 25 mm overhead); the face
plate CANNOT pass (plate_w 190 mm > opening 164 + 20 mm) and bears on the jamb
fronts — the flush pose is only producible by a real slide through the opening;
the plate stops the travel before the box rear can reach the rear wall (bay_d
172 > box_l 160 + tol) so the judged stop is the visible face-frame contact; the
50 mm block fits the basin and rides fully below the box top, but the roof gap
over an installed drawer (open_h − box_h = 30 mm) is sub-block AND the flush
plate seals the whole opening (plate_h = open_h) — loading after closure is
physically impossible, so the load-before-close order is inherent, not decreed;
a block left loose in the bay arrests the drawer >= 10 mm short of flush
(box_l + cube > bay_d + tol) — "just dump the block in the bay" jams the very
travel being judged; the tag chip sits clear of the plate's flush sweep; the two
bay sills are separated far beyond the z tolerance so the target-bay height band
cannot confuse them.

## Teleport solution (`solve.py`) — teleports are transport only

- P0 — settle; read back board pose/yaw, WHICH bay is tagged (verified against
  the tag chip's height), drawer and block ground poses; baseline ~0.
- P1 — LOAD: PD force + gravity feedforward (<= 4 N, a firm grasp) lifts the
  block into free air; the held block is teleported over the grounded drawer's
  open top and RELEASED — gravity drops it into the basin; the block latch is
  judged on that settled contact outcome. Score 0.15.
- P2 — TRANSPORT: the same carry (<= 12 N) lifts the LOADED drawer into free air
  — the block rides inside on real contact — then the held assembly is teleported
  (block's drawer-relative pose preserved) to a hover at the tagged bay's mouth,
  aligned with the board, and PD-held there.
- P3 — INSERTION (applied force, three stages so the box can never wedge-climb
  the sill lip): (A) enter with the nose 6 mm above the sill; (B) set the box
  down onto the bay floor and press (feedforward at 0.9x weight = a steady
  down-preload); (C) slide flush along the floor — jambs guiding laterally —
  until the plate lands on the jamb fronts and the PD presses it home
  (settled gap ~1.4 mm < 8 mm tol). Score 1.0.
- Persistence — forces cleared, >= 3.3 simulated seconds hands-off, then
  `SIM_GEN_SOLVE: SUCCESS`.

Passes on forge on all nine seeds 0–8, covering both bays (bay 0: seeds
2,4,5,7,8; bay 1: 0,1,3,6), yaws spanning −132° to +164°, distinct board xy and
drawer/block ground poses. `SIM_GEN_SCORE` printed at each boundary,
non-decreasing (0.00 → 0.15 → 0.15 → 1.00 → 1.00).

## Embodiment argument (Franka, single arm)

Base ~0.55 m in front of the face plane on the apron (+x) side; everything
touched lives at world z 0.03–0.40 within a 0.5 m disc — comfortably inside
Franka's workspace at any episode yaw (the ±5 cm xy jitter and free yaw only
rotate the approach; heights never change).

- GREEN BLOCK: 50 mm cube (< 80 mm max jaw) standing free on open ground — a
  top-down side pinch, a 0.2 m carry, and an unconstrained release over the
  drawer's fully open top (basin mouth 144 × 124 mm vs the 50 mm block: 37 mm
  xy slack per side).
- DRAWER: 0.36 kg loaded — the yellow T-HANDLE bar (16 mm square section, 64 mm
  wide, standing 72–88 mm above the drawer bottom, protruding 57 mm clear of the
  plate face) gives a clean top-down parallel-jaw grasp on a bar much thinner
  than the jaw span; the oversized plate's top edge (12 mm thick, standing
  30 mm proud of the box walls) is a second pinch affordance for re-grips. The
  insertion is a straight horizontal slide at sill height (0.160 or 0.284 m —
  both in Franka's sweet band) with 12 mm lateral and 30 mm vertical clearance,
  and the fixture forgives: jambs funnel laterally, the bay floor carries the
  weight during the final push, and the flush stop is a hard contact the arm
  simply presses onto — no precision hold at the end. The final push force
  (~1 N of sliding friction) and the 3.5 N carry are trivial loads.
- Nothing must be reached inside the bay: the handle stays fully outside the
  face plane at flush (asserted), so the grasp never enters the cavity.

## Ordering

The rubric imposes no order; the only ordering is physically inherent: the block
cannot be loaded after closure (sub-block roof gap + the flush plate sealing the
opening — demonstrated by smoke with a real 8 N held press against the closed
face), so LOAD must precede CLOSE. Between-bay choice is perception, not order.
The three latches (`_cube_l`, `_eng_l`, `_ins_f` max-fraction) only anchor
demonstrated progress; success itself is latch-free and judged live (flush in
the tagged bay & block in the basin & settled & finite).

## Rubric

- 0.15 — block ever settled inside the drawer basin (latched).
- 0.10 — drawer nose ever >= 2 cm past the face plane, upright, in-lane, at the
  target bay's height (latched).
- 0.55 × latched max insertion fraction into the TARGET bay (gated on the same
  upright/in-lane/height band — ground-level pushing earns nothing).
- Cap 0.80 without success; 1.0 iff `success()`: plate gap < 8 mm, centred, at
  the tagged bay's floor, upright, facing out, block inside the basin, all
  settled and finite. Null policy ~0; the seed's strategy (push the drawer) ~0.

## Checks (`smoke.py`, rejection-only, 15 checks)

settle/no-NaN (drawer and block on the ground, bays empty, ~0); randomization A
(yaw span > 90°, xy jitter); randomization B (tagged bay varies, tag height
tracks it every reset, drawer/block ground poses vary); null policy ~0; SEED
strategy (real 8 N push — slides into the plinth, stays on the ground, ~0, moved
verified); wrong bay (loaded drawer flush in the UNTAGGED bay → block credit
only, <= 0.20); inverted seat (flush in the target bay upside down → upright
clause refuses, ~0); near-miss proud (25 mm short, loaded → 0.71, < 0.80 cap, no
success); flush but empty (seated True, containment refuses success); block
loose in the bay (cube-in-DRAWER refuses, ~0); sealed front (real 8 N held press
against the closed face — the block moved >= 15 cm, pressed, and stayed outside;
no block credit); latched credit survives theft (no success); rejection audit
(success never observed anywhere in the battery); final no-NaN; frames.npz video
recorded.
