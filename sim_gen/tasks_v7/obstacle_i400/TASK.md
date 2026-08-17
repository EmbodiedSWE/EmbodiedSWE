# obstacle_i400 — Breach Doorway (scene `breach_doorway`)

Get the blue 55 mm cargo cube into a court that is sealed on every side — side walls,
back wall, and a full ROOF — so "carry it over" does not exist. The only way in is a
single floor-level doorway through the front wall, and that doorway starts PLUGGED by
a stack of three loose terracotta bricks sitting in it like drawers in a slot. The
lintel leaves only ~9 mm above the stack, so no brick can be lifted out: each must be
SLID horizontally out of the doorway by its protruding yellow handle tab, top brick
first (pulling a lower brick first just drops the ones above it into the vacancy —
gravity re-plugs the doorway). Only then can the cargo be pushed along the floor
THROUGH the bore until it rests fully inside the covered court.

## Provenance

- **Seed:** `pick_place/obstacle`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/obstacle.py`) — pick a cube and
  carry it OVER a wall standing between it and the goal zone; the wall is a detour in
  an otherwise free transport, and the checker is the cube's final resting pose.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene `breach_doorway`,
  env `simgen.breach_doorway`, robot `"null"`), `solve.py` (teleport solution),
  `smoke.py` (rejection battery), all procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed's wall is a soft obstacle — lift the payload over it; one
  object, one grasp, one carried trajectory, and the wall itself is never touched.
  Here the vertical route is deleted outright (the goal area is roofed and sealed)
  and the obstacle itself becomes the WORK: the doorway is blocked by a multi-body
  plug that must be DEMOLISHED brick by brick before the payload can move at all. The
  plan changes shape — N ordered tool-like extractions (drawer slides under a lintel
  that forbids lifting, the seed's one primitive) followed by a floor-level threading
  push, never a carry — and the code changes shape with it (a per-brick extraction
  loop plus a guided push, not a single pick-lift-place trajectory). The seed's end
  state, the cargo resting in the goal region without having transited the bore, is
  explicitly constructed and REJECTED by the rubric's transit latch (smoke check 6).
- **vs `obstacle_i17` (skittle gallery, the sibling from this seed):** i17 is a
  ballistic strike — release a projectile, and the judged outcome is a toppling EVENT
  on untouchable pins behind a perception gate. Here nothing is ballistic and nothing
  is untouchable: every interaction is quasistatic manipulation of bodies the agent
  handles directly, the judged body is the handled cargo itself, there is no
  perception flip, and the core novelty is the ORDERED DEMOLITION of a blocking
  structure — obstacle-as-workpiece rather than obstacle-as-wrapper.
- **vs the pick-and-place family** (`libero_*`, `pick_single_egad_*`,
  `approach_grasp_spoon_i12`, `setup_checkers_i2`): those judge final poses reached
  by direct free-space carries. Here the goal pose is unreachable by any carry; the
  bulk of the work is on NON-goal bodies (the bricks), whose final poses are almost
  irrelevant (anywhere out of the way), and the goal object's route is constrained to
  a floor push through a bore.
- **vs the articulated family** (`close_microwave_*`, `open_oven_i6`,
  `close_grill_i8`): no joints anywhere — the "mechanism" is a stack of free rigid
  bodies whose slot geometry plus gravity enforce the extraction order.
- **vs `peg_insertion_side_i1/i2`:** those thread a held rod through an existing
  passage. Here the passage must first be CREATED by removing three bodies, and what
  goes through is the free-sliding payload, pushed, not a held tool.

## Execution-order declaration

Extraction order is REQUIRED in practice and enforced by gravity, not by the rubric:
pulling a lower brick first drops the bricks above it into the vacancy and the
doorway stays plugged (smoke check 8 constructs exactly this and shows the plug
survives with the push still blocked). The rubric itself is order-agnostic — it pays
for bricks OUT, however achieved — so an agent that wastes pulls on lower bricks is
punished with re-plugging, not with an artificial foul. Demolition must fully precede
the push (the push is physically blocked otherwise — smoke check 7).

## Randomization (per episode, verified by readback in smoke)

Fixture xy jitter ±100/±40 mm + yaw ±10° (the doorway moves — the solver must read
the fixture pose), per-brick lateral jitter inside the slot, cargo scattered
±180/±80 mm on the near floor.

## Rubric

`success()` iff, settled (cargo lin < 0.05 m/s, bricks < 0.10 m/s): the cargo centre
is past `wall_t + cargo/2 + 5 mm` in the fixture frame (trailing face beyond the
wall's inner surface), inside the court bounds, AND the bore-transit latch is set —
latched only while the cargo centre was physically inside the doorway bore at floor
height. The court is sealed everywhere else, so the bore is the only physical route;
the latch makes the RUBRIC itself reject a hypothetically constructed fly-over end
state (the seed's strategy).

`score()` (latched every physics substep in `post_step`): `0.10` per brick ever out
of the doorway plug (max 0.30) `+ 0.25` once the cargo has entered the bore, capped
at 0.55; exactly 1.0 iff `success()`. Doing nothing scores ~0.

Cfg `__post_init__` asserts the geometry that keeps the task honest: the stack fits
under the lintel with only millimetres to spare (no vertical lift-out) yet slides
freely; the cargo cannot squeeze over or beside the plug; bricks fit the bore depth
and slot width with real clearance; the cleared doorway passes the cargo with ≥30 mm
margin both axes; handle tabs are pinchable, protruding, and staggered per level so a
dropping brick cannot snag the tab below; and the success band lies strictly inside
the court.

## Teleport solution (`solve.py`) — transport only; all load-bearing work is contact

- **P1–P3 — demolition, top brick first (2, 1, 0):** each brick is slid out of the
  slot by a horizontal velocity-servo force along the fixture-local −y axis
  (`is_global` — the body-frame default drags with the body), cut 60 mm clear of the
  outer face; the brick tips off its support and lands under gravity. Only after it
  is free is it teleported to a parking spot (pure transport of an already-free
  body). Score latches monotonically to 0.30 (friction may drag the next brick
  partway out with the pulled one, latching its credit a phase early — legitimate
  contact physics, and the latch is non-decreasing either way).
- **P4 — threading:** the cargo is teleported from its scatter spawn to a staging
  point on the doorway axis, 160 mm in front of the wall, on open floor (the only
  cargo transport). From there a forward force — a friction feed-forward bias plus a
  velocity servo, capped at 0.90 N, under the mg ≈ 0.98 N CoM-push tipping bound so
  the cube slides rather than tumbles — plus a lateral PD holding the doorway
  centreline pushes it along the floor through the bore; the force is cut 12 mm past
  the success threshold and the cube coasts to rest inside the court. The transit latch is set by the push itself — the cargo is
  never teleported into, past, or over the wall. Score → 1.0.
- `SIM_GEN_SCORE` printed at every phase boundary is non-decreasing
  (0.000 → 0.200 → 0.200 → 0.300 → 0.300 → 1.000 → 1.000), ≥3.3 simulated seconds
  hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`. **Verified on the forge on
  three seeds with provably distinct layouts by stdout readback** (seed 0: fixture
  (−0.020, +0.138) yaw −9.6°, cargo (+0.077, −0.491); seed 1: cargo (−0.020,
  −0.390); seed 2: cargo (−0.177, −0.415) — all three SUCCESS, ~19 s each).

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (0.0, −0.85), facing the wall: the cargo scatter zone, the staging
line, and the doorway front are all within ~0.7 m reach, and nothing the task needs
is deeper — the push ends with the fingertip ~65 mm inside a 120 × 150 mm opening,
comfortably within a Franka hand's reach-through. Per-object contact strategy:
bricks — each 24 mm yellow handle tab is a clean parallel-jaw pinch (< 80 mm stroke;
the lowest tab sits ~36 mm above the floor, graspable with the hand horizontal), and
the extraction is a straight horizontal pull of ~120 mm at drawer speeds; the tabs
are staggered per level so the grasped tab is never shadowed by the one above.
Cargo — the 55 mm cube side-pinches (< 80 mm stroke) for staging, then is pushed
along the floor with the fingertips; the 120 mm doorway width forgives ±30 mm of
lateral error. Forces are tiny (≈1 N pulls, ≈0.7 N push). The task is genuinely
completable and its constraints genuinely bind.

## Checks (`smoke.py` — rejection battery, 14 named checks, ALL PASS on the forge)

1. settle: states finite; three bricks stacked in the plug, cargo on the near floor.
2. settle: score ~0 at reset, no success.
3. randomization readback: fixture xy + yaw vary.
4. randomization readback: cargo scatter + per-brick slot jitter vary.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: cargo teleported to rest INSIDE the court (through + in-court +
   settled all verified by readback) → NOT success (transit latch never set), score
   ~0 — the fly-over end state is rejected by the rubric itself.
7. blocked door: the solve's own force-push run against the intact plug — the cargo
   genuinely presses the stack (moved-assert, no vacuous probe) but never nears the
   bore; bricks stay put, no entry credit, no success.
8. out-of-order: bottom brick magicked out → the bricks above DROP into the vacancy
   (middle brick lands at floor height, still in the plug, by readback), the push is
   still blocked, and score is exactly the one-brick credit (0.10).
9. bore-rest: bricks cleared + cargo at rest inside the bore = credit exactly 0.55,
   NOT success (not through).
10. near-miss: cargo settled 8 mm short of the through threshold → NOT success.
11. wrong object: a BRICK delivered into the court instead of the cargo → NOT
    success, ≤ 0.15.
12. latched credit: full partial progress constructed, then cargo removed AND a brick
    re-plugged → the latched 0.55 survives unchanged, still no success.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN. Plus `frames.npz` (194 × 600 × 960 × 3) recorded and saved in CWD.
