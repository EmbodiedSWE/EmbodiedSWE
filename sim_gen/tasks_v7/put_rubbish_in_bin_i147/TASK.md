# put_rubbish_in_bin_i147 — `rubbish_chute`

Get the white paper ball (the rubbish) into a SEALED green bin that the arm cannot
reach and cannot open: load the ball into an elevated gravity chute's open trough,
then lift the chute's captive guillotine gate by its pinch tab and HOLD it while the
ball rolls under, down the covered tunnel, through the bin's only window, into the
bin — then let go: the gate falls shut on its own. The red tomato (same size) must
stay out of the bin.

- **Env name:** `simgen.rubbish_chute` (robot `"null"`, procedural assets only)
- **Package:** `scene.py` (scene + rubric), `solve.py` (legitimacy certificate),
  `smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `rlbench/put_rubbish_in_bin`
(`RoboVerse/roboverse_pack/tasks/rlbench/put_rubbish_in_bin.py`): a Franka grasps
the rubbish off a table among two tomato distractors, carries it above an OPEN bin,
releases. One grasp, one carry, one release — the bin is passive and the drop is the
delivery.

## What changed and why it is strategically different

| Axis | Seed | This task |
|---|---|---|
| Bin | open-topped, passive, in reach | **SEALED and out of reach**: roofed, walls closed, its only window completely filled by the chute's discharge tunnel (every residual gap < ball diameter, asserted in cfg) — the direct carry-and-drop plan is physically dead |
| Delivery | the arm's own release above the bin | **operated infrastructure**: a pitched gravity chute delivers the ball; the arm never approaches the bin at all |
| Mechanism | none | **captive gravity-return guillotine gate** in the chute: jointless (slot geometry IS the mechanism), retained by caps (cannot be removed), recloses under gravity the moment it is released (cannot be propped) |
| Plan shape | one open-loop pick-and-drop | **forced two-stage order**: the gate auto-closes and one arm cannot hold it and fetch the ball at once, so the ball must be loaded FIRST, then the gate lifted and **held** (a sustained, transient hold — not a state change) while gravity does the transport |
| Distractor role | visual clutter beside the target | **contamination constraint**: success requires the tomato NOT inside the bin, and a reseated gate at the end |

Distinct from the corpus neighbours:
`living_room_scene1_..._i8` (slidelid_hamper — slide a captive lid open, drop in,
slide it shut: both lid states are stable, the lid is repositioned, the container is
in reach); `basketball_in_hoop_i128` (crater_run — sustained anti-gravity push
conveyance of the ball itself); `hockey_i325` (barrier extraction then floor push).
Here the manipulated object (gate) is NOT the delivered object (ball), the held
state is *unstable by design* (gravity-return, so "open the lid then do the rest"
cannot work), and the delivered object is never touched between the trough and the
goal — delivery is entirely machine-mediated.

## Rubric

- `success()`: paper ball settled inside the bin interior (fixture frame:
  `|x−bin_cx| ≤ 0.11`, `|y| ≤ 0.11`, `z ∈ (0.013, 0.09)`), tomato NOT inside, gate
  reseated (`lift < 0.030` and laterally seated), ball + gate settled
  (`|v| < 0.05`). `__post_init__` asserts the geometry backs every claim: the caps
  still pass the ball, a shut gate blocks it, all three window gaps around the
  tunnel are smaller than the ball (the bin really is sealed), every physically
  possible in-bin rest is accepted, and the discharge lip inside the window is
  rejected by the z band.
- `score()`: latched — 0.25·loaded (ball ever in the trough) + 0.25·gate progress
  (running max of lift/0.060) + 0.30·passed (ball ever past the gate) — capped at
  0.80; exactly 1.0 iff `success()`. Null policy scores ~0; credit never evaporates
  (verified in smoke).

## Teleport-solution outline (`solve.py`)

1. **P0** settle + readback (layout, gate-mass 0.30 kg from `get_masses()`).
2. **P1 transport** — the ONLY pose write on the ball: ground → 118 mm above the
   trough floor, outside the judged trough band (asserted against the scene's own
   predicate). Both endpoints free space.
3. **P2 load (contact)** — gravity drops it in; it rolls down the 16° floor and
   beds against the closed gate (`loaded` earned by physics).
4. **P3 release (contact)** — vertical force at the gate CoM: gravity feedforward
   (m·g from readback) + PD on measured slide-lift (kp·dt/m = 0.22 ≪ 1). A vertical
   command is invariant under every pod force-frame encoding (world: exact gravity
   cancellation; body/drag: z maps through qz(yaw)·qy(θ) onto the slide axis), and a
   progress probe guards it anyway. Held ~66 mm while the ball rolls under and down
   the tunnel; released only once the ball is irrecoverably downhill.
5. **P4 restore (contact)** — force cleared; the gate free-falls back onto its seat;
   ball flies the window and settles mid-bin. success() first True here.
6. **P5 persistence** — ≥3.3 s hands-off, success must hold.

`SIM_GEN_SCORE` 0.00 → 0.00 → 0.25 → 0.50 → 0.80 → 1.00 → 1.00,
`SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1, 2 (yaw −2.4° / +9.4° / distinct spawns —
randomization provable from stdout). Iteration history: attempt 1 wedged the gate on
release (friction jam in the 2.5 mm grooves at default μ≈0.5); fixed by authoring a
slick material (μ=0.06, combine-mode MIN) on the gate — the self-closing claim is
now physically robust, not narrated.

## Embodiment argument (single Franka + parallel jaw, OSC)

Base at world ≈ (−0.15, 0), facing +x — the trough head and both ball bands in
front of it.

- **Balls (Ø56 mm):** graspable (< 80 mm jaw). Pick from the ground bands at
  0.15–0.31 m reach, ground height — comfort zone. Drop into the open trough from
  above: trough centreline at local x 0.05–0.20 → world 0.17–0.32, release height
  ~0.37 m, walls only 75 mm high and the channel 110 mm wide — an easy top-down
  release at ≤ 0.45 m reach.
- **Gate tab (the one mechanism):** black tab, 32×50 mm, top edge at 0.36–0.43 m
  height at world x ≈ 0.38–0.40 — pinch it (10 mm plate < 80 mm jaw) and slide
  66 mm along the near-vertical slot at ~0.45 m reach: well inside the work envelope,
  and the 3 N hold (0.30 kg gate) is trivial for the arm. The hold-then-nothing
  sequencing (ball already loaded) means one arm suffices.
- **Bin:** never manipulated — its nearest face is at world x ≈ 0.85 and it is
  sealed; the task never requires reaching it (that is the point).
- **Structures:** kinematic, never manipulated.

## Execution order

1. `scene.py` written first (geometry derived from cfg constants; honesty asserted
   in `__post_init__`).
2. `solve.py` iterated on the forge until `SIM_GEN_SOLVE: SUCCESS` on seeds 0/1/2.
3. `smoke.py` rejection battery on the forge.
4. `TASK.md` — all green runs above are on the final, unmodified package.

## Check list (smoke, 18/18)

1. settle/no-NaN + layout sanity (gate seated shut, balls on the ground, at rest)
2. score ~0 at reset, no success
3. randomization readback: structure yaw + xy vary, gate follows seated (8 seeds)
4. randomization readback: ball spawns vary, side bands really swap, ≥ 9 cm apart
5. null policy: 240 idle steps → score ~0
6. seed strategy dead: dropped above the SEALED bin the ball settles on the roof,
   never inside — the chute is the only way in
7. wrong object (load): tomato in the trough (readback) earns NO load credit
8. wrong object (bin): tomato constructed inside → no success, score ~0
9. contamination: BOTH balls inside → NOT success, score ≤ 0.80
10. order forced: trough drop rests AGAINST the closed gate, never past (0.25)
11. latched credit survives removing the loaded ball to the ground
12. gate captive (non-vacuous): 1.3×-weight overpull peaks at the caps (0.0745 m),
    never escapes
13. gravity return: force cleared → gate reseats ON ITS OWN; gate credit caps 0.25
14. incomplete: ball in bin + tomato out + gate HELD open → NOT success (reseat is
    load-bearing)
15. monotonicity: partial lift latches strictly less gate credit (0.39 < 1.00)
16. rejection audit: success() never True at any judged point
17. final no-NaN
18. camera ≥ 20 rgb frames → `frames.npz` (245 saved)
