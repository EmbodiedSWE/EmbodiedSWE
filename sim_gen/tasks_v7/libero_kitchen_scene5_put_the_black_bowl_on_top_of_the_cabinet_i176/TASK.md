# ballast_rocker — seat the counterweight, then place the bowl on the rocking tray

**Env key:** `simgen.ballast_rocker` · **Robot slot:** `null` (scene-level task)
**Package:** `sim_gen/tasks_v7/libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i176`

## Seed provenance

Derived from RoboVerse `libero_90/kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet`
(`roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet.py`):
a Franka picks the akita **black bowl** off the table and sets it down **on top of the
cabinet**; success is the bowl inside a bbox above the fixed cabinet top.

Kept from the seed: the cabinet as the goal fixture, the black bowl as the goal object, and
the goal phrase itself — put the black bowl on top of the cabinet.

## What changed and WHY it is strategically different

The cabinet's top is no longer a surface you can just use. It is a **see-saw tray** on a
central revolute hinge between two pylons, deliberately unbalanced (authored CoM offset on
the socket side) so it rests tilted +12° against its upper joint stop: the red-fenced
ballast **socket** end low, the green **bowl zone** end raised. The seed's entire plan —
carry the bowl up and set it down — is preserved as a live **trap**: the bowl's weight on
the raised zone out-torques the tray's bias 2:1, the tray pivots to its −35° stop (steeper
than the friction angle), **dumps the bowl on the floor**, and rocks back empty. The only
winning plan is a **counterweight interlock**: first seat the 0.9 kg steel-blue ballast
cube in the fenced socket on the low end (with the bias, ~5.5× the bowl's torque — pins the
tray at its rest stop through the landing transient), and only then place the black bowl
upright on the zone.

Strategic difference, not variation:

- **vs the seed** — the seed is a single pick-and-place onto a static surface. Here the
  same surface is a passive mechanism that *rejects* that plan; solving requires torque
  reasoning, a second object as a physical *precondition*, and a gravity-forced order.
- **vs sibling i306 (hatch_shelf, same seed)** — i306 *directly actuates* a bistable lid
  through its over-centre arc to *create* the support surface (plus decoy-object
  discrimination). Here the mechanism is **never actuated directly**: nothing pushes the
  tray; it is *stabilized by loading it* with a counterweight. Different physics
  (torque-balance interlock vs over-centre actuation), different failure trap (dynamic
  dump-and-reset vs falling into the cavity), different precondition structure (second
  object's placement vs mechanism state).
- **vs the exemplars read** (pen_holder: fill a cup) — no aggregation/counting; a
  two-object ordered interlock on a live mechanism.

## Teleport-solution outline (solve.py, honest)

Teleports are **transport only** (single free-space pose writes into asserted-dead hovers
above every scoring band); every load-bearing interaction is contact dynamics. The tray is
never teleported and never pushed.

1. **P0** reset + settle; assert tray at rest stop, both objects on the floor, score ≤ 0.05.
2. **P1** ballast teleported to a hover 3 cm above the socket's z band (asserted: no gate,
   no seat) → **gravity drop into the fenced socket**; landing impulse, fence funnelling
   and the hinge's response are pure physics; assert `ballast_seated` (tray resting still
   on its stop under the counterweight).
3. **P2** bowl teleported to a hover 2 cm above the zone's z band, pre-tilted to the tray
   angle (asserted: no gate, no success).
4. **P3** **gravity drop onto the zone**: the ballasted tray absorbs the landing nod and
   demonstrably *carries* the bowl on the hinge — assert `bowl_in_zone` and `success()`.
5. **P4** hands-off persistence ≥ 3.3 simulated s, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, asserted non-decreasing (latched credit).
Verified on the forge: **seed 0 and seed 1 both SUCCESS** (rc=0, ~25 s each).

## EMBODIMENT ARGUMENT (single Franka, one plausible base pose)

Base at the world origin facing +x (cabinet centre 0.50 m away, hinge line at z 0.375):

- **Ballast cube** (5 cm, 0.9 kg): floor slots at (0.32, ±0.22), z 0.025 — top-down
  parallel-jaw pinch across the 5 cm cube fits the Franka's 8 cm jaw span; carry to the
  socket at ~(0.50, −0.15, 0.37) and release 2–3 cm above the fence mouth (the 8.5 cm
  fenced interior vs 5 cm cube leaves ±1.75 cm; the fences funnel the drop). 0.9 kg is
  well inside payload.
- **Black bowl** (~12 cm wide, 4.5 cm tall, 180 g): rim pinch — one finger inside, one
  outside a 1 cm wall (standard LIBERO bowl grasp); place onto the zone at
  ~(0.50, 0.14, 0.41) with a low-drop release; the cleat guards downhill creep.
- All grasp/release points lie in x 0.30–0.55, |y| ≤ 0.25, z 0.02–0.48 — inside the
  Franka's comfortable workspace from that base pose, no reorientation needed beyond yaw.

## Execution-order declaration

Physics forces **ballast → bowl**: the reversed order (bowl first) tips the tray, dumps the
bowl, and rocks back — constructed and asserted rejected in smoke check 6; ballast-after-
dump (out-of-order end state) is partial credit only (check 8).

## Checks (smoke.py — rejection battery, 16 checks)

1. settle: finite states, empty tray on its +12° stop, objects standing on the floor, still
2. settle: score ~0, no success
3. randomization READBACK: slot swap mean strictly inside (0,1), xy jitter spread real
4. randomization READBACK: free yaw varies across seeds
5. null policy: 240 idle steps → score ~0, no success
6. seed strategy: bowl set on the unballasted tray → tray tips past −20°, bowl dumped off,
   no success, score ≤ 0.25
7. trap aftermath: emptied tray rocks back and re-settles on its rest stop
8. out-of-order: ballast seated after the dump, bowl on the floor → no success, ≤ 0.50
9. swapped roles: ballast dropped on the zone tips the tray and is dumped itself → no success
10. near-miss socket: ballast ON the tray but outside the socket xy tolerance while the bowl
    settles in the zone (off-spec stack that happens to hold) → no success, ≤ 0.30
11. near-miss zone: seated ballast, upright bowl on the tray but inboard of the cleat,
    outside the zone band → no success
12. wrong orientation: upside-down bowl on the zone → upright clause rejects
13. latched credit: bowl removed to the floor → latched credit unchanged, no success
14. hover fly-through: bowl held above the zone, judged without stepping → no success
15. rejection audit: success() never True at any judged point in the battery
16. final no-NaN

Video frames recorded via the viewport rgb annotator and saved to `frames.npz`.
