# service_carousel — ferry the plate out, load the MIDDLE bowl, ferry it back to the serve mark (i78)

**Env name:** `simgen.service_carousel` (scene `service_carousel`, registered with robot
`null`; solve.py and smoke.py build the same scene-level env).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate.py`)
- Seed plan: among three identical black bowls on a table, select the MIDDLE one by table
  position, grasp it, carry it across free space, set it down on a flat, openly reachable
  plate. Success = one geometric relation (bowl within 6 cm of the plate axis), reached by
  a single unconstrained grasp-carry-place.

## What changed, and why it is strategically different

The "middle black bowl of three, onto the plate" premise is kept (three identical black
bowls in a row, target = the middle slot, identity permuted per episode); everything that
made the seed's one-step plan sufficient is removed:

1. **The placement surface must be TRANSPORTED — twice — through a 1-DoF MECHANISM.** The
   plate sits flush in a pocket on a rotating carousel disc (a dynamic body on a
   spawn-authored revolute joint, angular damping, no motor). It starts parked under a low
   service hutch where no hand fits, 25–45° off the serve mark. The only way to move the
   plate is to rotate the whole carousel by its peg handles: ~180° out to the open loading
   side, and — after loading — back.
2. **The goal region is access-BLOCKED.** The hutch roof leaves an 83 mm slot over the
   disc: the pegs (70 mm) and the loaded plate+bowl (62 mm) sweep beneath it by
   construction, but a hand holding a 50 mm bowl cannot deliver at the serve point. Direct
   placement at the goal is geometrically impossible; the payload must RIDE the mechanism
   in.
3. **The goal pose differs from the start pose.** The seed's end state — "bowl on the
   plate where the plate is" — is a REJECTED outcome here: the plate starts outside the
   15° delivery window by construction (25–45° offset), and the smoke constructs exactly
   that state and shows success False, score 0.
4. **The load must survive dynamics.** The return ride carries the bowl on the plate by
   friction alone; a bowl that tips or slides off during the ~180° swing fails (upright +
   centring + pocket gates on the settled end state).
5. **A solver needs different code structure**: a closed-loop rotation controller
   (push/pinch pegs, regulate disc angle to two set-points ±15°/±40°) bracketing one
   pick-and-place, instead of the seed's single place-on-target primitive.

Corpus differentiation (tasks read this batch): `spindle_serve` (i2) claims stack-order
identification + ordered de-stacking + dowel threading — static fixtures, vertical
insertions, no mechanism. `hanoi_plates` (i56) claims a rule-governed multi-move relay on
static pegs — order enforced by the rubric, not by geometry. `trestle_service` (i7) claims
push-to-marks + bridge building on static furniture. `track_bowl` (i27) claims capturing
and herding a free-rolling ball with a movable pocket tool. robobench's own `pen_holder`
is container filling. This task claims **mechanism-mediated ferrying through an
access-blocked goal region (rotate → load → rotate-back)**: the load-bearing motion is the
rotation of a scene mechanism that carries the payload BOTH ways, something none of the
above contains; nothing here is de-stacked, threaded, rule-sequenced, pushed to a mark,
bridged, or chased.

## Embodiment argument (single Franka arm + parallel jaw, OSC)

Intended base pose: **(-0.45, 0.0, 0.20)** — mounted on the counter slab (the sibling
tasks' verified on-counter mount).

Per manipulated object (the hub, hutch and counter are kinematic; the plate is never
touched directly):

- **Peg handles (the carousel drive):** four Ø18 mm × 70 mm steel pegs at r = 170 mm; with
  90° spacing there is ALWAYS one peg in the near sector (within ±45° of the robot side),
  i.e. at 0.43–0.53 m from the base — inside the proven 0.35–0.65 m comfort band. Ø18 mm
  pinches inside the 80 mm jaw span, or the closed fingertips simply push a peg
  tangentially. Required force is trivial: the reference torque (0.06 N·m) at the peg
  radius is ~0.35 N. Peg tops sit 122 mm above the counter — open top-down access on the
  loading side (the hutch covers only the far serve sector).
- **Middle bowl (the payload):** bowls stand at 0.23–0.28 m from the base — close-range
  but inside Franka's dexterous envelope for a top-down rim pinch (elbow-up). The bowl is
  Ø~110 mm (too wide to span) with a 7 mm rim wall 42 mm tall: the standard mug-style rim
  pinch (7 mm << 80 mm span, ≥20 mm of finger engagement). Load point: the fetched plate
  parks at ~0.49 m from the base, in the open, top-down free; set-down tolerance is 35 mm
  radial / release from 25–30 mm — far above closed-loop OSC placement noise (1–3 mm in
  prior tasks).
- **Plate:** intentionally NOT graspable (flush in its pocket, 3 mm radial slack, wall top
  level with the plate top — no rim to pinch) and never needs to be: the carousel moves it.

No contact is required near the ground, under the hutch, or through any aperture; all
interactions happen 200–330 mm up on the open loading side of the counter.

## Execution order

**Fully ordered, enforced by PHYSICS, not by the rubric:** the bowl cannot be placed on
the plate while the plate is under the hutch (no hand + bowl fits the 83 mm slot), so
FETCH must precede LOAD; the bowl cannot be added at the serve mark for the same reason,
so LOAD must precede RETURN. success() is a pure settled-state predicate with no order
clause; the latches (fetch, load-in-window) are additive and monotone along the only
feasible order. The smoke's seed-strategy probe is the order-violation case (load without
fetch → the LOADED latch cannot fire outside the loading window).

## Rubric (graded, latched in post_step, additive)

| score | state |
|-------|-------|
| 0.00  | nothing done (null policy; also the seed-strategy end state) |
| +0.25 | FETCHED: the plate has been carried into the loading window (azimuth within 40° of the robot side), disc slow |
| +0.35 | LOADED: the middle bowl seated on the plate WHILE in the loading window, bowl slow |
| 1.00  | success(): middle bowl upright & centred on the plate + plate in its pocket within 15° of the serve stripe + everything settled |

Honesty: `bowl_xy_tol` 35 mm vs a max fully-on offset of ~27 mm (plate r 75 − bowl base
r 48) — a settled 55 mm off-centre bowl (resting, bridging the flush pocket wall) is the
constructible rejected near-miss. Delivery 15° vs a 25–45° start offset — the reset state
is rejected by construction; 20° parked is the near-miss, a 10° EMPTY plate is the
accepted-direction twin (never full success). Pocket gates (±30 mm radial, ±8 mm height)
reject a plate lying anywhere but seated in its pocket. Upright gates reject an inverted
or tipped bowl. Both latches are velocity-gated (disc < 0.5 rad/s, bowl < 0.10 m/s): a
plate swept through the window on a fast-spinning disc latches NOTHING (smoke-verified).
Commissioning note: dynamic bodies run 4 solver velocity iterations — at 1, GPU TGS
leaves a constant ~0.04 rad/s phantom disc creep under zero torque that would have let
`settled()` ride an artifact.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move the free bowl through free air ONLY; every load-bearing interaction runs
through contact dynamics:

- Phase 0: `env.reset(seed=N)`, settle, layout readback (start azimuth, middle-bowl
  identity, plate position) → `SIM_GEN_SCORE 0.000`.
- Phase 1 FETCH: a clamped velocity-servo z-torque on the disc via
  `set_external_force_and_torque` (the finger-on-peg surrogate; τ ≤ 0.06 N·m,
  ω ≤ 0.55 rad/s — tangential accel at the pocket ~0.26 m/s², far under the friction
  budget), active braking tail, torque cut. The plate rides pocket contact + friction the
  whole way; its state is never written → `SIM_GEN_SCORE 0.250`.
- Phase 2 LOAD: middle bowl — incremental raise off the counter, one hop at 0.42 m (above
  the hutch roof top 0.347 and peg tops 0.322), incremental descent in the open loading
  column, RELEASE ~28 mm above the plate; gravity + contact seat it (retry with small
  lateral offsets; never spawned seated) → `SIM_GEN_SCORE 0.600`.
- Phase 3 RETURN: same torque servo back to azimuth 0 with the bowl riding on friction;
  brake, cut, settle → success → `SIM_GEN_SCORE 1.000`.
- Phase 4: ≥3.3 simulated seconds hands-off; success() must hold → `SIM_GEN_SOLVE:
  SUCCESS`. Any score decrease or exhausted retries prints FAIL. Hard exit behind
  watchdog Timers.
- **Verified on the forge: seeds 0 and 1 both print `SIM_GEN_SOLVE: SUCCESS` (no retries;
  score ladder 0 → 0.250 → 0.600 → 1.000 on both; parked within ±0.6° and persistence
  held with zero drift after the vel-iters fix). Smoke: `SIM_GEN_SMOKE: ALL PASS 20/20`
  (run 1), 82 frames recorded.**

## Check list (smoke.py — rejection battery; solve.py is the acceptance proof)

1. settle/no-NaN: authored layout settles finite; plate seated in pocket, OFF the mark,
   outside the loading window
2. rubric clean at reset (score 0, no success)
3. determinism: same seed → identical layout readback
4. randomization: middle-slot bowl identity varies across seeds (readback)
5. randomization: plate start azimuth varies across seeds (readback)
6. randomization: bowl positions vary across seeds (permutation + jitter readback)
7. null policy: score ~0, no success after 240 idle steps
8. SEED STRATEGY (middle bowl seated on the un-fetched plate at its start azimuth):
   on_plate True, success False, score 0
9. azimuth near miss: loaded plate parked at 20° (tol 15°) → not delivered, no success
10. tolerance twin: EMPTY plate parked at 10° IS delivered — no bowl, no success
11. wrong bowl: an OUTER bowl delivered at the serve mark → success False
12. inverted bowl on the plate → upright gate rejects, LOADED never latches
13. tolerance twin: 25 mm off-centre set-down counts as on_plate (gate honest)
14. near miss: settled 55 mm off-centre set-down (resting, bridging the pocket wall) NOT
    on_plate
15. latched credit: fetch+load latched 0.60 while the bowl sat on the plate in the window
16. latched credit survives removing the bowl back to the counter (still 0.60)
17. velocity gate: plate swept through the window on a fast disc (2.0 rad/s) latches
    NOTHING
18. velocity gate twin: the same pose with the disc parked DOES latch (0.25)
19. out of pocket: plate lying on the counter at azimuth 0 → not delivered
20. finite at the end

Teleports in the smoke are instrumentation (constructed settled states, real physics
steps before judging); no probe constructs full success.

## Scene / assets

Fully procedural, one rigid body per object, explicit MassAPI on dynamics: carousel disc
(Ø400 × 20 mm, 1.2 kg, four Ø18 mm peg handles, an 8-box pocket ring inner r 78 mm at
r 115) on a per-env revolute joint (axis Z, body0 = the kinematic hub, which is never
re-posed — the world-fixed anchor quirk is moot; the hutch is a separate body so
disc/payload↔hutch collisions stay live); kinematic hutch (roof slab at 135 mm clearance,
back wall and posts outside the disc rim, green serve stripe); white plate (Ø150 × 12 mm,
250 g, friction material 0.7/0.6), flush in the pocket; three identical black octagonal
bowls (Ø~110 × 50 mm, 150 g, 7 mm rim wall); counter = kinematic slab (top at 0.20 m).
Randomized per episode: which bowl body occupies the middle slot (permutation), plate
start azimuth (signed 25–45°), per-bowl xy jitter and free yaw.
