# gondola_wheel — ride the black bowl over the crank wheel into the roofed rooftop gallery

`libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i362`
Env: `simgen.gondola_wheel` (robot="null", scene-level)

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet`
(RoboVerse `roboverse_pack/tasks/libero_90/...put_the_black_bowl_on_top_of_the_cabinet.py`).
The seed is a single pick-and-place: grasp the akita black bowl (a plate is a
distractor) and set it down on the cabinet's open top surface — success is a bbox
check above the cabinet.

Kept from the seed: the cast (BLACK BOWL = goal object, WHITE PLATE = distractor,
cabinet = destination) and the goal *predicate shape* ("bowl resting on top of the
cabinet"). Everything about *how the bowl can get there* is replaced.

## What changed and why it is strategically different

1. **The seed's strategy is dead by construction.** The cabinet top is a ROOFED
   ROOFTOP GALLERY: a parapet-walled deck sealed by a solid roof slab. Nothing can be
   lowered onto the deck from above (smoke check: a bowl dropped from above rests on
   the roof, scores ~0). The only opening is a narrow 15 cm window in the front
   parapet, 10 cm tall — the bowl can only *slide in horizontally*.
2. **A drive mechanism is the only sanctioned route.** A hand-cranked GONDOLA WHEEL
   (crimson disc r=0.22 on a pylon, four rim pegs as crank handles) carries a
   free-swinging pendulum GONDOLA TRAY on a rim pin. The wheel travels 30°→186°
   between hard joint stops; the upper stop is 6° PAST top-dead-center, so a loaded
   tray *parks itself* there by its own hanging weight (over-center gravity latch —
   no holding torque). At the park, the tray's open front edge sits 15 mm from the
   window sill, 9 mm above the deck.
3. **Declared execution order, latch-enforced.** The bowl must REACH THE DECK BY
   RIDING: aboard the tray at the bottom stop (`loaded`), aboard through mid-arc
   95–125° (`rode`), aboard at the ≥183° park (`hoisted`), then on the deck
   (`delivered`) — each latch requires the previous one. A bowl hand-carried through
   the window produces an end state *identical* to success (wheel parked, bowl on
   deck) but scores ~0 with success() false.

Different from every examined neighbor:
- **seed**: one pick-and-place onto an open top — here the top is sealed and the
  transport is a constrained mechanism ride with an order rule.
- **i297 "bowl_airlock"**: sequential sliding-door interlock (translation gates) —
  here the mechanism is a *rotating* wheel + passive pendulum with an over-center
  gravity park; no doors, no interlock.
- **i61 "cart_ferry"**: horizontal prismatic ferry over a void slot — here the ride
  is a 156° *vertical arc* on a revolute crank with a self-leveling pendulum carrier,
  and the destination is roof-sealed rather than slot-guarded.
- **i5 "counterweight_shelf"**: counterweight statics (balance decides the shelf
  angle) — here nothing is counterweighted; the physics story is pendulum
  self-leveling + over-center parking + a window-gauge horizontal serve.

## Scene

- ONE kinematic compound fixture (never teleported): wheel pylon + axle stub,
  staging table (top z=0.15), cabinet plinth (deck z=0.52), parapet walls with the
  front window (x∈[−0.055,0.095], z∈[0.52,0.62]), roof slab (z=0.65–0.67).
- Wheel disc (r=0.22, m=0.8) on a per-env authored USD revolute (axis Y) at the hub
  (0,0,0.435), authored at mid-travel 108° so the joint limits are ±78° — far from
  the PhysX ±180° wrap. Rim pin at radius 0.20; four crank pegs at radius 0.14.
- Gondola (m=0.25) on a second, free revolute at the rim pin; CoM authored below the
  pin → stable pendulum, self-levels. Tray basin has curbs on ±x and the back; the
  FRONT (toward the gallery) is open. All jointed-pair clearances are REAL authored
  gaps (≥10 mm; the joint collision filter is not load-bearing).
- Black bowl: procedural open octagonal cup (outer r 0.051, h 0.055, m 0.35). White
  plate: cylinder r 0.08 (distractor).
- Randomization (readback-verified): wheel start angle 32–46° (falls to the 30°
  stop), bowl/plate staging-slot assignment sampled, ±3 cm xy jitter, free yaw.

## Rubric

Latched, order-chained (post_step): 0.15 `loaded` + 0.15 `rode` + 0.25 `hoisted` +
0.40 `delivered`, capped at 0.95; exactly 1.0 iff `success()` = bowl upright at rest
inside the deck band ∧ everything settled ∧ `hoisted_ever`.

## Teleport solution (solve.py — teleports are TRANSPORT ONLY)

1. **Reset/settle** — wheel falls from its sampled start angle onto the 30° stop.
2. **LOAD** — teleport the bowl to 2.4 cm ABOVE the tray basin (computed from the
   live gondola pose, deliberately outside the aboard band so the teleport latches
   nothing); gravity drops it in; it rests by contact. `SIM_GEN_SCORE 0.150`.
3. **CRANK** — torque servo on the disc (gravity feedforward M·g·R·sinθ for the
   hanging load + clamped P velocity term, |τ|≤2.6 N·m ≈ 8 N at the pegs), cut at
   182°; gravity alone carries the wheel onto the 186° over-center stop and holds it.
   The pendulum tray self-levels; the bowl rides by contact/friction only.
   `SIM_GEN_SCORE 0.550`.
4. **SERVE** — small regulated horizontal force (≤5 N) pushes the bowl off the tray's
   open front edge, through the window; it drops 9 mm onto the deck. Forces off.
   `SIM_GEN_SCORE 1.000`.
5. **Persistence** — ≥3.5 simulated seconds hands-off; success must hold →
   `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge, seeds 0 and 1 (monotone 0.000/0.150/0.550/1.000/1.000, rc=0).

## Embodiment argument (single Franka arm, one base pose)

Base at ≈ (−0.32, −0.02, 0), between the staging table and the wheel, facing +x/+y:

- **Load**: the bowl (10.2 cm across, 11 mm rim) is rim-pinched from the staging
  table (top 0.15 — comfortable tabletop height) and set into the tray basin at the
  bottom stop (tray floor ≈0.10, well inside the workspace envelope).
- **Crank**: the four 2.2 cm rim pegs on the wheel's NEAR (−y) face are grasp
  handles at radius 0.14 around a hub at height 0.435 — every peg position is within
  reach. Required tangential force ≈ τ_max/0.14 ≈ 8 N, far under Franka payload;
  the arc is cranked in quarter-turn strokes with regrasps (the over-center stop
  means the wheel needs no holding between strokes above 180°, and below TDC the
  wheel back-drives only to the 30° stop — recoverable).
- **Serve**: the parked tray floor is at 0.529, the window at y≈0.18, z 0.52–0.62 —
  the arm pushes the bowl's side wall horizontally (fingertip push, ≈2 N) along +y
  through the window. No regrasp inside the gallery is needed (nothing must enter
  it but the bowl).

Each interaction is a single-object, single-contact-strategy primitive (pinch-place,
peg cranking, side push) from one base pose.

## Execution-order declaration

Declared in `describe()`/`instruction()` and enforced by the order-chained latches:
LOAD (aboard at ≤60°) → RIDE (aboard through 95–125°) → PARK (aboard at ≥183°) →
DELIVER (on deck). Success additionally requires `hoisted_ever`. An end state with
the bowl on the deck but no ride provenance scores ~0 and is not success.

## Smoke battery (smoke.py, recorded to frames.npz)

15 checks: settle/no-NaN + zero-score reset (2); randomization READBACK across 6
seeds (1); null policy 240 steps (1); seed-strategy roof drop fails (1);
hand-delivery through the window fails despite geometric on-deck (1); flagship
end-state-identical illegal run — EMPTY ride to the park + hand-delivered bowl —
fails (1); settled near-miss in the window mouth outside the deck y-band (1);
inverted bowl fails the upright tolerance (1); wrong object (plate delivered) fails
(1); oracle full ride on 3 seeds with 240-step persistence (3); score monotonicity
ladder 0 < 0.15 < 0.30 < 0.55 (parked near-miss, no success) < 1.0 (1); partials
< 1.0 (1).
