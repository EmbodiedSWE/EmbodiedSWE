# domino_relay — cascade-delivered ball into a sealed gallery pit

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet`
— pick up a black bowl and place it inside the top drawer of a cabinet (direct grasp,
transport, containment).

## What changed, and why it is strategically different

The seed's strategy is **grasp-and-place into an openable container**: the entire task is
solved by one pick, one carry, one release, with the drawer as a passive receptacle.

This task keeps the seed's *skeleton* — "get the round-ish object INTO the enclosed
receptacle" — but makes direct manipulation of the payload **physically impossible** and
replaces it with a **multi-body kinetic relay that the robot must construct first**:

- The orange ball sits on a shelf inside a **sealed gallery** (roofed, walled); the only
  opening is a 55 mm letterbox window shadowed by an overhanging eave. No gripper or tool
  fits through; the ball can never be grasped, poked, or dropped in from above.
- The only permitted impulse on the ball is a **domino cascade**: the robot must stand
  8 scattered tiles into a spaced relay line from a fixed crimson trigger tile to the
  window, then tip *only* the trigger. The falling relay runs tile-to-tile, the last tile
  pitches through the window and bats the ball off its shelf into the sunken pit.
- The rubric enforces the *mechanism*, not just the end state, with latched causal
  timestamps: the ball must leave its seat within **tau = 2.75 s AFTER** the trigger tile
  falls (`causal`), at least **4 tiles** must lie fallen inside the pad-to-window corridor
  (relay actually existed), and the ball must rest on the pit floor, all judged at
  settlement. Dislodging the ball first, toppling tiles by hand one at a time, or a
  hand-delivered ball (the seed strategy's analogue — verified in smoke check 6) all
  fail permanently.

Versus the rest of the corpus: this is not a stacking/static-structure task (the structure
is built *to be destroyed*), not a ballistic/throwing task (energy is stored as tile
potential energy and released by a fingertip nudge), and not a tool-use task (no tool
touches the payload — an unbroken chain of free rigid bodies does). The scored skill is
**precision placement of many identical objects whose value is only realized through a
timed, self-propagating collision chain**, plus one minimal triggering contact.

## Teleport-solution outline (solve.py)

Teleports are transport-only, always ending in free space above the floor; every
load-bearing interaction is contact dynamics.

- **P0** — settle 60 steps; read back the randomized layout (gallery pose/yaw, pad
  position, span 0.30–0.40 m, bearing up to ±20°). `SIM_GEN_SCORE 0.0`
- **P1** — teleport tiles one at a time to standing poses (2 mm drop into free space) on
  the pad→window line: n_chain = tiles sized for ~48 mm gaps (6 tiles at the observed
  spans), faces square to the line; leftover tiles parked flat far from the corridor.
  Settle. Latches `built` = 4 (0.30). `SIM_GEN_SCORE 0.30`
- **P2** — apply a 0.16 N horizontal force at the trigger's COM along the chain direction
  (fingertip-scale; tip threshold is ~0.065 N) until it leans 15°, then clear the wrench —
  hands off. The cascade runs unaided: trigger → 6 tiles → last tile pitches through the
  window → ball batted off the shelf (dt_trig→ball ≈ 0.23–0.28 s ≪ tau) → rolls off the
  open back edge of the shelf into the pit. Poll to settlement. `SIM_GEN_SCORE 1.0`
- **P3** — 3.3 sim-seconds hands-off persistence with per-interval success re-checks,
  then `SIM_GEN_SOLVE: SUCCESS`.

Passed on the forge with seeds 0 and 1 (spans 0.302/0.325, bearings +11.5°/+10.3°).

## Embodiment argument (single Franka, parallel jaw)

- **Tiles** (14 × 36 × 105 mm, 50 g): grasped across the 14 mm faces — well inside the
  ~80 mm jaw span, light enough for any grip force. Standing a tile upright is a standard
  top-down place; chain geometry tolerates ±15–20 mm placement error and ~10° yaw error
  (gaps 42–48 mm vs. 105 mm tile height, window 55 mm vs. 36 mm tile width, and the
  straight-line chain self-centers the last tile into the window — observed on both seeds).
- **Trigger nudge**: a one-finger 0.16 N push at mid-tile height until ~15° lean; no grasp
  needed, timing not critical (the cascade is self-propagating once past the 7.6° balance
  point).
- **The ball is intentionally *out* of the embodiment**: the task is designed so the arm
  never needs to reach it.
- **Base pose**: Franka base at ≈ (0.50, −0.15, 0) — beside the corridor, ~0.35–0.75 m
  reach to every tile scatter position, the pad, and every chain slot, without the arm
  ever crossing above the standing chain (approach from the side).

## Execution order declared

1. `scene.py` written; layout/threshold math validated standalone.
2. `solve.py` iterated on the forge to `SIM_GEN_SOLVE: SUCCESS` on seed 0, then seed 1.
3. `smoke.py` battery run on the forge: **16/16 PASS** (frames.npz recorded).
4. `TASK.md` written; final clean runs are the three above (package unchanged).

## Check list (smoke.py, 16 checks)

1. Settle readback — trigger standing on pad, ball seated at shelf seat, 8 tiles flat; all finite.
2. Score ≈ 0 at reset, no success.
3. Randomization readback A — gallery xy/yaw, span, bearing vary across seeds 21–26.
4. Randomization readback B — trigger yaw and tile scatter vary across seeds.
5. Null policy — 240 idle steps: score ≈ 0, no success.
6. **Seed strategy** — ball hand-set at rest on the pit floor (in-pit verified): NOT success, score ≤ 0.02.
7. **Out of order** — trigger down + 4 tiles fallen in corridor + ball in pit, but ball moved BEFORE the trigger: causal 0, NOT success.
8. **Too slow** — ball dislodged 3.5 s (> tau) after the trigger fell: causal 0, no pit credit.
9. **Relay clause** — causal genuinely latched but ZERO tiles fallen in corridor: NOT success, score ≤ 0.75.
10. Near miss — ball dislodged but resting ON the shelf: pit False.
11. Near miss — ball on the floor OUTSIDE the front wall: pit False.
12. Wrong place — 4 tiles fallen laterally outside the corridor: fallen_in_corridor 0.
13. Latched credit — standing relay latches `built`; removing tiles leaves it latched, still no success.
14. Monotonicity — more standing relay tiles ⇒ strictly more latched credit.
15. Rejection audit — success() never True at any judged point in the battery.
16. Final no-NaN across all task objects.
