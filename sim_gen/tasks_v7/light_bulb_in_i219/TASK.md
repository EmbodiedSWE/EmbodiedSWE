# light_bulb_in_i219 — Roll the globe into the lantern shelter and shutter it in

**Scene:** `simgen.lantern_shelter` (`LanternShelterScene`, robot="null")
**Seed:** `rlbench/light_bulb_in`
(`RoboVerse/roboverse_pack/tasks/rlbench/light_bulb_in.py`)

## Seed provenance and what changed

The RLBench seed is a direct insertion: a small bulb stands in a holder, the
plan is **grasp the bulb, carry it over the lamp, lower it into the socket and
screw it in with a wrist rotation** — one grasp ON the goal object, one
free-space transport, one insert-plus-twist at the goal. There is also a decoy
bulb the agent must not take.

This task keeps the goal predicate ("the correct bulb ends up installed in the
lamp's socket, distractor excluded") but replaces the entire plan skeleton:

1. **The bulb cannot be grasped.** The lamp's globe is a 90 mm sphere against
   an 80 mm parallel-jaw span — asserted in `__post_init__`. The seed's
   grasp-and-carry schema transfers zero: the only way to move the globe is
   non-prehensile rolling by pushing on it.
2. **The socket cannot be approached from above.** The four-post socket cradle
   sits INSIDE a roofed shelter (roof underside 135 mm; a seated-globe top is
   ~94 mm, but a globe held above the seat cannot descend through the roof —
   smoke check 6 drops it from above and it rolls off the roof, no credit).
   The only route is the floor-level doorway in the front face (110 mm wide ×
   105 mm tall vs. the 90 mm globe): the whole approach is a planar roll
   through a portal, not a vertical insert.
3. **A real force threshold replaces the screw.** The seat is four brass ball
   posts. Rolling in from the doorway the globe must climb a 10.25 mm entry
   pivot barrier (quasi-static ~0.52 N for the 150 g globe; smoke check 7
   pushes with a 0.30 N cap for 4 s: it travels ≥ 20 mm, stalls at the posts
   and rolls back — no seat, no enter credit). A firm shove pops it over; the
   back wall 3 mm behind the seat window geometrically caps overshoot, so a
   super-threshold shove is captured, not lost. Retention: the exit barrier is
   6.55 mm, so the seated pose survives the persistence window and settling.
4. **Closure is part of the goal.** A free-sliding storm shutter (a slab
   captured purely by curb/rail channel geometry — the scene has NO joints)
   must be slid along its channel to cover the doorway; pushing it to the near
   end stop is exactly closed (stop face at −0.074, tolerance 8 mm). Success
   = globe seated on all four posts AND shutter covering the doorway AND
   everything settled.
5. **The decoy is excluded by physics, not by fiat.** The 48 mm spare bulb's
   contact reach (24 + 8 = 32 mm) is less than the post diagonal half-spacing
   (33.94 mm): placed over the cradle it falls straight through to the shelter
   floor and can never seat (smoke check 11).

So the plan skeleton changes from *"grasp, carry, insert, screw"* to
*"roll a too-big-to-grasp object across the floor, through a floor-level
portal, shove it over a retention barrier onto a post cradle, then slide a
free closure panel to seal the portal"* — whole-route non-prehensile
transport with a force-threshold seating and a separate closure act, instead
of prehensile pick-and-insert.

## Why strategically different from every examined sibling

- **Seed (`rlbench/light_bulb_in`):** see above — grasp denied (90 > 80 mm),
  top-down insert denied (roof), screw replaced by a pivot-barrier shove,
  and a closure step added; the goal object is only ever pushed.
- **`light_bulb_in_i120` (turnstile delivery):** there a *jointed carrier
  mechanism* (free revolute turnstile) is operated out–load–in and the arm
  PLACES the grasped bulb onto the carrier; the bulb rides the mechanism
  through the wall. Here there are no joints anywhere, nothing is ever
  grasped or placed, and no mechanism carries the object: the arm pushes the
  goal object itself along the whole route and then separately pushes a free
  slab closed. Disjoint machinery (rolling + portal + detent + free-slide
  closure vs. jointed carrier), disjoint forced structure (roll-shove-seal
  vs. out–load–in).
- **`light_bulb_out_i204` (rear-port tool ejection):** there a TOOL (push
  rod) is threaded through a service port to push the bulb OUT, and gravity
  performs the transfer ballistically. Here there is no tool intermediary —
  contact is direct on the goal object — the delivery is INWARD and uphill
  (over a barrier onto a raised seat, the opposite of gravity delivery), and
  the task ends by sealing the portal, which i204 has no analogue of.
- **`lamp_off_i101` (twistlock unplug):** there the challenge is identifying
  the right plug by cord-trace among decoys, then a grasped key-twist-pull.
  Here identification is by physical size (the decoy self-rejects by falling
  through the posts), nothing is grasped, and there is no keying: the
  challenges are portal-constrained rolling and a force threshold.
- **`robobench/packing/pen_holder` (exemplar, not corpus):** direct drop of
  free objects into an open receptacle — exactly the schema removed here
  (the roof denies any drop; the seat is reached only by rolling through the
  doorway and shoving over the barrier).

## Solution outline (solve.py — NOTHING is teleported)

External-wrench plant recipe: `enable_external_forces_every_iteration: True`
in the scene's physx cfg; authored diagonal inertias so both servos are
auditable and discretely stable — globe kv·dt/m = 6/(120·0.15) = 0.33 < 1,
shutter kv·dt/m = 25/(120·0.30) = 0.69 < 1. Wrenches are encoded body-frame
via `quat_apply_inverse` (the rolling globe's orientation is arbitrary).

- **Phase 0 (reset):** settle 0.5 s, read back layout, assert score ≤ 0.02.
  `SIM_GEN_SCORE` 0.000.
- **Phase 1 (stage, dynamics):** a horizontal CoM velocity-servo (cap 4 N,
  z-force zeroed — the ground carries the weight) rolls the globe from its
  random spawn to the staging point 0.20 m in front of the doorway on the
  door axis (|y| < 12 mm asserted). Score still 0.000.
- **Phase 2 (shove, dynamics):** the servo drives the globe +x through the
  doorway with a y-guard, releases at the seat line, and lets it fly over
  the entry barrier; escalating speeds (0.55 → 0.90 m/s) with retreat and
  restage between attempts. The posts catch it; the back wall caps
  overshoot. Assert seated + enter latch. `SIM_GEN_SCORE` 0.500.
- **Phase 3 (close, dynamics):** the shutter servo pushes the slab −y along
  its channel to the near end stop (stop = exactly closed), releasing when
  inside the closed window and slow. Assert door_closed. `SIM_GEN_SCORE`
  ≥ 0.75 → success → 1.000.
- **Phase 4 (settle):** all bodies at rest, success live. 1.000.
- **Phase 5 (persistence):** 420 substeps (3.5 s) hands-off; `success()` is
  live state every step. Only then `SIM_GEN_SOLVE: SUCCESS`.

Verified end-to-end on the forge on seeds 0 and 1 (18.0 s and 21.5 s, both
`SIM_GEN_SOLVE: SUCCESS`, scores non-decreasing 0.000 → 0.500 → 1.000).

**Execution order is REQUIRED and physically forced:** the shutter spawns
parked open and the closed shutter leaves no globe-passable gap (asserted:
doorway is the only ≥ 90 mm aperture and the closed slab covers it with 2 mm
channel slack), so enter-then-close is the only order; the seat cannot be
reached from above (roof, smoke 6) or by a gentle push (barrier, smoke 7),
so the doorway roll + firm shove is the only seating path; the decoy can
never satisfy the seat window (fall-through, smoke 11). The rubric's latches
mirror exactly this chain.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `enter_latch` (0.20): globe center ever past the doorway plane inside the
  chamber band (local x ≥ −0.014, |y| ≤ 0.040, z ≤ 0.070). No floor-resting
  globe pose exists with center x ≥ −0.033 (post interpenetration —
  asserted), so this fires only during a genuine transit or climb.
- `seat_latch` (0.30): globe center ever inside the seat window (|xy| ≤ 6 mm,
  z ∈ [0.0465, 0.054] local) — the four-post rest pose (z = 0.04871).
- `shut_latch` (0.25): shutter ever closed WHILE the globe is seated (a
  closed-empty shelter earns nothing — smoke 10).
- success (to 1.0): globe seated AND shutter closed (slide offset ≤ 8 mm,
  flush ≤ 3 mm, upright ≤ 10°) AND everything settled.
- Latches are transient-achievement credit (`torch.maximum`); cap 0.75
  without success (float32 boundary honoured with +eps in smoke). Null
  policy ≈ 0 (globe and decoy spawn on the open floor, shutter parked open).

## Embodiment argument (single Franka, parallel jaw 80 mm, OSC)

Base at (0, 0) on the floor, facing +x: the globe spawn band (world
x ≈ 0.24–0.42, |y| ≤ 0.08 around the shelter at x ≈ 0.58–0.66), the whole
rolling route, the doorway apron and the shutter channel (world x ≈
0.49–0.59, |y| up to ~0.29) all sit inside a 0.75 m reach disc at floor-to-
0.15 m heights the Franka reaches comfortably from above and from the side.

- **Rolling the globe:** a 90 mm sphere is pushed with the closed fist /
  jaw side at equator height (45 mm) — the classic non-prehensile planar
  push; the CoM servo's decelerating profile is exactly a guarded move with
  velocity feedback the OSC controller provides.
- **The shove:** 0.55 m/s at 150 g is a light dynamic push; the arm releases
  at the doorway plane and NEVER reaches inside — at first post contact the
  globe's rear pole is at local x = −0.098, still outside the front wall's
  outer face (−0.072), so the fist stays clear of the jambs at all times.
- **The shutter:** a 115 mm slab pushed along a straight channel to a hard
  stop — push-to-stop is exactly closed, so no fine positioning is needed;
  8 mm tolerance vs. a rigid end stop is a contact-terminated guarded move.
- **No blind reach:** shelter, doorway, amber shutter, brass posts (visible
  through the doorway), and both bulbs are visible from outside; the seat is
  confirmed by the felt pop-over and the globe disappearing from the apron.

## Checks (smoke.py — rejection battery, 17 checks)

1. Clean reset: states finite; globe and decoy in their floor spawn bands,
   shutter parked open in the channel, doorway clear.
2. Score ~0 / no success at rest.
3. Randomization (8 seeds): shelter xy + yaw + shutter park offset vary by
   READBACK; the shutter tracks the shelter's channel frame.
4. Randomization: globe spawn spreads; decoy spawns on both sides.
5. Null policy: 240 idle steps, score ≤ 0.02.
6. **Seed-strategy family (from above):** globe dropped from z = 0.45 over
   the seat — bounces off the roof, never enters, ~0.
7. **Barrier retention (gentle push):** 0.30 N capped push (< 0.52 N
   threshold) held 4 s — travels ≥ 20 mm (non-vacuous), stalls at the posts,
   never seats, rolls back; no enter/seat credit.
8. Doorway lodge: globe pushed just into the doorway mouth and released —
   no seat credit from a portal-lodged globe.
9. Seated-but-open: globe placed on the posts, shutter open — score exactly
   0.50 (+eps), no success.
10. Closed-empty: shutter slid closed with no globe — score ≤ 0.02.
11. Wrong object: the DECOY placed over the cradle falls through the posts
    to the shelter floor — never seats, ~0.
12. Settle gate: shutter written closed but moving (0.5 m/s) — shut latch
    fires, not settled, not success; it slides out of the closed window.
13. Cap: all latches constructed, then the globe yanked outside → score ==
    0.75 cap (+eps), NOT success.
14. Latched credit: 40 further steps, score unchanged.
15. Rejection audit: success() never True at any judged point.
16. Final no-NaN.
17. Camera: ≥ 20 rgb frames captured → `frames.npz`.
