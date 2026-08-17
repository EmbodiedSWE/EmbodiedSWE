# airlock_transfer — cycle the anti-phase gate to pass the soup can through a sealed vault's airlock

`living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_i236` · scene `airlock_transfer` · env `simgen.airlock_transfer`

## Seed provenance

Seed task: `roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.py`
(RoboVerse / LIBERO-90). The seed is a pick-and-place: lift the alphabet-soup can
off a table shared with look-alike grocery cans and lower it into an open tray;
`_terminated` is bounding-box containment on the tray's `contain_region`.

Kept from the seed: the protagonist (a 66 mm x 90 mm soup can), the "identify the
red can among identical-shape distractors" framing (corn yellow, cream white,
slot-shuffled per episode), and the goal readout "the alphabet-soup can inside the
tray".

Changed: the tray is sealed inside a **vault with no door and no open top** — its
only aperture is a transfer window low in a divider wall; the reach-in drop became
a **two-opening airlock protocol** driven by ONE gate slider whose roof blade and
hanging window blade are welded in strict **anti-phase**; the containment readout
stayed, but no transport motion can satisfy it — the can must be *routed through a
mechanism* whose invariant forbids carrying it in.

## Strategic difference argument

- **vs the seed:** the seed's whole strategy is transport — carry the can over the
  tray and release. Here that exact move is constructed in smoke check 5 and
  REJECTED: released over the tray, the can settles on the sealed vault roof, no
  credit. Success requires sequencing a mechanism: open A, feed, close A (which IS
  opening B — one rigid body), let gravity finish. The strategy is a protocol, not
  a placement.
- **vs sibling i158 (shuttle-drawer kiosk):** i158's shuttle is a container that
  MOVES the can; both of its positions are can-accessible and the agent chooses
  when to push. Here the gate never touches or carries the can — it only permutes
  which aperture exists; the can travels by gravity alone, and the 29 mm dead band
  makes the two apertures PROVABLY never coexist (smoke checks 6/7/9 exercise both
  blades and the dead band against real drops and 5 N pushes).
- **vs i139 (beam balance), i166 (roll-chute), i207 (vertical queue):** those are
  continuous mechanisms — a scalar readout (tilt, roll distance, queue depth)
  advances monotonically with effort. The airlock is *discrete and non-monotone*:
  driving the gate toward the goal position first destroys the aperture you fed
  the can through, and c must REVERSE direction between phases (right +0.065, then
  left −0.055). No sibling requires undoing your own first move.
- **vs i33 (flap chute):** the flap is one passive one-way gate that re-closes by
  gravity; a single drop defeats it. Here both blades are rigid, actively driven
  through a captive handle, and each of the two openings is sealed exactly when
  the other is usable — a single drop can never reach the tray (checks 6 and 9).

## The mechanism (numbers)

- Station (25 kg static-by-mass dynamic body, footprint ~194 x 409 mm, roof plane
  z 0.257): open-topped ANTECHAMBER with a 10° slick ramp floor draining through a
  window (140 x 130 mm) in the divider; sealed VAULT behind it holding the amber
  tray (interior 130 x 130 mm, walls 55 mm) on a fenced seat.
- Gate (0.6 kg, free rigid body, captive): rides the roof between rails (z
  0.257..0.269), rise-capped by keepers (z 0.269..0.277), stroke c ∈ ±0.076
  between end stops; a support shelf continues the slide plane past the plate edge
  (an unsupported blade seesaws over the roof edge — measured 13° pitch, fixed by
  geometry). Handle post 26 x 26 x 34 mm on top.
- Anti-phase invariant (station-local, blade A = roof plate x span c−0.010..c+0.140
  over the 110 mm hatch; blade B = hanging plate x span c−0.150..c in the window):
  A passes the 66 mm can iff c ≥ +0.027; B passes iff c ≤ −0.002 and seals for
  c ≥ +0.005 — **disjoint, 29 mm dead band**. Every leak path (12 mm gate slot,
  ~18 mm blade corridor, 4 mm ramp-lip slit) is far below the can diameter.
- Gravity feed: the ramp slab extends THROUGH the divider plane to a drop edge at
  y −0.012 / z 0.1026 on the vault side — a quasi-statically creeping can never
  hands over to a flat sill (it parks there — measured); it rides the tilted
  surface until its CoM passes the edge and tips ~75 mm down into the tray.

## Rubric

`success()` (all clauses live): alphabet can inside the tray interior AND inside
the vault airspace AND everything settled AND finite.

`score()`: latched partial credit 0.30 in_chamber + 0.15 staged (at the window,
resting on the ramp bottom) + 0.30 in_vault, capped at 0.75; exactly 1.0 iff
success() live. Null policy ~0 (cans spawn on ground slots outside every credit
volume).

**Ordering declaration:** the rubric imposes NO step ordering — the airlock
geometry does. Each credit volume is reachable only through an aperture that the
gate must be posed to create, and no prefix of any probe sequence satisfies the
goal (smoke checks 5–10).

## Solution (solve.py) — teleport for transport only

- P0 settle; mass + layout readback asserts (station yaw/xy, gate c, slot
  shuffle). Score 0.
- P1 drive the gate RIGHT to c=+0.065 by applied force on the handle (overdamped
  position servo, ≤8 N — the force a Franka applies through a grasp): hatch
  passes, window sealed.
- P2 TRANSPORT: teleport the red can once to a zero-velocity hover over the open
  hatch (what pick-and-carry delivers), then hands-off: it falls through the
  hatch, slides down the ramp, and stages against the closed window blade. Score
  latches 0.45; asserted NOT success.
- P3 drive the gate LEFT to c=−0.055: the window opens (hatch now sealed), the
  can creeps down the ramp, tips off the drop edge, lands in the tray. Score 1.0.
- P4 ≥3 simulated seconds hands-off persistence (10 x 40 steps, success asserted
  every block), then `SIM_GEN_SOLVE: SUCCESS`.

Passes on ≥2 seeds; `SIM_GEN_SCORE` printed at each phase boundary is
non-decreasing (0 → 0 → 0.45 → 1.0 → 1.0).

## Franka embodiment argument

- **The gate** is the only actuated fixture: its 26 x 26 mm handle post stands
  proud of the guideway (top z ≈ 0.30 m), within the Franka's 80 mm jaw span, with
  clear air above and beside it; the drive is a horizontal slide of ≤8 N over
  132 mm — well inside Franka payload/force limits, and the captive channel means
  imprecise pushes cannot derail it (smoke check 11 yanks it with 8 N both ways).
- **The cans** (66 mm diameter, 0.35 kg) fit the 80 mm jaw opening; they start in
  open ground slots in front of the station with ≥120 mm spacing — a top-down or
  side grasp with no clutter contact.
- **The drop** requires holding the can over the hatch at z ≈ 0.31 and releasing —
  a point ~0.45 m from the station front edge at modest height; no insertion, no
  in-chamber reach is ever needed (the geometry forbids it, and the solve never
  does it).
- **Base pose:** on the antechamber side (station-local +y), base origin ~0.55 m
  from the station center, facing the hatch. From there the ground slots
  (0.30–0.45 m), the handle (~0.45 m, z 0.30) and the hatch hover pose are all
  inside the ~0.85 m reach envelope at comfortable elbow-up postures.

## Smoke battery (12 checks, `SIM_GEN_SMOKE: ALL PASS 12/12`)

1. settle/no-NaN — seeded reset settles finite; gate on stroke, cans grounded,
   tray seated; score ~0, no success.
2. randomization readback — station yaw, station xy, gate start c, can xy all
   differ across two seeds.
3. slot shuffle — ≥2 distinct alphabet slots over 10 resets.
4. null policy — 240 idle steps: nothing moves, score ~0.
5. SEED STRATEGY REJECTED — the can released over the tray ends ON the sealed
   vault roof; no credit.
6. anti-phase A — window open ⇒ a hatch drop lands ON the closed roof blade,
   never enters the chamber.
7. anti-phase B — hatch open ⇒ a mid-ramp can slides ≥50 mm and stages against
   the closed window blade (0.45 latched); a 5 N push toward the vault never puts
   it through.
8. latch persistence — staged can teleported back to the ground: 0.45 survives,
   success stays False (success is live).
9. dead band — at c=+0.012 neither opening passes: the staged push fails AND,
   with the gate held there (an agent gripping the handle), a drop over the
   uncovered 57 mm hatch strip cannot end inside the chamber (unheld, the drop
   impact merely shoves the captive slider to its stop — i.e. opens the hatch).
10. wrong object — CORN in the tray, alphabet on the ground: no success, no
    credit.
11. captive gate — 8 N yanks traverse the full stroke both ways; the gate never
    leaves the stops, rails, or roof plane.
12. frames.npz — video captured to CWD.
