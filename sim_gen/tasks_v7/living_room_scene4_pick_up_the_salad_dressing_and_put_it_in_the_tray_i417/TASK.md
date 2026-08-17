# decant_return — pour the sealed marble into the tray, re-seat the empty bottle

## Seed provenance

Seed: `libero_90/living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray`
("pick up the salad dressing and put it in the tray").

## What changed, and why it is strategically different

The seed is a prehensile transport task: grasp a bottle, carry it, release it inside
a tray. Here the **bottle is never the deliverable — its CONTENT is**. A 30 mm marble
is sealed inside a square amber bottle (70 mm body, 130 mm tall) whose only opening
is a 40 mm square mouth at the top of a funnel shoulder; the marble rests > 100 mm
below the mouth, so no gripper can reach in or pinch it out (mouth 40 mm < marble
30 mm + 2×8 mm jaw fingers). The goal state is a CONJUNCTION at two different
fixtures: the marble at rest inside the walled tray AND the empty bottle standing
seated in a separate 82 mm coaster socket on the opposite side of the rig.

The seed's own strategy is an explicit failure: putting the bottle in the tray
scores nothing (and the marble, still sealed inside, is not "in the tray" — the
in-bottle exclusion is part of the predicate). The only solution channel is:

1. pick the bottle up (it fits the jaw),
2. hold it over the tray, above the 55 mm walls,
3. roll it far past horizontal (discharge at ~100–135° depending on friction
   draw) so the marble rolls down the funnel and pours out of the mouth,
4. aim the pour so the marble lands between the walls (a marble on the ground
   cannot get over the 55 mm walls — verified by a smoke probe),
5. re-erect the empty bottle and drop-seat it in the socket (square-in-square
   with 6 mm/side slack; a 45°-yawed base diagonal 99 mm > 82 mm interior, so
   seating requires yaw registration too).

Difference from the sibling tasks built in this campaign: i18 drains via a wedge
jack, i307 is non-prehensile push routing, and i316 (carafe pour) delivers a CAN
into a tray by pouring it out of a held carafe. This task inverts the i316
emphasis: the pour is only half the goal — credit for seating is **gated on the
decant having happened first** (`_seated |= _decant & bottle_seated()`), making it
an ordered two-fixture conjunction with an irreversible-evidence latch: the decant
latch only fires when the marble leaves the bottle interior CONTINUOUSLY —
per-step travel < 60 mm for BOTH bodies (teleporting the marble out jumps too far,
and teleporting the bottle away from around a stationary marble is refused too) —
so any end-state-identical bypass fails even though the final poses are
pixel-identical (smoke check 5).

## Physical soundness

- Pour possible: marble Ø30 < mouth 40; funnel plates at 45° guide it out.
- Pinch-out impossible: mouth 40 < 30 + 16 (two 8 mm fingertips), and the marble
  sits > 100 mm deep.
- Discharge: cavity wall slope exceeds the friction angle (~8.5°) once the roll
  passes ~99°; observed discharge 109.6° (seed 0) and 135.0° (seed 3).
- Exit drift: ~40–50 mm along −y at discharge; the aim point is biased +30 mm and
  the 170 mm interior + 55 mm walls catch the spread.
- Socket: 82 mm interior vs 70 mm base = 6 mm/side drop-in slack; the 25 mm lip
  retains the seated bottle (persistence holds 3.3 s hands-off).

## Solution outline (solve.py, transport-only teleports)

P0 settle + layout/mass readback asserts → transport-teleport the bottle (marble
riding inside, relative pose preserved) to an upright hover over the tray →
**kick probe**: measure the pod's external-force frame (weight-cancelling vertical
ff + 0.6 N horizontal kick; the direction the bottle travels is the drag yaw of
`R_now·R_ref^T`; on this pod −76° to −122°, i.e. `R_ref` = the bottle's random
spawn yaw) and build the exact per-step pre-encoding `R_ref·R_now^T` → engage a
6-DOF CoM wrench servo (force kp 50/kd 10/cap 12 N + live gravity ff including
the resident marble; torque kq 0.8/kdw 0.02/cap 1 N·m + marble-offset gravity
moment) with tight rotation/velocity bails (the rotation loop diverges at
~sqrt(kq/I) ≈ 54 rad/s — guarding position alone lets a wrong frame mode
centrifuge the marble out) → 35° tilt probe (drag invisible upright) → mouth-
anchored escalating roll 115/135/155/170° at 26°/s with an in-pour divergence
guard (0.10 m / 0.6 rad, toggle-and-retry with marble re-insertion + un-earning
of accidental latch credit) → hold through the landing until the marble settles
in the tray → re-erect, transport-teleport the empty bottle over the socket,
yaw-aligned drop-in with measured-offset retries → success live → 3.3 s
hands-off persistence. All load-bearing interactions (the pour, the landing,
the drop-seating) are contact dynamics; teleports only transport.

SUCCESS on seeds 0 and 3.

## Franka embodiment argument

A single Franka with an 80 mm parallel jaw, base at rig-frame ≈ (0.0, −0.42, 0)
(facing the rig center, tray and socket both within ~0.65 m reach):

- **Bottle**: 70 mm square body fits the 80 mm jaw with 5 mm/side engagement;
  mass 0.18 kg loaded. Grasp mid-body, lift, hold over the tray, and roll the
  wrist (joint 7 continuous ±166° plus wrist re-orientation reaches the ~110–135°
  discharge) — exactly the held-pour the wrench servo emulates. Re-erect and
  lower into the socket; the 6 mm/side slack tolerates Franka repeatability.
- **Marble**: never needs grasping — it is delivered ballistically by the pour.
  (It could not be grasped inside the bottle anyway; that is the point.)
- **Tray / socket / rig**: kinematic fixtures, never manipulated.

## Ordering declaration

Decant BEFORE seat: the seat latch is gated on the decant latch, so seating the
still-full bottle first earns nothing until the pour has happened (and pouring
requires picking the bottle back OUT of the socket — physically possible, but the
credited order is decant-first; the success clause itself is an end-state
conjunction plus the decant evidence latch).

## Rubric

Latched, non-evaporating: lift 0.10, hover-over-tray 0.10, decant observed 0.20,
marble ever in tray 0.15, seat-after-decant 0.15 (cap 0.70); exactly 1.0 iff
`success()` (marble settled in tray outside the bottle ∧ bottle settled seated
upright ∧ decant latch ∧ finite).

## Smoke battery (10 checks)

1. settle / no-NaN;
2. randomization readback, 3 seeds max-pairwise (rig yaw/xy, bottle xy/yaw,
   marble in-cavity);
3. null policy 240 steps ≈ 0;
4. SEED strategy refused: bottle (marble inside) settled in the tray — no credit;
5. BYPASS flagship: end-state-identical teleport (marble into tray + bottle
   drop-seated) — every live predicate true, refused for missing decant latch;
6. wall probe: ground marble rolled at the tray with velocity kicks (drag-proof —
   force pushes on a rolling ball drift under the pod's wrench frame drag; kick
   speed capped below the wall-hop bound sqrt(2·g·Δh) ≈ 0.9 m/s) — moves > 30 mm,
   arrested by the outer wall, never enters;
7. socket near-miss: bottle upright 60 mm off the socket — `bottle_seated()`
   refuses;
8. never-poured: bottle genuinely drop-seated with the marble still inside —
   seat is real, credited seat latch stays false, score ≤ 0.01;
9. rejection audit: `success()` never fired during checks 4–8;
10. frames.npz saved (> 10 frames).
