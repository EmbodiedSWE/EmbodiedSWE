# spit_roast (open_oven_i378)

Thread a steel spit rod through the bore of a meat block at a loading station, carry the
loaded spit across the room, and seat both rod ends in the slotted brackets of a
free-standing roasting rack so the meat hangs between the posts.

## Seed provenance

Derived from `rlbench/open_oven` (RoboVerse pack): a Franka pulls a hinged oven door
open — one articulated joint, one pull, success = joint angle.

## What changed and why it is strategically different

The seed (and the sibling tasks in this family) interact with a PRE-BUILT articulated
mechanism: grasp a handle/knob, actuate an existing joint (door pull i213/i6, spindle
dials i7). Here there is **no articulation and nothing opens**. The strategy is
inverted: the solver must **manufacture a compound object** — thread a loose rod
through a passive block's through-bore against a backstop (contact insertion with a
physically forced direction: the 32 mm end knob cannot pass the 26 mm bore) — and then
**create a gravity-borne suspension**, seating the rod ends in two funnel-flared slots
so the block hangs by its bore between the posts. The load-bearing physics is
peg-in-hole insertion, carrying a swinging suspended load, and a two-point
simultaneous seating — none of which appear in the seed or siblings. It is also
strategically distinct from `pick_up_cup_i364` (single-object lift to a height band):
success here is a three-body assembly relation, not a pose of one object.

## Scene

- **Rack** (static, at x = 0.45, y = 0): two posts 0.26 m apart, each topped by a
  15 mm slot between cheeks with 45-deg funnel plates (mouth 128 mm, funnel top
  0.267 m). Seated rod centre height 0.200 m.
- **Loading station** (kinematic, at y = ±0.30 with ±30 mm xy jitter and ±35-deg yaw):
  a 0.10 m cradle cube with a low backstop plate on its local −x side (top 0.128 m —
  below the resting block's bore).
- **Roast block** (dynamic, 0.7 kg): 90 mm cube with a 26 mm square through-bore,
  resting on the cradle, yaw = station yaw ± 4 deg.
- **Spit rod** (dynamic, 0.4 kg): Ø12 mm × 410 mm rod, bare tip at one end, Ø32 mm
  knob at the other, lying on the floor on the opposite side of the rack from the
  station (spawn rejection-sampled clear of rack + station footprints).

Randomization (verified by readback in smoke): station side (left/right of the rack),
station yaw + xy jitter, block yaw + bore-axis flip, rod spawn pose.

## Success

All simultaneously, settled and finite:
- rod **threaded**: protrudes ≥ 20 mm past BOTH bore faces, within 11 mm of the bore
  axis at both faces (exceeding that requires wall penetration);
- rod **seated at both notches**: rod x-extent covers each notch plane by ≥ 20 mm,
  |y| ≤ 6 mm and |z − 0.200| ≤ 8 mm at each notch plane;
- block **hanging between the posts**: centre z > 0.170 (hanging rest: 0.193, cradle
  rest: 0.145) and |x − rack_x| ≤ 45 mm (strictly inside the 65 mm post jam limit, so
  shoving the block along the rod against a post revokes success — it stays live
  state, never a latch).

Score: 1.0 iff success holds now; otherwise latched best stage — entered 0.15,
threaded 0.40, carried (threaded + hanging) 0.65. Null policy ≈ 0.

## Teleport solution (solve.py)

Teleports = transport only. ONE root-pose write: park the rod in free air, collinear
with the block's bore axis, 280 mm short of the bore, zero velocity (the arm's
carry-and-present). Everything else is contact dynamics through the scene's external
wrench buffers (world-frame by contract; the scene pre-encodes per step into the body
frame — the grasp-wrench stand-in for the Franka's hand on the knob; the block's
buffers are never touched):

1. **THREAD** — a lag-clamped carrot walks the rod tip down the live bore axis at
   50 mm/s; the rod pushes the block against the backstop and passes through until it
   protrudes past both faces, centred.
2. **LIFT** — raise to 0.30 m; the block is picked up BY the rod (bore top rides the
   rod). Load feedforward gates on the block actually rising; a gravity-moment
   feedforward cancels the hanging load's pitch.
3. **TRAVERSE** — carry to above the rack, slewing the rod axis to the rack x axis at
   0.5 rad/s (0.30 m clears the 0.267 m funnel tops; the hanging block passes between
   the notch assemblies).
4. **SEAT** — descend at 30 mm/s; the funnels gather both ends into the slots; the
   wrench ramps to zero over 30 steps and gravity owns the seat.
5. Settle to success, then ≥ 3.5 s hands-off persistence (all wrench buffers asserted
   zero) before `SIM_GEN_SOLVE: SUCCESS`.

## Execution-order declaration

The order **thread → lift → carry → seat** is physically forced, not conventional:

- *Seat first, then thread?* Impossible: a block threaded onto a seated rod's outboard
  overhang cannot travel into the between-posts region — the post blocks it in z
  (hanging block bottom 0.148 < post top 0.194, asserted) and the cheek assembly
  blocks it in y (block half-width 45 mm > cheek outer span 19.5 mm, asserted) — and
  success requires the block to hang BETWEEN the posts (roast_x_tol ≤ inter-post
  clearance, asserted).
- *Thread direction*: forced by the knob (32 mm > 26 mm bore, asserted) — tip first.
- *Carry height*: the funnel tops (0.267 m) force a lift before the rack approach can
  end in a seat.

Smoke check 5 (bare rod seated without the block → no credit) and check 6 (block
dropped onto the racked rod → never threads: the bore is closed, lateral capture is
impossible) exercise the two halves of the order argument.

## Embodiment argument (single Franka, parallel jaw, OSC)

One plausible base pose: on the rod-spawn side of the room at (0.15, −0.05) facing
+x/+y (station side +y; mirror for −y), putting the rod spawn band (x 0.08–0.28,
|y| 0.18–0.34), the station (±0.33, 0.30) and the rack seats (0.32/0.58, 0, 0.20)
all inside a 0.85 m reach envelope at working heights 0.0–0.30 m.

- **Spit rod**: the only object the hand touches. Grasp the Ø12 mm shaft next to the
  knob (parallel jaws close to 12 mm; the knob is a natural end-stop preventing axial
  slip toward the hand). Wrist-yaw + arm translation reproduce every motion the servo
  performs: present collinear to the bore, push axially (≤ 30 N, well inside Franka's
  continuous wrench), lift 1.1 kg total (≪ 3 kg payload), slew the axis at 0.5 rad/s,
  lower into the funnels. The final release is a plain gripper open with the rod
  already resting in both slots.
- **Roast block**: never grasped — it is positioned by the scene and moved only
  through rod contact (push against backstop, hang on rod). A 90 mm cube would not
  fit the 80 mm jaw span anyway; the task is designed so it never needs to.
- **Rack / station**: never touched; static/kinematic fixtures.

## Checks (smoke.py)

1. settle: finite, block on cradle, rod on floor, score < 0.05
2. randomization readback across seeds 11/12/13 (station side/yaw/xy, rod pose differ;
   station teleport-target readback < 5 mm)
3. null policy 240 steps → score < 0.05
4. seed-strategy analogue (haul + door-like swing of the bare rod) → no credit
5. bare rod seated in both notches without the block → seated() true but score < 0.05,
   no success (wrong-object outcome rejected)
6. block dropped onto the racked rod → never threads (closed bore), score < 0.05
7. partial insertion (25 mm) → entered credit band only, not threaded
8. threaded assembly on the floor → threaded credit, not seated, no success
9. seat capture from y-offsets 20/40 mm → funnels gather both ends (tolerance is real)
10. seat from centred hover → full success, score 1.0
11. seat miss at 80 mm y-offset (beyond mouth) → not seated, no success, latched 0.65
12. knock-off after success: axial shove on the block → success revoked, score falls
    back to the 0.65 latch (success is live state, not a latch)
13. ≥ 20 video frames recorded → frames.npz
