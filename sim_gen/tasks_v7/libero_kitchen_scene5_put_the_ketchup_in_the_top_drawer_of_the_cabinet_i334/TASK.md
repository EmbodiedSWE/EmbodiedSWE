# letterbox_cradle_vault — lay, slide, and crank-erect the bottle inside the sealed vault

**Task id:** `libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_i334`
**Seed task:** `libero_90/libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet`
(RoboVerse `roboverse_pack/tasks/libero_90/...`): pull the cabinet's top drawer open, pick up
the upright ketchup bottle, drop it into the drawer.

## What changed, and why it is strategically different

The seed is an *open-receptacle pick-and-place*: the container is opened by a translation
(drawer pull), the bottle keeps its upright pose the whole way, and success is a bounding-box
containment check on the drawer.

Here the receptacle **never opens** and the **bottle's pose is the puzzle**:

- The vault's only aperture is a wide, **low letter slot** (15 cm x 9 cm). The bottle
  (Ø6 cm x 16 cm) is nearly twice the slot's height: **upright entry is physically refused
  by the slot header** (smoke-proven with a real shove). The seed's whole skill — carry the
  upright bottle to the receptacle — earns ~0 here.
- Entry demands a **reorientation**: lay the bottle on its side in the apron's guide lane and
  slide it **base-first** through the slot onto a hinged **tilt cradle** inside.
- The goal pose (upright, inside) is **produced by a mechanism, not by the hand**: an external
  crank (through-wall axle) raises the cradle; past its loaded **over-centre angle (~68°)**
  the cradle tips onto its 90° stop *by itself* and stands the bottle on its end plate —
  upright inside a chamber no hand can enter. The cradle is gravity-**bistable** (authored
  CoM offset), so there is no stored energy against the goal and the terminal state presses
  its stop (~0.14 N·m margin) hands-off.

Versus the sibling tasks from the same seed and my other packages: **i154** (carousel: azimuth
alignment, upright insertion through a window, rotate-to-stow) shares neither the aperture-
defeat reorientation nor the over-centre mechanism; **i163** (counterweight press closes a
drawer) is a payload-driven transmission with no cargo reorientation; the pen-holder exemplar
is upright insertion into an open cup. Plan structure (lay → slide → crank-erect), rubric
(pose-transformation of the cargo itself), and code structure (bind-time revolute joint with
limits-as-stops + authored CoM bistability + aperture interlock) are all distinct.

## Solution outline (solve.py — teleport = transport only)

- **P0 perception** — settle; read back the *sampled* staging pose from the episode state
  (never hard-coded); assert baseline score ~0.
- **P1 transport (the only teleport)** — the bottle is placed at rest **lying** over the
  guide lane (what a pick-and-lay-down produces); it drops 15 mm and settles. Earns exactly
  the `laid` credit (asserted).
- **P2 slide (real force)** — a velocity-capped horizontal push (|F| <= 4.5 N, body-frame-
  corrected each step) slides it base-first through the slot until the base seats against the
  cradle's end plate. The asserted design contract guarantees this push can never torque the
  resting cradle off its stop.
- **P3 crank (real torque) + release** — a ramped torque servo (|tau| <= 1.2 N·m) raises the
  cradle to ~80°, past loaded over-centre, then releases; gravity tips it onto the 90° joint
  stop and erects the bottle. Success arrives hands-off.
- **P4 persistence** — >= 3.3 simulated seconds with zero wrench; then the whole episode is
  repeated on a second seed (no score prints; the monotone `SIM_GEN_SCORE` stream is from
  the first).

## Embodiment argument (Franka, single arm)

Base placed at the front-right of the apron, e.g. (0.75, −0.25, 0) facing the vault; all
contact sites lie in a ~0.85 m reach envelope, all **outside** the vault:

1. **Lay-down** — the standing Ø6 cm bottle is within the Franka gripper's 8 cm span; side
   grasp at mid-height, lift, rotate the wrist 90°, set down lying in the guide lane
   (counter height 0.47 m). Standard pick-and-reorient.
2. **Slide** — closed-fingertip planar push on the bottle's base end along the lane
   (~4.5 N horizontal at 0.50 m height); the fences form a tolerance funnel into the slot.
   The arm never crosses the wall plane — only the bottle passes the slot.
3. **Crank** — the orange knob (2.4 x 4 cm) rides *outside* the +y side wall on a 0.20 m arm.
   Pinch the knob and sweep a quarter-circle arc (radius 0.20 m, x–z plane at y ≈ +0.20)
   from horizontal-forward to vertical-up; 1.2 N·m at the hinge = 6 N at the knob, trivially
   within payload. Release above the tipping angle — the mechanism finishes the task.

No phase requires reaching inside the chamber, bimanual coordination, or force beyond
fingertip effort.

## Execution-order declaration

Order is REQUIRED and physics-enforced: **(1) lay, (2) slide in, (3) crank past over-centre,
then release.**

- Upright entry (skipping the lay) is blocked by the slot header — smoke check 4, a real shove.
- Cranking first (empty cradle) strands the bottle lying on the chamber floor with the bed
  vertical; the erect credit is guarded by "bottle riding the cradle", so the wrong order also
  earns nothing for the cranking — smoke check 5, real crank + real push.
- Releasing the crank below over-centre drops the loaded cradle back to rest and lays the
  bottle down again — smoke check 6, real crank to ~50° + release.

## Rubric

Latched: 0.15 laid + 0.35 inside (fully through the slot) + 0.40 x max erect fraction while
the bottle rides the cradle; capped at 0.90. Exactly 1.0 iff live success: bottle upright
(<= 15° of vertical), root inside the chamber box, settled, finite. Success is judged on the
**outcome state of the bottle only** — the aperture makes it reachable only through the plan.
Null policy ~0; the seed's strategy ~0.

## Checks (smoke.py — rejection-only battery, 11 checks)

1. settle / no-NaN (bottle at sampled pose, cradle on its stop, score ~0)
2. randomization readback (x span > 40 mm, y span > 60 mm; settled bottle tracks the sample)
3. null policy (240 idle steps, score ~0)
4. SEED strategy executed for real: upright push refused by the header — never inside∧upright
5. wrong order: empty crank (bistable, no erect credit) then insertion strands the bottle
6. under-crank: raise to ~50° < over-centre, release — falls back, partial credit < cap
7. lying-inside fake (constructed): inside but not upright — no success
8. roof-perch fake (constructed): upright but outside — no success
9. rejection audit: success() never observed anywhere in the battery
10. final no-NaN
11. video: frames.npz (> 10 frames)
