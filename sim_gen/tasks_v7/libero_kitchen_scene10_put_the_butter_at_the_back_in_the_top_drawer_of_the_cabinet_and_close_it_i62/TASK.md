# rammer_gallery (i62)

**Seed:** `libero_90/libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it`
(pick the butter → place it at the back of the top drawer → close the drawer).

**Env id:** `simgen.rammer_gallery` — scene-level task, `robot="null"`; all assets
procedural (raw-pxr compound spawners, no external files).

## The task

A slate-gray covered GALLERY (roofed tunnel on a plinth) is fixed to the floor. Its
single MOUTH (90 × 75 mm side aperture at 10 cm apron height) is open from the start;
the far end holds a sunken WELL 25 cm past the mouth, 10 mm below the channel floor —
far beyond direct reach through the aperture. Along the channel axis rides a captive
RAMMER TROLLEY (green knob on top, orange blade hanging to 15 mm above the deck) that
parks OUTBOARD over the apron. Across the ram's path, just outside the mouth, a BLUE
GATE plate on a transverse slide starts parked open and slides sideways to its y=0
stop, fully covering the mouth. Two identical 90 mm sticks lie on the floor, shuffled
per episode — a YELLOW butter and a RED clay brick.

Goal: lay the butter flat on the apron in the loading zone (between the parked blade
and the open gate plane), drag the trolley's green knob rearward so the blade sweeps
the butter through the open gate plane, through the mouth, down the covered channel
until it drops over the lip into the sunken well, then slide the blue gate to its
stop. The brick must stay outside. Success = butter settled flat in the well + gate
fully closed + brick out + everything at rest (live physical readback, no latched
shortcut).

## Strategic difference

Vs the **seed** (open drawer → lower in from above → push shut): nothing here is ever
opened — the receptacle is roofed and permanently open at one side aperture; placement
from above lands on the roof and earns nothing (smoke check 7). "At the back" is not a
placed pose but a **machine-mediated depth**: the well is unreachable except by the
built-in rammer, so the payload's final pose is produced by a tool stroke, not a
carry. "Close it" is a **transverse slide across the ram's path**, not a prismatic
push along the access axis — and the order deposit-before-seal is forced by geometry
(a closed gate blocks butter and blade alike; smoke check 13), not by rubric fiat.

Vs the rest of the corpus read: **i307** (butter_cellar) is a vertical gravity-press
with a one-way pawl ratchet — here there is no press, no ratchet, no gravity-driven
mechanism, and the payload moves horizontally by a driven ram. **i332** (drawbridge
vault) hinges a wall down to bridge into a vault — here nothing rotates and no access
is ever created or removed except the final seal. **i9** (carousel airlock) rotates a
turntable through an airlock — no rotary indexing here. **i3** (stopper vault) is
about extracting a blocking plug — here the gate is the goal-state closure, not an
obstacle to remove (it starts open). **i34** (gumball meter) meters balls through a
column — no gravity feed, no counting. **i237** (tilt_bin_stow) tips a bin to roll the
payload in — the gallery is rigid and fixed; depth comes from the trolley, not from
reorienting the receptacle. **i45** (domino relay) is a chain of falling elements — no
chain reaction here; the two driven mechanisms (ram, gate) are independently and
deliberately actuated, in a geometrically forced order. The color-decoy brick
additionally makes identification load-bearing (smoke check 10).

## Teleport-solution outline (solve.py; passes seeds 0 and 1)

Teleport is used ONLY to transport the free butter through open space onto the apron;
every load-bearing interaction is contact/force physics on the scene's own joints.

- **P0** settle + layout readback asserts (trolley in its randomized outboard zone,
  gate open in its randomized range, blocks outside, score ≤ 0.02).
- **P1** pose-write the butter to 2 cm above the apron loading point (−0.245, 0) and
  let it FALL and settle — a contact landing, endpoint verified outside all geometry.
- **P2** ram: velocity-servo the trolley (+x) by external force on its own prismatic
  slide (`F = gain·(v_des − v)` + stiction floor, capped 3 N; 0.10 m/s cruise,
  0.05 m/s near the well) until the butter crosses into the well band; blade↔butter
  contact does all the work. Clear forces, settle, assert `in_well`.
- **P3** seal: same servo on the gate (−y, capped 2 N) to its y=0 stop; assert
  `gate_closed & success`, score = 1.0.
- **P4** hands-off persistence 3.3 s (10 × 40 steps), success re-asserted every
  window, then `SIM_GEN_SOLVE: SUCCESS`.

(The pod's external-force frame-drag quirk is moot here: both driven bodies ride
prismatic joints and never rotate, so world +x/−y stay themselves.)

## Embodiment argument (single Franka, parallel jaw, OSC)

One plausible base pose: **(−0.10, −0.50, 0)**, facing +y toward the gallery flank.
All grasp/contact points then lie within ~0.20–0.59 m reach, approached from the open
−y half-space, at heights 0.02–0.25 m:

- **Butter (and brick avoidance):** 90 × 45 × 45 mm stick on open floor at the two
  spawn slots (~0.30–0.40 m from base) — pinch across the 45 mm faces (fits the
  parallel jaw), lift, lay flat on the OPEN apron shelf at z ≈ 0.145, approach from
  above/−y; the loading zone is outside the mouth, unobstructed (the trolley parks
  behind it, the gate plane is 5 cm forward of it).
- **Trolley:** hook the gripper over the 60 × 30 × 30 mm GREEN knob (top z ≈ 0.245,
  always outside the roofed span or riding the open slot) and drag +x ~0.39 m at
  constant height — a planar pull well inside the workspace; required force ~O(2 N).
- **Gate:** pinch or side-push the dark-blue grip knob (on the −x face of the plate,
  z ≈ 0.15) and slide −y ~0.11 m — a short lateral stroke on the near side of the
  gallery, nothing overhead.

No step needs a second arm, regrasp-in-flight, or reach into the covered bore: the
arm only ever touches the butter (on open floor/apron), the green knob, and the blue
knob.

## Execution order

`scene.py` (minimal scene, registered) → `solve.py` iterated on the forge to SUCCESS
(seed 0: score trace 0.00 → 0.10 → 0.30 → 0.749 → 1.00, SUCCESS in 53 s; seed 1:
SUCCESS in 50 s after moving the gate plane 5 mm outboard so it can never catch the
gallery front face) → final rubric (latched partials, 0.85 non-success cap, 1.0 iff
live success) → `smoke.py` rejection battery.

## Smoke battery (15 checks)

1. settle/no-NaN: trolley parked outboard, gate open, blocks outside, still
2. score ≤ 0.02, no success at reset
3. butter/brick slot assignment varies (readback, 8 seeded resets)
4. per-slot xy jitter and spawn yaw vary (readback)
5. gate opening y0 and trolley carriage x0 vary (readback)
6. null policy: 240 idle steps → score ≤ 0.02, no success
7. seed strategy (drop from above over the well + slide gate shut): butter lands ON
   THE ROOF bridging the 26 mm slot, never inside → rejected
8. mouth camp (inside, not fully entered) → approach sliver only, ≤ 0.12
9. mid-channel (entered + depth, short of the well) → no success, ≤ 0.58
10. brick smuggled inside with everything else right → rejected at the 0.85 cap
11. gate ajar 20 mm with butter in the well → rejected
12. latched credit survives the butter being yanked back out (score holds, ≥ 0.5)
13. forced order: rammed against a CLOSED gate, the butter stops OUTSIDE and the
    trolley stalls; gate holds → no success, score stays ≤ 0.15
14. rejection audit: success() never True at any judged point
15. final no-NaN
