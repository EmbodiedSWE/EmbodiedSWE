# combo_shutter_pantry — register two shutter plates, slide the pan through the gate into the roofed pantry

Seed: `libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf`
Task id: `libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf_i336`
Env: `simgen.combo_shutter_pantry` (robot="null")

## What the task is

A plinth-mounted pantry cabinet has a low front doorway (170 mm wide x 55 mm tall)
leading into a fully roofed alcove. The doorway is blocked by **two independent
sliding shutter plates** riding in two parallel guide slots just behind the front
wall. Each plate is a "staple": two solid blocks flanking a central notch, joined
by a bridge that passes OVER the doorway header. Each plate spawns with a random
lateral offset (55–125 mm, random sign, independent per plate), so its blocks
cover part of the doorway. Only when **both** plates are slid to their registered
(centered) position do the two notches line up with the doorway and open a clear
through-channel. The frying pan (130 mm disc — too tall to fit through on edge,
too wide to enter anywhere but the gate) must then be slid FLAT along the plinth,
through the combined opening, into the alcove. A decoy saucer lies nearby and must
NOT end up inside. The alcove roof forbids any top-down placement; the doorway
height forbids rolling the pan through on its rim.

This is a **combination-lock alignment puzzle feeding a constrained transport**:
the mechanism credit is for registering each plate, the flagship credit is for the
pan transiting the doorway slab *while both plates are registered* (a state-based
pathway latch that teleport-injection cannot fake).

## Strategic difference — vs the seed

The seed asks the robot to pick up a frying pan and place it on top of an open
cabinet shelf: a single pick-and-place with a free vertical approach. Here there
is **no reachable open shelf**: the goal region is sealed under a roof and behind
a doorway shorter than the pan's diameter, and the doorway itself is blocked by a
two-plate combination shutter. The solution is (1) solve a two-degree-of-freedom
alignment puzzle (two independent randomly-offset plates, each with its own
registered position), then (2) transport the pan by planar sliding through the
opened gate — never by lifting it onto anything. Placement-from-above, the seed's
entire strategy, is physically impossible here.

## Strategic difference — vs siblings

- **i11** (open the drawer, load the pan, re-close): articulated prismatic drawer,
  single mechanism, cargo rides INSIDE the moving part. Here nothing containing
  the cargo moves — two free-sliding plates form a static-when-registered gate,
  and the pan crosses through them on the fixed plinth.
- **i9** (rake the pan out of a roofed cubby onto the burner): extraction FROM a
  roofed lane using a drag/rake, goal is an open destination. Here the direction
  is inverted (insertion INTO the roofed space) and the core difficulty is the
  upstream two-plate combination lock, which i9 has nothing like.
- **i260** (drop-leaf shelf + brace): erect a gravity-braced support surface, then
  place on top. Vertical placement onto a built structure; no aperture, no
  alignment puzzle, no transit gating. Here the surface is fixed and sealed, and
  all credit flows through the gate mechanism.

No sibling combines (a) a multi-element alignment lock, (b) an aperture that
excludes every orientation but one, and (c) a transit latch conditioned on the
lock state.

## Scene (procedural, cabinet frame: origin = plinth top center, +x = push-through)

- Plinth 0.80 x 1.00 x 0.24 m; cabinet pose randomized per seed: xy jitter ±5 cm,
  free yaw (±180°), verified by readback.
- Front wall at x∈[0.06,0.09], doorway |y|<0.085, h=0.055, header above; roofed
  alcove interior x∈[0.09,0.34], |y|<0.13, h=0.10 with 30 mm walls and roof.
- Two guide slots (front slot_x=0.0175, rear slot_x=0.051) formed by three ridges
  + ledge; segmented so the doorway span stays clear; end stops at |y|≈0.47.
- Plates: 13 mm thick staples, notch half-width 85 mm (doorway is 85 mm →
  registered notch passes the 130 mm pan with 5 mm margin per side via the
  combined channel), tabs (13 x 32 mm posts) at y=±0.24 for manipulation,
  mass 0.25 kg, random offsets |a|∈[0.055,0.125] m with independent random signs.
- Pan: disc r=0.065 m, h=0.035 m + stub handle, 0.40 kg, spawns in front of the
  cabinet at x∈[−0.26,−0.16], |y|≤0.08, free yaw. Decoy saucer (r=0.045) spawns
  in a side band on a random side.
- Honesty asserts in `__post_init__` (~30): registered gate passes the pan;
  a single misregistered plate blocks it; on-edge pan taller than the doorway;
  bridge clears the sliding pan; ledge always supports plate blocks; etc.

## Success and score

`success()` = pan fully inside the alcove footprint, flat (upright < 20°),
`transit_latch` set (pan crossed the doorway slab while BOTH plates were
registered within pass_tol), decoy NOT inside, everything settled.

```
score = clamp(0.15*alignF + 0.15*alignR + 0.20*both_simultaneous + 0.20*transit, 0, 0.70)
        → 1.0 on success
```

All four terms are post_step latches (state-based, order-aware). Null policy
scores 0.0.

## Teleport solution (solve.py)

- P0: settle, read back the randomized layout (cab pose, plate offsets, pan pose).
- P1: align the REAR plate — friction-feedforward velocity servo (ff=0.7 N,
  gain 20, cap 3 N, K·dt/m≈0.67) pushing along the cabinet-local y axis until
  |offset| < 4 mm. Score → 0.15.
- P2: same for the FRONT plate. Score → 0.50 (both + simultaneous latches).
- P3: single transport teleport of the pan to the staging point (−0.150, 0, 0.004)
  in cabinet frame, free space, handle trailing; then a bounded position-carrot
  push (spring toward a waypoint advancing at 5 cm/s along the doorway
  centreline, force capped at 5 N / 4.5 N lateral) through the doorway to
  cab-x > 0.210. Transit latch fires in the slab → score 0.70 → success → 1.0.
- P4: hands-off ≥3.3 s persistence, re-verify, `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds 0/1/2: score ladder 0.00 → 0.15 → 0.50 → 1.00 on each,
with provably distinct layouts (yaws −173.0°/−90.9°/+12.2°, differing plate
offsets/signs, pan and decoy positions).

## EMBODIMENT ARGUMENT (single Franka + parallel jaw, OSC)

- **Plates**: each plate has two 13 x 32 mm tab posts whose tops rise to ~0.167 m
  above the plinth top ≈ 0.407 m in world — comfortably inside an 80 mm parallel
  jaw's pinch envelope, or pushable on their flat faces. Registering a plate is a
  pure lateral slide of ≤125 mm at constant height along an unobstructed span
  (the bridge passes over the header, tabs stay outside the wall footprint).
- **Pan**: moved entirely by planar pushing on its rim/side wall at plinth height
  (0.24–0.275 m world) — no grasp of the pan is ever required; the solve's
  bounded-force planar push is exactly what a stiff OSC end-effector push
  achieves. The staging teleport is plain transport across open floor.
- **Base pose**: a Franka based ~0.55 m in front of the gate (−x side), facing +x,
  reaches the pan spawn band (0.16–0.26 m in front of the wall), both tab travel
  spans (y = ±0.24 ± 0.125), and the push line through the doorway, all within
  a ~0.85 m radius workspace at heights 0.24–0.41 m. No reorientation of the pan
  and no reach into the roofed alcove is needed (the pan is released before the
  doorway and pushed through; final position is reached by the push, not by
  placing inside).

## Execution-order declaration

Plate order is free (front/rear interchangeable). **Both plates registered
strictly before the pan transits** — this is physically forced (either
misregistered plate's block spans the doorway) and rubric-enforced (transit latch
requires both registered at the moment of crossing).

## Smoke battery (16 rejection-only checks)

1. settle: finite state, plates seated in slots, pan flat.
2. fresh reset: score ≤0.02, no success.
3. cabinet xy+yaw randomization readback across seeds.
4. plate offsets vary, both signs occur, in-band.
5. pan xy varies; decoy occurs on both sides.
6. null policy 240 steps: score ≤0.02.
7. seed-strategy end state (pan placed ON the cabinet roof): rejected ~0.
8. blocked gate: misregistered plates, honest 400-step push — pan moves ≥3 cm
   but never enters the slab; plates not walked open; score ≤0.02.
9. single plate registered: still blocked; score = align credit only.
10. straddle: both registered, push stopped mid-slab — transit latches (honest)
    but no success; score ≤0.70.
11. teleport-injection into the alcove: pan_inside true but transit never
    latched → score ≤0.02.
12. wrong object: decoy inside first, pan honestly delivered → success False.
13. flipped pan pushed in → upright rejects.
14. latched credit: register both, then shove a plate open — score unchanged.
15. `ever_success` audit: no probe ever reached success.
16. final no-NaN.

Frames recorded to `frames.npz`.
