# flap_postbox (`box_task_replay_i299`)

**Seed:** `box_task/box_task_replay` —
`sim_gen/RoboVerse/roboverse_pack/tasks/box_task/box_task_replay.py`: a bimanual
OpenArm Wuji robot replays a demo trajectory that picks a soda can and a scented
candle off a table and places them down into an OPEN cardboard box.

**This task:** `simgen.flap_postbox` — push both parcels through the one-way flap
door of a SEALED drop box.

## What changed, and why it is strategically different

The seed's whole plan is *pick item, carry it over the open box mouth, lower it in* —
top-down placement into an open container, twice, with no mechanism anywhere. Here the
container relation is inverted into a mechanism the seed never touches:

- The receiving box is **sealed on every face including the top** (walls, jambs,
  lintel, roof). There is no mouth to lower anything into; the seed's plan is
  architecturally impossible on the target container.
- The only entry is a **gravity-shut swinging flap** (letterbox / cat-door) over a
  doorway at porch level. Delivery is a **planar push-through**: slide the parcel
  across the raised porch, press it against the flap until the flap yields inward,
  the parcel crosses the sill under the swinging plate and **falls 70 mm** into the
  interior pit, and the flap falls shut again behind it. The load-bearing interaction
  is a moving-obstacle contact (parcel vs. hinged flap) — a class of interaction the
  seed does not contain at all.
- The seed's strategy is present as a **decoy**: an open-top gray crate stands beside
  the porch, and parcels placed into it (the literal seed end-state: items settled
  inside an open box) score ~0 — verified by a dedicated smoke check.
- No grasping is required anywhere; the winning plan is nonprehensile pushing. A
  solver replaying the seed needs a different plan (push-through-valve delivery, not
  pick-and-lower) and different code structure (force-regulated pushes through a
  yielding door, not free-space place poses).

## Teleport-solution outline (solve.py)

1. **P0** reset (seeded), settle, layout readback, `SIM_GEN_SCORE`.
2. **P1 transport (teleport)**: block parked at a porch corner out of the lane; can
   teleported to the runway start in front of the doorway. All endpoints are on the
   open porch — outside the box, no gate satisfied.
3. **P2 contact**: regulated horizontal force (≤0.9 N, velocity-servoed at 0.25 m/s)
   pushes the can into the flap; the flap swings inward about its real hinge (the
   solve **asserts the flap opened > 15°** during passage), the can crosses the sill,
   free-falls into the pit, settles under pure physics; the flap falls shut by
   gravity. Force is cut at the inner face.
4. **P3** same transport + push for the block.
5. **P4** hands off until the flap hangs shut and everything is still → success().
6. **P5** ≥3.3 s persistence, then `SIM_GEN_SOLVE: SUCCESS`.

Teleports move parcels across the open porch only; every doorway crossing happens
under contact dynamics against the hinged flap. Passed on seeds 0 and 1 (see forge
logs).

## Embodiment argument (single Franka + parallel jaw, OSC)

Plausible base pose: **(-0.05, 0.0, 0)** facing +x; the porch surface (0.20–0.50 m,
top at 0.07 m) and the doorway plane (x = 0.50, push height ~0.10 m) are inside a
comfortable reach envelope, with the whole approach lane open from above.

- **Blue can (52 mm dia × 65 mm)** — nonprehensile push: closed fingertips against
  the barrel at ~0.10 m height, straight-line OSC push along +x through the doorway
  until the fingertips reach the wall plane; the can's leading half is then past the
  inner edge and gravity finishes the delivery. Lateral tolerance is huge (doorway
  130 mm vs. parcel ≤52 mm); the flap yields at ~0.3 N, far below arm capability.
  Fallback grasp exists (52 mm < 80 mm jaw span) but is never required.
- **Yellow block (42 × 42 × 65 mm)** — same push on a flat face (even easier: no
  rolling). If it topples during the push it still fits the doorway lying down
  (42 mm < 100 mm opening) and the push completes identically.
- **White flap** — never needs direct hand contact: the pushed parcel displaces it.
  The hand never has to enter the doorway; the parcel's own length past the
  fingertips provides the through-travel.
- Required precision: end the push with fingertips anywhere near the wall plane,
  parcel roughly centred (±40 mm works). No tolerance is near control noise.

## Execution order

**None required.** Either parcel may be delivered first (both smoke and solve
exercise can-first; the rubric is order-blind by construction).

## Randomization (readback-verified in smoke)

- Bernoulli spawn-slot swap of the two parcels + per-parcel xy jitter (±3 cm) + free
  yaw on the block.
- Decoy crate side (Bernoulli left/right) + xy jitter (±3 cm).
- The postbox fixture itself is FIXED: it anchors the flap's revolute joint, and
  jointed mechanisms are never teleported per episode (drawer_stash lesson);
  randomization lives in the movable objects.

## Rubric

- `success()`: BOTH parcels inside the interior volume (reachable only through the
  flap), flap hanging shut (≤6°), parcels and flap at rest.
- `score()`: 0.05/parcel latched approach progress (from spawn distance; ~0 for
  null) + 0.20/parcel latched delivered-inside + 0.15 latched both-in-with-flap-shut,
  capped 0.85; exactly 1.0 iff `success()`.

## Check list (smoke.py — rejection battery, 14 checks)

1. reset settles finite, parcels on porch, flap shut, still
2. reset score ~0, no success
3. randomization: parcel slot swap + jitter (readback over 8 seeds)
4. randomization: crate side flip + jitter (readback)
5. null policy → score ~0, no success
6. **seed strategy**: both parcels settled inside the OPEN gray crate → score ~0
7. near-miss: can settled against the closed flap → not inside, no success
8. half-through: block resting in the doorway propping the flap (force-constructed)
   → no success
9. flap-ajar: both parcels inside while the flap is held ajar → no success (shut-flap
   clause is load-bearing)
10. latched credit survives removing the parcels (score unchanged, no success)
11. roof cheat: parcel settled on the box roof → not inside, no success
12. one delivered, one on porch → no success (both required)
13. rejection audit: success never True anywhere in the battery
14. final no-NaN
