# native_libero_i157 — Domino Relay

Build a correctly **spaced** row of domino tiles between two floor marks, then tip the
first tile so a **self-propagating cascade** carries the fall to the far mark.

- Scene: `domino_relay` (`scene.py`), env `simgen.domino_relay`, robot `"null"`.
- Solve: `python -m simgen_tasks.native_libero_i157.solve --headless [--seed N]`
- Smoke: `python -m simgen_tasks.native_libero_i157.smoke --headless`

## Seed provenance

Derived from `libero/native_libero`
(`RoboVerse/roboverse_pack/tasks/libero/native_libero.py`): the MuJoCo-backed LIBERO
family, where every task is a single-object relocation or fixture articulation —
*pick object X, place it in/on region Y* (or open/close a drawer/appliance) — judged
by static BDDL `on`/`in` predicates over per-object goal regions, solved by OSC
pick-and-place.

## What changed / strategic difference

Kept from the seed: a flat tabletop-scale floor workspace, small graspable rigid
objects, visually marked start/goal regions.

Changed — and why a solver needs a genuinely different plan and code structure:

1. **No object has a goal region of its own.** The green plate is where a *fall* must
   arrive, not where any tile belongs. No sequence of single-object relocations of the
   LIBERO form reaches the goal; a one-tile "place at goal" earns crumbs only (cfg
   asserts one tile can never both anchor at A and reach B — `__post_init__`).
2. **The task content is relative geometry between interchangeable objects.** Six
   identical tiles must be stood at face-to-face gaps inside the topple window
   (18–42 mm) along a per-episode segment A→B whose distance *and* heading are drawn
   fresh — the solver must *plan* the pitch `(D − reach)/(n − 1)` from the draw, not
   look up fixed goal poses.
3. **A triggered, self-propagating dynamic event is the deliverable.** After one tip
   of the first tile, gravity + tile-on-tile contact must carry the fall tile-by-tile
   to the plate hands-off. LIBERO has no analogue: nothing propagates, nothing is
   judged about how an outcome was reached.
4. **The rubric is a chain-topology audit, not object-in-region checks:** lane tiles
   sorted along A→B must ALL be fallen toward B, first foot anchored at A, each head
   propped **on** the next tile (`lap_z_min` — the lapped carpet only a real cascade
   leaves; hand-laid flat tiles rest at ~6 mm and are rejected, cascaded heads rest at
   ~21 mm per solve readback), last head on the plate, everything at rest.

Also distinct from the other corpus tasks I inspected: `base_i88` (see-saw ballast
lift: hinge moment balance, ungraspable cargo, force-servo transfer — no construction
of relative geometry, no propagating cascade) and `robobench/packing/pen_holder`
(pack pens into a holder — pure single-object placement).

## Solution outline (solve.py, passes seeds 0 and 1)

1. **P0** reset + settle; assert score ≈ 0 (no rubric leak).
2. **P1 build** (transport only): teleport each tile from the depot to free space
   3 mm above its lane station (foot of tile 0 at A, even pitch along u, faces normal
   to u); gravity seats it. Score climbs 0.05/tile (latched build credit).
3. **P2 trigger** (applied force): 0.12 N horizontal CoM force along u on tile 0 —
   above the 0.059 N tipping threshold `m·g·t/h`, below the 0.16 N sliding threshold
   `μ·m·g` — re-set every step in the tile's current frame, cut past ~20° tilt
   (critical angle `atan(t/h)` ≈ 11.3°).
4. **P3 cascade** (pure dynamics, hands-off): the chain runs tile-to-tile; settle;
   success() → score 1.0.
5. **P4 persistence**: ≥ 3.3 sim-seconds hands-off with success() held, then
   `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka, parallel-jaw gripper)

- **Tiles** (12 × 36 × 60 mm, 30 g): pinch-grasp across the 12 mm thickness from
  above — well inside the Franka jaw span (80 mm) and payload. A lying tile is
  grasped on its long edges, reoriented in-hand or by regrasp, and set down upright;
  releasing 2–3 mm above the ground and retracting vertically clears the 18–42 mm
  inter-tile gaps because the fingers approach along the row normal (jaw axis
  perpendicular to the lane), never sweeping through a neighbour.
- **Trigger**: a fingertip push at the first tile's upper half toward the plate —
  the demonstrated 0.12 N is far below arm force limits; any gentle poke past ~12°
  works, after which the arm withdraws.
- **Base pose**: base at ≈ (−0.25, −0.10): every depot slot (y ∈ [−0.40, −0.28]) and
  every lane point (x ∈ [0.03, 0.38], y ∈ [−0.16, +0.07]) lies within 0.3–0.65 m
  reach at tabletop height — comfortably inside the Franka workspace.
- **Perception**: tiles orange, start disc red, target plate green — all
  distinguishable by colour; per-episode poses are found by looking.

## Execution order

No strict total order is imposed beyond *build before trigger*: tiles may be stood
in any order (the rubric audits the settled end state). Triggering before the row is
complete simply fails to deliver (the smoke "stops short" probe demonstrates the
outcome). Extra tiles must not be left inside the lane.

## Checks (smoke.py — rejection-only battery, 15 checks)

1. settle/no-NaN: reset finite, tiles flat in depot below the lane, markers at the
   stored (judged) A/B.
2. reset score ≈ 0, no success.
3. randomization readback: A moves, heading and distance vary (every pitch plan
   differs), markers match the judged A/B across 6 seeds.
4. randomization readback: depot slot permutation, xy jitter, free yaw.
5. null policy: score ≈ 0 after 240 idle steps.
6. seed-strategy analog: ONE tile laid head-on-plate → crumbs only; foot provably
   far from A (one tile can never bridge); no success.
7. **flat carpet reject** (the discriminator): tiles hand-laid flat end-to-end A→B
   pass anchor, direction, lane, lap-xy and delivery, but heads rest on the ground
   below `lap_z_min` → rejected; only a real cascade laps.
8. mechanism reject side: 75 mm pitch + solve's own trigger → tile 0 topples
   (non-vacuous) but strikes air; chain dies; tiles 1–2 still standing.
9. direction: correct row triggered backward → tiles topple (non-vacuous) with
   dir_dot ≈ −1 → zero cascade credit, no delivery.
10. stops short: 4-tile correct-pitch row cascades for real but the nearest head
    stops ≫ pad_r from the plate → no delivery.
11. extra tile spoils: tiles 0–4 cascade while tile 5 stands in the lane → all-fallen
    audit fails.
12. latched credit: removing every tile to the depot leaves the score unchanged.
13. settle gate: a freshly teleported moving tile is not settled (windowed
    pose-drift stillness — velocity thresholds alone cannot reject the GPU
    edge-contact phantom limit cycle on lapped tiles).
14. rejection audit: success() never True anywhere in the battery.
15. final no-NaN.
