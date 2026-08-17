# pick_single_egad_i216 — go/no-go gauge sort

Sort four bars with a physical go/no-go gauge: every bar THIN enough for this
episode's slot gap must be dropped through the slot — the only opening of an
otherwise closed bin — so it rests inside; every bar TOO THICK for the gap must
be laid in the open reject tray. Success = all four bars settled in their
gauge-correct destinations.

## Seed provenance

Seed task: `maniskill/pick_single_egad` (RoboVerse
`roboverse_pack/tasks/maniskill/pick_single_egad.py`): grasp ONE odd-shaped EGAD
object on an open table and lift it — judged by a `PositionShiftChecker` on the
picked object itself (`distance=0.075, axis="z"`). One object, no destination,
no decision; the judged end state is the object held aloft.

## What changed, and why it is strategically different

- **From one lift to a classification + routing problem.** Nothing here is
  judged by lifting. FOUR bars — identical color and length, differing only in
  cross-section (16x40, 30x48, 44x56, 58x66 mm) — must each be CLASSIFIED
  against a physical gauge (a slot channel whose gap is re-sampled every
  episode from three classes: ~23 / 37 / 51 mm) and ROUTED to one of two
  destinations. Which of the middle bars fits flips with the sampled gap, so a
  memorized routing fails; the plan is measure-then-route, and its code shape
  is a per-object decision loop, not a single pick.
- **The aperture is load-bearing.** The bin is closed everywhere except the
  slot (four 140 mm walls + two roof plates flush with the wall tops), so the
  in-bin readout is reachable only by a real transit through the gauge. A
  rejected bar is refused *mechanically* (its minimum dimension exceeds the gap
  by >= 5 mm), not by a soft rubric term.
- **Both routes must be used.** Dumping everything in the tray fails on the
  fitting bars; presenting everything at the slot strands the thick ones.
- **Vs the corpus references examined:** the pen-holder exemplar fills ONE open
  cup with all pens tip-up (single destination, orientation rubric, no
  classification); `poke_cube_i83` arms a latch and fires a hands-off domino
  cascade (causal relay, no routing). Neither has a per-episode-randomized
  physical gauge that flips a routing decision.
- **The seed's end state has no analog here.** A held-aloft bar is not a
  settled outcome; the nearest expressible state — the bar at rest ELEVATED on
  the bin roof — is explicitly rejected by the below-sill gate (smoke check 6).

## Scene (all procedural, no external assets)

- **Station** (kinematic compound): closed gray bin, inner 200 x 140 mm,
  walls 140 mm tall; open red tray, inner 200 x 200 mm, 32 mm lip, side by
  side. Anchor xy +/- 3 cm, yaw +/- 20 deg per episode.
- **Gauge plates** (two separate kinematic boxes): the bin roof; repositioned
  in `reset()` so the slot between them opens exactly the sampled gap W
  (spawn-time geometry cannot change per episode — randomized geometry is done
  by kinematic-body repositioning, verified by world-pose readback in smoke).
- **Bars** `bar_a..bar_d` (dynamic, 60 g): 95 mm long, thickness x width
  16x40 / 30x48 / 44x56 / 58x66 mm, all the same orange — identity is SIZE.
  Scattered lying flat at shuffled slots with jitter + free yaw
  (overlap-rejected via segment-distance test).

Geometry honesty is asserted in `__post_init__`: t < w < l per bar (min
dimension = thickness, so no orientation sneaks a thick bar through); every
sampled gap >= 5 mm from every thickness ("fits" is crisp); bar_a always fits,
bar_d never; plates cover the walls at the narrowest gap; a bar (95 mm)
submerges fully below the sill (140 mm); all rubric gates clear the geometry;
tray capacity and jaw span hold.

## Rubric

- `fits()`: thickness < this episode's W (>= 5 mm margin either way).
- `in_bin()`: bar center inside the inner walls AND below sill - 20 mm — only
  reachable through the slot.
- `in_tray()`: center within the lip, down at floor level (a bar on the lip or
  on the bin roof fails).
- `correct()` = routed to the gauge-correct destination AND settled;
  `success()` = all four correct; `score()` = 0.9 x correct fraction, 1.0 iff
  success. Null policy ~0. Credit is state-based and non-evaporating (walled
  destinations).

## Teleport solution (solve.py)

Teleports are TRANSPORT ONLY; the load-bearing interaction — the aperture
transit — is pure contact dynamics:

- **P0** settle, read the episode (gap class, fits, station pose).
- **P1** each thick bar teleported to 15 mm above the tray floor, lying flat,
  released to settle inside the lip.
- **P2** each fitting bar teleported to a nose-down hover with its tip 5 mm
  ABOVE the roof plates, centered on the slot, thin side across the gap, and
  RELEASED: it free-falls, threads the gauge, drops 140 mm, topples and
  settles inside the closed bin. A bounce astray is re-grasped (teleport above
  the slot again) and re-dropped, retries with a small guided downward
  velocity. No write ever puts a bar below the sill or inside a destination.
- **P3/P4** settle, `success()`, then >= 3.3 s hands-off persistence before
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing.
Verified on the forge: seed 0 (gap 23.6 mm, 1 fitting bar) and seed 3 (gap
50.3 mm, 3 fitting bars) — the routing decision demonstrably flips between the
two runs. rc=0, ~17 s each.

## Embodiment argument (Franka, one fixed base pose)

Base at the world origin facing +x (station anchor ~0.42 m, all interaction
points within ~0.63 m reach, approached from above — no overhead obstruction):

- **Grasp:** every bar is grasped across its THICKNESS (16-58 mm), all under
  the 78 mm usable jaw opening with >= 15 mm margin (asserted in cfg). Bars
  rest lying flat on open floor with >= 6 mm scatter clearance — top-down
  pinch on the exposed long faces.
- **Gauge/insert:** hold the bar nose-down over the slot (hand ~0.25 m above
  the plates' 0.148 m top — free air), thin side across the gap, and release;
  the fitting classes clear the slot by >= 5 mm total at the tightest
  class-boundary pairing, and the slot is 224 mm long so x-placement tolerance
  is generous. Gauging can be done visually (bar cross-sections differ by
  >= 12 mm in thickness) or by touch-testing against the slot — the task text
  allows trying.
- **Tray place:** lower and release anywhere within the 200 x 200 mm tray over
  its 32 mm lip — a trivial place.
- **Order:** bars may be sorted in ANY order (no execution-order requirement);
  destinations are spatially separate, nothing occludes anything.

## Files

- `scene.py` — cfg (+honesty asserts), spawner, scene, rubric; registers
  `simgen.gauge_sort` with `robot="null"`.
- `solve.py` — teleport-transport solution, phase scores, persistence hold.
- `smoke.py` — 15-check rejection battery (below), records `frames.npz`.

## Smoke battery (15 checks, all passing on the forge)

1. settle: finite, all four bars low on the open floor, in neither destination
2. settle: score ~0, no success
3. randomization: slot gap READBACK from plate world poses matches sampled W
   (max err 0.00 mm), plates at the sill, >= 2 fit-count classes over 12 seeds
   (saw all three: 1/2/3 bars fit)
4. randomization: anchor + yaw + scatter permutation + bar poses vary
5. null policy: 300 idle steps -> score ~0
6. seed strategy (lift analog): bar at rest ELEVATED on the bin roof rejected
   by the below-sill gate
7. gauge refuses oversize: smallest too-thick bar dropped onto the slot with
   downward velocity at 3 spots demonstrably descends onto the gauge
   (actuator-moved assert) yet is NEVER in_bin at any sampled step
8. gauge admits fitting: bar_a dropped identically genuinely threads the slot
   into the bin (mechanism proven both ways; still no success)
9. roof rest: fitting bar lying across the slot passes the xy gates but is
   rejected by the z gate
10. wall-hug + beside-tray rests rejected by the xy gates
11. dump-all-in-tray: only genuinely thick bars credited, fitting bars in the
    tray earn nothing, no success
12. near-complete (3/4 routed, one fitting bar on the floor): score 0.675, no
    success — the last routing is load-bearing
13. fly-through: a bar inside the bin volume at 1.2 m/s earns nothing while
    moving (settle gate)
14. rejection audit: success() never True anywhere in the battery
15. final no-NaN
