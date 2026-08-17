# track_spoon_i335 — Sieve Sorter: feed the batch through the size classifier (`simgen.sieve_sorter`)

## Seed provenance

Derived from **pick_place/track_spoon**
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_spoon.py`): a Stage-3
trajectory-tracking task — the spoon starts ALREADY RIGIDLY GRASPED in the Franka's
closed gripper, and reward is dense per-step position+rotation tracking of a
prescribed free-space waypoint path toward a basket. The seed's whole strategy is
*transport fidelity of ONE held payload along given waypoints*; where the payload
ends up has no physical consequence, and no property of the payload ever matters.

## What changed, and why it is strategically different

Nothing of the seed survives but "objects must be moved somewhere". The judged skill
is inverted from transport fidelity to **understanding and operating a passive
classifying MACHINE on a multi-object batch**:

- **The goal cannot be reached by placement at all.** Every bead must rest in the
  LOWER bin and every marble in the roofed END bin — but the lower bin's only
  ceiling IS the sieve, and the end bin is roofed (the roof starts a marble-diameter
  upstream of the divider sill, tilts so strays roll back onto the rails, and its
  lip slot is narrower than a marble — all cfg-asserted). Every route into either
  bin runs THROUGH the machine; smoke proves a vertical drop over the end bin lands
  on the roof and can only enter by riding the rails.
- **One action, two outcomes — the machine decides, not the solver.** The sieve is a
  bank of knife-edge diamond rails (rolled 45°, pitched 12° downhill) with 21 mm
  waist gaps: a 13 mm bead dropped anywhere on it falls straight through into the
  lower bin; a 34 mm marble cannot pass, seats on the 45° faces of a V-groove
  (contact 1.5 mm up the faces — cfg-asserted), rolls downhill, clears the sill by
  ~8.5 mm and drops off the rail ends into the end bin. The solver's insight is that
  the SAME feed zone serves BOTH destinations — the classification is a property of
  the object, not of the trajectory. The seed's strategy (carry objects along a path
  and set them down) scores ~0.
- **A batch, not a payload.** Per-episode random bead count (2–3), marble count
  (1–2), slot permutation + jitter, tray xy + free yaw, apparatus xy + yaw — the
  rubric is judged per object in the LIVE apparatus frame, and score is latched per
  object, so partial work earns exactly its fraction.
- **Versus the read corpus:** `track_spoon_i160` (same seed) is hidden-information
  discovery through a jointed two-pan balance; `track_spoon_i265` (same seed) is
  quantitative statics prediction on a knife edge. Here there is nothing hidden and
  nothing to compute — the skill is recognizing and exploiting a passive sorting
  mechanism under batch/count randomization. No corpus task judges routing a mixed
  multi-object batch through a size-selective machine into two mutually
  placement-denied destinations.

## Apparatus (fully procedural, boxes + spheres only — no meshes, no joints)

ONE heavy dynamic compound (25 kg, teleported at reset; explicit MassAPI mass+CoM —
custom spawn funcs ignore cfg mass_props, and MassAPI-only mass leaves the CoM at
the origin): 0.66 × 0.30 × 0.02 base slab; 5 diamond rails (14 mm square section
rolled 45°) pitched 12° toward +x spanning local x −0.298..0.130, gaps 21 mm between
rails and to each capped side wall; LOWER BIN under the rails (blue floor tilted 3°
back toward the upstream wall — settled beads gravity-pin and stop rolling); red
divider SILL at x = 0.10 (top 0.118); roofed END BIN x 0.106..0.305 (green floor
tilted 4° downstream to pin marbles against the end wall); red ROOF (14°,
upstream-low) from x = 0.055; 45° ridge caps on every wall top (nothing parks on a
wall). Every classifying quantity (gap vs diameters, groove seat height, sill
clearance, roof lip slot, bin z-caps vs sill rests, feed-zone openness, slot/tray/
depot clearances, jaw fit) is DERIVED in cfg from part dimensions — one source of
truth shared by the spawner, the rubric and the solver — and guarded by 28
`__post_init__` honesty asserts. Tray: open 0.21 m dynamic box, 5 slots on a cross.
Beads: yellow 13 mm / 10 g spheres; marbles: blue 34 mm / 60 g spheres.

**Randomization (readback-verified):** apparatus xy ±3 cm + yaw ±15°, tray xy ±4 cm
+ FREE yaw, bead count 2–3, marble count 1–2, slot permutation + ±8 mm jitter;
absent objects park in a far depot.

## Rubric

Judged on the settled state in the LIVE apparatus body frame:

- `success()` = every present bead inside the lower-bin box ∧ every present marble
  inside the end-bin box (boxes admit wall-resting objects and exclude sill/sieve/
  slab-rim rests via z-caps and the y bound — cfg-asserted) ∧ apparatus upright
  (≤ 5°) ∧ everything at rest ("at rest" = velocity thresholds set above the GPU
  phantom-velocity band PLUS a pose-stillness window: every body's pose frozen over
  the last 0.25 s — falling, rolling and freshly-teleported states all break it).
- `score()` (monotonic, latched per object): 0.55 · (fraction of present objects
  that ever rested correctly binned and quiet) + 0.15 · success() ever held, capped
  at 0.70; exactly 1.0 iff success() now. Null policy ~0; the seed's strategy
  (transport + set-down) ~0.

## Solution outline (solve.py — the legitimacy certificate)

Teleport = TRANSPORT ONLY, always ending in a non-contact hover; the entire SORTING
goes through contact dynamics:

- **P0 SETTLE** — reset scatters apparatus/tray/batch; everything rests. Score ~0
  (asserted).
- **P1 FEED MARBLES** — each present marble is hovered 20 mm above a V-groove on the
  open upstream stretch and RELEASED; the machine seats it, rolls it over the sill
  and drops it into the roofed end bin (per-object latch polled by readback).
- **P2 FEED BEADS** — each present bead is hovered 30 mm above the SAME groove line
  and RELEASED; it falls straight through the 21 mm gap into the lower bin. Same
  feed zone, opposite verdict — the classification is the machine's.
- **P3 PERSISTENCE** — hands off ≥ 3.6 simulated seconds; verdict only if success
  still holds.

`SIM_GEN_SCORE` printed at each phase boundary, non-decreasing (seed 0: 0 → 0.275 →
0.55 → 1.0; seed 1: 0 → 0.183 → 0.55 → 1.0). Verified on the forge for seeds **0**
(2 beads + 2 marbles) and **1** (2 beads + 1 marble), rc=0 — different apparatus
poses, tray yaws, counts and slots.

## Embodiment argument (Franka, one base pose)

Base on the floor at (0.42, −0.10) facing the gap between tray and machine: the tray
centre is at (0.25 ± 0.04, −0.40 ± 0.04) (reach 0.30–0.40 m) and the apparatus feed
zone — the open upstream stretch, local x −0.29..0.04 — lies within 0.15–0.45 m of
the base at hover heights 0.21–0.28 m, all inside the 0.855 m envelope. Per-object
contact strategy:

- **Marble (34 mm, 60 g):** standard parallel-jaw pinch (80 mm stroke — cfg-asserted
  fit) from above the open tray; carry to a hover just above the rails over any
  groove line in the feed zone and release — exactly the motion solve certifies.
  The walls flanking the feed zone are open-topped (no roof upstream of x = 0.055,
  cfg-asserted ≥ 25 cm of open span), and a feed hover fits inside the wall height.
- **Bead (13 mm, 10 g):** same pinch, same hover-and-release over a gap; the
  gap-centred drop has ≥ 3 mm lateral slack per side (cfg-asserted), comfortably
  inside closed-loop arm precision.
- **The machine and the tray:** never need to be touched; the machine performs the
  classification, transport is the only manipulation.

## Execution-order declaration

NO required execution order. Objects may be fed in any order (marbles and beads use
the same feed zone and never interact — beads leave the sieve plane instantly); the
rubric judges only the settled terminal state plus per-object latched credit.
Nothing procedural is checked.

## Checks (smoke.py — rejection battery, recorded, `SIM_GEN_SMOKE: ALL PASS 14/14`)

1. Settle/no-NaN: apparatus upright and settled, batch resting in the tray, score
   ~0, no success.
2. Authored-mass readback: apparatus 25 kg, tray 1.2 kg, bead 10 g, marble 60 g via
   get_masses().
3. Randomization readback (8 seeds): apparatus xy + yaw, tray xy + free yaw, bead
   and marble counts, slot assignment all vary.
4. Null policy (240 steps): score ~0, no success.
5. Seed-strategy analog: every object carried and SET DOWN on the floor around the
   machine — score ~0, no success.
6. Wrong-bin swap: a marble settled in the LOWER bin and a bead in the END bin —
   both genuinely in-a-bin and quiet, but correct_bin is per-type: no latch, ~0.
7. Roof denial: a marble dropped vertically over the end bin lands ON the roof
   (readback above the bin cap), rolls back onto the open sieve, and its first
   sub-cap sample is already past the sill — entry is only THROUGH the machine.
8. Settle gate: a bead mid-fall INSIDE the lower-bin box reads geometrically
   contained but unsettled — success refused at that judged instant.
9. Sieve passes beads: one fed bead falls through and latches exactly 0.55/n_present
   — partial credit is honest, no success with the batch unfinished.
10. Sieve blocks marbles: the marble fed over the SAME groove line never enters the
    lower-bin box at any polled instant, rides over the sill and latches in the END
    bin — same feed zone, opposite verdict; still no success.
11. Latched credit survives removing the delivered marble to the depot.
12. Slab-rim near-miss: a bead resting on the machine's own slab rim outside the
    side wall never reads in-bin and never latches.
13. Rejection audit: success() never True at any judged point in the battery.
14. Final no-NaN; frames.npz saved.
