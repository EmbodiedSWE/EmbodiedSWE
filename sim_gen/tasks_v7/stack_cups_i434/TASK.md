# stack_cups_i434 — shroud posts: dismantle the cup nest (forced LIFO) and cap each size-keyed post

Scene `shroud_posts`, env `simgen.shroud_posts` (robot="null").

## Seed provenance

Derived from **rlbench/stack_cups**: three cup meshes and a franka, no checker; the
implied strategy is transport + **NESTING** — pick each cup up and stack/nest it
into another cup, success being the cups piled together.

## Strategic difference

The seed's goal state is this task's STARTING obstacle, and the seed's whole move
(pile cups together) is worth exactly nothing — the reset state itself IS a nest
and scores 0 (checked). Three graded cups (red 44 mm / yellow 57 mm / blue 70 mm
outer Ø) spawn shrouded concentrically mouth-DOWN over a storage mast; three
free-standing stations each carry a cone-tipped post of a matching size class.
The goal is the inverse of the seed: take the nest APART and redistribute the
cups as covers, one over each post, rim seated on the post's pad.

A solver needs a different plan and different code, not new constants:

- **forced LIFO disassembly** (topological, never declared as a rule): a shrouded
  inner cup is caged — the next cup's wall blocks lateral escape (≈2.5 mm
  effective gap) and lifting it presses its closed top into the next cup's
  ceiling after ~20 mm, entraining the stack. Only the outermost cup is ever
  free, so the nest can only open big → mid → small (proved in smoke with REAL
  force probes, not constructs);
- **size-keyed deployment** (radius interference + pigeonhole): post shaft radii
  15.5/22/28.5 mm against cup mouth radii 19/25.5/32 mm mean a cup physically
  CANNOT enter any post larger than its own (≥3 mm interference — it perches
  high on the cone flank, asserted). A bigger cup DOES drop loosely over a
  smaller post and genuinely caps it — but that locks the smaller cup out of
  every remaining post, so the only end state with ALL THREE posts capped is the
  identity assignment. The rubric never reads identities; geometry forces them;
- **no reorientation anywhere**: cups spawn mouth-down and are carried and
  seated mouth-down (the body +z axis IS the task orientation) — the plan is
  sequencing + matching, not pose work.

Also strategically different from the constructed tasks read while building it:
`stack_cups_i92` (flip each cup upside-down over its color-matched die — mandatory
180° reorientation, a separate enclosed target family, per-pair color identity
judged by the rubric; here nothing is enclosed on the floor, nothing is ever
flipped, and identity is forced by interference geometry, not judged) and
`packing/pen_holder` (insert pens INTO an upright cup — rim-up containment; here
containment of any kind scores nothing).

## Execution order

The DISMANTLING order is physically forced LIFO (big → mid → small) by the
captivity geometry — never declared, only enforced (and stated in describe()).
The deployment order across posts is free; solve.py caps big, mid, small as each
cup comes off the nest.

## Teleport-solution outline (solve.py — teleports for transport only)

- **P0** settle + layout readback; assert score ≈ 0 (rubric-leak guard).
- Per cup, big → mid → small (×3):
  - **UNSHROUD (contact dynamics)**: velocity-regulated vertical force (target
    +0.10 m/s, mg feed-forward, force at the CoM only) lifts the outermost cup
    off the nest while an xy PD holds it on the storage axis through the 3.5 mm
    annulus; asserted: the inner cups never move (>1.5 mm fails), the cup stays
    upright, the lift clears the nest.
  - **CARRY (transport)**: teleport to a hover pose concentric above the matched
    post, mouth ~15 mm above the cone tip (nothing touches); 10 regulated
    zero-velocity hold steps.
  - **SEAT (contact dynamics)**: velocity-regulated descent (−0.06 m/s target,
    xy PD to the post axis); the cone funnels the mouth, the shaft slides in,
    the rim lands on the pad. Wrench DROPPED at the seat; 60 hands-off steps;
    `capped(post)` asserted.
- **Final**: hands off; success(); ≥3.3 s persistence window checked every
  substep; `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` printed at every phase boundary, non-decreasing
(0 → 0.05 → 0.30 → 0.35 → 0.60 → 0.65 → 1.0). Passes on seeds 0 and 1 on the
forge (persistence clean on both).

## Rubric

`success()` = all three posts `capped` NOW; post j capped ⇔ some cup i with

- **mouth-down**: cup body +z within 12° of world UP;
- **rim seated**: mouth center within (−4, +7) mm of the pad top;
- **post through the mouth**: cup axis within `inner_r_i − shaft_r_j + 1 mm` of
  the post axis — the pairwise threshold is NON-POSITIVE for every
  impossible-fit pair (asserted in `__post_init__`), so a too-small cup can
  never read capped on a bigger post, while any pose with the post physically
  inside the mouth passes;
- **settled**: stillness SUSTAINED 20 consecutive substeps (counter in
  post_step).

A mismatch perch is doubly dead: the concentricity threshold is ≤0 AND the rim
hovers ≥15 mm above the band (post_h_j − depth_i, asserted). The upright gate is
redundant armor — a mouth-up cup with a post through its mouth is not
constructible (the post is floor-anchored).

`score()`: latched 0.05 per cup ever carried > 0.15 m off the storage axis +
latched 0.25 per post ever capped (cap 0.90, clamp 0.95); exactly 1.0 iff
success() now. Null policy ≈ 0 (the intact nest sits at storage — nothing
cleared, nothing capped).

## Embodiment argument (single Franka, parallel jaw, OSC)

Only the three cups are ever manipulated (the stations are fixtures):

- **cups**: outer Ø 44/57/70 mm — all inside the ~80 mm jaw span, side
  power-grasp around the wall at mid-height (rigid 12-gon shell). Per cup:
  grasp the exposed outermost cup, lift straight up ~120 mm (clears the 90 mm
  remaining nest), carry level, lower over the post, open, retract vertically.
  No wrist reorientation is ever needed — the cup stays mouth-down for the
  whole task.
- **precision**: the seat needs the cup axis within `inner_r − shaft_r` ≈
  3.5 mm plus the CONE funnel (capture radius ≈ inner_r, 19–32 mm) — an
  off-center descent contacts the cone flank and self-centers; comfortably
  inside closed-loop OSC accuracy, and a bad drop just perches and can be
  re-lifted.
- **finger clearance**: holding a cup at mid-height, the fingertips end
  30–55 mm above the pad at full seat — clear of the pad, and the post is
  inside the cup, never near the fingers.
- **base pose**: Franka base at the ring center (0, 0) facing any station; all
  four stations lie 0.32–0.36 m out at floor height — inside the comfortable
  envelope; the tallest lift (cup top ~0.27 m) is trivial.

## Randomization (readback-verified in smoke)

The four stations are dealt onto the four ring slots by a random permutation,
the whole ring gets a free random phase rotation, and every station gets ±20 mm
xy jitter.

## Check list (smoke.py — 14 checks, rejection only + real-force captivity proofs)

1. settle/no-NaN + layout sanity (nest concentric at storage, stations on
   distinct slots); 2. score ≈ 0 at reset; 3. station-slot permutation readback
   over 8 seeds; 4. ring phase + radial jitter readback, every reset sane;
5. null policy ≈ 0; 6. captivity LIFT probe — a real 1.5×-weight pull on the
   shrouded small cup rises only the ~20 mm ceiling headroom (23 mm measured, big
   cup 0 mm) and jams; 7. captivity SHOVE probe — a real 1.0 N lateral shove
   moves it 7 mm relative to the big cup, clear credit never fires; 8. seed
   strategy — the cups re-nested on the open floor (genuine settled shroud
   stack): nothing capped, no cap credit; 9. radius lockout — small cup dropped
   over the MID post (tightest 3 mm interference) perches +68 mm above the band,
   rejected; 10. pigeonhole — big cup genuinely CAPS the small post (loose drop
   counts; rubric anchor) but the mid cup is then locked out of the big post:
   the wrong assignment can never reach success; 11. settle gate — a capped pair
   with injected velocity is not capped while moving; 12. latched cap credit
   survives teleport-away while capped() drops; 13. rejection audit (success
   never True in the battery); 14. final no-NaN.
