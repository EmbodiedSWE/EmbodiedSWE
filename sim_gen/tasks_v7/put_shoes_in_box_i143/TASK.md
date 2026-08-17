# put_shoes_in_box_i143 — LiddedShoeBoxScene (`simgen.shoe_box_lid`)

Pack a PAIR of heel-counter shoes into a box that is one size too small for naive
packing, under a removable PLUG LID: take the lid OFF (it starts seated, sealing the
mouth), lay the shoes flat HEEL-TO-TOE in opposite directions (the only nesting that
fits the interior), and reseat the lid FLUSH — the lid is the physical verifier: it
only closes over a correctly packed box.

## Seed provenance

- **Seed task**: `rlbench/put_shoes_in_box` (RoboVerse
  `roboverse_pack/tasks/rlbench/put_shoes_in_box.py`) — "put the shoes in the box".
  An always-OPEN box (articulated base whose lid plays no role in the trajectory) and
  two free shoes; the demo picks each shoe, carries it over the mouth and releases.
  No closure, no fit constraint, no ordering, no relation between the two shoes: any
  pose inside counts, the box is never operated.

## What the task is

A TAN KINEMATIC BOX (interior 165 × 132 × 58 mm, 12 mm walls) stands on the floor,
CLOSED by a free-body STEEL-BLUE LID (250 g): a 205 × 172 × 12 mm plate with a
157 × 124 × 8 mm plug underneath (4 mm per-side clearance in the mouth) and a
56 × 16 × 24 mm handle ridge on top. Per episode the box center jitters ±50 mm and
its yaw is FREE (±180° — the packing axis must be read, not memorized); the lid
starts seated with micro jitter; two CRIMSON SHOES scatter on the floor with free
yaw in a 0.20–0.55 m annulus (keep-out resampled off the box footprint, pairwise
separated, deterministic fallback). Each shoe is a narrow sole (150 × 32 × 18 mm)
carrying a WIDE HEEL COUNTER (50 × 72 × 26 mm) at the rear — flat it is 44 mm tall,
150 g.

Goal: both shoes fully inside the cavity below the rim, the lid seated flush
handle-up (plug engaged: centered ≤ 6 mm, plate height ≤ 2 mm of nominal, upright
≤ 5°), everything at rest.

## Why strategically different

- **vs. the seed**: the seed's whole skill is *transport-and-release into an open
  container*. Here that plan is constructed verbatim in smoke 4 — both shoes carried
  over the box and released — and delivers NOTHING: they land on the CLOSED lid and
  slide off (score 0). Three new obligations replace it: (1) OPERATE the container
  (remove the lid first — the mouth is sealed; smoke 5 presses a shoe down onto the
  seated lid with 5 N ≈ 3.4× its weight and it never enters, so the open-first order
  is enforced by geometry, not rubric fiat); (2) REASON about headings — the
  interior (132 mm) admits both shoes flat only ANTIPARALLEL heel-to-toe (needs
  124 mm; same-heading needs 2 × 72 = 144 mm and physically jams — smoke 6; stacked
  stands 62 mm proud of the 50 mm plug line — smoke 7; the fit inequalities are
  asserted in `__post_init__`); (3) CLOSE AND VERIFY — the plug lid only seats flush
  when nothing pokes above the plug line, so a mis-packed box leaves it ≥ 8 mm proud.
- **The lid is a physical verifier, not bookkeeping**: success' closure clause is
  live — teleport the lid off a constructed success and success collapses while the
  latched credit survives (smoke 13); a lid ajar 15 mm, upside-down, or perched on a
  rim-crossing shoe is refused (smoke 8–10).
- **vs. corpus tasks read**: `pour_water_i7` (ramp force-chain equilibrium),
  `pen_holder` (upright inserts into an open cup) and the robobench packing/pouring
  suites share no mechanic — none has a sealed-then-resealed container whose lid
  doubles as the packing verifier, nor an interference-fit orientation puzzle
  between two identical objects. Drawer tasks (open-insert-close) operate a captive
  prismatic joint and impose no fit/heading constraint between the inserted objects;
  here the closure is a free body that must be parked flush, and the two shoes
  constrain EACH OTHER.

A solver therefore needs a different PLAN each episode (read the box axis, clear the
lid, flip one shoe 180°, nest, reseat flush) and different CODE STRUCTURE (box-frame
containment predicate over shoe body points, seated-lid pose predicate, latched
open/packed credit) — not a waypoint pick-and-drop.

## Solution outline (solve.py = the legitimacy certificate)

1. **P0** settle 1 s; READBACK the layout (box xy + yaw, shoe spawns); score 0.
2. **P1** (teleport = transport only): one root-state write carries the lid to a
   free-space hover over a dynamically-chosen clear floor spot (≥ 0.35 m from the
   box, ≥ 0.22 m from each shoe); hands-off it falls onto its plug face. Open latch
   → score 0.15.
3. **P2** shoe A teleported to a hover 5 mm above the OPEN mouth at box-local
   y = −26 mm, heading = box yaw; falls 60 mm onto the box floor; then butted flush
   against its wall with escalating CoM force pulses (0.8 → 4 N max, cleared after
   each; the force frame is PROBED from measured displacement and flipped if the
   pod's rotation-drag quirk shows). Packed latch → score 0.40.
4. **P3** shoe B likewise at y = +27 mm with heading FLIPPED 180° — the antiparallel
   nest, each heel beside the other's sole → score 0.65.
5. **P4** lid teleported to a centered hover, plug underside 2 mm above the rim,
   box-matched heading; released: the plug funnels in and the plate settles flush
   (retry with ±1.5 mm jitter if it perches) → success, score 1.0.
6. **P5** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

No judged clause is ever written: the lid's landings, both shoe rests and the flush
seat are all outcomes of gravity, contact and friction. Monotone `SIM_GEN_SCORE`:
0.0000 → 0.1500 → 0.4000 → 0.6500 → 1.0000 → 1.0000.

## Rubric

- 0 → 0.65 latched shaping (never decays): 0.15 lid ever fully CLEAR of the mouth
  + 0.25 per shoe ever inside-and-settled (cap 0.65). Doing nothing scores ~0 (the
  lid starts seated).
- 1.00: success — live: both shoes' three body points (toe end, heel end, heel top)
  inside the interior volume below the rim in the BOX FRAME; lid seated flush
  (xy ≤ 6 mm, z ≤ 2 mm, upright ≤ 5°, settled); everything settled and finite.

Unfakeable without the mechanic: nothing enters a closed box (smoke 4–5), no packing
except the flat antiparallel nesting leaves the lid seatable (smoke 6–8), and the
closure is judged live (smoke 13).

## Franka embodiment (single arm, parallel jaw 80 mm, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m). The
box center sits at 0.34 ± 0.07 m; shoes spawn in the 0.20–0.55 m annulus around the
base; the lid's set-aside spot is chosen on a 0.45–0.62 m ring. All manipulation is
below 0.12 m height and inside the envelope; joint 1 spans ±166° so the annulus is
covered (elbow-flip for the rear wedge).

- **Lid** (250 g): grasp the HANDLE RIDGE — 16 mm wide, ≪ 80 mm jaw, standing 76 mm
  above ground when seated (fingers never scrape the plate). Removal is a straight
  lift (plug clearance 4 mm/side, no latch force); reseating is a hover-and-release
  2 mm above the rim with a wrist yaw matched to the box heading read from the
  scene — the plug's 4 mm funnel does the fine centering, and the ±6 mm/±5°
  tolerances are coarse by placement standards.
- **Shoes** (150 g): pinch the HEEL COUNTER across its 50 mm length (fingertips at
  ~20–40 mm height — well off the ground) or the 32 mm sole. Insertion is a
  release from a hover above the open mouth: the shoe falls 60 mm and lands flat;
  the wall-flush nudge is a fingertip side-push (≤ 4 N). Shoe B needs a 180° wrist
  yaw before release — exactly the reasoning step the task tests. The drop corridor
  after flushing A is ≥ 3 mm per side; walls guide the fall.
- **Ordering** (declared): lid off FIRST (geometry-enforced — the mouth is sealed);
  shoes in either order; lid back LAST. Between the shoes there is no required
  order.
- **Forces**: heaviest lift 250 g; the box is a kinematic fixture, so incidental
  contact cannot move the goal frame.

## Validation evidence (forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (box yaw −36.3°): SUCCESS, scores
  0.0000/0.1500/0.4000/0.6500/1.0000/1.0000, lid reseats to < 1 mm error, 18.3 s wall.
- `solve --seed 1` (box yaw +140.5°): SUCCESS, same monotone scores, 18.2 s.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 15/15**, frames.npz saved:
  1. settle/no-NaN: lid READBACK seated, shoes on the floor, score 0
  2. randomization readback: box yaw (Δ145°), box xy (Δ49 mm), shoe_a spawn
     (Δ1.06 m) all differ across seeds
  3. null policy: 240 idle steps, score 0, no success
  4. SEED STRATEGY: both shoes carried over the box and released onto the CLOSED
     lid — nothing enters, lid stays seated, score 0
  5. CLOSED-LID PRESS: a shoe falls 30 mm to contact on the plate, then pressed
     with 5 N (~3.4× its weight) — never enters, lid holds seat: open-first is
     geometric
  6. PARALLEL packing: same-heading heels (144 mm) jam the 132 mm interior; the
     second shoe perches tilted, lid rides 17 mm proud — not seated
  7. STACKED packing: top shoe above the plug line, lid 9 mm proud — not seated
  8. RIM near-miss: shoe flat across the rim is not inside; the dropped lid perches
     22 mm proud — never counted seated while the shoe crosses the mouth
  9. LID AJAR: proper pack, lid dropped 15 mm off-axis catches the rim — rejected
     (xy clause), score stays at the latched 0.65
  10. UPSIDE-DOWN lid: upright clause refuses regardless of resting height
  11. lid-off only: score 0.15, no success
  12. packed but box open: score 0.65 < 0.9, no success — closure is required
  13. REMOVE-LID: constructed success is live (1.0); lid teleported away → success
      COLLAPSES, latched score survives at 0.65
  14. rejection audit: success never fired during checks 4–12
  15. video frames.npz saved (270 frames)

## Files

- `scene.py` — LiddedShoeBoxScene + kinematic box / plug-lid / heel-counter-shoe
  compound spawners + rubric; registers `simgen.shoe_box_lid`.
- `solve.py` — teleport-transport + gravity pack/seat certificate (`--seed N`).
- `smoke.py` — 15-check rejection battery + video.

Run (forge):
`python -u -m simgen_tasks.put_shoes_in_box_i143.solve --headless [--seed N]`
`python -u -m simgen_tasks.put_shoes_in_box_i143.smoke --headless`
