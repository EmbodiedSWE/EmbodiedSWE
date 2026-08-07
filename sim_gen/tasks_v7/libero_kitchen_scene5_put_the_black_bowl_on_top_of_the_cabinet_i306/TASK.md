# hatch_shelf — close the hatch to CREATE the cabinet top, then put the BLACK bowl on it

**Task id:** `libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i306`
**Scene:** `hatch_shelf` (`register_env("simgen", ...)` → `simgen.hatch_shelf`, robot="null")

## Seed provenance

`libero_90/libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet`
(`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet.py`).
Seed strategy: pick the akita black bowl off the table, place it on the white
cabinet's FIXED flat top; success = bowl inside a bbox above the cabinet. One
pick-and-place onto a surface that exists from t=0.

## What changed, and why it is strategically different

The destination surface is REMOVED from the initial scene. The cabinet's top face is
a hinged, bistable HATCH LID standing open ~104-118 deg (leaning past vertical toward
the robot against its hinge stop); under it the cabinet is an open-mouthed 26 cm pit.
The seed's entire plan — carry the bowl to the top of the cabinet and set it down —
is a live trap here: a bowl released "on top of the cabinet" falls through the mouth
into the cavity and scores nothing (smoke check 6 constructs exactly this). A second
discrimination the seed never asks for: an identical-geometry WHITE decoy bowl that
must not be used.

A solver therefore needs a different PLAN, not different numbers:
1. actuate a mechanism (drive the standing lid over-centre so gravity drops it flat)
   to CONSTRUCT the destination surface;
2. only then pick the BLACK bowl (not the white one) and place it upright, centered
   on the lid it just closed.

Different code structure too: bind-time revolute joint with a gravity-bistable rest
pair (open stop / closed stop), an over-centre actuation phase, latched
surface-creation credit, and object-discrimination + orientation clauses — none of
which exist in the seed (fixed articulated-asset cabinet, single bbox check).

## Teleport-solution outline (solve.py)

- P0: reset (seed via `env.reset(seed=...)` after build), settle; assert lid fully
  open (> 95 deg), bowls on the floor, score ~0.
- P1 (contact/joint dynamics — the core interaction): regulated +x force at the
  lid's CoM (velocity-servoed on the hinge rate, ≤ 5 N) drives the lid about its
  REAL hinge past over-centre; force is CUT at 25 deg and the fall + slam onto the
  closed stop are pure gravity/joint-limit physics. Asserts the lid settled closed.
- P2 (transport ONLY): one pose write moves the black bowl across free space to a
  hover 5 cm ABOVE the closed lid's centre — touching nothing and satisfying no gate
  (the on-lid z band tops out 2.5 cm over the lid; the solve asserts the hover is
  NOT on-lid and NOT success). The white bowl is never touched. The lid is NEVER
  teleported outside reset re-pose.
- P3 (contact dynamics): the bowl free-falls onto the lid and settles; the closed
  lid physically carries the bowl's weight on the hinge stop.
- P4: hands-off persistence ≥ 3.3 simulated s, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing
(all credit is latched). Passes seeds 0 and 1 on the forge (see run logs).

## Embodiment argument (single Franka, parallel jaw, OSC; base at the origin)

- Plausible base pose: (0, 0, 0), facing +x. Cabinet footprint centre at
  x = 0.56 m; every required contact lies in x 0.26-0.66 m, z 0.0-0.50 m — inside
  the 0.45-0.71 m comfortable envelope for the mid-height work and generous for the
  near/low points.
- Lid (must move): NO grasp needed — a PUSH on the open lid's broad raised face
  (30 x 30 cm plate, face centre at ~(0.33, 0, 0.37), tip at ~(0.26, 0, 0.50),
  tilted 32 deg past vertical toward the robot). Closed-fingertip forward push,
  ~1.6 N over-centre threshold, ~16 cm of travel; after over-centre gravity finishes
  the close. Precision required: "anywhere on the upper half of a 30-cm-wide face" —
  far above control noise.
- Black bowl (must move): rim pinch. The octagonal wall is 12 mm thick and 35 mm
  tall with an open 8 cm interior — one finger inside the cup, one outside, standard
  top-down rim grasp with full hand clearance (bowl stands in the open floor at
  x ≈ 0.40, |y| ≈ 0.22). Place target: anywhere within a ±9 cm square at the lid
  centre (0.568, 0), release at ~0.28 m height — a 18 x 18 cm landing zone for an
  11 cm bowl, comfortably above control noise. The closed lid rests on its hinge
  stop and carries the 0.15 kg bowl without moving.
- White bowl: never needs to be touched.
- Clearances: the lid push happens at 0.37-0.50 m height in open air; the bowl grasp
  is a floor-level rim pinch with no overhangs nearby (the open lid leans over
  x < 0.42, y within ±0.15 — the bowl slots at |y| = 0.22 are clear of it).

## Execution order

Ordering is FORCED BY PHYSICS, not declared: the bowl cannot rest on the lid before
the lid is closed (there is no surface), and a bowl dropped early is lost into the
cavity (recoverable only by re-extraction). No additional artificial ordering.

## Check list (smoke.py — rejection only; solve.py is the acceptance evidence)

1. settle/no-NaN: lid rests fully open, bowls stand on the floor, all states finite.
2. score ~0 at reset, no success.
3. randomization readback: black/white slot swap + xy jitter across 8 seeds.
4. randomization readback: lid initial open angle varies (> 3 deg spread).
5. null policy: 240 idle steps → score ≤ 0.02, no success.
6. SEED STRATEGY: bowl released over the cabinet top with the lid as found (open) →
   falls through the mouth into the cavity → no success, score ≤ 0.25.
7. buried ≠ on: lid closed over the trapped bowl → no success (bowl under, not on).
8. wrong object: WHITE decoy centered on the closed lid → no success.
9. wrong orientation: BLACK bowl upside-down, centered on the closed lid → no success.
10. near-miss: BLACK bowl upright on the lid but at its front edge, outside the
    ±9 cm centered tolerance → no success.
11. latched credit: bowl removed to the floor → latched score unchanged, no success.
12. open lid is not a shelf: bowl released against the standing lid's face slides
    off → no success.
13. rejection audit: success() never True anywhere in the battery.
14. final no-NaN.

## Rubric

score = 0.10·close-progress(running max from the episode's own initial angle)
      + 0.25·lid-ever-settled-closed (latched)
      + 0.15·black-bowl carry progress toward the goal point (running max)
      + 0.20·black-bowl ever upright+centered on the CLOSED lid (latched)
      capped at 0.85; exactly 1.0 iff success() holds live.
success = lid closed (≤ 8 deg) and still, black bowl upright within ±9 cm of the lid
centre at lid-top height, at rest. Null policy ≈ 0 (the lid at rest falls AWAY from
closed, so close-progress latches nothing).
