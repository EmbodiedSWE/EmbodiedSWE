# roof_shuttle — slide the only roof open, then stand the bowl on it

`sim_gen` task `libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i371`
· env key `simgen.roof_shuttle` · robot slot `null` (scene-level, bodies driven
through scene handles)

## Seed provenance

Seed: `libero_90/libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet`
(RoboVerse `roboverse_pack/tasks/libero_90/...`). In the seed, a Franka picks a
freely accessible akita black bowl off the table and places it on the FIXED flat
top of a white cabinet; the terminated check is a bbox test in the cabinet frame.
One grasp, one carry, one set-down onto a surface that exists from t=0.

## What changed and why it is strategically different

The seed's two pillars are inverted:

1. **The source object starts captive, not free.** The black bowl begins SEALED
   inside the cabinet's front compartment (on a pedestal), roofed over by the
   cabinet's only top surface. In the seed you can touch the bowl at t=0; here
   the first thing you must manipulate is the cabinet itself.
2. **The destination does not exist as a fixed surface.** The cabinet has NO
   fixed top. Its only top surface is a captive SLIDING ROOF PANEL riding a
   prismatic rail along the cabinet's long axis (yellow handle bar, ~29 cm of
   travel). The rear half of the cabinet is a topless shaft with a deep-red
   floor 30 cm down — the seed's move ("release the bowl over the cabinet top")
   aimed at the only open area just drops the bowl into the shaft.

One slide does BOTH halves of the work: it uncovers the bowl bay (releasing the
source) and carries the only top surface clear of it (providing the
destination). Ordering is forced by geometry, not by a declared rule: the bowl
cannot rest on the panel's top face without having left the bay, and it cannot
leave the bay while the panel roofs it (smoke presses a real ~5 N hoist against
the closed roof and the joint-locked panel holds).

Versus the same-seed siblings: **i176** (ballast_rocker) is a see-saw
counterweight torque interlock; **i246** (crown_socket) evicts a plug from a
socket under a slick gable roof; **i306** (hatch_shelf) creates the top by
flipping an over-center hinged lid, with a decoy bowl. roof_shuttle has no
lever arithmetic, no occupancy eviction, no revolute/over-center mechanism and
no decoy — its core is a translating dual-role panel plus a captive source.

## Teleport-solution outline (solve.py, seeds 0 and 1)

- **P0** settle 1 s; read back panel start offset, pedestal and bowl poses
  (never hard-coded); assert captive start, score ~0.
- **P1 (applied force)** velocity-servo push (≤12 N) on the panel along its
  rail until ≥0.27 m of travel, brake, hands off. Latches: slide + open.
- **P2 (applied force)** PD force + gravity feedforward (≤8 N grasp analog,
  righting torque) lifts the bowl straight up OUT through the opening P1
  created — the out latch fires during this force-driven exit.
- **P3 (transport + gravity)** the held bowl is teleported over the panel's top
  face (clear of the handle bar) and RELEASED 2 cm up; gravity seats it. The
  seat clause is judged on the settled contact outcome.
- ≥3.3 s hands-off persistence, then `SIM_GEN_SOLVE: SUCCESS`.

Teleports are transport-only; every judged interaction (slide, exit, seat) is
contact dynamics or applied force.

## EMBODIMENT ARGUMENT (single Franka + parallel jaw, OSC)

Franka base ~0.55 m in front of the cabinet's front face (−y side), centered on
x; the cabinet stands on the ground (rim plane 0.30 m, panel top 0.318 m, all
well inside a floor-mounted Franka's envelope).

- **Roof panel**: the raised yellow handle bar (16 cm long, 2.4 cm wide, 2 cm
  tall, 3.5 cm finger clearance underneath) is a canonical top-grasp for a
  parallel jaw; the 29 cm slide is a straight horizontal pull/push away from
  the robot along −y→+y, ≤12 N against 3 N·s/m damping — one continuous OSC
  motion, no regrasp.
- **Bowl**: 11 cm across, 5 cm tall, 9 mm rim wall — a standard top pinch on
  the rim. Once the panel is open the bay presents a 0.36 × 0.27 m opening;
  the grasp point (rim at 0.21 m) sits ~0.09 m below the rim plane, a shallow
  vertical reach-in with the wrist above the opening.
- **Set-down**: the panel's top face at 0.318 m is an open, flat, near-side
  surface; placing the bowl clear of the handle bar needs ~6 cm of clearance,
  available on the panel's inboard half.

## Execution order (declared)

1. Slide the roof panel back by its handle (uncovers the bay).
2. Lift the black bowl out through the opening.
3. Stand it upright on the panel's top face (any rail position), clear of the
   handle bar.

Steps 1→2 are physically forced (captivity); 2→3 is forced because the panel
top is above the rim plane while the bay interior is below it.

## Rubric

Latched partial credit (survives transients; success judged live):
0.25 slide fraction + 0.15 bay-opened + 0.25 bowl-out-of-bay + 0.20 seated,
capped at 0.85; exactly 1.0 iff success() = bowl upright, settled, resting on
the panel's top face, everything finite. Null policy ~0; the shaft drop (seed
analog) earns no seat credit.

## Check list (smoke.py, rejection-only)

1. settle/no-NaN captive start, score ~0
2. randomization readback (bowl yaw >90° span, bowl+pedestal xy, panel offset
   matches stored value every reset)
3. null policy 240 steps ~0
4. SEED-strategy trap: bowl released over the topless rear half falls 30 cm
   into the shaft — no seat credit
5. roofed extraction: real ~5 N hoist presses the bowl against the closed roof
   ~2 s — bowl demonstrably rises yet never leaves the bay, panel stays parked
6. inverted bowl on the panel — upright clause refuses
7. panel opened but bowl never extracted — out/seat refuse
8. bowl upright on the ground beside the cabinet — seat clause refuses
9. bowl lying on its side on the panel — upright + rest band refuse
10. latched credit: open+extract latches survive a full restore to the start
    pose; seat does not; no success
11. rejection audit (success never observed anywhere in the battery)
12. final no-NaN
13. video: frames.npz, >10 frames
