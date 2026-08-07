# libero_pick_chocolate_pudding_i46 — PuddingSiftScene (`simgen.pudding_sift`)

Pour a sealed bin's mixed contents onto a sieve grate so the small pearls fall
through into the hopper, then move the retained pudding ball to the dish and park
the bin.

## Provenance

Seed task: `libero/libero_pick_chocolate_pudding` — "pick up the chocolate pudding
and put it in the basket": one target box among tabletop distractors, one open
goal container, one grasp-carry-lower, one bbox check.

Kept from the seed: a brown "chocolate pudding" target that must end up inside an
open goal vessel, among other loose objects that are not the target.

## Strategic difference

The seed's distractors are scenery and its goal container is open from above; the
whole strategy is single-object transport. Here the "distractors" are half the
judged set, and NOTHING can be placed into the goal container by hand:

- The **hopper** (pearl goal) is sealed on every face except its top, which is a
  **grate** of twenty-five 26 mm holes between 6 mm bars — the only way in is to
  *fall through a hole*. A 42 mm pudding ball and an 80 mm parallel jaw both
  out-span every hole.
- The judged batch (1 pudding ball + 3–6 pearls) starts **sealed inside a roofed
  carry-bin** whose only exit is a 72 mm side port behind a 28 mm sill (plus a
  matching roof notch above it). A level bin keeps everything; contents leave only
  when the bin is lifted and tilted port-down **past vertical**.

So the plan is *partition by geometric filtration*: carry the loaded bin over the
grate, tilt-pour the whole mixed batch through the port onto the grate, let hole
geometry sort it — pearls thread the holes under gravity, the pudding ball is
retained on top — then pick the retained ball off the grate into the dish, and set
the bin down upright on the floor. None of the ~48 corpus tasks uses passive
size-sorting through a sieve as the load-bearing mechanism; the seed's strategy
(drop the target in the open goal vessel, ignore the rest) is a scored rejection
probe here (≤ 0.30, never success).

## Solution outline (solve.py — teleport = transport only)

1. **P0** settle + layout/seal readback.
2. **P1** carry + pour: a PD force/torque servo on the bin (mass readback includes
   contents; wrench pre-rotated by the rotation-since-anchor to defeat the
   rotating-frame quirk of `set_external_force_and_torque`) lifts the bin, hovers
   it over the grate, then tilts it port-down. Phase A: 112° with a theta rock —
   past vertical, because below 90° bin-local gravity presses contents into the
   floor and the sill retains them. Phase B: ramp to 142° (contents ride the roof
   underside) plus a **translational y-shake** of the carrier (±35 mm at 2.5 Hz):
   pearls jammed in the corner column beside the port pillar move rigidly with the
   bin under any rotation (rotational rock produces no bin-local y gravity — the
   up-axis servo has no yaw authority), but the inertial shake slides them along
   the roof into the open notch span, where they drop out onto the grate. Then
   ramp upright. Everything exits by contact dynamics through the port window.
3. **P2** park: transport the bin aside, lower, release; require `bin_parked` by
   readback.
4. **P3** sift assist: any pearl resting on a bar is re-dropped from free space
   directly over an open hole — the grate does the passing; a pearl (or the
   pudding) still *inside the bin* is a HARD FAIL (no teleport may bypass the
   sealed container). The pudding, if displaced, is re-dropped over the grate
   center — the grate does the retaining.
5. **P4** deliver the pudding ball into the dish (transport to free space above
   the dish, gravity does the seating).
6. **P5** hands-off persistence ≥ 3 s, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds 0, 1, 2 (3, 4, 6 pearls; both dish sides; varied port
headings, e.g. −18° and −122°): score 1.000 and SUCCESS on all three.

## Embodiment argument (Franka, 80 mm parallel jaw)

Suggested base pose: `(0.10, 0.0, 0.0)`, facing +x — the hopper center is at
~0.55 m, the bin and dish nominals at ~0.38–0.40 m reach, all within a 0.855 m
Franka envelope at comfortable heights (grate top 0.208 m, hover ~0.36 m).

- **Bin**: grasped by the 12 mm handle bar bridged 40 mm above the roof — 12 mm ≪
  80 mm jaw, and 40 mm posts give finger clearance under the bar. Bin + full
  contents ≈ 0.47 kg, far under payload. The pour (tilt past vertical + a small
  lateral shake) is a wrist rotation + end-effector oscillation.
- **Pudding ball**: 42 mm sphere ≤ 80 − 15 mm — a comfortable sphere pinch; it
  sits proud on the grate (center 21 mm above the bars, within a 12 mm lip that
  does not swallow it).
- **Pearls**: never need grasping — the intended interaction is a fingertip nudge
  of a pearl resting on a bar until it drops through a 26 mm hole (a Franka
  fingertip is ~18 mm wide: it can press a pearl off a bar but can NOT pass
  through a hole, so the basin stays sealed to the robot).
- **Sealing vs the jaw**: the 72 mm port/notch window is narrower than the 80 mm
  jaw, so an open jaw cannot enter the bin; contents rest ~90 mm below the roof —
  beyond fingertip reach through the notch. The only way to empty the bin really
  is to pour it.
- **Dish**: never needs manipulation (30 mm walls retain the delivered ball).

## Order requirement

**No fixed order is declared.** The instruction says so explicitly. Pouring before
parking, parking before sifting, delivering the pudding before or after the last
pearl — all permutations that end in the goal state are accepted; success() is a
pure settled-state predicate. (Physics itself imposes the only real precedence:
contents cannot be sorted until they leave the bin.)

## Scoring

Latched every physics substep (`post_step`): 0.10 · pudding ever left the bin +
0.50 · fraction of present pearls ever in the basin + 0.20 · pudding ever in the
dish, capped at 0.80; exactly 1.0 iff success() (all present pearls in the basin
∧ pudding in the dish ∧ bin parked upright at ground level ∧ settled).

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 15/15` on forge)

1. settle/no-NaN: contents sealed in the parked bin by readback, settled.
2. reset score ≤ 0.03, no success.
3. randomization readback A: hopper xy + yaw vary; dish side takes both signs.
4. randomization readback B: bin xy + free yaw, dish xy, pearl count vary.
5. null policy: 240 idle steps → score ≤ 0.03, no success, bin still sealed.
6. seed strategy (pudding alone to the dish): pearls sealed in bin → ≤ 0.32,
   no success.
7. filter honesty A: 42 mm ball dropped directly over an open hole is retained
   above the grate (readback), no success.
8. filter honesty B: 18 mm pearl dropped over the same hole passes into the
   basin (readback), no success.
9. dump cheat (whole batch into the dish): contamination, 0 in basin → ≤ 0.32,
   no success.
10. near miss (full sift, pudding retained on grate, bin parked): score ~0.60,
    no success.
11. bin not parked (full goal but bin on the grate): bin_parked alone rejects.
12. latched credit survives a delivered pearl teleported back out.
13. monotonicity: second sifted pearl latches strictly more than the first.
14. rejection audit: success() never True at any judged point in the battery.
15. final no-NaN. Video recorded to frames.npz (260 frames @ 960×600).
