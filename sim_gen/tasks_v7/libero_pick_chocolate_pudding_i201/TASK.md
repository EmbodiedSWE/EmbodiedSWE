# libero_pick_chocolate_pudding_i201 — ShuttleVaultScene (`simgen.shuttle_vault`)

Load the chocolate pudding through a sealed vault's roof window into an internal
shuttle pocket, convey it along a hidden rail by pushing an external T-knob, let
it gravity-drop through an offset floor hole into the lower chamber, then pull
the shuttle back home.

## Provenance

Seed task: `libero/libero_pick_chocolate_pudding` — "pick up the chocolate
pudding and put it in the basket": one target box among tabletop distractors,
one open goal container, one grasp-carry-lower, one bbox check.

Kept from the seed: a brown "chocolate pudding" target that must end up inside a
goal container, among other loose objects that are not the target.

## Strategic difference

The seed's basket is open from above — the whole strategy is single-object
transport to a free-space release point. Here the goal container is the **lower
chamber of a sealed, floor-fixed vault** that no gripper, jaw, or dropped object
can reach directly:

- The chamber's only entrance is an 80 mm **drop hole** in the mid-level gallery
  floor at the FAR end of the vault (x_b = +0.052 in vault frame).
- The only outside opening above the gallery is a 72 mm **loading window** in
  the roof at the NEAR end (x_a = −0.052) — horizontally offset from the hole
  with 28 mm of solid roof overlap between their projections, so nothing dropped
  (or bounced: restitution 0 everywhere) from outside can reach the chamber.
- The bridge between the two apertures is an internal **shuttle car**: an
  open-top open-bottom pocket collar on a spawn-authored prismatic rail in the
  gallery, operated only via a 12 mm handle bar that passes through a 24 mm end
  slot to a yellow T-knob outside.

So the plan is *load → convey → gravity-deposit → re-home*: drop the pudding
through the window into the home-parked pocket, push the knob in so the pocket
travels over the drop hole (the pocket has no floor — the gallery floor is its
floor — so at B the pudding simply falls through), then pull the knob back out.
No corpus task uses an offset-aperture airlock with an internal conveyor as the
load-bearing mechanism: the sibling same-seed task `..._i46` (PuddingSiftScene)
is carry-bin tilt-pour + sieve size-sorting of a mixed batch; `close_box_i26`
traps a free crate; here the target is never carried while contained, the
container never moves, and the sorting is positional (offset apertures), not by
size. The seed's strategy (drop the target straight down into the goal) is a
scored rejection probe (roof drop over the hole's xy scores 0; the direct path
is geometrically sealed).

## Solution outline (solve.py — teleport = transport only)

1. **P0** build(seed) + reset(seed), settle, layout readback (vault build pose,
   shuttle q0, item xy); assert score ≤ 0.03, no success.
2. **P1** pull home: a velocity-regulated force servo on the T-knob (K = 4 N per
   m/s of rail-rate error, v_des 0.10 → 0.04 near target, stiction floor
   escalation 0.4 → 2.5 N, cap 6 N, force pre-encoded into the probed rail
   frame) pulls the shuttle to q ≤ 0.015; the authored joint limit is the hard
   stop, press-and-hold 30 steps to park.
3. **P2** load: teleport the pudding to free space 65 mm above the roof window
   (housing-frame point, housing yaw applied) and let gravity drop it through
   the 72 mm window into the pocket (interior 78 mm; pudding tumble diagonal
   67 mm — cannot jam). Perched-on-rim fallback: a 1.5 N horizontal nudge toward
   the pocket center; missed-window fallback: re-drop.
4. **P3** convey: servo the knob IN to q = travel − 4 mm; the pocket carries the
   pudding over the drop hole where it falls into the chamber by gravity
   (hole 80 ≥ pocket interior 78 guarantees the drop footprint). Rocking
   fallback if the deposit doesn't register.
5. **P4** re-home the shuttle, settle; assert success and score 1.0.
6. **P5** hands-off persistence ≥ 3 s, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds 0, 1, 2 (vault pose/yaw, shuttle q0, and item layouts
all vary): score 1.000 and SUCCESS on all.

## Embodiment argument (Franka, 80 mm parallel jaw)

Suggested base pose: `(-0.10, 0.0, 0.0)`, facing +x — vault center ~0.45 m,
item slots 0.28–0.44 m, all inside a 0.855 m envelope at comfortable heights
(knob at z 0.162, window plane z 0.215).

- **T-knob**: a 32 mm cap on the 12 mm handle bar — pinched across ±y (32 ≤
  80 − 15) at every rail position: the knob protrudes ≥ 15 mm from the wall even
  fully pushed in, up to ~129 mm at home. Push/pull forces are 2–6 N along the
  rail at z 0.162 — a straight horizontal EE translation, no wrist gymnastics.
- **Pudding**: 45 mm square block ≤ 80 − 15 mm — comfortable top-down pinch from
  its floor slot; released centered above the 72 mm window (±13.5 mm lateral
  tolerance) it falls in.
- **Sealing vs the jaw**: the 72 mm window is narrower than the 80 mm open jaw,
  so the jaw cannot enter the vault; chamber contents rest ≥ 90 mm below the
  window plane and are horizontally offset under solid roof — beyond fingertip
  reach. The only way into the chamber really is the shuttle path.
- **Distractors** (ketchup, milk): never need manipulation — success requires
  them merely left outside the vault, where they start.

## Order requirement

**No fixed order is declared** — success() is a pure settled-state predicate.
Physics imposes the only real precedence: the pudding cannot enter the chamber
except through the hole, cannot reach the hole except in the pocket, and cannot
enter the pocket except through the window with the pocket home — so
home → load → convey → deposit → re-home is forced mechanically, not by rubric.

## Scoring

Latched every physics substep (`post_step`), latches cleared on reset:
0.20 · pudding ever in the pocket in the gallery + 0.15 · ever conveyed while
loaded + 0.30 · ever in the lower chamber + 0.10 · in chamber with shuttle
home, capped at 0.75; exactly 1.0 iff success() (pudding in chamber ∧ shuttle
home ∧ ketchup and milk outside the vault ∧ settled ∧ finite).

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 16/16` on forge)

1. settle/no-NaN: vault sealed, everything outside, settled.
2. randomization readback A: pudding xy+yaw, ketchup xy, milk xy, shuttle q
   differ across seeds.
3. rail-coordinate coverage over 8 resets (spread > 0.030).
4. null policy: 300 idle steps → score ≤ 0.03, no success.
5. seed strategy (drop pudding into the home pocket and stop): ≤ 0.20 + ε,
   no success — the open basket-style drop alone earns only the load latch.
6. sealing honesty A: pudding dropped from above the drop-hole's xy lands ON
   the roof (readback), score 0.
7. sealing honesty B: shuttle parked at B, window drop strands the pudding on
   the gallery floor — not loaded, score 0.
8. near miss: load + half-convey → ≈ 0.35, no success.
9. mechanism honesty: knob force-push to B moves the rail and the contained
   pudding (+> 60 mm) and deposits it in the chamber — all by readback.
10. not-home rejection: full deposit, shuttle left at B → ≤ 0.65, no success.
11. latched credit survives the pudding teleported back out.
12. wrong object: milk in the chamber → ≤ 0.01.
13. contamination: pudding AND milk inside, shuttle home, settled → the
    distractor clause alone refuses success (≤ 0.75 + ε).
14. settle gate: goal state with the pudding sliding at 0.4 m/s → no success.
15. rejection audit: success() never True at any judged point in the battery.
16. video: frames.npz recorded (960×600).
