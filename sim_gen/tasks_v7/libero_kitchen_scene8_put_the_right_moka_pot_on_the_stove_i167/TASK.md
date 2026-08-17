# libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove_i167 — prop the stove lid open with a rod, then put the copper pot on the burner

## Provenance

Seed: `libero_90/libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove` —
pick "the right" moka pot (a discrimination between two pots) and set it down
ON the open, flat stove: a single pick-and-place whose success is a pose
predicate on one object, satisfiable the moment the pot rests on the burner.

## Strategic difference from the seed

The burner still exists and the pot must still end up on it — but the stove is
now a deep **tub sealed by a heavy gravity-closing lid** (rear hinge, limits
[−80°, +0.5°], never over-center: nothing about the hinge can hold it open).
The pot placement — the seed's entire task — is unreachable until the solver
**builds falsework**:

1. Wrench the lid open and hold it (holding is *instrumental*, not terminal —
   an anti-pinning clause refuses credit while an external wrench carries the
   lid).
2. Erect a brace: stand the loose 232 mm **prop rod** in a small **socket
   well** sunk in the tub floor and seat its red tip in a **capture pocket**
   fenced on the lid's underside. The braced rod is the ONLY hands-free way
   the lid stays open (~60°); the fence is asymmetric — a deep fore wall
   catches the loaded tip (a plumb strut on an inclined panel is statically
   unstable and slides fore), a shallow aft wall clears the slanting shaft —
   so a rod placed anywhere else on the panel or floor collapses.
3. Only then lower the **COPPER** pot (an identical-shaped STEEL distractor
   stands beside it; sides randomized per episode) upright onto the burner pad
   through the propped mouth, and take hands off everything.

A solver needs a different plan (an ordered *construction* — open, brace,
verify the brace carries the load hands-free, then place — where step N is
physically impossible before step N−1) and different code structure: the
load-bearing state is a two-point mechanical brace between two other bodies,
not any pose of the goal object. Distinct from every other task in this
corpus: no other task requires erecting temporary falsework that must carry a
live gravity load before the goal object can even be delivered. The siblings
i141 (balance-scale weighing) and i75 (aperture threading) share the LIBERO
kitchen provenance but neither has an instrumental structure whose stability
is the gate.

## Scene

Fully procedural (native PhysX box colliders), on a 900×750×24 mm kinematic
counter **deck**:

- **tub** (kinematic, fixed — the hinge anchor must stay world-fixed): outer
  box with 300×300 mm inner cavity, 12 mm walls 54 mm high; floor carries a
  raised octagonal **burner pad** (96 mm across) and a **socket well** (20 mm
  square inner, 22 mm yellow walls) near the front-left.
- **lid** (dynamic, 0.45 kg): 316 mm steel panel + front handle bar, on an
  authored revolute Y joint at the rear top edge, limits [−80°, +0.5°],
  angular damping 0.5. Underside carries the **capture pocket** 200 mm from
  the hinge: deep (30 mm) fore/side fences, shallow (12 mm) aft fence,
  64×32 mm inner aperture. Gravity torque closes it from every legal angle.
- **rod** (dynamic, 60 g): 14×14×232 mm square-section prop, red tip. Braced
  rest angle is the derived `catch_angle` ≈ 60.1° (fore fence arrests the
  fore-slide); the well caps base tilt at ~7.8°.
- **pot_copper / pot_steel** (dynamic, 280 g): identical octagonal moka pots
  (64 mm across, 90 mm tall, side handles); only color and trim differ.

Randomization (all readback-verified in smoke): copper/steel side coin flip,
both pot xy jitter + free yaw, rod base xy + azimuth, well y position on the
tub floor (mirrored into the lid pocket y so the brace geometry holds).

Honesty-by-construction asserts in `SceneCfg.__post_init__`: the pocket's
fore catch arrests the tip strictly between `open_min`+6° and the plumb prop
angle; the aft fence clears the slanting shaft at every braced angle; the
well admits the rod with play but caps its tilt below 10°; the closed lid's
fence clears the well, pad, and floor; the pot fits through the propped mouth
with margin; the drop corridor onto the pad is open at the catch angle; a lid
resting on a pot wedged on the pad sits far below `open_min`.

## Rubric

Latched partial credit, capped at 0.75 unless success holds live:

| credit | clause |
|---|---|
| 0.20 | lid ever open ≥ `open_min` 50° (latched — however achieved) |
| 0.30 | propped: tip in pocket + base in well + lid ≥ 50° + **no external wrench on the lid** + rod/lid quasi-static (latched) |
| 0.25 | copper pot upright at rest on the burner pad + steel pot not in the tub (latched) |
| 1.00 | `success()` live |

`success()` (all live) = lid ≥ 50° AND rod braced (tip inside the pocket
bands, base inside the well bands) AND copper pot upright (≤15°) on the pad
AND steel pot not inside the tub AND lid/rod/pots at rest AND no external
wrench anywhere AND finite. The no-wrench clause is the anti-pinning
backstop: a hand (wrench) holding the lid open passes every geometric clause
but not this one; conversely an honest brace passes automatically. A lid
*swinging through* 50° is rejected by the quasi-static gates plus a 240-substep
continuous-success requirement in the solve before its P4 boundary.

## Solution outline (solve.py — teleport-transport contract)

Teleports carry ONE object at a time through free space; the lid is driven
only by an external wrench (gravity-feedforward + PD servo on the hinge);
everything load-bearing is contact dynamics:

- **P0** settle 120 steps; assert lid sealed (< 3°), rod and pots on the
  counter, score ≈ 0.
- **P1** wrench the lid open to 68° and hold → open latch, 0.20.
- **P2** transport the rod to a hover with its base over the well and its tip
  a few mm under the lid's pocket mouth; release; it drops base-into-well,
  tip-into-pocket.
- **P3** quasi-static hand-off: servo the lid down to `catch_angle` + 0.5°,
  bleed the wrench to zero over 360 steps, drop it; the fore fence catches
  the tip and the free lid rests braced at ~60° → prop latch, 0.50. (Up to 3
  attempts; the final configuration is 100 % contact-made.)
- **P4** transport the copper pot to a hover over the burner pad through the
  propped mouth; it drops and seats upright; success must hold 240
  consecutive substeps (2 s) → 1.0.
- **P5** ≥ 3.3 s fully hands-off persistence (10 × 40 substeps, success at
  every checkpoint) → `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed non-decreasing at every phase boundary. Verified
on the forge for seeds 0 and 1 (`--args "--headless --seed 1"`).

## Embodiment argument (Franka)

Every required interaction is tabletop manipulation of hand-sized rigid
objects inside a 0.90×0.75 m counter. The lid's front handle bar is a
canonical pinch-grasp feature; opening is a slow guarded arc (0.45 kg,
gravity-closing — the arm holds it exactly as the wrench does). The rod is a
14 mm square section (well inside the Franka jaw span) grasped near its tip;
standing it base-first into the 20 mm well leaves 3 mm per side, and the
pocket mouth above is 64 mm fore-aft — both are vertical insertions with the
lid held steady by... nothing: the arm releases the lid only AFTER the rod is
seated, exactly the quasi-static hand-off the solve performs, which is a
single-arm-feasible ordering (hold lid at 68° with the grasp, prop the rod
with a regrasp after resting the lid onto the braced rod — or open past 60°,
place the rod, and lower the lid onto it, as the solve does). The pots are
64 mm octagonal bodies with 36×12 mm side handles; the drop onto the pad is a
vertical placement with ≥ 25 mm xy clearance through a ~250 mm propped mouth.
With the base at the counter's +x edge all targets (tub ~0.38 m, well/pad
inside it, pots 0.25–0.45 m) sit inside the Franka's 0.855 m reach, and the
propped lid leaves top-down access to the pad.

## Execution order

1. `solve.py` (seeds 0 and 1) — runs first; demonstrates the task and the
   monotone score trace.
2. `smoke.py` — rejection battery, 13 checks:
   1. settle/no-NaN (lid sealed < 3°, score ≤ 0.01);
   2. randomization readback (rod base xy + azimuth, pot yaws differ across
      seeds);
   3. copper-side coin flip (both sides seen over 10 resets);
   4. null policy 240 steps (score ≤ 0.05, no success);
   5. seed strategy on the new scene (copper pot set ON TOP of the closed
      lid over the burner: score ≈ 0 — the seed's own goal pose earns
      nothing);
   6. anti-pinning (wrench holds the lid at 65°, verified held and calm: no
      prop credit while held; released lid slams shut; ≤ 0.205);
   7. pot-wedge cheat (lid lowered onto a pot standing on the pad: rests far
      below `open_min`, no prop latch, ≤ 0.455);
   8. wrong pot (rod braced honestly, STEEL pot on the burner: no pot latch,
      no success, ≤ 0.505);
   9. right pot wrong place (brace restored, copper pot in the tub OFF the
      pad: ≤ 0.505);
   10. copper pot on its SIDE on the pad (not upright: ≤ 0.505);
   11. pot-prop cheat (steel pot used as the prop under the lid edge: lid
       rests well below the brace band, no prop latch, ≤ 0.205);
   12. strut-beside cheat (rod stood on the bare floor beside the well, tip
       on the bare panel beside the pocket: verified carrying the lid while
       held, then COLLAPSES on release — the well and pocket clauses are
       load-bearing; ≤ 0.205);
   13. frames.npz video.

Both modules print machine-readable verdicts (`SIM_GEN_SOLVE: SUCCESS`,
`SIM_GEN_SMOKE: ALL PASS 13/13`) and hard-exit; a top-level try/except prints
a FAIL verdict on any exception so the forge watchdog never hangs.
