# scene_d_i130 — carousel vault: shutter, rim-indexed turntable, window extraction

## Seed provenance

Seed task: `calvin/scene_D` — the CALVIN play-table D: a Franka at a desk with
four independent articulated primitives (`base__button`, `base__switch`,
`base__slide`, `base__drawer`) and three colored blocks (pink/blue/red) lying
on the open tabletop. Every CALVIN-D objective is either "actuate one
single-DOF primitive to its other end-stop" or "pick up / push a block sitting
in free space". Carried over here: a small articulated furniture piece with a
sliding element, colored cubes among which exactly one is the target, and a
place-on-goal finish.

## Strategic difference

**vs. the seed.** CALVIN-D's primitives are independent and binary (either
end-stop is "done"), and its blocks are graspable in free space from the very
first frame. Here *nothing* is graspable in free space and no DOF has a useful
end-stop: both cubes ride pockets on a free-spinning **turntable sealed inside
a vault** (solid walls, a roof with one 14×14 cm window, a sliding shutter over
the window). The disc is touchable only where its rim protrudes ~3 cm through
a shallow front slit as an exposed thumbwheel lip. The goal state needs a
**chain the seed never requires**: (a) slide the shutter clear (its analogue of
the seed's slider — but here merely an *enabling* move worth 0.15, not a goal),
(b) rotate the disc by its rim to a **continuous angular alignment target** —
the red pocket centred under the window, sampled anywhere in a ±[50°, 310°]
band, *not* an end-stop push, (c) lift the red cube out through the roof
aperture, (d) place it on the seeded ground pad. The partial order is enforced
by physics, not by the rubric: a closed shutter caps any lift under the window
(smoke check 5, real force), everywhere else the roof caps the cube (check 6,
real force), the 4 cm slit passes the 3 cm disc rim but never the 4.5 cm cube
(cfg contract), and the reset band starts the pocket ≥ 5.7 cm from the window
centre — beyond even the geometric pass-through bound (cfg contract).

**vs. the rest of the corpus.** `libero_..._i88` (read in full as the
structural template) is a tool-mediated bayonet key → drawer pull; here there
is no tool, no drawer, and the key skill is continuous rotary indexing of a
*hidden payload* under an aperture. `pen_holder` (robobench exemplar read) is
free-space pick-and-insert. No other task read has a turntable, a
rotate-to-bearing target, or an aperture-gated extraction. There is no stored
energy: the disc is a plain damped revolute rotor, the shutter a plain
prismatic plate with hard stops, the cubes free bodies — every intermediate
outcome persists hands-off.

## Scene (`carousel_vault`, env `simgen.carousel_vault`, robot="null")

Procedural geometry only (compound spawners; per-env USD joints authored once
at bind time; joint collision filtering applies only to the housing↔disc and
housing↔shutter pairs, so every cube contact — the containment — stays live):

- **Housing** (kinematic): plinth deck (top 0.10 m), back/side walls, slitted
  front wall (slit 0.10–0.14 m tall, |y| < 0.14), roof 0.19–0.21 m with a
  14×14 cm window centred over the front pocket point (0.075, 0).
- **Disc** (dynamic, revolute about z, no limits, ang. damping 4): r = 0.20 m
  cylinder at z = 0.119 whose rim protrudes 3 cm past the front face through
  the slit, carrying two open-top pockets at (±0.075, 0): inner half-width
  45 mm, walls 50 mm tall (overtop the cube; roof gap 6 mm traps it).
- **Shutter** (dynamic, prismatic along y, limits [0, 0.17] m): 164 mm plate
  floating 2 mm above the roof + orange grip knob; slick material.
- **Cubes**: red prize + blue decoy, 4.5 cm / 60 g, seated in opposite
  pockets, carried by the disc's sampled start yaw.
- **Pad** (kinematic): green 14 cm square, 12 mm tall, on the ground.

Per-episode randomization (readback-verified): disc start yaw
θ0 ∈ ±[50°, 310°] (never indexed, never window-extractable at reset) and the
pad's ground xy ∈ [0.36, 0.54] × [−0.44, −0.14].

Geometry contract (~27 asserts in `__post_init__`): pocket sweep clears the
front wall; indexed pocket presents the whole cube inside the window even at
the index tolerance edge; roof gap traps the cube; slit passes the disc, never
the cube; rim protrudes ≥ 15 mm (pinchable); shutter clears the window at
`open_thresh` and its closed air gap is 0.5–4 mm; the reset band exceeds the
window pass-through bound by ≥ 8 mm; jaw-fit checks; weights sum to the cap.

**Success** (live state, no memory): red cube ON the pad (xy < 5 cm, rest
height ±12 mm) ∧ settled ∧ finite. **Score**: latched partial credit — opened
0.15 (shutter past 0.15 m), indexed 0.20 (red pocket centre within 3 cm of the
window centre), out 0.25 (red cube above the roof plane or clear of the
footprint), capped at 0.60; exactly 1.0 iff success. Latches clear on reset.
Note (honesty): the index latch is a *conservative* marker — extraction is
geometrically possible out to ~4.7 cm of pocket offset, so a razor-edge policy
could in principle lift out without the 0.20 latch; success stays live-physical
and the score ladder stays monotone either way.

## Solution (`solve.py`) — one transport teleport, then applied wrench only

1. **P0 settle + perception**: 120 steps; read θ0 and the pad xy back from the
   live state (never hard-coded); assert the disc settled at its sampled yaw,
   the prize rides its pocket, shutter closed, score ≈ 0.
2. **P1 open** (applied force): PD y-force servo on the shutter (kp 60 N/m,
   kd 8, clamp 8 N — a hand on the knob), released only once past the
   threshold *and slow*; shutter stands at 0.164 m hands-off (score 0.15).
3. **P2 index** (applied torque): PD z-torque servo on the disc (kp
   0.8 N·m/rad, kd 0.25, clamp 0.6 N·m — a fingertip dragging the exposed rim;
   torque about the rotor's own axis is invariant under the pod wrench-frame
   drag), shortest-path to yaw 0, brake, settle: |yaw| ≈ 0.01 rad from starts
   of +2.65 and −1.37 rad (score 0.35).
4. **P3 extract** (applied force): PD lift (kp 8 N/m, kd 0.8, clamp 2.5 N +
   0.59 N gravity feedforward) raises the cube out of the pocket, through the
   open window, to z ≈ 0.29 (out latch, score 0.60). A stall guard asserts the
   cube actually rose within 90 steps.
5. **P4 transport** (the single permitted teleport): wrench zeroed, the
   airborne cube written to a hover 3.5 cm above the sampled pad and dropped.
6. **P5 hands-off**: it lands and settles in ~10 steps; `success()` turns True
   on the live state (score 1.0) and holds through a **3.3 s hands-off
   persistence** window before `SIM_GEN_SOLVE: SUCCESS`.

The whole episode repeats on a second seed (fresh reset; SIM_GEN_SCORE stream
non-decreasing: 0 → 0.15 → 0.35 → 0.60 → 1.0). Forge result: both seeds OK,
rc=0, ~38 s.

## Franka embodiment argument

Base pose: one fixed base on the ground at ≈ (0.55, −0.30), facing the vault's
front-right corner — all four contact zones below are inside a 0.30–0.75 m
reach annulus from there, none behind the vault.

- **Shutter knob** (24×40×48 mm orange tab on top of the plate, z ≈ 0.27):
  canonical parallel-jaw pinch (40 mm < 80 mm jaw), then a 15 cm lateral drag
  along +y at ≤ 8 N. Free air above and around it.
- **Disc rim** (the thumbwheel): protrudes 30 mm past the front face across
  the full slit width at z ≈ 0.104–0.134 with free air below (deck top 0.10)
  — a fingertip press-and-drag or a 30 mm-deep edge pinch; repeated strokes
  allowed since the rotor is free (no detent). The required tangential force
  is ≤ 0.6 N·m / 0.17 m ≈ 3.5 N.
- **Red cube** (in the indexed pocket): the 14×14 cm window admits the Franka
  hand — cube top sits 31 mm below the roof top, pocket inner width 90 mm vs
  the 45 mm cube leaves 22 mm per side for the fingertips, and the cube's top
  20 mm are graspable with the palm above the roof plane (no wrist insertion
  needed beyond ~35 mm). Straight vertical lift, ≤ 1 N payload.
- **Pad placement**: open-ground place at 0.36–0.54 m — free space, trivial.
- **Decoy cube**: identical geometry in the opposite pocket — reachability is
  never the discriminator; color-keyed selection is.

## Execution order declaration

Scene was designed first with the minimal success predicate; `solve.py` was
then run on the forge and passed both seeds on the first submission (rc=0,
~38 s). After the solve was demonstrated, one geometry-contract hole was found
on paper — the old ±[40°, 320°] start band could reset the pocket inside the
window's geometric pass-through bound — so the band was tightened to
±[50°, 310°] with a new cfg assert, and the solve was re-run clean before
`smoke.py` was written. No check was ever weakened to make a run pass; both
forge runs after the change passed first try.

## Smoke battery (`smoke.py`) — 14 checks, all rejection/health

1. **settle/no-NaN** — shutter at 0, disc at its sampled yaw, prize riding its
   pocket, score ≈ 0, no success.
2. **randomization A** — θ0 varies across 8 seeded resets (span 9.42 rad) and
   the live disc yaw *and* the pocketed prize track the sample every time.
3. **randomization B** — pad xy varies (std 0.067) and the live pad body
   tracks the sample.
4. **null policy** — 240 idle steps: vault sealed, cube captive, score ≈ 0.
5. **closed-shutter cap** — disc constructed indexed (whole rotor assembly
   written together), shutter left CLOSED; the real solve-grade lift raises
   the cube off its seat (asserted — the force pathway is live) and it jams on
   the shutter plate far below the out plane; out never fires, no success.
6. **un-indexed cap** — shutter written open, rotor assembly constructed at
   yaw π; the real lift raises the cube (asserted) and it jams on the ROOF;
   out never fires, no success.
7. **seed reflex** — the real push opens the shutter fully and it stays open
   hands-off — the whole CALVIN slider move scores exactly 0.15, no success.
8. **wrong object** — the BLUE decoy constructed resting on the pad: score ≈
   0, no success (the rubric keys the red body).
9. **near miss** — the red cube settled on the ground 11 cm from the pad
   centre: out credit only (0.25, capped), on_pad False, no success.
10. **on-pad but moving** — the red cube written at the pad rest pose with
    0.3 m/s lateral velocity: the settle gate refuses success (judged without
    stepping, then removed).
11. **hover above pad** — the red cube written 5 cm above the pad rest height:
    the z gate refuses success (judged without stepping, then removed).
12. **rejection audit** — `success()` observed False at every step of the
    entire battery.
13. **final no-NaN.**
14. **video** — frames.npz (299 rgb frames, 960×600) written to CWD.

Forge result: `SIM_GEN_SMOKE: ALL PASS 14/14` (rc=0, ~112 s).
