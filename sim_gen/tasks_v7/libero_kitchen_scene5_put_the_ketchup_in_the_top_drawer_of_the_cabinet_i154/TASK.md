# libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_i154 — CarouselCabinetScene (`simgen.carousel_cabinet`)

Stow the RED ketchup bottle in the GREEN sector of a crank-driven carousel cabinet,
rotated to the BACK — while the YELLOW mustard bottle stays outside. The cabinet is a
roofed drum with ONE fixed front window; inside spins a free lazy-susan carousel split
into three sectors by divider walls, driven by a green crank arm above the roof that
always points at the green sector. Access is granted by ROTATION, not by opening:
align the green sector with the window, stand the bottle in it through the window,
then rotate the loaded carousel half a turn so the green sector faces the back.

## Seed provenance

- **Seed task**: `libero_90/kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet`
  (RoboVerse `roboverse_pack/tasks/libero_90/...`) — "put the ketchup in the top
  drawer of the cabinet": USD cabinet articulation with prismatic drawers + a ketchup
  asset, robot=franka. The plan: pull the top drawer open along its rail, pick up the
  bottle, drop it in; judged by a drawer-attached bounding box. Any drawer moment, any
  order, one receptacle, no distractor consequence.

## What changed (scene and code structure)

| | seed | this task |
|---|---|---|
| storage mechanism | prismatic drawer in a cabinet | rotating 3-sector CAROUSEL (revolute bearing, free both ways) inside a roofed drum with ONE fixed window |
| access act | PULL the drawer open (translation) | nothing opens — ROTATE the correct sector into the fixed window (the roof + 260-deg wall make the window the only way in) |
| receptacle identity | "the top drawer" — fixed, visible | the GREEN sector — readable only via the green mat / the green CRANK ARM pointer above the roof; its heading is random per episode |
| end state | bottle inside the (still-open) drawer at the front | bottle STOWED: the loaded green sector rotated to the BACK, away from the window (a final act absent from the seed) |
| ordering | none (open the drawer whenever) | **physically forced**: align BEFORE insert (a non-aligned green sector is unreachable — roof interlock), rotate AFTER insert (sector membership cannot change without rotating — divider interlock, both smoke-proven) |
| distractor | none that matters | YELLOW mustard bottle: must stay OUTSIDE (restraint clause) |
| assets | USD cabinet + ketchup meshes | 100 % procedural (compound kinematic housing; dynamic carousel with authored MassAPI + world-anchored revolute joint; cylinder bottles) |
| judging | static bbox attached to the drawer link | live geometric success (bottle upright in the green wedge by carousel-frame azimuth/radius/height, green sector within stow tol of the back, mustard outside, settled) + latched stage credit |

Code structure shares nothing with the seed (a 20-line LIBERO cfg pointing at USDs):
`@SCENES.register` BaseScene, compound spawners, per-env USD revolute joint authored in
`bind()` (carousel -> WORLD at the fixed housing axis), scene-owned crank-torque
buffer applied in `post_step`, `register_env(..., robot="null")`.

## Why strategically different

The seed's skill is *pull open a drawer, then pick-and-place into it* — one prismatic
pull, an always-visible receptacle, no ordering, no distractor. Here that plan earns
almost nothing: smoke check 7 CONSTRUCTS the seed's exact move (drop the bottle into
whatever receptacle faces the opening, no alignment) and it scores 0.20 with no
success — the bottle lands on a bare sector floor. What the solver must bring instead:
(1) **perception** — read the green pointer/mat to identify the target sector (its
heading is random, ±40..150 deg either side) and tell RED from YELLOW bottles with
shuffled slots; (2) **mechanism control** — a closed-loop crank rotation to a bearing
angle (align within 15 deg), a skill class absent from the seed (and the carousel is
NEVER pushed shut — rotation is the access mechanism, not the goal); (3) **forced
order** — the roof interlock (smoke 5: a bottle dropped from above never enters) makes
the window the only way in, so align must precede insert; the divider interlock
(smoke 6: a 3x-weight shove held against the crank never crosses a divider) makes
sector membership rotation-only, so the stow rotation must FOLLOW insertion and carry
the bottle; (4) **a second, load-bearing rotation** — the goal is not "bottle in
receptacle" but "receptacle rotated away": leaving the green sector at the window
fails (smoke 10); (5) **restraint** — the mustard must stay out (smoke 12: a perfect
ketchup stow with the mustard smuggled inside fails). The seed's one skill (pull open,
drop in) is the one plan that scores ~0 here.

## Solution outline (as demonstrated by solve.py on the forge)

1. **P0** settle 1.5 s; readback layout (green azimuth ±40..150 deg, ketchup side ±y,
   bottle jitter) and the authored carousel mass (0.900 kg); baseline score 0.0, no
   success.
2. **P1** (align, closed loop): a velocity-capped P-servo writes the scene's
   crank-torque buffer (|tau| <= 0.3 N·m — a hand on the crank arm); the carousel
   rotates on its bearing until the green sector faces the window (converges to
   ~0.5 deg; both start signs exercised across seeds) → aligned latch, score 0.20.
   The carousel is never teleported after reset.
3. **P2** (teleport = transport only): carry the ketchup through the OPEN window
   (~290 x 220 mm vs a 60 mm bottle — free space) to 15 mm above the green mat, zero
   velocity; it FALLS and seats upright by contact → inserted + seated latches,
   score 0.60. The mustard is never touched.
4. **P3** (stow under load): the same servo (0.5 rad/s cap) rotates the LOADED
   carousel half a turn — the bottle rides the platform on friction — until the green
   sector is within 1 deg of the back → success, score 1.0.
5. **P4** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

Monotone `SIM_GEN_SCORE` prints: 0.0000 → 0.2000 → 0.6000 → 1.0000 → 1.0000.

## Franka embodiment (single arm, parallel jaw, OSC)

Proposed base pose: **(1.30, 0.00, 0.00), facing −x** (toward the window; nominal
reach 0.855 m). The window plane is ~0.49 m away, the green mat seat ~0.71 m, the
crank arm (z 0.42) ~0.68 m, the bottles ~0.42–0.48 m — all inside the dexterous shell.

- **Crank arm** (100 x 24 x 20 mm box at z 0.420, above the roof): pinch its free end
  (24 mm ≪ 80 mm jaw max) and orbit the wrist about the bearing axis, regripping every
  60–90 deg (the demonstrated torque is ~0.05–0.3 N·m — fingertip effort). The arm
  spins with the carousel and is always exposed above the roof; alternatively push a
  divider edge through the window. Free spin both ways — whichever direction is closer.
- **Ketchup bottle** (Ø60 x 160 mm, 150 g): vertical SIDE PINCH at the upper half
  (60 mm < 80 mm jaw), lift over the sill (top at 95 mm), insert through the window
  aperture (~290 mm wide x 220 mm tall — admits hand + bottle with margin), set down
  on the green mat at radius ~95 mm — a ±45 deg wedge, 95 mm radial band and 25 mm
  height tolerance: generous for closed-loop placement. Release and retract straight
  back out.
- **Stow crank**: same crank skill, now with the bottle riding inside — no contact
  with the bottle ever again.
- **Mustard bottle**: never touched — restraint, not manipulation.
- Housing is kinematic: incidental contact cannot move the goal frames; the carousel
  is the only mechanism, and its bearing is the only degree of freedom.

## Execution order (declared)

`align green sector to window → insert ketchup through window onto the green mat →
rotate carousel to stow the green sector at the back`. REQUIRED, and enforced by
physics rather than rubric fiat: the roof + wall make insertion impossible unless the
target sector faces the window (smoke 5), and the divider walls make sector membership
unchangeable without rotation (smoke 6), so the stow rotation must come last and carry
the bottle. The rubric mirrors this: stow progress only accrues while the bottle rides
seated in the green sector.

## Validation evidence (all on the forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0` (start +146.9 deg, ketchup +y): SUCCESS, scores 0.0/0.2/0.6/1.0/1.0.
- `solve --seed 1` (start −113.9 deg — opposite spin direction): SUCCESS.
- `solve --seed 2` (start +48.1 deg — short approach, ketchup −y): SUCCESS — three seeds,
  both rotation directions, both bottle sides; servo converges to ≤0.5 deg every time.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 13/13**, frames.npz (68 x 600 x 960) saved:
  1. settle/no-NaN; green sector 40..150 deg from the window, bottles outside; score 0
  2. randomization readback differs (green_az Δ154 deg, ketchup xy Δ507 mm)
  3. side swap: ketchup occupies both ground slots over 10 resets
  4. null policy: 240 idle steps, carousel drift 0.00 deg, score 0, no success
  5. ROOF INTERLOCK: bottle dropped from above the roof hole falls 120 mm onto the
     roof and NEVER enters (the inserted latch never fires) — window-only access
  6. DIVIDER INTERLOCK: 3x-weight tangential shove for 1.5 s (crank held) drives the
     bottle 112 mm into the divider — it never leaves the green wedge
  7. SEED STRATEGY: bottle dropped into the sector facing the window without
     aligning → bare floor, score 0.20, no success
  8. wrong sector: ketchup seated in a non-green sector at the back → no success
  9. wrong object: MUSTARD stowed in the green sector at the back → no success, score 0
  10. near-miss: green sector parked 40 deg short of the back (tol 20) → no success
  11. tipped: ketchup lying on its side on the green mat at the back → refused
  12. smuggled: ketchup perfectly stowed BUT mustard inside another sector → no success
  13. video frames.npz saved

## Files

- `scene.py` — CarouselCabinetScene + housing/carousel compound spawners + world-
  anchored bearing joint + rubric; registers `simgen.carousel_cabinet`.
- `solve.py` — crank-servo align + teleport-transport insertion + loaded stow
  rotation certificate (`--seed N`).
- `smoke.py` — 13-check rejection battery + video.
