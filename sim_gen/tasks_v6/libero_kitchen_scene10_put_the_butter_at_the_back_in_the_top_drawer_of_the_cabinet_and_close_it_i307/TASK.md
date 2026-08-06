# butter_cellar (i307)

## Seed provenance

- Seed id: `libero_90/libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it`
- Seed source: `sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it.py`
- Seed strategy: OPEN a receptacle (grasp the drawer handle, pull the prismatic drawer
  out), PICK-PLACE the butter into the exposed static volume (at the back), then CLOSE
  the receptacle by pushing the drawer shut. Success = butter inside the top-drawer
  region AND drawer joint near zero. Distractors (bowl, pudding, second butter)
  disambiguated by identity.

## What changed and WHY it is strategically different

Kept from the seed only the abstract goal ("a butter stick ends up concealed deep in a
closed-off receptacle, receptacle restored, distractor butter excluded") and rebuilt the
containment mechanism wholesale:

1. **Nothing opens and nothing closes.** The receptacle (an open-topped silo) has no
   handle, no door, no drawer, and no openable state. The deep interior is blocked by a
   spring-raised dumbwaiter PLATFORM riding a vertical prismatic slide.
2. **The loaded surface itself descends into the goal region.** In the seed the object
   is placed into a static cavity; here the butter is placed on the EXPOSED tray and
   then the whole receptacle surface is driven 17 cm down its slide, carrying the
   butter into the silo. Transport-into-containment is done BY the fixture, not by the
   hand.
3. **The closure is a one-way ratchet, not a reversible joint.** A brass pawl
   (revolute, limits [-85°, 0°], gravity-off + weak return spring) folds down under a
   descending tray by cam contact and jams an ascending tray at its 0° stop. Once
   pressed past it, the state is irreversible — unlike a drawer, which can be reopened.
   The spring makes the latch band unreachable as a rest state by any other route
   (smoke-tested: an under-press springs all the way back up).
4. **The press is tool-mediated dead weight.** The intended press is placing a 0.55 kg
   iron INGOT into the plunger's socket (gravity overwhelms the ~3.5 N spring surplus),
   and the ingot must afterwards be returned to the floor clear of the silo — a
   place/press/unload cycle with a required undo step that has no analogue in the seed.
   (A direct sustained fingertip push on the socket cap is an equally legal press.)
5. **Same-shape color discrimination is kept but repurposed**: a WHITE lard block of
   identical shape must stay out of the silo.

A solver needs a different PLAN (load a moving surface → drive the receptacle down past
a ratchet → restore the press weight) and a different CODE STRUCTURE (prismatic-slide
spring plant, revolute pawl cam, latch-band pose predicates) — not different parameters
of open/insert/close.

## Teleport-solution outline (solve.py)

- **P0** reset, settle; assert platform at its top stop, pawl horizontal, score ≈ 0.
- **P1 TRANSPORT (teleport)**: butter → hover 25 mm above the tray load point
  (asserted outside the on-tray band); gravity lands it on the tray (contact).
- **P2 PRESS (contact dynamics)**: ingot → hover 20 mm above the plunger socket; from
  there everything is gravity: the 5.4 N ingot beats the 6.0 N constant spring minus
  2.6 N of platform+butter weight, drives the platform down the real prismatic joint;
  the descending tray edge CAMS the pawl down (asserted ≥ 45° fold — the pawl is never
  pose-written), passes, and the pawl's return spring re-seats it. No script forces.
- **P3 RELEASE + LATCH (contact dynamics)**: ingot teleported out of the socket back to
  open floor (transport of a free body out of a resting contact, like any pick); the
  spring throws the platform up and the tray edge JAMS under the pawl at its 0° stop.
  Asserted: platform rose off the bottom stop (> 0.120 m) yet held below the latch
  ceiling (< 0.152 m) — only the pawl can produce that state — butter still aboard.
- **P4** ≥ 3.3 simulated seconds hands-off; `SIM_GEN_SOLVE: SUCCESS` only if success()
  still holds. `SIM_GEN_SCORE` printed at every phase boundary; non-decreasing
  asserted. Passes on seeds 0 and 1 (see forge logs).

Teleports move bodies across free space only; the spring/damper plant lives in the
scene's `post_step` and acts identically in every module (solve, smoke, policy).

## Embodiment argument (single Franka, parallel jaw, OSC)

Plausible base pose: arm base at the world origin (0, 0, 0) facing +x; the silo axis is
at (0.56, 0), spawn slots at 0.33–0.36 m. Everything actionable lies in the 0.35–0.60 m
radial band; the highest waypoint (socket, 0.51 m) is well inside the demonstrated
0.45–0.71 m comfort envelope.

- **Butter (yellow, 58×32×26 mm, 80 g)**: top-down pinch of the 32 mm faces (fits the
  jaw with room to spare), carried above the rim and released 1–2 cm above the tray
  load point — the tray sits only 4 cm below the rim at rest, recessed in a 14 cm bore,
  so the fingers dip at most ~3 cm below the rim over a >4 cm lateral clearance to
  every wall. Placement tolerance is the 6 cm on-tray band: far coarser than arm noise.
- **Ingot (dark iron, 45×45×75 mm, 0.55 kg)**: top-down pinch of the 45 mm faces
  (within the demonstrated 0.62 kg carry ceiling); dropped 1–2 cm above the plunger
  socket, whose 12 mm lip funnels a 2.5 mm-per-side fit — and the socket stays 3.5+ cm
  ABOVE the rim over the entire stroke, so no finger ever enters the bore. Removal is
  the same pinch in reverse: the ingot protrudes 63+ mm above the lip.
- **Press (alternative)**: a sustained ~3.5 N downward fingertip push on the socket cap
  through a 17 cm vertical stroke — the proven closed-fingertip spring-press pattern;
  the cap never drops below 0.33 m.
- **Nothing else must be touched.** The pawl operates purely by tray contact; the lard
  is a leave-alone distractor.

## Execution order

Partially ordered: the press-past-the-pawl must precede the latch (physics enforces
it), and the ingot — if used — must be removed AFTER the press (its removal is what
lets the spring seat the tray on the pawl; success requires the ingot on the floor
clear of the silo). Butter loading is intended BEFORE the press (reliable); loading
after latching by dropping the butter down the bore is not forbidden by the rubric but
risks the butter catching on the pawl ledge — describe() says so.

## Rubric

success(): butter settled ON the tray (relative-pose band) ∧ platform in the latch band
(0.082, 0.150) ∧ pawl seated (|θ| ≤ 10°) ∧ ingot on the floor clear of the silo ∧ lard
not in the bore ∧ platform and butter at rest. score(): 0.10 approach (per-episode
d_init, exactly 0 for null) + 0.20 loaded + 0.25 press-depth (rising-only min-z) +
0.20 latched + 0.10 ingot-cleared, all latched, cap 0.85; 1.0 iff success().

## Check list (smoke.py — rejection battery, 15 checks)

1. settle/no-NaN: platform at top stop, pawl horizontal, blocks outside, still.
2. reset score ≈ 0, no success.
3. randomization: object→slot permutation varies (readback over 8 seeds).
4. randomization: per-slot xy jitter + spawn yaw vary (readback).
5. null policy: 240 idle steps → score ≈ 0, no success.
6. SEED-STRATEGY end state: butter placed in the receptacle, receptacle left "shut"
   (platform still at its rest stop) → rejected.
7. empty press: platform latched with no butter → rejected.
8. one-way ratchet: latched platform never escapes past the pawl (240 steps).
9. under-press near miss: released above the pawl → springs back to the top → rejected.
10. wrong object: lard latched on the tray → rejected.
11. butter balanced on the socket cap (in the silo column, not on the tray) → rejected.
12. ingot left in the socket at full depth with butter aboard → rejected.
13. latched credit survives the butter being yanked back out; still no success.
14. rejection audit: success() never True anywhere in the battery.
15. final no-NaN.
