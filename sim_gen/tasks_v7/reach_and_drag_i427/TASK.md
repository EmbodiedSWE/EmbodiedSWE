# trolley_shunt (reach_and_drag_i427)

## Seed provenance

Seed: `rlbench/reach_and_drag` — a Franka grasps a stick tool and drags a cube across
the tabletop onto a colored target square.

Kept from the seed: the core act is still *moving one judged object along the floor to a
color-designated goal region by pushing/dragging contact* (no grasp-and-carry of the
judged object).

## Strategic difference

Everything that made the seed a one-line straight drag is replaced:

1. **The judged object is an articulated, nonholonomic VEHICLE**, not a block. The
   trolley is a 4-body cluster — deck + two fixed-axle wheels on free revolutes + a free
   caster ball (D6). It rolls easily along its heading and grips sideways, so it cannot
   be dragged straight at the target: it must be *steered* (push to translate, nudge to
   yaw) like a shopping cart. The seed's tool-drag strategy (plant a stick behind the
   object, pull straight) does nothing useful here — a sideways pull just skids and the
   route is not straight anyway.
2. **The route is order-forced, not straight.** A center divider wall splits the walled
   yard into two lanes; the trolley starts in the decoy lane *facing away from the
   goal*. The only rolling path is: east down the start lane past the divider tip,
   U-turn in the open east zone, west up the far lane, in over the shed sill. Score
   latches are order-chained (east zone → far lane → sill) and gated on rolling height +
   uprightness, so lifting the trolley over the divider earns nothing.
3. **The goal is perceptually keyed and flips per episode.** Two identical roofed sheds
   terminate the lanes; flat roof tiles (GREEN = goal, RED = decoy) are the only
   difference, and which lane is green is resampled every reset. The decoy shed sits
   *directly behind the start pose* — the seed's "move toward the target you see"
   heuristic drives you into the trap.
4. **Parking is one-way and must be retained.** The goal shed has a 10.5 cm roof
   (nothing can be lowered in from above) and a 12 mm sill — a long shallow ramp on the
   approach, a vertical step inside — so rolling in is easy (~3.2 N climb) and rolling
   back out statically needs ~11 N, above the task's 8 N force cap (asserted in the
   scene's `__post_init__` statics/energetics audit). Success requires the parked pose
   (band, lane-aligned within 20°, upright, still) *sustained* 60 substeps, and the
   smoke battery verifies a 3 N outward shove from dead contact with the step (zero
   run-up) cannot extract a parked trolley.

### Contrast with the closest corpus neighbors

- **cart_ferry**: the cart there is a *jointless* rigid box slid down a *straight*
  roofed lane, and the judged object is a *bowl riding on it*. Here the vehicle itself
  is the judged object, its wheels are real revolute joints that make it nonholonomic,
  the route has a forced 180° reversal around a divider, and the goal is a per-episode
  perceptual flip with a one-way sill. No shared mechanic beyond "something rolls".
- **roll_in_garage**: a *passive free roller* is pushed *straight* through a single
  doorway. Here there is no straight line to the goal (divider + U-turn), the trolley
  steers (fixed axle = heading is a controlled state, not free), there are two sheds
  with a color key and a decoy trap, and the sill makes entry irreversible.

## Solution phases

0. Settle; read the goal side from the roof tiles (`scene.flip` readback).
1. Drive EAST down the start lane past the divider tip (g1 latch, score 0.20).
2. U-turn in the open east zone; drive WEST up the far lane (g2 latch, 0.45).
3. Line up east of the ramp toe; roll in over the sill ramp (+3.5 N feed-forward on
   the ramp); brake to a DEAD stop inside the park band (g3 latch, 0.70).
4. Hands off; hold the parked pose 60 substeps → success (1.00).
5. Keep simulating ≥ 3 s hands-off; success must still hold.

The solver is force-only: a heading P-servo torque (τ = k_θ·err − k_ω·ω_z, caps 1 N·m)
plus a forward velocity servo (F = k_v·(v_des − v_fwd), cap 8 N) applied through the
scene's world-frame deck wrench plant — the same push-and-steer interaction a hand
delivers. **Zero teleports.**

## Embodiment argument (single Franka, OSC, parallel jaw)

Plausible base pose: on the yard's long (south) side, ~0.50–0.55 m from the yard
center, on a ~12 cm pedestal so the wrist clears the 12 cm perimeter walls and the
12.5 cm shed roofs. The 80 × 52 cm yard interior is then inside the ~85 cm reach
envelope.

Per-object contact strategy (all pushes 2–6 N, well inside Franka capability):

- **Drive pushes**: closed-gripper knuckles against the yellow rear POST (z 6–8 cm
  above floor) or the deck rear edge — forward force at deck height, exactly the
  plant's force channel.
- **Steering nudges**: fingertip taps on the deck's front corner skirts (low faces at
  z 2.4–5 cm, split left/right of the caster) — small yaw moments, the plant's torque
  channel.
- **U-turn**: alternate corner nudges and short pushes while the arm re-poses around
  the open east zone (fully reachable, 37 cm of open floor past the divider tip).
- **Final roll-in**: slow quasi-static feed on the post through the shed mouth (15 cm
  wide, 10.5 cm of headroom — fingertip-first hand fits) until the tail passes the
  sill; at 2–3 cm/s the trolley coasts only a few cm and settles in the band.
- **Perception**: the green/red roof tiles are top-facing 10 cm squares — visible from
  a wrist or overhead camera at episode start.

## Execution-order declaration

Progress is DRIVEN in this order and latched: (1) east open zone, (2) far lane, (3)
over the sill into the green shed — each latch requires the previous one, rolling
height (deck z < 7.5 cm) and uprightness, so fly-over/carry shortcuts earn nothing.
Success additionally requires the sustained parked pose and is monotone (score never
decreases; asserted at every solve phase boundary).

## Checks

- `solve.py`: SIM_GEN_SCORE at 6 phase boundaries (asserted non-decreasing), success
  only after a ≥ 3 sim-second hands-off persistence window; passes on ≥ 2 seeds.
- `smoke.py` (13 checks): settle/no-NaN + spawn-in-decoy-lane + score 0; randomization
  READBACK (yard xy/yaw spread, both flips seen, flags/lane consistent); null policy
  < 2 cm; nonholonomy probe (forward shove ≥ 3× lateral displacement); seed-strategy
  end state (parked in the RED shed) → score 0, no latches; overshoot (settled too
  deep, past the band) → latches bank at most 0.70, never success; short-stop 16+ cm
  before the sill → g3 never fires; fly-over (held-carried above the park spot,
  120 steps) → z-gate keeps g3 off throughout; sanctioned drive-in → success + score
  1.0; sill retention (wheels seated dead against the inner step, then 3 N outward for
  1 s) → never climbs the 12 mm step, stays ≥ 4.5 cm behind the sill line; recovery →
  the sanctioned servo re-parks and success returns; no accidental success anywhere
  else; final no-NaN. Records frames.npz.
