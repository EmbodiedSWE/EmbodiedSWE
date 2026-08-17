# ball_corral (`hit_ball_with_queue_i210`)

Trap the white ball under an open-mouth cage seated flush on a tilted plateau, then
slide the caged assembly — the ball rolling captive inside — until the cage is
centred on the green goal disc, and leave everything at rest there.

## Seed provenance

- **Seed:** `rlbench/hit_ball_with_queue`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/hit_ball_with_queue.py`) — grasp a
  cue stick and STRIKE a free ball across an open surface into a pocket.
- **Seed strategy:** tool-mediated impulse transfer. One ballistic contact; aim and
  momentum decide the outcome; the ball travels FREE the whole way to the goal.

## What changed and why it is strategically different

The seed's whole theory of the task — *propel the payload and let it fly free* — is
inverted into *never let the payload travel a single centimetre uncontained*:

1. **Free ball travel IS the failure mode.** The work surface is an elevated plateau
   tilted 1.8–2.6° with open drop-off edges. A free ball can rest ONLY in one of two
   shallow recessed cradles; anywhere else it rolls downhill, off the edge, to the
   floor — where it is **unrecoverable**: the 85 mm ball is wider than the 80 mm
   Franka jaw span (asserted in `BallCorralSceneCfg.__post_init__`) and the plateau
   face is a 0.24 m vertical cliff no ball can be rolled up.
2. **Striking is the losing move.** Any impulse pops the ball over its 4 mm cradle
   lip and gravity takes it off the edge (smoke check 6 executes exactly the seed's
   plan and shows it scores ~0). The goal marker is bare tilted deck: even a
   perfectly aimed or perfectly gentle bare delivery cannot ever REST there (smoke
   check 7).
3. **The only winning plan is capture-then-escort:** lower the open-bottom cage
   straight down over the white ball until its rim seats FLUSH on the deck (a real
   contact seating: 17.5 mm free lateral clearance, with the ball's dome gravity-
   funnelling moderate misses into a centred capture — but a badly aimed drop,
   past the dome apex, sheds off the ball's shoulder and captures nothing, smoke
   check 10), then SLIDE the caged assembly quasi-statically across
   the deck — the captive ball climbing its cradle lip and rolling enclosed —
   until the cage is centred on the goal disc. The payload reaches the goal only
   ever inside a closed container being pushed; nothing is ever aimed or launched.
4. **Identity and orientation are load-bearing:** an identical-size orange decoy
   waits in the other cradle (Bernoulli-swapped each episode) and scores nothing;
   an inverted mouth-up cage used as a basket is rejected by the flush up-axis gate.

Distinct from the corpus siblings read this session:

- **hit_ball_with_queue_i77 / skyway_bridge** (same seed): there the robot builds
  INFRASTRUCTURE (seats a bridge) and the payload then travels the whole way to the
  goal HANDS-OFF under gravity — the robot never accompanies it. Here the payload
  moves ONLY while contained and escorted: sustained pushing contact end to end,
  no ballistic/hands-off leg at all, and the manipulated object (the cage) travels
  WITH the payload to the goal.
- **hockey_i325** (gate extraction + push the puck to a roofed goal): the payload is
  pushed BARE across the floor. Here a bare push is a guaranteed loss — the payload
  must be enclosed before it may move.
- **track_bowl_i27** (bell cage drag): there the payload starts already captive in
  its carrier. Here the CAPTURE itself is the first scored act — a real vertical
  seating around a free ball — and a badly aimed drop sheds off the ball's
  shoulder, captures nothing, and risks knocking the ball loose for good.
- **obstacle_i17** (ballistic launch): that is aim-and-impulse — this task's
  explicit losing move.

Loophole audit: delivering the empty cage scores nothing (delivery credit requires
the ball enclosed at that moment — smoke check 8); caging/delivering the decoy
scores nothing (check 9); a ball dropped on the cage roof is not "inside" (z-band
gate, check 11); the mouth-up-basket cheat is rejected (check 12); capture credit
is latched but capped at 0.30 without a real enclosed arrival (check 13). The ball
can never be grasped, lifted, or placed — every path to the goal that does not run
through the seated cage ends on the floor.

## Teleport solution outline (solve.py)

1. **P0** reset + settle; layout readback printed (seed-provable: plateau xy/yaw/
   tilt, zone, ball/cage locals); score ~0 asserted.
2. **P1 (teleport = transport only):** one pose write stages the cage HOVERING with
   its rim 45 mm above the deck, centred over the white ball with a deliberate
   (6, 4) mm lateral offset — asserted NOT caged.
3. **P2 (contact dynamics):** release → the cage falls under gravity, its rim lands
   on the deck AROUND the ball and settles flush; the scene's `caged()` readback
   (rim flush + upright + ball enclosed at deck height) confirms →
   `SIM_GEN_SCORE 0.30`.
4. **P3 (contact dynamics):** a horizontal push at the cage CoM — authored AT RIM
   LEVEL, so this is exactly a low fingertip push on the wall with no tipping
   moment — slides the caged assembly to the goal. Force law: a feed-forward along
   the course (2.5 N, escalating on stall to break stiction/the cradle lip) plus a
   full-vector velocity-error brake (k=45 N·s/m, target ≤ 0.10 m/s, 0.03 m/s while
   the captive ball climbs its 4 mm cradle lip) — self-capping even if the pod's
   external-force API rotates the applied wrench. Guards: a displacement-direction
   probe toggles world/body force encoding if the measured course deviates, a
   runaway speed guard cuts and coasts, and a climb guard backs off whenever the
   cage starts riding up on the ball. Force CUT 15 mm from the goal point.
5. **P4 (pure physics):** everything settles caged + centred + still →
   `SIM_GEN_SCORE 1.0`.
6. **P5:** ≥ 3.3 further simulated seconds hands-off; success must persist →
   `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge on **seeds 0 and 1** (rc=0; distinct layouts in the readback).

## Embodiment argument (single Franka, OSC, parallel jaw)

- **Base pose (plateau frame):** (x, y) = (−0.10, −0.52), beside the plateau's
  y edge. All contacts below are within ~0.73 m reach: the cradles at
  (0.08, ±0.13), the parking spot (−0.05, −0.20), the whole goal box
  x ∈ [−0.23, −0.14], y ∈ [−0.12, 0.12], all at deck height 0.24 m — a comfortable
  Franka workspace annulus.
- **Cage:** the 12 mm carry BAR sits raised on posts above the roof — a clean
  top-down parallel-jaw pinch (12 mm ≪ 80 mm opening) with free jaw clearance on
  both sides. Lift, carry over the ball, lower STRAIGHT down (17.5 mm free mouth
  clearance, and the ball's dome gravity-funnels moderate lateral error into a
  centred capture — OSC placement error of a few cm is forgiven), release when
  the rim takes the deck. The escort is a low fingertip/knuckle push on a wall face at
  rim level (the CoM is authored there; the solve's CoM force is exactly this
  push): 3–5 N overcomes cage friction (~1.7 N) + the ball's 4 mm lip climb
  (~1.2 N through the wall).
- **Balls:** 85 mm > 80 mm jaw span — never grasped, and never legitimately pushed
  bare (that loses). The white ball is only ever contacted BY THE CAGE.
- **Decoy:** never needs touching.

## Execution order (declared)

REQUIRED order: (1) seat the cage over the white ball, (2) slide the caged assembly
to the goal. The order is enforced by physics, not by the rubric: the goal zone is
bare tilted deck, so the ball can only ever BE at the goal inside the already-seated
cage; moving the ball first (bare) loses it off the plateau; delivering the cage
first arrives empty and there is then no way to bring the ball (it cannot be
grasped, and pushed bare it is lost). The rubric's delivery latch additionally
credits only an ENCLOSED arrival (caged ∧ in-zone at the same instant).

## Rubric

- `0.30 caged` (latched) — the WHITE ball ever enclosed under the flush-seated
  cage (rim at deck level, cage upright, ball inside at deck height).
- `0.35 delivered` (latched) — ever caged AND the cage centred within `zone_tol`
  (4.5 cm) of the goal point at the same instant.
- `1.0 iff success()` — caged ∧ centred ∧ still. Non-success cap 0.65. Null ~0.

## Checks (smoke.py — rejection-only battery, recorded to frames.npz)

1. settle/no-NaN: both balls at cradle rest height, cage parked flush, all still
2. reset score ~0, no success
3. randomization readback: plateau yaw + xy + TILT spreads real
4. randomization readback: cradle swap flips; zone moves and the marker disc
   tracks the stored zone; cage jitter + in-cradle ball jitter real
5. null policy (240 steps) → score ~0
6. SEED STRATEGY: white struck downhill at 1.4 m/s → pops the lip, off the edge,
   to the floor → score ~0 (striking loses)
7. bare ball at the goal: gently placed at rest on the goal half → rolls off the
   edge → a bare ball can NEVER rest at the goal
8. empty container: cage settled flush ON the zone (real readback) → no credit
9. wrong object: DECOY trapped under the flush cage ON the zone (full delivered
   geometry, wrong ball) → no credit
10. near-miss capture: cage dropped 75 mm off-axis (beyond the dome-funnel
    forgiveness) sheds off the ball's shoulder → never caged
11. roof gate: white dropped ON TOP of the closed cage → z-band gate holds
12. inverted container: mouth-up cage holding the ball at the zone → flush up-axis
    gate rejects (orientation load-bearing)
13. latched credit: probe-caged ball earns exactly 0.30, survives cage removal,
    never success (cap holds)
14. rejection audit: success() never True anywhere in the battery
15. final no-NaN

`SIM_GEN_SMOKE: ALL PASS 15/15` expected; solve.py separately proves acceptance.
