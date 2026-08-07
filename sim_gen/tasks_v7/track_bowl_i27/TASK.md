# track_bowl_i27 — Bell-Herd: cage, drag, deliver, reveal

## Provenance

Seed: `roboverse_pack/tasks/pick_place/track_bowl.py` (`pick_place/track_bowl`).
The seed grasps a bowl, lifts it, and carries it through free space along a dense
interpolated waypoint trajectory; reward is per-step position+rotation tracking of
the current waypoint, and the bowl itself is the judged payload in the hand the
whole way.

## Strategic difference

This task keeps the seed's central object — a bowl-shaped body — and inverts its
role and the entire plan:

- **Payload -> tool.** The bowl becomes an open-mouthed, knob-topped BELL. It is
  the only graspable object in the scene, but it is judged only as a tool: where
  it ends up matters only in that it must be parked *clear* of the goal.
- **Graspable payload -> ungraspable payload.** The judged payload is an 85 mm
  ball, wider than the 80 mm parallel-jaw span (asserted in cfg): it can never be
  held, only caged under the bell or nudged.
- **Prescribed trajectory -> free-route terminal state.** There is no waypoint
  list and no per-step tracking. The plan is CAPTURE (lower the bell over the red
  ball; the mouth swallows it with 32 mm slack) -> HERD (drag the loaded bell
  flat across the board; the skirt wall pushes the rolling ball through contact —
  the bell never leaves the surface while loaded) -> DELIVER (steer the cage over
  a sunken 107 mm pocket; the ball falls through under gravity; the bell's rim
  ring out-spans the pocket diagonal, so the tool itself can never fall in —
  asserted) -> REVEAL (lift the empty bell away and park it > 0.20 m from the
  pocket, else the delivery does not count).
- **Distractor with irreversible failure.** A blue twin ball must stay out of the
  pocket; dropping it in cannot practically be undone (no ball can be grasped,
  and it plugs the hole — smoke check 8).

A solver therefore needs a different plan (tool acquisition, containment,
contact-dragging, gravity delivery, tool removal) and different code (containment
predicates and drag control instead of a waypoint follower). Against the rest of
the corpus: no other task uses an enclosure as a *movable capture tool* dragged
along the support surface — enclosure elsewhere is a goal state (smothering), and
transport elsewhere is carrying the payload itself.

## Scene

Procedural only. One kinematic fixture (0.70 m fenced board, play surface 80 mm
up, square pocket sunk 60 mm with a yellow floor at board-local (0.20, 0)); one
dynamic bell (octagonal skirt, inner mouth inradius 75 mm, 0.5 kg, damped, 22 mm
grasp knob with 36 mm cap flange); red target + blue distractor balls (85 mm,
150 g). Randomized per episode (readback-verified): board xy + yaw, coin-flip of
which y-side holds the red ball (blue and bell mirror), xy scatter for all three
movables with pocket/mutual keep-outs, and the progress baseline d0.

## Rubric

Judged in the board's body frame, only when settled (< 0.05 m/s):

- `success()` = red ball inside the pocket (xy within the square, centre below
  the surface) AND blue ball not in the pocket AND bell axis > 0.20 m from the
  pocket centre.
- `score()` (latched every physics substep, so transient progress keeps credit):
  0.15 * ever-caged + 0.25 * best pocket progress (normalized by the episode's
  own spawn distance) + 0.30 * ever-in-pocket, capped at 0.70; exactly 1.0 iff
  `success()`. Null policy ~0; the seed's strategy (carry the bowl and set it
  somewhere) ~0.

Geometric honesty is asserted in `BellHerdCfg.__post_init__`: ball defeats the
jaw; cage slack (32.5 mm) exceeds closed-loop arm precision; skirt fully
encloses the ball; ball drops through the pocket with >= 10 mm side clearance;
bell rim ring always bridges the pocket; a caged ball centred on the pocket is
centred on the hole; in-pocket vs on-surface heights cleanly separated; fence
contains a rolling ball; `bell_clear` puts the whole bell off the pocket.

## Solution outline (solve.py)

Teleports are transport-only, ending in free space or a non-contact hover; every
load-bearing interaction is contact dynamics:

- **P1 cage** — bell teleported to a hover centred on the red ball, mouth 25 mm
  above the surface (the 150 mm mouth around an 85 mm ball intersects nothing),
  then DROPPED; the cage closes by gravity + rim contact.
- **P2 herd** — external forces on the bell only (floating-hand drag): a carrot
  moves at 0.10 m/s along the straight board-frame line from the caging point to
  a target 5 cm PAST the pocket centre (sliding friction leaves the PD a
  ~mu*g/kp steady-state lag, and the ball leaning into the hole rests against
  the bell's front inner wall, which must advance past centre to release it —
  overshoot is safe because the loop breaks on delivery and the bell can never
  fall in); PD force (kp 240, kd 20 per kg, clamp 8 N, zero z) plus a weak
  upright torque servo. World wrench pre-rotated by q_ref * q_now^-1 (this
  stack applies external wrenches in a rotating body frame). The skirt pushes
  the rolling ball; when the cage crosses the pocket the ball falls through
  under gravity. No force ever touches either ball.
- **P3 reveal** — forces cleared; the empty bell teleported up off the board and
  parked at board-local (-0.22, +0.20 * side) — the red ball's vacated side,
  clear of both the pocket and the blue ball — then dropped 60 mm and settled.
- **P4 persistence** — 3.3 simulated seconds hands-off; success must hold.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing;
verdict `SIM_GEN_SOLVE: SUCCESS`. Verified on the forge on seeds 0, 1, 2
(covering both mirror sides of the coin-flip randomization).

## Embodiment argument (single Franka + parallel jaw)

Mount the Franka base on the floor at world (-0.05, 0, 0) facing +x; the board
centre is at world (0.40, 0) with its surface at 0.08 m, so the whole 0.70 m
field sits inside the ~0.85 m reach envelope at a comfortable height.

- **Bell**: grasped by the 22 mm knob shaft (jaw span 80 mm closes to 22 mm with
  wide margin); the 36 mm cap flange above the shaft makes the grasp hook-proof
  against the downward drag loads. Caging = lower over the ball and release
  (25 mm drop — the same motion the solution certifies). Herding = re-grasp the
  knob and slide the bell along the surface; the 0.5 kg damped bell glides at
  the ~8 N lateral forces used in solve, well inside Franka payload/force limits.
  Reveal = lift 60 mm and place 0.4 m away — a plain pick-and-place.
- **Balls**: never hand-contacted. 85 mm > 80 mm jaw span by design; the fence
  keeps them on the board, so no recovery beyond reach is ever needed.
- All interaction heights are 0.08–0.22 m above the floor in a 0.7 m disc —
  no wrist gymnastics, no bimanual need.

## Order requirement

NO required execution order is declared. Any route and any phase interleaving
that reaches the terminal state counts; the rubric judges only the settled end
state (plus latched partial credit). The blue-ball clause is a state constraint,
not an ordering constraint — though in practice plugging the pocket with the
blue ball first is unrecoverable, which smoke check 8 demonstrates.

## Checks (smoke.py — rejection battery, recorded)

14 checks: settle/no-NaN + score~0 (2), randomization readback over 6 seeds
(board xy/yaw + both sides; red/blue/bell xy + d0) (2), null policy (1), seed
strategy = bell itself set down on the pocket bridges the hole and scores ~0 (1),
near miss at the pocket edge (1), wrong object: blue plugs the pocket and blocks
the red (1), covered delivery rejected by the reveal clause (1), latched credit
survives removing the delivered ball (1), cage alone is only partial credit (1),
progress monotonicity (1), success-never-True audit (1), final no-NaN (1).
Verdict `SIM_GEN_SMOKE: ALL PASS 14/14`.
