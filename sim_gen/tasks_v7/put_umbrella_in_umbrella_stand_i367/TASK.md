# put_umbrella_in_umbrella_stand_i367 — bayonet-park the umbrella runner

Env id: `simgen.umbrella_bayonet` · robot: `null` · seed task: `rlbench/put_umbrella_in_umbrella_stand`

## Provenance and strategic difference

The RLBench seed is a free-space pick-and-insert: lift a free-standing umbrella and drop
it into the open throat of a floor stand. This task keeps the umbrella/stand *theme* and
inverts every mechanic:

- **The umbrella is never free.** It is a RUNNER permanently captive on the mast — a
  spawn-authored generic D6 joint (transZ 0–0.42 m travel + free rotZ; all other axes
  locked) means there is nothing to pick up and no insertion event at all. The
  manipulation vocabulary is *hoist + twist on a constrained track*, not transport.
- **A removable interlock gates the track.** A loose steel TRANSIT PIN rests in a
  through-channel in the pole at 24.3 cm; its protruding tails overlap the collar wall on
  both sides for every sampled seat depth, jamming the hoist at q≈0.058. The pin is held
  by nothing but gravity — it must be *slid out horizontally* before the track is usable.
  The seed has no interlock, no tool-like sub-object, and no forced ordering.
- **The goal is a bayonet park, not a drop-in.** The GREEN gallery shelf at 52 cm has a
  square hole that passes the collar at any yaw but passes the long YELLOW lug only
  through ONE open gap sector (+x). Above the shelf the runner must be twisted ≥55° so
  the lug overhangs the ring, then lowered until the lug *rests on the shelf top* and the
  whole mechanism is still, hands-off. Released un-twisted over the gap it falls all the
  way back down (gravity runs the runner to the bottom stop). The seed's success is
  containment in a cavity; here it is a keyed, orientation-dependent gravity catch.
- **A decoy tests object identity.** An identical SPARE pin lies on the floor; moving it
  achieves nothing (smoke check 13).

Difference from the sibling variants read this session: i220 parks an extracted umbrella
in a cradle (extraction + free transport; here nothing is ever free), i72 hangs by a
crook (hook topology; here a keyed lug + twist), pen_holder is free pick-and-insert, and
open_window_i283 is a gravity-pawl sash (ratchet one DOF; here a 2-DOF heave+yaw bayonet
with a removable interlock). No other task in tasks_v7 combines a captive 2-DOF runner, a
gravity-seated removable lock pin, and a keyed-aperture over-center park.

## Physically forced order

1. Pin out first: with the pin seated, the collar jams at q_block=0.0575 «« q_pass=0.37
   (smoke 6 drives the hoist servo against the seated pin and verifies the jam).
2. Align before hoist: the lug tip (75 mm) cannot pass the shelf plate anywhere but the
   +x gap (admission half-angle 29.6°).
3. Twist before park: released over the gap un-twisted, nothing supports the lug — the
   runner free-falls to the bottom stop (smoke 8). The unique support surface at
   q_park=0.359 is the shelf top, reachable only ≥55° away from the gap heading.

## Solution outline (transport-only teleports = none; all interaction via wrenches)

P0 settle + layout/readback asserts → P1 pull the pin +x with a HELD carry (mg
feedforward + z/y PD at the CoM — a bare horizontal pull drawer-jams once the CoM passes
the channel-floor edge; the held pull models a pinch grasp holding the pin level), wrench
off, pin drops clear (score 0.30) → P2 yaw-servo the lug to the gap heading → P3 heave-
servo (gravity-ff force) to q=0.405, lug rises through the gap (pass latch, score 0.60)
→ P4 twist +90° above the shelf → P5 lower onto the shelf ring, wrench off, ring-down →
success=1.0 → P6 400-step hands-off persistence → `SIM_GEN_SOLVE: SUCCESS`.

## Rubric (scene.py)

- Latched (anti-teleport) credit in `post_step`: `pin_latch` (pin ≥0.10 m off-axis,
  0.30) + `pass_latch` (q ≥ 0.37, 0.30), clamp 0.60.
- `success()` = parked (|q−0.359| ≤ 6 mm) ∧ aligned (lug heading ≥55° from gap) ∧ both
  latches ∧ settled (pose-stillness streak ≥30 steps). Score 1.0 iff success.
- The pass latch (0.37) sits ABOVE the park height (0.359): a teleport-to-park never
  fires it (smoke 12).

## Franka embodiment

- Transit pin: pinch the 24 mm RED knob (cube, exposed on +x, 24 cm up) and pull
  ~75 mm horizontally — a wrist-only linear slide, well inside the workspace.
- Runner: grasp the 16 mm-thick YELLOW lug plate side-on; slide 0.36 m up the pole
  (prismatic follow), orbit the grip ~90° at r≈0.05 m about the pole axis (wrist yaw),
  lower 46 mm, release. All grasp faces are ≥12 mm boxes; forces in the solve are
  ≤6 N — trivially within Franka payload.
- Base pose ≈ (0.55, 0, 0), facing −x: pin channel, gap sector, and the full hoist
  line are on the near side; max reach needed ≈0.75 m at z ≤ 0.60 m.

## Checks

- solve: seeds 0 and 1 on forge, `SIM_GEN_SCORE` non-decreasing 0 → 0.30 → 0.60 → 1.0,
  ≥3 s hands-off persistence after first success.
- smoke: 16 named checks — settle/no-NaN + layout, fresh-reset score ≈0, randomization
  readback (runner yaw >0.8 rad spread, pin seat >2 mm, spare >40 mm over 6 seeds), null
  policy, blocked hoist against the seated pin (vacuous-guarded), honest pin-pull credit
  0.30, untwisted release falls, 45° near-miss parks-but-not-aligned, below-shelf twist
  falls, latched credit persists, teleport-to-park rejected, wrong-object (spare) ≈0,
  settle gate (judged mid-ring: everything true but settled=False → no success),
  rejection audit (`ever_success` only in the honest runs), final no-NaN; frames.npz
  recorded.
- All ~22 `__post_init__` geometry asserts replicated and verified in standalone
  arithmetic before submission (pin-jam coverage for all seat jitter, keyed-aperture
  pass/block bounds, park uniqueness, hoist clearances).
