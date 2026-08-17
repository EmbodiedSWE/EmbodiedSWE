# insert_onto_square_peg_i268 — RingBalance

**Scene name:** `ring_balance` · **Env:** `simgen.ring_balance` (robot="null")

## Seed provenance

Seed task: `rlbench/insert_onto_square_peg` — "put the ring on the spoke": a
square ring is picked up and dropped over one square peg among several. One
pick, one insertion, judged by "is the ring on the (right) peg" — a pure
place-detector; a second, third or zeroth ring changes nothing.

## Strategic difference

The seed is satisfied by the *existence* of a ring on a peg. Here that exact
end state — one ring threaded, count ignored — is a **judged failure**: smoke
check 5 constructs it (one gold ring on the free post of a k=2 episode) and
the beam stays pressed on its ballast stop, no success. The pegs live on the
two pans of a working **beam balance** (spawn-authored revolute joint, hard
stops at ±12°, all restoring moment from real below-pivot mass), one pan
pre-loaded with a hidden-count stack of k ∈ 1..4 black ballast rings. The
judged outcome is a *mechanism equilibrium*: the beam floating level (|angle|
≤ 5.5°, 1 s settled streak) — which only the **exact count** of gold rings
produces:

- k−1 rings (the seed's "a ring is on the peg"): still pressed on the ballast
  stop (smoke 5);
- k+1 rings: tips onto the OPPOSITE stop — over-insertion is a real failure,
  not "more is better" (smoke 6);
- ring on the wrong post: adds ballast-side moment, zero credit (smoke 7);
- unloading the ballast to fake balance: beam levels but the intact-ballast
  gate refuses (smoke 8);
- threading parked BLACK rings: weight is weight, the beam levels — but
  success demands gold (n_free counts gold only, smoke 11).

So the insertion skill of the seed is still required (30 mm square bore over a
24 mm square shaft, stepped self-centering tip, ≤ ~18° relative-yaw admission)
— k times, onto a pan that *moves and swings* as it is loaded — but the task
is decided by counting / physical reasoning (count the stack, or read the
beam's response), not by a place-detector. No other examined tasks_v7 package
judges a balance-scale equilibrium or an exact-count objective.

## Assets (fully procedural)

- **stand** — DYNAMIC 30 kg pedestal (root MassAPI, CoM at ground level;
  dynamic because a joint anchored to a kinematic body0 stays world-fixed when
  teleported at reset): base 0.34×0.26×0.04, column, fork plates, visual axle
  pin at pivot_z = 0.35.
- **beam** — DYNAMIC rotor, origin ON the axle (pure-quat angle writes; the
  stand+beam linkage is always written together). SHORT crossbar ending at
  |x| = 0.100 (a 45°-yawed descending ring reaches 0.1105 inboard — the
  airspace above each post is completely open for the full bore travel), a
  hanger dropping at each bar end, an under-arm running BELOW the pan floor to
  each square-rimmed PAN at |x| = arm = 0.16, each with a square centre POST:
  24 mm shaft + 20/13 mm stepped funnel tip. Spawn-authored
  `UsdPhysics.RevoluteJoint` to the stand, axis Y, joint limits ±12° ARE the
  stops (no contact stops → nothing to wedge or prop). Angular damping 3.0.
- **rings** — 9 identical square rings (70 mm outer, 30 mm bore, 12 mm thick,
  steel density → 0.377 kg each): 5 GOLD (supply, scattered on a floor arc),
  4 BLACK (ballast; k on the ballast post, the rest parked far off-field).

Randomization (readback-verified, smoke 2–3): stand xy ±50 mm + yaw ±25°;
ballast side flip; k ∈ 1..4 (all four values observed over 12 seeds); gold
scatter slots + jitter + yaw. Every load-bearing geometry/statics claim is
asserted numerically in `RingBalanceSceneCfg.__post_init__`: one-ring
imbalance always presses a stop; worst-case bore-slack bias + measured settle
overshoot stays inside the success band; funnel/bore/step slacks clear the
speculative-contact envelope; a full stack stays captured at the stop angle;
the descent corridor above each post is open (bar end vs 45°-corner reach);
pans/posts clear the stand fork at both stops.

## Rubric

- 0.10 `lifted` — any gold ring ever above 0.15 m (latched)
- 0.50 `load` — latched max of min(n_gold_on_free_post, k)/k (over-loading
  never earns beyond k; progress never evaporates)
- non-success capped at 0.60; **1.0 iff success()**: |beam angle| ≤ 5.5° with
  beam/rings/stand settled for a continuous 1 s streak (the streak gate
  rejects a beam swinging through level — smoke 9), ballast stack intact on
  its post, ≥ 1 GOLD ring on the free post, all finite — judged live
  (revocation: smoke 10). Null policy ~0 (beam pinned on its stop, nothing
  lifted — smoke 4).

The 5.5° band is set above the natural hands-off settle envelope (worst-case
bore-slack bias ≈ 2.6° plus the forge-measured decaying overshoot ≈ 2.6°
grazes ~5.3°) and far below the ±12° stops that any wrong count presses.

## Teleport solution (solve.py) — the legitimacy certificate

- **P0** reset, 240-step settle; layout readback (k, side, pressed angle) and
  mass readbacks (ring 0.3768 kg, stand 30 kg — custom spawners apply no cfg
  mass schemas, so masses are asserted); free post empty; score ~0.
- **Per ring j = 1..k** (only k of the 5 gold rings are ever touched):
  - *(applied force)* PD force + gravity feedforward (kp=15, kd=6, clamp
    7.5 N — sized to the 1-substep wrench-delay bound) lifts ring j off the
    floor to z = 0.24; `lifted` latches during the force lift;
  - *(transport)* ONE root-state write to a hover 60 mm above the free post's
    funnel tip, bore aligned — free air, nothing judged satisfied (asserted:
    not on post);
  - *(guided descent)* the same force-limited carry lowers the ring; a mod-90°
    yaw servo holds bore-over-shaft alignment while the stepped tip funnels
    the bore; contact does the threading. Escalating retry (4 attempts,
    longer/stronger) with a recovery lift if a descent stalls;
  - release, 180-step hands-off settle; asserts: captured on the post, ballast
    intact, and — while j < k — the beam still PRESSED on its stop (no early
    credit) with success False. Score after j rings: 0.10 + 0.50·j/k.
- **P(k+1) equilibrium (pure mechanism physics)** after the k-th release the
  beam ALONE swings up and settles level; the 1 s streak accumulates entirely
  hands-off; asserts |angle| ≤ 5.5°, score 1.0.
- **Pfinal** ≥ 3.3 simulated seconds hands-off with a per-step diagnostic that
  prints any settle-component break; success must hold at every checkpoint →
  `SIM_GEN_SOLVE: SUCCESS`.

Teleports are transport-only: lifting, every threading, and the balance
equilibrium are produced by applied forces, contacts, gravity and the hinge.

## Execution order

Declared order: fetch → thread, k times, then hands-off equilibrium. The
*count* is the real constraint and it is order-free but exactness-forced by
physics: the stops are joint limits (nothing props an airborne pan), only mass
on the free post moves the settled angle, and the success band is reachable
only at n = k (n < k and n > k both press a stop — smoke 5/6). Partial credit
latches per ring; full credit exists only at the exact count.

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin; the stand stands 0.35–0.45 m out (nominal
xy ± 50 mm, yaw ±25°), gold rings scattered on a 0.42 m arc in front — all
within a 0.85 m reach envelope. Each pick is a pinch of a 70 mm square ring
lying flat (12 mm thick — a standard top-down edge grasp with 40 mm-wide
Franka jaws). Each place is a lower of the held ring over a post whose funnel
tip + 6 mm/side bore slack forgive several mm and ~18° of yaw error — exactly
the tolerance the solve's guided descent certifies; the highest seat is at
world z ≈ 0.30, the hover at ≈ 0.41 — comfortable workspace. The pan swings
under load by design ≤ 12°, and the solve shows a release from 60 mm above
the tip threads reliably. Counting requires no manipulation: the black stack
is visible from the front (4 × 12 mm rings), or the robot can iterate using
the beam's response — thread, watch, repeat — which the rubric supports
(per-ring latched credit, no ordering trap). No step needs a second arm, a
regrasp in flight, or simultaneous contacts.

## Files

- `scene.py` — cfg (+ 11 numbered statics/geometry asserts), spawners (stand,
  beam + revolute joint, rings), scene (rubric, latches in `post_step`),
  `register_env`.
- `solve.py` — phased solution (force fetch, transport write, force-guided
  contact threading × k, hands-off equilibrium + persistence diagnostics);
  watchdog + hard exit.
- `smoke.py` — 13-check rejection battery (settle, randomization readback,
  flag coverage, null policy, SEED-strategy/under-load, over-load,
  wrong-post, ballast-removal cheat, streak gate, positive control,
  revocation, wrong-object black rings, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 38.1 s; k=4 on the −x
  pan; scores 0.000 → 0.225 → 0.350 → 0.475 → 0.600 → 1.000, non-decreasing;
  beam pressed at −12.0° through ring 3, swung to −4.0° settling after ring
  4, persisted 3.3 s hands-off with streak 520)
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 25.2 s; k=1, +19.3°
  stand yaw; balanced at −0.27° with streak 520)
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 13/13` (rc=0, 96.3 s; k values
  {1,2,3,4} and both sides observed; seed-strategy ring left the beam at
  +12.0°; over-load tipped to −12.0°; ballast cheat leveled at +0.08° with
  success False; wrong-object black rings leveled at +0.18° scoring 0.000;
  revocation re-tipped to +12.0° at the 0.60 cap; 362 frames saved)
