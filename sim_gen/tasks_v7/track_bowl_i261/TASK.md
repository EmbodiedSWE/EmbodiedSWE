# track_bowl_i261 — Covered Dish (seat the bowl, fill it, cap it with the RIGHT lid)

## Seed provenance

Derived from `pick_place/track_bowl`
(`sim_gen/RoboVerse/roboverse_pack/tasks/pick_place/track_bowl.py`): grasp a bowl,
carry it through free space along 121 interpolated waypoints, with a per-step
position+rotation tracking reward (K_POS=10, K_ROT=5) toward the current waypoint,
from a hardcoded initial state with `object_grasped` forced.

## Strategic difference

The seed is prescribed TRANSPORT: the entire task is following a dense trajectory
with the bowl in hand, and the reward is a tracking distance evaluated every step —
there is no contact goal at all; the terminal state is just wherever the path ends.

`covered_dish` keeps the bowl but inverts everything else. Nothing is prescribed and
nothing is tracked; the goal is a settled three-piece CONCENTRIC ASSEMBLY on a
fixture, every stage a contact-mated insertion with a real tolerance, judged in body
frames on the END STATE only:

1. **SEAT** — the bowl's 76 mm bottom boss must drop into the 90 mm octagonal socket
   of a pedestal stand (7 mm radial slack; stand-frame axis offset + height band +
   upright cone).
2. **FILL** — the 34 mm egg must come to rest INSIDE the seated bowl (bowl-frame
   containment on the bowl floor).
3. **CAP** — the LARGE lid must seat on the bowl: its octagonal centering skirt drops
   into the 104 mm mouth (8.7 mm worst-case radial slack) and its 132 mm cover disk
   rests level on the rim (bowl-frame concentric + height band + tilt cone).

A wrong-object trap makes identification part of the task: a DECOY lid of the same
color but smaller diameter cannot cover the bowl — dropped on the mouth it falls
INSIDE, and any decoy resting in the bowl pokes its tall knob above the rim plane by
construction (asserted in cfg), so the real lid then rides ~11 mm too high and can
never seat until the decoy is lifted back out by its knob.

A solver needs a different plan (multi-object assembly with fixture + payload +
cover identification) and different code (body-frame mating predicates instead of a
waypoint follower). The seed's whole strategy — carry the bowl somewhere and set it
down — is expressible here (smoke check 6 constructs exactly that, a genuine
gravity-seat of the bowl alone) and earns only the 0.20 seat credit, never success.

Distinct from every other task read this session:

- `track_bowl_i27` (bell herd): capture/herd a rolling ball under an inverted bell
  by force-dragging — no assembly, no wrong-object trap. Here nothing is herded and
  no forces are needed; all three mates are gravity drops.
- `track_banana_i79` (clamp release + crate catch): a stored-energy release and
  ballistic catch — nothing like a concentric insertion stack.
- `track_spoon_i160` (balance scale): mass identification by measurement; here
  identification is geometric (lid diameter) and the penalty is a physical block.
- `track_ceramic_teapot_i22` (mug hooks): hanging on cantilevered hooks — pendant
  rests, not nested insertions.
- `track_banana_i240` (ram-feed tunnel): forced translation through a constrained
  channel; here every mate is an open-air vertical drop.
- `pen_holder` exemplar (fill container + set down): single container filled with N
  pens; here the container itself must be fixtured, the fill is one payload, and the
  distinguishing stage is the lid mate + decoy trap.

## Rubric

- `success()` (current state): bowl seated AND egg inside AND real lid seated AND
  all dynamics settled (< 0.08 m/s — above GPU phantom-velocity artifacts).
- `score()`: per-substep latches — 0.20 × bowl ever seated + 0.25 × egg ever inside
  the seated bowl + 0.25 × full stack ever mated, capped at 0.70; exactly 1.0 iff
  `success()`. Null policy ≈ 0. Non-decreasing by construction.
- Honesty asserts in `CoveredDishCfg.__post_init__`: every tolerance covers ALL
  physically-mated rests (e.g. mouth-corner egg rests, any-yaw skirt seats) and
  rejects the nearest wrong rest with margin (collar perch, decoy-propped lid);
  the decoy provably falls through the mouth AND provably blocks capping; the
  seated lid's skirt always clears the egg; every grasp feature fits the jaw.

## Teleport solution (solve.py) — passes seeds 0 and 7

Teleports are TRANSPORT ONLY (always ending in a free-space hover); every
load-bearing interaction is gravity + contact:

- **P0** settle, layout readback (stand pose, slots, lid/decoy coin flip).
- **P1 SEAT** — hover the bowl over the socket (boss bottom 15 mm above the collar,
  4 mm off-axis; the whole fall path intersects nothing), drop: the boss lands on
  the socket floor. Readback: stand-local z = 0.036 (designed 0.036).
- **P2 FILL** — hover the egg 40 mm above the seated rim on the bowl axis, drop:
  it falls through the mouth, lands on the floor, settles at bowl-local z = 0.017.
- **P3 CAP** — hover the REAL lid with its skirt 12 mm above the rim, 4 mm off-axis,
  flats aligned, drop: the skirt self-centres in the mouth and the disk lands on
  the rim at bowl-local z = 0.055. The decoy is never touched.
- **P4** hands-off persistence 3.33 s (400 substeps); `SIM_GEN_SOLVE: SUCCESS` only
  if `success()` still holds. `SIM_GEN_SCORE` printed at every boundary
  (0.00 → 0.20 → 0.45 → 1.00 → 1.00), asserted non-decreasing.

## Embodiment argument (Franka, base ≈ (0.12, 0.0), facing +x)

All nominal object slots lie 0.20–0.60 m from that base (stand at ~0.50 m, scatter
slots 0.17–0.55 m), inside the ~0.855 m reach with elevation ≤ 0.24 m:

- **Bowl**: top-down parallel-jaw grasp on the 6 mm octagonal rim wall (55 mm deep
  jaws not required — rim protrudes freely; 80 mm jaw span ≫ 6 mm). Carry upright,
  hover over the socket, lower until the boss enters, release: the 7 mm radial slack
  and 10 mm xy tolerance absorb standard visual-servo error; the mate is yaw-free
  (round boss in an octagonal socket).
- **Egg**: direct pinch of the 34 mm sphere (jaw 80 mm), release 30–40 mm above the
  open 104 mm mouth — the drop is exactly the solve's P2.
- **Lids**: both are grasped by the 22 mm knob shaft under a 36 mm cap flange
  (hook-proof); identification is visual (132 mm vs 80 mm disk). Lower the real lid
  over the mouth and release with the skirt just above the rim; 8.7 mm worst-case
  radial slack self-centres it. If the decoy was dropped in by mistake, its knob
  stands ~11 mm proud of the rim — graspable for recovery.
- No stage needs force beyond carrying 0.3 kg parts; every mate closes under
  gravity, as demonstrated by the solve.

## Execution order

No order is imposed by the rubric beyond what physics forces: the egg cannot enter
a capped bowl (smoke check 9 demonstrates the lid genuinely seats on an empty bowl
and the egg then stays outside), and egg/assembly credit is gated on the bowl being
seated. Seat → fill → cap is the natural order; fill-then-seat (egg into the
grounded bowl, then carry both) is physically possible but riskier and earns egg
credit only once the bowl is seated.

## Checks (smoke.py — `SIM_GEN_SMOKE: ALL PASS 15/15`)

1. Settle/no-NaN: reset rest heights by readback, everything settled.
2. Score ~0 at reset, no success.
3. Randomization: stand xy + yaw vary; lid/decoy slot coin flip takes both values.
4. Randomization: bowl/egg/lid/decoy stand-local xy all vary (readback).
5. Null policy (240 steps): score ~0, no success.
6. SEED strategy (bowl alone carried + set down seated): 0.20 only, not success.
7. Fouled socket: egg debris in the socket → bowl rides 20 mm high, seat rejected.
8. Inverted bowl over the socket: rejected (upright cone + height band).
9. Egg dropped on the capped lid: lid genuinely seated, egg outside, asm never
   latches (physics forces egg-before-lid).
10. Lid rim-perch 45 mm off-axis: rejected by the concentric band.
11. Decoy trap: decoy falls INSIDE (readback z=0.012), real lid rides on its knob
    at z=0.066 vs band 0.055±0.004 → capping physically blocked.
12. Off-fixture dish: egg in + lid seated on the GROUNDED bowl → score 0, no
    success (the fixture clause is load-bearing).
13. Latched credit survives the egg being removed (0.45 unchanged, no success).
14. Rejection audit: success() never true at any judged point in the battery.
15. Final no-NaN.

Verified on the forge (RTX 4090, Isaac Sim 5.1): solve SUCCESS on seeds 0 and 7
(26 s each), smoke ALL PASS 15/15 with 262 recorded frames (frames.npz).
