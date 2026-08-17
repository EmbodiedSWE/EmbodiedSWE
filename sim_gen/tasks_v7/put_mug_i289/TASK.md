# Task `put_mug_i289` — trap the ball under the flipped mug, then herd it into the green disc

## Seed provenance

Seed: `embodiedgen/put_mug`
(`sim_gen/RoboVerse/roboverse_pack/tasks/embodiedgen/put_mug.py`) — a Franka
pick-and-place: lift a mug off the table and set it down inside a marked
bounding box (`RelativeBboxDetector`, ±0.1 m, orientation ignored).

## What changed, and why it is strategically different

The seed (and every same-seed sibling read this session) treats the mug as
**cargo**: the mug itself must end up somewhere. Here the mug is a **tool**
and is worth nothing at the goal — a loose ball is the payload:

1. **Flip**: the mug must be turned mouth-down (inverted) — the seed ignores
   orientation entirely.
2. **Trap**: the inverted mug is dropped over a free ball so the rim ring
   cages it against the floor.
3. **Herd**: the capped mug is slid along the floor; its interior wall pushes
   the captive ball ahead of it until the ball sits inside a green floor
   disc. The ball is **never grasped, carried, or pose-written** — it travels
   only by contact with the mug's interior wall.

Success is judged on the **ball** (covered by the resting inverted mug AND
inside the disc, settled), not on the mug's own position. Contrasts with the
neighbours:

- vs **seed** (`put_mug`): goal predicate is about a second, never-carried
  object; the mug's final position is irrelevant except that it must still
  cover the ball; mug orientation (inverted) is load-bearing, not ignored.
- vs **`put_mug_i173`** (hang mug by its handle window on a wall peg):
  that is suspension/threading of the mug itself above the floor; here
  everything happens on the floor plane and the mug is a pushing tool.
- vs **`put_mug_i228`** (mug as upright receptacle catching a payload from a
  silo above): that mug is a passive upright container that must END
  upright in a dock; this mug is an active inverted pusher — smoke check 9
  explicitly rejects the i228 topology (ball inside an UPRIGHT mug standing
  in the disc scores ≤ 0.02).
- vs **`pen_holder`** (insert many items into a container): no insertion,
  single payload, and the "container" is upside-down and mobile.

## Teleport solution (solve.py phases)

Teleport is TRANSPORT ONLY (one pose write on the mug, both endpoints
contact-free); every load-bearing interaction is contact dynamics:

- **P0** reset + 0.5 s settle; mass readbacks; layout printout (seed-varying).
- **P1 TRANSPORT** (the only teleport): mug written inverted, hovering with
  its rim 30 mm above the floor, centred on the ball (19 mm radial
  clearance, no contact). The written state cannot judge: mug aloft
  (rest-band false), ball far from the disc, jump guard resets the streak.
- **P2 CAPTURE** (contact): gravity drops the mug 30 mm; the rim lands
  around the ball — genuinely trapped.
- **P3 HERD** (contact): horizontal external force at the mug CoM
  (velocity-regulated bang-bang toward the disc, 1.5→6 N stall escalation)
  slides the capped mug; the interior wall pushes the ball. Break on the
  BALL's own distance-to-disc readback.
- **P4** hands-off settle to `success()` (re-herd if the coast-out parks the
  ball marginally outside — still pure contact).
- **P5** ≥ 3 simulated seconds hands-off persistence, then
  `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at each phase boundary and is non-decreasing
(0.00 → 0.00 → 0.30 → 0.60 → 1.00 → 1.00): flip/capture/delivery latches are
streak-gated (10 still post-steps) so cold teleports can't latch credit.

## Embodiment argument (single Franka, parallel jaw, OSC)

Everything the teleport solution does maps onto one Franka standing at
base ≈ (−0.15, 0, 0) facing +x, workspace x ∈ [0.0, 0.75], |y| ≤ 0.45 —
all sampled layouts fit inside a 0.75 m reach envelope:

- **Flip + carry (replaces P1)**: pinch the mug rim/wall (wall thickness
  6 mm, body height 96 mm — a standard rim pinch), lift, rotate the wrist
  π about a horizontal axis (well within the wrist's range), carry to above
  the ball, lower to a ~30 mm hover.
- **Drop/press (P2)**: release, or press down gently; the capture funnel is
  forgiving — the ball may be anywhere within 19 mm of the rim-ring centre
  (interior radius 38 mm vs ball radius 19 mm), a loose visual-servo
  tolerance.
- **Herd (P3)**: push the capped mug horizontally at its wall or via the
  handle bar with the closed fingertips. Required force ≤ 6 N at CoM height
  48 mm — far below the ~82 mm tipping threshold at μ ≈ 0.5, so the mug
  slides without tipping and the trap holds; the Franka can exert this
  trivially in OSC.
- **Retreat (P4/P5)**: withdraw; nothing is held at success.

No bimanual squeeze, no regrasp-in-air, no force beyond a light push.

## Execution order

**Required, and physically enforced** — declared here:
flip must precede capture (an upright mug cannot cage the ball), capture
must precede herding (an uncovered ball squirts away instead of being
pushed; the `covered` predicate is part of success), and delivery is judged
only while the trap holds. The latches (`flipped → captured → delivered`)
are monotone along exactly this order and smoke verifies no shortcut
(bare-ball delivery without the mug scores ≤ 0.02).

## Scoring rubric

- 0.10 — mug latched resting inverted (streak-gated).
- +0.20 — ball latched trapped under the resting inverted mug.
- +0.30 — trapped ball latched inside the disc.
- Partial credit capped at 0.85; exactly 1.0 iff `success()` holds live
  (covered ∧ in-zone ∧ 45-step stillness ∧ speed bands ∧ no pose jump).

## Smoke check list (17)

1. reset settles, layout readback sane
2. fresh reset: score ≤ 0.02, not success
3. randomization: zone/ball move across seeds, disc body tracks `_zone_xy`, min separation holds
4. mug spawn varies, side flips, ≥ 0.15 from ball
5. null policy 240 steps: nothing happens
6. seed strategy (mug upright centred ON the disc) rejected ≤ 0.02
7. bare ball delivered to disc centre without mug rejected ≤ 0.02
8. capture-only far from disc: 0.28–0.32, not delivered
9. wrong topology (ball inside UPRIGHT mug standing in disc — the i228 end state) rejected ≤ 0.02
10. beside-rim: inverted mug in disc, ball 68 mm away (in zone, not covered): 0.09–0.12
11. near miss: real capture parked just outside the zone edge: captured, NOT delivered
12. fly-through: cold-written goal pose judged without stepping → not success
13. yank-out after real success-shaped run: success never fired mid-yank, latches persist (0.55–0.65)
14. held-aloft inverted mug judged cold → rejected, no latch
15. ball perched on the upturned mug base → not captured/delivered, ≤ 0.12
16. rejection audit: `success()` never fired during any rejection construct
17. no NaNs anywhere

Frames for all phases recorded to `frames.npz`.
