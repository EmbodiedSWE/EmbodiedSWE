# stack_wine_i202 — Hang the Carafes on the Keyhole Rail (scene `bottle_hang_rail`)

Three identical green carafes (56 mm cylinder body, 14 mm neck, 34 mm tan collar
disc) stand on the floor in front of an overhead rail: a walnut plate 300 mm up on
two corner posts, pierced by three KEYHOLES — a wide 46 mm square PORT at the front
edge continuing into a narrow 24 mm SLOT running to the back. The collar passes the
port but NOT the slot; the body passes neither. Success = every carafe HANGING below
the rail by its collar, one per keyhole: collar seated on the plate top at least
15 mm down the slot (past the lift-out region), carafe plumb, body below the plate,
everything settled. The only way in is the three-stage keyhole maneuver — raise the
collar up through the port, slide the neck down the slot, lower and release. A
carafe released while still at the wide port simply falls back through (physics
rejects it); a bottle laid ON TOP of the rail — the seed's whole strategy — is
explicitly rejected by the rubric.

## Provenance

- **Seed:** `rlbench/stack_wine`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/stack_wine.py`) — grasp THE wine
  bottle and LAY it on THE rack: one pick, one support-from-below set-down, no
  mechanism between grasp and goal.
- **Files:** `scene.py` (cfg + scene + rubric, registered as scene
  `bottle_hang_rail`, env `simgen.bottle_hang_rail`, robot `"null"`), `solve.py`
  (teleport-transport + force-servo solution), `smoke.py` (rejection battery), all
  procedural geometry — no external assets.

## Strategic difference (vs the seed and vs every task read this session)

- **vs the seed:** the seed's goal contact is "object rests ON TOP of the fixture"
  and its plan is approach-grasp-carry-set — the rack imposes no constraint on the
  path. Here the goal contact is INVERTED (the fixture holds the object from ABOVE,
  by a collar hanging on a plate) and the approach is gated by a keyhole mechanism
  that forces a three-stage plan PER OBJECT: insert-through-aperture (collar up
  through the port), captive slide (neck down the slot the collar cannot pass), then
  hanging release. The final state is reachable only through that gate: releasing at
  the port drops the carafe straight back through (smoke check 7 constructs it), and
  the seed's strategy transplanted verbatim — all three carafes laid on top of the
  plate, settled — is physically built by smoke check 6 and scores <= 0.20, never
  success. Place-on-top versus thread-through-and-hang; one set-down versus
  insert -> slide -> release, three times.
- **vs the corpus task read in full this session:** `stack_wine_i48`
  (`cask_weight_sort`) derives from the same seed but its load-bearing problem is
  EPISTEMIC — visually identical kegs with hidden shuffled masses must be probed and
  sorted; its motor act is a plain drop into an open V-cradle. Here there is no
  hidden state at all: everything is visible, and the load-bearing problem is the
  MECHANISM — a multi-stage, geometry-gated insertion in which each stage
  (port alignment within 6 mm radial, captive slide past the lift-out bound, release
  onto the seat) only makes sense given the next. Measure-then-place versus
  thread-a-keyhole; i48's rubric judges an assignment, this rubric judges a
  retention state that only the mechanism can produce.
- **Execution order (declared):** within one carafe the order
  insert -> slide -> release is mechanically forced by the keyhole itself, not by
  the rubric; across carafes any order and any carafe->keyhole bijection is
  accepted (success requires only that all three keyholes end up used, which
  physics already forces since one keyhole cannot hold two collars — smoke
  check 10). There is no hidden ordering constraint.

## Randomization (per episode, verified by READBACK in smoke)

Rail pose: xy jitter (±30/±20 mm) + yaw ±8°; the rubric is judged in the RAIL's
frame so all keyholes move and rotate together. Carafe spawns: three floor slots
permuted per episode + xy jitter (±20/±30 mm) + free yaw (carafes are symmetric).
Smoke check 3 reads back rail x/y/yaw and carafe xy across 6 seeded resets: rail
spread 42/17 mm, 7.3°, and ≥ 3 distinct spawn x-orderings.

## Rubric

`success()` iff, judged live on physical poses in the rail's frame, for EVERY
carafe (with the three occupied keyholes distinct):

- collar centre within 8 mm of its slot's centreline (x) — the channel physically
  bounds a hanging neck at 5 mm, so every true hang passes by construction;
- collar centre 15–34 mm down the slot (past the port/channel datum) — the collar
  can lift back out only while over the port (bound ≤ −17 mm), so the band starts
  ≥ 25 mm beyond escape (asserted in `__post_init__`);
- collar centre in the seat band 4–18 mm above plate mid-thickness (resting on the
  plate top at 10 mm; rejects standing-on-floor at −140 mm, on-top-of-plate at
  ~+35 mm and above, and anything aloft);
- carafe plumb (axis within 15° of vertical) with the body centre ≥ 60 mm below the
  plate (rejects any on-top or through-port spoof);
- settled (|v| < 0.10 m/s — above the GPU phantom-velocity artifact, the tight
  position window does the work — |ω| < 0.60 rad/s).

`score()` (latched every physics substep in `post_step`): per carafe `0.05 ×` ever
lifted to rail height + `0.10 ×` collar ever inside a keyhole above the plate
+ `0.10 ×` ever in the full hang clause, slow, for 24 CONSECUTIVE substeps, capped
at 0.75; exactly 1.0 iff `success()`. Doing nothing scores ~0; the seed strategy
(all three on top of the plate) is capped at 0.15–0.20.

## Teleport solution (`solve.py`) — transport only, every load-bearing act by contact

Per carafe (assigned keyhole = its index): **T** teleport to a hover in free air
28 mm BELOW the plate under the assigned port, upright, velocities zeroed (never
into the keyhole, never hanging); **A** a CoM force + uprighting-torque servo
converges the collar onto the port centreline (< 3.5 mm); **B** a tapered
velocity-servo rise (≤ 0.10 m/s) carries the collar up THROUGH the port — real
geometry, with a stall detector that backs off if the collar butts the plate;
**C** a capped 0.9 N pull slides the neck down the channel until it stalls on the
channel's far end (along ≈ 26 mm, inside the hang band — the channel walls, not the
servo, provide the guidance); **D** release only once slow: all wrenches zeroed,
the carafe drops the last 2.5 mm and hangs by collar-on-plate contact, verified by
rubric readback with full retries. Wrench-frame discipline per the forge pods'
rotation-drag quirk: identity-orientation teleports + plumb hold keep R ≈ I,
commands are pre-encoded into the current body frame, and stage A doubles as a
divergence probe that flips the encoding. Gains are audited in the docstring
(all discrete-stability ratios ≪ 1). `SIM_GEN_SCORE` at every phase boundary is
non-decreasing (0.00 → 0.25 → 0.50 → 1.00, success turning true at the third
hang), ≥ 3.3 simulated seconds hands-off persistence, then `SIM_GEN_SOLVE:
SUCCESS`. **Verified on the forge: seeds 0 and 1, both SUCCESS, provably distinct
by stdout readback** — seed 0: rail at (−0.006, +0.019) yaw −7.7°; seed 1:
(+0.023, +0.007) yaw −4.0°; different spawn permutations; every hang lands at
along = +26.3 mm, collar z_loc = +10.0 mm (the seat), first attempt, all six.

## Embodiment sanity (single-arm Franka feasibility)

Base at roughly (0.0, −0.60): carafe spawn row at y ≈ −0.30 (x ∈ ±0.17) and the
three hang points at y ≈ 0.02, z = 0.30 (x ∈ ±0.15 with jitter) all sit within a
~0.75 m reach envelope at comfortable heights. Per-object contact strategy: a
simple side grasp of the 56 mm body with the 80 mm jaw (24 mm margin; 0.3 kg ≪
3 kg payload), carry upright, position under the chosen port — the standing carafe
is 158 mm tall against a 288 mm plate bottom, so there is 130 mm of free air below
the plate for wrist and fingers, and the hand NEVER has to enter the keyhole: the
jaw holds the body, whose top stays ≥ 35 mm below the plate even at full insertion
(only neck and collar pass through). Then: raise ~150 mm (collar through the port —
6 mm radial clearance against a 46 mm opening is generous for a guarded vertical
move), slide ~50 mm toward the back until the neck stops at the slot end (the slot
itself funnels the 14 mm neck with 5 mm slack per side — compliance, not
precision), lower ~3 mm and open the jaw; the collar catches on the plate and the
carafe hangs. The rail is kinematic and cannot be knocked over; hung carafes hang
54 mm apart from their neighbours' bodies, leaving a clear approach corridor for
the remaining ports.

## Checks (`smoke.py` — rejection battery, 14 named checks, ALL PASS on the forge)

1. settle: states finite; all three carafes standing on the floor at CoM height
   (readback z), settled.
2. settle: score ~0 at reset (≤ 0.02), no success.
3. randomization readback: rail xy + yaw vary across 6 seeded resets; carafe spawn
   xy and x-order (slot permutation) vary.
4. hanging principle is physical: PhysX mass readback matches the authored mass,
   and a carafe CONSTRUCTED into a mid-band hang stays hanging through 2 s of real
   contact (the collar cannot pass the slot); one hung of three → NOT success.
5. null policy: 240 idle steps → score ~0, no success.
6. SEED strategy: all three carafes laid ON TOP of the plate (readback z ≈ plate
   top + body radius), settled — NOT success, score ≤ 0.20 (support-from-below is
   exactly what this task rejects).
7. released-at-port: a carafe released at the wide port falls straight back through
   (collar ends 140 mm below the plate, carafe back on the floor) — NOT hung, NOT
   success, and the 24-substep hang latch stays 0.
8. shallow-hang near-miss: physically hanging by its collar (seat-band readback)
   but only 8 mm down the slot, still in the lift-out region — NOT hung, NOT
   success (depth is what makes the hang captive).
9. floor-below near-miss: standing on the floor directly under its keyhole, x- and
   slot-depth-aligned — only the seat-band clause rejects it — NOT success.
10. two-in-one-keyhole: a second carafe dropped from above the plate onto an
    occupied keyhole never hangs there; the first stays captive — NOT success.
11. partial completion: exactly one carafe hung → NOT success, score ≈ 0.25 (one
    carafe's stage credit, no more).
12. latched credit: removing the one hung carafe leaves the latched score unchanged
    (0.25 → 0.25), live hang gone, success stays gone.
13. rejection audit: success() never True at any judged point in the battery.
14. final no-NaN. Plus `frames.npz` (332 frames) recorded and saved in CWD.

Cfg `__post_init__` additionally asserts the geometry that makes the task honest:
the collar passes the port but not the channel, the neck slides the channel freely,
the body passes neither (no drop-through), the hang band starts ≥ 25 mm beyond the
collar's lift-out bound and is shorter than a collar diameter (one carafe per
keyhole), a standing carafe fits under the plate with insertion headroom, a hanging
carafe swings clear of the floor, the x tolerance admits every physically hanging
carafe, carafes are trivially graspable and liftable, and the spawn row stays clear
of the rail under worst-case jitter.
