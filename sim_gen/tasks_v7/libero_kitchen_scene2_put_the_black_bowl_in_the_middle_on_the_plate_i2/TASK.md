# spindle_serve — serve the MIDDLE ring of a dowel-stacked pile; re-thread the rest (i2)

**Env name:** `simgen.spindle_serve` (scene `spindle_serve`, registered with robot `null`;
solve.py and smoke.py build the same scene-level env).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate.py`)
- Seed plan: among three identical loose bowls on a table, select ONE by TABLE POSITION
  ("the middle"), grasp it, carry it across free space, set it down on a flat plate.
  Success = one geometric relation (bowl within 6 cm of the plate axis), reached by a
  single unconstrained grasp-carry-place.

## What changed, and why it is strategically different

The "middle of three, onto the plate" premise is kept; everything that made the seed's
one-step plan sufficient is removed:

1. **"Middle" is a STACK-ORDER percept, not a table position.** The three candidates
   (square napkin rings: red, yellow, blue, order resampled every episode) are threaded
   onto a vertical dowel. The target is the ring with exactly one ring above and one
   below — identifying it requires reading the stack, not comparing table coordinates.
2. **Access is mechanically ORDERED.** Rings enter and leave a dowel only over its tip.
   The target starts BURIED: the top ring must come off before the middle one can. The
   seed's plan (one free pick of the chosen object) is physically impossible here; smoke
   check "seed strategy" shows its closest executable analog (grab the one accessible
   ring, set it on the plate) scores exactly 0.
3. **The goal is a three-body ASSEMBLY with two different terminal interactions.** The
   two non-middle rings must be THREADED onto the spare stand's dowel (hole-over-post
   insertion — a contact-guided descent down a 110 mm shaft), while the middle ring must
   end lying FLAT on the plate. The seed has no insertion of any kind and moves one body.
4. **A solver needs different code structure**: read the stack order, plan three
   transports with a topological constraint (top before middle), execute two threading
   insertions plus one flat placement — vs. the seed's single place-on-target primitive.

Sibling differentiation (earlier batches on this seed): `plated_meal` (old i2) claimed
aggregation-into-a-carried-container + composite nesting; `serve_ingredient` claimed
pouring/extraction; `dish_rack` claimed reorientation-into-rack. This task claims
**stack-order target identification + mechanically ordered de-stacking + peg threading**;
nothing is aggregated, poured, or reoriented into a slotted rack, and the plate receives
a single flat object.

## Embodiment argument (single Franka arm + parallel jaw, OSC)

Intended base pose: **(-0.42, 0.0, 0.20)** — mounted on the counter slab (the
plated-meal-verified on-counter mount). All action sites lie 0.42–0.60 m from that base
(source stand and spare stand at ~0.57 m worst-case jitter, plate at ~0.45 m), inside the
proven 0.35–0.65 m comfort band.

Per manipulated object (all three rings; the stands and plate are never moved):

- **Grasp:** pinch the ring across its outer flats — 64 mm square < the 80 mm jaw span
  (open ~76 mm, close to 64; the 54 mm-can/60 mm-carton pinch load case). The grasped
  ring is always the CURRENT TOP of its stack, so the pads take the upper ~10 mm of the
  16 mm face and the ring below stays untouched. Hand clearance over the dowel: grasping
  the top ring of the full stack puts the palm ~165 mm above the counter while the dowel
  tip is at 130 mm — the dowel passes between the fingers with >30 mm of palm clearance.
- **Extraction:** pure vertical lift of ~80–110 mm until the ring bottom clears the dowel
  tip — free sliding, no contact force needed.
- **Threading (the precision interaction):** hover the ring over the spare dowel tip and
  lower/release. Capture tolerance is ±10 mm radial (32 mm hole inradius 16 mm over the
  12 mm dowel) — comfortably above closed-loop OSC placement noise (1–3 mm demonstrated
  in prior tasks); after capture the dowel itself guides the 110 mm descent.
- **Plate set-down:** release the middle ring 20–30 mm above the plate; the 36 mm centring
  tolerance is far above control noise.

No contact is required near the ground (everything happens 200–330 mm up, on the
counter), under an overhang, or through a tight aperture; the only aperture (hole over
dowel) has 10 mm of radial slack.

## Execution order

**Partially required, enforced by physics, not by the rubric:** the top ring must leave
the source dowel before the middle ring can (rings only pass over the dowel tip). Beyond
that, order is free — e.g. top→spare, middle→plate, bottom→spare (the reference solve) or
top→spare, bottom-after-middle, etc. The rubric judges the settled END STATE plus latched
milestones; no order clause exists, so no out-of-order smoke case applies.

## Rubric (graded, latched in post_step, additive — monotone under any legal order)

| score | state |
|-------|-------|
| 0.00  | nothing done (null policy; also the seed-strategy end state) |
| +0.15 | a non-middle ring has been threaded on the SPARE dowel (at rest) at least once |
| +0.25 | BOTH non-middle rings threaded on the spare dowel at the same time |
| +0.30 | the middle ring has rested flat on the plate at least once |
| 1.00  | success(): middle ring flat on the plate + both others threaded on spare + settled |

Honesty: `threaded` accepts at most 10 mm of physical off-axis play (gate 15 mm) and a
ring beside the dowel sits at ≥38 mm; the z-band and xy gate together reject tip-balanced
rings. `on_plate` (gate 36 mm) can only accept a ring physically on the plate (max
fully-on offset 40 mm); a settled 50 mm off-centre set-down and a ring standing on its
edge face are constructible rejected near-misses. Latches are velocity-gated
(`latch_speed` 0.10 m/s), so a ring falling through a gate band latches nothing.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports = TRANSPORT ONLY; both load-bearing interactions run through contact dynamics:

- Phase 0: `env.reset(seed=N)`, settle, print the layout readback (stack color order,
  mirror side, stand/plate positions) and `SIM_GEN_SCORE 0.000`.
- Phase 1: TOP ring — incremental vertical pose-write raise up the source dowel (the
  arm's own free path; small steps with physics between), one free-space hop ABOVE both
  dowel tips, release ~30 mm above the SPARE dowel tip with a deliberate 4 mm lateral
  offset: the ring contacts the dowel, self-centres, slides the full shaft under contact
  guidance, lands on the base. Settle → `SIM_GEN_SCORE 0.150`.
- Phase 2: MIDDLE ring (now the exposed top) — raise, hop, release ~30 mm above the
  plate, settle flat → `SIM_GEN_SCORE 0.450`.
- Phase 3: BOTTOM ring — raise, hop, drop-thread onto the spare (lands on ring 1),
  settle → success → `SIM_GEN_SCORE 1.000`.
- Phase 4: ≥3.3 simulated seconds hands-off; success() must hold at every poll →
  `SIM_GEN_SOLVE: SUCCESS`. Missed drops retry (re-raise, re-drop) up to 4 attempts;
  any score decrease or exhausted retries prints FAIL. Hard exit behind watchdog Timers.
- **Verified on the forge: seeds 0, 1 and 2 all print `SIM_GEN_SOLVE: SUCCESS` (run 1,
  no retries needed; score ladder 0 → 0.150 → 0.450 → 1.000 on every seed, persistence
  held). Smoke: `SIM_GEN_SMOKE: ALL PASS 18/18` (run 1).**

Nothing is ever spawned seated: threading always starts above the dowel tip, the plate
set-down always starts above the plate.

## Check list (smoke.py — rejection battery; solve.py is the acceptance proof)

1. settle/no-NaN: authored stack settles finite, all three rings threaded on SOURCE
2. rubric clean at reset (score 0, no success)
3. determinism: same seed → identical layout readback
4. randomization: middle ring's color varies across seeds (readback)
5. randomization: stand mirror side varies across seeds
6. randomization: plate position jitters across seeds
7. null policy: score ~0, no success after 240 idle steps
8. SEED STRATEGY (top ring set on the plate, all else untouched): no success, score 0
9. incomplete (middle served, others still on SOURCE dowel): no success, score 0.30
10. tolerance twin: 20 mm off-centre set-down counts as on_plate (gate honest)
11. near miss: settled 50 mm off-centre (resting, overhanging) NOT on_plate
12. edge stand: middle ring standing on its edge face on the plate → not served, score 0
13. beside dowel: non-middle rings next to the spare stand → not threaded, no success
14. wrong ring: top ring on plate, middle wrongly parked on spare → no success, score 0.15
15. latched credit: one legit drop-threading scores 0.15
16. latched credit survives removing that ring again (still 0.15)
17. dumped pile: all three rings piled on the plate → no success
18. finite at the end

Teleports in the smoke are instrumentation (constructed settled states, real physics
steps before judging); no probe constructs full success.

## Scene / assets

Fully procedural, one rigid body per object: rings = 4-box square annuli (outer 64 mm,
hole 32 mm, 16 mm thick, 80 g, explicit MassAPI, 0.5 m/s depenetration cap, light
damping); stands = kinematic compounds (100 mm walnut/birch base + 12 mm × 110 mm steel
dowel); plate = white cylinder (170 mm × 12 mm, 300 g); counter = kinematic slab (top at
0.20 m). Randomized per episode: stack color order (which color is the middle ring),
layout mirror (which side each stand is on), stand/plate xy jitter, free yaw on rings
and plate.
