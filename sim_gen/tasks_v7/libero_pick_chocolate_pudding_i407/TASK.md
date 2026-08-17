# cleat_pier — build a cantilever perch off the pier's cliff, then set the pudding crate on it

**Task id:** `libero_pick_chocolate_pudding_i407` · **Scene:** `cleat_pier` (`CleatPierScene`) ·
**Env:** `simgen.cleat_pier` (robot `null`)

## Provenance

Derived from **libero/libero_pick_chocolate_pudding** ("pick up the chocolate pudding and place
it in the basket"): a Franka picks one food item from among five distractors and sets it into a
basket — one grasp, one carry, one place, onto a goal support that already exists.

## What changed and WHY it is strategically different

- **The goal support does not exist — it must be BUILT.** The seed places the payload into a
  waiting basket. Here the goal band is **empty air beyond a cliff edge** (a pier deck 400 mm
  above the floor). Before any pick-and-place can happen, the solver must construct the perch:
  slide a 480 mm plank along a railed guide lane, **under a low fixed hold-down bar (the
  cleat)**, until its nose cantilevers past the edge while its tail still runs beneath the bar.
- **A statics mechanism, not a container.** The cleat is a purely geometric anchor: when the
  loaded plank tips about the cliff-edge fulcrum, its rising tail presses the cleat's underside
  and the tip-up moment is reacted as a **force couple** — no joints, no counterweights, no
  ballast (deliberately NOT the corpus's counterweight/ballast family; nothing is weighed down,
  the anchor is pure geometry).
- **Order is forced by PHYSICS, not by the rubric.** An unanchored plank (tail clear of the
  cleat) see-saws off the cliff when loaded: the payload's tipping moment (0.6 kg x >=100 mm)
  out-weighs any unanchored hold-down — including the plank's own tail and even the decoy piled
  on it — by an asserted **>= 1.8x margin** (`__post_init__`). Load-after-anchor is the only
  order that survives; success() itself stays a stateless physical predicate.
- **Different plan and different code structure.** The seed is one grasp-carry-place
  trajectory. This needs: a long guided slide with a stop **window** (tail must stay under the
  cleat while the nose reaches past the edge), then a precision set-down on a 15 mm-thick
  overhanging ledge. The wrong-object decoy (pale-yellow butter box, 10x lighter) doubles as a
  useless counterweight, closing the "weigh the tail down instead" route.
- **vs. the same-seed siblings:** `_i201` (ShuttleVaultScene: offset-aperture airlock +
  internal shuttle conveyor) and `_i46` (PuddingSiftScene: tilt-pour through a sieve) share no
  mechanism, no goal topology, and no plan shape with a cantilever-and-hold-down statics build.

## Teleport-solution phases (solve.py — teleports are TRANSPORT ONLY)

- **P0** reset + settle; layout readback (fixture pose, plank depth, item strips).
- **P1 cantilever build (contact physics).** Velocity-servo horizontal force at the plank CoM
  (friction feed-forward + capped servo, lateral PD on the lane centreline) slides the plank
  ~0.5 m forward under the cleat; force cut when the tail reaches 20 mm under the cleat front
  (anchored window), plank coasts and settles flat. Nose ends ~185 mm past the edge.
- **P2 loading (gravity + contact).** Payload teleported to a **hover** 40 mm above the nose —
  open air, touching nothing — then dropped: it lands on the nose, the plank pitches by the
  6 mm cleat gap, the tail presses the cleat and the couple holds. The payload is never
  teleported into a supported or goal contact state.
- **P3 persistence.** >= 3.3 s hands-off; success() must hold throughout →
  `SIM_GEN_SOLVE: SUCCESS`. Verified on forge for **seeds 0 and 1** (rc=0 both).

## Rubric

`success()`: payload centre >= 100 mm beyond the cliff face, |y| <= 80 mm of the lane
centreline, at plank-top height (40 ± 30 mm above deck), payload AND plank settled — a pose
only a loaded, anchored cantilever can hold (goal region is pure void; leaning-plank ramps,
upended-plank pedestals, and payload-on-decoy stacks are excluded by asserted geometry).
`score()` (latched in `post_step`, never evaporates): 0.25 nose ever cantilevered flat past
the edge + 0.25 ever cantilevered while anchored + 0.20 payload ever rode the plank's forward
zone (requires the flat plank actually under it); capped at 0.70; exactly 1.0 iff success().
Null policy ~0.

## Embodiment argument (Franka, single arm)

- **Plank (480 x 100 x 15 mm, 0.40 kg):** slid, not lifted — push on its exposed rear end
  face, or pinch its 15 mm top edge in the open lane sections and drag; the ~0.5 m travel is
  a repeated push-regrasp sequence down the lane. The 6 mm cleat gap only ever admits the
  plank itself; the hand never needs to enter it (push from behind).
- **Payload (50 mm cube, 0.60 kg) / decoy (55 mm cube):** comfortably inside the ~80 mm
  parallel-jaw opening; top-down grasp from the open side strips, set-down onto the nose from
  above in free space over the void.
- **Base pose:** fixture-local ≈ (−0.45, −0.30) — beside the deck at mid-lane. From there the
  lane run (x ∈ [−0.85, 0]), the cleat region, both side strips, and the set-down point
  ~140 mm beyond the edge are all within a ~0.75 m reach envelope at deck height (0.40 m).

## Execution-order declaration

**Anchor-before-load, enforced by physics:** the plank must be slid to the anchored window
BEFORE the crate is placed; loading an unanchored plank destroys the perch (see smoke check 9).
The rubric contains no order latch for it — the order is a consequence of statics, asserted at
import time with >= 1.8x margin.

## Checks (smoke.py — rejection battery, 14/14 on forge, frames.npz recorded)

1–2. reset settles finite (plank in lane, items on opposite strips), score ~0, no success ·
3–4. randomization readback over 6 seeds (fixture xy+yaw, plank depth, item scatter, opposite
strips) · 5. null policy ~0 · 6. **seed strategy** (payload set down at the exact goal pose,
no plank) falls 400 mm into the pit, ~0 · 7. plank-alone anchored cantilever = 0.50, no
success · 8. near-miss (payload only 60 mm past the edge) = 0.70, no success · 9. **no-anchor
collapse** (order forcer): unanchored-but-stable plank see-saws into the pit when loaded ·
10. wrong object (decoy at goal on anchored plank, stable) rejected by identity · 11. z-band
stack cheat (payload on decoy pedestal, 92 mm high) rejected · 12. latched 0.70 survives
regression (payload removed, plank slid back), still no success · 13. success() never True at
any judged point · 14. no NaN.
