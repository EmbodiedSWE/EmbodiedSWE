# libero_pick_ketchup_i295 — `postbox_deposit`

Push the red ketchup bottle through a one-way swing flap into a roofed deposit box.

## Seed provenance

- Seed: `libero/libero_pick_ketchup`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero/libero_pick_ketchup.py`)
- Seed strategy: pick the ketchup bottle from among grocery distractors (bbq sauce,
  salad dressing, alphabet soup, cream cheese, milk, basket), lift it, and DROP it
  into an OPEN-TOP basket; the checker is a bbox test relative to the basket.

## What changed and WHY it is strategically different

The receptacle is replaced by a mechanism that makes the seed's entire plan
impossible, not just re-parameterized:

- The open basket becomes a fully ROOFED deposit box. The seed's move — carry the
  bottle above the receptacle and release — dies on the roof (smoke check 6
  constructs exactly that end state and asserts rejection).
- The only entrance is a side APERTURE (130 x 120 mm) at 16 cm sill height, covered
  from inside by a ONE-WAY SWING FLAP hinged along its top edge: it swings inward
  under push, a joint limit stops it swinging outward, gravity plus a weak spring
  return it hanging shut.
- A LOADING TRAY with guide rails juts out from the aperture, flush with the sill;
  inside, a low-friction 36-degree ramp descends from the sill to the box floor.
- The seed's vertical pick-and-drop becomes: identify the RED bottle among a
  permuted line-up (yellow mustard, white mayo decoys), REORIENT it horizontal onto
  the tray, and PUSH it through the flap — a horizontal push-through-a-sprung-door
  interaction with a gravity-fed deposit, plus a "flap must hang shut again" outcome
  clause. A solver needs a different plan (stage-then-push, not lift-and-drop) and
  different code structure (flap hinge state, sill crossing, descent) — none of the
  seed's checker or trajectory logic transfers.

## Teleport-solution outline (solve.py)

1. P0: reset, settle, assert flap hanging shut and score ~0.
2. P1 TRANSPORT (teleport): one pose write lays the ketchup on the loading tray,
   cap toward the box — supported free pose, provably outside the box (asserted),
   flap untouched (asserted).
3. P2 PUSH (contact dynamics): a bounded horizontal force (4 N bang-bang with a
   0.15 m/s speed cap and a small rail-centering y term — the tray has a DEFINED
   friction material, mu 0.4, so the push budget is known) slides the bottle along
   the tray; its cap shoves the flap inward THROUGH CONTACT until the bottle's
   centre crosses the sill; the force is then removed. The flap's whole trajectory
   is hinge physics; no pose write touches the bottle after P1.
4. P3 DESCENT (contact dynamics, hands off): the bottle tips over the sill, slides
   down the low-friction ramp to the box floor; the flap swings shut behind it.
   success() first turns True here.
5. P4: persistence >= 3.3 simulated seconds hands-off, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing
(all rubric credit is latched). Verified on seeds 0 and 1 on the forge.

## Embodiment argument (single Franka, parallel jaw, OSC)

- Base pose: Franka base at the world origin, facing +x. All required contacts lie
  in x 0.26–0.46, |y| <= 0.30, z 0.08–0.25 — comfortably inside the reach envelope.
- Ketchup bottle (the only object that must be moved): body diameter 56 mm — fits
  the parallel jaw with clearance; grasp the standing bottle around its body
  (approach from above/side, nothing overhead), lift, rotate the wrist to
  horizontal, lay it between the tray rails (tray is 20 cm long, rails 13 cm apart
  vs 5.6 cm bottle — lateral tolerance +/-3.7 cm; the rails and aperture pillars
  funnel residual misalignment). Then push its base with closed fingertips at
  z ~ 0.19: a straight-line 10–12 cm OSC push ending ~6 cm short of the front wall.
  The fingers NEVER need to enter the aperture: once the bottle's centre passes the
  sill, gravity and the ramp finish the deposit (demonstrated by solve.py, which
  cuts its force at exactly that point). Over-pushing is harmless.
- Flap: never touched by the hand — actuated purely through the bottle.
- Decoys: never touched.
- Precision budget: the only tolerance-critical value is lateral alignment
  (+/-3.7 cm), far above OSC control noise; push depth has centimetres of slack.

## Execution order

No discrete order is declared beyond what the mechanism physically forces
(stage on the tray, then push through the flap). Only the ketchup bottle needs to
be moved; touching the decoys is allowed as long as they end outside the box.

## Check list (smoke.py — rejection battery, 14 checks)

1. settle: finite states, flap hanging shut, bottles standing at slots, still,
   nothing inside the box
2. settle: score ~0, no success
3. randomization: bottle->slot permutation varies across seeds; three distinct
   slots every reset (readback)
4. randomization: per-slot xy jitter > 4 mm (readback)
5. null policy: 240 idle steps -> score ~0, no success
6. seed strategy: bottle released from above settles ON the roof -> rejected
7. near-miss: staged on the tray at the aperture mouth, flap shut -> tray credit
   only, no success
8. jam: bottle bridging the sill (centre outside), flap propped on it -> rejected
9. flap propped open by the decoy while the ketchup is inside -> the flap-shut
   clause alone rejects
10. wrong object: mustard deposited instead -> score ~0, no success
11. decoy exclusion: ketchup AND mustard both inside the shut box -> rejected
12. latched credit survives teleporting the ketchup back out; still no success
13. rejection audit: success() never True anywhere in the battery
14. final no-NaN

The seed's literal end state (bottle in an open basket) is not expressible in this
scene (there is no open receptacle); its analog — release from above — is check 6.
