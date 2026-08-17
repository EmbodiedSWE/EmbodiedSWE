# put_money_in_safe_i259 — SeesawVaultScene ("bank the brick through the counterweight-tolled lid")

Env: `simgen.seesaw_vault` (robot slot: `null`; bodies driven through scene handles).

## Seed provenance

Seed: `rlbench/put_money_in_safe` — "put the money away in the safe on the top
shelf". In the seed, a stack of dollars sits on a table and a safe stands with
its door **already open**: the demonstrated strategy is a single
grasp–transport–place of the money onto a shelf inside a static, always-open
receptacle. Nothing about the receptacle ever moves.

## Strategic difference

- **vs the seed:** the receptacle is never open by default and can never be
  *held* open by the hand that carries the money. The vault has no door and no
  open face — the only way in is a roof deposit mouth covered by a heavy lid on
  a gravity-biased SEE-SAW lever. The seed's plan (carry the money to the safe
  and set it down inside) is physically impossible here: setting the brick on
  the vault parks it ON the closed lid (smoke check 5), and the solve's own
  deposit drop against the closed lid is refused outright (check 6). The task
  is a **third-hand puzzle**: a tool object (the brass counterweight) must be
  parked on the lever's pedal pan so its *standing weight* — not the gripper —
  holds the lid open while the hand is free to fetch and drop the brick; then
  the tool must be removed again so gravity reseals the mouth. One
  grasp-and-place became three, with a mechanism between them.
- **vs sibling i174 (same seed, rotary turret airlock):** i174's differentiator
  is a *sequenced airlock* — crank a turret to receive, force-push the brick
  through a side window into a pocket, crank back so the sweep dumps it into
  the bin and reseals; the hand interacts with the mechanism directly and the
  brick is pushed through a sealed boundary. i259's differentiator is a
  *counterweight toll*: the hand never needs to touch the mechanism at all —
  the lever is driven purely by what is set on its pan, entry is a free drop
  through the top, and the torque budget (brick too light, weight heavy
  enough) is itself the puzzle. Different mechanism (see-saw vs turret),
  different entry path (roof drop vs side push), different resource (a
  dedicated tool object vs cranking), different failure modes.

## Solution phases (solve.py — no applied wrench anywhere; teleports are transport only)

1. **P0 settle + audit** — 180 steps; readback of layout randomization and of
   every mass (per-child density / root MassAPI regression guard); lid on its
   closed stop; baseline score ~0.
2. **P1 PARK** — one write carries the counterweight to a 6 mm free-air hover
   over the pedal pan's centre (inside the open-top cage; nothing judged). It
   lands; its standing weight back-drives the see-saw and swings the lid to the
   60° open stop. `SIM_GEN_SCORE` after: ≥ 0.25 (open credit, latched).
3. **P2 DEPOSIT** — one write carries the brick to a free-air hover 0.42 m up
   over the REAR part of the mouth the raised lid has uncovered (long axis
   across the mouth). It free-falls through and settles on the vault floor.
   Score after: ≥ 0.60 (`in` latch).
4. **P3 UNPARK** — one write lifts the weight out of the cage back onto the
   free pedestal top. The lid's own gravity bias closes the see-saw onto the
   0° stop: reseal is pure mechanism dynamics. Score after: 1.0 (success live).
5. **P4 persistence** — 3.3 simulated seconds hands-off; success must hold
   through the whole window; then `SIM_GEN_SOLVE: SUCCESS`.

Validated on the forge, seeds 0 and 1 (different vault yaw/xy, stand bearings,
and brick/weight side assignment), scores strictly monotone
0.00 → 0.25 → 0.60 → 1.00 → 1.00.

## Execution order (physically forced)

`park weight → drop brick → unpark weight`, declared in `describe()` /
`instruction()` and enforced by gravity, not by the rubric:

- drop before park: the closed lid covers the whole mouth with a 13 mm hover
  gap vs a 30 mm brick — refused (smoke 6);
- park the *brick* instead of the weight: 0.40 N·m at the pan's outer fence
  vs a 0.60 N·m closed bias — the lid never opens (smoke 7);
- skip the unpark: the weight in the pan holds the lid open — `success()`
  refuses and the score caps at 0.60 (smoke 8).

## Embodiment (single Franka, parallel-jaw gripper)

Plausible base pose: on the ground at vault-local **(+0.62 m, 0, 0)** — i.e.
between the two pedestals, facing the vault front (the pedestals stand at
0.62 m from the vault centre, ±15–35° off the front axis, so both stand tops
and the pedal pan are inside a ~0.55–0.75 m reach envelope at comfortable
heights).

- **counterweight** (⌀48 × 100 mm, 1.28 kg): dedicated 20 mm grasp knob for a
  parallel-jaw pinch; 1.28 kg is well inside Franka payload; carried from a
  0.10 m pedestal top to the pan rim at ~0.30 m height, a short free-air arc.
  Release 6 mm above the pan floor inside the open-top cage — no insertion
  precision beyond the ±23 mm xy slack of the cage.
- **cash brick** (120 × 60 × 30 mm, 0.24 kg): 60 mm across the grasp faces —
  inside the Franka's 80 mm jaw span; picked flat off a pedestal, released
  from a hover above the uncovered rear mouth (the drop zone is a 60 × 150 mm
  clear rectangle; the brick falls in flat with ≥ 15 mm side margins). No
  in-chamber manipulation is ever needed — entry is ballistic.
- **lever / lid**: never touched by the hand. The pan hangs in free air in
  front of the vault at ~0.28–0.30 m height, reachable for the set-down and
  the pick-off; the lid does all its moving under gravity.
- **forces**: the solve applies no external wrench at all — every load-bearing
  interaction is standing weight + contact through the pivot, so nothing
  exceeds what a set-down/pick-up can deliver.

## Smoke battery (11 checks, `SIM_GEN_SMOKE: ALL PASS 11/11`)

1. settle/no-NaN — seeded reset settles finite, lid on its closed stop, score ~0;
2. randomization-is-real — vault yaw / vault xy / stand xy / brick yaw readback
   differ across seeds;
3. side-flip — brick/weight pedestal assignment flips across 6 seeds and the
   brick readback sits on the flagged stand every time;
4. null-policy — 240 idle steps: score ~0, no success;
5. SEED strategy — the brick set down on the vault lands ON the closed lid over
   the mouth: refused entry, no credit (the seed's one-place plan is dead);
6. unpaid-toll drop — the solve's own deposit drop vs the closed lid: refused,
   lid barely moves (order forced);
7. brick-cannot-pay — brick laid on the pan at full radius: lid stays closed
   (the torque budget is real; the counterweight is the only tool);
8. banked-unsealed — brick genuinely inside + weight left in the pan: success
   false, score ≤ 0.60 (reseal clause has teeth);
9. wrong-object — counterweight inside the chamber, brick on its pedestal:
   score ~0, no success;
10. stop-reality — parked weight holds the lid through a 1 s window on the 60°
    stop; gravity alone reseals after the unpark;
11. video — frames.npz captured and saved.

## Rubric

`0.25·open_frac (latched max lid angle / 55°) + 0.35·brick_inside (latched,
3-step streak) + 0.15·closed_after (lid closed AFTER the in-latch, 3-step
streak)`, capped at 0.75; exactly 1.0 iff `success()` holds live: brick CoM
inside the chamber footprint below z = 0.20 (containment below the aperture),
lid ≤ 5°, everything still and finite.
