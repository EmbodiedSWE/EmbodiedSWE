# ledge_catch — stage the basket, then drop the orange carton into it off the shelf edge

Task id: `libero_pick_orange_juice_i6` · Scene: `ledge_catch` · Env: `simgen.ledge_catch` (robot="null")

## Seed provenance

- Seed: `libero/libero_pick_orange_juice`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero/libero_pick_orange_juice.py`)
- Seed plan: identify the orange juice among six distractor groceries, **grasp it,
  carry it through free space, place it in a basket that never moves**; success is a
  relative-bbox containment check on the static basket.

## What changed, and why it is strategically different

The seed's whole transport model is inverted; a solver needs a different plan and a
different program, not different numbers:

| | seed | this task |
|---|---|---|
| item | graspable bottle, carried by the gripper | ORANGE carton **100 mm on every horizontal axis > 80 mm Franka jaw span — ungraspable, never held** |
| receptacle | static basket, never touched | the basket is the **mobile** body: it must be fetched and staged under the shelf edge FIRST |
| transport of the item | prehensile carry through free space | **non-prehensile slide** along the slab + **ballistic free fall** into the waiting basket |
| ordering | none | **required**: stage the basket, then push (see below) |
| distractors | six groceries to ignore | one WHITE carton that must be **left standing** on the shelf (a disturb-nothing clause, not just an identification clause) |
| success | bbox containment relative to the basket | settled containment **at the catch station** + upright grounded basket + fell-in gate + decoy still shelved |

The seed's strategy is expressible in this scene and is explicitly rejected: carton
dropped into the never-moved basket at its spawn → smoke check 6 (geometric
containment verified TRUE, rubric still rejects).

## Execution order — REQUIRED, and how it is enforced

Stage the basket under the target's slab edge BEFORE pushing the carton off. It is
enforced twice:

1. **Physics**: a carton pushed off with no basket below lands on the ground, where
   it is ungraspable and the basket walls block any ground-level push-in.
2. **Rubric (`caught` fell-in gate)**: success requires the carton to have been
   observed *descending* (vz < −0.25 m/s) into the interior of the already-staged
   basket at the catch station. Lowering the basket over a grounded carton
   ("capping"), or any at-rest constructed containment, never sets it (smoke checks
   9 and 10). Staging credit itself is order-gated: the staged latch only sets while
   the carton is still on the shelf.

## Teleport-solution outline (solve.py — passes seeds 0, 1, 2)

- **P0** — reset, settle 0.5 s, baseline: score 0.000, no success.
- **P1 — TRANSPORT (the only teleport)**: one root-state write stages the basket
  upright on the ground at the catch point (~6 cm outboard of the target's slab
  edge, aligned to the shelf yaw). Receptacle staging across free ground — exactly
  what the arm does by carrying the basket. Asserts containment is NOT satisfied by
  the write. Score 0.400 (approach + order-gated staged latch).
- **P2 — contact dynamics only**: the carton is driven outboard by a horizontal
  external force at its CoM (bang-bang velocity regulation at 0.10 m/s, 0.55 N
  nominal, escalating to ≤0.95 N — below the ~1.0 N tipping bound so it slides).
  The force is CUT the instant the CoM crosses the slab edge; gravity tips it over,
  it free-falls ~15 cm and the basket catches it. **No pose write ever touches the
  carton.** Score 1.000 (caught latch set during the fall; success live).
- **P3 — persistence**: ≥3.3 simulated seconds hands-off; success still holds →
  `SIM_GEN_SOLVE: SUCCESS`. `SIM_GEN_SCORE` is printed at every phase boundary and
  never decreases (0.000 → 0.400 → 1.000 → 1.000).

## Embodiment argument (single Franka arm, parallel-jaw, OSC)

Plausible base pose: **base at world (0, 0), facing +x** (shelf centre at x ≈ 0.55).
All contacts lie in the 0.10–0.63 m radial band at heights 0–0.32 m.

- **Basket** (the only carried object): pinch the 8 mm top rim from above — fits any
  parallel jaw with the hand in free air above the open top; carry at ground level
  and set down at the catch point. The catch point is ~6 cm outboard of the slab
  edge and the slab underside is at 248 mm while the basket is 108 mm tall, so the
  set-down needs no reach under the overhang; a ground slide-push is a fallback.
  Placement tolerance is the stage radius (9 cm) — far above OSC noise.
- **Orange carton** (never grasped): a low, slow fingertip/closed-jaw push on its
  100 × 115 mm face, at slab height (0.26–0.32 m), driving it ~9 cm to the edge.
  The required force (~0.5 N) and speed (~0.1 m/s) are gentle; pushing below the
  ~1.0 N tip bound keeps it sliding. Overshoot margin: the basket interior is
  264 mm for a 100 mm carton falling from a known edge — generous.
- **White carton (decoy)**: must not be contacted at all; it stands at the opposite
  edge, well clear of the push lane.

## Rubric (anchored in the demonstrated solution)

`score() = 0.15·approach + 0.25·staged + 0.35·caught` (all latched, capped 0.85),
`1.0` iff `success()`:

- `approach` — running max of basket progress toward the catch point, normalized by
  its own spawn distance (exactly 0 for the null policy).
- `staged` — basket upright on the ground within 9 cm of the catch point WHILE the
  carton is still shelved (the order gate).
- `caught` — the fell-in gate described above.
- `success()` — carton settled fully inside the upright grounded basket, basket
  still at the catch station (≤13 cm), caught latch set, decoy still standing on
  the shelf, everything below settle speed.

## Check list (forge, dedicated GPU server)

- `solve` seeds 0, 1, 2 → `SIM_GEN_SOLVE: SUCCESS`, scores 0.000 → 0.400 → 1.000 →
  1.000 (non-decreasing), landing dead-centre (basket-frame |xy| ≤ 0.07).
- `smoke` → `SIM_GEN_SMOKE: ALL PASS 15/15`, frames.npz (147 × 600 × 960) saved:
  1. settle/no-NaN: cartons shelved, basket upright far from catch, all still
  2. reset score ~0, no success
  3. randomization: target slab side flips AND basket ground side flips (readback)
  4. randomization: shelf yaw + xy, target slot x + rel yaw, basket x vary (readback)
  5. null policy 240 steps → score ~0, no success
  6. seed strategy (item into the spawned basket) → rejected despite verified
     geometric containment
  7. upside-down basket at the catch point → staged latch refuses (score 0.15)
  8. overshot drop: carton grounded just outboard of the staged basket → rejected
  9. capping: basket lowered over the grounded carton at the catch point →
     containment TRUE, fell-in gate alone rejects (score 0.15)
  10. capped after a legitimate stage (missed-drop + re-cap recovery) → rejected
  11. wrong object: WHITE decoy in the basket, ORANGE shelved → rejected
  12. decoy clause isolation: target genuinely fell in (caught verified SET), decoy
      off the shelf → that clause alone rejects (score 0.75 ≤ 0.85)
  13. latched credit survives the basket being yanked away (0.400 → 0.400)
  14. rejection audit: success() never True at any judged point
  15. final no-NaN
