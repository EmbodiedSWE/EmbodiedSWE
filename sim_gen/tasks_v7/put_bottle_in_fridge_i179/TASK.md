# put_bottle_in_fridge_i179 — HangChillerScene (`simgen.hang_chiller`)

## Seed provenance

- **Seed task:** `rlbench/put_bottle_in_fridge`
  (`RoboVerse/roboverse_pack/tasks/rlbench/put_bottle_in_fridge.py`): a rigid
  `bottle_visual` and an articulated `fridge_base`; the demonstrated strategy is *open
  the fridge's revolute door, then carry the bottle in and STAND it upright inside* —
  one grasp-carry-release once the door is out of the way (no checker in the seed).

## What changed, and why it is strategically different

The STORAGE PRINCIPLE changed, not just the fixture. Both of the seed's load-bearing
subgoals are removed and replaced:

1. **No door.** The chiller is open-fronted — nothing to articulate before access.
2. **Standing storage is physically impossible.** The interior floor is a sparse drip
   GRATE over a drain pan; the gaps between the rungs (~70 mm) exceed the bottle's
   diameter (60 mm), so the seed's end state — a bottle stood upright inside — falls
   straight through into the pan and earns nothing (smoke check 6).
3. **The only storage is SUSPENSION**, and it is end-load-only *by topology*: an
   overhead slotted rail (two bars, 32 mm gap) runs from a loading tongue proud of the
   fascia back into the cold zone. The bottle's 60 mm body can never pass down through
   the 32 mm slot, so it cannot be dropped in from above (smoke check 10) — it must be
   threaded upright at the exposed tongue so the 56 mm cap flange seats on the two
   bars, then SLID rearward, hanging, under the low roof (24 mm over the rail plane:
   no top access, smoke check 8) at least 10 cm behind the fascia.

Distinct from the sibling same-seed tasks: `i311` (lay the bottle into a captive
sliding rack, push the rack through a letterbox — the manipulated DOF is the rack's
prismatic travel; here there is no captive carrier at all, the bottle itself slides
while suspended, and it must stay VERTICAL, never lie down), `i89` (payload rides a
door being closed under a speed bound), `i52` (count-metered dosing valve).

## Files

- `scene.py` — `HangChillerScene` (`SCENES.register("hang_chiller")`,
  `register_env("simgen", ... robot="null")`). Fully procedural compound spawners:
  kinematic cabinet (pan + grate + walls + roof), kinematic rail (bars + entry flares
  + below-plane neck stop; authored low-friction material on the bar tops), dynamic
  compound bottle (body + neck + flanged cap), plain dynamic red can (distractor: no
  neck, 56 mm > slot — it can never hang, and the rubric only watches the bottle).
  Randomization (readback-verified): cabinet yaw ±8° + xy jitter ±2 cm, rail slot
  lateral offset ±5 cm, bottle/can positions in the front field with a 26 cm
  separation floor (rejection resampling + deterministic fallback).
- `solve.py` — teleport solution (see below).
- `smoke.py` — 16-check rejection battery, records `frames.npz`.

## Rubric

- `_hung_now()`: cap-flange centre in the RAIL frame — seated z-band over the bar
  tops, |y| within the slot tolerance, x on the rail run — AND the bottle vertical
  (≤ ~12°). Geometry makes this suspension: a seated cap with the bottle vertical
  puts the body hanging in free air.
- `score() = 0.30·hung(latched) + 0.50·depth_max(latched running max of cap depth
  behind the fascia / chill_depth, counted ONLY while currently hanging)`, capped at
  0.80; exactly 1.0 iff `success()`.
- `success()` = live `_hung_now()` ∧ cap ≥ 10 cm behind the fascia (cabinet frame)
  ∧ settled (lin AND ang velocity — a swinging pendulum is not a stored bottle).

## Teleport-solution outline (solve.py)

- **P0** reset(seed), settle, layout readback; score ~0 asserted.
- **P1 TRANSPORT (the only pose write on the bottle):** one root-state write to a
  hover pose over the loading tongue — neck centred in the slot gap, cap flange 8 mm
  ABOVE the bar tops (above the seated band ⇒ not hung, no credit; both endpoints
  free space).
- **P2 SEAT (contact):** gravity drops the last 8 mm; the flange lands on the bars
  and the bottle hangs — `hung` latches from real resting contact (score 0.30).
- **P3 SLIDE (contact):** world-frame horizontal force at the bottle CoM along the
  cabinet's inward axis (velocity-regulated bang-bang, 1.5→5 N stall escalation,
  fell-off guard) drags the hanging bottle to cap depth ≥ 15 cm; release, settle;
  success asserted (score 1.0).
- **P4** ≥ 3.3 simulated seconds hands-off; success still holds ⇒
  `SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka arm)

Base at ~(0, 0) facing the cabinet fascia at x ≈ 0.50 (front toward the robot).
Everything a solver must touch lies in a 0.14–0.56 m annulus: the two ground spawns
(x 0.14–0.32, |y| ≤ 0.32), the loading tongue (fascia + 6 cm, rail top 0.345 m), and
the slide handle point — the solver grips the bottle BODY below the rail (body top is
30 mm under the bars) and walks it rearward; the wrist stays outside the letterbox
front opening for the first ~6 cm and the final 10 cm of cap travel needs reach only
to x ≈ 0.40 (cap at fascia −0.10, bottle body graspable at the front opening plane).
One grasp strategy per object: power grasp on the 60 mm bottle body (kept vertical
throughout — thread, seat, then push/pull along the slot); the can is never touched.
No regrasp, no bimanual coordination, no articulation.

## Execution order declared

1. scene.py (minimal goal predicate first), 2. solve.py iterated on the forge
(SUCCESS on `--seed 0` and `--seed 1`, first attempt each), 3. rubric finalized
(latched partials + cap), 4. smoke.py battery, 5. full re-verify after any scene
edit.

## Check list (smoke.py, 16)

1. settle: finite states, rail at its written lateral offset (readback vs cabinet
   pose), bottle upright on the ground, at rest
2. settle: score ~0, no success
3. randomization: cabinet yaw + xy and rail slot offset vary (readback, 6 seeds)
4. randomization: bottle + can positions vary, separation floor respected
5. null policy: 240 idle steps → score ~0, no success
6. seed strategy A: bottle stood upright inside falls THROUGH the grate (static
   assert: gap > bottle dia) → ~0
7. deep-but-not-hanging: bottle standing in the pan, cap deeper than chill_depth →
   ~0 (depth credit gated on suspension)
8. seed strategy B: carry-drop above the cabinet settles ON the roof (no top
   access) → ~0
9. wrong object: red can over the slot rests on the bars (static assert: can dia >
   slot) → ~0
10. topology: top-load attempt — bottle upright over the slot rests on its BODY
    (static assert: body dia > slot), never hangs → ~0
11. wrong place: bottle lying across the grate rungs in the cold zone → ~0
12. near-miss: properly hung at 60 mm depth (< 100 mm) → partial (~0.60), NOT
    success
13. incomplete: hung but still out on the tongue → hung credit only (~0.30), NOT
    success
14. monotonicity: deeper hang latches strictly more credit (30 → 70 mm), still no
    success
15. rejection audit: success() never True anywhere in the battery
16. final no-NaN

## Verification (forge, RTX-4090 pod)

- `solve --seed 0`: SUCCESS (depth 0.152, score 1.0, persisted 3.3 s)
- `solve --seed 1`: SUCCESS (depth 0.153, score 1.0, persisted 3.3 s)
- `smoke`: ALL PASS 16/16 with frames.npz recorded
