# put_shoes_in_box_i379 — CrateFlipPackScene (`simgen.crate_flip_pack`)

Put the shoes in the box — but the box starts MOUTH-DOWN ON TOP OF THE SHOES.
The crate is a trap that must become the receptacle: tip it up onto its side,
tip it again onto its base (two nonprehensile quarter-rolls of a 0.6 kg bin),
and only then pack the two shoes it was covering. The rubric's containment
predicate is geometrically TRUE at spawn — credit is gated on the crate being an
ERECTED receptacle, so the spawn state and every shortcut score 0.

## Seed provenance

- **Seed task**: `rlbench/put_shoes_in_box` (RoboVerse
  `roboverse_pack/tasks/rlbench/put_shoes_in_box.py`) — "put the shoes in the
  box". An always-open, ready-to-use box and two free shoes; the demo picks each
  shoe, carries it over the mouth, releases. The box is never operated, never
  moved, never in the way.

## What the task is

A SLATE-BLUE CRATE (interior 340 × 260 × 120 mm, 12 mm walls/floor, 0.6 kg,
low CoM) lies MOUTH-DOWN on the floor with two ORANGE SHOES trapped beneath it
(sole 140 × 55 × 16 mm + heel counter 50 × 55 × 30 mm, 46 mm tall, 140 g each).
Per episode the crate center jitters ±60 mm around (0.40, 0) with FREE yaw
(±180°); the shoes sit in two slots at crate-local x ≈ ∓75 mm with their own
jitter, segment-separated ≥ 61 mm (resampled, deterministic crosswise fallback).

Goal: crate UPRIGHT AT REST ON ITS BASE (up-axis within 8° of world up, root
height within 8 mm of the base rest height, settled) with both shoes fully
inside the interior below the rim (3 body points per shoe in the crate frame),
everything settled and finite.

## Why strategically different

- **vs. the seed**: the seed's skill is *transport into a ready container*.
  Here the container is the OBSTACLE: at spawn it seals the shoes under
  itself. Smoke 4 runs the seed plan (carry a shoe over the crate, release) —
  it lands on the upturned base and slides off, score 0. Smoke 5 shoves a
  trapped shoe with 4 N: it drags/jams and shoves the whole crate 0.52 m along
  the floor without ever escaping the footprint — the cover is physical, not
  bookkeeping. Two new obligations precede any packing: (1) UNCOVER — roll the
  crate off the shoes; (2) ERECT — continue the roll until the crate rests on
  its base, mouth up. Both are nonprehensile quarter-tips: pivot on a ground
  edge, drive past the balance angle (~2.06 rad from mouth-down, ~1.19 rad from
  the side plateau), catch the fall. Only then does the seed's transport skill
  apply.
- **The containment predicate is honest by gating, and the gate is honest by
  geometry**: `inside` is TRUE for both shoes at spawn (they are inside the
  inverted interior — smoke 6, the flagship: containment TRUE, score 0).
  Packed credit and success additionally require the ERECTED receptacle, and
  `__post_init__` asserts cover (up_z < −0.10) and erected (up_z > cos 8°)
  are disjoint over all poses — no pose earns both, so the uncover credit
  needs a real roll. A crate PERCHED on a shoe rests 38 mm high and fails the
  8 mm rest-z clause (smoke 8); a crate on its SIDE with shoes tucked into the
  sideways cavity satisfies `inside` and still scores no packed credit
  (smoke 10); a shoe lying on the rim wall fails the z clause (smoke 9).
- **vs. the corpus read**: `put_shoes_in_box_i143` keeps the box ready-made and
  moves the difficulty into a plug lid + antiparallel-fit puzzle — its box is
  kinematic and never moves; here the container itself is the manipulandum
  (a free body rolled 180° across two balance points, translating ~0.4 m in
  the process, so the solver must re-read its pose before packing). Flip/tip
  corpus tasks (`die_quarter_roll`, `crate`-style flips) reorient the GOAL
  OBJECT; here the tipped body is the CONTAINER and the goal objects start
  captive under it — uncover-then-erect-then-pack is an order forced by
  geometry, not by rubric fiat.

A solver therefore needs a different PLAN each episode (read the crate yaw,
pick a tip axis, roll twice, re-locate the displaced crate mouth, then pack)
and different CODE STRUCTURE (latched uncover/erect/packed credit, an
erected-receptacle gate on containment) — not a waypoint pick-and-drop.

## Solution outline (solve.py = the legitimacy certificate)

1. **P0** settle 1 s; READBACK layout (crate xy + yaw + mouth-down, shoes
   covered); assert score ≤ 0.03.
2. **P1** TORQUE-FRAME PROBE: a yaw-spin torque pulse about world z fights only
   friction (never gravity), so the SIGN of the resulting ω_z identifies
   whether this pod's wrench API takes world or body vectors (mode 1
   pre-encodes via `quat_apply_inverse`). Then QUARTER-TIP 1: torque about the
   horizontal projection of the crate's local x̂ (bang-bang, ω-gated at
   2.5 rad/s, stall-escalation ×1.45 capped 3.0 N·m from τ₀ = 1.0) until θ
   from upright passes 1.85 rad, then rate-damped descent onto the side.
   Shoes uncovered → score 0.15.
3. **P2** QUARTER-TIP 2 from the side plateau (gate 1.05 rad, τ₀ = 0.55), same
   axis sign so the roll continues one way; crate lands on its base, settles →
   erected, score 0.30. Peak equivalent wall force ≈ 6.3 N (asserted < 12 N in
   cfg).
4. **P3** shoe A teleported (transport only) to a hover 8 mm above the rim at
   crate-local x = −0.08, crate-matched yaw; released, falls ~130 mm to the
   floor of the crate (jog-retry with fresh crate readback if it perches) →
   score 0.50.
5. **P4** shoe B likewise at x = +0.08 → success, score 1.0.
6. **P5** hands-off persistence 400 steps (> 3.3 s); success holds →
   `SIM_GEN_SOLVE: SUCCESS`.

Teleports only ever TRANSPORT a free shoe through open air; the crate is moved
exclusively by external torques through contact with the ground, and every
landing (both tips, both shoe drops) is gravity + contact. Monotone
`SIM_GEN_SCORE`: 0.0000 → 0.1500 → 0.3000 → 0.5000 → 1.0000 → 1.0000.

## Rubric

- 0 → 0.70 latched shaping (never decays): 0.15 shoes ever UNCOVERED + 0.15
  crate ever ERECTED (upright + rest height + settled) + 0.20 per shoe ever
  inside-the-ERECTED-crate and settled (cap 0.70). Doing nothing scores 0
  (the crate starts covering the shoes).
- 1.00: success — live: both shoes' three body points inside the interior in
  the CRATE FRAME below the rim, crate upright at rest on its base, everything
  settled and finite.

Unfakeable without the mechanic: nothing enters a mouth-down crate (smoke
4–5), containment without the erected receptacle earns nothing (smoke 6, 10),
a perched or side-resting crate is refused (smoke 8, 13), and success is
judged live — tipping the crate back collapses it (smoke 13).

## Franka embodiment (single arm, parallel jaw 80 mm, OSC)

Proposed base pose: **(0.00, 0.00, 0.00), facing +x** (nominal reach 0.855 m).
The crate spawns at 0.40 ± 0.06 m; over the two tips it translates ~0.4 m
(roughly half the outer height + half the outer length per quarter-roll), so
the tip DIRECTION is chosen toward/across the base to keep the landed crate
inside the reach band — exactly the axis-sign reasoning solve.py encodes.

- **Tipping** (nonprehensile): plant the closed fingertips against the crate's
  upper side wall near the top edge (~130 mm height, well off the ground) and
  push horizontally; the far ground edge is the pivot. Breakaway force is
  ~6.3 N at that lever arm (cfg-asserted < 12 N) — trivial for the arm.
  After the balance angle the crate falls on its own; the arm retracts and the
  crate's own walls/floor catch the landing (rate-damped in solve.py, and a
  passive landing is equally accepted — the rubric judges the settled pose).
  Repeat once from the side plateau at a lower breakaway (~1.19 rad balance).
- **Shoes** (140 g): pinch across the 55 mm sole/heel width (≪ 80 mm jaw),
  fingertips at 8–46 mm height. Insertion is a release from a hover above the
  open mouth (interior 340 × 260 mm — a generous drop corridor per shoe slot);
  the shoe falls ~130 mm and lands flat. The crate rim is 132 mm high — the
  wrist clears it with the standard top-down grasp.
- **Ordering** (declared): uncover/erect FIRST — geometry-enforced (the mouth
  is sealed against the floor; smoke 4–5); the two tips are ordered by
  physics (mouth-down → side → base); shoes in either order afterwards.
- **Forces**: heaviest interaction ~6.3 N side push; heaviest lift 140 g.
  All manipulation is below 0.20 m height and inside the envelope.

## Validation evidence (forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0`: SUCCESS, scores 0.0000/0.1500/0.3000/0.5000/1.0000/1.0000,
  spin probe ω_z ≈ +4.4 → mode 0, tip 1 lands θ = 1.571, tip 2 lands θ = 0.000
  with no torque escalation (τ_end 1.00/0.55), 19.1 s wall.
- `solve --seed 1` (different crate yaw/xy): SUCCESS, same monotone scores,
  19.1 s.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 15/15**, frames.npz saved:
  1. settle/no-NaN: crate READBACK mouth-down, both shoes covered, score 0
  2. randomization readback, 3-seed max-pairwise: crate yaw Δ98.6°, crate xy
     Δ84.2 mm, shoe_a xy Δ148.9 mm
  3. null policy: 240 idle steps, score 0, no success
  4. SEED STRATEGY: a shoe carried over the crate and released — lands on the
     upturned base and slides off, nothing enters, score 0
  5. CAPTIVITY PUSH: a trapped shoe shoved with 4 N along the crate's long
     axis — it drags/jams and shoves the whole crate 517.5 mm along the floor,
     never escapes the footprint, stays covered, score 0: the cover is physical
  6. GEOMETRIC CONTAINMENT (flagship): at spawn `inside` is TRUE for both
     shoes, yet score 0 — containment without the erected receptacle is void
  7. uncover-only (crate parked on its side away from the shoes): 0.15
  8. PERCHED: crate dropped upright with a wall over a shoe rests at
     z = +37.9 mm — receptacle refused by the 8 mm rest-z clause
  9. RIM near-miss: shoe balanced lying on the rim wall — z clause refuses
  10. crate-on-side pack: shoes tucked into the sideways cavity satisfy
      `inside` — no packed credit, no success
  11. erect-only: 0.30, no success
  12. one shoe packed: 0.50, no success
  13. TIP-BACK liveness: constructed success reads 1.0; crate tipped back onto
      its side → success COLLAPSES, latched score survives at 0.70
  14. rejection audit: success never fired during checks 4–12
  15. video frames.npz saved (144 frames)

## Files

- `scene.py` — CrateFlipPackScene + procedural crate/shoe compound spawners +
  gated latched rubric; registers `simgen.crate_flip_pack`.
- `solve.py` — torque-frame probe + double quarter-tip + drop-pack certificate
  (`--seed N`).
- `smoke.py` — 15-check rejection battery + video.

Run (forge):
`python -u -m simgen_tasks.put_shoes_in_box_i379.solve --headless [--seed N]`
`python -u -m simgen_tasks.put_shoes_in_box_i379.smoke --headless`
