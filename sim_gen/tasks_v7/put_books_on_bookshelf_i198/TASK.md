# put_books_on_bookshelf_i198 — Slanted Display Shelf: Buttress and Lean

**Scene:** `tilt_shelf_buttress` · **Env:** `simgen.tilt_shelf_buttress` · robot="null" (scene-level)

## Seed provenance

Seed: `rlbench/put_books_on_bookshelf` (RoboVerse `roboverse_pack/tasks/rlbench/put_books_on_bookshelf.py`).
The seed is an additive pick-and-place: three USD books start flat on a table and are moved onto an
empty, level bookshelf. There is no checker; the shelf holds a book anywhere it is put.

## What changed, and why it is strategically different

**vs the seed:** the shelf here is a wall-mounted display plank slanted 13.5–16.5° along its length
(the downhill end flips left/right per episode). Every book's topple angle atan(t/h) is 7.2–10.6°,
strictly below the minimum tilt, so **no book can free-stand anywhere on the plank** — the seed's
strategy (put each book down on the shelf independently) is physically impossible, and smoke check 5
proves it (a book placed upright on the bare plank topples). The task is
construction-under-instability: first anchor a heavy (1.2 kg, high-friction) bookend at the sampled
downhill end, then build a contiguous leaning row of four books growing uphill from it, each new
book caught by the structure built so far. Support is transitive through contact — remove the
anchor and the whole row cascades (smoke 6).

**vs sibling i190 (Librarian's Extraction):** i190 is subtractive/selective — a packed row already
exists and one requested book must be tipped out and laid in a tray without disturbing neighbors.
i198 is the constructive dual with a different physics core: there is no pre-built row, no
distractor-preservation constraint, and the difficulty is that intermediate states are statically
infeasible without the right build order. i190's rubric is about *not* touching things; i198's is
about creating a mutually supporting structure on a slope where nothing stands alone.

**vs the rest of the corpus read:** no other task uses transitive support chains on an incline as
the goal predicate (chain-contiguity anchored at a buttress, judged in the plank frame against the
plank normal).

## Teleport-solution outline (solve.py)

Teleports are TRANSPORT ONLY: every body is released slightly ABOVE its placement spot; seating is
pure contact dynamics (drop + friction catch). Phases (SIM_GEN_SCORE at each boundary,
non-decreasing asserted):

1. **reset** (0.000) — assert clean state, latches dark, drives zero.
2. **buttress** (0.250) — bookend released just above the bare slope at u=−0.13 (downhill end);
   friction seats it (μ=0.6 capacity ≈ 3.4 N ≫ worst-case row push ≈ 2.45 N).
3. **book0..book3** (0.400 / 0.550 / 0.700 / 0.850) — each book released with a ~3.4° downhill
   pre-lean just uphill of the current row face; it falls INTO the structure and is caught
   (bookend, then book-on-book). Without the support it would topple (smoke 5/9).
   After book1: **snug press** (0.550, unchanged) — a fingertip-scale 0.8 N CoM force via the
   scene's `drive_sel`/`drive_f` wrench slot compacts the 2-book row downhill against the bookend,
   then releases. The press is mid-build deliberately so a transient un-settle cannot drop a
   printed score after success is reachable.
4. **success** (1.000) — live predicate: buttress ok + all 4 books ok + chain of 4 + settled.
5. **persistence** — drives asserted zero, ≥3.5 simulated seconds hands-off, success re-checked,
   then exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) + watchdog Timer.

## Execution order: FORCED

The bookend must be first (nothing stands without it — smoke 6), and the row must grow uphill one
book at a time (each book's support is the previous face; placing uphill books first gives them
nothing to lean on and they topple — smoke 7 shows the wrong-end variant collapsing). The rubric's
chain predicate is anchored at the buttress and counts only contiguous links, so no prefix or
reordering scores as success (smoke 9: a book leaning on the bookend's top edge at 18° is not
counted).

## Embodiment argument (single Franka, parallel jaw)

Base pose: **(0.45, 0.0, 0.40)** on the table surface, facing the wall (−x). Reach: plank center is
at x=−0.18, z=0.53, books start at x=+0.10 — all within ~0.75 m of the base, no wall collision
(wall at x=−0.31, plank protrudes toward the robot).

- **Books** (0.11 × 0.022–0.028 × 0.15–0.175 m, 200–260 g): pinch across the 22–28 mm thickness —
  comfortably inside the Franka jaw span (80 mm); grasp the top third of the spine so the fingers
  clear the neighbor when lowering into the lean. Place-then-release-into-lean: hold the book just
  above the row with a small downhill tilt, open the jaws, and the book falls ~6 mm into the
  structure — exactly what solve.py does with `tp` + gravity.
- **Bookend** (0.09 × 0.055 × 0.085 m, 1.2 kg): pinch across the 55 mm depth (within span); 1.2 kg
  is within Franka payload (3 kg). Lower to ~6 mm above the slope and release.
- **Snug press**: 0.8 N fingertip push at the book's CoM height along the downhill slope direction —
  trivially within Franka force control range; the scene's wrench slot is the stand-in.
- **Tolerances**: chain gap 15 mm, placement clearance 3 mm + settle-by-contact means the arm needs
  only ~5 mm placement accuracy — realistic for Franka position control.

## Randomization (verified by readback, smoke 3)

Tilt sign (both directions across seeds 2–9), tilt magnitude 13.5–16.5° (≥5 distinct, spread >1°),
plank y ±0.05 (≥5 distinct, spread >0.03), book table-slot permutation (≥3 distinct) + per-slot
jitter and yaw. Plank pose readback: quat dot >0.99999 vs R_x(φ), position error <2 mm (smoke 2).

## Rubric

score = clamp(0.25·but_latch + 0.15·k_latch, ≤0.85); exactly 1.0 iff success(). Latches are
slow-gated (all body speeds < 0.08 m/s) max-latches — no fly-through credit, no credit evaporation
below the cap; success itself is live, so removing a book after success revokes it (smoke 11 →
score back to ~0.85).

## Smoke battery (12 checks)

1. Clean reset: settled, finite, books upright at their table slots, latches dark, score < 0.05.
2. Plank pose readback matches sampled (φ, y): quat dot, 2 mm.
3. Randomization sweep seeds 2–9 (signs, tilts, plank_y, permutations) by readback.
4. Null action 240 steps: score < 0.05.
5. Seed-strategy negative: book stood upright on the bare plank topples (align > 30°), score < 0.05.
6. No-anchor row: 4-book row without bookend cascades, chain 0, score < 0.05.
7. Wrong-end bookend (uphill) with books downhill: collapse, score ≤ 0.30.
8. Flat books lying uphill of a correct bookend: not upright, not chained, score 0.20–0.30.
9. Settled near-miss: book leaning 18° on the bookend top edge — settled but align outside the
   upright cone, not counted, score ≤ 0.30.
10. Exactness: physical goal build (drops mirroring solve) → success, score == 1.0, persists 1 s.
11. Latch/revocation: teleport book3 back to the table → success revoked, 0.83 ≤ score ≤ 0.87.
12. frames.npz saved in CWD with ≥ 20 RGB frames.

Prints exactly `SIM_GEN_SMOKE: ALL PASS 12/12`.
