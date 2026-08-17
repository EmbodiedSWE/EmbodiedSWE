# put_books_on_bookshelf_i190 — Librarian's Extraction from a Packed Shelf

**Env:** `simgen.librarian_extract` (scene `librarian_extract`, robot `"null"`)
**Seed:** `rlbench/put_books_on_bookshelf` (RoboVerse `roboverse_pack/tasks/rlbench/put_books_on_bookshelf.py`)

## What the task is

A small one-slot bookcase stands at the back of a table, open side toward the robot.
Five hardcover books of different colors and sizes (thickness 22–40 mm, height
160–190 mm) stand upright in ONE tight row between two full-height side cheeks: the
side gaps between neighbors are ~3 mm and the cheeks run all the way up to the case
ceiling — there is no room to pinch a book's covers where it stands and no way to
lift a book clear of the case. On the table directly in front of ONE slot lies a
small ORANGE REQUEST CARD; the book standing straight above it is the requested one.
An open shallow gray tray also sits on the table.

Goal: take ONLY the requested book out of the case and lay it FLAT, covers down,
inside the tray — while every other book remains standing upright in its own slot.
Toppling, extracting or shifting any other book fails the task. The geometry forces
the librarian's move: tip the requested book's top edge forward so it pivots on its
bottom-front edge out of the row (~30°), grip its now-exposed covers, draw it out,
and lay it flat in the tray. Success is judged on the settled physical scene.

**Execution order: FORCED BY GEOMETRY (tip → extract → place), but no order is
declared or checked between independent sub-goals — there is only one object to
move.** The order is a physical consequence (nothing else exposes a graspable
surface), not a rubric clause.

## Seed provenance and why this is strategically different

The seed (`put_books_on_bookshelf`) is an **additive pick-and-place**: loose books
start on an open table, the shelf is empty, and the robot piles them ONTO it. Every
strategic property is inverted here:

- **Subtractive, not additive.** Books start dense-packed IN the case; exactly one
  must come OUT. The seed's end state (more books on the shelf) is this task's
  tested negative (smoke check 5: a book laid on top of the case earns ~0).
- **A selection problem.** The seed moves every book with no identity constraint.
  Here the requested book is sampled per episode and physically indicated by the
  re-posed request card — a memorized motion toward a fixed slot can never work.
- **A do-not-disturb constraint.** The seed's shelf is empty; nothing can be
  collaterally damaged. Here the four non-target books are live rubric state: the
  extraction must not disturb the rest of the row (smoke check 9).
- **A forced non-prehensile preparatory maneuver.** The seed is grasp-first. Here
  the packing (3 mm side gaps, full-height cheeks, 5 mm back-wall gap) makes an
  immediate grasp impossible: the only opening move is the fingertip tip-out pivot
  — a regulated rotation against gravity with a stopping problem (the book topples
  out of the case past ~35°, so the servo must stop in the 30–33° window where the
  book is still gravity-restoring and held, then the covers are gripped).
- **Different goal predicate.** Seed: books upright on a shelf. Here: one book flat
  (covers down) in a bounded tray, the REST upright — success couples the moved
  object and the untouched ones.

It is also deliberately unlike the corpus tasks I read: press_switch_i161 lives in
two rotational setpoints (nothing transported, nothing selected); pen_holder is
containment-filling of interchangeable pens. Neither has selective extraction from
a packed row under a collateral constraint.

## Mechanics

Plain rigid bodies, no joints. The case (floor/ceiling/back/two cheeks), tray
(floor + 4 walls) and request card are kinematic boxes re-posed every reset; the
five books are dynamic boxes (density 500 kg/m³ → 0.25–0.48 kg, friction 0.6,
restitution 0, sleep thresholds zeroed). Contact offset 1 mm < half the 3 mm side
gap — no phantom neighbor contacts. The case ceiling sits 240 mm above the shelf
floor: the tallest book's top-rear corner rises ~40 mm during the tip sweep
(h·cosθ + d·sinθ − h), leaving ≥10 mm of margin at full tip, while the full-height
cheeks still make vertical extraction impossible. `post_step` owns the target
book's wrench slot: it applies the world-frame `drive_f`/`drive_t` buffers every
substep (converted with the fresh quat — this stack applies wrenches in the body's
current frame) and advances the rubric latches.

## Rubric (score 0..1, anchored in the solve trajectory)

- **0.35 · tip_latch** — best forward-pivot progress `clamp(pitch/30°, 0, 1)` of the
  target, latched ONLY while the book is still in its slot (x, y, and z **below the
  ceiling** — a book parked on top of the case latches nothing) and every other
  book is undisturbed.
- **0.25 · out_latch** — target carried clear of the case (x > −0.10, above the
  table) with the row intact.
- **0.30 · tray_ok (live)** — target flat (cover normal within 15° of vertical),
  centered within 55 mm, resting at tray-floor height.
- **success()** (physical, live): tray_ok AND all four others upright (12°) in
  their own slots (±20 mm) AND everything settled → **score exactly 1.0**;
  otherwise the staged sum is capped at 0.90. Null policy ≈ 0.

## Solution outline (solve.py)

Teleports for TRANSPORT ONLY. The load-bearing interaction — the tip-out — is pure
contact dynamics: the solver writes a fingertip stand-in wrench (horizontal pull F
at body point (d/2, 0, h/2−0.012), applied as the equivalent CoM wrench F + r×F)
and the book pivots on its bottom-front edge. Outer loop: rate command
`w_des = clamp(2.5·(31.5° − θ), 0.05, 0.5)`; inner: `F = k_ff·F_ff(θ) + KF·(w_des −
w_fd)` with the quasi-static feedforward `F_ff = mgL·sin(α−θ)/(lever·cosθ)`
(α = atan(d/h) is the topple angle; 31.5° stays ≥2.9° below every book's α, so the
book is gravity-restoring the whole way — a grab-at-the-peak, not a park). F is
capped at 0.9·μmg (breakaway needs only ~0.62·μmg, so the base pivots and never
slides); pivot rate from finite differences; the gain respects the one-substep
wrench delay (KF·lever·dt/I_pivot ≈ 0.35); stalls escalate the feedforward only.
At the peak the drives are zeroed and the held book is teleported (transport) to
2 cm above the tray floor, covers down; it drops, settles, success() holds, then
≥3.5 s hands-off persistence with asserted-zero drives → `SIM_GEN_SOLVE: SUCCESS`.
Phases print non-decreasing `SIM_GEN_SCORE`: reset (~0) → tipped (0.350) → placed
(1.000) → final (1.000). Verified on seeds 0 and 1 (violet and yellow targets).

## Embodiment argument (single Franka, parallel jaw)

- **Base pose:** on the table at ~(0.55, 0.0, 0.40), facing the case (−x). Books
  stand at x = −0.21, y ∈ ±0.12, tops at z ≈ 0.58–0.61; the tray at x = 0.16.
  Everything lies inside a 0.45–0.80 m reach envelope with the approach axis
  horizontal into the open case mouth.
- **The tip:** one fingertip (closed jaws) hooks the top of the requested spine at
  z ≈ 0.60 — the case mouth is fully open above the books' front faces (front face
  15 mm behind the case front edge, ceiling 50–80 mm above the tops, neighbor tops
  ≥3 mm to the side; the fingertip is ~15 mm wide). Required pull ≤1.8 N at ≤0.5
  rad/s — fingertip-scale, and the wrist only translates ~9 cm along +x.
- **The grab:** at 30–33° the book is still gravity-restoring, so the fingertip
  HOLDS it leaning; its top ~9 cm now stands proud of the case front with free air
  on both cover faces (the neighbors are upright, this book is not) — a canonical
  parallel-jaw pinch across the 22–40 mm covers near the top edge. Jaw opening
  40 mm suffices for every book.
- **The carry + place:** lift out along the tip direction (the bottom edge slides
  free of the 3 mm slot with the covers gripped), a free-space transport, then
  lower flat into the 220×280 mm open tray (book footprint ≤130×190 mm) and
  release from ≤2 cm. No forces beyond fingertip/jaw scale anywhere; total moved
  mass ≤0.48 kg ≪ Franka payload.

## Randomization (readback-verified)

Per episode: the row permutation of the five books (rand-argsort), the requested
index (card physically re-posed under that slot), the case row center
(±30 mm) and the tray y (±80 mm). `torch.rand` only (first-`randint` degeneracy
on this stack). Smoke verifies by world-pose readback: books at their permuted
slots, card under the target slot, cheeks/tray at the sampled centers.

## Checks (smoke.py — 12, one linear run, frames.npz recorded)

1. **settle** — clean reset: finite, books physically AT their sampled slots
   (readback <6 mm), latches dark, score <0.05, no success.
2. **readback** — card under the TARGET slot, cheeks at row_c ± span/2, tray at
   (tray_x, tray_y): world pose vs formula <2 mm.
3. **random** — 8 seeds: ≥3 distinct targets, ≥4 distinct permutations, row_c and
   tray_y spread; books at their slots on every draw.
4. **null** — 2 s of nothing: score <0.05, no success.
5. **negative (seed strategy)** — target laid flat ON TOP of the case above its own
   slot (the additive move): settled there, tip latch dark (z-gate), score <0.05,
   no success.
6. **negative (orientation near miss)** — target STANDING upright in the tray
   center: not flat → no success, only latched carry-out credit (0.10–0.40).
7. **negative (position near miss)** — target flat INSIDE the tray walls but 58 mm
   off-center (> 55 mm tol; probe seed with the short yellow book, the only one
   with wall room past the tolerance): no success.
8. **negative (wrong object)** — a non-requested book in the tray, target still
   shelved: score <0.05, no success.
9. **negative (collateral)** — target correctly flat in the tray BUT another book
   toppled out onto the table: no success, only the live tray credit (~0.30).
10. **exactness** — the goal state settled → success() and score == 1.0 exactly,
    still true 1 s later.
11. **latch** — standing the book back in its slot revokes success (live state);
    only the latched carry-out share remains (0.20–0.30).
12. **frames** — ≥20 video frames saved to `frames.npz`.

## Files

- `scene.py` — cfg + scene + registration (`librarian_extract`, `simgen.librarian_extract`)
- `solve.py` — `python -u -m simgen_tasks.put_books_on_bookshelf_i190.solve --headless [--seed N]`
- `smoke.py` — `python -u -m simgen_tasks.put_books_on_bookshelf_i190.smoke --headless`
