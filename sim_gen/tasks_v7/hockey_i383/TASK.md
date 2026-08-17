# hockey_i383 — PolarityDock: orient, seat, and lock two batteries

**Env:** `simgen.polarity_dock` (`scene.py`, robot `"null"`) · **Seed:** `rlbench/hockey`

## Seed provenance

`RoboVerse/roboverse_pack/tasks/rlbench/hockey.py`: a Franka grasps a hockey
stick and **strikes a ball across open floor into an open-mouthed goal**; success
is a bbox containment check on the goal. Essence: one ballistic tool swing at a
passively available receptacle.

## Why this task is strategically different

Every element of the seed's plan is removed or inverted:

- **Nothing is propelled.** The two payloads (green battery cylinders) must be
  picked up, **reoriented end-for-end as needed**, and **laid** into open-top
  bays. A battery slid/shot along the floor at the dock just bounces off the
  cassette outer wall — the smoke battery fires exactly that probe (check 6).
- **The receptacle is a mechanical verifier, not a passive goal.** Each bay
  accepts a battery lying flat in only ONE axial orientation: the silver 12 mm
  tip must enter the bay's silver split terminal (18 mm slot between two silver
  pillars). Correct: the battery sinks 1 mm below the deck. Reversed: body 70 +
  tip 12 = 82 mm against a 76 mm trough — it can only rest tilted ≥ 22°,
  ~30 mm proud. **Which end carries the terminal is randomized per bay per
  episode** (Bernoulli 180° cassette yaw), so the required orientation must be
  perceived, not memorized.
- **The final act is a closure, not a goal crossing:** an amber lid on a D6
  prismatic slide (108 mm travel, z locked — it cannot lift) must be slid fully
  home by its red handle. A proud (reversed/unseated) battery stands ≥ 10 mm
  into the lid plane and **jams the slide**; the lid closes only over a
  correctly polarized, fully seated pair.
- Different plan shape (perceive terminal side → reorient → place-from-above ×2
  → prismatic-slide closure) and different code structure (orientation readback
  + seat placements + slide servo vs. grasp-stick + swing). Also distinct from
  the sibling hockey derivatives (i185 tip-feeder, i294, i325) and the corpus:
  the core mechanic here is **orientation-keyed admission with a
  mechanically-verifying lid**, not tipping/launching/routing.

## Teleport-solution outline (`solve.py`)

1. **P0** settle; layout readback (frame pose, per-bay terminal side recovered
   from the cassette quaternions, battery poses); assert score ≈ 0, lid open.
2. **P1/P2** per battery: one pose write stages it **above** its bay — lying
   flat, tip toward the terminal (orientation computed from the bay's own
   quat), bottom ~5 mm above the deck (the arm's place-from-above after a
   wrist-yaw reorient). It **falls ~18 mm** into the trough and settles under
   contact; a velocity-capped 0.35 N axial press squares the tip into the slot.
   Seat verdict earned by contact, never written. `SIM_GEN_SCORE` 0.28 / 0.56.
3. **P3** lid closed by a world-frame force along frame-local −y on the lid
   body (the push on the red handle): bang-bang velocity servo (0.10 m/s cap)
   with a stall-escalating force cap; force cut at the stop; re-press guard if
   the limit solver kicks it > 2 mm back off the stop. `SIM_GEN_SCORE` 1.0.
4. **P4** hands-off ≥ 3.3 sim-seconds, success must hold → `SIM_GEN_SOLVE: SUCCESS`.

Forge-verified on seeds 0 and 1 (different terminal sides, offsets, lid parks,
battery poses; scores non-decreasing 0.00 → 0.28 → 0.56 → 1.00 → 1.00).

## Embodiment argument (Franka, one base pose)

Base at ≈ (0, −0.42, 0), facing +y: everything (battery spawn zone at
y ≈ −0.20 ± 0.02, dock at y ∈ [−0.05, +0.05], lid handle park at y ≈ +0.10)
lies within a 0.15–0.55 m annulus at table height.

- **Batteries (grasp + reorient + place):** 26 mm diameter cylinder — squarely
  inside the Franka parallel-jaw span (~80 mm). Top-down grasp on the lying
  body at mid-length; wrist roll/yaw performs the end-for-end reorient (the
  tip is visually silver vs. the green body). Place-from-above: hold the
  battery horizontal, tip toward the perceived terminal, and release ~5 mm
  above the deck centered over the 32 mm-wide trough (±3 mm lateral tolerance
  vs. the 6 mm y-clearance; the drop is 18–27 mm — exactly what the solve's
  teleport stages). The final squaring is a fingertip nudge along the trough
  (the solve's 0.35 N axial press).
- **Lid (slide closure):** red handle bar (90 × 14 × 30 mm) on top of the
  plate, graspable from above or pushable from its +y face; the required
  motion is a straight horizontal 108 mm stroke at constant height (the
  joint carries all constraint forces), ending pressed lightly against the
  stop — the solve's velocity-capped push, well under arm force limits (the
  escalated caps ≤ 20 N are far below the Franka's ~87 N).
- No interaction requires reaching under, through, or behind anything; the
  troughs and the handle are open from above at all times.

## Execution order declaration

Seat order between the two batteries is free (they are interchangeable and the
bays are interchangeable); the **lid must be closed last** — it is not an
arbitrary declared rule but is physically forced: the closed lid roofs both
troughs (a battery dropped on it never seats — smoke check 8), so seating after
closing is impossible, and the lid credit (`_lid_ever`) only latches while both
batteries sit seated.

## Check list (smoke.py — rejection-only battery, 17 checks)

1. Settle/no-NaN + mass readback (frame > 30 kg, lid ≈ 0.30 kg, batteries ≈ 0.08 kg).
2. Reset score ≈ 0, no success.
3. Randomization: per-bay terminal-side Bernoulli flips (readback, both bays).
4. Randomization: battery jitter + heading spread, apparatus yaw + offset spread.
5. Null policy 240 steps: score ≈ 0, lid still open.
6. **Seed-strategy analog:** battery fired along the floor at the dock bounces
   off the cassette outer wall — never seats, score ≈ 0.
7. **Empty-close (free-travel proof):** the lid servo closes the empty dock
   fully — `lid_closed()` true but NO lid credit, NOT success.
8. Battery dropped onto the closed lid: never seated.
9. **Reversed polarity:** dropped tip-away rests tilted/proud — never seats.
10. **Lid jam:** the same escalating-force servo (cap escalated to 20 N) is
    stopped ≥ 30 mm short of closed by the proud battery — no credit.
11. Battery laid across the deck (on the side walls): z band rejects.
12. Battery standing vertical in the trough: flatness/z gates reject.
13. Battery beside the cassette outer wall: bay-frame math rejects.
14. One-seated partial = exactly 0.28; lid closed over the half-empty dock
    earns no lid credit, NOT success (both bays must be filled).
15. Latched credit survives removing the seated battery; still no success.
16. Rejection audit: success() never True anywhere in the battery.
17. Final no-NaN. — plus `frames.npz` recorded from the viewport annotator.
