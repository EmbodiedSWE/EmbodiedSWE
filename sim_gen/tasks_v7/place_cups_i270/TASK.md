# place_cups_i270 — cargo_shuttle

Seat three color-coded canisters into the matching sockets of a rail shuttle, then push
the loaded shuttle down its guide channel, under the low canopy of a covered delivery
bay, until it docks against the end stop.

- Scene: `cargo_shuttle` (`simgen.cargo_shuttle`, robot="null")
- Modules: `scene.py`, `solve.py`, `smoke.py`
- Verified on forge: `solve` → `SIM_GEN_SOLVE: SUCCESS` on seeds 0 and 1 (~21 s each);
  `smoke` → `SIM_GEN_SMOKE: ALL PASS 17/17` (frames.npz recorded, 469 frames).

## Seed provenance

Derived from **rlbench/place_cups** (`RoboVerse/roboverse_pack/tasks/rlbench/place_cups.py`):
pick each of N identical mugs off a table and hang it on its own peg of a static
mug-tree — N repeated, interchangeable prehensile transports, judged per-object by
proximity to a dedicated **static** target.

## Strategic difference

The seed's whole content is "N times: move object i to static target i". Here:

1. **No target is static.** The sockets ride a VEHICLE (a low shuttle captive in a
   guide channel). Socket membership is judged in the shuttle's moving body frame
   (`_cup_local` via `quat_apply_inverse`), not by distance to a world-fixed peg.
2. **Objects are not interchangeable.** Socket rims are colored; only the
   color-matched canister counts (`seated(nm)` checks that socket's center). The
   seed's mugs are identical and any-mug-on-any-peg would do.
3. **Placement is only half the task.** Nothing succeeds until the loaded shuttle is
   itself transported: pushed by its rear handle down the channel, under the bay roof,
   onto the end stop (`docked()` on the shuttle's own x readback). The final stage is
   a captive prismatic transport of the whole assembly — a skill the seed never asks
   for.
4. **Execution order is geometry-enforced, not rubric-enforced.** The bay roof passes
   ~18 mm above a seated canister's lid, so a canister can NEVER be lowered into a
   socket once the shuttle is docked — it lands on the roof (proven live by smoke
   check 14). Load first, ship second. The seed has no ordering structure at all.
5. **Different code structure.** One dock predicate + three moving-frame membership
   predicates + a stillness counter-latch, instead of N object-near-static-target
   checks; the solve is 3 gravity-seated drops + a velocity-servo'd force push,
   instead of N identical pick-and-hang transports.

It is also different from every other tasks_v7 package (in particular from
`place_cups_i82`, same seed: a mass-partition balance-scale task judged by statics —
here there is no balance, no mass reasoning; the physics content is guided transport
under clearance constraints).

## Solution outline (solve.py, teleport = transport only)

- P0: reset, settle, mass/layout readback, plan from `describe()`; assert score ~0.
- P1–P3: for red, green, blue — teleport the canister (zero velocity, shuttle-aligned)
  to 20 mm above its color-matched socket floor computed from the LIVE shuttle pose,
  release; **gravity and the socket rims seat it through contact**. Assert `seated`,
  score non-decreasing (0.10 / 0.10 / 0.30 with the all-seated latch).
- P4: **the shuttle is never teleported.** A velocity-servo'd horizontal external
  force (the arm's stand-in at the rear handle; `set_external_force_and_torque`,
  gain escalation on stall, slow final approach) pushes the loaded shuttle down the
  channel, under the canopy, onto the end stop. Assert `docked` + all still seated.
- P5–P6: wait for `success()` (stillness counter fills), then ≥3.3 simulated seconds
  hands-off; `SIM_GEN_SOLVE: SUCCESS` only if it persists.

Score is latched (credit never evaporates): 0.10/canister ever seated matched +
0.10 ever docked + 0.20 all seated at once + 0.30 ever delivered
(docked & loaded & settled), capped 0.90; 1.0 iff `success()` live.

## Embodiment argument (single Franka arm)

Base at ~(-0.15, -0.50, 0): the canister ground row (y ≈ -0.30, x ∈ [-0.41, 0.09])
and the shuttle loading zone (x ∈ [-0.30, -0.24], y = 0) are all within ~0.55 m reach.

- **Canisters** (⌀60 mm × 60 mm, 80–140 g): side pinch fits a parallel jaw (~80 mm
  opening); lift over the 50 mm socket rim, lower to ~20 mm above the seat, release —
  exactly the release height the solve demonstrates; gravity finishes the seat.
- **Shuttle push**: the orange handle (20 × 60 mm post, 150 mm tall) at the shuttle's
  rear is a flat pushing face; the arm presses its closed fist / gripper flat against
  it and walks the shuttle +x. The handle top stands above the roof line and stops
  ~10 mm short of the canopy at full dock, so the end-effector never has to enter the
  bay — the last 0.33 m of shuttle travel happens under the roof while the contact
  point stays outside. Push force demonstrated: ≤ 12 N cap, ~2.5 N sustained — well
  inside Franka capability.
- **Order**: load-before-ship is forced by the canopy geometry; the arm loads in the
  open zone (full overhead clearance), then only pushes horizontally.

## Execution-order declaration

Load-before-ship is REQUIRED and geometry-enforced (canopy clearance; smoke check 14
proves a post-dock load attempt is physically impossible). The loading order among the
three canisters is free. Docking the empty shuttle first is legal but useless — it
must come back out to be loaded (and earns only the 0.10 dock latch).

## Smoke battery (17 checks, all PASS on forge)

1. settle: finite, canisters upright on the floor, shuttle at rest in the start band
2. settle: score ~0, no success
3. randomization: canister masses vary (PhysX-view readback, cache agreement)
4. randomization: shuttle start x varies, slot permutation varies, xy jitter real
5. null policy: 240 idle steps → score ~0
6. seed-analog (each object at its own static spot: row on the bay roof) → rejected
7. loaded but not docked → all_seated true, NOT success, score 0.50
8. docked but empty → docked true, NOT success, score ≤ 0.15
9. wrong colors (cyclic mismatch, physically in pockets) → nothing seated, score ~0
10. stacked (red on seated green, local z ≈ 0.100) → z window rejects
11. toppled inside the socket (z INSIDE the window!) → upright cone rejects
12. near-dock miss (40 mm short) → no dock credit, NOT success
13. latched credit survives removing a seated canister
14. canopy reality: canister dropped over the docked socket lands ON the roof
15. settle gate: full goal state judged 2 steps after arrival → NOT success
16. rejection audit: success() never true anywhere in the battery
17. final no-NaN

## Run

```
python -u -m simgen_tasks.place_cups_i270.solve --headless [--seed N]
python -u -m simgen_tasks.place_cups_i270.smoke --headless
```
