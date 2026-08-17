# approach_grasp_ceramic_teapot_i109 — Mug Rack Hang (scene `mug_rack_hang`)

Hang the WHITE mug by its closed handle window on the GREEN-marked peg of a
three-peg wall rack, and let go: the judged outcome is a free SUSPENDED
EQUILIBRIUM — mug airborne, peg through the handle window, hanging still with
nothing but peg contact and gravity holding it (env `simgen.mug_rack_hang`,
robot `null` · files: `scene.py`, `solve.py`, `smoke.py`).

## Seed provenance

Seed: `pick_place/approach_grasp_ceramic_teapot` (`RoboVerse/roboverse_pack/
tasks/pick_place/approach_grasp_ceramic_teapot.py`) — a Franka approaches a
ceramic teapot among table clutter and closes its jaw around it; success is a
gripper-object distance relation held a few frames, then a small lift. The
seed's plan is "reach the known object and acquire it"; the episode ends
HOLDING the object.

## What changed, and why it is strategically different

The seed's object-acquisition plan is discarded entirely:

1. **Anti-seed end state.** The seed ends holding; here holding is worth at
   most the two small carry latches (0.20) and the episode only succeeds after
   the hand LETS GO. Smoke check 5 holds the white mug perfectly still in the
   air right beside the target peg with a bounded wrench hand — rejected
   (threading clause) and capped at the carry latches. Smoke check 4 executes
   the full seed strategy (approach, carry, set down on the rack's base plate)
   — rejected (airborne clause). No gripper-object relation is ever read.
2. **The judged state is a built equilibrium in suspension.** Corpus contrast:
   the beam-balance family (`coke_task_i15`, `pick_and_lift_i16`,
   `pull_cube_i20`) adds weights to pre-built pivoting mechanisms and reads a
   mechanism angle; `approach_grasp_i29` (ridge poise) builds a standing stack
   on supports below; `track_bowl_i27` and `hockey_i325` operate caged balls;
   `approach_grasp_banana_i69` gravity-steers a sealed maze. None judges an
   object hanging FREE on a cantilever through its own closed aperture — the
   support is a 14 mm peg threading a 26 x 39 mm window in the payload itself,
   and the mug's rest pose (slid to the peg root, leaning on the panel) is
   found by contact dynamics, not placed.
3. **Per-episode target selection by a visual marker + a decoy.** A green
   marker block on the panel top edge is re-posed every reset above 1-of-3 peg
   columns (torch.rand draw); an identical BLACK decoy mug spawns beside the
   white one. Hanging the black mug on the green peg (check 6) or the white
   mug on a wrong peg (check 7) are both genuine, settled hangs — each is
   rejected by exactly one identity clause, and the wrong-peg hang leaves the
   thread/hang latches unfired.
4. **Success is live and streak-gated, never bookkept.** `success()` requires
   the peg axis to cross the open window rectangle (computed in the mug's
   frame — tilt-invariant), the mug center above every stand-state height in
   the scene (floor, base plate, even one mug stacked on the other — all
   asserted >= 15 mm below the gate), and an uninterrupted 60-substep still
   streak counted in `post_step`. Teleporting the mug into the perfect hang
   pose and sampling immediately earns nothing (check 9); disturbing a real
   hang collapses success to the latched 0.60 (check 10).

## Scene (engineered numbers, asserted in `__post_init__`)

- KINEMATIC rack at nominal (0.45, 0), yaw ±25°, xy jitter ±30 mm: base plate
  160 × 300 × 30 mm, upright panel 30 × 260 × 350 mm (top at 0.380). Three
  square pegs (14 mm, length 85 mm) cantilever from the panel face, tilted 12°
  UP, at (y, z) = (−0.08, 0.24), (0, 0.33), (+0.08, 0.285). Pegs bind a slick
  material (μ 0.15/0.12) — tan 12° > 1.3 μ, so a hung handle slides to the peg
  root and rests leaning on the panel (kills pendulum swing; asserted).
- Green marker cube (24 mm, KINEMATIC) on the panel top edge directly above
  the target peg's column, re-posed at reset; sits above the swept volume of
  any hung mug (asserted).
- Mugs ×2 (white judged, black decoy; DYNAMIC, 0.25 kg, explicit mass / CoM /
  diagonal inertia): 64 mm square cup (4 walls + floor) + closed 3-bar loop
  handle forming a 26 × 39 mm open window. Window beats the peg diagonal with
  clearance (asserted); jaw-fit asserted.
- Height-band statics asserts: worst-case hang (60° handle tilt) stays ≥ 10 mm
  ABOVE the airborne gate (0.155 m); tallest stand state (mug stacked on mug,
  0.135 m) ≥ 15 mm BELOW it; hang top clears the panel top; peg lateral
  separation clears a hung cup body; weights sum to the cap.
- Randomization (readback-verified in smoke): target peg 1-of-3 via torch.rand
  comparison (marker readback matches the drawn peg), rack yaw/xy, both mug
  floor spawns in a 0.16–0.42 m annulus with rack keep-out 0.30 m, mug-mug
  separation 0.22 m, free yaw.

## Rubric (`success` / `score`)

`success()` — all live: (a) THREADED — the green peg's centerline, transformed
into the white mug's frame, crosses the handle-window rectangle (shrunk 4 mm)
within the peg's span (4 mm end margins), direction gate |d·ŷ| > 0.5;
(b) AIRBORNE — mug center z > 0.155; (c) HUNG STILL — 60 consecutive substeps
with |v| < 0.05 m/s and |ω| < 0.40 rad/s while (a) ∧ (b) hold (streak counted
in `post_step`, reset on any violation).

`score()` — latched (credit never evaporates): 0.10 carried-near (aperture
within 0.10 m of the green peg mid) + 0.10 lifted-near (airborne + within
0.15 m) + 0.20 ever-threaded + 0.20 ever-hung-settled, cap 0.60; exactly 1.0
iff `success()` live. Null policy scores 0 (floor spawns are far from every
peg; asserted).

## Solution (`solve.py`) — teleport is transport only

- **P0** — settle 150 steps; READBACK asserts: rack pose/yaw, green peg index,
  marker sits over the drawn peg's column (|Δy| < 5 mm), both mugs on the
  floor, score ≤ 0.03, no success.
- **P1 STAGE** — teleport the WHITE mug through free space to a staging pose:
  window center 12 mm off the green peg tip, window normal along the peg axis,
  mug upright (pure transport — no contact at the staging pose). Score
  latches near + lift → 0.20.
- **P2 INSERT** — a bounded external-wrench "hand" (weight hold + ≤ 2.5 N
  velocity-regulated push along the peg axis + ≤ 2 N lateral PD + ≤ 0.15 N·m
  attitude torque) slides the window over the peg to 32 mm depth. Threading
  latches → 0.40. All load-bearing interaction is contact dynamics.
- **P3 RELEASE** — zero the wrench; gravity slides the handle down the slick
  tilted peg to the root, the cup leans on the panel and settles; the
  60-substep streak fills → score 1.0, success live.
- **Persistence** — 400 steps (≥ 3.3 sim-seconds) fully hands-off, success
  re-sampled 10×, all required, then `SIM_GEN_SOLVE: SUCCESS`. Watchdog
  hard-exit. Up to 5 staged retries (none needed on any verified seed).

Verified on the forge: seeds 0, 1, 2, 3 — all rc=0, first-attempt, monotone
`SIM_GEN_SCORE` ladder 0.00 → 0.20 → 0.40 → 1.00, hang settles ~28 mm along
the peg with ~10 mm lateral offset (slid to root, leaning on the panel).
Seeds 0/1 drew the high peg, seeds 2/3 the low peg — the draw varies.

## Embodiment argument (Franka)

A Franka based at the origin covers the workspace: the rack at (0.45, 0) ± 3 cm
and the whole 0.16–0.42 m spawn annulus sit inside its ~0.85 m reach; the
highest peg tip is at ~0.35 m, mid-workspace. The mug is a 64 mm cup with a
protruding handle: grasp the cup body across two opposite walls (64 mm < 80 mm
jaw) or pinch the 12 mm handle bar; 0.25 kg ≪ payload. Threading needs the
26 × 39 mm window over a 14 mm peg — 6 mm per-side clearance at the tightest
axis, and the solve's own hand achieves it with ≤ 2.5 N regulated pushes and
coarse (±5 mm-class) lateral control, watching the high-contrast white-on-dark
window against the peg. Release is trivial (open jaw and retract); the slick
tilted peg self-centers the hang, so no precision set-down exists anywhere.
The marker is a 24 mm saturated-green cube on the panel top edge — an easy
color read. No moving mechanism threatens the arm; the tallest structure is
0.392 m.

## Execution order (declaration)

1. Designed the geometry/statics and a *minimal* `success()` first.
2. Wrote `solve.py` and iterated it on the forge until the hang was reached
   robustly — seeds 0–3 SUCCESS, first-attempt, monotone scores.
3. Only then finalized the rubric (latches, streak gate, score shape) and the
   `__post_init__` statics asserts.
4. Wrote `smoke.py` last, as an adversarial battery against the final rubric.
5. Re-ran solve (seeds 0 and 1) and smoke clean on the final submitted code.

## Checks

`solve.py` (per run): P0 readback asserts, per-phase monotone score asserts,
final score ≥ 0.999 with success live, 3.3 s hands-off persistence sampled
10×. Passed on seeds 0, 1, 2, 3.

`smoke.py` — `SIM_GEN_SMOKE: ALL PASS 12/12` on the forge:
1. Settle / no-NaN: mugs upright on the floor, score 0, no success.
2. Randomization readback over 6 resets: ≥ 2 distinct green pegs, marker
   column matches the drawn peg every reset, rack yaw spread, mug spawn spread.
3. Null policy 300 steps → score ~0, no success.
4. SEED STRATEGY (set-down): white mug carried and set down on the rack base
   plate — grounded, rejected by the airborne clause, score ≤ 0.20.
5. SEED STRATEGY (hold): white mug held perfectly still in the air beside the
   green peg by a bounded wrench hand — rejected (not threaded), ≤ 0.20.
6. BLACK DECOY: the black mug genuinely hung on the green peg (settled,
   verified hanging) — rejected purely by mug identity, white latches unfired.
7. WRONG PEG: the white mug genuinely hung on the farthest gray peg —
   rejected by peg identity; thread/hang latches unfired, score ≤ 0.20.
8. RIM HANG cheat: white mug hung on the green peg by its RIM (peg inside the
   cup body, a genuine stable hang) — rejected by the window clause.
9. NO INSTANT CREDIT: white mug teleported into the exact final hang pose and
   sampled immediately — streak not teleportable, no success at sample time.
10. DISTURB/RECOVER: a real hang built (score 1.0, success live), then the mug
    is knocked; success collapses to the latched 0.60 floor.
11. Rejection audit: success() never fired during any negative probe (4–9).
12. frames.npz recorded (≥ 20 frames, 600 × 960 × 3).
