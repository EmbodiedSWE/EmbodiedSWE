# open_grill_i343 — Shift-Park Grill

**Scene name:** `shift_park_grill` · **Env:** `simgen.shift_park_grill` (robot="null")

## Seed provenance

Seed task: `rlbench/open_grill` — "open the grill": a barbecue USD with a hinged lid;
the robot grasps the lid handle and rotates it up about its single revolute hinge,
judged by the lid joint angle. The seed's whole interaction is one grasp + one
monotone hinge rotation, and "open" is a joint-angle threshold on that one DOF.

## Strategic difference

The lid here rides a **worn 2-DOF mount** — a spawn-authored D6 joint with a
translational slide (0..80 mm along the grill's fore axis) *and* the pitch hinge
(0..85°) — and the seed's strategy (grab the handle, swing up) **jams immediately**:
the lid spawns slid FORWARD with its plate tip tucked under an overhanging CATCH
bar, so the upswing hits the catch underside at **1.7°** (predicted `jam_deg`,
measured 1.70° peak in smoke check 4). Opening is a forced 4-beat sequence on the
two coupled DOFs, and the goal is compound (park the lid open, then deliver a patty
through the mouth it exposes):

- **Slide BACK first** (the only free motion at spawn): 80 mm rearward pulls the
  tip out from under the catch (tip must retreat past `catch_x0` — asserted with
  jitter margin) and only then does the pitch DOF open up.
- **Raise at the rear**: the rear-pin sweep radius of every lid point
  (`max_sweep_r` 0.270 m) clears the high REST BAR (`rho_rear` 0.289 m,
  clearance asserted ≥ 5 mm incl. contact offsets), so the lid can swing to ~82°
  without touching anything.
- **Slide FORWARD while raised**: at the forward pin the bar radius `rho_fwd`
  0.240 m falls *inside* the plate span — the bar is now in the fall path.
- **Release → gravity-park**: the pitch cap (85°) is strictly under over-center,
  so gravity always closes a free lid; the falling forward-slid plate lands on the
  rest bar at `theta_bar` ≈ 60.0° and — because tan 60° beats pair-averaged
  friction (asserted at 1.5×) — creeps back down the slide until the underside
  CLEAT seats against the bar (settled slide ≥ `slide_park_lo` 56 mm > the 50 mm
  `parked` clause). The park is a *real force balance* on the bar, not a joint
  stop: released at the REAR instead, the identical raise ends fully closed
  (smoke check 5).
- **Patty through the mouth**: the parked lid exposes the chamber's only
  patty-sized aperture — the closed-lid hover gap (24 mm) under-sizes the patty
  (34 mm, asserted), grate slat gaps under-size it, and a patty dropped ON the
  closed lid rests ~120 mm above the accepted on-grate z window.

Differs from every corpus task read: the seed and `open_grill_i260` (same seed:
prop-rod propping of a plain hinge lid) operate one revolute DOF;
`close_grill_i8` closes a hinge with a slam; `open_window_i283` / `close_door_i274`
are single-DOF swing panels with different gating; `open_box_i184` rolls a
gravity-seated roller; `meat_off_grill_i64` extracts cargo from a static grill.
**No corpus task couples a translational slide and a pitch hinge on one body so
that opening requires slide-back → raise → slide-forward → gravity-park onto a
rest bar**, and none makes "open" a bar-supported force balance instead of a
joint angle.

## Assets (fully procedural)

- **grill** — DYNAMIC 40 kg compound (effectively stationary): chamber
  260×200×160 mm with 10 mm walls, 4-slat grate at z 0.075, side BOARD (patty
  staging, y 0.18..0.30, h 0.10), front CATCH bar (x 0.150..0.185, underside
  z 0.204) on posts, rear slide RAILS, and a high REST BAR (12 mm square, centre
  x 0.020 / z 0.398) between towers.
- **lid** — DYNAMIC 0.35 kg plate (250×240×12 mm) with handle, underside CLEAT
  (x 0.256..0.268, h 14 mm), and mount lugs; a spawn-authored `UsdPhysics.Joint`
  (D6) with per-axis LimitAPI gives transX ∈ [0, 0.08] and rotY ∈ [0°, 85°],
  all other axes locked. Explicit CoM + diagonal inertia authored.
- **patty** — DYNAMIC 60×60×34 mm 0.10 kg brown block, spawns on the board.

Randomization (readback-verified, smoke check 2): grill yaw ±20° + xy ±30 mm,
lid spawn slide back-off 0..6 mm, patty board-frame xy jitter.

Honesty geometry asserted in `__post_init__` (31 asserts): catch jams at 1.7° and
the tip can't exit mid-jam; `raised` unreachable while caught; rear sweep clears
the bar, forward pin puts it inside the plate; park angle centred in the band with
a ≥15° fall gap; creep-to-cleat friction inequality; settled slide beats the
forward clause; mouth-is-the-only-aperture (hover gap, slat gaps, on-lid rest vs
z window); patty drop corridor past the parked plate; cleat/towers/board
clearances; board rest fails the grate window.

## Rubric

Latched partial credit (never evaporates), finite-guarded:

- 0.15 `released` — lid ever at rear slide (≤ 15 mm) with pitch under 15° after
  having moved (freed from the catch)
- 0.20 `raised` — pitch ever past 70° (unreachable while caught: jam at 1.7°)
- 0.30 `parked` — lid settled in the 50–80° band at forward slide (≥ 50 mm) —
  only the bar can hold this (pitch cap < over-center)
- non-success capped at 0.70; **1.0 iff success()**: lid parked AND patty inside
  the on-grate window (grill-frame |x| ≤ 0.105, |y| ≤ 0.080, z 0.055..0.130)
  AND grill upright AND everything settled and finite — all judged live.

## Teleport solution (solve.py)

Teleports for transport only (the patty carry); every lid interaction is applied
CoM forces through the D6 joint's contact dynamics:

- **P0** reset, settle: lid forward-closed under the catch, patty on the board.
- **P1 SLIDE BACK (applied force)** velocity-regulated rearward CoM force
  (stiction floor with stall escalation, force-frame mode probed from progress)
  to the rear stop. Score 0.15.
- **P2 RAISE (applied force)** gravity-feedforward + rate-PD vertical CoM force
  swings the lid to ~82° at the rear. Score 0.35.
- **P3 SLIDE FORWARD + RELEASE (applied force)** forward slide servo with an
  exact `f_x·tanθ` closing-torque feedforward holding pitch high; at the forward
  stop, re-steady ≥ 78°, then ramp the lift out over 0.5 s — the lid falls onto
  the rest bar and creep-seats on the cleat at 60.2° / 64.5 mm. Score 0.65.
- **P4 PATTY (teleport transport)** patty carried to a hover above the exposed
  front grate (drop x computed live past the leaning plate's max x), released,
  settles between slats. Score 1.00.
- **P5** ≥ 3.3 simulated seconds hands-off; success persists →
  `SIM_GEN_SOLVE: SUCCESS`.

## Execution order

The order is **physically forced**: raise-before-slide-back jams on the catch at
1.7° (assert + smoke check 4); slide-forward-before-raise just re-enters the
catch pocket; release-before-slide-forward falls fully closed past the bar
(rear-pin clearance assert + smoke check 5); patty-before-park has no aperture —
the closed-lid hover gap and slat gaps both under-size the patty and an on-lid
rest is 120 mm above the z window (asserts + smoke checks 6/7). Declared order:
slide back → raise → slide forward → release/park → patty through the mouth.

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin, facing +x; grill at (0.35, 0). The lid handle
crossbar (grill-frame x 0.20, z ≈ 0.22 closed) sits ~0.55 m out at spawn —
reachable, and a parallel-jaw grasp across the crossbar is the natural hold.

- **One continuous grasp does the whole lid sequence**: slide back (80 mm
  horizontal pull), raise (the handle arcs on a 0.38 m radius to ~0.5 m height —
  inside the Franka envelope with the wrist pitching to track), slide forward
  raised, and lower until the bar takes the load, then open the fingers. Peak
  applied force in the solve is ≤ 6·m·g ≈ 21 N vertical and ≤ 2.5 N horizontal —
  trivial for the arm. No regrasp is needed because the handle never leaves the
  gripper's approach half-space (pitch stays ≤ 85°).
- **Patty**: a top pick off the open side board (z 0.117) and a top drop through
  the exposed mouth — the drop corridor is asserted ≥ patty + 20 mm wide and the
  parked plate leans AWAY from the corridor.
- No step needs a second arm; the grill is 40 kg and never moves.

## Files

- `scene.py` — cfg (+31 honesty asserts), compound grill/lid/patty spawners
  (spawn-authored D6 slide+pitch joint), scene (latched rubric in `post_step`),
  `register_env`.
- `solve.py` — force-driven lid sequence + transport-only patty teleport;
  watchdog + hard exit.
- `smoke.py` — 12-check rejection battery (settle, 3-seed randomization
  readback, null policy, seed-strategy handle press — jams at 1.70°, rear
  release falls closed, patty-inside-closed-lid, patty-on-lid, parked-but-patty-
  elsewhere (board + ground), moving-patty settle gate, success-never-True
  audit, score-cap audit, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, ~22 s), score trace
  0.00 → 0.15 → 0.35 → 0.65 → 1.00
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, ~22 s), same trace
- forge solve seed 7: `SIM_GEN_SOLVE: SUCCESS` (rc=0, ~22 s), same trace
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 12/12` (rc=0, 43.7 s); catch press
  peaks at 1.70° (predicted 1.7), rear release ends closed at 0.0°, park probe
  settles at 60.9° / 61.1 mm, battery-wide max score 0.35
