# scoop_with_spatula_i97 — Weighbridge: tip the beam by loading iron cubes into its pan

## Seed provenance

Seed task: `rlbench/scoop_with_spatula` (RoboVerse
`roboverse_pack/tasks/rlbench/scoop_with_spatula.py`): "scoop up the cube and lift it
with the spatula". One 20 mm cube, one spatula tool (USD asset), a Franka, and a
trajectory file; the whole skill is TOOL MEDIATION — grasp the thin blade, slide it
under the cube, lift, and be judged on the cube riding aloft on the blade.

## What changed and why it is strategically different

The manipulation model is replaced wholesale, not perturbed:

| | seed | this task |
|---|---|---|
| plan | tool-mediated scooping (slide blade under, lift) | mechanism actuation by ACCUMULATED LOAD |
| objects | 1 cube + 1 spatula | 6 visually identical cubes, 3 iron / 3 foam, + a hinged weighbridge |
| perception | locate the cube | MASS IDENTITY: iron vs foam, same size/shape, colour-coded, slots permuted per episode |
| reasoning | none beyond the scoop | lever causality: load counts ONLY inside the pan, on the pan arm; one cube is NOT enough — the sum of two is |
| judged on | cube carried aloft on the blade (transient, hand-supported) | the MECHANISM's settled state: beam tipped past `tip_deg` and HOLDING there hands-off |

Nothing is scooped, slid under, or held aloft; there is no tool. It is also different
from every other task package read while building it (pen_holder: containment fill;
close_grill_i8: structure building / load path to a height band): here the deliverable
is the state of a constrained mechanism, reached only by selecting the heavy cubes and
accumulating their weight at the right point of a lever.

The seed's own end state — a cube held aloft on a hand-supported blade — is not a
settleable state under a hands-off judge, so its nearest constructed analogs in the
smoke battery are the "weight in the wrong place" probes (checks 8–9) and the forced
empty tip (check 7), all rejected.

## The scene

Fully procedural. A kinematic pillar (fixed pose) carries a 56 cm dynamic compound
beam on a spawn-authored Y-axis revolute joint with hard stops at ±12°. The beam's
local origin IS the hinge point; explicit MassAPI (mass 0.55 kg, CoM offset 0.16 m
toward the counterweight arm, diagonal inertia) makes the counter-torque exact. One
arm carries a black counterweight block (the visual mass argument), the other an open
orange pan tray (10 cm square cavity, 32 mm walls). Six 40 mm cubes lie on ground
scatter slots: iron 0.32 kg (dark), foam 0.006 kg (pale yellow).

Torque budget — asserted in `WeighbridgeSceneCfg.__post_init__`, including worst-case
in-pan cube positions and the CoM-height correction at the rest angle:

- counter-torque ≈ 0.863 N·m;
- 2 iron at the inner pan wall ≈ 0.974 N·m → tips (≥ +8 % margin);
- 1 iron + all 3 foam at the outer wall ≈ 0.763 N·m → stays (≥ 8 % margin);
- 3 foam anywhere ≤ 0.054 N·m → nowhere close.

So identity errors and under-accumulation fail by physics, not fiat.

Per-episode randomization (readback-verified in smoke checks 2–3): the six cubes are
permuted over the six scatter slots with ±30 mm xy jitter and free yaw. The
weighbridge fixture is fixed ON PURPOSE: the hinge anchor is spawn-authored in the
pillar's frame, and a teleported fixture leaves its joint anchor behind (measured
quirk), so the pillar is never randomized or teleported.

## Teleport solution (`solve.py`)

Teleports are TRANSPORT ONLY; every load-bearing interaction is contact/joint physics:

- **P0** reset + settle: beam drops onto its counterweight stop (θ ≈ −12°), cubes seat
  on the ground. `SIM_GEN_SCORE 0.00`.
- **P1** carry `iron_0` to a free-space hover a few cm above the open pan cavity
  (computed in the beam's body frame so it tracks the tilt), zero velocity; it FALLS
  into the tray and settles. The counterweight still wins — the mid-point θ < 0 is
  itself part of the certificate (the accumulation claim). `SIM_GEN_SCORE 0.25`.
- **P2** same carry for `iron_1`; the summed load tips the beam through horizontal to
  the pan-side stop — pure lever dynamics, the beam is never written.
  `SIM_GEN_SCORE 1.00`.
- **P3** hands-off persistence ≥ 3.3 simulated seconds; success must still hold.
  `SIM_GEN_SOLVE: SUCCESS`.

Scores are monotone by construction (latched partial credit). Verified on the forge
for seeds 0 and 1.

## Rubric

Latched partial credit (survives transients), success judged live on the settled state:

- 0.10 `appr` — an iron cube ever carried within 0.20 m of the pan centre;
- 0.15 `iron1` — one iron cube ever calm inside the pan cavity;
- 0.15 `iron2` — two iron cubes ever calm inside the pan cavity;
- 0.15 `cross` — the beam ever reached horizontal WITH load in the pan;
- 0.20 `high` — the beam ever within 2° of the low stop with 2 iron aboard;
- non-success capped at 0.75; exactly 1.0 iff `success()`: θ ≥ 10° pan-down, ≥ 2 iron
  cubes inside the pan, everything settled and finite.

The angle latches are gated on load-in-pan, so pressing (or teleporting) the empty
beam down earns nothing — proven by smoke check 7.

## Smoke battery (`smoke.py`, 10 checks)

1. settle/no-NaN baseline; 2. randomization readback (permutation + iron_0 xy);
3. permutation coverage over 10 resets; 4. null policy ~0; 5. FOAM CANNOT TIP (3 foam
in the pan, beam stays down, score ~0); 6. ONE IRON SHORT (1 iron + 3 foam: never
crosses horizontal, only appr+iron1 ≤ 0.26); 7. EMPTY TIP REVERTS (beam forced to the
pan-down stop, counterweight rights it, load-gated latches stay dark); 8. WRONG SIDE
(2 iron on the counterweight block, score ~0); 9. ARM OUTSIDE PAN (2 iron near the
hinge: insufficient torque AND outside the tray, ≤ 0.11); 10. frames.npz video.

## Embodiment argument (single Franka + parallel jaw, OSC)

- Every graspable object is a 40 mm rigid cube ≤ 0.32 kg — a canonical parallel-jaw
  pick (Franka jaw opening 80 mm, payload 3 kg).
- The pan tray is open on top; its cavity is 100 mm square with only 32 mm walls, and
  it rides at ~0.19 m (pan up) to ~0.11 m (pan down) — a plain top-down place with
  ample clearance; the solve's release-from-hover mirrors a gripper release a few cm
  above the tray.
- Scatter slots span x ∈ [−0.05, 0.61], y ∈ [−0.41, −0.27]; the pan centre sits near
  (0.60, 0.10). A base at ≈ (0.30, −0.05, 0) puts every pick and the place inside a
  comfortable 0.35–0.55 m reach annulus, with no obstacles between zones.
- Forces: tipping needs no applied force at all — gravity does the actuation; the
  robot only transports.

## Execution order

No required order beyond the goal state: either iron cube may go first, foam may be
ignored entirely, and the third iron cube is never needed. (The one-iron midpoint in
`solve.py` is a property of ANY two-cube trajectory, not an ordering constraint.)

## Checks

- forge `solve` seed 0: `SIM_GEN_SOLVE: SUCCESS` (scores 0.00 → 0.25 → 1.00 → 1.00);
- forge `solve` seed 1: `SIM_GEN_SOLVE: SUCCESS`;
- forge `smoke`: `SIM_GEN_SMOKE: ALL PASS 10/10` + `frames.npz`.
