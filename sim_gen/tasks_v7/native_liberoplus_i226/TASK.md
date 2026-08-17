# native_liberoplus_i226 — Balance Verdict

**Env:** `simgen.balance_verdict` (scene `balance_verdict`, robot `"null"`)
**One line:** Two visually identical canisters, one secretly ~3.5x heavier — weigh them
both on a two-pan balance scale (the required FIRST step), read the verdict from the
beam's tilt, then place the heavier one upright on the red pedestal with the lighter
one kept clear.

## Seed provenance

Seed: `libero/native_liberoplus` (RoboVerse
`roboverse_pack/tasks/libero/native_liberoplus.py`) — the LIBERO-plus corpus: LIBERO
pick-and-place scenes replayed under scene-level perturbations (moved distractors,
altered layouts, camera/lighting changes). The seed's strategy is invariant across the
whole family: the target object's identity is GIVEN visually by the instruction, the
perturbations are nuisance factors to be ignored, and the plan is always
*locate visually → grasp → transport → release on the goal*.

## Strategic difference

The defining move of the seed is *robust visual grounding*; the defining move here is
a *physical information-gathering experiment* — and the two are mutually exclusive:

1. **The perturbation IS the task and is invisible.** The two canisters share size,
   shape and color; only mass differs (0.42 kg vs 0.12 kg), and which pad holds the
   heavy one is sampled per episode. No camera pixel reveals the target, so the seed's
   entire strategy (ground the named object visually, ignore the rest) is inapplicable
   by construction. Smoke check 4 executes the seed's plan — put the (luckily correct)
   canister straight on the goal — and it earns no success.
2. **An instrument must be operated and READ.** The scene provides a two-pan balance
   comparator: a free-tilting beam on a D6 pivot with hard stops at ±10°, CoM below the
   pivot so it self-levels when unloaded and parks heavy-side-down under any unequal
   load. The verdict exists only as a physical state (beam tilt) that the solver's later
   actions must be CONDITIONED on — a sense→decide dependency no LIBERO(-plus) task has.
3. **Declared, latched ordering.** `weighed` latches only while BOTH canisters rest on
   OPPOSITE pans with the beam settled tipped past 6°; success requires that latch.
   Weighing must precede placement; a single-load tilt (check 5) or an unweighed lucky
   guess (check 4) never satisfies it. The seed family has no ordering constraint at
   all beyond "pick then place".

Also distinct from the other tasks_v7 packages examined while building it (e.g.
`press_switch_i161` = dial setpoint regulation against flags; packing/pen-holder =
fill-a-container): nothing else in the corpus makes the goal object's identity hidden
state recoverable only through a comparator instrument.

## Scene

- Wooden table (top z = 0.40). At its centre a charcoal pillar carries the **beam**:
  ONE dynamic compound rigid body (bar + two pan discs + 8 rim blocks per pan, authored
  per env) hung on an authored D6 joint that frees exactly rotX within ±10° (physical
  end stops; all other axes locked, low > high). Beam CoM sits 2.5 cm below the pivot
  → gravity restores level unloaded; one canister's weight on either pan (Δτ ≈ 0.4 N·m)
  overwhelms the restoring torque and parks the beam on a stop. Angular damping 3.0
  tames the slam; rims keep a canister riding a pan that flips under it.
- Two **blue canisters** (r 2.4 cm, h 10 cm), identical spawn cfgs except mass
  (`can0` = 0.42 kg heavy, `can1` = 0.12 kg light), standing on two dark pad markers.
- A red kinematic **pedestal** (r 5 cm, h 5 cm) to the side.
- **Randomization (readback-verified in smoke check 2):** which pad holds the heavy
  canister (`torch.rand < 0.5` — the hidden bit), ±2 cm xy jitter per canister,
  ±2.5 cm xy jitter on the pedestal.

## Rubric (score 0..1, latches via 24-calm-substep streaks in `post_step`)

| share | term | meaning |
|------:|------|---------|
| 0.15 | mean(`pan_latch`) | each canister has rested on a pan (latched) |
| 0.25 | `weighed` | both at rest on OPPOSITE pans, beam settled tipped > 6° (latched) |
| 0.20 | `heavy_on_pedestal()` | CURRENT state: heavy upright (<15°), centered (<3 cm), seated (±1.2 cm), still |
| 0.40 | `success()` | `weighed` ∧ heavy on pedestal ∧ light ≥ 9.5 cm from pedestal axis ∧ light slow |

Null policy ≈ 0; score = 1.0 iff `success()`; latched shares never evaporate under
correct behavior (revocation keeps ~0.40).

## Solution outline (solve.py — demonstrated on the forge)

Teleports are TRANSPORT ONLY (release ≥1.2 cm above every seat, outside all latch
z-windows); landings, the beam's verdict and the seating flow through contact dynamics.

1. **Weigh:** drop canister 0 on its near pan (beam slams to that stop), then
   canister 1 on the OTHER pan — released beam-aligned (quat + up-axis offset) so it
   lands flat on the tilted pan; the beam re-settles heavy-side-down and `weighed`
   matures. `SIM_GEN_SCORE` at each phase boundary, asserted non-decreasing.
2. **Verdict:** read `down_side()` (the scene's own physical readout) and pick the
   canister standing on the sunken pan; assert it matches the ground-truth masses.
3. **Place:** teleport the verdict canister to 1.5 cm above the pedestal seat and
   release; it seats under gravity. The light one stays on its pan (clear by
   construction).
4. **Persist:** ≥3.5 simulated seconds hands-off (probe buffers asserted zero), then
   `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge: seeds 0 (heavy at pad 1) and 1 (heavy at pad 0), both
`SIM_GEN_SOLVE: SUCCESS`, score trajectory 0 → 0.075/0.15 → 0.400 → 1.000, rc=0.

## Franka embodiment argument

Base pose: **(−0.44, 0.0, 0.40)** on the table's −x edge (world z = table top), facing
+x. Everything the task touches lies in a 20–75 cm reach annulus in front of the base:
pads (−0.20, ±0.13), pans (0.02, ±0.13, z≈0.52), pedestal (0.06, 0.30, top 0.45).

- **Canisters** (r 2.4 cm, h 10 cm, ≤0.42 kg): a 4.8 cm diameter cylinder is squarely
  inside the Franka gripper's ~8 cm max aperture; 0.42 kg is far under payload. Top-down
  grasp at the barrel's upper half, OSC transport at ~0.5 m height clears the beam
  (pivot z = 0.53, rims top ≈ 0.526).
- **Placing on a pan** (r 5.5 cm, rim gap ≈ 5 cm inner): release the canister 1–2 cm
  above the pan centre, aligned with the beam's current tilt (readable visually from
  the beam); the ±10° stops keep pan tilt small enough that a flat-bottomed cylinder
  seats inside the rims — exactly what the beam-aligned drop in solve.py demonstrates.
- **Reading the scale:** the verdict is a ≥6° beam tilt over a 34 cm bar — a ~3.5 cm
  height difference between pan ends, trivially visible.
- **Re-grasping from a tilted pan:** the rims are 1.8 cm tall vs the canister's 10 cm —
  plenty of exposed barrel for the same top grasp; lifting straight up clears the rims.
- **Pedestal placement** (r 5 cm, tol 3 cm): lower until light contact, release —
  standard place; upright tolerance 15° is generous for a gripper-held cylinder.
- **Order/clearance:** the light canister may simply be LEFT on its pan (≥9.5 cm from
  the pedestal by construction), so the arm handles each object at most twice.

**Declared execution order (required):** (1) place BOTH canisters on opposite pans and
wait for the settled tilt (`weighed` latch); (2) only then move the heavy one to the
pedestal. Violations rejected by smoke checks 4 (unweighed placement) and 5 (single
load).

## Files

- `scene.py` — cfg + `BalanceVerdictScene` (assets, D6 authoring, reset randomization,
  readings, latches, rubric, describe/instruction, get/set_state), registration
  `balance_verdict` / `simgen.balance_verdict`.
- `solve.py` — the demonstrated teleport-transport solution (above).
- `smoke.py` — 11-check rejection battery, records `frames.npz`.

## Smoke battery (11 checks)

1. **settle** — clean reset: finite, canisters physically AT the sampled pads
   (readback), beam level, hidden masses real (`get_masses` readback), score ~0.
2. **random** — 8 seeds, readback: heavy pad takes both values, canister/pedestal xy
   vary, can0's side matches the draw.
3. **null** — 2 s of nothing: score < 0.02, no success.
4. **order violation (the seed's strategy)** — heavy straight onto the pedestal without
   weighing: seats fine, no success, score ~0.20.
5. **single load** — only the light canister on one pan: beam tips > 6° yet `weighed`
   never latches, score < 0.12.
6. **wrong object** — full weigh, then the LIGHT canister on the pedestal: no success,
   latched credit only (~0.40).
7. **near miss** — heavy released 5 cm off the pedestal axis: not seated, no success.
8. **exactness** — full strategy; beam verdict asserted to agree with ground truth;
   success ∧ score == 1.0, stable 1 s hands-off.
9. **latch/revocation** — heavy shoved off the pedestal (scene-owned `drive_f` probe):
   success revoked, latched ~0.40 remains.
10. **clearance** — light stacked on the seated heavy (inside 9.5 cm): success revoked.
11. **frames** — ≥20 rgb frames saved to `frames.npz`.
