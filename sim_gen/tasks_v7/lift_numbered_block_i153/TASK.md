# lift_numbered_block_i153 — `beam_hoist`

Raise the placard-matching numbered block ~10 cm WITHOUT ever lifting it to the goal
yourself: seat it in the amber cradle of a hinged balance beam, then load BOTH other
numbered blocks into the weight pan on the far arm until the pan side out-torques
the cradle side and the beam tips up against its stop, hoisting the seated block.

- **Env name:** `simgen.beam_hoist` (robot `"null"`, `env_spacing=6.0`)
- **Package:** `scene.py` (scene + rubric), `solve.py` (legitimacy certificate),
  `smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `rlbench/lift_numbered_block`
(`RoboVerse/roboverse_pack/tasks/rlbench/lift_numbered_block.py`): three numbered
blocks on a table; the Franka must identify the commanded number, grasp that block,
and lift it — one grasp, one vertical lift, distractors are pure clutter to be
avoided. The arm itself is the lifting mechanism.

## What changed and why it is strategically different

| Axis | Seed | This task |
|---|---|---|
| How the target rises | the gripper lifts it | **a machine lifts it**: the arm may only *place* blocks; the hoist is hinge statics — two counterweights out-torque the beam's built-in cradle-side tare bias |
| Role of distractors | clutter to avoid | **load-bearing**: both non-target blocks are required counterweights; one is provably insufficient (torque margins asserted in `__post_init__`), so "avoid the distractors" becomes "use every distractor" |
| Identification | numbered textures | **placard matching**: a kinematic sign on the rig shows a pip count (1–3, redrawn per episode); the blocks carry matching pip studs; block scatter is permuted + jittered, so identity must be read, not memorized |
| Plan shape | one pick-and-lift | three placements with a **hard ordering constraint on the goal side** (target must be seated for the hoist to score) and a physics threshold crossed only by the *second* pan block |
| Success | trajectory-driven lift height | rubric: beam-frame containment windows for cradle + pan occupancy, beam raised past +10°, all settled, ≥3.5 s persistence |

Distinct from the corpus: `crater_run` (`basketball_in_hoop_i128`) is sustained
push-conveyance of an ungraspable ball; `oven_dials` (`open_oven_i7`) is
dial/knob articulation; `pen_holder` is container filling. None make the
*distractors* the actuator, and none score a lever-statics threshold (N−1
counterweights must demonstrably fail).

## Rubric

- `success()`: target block inside the cradle window (beam-body frame:
  `|x−cradle_cx| ≤ 0.032` per axis, `|z−seat_z| ≤ 0.012`) AND both distractors
  inside the pan window (`|x| ≤ 0.036`, `|y| ≤ 0.058`, z window admits a two-high
  stack) AND beam raised (`θ ≥ +10°`, travel stop at ±14°) AND everything settled
  (blocks lin < 0.05, ang < 1.5; beam ω < 0.25). `__post_init__` asserts window
  honesty (every seated rest incl. tilted edge-rest accepted; wall-perch, bar-rest,
  floor rejected), the torque budget (one counterweight insufficient / two
  sufficient at worst-case in-tray arms), graspability, sweep clearance, and
  scatter-slot separation.
- `score()`: latched stages, 0.15 each — target seated, first pan block, second pan
  block, beam-up — capped at 0.60; 1.0 iff `success()`. Credit never evaporates
  (verified in smoke); null policy scores ~0 (blocks spawn on floor slots).

## Teleport-solution outline (`solve.py`)

Teleports are TRANSPORT ONLY (the stand-in for pick-and-carry of each 45 mm,
jaw-graspable block); every load-bearing event is contact/joint dynamics:

1. **P0** settle 0.5 s; read target identity from the scene (placard/`target_idx`);
   assert score ≤ 0.02 and the empty beam rests cradle-down (θ < −8°).
2. **P1 seat** — carry the matching block to a hover pose over the cradle
   (`beam_to_world`, beam-aligned quat, zero vel) and RELEASE; gravity + contact
   seat it. Score 0.15.
3. **P2 counterweight A** — release distractor 1 into the +y pan slot; assert the
   beam STAYS down (torque-margin proof in vivo). Score 0.30.
4. **P3 counterweight B** — release distractor 2 into the −y pan slot; the hinge
   does the hoisting — the beam tips to its upper stop, raising the cradle and the
   target riding in it. No wrench ever touches the beam; the success state is
   never spawned. Score 1.00.
5. **P4 persistence** — 420 substeps (3.5 s at 120 Hz) hands-off; success holds.

Drop poses are computed from the live beam pose per episode, so one script serves
all seeds. Passes seeds 0, 1, 2 — `SIM_GEN_SCORE` 0.00 → 0.15 → 0.30 → 1.00 → 1.00,
`SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka + parallel jaw, OSC)

Base at env-frame ≈ (0, −0.55), facing +y toward the rig; the scatter row
(y = −0.35) lies directly in front of it and the sign board faces it (−y face).

- **Numbered blocks (the only manipulands):** 45 mm cubes < 80 mm jaw span
  (asserted in cfg) — top-down grasps on the floor slots (reach 0.40–0.43 m).
  Drop-offs: cradle center at radius ≈ 0.62 m, height ≤ 0.30 m even beam-down;
  pan center ≈ 0.62 m on the other arm. Tray walls are 35 mm — the gripper
  releases from a hover above them, exactly as the solve does. Loading order
  within the pan is free; only "target seated" gates the hoist credit.
- **Beam / rig / placards (kinematic base, hinged beam):** never manipulated —
  the beam is the machine, the placard is read (visual), the rig is fixed at the
  env origin. Nothing requires touching them.

## Execution order

1. `scene.py` first (torque budget + window honesty derived from cfg constants and
   asserted in `__post_init__`).
2. `solve.py` iterated on the forge until `SIM_GEN_SOLVE: SUCCESS` on seeds 0/1/2
   (one iteration: the multi-collider beam needed its CoM authored explicitly at
   the body origin — PhysX otherwise derives it from collision geometry and the
   pan tray outweighed the tare bias).
3. `smoke.py` rejection battery (16 checks) on the forge.
4. `TASK.md` + final clean runs.

## Check list (smoke, 16/16)

1. settle/no-NaN + layout sanity (rig at origin, beam cradle-down, blocks on
   slots, target placard mounted, others parked, no tray occupied)
2. score ~0 at reset, no success
3. randomization readback: all three targets occur across 14 seeds, placard
   tracks the target, layout stays sane
4. randomization readback: scatter permutation + jitter really vary
5. null policy: 240 idle steps → score ~0, beam stays down
6. seed strategy (grasp-and-lift the target alone) fails: seated target, beam
   stays down, score exactly the seat stage, no success
7. one counterweight is insufficient: lands in the pan (non-vacuous) but the beam
   stays down; score 0.30, no success
8. wrong-block hoist rejected: distractor seated + target in the pan tips the
   beam (non-vacuous) but seat latch stays 0, score ≤ 0.30, no success
9. all-three-in-pan rejected: beam tips but the cradle is empty — no seat credit
10. bar-rest rejected: target resting on the bare bar outside the cradle fails
    the x window
11. wall-top perch rejected by the z window
12. settle gate: full load mid-tip is not success while the beam still swings
13. latched credit survives removal: pan block teleported away mid-tip → beam
    returns down, all four latches hold, score == 0.60 cap, not success
14. rejection audit: success() never True at any judged probe
15. final no-NaN
16. camera ≥ 20 rgb frames → `frames.npz`
