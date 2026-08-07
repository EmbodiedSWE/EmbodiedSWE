# close_microwave_i5 — Bayonet-lock the canister

**Scene:** `simgen.bayonet_canister` (`BayonetCanisterScene`, registered as `bayonet_canister`)
**Seed task:** `rlbench/close_microwave`
**Files:** `scene.py`, `solve.py`, `smoke.py`, `TASK.md` (this file). Procedural geometry only — no external assets.

## Seed provenance and what changed

The RLBench seed is *close_microwave*: a microwave with a hinged door stands on a table and the
robot pushes the door shut — a single unconstrained push on a 1-DoF revolute panel, success = door
angle near zero. This task keeps the seed's **abstract goal** — *close an open container by moving
its closure into the sealed configuration* — and replaces everything about *how* closure is
achieved:

| Aspect | Seed (`close_microwave`) | This task |
| --- | --- | --- |
| Closure mechanism | hinged door already attached (1 revolute DoF) | free-body lid that must be fetched, aligned, dropped through a keyed flange, then twisted |
| Motion class | single planar push | transport → rotational key alignment → vertical insertion → press-and-twist to a hard stop |
| Kinematic constraint | hinge does all the guiding | no joint at all: the bayonet capture is produced purely by contact between lug feet and flange |
| Distractors | none | RED four-winged decoy lid that geometrically cannot pass the three-notch flange |
| Success test | door angle threshold | lid seated on the rim (xy/z/upright tolerances) AND twisted into ψ ∈ [40°, 58°] AND at rest — judged live on physical poses |
| Assets | RLBench microwave USD + trajectory file | fully procedural compound rigid bodies |

## Strategic difference (vs seed and vs the corpus)

- **Vs the seed:** the seed's entire strategy is "push the panel about its hinge until it latches".
  Here there is no hinge and no panel — nothing in the scene can be solved by a push. The core
  skill is a *rotational key-fit closure*: align a 3-fold winged lid with 3 entry notches (~±10°
  window), insert vertically, then twist ~48° under press contact until the lug feet slide beneath
  the flange and hit under-flange lock stops. The seed's solution strategy is inexpressible in this
  scene, and this scene's strategy (grasp–align–insert–twist) is inexpressible in the seed.
- **Vs sibling `close_microwave_i4` (same seed):** i4 is a *mechanism-release* task — remove
  contents, pull a prop post, and gravity swings a drop-gate shut (subtractive, forced ordering,
  articulated gate). i5 is *additive locking* — nothing is removed, no prop, no articulation, and
  the closure element is carried in and locked by twist. No shared mechanism, ordering structure,
  or motion class.
- **Vs the rest of the corpus read this session:** bowl_decant (pouring + inverted park),
  peg_insertion_side_i1 (ramrod ball eject), pick_single_egad_i3 (gate-pin tunnel slide),
  setup_checkers_i2 (silo drop ordering) — none involve rotational locking / threading / bayonet
  capture. The twist-to-lock-under-a-flange interaction is new to the corpus.
- **Key-fit selectivity is provable, not tuned:** the decoy's four lugs sit at 90° spacing, the
  notches at 120°. gcd(90°,120°)=30°, and at no global yaw do all four lugs coincide with notch
  arcs (each notch subtends ~30°, lugs would need simultaneous alignment at 0/90/180/270 vs
  0/120/240) — so the red lid *perches* on the flange at every yaw. smoke check 8 verifies this
  empirically.

## The scene

A kinematic grey canister (mouth ~104 mm, rim at z = 0.120) with a bayonet flange: a flat ring at
z ≈ 0.110 interrupted by three 30°-wide entry notches at 120° spacing. Under the flange, three CW
blocker posts and three lock-stop posts bound the lid twist to ψ ∈ (≈−5°, ≈+50°). Two free lids
rest on their lug feet on the ground, one per side of the canister (side assignment randomized):
the BLUE lid (3 wings, matches the band color and the notch count) and the RED decoy (4 wings).
Each lid has an L-lug under each wing tip and a T-bar handle on top.

**Randomization (seed-driven, verified by readback in smoke check 2/3):** canister xy jitter
(±4 cm) and full yaw (±180°, which rotates the notch pattern), lid slot swap (blue left/right),
per-lid xy jitter (±3 cm). The agent must read the notch orientation from the scene every episode.

**Rubric (latched partial credit, updated in `post_step`):**
0.15 lid ever over the mouth · +0.30 lid seated (feet through the notches, plate on the rim) ·
+0.30 twist latched past ψ=20° while seated → capped at 0.75 unless `success()`;
`success()` = `lid_locked()` (seated ∧ ψ ∈ [40°, 58°]) ∧ settled ∧ finite, judged live. Score 1.0
only on success. Because z_tol = 6 mm and a not-through-the-notches lid rests at z_perch = 0.144
vs z_seat = 0.125, every perched/misaligned state fails `lid_seated` by construction.

## Solution outline (`solve.py`)

Teleports carry the lid through free space only; every load-bearing interaction is contact.

- **P0** settle 1.5 s, layout readback (canister yaw / lid slot), assert baseline score ≈ 0.
- **P1 transport + gravity seating:** one root-state write puts the blue lid 30 mm above its seat
  on the canister axis, wings at entry alignment (ψ₀ ≈ +2°, computed from the *randomized* canister
  yaw readback). At hover height the lug feet are above the flange — free space, no rubric clause
  satisfied. The lid then **falls**: feet drop through the notches, plate lands on the rim; a 3 N
  downward press (fingertip stand-in) confirms seating. Misses are detected by height readback and
  retried with a fresh hover (transport only).
- **P2 twist under contact:** with 1 N hold-down, a +z torque (0.06 N·m, ramping on stall,
  angular-velocity-capped at 2 rad/s) rotates the seated lid CCW; the lug feet slide under the
  flange until the lock stops at ψ ≈ +50°. The lid is never teleported after entry.
- **P3 persistence:** all wrenches cleared, ≥3.3 simulated seconds hands-off, success must hold.

Prints `SIM_GEN_SCORE` at each phase boundary (non-decreasing: 0 → 0.45 → 1.0 → 1.0) and
`SIM_GEN_SOLVE: SUCCESS` only if success persists. Verified on the forge on seeds 0 and 1.

## Embodiment argument (Franka, table-top base)

- **Base pose:** robot base at the origin; canister at ~(0.45, 0) m, lid slots at (0.12, ±0.30) m —
  all inside a Franka's ~0.85 m reach envelope, objects 0–0.15 m high.
- **Blue lid (the only object touched):** grasped by its **T-bar handle** — a 16 mm square bar
  ~35 mm above the plate, sized for the Franka parallel gripper (80 mm max opening). Carry is a
  free-space transport. Entry alignment (~±10° window) is a wrist-roll setpoint. Seating is a
  straight-down guarded move; the 3 N press is a normal-force press well inside Franka limits.
  The ~48° twist is a single wrist-roll (joint 7 range ±166°) while pressing — no regrasp needed;
  required torque ≲ 0.5 N·m at the axis, trivial for the wrist.
- **Canister:** kinematic; never manipulated.
- **Red decoy:** never touched.

## Ordering declaration

No inter-object ordering: exactly one object is manipulated. The seat-before-twist sequence is
*intrinsic to the mechanism* (the lugs cannot pass under the flange without first entering through
the notches), not an imposed rubric ordering.

## Checks (`smoke.py` — 11 checks)

1. Settle/no-NaN baseline; blue lid starts >0.20 m from the canister; score ≈ 0.
2. Randomization readback across seeds (canister yaw / canister xy / lid xy all move).
3. Lid slot swap occurs in both directions over 10 resets.
4. Null policy 2 s → score ≤ 0.05.
5. **Seed-strategy / wrong-outcome rejection:** lid seated but untwisted → no success, score ≤ 0.75,
   and a 3·m·g pull removes it (untwisted lid is not captive).
6. **Lock capture:** lid at ψ=48° → locked; the same pull raises it <8 mm; still locked after release.
7. Near-miss: seated at ψ=30° (past rot-latch, short of lock) → no success.
8. Wrong lid: red decoy dropped aligned → perches above the seat, no success.
9. Wrong place: blue lid twisted to ψ≈48° *on the ground* → no success.
10. Perch: blue lid dropped misaligned (ψ=60°) → rests above the seat, not seated.
11. Video frames saved (`frames.npz`).
