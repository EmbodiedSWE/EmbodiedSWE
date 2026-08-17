# obstacle_i284 — unlock the bolted vault, deposit the red cube, seal it again

## Seed provenance

Seed task: `pick_place/obstacle` (RoboVerse `roboverse_pack/tasks/pick_place/obstacle.py`)
— a Franka picks a 4 cm cube and carries it OVER a free-standing wall
(0.8 × 0.05 × 0.3 m) along waypoint markers, setting it down in an open goal
region on the far side. The judged outcome is the carried cube's final resting
pose; the obstacle is passive scenery to be flown over.

## Strategic difference

**vs. the seed.** The seed's obstacle is a static wall in an otherwise open
transport problem: the entire strategy is *go over the blocker and set the
payload down in a goal region that is always open*. Here the goal region is the
inside of a VAULT that is sealed from above — a captive lid rides in rails over
the bay, and the lid itself is blocked by a cross-sliding bolt standing in its
track. "Go over" does not exist: the obstacle must be *reconfigured in a fixed
order* (retract the bolt → slide the lid open → drop the payload through the
mouth → slide the lid SHUT again), and the mechanism state is *judged* —
success reads the lid's restored closed pose as well as the payload's settled
containment. The seed's whole plan, executed faithfully (carry the cube over
everything and set it down at the goal xy), deposits the cube ON the closed lid
— explicitly constructed and rejected in smoke (check 5). Two more inversions:
a same-size wrong-color decoy makes the payload a perception choice (the seed
has a single cube), and the finish requires *restoring* the obstacle, which has
no seed analogue.

**vs. the corpus tasks read this session.**
- `obstacle_i17` (same seed; skittle gallery): a RELEASED ballistic strike
  through a pre-existing floor tunnel, judged on toppling an untouchable pin
  plus a perception coin-flip. Here nothing is thrown and nothing rides on
  momentum: every interaction is sustained quasi-static contact (two orthogonal
  slides and a gravity drop), the judged payload is handled directly, and the
  blocking structure must be reconfigured and restored — i17 never touches its
  tunnel.
- `pull_cube_i20` (beam scale): goal is a torque threshold on a body the robot
  never touches, with an episode-dependent stopping rule. Here there is no
  equilibrium threshold and no indirection — the difficulty is the ORDERED
  INTERLOCK (bolt gates lid gates deposit gates re-close), all driven by direct
  pushes/pulls.
- `libero_..._i87` (rocker cabinet): force transmission through a hidden
  pivoting link to an ungraspable judged drawer. Here every mechanism part is
  graspable and directly driven; the challenge is sequence, not transmission.
- `pen_holder` (robobench exemplar): open cup, multi-object insertion. Here a
  single deposit is gated by a locked, roofed aperture, plus a decoy and a
  mechanism-restore end state.

## Scene (`vault_deposit`, env `simgen.vault_deposit`, robot="null")

Procedural geometry only — three compound custom spawners + two `CuboidCfg`
cubes; NO USD joints (both slides are contact-geometry captives, so the whole
scene teleport-randomizes cleanly at reset):

- **Housing** (kinematic): open-top bay (16 × 16 cm cavity, 13 cm walls, rim at
  z = 0.140) with a grippy floor and slick walls; a rail track running +x from
  the bay over an open shelf — support rails + guide fences + cap strips make
  the lid captive (slides in x only; cannot lift, yaw out, or leave), with
  closed/open end stops; a raised bolt channel crossing the track's +y side
  (pedestal, x-walls, keeper lips, y end stops — the lock-side stop is kept
  below the rail plane because it crosses the lid corridor). The +y
  rail/fence/cap runs are split around x ∈ [0.098, 0.134] so the blade and
  knob pass through the rail plane.
- **Lid** (dynamic, 0.35 kg): 200 × 205 × 12 mm slab + 30 mm handle knob,
  slick. Closed at x_local ≈ 0 (sampled), fully open needs x_local ≥ 0.190.
- **Bolt** (dynamic, 0.12 kg): base slab with bottom FLANGES captive under the
  channel's keeper lips (the bolt slides ~9 cm in y but cannot be tipped,
  folded, or lifted out — the interlock is force-proof: a hard press on the
  blade tips it ≤ 3 mm before the flanges catch), a blade rising through the
  rail gap into the lid's corridor (blade z [0.124, 0.172] covers the lid slab
  z-span), and a graspable knob (⌀22 × 56 mm) raised clear of the lips.
- **Red cube** (payload) and **blue cube** (decoy): both 45 mm / 60 g, on
  jittered polar slots on the bay's −x side, slot assignment randomly SWAPPED.

Geometry contract (all asserted in `__post_init__`): locked blade fully inside
the lid's swept corridor and covering its z-span; retracted blade clears the
worst-case (guide-play-shifted) lid edge with margin, and the channel allows
retraction beyond that; blade+knob fit the rail gap; open lid clears the whole
mouth and the open stop allows it; any lid inside the closed window fully
covers the bay; containment window honest by construction (accepts every
physical in-bay rest, rejects wall-outside and rim/lid rests via the z window);
flange/lip cage overlaps; everything the arm moves fits the Franka jaw.

Per-episode randomization (readback-verified): housing xy ± 4 cm and yaw =
{0 | 180°} flip + ±25° jitter (the whole track/bolt side swaps across the
scene); bolt lock position ±4 mm; lid closed position ±6 mm; cube slots
(radius 0.30 ± 0.03 m, angles 150°/210° ± 12°, random swap, free cube yaw).

**Success** (live state): red cube inside the bay window (|x|,|y| ≤ 0.065,
z ∈ [0.018, 0.105] housing-local) ∧ cube settled ∧ lid settled inside its
closed window (|x_local| ≤ 0.020, seated on the rails) ∧ decoy NOT in the bay.
**Score**: latched stage credit — 0.10·(bolt retraction progress, latched max)
+ 0.15·(lid-open progress, latched max) + 0.30·(red cube has been in the bay,
latched), capped 0.55; exactly 1.0 iff success. Latches reset on `reset()`;
null policy scores ~0.

## Solution (`solve.py`) — one transport teleport, everything else applied force

1. **P0 settle + perception**: 120 steps; read back housing pose/yaw, sampled
   bolt lock and lid closed positions, and both cube slots from the live state
   (never hard-coded); assert locked/closed/score ≈ 0.
2. **P1 unlock**: force-limited velocity servo (kv 5, clamp 2 N — a fingertip
   hooked on the brass knob) pulls the bolt along housing +y until the blade
   clears the lid corridor (travel ≥ 0.082 m; it coasts onto the far stop).
   Score latches ≥ 0.10.
3. **P2 open**: the same servo (kv 20, clamp 6 N, on the lid) slides the lid
   along its rails to x_local ≈ 0.196 — the rails/fences/caps carry it the
   whole way. Score ≥ 0.25.
4. **P3 deposit**: the red cube is TELEPORTED (transport only) to free air
   0.22 m above the bay center and RELEASED — it free-falls through the open
   mouth, lands on the grippy floor through real contact, settles. Score
   ≥ 0.55.
5. **P4 re-close**: servo drives the lid back (−x) until x_local ≤ 0.006; it
   coasts onto the closed stop. Forces off.
6. **Hands-off**: success() turns True on the settled live state (score 1.0),
   then holds through a **3.3 s hands-off persistence** window before
   `SIM_GEN_SOLVE: SUCCESS`.

The full episode is repeated on a second seed (fresh reset; no score prints, so
the announced SIM_GEN_SCORE stream stays non-decreasing) to certify seed
robustness. Forge result: seeds 0 and 1 both OK, rc=0 (~22 s).

## Franka embodiment argument

Base pose: on the floor ~0.50–0.55 m from the bay center, on the side the
*read-back* track direction dictates (the 0/180° flip means the robot must look
first) — e.g. housing-local (+0.15, −0.55) faces the track, the channel, and
both cube slots. Every interaction point lies within 0.40 m of the bay center
at heights 0.02–0.20 m, inside the Franka's ~0.85 m reach envelope.

- **Bolt**: pinch or fingertip-hook the ⌀22 × 56 mm knob (jaw 80 mm), pull
  ~9 cm horizontally along the channel. Required force ≈ 0.2 N (0.12 kg on a
  μ≈0.055 pair) — the solve's 2 N-clamped servo is a generous proxy for a
  force-limited fingertip.
- **Lid**: pinch the 30 mm handle block (or press against it) and slide 19 cm
  along the rails at ≈ 0.3 N sliding friction (6 N clamp in the solve); the
  captive rails remove any need for precise vertical control.
- **Red cube**: 45 mm / 60 g — canonical Franka top grasp from the open floor;
  carry over the rim and release above the 16 cm-square exposed mouth (>5 cm
  clearance per side); gravity finishes the deposit, as in the solve.
- **Decoy**: same grasp affordance, must simply be left alone — a perception
  requirement, not a dexterity one.
- **Re-close**: the reverse of the open slide; the closed stop provides a hard
  detent so millimeter precision is unnecessary (±20 mm window).

## Execution order declaration

Scene was designed first with the minimal success predicate; `solve.py` was
iterated on the forge until the task was *physically* solved on two seeds; only
then were the rubric weights anchored and `smoke.py` written. No check was ever
weakened to make a run pass. Forge iterations fixed *geometry* bugs, not the
rubric: (1) a float-boundary cfg assert (open-need margin exactly at its
bound), (2) the lock-side channel stop originally crossed the lid corridor
above the rail plane and wedged the lid shut for good, (3) the keeper cage
(flanges + lips) was added after a torque audit showed a 6 N press on the blade
could fold the un-caged bolt over its 0.13 m channel wall — the interlock is
now force-proof by geometry, so the *only* unlock is the intended retraction.
The task order (bolt → lid → deposit → re-close) is PHYSICALLY enforced: the
locked lid jams on the blade after ~1 cm (smoke 6, non-vacuous force probe),
no cube fits the bay while the lid covers it, and the deposit must precede the
re-close because the closed lid roofs the mouth.

## Smoke battery (`smoke.py`) — 15 checks, all rejection/health

1. **settle/no-NaN** — bolt at its sampled lock, lid in its sampled closed
   window, cubes outside, score ≈ 0.
2. **randomization A** — housing xy jitters AND both yaw flip clusters (track
   pointing either way) appear across 10 seeded resets (readback).
3. **randomization B** — bolt lock and lid closed positions jitter; the
   red/blue slot assignment swaps across resets.
4. **null policy** — 240 idle steps: locked, closed, nothing deposited, ≈ 0.
5. **SEED strategy rejected** — the seed's carry-over-and-set-down end state
   (red cube settled ON the closed lid, over the bay): not in the bay (z
   window), no deposit latch, score ≈ 0.
6. **locked interlock (non-vacuous force probe)** — the solve's own 6 N lid
   servo pushes the closed lid while locked: the lid MOVES ≥ 2 mm (force
   pathway live) but jams ≤ 35 mm on the blade; the bolt stays locked; ≈ 0.
7. **wrong object** — the full correct sequence executed with the DECOY (it
   demonstrably lands and settles in the bay — the containment window accepts
   real deposits — and the lid re-closes): no deposit latch, no success,
   credit ≤ 0.25.
8. **both cubes in** — red settled beside the decoy, lid closed: every other
   gate holds and the decoy gate ALONE refuses success (≤ 0.55).
9. **unsecured deposit** — red settled in the bay, lid left OPEN: ≤ 0.55.
10. **lid ajar** — same deposit, lid at x = +0.035 (outside the ±20 mm closed
    window, mouth mostly covered): no success.
11. **latched credit** — bolt REALLY retracted by force (w_bolt latches), then
    teleport-stolen back to its lock: latch survives, success does not.
12. **settle gate** — full success layout with the cube written MOVING: judged
    without stepping, refused; layout destroyed before it can settle.
13. **rejection audit** — success() observed False at every step of the battery.
14. **final no-NaN.**
15. **video** — frames.npz (278 rgb frames, 960 × 600) written to CWD.

Forge result: `SIM_GEN_SMOKE: ALL PASS 15/15` (rc=0, ~62 s).
