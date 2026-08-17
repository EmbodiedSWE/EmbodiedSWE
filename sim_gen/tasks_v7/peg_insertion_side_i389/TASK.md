# peg_insertion_side_i389 — `line_bore`

Select the silver wide-channel COUPLER (not its copper narrow-channel look-alike),
seat it into the keyed pocket between two window-pierced pylons — only then do the
three square bores line up into one continuous passage — and thread the headed shaft
through window → coupler channel → window until its red head sits flush against the
entry pylon, everything at rest.

- **Env name:** `simgen.line_bore` (robot `"null"`, `env_spacing=4.0`)
- **Package:** `scene.py` (scene + rubric), `solve.py` (legitimacy certificate),
  `smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `maniskill/peg_insertion_side`
(`RoboVerse/roboverse_pack/tasks/maniskill/peg_insertion_side.py`): a Franka picks a
peg off the ground and pushes it sideways into a hole in one fixed box; a
`DetectedChecker` with a `RelativeBboxDetector` at the hole judges the insertion. One
object, one grasp, one alignment, one insertion — **the hole exists from the start**.

## What changed and why it is strategically different

| Axis | Seed | This task |
|---|---|---|
| The hole | pre-exists in a fixed block; the task IS the insertion | **the through-passage does not exist until built**: an empty pocket between two pierced pylons must first be filled by seating the coupler; only then do window/channel/window become one bore |
| Objects & plan shape | one peg, one fixed block; single insert | **three-body ordered assembly**: choose between two look-alike blocks (16 mm decoy channel refuses the 20 mm shank — cfg-asserted), place-and-key the coupler, then a chained three-gate threading with a head-flush depth condition |
| Selection | none — the peg is the only candidate | **perception-forced choice**: which start slot holds the coupler vs the decoy is a per-episode coin flip; identity must be read from color/channel size, not position |
| Ordering | trivially single-step | **topologically forced order**: a shaft threaded first lies across the pocket at bore height; the closed-profile coupler can then only rest ON it, ~67 mm above the seat band (cfg-asserted) — seat-first is the only physical route |
| Judged condition | bbox containment at the hole | conjunction over the joint state: coupler seated (keyed bands + upright) ∧ shaft aligned/in-band/**spanning both outer faces**/**head flush** ∧ shaft axis **encircled by the seated coupler's channel** ∧ both settled |

Distinct from the corpus: `basketball_in_hoop_i128` (push-only ball conveyance up a
switchback), the shunt-yard/roofed-dock routing tasks (single cargo through static
geometry), and the gauge-adapter chain (mating fingers, no decoy selection). None
combine decoy selection + build-the-passage seating + chained multi-gate threading
to a flush stop.

## Rubric

- `success()`: coupler seated (canonical `|x|≤0.009, |y|≤0.008, |z−0.075|≤0.010`,
  up ≥ 0.98, channel axis ‖ bore ≥ 0.98) **and** shaft installed (axis ≥ 0.97,
  `|y|≤0.012`, `z∈[0.062,0.088]`, tip past the far outer face by ≥ 5 mm, head inner
  face within 12 mm of the entry face) **and** the shaft axis passes through the
  seated coupler's channel (≤ 12 mm offset at the coupler's station) **and** shaft +
  coupler settled (lin < 0.08, ang < 0.80).
- `score()`: latched — 0.30 coupler ever seated; +0.20 shaft ever engaged in a
  window **while seated**; +0.30 × max flush-approach fraction while seated+engaged;
  1.0 iff `success()`. Credit never evaporates (verified in smoke); null policy ~0.
- `__post_init__` asserts every honesty premise: windows/channel admit the shaft,
  decoy refuses it, the head passes no gate, the seated coupler covers the whole
  window projection, a crosswise block cannot enter the pocket, the bore z-band
  accepts the sill rest, encircle_tol covers every physical in-channel offset, the
  ORDER FORCER (coupler dropped onto a pre-spanned shaft rests far above the seat
  band), and jaw-fit for the embodiment.

## Teleport-solution outline (`solve.py`)

Teleports are transport only; every load-bearing interaction is contact dynamics:

1. **P1 seat** — the coupler is teleported to a hover 15 mm above the seat and
   DROPPED; gravity + rail/pylon contacts key it into the pocket (up to 4 retries;
   passes first drop on seeds 0/1/2). Assert seated, score ≥ 0.30.
2. **P2 thread** — the shaft is teleported bore-aligned with its tip 6 mm OUTSIDE
   the near window (no gate pre-entered), then a held-carry PD wrench (kp 60 with
   stall-escalation to 240, kd 4, cap 8 N, gravity feedforward, axis-alignment
   torque kr 0.4/kdw 0.01, lag-clamped 30 mm/s carrot) drags it through all three
   gates to `remaining ≤ 5 mm`. Gains respect the one-substep wrench delay
   (kd·dt/m = 0.22 ≪ 1). Assert spanning + flush, score ≥ 0.70.
3. **P3 release** — wrench cleared, 150 settle steps, success() must be True.
4. **P4 persistence** — 3.5 s hands-off, success must hold every step.

`SIM_GEN_SCORE` 0.00 → 0.30 → 1.00 → 1.00 → 1.00, `SIM_GEN_SOLVE: SUCCESS` on
seeds 0, 1, 2 (~20 s each on the forge).

## Embodiment argument (single Franka + parallel jaw, OSC)

Base at fixture-canonical ≈ (0.0, −0.35), facing the fixture; every manipulation
point is within ~0.60 m reach at heights 0.02–0.19 m.

- **Coupler (60 × 50 × 110 mm, 0.25 kg):** side grasp across the 60 mm width (jaw
  80 mm, cfg-asserted margin), lift over the 30 mm rails, lower into the pocket —
  the drop-funnel in solve is exactly this place action.
- **Decoy:** same geometry — graspable, but the task never requires touching it
  (and seating it dead-ends the episode until removed).
- **Shaft (Ø20 mm shank, Ø44 mm head, 0.15 kg):** grasp the shank near the head,
  present the tip to the near window, push axially; regrasp/push on the head disc
  for the final flush press. The held-carry PD wrench is the stand-in for this
  grip. The window gates leave 8 mm radial slack — within OSC tracking.
- **Fixture (kinematic):** never manipulated; it is the jig.

## Execution order

1. `scene.py` written first (geometry + honesty asserts derived from cfg
   constants).
2. `solve.py` on the forge — `SIM_GEN_SOLVE: SUCCESS` on seeds 0/1/2 (first
   submission).
3. `smoke.py` rejection battery — 17/17 after fixing a probe-side NaN (the decoy
   press probe now starts servoing at the hover instant and stops grinding once the
   stall is confirmed).
4. `TASK.md` + final clean runs.

## Check list (smoke, 17/17)

1. settle/no-NaN + layout sanity (fixture at workspace, blocks upright at slots,
   shaft lying at its slot)
2. score ~0 at reset, no success
3. randomization readback (8 seeds): BOTH slot assignments occur, fixture yaw/xy
   jitter real
4. randomization readback: object slot jitter real, blocks track the fixture
5. null policy: 240 idle steps → score ~0
6. seed strategy (partial): shaft tipped into the near window only, no coupler →
   score ~0
7. FLAGSHIP seed strategy (full): shaft threaded through BOTH windows to a
   spanning, flush sill rest — `installed()` True — with the coupler at its slot →
   seated/encircle reject, score ~0
8. decoy refusal (force probe): decoy seated in the pocket, regulated velocity-servo
   press drives the shaft ≥ 8 mm into the window then stalls at the decoy face,
   never spanning, score ~0
9. seat near-miss: coupler against the pylon outer face → seated False, score ~0
10. seat-only credit: coupler keyed into the pocket → score == 0.30 band, no success
11. depth near-miss: threaded but stopped ~30 mm short — spanning yet not flush →
    no success, score ≈ 0.71
12. latched credit survives teleporting the shaft away
13. roof route: shaft across the pylon tops with coupler seated → z-band reject,
    score stays 0.30
14. settle gate: assembled geometry moving axially is not success at the judged
    instant; removed before settling
15. rejection audit: success() never True at any judged point
16. final no-NaN
17. camera ≥ 20 rgb frames → `frames.npz`
