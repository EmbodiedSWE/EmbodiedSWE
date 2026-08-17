# plug_charger_i337 — gauge-mismatch adapter chain

Scene `gauge_adapter`, env id `simgen.gauge_adapter`.

## Seed provenance

Seed: `maniskill/plug_charger`
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/plug_charger.py`).
The seed is a terminal DIRECT-insertion alignment task: a free two-prong charger
lies on the table, a fixed wall receptacle with two (slightly enlarged) holes is
always compatible with it, and the goal is one pose — prongs aligned with the
holes, charger pushed in.

## What changed and why it is strategically different

The charger **does not fit the wall outlet — and never can**. The outlet is wide
gauge (two 20 mm slots at ±22 mm with a solid center divider), the charger is
narrow gauge (10 mm prongs at ±12 mm): each prong overlaps the divider by 5 mm and
the 24 mm prong pitch can never match the 44 mm slot pitch at ANY planar pose
(asserted in `__post_init__`, force-proven in smoke). The task is **mediated
mating through a gauge converter** with a decoy forcing gauge diagnosis:

| | seed | this task |
|---|---|---|
| plug↔socket compatibility | always compatible | **geometrically excluded**: direct wall mating is impossible at every pose |
| goal | one inserted pose | an assembled **two-interface chain**: green adapter seated flush in the outlet (horizontal slide-to-flush) AND charger plugged prongs-down into the adapter's top sockets (vertical drop-press) |
| object selection | none (one plug, one socket) | **two visually-twin adapters**: green (16 mm sockets, admits the prongs) vs orange decoy (8 mm sockets, refuses them) — the correct converter must be selected |
| mating directions | one horizontal push | two **heterogeneous** mates in different directions and frames: fingers→slots along the dock axis, prongs→bores along the adapter's up axis |
| judged frames | charger pose in base frame | two relative-frame predicates composed: finger tips in the DOCK frame + prong tips in the ADAPTER frame |
| naive strategy | align and push | the seed's whole strategy scores ~0 here (smoke presses the charger at the outlet with 4 N for 2 s — the tips never pass the panel face) |

No execution order is required (unlike order-gated siblings): adapter-first is the
natural route, but a pre-coupled pair slid in together is equally valid — the
defining constraint is the impossible direct mate, so a solver needs a different
PLAN (diagnose the gauge mismatch, pick the compatible converter, assemble the
chain), not a mechanism-operation sequence.

Also different from every other task read this session:
- `plug_charger_i230` (shutter garage): access-gating + airlock cycle around ONE
  compatible insertion; here access is never blocked — compatibility itself is the
  obstacle, there is no mechanism to operate, and the goal adds a second, upward
  mating interface.
- `plug_charger_in_power_supply_i21` (keyhole unplug): unlock–EXTRACT–stow of a
  captive plug; here both bodies start free and two insertions are ADDED.
- `peg_insertion_side_i2` (hasp bar + pin ordered fastening): geometry forces an
  order of two mates on one axis chain; here no order is forced and the two mates
  live on different axes of different bodies.
- `turn_key_i251` (key quarter-turn): tool-torque transmission; no rotation here.
- `lift_hook_i148` (hook escape), `tote pour i293`, `beam balance i268`: extraction /
  pouring / counting strategies, none of which involve building a mating chain.

## Scene

Fully procedural (axis-aligned `UsdGeom.Cube` parts, explicit 1 mm contact
offsets — the ~2 cm default would eat the 2–3 mm mating clearances). Dock frame:
origin at the panel front-face centre at floor level, +y into the panel.

- **station** — KINEMATIC compound: 240×160 mm panel (40 mm thick) with the twin
  wide-gauge slots (20 w × 22 h mm at ±22 mm, 30 mm deep, back-stopped, center
  divider) just above a shelf whose top (60 mm) is flush with the slot floors.
- **adapter (green)** — DYNAMIC 64×50×46 mm block, 0.35 kg: two brass fingers
  (14×26×18 mm at ±22 mm, lifted 2 mm off the body bottom so they ride centred in
  the slot band instead of skimming the coplanar slot-floor edge) on the front
  face match the slots (3 mm/side clearance);
  two 16 mm square bores (20 mm deep, ±12 mm, 20 mm funnel mouths) open in the top
  deck and match the prongs (3 mm/side).
- **decoy (orange)** — same body/fingers, but 8 mm bores / 10 mm mouths: seatable,
  yet its sockets refuse the 10 mm prongs.
- **charger** — DYNAMIC white 50×40×32 mm brick, 0.25 kg, two brass prongs
  (10 mm sq, 18 mm proud, ±12 mm) on one end face.

Flush stops are structural: fingers (26 mm) shorter than slots (30 mm) → the seat
stop is body-face-on-panel; prongs (18 mm) shorter than bores (20 mm) → the mate
stop is body-on-deck. Both margins asserted in `__post_init__`.

Randomization (verified by readback in smoke): station xy ±40 mm + yaw ±25°; the
two adapters SWAP sides at random with xy jitter + free yaw; charger spawn xy +
free yaw. All predicates are relative-frame, so nothing is world-anchored.

`success()` iff, with adapter AND decoy AND charger settled (< 0.05 m/s): adapter
SEATED (finger-tip dock-y > 21 mm, centred, riding the shelf, aligned ±10°,
upright) AND charger MATED (both prong tips > 12 mm below the deck in the
adapter frame, inside the bore footprints with opposite x signs, prong axis within
15° of down-the-bores). `score()` = 0.35·latched gated seating progress +
0.30·latched gated mating progress (cap 0.65); exactly 1.0 iff success. Null ~0;
seed strategy ~0; seated decoy 0 (credit tracks the green adapter).

## Solution outline (solve.py, the legitimacy certificate)

Transport-only teleportation: TWO pose writes, each moving a free body across free
space to a staging pose (never into a mated state): T1 floor→shelf, fingers 10 mm
short of the panel; T2 floor→hover 8 mm above the seated adapter's deck, tips
above the funnel mouths. Everything load-bearing is contact dynamics via
`set_external_force_and_torque`:

- **P1 seat slide** — velocity-regulated push (≤ 6 N, friction feedforward with
  stall escalation) along dock +y: the adapter slides the real shelf, fingers
  thread the real slots (lateral PD centring, yaw/upright steadying torque ≤
  0.12 N·m), until the **panel face physically stops the body** (stall inside the
  seat band under forward press = flush); release, settle → seated, score 0.35.
- **P2 plug press** — gravity-feedforward vertical regulation (net press ≤ ~2.5 N
  beyond weight) with adapter-tracking lateral PD and full orientation PD holding
  prongs-down: the prongs thread the funnel mouths and bores until the **deck
  physically stops the body**; release, settle → mated + seated = success, 1.0.
- **P3 persistence** — ≥ 3.4 simulated seconds hands-off; SUCCESS only if
  `success()` still holds and the phase-boundary `SIM_GEN_SCORE` prints never
  decreased.

## Embodiment argument (single Franka + parallel jaw, recorded not simulated)

- Base pose ≈ (0.0, −0.45) facing the station; everything (station ±4 cm around
  (0, 0.10), spawns at dock-y −0.17…−0.36) is within ~0.65 m reach, work heights
  0–150 mm.
- Adapter: 64 mm width or 50 mm depth pinch — inside the 80 mm jaw stroke; grasp
  the upper half so the fingers clear the shelf on set-down; the seat slide is a
  1–6 N straight push with the wrist supplying the ≤ 0.12 N·m steadying moment a
  rigid grasp gives for free.
- Charger: 32 mm-tall body side pinch; the plug press is a vertical guided press —
  the 20 mm funnel mouths give ±5 mm capture on a 10 mm prong before the 3 mm/side
  bores take over, so hand precision ~5 mm suffices; press force ≤ weight + 2.5 N.
- Selection: green vs orange is a color/socket-width discrimination, no extra
  dexterity. All forces are far inside Franka limits; the mm-scale rubric bands
  are enforced by the stops (panel face, deck), not by the hand.

## Checks (smoke.py — recorded to frames.npz)

1. settle/no-NaN: all three loose objects read back OUTSIDE, adapters on opposite
   sides, everything settled.
2. SEED strategy CONTACT probe: 4 N orientation-held push of the charger at the
   outlet for 2 s — non-vacuous (advances > 8 mm) yet the prong tips never pass
   the panel face, no credit, score ≤ 0.02.
3. randomization readback: station xy + yaw vary over 8 seeds.
4. randomization readback: green adapter's side FLIPS, spawns + relative yaw vary,
   adapters opposite every reset.
5. null policy 240 steps → score ≤ 0.02, no success.
6. decoy dead end CONTACT probe: seated decoy + 2.5 N laterally-tracked held press
   of the charger on its deck for 2 s — a prong really bears down AT a socket
   mouth (non-vacuous) but no tip inside a socket footprint ever passes the
   funnel-mouth ledge (> −8 mm, far short of the 12 mm mate depth), no credit.
7. seated-only → 0.33 ≤ score ≤ 0.37, no success.
8. removal latch: pulling the seated adapter back out leaves credit unchanged.
9. floor mate: charger constructed in the sockets with the pair loose on the
   floor → mated() true but 0.27 ≤ score ≤ 0.33, no success.
10. near-miss seat: fingers 11 mm short → partial gated credit only, not seated.
11. deck percher: charger dropped prongs-down yawed 90° onto the seated adapter's
    deck → tips land on SOLID deck, mate latch stays 0.
12. wrong station spot: adapter pushed at the panel beside the slots (4 N, 2 s) —
    non-vacuous, tips never pass the face, seat latch stays 0.
13. seat progress is monotone in depth (two-depth comparison).
14. rejection audit: success() never True anywhere in the battery.
15. final no-NaN.

## Verification (forge server)

- `solve --seed 0`: SIM_GEN_SCORE 0.0000 → 0.0000 → 0.3500 → 0.3500 → 1.0000 →
  1.0000, flush seat at tip_y +0.0252 (stall-escalation 1.0→1.8 N), mate stop at
  tipz +0.0050, `SIM_GEN_SOLVE: SUCCESS` (rc=0, 23 s).
- `solve --seed 1`: same trace (flush tip_y +0.0252, mate tipz +0.0050),
  `SIM_GEN_SOLVE: SUCCESS` (rc=0, 25 s).
- `smoke`: `SIM_GEN_SMOKE: ALL PASS 15/15` (rc=0, 36 s), frames.npz saved; decoy
  probe's deepest in-footprint tip excursion +0.3 mm (tips ride the mouth ledge).
