# handover_i339 — `escrow_clamp`

Hand the green spool over to a MECHANICAL RECEIVING HAND in escrow order: stand it
on the escrow ledge between the receiver's open jaw prongs, close the jaw around its
waist, THEN withdraw the ledge — the spool drops 9 mm, its top flange lands on the
prong tops, and the receiver alone bears it, suspended over the reject basin.

## Seed provenance

- **Seed task:** `mujoco_playground/handover`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/mujoco_playground/handover.py`): two ALOHA
  arms; one gripper picks a 4 cm box, hands it to the other gripper at a fixed
  handover point, which carries it to a floating target sphere. The receiver is
  always ready; "letting go at the handover point" is the whole transfer; success is
  a pose match reached by carrying.

## What changed, and why it is strategically different

| | seed | this task |
|---|---|---|
| the receiver | a second gripper, always ready to take | a passive-jawed MECHANISM the agent must itself close around the cargo before the giver may let go |
| what "handover" means | release at a fixed free-space point | an **escrow protocol**: present on a support, transfer the grip, withdraw the support — the load-bearing party changes only while both hold it |
| ordering | pick → handover → carry (given by the script) | **physically forced 3-phase order**: clamp-first makes the spool PERCH on the roofed prongs (rejected by height); release-first hands the spool to nobody — it rides out with the shelf or falls into the basin |
| success criterion | object at pose | a settled **suspension invariant**: jaw closed ∧ ledge fully out ∧ spool upright at hang height in the capture cell — a state where NOTHING but the receiver can support the load (audited) |
| object identity | one box | the waisted spool vs a same-colour, same-height, same-mass **decoy** with no waist — the jaw physically stalls on it 35 mm short of closed (smoke-measured 47.5 mm vs the audit's predicted 47.4 mm) |
| failure structure | miss the pose | wrong order, wrong object, partial clamp and no-release near-misses each rejected by a distinct clause on the settled end state |

The seed's plan — carry the object to the receiver and let go — executed verbatim
here latches 0.25 and never succeeds (smoke check 5). It is also different from
every sibling read during construction: `handover_i124` is equilibrium metrology on
a passive balance; `franka_handover_i115` is a sealed-boundary carousel transit.
Here the difficulty is a **grip-transfer protocol on an actuated receiver**: the
agent operates the receiving hand itself, and the order is enforced by geometry
(roofed prongs, escrow shelf), not by a rule latch.

## Scene (procedural, single station)

Deck + walled reject basin + fixed anvil wall; ORANGE jaw carriage (two prong
fingers with a 32 mm U-channel, bridge, upright handle) on an X prismatic slide
(limits [0, 92] mm, + = open); YELLOW escrow ledge (thin shelf + low pull tab) on a
crossing Y prismatic slide (limits [0, 105] mm, + = out); GREEN waisted spool
(60 mm flanges / 26 mm waist, 0.12 kg) and GREEN plain decoy (same Ø/height/mass) on
two apron slots (permuted per episode). Both slides are honest PhysX prismatic
joints (frictionless outside articulations; settling by authored damping); the
scene's post_step owns both external-wrench slots and exposes scalar sanctioned
drives (`jaw_drive` + closes, `ledge_drive` + withdraws), clamped to 6 N and
re-encoded into each body's LIVE frame every step. ~20 `__post_init__` audits pin
the escrow contract: closed aperture < waist Ø; open channel admits the flange;
perch above / basin far below / hang & stand inside the z band; withdrawn shelf
parks a riding spool outside the cell; decoy stall-q ≥ closed + 25 mm; tab, slots
and shelf clear of the jaw sweep.

## Teleport solution (solve.py) — phases

1. **Settle + perception** — read back WHICH slot holds the spool (station-local
   box test, asserted against the episode's `swap`) and both slide seats (asserted
   against `q_jaw0`/`q_ledge0`). `SIM_GEN_SCORE ~0.00`.
2. **PRESENT** — transport-only teleport of the spool to free air 25 mm above its
   standing pose on the ledge, released upright with the station heading; gravity
   stands it between the open prongs. `SIM_GEN_SCORE 0.25`.
3. **CLAMP** — sanctioned-drive velocity servo on the jaw slide (K = 15 N·s/m,
   v_des = 0.08 m/s, K·dt/m = 0.25 ≪ 1; the jaw closes in −q so the
   negative-feedback law is K·(v_des + q̇)) until q_jaw ≤ 4 mm: the notch flanks the
   waist between the flanges. `SIM_GEN_SCORE 0.55`.
4. **RELEASE** — the same servo law withdraws the ledge to its stop; the spool drops
   9 mm and hangs by its top flange on the prong tops over the basin.
5. **VERIFY** — hands off until `success()` (jaw closed ∧ ledge out ∧ upright in
   cell ∧ settled ∧ finite), `SIM_GEN_SCORE 1.00`, then ≥ 3.3 more simulated seconds
   hands-off; success must persist → `SIM_GEN_SOLVE: SUCCESS`. Passes on forge seeds
   0, 3 (spool in slot 0) and 1, 2 (spool in slot 1).

## Embodiment argument (single Franka arm, parallel-jaw gripper)

Base pose: ground-mounted at station-local ≈ (0.50 m, 0), facing −x. Every grasp
and press point lies within 0.56 m horizontal reach at ≤ 0.30 m height — inside the
Franka's 0.855 m envelope; every load ≪ the 3 kg payload.

- **Spool (0.12 kg):** top-down grasp on the 60 mm top flange across y (open sky
  above the apron and above the open channel), or a waist grasp (26 mm ≪ 80 mm jaw
  span). Set-down on the ledge reproduces the demonstrated release-from-25 mm; the
  open prongs leave the capture spot approachable from above (clamp-first would
  deny it — which is exactly the task's order).
- **Jaw handle:** upright 36 × 36 mm post, top at station z ≈ 0.25 m, always in
  open air behind the prongs; push −x with the closed gripper or a pinch grasp;
  the servo force scale (≤ 6 N) is trivial for the arm. The handle is the only
  contact needed — never the prongs.
- **Ledge tab:** 30 mm bar offset in +y, audited clear of the jaw sweep and of both
  cylinders for the whole stroke; pinch on its x-faces and pull +y ~105 mm at
  constant height.
- **Perception:** the spool and decoy differ ONLY in silhouette (waist vs straight
  wall — 17 mm of radius over 30 mm of height at wrist-camera range);
  receptacle/actor colour coding (ORANGE jaw, YELLOW ledge, GREEN cargo) makes the
  three interaction sites unambiguous.

## Execution order

**Required, and physically forced** (this is the point of the task): PRESENT while
the jaw is open → CLAMP → RELEASE. Clamp-first perches the spool on the roofed
prongs above the success band (smoke check 7); release-first loses the spool — it
rides out with the shelf or falls into the basin (smoke check 6); the rubric's
clamp latch is additionally order-gated on the present latch, so closing an empty
jaw pays nothing.

## Rubric (scene.py)

- 0.25 — presented: the spool ever stood upright at the capture spot (latched)
- 0.30 — clamped: the jaw ever closed around the spool in the capture cell
  (latched, order-gated on presented)
- capped at 0.55; exactly 1.0 iff live `success()`: jaw closed (≤ 12 mm) ∧ ledge
  out (≥ 98 mm) ∧ spool upright at hang height in the capture cell ∧ settled
  (FD slide rates + body velocities) ∧ finite.

## Checks

- `solve.py`: `SIM_GEN_SOLVE: SUCCESS` on forge seeds 0/1/2/3 (both slot
  permutations), monotone `SIM_GEN_SCORE` prints at phase boundaries
  (0 → 0.25 → 0.55 → 1.0), ≥ 3.3 s hands-off persistence.
- `smoke.py`: 14 rejection-only checks — settle/no-NaN; randomization spans (yaw
  151°, xy, both swap values, jaw/ledge seat bands) + READBACK (slots vs `swap`,
  q vs `q_jaw0`/`q_ledge0`); null policy ~0; SEED strategy (present-and-let-go →
  0.25 cap); premature release (rides out with the shelf, 0.25, no success);
  clamp-first (perch at 0.209 above the band, empty close pays 0, ledge-out changes
  nothing); partial close (q ≈ 47 mm, spool lost to the basin, jaw clause refuses);
  wrong object (decoy stalls the full servo at 47.5 mm — moved-assert so the probe
  is not vacuous); ledge-in near-miss (every clause true but ledge-out, 0.55 cap);
  latched credit survives theft (0.55 after the clamped spool is stolen); rejection
  audit (success never observed); final no-NaN; frames.npz video (563 frames).
  `SIM_GEN_SMOKE: ALL PASS 14/14`.
