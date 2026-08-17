# caliber_vault — sort two marbles into a roofed vault through the only aperture each one fits

**Task id:** `libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i404`
**Scene:** `simgen.caliber_vault` (`scene.py`, robot="null" scene-level env)

## Seed provenance

Derived from **libero_90 / `libero_kitchen_scene1_put_the_black_bowl_on_the_plate`**: pick one
payload (black bowl) and place it on an open, passive goal surface (the plate). Kept from the
seed: the skeleton "move payload(s) to a designated goal region and leave them at rest there",
a tabletop-scale workspace, and free rigid payloads. Changed: the single bowl becomes **two
spheres of different caliber** (d = 60 mm and d = 32 mm), and the open plate becomes a
**fully roofed and walled vault** with exactly two size-gated ways in.

## What the task is

A single kinematic compound rig (re-posed each reset with xy jitter + full yaw, so every
predicate lives in the rig body frame) contains:

- **Route 1 — the rail run.** Two horizontally *diverging* rails (gap 62 mm → 92 mm) enter
  the vault under its canopy. The **big marble (d=60)** bridges the gap and rides; because
  the gap widens, its center height h(x) = √((R+r_rail)² − (s(x)/2)²) falls covertly, so it
  self-accelerates on visually level rails and **drops through the gap** exactly over the
  vault interior (geometric release at s = 2(R+r_rail), x ≈ 0.32, verified against a dynamic
  departure model). The **small marble (d=32) never bridges** (62 > 2(r+r_rail) = 56 mm):
  dropped on the rails it falls straight through onto a return ramp that rolls it back out.
- **Route 2 — the side port.** A 44 × 44 mm doorway low in the vault's +y wall, fed by an
  external open-top runway lane (inner width 45 mm) ending in a 10 mm drop-in sill. The
  small marble rolls down the lane and through the port; the big marble does not even fit
  into the lane (45 < 60) and is refused by the port.

**Goal:** both marbles at rest inside the vault interior.

**Randomization (seed-driven, verified by readback in smoke):** rig xy jitter ±50 mm + full
yaw; the two marbles spawn in two staging slots with the **slot assignment swapped** per
seed plus per-slot jitter — the solver must identify which marble is which, not replay poses.

## Strategic difference

- **vs the seed:** the seed's entire plan — carry the payload over the goal and set it
  down — is *physically impossible* here: the goal region is under a roof. Success requires
  matching each object to the one aperture that admits it and letting the structure's
  physics (ride-and-release, lane-and-drop) finish the delivery. The scene is a size
  classifier the agent must route through, not a surface it can reach.
- **vs sibling i187 (hidden-mass beam balance):** no hidden state, no measurement — the
  discriminating property (diameter) is visible; the challenge is routing, not sensing.
- **vs sibling i221 (cloche uncover/re-cover):** no removable cover and no
  restore-the-cover ordering clause; the roof here is permanent and never comes off.
- **vs sibling i281 (berry pour):** no container-content relation and no held-container
  manipulation; both payloads are free rigid bodies delivered through fixed machine paths.
- **vs exemplar pen_holder:** no insertion into a socket; the goal is a *region* entered
  through interference-gated apertures, and half the delivery (rail ride, covert
  self-acceleration, geometric release) is done by the structure, not the mover.

An aimed ballistic throw through the entrance fascia slot and over the bulkhead is
geometrically conceivable but requires precision far beyond a plausible strategy; the rails
and the port are the intended (and rubric-latched) routes, and both latch chains
(`ride→canopy→vault`, `lane→vault`) are asserted by the solver.

## Solution phases (mirrors `solve.py`)

1. **Reset + audit** — settle, mass readbacks (rig 60 kg, big 0.120 kg, small 0.030 kg),
   staging sanity, score ≈ 0. `SIM_GEN_SCORE 0.000`.
2. **Phase A, rail run** — transport the big marble to a hover above the rails in the *open*
   loading bay and release. It seats into the groove, self-accelerates down the diverging
   run, passes under the canopy, clears the bulkhead notch, and falls through into the
   vault. Retry by re-release if needed. `SIM_GEN_SCORE 0.500`.
3. **Phase B, lane push** — transport the small marble to a hover over the start of the
   open-top lane (outside every scoring band) and release; push it down the lane with a
   contact-scale external force (0.06 N bang-bang, 0.30 m/s cap, world direction
   re-expressed in the spinning body frame every step), cut the force when its center
   passes the wall plane; it drops over the sill into the vault. Stalls escalate the force
   cap. `SIM_GEN_SCORE` ≥ 0.900 latched, then `1.000` at success.
4. **Persistence** — hands off ≥ 3.5 simulated seconds; success() must still hold. Then
   exactly `SIM_GEN_SOLVE: SUCCESS`.

Teleports are TRANSPORT only (always to non-scoring hover points outside the vault); every
scoring displacement happens through gravity and contact.

## Scoring (latched, monotone)

ride 0.10 + under-canopy 0.15 + big-in-vault 0.25 + in-lane 0.15 + small-in-vault 0.25,
capped at 0.90; 1.0 iff `success()` (both marbles inside the vault interior box and
settled). Latches only set in `post_step`, never cleared, so credit cannot evaporate and
the printed score sequence never decreases.

## Embodiment argument

A single **Franka with OSC from one base pose** can do everything the solve does:

- The 60 mm marble is grasped with the jaws near full open (~80 mm max aperture) from its
  open staging slot and set down in the **open loading bay** on the rails — an unobstructed
  top-down place under open sky (the canopy starts 90 mm further along the run).
- The 32 mm marble is pinch-grasped and set into the **open-top lane** at its start, then
  advanced with a fingertip push along the lane — the lane is uncovered along its whole
  length, and the 0.06 N contact push in solve.py is a stand-in for exactly that fingertip
  contact.
- Both drop-off points (bay at rig-local x ≈ 0.05 and lane start at y ≈ 0.27) are within
  ~0.35 m of the rig origin, comfortably inside one Franka reach envelope from a single
  base pose beside the rig.

No regrasp gymnastics, no dual arm, no in-hand rotation is needed; everything past the two
releases is done by the structure.

## Execution order

**Order-free.** Either marble may be delivered first; the two routes do not interact and no
latch is order-gated across routes (within each route the chain ride→canopy→vault /
lane→vault is causal, which the honest paths produce automatically).

## Checks (`smoke.py`, 11)

1. Reset settles, no NaNs, score ~0, no latches pre-fired.
2. Mass readback: rig 60, big 0.120, small 0.030 kg (custom-spawner MassAPI honesty).
3. Randomization by READBACK across seeds: rig x/y spread, yaw spread, slot swap exercised
   (both assignments observed).
4. Null policy: 240 steps hands-off, score ≤ 0.02.
5. Rails refuse the small marble: released at the big's drop point it falls straight
   through — never rides, never reaches the vault.
6. Port refuses the big marble: pushed at the doorway with 0.40 N it *advances* (probe is
   not vacuous) then stops outside the wall plane; no vault latch.
7. Seed-strategy end state (place from above): both marbles teleported over the goal rest
   ON the canopy — score ≤ 0.005, no success.
8. Latch persistence: after an honest rail delivery (score exactly 0.50), removing the big
   marble leaves the latched score unchanged.
9. Near miss: big delivered honestly + small parked mid-lane → score exactly 0.65, no
   success.
10. Lone small marble in the vault → partial credit < 0.50 and ≤ 0.90, no success.
11. Positive control: both marbles in the vault → success True, score 1.0.

Frames are recorded to `frames.npz`. Final marker: `SIM_GEN_SMOKE: ALL PASS 11/11`.
