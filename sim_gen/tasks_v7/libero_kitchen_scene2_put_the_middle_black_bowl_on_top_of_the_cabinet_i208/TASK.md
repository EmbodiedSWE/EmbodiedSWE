# letterbox_cabinet — clear the roofed seat out the discard chute, then drop the middle bowl through the port

**Task id:** `libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i208`
**Env:** `simgen.letterbox_cabinet` (scene-level, `robot="null"`)
**Files:** `scene.py` (scene + rubric), `solve.py` (force-eviction + teleport-drop solution),
`smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet`
(RoboVerse `roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet.py`).
The seed places three identical akita black bowls and a plate beside a wooden cabinet; the
goal is one vertical pick-and-place — grasp the MIDDLE bowl and set it inside a bbox above
the cabinet's always-free flat top. Success is a static pose test on that bowl alone.

Kept from the seed: three identical black bowls in a row with the MIDDLE one as the target
(identity discrimination), a distractor plate, a wooden cabinet, and the goal of placing the
middle bowl onto the cabinet's top level — even the seed's own vertical release is kept as
the finale.

## What the task is

The cabinet's top compartment is roofed; the roof has one square **DROP PORT** (15 × 15 cm)
directly above the goal seat — the only entry that counts (latch-gated). But the seat is
**OCCUPIED**: two flat gray **SLABS** (13 × 15.5 × 4.5 cm — wider than a parallel jaw in
both horizontal spans, and under the roof, so they are push-only) lie on the compartment
floor beneath the port. A bowl released over the cabinet right now lands ON the slabs,
above the seat band: the seed's exact plan is a settled, scored failure (smoke check 12).

The front face has a service **MOUTH** (25 × 11.5 cm) that admits a gripper hand + wrist
for pushing. The back wall has a **DISCARD CHUTE** opening (23 × 8.5 cm) onto a slick
30.8° slide that runs down into a green **CATCH BIN** behind the cabinet. Three identical
black bowls stand in a row on a staging block; which physical body occupies which slot is a
fresh random permutation every episode (readback-verified), so "the middle bowl" is a
per-episode identity. The goal:

> "Clear the cabinet's top compartment by pushing the two gray slabs out through the back
> discard chute so both land in the catch bin, then drop the MIDDLE black bowl of the three
> through the roof port so it rests upright on the cleared seat. Do not put the other bowls
> or the plate in the compartment or the bin."

`success()` = middle bowl upright at rest in the seat band ∧ the **port-passage latch**
fired (a bowl brought in through the mouth or teleported onto the seat does not count)
∧ both slabs at rest **inside the bin**, each having physically passed **through the
chute** (per-slab passage latches — a slab teleported into the bin earns nothing) ∧ no
decoy bowl in the compartment or the bin ∧ everything settled.

## Strategic difference — vs the seed and vs every corpus neighbor

- **vs the seed:** the seed's whole plan is one grasp and one vertical release. Here that
  release still happens — but only as the finale, and executed immediately it is a measured
  failure (parks on the slabs, 0.0). The load-bearing work is inverted onto the NON-target
  objects: evict the obstructions through a constrained rear passage, with a required
  destination (the bin) for the debris itself.
- **vs sibling i140 ramp_hutch:** i140 seals the top and routes the TARGET bowl up a ramp
  through a lateral one-way doorway — the target itself travels a constrained path, and
  transport alone can never finish. Here the target's path is the seed's own vertical drop;
  the strategic novelty is the opposite object-of-work: the DEBRIS must be routed (contact
  chain → chute → gravity slide → bin) before the target's one-step placement can score.
- **vs i306 hatch_shelf / i5 counterweight_shelf:** those reconfigure the fixture (close a
  lid, ballast a shelf) to create/level the destination. Here the fixture never moves;
  the destination exists from the start but is *occupied*, and the occupiers have their own
  goal region with passage-gated credit.
- **vs i61 cart_ferry:** cart_ferry repositions a vehicle twice and ferries the target on
  it. Here nothing carries anything; disposal is one-way gravity delivery, and the target
  is handled once.
- **vs i53 slab_easel / i14 chute_switch / i48 cask_weight_sort / i17 skittle_gallery:**
  those choose *which* terminal state to produce (orientation/routing/sorting decisions).
  Here the terminal state is unique; the challenge is a mandatory clearing sub-task whose
  own product (the slabs) must be delivered to a specific container.
- **vs i57 hanoi_rings / i27 bell-herd:** no symbolic ordering or multi-agent component;
  the one ordering constraint (clear before drop) is enforced physically (the early drop
  parks on the slabs) and by rubric gating, not by rule.
- **vs robobench suite (balance_scale, combination_safe, syringe, pen_holder):** no
  articulated mechanism; the "mechanism" is emergent rigid-body geometry (contact-chain
  push, passage clearances, gravity slide, one-way bin lip).

No task in the read corpus makes "the destination is blocked; remove the blockers through
their own constrained passage into their own required container" the core strategy, and
none keeps the seed's exact motion as a finale that scores 0 until the clearing is done.

## Rubric (latched credit, `score()`)

| latch | condition (fixture frame) | weight |
|---|---|---|
| `_chute_ever[i]` | slab *i* center ever inside the chute opening (x −0.22..−0.145, |y|<0.12, z 0.25..0.345) | 0.125 each |
| `_binned_ever[i]` | slab *i* ever inside the bin **and** its own chute latch already fired | 0.125 each |
| `_seated_ever` | target bowl ever seated (upright, seat band) **and** the port latch fired **and** both chute latches fired | 0.30 |

`score = Σ weights`, capped at 0.95 unless `success()`, which returns exactly 1.0. Latches
are evaluated in `post_step` every physics substep; the bin latches are **gated on the
chute passage** (teleporting a slab into the bin earns nothing — smoke check 13) and the
seat latch is **gated on the port passage and both chutes** (mouth entry or a seat teleport
earns nothing — smoke checks 13, 14). Success additionally requires bin residency LIVE, so
latched credit cannot fake a terminal state (smoke check 15). The printed `SIM_GEN_SCORE`
sequence is non-decreasing by construction.

## Solution (`solve.py`) — teleport = transport only

1. **EVICT B** — pulsed 6 N force on the FRONT slab along the fixture's −x axis (force on
   only below 0.08 m/s; chain friction ≈ 2.8 N resists; 6 N × 0.0225 m tipping moment <
   mg × 0.065 m restoring). The front slab pushes the back slab — a contact chain —
   through the chute opening (passage latch), over the plinth edge; it tips onto the
   30.8° slide (pair μ ≈ 0.30 ≪ tan 30.8° = 0.596) and gravity delivers it into the bin.
   Cut at back-slab x < −0.215, settle. → 0.250
2. **EVICT A** — same pulsed push (4 N) until the front slab's own center passes
   x < −0.215; cut; it slides into the bin beside the first. → 0.500
3. **DROP** — teleport the MIDDLE bowl (scene.target_idx) from the staging row to fixture
   (0.03, 0, 0.44): above the port, ABOVE the port passage band (z 0.32..0.40), zero
   velocity — pure transport through free air; the solve *asserts the score is still
   0.500* after the write. Hands off: the bowl free-falls through the port (latch fires
   mid-fall, ~10 samples in the band), lands upright on the cleared seat, settles into
   success. → 1.000
4. **PERSISTENCE** — ≥ 3.5 simulated seconds fully hands-off; `success()` must still hold
   before `SIM_GEN_SOLVE: SUCCESS` is printed.

Because the pod's external-force API may interpret wrenches in the body's current frame,
`drive()` probes force encoding at runtime from measured progress and toggles between
raw-world and `quat_apply_inverse(q_now, f)` if the pushed slab stalls.

## Embodiment argument (single Franka, 8 cm parallel jaw, OSC)

The same plan executes with one arm:

- **Evict the slabs:** both horizontal spans (13, 15.5 cm) exceed the 8 cm jaw and the
  roof (11.5 cm of headroom) blocks any top approach — the slabs are push-only, exactly as
  the solve treats them. The front mouth is 25 cm wide × 11.5 cm tall: the Franka hand
  (~9 cm wide, ~6.5 cm tall) and wrist flange (⌀ ~10 cm) pass through it flat. Pushing the
  front slab until its center crosses the plinth edge puts the fingertip at fixture
  x ≈ −0.125; with fingers + hand extending ~16 cm ahead of the flange, the flange sits at
  x ≈ +0.035 — inside the mouth, with ~5 cm of vertical slack. Required force ≈ 1.4–2.8 N.
  The chute and slide deliver each slab by gravity; the hand never enters the chute or
  goes near the bin.
- **Drop the middle bowl:** the bowl is a 9.8 cm-wide open cup — too wide for an outside
  pinch, but the open top admits a **rim pinch** (one fingertip inside the 0.9 cm wall,
  one outside), the standard way to lift an open cup. Read the row for the middle
  identity, rim-pinch it, carry it ABOVE the roof (top z 0.39), center it over the port
  and release at z ≈ 0.46; it free-falls through the port. The hand stays above the roof
  the whole time — it never needs to fit through the port or enter the compartment.
- **One plausible base pose:** base at fixture-frame ≈ (0.60, −0.45), facing the cabinet.
  Flange-in-mouth point (0.035, 0, 0.315) is at 0.72 m; the middle bowl (0.46, ~0, 0.13)
  at ~0.47 m; the port release point (0.03, 0, ~0.50) at ~0.75 m — all within the 0.855 m
  reach, with every required contact at z ≤ 0.50.

No stage admits a shortcut: the slabs cannot be grasped and the bin's 14 cm walls and 6 cm
entry lip cannot be crossed by a floor-level push, so the chute is the only physical route
to bin residency (and the bin latch is chute-gated anyway); a bowl slid in through the
mouth rests on the seat but earns nothing (port gate, smoke check 14); grabbing an
arbitrary bowl fails the decoy vetoes, so the row must be read for the middle identity.

## Execution order

Clear-before-drop is enforced twice: physically (a bowl dropped first parks on the slabs
above the seat band and scores 0 — smoke check 12) and by rubric gating (the seated latch
requires both chute latches and the port latch). Slab order is forced by geometry: the
back slab can only leave first (the front slab is between it and the mouth), so the chain
push is the only admissible eviction.

## Checks (`smoke.py`) — 16

1. reset settles, states finite, slabs on the seat, three bowls in a row on the staging
block, bin empty; 2. score ~0, no latches at reset; 3. randomization READBACK across 6
seeds (fixture xy/yaw, TARGET BODY INDEX varies, slab y, row y, bowl yaw); 4. null policy
240 steps ≈ 0; 5–7. oracle on seeds 0/1/2 (chain-evict 0.25 → 0.50, port drop → success
1.0, persists 240 steps); 8. occupied forfeit (decoy teleported into the compartment flips
success off, score falls to the 0.80 latch sum); 9. discard forfeit (decoy moved to the
BIN keeps success off); 10–11. monotone ladder 0 < 0.125 (chuted) < 0.25 (binned) < 0.50
(both out) < 1.0, and every partial state < 1.0; 12. seed-strategy negative (bowl released
over the cabinet parks ON THE SLABS above the seat band, readback z, score 0);
13. teleport-bypass negative (slabs placed straight in the bin + target placed on the
seat: live predicates TRUE but every latch refuses — score 0); 14. mouth-entry negative
(bowl pushed in through the front mouth seats live but the port gate refuses — score
stays 0.50); 15. permanence + live bin residency (slab removed from the bin keeps latched
0.50; a legit port drop then raises latched credit to 0.80 but success is refused);
16. inverted negative (upside-down bowl readback-inside the seat band, live predicate
refuses). Frames recorded to `frames.npz`.

## Forge verification

- `solve --headless --seed {0,1,2}`: all three print the ladder
  `SIM_GEN_SCORE 0.000 → 0.250 → 0.500 → 1.000 → 1.000`, hold success through the
  420-step (3.5 s) hands-off window, and end `SIM_GEN_SOLVE: SUCCESS`, rc = 0
  (~19 s each). `target_idx` readback 2/2/1 confirms the bowl→slot permutation varies.
- `smoke --headless`: `SIM_GEN_SMOKE: ALL PASS 16/16`, rc = 0 (69.9 s),
  `frames.npz` (500 × 600 × 960 × 3) saved.
