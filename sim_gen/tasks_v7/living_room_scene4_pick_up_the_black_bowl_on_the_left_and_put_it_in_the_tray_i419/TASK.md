# living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray_i419 — Walk the Stele

Scene: `stele_walk` · Env: `simgen.stele_walk` · Robot slot: `null` (scene-level task;
solve drives one external wrench on the stele — the sanctioned proxy for two hands on
the stone)

## Seed provenance

Seed: `libero_90/living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray`
("pick up the black bowl on the left and put it in the tray"): two identical black bowls
disambiguated only by POSITION; the solver grasps the left one and drops it into a
stationary open tray. The bowl weighs grams, success is a bbox containment readout, and
the whole strategy is one free-space grasp-carry-release — the object goes wherever the
gripper takes it.

## What changed and why it is strategically different

The seed's outcome shape is kept — two IDENTICAL black objects, only the LEFT one belongs
in the receptacle — but the object is a stout black basalt STELE (120 × 120 × 240 mm,
2.2 kg) and **every continuous transport mode is removed**, each by a different mechanism:

- **No grasp-carry:** 120 mm across > an ~80 mm parallel jaw (asserted), and CARRYING is a
  declared, readback-measured **permanent foul** — all four base corners more than 32 mm
  clear of the deck over the causeway (fixture-x ∈ [0.05, 0.50]) for ≥ 12 substeps
  forfeits the episode forever (score 0, success unreachable). The seed's entire strategy
  is not merely unhelpful here; it is the flagship smoke rejection (checks 5–6: a carried
  stele later standing PERFECTLY in the socket still scores 0).
- **No slide:** two full-width 20 mm KERBS cross the causeway. A flat stele pushed hard
  and low (20 N at 2 cm height — above sliding friction, far below the 65 N low-push tip
  bound) jams its leading base edge on the kerb face and pins there (smoke check 7).
- **No tumble:** rolling it end-over-end crosses tilt 30° — between the 26.6° edge-topple
  and the 35.3° diagonal-topple, i.e. exactly "it is falling over" — which is the same
  permanent foul (smoke check 8; the asserts in `__post_init__` pin the window).

What remains is the ancient furniture-mover's answer: **WALK it**. Tip the stele ~15°
onto one base edge (raising the free edge 25–44 mm — over the 20 mm kerbs and the 12 mm
socket threshold), pivot the raised side forward about the planted corner, set it down,
alternate sides. Locomotion is a property of the object's own alternating contact set —
the support corner migrates step by step, and progress per half-step (~2–3 cm) is bought
by rotation about a planted corner, never by translation of a free body.

Distinct from the corpus (survey of the tasks_v7 cards): the same-seed sibling **i73
Roller Freight** interposes rolling elements under an un-pushable slab — the freight
still SLIDES (rides) along a straight channel under a capped push; **i61 / i368 / the
ferry-depot family** move carts and receptacles on rails; sled/trolley/shuttle tasks tow
or push bodies along guided paths; the die quarter-roll task tips a cube for an SO(3)
goal but tipping THERE is the allowed move, with no gait, no kerbs, and no foul framing.
No corpus task locomotes cargo by a rock-and-swivel walking gait, none makes BOTH lifting
and toppling permanent readback fouls so that only the gait survives, and none steps an
object OVER raised full-width barriers that jam the slide by construction. The solve is
also the only one of its siblings with **zero teleports of any kind** — every millimetre
of progress is contact dynamics.

## Scene

Procedural only. Kinematic **causeway**: deck slab 0.98 × 0.42 m (deck top = fixture
z 0), two orange kerbs (20 mm tall) at fixture x 0.14 and 0.32, and a green **socket** at
x 0.58 — a 170 mm square court walled 25 mm on three sides with a low 12 mm threshold
facing the berth (arrival is itself a step, not a slide). Two identical dynamic black
**steles** (120 × 120 × 240 mm, 2.2 kg, μ 0.9/0.8, restitution 0) side by side in the
berth at y ± 0.09.

Randomization (verified by smoke readback): causeway xy ± 40 mm + yaw ± 25°, fair draw of
WHICH body spawns on the left, per-stele xy ± 8 mm + yaw ± 4°.

## Rubric (latched, monotone; score in [0, 1])

| credit | stage |
|---|---|
| 0.10 | latched: left the berth (fixture x > 0.06) |
| +0.20 | latched: whole base past kerb 1, upright-ish |
| +0.20 | latched: whole base past kerb 2, upright-ish |
| +0.15 | latched: reached the socket apron (x > 0.44, in lane) |
| cap 0.65 | pre-success cap |
| 1.00 | iff `success()`: never fouled ∧ all 4 base corners inside the socket ∧ upright (< 8°) ∧ decoy NOT in the socket ∧ settled |

Latches are evaluated every physics substep and gated on not-fouled; **a foul zeroes the
score permanently**. Null policy scores 0 (berth x ≈ 0 < 0.06; nothing moves by itself).

## Walking solution (solve.py) — ZERO teleports

One external wrench on the target stele, every substep:

1. **P0** settle + layout readback (target LEFT, decoy RIGHT asserted); `SIM_GEN_SCORE 0`.
2. **Walk loop** (≤ 64 half-steps): PD orientation servo (τ = 14·axis-angle error −
   1.0·ω, capped 3 N·m — 2.3× below the 1.29 N·m static topple hump is the margin the cap
   enforces dynamically) + gravity feed-forward along the lean axis + a ≤ 2 N fixture-frame
   lane-trim force (~0.09 g — cannot drag or lift the stone). Each half-step:
   TIP (ramp to 15° on a base edge) → SWING (yaw ~0.3–0.6 rad about the planted corner) →
   DOWN → hands-off SETTLE. Lean direction is chosen from the corner readback: REAR lean
   by default (plants a rear corner, the whole FRONT edge flies 25–44 mm — over kerbs and
   threshold); FORWARD lean only when the rear edge is at an obstacle and the front is
   clear; a frozen (x, ψ) history triggers a back-off half-step. Steering shrinks near the
   walled socket. `SIM_GEN_SCORE` printed as each latch fires: 0.10 → 0.30 → 0.50 → 0.65.
3. **Arrival** — walks over the threshold, squares up inside; settle → success;
   `SIM_GEN_SCORE 1.0000`.
4. **Persistence** — 400 substeps (3.3 s) fully hands-off; `SIM_GEN_SOLVE: SUCCESS` only
   if success still holds.

Verified on the forge: **seeds 0, 1 and 2** (21–22 half-steps), monotone score ladder
0 → 0.10 → 0.30 → 0.50 → 0.65 → 1.0, `SIM_GEN_SOLVE: SUCCESS` each.

## Embodiment argument (single Franka, OSC, 80 mm parallel jaw)

Plausible base at **(0.24, −0.50)** in the fixture frame (berth far corner, socket back
corner and kerb ends all asserted < 0.76 m reach in `__post_init__`):

- **The stele is never grasped** (cannot be: 120 mm > 80 mm jaw, asserted). Walking
  furniture is a two-contact skill an arm performs by pressing on the upper third of two
  adjacent faces: one contact loads a base edge (the tip), the other sweeps the raised
  side around the planted corner (the swing). The solve's capped wrench (≤ 3 N·m, ≤ 2 N)
  is the sanctioned proxy for exactly those fingertip/palm contacts — 3 N·m at the
  ~0.2 m lever of the upper faces is ~15 N of pressing force, well inside arm scale, and
  the 15° working tilt keeps the stone statically recoverable (edge-topple 26.6°) so the
  hand never bears its weight.
- **Nothing else is touched:** kerbs, threshold and socket are kinematic furniture; the
  decoy stays where it stands.
- All work happens between deck height and 0.24 m, within 0.76 m of the base; no bimanual
  lift (the foul framing removes the very move that would need it).

## Execution order (declared)

Built in this order: (1) scene with minimal `success()` + foul machinery, (2) walking
solve iterated on the forge (3 iterations: side-lean gait jammed at kerb 1 → rear/forward
lean with front-blocked-wins fixed kerbs but wedged diagonally in a 123 mm apron → apron
widened to 153 mm (sock_x 0.58) + cfg-derived obstacle windows + wedge back-off), (3)
final latched rubric, (4) smoke battery. In-task order is geometry-forced: kerb 1 before
kerb 2 before threshold (the latches' whole-base-past predicates can only fire in
sequence along the causeway).

## Checks

- **solve.py** — forge, seeds 0, 1, 2: `SIM_GEN_SOLVE: SUCCESS`, monotone scores, zero
  teleports.
- **smoke.py** — forge: `SIM_GEN_SMOKE: ALL PASS 16/16`:
  1. settle/no-NaN, both steles standing at the berth;
  2. layout readback: target LEFT, decoy RIGHT, latches clear, score ~0;
  3. null policy 240 steps → ~0;
  4. hover calibration: 20 mm clearance (< the 32 mm carry line) never fouls;
  5. SEED strategy: an 80 mm lift-and-carry down the causeway fires the carried foul;
  6. carry is forfeit: the fouled episode's stele placed PERFECTLY in the socket
     (in-socket readback True) → success False, score 0;
  7. slide jam: 20 N low push pins the base edge on kerb 1 (readback), kerb-1 latch
     never fires, score ≤ 0.10;
  8. tumble: rolled past 30° → foul, score 0;
  9. wrong object: the DECOY standing in the socket → no success, score ~0;
  10. near-miss: perched on the threshold, rear corners outside → no success, ≤ 0.65 cap;
  11. latched credit: departure 0.10 latched, return to berth keeps exactly 0.10;
  12. ACCEPTANCE (non-audited): target standing centred in the socket → success True,
      score 1.0;
  13. randomization real (3-seed max-pairwise readback: causeway xy/yaw + target xy);
  14. swap coverage: LEFT body is A on some seeds, B on others;
  15. rejection audit: success never True at any audited point;
  16. final no-NaN; frames.npz saved.
- **scene.py `__post_init__`** — ~19 asserts (no-grasp, topple-foul window between edge
  and diagonal topple, gait tip clears kerb + threshold, kerb jams a flat slide, carry
  line above every legitimate rest, bays/socket admit the stele with walking room, berth
  spacing, one base pose reaches all required contacts).
