# waiters_pull — seat the bowl by yanking its slat out (the waiter's pull)

`sim_gen` task `libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_i117`
· scene name `waiters_pull` · env `simgen.waiters_pull` · robot `null`

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle`
(RoboVerse pack, `roboverse_pack/tasks/libero_90/...stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.py`).
The seed is a plain pick-and-place: grasp `akita_black_bowl_1`, hover it over
`akita_black_bowl_2`, release; success is an xy/height proximity predicate on the two bowl
origins. Its whole difficulty lives in grasping and carrying the payload.

## What this task is

A **waiter's pull** (tablecloth trick). At reset a slick, low-friction slat bridges the rim of
an octagon-walled pedestal, and a wide-flanged bowl rides on TOP of the slat — a
bowl-on-slat-on-pedestal sandwich. The goal "stack" (bowl seated flange-on-rim in the pedestal
recess) is reached **by removing the interposed support, never by moving the bowl**: yank the
slat out from under the bowl by its red tab, fast, so kinetic friction (pair μ ≈ 0.08,
min-combine material) has no time to drag the bowl off the 24 mm radial capture window; the
bowl loses support and drops ~20 mm into the recess under gravity, self-centred by the octagon
wall. Then stow the slat flat between the rails of a kinematic tray parked at the side of the
counter. Success = bowl seated + slat stowed + pedestal undisturbed + slat clear + everything
settled, judged on the settled state.

The documented failure mode is physical, not geometric: a slow, cautious pull lets friction
walk the bowl along with the slat and off the pedestal — drag distance grows as ½·μg·t², so
the skill being tested is **speed/impulse management of a loaded shear extraction**, the exact
opposite of the seed's slow careful transport.

## Why it is strategically different

- **vs the seed**: the seed moves the payload (grasp–carry–release, proximity predicate).
  Here the payload is *never touched* — the rubric's draw latch requires the bowl to be
  near-still and the slat in real motion while the support leaves. The manipulated object is
  the slat; the bowl only ever moves by gravity. Also unlike the seed, the goal has an ordered
  second clause (stow the slat) so "just put the bowl in the recess by hand" scores 0.25 and
  never succeeds (smoke check 8 constructs precisely that seed-strategy end state and asserts
  rejection).
- **vs the corpus**: no existing task does a *loaded horizontal shear extraction with a
  payload landing*. `falsework_tent` removes a prop, but vertically, unloaded, and the
  structure is static; `play_jenga_i31` pivoted away from extraction to compass dials;
  `ledge_catch` pushes the item itself off an edge and catches it ballistically; the pyramid /
  bowl-stack tasks all transport the payload. The friction-race dynamic (extraction speed vs
  payload drag) appears nowhere else.

## Scene (procedural, no external assets)

Counter slab (kinematic, top z = 0.20). All dynamic bodies are compound rigid bodies from
procedurally spawned USD prims with authored masses and a bound low-friction physics material
on the slat (static/dynamic 0.08, restitution 0, combine-mode *min*):

- **pedestal** (0.80 kg): disc r 70 mm × 14 mm plus an 8-segment octagon wall
  (recess r 54 mm, wall 12 mm thick, 32 mm tall; rim top 46 mm above the counter). Free —
  a clumsy draw can knock it off station (`pedestal_home` gate, ±8 mm z).
- **bowl** (0.15 kg): foot cylinder r 30 mm × 12 mm under a wide flange r 62 mm × 14 mm.
  The 124 mm flange overhangs the recess and seats on the wall-top annulus at
  seat_dz = 34 mm ± 7 mm; the foot drops inside the recess (24 mm radial margin).
- **slat** (0.06 kg): 300 × 90 × 8 mm slick board with a red pull tab
  (30 × 24 × 30 mm) at +115 mm — the only intended grasp point in the scene.
- **tray** (kinematic): floor 360 × 130 × 10 mm with two side rails and one end rail
  (34 mm tall); one end open so the slat can be slid or laid in.

Per-seed randomization (all verified by readback in smoke): pedestal xy jitter ±30 mm, pull
axis yaw π ± 25°, bowl-on-slat jitter ±8 mm, tray side mirrored ±y with x jitter ±20 mm and
yaw ±12°.

## Rubric

Milestone latches in `post_step`, additive score, all rest latches velocity-gated
(`latch_speed` 0.10 m/s):

- **0.20 drawn** — latches only during a *real* draw: slat displaced > 60 mm along the pull,
  bowl within 25 mm of the pedestal centre, bowl still riding above the seat band, bowl
  near-still (< 0.30 m/s), **and the slat itself moving > 0.15 m/s**. A teleported slat has
  zero written velocity and can never latch this.
- **0.25 seated** — bowl flange on the rim: xy ≤ 20 mm, z within 34 ± 7 mm of the pedestal
  origin, upright ≤ 10°, at rest.
- **0.25 stowed** — slat flat in the tray: tray-frame x ≤ 22 mm / y ≤ 20 mm, z within 6 mm of
  the floor-rest height, yaw ≤ 25° (mod 180), tilt ≤ 8° (a slat propped on a rail fails), at
  rest.
- **1.00** iff `success()` = seated ∧ stowed ∧ pedestal_home ∧ slat_clear (≥ 160 mm) ∧
  settled — judged on the settled configuration.

## Execution order (declared)

1. Minimal goal predicate + scene brought up on the forge.
2. `solve.py` made to pass on the forge (this drove the flange-radius fix below).
3. Rubric finalized (draw latch, honesty gates) — then `smoke.py` written against it.

Design iteration worth recording: the first flange (r 52 mm) was *smaller* than the recess
(r 54 mm), so the drawn bowl fell straight to the recess floor (dz ≈ 14 mm) instead of
seating on the rim — caught by the solve's dz readback on the forge, fixed by widening the
flange to 62 mm. And the first draw latch could be teleport-satisfied by writing the slat
away while the bowl hovered; fixed with the `draw_slat_speed > 0.15` in-motion gate
(smoke checks 8/10/13 pin the score at 0.25 for every such construction).

## Teleport solution (solve.py) — phase outline

Teleports transport only; both load-bearing interactions are pure contact dynamics.

- **Phase 0** — reset(seed), settle the authored sandwich, read back layout
  (pull yaw, pedestal xy, tray pose). `SIM_GEN_SCORE 0.000`.
- **Phase 1 — the draw (forces only)**: bang-bang horizontal external force on the *slat*
  (the body the arm would pull by the tab) along the pull axis under a speed cap
  (attempt ladder v_cap/F = 2.0/6 N → 2.5/8 N → 3.0/10 N) until 350 mm of travel, then an
  active reverse-force brake. The bowl is never written or forced: it loses support and
  seats by gravity (measured landing error 6.7–11.4 mm across seeds vs the 24 mm capture
  margin). Failed attempts rebuild the *start* sandwich by transport teleports and retry
  through physics. `SIM_GEN_SCORE 0.450`.
- **Phase 2 — the stow (transport + gravity)**: slat teleported to a hover 20 mm above the
  tray floor (outside the 6 mm scoring z band), released, lands and settles between the
  rails. `SIM_GEN_SCORE 1.000`.
- **Phase 3 — persistence**: ≥ 3.3 simulated seconds hands-off, success must hold.
  `SIM_GEN_SCORE 1.000`, then `SIM_GEN_SOLVE: SUCCESS`.

Forge-verified: `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1, 2 (first-attempt draws, rc = 0),
re-verified on seeds 0 and 1 after the draw-latch change.

## Embodiment argument (Franka, one plausible base pose)

Base plausibly at **(−0.42, 0.00, 0.20)** on the counter, facing +x: the pedestal
(≈ 0.06, 0) sits ~0.48 m away and the tray park (0.14, ±0.34) ~0.65 m — inside a Franka's
~0.85 m comfortable reach, and the pull axis (π ± 25°, i.e. *toward* the robot) means the
draw is a natural pull-in stroke of the arm.

- **slat**: grasped ONLY by the red tab — 24 mm wide × 30 mm tall, a comfortable Franka
  two-jaw pinch (< 80 mm stroke), sticking up clear of the bowl flange. The draw is a
  straight-line cartesian pull at ~2 m/s over 0.35 m — brisk but inside Franka joint-velocity
  limits; the brake is just decelerating the light (60 g) slat. The stow is a carry and a
  20 mm drop.
- **bowl**: 124 mm flange — wider than the Franka's maximum jaw opening, deliberately
  *ungraspable*, and nothing in the task requires touching it.
- **pedestal**: 0.80 kg free fixture; never manipulated, only not-to-be-disturbed.
- **tray**: kinematic furniture; the slat is laid into its open top / open end.

## Smoke battery (16/16 on the forge, frames.npz recorded)

1. Authored sandwich settles finite and HOLDS (bowl rides ~20 mm above the seat band).
2. Rubric clean at reset: score 0, no success.
3. Determinism: same seed → identical layout readback.
4. Randomization readback: pull-axis yaw varies across seeds.
5. Randomization readback: pedestal centre jitters.
6. Randomization readback: tray side takes both values, tray yaw varies.
7. Null policy: 240 idle steps → score ~0, no success (sandwich does not self-solve).
8. **Seed strategy rejected**: bowl PLACED seated by hand, slat left on the counter →
   seated TRUE but success FALSE, score pinned at 0.25 (no draw credit).
9. Tolerance twin / near miss: a 12 mm off-centre seat counts; a *settled* 22 mm off-centre
   seat (foot physically inside the recess) is rejected by the xy gate.
10. Stow without the seat: slat stowed, bowl dumped on the counter → stowed TRUE, success
    FALSE, no draw credit.
11. Stow twin: slat dropped centred (8 mm off) in the tray counts as stowed.
12. Slat laid CROSSWISE resting on both tray rails (~24 mm high) → not stowed (z + tilt).
13. Slat protruding 60 mm from the tray's open end → not stowed; score stays 0.25.
14. Latched credit is a latch: the 0.25 seat credit survives the bowl being knocked back off.
15. All states finite at the end.
16. (battery total asserted) `SIM_GEN_SMOKE: ALL PASS 16/16`.

## Checks

- solve: forge `SIM_GEN_SOLVE: SUCCESS`, seeds 0/1/2 (and 0/1 re-run post-latch-change),
  score trace 0.000 → 0.450 → 1.000 → 1.000, monotone.
- smoke: forge `SIM_GEN_SMOKE: ALL PASS 16/16`, rc 0, frames.npz written.
