# baton_stow — thread the too-long baton into the drawer, then seal it shut

`sim_gen` task `libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i199`
— env `simgen.baton_stow`, scene `baton_stow`, robot slot `null` (scene-level task).

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet`
(`roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.py`):
pick the akita black bowl off the table and place it inside the top drawer of the white
cabinet; `_terminated` is a single bowl-centre-in-drawer-OBB test, drawer state comes
from the shipped demo trajectory, distractors are inert.

## What changed, and why it is strategically different

The seed is one pick, one carry, one vertical release into a receptacle whose open mouth
is far larger than the payload. This task keeps the skeleton — "get the black object
INTO the drawer" — but makes the **vertical drop geometrically impossible** and adds a
**closure + exclusion** clause, so the scored skill becomes *oriented threading under a
clearance constraint*, i.e. keyed insertion, not place-and-drop:

- The payload is a rigid **black baton, 200×30×30 mm** — *longer than any exposure of
  the drawer mouth*. The drawer (interior 240×120×60 mm) slides out from under a fixed
  **roof** whose underside is 88 mm above the drawer floor; at full travel (165 mm) the
  exposed mouth strip is 157 mm long (max diagonal ≈ 163 mm < 200 mm). Dropped flat —
  axis-aligned or diagonal — the baton always bridges the rim (smoke checks 6–7). The
  only way in is the **cam-clearance thread**: enter pitched steeply (~50°), low tip on
  the drawer floor, then slide aft under the roof lip while pitching down (50°→14°),
  keeping the highest proud corner below the roof plane the whole way, and finally push
  the baton flat so both endpoints are inside the cavity.
- **Closure clause**: success additionally requires the drawer pushed back SHUT
  (opening ≤ 10 mm). "Inside but open" is explicitly not success (smoke check 9), and
  when shut, the 4–6 mm residual gaps are far under the 30 mm cross-section, so the
  enclosure is verifiably escape-proof.
- **Exclusion decoy**: a white baton (120×30×30 mm — short enough to drop in flat!)
  starts next to the black one; if it ends up in the drawer the score is crushed to
  ≤ 0.10 even with everything else perfect (smoke checks 11–12). "Shove both in" fails.
- Both batons spawn ON THE ROOF of the stow bay (randomized left/right slot swap, ±2 cm
  jitter, ±20° yaw), so nothing starts in or near the mouth.

Versus the corpus: not a contents-transfer/pour (i134 `drawer_bead_pour` — same cabinet
family, but its payloads fit trivially and the skill is pouring), not a kinetic relay
(i45 `domino_relay` — same seed, but there the payload may never be touched; here the
payload is directly manipulated and the difficulty is its *orientation trajectory*), not
a stacking, ballistic, bistable-closure, or re-installation task. No other task read
scores a payload that is **too long for the receptacle's opening**.

All assets are procedural compound spawners (no external files): kinematic shell
(plinth, side walls, back wall, roof slab), dynamic drawer (floor + 4 walls + face
plate + handle posts/crossbar) bound to the shell by a bind-time
`UsdPhysics.PrismaticJoint` (limits [−0.165, 0] m, joint-pair collision off, heavy
linear damping = springless slide that stays where it is left), and the two dynamic
batons (0.12 kg / 0.07 kg, explicit MassAPI + friction material).

**Rubric** (latched, monotone): 0.15·open_max + 0.25·thread (an endpoint enters the
cavity airspace) + 0.30·inside (both endpoints in) + 0.15·close_max (shut progress,
gated on live inside), cap 0.85; decoy-in caps everything at 0.10; exactly 1.0 iff
`success()` holds live (both endpoints in ∧ shut ∧ decoy out ∧ all still).

## Teleport-solution outline (solve.py)

Teleports are transport-only; every load-bearing interaction is contact dynamics or a
bounded applied wrench. The drawer is **never pose-written**; the black baton's pose is
written exactly **once** (the staging teleport into free air).

- **P0** — settle, readback asserts: drawer shut, both batons still on the roof.
  `SIM_GEN_SCORE 0.0`
- **P1** — bounded (≤10 N) horizontal force-PD on the drawer body (the wrench emulation
  of pulling the handle crossbar) pulls the slide out to 157 mm; drive removed; the
  springless slide holds. `SIM_GEN_SCORE 0.15`
- **P2** — the ONLY baton pose write: teleport to the staging pose (pitch 50°, low tip
  6 mm above the exposed drawer floor, in free air — verified non-penetrating), then a
  90-step wrench "grab" (force PD + gravity feedforward, cap 6 N; attitude PD,
  cap 0.3 N·m — the emulation of a pinch grasp). `SIM_GEN_SCORE 0.40`
- **P3** — the thread: 7 smoothstep waypoints take the held baton aft and down
  (50°→14°), the low tip riding the drawer floor through real contact; then a
  closed-loop final push (+0.4 mm/step on the *measured* aft-endpoint readback) until
  the deep endpoint passes drawer-local x = 0.225; wrench released; settle flat inside.
  `SIM_GEN_SCORE 0.70`
- **P4** — drawer force-PD pushes the slide SHUT (≤10 N); drive removed; settle;
  assert `success()`. `SIM_GEN_SCORE 1.0`
- **P5** — 3.3 sim-seconds hands-off persistence with success re-checks every interval,
  then `SIM_GEN_SOLVE: SUCCESS`.

Scores print at every phase boundary and are asserted non-decreasing. Verified on the
forge: seeds 0 and 1 (distinct layouts by readback: slot swap visible), both
`SIM_GEN_SOLVE: SUCCESS`.

## Embodiment argument (single Franka, parallel jaw, base at the world origin)

Everything lives in a 0.25–0.70 m annulus in front of the bay (shell front at
x = 0.42, mouth strip x ≈ 0.26–0.42, roof-top spawn slots x ≈ 0.55–0.59, all |y| ≤ 0.15).

- **Roof pick**: the batons rest on the flat roof top (z = 0.255) under open sky — a
  plain top-down pinch across the 30 mm square section (jaw span ~80 mm), 0.12 kg.
- **Drawer pull/shut**: the handle crossbar (14 mm section between two posts, proud of
  the face plate) is a pinch target; the pull is a straight −x drag of ≤10 N over
  157 mm. Shutting is a palm push on the face plate. The solve's bounded drawer wrench
  is exactly this interaction.
- **The thread is single-wrist-feasible**: grip the baton near its fore (outboard) end.
  Throughout the thread the fore endpoint stays in the exposed strip *in front of* the
  roof lip with open sky above it — only the far tip travels under the roof — so the
  wrist never enters the covered volume. At 50°→14° pitch the grip point traces a
  smooth arc, a standard 6-DoF Cartesian path. The final few mm of seating (fore
  endpoint at drawer-local x ≈ 0.03, still ~13 cm clear of the roof lip) is a
  fingertip push on the butt end, ≤6 N. The wrench servo's caps (6 N, 0.3 N·m) are
  well inside Franka wrist limits.
- No bimanual, no regrasp-in-cavity, no reach under the roof is ever required.

## Execution order declared

1. `scene.py` written first (geometry invariants checked numerically: mouth 157 mm <
   baton 200 mm; thread-curve roof margins ≥ 5 mm; shut gaps ≤ 6 mm ≪ 30 mm).
2. `solve.py` iterated on the forge to `SIM_GEN_SOLVE: SUCCESS` on seeds 0 and 1.
3. Rubric finalized against the observed solution trace (latches fire where designed).
4. `smoke.py` battery run on the forge: **ALL PASS 15/15** (frames.npz recorded).
5. `TASK.md` — this file; final clean runs with the unchanged package.

## Check list (smoke.py, 15 checks)

1. Settle: states finite, drawer shut, both batons resting on the roof, all still.
2. Score ~0 at reset, no success.
3. Randomization readback: roof-slot side flips across seeds AND black yaw range > 5°.
4. Randomization readback: per-slot spawn jitter spread > 4 mm.
5. Null policy (240 idle steps): score ~0, no success.
6. **Seed strategy A**: baton dropped flat, axis-aligned, over the fully-open mouth —
   settles bridging the rim, NOT inside, NOT success (endpoint readback printed).
7. **Seed strategy B**: same drop at the best diagonal yaw — still NOT inside.
8. Leaning near-miss (low end on the floor, upper on the rim): thread fires but NOT
   inside, NOT success, score ≤ 0.85.
9. Inside-but-open: baton flat inside the OPEN drawer — inside holds live but NOT
   success (shut clause load-bearing), 0.69 ≤ score ≤ 0.85.
10. Latched credit: yanking the enclosed baton back out leaves the score unchanged.
11. Decoy sealed in the SHUT drawer (black out): latched 0.70 crushed to ≤ 0.105.
12. Both batons sealed in the SHUT drawer: NOT success, score ≤ 0.105.
13. Crosswise interlock: baton laid across the rims, a REAL bounded closing push —
    inside never latches, the drawer verifiably moved (157 → 28 mm), never success.
14. Rejection audit: success() never True at any judged point in the battery.
15. Final no-NaN across all task objects.
