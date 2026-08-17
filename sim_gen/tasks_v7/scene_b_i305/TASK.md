# scene_b_i305 — marble router: write the routing program, then commit

## Seed provenance

Seed task: `calvin/scene_B` — the CALVIN play-table B: a Franka at a desk with
four independent articulated single-DOF primitives (`base__button`,
`base__switch`, `base__slide`, `base__drawer`) and three colored blocks on an
open tabletop. Every CALVIN-B objective is either "actuate one single-DOF
primitive to its other end-stop" or "pick / push / lift a block in free
space". Carried over here: a desk-scale scene whose interaction vocabulary is
exactly CALVIN's — throw small hinged flippers between their end-stops
(switch), slide a handle along its prismatic axis to its stop (slider/drawer)
— plus a color/marker cue for goal identification.

## Strategic difference

**vs. the seed.** In CALVIN every DOF is a TERMINAL goal: push it to the
other end-stop and you are done, in any order, reversibly, judged on the DOF
itself. Here every DOF the robot may touch is purely INSTRUMENTAL and none is
judged: the goal is a place-outcome of a body the arm can never touch — a
40 mm marble sealed inside a roofed, 10°-pitched gravity switchyard. The two
railroad-switch fins on the route (revolutes between hard stops, exactly
CALVIN's "switch" motor primitive) form a **2-bit routing program that must
be written FIRST**; the sliding gate (CALVIN's "slider" primitive) is a
**commit**: pulling it is irreversible — the marble outruns any intervention
and lands wherever the fins pointed. The seed's reflex — yank the slider to
its end-stop first — permanently fails the episode with ~0 score. No CALVIN
objective conditions one DOF's meaning on another, none is irreversible, and
none is judged by where a never-touched object comes to rest.

**vs. the rest of the corpus.** `scene_c_i171` (read in full as the
structural template) is mass arithmetic judged by the equilibrium ANGLE of an
unactuated beam — nothing there is sequenced or irreversible, and the loaded
cubes are the judged objects. `scene_d_i130` is aperture-gated extraction
(rotate a hidden payload under a window, lift it out). `libero_..._i87/_i88`
are drawer tasks (tool-mediated or interlocked). `pen_holder` (robobench
exemplar) is free-space pick-and-insert. No other task read has: a routing
PROGRAM written into mechanism state that a later irreversible release
executes, an untouchable judged payload, or an order-aware credit chain where
acting out of order forfeits everything. There is no stored energy: fins are
plain damped revolutes between stops (the board's own tilt presses them into
whichever stop they are near — gravity-bistable, no springs), the gate a
damped gravity-neutral prismatic; every intermediate configuration persists
hands-off.

## Scene (`marble_router`, env `simgen.marble_router`, robot="null")

Procedural geometry only. One KINEMATIC compound rig — 100 cm board pitched
10° on four legs, walled channels under a grey roof: start chute with a
both-walls gate slot, junction A forking into two junction Bs, four parallel
roofed BINS with narrow viewing slits and colored flags (crimson/green/blue/
amber). Three DYNAMIC fins (orange blade + yellow post rising through a roof
slot) on bind-time per-env USD revolute joints about the board normal, hard
stops ±ψ (≈15°); one DYNAMIC teal gate (plate in the slot + yellow knob
outside the chute wall) on a prismatic joint, travel 0→10 cm laterally. Joint
collision filtering applies only to each jointed pair, so every marble
contact stays live. A 40 mm / 60 g marble; a white kinematic marker tile on
the ground past the low end, lined up with the target bin.

Geometry contract (~30 asserts in `__post_init__`): every roof hole, slit and
sealed gap keeps its minimum dimension < the 40 mm marble (a rectangle passes
a sphere only if BOTH sides exceed the diameter) while every intended passage
keeps ≥ 48 mm; anti-tunneling v_max·dt ≤ wall_t + contact_offset; fin blades
clear the flare walls at both stops by ≥ 2 mm (no prop masking the stop);
gravity torque presses an upstream-pointing blade into its stop (bistable,
τ_g > 0); the marble's wall-rest point sits inside the success box; the
gate plate at full open clears the channel; rubric weights sum to the cap.

Per-episode randomization (readback-verified): target bin t ∈ 0..3, BOTH
path fins written at their WRONG stops (off-path fin at a random stop), gate
shut, marble start jitter, marker lateral jitter.

**Success** (live state, no memory): the marble at rest (|v| < 0.06 m/s)
inside the TARGET bin's board-frame box (x 0.870..0.975, |y − y_bin| < 0.030,
z 0.006..0.045), finite. The yard is sealed — the marble can only be there
via the forks the fins spelled. **Score**: latched partial credit, ORDER-
AWARE chain — `routed` 0.20 (both path fins correct while the gate is still
shut and the marble still waits), `released` 0.20 (requires `routed` already
latched; gate past the open threshold with fins correct), `branched` 0.15
(requires `released`; marble crosses junction A into the correct half),
capped at 0.55; exactly 1.0 iff success. Gate-first play collects nothing,
however it ends. Latches clear on reset.

## Solution (`solve.py`) — applied wrenches only, ZERO teleports

1. **P0 settle + perception**: 120 steps; read t, half, and the fin start
   angles back from the LIVE state; `get_masses()` readback on ball, all
   three fins, and the gate guards the density-mass trap; assert the marble
   waits behind the shut gate and both path fins rest at their WRONG stops;
   score ≈ 0.
2. **P1 write the program**: PD torque about each path fin's own hinge axis
   (body z = board normal, clamp 8 mN·m — a fingertip on the post) throws
   fin A to half·ψ_A and the path fin B to its correct stop; torque off,
   gravity seats each on its stop. `routed` fires (0.20).
3. **P2 commit**: PD force along the gate's prismatic axis (body y, clamp
   4 N — a pinch on the knob) pulls it to 9.5 cm; `released` fires (0.40);
   the marble rolls out; force off once it is 30 cm downboard.
4. **P3 hands off**: the marble rolls junction A (`branched`, 0.55) then B
   and stops against the end wall inside the marked bin; success() turns
   True live (1.0) and holds through a **3.3 s hands-off persistence**
   window before `SIM_GEN_SOLVE: SUCCESS`.

Repeats on a second seed (fresh reset; SIM_GEN_SCORE non-decreasing). Forge:
both seeds OK (bins 3 and 2), rc=0, ~21 s — **first submission, no
iteration**.

## Franka embodiment argument

Base pose: one fixed base beside the yard at ≈ (0.45, +0.55), facing the
board's +y flank — the gate knob (protruding from the +y chute wall at world
y ≈ +0.26, pull direction +y i.e. TOWARD the robot), both fin posts (board
x 0.46–0.68, world z ≈ 0.40, reaching over the roofline from above), and the
bin roof slits are inside a 0.30–0.75 m reach annulus. The marble is never
touched.

- **Fin posts** (10 mm square, rising through 48 mm-wide roof slots, ≈ 25 mm
  proud of the roof): canonical fingertip push or two-finger pinch; the full
  throw is a 28 mm lateral arc at the post — one short push to the far side
  of the slot; the slot walls and hard stops make the action discrete and
  self-terminating; gravity holds the result, so no regrasp precision.
- **Gate knob** (30 mm cube on a stem outside the chute wall, in free air
  above open unroofed chute): parallel-jaw pinch (30 mm < 80 mm jaw) and a
  10 cm lateral pull along the slot — a drawer-pull, CALVIN's own slider
  skill; the prismatic rails take all misalignment, and the required force
  is < 4 N.
- **Perception**: the target is indicated by a 70 mm white tile on the
  ground lined up with the bin mouth, the bins are flagged in four distinct
  colors, roof slits show the marble's final lane, and the orange blades +
  yellow posts make fin state legible from above — all shoulder-camera
  visible; nothing requires seeing through the roof.
- **No interaction with the marble**: the rubric's sealed-yard + hands-off
  delivery means the arm only ever touches the two posts and the knob — three
  discrete, sturdy, purpose-built handles in open space.

## Execution order declaration

The scene was designed first with the minimal success predicate and the full
geometry contract; `solve.py` ran on the forge and passed both seeds on the
**first submission** (rc=0, ~21 s), empirically confirming the fin→branch
sign convention. The rubric was then finalized by STRENGTHENING the latch
chain (released requires routed; branched requires released and mid-board
position) so out-of-order play collects nothing — the solve is unaffected
(its phases fire the chain in order). Only then was `smoke.py` written; its
first forge run scored **15/15** with no fixes. No check was weakened at any
point to make a run pass.

## Smoke battery (`smoke.py`) — 15 checks, all rejection/health

1. **settle/no-NaN** — marble waits behind the shut gate, path fins wrong,
   score ≈ 0, no success.
2. **randomization A** — target bin varies (≥3 distinct in 10 seeded
   resets) and the live marker tile tracks the sampled bin every time.
3. **randomization B** — marble start varies (std > 8 mm), the off-path fin
   takes both stops across seeds, live fins/marble track the sample.
4. **null policy** — 300 idle steps: marble still waiting, gate shut,
   score ≈ 0.
5. **seed reflex** — PHYSICAL: the gate really pulled open with the fins
   unset (the CALVIN slider move); the marble really rolls out — into the
   WRONG half; no latch fires, score ≈ 0, irreversibly failed.
6. **one fin wrong** — fin A really thrown (asserted), path B left wrong;
   gate pulled: marble takes the correct half but the SIBLING bin; no
   credit, no success.
7. **out of order** — gate first, fins thrown correctly AFTER: fins_correct
   holds with the gate open, but the order-aware chain refuses all three
   latches; score ≈ 0.
8. **wrong-bin rest** — marble constructed at rest in the sibling bin: in a
   bin box in x/z, only the bin identity fails; no success, also after
   settling against the end wall.
9. **short of the box** — at rest in the TARGET lane at x = 0.84 < x_lo:
   the x clause refuses (judged without stepping).
10. **rolling through** — INSIDE the target box at 1.2 m/s: the settle gate
    refuses (judged without stepping).
11. **on-roof cheat** — at rest on the roof directly over the target bin:
    the board-frame z clause refuses (judged without stepping).
12. **ground-by-marker** — resting on the marker tile outside the yard (the
    world-frame look-alike): board-frame x/z refuse, also after 20 steps.
13. **rejection audit** — success() observed False at every step of the
    entire battery.
14. **final no-NaN.**
15. **video** — frames.npz (243 rgb frames, 960×600) written to CWD.

Forge result: `SIM_GEN_SMOKE: ALL PASS 15/15` (rc=0, ~62 s). The solve was
re-run clean on the final submission after the latch-chain strengthening.
