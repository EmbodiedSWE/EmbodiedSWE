# living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i18 — WedgeHopperScene (`simgen.wedge_hopper`)

Drain an amber ball out of a SEALED, back-tilted hopper into a walled basin — without
ever touching the ball — by driving a blue wedge along a guided runway under the
hopper's raised back edge: the inclined plane converts the horizontal push into lift
under load, the hopper pitches past level about its pivot bar, the wedge SELF-LOCKS by
friction, and gravity rolls the payload out of the mouth, over a deflector, into the
basin. The payload is untouchable by design; the only route to the goal is actuating a
force machine.

## Seed provenance

- **Seed task**: `libero_90/living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray`
  (RoboVerse libero_90) — "pick up the salad dressing and put it in the tray". A direct
  prehensile transport: identify the target bottle among distractors, grasp it, carry
  it through free space, release it inside a tray's containment bbox. One
  grasp-carry-release, no mechanism, the hand touches the payload the whole way.

## What the task is

A kinematic gray RIG stands on the floor (per episode its yaw swings ±30° and its
foot jitters ±40 mm — the push axis must be READ from the scene, not memorized). The
rig carries, in its own frame (+x toward the basin):

- a **cradle pedestal** (top 0.100 m) with bar stops — the front support;
- a **low-friction runway** (μ≈0.15, top 0.093 m, 0.640 m long) with guide rails
  (±4 mm lateral slack) — a 1-DOF push channel that abuts the pedestal;
- a **walled basin** (interior x ∈ [0.025, 0.205], |y| < 0.088, walls 55 mm) fronted
  by a 46° deflector.

A free dynamic **HOPPER** (264 × 180 mm floor, sealed by a roof; mouth aperture
164 × 70 mm at the front; mass 0.50 kg) rests bridged across the rig: its pivot bar
sits in the cradle slot, its two 10 mm feet stand on the runway, holding it ~2° BACK
tilted — the Ø40 mm, 60 g BALL inside parks against the back wall (lateral start
±50 mm randomized) and the null policy drains nothing. On the floor, scattered in a
0.22–0.62 m annulus with free yaw and keep-out resampling: the blue **WEDGE**
(130 mm long, 6 → 62 mm, 23.3°, 250 g; high-friction top μ≈0.80, low-friction
bottom) and a red **SHIM** (130 × 60 × 9 mm, 150 g — the decoy).

Goal: the ball settled INSIDE the basin, having drained THROUGH THE HOPPER'S MOUTH.

## Why strategically different

- **vs. the seed**: the seed's whole skill is *touching the payload* — grasp, carry,
  release into a container. Here the payload is UNTOUCHABLE BY DESIGN: the roof and
  the deep mouth leave no grasp line to the ball, so "pick it up and put it in" is
  physically impossible. The seed's plan is constructed verbatim in smoke 4 — the
  ball teleported straight into the basin, where it genuinely settles — and the
  pathway latch refuses it: no `_via_mouth`, score ≤ 0.21, never success. The
  deliverable is not a delivery; it is an ACTUATED MECHANISM STATE plus the drain it
  causes.
- **The physics is load-bearing, not rubric fiat**: (1) the 10 mm foot gap admits
  only the 6 mm wedge tip — lifting requires the inclined plane, under load (hopper
  + ball weight carried through the wedge contact); (2) the wedge self-locks because
  avg top friction 0.70 > tan 23.3° = 0.43 — the tilt survives hands-off, asserted
  in solve P2 with a 60-step force-free hold; (3) the ~2° back-tilt containment is
  live: with no wedge (smoke 3) or a ball placed at the mouth lip (smoke 7) the ball
  rolls BACK inside — gravity actively opposes the goal until the mechanism is
  actuated; (4) drain is irreversible (deflector + 55 mm walls), so the end state is
  a settled equilibrium, not a held pose.
- **The decoy is physically self-rejecting**: the 9 mm shim is thinner than the
  10 mm gap — staged and pushed hard through the channel (smoke 6) it slides clean
  under the raised edge and lifts NOTHING: tilt never passes −0.6°, the ball stays
  inside, score ≤ 0.01.
- **vs. corpus tasks read**: no die-tipping (`approach_grasp_spoon_i12`), no
  log-cabin crib (`close_grill_i8`), no gravity-gate drop (`close_microwave_i4`),
  no bayonet twist (`close_microwave_i5`, `libero..._i2`), no beam balance / mass
  select (`coke_task_i15`, `..._i14`), no carousel sweep (`..._i9`), no
  pour-then-invert-park (`..._i19`), no kettle-snuffer flip (`..._i3/_i13`), no
  drawer slide-load-close (`..._i11`), no pin-topple tunnel shot (`obstacle_i17`),
  no dice tumbling (`open_oven_i6`), no ramrod ejection (`peg_insertion_side_i1`),
  no bar-eye pin drop (`peg_insertion_side_i2`), no seesaw hoist
  (`pick_and_lift_i16`), no pin-unlock shuttle (`pick_single_egad_i3`), no
  thread-and-hang (`pick_single_egad_i4`), no ramp-chock equilibrium
  (`pour_water_i7`), no silo drop sequence (`setup_checkers_i2`). *A self-locking
  wedge jack — an inclined-plane machine that lifts a loaded body so an untouchable
  payload drains by gravity* — appears in none of them.

A solver therefore needs a different PLAN each episode (read the rig heading, stage
the wedge in the channel, push along the rig's −x→+x axis while monitoring pitch,
stop when self-locked past +5°, wait out the drain) and different CODE STRUCTURE (a
rig-frame push-axis controller with speed cap and stall escalation, a hopper-frame
mouth-window pathway latch, tilt telemetry) — not a grasp-carry-release routine.

## Solution outline (solve.py = the legitimacy certificate)

1. **P0** settle 1 s; READBACK the layout (rig xy + yaw, wedge/shim/ball spawns);
   assert ball in hopper, resting tilt ∈ (−4.5°, −0.6°); score 0, no success.
2. **P1** (teleport = transport only): one root-state write carries the wedge from
   the floor to a FREE-SPACE hover 2 mm above the runway at rig (−0.31, 0), tip
   toward the hopper — the pose a gripper would release from. It falls onto the
   runway between the rails. Staged latch → score 0.10.
3. **P2** (contact dynamics does ALL the work): a per-step speed-capped push
   (start 1 N along rig +x, force cleared whenever |v| > 0.10 m/s, stall-only ×1.4
   escalation capped at 14 N, working-zone aborts, re-stage retries) drives the
   wedge under the back edge until drain tilt ≥ +6°. Forces off, 60 steps: the
   wedge SELF-LOCKS, tilt holds. Gravity drains the ball through the mouth, over
   the deflector, into the basin (up to 1200-step wait; one mid-wait re-push to
   ≤ 9° if the ball stalls inside). Mouth + basin latches → score 1.0 live.
4. **P3** settle 0.75 s; success still live; score 1.0.
5. **P4** hands-off persistence 3.33 s; success holds → `SIM_GEN_SOLVE: SUCCESS`.

The ball is never written to, pushed, or touched: its entire trajectory — rolling
off the back wall, crossing the mouth window, dropping onto the deflector, settling
in the basin — is an outcome of the tilt the wedge produced. Monotone
`SIM_GEN_SCORE`: 0.0000 → 0.1000 → 1.0000 → 1.0000 → 1.0000.

## Rubric

- 0 → 0.75 latched shaping (never decays): 0.10 wedge ever staged in the channel +
  0.15 hopper ever pitched past +1.5° + 0.15 ever past +5.0° + 0.15 ball ever
  through the mouth window (hopper-frame) + 0.20 ball ever inside the basin
  (cap 0.75).
- 1.00: success — live: ball settled inside the basin (rig-frame containment
  x ∈ (0.025, 0.205), |y| < 0.088, resting-height window, velocity gates
  |v| < 0.05 m/s, |ω| < 2.5 rad/s) AND the drain pathway went through the mouth
  (`_via_mouth` latch) AND all states finite.

Unfakeable without the mechanic: a ball delivered into the basin any other way
fails the pathway latch (smoke 4), and nothing but the wedge can produce the tilt
(smoke 3, 6).

## Franka embodiment (single arm, parallel jaw 80 mm, OSC)

Proposed base pose: **rig-frame (−0.42, −0.45, 0.00), facing the runway** (nominal
reach 0.855 m). The wedge/shim spawn annulus (0.22–0.62 m about the rig origin) and
the whole push channel (rig x ∈ [−0.67, −0.15] at height 0.093 m) sit inside the
envelope; all manipulation happens below 0.16 m.

- **Wedge** (130 × 60 mm, tallest face 62 mm, 250 g): pinch across the 60 mm width
  or the 50 mm back block — both ≪ 80 mm jaw stroke; the flat back block gives a
  square contact pad. Staging is a lay-down into the runway channel; the rails'
  ±4 mm slack and the 0.16 m staging window are coarse by placement standards.
- **The push**: close the jaw on (or press the closed jaw against) the wedge's back
  face and slide along the channel — the rails make it 1-DOF, the runway's μ≈0.15
  keeps the required force ≈ 1–5 N (solve's controller caps at 14 N for worst-case
  stalls), well inside Franka's ≈ 90 N envelope, at a comfortable 0.093 m height.
  Progress is verifiable from wrench feedback and the hopper's visible pitch.
- **Payload untouched**: the ball is sealed away and the shim need never be
  touched; incidental contact with the kinematic rig cannot move the goal frame.
- **Ordering** (declared): stage the wedge, then push — the order is
  mechanism-enforced (pushing nothing lifts nothing; the gap admits only the tip
  first). The ball's drain follows by gravity alone.
- **Forces**: heaviest lift 250 g; push ≤ 14 N horizontal.

## Validation evidence (forge, RTX 4090, Isaac Sim 5.1)

- `solve --seed 0`: SUCCESS, scores 0.0000/0.1000/1.0000/1.0000/1.0000, self-locked
  at +6.1°, wedge parked at rig x = −0.160 (matching the analytic 91 mm-stroke
  prediction), ~24 s wall.
- `solve --seed 3` (rig yaw differs; ball starts 38 mm off-center): SUCCESS, same
  monotone scores, self-locked at +6.0°, ~24 s.
- `smoke`: **SIM_GEN_SMOKE: ALL PASS 10/10** (35.1 s), frames.npz saved:
  1. settle/no-NaN: hopper seated back-tilted (−4.5° < tilt < −0.6°), ball parked
     inside, wedge + shim on the ground outside; score ≤ 0.01
  2. randomization readback: rig yaw, rig xy, wedge spawn, ball world position all
     differ across seeds 101/202
  3. null policy: 240 idle steps, back-tilt containment holds, score ≤ 0.01, no
     success
  4. SEED STRATEGY / bypass: ball teleported directly into the basin — genuinely
     settles there, but `_via_mouth` never latched: score ≤ 0.21, NO success (the
     seed's grasp-carry-drop plan is worth at most partial credit, never success)
  5. wedge staged only: score 0.09–0.11, hopper still back-tilted, no success
  6. SHIM decoy: staged and pushed through the channel — slides clean under the
     10 mm gap, lifts nothing: tilt never passes −0.6°, ball stays inside,
     no lift credit, score ≤ 0.01
  7. mouth-only near-miss: ball placed at the mouth lip of the back-tilted hopper
     rolls BACK inside (containment is live); `_via_mouth` latches but score
     ≤ 0.16, no success
  8. outside-basin near-miss: ball settled on the rig outside the basin walls —
     a REAL settled state, rejected for place: score ≤ 0.01, no success
  9. rejection audit: success never fired during checks 3–8
  10. video frames.npz saved ((81, 600, 960, 3))

## Files

- `scene.py` — WedgeHopperScene + kinematic rig / dynamic hopper / wedge / shim
  compound spawners + rubric; registers `simgen.wedge_hopper`.
- `solve.py` — teleport-stage + speed-capped contact push certificate (`--seed N`).
- `smoke.py` — 10-check rejection battery + video.

Run (forge):
`python -u -m simgen_tasks.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i18.solve --headless [--seed N]`
`python -u -m simgen_tasks.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray_i18.smoke --headless`
