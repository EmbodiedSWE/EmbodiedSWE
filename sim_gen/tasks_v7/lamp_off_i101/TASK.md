# lamp_off_i101 — Cut the lamp's power at the outlet (childproof twist-lock unplug)

**Scene:** `simgen.twistlock_unplug` (`TwistlockUnplugScene`, robot="null")
**Seed:** `rlbench/lamp_off`
(`RoboVerse/roboverse_pack/tasks/rlbench/lamp_off.py`)

## Seed provenance and what changed

The RLBench seed is a one-shot poke: a lamp with a push button on its own body,
and the plan is **press the button on the target device** — a single free-space
reach-and-press, no mechanism, no discrimination, no failure mode beyond "missed
the button".

This task keeps the goal predicate ("the lamp ends up off") but replaces the
entire plan skeleton:

1. **The device is not the interface.** The lamp here has NO switch — its power
   is cut at the **outlet**, by unplugging it. Acting on the lamp itself is
   physically futile (it just slides across the desk; smoke check 8 shoves it
   15 N and it earns nothing). The seed's press-the-device schema transfers zero.
2. **The target must be *inferred*, not perceived directly.** The two-socket
   outlet holds two **identical** twist-lock plugs. Nothing on the plugs or
   sockets says which is which: the only discriminating evidence is the **cord**
   — a red cord runs from the lamp to its socket, a blue cord from a desk fan to
   the other. Which socket the lamp uses is sampled per episode, so the plan
   needs a cord-tracing step before any manipulation.
3. **A live keep-alive constraint on the decoy.** The fan must STAY powered:
   unplugging the fan's plug — even *in addition to* correctly unplugging the
   lamp — quarters all non-success credit and blocks success (smoke check 11).
   "Unplug everything" is not a workaround for skipping the identification step.
4. **The extraction is gated by a physical mechanism.** Each plug is a
   childproof twist-lock: a key bar behind the cap sits 3 mm behind a slotted
   faceplate. Straight pulling a seated plug JAMS on the plate by real collision
   (~3 mm of free travel, then metal-on-plastic; smoke check 6). The plug must
   first be **rotated to the episode's sampled slot angle** (±30–50°, sign and
   magnitude random) so the bar passes through the slot (~13.5° of tolerance,
   live clearance geometry — a 25°-off twist still jams, smoke check 7), and
   then pulled ≥ 40 mm along its axis while alignment is *held* — the contact
   normal at the slot is axial, so nothing self-aligns the bar en route.

So the plan skeleton changes from *"reach and press the target device"* to
*"trace a cord to identify the correct of two identical mechanisms, unlock it
with a rotation-then-translation sequence against real interlock geometry, and
leave the decoy untouched"* — identification + mechanism-gated extraction with a
negative constraint, instead of a poke.

## Why strategically different from every examined sibling

Corpus tasks examined (tasks_v7): charger insertion, push-button (×3), jar
close/open, wine-bottle open, nail screwing, peg insertion (×2), stove knob off,
egg-on-door fridge close, drop-gate release, log-cabin stacking, pin + tray,
crate inversion, shuttle metering, letterbox rack, dice tumbling, detent knob,
rammer gallery, oven door/dials. The closest neighbours differ structurally:

- *plug_charger_in_power_supply_i21*: **insertion** of a plug — one object, one
  receptacle, no discrimination, no mechanism gate, opposite direction. Here the
  hard parts are exactly what that task lacks: which plug (cord trace), the
  twist-lock gate, and the keep-alive decoy.
- *push_button_i6 / i35 / push_buttons_i40*: the seed's own strategy family.
  This task's smoke explicitly rejects the act-on-the-device schema (check 8).
- *close_jar_i54 / open_jar_i65 / screw_nail_i59*: rotation **is the goal**
  (continuous screwing to a depth/torque). Here rotation is a **key** — a small
  bounded twist whose only purpose is to gate a translation, and it must be
  *held* during that translation or the bar jams mid-slot.
- *peg_insertion_side_i1/i2*: geometric mating with no target selection and no
  constraint on a second object.
- *libero…turn_off_the_stove_i13*: a knob **on the device itself**; this task's
  point is that the device carries no interface at all.
- *close_fridge_i89*-style ordering tasks: order there is forced by containment;
  here the sequencing (twist, then pull, twist held) is forced by an interlock,
  and the decisive cognitive step is upstream — identification by cord tracing.

## Solution outline (solve.py — NOTHING is teleported)

All interaction flows through the live plant: the solver only writes the
scene's `drive_f` / `drive_t` buffers (the stand-in for the Franka's wrist roll
and straight pull on the grasped cap), applied in `post_step` as wrenches along
the plug's ONE free axis, so the drive never drifts with plug rotation. Because
wrenches take effect one substep late, both servos are sized for discrete
stability at dt = 1/120 s:

- **Phase 0 (reset):** settle 0.5 s, read the lamp's socket and sampled slot
  angle, assert drive buffers zero. `SIM_GEN_SCORE` ≈ 0.000.
- **Phase 1 (twist):** cascaded velocity servo — outer rate command
  ω_des = clamp(4·(α − twist), ±1 rad/s), inner τ = 3e-4·(ω_des − ω),
  |τ| ≤ 0.02 N·m. The plug's ~7e-6 kg·m² inertia makes any stiffness servo ring
  against the sampled loop; the rate cascade is discretely monotone
  (KW·dt/I ≈ 0.35 < 1) and parks the bar at the slot angle (either sign) without
  ringing. The bar never touches the plate during the twist (it sits 3 mm behind
  it) — alignment credit is earned by rotation alone. `SIM_GEN_SCORE` 0.250.
- **Phase 2 (pull):** velocity servo f = kv·(v_des − v),
  v_des = min(0.06, 3·(x_des − ext)), |f| ≤ 6 N, kv = 2.0 (kv·dt/m = 0.28 —
  kv = 8 rang bang-bang at ±0.35 m/s against the one-substep delay), while the
  twist servo keeps holding alignment through the slot. If progress stalls the
  GAIN escalates (×1.5 up to 3.5), not the cap. The drive is released only past
  ext_goal + 5 mm **and** slow (|v| < 0.02 m/s — never release mid-swing); body
  damping parks the plug. The fan's plug is never touched. `SIM_GEN_SCORE` 1.000.
- **Phase 3 (persistence):** drive buffers asserted zero, ≥ 3.5 simulated
  seconds hands-off; `success()` is live state. Only then `SIM_GEN_SOLVE:
  SUCCESS`.

Verified on the forge: seeds 0 (socket 1, slot +45.8°), 1 (socket 1, −34.6°)
and 5 (socket 0, +36.9°) all print the monotone sequence
0.000 → 0.250 → 1.000 → 1.000 and `SIM_GEN_SOLVE: SUCCESS`.

**Declared abstraction:** each plug rides a D6 joint with one free translation
(0…110 mm) and one free rotation (±60°) — the plug is "leashed", never a fully
free body after extraction. The generous 110 mm limit keeps the joint bound far
from the 40 mm goal so no limit-contactDistance impulses pollute the release.

## Rubric (score 0..1, latched; score == 1.0 iff success())

- `align_latch` (0.25): best angular alignment of the LAMP plug's bar with its
  sampled slot (1 at aligned, 0 beyond 2× tolerance).
- `pass_latch` (0.15): the bar has ever physically cleared the faceplate.
- `ext_latch` (0.30): best extraction fraction (ext/ext_goal) of the lamp plug.
- current success (0.30): lamp plug out ≥ 40 mm AND fan plug still seated AND
  both settled.
- The whole non-success part is **quartered while the fan's plug is unseated** —
  the keep-alive constraint bites the score, not just success.
- Latches are transient-achievement credit (NaN-safe via torch.maximum); credit
  only evaporates under incorrect behavior (unseating the fan). Null policy ≈ 0.

## Embodiment argument (single Franka, parallel jaw, OSC)

Base at ≈ (0.30, 0.0) on the desk (top z = 0.40), facing −x toward the outlet
face at x = −0.25; sockets at y = ∓0.07, z = 0.51 — everything is inside a
0.75 m reach disc at comfortable elbow height.

- **Cord tracing:** purely visual — the red/blue cords run in ~8 catenary
  segments across the desk from appliance to socket riser; a wrist-camera sweep
  along either cord resolves the target without touching anything.
- **Plug cap (Ø 26 mm × 22 mm knurled cylinder):** a textbook parallel-jaw
  pinch grasp for the 80 mm jaw, approached along −x with the wrist axis
  colinear with the plug axis.
- **Twist:** pure wrist-roll (joint 7) of the grasped cap; the 0.02 N·m torque
  cap in solve.py is fingertip-scale — it only needs to beat ~4e-5 N·m of
  damping torque. Slot angles (±30–50°) are well inside the ±166° wrist range.
- **Pull:** a straight −x Cartesian retreat of ≤ 50 mm at ≤ 0.06 m/s under the
  6 N cap — a light OSC impedance pull; alignment hold during the pull is the
  same wrist-roll servo. This grasp-twist-pull is exactly what the drive-buffer
  wrenches stand in for.
- **Keep-alive:** satisfied by *not* acting on the fan's plug — correct target
  selection, no extra dexterity.

## Checks (smoke.py — rejection battery, 14 checks)

1. Clean reset: finite state, both plugs seated & untwisted, score ~0.
2. Readback: faceplates physically posed at the sampled slot angles (< 1.5 mm).
3. Cord routing readback: lamp cord lands at the lamp's socket, fan at the
   other; appliances on opposite y-bands.
4. Randomization is real (8 seeds): both sockets drawn, ≥ 4 distinct slot
   magnitudes, both signs, ≥ 4 appliance positions.
5. Null policy: 2 s of no action, score < 0.05.
6. **Seed-strategy family (locked pull):** an untwisted 1.5 N yank moves the
   plug ~3 mm then JAMS on the plate by real collision — ext < pass threshold,
   `pass_latch` 0, score < 0.10.
7. Mis-twist: rotated but 25° OFF the slot — still jams, no pass credit.
8. **Seed-strategy rejection (act on the device):** shoving the LAMP with 15 N
   slides it ≥ 10 mm and earns nothing (score < 0.05).
9. Near miss: aligned pull landed at ~30 mm (< goal) — partial credit
   (0.30 < score < 0.90), no success.
10. Wrong plug: the full twist-and-pull strategy executed on the FAN's plug —
    ~0 score, no success.
11. Keep-alive: lamp correctly out but fan unplugged too — rejected, non-success
    credit quartered (score ≈ 0.175).
12. Exactness: full correct strategy → success() and |score − 1.0| < 1e−3.
13. Achievement latch: reseating the lamp plug revokes success; latched 0.70
    remains.
14. Camera: ≥ 20 rgb frames captured across the checks → `frames.npz`.
