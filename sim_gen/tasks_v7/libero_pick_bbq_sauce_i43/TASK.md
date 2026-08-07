# cellar_tow — free the bottle from the cellar, then stand it on the pad

Env id: `simgen.cellar_tow` · Scene: `cellar_tow` · Robot: `null` (scripted physics solution)

## Seed provenance

Derived from `libero/libero_pick_bbq_sauce` — *"pick up the bbq sauce and place it in the basket"*.
What the seed tests: identify a target bottle among visually similar clutter, prehensile pick,
transport, containment placement.

What is kept from the seed:

- A target **bottle** that must be identified against a same-shape distractor (brown vs. blue).
- A prehensile **pick-and-place** finish: the bottle ends standing on a designated goal region.
- Clutter discipline: the distractor must not be disturbed (≤ 5 cm drift, must stay upright).

## What changed, and why it is strategically different

The seed (and the corpus tasks around it) let the manipulator reach the target directly: the
challenge is grasp selection and placement. Here **direct access is physically impossible at
reset** and the solver must *construct a tool coupling between two free bodies* to create access:

1. The bottle starts cradled on a free-sliding **sled** parked under a low **canopy** ("cellar").
   The roof underside is 11.5 cm up; cradle walls + bottle stand 14.0 cm — a geometric guarantee
   (asserted in code) that the bottle cannot be lifted out while the sled is under the roof.
2. The only access is a 24 mm **slot** through the roof, over an open-topped 32 mm **socket tube**
   on the sled. The solver must pick up a loose 30 cm **tow probe** from the open floor, thread it
   down through the slot into the socket (peg-in-hole through an intermediate aperture), then
   **tow**: dragging the handle along the slot pulls the coupled sled — cargo and all — out from
   under the canopy. Force is transmitted purely through probe-wall / socket-wall contact.
3. The coupling must then be **unmade** (probe withdrawn from the socket and set aside) before the
   ordinary pick-and-place finish is allowed to count.

Nothing in the corpus builds a *temporary* rigid coupling between two initially separate free
bodies and then dissolves it again: corpus mechanisms are pre-existing (hinges, dials, captive
shuttles, tracks) — here the "mechanism" itself is assembled by the solver out of loose parts,
used, and disassembled. Relative to the seed, "pick up the sauce" becomes the *last* 20% of the
task; the first 80% is tool-mediated extraction under a hard geometric occlusion.

## Scene (procedural geometry only)

- **Canopy** (dark, 40 kg dynamic fixture): two roof plates on four corner legs, roof top 13.5 cm,
  underside 11.5 cm, full-length 24 mm slot between the plates along local +x.
- **Sled** (yellow, free-sliding): deck plate carrying a walled cradle pocket (bottle seat) and an
  open-topped orange square socket tube (32 mm bore, 8.0 cm tall) at its front, under the slot.
- **Tow probe**: 30 cm steel rod, red T-handle (14 mm grip bar), lying loose on the floor.
- **Brown bottle** (target, 5.0 cm across, 8.5 cm tall) upright in the cradle; **blue decoy**
  bottle of identical shape loose on the floor; flat **green goal pad** (12 cm) on the floor.
- Per-seed randomization (verified by state readback in smoke): canopy xy jitter ±3 cm and yaw
  ±15°, sled park depth x ∈ [−0.15, −0.04] m, probe / decoy / pad placed on world arcs with
  randomized angle and radius. Arc bounds are asserted clear of the +x tow corridor.

## Teleport solution (solve.py) — teleport is transport only

- **P0** settle + layout readback; assert bottle cradled, sled not clear, score ≈ 0.
- **P1** teleport-carry the probe to a hover over the roof slot, tip above the tube (transport of
  a free body the gripper would be holding; released with zero velocity).
- **P2** contact-guided descent: xy servo + damped downforce threads the shaft through the slot
  into the socket. Engagement is detected from state readback; the external-force frame mode is
  probed at runtime and locked (see memory: rotated-wrench quirk).
- **P3** tow: waypoint slides along the canopy's local +x at 10 cm/s; the applied wrench acts on
  the **probe only** — the sled moves exclusively through peg/socket contact, cargo riding its
  cradle, until the sled is fully clear (sled_x ≥ 0.47 m). Hands-off coast to rest.
- **P4** retract: upward force pulls the shaft out of the socket through contact; the freed probe
  is then teleport-carried aside and laid down.
- **P5** teleport-carry the bottle from the cradle to 8 mm above the pad; gravity lands it.
  Hands-off persistence 400 steps (3.3 s) with success held, then `SIM_GEN_SOLVE: SUCCESS`.

Verified on forge seeds 0, 1, 2 — score trace 0.00 → 0.15 → 0.55 → 0.70 → 1.00, monotone,
success persists ≥ 3.3 s hands-off.

## Rubric (anchored in the demonstrated solve)

Latched milestones (updated only in `score()`, non-decreasing, in `get_state`/`set_state`):
engaged (0.15) → tow fraction (0.25·frac) → sled extracted with cargo cradled (0.15) → probe
disengaged after extraction (0.15) → success (1.0). `success()` requires ALL of: bottle upright
(≤ 10°) centered on the pad (≤ 3 cm, correct height band) and settled; sled fully clear of the
canopy (≥ 0.44 m local x) and bottle off the sled; probe out of the socket and clear; decoy
upright and within 5 cm of spawn; extraction/disengage latches set in order (so a state that
merely *looks* final but skipped the tow is rejected).

## Embodiment argument (Franka, 80 mm parallel jaw)

- The probe's T-handle grip bar is 14 mm diameter — a natural full-closure pinch; the 30 cm shaft
  gives the wrist standoff to reach over the 13.5 cm roof and push/pull along the slot.
- The bottle is 5.0 cm across — comfortably inside the 80 mm jaw span for the final pick.
- Tow forces are ≤ 5 N horizontal, well inside Franka payload; the slot runs at table height
  (≤ 13.5 cm), so a base posted at the open (+x) side of the plaza reaches hover, insertion,
  the full tow stroke, retraction, and the pad without exceeding ~85 cm reach.

## Execution order is physically forced

Cradle walls + bottle (14.0 cm) > roof underside (11.5 cm), asserted in `__post_init__`: the
bottle cannot leave the cradle while the sled is under the canopy — extraction MUST precede the
pick. Success additionally gates on the ordered latches (engage → extract → disengage), so
bypasses (placing a look-alike, sliding the loaded sled onto the pad, leaving the probe in the
socket) are rejected — each of these is constructed and asserted rejected in smoke.py.

## Checks (smoke.py, 20)

1 settle/no-NaN · 2 baseline score ≈ 0 · 3–5 randomization differs across seeds via readback
(canopy pose, sled depth, arc props) · 6 null policy 300 steps ≈ 0 · 7 coupling is real (probe
seated in socket, velocity-driven along the slot, drags the sled 28 cm with cargo cradled) ·
8 cellar blocks escape (kicked
bottle stays under the canopy footprint) · 9 wrong object (decoy on pad) rejected · 10 cellar
bypass (bottle on pad, sled still parked) rejected · 11 acceptance state accepted · 12/13 probe
re-engaged flips success off / removed flips back on · 14/15 decoy disturbed flips off / restored
flips on · 16 near-miss (bottle upright on the ground 11 cm off the pad) rejected · 17 unsettled
placement rejected ·
18 loaded-sled-on-pad (bottle still cradled) rejected · 19 no spurious success ever observed ·
20 final finite-state audit. Frames recorded to `frames.npz`.
