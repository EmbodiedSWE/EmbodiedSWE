# open_box_i184 — Roll-Away Vault

**Scene name:** `roll_away_vault` · **Env:** `simgen.roll_away_vault` (robot="null")

## Seed provenance

Seed task: `rlbench/open_box` — "open the box": an articulated box USD stands with
its hinged lid shut; the robot swings the lid open about its BUILT hinge, judged by
the lid joint angle. One pulling/pushing contact on a hinged panel, ending in a
joint readout.

## Strategic difference

The seed's closure model is replaced wholesale: **there is no joint anywhere in the
scene** — nothing articulated, no hinge to swing, no joint angle to read (the seed's
literal end state is inexpressible here; documented N/A in smoke check 6). The box's
only opening is a slot in the top plate, sealed by a free, heavy **crimson roller**
(Ø100 mm — asserted wider than the 80 mm jaw, so it can never be picked, only
ROLLED). "Opening the box" is a **nonprehensile act with a real force threshold**:
push the roller over a 4 mm detent lip (~5.9 N quasi-static, asserted arm-scale in
`__post_init__`; seed 1's solve run measured the break between 5 and 6 N with no
run-up) and the scene finishes the opening **by itself** — the roller self-rolls
down a fenced 25° ramp into a walled ground dock. The opening is then consumed: the
gold prize cube is lifted out through the uncovered slot onto a pad while a grey
distractor stays inside. The seal is physics, not fiat: the largest crescent
clearance under the seated roller is 20 mm < the 40 mm cube (asserted), and smoke
check 5 shoves the prize upward at 3× its weight for 2 s — it demonstrably rises
30 mm into the slot and is stopped by the roller.

Differs from every corpus task read: `close_box_i26` *manufactures* a closure by
inverting a crate and slides it under a lift-forbidden constraint; `close_microwave_i4`
*releases* a prop so gravity drops a gate; `close_laptop_lid_i152` loads ballast into
a tray on a hinged panel; the libero/rlbench pick-place family transports grasped
objects freely. **No corpus task removes a gravity-seated closure by a
threshold-gated nonprehensile roll whose completion is handed off to the scene's own
geometry (ramp + fences + dock)** — and none has retrieval gated on that removal.

## Assets (fully procedural)

- **vault** — KINEMATIC compound (170×240 mm footprint, 95 mm tall): raised plinth
  floor (cavity floor at z 30 mm keeps the grasp shallow), 4 walls, top plate with an
  80×170 mm slot, two flush bridge bars at y=±78 mm (the roller's seat rails; they
  leave an 80×144 mm centre opening the cubes pass through), 4 mm detent lip strip
  (x 40..52 mm), back stop, 25° ramp slab (rotated child) from the plate edge to the
  ground, dock far wall (95 mm > roller centre 50 mm — retention asserted), two
  full-length side fences (6 mm guide clearance per roller end, asserted).
- **roller** — DYNAMIC crimson cylinder, r 50 mm × 200 mm, 1.4 kg, axis along vault
  y, `solver_velocity_iteration_count=4` (GPU free-cylinder creep fix).
- **prize / distractor** — DYNAMIC 40 mm cubes (0.10 kg), GOLD and GREY, standing on
  the cavity floor at interior slots y=±38 mm; which slot holds which is shuffled
  per episode (`torch.rand < 0.5` — first-randint trap avoided).
- **pad** — KINEMATIC blue disc (r 55 mm) on the ground opposite the ramp.

Randomization (readback-verified, smoke check 2): vault yaw ±12° + xy ±30 mm, pad
xy ±40 mm, roller seat x ±8 mm, cube swap + per-cube jitter + free yaw.

## Rubric

Latched partial credit (never evaporates), all latches transient-guarded:

- 0.15 `unseated` — roller ever driven past the detent lip (x > 30 mm; the null
  policy's bounded solver walk stops at lip contact ~20 mm — smoke check 4)
- 0.20 `cleared` — mouth ever uncovered with the roller at REST (6-step still streak;
  a roller flying across the slot opens nothing)
- 0.15 `out` — prize ever outside the cavity WHILE the mouth is clear (a prize
  teleported past a seated roller latches nothing — smoke check 7)
- 0.25 `placed` — prize ever settled on the pad, gated on `out` already latched
- non-success capped at 0.75; **1.0 iff success()**: mouth clear, prize flat on the
  pad, distractor still inside, everything settled and finite — all judged live.

## Teleport solution (solve.py) — the legitimacy certificate

- **P0** reset, 180-step settle: roller seated over the slot, cubes inside, score 0.
- **P1 (opening — applied force + gravity)** an escalating horizontal force at the
  roller CoM (vault-local +x, 3→9 N) pushes it over the lip; the force is CUT the
  moment it tips onto the ramp; gravity rolls it down between the fences into the
  dock, hands-off, to rest. The roller is never teleported. Score 0.35.
- **P2 (transport)** one root-state write carries the prize to free space 30 mm above
  the pad — the slot is uncovered at this point, so the path is free space (the
  straight-up pinch-lift the embodiment argument describes). Gravity seats it on the
  pad. Score 1.0.
- **P3** ≥ 3.3 simulated seconds hands-off; success persists → `SIM_GEN_SOLVE:
  SUCCESS`.

Teleports are transport-only; every rubric fact (roller unseated/cleared, prize out,
prize on pad) is produced by force, gravity and contact. Verified on forge, seeds 0
and 1 (rc=0, 18.6 s / 19.0 s), score trace 0.00 → 0.35 → 1.00 → 1.00 each.

## Execution order

Open-then-retrieve is **physically forced**, not declared: while the roller is
seated nothing passes the slot (smoke check 5), and pad credit is gated on a real
opening (`out` requires a clear mouth — smoke check 7). Within that, the roller's
final resting spot (dock is the natural attractor) and when the prize is placed are
free; the distractor is simply never touched.

## Embodiment argument (single Franka + parallel-jaw gripper)

Plausible base pose: world origin, facing +x; everything lies at reach 0.28–0.62 m.

- **Open (nonprehensile push):** the roller seat is at world ≈ (0.46, 0.02),
  centre height 145 mm. The closed gripper's fingertips push horizontally at the
  CoM height along the fence direction: ≤ 8 N at quasi-static speed — trivially
  inside the Franka force envelope. The roller is 100 mm wide vs the ~80 mm jaw
  (asserted): rolling is the only strategy. After the lip pops, the robot only
  retracts — the roller rolls AWAY from the arm (down-ramp, +x) into the dock.
- **Retrieve (top pinch through the slot):** the centre opening is 80×144 mm; the
  40 mm cube stands with its top 25 mm below the plate (plinth floor keeps the grasp
  shallow) and grasp mid-height 45 mm below the plate — within Franka finger length,
  with 20 mm per-side x-clearance for the 40+8 mm jaw+finger envelope and ≥ 52 mm
  y-clearance. Colors disambiguate the shuffled cubes.
- **Place:** the pad at ≈ (0.30, −0.30) ± 40 mm is an open-ground set-down at
  0.42–0.45 m reach. No step needs a second arm, regrasping, or in-flight handoff.

## Files

- `scene.py` — cfg (+ seal/lip-force/aperture/dock asserts in `__post_init__`),
  compound vault spawner, scene (latched rubric in `post_step`), `register_env`.
- `solve.py` — phased teleport solution; watchdog + hard exit.
- `smoke.py` — 16-check rejection battery (settle, randomization readback, cube
  swap, bounded null-policy walk, SEAL raid, bare opening, closure bypass,
  coverage/pad near-misses, wrong object, restraint, settle gate, mechanism roll,
  latch persistence, success-never-True audit, frames.npz).

## Checks

- forge solve seed 0: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 18.6 s)
- forge solve seed 1: `SIM_GEN_SOLVE: SUCCESS` (rc=0, 19.0 s)
- forge smoke: `SIM_GEN_SMOKE: ALL PASS 16/16` (rc=0, 35.5 s)
