# libero_kitchen_scene6_close_the_microwave_i114 — `microwave_ballast_door`

Close the counterweighted microwave door WITHOUT touching it: load both mugs into
the tray hanging from the door's raised edge until their weight overpowers the
counterweight and the door swings shut on its own.

## Provenance

- **Seed task**: `libero_90/libero_kitchen_scene6_close_the_microwave`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene6_close_the_microwave.py`)
  — a Franka pushes the microwave's hinged door shut and is judged by a single
  articulation joint angle crossing a threshold. One push, one panel, monotone.
- **Scene**: `microwave_ballast_door`, registered as
  `simgen.microwave_ballast_door` with `robot="null"`.

## Strategic difference (vs the seed and the corpus read this session)

The seed is **direct articulation actuation**: push THE door past a joint-angle
threshold. Here pushing the door is *physically refuted*, not just unrewarded:

- the door is a top-hinged awning panel held OPEN by a **brass counterweight drum**
  behind the hinge line — an empty door pushed flush **springs back open** from any
  angle (statics condition A, margin ≥ 3.69×; live-verified by smoke check 6 with a
  2.2 N·m hold-then-release probe);
- the only way to close it is **indirect, via ballast**: a self-leveling tray hangs
  on a free pivot from a pin on the door's raised edge; loading BOTH 0.35 kg mugs
  into the tray overpowers the counterweight at every angle (condition C, worst
  margin 1.49× at the 85° open stop) and the door swings shut by pure mechanism
  physics with nothing applied;
- **one mug is provably insufficient** (condition B: loses 1.22× at the open stop)
  — the door visibly stays open after the first mug, making the load count
  load-bearing, not incidental;
- success also demands the cargo arrive intact: both mugs inside the tray's
  containment box, everything settled (streak-based stillness), door < 6°.

A solver needs a different **plan** (transport cargo into a moving, self-leveling
receptacle hanging off the very mechanism it must actuate — while never touching
that mechanism) and a different **code structure** (pan-frame containment + door
angle + settle streak, not a joint readout after a push). Unlike the corpus read
this session: `i106 microwave_carousel` (bidirectional precision rotation of a
free platter; the door there is static scenery and cargo rides friction — here the
door IS the goal DOF and cargo is placed, not ridden), and no other task read is a
ballast/counterweight machine.

## The mechanism is real (numbers)

All statics are asserted numerically in `__post_init__` over the full angle range
and were verified live on the forge with teleport probes:

- Door: blade 12×360×250 mm, mass 2.36 kg, authored CoM (0.0015, 0, 0.080) in door
  frame, explicit diagonal inertia — the brass drum (r 40 mm) rides 104.5 mm above
  the hinge on the back side. Opening torque
  τ_open(φ) = 2.36·9.81·(0.0015 cos φ + 0.080 sin φ).
- Tray pin at door-local (−0.012, 0, −0.36): IN the blade plane, 30 mm beyond the
  blade tip, with the open stop capped at 85° < 90° — so every blade point stays
  above the pin plane through the whole swing (blade-clearance certificate, 2.6 mm
  at the stop; asserted). The tray (0.08 kg) self-levels on a free revolute pivot.
- Closing torque with load m: τ_close = (0.08 + m)·9.81·(0.012 cos φ + 0.36 sin φ).
  - (A) empty door reopens from flush: min margin 3.69× at φ = 0;
  - (B) ONE mug loses at the 85° stop by 1.22× (equilibrium would be 2.7°, but the
    stop wins) — door stays open;
  - (C) TWO mugs beat the counterweight at every φ, worst 1.49× at 85°, net
    ≈ 0.9 N·m at the stop, and hold flush with +0.057 N·m at 0°.
- Effective inertia about the hinge with the hanging load ≈ 0.15 kg·m²,
  angular damping 1.5/s → the door swings shut in a couple of seconds and damping
  kills the residual pendulum swing on the closed stop.
- Anti-friction-lock geometry asserted: the hinge axle (±0.18 m) clears the shell
  towers (inner face at |y| = 0.19) with ≥ 5 mm — clearance invariant under the
  Y-axis hinge motion, so overlapping-collider friction lock (which silently holds
  the joint at any angle) cannot occur.

## Solution outline (solve.py — teleports are TRANSPORT ONLY)

Each mug is teleported ONCE through free air to a hover pose 2 cm above its slot in
the hanging tray (read in the LIVE pan frame), released with zero velocity. The
door and the pan are never written; everything load-bearing is contact dynamics.

1. **P0**: reset, 1 s settle; door drifts to the 85° open stop; assert score ≤ 0.02
   and tray empty.
2. **P1**: mug A → tray; gravity + friction seat it; 2 s statics check — the door
   MUST stay ≥ 70° (condition B live). `SIM_GEN_SCORE` 0.20.
3. **P2**: mug B → tray (0.40); hands-off — the ballast overpowers the
   counterweight, the door crosses the 45° swing gate (0.70) and presses onto the
   closed stop; success at door < 6° with both mugs contained and settled (1.00).
4. **P3**: 3.3 s (400-substep) hands-off persistence → `SIM_GEN_SOLVE: SUCCESS`.

Passes seeds 0 and 1 on the forge (`SIM_GEN_SOLVE: SUCCESS`, zero persistence
flickers).

## Rubric

- `success()`: door angle < 6° AND both mugs inside the pan-frame containment box
  AND everything settled (streak of 30 consecutive still substeps — a turning-point
  velocity dip cannot fire it, and a teleported pose breaks the streak via a > 5 cm
  per-substep jump detector even though its written velocity is zero) AND all
  states finite.
- `score()` (latched in `post_step`, never evaporates): 0.20 first mug seated in
  the tray + 0.20 second mug (0.40) + 0.30 when the LOADED door first drops below
  45° (gated on both mugs in the tray at that instant — pushing the empty door
  earns nothing), cap 0.70; 1.0 iff `success()`. Null policy ~0 (mugs spawn on the
  counter, door rests at the open stop).
- Per-episode randomization (readback-verified in smoke over 6 seeds): mug A/B
  counter positions (x, y) and yaws, and a side swap that exchanges which mug
  spawns on which side of the plinth.

## Embodiment argument (single Franka arm, OSC, one base pose)

Base at the env origin (0, 0) on the floor, facing +x. The mugs spawn on the open
counter at x ≈ 0.28 m, z ≈ 0.20 m — a standard tabletop top-grasp (mug body Ø
64 mm, handle along x, well inside reach). With the door at the 85° open stop the
hanging tray's floor sits at z ≈ 0.354 m, its opening facing straight up,
x ≈ 0.42 m from the base — the arm carries each mug over the tray and lowers it
2 cm above the slot, exactly the hover pose the teleport stands in for; nothing
blocks the vertical approach corridor (the raised blade tip is 30 mm above the pin
and behind it). **Contact strategy per object**: MUGS are top-grasped and lowered
into the tray; the DOOR, TRAY, and SHELL are never touched. **Execution order:
mugs in any order between themselves** — but both must be loaded before the door
will move (physics-mandated ordering: the mechanism itself enforces "load ballast
→ door closes"; declared, and check 7/12 of smoke verify one mug and removed
ballast fail).

## Files

- `scene.py` — cfg + scene: procedural compound spawners (kinematic shell with
  recessed plinth and hinge towers, counterweighted door with drum + axle + pin
  arms, self-leveling tray, two mugs), spawn-authored hinge with ±(−1°, 85°)
  limits and free pan pivot, full statics certificate in `__post_init__`, latched
  score, streak-based settle. Registers `microwave_ballast_door` and
  `simgen.microwave_ballast_door`.
- `solve.py` — transport-teleport solution (above); watchdog + hard exit.
- `smoke.py` — 14-check rejection battery (records `frames.npz`): reset/settle
  sanity, randomization readback over 6 seeds (spreads asserted, side swap seen
  both ways), null policy, the seed strategy refuted live (2.2 N·m push holds the
  door flush → zero credit → released door REOPENS to the stop), one-mug statics
  (door stays open, exactly the 0.20 milestone), latched credit survives mug
  removal, mugs-in-chamber and mugs-on-floor near-misses, teleported-shut linkage
  rejected by the settle/window gates, ballast-removed door reopens with score
  still latched and no success, success-never-True audit, no-NaN.
  `SIM_GEN_SMOKE: ALL PASS 14/14`.

## Check list

- [x] Strategically different from the seed (indirect ballast actuation with the
      direct push physically refuted vs a direct door push) and from every task
      read this session.
- [x] Teleports transport-only (mugs through free air to a hover above the tray);
      the door/pan/shell are never written; the closure is pure mechanism physics.
- [x] `SIM_GEN_SCORE` non-decreasing at phase boundaries; ≥ 3 s persistence after
      success; hard exit with watchdog.
- [x] Solve passes on ≥ 2 seeds (0 and 1) on the forge.
- [x] Rubric rejection battery passes on the forge; success never True in it.
- [x] describe()/instruction() state goal, mechanism, randomization, and what does
      not count; all claims physically true (statics asserted in `__post_init__`,
      refutation probes run live in smoke).
- [x] Fully procedural geometry; no external assets; files only in the task dir.
