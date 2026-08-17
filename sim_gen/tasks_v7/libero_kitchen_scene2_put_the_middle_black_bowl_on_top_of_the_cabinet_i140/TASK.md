# ramp_hutch — slide the middle bowl up the service ramp into the roofed hutch

**Task id:** `libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet_i140`
**Env:** `simgen.ramp_hutch` (scene-level, `robot="null"`)
**Files:** `scene.py` (scene + rubric), `solve.py` (teleport-transport + force-push solution),
`smoke.py` (rejection battery), this file.

## Seed provenance

Seed task: `libero_90/libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet`
(RoboVerse `roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet.py`).
The seed places three identical akita black bowls and a plate on a table beside a wooden
cabinet; the goal is one vertical pick-and-place — grasp the MIDDLE bowl and set it inside a
bbox above the cabinet's always-open flat top. Success is a static pose test on that bowl alone.

Kept from the seed: three identical black bowls in a row with the MIDDLE one as the target
(identity discrimination), a distractor plate, a wooden cabinet, and the goal of relocating the
middle bowl onto the cabinet's top level.

## What the task is

The cabinet's top is a walled **tray** sealed under a fixed **roof** — there is NO opening from
above; anything released over the cabinet lands on the roof, which is not a goal surface. The
tray's only entrance is a **doorway** (15 cm wide, 8.5 cm tall) cut into the front wall, reached
by a 20.4° curb-fenced service **ramp** running from the floor up to the door **sill**. The ramp
crest sits 5 mm above the sill top, and the tray floor lies 3 cm below the sill — a one-way
step: a bowl that crosses the crest drops in and cannot slide back out (smoke check 15 proves
the exit is blocked under a sustained pull).

Three identical black bowls stand in a floor row; which physical body occupies which slot is a
fresh random permutation every episode (readback-verified), so "the middle bowl" is a
per-episode identity. The goal:

> "Put the middle bowl on top of the cabinet: the top compartment is roofed, so slide the bowl
> up the yellow-curbed ramp and through the doorway until it drops inside. Leave the other two
> bowls and the plate outside; the bowl must end upright."

`success()` = middle bowl upright at rest inside the tray's seat band ∧ the **doorway-passage
latch** fired (the only physical way in — a state teleported into the tray does not count)
∧ neither decoy bowl inside the tray ∧ everything settled.

## Strategic difference — vs the seed and vs every corpus neighbor

- **vs the seed:** the seed's whole plan is a grasp and a vertical release above the cabinet.
  Here that exact plan is a measured failure mode: the released bowl parks on the ROOF and
  scores 0 (smoke check 11). The destination's access direction is **inverted** — top sealed,
  lateral entry only — so the required behavior is a sustained, friction-loaded uphill *slide*
  through a doorway, terminated by a one-way drop. Transport gets the bowl to the ramp;
  transport alone can never finish the task.
- **vs i306 hatch_shelf / i5 counterweight_shelf:** those tasks *reconfigure the fixture*
  (close a lid, ballast a shelf) to create/level the destination, then place by transport. Here
  the fixture never moves and nothing is placed by release; the challenge is the constrained
  approach path of the object itself.
- **vs i61 cart_ferry:** cart_ferry is a multi-stage logistics chain whose destination is a
  vehicle the agent repositions twice. Here there is a single moving object and zero stages of
  environment reconfiguration — the difficulty is the geometry of the only admissible
  trajectory (ramp channel → doorway → one-way sill), plus which-of-three identity.
- **vs i53 slab_easel / i14 chute_switch / i48 cask_weight_sort / i17 skittle_gallery:** those
  are orientation/routing/sorting decisions about *which* state to produce; here the terminal
  state is unique and the task is *how* to physically reach it.
- **vs i57 hanoi_rings / i27 bell-herd:** no ordering or multi-agent component; the decoys must
  simply stay out (occupancy veto, smoke check 8).
- **vs robobench suite (balance_scale, combination_safe, syringe, pen_holder):** no articulated
  mechanism; the "mechanism" is emergent rigid-body geometry (incline friction hold, doorway
  clearance, one-way step).

No task in the read corpus gates its destination behind an inclined approach channel with a
one-way entrance, and none makes the seed's own strategy a settled, scored failure state.

## Rubric (latched credit, `score()`)

| latch | condition (fixture frame, target bowl only) | weight |
|---|---|---|
| `_hi_ever` | ever high on the ramp (x −0.33..−0.15, z 0.17..0.31), upright | 0.15 |
| `_doored_ever` | center ever inside the doorway cut (the passage) | 0.25 |
| `_seated_ever` | ever seated in the tray **and** the passage latch already fired | 0.35 |

`score = Σ weights`, capped at 0.95 unless `success()`, which returns exactly 1.0. Latches are
evaluated in `post_step` every physics substep; the seated latch is **gated on the passage
latch**, so a teleport into the tray earns nothing (smoke check 12). The printed
`SIM_GEN_SCORE` sequence is non-decreasing by construction.

## Solution (`solve.py`) — teleport = transport only

1. **TRANSPORT** — teleport the middle bowl (scene.target_idx) from the floor row onto the
   LOWER ramp (fixture x = −0.55), tilt-matched to the slope, zero velocity. This point is far
   outside every credit band; after settling the solve *asserts score is still 0.000*.
2. **PUSH P1** — pulsed 2.5 N force along the fixture's uphill tangent (force on only below
   0.08 m/s; mg(sin θ + μ cos θ) ≈ 1.27 N resists) until the bowl passes x > −0.30, inside the
   hi band. Release; static friction (μ 0.55 > tan 20.4° = 0.372) parks it. → 0.150
3. **PUSH P2** — same pulsed push through the doorway (passage latch fires in the door cut),
   over the crest/sill, 3 cm drop into the tray, creep to x > −0.12 in the seat band. Release,
   settle → success. → 1.000
4. **PERSISTENCE** — ≥ 3.5 simulated seconds fully hands-off; `success()` must still hold
   before `SIM_GEN_SOLVE: SUCCESS` is printed.

Because the pod's external-force API may interpret wrenches in the body's current frame,
`drive()` probes force encoding at runtime from measured progress and toggles between raw-world
and `quat_apply_inverse(q_now, f)` if the body stalls.

Verified on the forge: seeds 0, 1, 2 all print 0.000 → 0.000 → 0.150 → 1.000 → 1.000 →
`SIM_GEN_SOLVE: SUCCESS`, rc = 0.

## Embodiment argument (single Franka, 8 cm parallel jaw, OSC)

The same plan executes with one arm:

- **Pick the middle bowl off the floor row:** the bowl is a 10.4 cm-wide open cup — too wide
  for an outside pinch, but the open top admits a **rim pinch** (one fingertip inside the wall,
  one outside; wall is 1.0 cm thick), the standard way to lift an open cup. Carry it to the
  ramp foot and set it down inside the curb channel, tilt-matched by compliance.
- **Uphill push:** fingertip or knuckle against the bowl's downhill wall, slide up the ramp.
  The curbs (6 cm tall) fence the channel so only forward servoing is needed; required force
  ≈ 1.3 N, trivial for the arm. Near the top the hand stays BELOW the roof line: the bowl's
  downhill face at the crest is at z ≈ 0.24–0.30, fully outside the doorway — and once the
  bowl's CoM crosses the crest it **drops in by gravity**, so the hand never needs to enter
  the doorway or the tray.
- **One plausible base pose:** base at fixture-frame ≈ (−0.45, −0.50), facing the ramp. The
  floor row (x −0.56 ± 0.2, y −0.38), the ramp foot (−0.82, 0) and the crest (−0.19, 0,
  z 0.24) are all within ~0.85 m reach; every required contact lies at z ≤ 0.30.

No stage admits a shortcut: there is no top opening to drop through (the roof is sealed), the
doorway is the only entrance, and a decoy bowl in the tray forfeits success, so grabbing an
arbitrary bowl does not work — the row must be read for the middle identity first.

## Execution order

No multi-object order is declared — this is a single-target task. The stage order
(high-on-ramp → doorway → seated) is enforced physically and by the rubric gating: the seated
latch requires the passage latch first, and the passage is unreachable except up the ramp.

## Checks (`smoke.py`) — 15

1. reset settles, states finite, three bowls in a floor row, tray empty; 2. score ~0, no
latches at reset; 3. randomization READBACK across 6 seeds (fixture xy/yaw, TARGET BODY INDEX
varies, row x, bowl yaw); 4. null policy 240 steps ≈ 0; 5–7. oracle on seeds 0/1/2 (transport
earns zero, climb → doorway → tray = success 1.0, persists 240 steps); 8. occupied forfeit
(decoy teleported into the tray flips success off, score falls to the 0.75 latch sum);
9. monotone ladder 0 < 0.15 (parked hi, no success) < 0.40 (doorway latch) < 1.0; 10. partials
< 1.0; 11. seed-strategy negative (bowl released above the cabinet lands ON THE ROOF, readback
z, score 0); 12. teleport-bypass negative (target placed in the tray: live seat predicate TRUE
but latch refuses without the passage — score 0); 13. inverted negative (upside-down bowl
readback-inside the hi band, latch refuses); 14. credit permanence (hi credit survives removal
to the floor, no success); 15. sill one-way (sustained pull toward the doorway travels ≥3 cm
— non-vacuous — but cannot exit; success recovers). Frames recorded to `frames.npz`.
