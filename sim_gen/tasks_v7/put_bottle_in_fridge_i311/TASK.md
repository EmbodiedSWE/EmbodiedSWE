# chill_rack — lay the bottle into the captive sliding rack, push the rack in to shelve it

**Seed:** `rlbench/put_bottle_in_fridge`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/put_bottle_in_fridge.py`)
**Tier:** medium — **2 stages with a forced reorientation**. **Execution order:
REQUIRED — lay-in before push-in.** The rack's cradle is only exposed while the rack
is drawn out; once it is inside, the roofed interior and the 105 mm letterbox mouth
make the trough unreachable (for an arm and for a drop alike), so loading must happen
first. The rack can always be pulled back out by its protruding handle, so the order
is recoverable but not skippable.
**Env name:** `simgen.chill_rack` (scene `chill_rack`, robot `null` — scene-level;
solve.py and smoke.py build this same env).

## What changed vs the seed

The seed's fridge is an articulated cabinet with a revolute DOOR: the plan is "open
the door, then place the bottle UPRIGHT inside" — one fixture actuation that merely
*clears the way*, followed by a standard grasp-carry-release at the goal region, with
the bottle kept upright the whole way.

Here every element of that plan is dead, by geometry:

- **There is no door.** The locker's only opening is a fixed letterbox mouth 105 mm
  tall; the interior is roofed. Nothing can be dropped in from above (smoke #7 — a
  released bottle settles ON the roof), and no fixture-clearing move exposes the goal.
- **Upright placement is impossible, not just unscored.** The standing bottle
  (190 mm) is nearly twice the mouth height; stood on the rack it can never pass the
  header (smoke #6). The solver MUST reorient the bottle to lying — the seed never
  changes the bottle's orientation.
- **The goal is unreachable by direct placement.** The shelf position is served
  exclusively by a captive sliding rack (retaining lips — it slides, it cannot be
  lifted out). The bottle is laid into the rack's cradle trough while the rack is
  drawn OUT on the apron, and the final placement is achieved by **actuating the
  fixture with the payload aboard** — pushing the loaded rack through the mouth until
  it seats against the back wall, the bottle riding on friction/ridges/chock the
  whole ~22 cm. In the seed the fixture actuation and the object motion are separate
  strategy steps; here the object's goal motion IS the fixture actuation.
- A **red can distractor** must be left out (smoke #8), and a bottle shoved along the
  interior floor OFF the rack fails (smoke #10) — "in the fridge" is not enough; it
  must be racked.

A solver therefore needs a different plan (reorient → load a carrier → drive the
carrier to the goal) and different code structure (a load-then-transport program
where the last stage manipulates the fixture, not the bottle), not different
parameters. Same-strategy-different-numbers this is not: the seed's door-open
subroutine has no analogue here, and the seed's place-upright subroutine is
physically rejected.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move objects; they never do the task. The bed-down and the entire ride to
the seat go through contact dynamics. Phases (each boundary prints `SIM_GEN_SCORE`,
non-decreasing — all credit is latched):

- **P0** reset (seeded), settle 0.5 s, layout readback (locker pose + yaw, rack
  draw-out travel t0, bottle + can positions — seed provenance in stdout), baseline
  score ~0.
- **P1 TRANSPORT (teleport, the only pose write on the bottle)**: one root-state
  write carries the bottle from its upright ground spawn to a LYING pose (axis along
  the trough, neck outward) **20 mm above** the drawn-out rack's cradle — above the
  trough z-band, so the freshly-written state earns no credit (asserted). The
  reorientation happens in free air, exactly what a wrist rotation does.
- **P2 LAY (contact dynamics)**: gravity drops the bottle the last 20 mm; it lands
  between the cradle ridges and beds down under physics (1 s). The `laid` latch first
  sets here, from real resting contact.
- **P3 RIDE (contact dynamics)**: a world-frame horizontal force at the rack's CoM,
  aligned with the channel axis (velocity-regulated bang-bang: 4 N while axial speed
  < 0.08 m/s, +2 N per 2 s stall up to 12 N), drives the loaded rack through the
  mouth until its travel < 6 mm (plus a gentle trim if it rebounds off the back
  wall). The bottle is coupled to the rack ONLY by gravity/friction/ridges/chock — if
  it toppled or slipped off during the ride, success() could never turn True. No pose
  write ever touches the rack. `success()` first turns True here, judged on the
  settled physical state.
- **P4 persistence**: 3.3 more simulated seconds hands-off; only if `success()` held
  prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 and 1 (see check list).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

**Base pose:** `(-0.32, 0.0, 0.0)`, facing +x (the locker mouth faces the robot; the
apron runs toward it). Working radii from this base: bottle/can spawns 0.46–0.66 m,
the drawn-out cradle ~0.55–0.63 m, the handle across its whole push stroke
0.44–0.72 m — all inside the proven ground-level comfort envelope.

**Bottle (stages 1–2): side pinch, lift, lay down into the trough.** A free-standing
60 mm cylinder on open ground — a canonical side pinch (60 mm across the 80 mm jaw,
grasp height ~75 mm, all approaches unobstructed). Reorientation is a single wrist
rotation during the carry; release from ~2 cm above the cradle, centred over the
70 mm-wide trough gap — ±5 mm of lateral slack plus ridge funnelling, and the gripper
stays above the drawn-out rack in open air (nothing overhead on the apron). Required
precision ~1 cm — far above OSC control noise.

**Rack (stage 3): push the yellow handle block.** The handle is a 50 × 60 × 64 mm
block at the rack's outer end, top face at ~90 mm height, always OUTSIDE the mouth —
push its outer 60 × 64 mm face with closed fingertips (or pinch it, 50 mm across the
jaw, to pull the rack back out for a retry). The stroke is a straight ~22 cm
horizontal push at constant height; the channel absorbs ±3 mm of lateral error and
the back wall is a hard stop, so overshoot is impossible and the 12 mm seat tolerance
is reached by simply pushing until it stops. The hand never enters the mouth (the
handle never crosses the fascia plane).

**Can, locker:** never need to be touched (the locker is a kinematic fixture; the can
must merely be left alone — it spawns on the opposite side band, ≥ 30 cm from the
bottle, verified by readback).

Every contact the task requires is one the arm can make.

## Success and rubric (physical outcomes only)

All geometry is judged in body frames (locker yaw is randomized): the trough test in
the RACK's frame, containment and travel in the LOCKER's frame. `success()` iff,
simultaneously and settled (|v| < 5 cm/s on bottle and rack): the bottle lies bedded
in the cradle trough (|x−15 mm| < 32 mm, |y| < 25 mm, axis height in (24, 52) mm,
axis within ~20° of horizontal and ~30° of the trough direction — either end may
point outward), AND the rack is seated within `seat_tol = 12 mm` of the back wall
(plus channel y/z sanity gates), AND the bottle is fully inside under the roof
(locker-frame x < 50 mm, |y| < 60 mm, z < 100 mm). `score()` ∈ [0,1], latched each
physics substep: `0.30·laid + 0.45·ride_max`, where `ride_max` is the running max of
inward travel progress `(t0 − t)/t0` **counted only while the bottle is currently in
the trough** (pushing an empty rack earns nothing — smoke #13); capped 0.85; **1.0
iff success()**; ~0 for doing nothing (the rack starts drawn out and the ride term is
gated, so reset drift earns exactly 0). Latched credit never evaporates; the solve's
phase prints are monotone.

## Randomization

Per episode (verified by sim READBACK in the smoke): locker yaw 180° ± 8° + xy jitter
± 2 cm; rack initial draw-out travel t0 ∈ [0.200, 0.235] m; bottle and can upright in
disjoint side bands (x ∈ [0.10, 0.24], y ∈ ±[0.19, 0.30]) whose sides SWAP 50/50 (so
"the object on the left" is not memorizable), always ≥ 30 cm apart.

## Check list (smoke.py — rubric REJECTION battery, 15 checks; no probe reaches
success(), enforced by the audit check)

1. settle: reset finite — rack drawn out at its written travel (readback), bottle
   upright on the ground, everything at rest
2. settle: score ~0 at reset, no success
3. randomization readback: locker yaw + xy and rack draw-out travel vary
4. randomization readback: bottle + can positions vary (incl. band swaps); ≥ 30 cm
   apart at every seeded reset
5. null policy: 240 idle steps → score ~0, no success
6. SEED-STRATEGY control A (upright placement): bottle stood UPRIGHT on the rack
   cradle earns nothing (the trough demands lying alignment) — and upright it can
   never pass the mouth (static geometry: bed top + 190 mm ≫ 105 mm header)
7. SEED-STRATEGY control B (carry-drop): bottle released above the locker settles ON
   the roof — the interior is roofed and door-less; no success, score ≤ 0.02
8. wrong object: RED can laid in the trough and racked inside, bottle outside → no
   success, score ≤ 0.02
9. near-miss: bottle in the trough but rack settled ~45 mm short of the 12 mm seat
   tolerance → NOT success, score ≤ 0.85 (the seat tolerance is load-bearing)
10. wrong place: bottle lying on the interior floor plate — inside the locker but OFF
    the rack → no success, score ≤ 0.02 ("in the fridge" is not enough)
11. incomplete: bottle laid in the trough but rack never pushed → laid credit only
    (0.20–0.35), NOT success
12. monotonicity: the loaded rack constructed at deeper travels (0.10 → 0.03 m)
    latches strictly more ride credit and score, still no success outside tolerance
13. ride gating: an EMPTY rack seated inside earns nothing, no success
14. rejection audit: success() never True at any judged point of this battery
15. final no-NaN

N/A notes: the seed's exact end state (bottle upright inside) is **not constructible
as a settled state** — the roof (115 mm) is below the standing bottle (190 mm), which
is precisely the interlock; #6 and #7 probe its two nearest expressible analogues.
"Bulldozing" (pushing the loose bottle along the channel floor with the rack's edge)
ends with the bottle inside but OFF the rack — rejected by the same clause as #10 —
and a 60 mm bottle jammed between bed edge and back wall also blocks the rack from
ever reaching the 12 mm seat tolerance, so the cheat fails twice.

Video frames are recorded throughout and saved to `frames.npz` in the working
directory.
