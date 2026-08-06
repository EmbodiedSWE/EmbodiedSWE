# plug_bayonet_lock — drop the lugged plug into the socket well and TWIST it locked

**Seed:** `maniskill/plug_charger`
(`sim_gen/RoboVerse/roboverse_pack/tasks/maniskill/plug_charger.py`)
**Tier:** medium — 2 stages (seat, then twist), one manipulated object.
**Execution order:** partially ordered by physics and by the rubric gates: the plug can
only enter the well with its lugs aligned to the slot, and lock-rotation credit is gated
on being SEATED (twisting in the air or above the plates earns nothing). Twist direction
is free (the lock zone is symmetric, 55–90 deg either way). No other ordering.
**Env name:** `simgen.plug_bayonet_lock` (scene `plug_bayonet_lock`, robot `null` —
scene-level; solve.py and smoke.py build this same env).

## What changed vs the seed

The seed is a precision **pick-align-and-push insertion**: grasp a charger, align its
two prongs with tight holes in a fixed wall base, translate it straight in; the checker
is a pure relative-bbox containment of the charger in the base frame (the ManiSkill
asset even enlarges the holes to keep the translational tolerance feasible). All the
difficulty is translational alignment; the final state is reached by pushing along one
axis.

Here the socket is a **bayonet (twist-lock) fitting**, fully procedural: a kinematic
block with a rectangular well (90 x 84 mm inside — the 48 mm plug rattles around, so
translational precision is deliberately trivial), whose mouth is covered by two orange
overhang plates leaving a single 54 mm slot strip. The blue plug carries two radial
lugs (tip-to-tip 76 mm, wider than the slot is narrow) and an elongated cap whose long
axis shows the lug direction. The lugs pass the slot only when aligned with it; once
the plug rests on the well floor, the lugs sit 6 mm below the plate undersides, and
rotating the plug >= 55 deg carries them under the plates. A smooth red cylinder
distractor fits the well but has no lugs and can never lock.

## Why strategically different

The seed's plan succeeds by **translation alone** — align, push in, done — and its
checker is containment. That plan is expressible here and REJECTED: a plug pushed
straight in to full depth but never twisted is not success and its score caps at ~0.5
(smoke control #6). Success requires a **rotation under contact as the load-bearing
final action**: `success()` is a yaw-latch predicate (relative lug-axis angle to the
socket's slot axis, mod 180 deg, at least `lock_deg = 55 deg`) on a SEATED, settled
plug — a different plan (read the slot direction from the randomized socket yaw, align
the lugs, lower through the slot, then wrist-twist ~60 deg while holding depth) and a
different code structure (a regulated twist loop and an angle predicate, not an
insertion servo and a bbox check). The lock is physically real, not just a number: the
geometric extraction limit is ~39 deg (asserted `lock_deg >= extract + 10` in the
config), so any state the rubric calls locked genuinely cannot be lifted out.

## Teleport solution outline (solve.py — the legitimacy certificate)

Teleports move the plug; they never do the task. Phases (each boundary prints
`SIM_GEN_SCORE`, non-decreasing — all credit is latched):

- **P0** reset (seeded), settle 0.5 s, layout readback (socket xy/yaw, plug spawn,
  extraction limit) — seed provenance in stdout.
- **P1 TRANSPORT (teleport)**: one pose write carries the plug across open space to a
  hover pose centred over the mouth, bottom 4 mm ABOVE the plate tops, lugs aligned
  with the slot (plug yaw = socket yaw). Entirely outside the well; satisfies no
  scoring gate; bypasses no required interaction.
- **P2 SEAT (contact dynamics)**: regulated downforce (2.5 N while descent speed
  < 0.10 m/s) plus small socket-frame lateral centring (clamp(-15*xy, +/-0.6 N),
  rotated to world) lowers the plug through the slot and 30 mm down the well onto the
  floor under real contact. Force cut at floor contact; 0.25 s settle.
- **P3 TWIST (contact dynamics)**: bang-bang torque about world z (0.05 N*m while
  yaw rate < 1.5 rad/s, +0.05 per 2.5 s stall up to 0.30) with a 1.5 N hold-down,
  twisting the seated plug against floor friction until the lug axis is 68 deg from
  the slot — inside the 55–90 deg lock zone with margin both ways. Torque cut; 1 s
  settle → `success()` (seated + locked + settled).
- **P4 persistence**: 3.3 more simulated seconds with no intervention; only if
  `success()` held prints `SIM_GEN_SOLVE: SUCCESS`; hard exit (watchdog + `os._exit`).

Verified on the forge on seeds 0 and 1 (see check list).

## Embodiment argument (single Franka arm, parallel jaw, OSC)

- **Base pose:** `(0.0, -0.55, 0.0)`, facing +y. Working radii: plug spawn ~0.39 m,
  socket well ~0.65 m, distractor (never touched) ~0.60 m — inside the proven
  0.30–0.71 m comfort envelope, and the working heights (cap grasp at 0.075–0.105 m,
  mouth at 0.036 m) are well off the ground-contact danger zone.
- **Plug (the one manipulated object):** grasp the CAP — a 34 x 20 x 30 mm box on top
  of the plug, standing 68–98 mm above the table at spawn; the 20 mm faces fit the
  parallel jaw with the fingers fully above the socket plates at all times (fingertips
  at ~66+ mm when the plug is seated, plates top out at 36 mm — the hand never enters
  the well or passes under anything). Sequence: pinch the cap across its short axis,
  lift, yaw the wrist so the cap's long axis (= lug axis) matches the visible slot
  direction, lower through the slot, keep light downforce, twist the wrist ~60 deg,
  release. Wrist roll range needed (~60–90 deg) is a fraction of the Franka joint-7
  range.
- **Precision demanded:** entry needs the plug centred within +/-7 mm along the slot
  (lug tips 76 mm vs 90 mm well) and +/-3 mm across it (48 mm body vs 54 mm slot),
  with yaw within ~+/-25 deg (lug corner clears the strip) — all far above OSC control
  noise, and the slot walls funnel residual error. The lock band is 35 deg wide
  (55–90 deg) with the physical stop at 90 deg usable as a hard reference: twist until
  it stops is itself a success strategy.
- **Distractor:** never needs to be touched; it spawns ~0.3 m from the well.

## Success and rubric (physical outcomes only)

Judged in the socket's body frame (its yaw is randomized, so the slot direction must be
read from the scene). `success()` iff the plug is SEATED (axis inside the well
footprint, centre within 6 mm of seated height, upright within 10 deg) AND LOCKED
(lug-axis slot-distance >= 55 deg, mod-180 symmetric) AND settled (< 0.05 m/s,
< 0.5 rad/s). `score()` in [0,1], latched each physics substep:
`0.15*best-approach + 0.35*best-seating-depth + 0.35*best-lock-rotation` (depth gated
inside-well + roughly upright; rotation gated SEATED), capped 0.85; 0.9 once seated +
locked; **1.0 iff success**; ~0 for doing nothing (approach normalized by the episode's
own spawn distance). Latched credit never evaporates (smoke #11); the solve's phase
prints are monotone. Honesty knobs asserted in the config: lock threshold >= 10 deg
beyond the geometric extraction limit; entry, rotation and under-plate clearances all
positive with margin.

## Randomization

Per episode: socket xy jitter +/-3 cm + yaw +/-20 deg (the slot direction moves), plug
spawn xy jitter +/-5 cm + free yaw, distractor xy jitter +/-3 cm. Verified by sim
READBACK in the smoke.

## Check list (smoke.py — rubric REJECTION battery, 13 checks; no probe reaches
success(), enforced by the audit check)

1. settle: reset finite, plug at rest upright on the table
2. settle: score ~0 at reset, no success
3. randomization readback: plug spawn xy + yaw vary
4. randomization readback: socket xy + yaw vary
5. null policy: 240 idle steps → score ~0, no success
6. SEED-STRATEGY control: plug pushed straight in to full depth, never twisted →
   seated but NOT success, slot-distance ~0, score <= 0.6
7. misaligned drop: lugs ACROSS the slot → the plug rests ON the plates, centre stays
   high, no rotation credit, no success, score <= 0.35
8. near-miss rotation: seated + twisted to 40 deg (past the ~39 deg extraction limit,
   short of the 55 deg lock zone) → NOT success, score < 0.9 (the load-bearing
   rotation tolerance)
9. monotonicity: deeper twist latched strictly more rotation credit than 20 deg
10. wrong object: red lug-less distractor seated in the well → score ~0, no success
11. latched credit survives pulling the plug back out (score unchanged, no success)
12. rejection audit: success() never True at any judged point of this battery
13. final no-NaN

Near-miss N/A notes: a settled plug at lock rotation but WRONG DEPTH is not
constructible — between the floor and the plates a rotated lug intersects the plates
(no settled pose exists), and above the plates the misaligned-drop control (#7) covers
the resting state; the depth gate is otherwise exercised by #7. The seed's literal
scene (prong holes in a wall base) is not expressible in this geometry; its STRATEGY
end state (straight full-depth insertion) is, and is rejected by #6.

Video frames are recorded throughout and saved to `frames.npz` in the working
directory.
