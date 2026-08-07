# piston_jar (open_jar_i65)

Open a factory-SEALED jar with no grippable closure and deliver its contents:
carry the jar onto a fixed pedestal spike, press it down so the spike drives the
jar's own internal ejector piston, shear the seal from the inside, and place the
freed red ball in the blue dish.

## Seed provenance

Seed: `rlbench/open_jar` — grasp the jar's lid, unscrew it (one continuous
rotation about the jar axis), lift it off.

What was kept from the seed:

- a closed jar whose mouth must end up open;
- "get at the contents" as the point of opening (the seed lifts the lid away;
  here the contents are the deliverable).

What was changed:

- **The lid affords nothing.** The seed's entire plan lives on a grippable lid.
  Here the lid is a bare plate recessed 4 mm below the rim with a 2 mm annular
  gap (nothing to pinch), and it is WELDED to the jar by a PhysX breakable
  fixed joint (breakForce 10 N): it does not lift, pry, twist or shake off.
- **Nothing rotates.** The seed is one continuous rotation; here no part of the
  opening involves rotation at all — opening is a straight vertical press.
- **The tool is the environment + the jar's own mechanism.** The jar's bottom
  plate has a 36 mm through-hole onto a captive internal piston (80 mm
  prismatic travel). Opening = sleeving the jar over a fixed orange spike and
  pressing DOWN: descent of the jar is converted into ascent of the contents,
  the piston presses the ball into the lid's underside, and past ~10 N the
  seal shears from the INSIDE.
- **Direction sensitivity is geometric, not scripted**: the lid hovers 0.5 mm
  above a supporting shoulder, so pressing the lid from outside is carried by
  shoulder contact and never loads the weld — only the inside push does.
- **A delivery goal on the contents**: the freed red ball must rest in a walled
  dish; the jar standing open is only partial credit.

## Strategic difference vs seed and corpus

- vs the seed (`open_jar`): the seed's winning strategy — take the lid off —
  is not executable here (nothing to grip, weld holds 10 N ≫ every gravity or
  carry load), and the seed's *outcome* ("lid off, contents still in the jar")
  scores only the 0.25 opened credit with no success (smoke check 9).
- vs `close_jar_i54` (latch canister): i54 *closes* a container by placing a
  lid and swinging two revolute turn-tabs; here a container is *opened*, there
  are no revolute DOFs at all, and the closure is destroyed (breakable joint),
  not articulated.
- vs `put_knife_in_knife_block_i41` (kerf chop): i41 also uses a breakable
  joint, but as a *target to sever from outside* — pick the one real blade and
  drive it through a slot onto a bridged rod. Here the breakable joint is a
  *seal that cannot be attacked directly*: no tool is picked up, the load path
  to the weld runs through the jar's own internal piston-and-ball train, and
  the press is applied to the jar body, not to anything sharp. The weld also
  gates a *retrieval* (contents out), not a *cut*.
- vs `screw_nail_i59` / slide-latch tasks: no latch is slid and no fastener is
  turned; the single motion is a vertical press of the whole container against
  a fixed feature of the world.
- vs `open_oven_i7`, `setup_checkers_i39` and other open/pry tasks: no hinge,
  no lever, no prying wedge; the lid leaves through the mouth, ejected from
  inside.
- The press-to-eject inversion — moving the CONTAINER against a fixed spike to
  move the CONTENT upward — appears nowhere else in the corpus.

Physics the task rests on (probe-verified in smoke): the seal holds against
gravity, carrying, full inversion + hopping (check 4 — the damped piston
cannot impulse-hammer it open) and a 6 N direct pull that lifts the whole
0.58 kg stack hanging by the weld (check 7), shears at 15 N (check 8,
threshold bracketed); merely parking the jar on the spike hands-off loads the
seal only ~4.4 N and does NOT open it (check 5 — the press must be forceful
and sustained); a sealed mouth makes delivery unearnable (bypass check 6).

## Scene

Procedural geometry only (UsdGeom cubes/spheres under compound rigid roots;
prismatic + breakable fixed joints authored at spawn):

- **pedestal** (12 kg dynamic compound): 170×170×30 mm base slab, 16 mm square
  orange spike to z 106 mm with an 8 mm pilot tip to z 116 mm; low-friction
  material on the spike (μ 0.10/0.08).
- **jar** (0.45 kg, origin at footprint centre): 64 mm square tube, bottom
  plate z 0–8 mm with a central 36 mm through-hole (low-μ hole edges), 48 mm
  bore with walls to the shoulder at z 92 mm, mouth collar to the rim at
  z 102 mm (interior 56 mm).
- **piston** (60 g): 46×46×10 mm plate + 6 mm tray fences, Z prismatic joint
  to the jar (limits −1…+80 mm, rests hanging at the bottom) with a pure
  viscous damper on the slide (2 N·s/m — a real ejector is not frictionless:
  invisible to the slow press at ~0.1 N, but the piston cannot free-fall its
  stroke and impulse-hammer the ball into the lid when the jar is inverted).
  Piston↔jar keeps the USD joint-pair collision filter — alignment is by the
  joint, stops are the limits; every load-bearing contact
  (spike→piston→ball→lid) is between unjointed pairs and live.
- **lid** (40 g): bare 52×52×6 mm plate, welded (breakable fixed joint,
  breakForce 10 N, breakTorque 0.5 N·m, collision ENABLED) 0.5 mm above the
  shoulder; centre of mass offset 10 mm so a centre-lifted lid tips off the
  rim instead of balancing on the ball.
- **ball**: red 30 mm sphere, 30 g, riding in the piston tray under the lid.
- **dish**: blue walled tray, 120 mm square, 14 mm fences, 1.2 kg dynamic.

Geometry chain at full press: jar bottomed on the base (z 30 mm) puts the
spike tip 86 mm up the jar — piston top at 96 mm (above the 92 mm shoulder),
ball centre at 111 mm, nine millimetres proud of the rim, presented in the
tray for a pinch grasp. The prismatic limit (90 mm) is never reached.

Randomization (verified by readback in smoke check 2): pedestal xy + yaw
(±25°), jar xy + free yaw, dish xy + yaw. The sealed stack is written
coherently on the jar axis.

## Rubric

`success()` (all live): `ball_in_dish & jar_open & still & finite`, where

- `ball_in_dish`: ball centre inside the dish pocket (dish frame, 42 mm xy,
  10–45 mm z), dish upright;
- `jar_open`: the lid centre NOT in the jar-frame mouth box (30 mm xy,
  70–135 mm z) — a ball delivered past a still-sealed jar does not succeed;
- `still`: 60-consecutive-substep stillness counter (not instantaneous).

`score()`: latched — 0.15·engaged_ever (spike inside the jar through the
bottom hole) + 0.25·opened_ever (lid physically clear of the mouth) +
0.20·presented_ever (ball riding above the shoulder; order-coupled on opened —
physically downstream of the shear), capped at 0.60; exactly 1.0 iff
`success()` holds live. Latches are included in get_state/set_state. A sheared
weld is irreversible across reset, so the smoke battery orders intact-seal
probes first.

## Solution outline (solve.py)

Teleports for TRANSPORT ONLY; the opening is applied force through contact:

1. settle + layout readback asserts (sealed, ball captive, score ≤ 0.03);
2. the sealed assembly translated rigidly to a hover 5 mm ABOVE the spike tip
   (nothing engaged by the write), then a **guarded lower**: velocity-regulated
   vertical hold (press term capped 3 N so the seal never nears 10 N), xy
   spring to the live spike axis, attitude PD (the sleeved jar is an inverted
   pendulum — held exactly as a wrist would); wrenches pre-encoded with
   R_ref·R_now^T against the pod's force-frame drag. The spike enters the
   hole, lifts the piston until ball meets lid → STALL, seal intact → 0.15;
3. **press to shear**: same servo, press released to 25 N (escalating on
   stall) — the weld shears, the ball shoves the lid out of the pocket, the
   offset CoM tips it off the rim; if the freed thin plate balances covering
   on the ball's top (statically stable: CoM height 3 mm < ball radius), it is
   swiped off — escalating lateral pushes with partial gravity relief,
   direction rotating 45°/attempt, the fingertip swipe a Franka would use on
   the ungrippable loose plate — the jar bottoms out, the ball ends
   presented → 0.60;
4. the freed ball teleported to a 75 mm hover over the dish (above the judged
   window) → **gravity drop** into the pocket → settle → success, 1.0;
5. persistence: ≥3.3 s fully hands-off, success at every sample →
   `SIM_GEN_SOLVE: SUCCESS`.

Verified on the forge for seeds 0 and 1: non-decreasing SIM_GEN_SCORE
0.00 → 0.15 → 0.60 → 1.00 → 1.00 → 1.00.

## Embodiment argument (Franka, single arm)

One base pose at ~(−0.15, 0, 0) faces all three fixtures (everything within
~0.55 m reach, below z 0.16):

- **jar** (64 mm square body, 0.45 kg): whole-hand wrap grasp well within the
  80 mm jaw span; carried over the spike and pressed — the press force
  (10–25 N downward, guided by the spike/hole funnel with a 10 mm-tolerant
  36 mm hole over an 8 mm pilot tip) is a fraction of Franka's payload, and
  the xy-spring + attitude hold in the solve is exactly a stiff wrist;
- **lid**: never needs to be (and cannot be) grasped;
- **ball** (30 mm): after the press it sits 9 mm proud of the rim in the
  57 mm-wide mouth with the tray fence top level with the rim — a standard
  two-finger pinch from above, then release above the dish;
- **dish**: never needs to move.

## Execution order declaration

scene → solve (verified on forge) → rubric finalized → smoke. There is one
implicit causal order in the task itself — press-before-retrieve — enforced
physically: the ball cannot be pinched until the press has ejected the lid and
raised it out of the bore. Tested exclusively through the forge client; final
artifacts re-verified after the last edit.

## Checks

- solve: forge seeds 0/1 → `SIM_GEN_SOLVE: SUCCESS`, non-decreasing scores,
  ≥3.3 s hands-off hold.
- smoke: forge → `SIM_GEN_SMOKE: ALL PASS 16/16` — settle/no-NaN + stack
  coherence, randomization readback, null policy, INVERSION dump rejected
  (weld holds upside down + two coherent hops; the damped piston cannot
  hammer it open), MECHANISM sub-critical (parked on the spike hands-off
  stays sealed, ≤ engage credit), bypass while sealed (ball in dish earns
  nothing — open-mouth clause load-bearing), seal holds 6 N (the whole stack
  lifted hanging by the weld, rise-limited), seal shears 15 N (threshold
  bracketed; first irreversible break, ordered after all intact-seal probes),
  SEED-strategy end state (lid off, ball captive ⇒ ≤ 0.25, no success),
  near-miss (ball beside dish), wrong object (LID in the dish),
  presented-only press end-state (score caps 0.60), settle gate (rolling ball
  defeats stillness), latched credit survives dismantle, whole-battery
  success()-never-True audit, frames.npz.
