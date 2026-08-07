# blade_guard — the knife is IMMOVABLE: sheath it by threading a guard sleeve DOWN OVER the blade

**Task id:** `put_knife_in_knife_block_i312`
**Scene:** `blade_guard` (`register_env("simgen", ...)` → `simgen.blade_guard`, robot="null")

## Seed provenance

`rlbench/put_knife_in_knife_block`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/put_knife_in_knife_block.py`).
Seed strategy: pick the MOVABLE knife off the chopping board by its handle and insert
it tip-first DOWN into a slot of the fixed knife block. Peg-into-hole with the peg in
the hand; the block is passive scenery.

## What changed, and why it is strategically different

The roles are INVERTED and the seed's plan is physically impossible here. The knife
is clamped BLADE-UP in a kinematic bench vise and CANNOT BE MOVED at all — "pick up
the knife", the first step of the seed's plan, does not exist as an action. What
moves instead is the APERTURE: a green guard sleeve (a rectangular tube open at both
ends, 2.2 x 6.2 cm channel, 4.6 x 8.6 x 13 cm outer) standing on the table. The
solver must bring the hole to the peg: pick the sleeve by its outer walls, carry it
above the 23-cm-high blade tip, rotate it so the channel's long dimension lines up
with the blade's per-episode RANDOMIZED yaw (the 4.5 cm blade width does not pass the
2.2 cm channel dimension — the tube fits one way modulo 180 deg), thread it down over
the blade through contact, and seat it flat on the vise shoulders with the blade
fully sheathed. A solid ORANGE block with identical outer dimensions is a decoy that
can only balance on the tip.

A solver therefore needs a different PLAN, not different numbers:
1. grasp a tube by its outer walls (not a knife handle) and select it against an
   identically-sized solid decoy (hollow-vs-solid discrimination);
2. align a MOVING aperture over a FIXED peg — the visual-servoing target is inverted
   relative to the seed;
3. actively match the blade's randomized yaw before descending (an orientation-
   matching step the seed never asks for; the seed's block slots are axis-fixed);
4. seat ON structure: success is the sleeve standing on the vise shoulders around
   the blade, not an object dropped into a cavity.

Different code structure too: a kinematic fixture re-posed with randomized yaw, a
tip-point containment test evaluated in the SLEEVE's body frame, a yaw-error clause
folded modulo 180 deg, an insertion-depth running max, and a seat-height band — none
of which exist in the seed (three USD assets + a recorded trajectory, no checker).

## Teleport-solution outline (solve.py)

- P0: reset (seed via `env.reset(seed=...)` after build), settle; assert sleeve and
  decoy standing on the table, score ~0; print the vise xy/yaw readback.
- P1 (transport ONLY): one pose write moves the sleeve across free space to a hover
  with its bottom face 2 cm ABOVE the blade tip, centered on the blade axis,
  yaw-matched. The solve asserts the blade is NOT in the channel and no seat/success
  gate is satisfied by the teleport. The vise is never teleported outside reset;
  the decoy is never touched.
- P2 (contact dynamics — the core interaction): a velocity-servoed vertical force
  (-0.25 m/s, |F| <= 2.5 N) lowers the sleeve; the fixed blade enters the channel and
  guides it through REAL contact for ~8.5 cm of overlap (the solve asserts tip-entry
  happened DURING the descent). The force is CUT ~2.5 cm above the seat; the final
  drop onto the shoulders and settling are pure gravity + contact.
- P3: hands-off persistence >= 3.3 simulated s, then `SIM_GEN_SOLVE: SUCCESS`.

`SIM_GEN_SCORE` is printed at every phase boundary and asserted non-decreasing
(all credit is latched). Passes seeds 0 and 1 on the forge (see run logs).

## Embodiment argument (single Franka, parallel jaw, OSC; base at the origin)

- Plausible base pose: (0, 0, 0), facing +x. Vise centre at x ~ 0.52 m, object slots
  at (0.36, +/-0.20); every required contact lies in x 0.33-0.58 m, z 0.0-0.27 m —
  inside the 0.45-0.71 m comfortable envelope for the vise work, generous elsewhere.
- Sleeve (the only object that must move): side pinch across its NARROW outer faces —
  4.6 cm across, well inside the ~8 cm jaw span, with 13 cm of wall height to grip
  anywhere along; both ends of the tube are open so a top-down approach with fingers
  straddling the narrow walls also works. Grip in the sleeve's upper half so at
  seating (sleeve bottom at 0.12 m) the fingertips stay >= 6 cm above the vise
  shoulders — no low-clearance contact anywhere.
- Alignment: yaw-match is a pure wrist rotation (blade yaw within +/-35 deg; joint-7
  range is far larger). Required precision: the channel leaves ~7 mm of free play
  per side around the 8 mm blade thickness and ~8 mm around the width; threading
  works up to ~18 deg of yaw error (rubric tolerance 25 deg). Both are far above
  closed-loop OSC noise, and the blade tip self-centres the descending tube.
- Descent: 11 cm straight-line lower onto a 23-cm-high tip in open air — no
  overhangs, no near-ground contact; release above the seat and gravity finishes.
- Decoy: never needs to be touched. Knife/vise: immovable fixture, never touched.

## Execution order

No ordering constraint is declared; the task is a single threading interaction.
(Yaw-matching must happen before the descent completes, but that is forced by the
geometry, not asserted as an order.)

## Check list (smoke.py — rejection only; solve.py is the acceptance evidence)

1. settle/no-NaN: sleeve and decoy stand on the table, vise in place, all finite.
2. score ~0 at reset, no success.
3. randomization readback: sleeve/decoy slot swap + xy jitter across 8 seeds.
4. randomization readback: vise xy and yaw vary (> 10 deg spread, within the band).
5. null policy: 240 idle steps → score <= 0.02, no success.
6. SEED-ANALOG / near-miss: sleeve standing ON the vise shoulders BESIDE the blade —
   seat-height without containment → no success, score <= 0.45. (The seed's literal
   end state — the knife moved into a block — is N/A here: the knife is immovable by
   construction; "brought to the block but not threaded" is its closest analog.)
7. near-miss (load-bearing orientation tolerance): sleeve dropped over the tip
   rotated 90 deg — cannot thread, balances or tumbles → never seated, no success.
8. wrong object: solid ORANGE decoy dropped yaw-matched over the tip → no success,
   sleeve untouched, score ~0.
9. wrong place: sleeve lying on its side on the table next to the vise → no success.
10. latched credit: a genuine partial insertion latches lift/entry/depth credit;
    removal back to the table does not evaporate it; never success.
11. hover is not seated: sleeve HELD still 2 cm above the seat with the blade deep in
    the channel → seat band rejects at every step (no fly-through/hover success).
12. rejection audit: success() never True anywhere in the battery.
13. final no-NaN.

## Rubric

score = 0.10·lifted (sleeve ever clearly off the table, latched)
      + 0.15·tip-entry (blade tip ever inside the channel, latched)
      + 0.20·insertion depth (running max of tip height above the sleeve bottom /
        blade length, gated on the tip being in the channel)
      + 0.25·seated (ever seated+centered+upright+yaw-matched+contained, latched)
      capped at 0.85; exactly 1.0 iff success() holds live.
success = sleeve root within 1.5 cm of shoulder height, within 2 cm of the blade
axis, upright (r33 >= 0.95), yaw error <= 25 deg (mod 180), blade tip >= 9 cm above
the sleeve's bottom face inside the channel cross-section, at rest. Null policy = 0
(the sleeve starts on the table: never lifted, never entered, no depth, no seat).
