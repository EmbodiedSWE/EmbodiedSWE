# pull_cube_tool_d1 — bridge the trench (tool as structure, not as hook)

| Field | Value |
|---|---|
| Seed | `RoboVerse/roboverse_pack/tasks/maniskill/pull_cube_tool.py` (ManiSkill PullCubeTool) |
| Mutation | The tool's **role** changes from *hook* to *structural bridge*, and the objective grows a full delivery-and-cleanup chain. The out-of-reach cube sits on a far platform separated from the near platform by an open trench; the seed's plan — drag the cube along the surface toward the base — drops it into the pit. Instead the plank must be laid **across** the trench, the cube must cross **on** the bridge, be deposited in a walled bin, and the plank must then be returned to its rest pad. |
| Stages | **4** (ordered): ① bridge the trench with the plank → ② move the cube across the trench on the bridge → ③ deposit the cube inside the bin → ④ return the plank to the rest pad. |
| Ordering | This task **requires a specific order.** ② is physically impossible before ① (no support over the pit); ③/④ credit is rubric-gated on the ②-crossing latch, and success demands the plank *back on the pad* at the end, so ④ can only complete after ②③. Wrong-order execution (park the plank on the pad first, then attempt the crossing) is a negative control and fails. |

## Why this is strategically different

The seed's whole plan is *tool-mediated dragging*: hook the L-tool behind the cube and
pull it along a continuous table surface into the reach region; success reads only the
cube's final xy. Here:

- **The surface is not continuous.** Executing the seed's strategy — sliding the cube
  along the surface toward the near side — loses support at the trench and the cube
  falls into the pit (negative control A; enforced by gravity, not by a rule).
- **The tool is load-bearing, not force-transmitting.** The solver must reason about
  the plank as a structure: it must *span* the gap (both ends overhanging the trench
  edges — a cantilevered near-placement tips under the cube's weight, negative control
  E) before the cube ever moves.
- **"Get the cube to my side" is not success.** The cube must (a) have crossed the
  trench while supported by a genuinely spanning bridge (a latched contact-over-trench
  event — flying the cube over the trench without a bridge is negative control B),
  (b) end settled inside a walled bin, and (c) the plank must be back on its rest pad
  (leaving the bridge in place is negative control D). A pull-into-reach solver
  satisfies none of these.

So a solver needs a different plan — place structure, transport over it, deliver,
clean up — not different parameters for a drag.

## Semantics

- **Scene**: two fixed platforms with tops at z=0, separated by a 12 cm open trench
  (pit floor 12 cm down). A 4 cm red cube starts on the far platform; a 22×7×1.2 cm
  plank starts on the near platform; a walled bin (10 cm interior, 3 cm walls) and a
  green rest pad sit on the near platform.
- **Instance distribution**: cube xy, plank xy+yaw, bin y, and rest-pad y are all
  randomized (with min-separation rejection so the furniture never overlaps).
- **Success** (all three): the ②-crossing latch has fired (cube touched the plank while
  over the trench *and* the plank was spanning at that instant); the cube is settled
  inside the bin (per-axis |Δxy| < 3 cm, at rest height, slow); the plank is settled on
  the rest pad (center within 5 cm, on the table, slow).
- **Score** (latched stage credits + gated shaping, `min(0.95, ·)` unless success):
  0.20·bridged_ever + 0.25·crossed_ever + 0.25·delivered_ever + 0.10·best-approach-to-bin
  (gated on crossed) + 0.15·(plank on pad, gated on delivered). Latches are cleared in
  `reset_instance` and carried through `get_state`/`set_state`, so credit never
  evaporates under correct behavior; 1.0 exactly iff `success()`.

## Seed-strategy negative control

The seed's strategy **is expressible** in this scene (drag the cube along the surface
toward the near platform) and is implemented as negative control A: the cube loses
support at the unbridged trench and ends in the pit — success False, score ~0. Even a
kinematic drag that somehow crossed would fail: no spanning-bridge contact ⇒ no
crossing latch ⇒ delivery credit and success both unreachable.

## Checks (smoke.py)

settle/no-drift (cube + plank) · no-NaN · physics readback (plank long enough to
out-span the trench; cube/plank masses as configured — the cantilever control depends
on them) · determinism · randomization-is-real (cube/plank/bin/pad all vary) ·
null-policy fails with score ~0 · oracle succeeds on 3 seeds · rubric monotone across
all 4 stage boundaries on 3 seeds · success stable over a 300-step hold · negative
controls: (A) the seed's own drag strategy — cube ends in the pit, score ~0; (B)
fly-the-cube-into-the-bin without ever bridging — cube *is* physically in the bin but
success False, score ~0; (C) wrong order — plank parked on the rest pad first, then
the crossing attempted — fails; (D) full delivery but the plank left in place as a
bridge — fails with partial credit strictly between 0.5 and 1.0; (E) cantilevered
"bridge" (near end short of the trench edge) — tips under the cube, crossing latch
must NOT fire · calibration sweep: bin drop-offsets at {0, 0.6, 3.0}× the bin
tolerance — in-tolerance offsets succeed, the far offset lands outside the bin and
fails, pinning the knee at the bin wall.
