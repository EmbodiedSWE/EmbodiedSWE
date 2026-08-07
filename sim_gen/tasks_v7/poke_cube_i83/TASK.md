# poke_cube_i83 — Keystone Cascade

Stand the red keystone on the blue trigger pad, then poke it over so it falls into
the tunnel mouth and sets off a hands-off chain — six dominoes, a hammer, and a
caged ball that must end up potted on the kiosk pit floor (scene
`keystone_cascade`, env `simgen.keystone_cascade`).

## Seed provenance

Seed: `maniskill/poke_cube` — grasp a peg tool and poke a red cube a few
centimeters across the plane into a painted goal region. One planar
non-prehensile act, judged by an xy-distance readout on the poked object itself.

The seed's end state (the poked cube resting at a floor goal spot) is **not
expressible** here — there is no floor goal region at all. The nearest naive plan
from the seed's family — slide the keystone flat across the floor to some spot —
is constructed as a settled state in smoke check 6 and scores ~0.

## What the task is

A procedural tabletop arcade sits along a per-episode "alley" axis (anchor
jittered ±3 cm in xy, yaw ±25°): a blue trigger pad, then a slick-walled tunnel
whose mouth (58 mm gap) is too narrow for the 63 mm parallel-jaw hand, then a run
of six 18×40×80 mm dominoes at 56.4 mm pitch, then a 140 mm hammer, all leading
to a kiosk with a 52×145 mm letterbox slot in its front wall. Inside the kiosk a
14 mm ball is perched on a 48 mm column with a three-sided retaining lip; the
only opening faces the letterbox. The red keystone (22×40×140 mm) starts lying
flat in one of three shuffled scatter slots on the open floor; two same-size tan
decoys lie in the other slots.

The pad is ARMED (a latched bit) only after the keystone stands still on the pad
with the whole run intact for 12 consecutive substeps — a fly-through or a
knocked-over arrival never arms it, and arming after the run is disturbed is
impossible. Goal: with the pad armed, topple the keystone downstream so it falls
through the tunnel mouth and strikes the first domino's top corner; the run
relays domino to domino, the last domino fells the hammer, the hammer swings
through the letterbox and its shaft sweeps the ball off the perch onto the pit
floor. Success = armed + keystone down + all seven run pieces down + ball at
rest on the pit floor.

Geometry gates every shortcut and is asserted in `__post_init__`: the decoys and
dominoes (82 mm diagonal) are too short to bridge the pad-to-run gap — only the
keystone (142 mm diagonal) reaches the first domino's top corner from the tunnel
mouth; the hand cannot enter the tunnel or the kiosk; the perch lip holds the
ball against any table-level disturbance except the hammer shaft arriving
through the letterbox.

## Strategic difference

- **vs. the seed**: the seed pokes a cube and reads the poked object's own xy
  position. Here the poke is the FIRST DOMINO of a five-stage energy relay — the
  predicate reads the state of bodies (the run, the hammer, the caged ball) that
  the robot never touches and physically CANNOT touch (tunnel and kiosk are
  hand-proof by asserted geometry). The reward-bearing motion is transitive
  contact three hops removed from the fingertip, and the poke only counts if a
  latched precondition (standing placement on the pad, run intact) was met first
  — an ordering constraint the seed does not have.
- **vs. corpus tasks read**: no corpus task's goal is a multi-body chain reaction
  behind hand-proof guards. `pull_cube_i20` (beam scale) converts placed mass
  into a static torque on one pivot — no kinetic relay, no ordering latch.
  `push_cube_i30` (silo scoop) and `pull_cube_tool_i1` (carousel ferry) read
  containment of the transported object itself. `obstacle_i17` (skittle gallery)
  knocks targets by a thrown/rolled projectile the robot launches directly —
  one hop, no armed precondition, nothing hand-proof. The pick-place and
  articulation families read containment or built-joint angles by direct grasp.
  **No corpus task requires arming a latch, then triggering a hands-off
  domino-hammer-ball cascade whose scored bodies are all unreachable.**
- The mechanism is physically honest and margin-checked in `__post_init__`:
  keystone tip reaches D1's top corner with ≥8 mm margin, each domino reaches
  the next with margin, the felled hammer's shaft sweeps the perch top with
  ≥8 mm margin, and the keystone is too short to hang up on the tunnel roof
  edge. The pad is grippy (μ≈0.6) so a poke TIPS the keystone rather than
  sliding it.

## Solution outline (solve.py — the legitimacy certificate)

Teleportation is TRANSPORT ONLY: the keystone is teleported from its floor slot
to a standing pose just above the pad (exactly what a pick-carry-place delivers)
with zero velocity and RELEASED; it settles under gravity and the armed latch
fires on its own after the 12-substep stillness streak. The poke is an APPLIED
WRENCH — a small torque about the alley cross-axis, the honest stand-in for a
fingertip push on the keystone's upper half — held only until the keystone
passes its ~9° balance angle, then cleared. Everything the rubric reads happens
hands-off after that. Nothing inside the tunnel or kiosk is ever teleported or
wrenched; the run, hammer, and ball are never touched by the solver at all.

The wrench frame on this stack is ambiguous, so the poke torque is CALIBRATED:
four candidate encodings of "torque about the alley +y axis" are probed at
0.020 N·m from a saved state (state restored after each probe) and the encoding
that actually tips the keystone downstream is used — the frozen-reference body
frame wins on every seed tried — then escalated (0.03 → 0.24 N·m) until the
keystone passes its balance angle. A single 0.03 N·m poke suffices in practice.

`SIM_GEN_SCORE` prints at every phase boundary (non-decreasing, asserted:
0.00 → 0.25 → 0.25 → 1.00 → 1.00), then ≥3.3 simulated seconds hands-off
persistence before `SIM_GEN_SOLVE: SUCCESS`.
**Verified on the forge: seeds 0, 1, 2 all SUCCESS (~18–23 s wall each),
covering both sampled keystone slots and yaws +23.6° / +8.6° / −21.3°.**

## Rubric

`score()` is stateless apart from the armed latch: 0.25 · armed + 0.30 ·
run-toppled-fraction (per-piece thresholds; the felled hammer rests propped on
the perch lip at up_z≈0.63, so it uses its own threshold 0.78 with an asserted
dead band) + 0.10 · ball-in-pit, capped at 0.65; 1.0 iff `success()`. Null
policy ≈ 0: nothing is armed, the run stands, the ball is perched.
`success()` = armed latch + keystone down + run fraction 1.0 + ball at rest on
the pit floor. The latch never clears; toppled pieces stay toppled; the potted
ball stays potted — so score is monotone along the intended solution.

## Embodiment argument (Franka, parallel-jaw)

- The keystone is a 22×40×140 mm, 100 g prism — grasp the 22 mm faces
  (comfortably inside the ~63 mm jaw, asserted) anywhere on its upper half.
  Decoys are the same stock, so the only pick is a color-directed pick.
- Placement tolerance is generous and latched, not instantaneous: the pad
  window is x −45/+50 mm × y ±40 mm around the pad center for a 22×40 mm
  footprint, and the armed latch waits for stillness — a slightly off-center or
  wobbling set-down still arms once it settles.
- The poke is a fingertip push on the keystone's standing upper half: 0.03 N·m
  about the base ≈ 0.4 N applied at 70 mm height — trivially within reach and
  force range, applied OUTSIDE the tunnel (the pad is 35 mm upstream of the
  mouth; the keystone's upper half stands proud of the 150 mm roof line
  upstream of the wall).
- Workspace: the arcade anchor is within ±3 cm of the origin, yaw ±25°; the
  scatter slots lie 0.18–0.44 m along the alley and ±0.25 m across it. With the
  base at ≈ alley (0.10, −0.35) facing the pad, the pick, the place, and the
  poke all lie within a 0.65 m reach; the robot never needs to reach past the
  tunnel mouth (0.535 m along-alley).
- Execution order: the ONLY ordering constraint is the task's own latch —
  stand the keystone on the pad (run intact) BEFORE poking it. The scatter
  pickup order is free; the decoys need never be touched.

## Checks (smoke.py — rejection battery, 16/16 PASS on forge, frames.npz recorded)

1. settle + no-NaN: hands-off from reset everything rests — run standing, ball
perched ABOVE the pit-z gate; 2. keystone flat in a scatter slot, not armed,
score ≤ 0.02, no success; 3. randomization readback seeds 21–32: keystone slot
shuffles (≥2 slots seen), anchor xy and yaw spread; 4. scatter slot positions
vary; 5. null policy 300 steps → score ~0; 6. seed strategy (poke/slide the
keystone flat across the floor to a spot — the seed family's move): settled
there → score ≤ 0.02, nothing armed, run intact; 7. decoy bridge: a DECOY stood
on the pad and flicked downstream topples (probe verified non-vacuous) but is
too short to reach D1 — run intact, nothing armed; 8. near miss: the keystone
poked over 75 mm UPSTREAM of the pad — falls, reaches nothing, not armed;
9. wand bypass: flicking D1 directly runs the whole cascade (run down, ball
potted — probe non-vacuous) but the latch never armed → no success, score
≤ 0.42; 10. out-of-order (same episode): placing + poking the keystone AFTER
the run is down cannot arm → no success; 11. ball teleport bypass: writing the
ball to the pit floor without the cascade → success False, score ≤ 0.12;
12. sideways poke: honest arming, then a flick along the WRONG axis (alley +x)
→ keystone falls beside the mouth, run intact, score ∈ [0.20, 0.27] (armed
credit only); 13. fly-through: the keystone written moving 0.8 m/s through the
pad window transits the window but never arms (streak latch); 14. geometry
audit: alley readback of tunnel mouth vs hand width, decoy reach vs D1 gap,
keystone reach margin, perch/ball pose; 15. rejection audit — success() never
True anywhere in the battery (the "accepts" evidence is solve.py itself);
16. final no-NaN.

Run (forge):
`python -u -m simgen_tasks.poke_cube_i83.solve --headless [--seed N]`
`python -u -m simgen_tasks.poke_cube_i83.smoke --headless`
