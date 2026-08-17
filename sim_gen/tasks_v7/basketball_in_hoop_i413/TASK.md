# basketball_in_hoop_i413 — CaromCourtScene ("aim the deflector, bank the ball into the bay")

## Seed provenance

Seed task: `rlbench/basketball_in_hoop`. The seed's plan is direct transport: grasp the
ball, carry it over the hoop, open the jaw — gravity delivers. The hand touches the ball
for the whole trajectory, "aiming" means hovering over the ring, and the goal test is a
sphere passing a fixed hoop plane.

## What changed and why it is strategically different

The delivery keeps the seed's spirit (get a ball into a raised goal receptacle by way of
a ballistic hand-off to gravity) but the *strategy space* is disjoint:

1. **The robot can NEVER touch the ball.** The ball starts outside the play volume,
   parked on a launch chute behind a gravity portcullis gate, and once released it is
   inside a **fully roofed, sealed circular court** — the only roof opening is a 26 mm
   shaft slot a 60 mm ball cannot pass (asserted at import). The seed's whole plan
   (carry it over and drop it in) is structurally impossible; the smoke battery drops
   the ball from above both roofs and proves it never gets in.
2. **Delivery is indirect and two-stage.** The hand operates two proxies that both stand
   outside the sealed volume: an overhead **pointer bar** (rigid with a hidden deflector
   blade under the roof, on a damped vertical-axis revolute) and the **gate knob** (a
   damped vertical slide; gravity re-closes it). The ball's entire trajectory — down the
   15° chute, through the wall port, a restitution-0 **carom** off the aimed blade, and
   a roll out through the orange doorway into a sealed exterior bay — is free physics.
3. **Aiming is continuous, not positional.** The required blade bearing `theta_star` is
   sampled from a continuous range and the whole delivery geometry (bay, doorway,
   beacon, both wall gaps) is re-posed from it every episode. The visual cue is a red
   beacon standing exactly on the correct pointer ray, over the bay.
4. **The order is physically forced** (declared here, as required): **aim FIRST,
   release SECOND.** A ball released before aiming crosses the court on the port line,
   misses the parked blade, and dies against the far wall at radius ~0.27 m — outside
   the blade's 0.17 m sweep (asserted at import) and unreachable under the roof.
   Nothing re-cocks the chute; the episode is spent. Smoke check 6 constructs exactly
   this run and shows that even a *correct aim performed afterward* earns partial
   credit only.
5. **The stopping problem is the skill.** The vane has no detent; the rubric's aim
   latch requires the pointer near the bearing AND slow for 24 consecutive substeps.
   Sweeping through the beacon ray at speed latches nothing (smoke check 8).

No task in the corpus I inspected uses an aim-a-deflector-then-release-a-captive-ball
scheme (nearest neighbours: `press_switch_i161` sets dial angles but moves nothing else;
ballistic-vault tasks aim a turret but the robot loads the projectile by hand).

## Solution outline (demonstrated by solve.py; forge-verified seeds 0, 1, 2)

1. **Aim** — torque P-servo on the pointer bar (τ = clamp(0.6·err, ±0.45 N·m); the
   joint's 0.35 N·m·s/rad damper is the D term, heavily overdamped). Release only when
   within 1.5° AND slower than 0.05 rad/s; wait 0.33 s for the slow-gate latch.
2. **Release** — hold ~1.5 N up on the gate knob (weight 0.49 N, D6 stop caps travel at
   85 mm). The ball rolls out from rest, drops ~4.6 cm down the chute, enters the court
   at ~0.8 m/s. Drop the gate once the ball is inside; gravity re-closes it.
3. **Hands off** — the ball caroms off the aimed blade and rolls through the doorway
   into the bay; nothing is driven. After success() first holds, solve keeps simulating
   3.5 s with all drive buffers asserted zero, then prints the verdict.

Nothing is teleported at any point in solve.py — there is no transport in this task.

## Embodiment argument (single Franka, 8 cm parallel jaw, OSC, one base pose)

Base the arm between the chute and the court edge (e.g. at (−0.45, +0.25, 0)); both
controls are within a 0.75 m reach and above all static geometry. The pointer bar is a
17 cm × 2 cm × 2 cm beam 16.5–18.5 cm off the ground: a comfortable side-grasp for an
8 cm jaw, and the aim servo is a pure wrist-roll about the shaft axis — the 0.45 N·m
torque cap used by solve.py corresponds to ~2.6 N of fingertip force at the bar tip.
The gate knob is a 2.8 cm cube 14–17 cm off the ground; lifting it 8.5 cm against a
0.49 N weight plus a 3 N·s/m damper is a trivial vertical jaw motion, and the hold force
(1.5 N) is far below any grip limit. Neither control requires regrasping; the two
manipulations happen at way-separated stations reachable from the single base pose, in
the declared order.

## Execution-order declaration

**Aim the pointer bar first; lift the gate second.** The scene physically enforces it
(see point 4 above); the rubric's latch chain (launch gated on aim, approach gated on
launch) mirrors it, so out-of-order execution caps well below full credit and can never
reach success.

## Rubric (latched, anchored in the demonstrated trajectory)

- 0.20 — aim latch: pointer within 7° of `theta_star` AND |ω| < 0.30 rad/s for 24
  consecutive substeps;
- 0.25 — launch latch (gated on aim): ball inside the roofed court;
- 0.25 — approach latch (gated on launch): ball within 12 cm of the computed carom
  exit point M;
- 0.30 — success(): ball inside the bay box (bay-frame x' ∈ [0.035, 0.170],
  |y'| ≤ 0.055, z ≤ 0.09) and slow. score == 1.0 iff success; ~0 for the null policy.

## Checks (smoke.py — forge: `SIM_GEN_SMOKE: ALL PASS 13/13`)

1. settle — clean reset readback (vane AT θ0, gate closed, ball parked, score ~0);
2. readback — beacon azimuth IS the sampled θ*, bay stands at the exit point M
   recomputed independently from the float carom formula;
3. random — θ* and θ0 via physical READBACK vary across 6 seeds, bay moves, bands hold;
4. null — 2.5 s of nothing: ball still parked, score < 0.05;
5. seed strategy — ball dropped from above the bay and above the court lands ON the
   roofs, never enters either volume;
6. out of order — unaimed release strands the ball (never within 12 cm of M), gate
   re-closes, late correct aim earns ≤ 0.45, no success;
7. wrong bearing — vane rested 20° off the ray: aim never latches, carom misses, ~0;
8. fast sweep — bang-bang sweep through the ray 4×, peak in-band |ω| > 2× slow gate:
   nothing latches;
9. near misses — settled placements just short of the band, in the court mouth, and on
   the bay roof: all rejected;
10. sealed bay — solve-scale pushes at the back and side walls never get in;
11. exactness — full correct strategy → success() and score == 1.0, stable 2 s later;
12. latch — removing the ball revokes success; latched 0.70 remains;
13. frames — ≥ 20 video frames saved (frames.npz).

Forge evidence: solve `SIM_GEN_SOLVE: SUCCESS` on seeds 0, 1, 2 (seeds 0 and 1 re-run
after the final scene edit); smoke 13/13.
