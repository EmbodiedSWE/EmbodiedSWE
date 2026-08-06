# stemware_glide — close the drawer without toppling the glass riding inside it (i50)

**Registered as:** `SCENES["stemware_glide"]`, env `simgen.stemware_glide` (robot="null",
scene-level). **Tier: easy — 1 manipulation stage** (close the drawer), with the
difficulty on the *manner* axis: the closing must be quasi-static. **Execution order:
N/A** (single stage; an optional brace-the-cargo preparatory move is allowed but not
required).

## Seed provenance

- Seed: `libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet`
  (`sim_gen/RoboVerse/roboverse_pack/tasks/libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet.py`)
- Seed plan: push the cabinet's top drawer shut. Its checker is a bare joint-position
  threshold (`joint_pos > -0.01`) — ANY contact that rams the drawer home wins, at any
  speed, and nothing in the scene can be damaged by doing so.

## What changed, and why it is strategically different

The drawer still ends CLOSED — but the drawer now carries **fragile cargo**, and the
judge reads the cargo, not just the joint:

1. **A tall slender stem glass (130 mm × Ø20 mm, uniform) stands loose on the drawer
   floor.** Its topple energy barrier is tiny by construction:
   `m·g·(sqrt(r² + h_com²) − h_com)` ≈ 0.45 mJ, i.e. a critical impact speed of
   **~0.123 m/s** (dry-computed, stored in cfg as `v_crit`, and measured by the smoke's
   speed sweep). The glass RIDES the actuated fixture — it must move, but gently.
2. **The seed's own plan is the tested failing control.** A one-stroke 0.50 m/s shove
   (the way every seed demo closes a drawer) carries 16× the barrier energy: the start
   jerk and the end-stop jolt tip the glass over mid-tray. The drawer ends flush — the
   seed's success condition is fully met — and the episode still fails (score ≤ 0.32).
3. **The required plan is speed-bounded transport, not a push:** glide the drawer
   under ~0.12 m/s (the oracle uses 0.06 m/s = ¼ barrier energy), or brace the cargo
   first. Partial credit only accrues while the closing is CONTROLLED — a per-substep
   `v_gate` on drawer speed gates the progress latch, so ramming banks nothing even
   before the glass falls.
4. **Success is a conjunction the seed never checks:** drawer flush-closed (physical
   back stop, live housing frame, settled) AND glass upright INSIDE the now
   roofed-over tray, settled. A toppled glass may be re-righted only while the drawer
   is still open — after closure the roof covers the tray and the mistake is
   irreversible, which is what makes the care constraint real.
5. **Two permanent anti-cheat latches** (post_step, per substep): `removed` — the
   glass ever leaves the tray volume (kills "take the glass out, close, put it back":
   physically impossible after closure, so only a state-write could fake it — tested);
   `warped` — the tray ever moves > 20 mm in one substep (kills "teleport the drawer
   shut around the glass" — tested with a picture-perfect faked terminal state).

A solver bringing the seed's plan ("push the drawer home") reliably produces
closed-drawer-with-toppled-glass and scores ~0.25; the required plan differs in kind —
regulate the *dynamics* of the actuation to protect a payload — not in parameters.

**Sibling-axis differentiation** (parallel batch, same or nearby seeds): unlike `i33
unjam_drawer` (same cabinet: solver never actuates the drawer; a spring closes it once
a jam is removed), here there is NO spring and NO obstruction — the solver performs the
closing itself, and the challenge is the manner of actuation. Unlike `i22 drawer_stash`
(spring-return + prop bar + deposit cargo), nothing is deposited and no temporary
fixture exists — the cargo starts inside and must merely survive the ride. Unlike `i19
barred_drawer` (drop-bar unlock then open-stow-shut) and `i15 drawer_fetch_restore`
(open, fetch, deliver, restore), there is no access/unlock sub-problem and no
multi-stage transport — one stage, one skill. Unlike `i25`'s fragile-bystander vase
(static object near the workspace), the fragile object here necessarily MOVES with the
actuated fixture — preservation under transport, not avoidance. No dwell timing
(i18/i21), no periodic obstacles (i48), no aperture keying (i14), no set-point
precision goal (i34's rotate-to-bearing). The speed-bounded / impulse-bounded actuation
axis ("gentle manipulation") is claimed by no completed sibling.

## Scene / physics honesty

- Fully procedural primitives: kinematic housing compound (guide walls, back-wall
  closed stop, roof, fascia above a 158 mm lintel) + a JOINTLESS dynamic one-piece tray
  sliding on the ground between the guides (the i15/i33 stack-proven pattern — bind-time
  prismatic joints are dead on this forge image) + a dynamic cylinder glass. No external
  forces, no springs: the null check asserts the drawer is statically stable.
- The topple cliff is real physics, not a scripted latch: the smoke drives the SAME
  kinematic close protocol at 0.06/0.10/0.16/0.28/0.50 m/s and publishes the measured
  stand/topple table around the dry-computed 0.122 m/s cliff; only the ≥4×-energy-margin
  extremes are asserted (0.06 stands 3/3 seeds; 0.50 topples 3/3 seeds). **Measured on
  the forge (run 1, ALL PASS 16/16 in 80 s): 0.06 → stands 3/3, 0.10 → stands,
  0.16/0.28/0.50 → topples (tilt 48–90°, drawer flush each time) — the physical cliff
  sits in (0.10, 0.16), bracketing the analytic 0.122 m/s.**
- The oracle teleport-drags the TRAY (kinematic pos+matched-velocity writes each
  substep); the glass is never touched — it rides on real friction. success()/score()
  judge settled physical outcomes in live body frames (housing yaw randomized ±25° about
  −45°).
- Sleep thresholds zeroed on tray and glass (kinematic drags raise no wake events — the
  i33 forge lesson). Contact offsets explicit 2 mm; standing-glass top clears the lintel
  by 17.5 mm.
- Rubric: cheat latches → 0.03 flat; glass toppled → 0.25·f; leaning → 0.40·f; standing
  → 0.65·max(f, banked-controlled-f); 1.0 iff success; closing fraction has a 2%
  deadband so the null policy reads exactly 0.

## Check list (smoke battery, 16 checks)

1. reset settles finite, glass standing, drawer open in band, score 0
2. randomization is real: housing xy + yaw (readback, 3 seeds)
3. initial opening + glass seat vary (readback, 8 resets)
4. null policy: drawer never creeps, glass stands, score exactly 0
5–7. oracle glide (0.06 m/s) reaches success() on seeds 0/1/2
8. milestone scores land in bands (mid ≈ 0.32, near-flush ≈ 0.56, success 1.0)
9. rubric strictly increases start → mid → near → success (all seeds)
10. negative A (seed strategy): 0.50 m/s shove rams the drawer home but topples the
    glass → no success, score ≤ 0.32
11. near-miss: controlled glide stopped 30 mm short → glass fine, not closed, mid score
12. tolerance probe: 1 mm gap closed, 20 mm gap not closed (authored poses)
13. removal latch: in-tray repositioning free; lifting the glass out latches
14. negative B: glass removed, drawer closed, glass teleported back in standing —
    perfect-looking terminal state refused (score ≤ 0.05)
15. negative C: tray+glass teleported to the flush terminal state in one write —
    `warped` latch refuses it (score ≤ 0.05)
16. calibration cliff: 0.06 m/s stands+succeeds 3/3, 0.50 m/s topples 3/3, full
    stand/topple speed table printed against dry-computed v_crit
