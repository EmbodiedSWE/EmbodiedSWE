# push_button_i217 — dock the recoil button cartridge, then press it latched

## Seed provenance

Seed task: `rlbench/push_button` — a button sits on a table; the robot pushes it down;
success the instant the button is depressed.

## What changed (and why it is strategically different)

The seed's entire challenge is one poke at a fixed fixture. Here the button is **not a
fixture**: it is a free-standing, spring-loaded **cartridge** (0.75 kg housing, 0.12 kg
horizontal plunger with a red 55 mm cap, 0.03 kg gravity pawl riding the plunger stem)
standing loose on the floor, and the fixture — a kinematic U-pocket **dock** (anvil) —
stands elsewhere, both with randomized poses. Three structural differences from the seed
and from every other examined task:

1. **The seed's own strategy is physically punished.** The spring's full-stroke load
   (~7.8 N) exceeds the cartridge's ground-sliding threshold (~3.5 N), so poking the cap
   where the button stands just shoves the whole cartridge across the floor — the plunger
   barely compresses (smoke check 4 measures 15 mm peak vs the 26 mm success line while
   the cartridge slides 2.5 m). The button can only be pressed after the cartridge is
   **docked** with its back against the dock backwall, which reacts the press force.
2. **The press outcome is an irreversible internal mechanism, not a pose.** Full stroke
   (34.5 mm) carries a stem groove under the gravity pawl; the pawl drops ~10 mm in; on
   release the spring sets the groove step against the pawl and the button stays latched
   hands-off at ~28.5 mm depth. A partial press (even 25 mm) springs all the way back
   out. No other examined task has a spring-return + drop-pawl click mechanism.
3. **Success is a conjunction of place AND internal state**: seated in the dock pocket
   (position + yaw window) ∧ latched depth ≥ 26 mm ∧ pawl dropped ≥ 7 mm ∧ settled,
   sustained 60 substeps. Latching the button *outside* the dock (constructible — smoke
   checks 9 and 10 do it for real) scores ~0: the task is dock-then-press, not press.

## Teleport solution outline (solve.py — passed on seeds 0 and 1)

Teleportation is transport only; every load-bearing interaction is contact dynamics via
the scene's world-frame probe buffers (frame-encoded plant-side in `post_step`):

1. **Transport**: one rigid transform moves the cartridge TRIO (cartridge + plunger +
   pawl, preserving internal state) to a staging pose 10 cm outside the dock seat,
   dock-axis aligned, zero velocity (asserted not seated). Approach latch fires: 0.150.
2. **Dock by contact**: capped velocity-servo force on the cartridge (kax=100,
   v_des=0.12 m/s, cap 12 N — stall authority ≫ the ~3.5 N sliding threshold; K·dt/m =
   0.93 < 1) plus a small lateral centring term pushes it into the pocket to the
   backwall (measured seat x = 0.059, exactly the design value). Force off, settle,
   `seated_now()` verified. Score 0.450.
3. **Press by contact**: capped force on the plunger along the press axis (spring
   feedforward + velocity servo, cap 16 N) to the hard stop; held ~48 steps while the
   pawl drops into the groove under gravity; release; the spring sets the step against
   the pawl. Measured hands-off latch: depth 28.5–28.6 mm, pawl_dz −13.8 mm. Score
   0.900 → 1.000.
4. **Persistence**: success awaited, then ≥ 3.5 more simulated seconds hands-off;
   success still holds; `SIM_GEN_SOLVE: SUCCESS`.

Score sequence both seeds: 0.000 → 0.150 → 0.450 → 1.000 → 1.000 → 1.000 (asserted
non-decreasing).

## Embodiment argument (single Franka, 80 mm parallel jaw)

Base at ≈ (−0.55, 0, 0) facing the workspace covers both the cartridge spawn annulus
(0.38–0.50 m) and the dock (~0.18 m ± jitter). One plausible motion sequence:

- **Carry**: top-down grasp of the 20 mm-square yellow **handle bar** (x span 90 mm,
  z 187–207 mm — well inside the 80 mm jaw, high enough that fingers clear the roof).
  Lift, carry, set down 10 cm in front of the dock mouth — exactly the solve's
  staging pose.
- **Dock**: push the housing back face (or keep the handle grasp) horizontally ~10 cm
  into the pocket; the 25° flared wings funnel ±12 mm/±8° error; required force ~4–12 N.
- **Press**: horizontal fingertip push on the red 55 mm cap at z ≈ 75 mm, 34.5 mm
  stroke, 10–16 N — reacted entirely by the dock backwall, so the cartridge does not
  slide. Release; the latch holds itself.

All forces used by the solve (≤ 16 N) and all grasp widths are within Franka limits;
every contact surface is a flat box face at graspable heights.

## Execution order

No strict sub-goal ordering beyond the physical one: the button must be **docked before
the press can latch** (undocked presses shove the cartridge; wrong-way presses latch but
never score). Shoving the cartridge toward the dock instead of carrying it is
legitimate — credit is for outcomes (approach, seat, depth, click), not for style.

## Checks (smoke.py — rejection-only battery, 13 checks)

1. settle/no-NaN: reset settles finite, plunger out, pawl up, not seated, score ~0
2. randomization readback: anvil xy/yaw, spawn distance/bearing, cartridge yaw all vary;
   spawn distance always in band; trio assembled at every spawn
3. null policy: 360 idle steps → score ~0, no success
4. seed strategy: 4.5 N cap poke on the free cartridge shoves it ≥ 40 mm (probe real),
   peak depth < 26 mm line, pawl never drops, no success
5. spring-back: released poke → plunger returns out, nothing latched
6. seated-no-press construct: 0.45-band credit only, no success
7. partial press (~15 mm) then release: springs back, pawl up, sub-click credit only
8. latch retention: carrying the credited cartridge out keeps the latched score,
   success stays False
9. latched-but-unseated construct on open ground: mechanism really persists hands-off
   (depth ≥ 26 mm, pawl in groove) yet scores ~0 — seat required
10. wrong-way dock: cartridge reversed, cap braced on the backwall, 12 N push really
    latches the button (probe real) but seated=False → no success, approach band only
11. fake latch: teleporting only the pawl down (plunger out) is depenetrated back onto
    the stem — no latch (planted pose verified from the physx view pre-step)
12. no accidental success anywhere in the battery
13. final: every body state finite

Verdict line: `SIM_GEN_SMOKE: ALL PASS 13/13`. Frames recorded to `frames.npz`.
