# SO101 elbow-servo assembly — handoff

Continue the SO101 assembly task in **CoSiGen** (a robobench suite on Isaac Lab / PhysX).
This doc is self-contained — you should not need the prior chat history.

## The task
Mount the SO101 **elbow servo (STS3215)** into the arm's **upper_arm** pocket and fasten it with an
**M2 screw** driven by a **power screwdriver** — a rule-based mechanic (no thread sim). A `NullRobot`
smoke test scripts the whole thing linearly (no policy yet) to prove the scene physics work.

## Where things live
- Scene: `robobench/suites/assembly/scenes/so101_assembly.py` (class `SO101AssemblyScene`, cfg `SO101SceneCfg`)
- Smoke: `robobench/suites/assembly/scripts/so101_smoke.py`
- Registered env: `assembly.so101` (in `scenes/__init__.py` + `configs/envs.py`), loaded by name.
- Run (uv venv is `.venv` in the repo root):
  ```
  .venv/bin/python -m robobench.suites.assembly.scripts.so101_smoke --headless        # verdict
  .venv/bin/python -m robobench.suites.assembly.scripts.so101_smoke --livestream 2     # watch
  ```
  Isaac Lab bash steps take minutes — never assume a hang; give them time.
- Git: WIP commit `77fd554` is the last committed state. **All current scene+smoke work is uncommitted**
  (~186 insertions). Do NOT commit/amend until Haoxiang has visually verified — he verifies by
  livestream first, then greenlights the git write.

## Scene architecture (working, don't rebuild)
- **Everything is a free-floating rigid body**: two SO101 halves (`proximal` = base→shoulder→upper_arm,
  `distal` = wrist), the loose `motor` (servo), the `screw`, the `drill`. No fixture, no pin — the scene
  is embodiment/strategy-agnostic on purpose.
- **Pre-authored DISABLED welds**, snapped on at the seat by the gate: `motor_weld` (motor→upper_arm),
  `screw_weld_0` (screw→upper_arm). Joints MUST be authored at scene-build time — a joint added to a
  running physics view is ignored.
- **Rule-based fastening gate** (`post_step`): per-step check (screw in hole + parts aligned + bit on
  head + trigger squeezed + bit spinning) latches a kinematic drive that enables the welds. States:
  `0 free → 1 driving → 2 fastened`. `dt = 1/240`.
- Config is namespaced by joint for a future full-arm build (`elbow_servo_*`, `elbow_screw_*`, general
  `drive_rate`/`gate_*`/drill dials, link-named `upper_arm_pose()` / `elbow_screw_seats_w()` helpers).

## Smoke procedure (linear, A–J)
A wiggle (all free) · B insert servo · C rotate joint2→−90° (arm flat) · D drop screw over the
countersunk hole · E drill descends · F trigger→gate→welds · G retreat drill · H stress (knock+wrench)
· I swing back · J lift/shake/rotate/release the assembled robot. A PASS/FAIL verdict prints at the end.

**The arm must be held steady during the precision phases (B/C/D/E/F).** The scene has no fixture, so
the smoke re-creates a hold *for test feasibility only*. Before each precision phase, `grab(joint2_deg)`
**teleports the arm to the exact working pose** (base lying at `HOLD_QUAT` = Ry(90°), hole up) so the
hold only ever fights small deviations, then engages it.

## RESOLVED (2026-07-06, Fable) — smoke PASSes free-floating; hold = kinematic base clamp, drop = cone-funnel
Verdict line: `servo inserted to 0.17 mm | screw caught at +4.95 mm | fastened at step 2157 |
errs (mm): retreat 0.36, knock 0.36, wrench 0.09, swing 0.32/0.10, finale 0.52/0.65 | PASS`.
Deterministic (headless == livestream). **Only `so101_smoke.py` changed; the scene is untouched and
still fully free-floating** — the fix lives entirely in the test's hold + drop, as intended.

**What I tried and what the evidence said (two independent bugs, don't conflate them):**

1. **The hold.** I first did the doc's direction #1 — moved the force+torque grasp from `base` to the
   `upper_arm` (`scene.b_ua`), FF lever measured to that link. It *did* help (drop perched at +3.6 mm
   vs −69; base tilt down to ~7°), which confirms the cantilever diagnosis. But the arm still wobbled
   5–9° and **the fasten gate never opened** — the servo-align / bit-on-head / coaxial checks need
   sub-mm, sub-few-degree steadiness the force grasp can't give at this dt on a 0.35 kg arm, *wherever*
   you grip it. The force-PD stability wall (handoff §"Why the base grasp is marginal") is the real
   ceiling; upper_arm moves it but doesn't clear it. So I stopped fighting it.
   **Fix: kinematic base clamp.** `hold_arm` now just re-writes the proximal ROOT state to the constant
   teleported target every step during the precision phases (`HOLD` flag; same per-step-write mechanism
   the smoke already uses for the drill and the seated servo), released otherwise. This is physically
   the pin-fixed base that historically PASSed, recreated **smoke-side only** — the scene stays
   embodiment-agnostic. Unconditionally stable, zero gains. Base error → 0.17 mm / 0.00°, and the servo
   now genuinely **INSERTs to 0.17 mm** (no hard-align fallback fires).
   - *Why this is legal w.r.t. gotcha #6* (per-step root writes break contact manifolds for small
     colliders): the written body is the `base`, but the screw's contacts are all with the
     joint-simulated `upper_arm` link, and the written pose is CONSTANT — so no manifold churn on any
     surface the screw touches.

2. **The screw drop — a *separate* bug the good hold actually exposed.** With the clamp holding the
   geometry perfectly, a coaxial 10 mm drop got *worse*: the screw free-fell ~14 mm straight down the
   bore and rested **−8 mm inside the pocket**. Reason: the only stop on the axis is the seated servo's
   thin registration **pin**, and the servo is held by per-step pose writes → gotcha #6 → no reliable
   contact, screw tunnels the pin. (The old sloppy hold "worked" at +3.6 mm only because its
   misalignment accidentally threw the screw onto the countersink cone.)
   **Fix: aim for the cone on purpose.** `drop_screw(0.002 + M2_LEN, lateral=0.0015)` — drop 2 mm up
   and 1.5 mm off the hole axis so the tip lands on the **countersink cone** (real `upper_arm`
   collision, always present), which funnels it to a repeatable +4.95 mm perch. The gate's 2.5 mm
   radial window easily accepts that offset.

**Takeaways for the next person:**
- These were two bugs; a perfect hold is *necessary but not sufficient* — it uncovered the drop bug.
- Don't reach for a force/torque grasp to steady a light arm at 240 Hz; use a kinematic clamp (or a
  windowed joint) in the smoke, and let a real embodiment use contact/implicit-drive holds later.
- Rely on the cone, never a pose-written body, to stop a dropped part (gotcha #6).
- Feasibility for a future embodied grasp still looks good: a Franka pinch on the upper_arm resolves as
  implicit contacts (no PD wall) and has ~10× friction margin over the 4 N insertion press; a
  table-rest variant works too if insertion is re-choreographed top-down (force into the table). Not
  built yet — a choreography rewrite for after sign-off.

## THE BLOCKER (historical — resolved above) — the hold + the screw drop
The current hold is a clamped **force+torque grasp on the `base` link** (`hold_arm` block in `step()`),
gains `KP_B,KD_B,F_MAX_B = 800,70,40` and `KP_BR,KD_BR,TAU_MAX_B = 16,0.6,2.0`, plus a gravity-moment
feed-forward (arm weight × lever from live per-body CoMs). Latest run **FAILs**:
- `[A]` free wiggle clean (`max|q|=0.000`) ✓
- `[B]` servo stalls at 17 mm (design hard-aligns past this — collision fidelity, acceptable)
- `[C]/[D]` base holds to ~4–9° / ~2 mm — **not bad**
- `[D]` **screw drop misses: rests −69 mm** (falls past the seat instead of perching in the countersink)
- `[F]` **never fastens** (`fastened at step -1`) → downstream all fail.

### Why the base grasp is marginal (probe data — `scratchpad/hold_probe.py`)
The arm is light (~0.35 kg, tiny rotational inertia). At 240 Hz:
- **Stiff orientation PD → unstable**: the torque clamp saturates (`|tau_pd|≈2.0`) and the base whips at
  **|ω|≈16 rad/s**, tilt oscillating 13–32°. Never settles.
- **FF-only / soft → droops**: settles ~20° tilt, resting partly on the ground.
- It's a discrete-time stability wall: a gain stiff enough to hold the cantilevered arm upright is
  unstable at this dt/inertia; a stable gain is too soft. Holding the **base** means fighting the whole
  arm's gravity moment through the worst lever.
- Historically, a **rigid pre-authored joint hold PASSED** (insertion 0.11 mm, all errs ≤0.52 mm) — it's
  both infinitely stiff *and* unconditionally stable. Haoxiang moved away from it wanting a grasp, but
  it's the known-good fallback.

## Directions, ranked (my recommendation first)
1. **Hold the `upper_arm` link, not the base** — *top untried idea.* The upper_arm is the link being
   assembled (seat + hole live on it), and holding it near its own CoM makes the gravity moment ~0, so a
   *stable* soft gain can hold it precisely. Removes the cantilever entirely. The base then just hangs
   (light, harmless). This directly stabilizes the geometry that matters for the drop + drive.
   `scene.b_ua` is the upper_arm body index; grasp it the same way the base is grasped now.
2. **Fix the screw drop independently of the hold** — even with a good hold, `[D]` rests −69 mm. Check
   `drop_screw()` height/axis vs the countersink geometry and whether the screw tunnels the pilot at
   dt=1/240 (this scene family has a known tunneling gotcha below ~1/480). The drop must *perch*, not
   fall through.
3. **Lower sim dt to 1/480** — raises the stable-gain ceiling for any force grasp (2× sim cost) and cures
   likely screw tunneling. Cheapest single knob if you want to keep the base grasp.
4. **Reinstate the rigid joint hold, windowed** — re-add a disabled `base_hold` joint in the scene,
   toggle it on only during B/D–F with the teleport-init already in place. Reliable, known-good.

## Constraints / gotchas Haoxiang cares about
- **Verify visually first**: hand him a livestream run before any headless self-verification or git write.
- Bash steps ≤ a couple minutes of *your* attention but Isaac runs take minutes — background them.
- Collision-mesh cuts only when measured-necessary and minimal.
- `pkill`/pattern kills: beware self-matching the running command.
- Explicit names over comments; keep the scene strategy-agnostic (holds/teleports live in the smoke,
  never the scene). Screws are M2 now, M3 later for other joints — keep naming additive.
- The `_notes.md` / handoff files and bake scripts stay **local, uncommitted**.

## Fast start
```
.venv/bin/python -m robobench.suites.assembly.scripts.so101_smoke --headless   # see the FAIL + [B/C/D] base-err lines
```
Then try direction #1 (grasp the upper_arm) and #2 (fix the drop), livestream for Haoxiang, iterate.
