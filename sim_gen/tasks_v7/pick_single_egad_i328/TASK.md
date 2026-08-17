# pick_single_egad_i328 — VaultLauncherScene (`simgen.vault_launcher`)

Aim the yaw turret's gravity chute at the sealed vault's snout mouth, drop the
steel ball into the chute's high end, and let gravity launch it across the gap,
through the mouth, over the one-way ridge, into the vault — where a 72 mm drop
keeps it forever.

## Seed provenance

`maniskill/pick_single_egad` — grasp a loose object resting on the table and
LIFT it 7.5 cm; success is a z-displacement check on a held object. One grasp,
one straight-up shift.

## What changed, and why it is strategically different

The seed's plan (grasp the loose object, lift it) is **insufficient by
construction**, not re-parameterized:

1. **The goal is not a lift.** The ball must end up INSIDE a fully sealed vault
   (floor, walls, roof); the only opening is a low snout tunnel facing the
   launcher. The smoke battery executes the seed's entire success criterion —
   force-lifts the ball 10 cm and sets it down — and earns exactly nothing.
2. **The hand cannot deliver it.** The tunnel is 124 mm deep and 90 mm wide; a
   Franka finger (~55 mm) holding the 32 mm ball reaches ~90 mm in, short of the
   ridge, and the gripper body (100 mm) cannot enter at all (both asserted at
   import time). Decisively: the tunnel floor RISES 12 mm toward the interior,
   so anything *released* inside the snout rolls straight back out — the smoke
   battery does exactly that. Entry demands ~1 m/s of arrival speed through the
   mouth (a one-way velocity gate, demonstrated by the slow/fast entry pair).
3. **The only tool that produces that arrival is the launcher**, which turns the
   episode into an AIMING task: the vault's bearing is randomized over a 100°
   fan and the turret always starts 18–55° misaimed, so the agent must read
   where the vault stands, swivel the turret by its rear red knob to within ~5°
   of physical tolerance (the import-time audit proves a 15° misaim misses even
   the flared mouth; the smoke battery launches at 18° off and misses), and only
   then load the ball. Gravity does the throw: the ball leaves the lip at
   ~1.1 m/s (measured; rolling model 1.4), flies the ~110 mm gap, drops into the
   90 mm mouth window, and carries the ridge with ~3× the required energy.

Also different from every sibling examined: i3 (gate-pin unlock + tunnel slide),
i4 (thread tags onto pegs), i216 (go/no-go gauge sorting), lamp_off_i247
(recessed overcenter isolator). None involves aiming a ballistic machine at a
randomized target.

## Scene

Fully procedural (compound spawners; masses, damping, and colliders authored in
the spawn funcs and verified by `get_masses()` readback in solve):

- **base** — heavy dynamic plinth+pedestal (root MassAPI 40 kg). Dynamic, not
  kinematic, so the turret's joint survives reset teleports.
- **turret** — dynamic, body origin ON the vertical axle (pure-quat yaw writes);
  inclined 40 mm chute with rails and backstop, flat launch run and lip, rear
  handle post with a red knob at 185 mm lever radius; spawn-authored
  `UsdPhysics.RevoluteJoint` to the base (axis Z, ±87° — inside the PhysX 180°
  wrap).
- **vault** — kinematic sealed box, interior 140×160 mm under a full roof; snout
  tunnel with rising floor, flare wings, low apron; solid plinths seal every
  under-floor cavity.
- **tray** — kinematic dish behind the launcher where the ball starts.
- **ball** — dynamic 32 mm sphere, density 2700 (~46 g), solver velocity
  iterations 4 (GPU sphere-creep fix).

Randomization (readback-verified in smoke): vault bearing ±50° at 0.51 m; turret
initial yaw = bearing ∓ 18–55° (toward center, inside joint travel); base xy
jitter; tray at 180°±40°, 0.35–0.50 m; ball jitter in the tray. The first
post-seed draw is burned (degenerate-draw quirk).

The launch ballistics, arrival window, ridge energy margin, aim cone, hand
exclusion, swing clearances, and rubric-weight arithmetic are all asserted
numerically in `VaultLauncherSceneCfg.__post_init__` before any sim runs.

## Rubric

Latched stage credit anchored in the demonstrated solve; success judged live:

- 0.20 `aimed` — turret ever within 4° of the vault bearing while slow (3-step
  persistence)
- 0.20 `loaded` — ball ever riding the chute floor between the rails (3-step
  persistence; a hover write above the chute does not count)
- 0.30 `entered` — ball ever inside the tunnel/interior airspace
- non-success capped at 0.70; **1.0 iff `success()`**: ball inside the vault
  interior below ridge height, settled, finite. Null policy ≈ 0.

## Solution outline (solve.py, teleport = transport only)

1. settle + readbacks (masses, misaim ≥ 12°, score ≈ 0);
2. torque-servo the turret onto the bearing via `set_external_force_and_torque`
   (velocity servo, stall-escalating gain), release, verify the latch;
3. teleport the ball to a hover 40 mm ABOVE the chute at the load point (honest
   transport — loading is a real drop);
4. hands off: the ball falls, rolls, launches (~1.1 m/s at the lip), flies the
   gap, threads the mouth, carries the ridge, drops in, settles;
5. assert success + score ≥ 0.999, then ≥ 3 s hands-off persistence →
   `SIM_GEN_SOLVE: SUCCESS`.

Passes on seeds 0 and 1 (identical launch physics both runs; monotone scores
0.00 → 0.20 → 0.40 → 1.00).

## Franka embodiment argument

Everything the task demands is single-arm parallel-jaw OSC: swivel the turret by
its 40 mm red knob at 185 mm lever radius (a few N tangential; the chute floor
top is 280 mm up, well inside the workspace), pick the 32 mm ball from an open
shallow tray, and drop it into the open-topped 40 mm channel from above. No
bimanual holds, no in-hand re-orientation, no forces beyond fingertip scale. The
one thing a hand must NOT be able to do — stuff the ball through the tunnel — is
excluded by geometry (finger reach and gripper width vs a 124 mm deep, 90 mm
wide, rising-floor tunnel), asserted at import time and probed in smoke.

## Execution order (declared, physically forced)

**Aim first, then load.** Loading first is self-defeating: the ball leaves the
chute in ~0.4 s, long before any turret swing completes — so an aim-after-load
run has already thrown the ball away at the initial misaim (≥ 18°, a guaranteed
miss). The smoke battery's misaim-launch check is exactly this failure. A miss
is recoverable (the ball lands on the open floor and can be fetched), so the
order rule is enforced by physics, not by a latch.

## Checks

- `__post_init__`: 20+ import-time geometry/ballistics asserts (launch speed
  margin ≥ 25%, arrival window, ridge energy ×3, 15°-miss cone, sweep
  clearances, hand exclusion, joint-travel and wrap margins, weight sum).
- `solve.py`: staged asserts each phase (readbacks, aim latch, hover-not-loaded,
  lip/mouth crossing telemetry, entered/in-vault, success + persistence), 2
  seeds.
- `smoke.py`: 13 checks — settle/no-NaN; 3-seed max-pairwise randomization
  readback; stored-vs-pose consistency (bearing, yaw, misaim band/sign); null
  policy; seed-strategy lift earns nothing; roof rest rejected; outside-wall hug
  rejected; hand-placement in the snout rolls back out; 18°-misaim launch
  misses; 0.25 m/s slow entry rolls back; 1.2 m/s fast entry commits (genuine
  success, full latch pipeline); revocation falls to the 0.70 cap; frames.npz
  video.
