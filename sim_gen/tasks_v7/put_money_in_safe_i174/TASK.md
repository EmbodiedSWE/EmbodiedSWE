# put_money_in_safe_i174 — bank the cash brick through the vault's rotary transfer hatch

## Seed provenance

Seed: `rlbench/put_money_in_safe`
(`sim_gen/RoboVerse/roboverse_pack/tasks/rlbench/put_money_in_safe.py` — "put the
money away in the safe on the top shelf"). The seed scene is a dollar stack and a
safe whose door already stands open; the demonstrated strategy is ONE
grasp-transport-place: pick the stack up, carry it through the open door, set it
down on a shelf inside a static receptacle.

## Strategic difference

The seed's entire plan is a single pick-and-place into an always-open container.
Here that plan is physically impossible and its end state is explicitly rejected:

- There is **no door and no reachable interior**: the vault is sealed on every
  face. The only opening, a 100 x 60 mm letterbox window, is blocked at rest by
  the seal slab of a vertical-axis revolving TURRET (an airlock, like a bank's
  rotary deposit hatch). Smoke check 5 pushes the brick at the sealed window with
  the solve's own real force controller — it is refused.
- The goal is **not "object placed on shelf"**. The seed-analogue state — brick
  resting on the transfer shelf inside the open pocket — is constructed in smoke
  check 4 and rejected (it is only the mid-plan `loaded` stage). Success requires
  the brick in the BIN below the shelf **and** the hatch resealed.
- The plan is a forced three-stage MECHANISM CYCLE with an ordering that geometry
  enforces (crank open -> insert -> crank shut to deliver-and-reseal), not a
  single transport. The final placement is performed by the mechanism itself (the
  pocket sweeps the brick past the shelf's half-disc edge and gravity drops it
  into the bin), not by the hand.

## Teleport-solution phases (solve.py)

1. **P0** settle + layout/mass readback; baseline: score ~0, no success.
2. **P1 open** — speed-capped PD torque about the pivot (pure-z wrench on the
   turret = a hand orbiting the crank knob) turns the turret 170 deg to the
   receive stop (the stop sits at 170 deg, strictly inside the PhysX +/-180
   angle-wrap boundary). `SIM_GEN_SCORE` ~0.20.
3. **P2 transport + insert** — ONE teleport carries the brick through free air to
   a hover just outside the window (nothing judged is satisfied by the write);
   then a PD force + gravity feedforward (a firm force-limited push) drives it
   through the window onto the shelf into the open pocket. Released; the `loaded`
   latch is earned by contact geometry. `SIM_GEN_SCORE` ~0.45.
4. **P3 deliver + reseal** — the same torque servo turns the turret back to the
   seal stop; the pocket walls sweep the brick across the shelf, past the
   half-disc edge it loses support and falls into the bin; the slab returns
   behind the window. Wrenches zeroed. `SIM_GEN_SCORE` 1.0.
5. **P4 persistence** — 3.3 simulated seconds fully hands-off; success must hold
   throughout -> `SIM_GEN_SOLVE: SUCCESS`.

Teleports are transport-only (stand -> free-air hover outside the vault). Every
load-bearing interaction — opening, insertion, delivery, resealing — is contact
dynamics under applied wrenches.

## Embodiment argument (single Franka, parallel jaw, OSC)

Base pose: on the floor at roughly (0.95, -0.35) world, facing the vault front
(vault at ~(0.35, 0), stand 0.55 m off the front at a random bearing) — both the
crank and the window face are inside a 0.85 m reach envelope.

- **Crank**: the red knob is a 28 mm diameter x 32 mm cylinder standing proud of
  the roof (top ~0.49 m), well under the ~80 mm jaw opening. Pinch-grasp the knob
  from above and drive a 90 mm-radius orbit with OSC; the wrist yaw range covers
  a half-turn done as two quarter-turn strokes with one regrasp. The measured
  drive torque (<= 1.2 N.m at 90 mm = ~13 N tangential) is far inside Franka
  payload.
- **Brick**: 120 x 60 x 30 mm, ~0.19 kg, lying flat on a 100 mm stand — top-grasp
  across the 60 mm width (fits the jaw) at comfortable height. Carry to the
  window and push it through long-side-leading with the closed fingertips; the
  window (100 x 60 mm) accepts brick + fingertip with clearance, and the 5 N
  insertion force budget is trivial for the arm. The insertion PD force + weak
  heading torque in solve.py mirrors exactly this fingertip push.
- **Reseal**: same crank motion in reverse.

## Execution order declaration

REQUIRED, and enforced physically: the sealed hatch refuses insertion (smoke 5),
a loaded pocket is not success (smoke 4), the bin is unreachable except by the
sweep (smoke 6: no path from above; scene geometry asserts no gap beside/under
the slab admits the brick), and success additionally requires the reseal (smoke
7). Open -> insert -> sweep-and-reseal is the only order that works.

## Checks

solve.py: layout/mass readbacks; baseline no-success + score ~0; monotone
`SIM_GEN_SCORE` at 5 phase boundaries; per-phase asserts (receive stop reached,
hover satisfies nothing, loaded latched but not banked, delivered + resealed +
success); 3.3 s hands-off persistence; watchdog + hard exit; passes on >= 2 seeds.

smoke.py (11 checks): settle/no-NaN; randomization readback (vault yaw/xy, stand
xy, brick yaw, theta0); null policy; SEED-strategy analogue rejected; sealed
hatch refuses a real force-push; roof drop never banks; in-bin-but-unsealed
rejected; stranded-on-shelf rejected; window-jam rejected; both travel stops are
real under steady torque; frames.npz video saved.
