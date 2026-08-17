# Task `libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i98` — Cam-Gate Cabinet

Scene: `cam_gate_cabinet` (env `simgen.cam_gate_cabinet`, robot `"null"`).

## Provenance

Seed task: `libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet` —
"close the top drawer of the cabinet": the Franka pushes the drawer front along its
prismatic travel until the cabinet joint reads closed. One guided translation OF the
judged part; the drawer's position IS the goal.

Kept from the seed: a cabinet with a sliding drawer and the identical judged OUTCOME —
the drawer's front face flush with the cabinet face, "drawer closed", judged on the
drawer's terminal position. The drawer here is a free rigid body captured between
channel walls under a roof (contact physics, not an articulation), reproducing the
seed's drawer by mechanism rather than by joint.

## Strategic difference

**vs the seed:** in the seed the robot TRANSLATES the judged part — the whole skill is
a straight push on the drawer front. Here the intended skill is a ROTATION OF A
DIFFERENT PART: a wide blue GATE hangs on a real geometric hinge (two knuckle rings
around a kinematic steel pin, resting on collars, captured under a cap) beside the
open drawer. Swinging the gate shut makes its inner face press the red ROLLER standing
on the drawer's front, and the sliding cam contact converts the swing into the
drawer's closing translation — a rotation-to-translation transmission living entirely
in contact dynamics (no joint primitives anywhere). Success requires BOTH terminal
states: gate flush on its stop post (within 5°) AND drawer seated (within 12 mm),
settled and judged live. The seed's exact skill executed for real — a force-limited
hand-push that seats the drawer — is demonstrated by smoke to earn the drawer credit
only (~0.45) and can never succeed: the gate still stands open.

**vs the corpus surveyed this session:**
- `close_drawer_i58` (the sibling drawer task, read in full): there the robot NEVER
  actuates anything on the drawer's axis — it removes two obstructions and gravity
  closes the tray on an inclined runway. Here there is no incline, no stored energy
  and no obstruction: the robot actively DRIVES the transmission's input link through
  a ~70° arc, and the drawer moves exactly as far as the cam is driven. Opposite
  division of labor: i58 = "clear the path, hands off the mechanism"; i98 = "drive
  the mechanism's OTHER degree of freedom".
- `pen_holder` (robobench packing exemplar): pick-and-insert into a passive container;
  nothing transmits motion. Here the only manipulated object is a mechanism input
  whose output is the judged part.

## The machine (honesty by construction)

`scene.py` asserts the cam contract in `__post_init__`: the stop post's protrusion
EQUALS `pin_x − panel_offset`, so θ=0 (flush on the post) is the exact rest; at that
rest the cam relation `q(θ) = nose_setback + pin_x + ((pin_y−nose_y)·sinθ −
(panel_offset+nose_r))/cosθ` presses the drawer to q ≈ 3 mm — inside the 12 mm success
tolerance — while the drawer's own rear hard stop sits deeper (−4 mm), so the terminal
contact chain is gate-on-post + gate-on-roller. A support-function sweep over 0–52°
proves the roller beats BOTH front corners of the drawer by >2 mm everywhere (the
hinge-side front section is stepped back 15 mm for this) — the roller is the sole cam
contact. The transmission margin `cosθc − 2μ·sinθc > 0.25` at first touch is only
positive because all three compounds are bound to a slick material (μs 0.10 / μd 0.08,
min combine): with PhysX's default ~0.5 friction the cam would JAM at first touch —
the material is load-bearing and asserted. The gate at its minimum start angle (58°)
clears the widest sampled drawer by >10 mm (no spawn overlap); the roller clears the
channel wall; the hinge rings ride the pin with 2 mm clearance under collar/cap
capture. The gate's CoM sits on the hinge axis, so `settled()` judges the gate on its
ANGULAR velocity (linear velocity is ~0 even mid-swing — asserted honest in code).

## Teleport solution (`solve.py`) — ZERO teleports

The solve never teleports anything after reset. One applied wrench (a z-torque servo
on the gate, clamped 1.2 N·m — a hand's push on the panel or the yellow handle,
force-limited; a z-torque is frame-encoding invariant for a yaw-only body):

- P0 — settle; read back station pose/yaw, q0, θ0; invert the scene's cam relation at
  the MEASURED opening to get the touch angle; baseline ~0.
- P1 — APPROACH: ramped-target servo swings the gate from θ0 to just above the touch
  angle (~0.30 rad/s). The drawer has not moved (asserted < 8 mm); score ≈ 0.10–0.17
  (gate credit only).
- P2 — CAM DRIVE: the servo swings through touch to just past flush; the gate's inner
  face presses the roller and the cam translates the drawer from q0 ≈ 120 mm to
  ≈ 5 mm. Wrench released; hands-off settle; success() live. Score 1.0.
- Persistence — ≥3.3 simulated seconds hands-off, then `SIM_GEN_SOLVE: SUCCESS`.

Passes on forge on seeds 0–4 (yaws −36°…+140°+, q0 0.090–0.130, θ0 58–80°; distinct
xy). `SIM_GEN_SCORE` printed at each boundary, non-decreasing (0.00 → ~0.1 → 1.00 →
1.00). The drawer is never wrenched by the solver: its closing translation is
delivered by the machine's transmission.

## Embodiment argument (Franka, single arm)

Base ~0.55 m out on the apron (+x) side of the station; everything touched lives at
world z 0.27–0.44 (the station stands on a 0.25 m plinth) within a 0.45 m disc —
comfortably inside Franka's workspace at any episode yaw (the ±5 cm jitter and free
yaw only rotate the approach direction; heights never change).

- BLUE GATE / YELLOW HANDLE: the only thing the robot must touch. The handle is a
  20 mm proud bar, 20 mm wide, spanning z 0.275–0.395 world on the gate's outer face
  near the far edge (0.32 m lever arm) — a side grasp with the parallel jaw (20 mm ≪
  80 mm max), or simply a fingertip/knuckle PUSH on the panel's outer face, swept
  along the arc. The required torque is ~0.15 N·m of cam+friction load — under 1 N at
  the handle; the arc is a 0.32 m radius quarter-circle at constant height, a
  standard impedance-controlled sweep for a 7-DoF arm. The gate's swing envelope is
  open air over the apron (asserted clear of the drawer until the cam touch).
- The DRAWER is never touched in the intended plan. It has no handle facing the
  robot; its red roller is a cam follower, not a grip point. A hand CAN push the
  drawer front (smoke does, with a real 8 N PD force) — it seats the drawer but earns
  no success, exactly the seed's plan demonstrated useless.

## Ordering

The rubric imposes no order — it judges two live terminal states. The task's only
ordering is physically inherent: the drawer cannot END closed with the gate ajar
mid-arc and still satisfy success, because success also demands the gate flush; and a
gate driven flush ALWAYS seats the drawer (the cam relation at θ=0 presses it inside
tolerance — geometry excludes "gate flush, drawer open" while the drawer is in its
channel; the off-hinge and stolen-drawer fakes are separately refused by the
`gate_on_hinge` and `drawer_in_channel` guards). Pushing the drawer shut FIRST and
then swinging the gate flush is an honest alternate route and would legitimately
succeed — it is strictly more work than the intended plan and still requires driving
the gate through its full arc, which is the novel skill being tested. The two
partial-credit latches (`_fgate`, `_fdrawer`) only anchor demonstrated progress;
success itself is latch-free and judged live on the settled state.

## Rubric

- 0.45 × latched max closure fraction of the gate's initial angle (on-hinge guarded).
- 0.45 × latched max closure fraction of the drawer's initial opening
  (channel-guarded: a drawer stolen out of the cabinet earns nothing).
- Cap 0.90 without success; 1.0 iff `success()`: gate flush on its stop post (< 5°,
  on its hinge) AND drawer seated (front face < 12 mm from the cabinet face, centred
  in its channel), all settled (gate judged on angular velocity) and finite, live.
- Null policy ~0; the seed's strategy (push the drawer shut) ~0.45, never success.

## Checks (`smoke.py`, rejection-only, 12 checks)

settle/no-NaN (gate on pin at θ0, drawer at q0, score ~0); randomization A (yaw span
>90°, xy jitter); randomization B (q0 span AND θ0 span vary, settled readback tracks
both every reset); null policy ~0; SEED strategy (real 8 N PD push seats the drawer —
gate still open, ~0.45, no success); both-ajar near miss (gate 10°, drawer 20 mm,
settled → both clauses refuse, credit ≤ 0.90 cap); channel guard (drawer stolen to
the ground + gate flush → drawer latch refuses, gate credit only); latch regression
(gate to mid-arc then back to θ0 → gate latch survives, live clauses read open, no
success); off-hinge fake (gate laid flush-ORIENTED on the apron off its pin + drawer
seated → `gate_on_hinge` refuses, no success); rejection audit (success never
observed anywhere in the battery); final no-NaN; frames.npz video recorded.
