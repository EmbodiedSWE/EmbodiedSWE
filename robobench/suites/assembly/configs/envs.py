"""Canonical runnable env configs for the assembly suite — registered in `ENVS` by name.

Each binds the suite's scene to an embodiment + control mode + sim, so a run or smoke test loads one
by name, instead of wiring scene/robot/mode by hand. `register_env` derives the canonical name by the
convention ``suite.scene[.robot[.control_mode]]`` (segments dropped from the right when default/absent),
so names stay consistent as scenes/robots multiply — e.g. this suite's `SUITE="assembly"` + scene
`ikea_table` + no robot -> ``assembly.ikea_table``. Factories (a fresh `EnvCfg` per call) keep one
build from mutating another's cfg.

These are the *baseline* bindings; curriculum/debug variants are cheap `dataclasses.replace(cfg, ...)`
derivations the harness or agent can make — nothing here is locked.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import (
    AlohaCfg,
    BimanualFrankaCfg,
    BimanualPiperCfg,
    FrankaRobotCfg,
    G1RobotCfg,
    GR1T2RobotCfg,
    PiperRobotCfg,
    WxaiRobotCfg,
)
from robobench.robots import MultiRobotCfg
from robobench.suites.assembly.scenes import (
    AllenBoltAssemblySceneCfg,
    BulbAssemblySceneCfg,
    ChairAssemblySceneCfg,
    IkeaTableAssemblySceneCfg,
    NutThreadAssemblySceneCfg,
    PcGpuAssemblySceneCfg,
    PcGpuRamAssemblySceneCfg,
    PcMotherboardAssemblySceneCfg,
    PcRamAssemblySceneCfg,
)

SUITE = "assembly"

register_env(SUITE, lambda: EnvCfg(scene="ikea_table", robot="null", env_spacing=3))  # scene physics only

# Fixed-bolt + loose-nut threading scene, scene physics only for now (a robot is added later).
# -> "assembly.nut_thread"
register_env(SUITE, lambda: EnvCfg(scene="nut_thread", robot="null", env_spacing=2))

# Fixed lamp-socket + loose light-bulb screw-in scene, scene physics only for now.
# -> "assembly.bulb"
register_env(SUITE, lambda: EnvCfg(scene="bulb", robot="null", env_spacing=2))

# Fixed threaded platform + a loose allen bolt + an allen key that drives the bolt down into the
# platform's threaded hole, scene physics only for now.
# -> "assembly.allen_bolt"
register_env(SUITE, lambda: EnvCfg(scene="allen_bolt", robot="null", env_spacing=2))

# PC case lying on its side, motherboard up: 7 threaded case-mount holes, 7 loose allen bolts, and
# one allen key that drives each bolt down into its hole, scene physics only for now.
# -> "assembly.pc_motherboard"
register_env(SUITE, lambda: EnvCfg(scene="pc_motherboard", robot="null", env_spacing=2))

# Franka arm at the pc-motherboard scene: one key fastens all 7 board bolts. The base stands
# WEST of the case at (0.07, 0), facing +x, and the whole table (case included) slides 40 mm
# east (`workbench_pos`) so the arm's foot clears the case's west edge by 5 cm — the 7 holes
# then span the arm's 0.36-0.60 m top-down band, where a reach probe (yaw 0 / +-120 deg,
# z 0.28-0.40) tracks to a few mm; an east base instead put the nearest holes at 0.19-0.26 m,
# inside the folded arm's close-in cliff, and the reinsert plunge saturated ~1 cm short. The
# key stands UPRIGHT in a four-wall stand in the south-west strip (`key_stand`), tip down and
# handle 210 mm up — the very grip the ratchet cranks with, so the arm lifts it out and screws
# with a single grasp (a flat-lying key instead demands a low pinch, a 90 deg in-hand
# reorientation, and a release+regrasp in the first socket). The bolt row lies along the SOUTH
# table edge (the smoke stages bolts kinematically in their holes, so the row is scenery).
# Deterministic spawn; sim dt 1/240 (no SDF threads here — the smoke drives the scene's
# kinematic screw joints, cf. pc_motherboard_smoke).
# Three control modes, switchable by env name:
#   - "assembly.pc_motherboard.franka.osc"       — operational-space control (default)
#   - "assembly.pc_motherboard.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.pc_motherboard.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="pc_motherboard",
            scene_cfg=PcMotherboardAssemblySceneCfg(
                workbench_pos=(0.54, 0.0),
                bolt_init_xy=tuple((-0.24 + k * 0.075, -0.42) for k in range(7)),
                key_init_xy=(-0.24, -0.30),
                key_init_z=0.001,
                key_init_quat=(1.0, 0.0, 0.0, 0.0),
                key_stand=True,
                reset_pos_jitter=0.0,
            ),
            robot="franka",
            robot_cfg=FrankaRobotCfg(base_pos=(0.07, 0.0, 0.0)),
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 240.0},
        ),
    )

# The same PC case with its primary PCIe x16 slot empty and a loose graphics card beside it, to be
# stood upright and pressed straight down into the slot, scene physics only for now.
# -> "assembly.pc_gpu"
register_env(SUITE, lambda: EnvCfg(scene="pc_gpu", robot="null", env_spacing=2))

# PC case lying on its side, motherboard up: two empty DIMM slots (invisible grip channels) and
# two loose RAM sticks to press in, scene physics only for now.
# -> "assembly.pc_ram"
register_env(SUITE, lambda: EnvCfg(scene="pc_ram", robot="null", env_spacing=2))

# The full build: the same case with BOTH work sites open — the empty PCIe x16 slot (+ rear
# cutout) and the two empty DIMM slots — a loose graphics card and two loose RAM sticks beside
# it, scene physics only for now.
# -> "assembly.pc_gpu_ram"
register_env(SUITE, lambda: EnvCfg(scene="pc_gpu_ram", robot="null", env_spacing=2))

# Franka arm at the combined gpu+ram scene: the card goes into the PCIe x16 slot FIRST (placed
# inside the case, slid rearward through the I/O cutout, pressed to seat), then the two sticks
# go into the DIMM pair. The three parts stage side by side in ONE line on the table south of
# the case, every part's length along y — pointing away from the case, so no pick brings the
# wrist near its 22 cm wall: stick 0 at world (0.29, -0.32), the card lengthwise between the
# sticks at (0.365, -0.321), stick 1 at (0.44, -0.32). The sticks stand in their seated heading;
# the card stands yawed 90 deg and the smoke rotates it back during its carry, in free air over
# the case. The case sits 40 mm north of the table anchor (`case_xy`) so the 267 mm card fits
# lengthwise in the staging strip, and the base follows to (0.72, -0.30) yaw 180 — the whole
# work cell translates rigidly, keeping every case-relative reach in the arm's accurate band:
# PCIe seat 0.394 m, its placement point 0.404 m (the rearward slide runs slightly radially
# inward), DIMM seats 0.365/0.378 m, picks 0.43/0.36/0.28 m. The card is installed first, so
# its emptied holder never obstructs the later stick flights. All parts stage UPRIGHT in foam
# holders (their lying defaults are ungraspable — see the single-task envs); deterministic
# spawn (no jitter): the holders are static geometry authored at the spawn points. sim dt
# 1/240 — the depth both force-driven smokes validated.
# Three control modes, switchable by env name:
#   - "assembly.pc_gpu_ram.franka.osc"       — operational-space control (default)
#   - "assembly.pc_gpu_ram.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.pc_gpu_ram.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="pc_gpu_ram",
            scene_cfg=PcGpuRamAssemblySceneCfg(
                case_xy=(0.55, 0.04),  # case 40 mm north of the table anchor: stretches the
                # staging strip so the card fits lengthwise; the base follows (see below)
                card_init_xy=(-0.185, -0.321),  # table-rel -> world (0.365, -0.321): the middle
                # of the staging line, lengthwise between the sticks
                card_init_z=0.030,  # tab-bottom plane = the holder's floor top
                card_init_quat=(0.70711, 0.0, 0.0, 0.70711),  # upright, yawed 90 deg: staged
                # parallel to the sticks; the carry rotates it back to its seated heading
                ram_init_xy=((-0.26, -0.32), (-0.11, -0.32)),  # table-rel -> world
                # (0.29/0.44, -0.32): flanking the card, all three parts parallel along y
                ram_init_quat=(1.0, 0.0, 0.0, 0.0),  # upright, the seated orientation
                ram_init_z=0.030,  # blade-bottom plane = the holders' floor top
                reset_pos_jitter=0.0,
                card_stand=True,
                ram_stand=True,
            ),
            robot="franka",
            robot_cfg=FrankaRobotCfg(
                base_pos=(0.72, -0.30, 0.0), base_rot=(0.0, 0.0, 0.0, 1.0)  # yaw 180: faces -x;
                # 40 mm north with the case, the 154 mm rear foot points +x along the strip
            ),
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 240.0},
        ),
    )

# Franka arm at the pc-ram scene (the case/table preset sits at 0.55 here). Same north-strip
# placement family as pc_gpu.franka: the base stands at (0.72, -0.34) yaw 180 with its whole
# link0 footprint (x [-0.154, +0.072] x y +-0.095) on the top plate, and the two stick holders
# sit west of it at world (0.30, -0.36) and (0.42, -0.36) — reaches: picks 0.42 / 0.30 m, slots
# 0.376 / 0.365 m, all in the arm's accurate band. The sticks cannot start in the scene's lying
# default (flat, their 7.3 mm thickness points up — no parallel-jaw pinch off the table), so the
# gripper env stages them UPRIGHT in the scene's foam holders, already in the seated
# orientation. Deterministic spawn (no jitter): the holders are static geometry authored at the
# spawn points. sim dt 1/240, the depth the force-driven pc_ram smoke runs at.
# Three control modes, switchable by env name:
#   - "assembly.pc_ram.franka.osc"       — operational-space control (default)
#   - "assembly.pc_ram.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.pc_ram.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="pc_ram",
            scene_cfg=PcRamAssemblySceneCfg(
                ram_init_xy=((-0.25, -0.36), (-0.13, -0.36)),  # table-rel -> world (0.30/0.42, -0.36)
                ram_init_z=0.030,  # blade-bottom plane = the holders' floor top
                ram_init_quat=(1.0, 0.0, 0.0, 0.0),  # upright, the seated orientation
                reset_pos_jitter=0.0,
                ram_stand=True,
            ),
            robot="franka",
            robot_cfg=FrankaRobotCfg(
                base_pos=(0.72, -0.34, 0.0), base_rot=(0.0, 0.0, 0.0, 1.0)  # yaw 180: faces -x,
                # the 154 mm rear foot points +x along the strip
            ),
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 240.0},
        ),
    )

# Franka arm at the allen-bolt scene (base at the origin). Placement follows the solve-verified
# reach lessons of the sibling franka envs: the platform is pulled from the table preset's 0.50 m
# to 0.42 m (`platform_slots`) — the screwing happens under a TOP-DOWN hand, and beyond ~0.45 m
# the gravity-uncompensated arm saturates several mm short (pc_gpu note), more than the socket's
# 0.75 mm/side clearance; the loose key leaves the stock "+x row" (0.76 m, out of reach) for the
# proven ~0.36 m pick radius on the +y side (the bulb layout's band). The bolt spawn stays put —
# the smoke stages it upright over the hole (the robot's job is the KEY). Deterministic spawn
# (no jitter) so smoke iterations reproduce. bolt_friction 0.3 makes the M16 thread SELF-LOCKING
# (needs mu > tan(2.5 deg) ~ 0.044): at the scene's slick 0.01 the bolt spins back out whenever
# the ratcheting key lifts out of the socket between strokes (the force-driven smoke never
# disengages, so only the robot env needs it). sim dt 1/240 — the depth the force-driven smoke
# validated for a pressed M16 on the SDF threads (the scene's 1/120 is for parts at rest).
# Three control modes, switchable by env name:
#   - "assembly.allen_bolt.franka.osc"       — operational-space control (default)
#   - "assembly.allen_bolt.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.allen_bolt.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="allen_bolt",
            scene_cfg=AllenBoltAssemblySceneCfg(
                platform_slots=((-0.08, 0.0),),
                bolt_staged=True,  # the bolt spawns hand-started in its hole; the task is
                # the key work
                # The key inserts by its LONG arm (the 50 mm short arm then cranks at half the
                # swept diameter, and the grip rides a long vertical shaft instead of a low one).
                # Spawned yawed +90 deg — handle along +y, short arm along +x — so the erection
                # about the short-arm axis lands the hand in the proven -y-approach insertion
                # configuration; spawn pulled to y 0.18 so the 120 mm handle's far end (the
                # inserting tip, 0.43 m out) stays inside the arm's accurate pick band.
                key_init_xy=((-0.24, 0.18),),
                key_init_quat=(0.5, 0.5, 0.5, 0.5),
                bolt_friction=0.3,
                reset_pos_jitter=0.0,
            ),
            robot="franka",
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 240.0},
        ),
    )

# SO101 full-arm assembly (seat + screw the elbow servo, clip + screw the forearm fork onto its
# horn) on a workbench, scene physics only.
# -> "assembly.so101"
register_env(SUITE, lambda: EnvCfg(scene="so101", robot="null", env_spacing=2))

# Franka arm at the nut-thread scene (base at the origin, reaching the bolt on the table at +x).
# Baked in (measured at the natural flat layout — base at the table plane, no sink):
#   - nut spawn pulled to (-0.12, 0) — the stock "+x row" default puts it at 0.62 m, out of the
#     origin-mounted arm's reach (the bolt's default slot at 0.50 m is in-band and stays);
#   - nut_friction 0.4 — at the 0.01 default the jaws cannot transmit wrench torque to the nut;
#   - sim dt 1/480 — a pressed M16 TUNNELS through the SDF threads at the scene's 1/120, so nothing
#     can genuinely thread there (1/240 narrows the window, 1/480 clean).
# Three control modes, switchable by env name:
#   - "assembly.nut_thread.franka.osc"       — arm by operational-space control (inertia-shaped; default,
#                                              smooth on this arm)
#   - "assembly.nut_thread.franka.impedance" — arm by Jacobian-transpose task-space impedance (Isaac's form)
#   - "assembly.nut_thread.franka.joint"     — arm by direct joint position targets
# (all carry a 2-finger gripper by direct position target.)
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="nut_thread",
            scene_cfg=NutThreadAssemblySceneCfg(
                nut_init_xy=((-0.12, 0.0),),
                nut_friction=0.4,
            ),
            robot="franka",
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 480.0},
        ),
    )

# Franka arm at the bulb scene (base at the origin). The socket + loose bulb are pulled off the stock
# nut_thread "+x row" layout into the arm's measured reach band: the default row put the bulb at
# 0.63 m (out of reach -> REORIENT_STUCK) on the centreline (parks wrist q7 near its stop). Baked in:
# socket 9 cm closer, bulb at the ~0.43 m pick radius on the +y side (q7 margin).
# (Per-shape bulb friction is already the scene default — no override needed.)
# Three control modes, switchable by env name:
#   - "assembly.bulb.franka.osc"       — operational-space control (default)
#   - "assembly.bulb.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.bulb.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="bulb",
            scene_cfg=BulbAssemblySceneCfg(
                socket_slots=((-0.09, 0.0),),
                bulb_init_xy=((-0.24, 0.25),),
            ),
            robot="franka",
            control_mode=mode,
            env_spacing=2,
        ),
    )

# Franka arm at the pc-gpu scene. The base stands in the table's NORTH strip at (0.64, -0.34),
# yaw 180 deg, beside the case's north-east corner; the card holder sits west of it at
# (0.28, -0.36). Both fit fully on the lab table's top plate — x [-0.32, 0.96] x y [-0.47, 0.44]
# in world, with panda link0's footprint spanning x [-0.154, +0.072] x y +-0.095 around the base
# origin, so the 0.26 m-deep strip only fits it with the rear foot pointing +-x. Reach stays in
# the arm's accurate band: pick 0.36 m near dead-ahead, placement 0.393 m / seat 0.384 m at
# ~74 deg right, and the rearward slide runs slightly radially inward. (A base much beyond
# ~0.45 m from the seat saturates the top-down arm several mm short — more than the channel's
# 1.5 mm end-stop play.) The case stays at the table preset's 0.5 m. The loose card cannot
# start in the scene's lying default:
# flat on its backplate its only sub-80 mm dimension (the 36 mm body thickness) points UP, so no
# parallel-jaw pinch can take it off the table. The gripper env therefore stages it UPRIGHT in the
# scene's foam holder (`card_stand=True`), already in the seated orientation — one top-down
# fingertip grip on the card's top edge (see the smoke's grasp-geometry note) then serves pick,
# carry, slide and press, with no re-orientation anywhere near the case. Deterministic spawn (no
# jitter): the holder is static geometry authored at the spawn point, so a jittered card would
# spawn inside a rail.
# sim dt 1/240 — the depth the force-driven pc_gpu smoke validated for the 0.15 mm/side channel.
# Three control modes, switchable by env name:
#   - "assembly.pc_gpu.franka.osc"       — operational-space control (default)
#   - "assembly.pc_gpu.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.pc_gpu.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="pc_gpu",
            scene_cfg=PcGpuAssemblySceneCfg(
                card_init_xy=(-0.22, -0.36),  # table-relative -> world (0.28, -0.36): the pick band
                card_init_z=0.030,  # tab-bottom plane = the holder's floor top
                card_init_quat=(1.0, 0.0, 0.0, 0.0),  # upright, the seated orientation
                reset_pos_jitter=0.0,
                card_stand=True,
            ),
            robot="franka",
            robot_cfg=FrankaRobotCfg(
                base_pos=(0.64, -0.34, 0.0), base_rot=(0.0, 0.0, 0.0, 1.0)  # yaw 180: faces -x,
                # the 154 mm rear foot points +x along the strip (the only fit inside it)
            ),
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 240.0},
        ),
    )

# Two Frankas at the SO101 workbench as ONE robot (`BimanualFranka`: action = [left | right];
# address one arm via `env.robot["left"]`). Bases stand on the bench top (z = 0.994, the default
# packing table), 0.94 m apart — each arm works best 0.3-0.55 m from its own base, so the shared
# zone sits around (0.4, 0).
# Reach-verified (reach probes): in-zone tracking
# <= 0.6 mm, both arms simultaneously at the shared zone OK. Known gotcha: the default home pose
# parks each hand over the other arm's zone — tuck the idle arm.
# TO VERIFY: arm-arm collision limits when one arm stretches cross-body (> 0.55 m); per-task base
# retuning for hold+insert style work.
# -> "assembly.so101.bimanual_franka.{osc,impedance,joint}" (mode applies to both arms)
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="so101",
                robot="bimanual_franka",
                control_mode=mode,
                robot_cfg=BimanualFrankaCfg(robots={
                    "left": ("franka", FrankaRobotCfg(  # yaw -50 deg, faces the proximal/motor zone
                        base_pos=(0.0, 0.28, 0.994), base_rot=(0.90631, 0.0, 0.0, -0.42262))),
                    "right": ("franka", FrankaRobotCfg(  # yaw +135 deg, faces the drill/fixture zone
                        base_pos=(0.75, -0.28, 0.994), base_rot=(0.38268, 0.0, 0.0, 0.92388))),
                }),
                env_spacing=3,
            )
        ),
    )

# Bimanual pairs at the IKEA table, bases on the bench top (z = 0.994), facing each other.
# The small pairs face off ACROSS the bench (x = 0, yaw -/+90 deg), separation tracking reach
# (WXAI ~0.5 m -> y = +/-0.30, PiPER ~0.6 m -> +/-0.35); the Frankas face off ALONG the bench
# (y = 0) instead — the top is only +/-0.38 in y, too narrow for their bases. The Franka pair is
# asymmetric on purpose: the slab (x in [-0.75, -0.15]) is a keep-out, so the "pin/left-drag" arm
# sits behind it at (-0.95, 0.0) (slab side, faces +x) and the "threader/leg-cycle" arm at
# (0.25, -0.25) (leg-row side, yaw 180 deg), workspaces overlapping around the slab edge.
# The Franka bases + leg spawn below are load-bearing calibrated constants — do not move them.
# CAUTION: the small pairs' base poses / reachability are NOT fully verified yet — starting guesses.
# -> "assembly.ikea_table.aloha.{joint,osc,impedance}"           (bimanual WXAI, as ALOHA)
# -> "assembly.ikea_table.bimanual_piper.{joint,osc,impedance}"  (bimanual AgileX PiPER)
# -> "assembly.ikea_table.bimanual_franka.{joint,osc,impedance}" (bimanual Franka)
for _mode in ("joint", "osc", "impedance"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="ikea_table",
                robot="aloha",
                control_mode=mode,
                robot_cfg=AlohaCfg(robots={
                    "left": ("wxai", WxaiRobotCfg(base_pos=(0.0, 0.30, 0.994), base_rot=(0.7071, 0.0, 0.0, -0.7071))),
                    "right": ("wxai", WxaiRobotCfg(base_pos=(0.0, -0.30, 0.994), base_rot=(0.7071, 0.0, 0.0, 0.7071))),
                }),
                env_spacing=3,
            )
        ),
    )
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="ikea_table",
                robot="bimanual_piper",
                control_mode=mode,
                robot_cfg=BimanualPiperCfg(robots={
                    "left": ("piper", PiperRobotCfg(base_pos=(0.0, 0.35, 0.994), base_rot=(0.7071, 0.0, 0.0, -0.7071))),
                    "right": ("piper", PiperRobotCfg(base_pos=(0.0, -0.35, 0.994), base_rot=(0.7071, 0.0, 0.0, 0.7071))),
                }),
                env_spacing=3,
            )
        ),
    )
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="ikea_table",
                # Leg spawn baked: no reset jitter (deterministic) and the explicit row that
                # keeps every grip in the right arm's 0.31-0.43 m pick band.
                scene_cfg=IkeaTableAssemblySceneCfg(
                    reset_pos_jitter=0.0,
                    leg_init_xy=((-0.05, -0.03), (0.01, 0.18), (0.13, 0.18), (0.25, 0.18)),
                ),
                robot="bimanual_franka",
                control_mode=mode,
                robot_cfg=BimanualFrankaCfg(robots={
                    "left": ("franka", FrankaRobotCfg(base_pos=(-0.95, 0.0, 0.994))),  # slab side, faces +x
                    "right": ("franka", FrankaRobotCfg(  # leg-row/threading corner, yaw 180 deg (faces -x)
                        base_pos=(0.25, -0.25, 0.994), base_rot=(0.0, 0.0, 0.0, 1.0))),
                }),
                env_spacing=3,
            )
        ),
    )

# Fixed-base G1 at the IKEA table, upper-body joint control. Two placement tweaks so the G1 (pelvis
# ~0.75 m) can reach the work:
#   - the workbench is lowered to a ~0.7 m top (surface_z=0.7; the bench sinks below the floor, like
#     Isaac's pick-place env) so the parts sit around chest height instead of at the neck;
#   - the G1 stands slightly BACK of the table (base_pos -0.6 m in y, facing +y) so it reaches forward.
# Both `surface_z` and `base_pos` are dials the agent can retune.
# Two control modes (same scene + embodiment + placement), so you can switch by env name:
#   - "assembly.ikea_table.g1.joint"   — arm+waist by direct joint targets
#   - "assembly.ikea_table.g1.pink_ik" — arm+waist by whole-body Pink IK (action = wrist poses)
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="ikea_table",
                scene_cfg=IkeaTableAssemblySceneCfg(surface_z=0.7),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.6, 0.75)),
                env_spacing=3,
            )
        ),
    )

# GR1-T2 at the same lowered table, both control modes (like G1). Placement (base_pos behind, facing
# +y) is a STARTING guess — GR1T2's default facing differs from G1, so tune base_pos/base_rot after an
# in-sim look. -> "assembly.ikea_table.gr1t2.{joint,pink_ik}".
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="ikea_table",
                scene_cfg=IkeaTableAssemblySceneCfg(surface_z=0.7),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.55, 0.95), base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )


# ---- FurnitureBench chair (heterogeneous parts + ordering; the Franka long-horizon anchor) -----
# Scene physics only (NullRobot oracle/smoke). -> "assembly.chair"
register_env(SUITE, lambda: EnvCfg(scene="chair", robot="null", env_spacing=3))


# Robot bindings. Placements are STARTING guesses copied from the stacking-toy / pen-holder
# measured reach values for the same embodiments at the same bench — re-verify with the
# per-binding stress smoke before any agent run (only the null smoke validates the scene
# itself). The seat sits slightly forward of the robot; the five loose parts scatter on a
# front arc.
def _chair_g1_cfg() -> ChairAssemblySceneCfg:
    """G1 (short ~0.55 m arms): work on a 0.7 m bench, seat pushed +y, parts on a compact
    front arc."""
    return ChairAssemblySceneCfg(
        surface_z=0.7,
        seat_pos=(0.0, 0.18),
        spawn_radii=(0.26, 0.36),
        spawn_arc=(215.0, 325.0),
    )


def _chair_gr1t2_cfg() -> ChairAssemblySceneCfg:
    """GR1-T2 (longer arms): same bench, slightly wider arc."""
    return ChairAssemblySceneCfg(
        surface_z=0.7,
        seat_pos=(0.0, 0.20),
        spawn_radii=(0.30, 0.42),
        spawn_arc=(205.0, 335.0),
    )


def _chair_franka_cfg() -> ChairAssemblySceneCfg:
    """Franka (PRIMARY — the source platform): ground-level kit just in front of the base,
    compact arc inside ~0.75 m reach. Every part passes the 8 cm-jaw audit: leg shaft
    30 mm dia, backrest panel 20 mm thick, nut ring 60 mm across (pinch the 15 mm rim),
    seat slab 30 mm edge."""
    return ChairAssemblySceneCfg(
        seat_pos=(0.0, 0.16),
        spawn_radii=(0.24, 0.34),
        spawn_arc=(215.0, 325.0),
    )


def _chair_multi_cfg() -> ChairAssemblySceneCfg:
    """Dual Franka on a DOUBLE-WIDTH packing table (2.47 x 1.52 top, sunk to 0.25 m),
    one arm at each end of the long axis. PROP-FREE LYING LAYOUT (user redesign,
    2026-08-08): the chair base LIES ON ITS SIDE between the arms — legs pointing
    EAST, the stud pair VERTICAL (one over the other at surface +0.09 / +0.37), both
    studs pointing SOUTH into the middle lane. The backrest lies FLAT nearby, face
    up, crown east: its holes sit 0.23 -/+ 0.14 from its long edge, so rolled upright
    onto that edge they land at EXACTLY the stud heights (the +/-0.14 offsets match
    by construction). The assembly is a ground-supported 90-deg roll + a short
    horizontal slide north — no lift, no staging rack, no riser."""
    return ChairAssemblySceneCfg(
        # 0.75: the full-size shell sits at the parallel jaw's limit (54-72 mm
        # pinches vs the 80 mm jaw) and its 2.5 kg roll moment (5.6 Nm) exceeds
        # the wrist channel's measured authority; at 0.75 the pinches are
        # 40-54 mm and the roll moment 2.3 Nm — both comfortable (user-approved
        # feasibility gate, 2026-08-09).
        part_scale=0.75,
        # DUAL-ARM NORTHWARD LINE (user design 2026-08-10: "two arms to pinch
        # the seat back firmly and move it to mesh with the base"): base ON ITS
        # SIDE at the north-centre, legs WEST, studs pointing SOUTH at the arms
        # (quat = qz(-90) x the stable family; stud axis verified (0,-1,0)
        # numerically); the ladder lies posts-E-W in the middle lane, and BOTH
        # arms grip the TOP POST beam (west half / east half) from the south —
        # each at the proven perpendicular geometry — then slide it 0.2 m north
        # onto the stud stack. Chain reaches (measured on the footprint map):
        # grasp 0.34/0.39, slide end 0.55/0.57, stud threading 0.51 from the
        # left arm.
        # base pulled closer (user: shorter insert) and the line shifted west:
        # the right arm's pin (site 8 east end, x -0.06) then sits at its PROVEN
        # -0.48 lateral offset, and the left tests -0.10/-0.15/-0.19 offsets on
        # site 7 (a frontal pin is the measured elbow-fold failure).
        # base y stays 0.18: with back_seat_pos corrected to the rig's measured
        # flush (chair_assembly.py), the proven approach unrolls verbatim and
        # the funnel snap (measured: always lands AT the physical flush = the
        # corrected plane) finishes the last stretch the arm cannot reach.
        # A 0.151 spawn was tried and is unworkable: the left forearm's carry
        # corridor plows the 29 mm-closer base (fp298-301, drift telemetry).
        base_root_pose=(-0.55, 0.18, 0.2224, 0.0, -0.70711, 0.0, -0.70711),
        seat_pos=(-0.55, 0.18),  # slot origin only (spawn_slots stay seat-relative)
        # bench widened 2x IN WIDTH ONLY (2.47 x 1.52 top, same length and height)
        workbench_scale=(0.01, 0.02, 0.01),
        # ladder: SAME quat family (mate = pure northward translation). Spawn
        # 0.2 m south of the assembled pose; unit spans x -0.42..0.31,
        # y -0.34..-0.24; top-post beam at z 0.633, y -0.30.
        # MIDWAY between the arms (y -0.45) and the base's south edge (-0.025):
        # footprint y -0.23..-0.13, 10 cm clear of the arm plates, 0.22 m mate
        # slide north; x matches the base origin (mate = pure y translation)
        back_root_pose=(-0.55, -0.18, 0.189, 0.0, -0.70711, 0.0, -0.70711),
        # manifest [back, nut_0, nut_1], SEAT-RELATIVE x/y (world = slot + seat_pos);
        # the back's slot is superseded by back_root_pose; nuts staged BENCH-LEVEL
        # in the left arm's south field at the IK-VERIFIED pick spots (world
        # (0.10, -0.60) and (-0.02, -0.66); tilt-0/15 pick chains solve there,
        # measured 230/iw254). NO tray/riser: the earlier "bench unreachable"
        # readings were two script bugs (one-finger grip command + nut root-vs-
        # mesh offset), not geometry — the task stays platform-free by design.
        spawn_slots=((0.0, -0.40, 0.0), (0.65, -0.78, 0.0), (0.53, -0.84, 0.0)),
    )


# -> "assembly.chair.g1.{joint,pink_ik}" / ".gr1t2.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="chair",
                scene_cfg=_chair_g1_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.50, 0.75)),
                env_spacing=3,
            )
        ),
    )
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="chair",
                scene_cfg=_chair_gr1t2_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.48, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )

# -> "assembly.chair.franka.{osc,joint}"
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="chair",
                scene_cfg=_chair_franka_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.40, 0.0),
                                         base_rot=(0.7071068, 0.0, 0.0, 0.7071068)),
                env_spacing=3,
            )
        ),
    )

# -> "assembly.chair.multi.{osc,joint}" — two Frankas facing each other across the work
# (the pen_holder dual-arm pattern; EnvCfg.control_mode propagates to both children).
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="chair",
                scene_cfg=_chair_multi_cfg(),
                robot="multi",
                control_mode=mode,
                robot_cfg=MultiRobotCfg(robots={
                    # one arm at each END of the table's long axis, facing each other,
                    # centred (-0.05 in y) between the two work zones (body at +0.32,
                    # panel lane at -0.38). TUCKED home pose: the stock forward "ready"
                    # pose dangles the hand ~0.49 m ahead of the base — measured
                    # resting ON the chair seat (scooted it 2 cm). Wrist joint 5 is
                    # held OFF ZERO: j5=0 is the Franka wrist singularity (j4/j6 axes
                    # coplanar, one rotational DOF lost) and every commanded
                    # reorientation that needed the lost axis stalled 116-159 deg
                    # from target, measured across five grasp geometries
                    # disable_arm_gravity: the OSC torque law has no gravity term, so
                    # with gravity on, low extended poses spend the gains carrying the
                    # arm and commanded reorientations stall 56-76 deg out (measured;
                    # the real Panda gravity-compensates internally). kp_null 1.0
                    # keeps the posture spring from also fighting the task.
                    # control_mode "impedance": the OSC form shapes the task force by
                    # the op-space inertia Λ=(J M⁻¹ Jᵀ)⁻¹, and from the work configs
                    # its z-rotation channel dies (74-76 deg stalls, invariant to
                    # gravity, posture gain, and joint margins — measured); the
                    # impedance form applies kp*err directly and has no Λ to corrupt
                    # pos gains 500 on BOTH arms: the task-space spring is measured
                    # exactly F = kp * gap (panel hang sag 24.5 cm at kp=100, 4.9 cm
                    # at kp=500 under the 2.5 kg panel), so kp=100 tops out at the
                    # panel's own weight and every drag/lift stalled; kp=500 holds
                    # it with ~5 cm sag and transported it (measured). Orientation
                    # under load is held ONLY by a dual grip (the rot channel's
                    # authority cannot brace the side-held panel's 4.6 Nm gravity
                    # moment — solo carry toppled it 88 deg, measured), so both
                    # arms carry, both need the force. Rot gains 90 (3x stock).
                    # bases at y -0.15: split the difference between the staging
                    # (y ~-0.22..-0.28, where the pick force lives) and the studs
                    # (y ~-0.01); at y -0.05 the arms couldn't drag the leaning panel
                    # left base y 0.0: the staging moved 20 cm north (to the seat
                    # front), and each relocation re-rolls the wrist-feasibility
                    # pocket — so the base FOLLOWS the band, preserving the exact
                    # proven hand-to-band relative transform (+0.36, -0.21, +0.25)
                    # of the passing far-staging grasp (the unmoved base missed
                    # with qerr 0.98, measured).
                    # x -0.45: at -0.55 the reach to the face-down panel's north
                    # band (z 0.33) stalled positionally with perfect orientation
                    # — the same pose converges 8 cm higher, i.e. the forearm
                    # grazes the table at full extension. 10 cm less extension
                    # lifts the elbow.
                    # WORKCELL ARRANGEMENT (user layout, 2026-08-10): both arms SIDE
                    # BY SIDE on the south half, facing north (+y, yaw +90) over the
                    # parts lane — the back slides east along the lane to the base
                    # and never crosses an arm's footprint. Same yaw => same tuck
                    # (no mirror needed). Right arm owns insertion + both nut
                    # threads (stud line 0.38-0.52 m from its base, measured on
                    # paper); left assists at the back's west end.
                    "left": ("franka", FrankaRobotCfg(
                        base_pos=(-0.28, -0.45, 0.25),
                        base_rot=(0.70711, 0.0, 0.0, 0.70711), kp_null=1.0,
                        disable_arm_gravity=True, control_mode="impedance",
                        task_prop_gains=(500.0, 500.0, 500.0, 90.0, 90.0, 90.0),
                        default_dof_pos=(0.0, -1.05, 0.0, -2.55, 0.75, 1.65, 0.5))),
                    "right": ("franka", FrankaRobotCfg(
                        base_pos=(0.42, -0.45, 0.25),
                        base_rot=(0.70711, 0.0, 0.0, 0.70711), kp_null=1.0,
                        disable_arm_gravity=True, control_mode="impedance",
                        task_prop_gains=(500.0, 500.0, 500.0, 90.0, 90.0, 90.0),
                        default_dof_pos=(0.0, -1.05, 0.0, -2.55, 0.75, 1.65, 0.5))),
                }),
                env_spacing=3,
            )
        ),
    )

