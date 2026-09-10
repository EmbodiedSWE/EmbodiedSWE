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
    AttachedArmRobotCfg,
    BimanualFrankaCfg,
    BimanualPiperCfg,
    CobottaPro1300RobotCfg,
    FrankaRobotCfg,
    G1RobotCfg,
    GR1T2RobotCfg,
    Jaco2N7RobotCfg,
    PiperRobotCfg,
    WxaiRobotCfg,
    XArm7RobotCfg,
)
from robobench.suites.assembly.scenes import (
    AllenBoltAssemblySceneCfg,
    BulbAssemblySceneCfg,
    IkeaTableAssemblySceneCfg,
    NutThreadAssemblySceneCfg,
    PcGpuAssemblySceneCfg,
    PcGpuRamAssemblySceneCfg,
    PcMotherboardAssemblySceneCfg,
    PcMotherboardGpuRamAssemblySceneCfg,
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
# Five control modes, switchable by env name:
#   - "assembly.pc_motherboard.franka.osc"       — operational-space control (default)
#   - "assembly.pc_motherboard.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.pc_motherboard.franka.diff_ik"   — differential IK (joint position targets)
#   - "assembly.pc_motherboard.franka.pink_ik"   — Pink QP IK (joint position targets)
#   - "assembly.pc_motherboard.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "diff_ik", "pink_ik", "joint"):
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

# The SAME pc-motherboard cell for the transfer suite: gen3n7_panda + xarm7 (panda-hand dial).
# Scene layout verbatim from the franka binding (7 holes spanning the 0.36-0.60 m top-down band
# from the west base, key upright in its stand). gen3n7 (~0.9 m) keeps the franka's base spot;
# the xarm7 (~0.70 m) moves 5 cm east so the far holes pull from its reach edge (0.60 -> 0.55 m)
# while the near holes stay outside the close-in cliff.
#   -> "assembly.pc_motherboard.{gen3n7_panda,xarm7}.{osc,impedance,joint}"
_PC_MB_SCENE_KW = dict(
    workbench_pos=(0.54, 0.0),
    bolt_init_xy=tuple((-0.24 + k * 0.075, -0.42) for k in range(7)),
    key_init_xy=(-0.24, -0.30),
    key_init_z=0.001,
    key_init_quat=(1.0, 0.0, 0.0, 0.0),
    key_stand=True,
    reset_pos_jitter=0.0,
)
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="pc_motherboard",
            scene_cfg=PcMotherboardAssemblySceneCfg(**_PC_MB_SCENE_KW),
            robot="gen3n7_panda",
            robot_cfg=AttachedArmRobotCfg(base_pos=(0.07, 0.0, 0.0),
                                          arm_effort_limit=120.0, gravity_compensation=True,
                                          # probe-verified ready pose (2026-08-18): key stand +
                                          # both hole-row extremes track to <= 1.0 cm from here
                                          # (the class home marched 20 cm wide in a bad branch)
                                          default_dof_pos=(-0.1329, 0.2094, 0.1780, 1.9612,
                                                           -0.8736, 0.6256, -0.5926)),
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 240.0},
        ),
    )
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="pc_motherboard",
            scene_cfg=PcMotherboardAssemblySceneCfg(**_PC_MB_SCENE_KW),
            robot="xarm7",
            robot_cfg=XArm7RobotCfg(gripper="panda_hand", base_pos=(0.12, 0.0, 0.0),
                                    arm_effort_limit=120.0, gravity_compensation=True,
                                    # probe-verified ready pose (2026-08-18): <= 1.0 cm at the
                                    # key stand + hole extremes (the stock folded home could
                                    # not descend — 22 cm z-stall)
                                    default_dof_pos=(-4.4657, 0.8562, -1.9503, 0.6564,
                                                     0.3913, 0.5140, -2.3288)),
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
# Five control modes, switchable by env name:
#   - "assembly.pc_gpu_ram.franka.osc"       — operational-space control (default)
#   - "assembly.pc_gpu_ram.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.pc_gpu_ram.franka.diff_ik"   — differential IK (joint position targets)
#   - "assembly.pc_gpu_ram.franka.pink_ik"   — Pink QP IK (joint position targets)
#   - "assembly.pc_gpu_ram.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "diff_ik", "pink_ik", "joint"):
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

# The SAME pc-gpu-ram work cell for the arms proven on BOTH parent tasks (pc_gpu + pc_ram) —
# scene layout copied VERBATIM from the franka binding above so every embodiment faces the
# identical task; base placement follows each arm's pc_gpu spot shifted with the cell (the
# case moves 40 mm north here and the base follows, as the franka's did). xArm7 is excluded:
# its vendor hand cannot make the task's stick pinches (the reason its pc_ram binding was
# removed), and this task installs both sticks.
#   -> "assembly.pc_gpu_ram.{jaco2_n7, cobotta_pro_1300}.{osc,impedance,joint}"
_PC_GPU_RAM_ARMS = (
    ("jaco2_n7", Jaco2N7RobotCfg, dict(
        # the pc_ram cell's NORTH spot shifted with the cell, then 40 mm WEST: the far
        # (slot-0) align/press fold had no elbow margin from 0.60 (aligns escaped on
        # timeout and the press wandered ~86 mm; measured across the order-flip A/B) —
        # 0.56 keeps every pick at this short arm's proven 0.27+ m band and relieves the
        # far-slot fold
        base_pos=(0.56, -0.26, 0.0),
        base_rot=(0.0, 0.0, 0.0, 1.0),
        default_dof_pos=(0.1629, 2.2989, -0.2511, 0.7993, -2.3822, -0.9424, -0.0279),
        arm_effort_limit=120.0,
        gravity_compensation=True,
    )),
    ("cobotta_pro_1300", CobottaPro1300RobotCfg, dict(
        default_dof_pos=(-0.2689, -0.2062, 2.3545, -0.0002, 0.9898, -0.2685),
        arm_effort_limit=150.0,
        # east+south of the franka spot: the near stick's pick would sit at a 0.28 m radius
        # from there and this 1.3 m arm's elbow tops out at its limit folding that close
        # (measured: j3 pinned at +2.46, margin 0.15, hand 237 mm short); from here the
        # stick picks sit at 0.34-0.49 m and the in-case work at ~0.55 m — all comfortable
        base_pos=(0.78, -0.34, 0.0), base_rot=(0.0, 0.0, 0.0, 1.0),
    )),
)
for _robot, _cfg_cls, _kw in _PC_GPU_RAM_ARMS:
    for _mode in ("osc", "impedance", "joint"):
        register_env(
            SUITE,
            lambda robot=_robot, cfg_cls=_cfg_cls, kw=_kw, mode=_mode: EnvCfg(
                scene="pc_gpu_ram",
                scene_cfg=PcGpuRamAssemblySceneCfg(
                    case_xy=(0.55, 0.04),
                    card_init_xy=(-0.185, -0.321),
                    card_init_z=0.030,
                    card_init_quat=(0.70711, 0.0, 0.0, 0.70711),
                    ram_init_xy=((-0.26, -0.32), (-0.11, -0.32)),
                    ram_init_quat=(1.0, 0.0, 0.0, 0.0),
                    ram_init_z=0.030,
                    reset_pos_jitter=0.0,
                    card_stand=True,
                    ram_stand=True,
                ),
                robot=robot,
                robot_cfg=cfg_cls(**kw),
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
# Five control modes, switchable by env name:
#   - "assembly.pc_ram.franka.osc"       — operational-space control (default)
#   - "assembly.pc_ram.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.pc_ram.franka.diff_ik"   — differential IK (joint position targets)
#   - "assembly.pc_ram.franka.pink_ik"   — Pink QP IK (joint position targets)
#   - "assembly.pc_ram.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "diff_ik", "pink_ik", "joint"):
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

# The SAME pc-ram work cell for the attached-gripper composites — the scene layout (stick
# holders at world (0.30, -0.36) and (0.42, -0.36), sticks staged upright, deterministic spawn,
# sim dt 1/240) is copied VERBATIM from the franka binding above so every embodiment faces the
# identical task; only the robot changes. Base placement and home posture are per-embodiment
# placement dials (see AttachedArmRobotCfg for the dial docs).
#   -> "assembly.pc_ram.<robot>.{osc,impedance,joint}" for each composite below.
_PC_RAM_COMPOSITE_KW: dict[str, dict] = {
    # default_dof_pos: per-embodiment ready posture at the work cell. base_pos overrides the
    # shared spot where an arm's reach calls for it.
    "z1_lite6g": dict(zero_joint_friction=True,  # the asset authors jointFriction 1.0-2.0
                      gravity_compensation=True,  # 2 kg-class arm (see the cfg dial docs)
                      arm_effort_limit=60.0,
                      base_pos=(0.58, -0.28, 0.0),  # 0.74 m reach: base beside the case
                      default_dof_pos=(0.4186, 1.5808, -0.7638, 0.7383, -0.0009, -1.1521)),
    "rizon4_panda": dict(arm_effort_limit=150.0,
                        gravity_compensation=True,  # the real device's controller actively
                        # gravity-compensates (as all the cobots here; see Jaco2N7RobotCfg)
                        default_dof_pos=(0.6362, -0.4773, -0.1640, 2.5330, 0.5056, 1.4094, -1.5786)),
    "gen3n7_panda": dict(arm_effort_limit=120.0,  # authored wrist ratings are 9 N*m
                        gravity_compensation=True,
                        default_dof_pos=(-0.0286, 0.3731, -0.1231, 2.0578, 0.0679, 0.7191, 1.3759)),
    "sawyer_egk25": dict(arm_effort_limit=150.0,
                         gravity_compensation=True,  # see rizon4's note
                         default_dof_pos=(0.2919, -1.4106, -0.8423, 2.1866, -0.6818, -0.8069, 2.3385),
                         nullspace_dof_pos=(0.2919, -1.4106, -0.8423, 2.1866, -0.6818, -0.8069, 2.3385)),
    "festo_panda": dict(arm_effort_limit=150.0,
                       gravity_compensation=True,  # authored masses are all zero
                       default_dof_pos=(-0.1269, -0.5584, 0.4606, 0.0000, 0.5567, 1.4435)),
}
for _robot in ("z1_lite6g", "rizon4_panda", "gen3n7_panda", "sawyer_egk25", "festo_panda"):
    for _mode in ("osc", "impedance", "joint"):
        register_env(
            SUITE,
            lambda robot=_robot, mode=_mode: EnvCfg(
                scene="pc_ram",
                scene_cfg=PcRamAssemblySceneCfg(
                    ram_init_xy=((-0.25, -0.36), (-0.13, -0.36)),  # table-rel -> world (0.30/0.42, -0.36)
                    ram_init_z=0.030,  # blade-bottom plane = the holders' floor top
                    ram_init_quat=(1.0, 0.0, 0.0, 0.0),  # upright, the seated orientation
                    reset_pos_jitter=0.0,
                    ram_stand=True,
                ),
                robot=robot,
                # shared base placement (yaw 180: faces -x); per-robot dials may override it
                # (the 0.74 m z1 sits closer to the cell)
                robot_cfg=AttachedArmRobotCfg(
                    **{"base_pos": (0.72, -0.34, 0.0), "base_rot": (0.0, 0.0, 0.0, 1.0),
                       **_PC_RAM_COMPOSITE_KW[robot]},
                ),
                control_mode=mode,
                env_spacing=2,
                sim_overrides={"dt": 1.0 / 240.0},
            ),
        )

# The same pc-ram work cell for the xArm7 (panda-hand dial) — completing the transfer suite's
# pc_ram coverage. Scene cfg verbatim as above; the ~0.70 m arm takes the Jaco2's closer base
# spot (the shared 0.72 m mount leaves the far DIMM slot at its reach edge). Ready posture =
# the pc_gpu xarm7 binding's, tuned for the same yaw-180 stance at this cell.
#   -> "assembly.pc_ram.xarm7.{osc, impedance, joint}"
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
            robot="xarm7",
            robot_cfg=XArm7RobotCfg(
                gripper="panda_hand",
                base_pos=(0.60, -0.30, 0.0), base_rot=(0.0, 0.0, 0.0, 1.0),
                arm_effort_limit=120.0,
                gravity_compensation=True,  # gravity-blind task-space laws (the bulb binding's note)
                default_dof_pos=(-0.0659, -0.3051, 0.0759, 0.6345, 0.0281, 0.9358, -0.0097),
            ),
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 240.0},
        ),
    )

# The same pc-ram work cell for two of the pc_gpu embodiments (Jaco2 N7 / Cobotta Pro 1300) —
# scene cfg verbatim as above so every embodiment faces the identical task; per-embodiment base
# placement + ready posture only.
#   -> "assembly.pc_ram.{jaco2_n7, cobotta_pro_1300}.{osc, impedance, joint}"
_PC_RAM_PCGPU_ARMS = (
    ("jaco2_n7", Jaco2N7RobotCfg, dict(
        base_pos=(0.60, -0.30, 0.0), base_rot=(0.0, 0.0, 0.0, 1.0),
        arm_effort_limit=120.0,
        gravity_compensation=True,  # see Jaco2N7RobotCfg.gravity_compensation
        default_dof_pos=(0.1629, 2.2989, -0.2511, 0.7993, -2.3822, -0.9424, -0.0279),
    )),
    ("cobotta_pro_1300", CobottaPro1300RobotCfg, dict(
        base_pos=(0.72, -0.34, 0.0), base_rot=(0.0, 0.0, 0.0, 1.0),
        arm_effort_limit=150.0,  # the vendored asset authors 60 N*m per joint
        default_dof_pos=(-0.2256, -0.1725, 2.3889, 0.0003, 0.9203, 1.3454),
    )),
)
for _name, _cfg_cls, _kw in _PC_RAM_PCGPU_ARMS:
    for _mode in ("osc", "impedance", "joint"):
        register_env(
            SUITE,
            lambda name=_name, cfg_cls=_cfg_cls, kw=_kw, mode=_mode: EnvCfg(
                scene="pc_ram",
                scene_cfg=PcRamAssemblySceneCfg(
                    ram_init_xy=((-0.25, -0.36), (-0.13, -0.36)),  # table-rel -> world (0.30/0.42, -0.36)
                    ram_init_z=0.030,  # blade-bottom plane = the holders' floor top
                    ram_init_quat=(1.0, 0.0, 0.0, 0.0),  # upright, the seated orientation
                    reset_pos_jitter=0.0,
                    ram_stand=True,
                ),
                robot=name,
                robot_cfg=cfg_cls(**kw),
                control_mode=mode,
                env_spacing=2,
                sim_overrides={"dt": 1.0 / 240.0},
            ),
        )

# Franka arm at the allen-bolt scene (base at the origin). Placement follows the reach limits the
# sibling franka envs are laid out around: the platform is pulled from the table preset's 0.50 m
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
# Five control modes, switchable by env name:
#   - "assembly.allen_bolt.franka.osc"       — operational-space control (default)
#   - "assembly.allen_bolt.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.allen_bolt.franka.diff_ik"   — differential IK (joint position targets)
#   - "assembly.allen_bolt.franka.pink_ik"   — Pink QP IK (joint position targets)
#   - "assembly.allen_bolt.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "diff_ik", "pink_ik", "joint"):
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

# The SAME allen-bolt cell for the transfer suite: gen3n7_panda + xarm7 (panda-hand dial).
# Scene layout verbatim from the franka binding — platform 0.42 m, key pick ~0.36 m with its
# far tip 0.43 m out — inside both reach envelopes (gen3n7 ~0.9 m, xarm7 ~0.70 m), so the base
# stays at the origin. arm_effort_limit + gravity_compensation follow each arm's bulb/pc_ram kw.
#   -> "assembly.allen_bolt.{gen3n7_panda,xarm7}.{osc,impedance,joint}"
_ALLEN_SCENE_KW = dict(
    platform_slots=((-0.08, 0.0),),
    bolt_staged=True,
    key_init_xy=((-0.24, 0.18),),
    key_init_quat=(0.5, 0.5, 0.5, 0.5),
    bolt_friction=0.3,
    reset_pos_jitter=0.0,
)
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="allen_bolt",
            scene_cfg=AllenBoltAssemblySceneCfg(**_ALLEN_SCENE_KW),
            robot="gen3n7_panda",
            robot_cfg=AttachedArmRobotCfg(arm_effort_limit=120.0, gravity_compensation=True),
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 240.0},
        ),
    )
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="allen_bolt",
            scene_cfg=AllenBoltAssemblySceneCfg(**_ALLEN_SCENE_KW),
            robot="xarm7",
            robot_cfg=XArm7RobotCfg(gripper="panda_hand", arm_effort_limit=120.0,
                                    gravity_compensation=True),
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
# Five control modes, switchable by env name:
#   - "assembly.nut_thread.franka.osc"       — arm by operational-space control (inertia-shaped; default,
#                                              smooth on this arm)
#   - "assembly.nut_thread.franka.impedance" — arm by Jacobian-transpose task-space impedance (Isaac's form)
#   - "assembly.nut_thread.franka.diff_ik"   — differential IK (joint position targets)
#   - "assembly.nut_thread.franka.pink_ik"   — Pink QP IK (joint position targets)
#   - "assembly.nut_thread.franka.joint"     — arm by direct joint position targets
# (all carry a 2-finger gripper by direct position target.)
for _mode in ("osc", "impedance", "diff_ik", "pink_ik", "joint"):
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

# The SAME nut-thread cell for the transfer suite: xarm7 (panda-hand dial) + the four panda-hand
# composites at the franka oracle's flat layout (bolt at world 0.50 m, nut pick at 0.38 m, dt
# 1/480 for the SDF threads). Ready poses re-aim the bulb bindings' hand-down branches at this
# scene's +x work line.
#   -> "assembly.nut_thread.<robot>.{osc,impedance,joint}"
_NUT_SCENE = dict(nut_init_xy=((-0.12, 0.0),), nut_friction=0.4)
_NUT_SCENE_BY_ROBOT: dict[str, dict] = {}
# rizon4/festo/sawyer: no nut binding — their posture creep forms off-axis hex grips the
# wrench recipe cannot rotate around.
_NUT_COMPOSITE_KW: dict[str, dict] = {
    "gen3n7_panda": dict(arm_effort_limit=120.0, gravity_compensation=True),
}
for _robot in ("gen3n7_panda",):
    for _mode in ("osc", "impedance", "joint"):
        register_env(
            SUITE,
            lambda robot=_robot, mode=_mode: EnvCfg(
                scene="nut_thread",
                scene_cfg=NutThreadAssemblySceneCfg(**_NUT_SCENE_BY_ROBOT.get(robot, _NUT_SCENE)),
                robot=robot,
                robot_cfg=AttachedArmRobotCfg(**_NUT_COMPOSITE_KW[robot]),
                control_mode=mode,
                env_spacing=2,
                sim_overrides={"dt": 1.0 / 480.0},
            ),
        )
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="nut_thread",
            scene_cfg=NutThreadAssemblySceneCfg(**_NUT_SCENE),
            robot="xarm7",
            # vendor gripper: the 24 mm M16 suits its linkage natively (unlike the bulb)
            robot_cfg=XArm7RobotCfg(arm_effort_limit=120.0, gravity_compensation=True),
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
# Five control modes, switchable by env name:
#   - "assembly.bulb.franka.osc"       — operational-space control (default)
#   - "assembly.bulb.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.bulb.franka.diff_ik"   — differential IK (joint position targets)
#   - "assembly.bulb.franka.pink_ik"   — Pink QP IK (joint position targets)
#   - "assembly.bulb.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "diff_ik", "pink_ik", "joint"):
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

# The SAME bulb cell for the xArm7, carrying the Franka hand (`gripper` is a robot-cfg dial; two
# driven prismatic fingers in METRES, 0 = closed, 80 mm aperture, action 6 pose deltas + 2 = 8 —
# a sim-only pairing used for controlled cross-embodiment comparison: threading this bulb needs
# a pad gap comfortably wider than the 48 mm glass belly, which rules out the arm's narrower
# real-gripper options).
# The scene layout (socket 9 cm closer at table-rel (-0.09, 0), loose bulb on the
# +y side at (-0.24, 0.25), default sim dt) is copied VERBATIM from the franka binding above so
# both embodiments face the identical task. Base at the origin facing +x, the franka's spot:
# both arms clear the socket at 0.41 m and the bulb at (0.26, 0.25) well inside their envelopes
# (xArm7 ~0.75 m reach vs the franka's ~0.85 m). arm_effort_limit is raised over the asset's
# authored ratings (50/50/30/30/30/20/20) for gravity-uncompensated torque control — the same
# dial the pc_gpu xarm7 binding uses.
#   - "assembly.bulb.xarm7.osc"       — operational-space control (default)
#   - "assembly.bulb.xarm7.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.bulb.xarm7.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="bulb",
            scene_cfg=BulbAssemblySceneCfg(
                socket_slots=((-0.09, 0.0),),
                bulb_init_xy=((-0.24, 0.25),),
            ),
            robot="xarm7",
            robot_cfg=XArm7RobotCfg(
                gripper="panda_hand",
                arm_effort_limit=120.0,
                gravity_compensation=True,  # the task-space laws are gravity-blind; this arm cannot
                # hold itself against gravity across the cell's reaches (the same dial the jaco2 and
                # the pc_ram composites set)
            ),
            control_mode=mode,
            env_spacing=2,
        ),
    )


# The SAME bulb cell for the four panda-hand composites — the transfer-learning suite: six
# arms (franka, xarm7, and these four), one task, one end-effector. Scene layout verbatim from
# the franka binding; base at the origin facing +x (socket 0.41 m / bulb 0.36 m sit inside every
# reach envelope); arm_effort_limit + gravity_compensation follow each arm's verified pc_ram kw.
#   -> "assembly.bulb.<robot>.{osc,impedance,joint}" for each composite below.
_BULB_COMPOSITE_KW: dict[str, dict] = {
    "rizon4_panda": dict(arm_effort_limit=150.0, gravity_compensation=True,
                         # 0.36 m is inside this arm's close-in cliff for a hand-down reach:
                         # base back 15 cm puts the bulb at 0.44 m and the socket at 0.56 m
                         base_pos=(-0.15, 0.0, 0.0)),
    "gen3n7_panda": dict(arm_effort_limit=120.0, gravity_compensation=True),
    # the stock home leaves the 6-DOF wrist in a branch where hand-down is unreachable
    # (a6 pins at -135deg); this home seeds the down-facing branch, a6 mid-range
    "festo_panda": dict(arm_effort_limit=150.0, gravity_compensation=True,
                        default_dof_pos=(0.7679, -0.6, -1.4, 0.0, 2.35, 0.0)),
    # ready pose for the corrected tool-axis mount: hand straight down over the table,
    # every joint mid-range (the Intera neutral leaves the EE half a metre from the work)
    "sawyer_panda": dict(arm_effort_limit=150.0, gravity_compensation=True,
                         default_dof_pos=(0.7679, -1.18, 0.0, 1.5, 0.0, 1.22, 0.0)),
}
for _robot in ("rizon4_panda", "gen3n7_panda", "festo_panda", "sawyer_panda"):
    for _mode in ("osc", "impedance", "joint"):
        register_env(
            SUITE,
            lambda robot=_robot, mode=_mode: EnvCfg(
                scene="bulb",
                scene_cfg=BulbAssemblySceneCfg(
                    socket_slots=((-0.09, 0.0),),
                    bulb_init_xy=((-0.24, 0.25),),
                ),
                robot=robot,
                robot_cfg=AttachedArmRobotCfg(**_BULB_COMPOSITE_KW[robot]),
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
# Five control modes, switchable by env name:
#   - "assembly.pc_gpu.franka.osc"       — operational-space control (default)
#   - "assembly.pc_gpu.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.pc_gpu.franka.diff_ik"   — differential IK (joint position targets)
#   - "assembly.pc_gpu.franka.pink_ik"   — Pink QP IK (joint position targets)
#   - "assembly.pc_gpu.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "diff_ik", "pink_ik", "joint"):
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

# The SAME pc-gpu work cell for the other single-arm embodiments — the scene layout (card holder
# at world (0.28, -0.36), card staged upright in the scene's foam holder, deterministic spawn,
# sim dt 1/240) is copied VERBATIM from the franka binding above so every embodiment faces the
# identical task; only the robot changes. Base placement and home posture are per-embodiment
# placement dials, retuned per arm exactly like every franka binding's base_pos (the dial docs
# live on each robot's cfg).
#   - "assembly.pc_gpu.xarm7.{osc,impedance,joint}"            — UFACTORY xArm7 + vendor gripper
#   - "assembly.pc_gpu.jaco2_n7.{osc,impedance,joint}"         — Kinova Jaco2 7-DOF, 3-finger hand
#   - "assembly.pc_gpu.cobotta_pro_1300.{osc,impedance,joint}" — Denso Cobotta Pro 1300 + RG6
_PC_GPU_ROBOT_KW: dict[str, dict] = {
    # xArm7 / Cobotta: the franka's north-strip spot (base defaults below). arm_effort_limit is
    # raised over the assets' authored ratings for gravity-uncompensated torque control — the
    # franka cfg documents the same dial.
    "xarm7": dict(
        default_dof_pos=(-0.0659, -0.3051, 0.0759, 0.6345, 0.0281, 0.9358, -0.0097),
        arm_effort_limit=120.0,
    ),
    "jaco2_n7": dict(
        # A short (0.9 m) assistive arm with a large 3-finger hand: it works from the SOUTH
        # strip between the card holder and the case (the north-strip spot lies outside its
        # comfortable envelope), yaw +90 so the arm faces the work to the north.
        base_pos=(0.42, -0.50, 0.0),
        base_rot=(0.70710678, 0.0, 0.0, 0.70710678),
        default_dof_pos=(-1.9644, 1.8449, 0.0412, 0.8507, -0.7280, 2.8869, 2.8141),
        arm_effort_limit=120.0,
        gravity_compensation=True,  # the real device's controller actively gravity-compensates
        # (see Jaco2N7RobotCfg.gravity_compensation)
    ),
    "cobotta_pro_1300": dict(
        default_dof_pos=(-0.2689, -0.2062, 2.3545, -0.0002, 0.9898, -0.2685),
        arm_effort_limit=150.0,  # authored 60 per joint; a 1.3 m arm needs more at full stretch
    ),
}
for _robot, _cfg_cls in (
    ("xarm7", XArm7RobotCfg),
    ("jaco2_n7", Jaco2N7RobotCfg),
    ("cobotta_pro_1300", CobottaPro1300RobotCfg),
):
    for _mode in ("osc", "impedance", "joint"):
        register_env(
            SUITE,
            lambda robot=_robot, cfg_cls=_cfg_cls, mode=_mode: EnvCfg(
                scene="pc_gpu",
                scene_cfg=PcGpuAssemblySceneCfg(
                    card_init_xy=(-0.22, -0.36),  # table-relative -> world (0.28, -0.36)
                    card_init_z=0.030,  # tab-bottom plane = the holder's floor top
                    card_init_quat=(1.0, 0.0, 0.0, 0.0),  # upright, the seated orientation
                    reset_pos_jitter=0.0,
                    card_stand=True,
                ),
                robot=robot,
                robot_cfg=cfg_cls(
                    # per-robot base placement: kwargs override the franka's north-strip default
                    **{"base_pos": (0.64, -0.34, 0.0), "base_rot": (0.0, 0.0, 0.0, 1.0),
                       **_PC_GPU_ROBOT_KW[robot]},
                ),
                control_mode=mode,
                env_spacing=2,
                sim_overrides={"dt": 1.0 / 240.0},
            ),
        )

# The SAME pc-gpu work cell for the panda-hand composites — scene layout copied VERBATIM from
# the franka binding above so every embodiment faces the identical task; shared base at the
# franka's north-strip spot, per-arm actuator/posture dials as in their pc_ram bindings.
#   -> "assembly.pc_gpu.{rizon4_panda, gen3n7_panda, festo_panda}.{osc,impedance,joint}"
_PC_GPU_COMPOSITE_KW: dict[str, dict] = {
    "rizon4_panda": dict(arm_effort_limit=150.0, gravity_compensation=True,
                         default_dof_pos=(0.6362, -0.4773, -0.1640, 2.5330, 0.5056, 1.4094, -1.5786)),
    "gen3n7_panda": dict(arm_effort_limit=120.0,  # authored wrist ratings are 9 N*m
                         gravity_compensation=True,
                         default_dof_pos=(-0.0286, 0.3731, -0.1231, 2.0578, 0.0679, 0.7191, 1.3759)),
    "festo_panda": dict(arm_effort_limit=150.0,
                        gravity_compensation=True,  # authored masses are all zero
                        default_dof_pos=(-0.1269, -0.5584, 0.4606, 0.0000, 0.5567, 1.4435)),
}
for _robot in ("rizon4_panda", "gen3n7_panda", "festo_panda"):
    for _mode in ("osc", "impedance", "joint"):
        register_env(
            SUITE,
            lambda robot=_robot, mode=_mode: EnvCfg(
                scene="pc_gpu",
                scene_cfg=PcGpuAssemblySceneCfg(
                    card_init_xy=(-0.22, -0.36),  # table-relative -> world (0.28, -0.36)
                    card_init_z=0.030,  # tab-bottom plane = the holder's floor top
                    card_init_quat=(1.0, 0.0, 0.0, 0.0),  # upright, the seated orientation
                    reset_pos_jitter=0.0,
                    card_stand=True,
                ),
                robot=robot,
                robot_cfg=AttachedArmRobotCfg(
                    **{"base_pos": (0.64, -0.34, 0.0), "base_rot": (0.0, 0.0, 0.0, 1.0),
                       **_PC_GPU_COMPOSITE_KW[robot]},
                ),
                control_mode=mode,
                env_spacing=2,
                sim_overrides={"dt": 1.0 / 240.0},
            ),
        )

# Two Frankas at the SO101 workbench as ONE robot (`BimanualFranka`: action = [left | right];
# address one arm via `env.robot["left"]`). Bases stand on the bench top (z = 0.994, the default
# packing table), 0.80 m apart, the layout the full-arm assembly was solved at: the left arm at
# the bench origin facing +x covers the screw band, the drill zone and the drive rests; the
# right arm at (0.75, -0.28) yaw 135 covers the workpiece drags, the fixture holds and the
# distal delivery corner. Each arm works best 0.3-0.55 m from its own base; the shared zone
# sits around (0.4, -0.1). Known gotcha: the default home pose can park a hand over the other
# arm's zone — tuck the idle arm.
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
                    "left": ("franka", FrankaRobotCfg(  # yaw 0, faces the work zone from the origin
                        base_pos=(0.0, 0.0, 0.994), base_rot=(1.0, 0.0, 0.0, 0.0),
                        nullspace_dof_pos=())),  # nullspace pulls toward the home pose
                    "right": ("franka", FrankaRobotCfg(  # yaw +135 deg, faces the drag/fixture zone
                        base_pos=(0.75, -0.28, 0.994), base_rot=(0.38268, 0.0, 0.0, 0.92388),
                        # home = the stock pose with q1 swung -1.0 rad: the hand spawns parked
                        # south of the bench center instead of looming over the shared work zone
                        default_dof_pos=(-1.0, -0.197, -0.0014, -1.976, -0.00028, 1.78, 0.786),
                        nullspace_dof_pos=(0.0015, -0.197, -0.0014, -1.976, -0.00028, 1.78,
                                           0.786))),
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


# ============================== pc_motherboard_gpu_ram (the complete build) =======================
# The COMPLETE PC install in one episode: the same gaming-PC case with ALL THREE work sites open
# at once — 7 case-mount bolts staged hand-started in the motherboard's holes (one allen key
# beside the case drives them), two empty DIMM slots with two loose RAM sticks, and the empty
# PCIe x16 slot (+ rear cutout) with a loose graphics card. Assembly order: fasten the board
# down, seat the dual-channel stick pair, then install the card through the rear cutout.
# Physics-only binding (no arm).
# -> "assembly.pc_motherboard_gpu_ram"
register_env(SUITE, lambda: EnvCfg(scene="pc_motherboard_gpu_ram", robot="null", env_spacing=2))

# Franka arm at the complete build — the pc_motherboard.franka work cell: base west of the case
# at (0.07, 0) facing +x, the table slid 40 mm east (`workbench_pos` = `case_xy` = (0.54, 0)).
# The allen key stands tip-down in its four-wall stand, and the card and both sticks stand
# upright in foam holders, all staged on the table's south-west side. Deterministic spawn (no
# jitter): the stands are static geometry authored at the spawn points. sim dt 1/240.
#   - "assembly.pc_motherboard_gpu_ram.franka.osc"       — operational-space control (default)
#   - "assembly.pc_motherboard_gpu_ram.franka.impedance" — Jacobian-transpose task-space impedance
#   - "assembly.pc_motherboard_gpu_ram.franka.joint"     — direct joint position targets
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(
            scene="pc_motherboard_gpu_ram",
            scene_cfg=PcMotherboardGpuRamAssemblySceneCfg(
                workbench_pos=(0.54, 0.0),
                case_xy=(0.54, 0.0),  # case at the table anchor — the motherboard cell layout
                key_init_xy=(-0.18, -0.30),  # table-rel -> world (0.36, -0.30)
                key_init_z=0.007,  # tip 1 mm above the stand's 6 mm floor pad
                key_init_quat=(1.0, 0.0, 0.0, 0.0),  # standing tip-down in the stand
                key_stand=True,
                card_init_xy=(-0.54, -0.44),  # table-rel -> world (0.00, -0.44): a second row
                # south-west of the stick holders
                card_init_z=0.030,  # tab-bottom plane = the holder's floor top
                card_init_quat=(1.0, 0.0, 0.0, 0.0),  # upright, the seated heading (length along x)
                ram_init_xy=((-0.41, -0.335), (-0.325, -0.335)),  # table-rel -> world
                # (0.13/0.215, -0.335): the south-west staging strip, west of the key stand
                ram_init_quat=(1.0, 0.0, 0.0, 0.0),  # upright, the seated orientation
                ram_init_z=0.030,  # blade-bottom plane = the holders' floor top
                reset_pos_jitter=0.0,
                card_stand=True,
                ram_stand=True,
            ),
            robot="franka",
            robot_cfg=FrankaRobotCfg(base_pos=(0.07, 0.0, 0.0)),
            control_mode=mode,
            env_spacing=2,
            sim_overrides={"dt": 1.0 / 240.0},
        ),
    )
