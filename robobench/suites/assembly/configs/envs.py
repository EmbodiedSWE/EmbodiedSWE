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
from robobench.suites.assembly.scenes import BulbAssemblySceneCfg, IkeaTableAssemblySceneCfg

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

# The same PC case with its primary PCIe x16 slot empty and a loose graphics card beside it, to be
# stood upright and pressed straight down into the slot, scene physics only for now.
# -> "assembly.pc_gpu"
register_env(SUITE, lambda: EnvCfg(scene="pc_gpu", robot="null", env_spacing=2))

# SO101 full-arm assembly (seat + screw the elbow servo, clip + screw the forearm fork onto its
# horn) on a workbench, scene physics only.
# -> "assembly.so101"
register_env(SUITE, lambda: EnvCfg(scene="so101", robot="null", env_spacing=2))

# Franka arm at the nut-thread scene (base at the origin, reaching the bolt on the table at +x). Three
# control modes, switchable by env name:
#   - "assembly.nut_thread.franka.osc"       — arm by operational-space control (inertia-shaped; default,
#                                              smooth on this arm)
#   - "assembly.nut_thread.franka.impedance" — arm by Jacobian-transpose task-space impedance (Isaac's form)
#   - "assembly.nut_thread.franka.joint"     — arm by direct joint position targets
# (all carry a 2-finger gripper by direct position target.)
for _mode in ("osc", "impedance", "joint"):
    register_env(
        SUITE,
        lambda mode=_mode: EnvCfg(scene="nut_thread", robot="franka", control_mode=mode, env_spacing=2),
    )

# Franka arm at the bulb scene (base at the origin). The socket + loose bulb are pulled off the stock
# nut_thread "+x row" layout into the arm's solve-verified reach band: the default row put the bulb at
# 0.63 m (out of reach -> REORIENT_STUCK) on the centreline (parks wrist q7 near its stop). Baked in:
# socket 9 cm closer, bulb at the ~0.43 m pick radius on the +y side (q7 margin). Evidence:
# experiments/bulb_franka_osc_fable/SCENE_IMPROVEMENTS.md. (Per-shape bulb friction is already the
# scene default — no override needed.) Three control modes, switchable by env name:
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

# Two Frankas at the SO101 workbench as ONE robot (`BimanualFranka`: action = [left | right];
# address one arm via `env.robot["left"]`). Bases stand on the bench top (z = 0.994, the default
# packing table), 0.94 m apart — each arm works best 0.3-0.55 m from its own base, so the shared
# zone sits around (0.4, 0).
# Reach-verified (P2/P2b probes, experiments/so101_bimanual_franka_osc_20260715): in-zone tracking
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
# The Franka bases + leg spawn below are VERIFIED by the four-leg solve (all tuned constants in
# experiments/ikea_bimanual_franka_fable/solve_four.py are calibrated to them — do not move them).
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
                # Leg spawn baked to the solve-verified layout: no reset jitter (deterministic) and the
                # explicit row that keeps every grip in the right arm's 0.31-0.43 m pick band.
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
