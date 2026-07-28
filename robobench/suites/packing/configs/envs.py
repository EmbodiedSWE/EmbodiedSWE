"""Canonical runnable env configs for the packing suite — registered in `ENVS` by name.

Same convention as the assembly suite: `register_env` derives ``suite.scene[.robot
[.control_mode]]``. Scene-physics-only first (NullRobot smoke/oracle), embodiments after
the physics is proven.
"""

from __future__ import annotations

from robobench.core import EnvCfg, register_env
from robobench.robots import (
    BimanualFrankaCfg,
    FrankaRobotCfg,
    G1RobotCfg,
    GR1T2RobotCfg,
    MultiRobotCfg,
    PiperRobotCfg,
    WxaiRobotCfg,
)
from robobench.suites.packing.scenes import CratePackingSceneCfg, PenHolderSceneCfg

SUITE = "packing"

# Crate packing, scene physics only (NullRobot oracle/smoke). -> "packing.crate"
register_env(SUITE, lambda: EnvCfg(scene="crate", robot="null", env_spacing=3))


def _gr1t2_scene_cfg() -> CratePackingSceneCfg:
    """Bench-height placement for the GR1-T2 (full-size manifest — its arms are longer):
    work raised to a 0.7 m bench; crate pushed slightly away (+y); cargo spawned on a
    front arc (the robot-facing side). Ring/base measured against reach: a 0.42 ring
    put the slab ~0.69 m out, beyond reach."""
    return CratePackingSceneCfg(
        surface_z=0.7,
        crate_pos=(0.0, 0.18),
        spawn_radius=0.36,
        spawn_arc=(205.0, 335.0),  # arc on the robot (-y) side of the crate
    )


# G1 manifest: the brief calls for a SMALLER manifest for the G1 (short ~0.55 m arms).
# Scaled ~0.7x: reference packing = slab (0.03) + tube layer (0.04) + brick (0.06) = 0.13;
# flat single-layer footprint (~0.10 m^2) still exceeds the derived floor (~0.066 m^2).
_G1_MANIFEST = (
    ("slab", "box", (0.28, 0.21, 0.03)),
    ("tube_0", "cyl", (0.02, 0.20)),
    ("tube_1", "cyl", (0.02, 0.20)),
    ("tube_2", "cyl", (0.02, 0.20)),
    ("tube_3", "cyl", (0.02, 0.20)),
    ("brick", "box", (0.12, 0.09, 0.06)),
)


def _g1_scene_cfg() -> CratePackingSceneCfg:
    """G1 variant: smaller manifest -> smaller crate -> everything within the short reach
    (slab spawn measured ~0.48 m from the base)."""
    return CratePackingSceneCfg(
        surface_z=0.7,
        crate_pos=(0.0, 0.14),
        spawn_radius=0.30,
        spawn_arc=(215.0, 325.0),
        manifest=_G1_MANIFEST,
        ref_stack_h=0.13,
    )


# Fixed-base G1 at the bench-height crate. -> "packing.crate.g1.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="crate",
                scene_cfg=_g1_scene_cfg(),
                robot="g1",
                control_mode=mode,
                robot_cfg=G1RobotCfg(base_pos=(0.0, -0.50, 0.75)),
                env_spacing=3,
            )
        ),
    )

# GR1-T2 at the same bench (bimanual slab carry). -> "packing.crate.gr1t2.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="crate",
                scene_cfg=_gr1t2_scene_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.48, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )


# ---- Franka binding ----
# Ground-level crate, gripper-sized manifest: every part graspable by the ~8 cm
# parallel jaw (slab grasped by its 0.02 m edge; tubes 0.03 m dia; brick by its
# 0.06 m side). Reference packing = slab (0.02) + tube layer (0.03) + brick (0.05)
# = 0.10 stack; slab down-scaled so the one-handed edge carry stays plausible.
_FRANKA_ROT = (0.7071068, 0.0, 0.0, 0.7071068)
_FRANKA_MANIFEST = (
    ("slab", "box", (0.20, 0.15, 0.02)),
    ("tube_0", "cyl", (0.015, 0.14)),
    ("tube_1", "cyl", (0.015, 0.14)),
    ("tube_2", "cyl", (0.015, 0.14)),
    ("tube_3", "cyl", (0.015, 0.14)),
    ("brick", "box", (0.09, 0.06, 0.05)),
)


def _franka_scene_cfg() -> CratePackingSceneCfg:
    """Franka variant: table-level crate just in front of the base, compact spawn ring
    inside the ~0.75 m reach."""
    return CratePackingSceneCfg(
        crate_pos=(0.0, 0.12),
        spawn_radius=0.26,
        spawn_arc=(215.0, 325.0),
        manifest=_FRANKA_MANIFEST,
        ref_stack_h=0.10,
    )


for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="crate",
                scene_cfg=_franka_scene_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.40, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )


# ---- Small single-arm bindings (piper / wxai) ----
# Same ground-level crate + gripper-sized manifest as the franka variant. Bases moved
# closer to match the shorter reaches (PiPER ~0.6 m, WidowX AI ~0.5 m); both face +y
# toward the crate like the franka binding.
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="crate",
                scene_cfg=_franka_scene_cfg(),
                robot="piper",
                control_mode=mode,
                robot_cfg=PiperRobotCfg(base_pos=(0.0, -0.30, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )

for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="crate",
                scene_cfg=_franka_scene_cfg(),
                robot="wxai",
                control_mode=mode,
                robot_cfg=WxaiRobotCfg(base_pos=(0.0, -0.25, 0.0), base_rot=_FRANKA_ROT),
                env_spacing=3,
            )
        ),
    )


# ---- Bimanual Franka crate binding: right arm at the proven single-franka pose
# (solutions transfer verbatim); left arm across the crate at +x, outside the spawn
# arc, available as a holder/assist.
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="crate",
                scene_cfg=_franka_scene_cfg(),
                robot="bimanual_franka",
                control_mode=mode,
                robot_cfg=BimanualFrankaCfg(robots={
                    "left": ("franka", FrankaRobotCfg(
                        base_pos=(-0.60, 0.0, 0.0), base_rot=(1.0, 0.0, 0.0, 0.0))),
                    "right": ("franka", FrankaRobotCfg(
                        base_pos=(0.0, -0.40, 0.0), base_rot=_FRANKA_ROT)),
                }),
                env_spacing=3,
            )
        ),
    )

# ---- RoboDojo fill-pen-holder (difficulty-floor tier, bimanual-friendly) ----------------------
# Scene physics only (NullRobot oracle/smoke). -> "packing.pen_holder"
register_env(SUITE, lambda: EnvCfg(scene="pen_holder", robot="null", env_spacing=3))


# Robot bindings. Placements are STARTING guesses copied from the stacking-toy / packing measured
# reach values for the same embodiments at the same bench — re-verify with the per-binding stress
# smoke before any agent run (only the null smoke validates the scene itself). The holder sits to
# the robot's right, pens scatter on a front arc to the left (the source's left/right split).
def _pen_holder_g1_cfg() -> PenHolderSceneCfg:
    """G1 (short ~0.55 m arms): work on a 0.7 m bench, holder right-front, pens on one front arc."""
    return PenHolderSceneCfg(
        surface_z=0.7,
        holder_pos=(0.18, 0.22),
        pens_center=(0.0, 0.16),
        spawn_radii=(0.26,),
        spawn_arc=(215.0, 325.0),
    )


def _pen_holder_gr1t2_cfg() -> PenHolderSceneCfg:
    """GR1-T2 (longer arms): same bench, slightly wider layout."""
    return PenHolderSceneCfg(
        surface_z=0.7,
        holder_pos=(0.20, 0.26),
        pens_center=(0.0, 0.18),
        spawn_radii=(0.30,),
        spawn_arc=(205.0, 335.0),
    )


def _pen_holder_franka_cfg() -> PenHolderSceneCfg:
    """Franka (single arm): ground-level work in front of the base. One arm cannot hold AND
    fill, so the honest single-arm strategy is inserting into the STANDING holder — the rubric
    never requires holding it. Pens (>= 20 mm dia) and the 8 mm holder rim both pinch under
    the 8 cm jaw."""
    return PenHolderSceneCfg(
        holder_pos=(0.12, 0.18),
        pens_center=(-0.10, 0.15),
        spawn_radii=(0.14,),
        spawn_arc=(120.0, 300.0),
    )


def _pen_holder_multi_cfg() -> PenHolderSceneCfg:
    """Dual Franka flanking the work (the genuinely bimanual binding, mapping the source's dual
    ARX X5): holder on the right arm's side, pens on the left arm's side."""
    return PenHolderSceneCfg(
        holder_pos=(0.14, 0.0),
        pens_center=(-0.14, 0.0),
        spawn_radii=(0.16,),
        spawn_arc=(100.0, 260.0),
    )


# -> "packing.pen_holder.g1.{joint,pink_ik}" / ".gr1t2.{joint,pink_ik}"
for _mode in ("joint", "pink_ik"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="pen_holder",
                scene_cfg=_pen_holder_g1_cfg(),
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
                scene="pen_holder",
                scene_cfg=_pen_holder_gr1t2_cfg(),
                robot="gr1t2",
                control_mode=mode,
                robot_cfg=GR1T2RobotCfg(base_pos=(0.0, -0.48, 0.95),
                                        base_rot=(0.7071, 0.0, 0.0, 0.7071)),
                env_spacing=3,
            )
        ),
    )

# -> "packing.pen_holder.franka.{osc,joint}"
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="pen_holder",
                scene_cfg=_pen_holder_franka_cfg(),
                robot="franka",
                control_mode=mode,
                robot_cfg=FrankaRobotCfg(base_pos=(0.0, -0.40, 0.0),
                                         base_rot=(0.7071068, 0.0, 0.0, 0.7071068)),
                env_spacing=3,
            )
        ),
    )

# -> "packing.pen_holder.multi.{osc,joint}" — two Frankas facing each other across the work
# (left at -x facing +x = the default facing; right at +x turned 180 deg). EnvCfg.control_mode
# propagates to both children.
for _mode in ("osc", "joint"):
    register_env(
        SUITE,
        (
            lambda mode=_mode: EnvCfg(
                scene="pen_holder",
                scene_cfg=_pen_holder_multi_cfg(),
                robot="multi",
                control_mode=mode,
                robot_cfg=MultiRobotCfg(robots={
                    "left": ("franka", FrankaRobotCfg(base_pos=(-0.55, 0.0, 0.0))),
                    "right": ("franka", FrankaRobotCfg(base_pos=(0.55, 0.0, 0.0),
                                                       base_rot=(0.0, 0.0, 0.0, 1.0))),
                }),
                env_spacing=3,
            )
        ),
    )
