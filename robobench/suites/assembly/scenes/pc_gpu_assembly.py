"""PcGpuAssemblyScene — insert a graphics card into the PC's PCIe slot.

The same gaming-PC case as `pc_motherboard` lies on its side on the table, opening up, motherboard
facing the ceiling — but here the board's PCIe area is the work site. Beside the case lies a loose
RTX 2060 (extracted from the PC model as its own rigid body, backplate down). Goal (carried here,
no task layer): stand the card upright over the primary x16 slot, line its PCB edge up with the
slot, and press it straight down until it seats.

The case is one kinematic body that never moves — the PC model stays visual-only; its physics is
an invisible fixture inside the case body: a channel whose walls grip the card's 4 mm PCB tab at
0.15 mm/side (flaring to a 1.2 mm/side funnel mouth — idealizing the real slot's spring
contacts, and capping the unscrewed card's gravity roll at ~2 deg), a floor whose top is the
model's own seated tab height, end stops (~1.6 mm play), a flush board plate so a dropped card
rests on the board face, and a REAR I/O PANEL frame around the expansion-slot cutout — so the
card must be placed inside the case, slid rearward until its bracket/ports pass through the
opening, and only then pressed down (a straight vertical drop is physically blocked by the
panel above the cutout, exactly like a real build). The card's origin is its
PCB-tab bottom CENTRE with axes equal to the case's, so the seated pose is just `seat_pos` +
identity orientation in the case frame, and insertion depth is a z difference.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class PcGpuAssemblySceneCfg(BaseCfg):
    """Config for `PcGpuAssemblyScene`. Each field is a `tunable()` curriculum/difficulty dial or
    an `info()` structural constant (see `robobench.core.BaseCfg`)."""

    # --- tunable: the curriculum / difficulty dials -----------------------------------------------
    # The card is "seated" when — in the case's frame — its tab is >= `seat_depth` below the slot
    # mouth, its origin is within `align_xy` of the seated point, and its axes are within
    # `align_axis_deg` (tilt) / `align_yaw_deg` (heading along the slot) of the case's. The full
    # stroke from the slot mouth to the channel floor is 5 mm, so 4 mm separates "seated" from
    # "merely resting in the mouth".
    seat_depth: float = tunable(0.004)  # min tab depth below the slot mouth (m) to count as seated
    align_xy: float = tunable(0.003)  # max distance (m) of the card origin from the seated point
    align_axis_deg: float = tunable(3.0)  # max tilt of the card's up axis off the slot axis (deg)
    align_yaw_deg: float = tunable(3.0)  # max heading error of the card's length axis (deg)
    reset_pos_jitter: float = tunable(0.01)  # uniform +/- xy jitter for the loose card at reset (m)
    # Part friction (static = dynamic), set on every shape at bind. The moving card runs moderately
    # slick against a grippier fixed case, so it slides down the channel but holds seat.
    card_friction: float = tunable(0.3)
    case_friction: float = tunable(0.75)

    # --- info: structure, reset layout, masses, asset paths (fixed) -------------------------------
    # Seated card origin (its PCB-tab bottom centre) in the case's local frame; orientation seated
    # = the case's own axes (identity). Baked into the committed USDs (keep in sync if they change).
    seat_pos: tuple[float, float, float] = info((0.03405, 0.0293, 0.0035))
    slot_mouth_z: float = info(0.0085)  # slot top in the case frame: depth datum (5 mm at full seat)
    board_top: float = info(0.0)  # board face height in the case frame (the asset's own origin)
    case_lift: float = info(0.0289)  # board face above the side panel the case lies on
    card_mass: float = info(1.0)  # dual-fan RTX 2060 (kg)
    light_intensity: float = info(2500.0)
    # Loose card start pose: lying backplate-down on the table beside the case (+x side; the case
    # spans x < 0.13 and the lying card's tail reaches origin_x - 0.132).
    card_init_xy: tuple[float, float] = info((0.28, 0.0))  # card start xy (table-rel.)
    card_init_z: float = info(0.0022)  # origin height lying backplate-down (backplate plane -2 mm)
    card_init_quat: tuple[float, float, float, float] = info((0.70711, 0.70711, 0.0, 0.0))  # flat
    card_contact_offset: float = info(0.0001)  # well below the 0.15 mm/side channel grip
    case_contact_offset: float = info(0.0001)  # ditto for the slot fixture's walls
    # Optional foam holder (a floor pad + two rails flanking the card's 36 mm body slab) that
    # presents the card UPRIGHT for a parallel-jaw grasp. The lying default is ungraspable by a
    # Franka gripper: flat on its backplate the card's only sub-80 mm dimension (the 36 mm body
    # thickness) points UP, so no top-down or side pinch can straddle it. Enable together with an
    # upright `card_init_quat` (identity = the seated orientation) and `card_init_z` = the
    # holder's floor top; the rails cap the free card's lean at ~3 deg and the pick pulls
    # straight up out of them.
    card_stand: bool = info(False)
    card_stand_gap: float = info(0.0025)  # rail clearance per side around the body slab (m)
    # Selectable work surface (same presets as the sibling scenes).
    table: str = info("lab_table")  # which work surface: "lab_table" | "packing"
    surface_z: float | None = info(None)  # table-top height (m); None -> the preset's
    workbench_pos: tuple[float, float] | None = info(None)  # xy the table sits at; None -> preset
    workbench_usd: str = info("")  # empty -> the preset's vendored USD
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.5, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }
    # Asset USDs; empty -> the prebuilt assets committed under `assets/`.
    asset_dir: str = info("")
    case_usd: str = info("")
    card_usd: str = info("")

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets)
        self.case_usd = self.case_usd or str(Path(self.asset_dir) / "pc" / "pc_case_gpu_assembly.usd")
        self.card_usd = self.card_usd or str(Path(self.asset_dir) / "pc" / "gpu_rtx2060.usd")
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(assets / "props" / preset["usd"][0] / preset["usd"][1])


@SCENES.register("pc_gpu")
class PcGpuAssemblyScene(BaseScene):
    cfg: PcGpuAssemblySceneCfg

    # Card-local y extent of the body collision slab (from gpu_rtx2060.usd `/gpu/collision/body`:
    # backplate plane -2 mm, fan-shroud plane +34 mm around the PCB-tab-centre origin). The stand's
    # rails flank THESE faces — the same pair a parallel-jaw grasp pinches.
    CARD_BODY_Y: ClassVar[tuple[float, float]] = (-0.002, 0.034)

    def __init__(self, cfg: PcGpuAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or PcGpuAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, dome light, table, the PC case lying on it (kinematic, with the invisible slot
        fixture), and one loose graphics card. The card loads with the high solver-iteration count
        the snug channel needs."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.case_usd, c.card_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — the pc-gpu assets ship with the repo under "
                    f"`suites/assembly/assets/`"
                )
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        table_z = c.surface_z - preset["top_offset"]
        ground_z = c.surface_z - preset["height"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd, scale=(preset["scale"],) * 3)
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        cx, cy = c.card_init_xy
        assets: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.9, 0.9, 0.9)),
            ),
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, table_z), rot=preset["orient"]),
                spawn=table_spawn,
            ),
            # The case: kinematic; the invisible fixture inside it (slot channel + rear-panel
            # cutout frame) is what the card mates with. Fixture contact offsets are set here
            # (not just authored in the asset) so the 0.15 mm/side channel grip never fights
            # speculative contacts.
            "case": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Case",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.case_usd,
                    activate_contact_sensors=True,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.case_contact_offset, rest_offset=0.0
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(wx, wy, c.surface_z + c.case_lift)),
            ),
            # Card contact offset must stay well below the channel grip (0.15 mm/side) or
            # speculative contacts choke the fit.
            "card": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Card",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.card_usd,
                    activate_contact_sensors=True,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.card_contact_offset, rest_offset=0.0
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=192,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.02,
                        linear_damping=2.0,
                        angular_damping=2.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.card_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + cx, wy + cy, c.surface_z + c.card_init_z), rot=c.card_init_quat
                ),
            ),
        }
        if c.card_stand:
            # Foam holder for a gripper env: three STATIC boxes (no rigid body). The floor pad's
            # top is the card spawn height (`card_init_z` = the tab-bottom plane), the two rails
            # flank the body slab's faces at `card_stand_gap` per side. Rail tops stay 35+ mm
            # below the pick grip band, so descending open fingers never meet them.
            y0, y1 = self.CARD_BODY_Y
            mid_y = wy + cy + 0.5 * (y0 + y1)  # body-slab mid-plane (the pinch/rail centre)
            half_gap = 0.5 * (y1 - y0) + c.card_stand_gap  # rail inner face off the mid-plane
            rail_h, rail_t = 0.055, 0.008
            foam = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.17, 0.17, 0.2), roughness=0.9)
            for name, size, pos in (
                ("card_stand_floor", (0.11, 0.09, c.card_init_z),
                 (wx + cx, mid_y, c.surface_z + 0.5 * c.card_init_z)),
                ("card_stand_rail_pcb", (0.11, rail_t, rail_h),
                 (wx + cx, mid_y - half_gap - 0.5 * rail_t, c.surface_z + c.card_init_z + 0.5 * rail_h)),
                ("card_stand_rail_fan", (0.11, rail_t, rail_h),
                 (wx + cx, mid_y + half_gap + 0.5 * rail_t, c.surface_z + c.card_init_z + 0.5 * rail_h)),
            ):
                assets[name] = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/" + "".join(p.capitalize() for p in name.split("_")),
                    spawn=sim_utils.CuboidCfg(
                        size=size,
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.001, rest_offset=0.0
                        ),
                        visual_material=foam,
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                )
        return assets

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab the case + card handles, cache env origins, and set the part frictions."""
        super().bind(env)
        self.case: RigidObject = env.iscene["case"]
        self.card: RigidObject = env.iscene["card"]
        self.env_origins = env.iscene.env_origins
        self._set_friction(self.case, self.cfg.case_friction)
        self._set_friction(self.card, self.cfg.card_friction)

    def _set_friction(self, asset, value: float) -> None:
        """Overwrite the static + dynamic friction on every shape of `asset` (across all envs)."""
        mats = asset.root_physx_view.get_material_properties()
        mats[..., 0:2] = value  # [static, dynamic, restitution]
        asset.root_physx_view.set_material_properties(mats, torch.arange(self.env.num_envs, device="cpu"))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start: the case pinned at spawn, the card lying backplate-down on
        the table beside it, with xy jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]  # (m, 3)
        wx, wy = c.workbench_pos
        cx, cy = c.card_init_xy

        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = origin + torch.tensor((wx + cx, wy + cy, c.surface_z + c.card_init_z), device=dev)
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 3:7] = torch.tensor(c.card_init_quat, device=dev)
        self.card.write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable scene state: world root states (13) of the case and the card."""
        return {
            "case": self.case.data.root_state_w[env_ids].clone(),
            "card": self.card.data.root_state_w[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned. The card's insertion depth is fully captured by its
        root state, so the channel holds it on restore."""
        self.case.write_root_pose_to_sim(state["case"][:, 0:7], env_ids)
        self.card.write_root_state_to_sim(state["card"], env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        if self.cfg.card_stand:
            card = (
                "Beside the case a loose graphics card stands upright in a foam holder, already "
                "in its installation orientation.\nGoal: grip the card by its top edge, lift it "
                "straight out of the holder, "
            )
        else:
            card = (
                "Beside the case lies a loose graphics card, backplate down.\nGoal: lift the "
                "card upright (fans toward the case front), "
            )
        return (
            "A gaming-PC case lying on its side on a sturdy table, opening up, its motherboard "
            "facing the ceiling. The board's primary PCIe x16 slot is empty, and the rear I/O "
            f"panel has an open expansion-slot cutout. {card}"
            "lower it into the case with its I/O bracket just forward of the rear panel, slide "
            "it rearward until the bracket and ports pass through the cutout, line its PCB edge "
            "connector up with the x16 slot, and press it straight down until it bottoms out. "
            "A seated card stays put on its own. The task is complete once the card is fully "
            "seated."
        )

    # ----- progress (public: seated()/engaged(); reads how far the assembly has got) -------------
    def engaged(self) -> torch.Tensor:
        """Tab depth below the slot mouth, shape (num_envs,), in metres (negative = still above
        the slot). The card bottoms out on the channel floor at 0.005 depth."""
        rel = self._card_offset_in_case()  # (n, 3), zero at the seated pose
        return (self.cfg.slot_mouth_z - self.cfg.seat_pos[2]) - rel[:, 2]

    def seated(self) -> torch.Tensor:
        """Whether the card is seated in the slot, shape (num_envs,): pressed down to `seat_depth`
        below the mouth, within `align_xy` of the seated point, and aligned in tilt AND heading."""
        c = self.cfg
        rel = self._card_offset_in_case()  # (n, 3)
        depth_ok = self.engaged() >= c.seat_depth
        xy_ok = rel[:, :2].norm(dim=-1) <= c.align_xy
        up_ok = self._axis_cos(2) >= math.cos(math.radians(c.align_axis_deg))
        yaw_ok = self._axis_cos(0) >= math.cos(math.radians(c.align_yaw_deg))
        return depth_ok & xy_ok & up_ok & yaw_ok

    def _card_offset_in_case(self) -> torch.Tensor:
        """The card origin's offset from the seated point, in the case's local frame, shape
        (num_envs, 3). Zero means the card origin sits exactly at the seated pose."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = quat_apply_inverse(
            self.case.data.root_quat_w, self.card.data.root_pos_w - self.case.data.root_pos_w
        )
        return rel - torch.tensor(self.cfg.seat_pos, device=rel.device)

    def _axis_cos(self, axis: int) -> torch.Tensor:
        """cos of the angle between the card's and the case's local `axis` (0=x: heading along the
        slot, 2=z: insertion axis), shape (num_envs,). Seated orientation = the case's own axes."""
        from isaaclab.utils.math import quat_apply

        e = torch.zeros(3, device=self.env.device)
        e[axis] = 1.0
        e = e.expand(self.env.num_envs, 3)
        case_ax = quat_apply(self.case.data.root_quat_w, e)
        card_ax = quat_apply(self.card.data.root_quat_w, e)
        return (card_ax * case_ax).sum(dim=-1)
