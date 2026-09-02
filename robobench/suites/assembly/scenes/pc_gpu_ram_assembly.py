"""PcGpuRamAssemblyScene — install the graphics card AND the dual-channel RAM pair in one build.

The same gaming-PC case as `pc_gpu`/`pc_ram` lies on its side on the table, opening up,
motherboard facing the ceiling — and BOTH work sites are open at once: the board's primary PCIe
x16 slot is empty (with the rear I/O panel's expansion-slot cutout), and all four memory slots
sit empty. Beside the case lie one loose RTX 2060 and two loose TridentZ sticks. Goal (carried
here, no task layer): install the card the way `pc_gpu` proved — place it inside the case
forward of the rear panel, slide it rearward so the bracket/ports pass through the cutout, press
it straight down to seat — and press each stick straight down into its DIMM slot the way
`pc_ram` proved, into the outermost and second-from-socket slots (the alternating pair a 2-stick
dual-channel kit populates).

The case is one kinematic body that never moves — the PC model stays visual-only; its physics is
the union of the two proven invisible fixtures inside the case body (this scene's case USD
composes both single-task overlays over the same base case): the PCIe channel + end stops +
rear-panel cutout frame from `pc_gpu`, and the two DIMM channels + end stops from `pc_ram`, each
gripping its part's PCB edge at 0.15 mm/side with a 1.2 mm/side funnel mouth, over a flush board
plate. Every part's origin is its PCB-edge bottom CENTRE with axes equal to the case's, so each
seated pose is just a translation in the case frame and insertion depth is a z difference.

The stick asset authors an inflated rotational inertia (see `pc_ram_assembly`); the card is the
`pc_gpu` asset unchanged.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import GraspWeldContract, SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class PcGpuRamAssemblySceneCfg(BaseCfg):
    """Config for `PcGpuRamAssemblyScene`. Nothing is locked — a variant is just a copy with a few
    fields changed. The gpu_* and ram_* gates
    and geometry carry the single-task scenes' proven values verbatim."""

    # --- the curriculum / difficulty dials -----------------------------------------------
    # Seating gates, per part family (see `pc_gpu_assembly`/`pc_ram_assembly` for their rationale;
    # the stick tilt gate is wider because a seated stick may legitimately rest leaned ~5 deg).
    gpu_seat_depth: float = 0.004  # min tab depth below the PCIe mouth (m) to count seated
    gpu_align_xy: float = 0.003  # max distance (m) of the card origin from its seated point
    gpu_align_axis_deg: float = 3.0  # max tilt of the card's up axis off the slot axis (deg)
    gpu_align_yaw_deg: float = 3.0  # max heading error of the card's length axis (deg)
    ram_seat_depth: float = 0.0037  # min blade depth below a DIMM mouth (m) to count seated
    ram_align_xy: float = 0.003  # max distance (m) of a stick origin from its seated point
    ram_align_axis_deg: float = 6.0  # max tilt of a stick's up axis off the slot axis (deg)
    ram_align_yaw_deg: float = 3.0  # max heading error of a stick's length axis (deg)
    reset_pos_jitter: float = 0.01  # uniform +/- xy jitter for every loose part at reset (m)
    # Part friction (static = dynamic), set on every shape at bind. Moving parts run moderately
    # slick against a grippier fixed case, so they slide down their channels but hold seat.
    card_friction: float = 0.3
    ram_friction: float = 0.3
    case_friction: float = 0.75
    # Weld-on-closure grasping (the benchmark's auto-weld contract, PhysX
    # form — the grasp-weld machinery at the end of this scene class):
    # close the fingers squarely across the card's body slab or flat across a stick's faces,
    # near the part's top edge, and it welds to the hand; open wide to release. Gripper envs
    # only (no-op under robot="null").
    grasp_weld: bool = True
    grasp_weld_dist: float = 0.010  # pinch-point-to-grip-band engage radius (m)

    # --- structure, reset layout, masses, asset paths (fixed) -------------------------------
    # Seated part origins (PCB-edge bottom centres) in the case's local frame; seated orientation =
    # the case's own axes (identity). Baked into the committed USDs (keep in sync if they change).
    # RAM slot 0 is the outermost (farthest from the CPU socket).
    gpu_seat_pos: tuple[float, float, float] = (-0.01595, 0.0293, 0.0035)
    gpu_slot_mouth_z: float = 0.0085  # PCIe slot top in the case frame (5 mm at full seat)
    ram_seat_pos: tuple[tuple[float, float, float], ...] = (
        (-0.1426893, -0.0678899, 0.0002058), (-0.1237320, -0.0678899, 0.0002058))

    ram_slot_mouth_z: float = 0.0046456  # DIMM channel wall top in the case frame
    board_top: float = 0.0  # board face height in the case frame (the asset's own origin)
    case_lift: float = 0.0289  # board face above the side panel the case lies on
    card_mass: float = 1.0  # dual-fan RTX 2060 (kg)
    ram_mass: float = 0.25  # keeps the press PD/solver in the proven stability class
    light_intensity: float = 2500.0
    # Loose part start poses (table-relative xy; see the single-task scenes for the lying
    # defaults' rationale — a gripper env instead stages every part upright in a foam holder).
    card_init_xy: tuple[float, float] = (0.28, 0.0)
    card_init_z: float = 0.0022
    card_init_quat: tuple[float, float, float, float] = (0.70711, 0.70711, 0.0, 0.0)  # flat
    ram_init_xy: tuple[tuple[float, float], ...] = ((0.27, -0.085), (0.27, 0.085))
    ram_init_z: float = 0.0042
    ram_init_quat: tuple[float, float, float, float] = (0.70711, 0.0, 0.70711, 0.0)  # flat
    card_contact_offset: float = 0.0001  # well below the 0.15 mm/side channel grips
    ram_contact_offset: float = 0.0001
    case_contact_offset: float = 0.0001
    # Optional foam holders that present the parts UPRIGHT for a parallel-jaw grasp (each part's
    # lying default is ungraspable: flat, its only sub-80 mm dimension points up). Enable together
    # with upright init quats (identity = seated orientation) and init z = the holders' floor top.
    card_stand: bool = False
    card_stand_gap: float = 0.0025  # rail clearance per side around the card's body slab (m)
    ram_stand: bool = False
    ram_stand_gap: float = 0.0012  # rail clearance per side around a stick's body slab (m)
    # Selectable work surface (same presets as the sibling scenes).
    case_xy: tuple[float, float] | None = None  # world xy the case sits at; None -> the
    # table anchor. Shifting the case (with the robot base following) stretches the staging
    # strip south of it without touching any case-relative work geometry.
    table: str = "lab_table"  # which work surface: "lab_table" | "packing"
    surface_z: float | None = None  # table-top height (m); None -> the preset's
    workbench_pos: tuple[float, float] | None = None  # xy the table sits at; None -> preset
    workbench_usd: str = ""  # empty -> the preset's vendored USD
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.55, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }
    # Asset USDs; empty -> the prebuilt assets committed under `assets/`.
    asset_dir: str = ""
    case_usd: str = ""
    card_usd: str = ""
    ram_usd: str = ""

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets)
        self.case_usd = self.case_usd or str(Path(self.asset_dir) / "pc" / "pc_case_gpu_ram_assembly_mb.usd")
        self.card_usd = self.card_usd or str(Path(self.asset_dir) / "pc" / "gpu_rtx2060.usd")
        self.ram_usd = self.ram_usd or str(Path(self.asset_dir) / "pc" / "ram_tridentz.usd")
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        if self.case_xy is None:
            self.case_xy = tuple(self.workbench_pos)
        self.workbench_usd = self.workbench_usd or str(assets / "props" / preset["usd"][0] / preset["usd"][1])

    @property
    def num_slots(self) -> int:
        return len(self.ram_seat_pos)


@SCENES.register("pc_gpu_ram")
class PcGpuRamAssemblyScene(BaseScene):
    cfg: PcGpuRamAssemblySceneCfg

    # Part-local extents of the body collision slabs (they match the visual shells): the card's
    # along its local y, a stick's along its local x. The holders' rails flank THESE faces — the
    # same pairs a parallel-jaw grasp pinches.
    CARD_BODY_Y: ClassVar[tuple[float, float]] = (-0.002, 0.0328)
    STICK_BODY_X: ClassVar[tuple[float, float]] = (-0.0037, 0.0036)

    def __init__(self, cfg: PcGpuRamAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or PcGpuRamAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, dome light, table, the PC case lying on it (kinematic, with BOTH invisible
        fixtures), one loose graphics card, and two loose RAM sticks. The moving parts load with
        the high solver-iteration count the snug channels need."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.case_usd, c.card_usd, c.ram_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — the pc assets ship with the repo under "
                    f"`suites/assembly/assets/`"
                )
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        table_z = c.surface_z - preset["top_offset"]
        ground_z = c.surface_z - preset["height"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd, scale=(preset["scale"],) * 3)
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        part_rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.02,
            linear_damping=2.0,
            angular_damping=2.0,
        )
        cx, cy = c.card_init_xy
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(__file__).resolve().parents[1] / "assets" / "props" / "ground" / "default_ground.usd")),
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
            # The case: kinematic; the invisible fixtures inside it (PCIe channel + rear cutout
            # frame, two DIMM channels, board plates) are what the parts mate with. Fixture
            # contact offsets are set here (not just authored in the asset) so the 0.15 mm/side
            # channel grips never fight speculative contacts.
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
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.case_xy[0], c.case_xy[1], c.surface_z + c.case_lift)
                ),
            ),
            # Part contact offsets must stay well below the channel grips (0.15 mm/side) or
            # speculative contacts choke the fits.
            "card": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Card",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.card_usd,
                    activate_contact_sensors=True,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.card_contact_offset, rest_offset=0.0
                    ),
                    rigid_props=part_rigid,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.card_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + cx, wy + cy, c.surface_z + c.card_init_z), rot=c.card_init_quat
                ),
            ),
        }
        for k, (ix, iy) in enumerate(c.ram_init_xy):
            out[f"ram_{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Ram_{k}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.ram_usd,
                    activate_contact_sensors=True,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.ram_contact_offset, rest_offset=0.0
                    ),
                    rigid_props=part_rigid,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ram_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + ix, wy + iy, c.surface_z + c.ram_init_z), rot=c.ram_init_quat
                ),
            )
        foam = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.17, 0.17, 0.2), roughness=0.9)

        def stand(name: str, size: tuple, pos: tuple) -> None:
            out[name] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + "".join(p_.capitalize() for p_ in name.split("_")),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
                    visual_material=foam,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            )

        if c.card_stand:
            # Foam holder presenting the card upright (floor pad + two rails flanking the 34.8 mm
            # body slab); geometry as in `pc_gpu_assembly`, with a slimmer floor pad. The holder
            # follows the staging yaw in `card_init_quat`: at identity the card stands in its
            # seated heading (length along x); at yaw 90 it stands lengthwise along y and the
            # rails flank the slab across x instead. Rail tops stay 35+ mm below the pick grip
            # band, clear of descending open fingers.
            y0, y1 = self.CARD_BODY_Y
            half_gap = 0.5 * (y1 - y0) + c.card_stand_gap
            rail_h, rail_t = 0.055, 0.008
            if abs(c.card_init_quat[3]) > 0.5:  # staged yawed 90 deg about z (local +y -> -x)
                mid_x = wx + cx - 0.5 * (y0 + y1)
                stand("card_stand_floor", (0.062, 0.11, c.card_init_z),
                      (mid_x, wy + cy, c.surface_z + 0.5 * c.card_init_z))
                stand("card_stand_rail_pcb", (rail_t, 0.11, rail_h),
                      (mid_x - half_gap - 0.5 * rail_t, wy + cy, c.surface_z + c.card_init_z + 0.5 * rail_h))
                stand("card_stand_rail_fan", (rail_t, 0.11, rail_h),
                      (mid_x + half_gap + 0.5 * rail_t, wy + cy, c.surface_z + c.card_init_z + 0.5 * rail_h))
            else:
                mid_y = wy + cy + 0.5 * (y0 + y1)
                stand("card_stand_floor", (0.11, 0.062, c.card_init_z),
                      (wx + cx, mid_y, c.surface_z + 0.5 * c.card_init_z))
                stand("card_stand_rail_pcb", (0.11, rail_t, rail_h),
                      (wx + cx, mid_y - half_gap - 0.5 * rail_t, c.surface_z + c.card_init_z + 0.5 * rail_h))
                stand("card_stand_rail_fan", (0.11, rail_t, rail_h),
                      (wx + cx, mid_y + half_gap + 0.5 * rail_t, c.surface_z + c.card_init_z + 0.5 * rail_h))
        if c.ram_stand:
            # Foam holders presenting the sticks upright (per stick: floor pad + two rails
            # flanking the 7.3 mm body slab); geometry as in `pc_ram_assembly`. The holders
            # follow the staging yaw in `ram_init_quat`: at identity a stick stands in its
            # seated heading (length along y); at yaw 90 it stands PARALLEL to the card
            # (length along x) and the rails flank the slab across y instead.
            x0, x1 = self.STICK_BODY_X
            half_gap = 0.5 * (x1 - x0) + c.ram_stand_gap
            rail_h, rail_t, stand_l = 0.018, 0.008, 0.130
            rot90 = abs(c.ram_init_quat[3]) > 0.5  # staged yawed 90 deg about z
            for k, (ix, iy) in enumerate(c.ram_init_xy):
                if rot90:
                    mid_y = wy + iy + 0.5 * (x0 + x1)
                    stand(f"ram_stand_{k}_floor", (stand_l, 0.022, c.ram_init_z),
                          (wx + ix, mid_y, c.surface_z + 0.5 * c.ram_init_z))
                    stand(f"ram_stand_{k}_rail_a", (stand_l, rail_t, rail_h),
                          (wx + ix, mid_y - half_gap - 0.5 * rail_t, c.surface_z + c.ram_init_z + 0.5 * rail_h))
                    stand(f"ram_stand_{k}_rail_b", (stand_l, rail_t, rail_h),
                          (wx + ix, mid_y + half_gap + 0.5 * rail_t, c.surface_z + c.ram_init_z + 0.5 * rail_h))
                else:
                    mid_x = wx + ix + 0.5 * (x0 + x1)
                    stand(f"ram_stand_{k}_floor", (0.022, stand_l, c.ram_init_z),
                          (mid_x, wy + iy, c.surface_z + 0.5 * c.ram_init_z))
                    stand(f"ram_stand_{k}_rail_a", (rail_t, stand_l, rail_h),
                          (mid_x - half_gap - 0.5 * rail_t, wy + iy, c.surface_z + c.ram_init_z + 0.5 * rail_h))
                    stand(f"ram_stand_{k}_rail_b", (rail_t, stand_l, rail_h),
                          (mid_x + half_gap + 0.5 * rail_t, wy + iy, c.surface_z + c.ram_init_z + 0.5 * rail_h))
        return out

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

    def __getattr__(self, name: str):
        # Legacy surface: the grasp contract's state used to live directly on the scene
        # (inline machinery era) and existing solutions read it there — forward to the
        # composed contract. `__getattr__` only fires for attributes not found normally.
        if name.startswith("_gw_") and "grasp_weld" in self.__dict__:
            return getattr(self.grasp_weld, name)
        raise AttributeError(name)

    def bind(self, env: BaseEnv) -> None:
        """Grab the case + part handles, cache env origins, and set the part frictions."""
        super().bind(env)
        self.case: RigidObject = env.iscene["case"]
        self.card: RigidObject = env.iscene["card"]
        self.rams: list[RigidObject] = [env.iscene[f"ram_{k}"] for k in range(self.cfg.num_slots)]
        self.env_origins = env.iscene.env_origins
        self._set_friction(self.case, self.cfg.case_friction)
        self._set_friction(self.card, self.cfg.card_friction)
        for ram in self.rams:
            self._set_friction(ram, self.cfg.ram_friction)
        self.grasp_weld = GraspWeldContract(self)  # composed, publishes self.grasp_held
        self.grasp_weld.bind()

    def grasp_sites(self) -> list:
        """One grip band per part — the card across its body slab (CARD_BODY_Y, 34.8 mm) high
        on its length, each stick across its blade (STICK_BODY_X, 7.3 mm) at the top edge."""
        cy = 0.5 * (self.CARD_BODY_Y[0] + self.CARD_BODY_Y[1])
        rx = 0.5 * (self.STICK_BODY_X[0] + self.STICK_BODY_X[1])
        sites = [("card", self.card, (-0.045, cy, 0.1005), (0.045, cy, 0.1005), (0.030, 0.039))]
        sites += [
            (f"ram{k}", ram, (rx, -0.055, 0.0401), (rx, 0.055, 0.0401), (0.005, 0.010))
            for k, ram in enumerate(self.rams)
        ]
        return sites

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Reconcile the weld-on-closure grasp contract every physics substep."""
        self.grasp_weld.step()

    def _set_friction(self, asset, value: float) -> None:
        """Overwrite the static + dynamic friction on every shape of `asset` (across all envs)."""
        mats = asset.root_physx_view.get_material_properties()
        mats[..., 0:2] = value  # [static, dynamic, restitution]
        asset.root_physx_view.set_material_properties(mats, torch.arange(self.env.num_envs, device="cpu"))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start: the case pinned at spawn, the card and both sticks loose on
        the table beside it, with xy jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]  # (m, 3)
        wx, wy = c.workbench_pos

        parts = [(self.card, c.card_init_xy, c.card_init_z, c.card_init_quat)]
        parts += [(ram, xy, c.ram_init_z, c.ram_init_quat) for ram, xy in zip(self.rams, c.ram_init_xy)]
        for part, (ix, iy), iz, quat in parts:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.tensor((wx + ix, wy + iy, c.surface_z + iz), device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 3:7] = torch.tensor(quat, device=dev)
            part.write_root_state_to_sim(st, env_ids)
        self.grasp_weld.release_all(env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable scene state: world root states (13) of the case, the card and both sticks."""
        out = {
            "case": self.case.data.root_state_w[env_ids].clone(),
            "card": self.card.data.root_state_w[env_ids].clone(),
        }
        for k, ram in enumerate(self.rams):
            out[f"ram_{k}"] = ram.data.root_state_w[env_ids].clone()
        out.update(self.grasp_weld.state(env_ids))
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned. A part's insertion depth is fully captured by its
        root state, so its channel holds it on restore."""
        self.case.write_root_pose_to_sim(state["case"][:, 0:7], env_ids)
        self.card.write_root_state_to_sim(state["card"], env_ids)
        for k, ram in enumerate(self.rams):
            ram.write_root_state_to_sim(state[f"ram_{k}"], env_ids)
        self.grasp_weld.restore(state, env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        if self.cfg.card_stand and self.cfg.ram_stand:
            parts = (
                "Beside the case a loose graphics card and two loose RAM sticks stand upright in "
                "foam holders, already in their installation orientations.\nGoal: install the "
                "card first — grip it by its top edge, lift it out of its holder, "
            )
        else:
            parts = (
                "Beside the case lie a loose graphics card (backplate down) and two loose RAM "
                "sticks (flat on the table).\nGoal: install the card first — lift it upright, "
            )
        return (
            "A gaming-PC case lying on its side on a sturdy table, opening up, its motherboard "
            "facing the ceiling. The board's primary PCIe x16 slot is empty (the rear I/O panel "
            f"has an open expansion-slot cutout), and all four memory slots are empty. {parts}"
            "lower it into the case with its I/O bracket just forward of the rear panel, slide "
            "it rearward until the bracket and ports pass through the cutout, line its PCB edge "
            "connector up with the x16 slot, and press it straight down until it bottoms out — "
            "then install each RAM stick: carry it over the case wall, line its gold edge "
            "connector up with its slot (the outermost and the second-from-socket, the "
            "alternating pair a dual-channel kit fills), and press it straight down until it "
            "clicks fully home. Every seated part stays put on its own. The task is complete "
            "once the card and both sticks are fully seated."
            + (
                " A part holds in a firm pinch: close the fingers squarely across the card's "
                "body slab or a stick's faces, near the top edge, and the grip locks; open "
                "wide to release."
                if self.cfg.grasp_weld
                else ""
            )
        )

    # ----- progress (public: seated()/engaged(); reads how far the assembly has got) -------------
    def engaged(self) -> torch.Tensor:
        """Depth of every part below its slot mouth, shape (num_envs, 1 + num_slots): column 0 is
        the card's tab depth (5 mm stroke), columns 1..S each stick's blade depth (4.44 mm
        stroke). Negative = still above the slot."""
        return torch.cat([self.gpu_engaged().unsqueeze(1), self.ram_engaged()], dim=1)

    def seated(self) -> torch.Tensor:
        """Whether every part is seated, shape (num_envs, 1 + num_slots): column 0 the card,
        columns 1..S the sticks (depth + xy + tilt + heading gates, per family)."""
        return torch.cat([self.gpu_seated().unsqueeze(1), self.ram_seated()], dim=1)

    def success(self) -> torch.Tensor:
        """(N,) bool: the card and every stick seated — this scene's assembled state
        (scene-level success alias, matching the other suites' surface)."""
        return self.seated().all(dim=1)

    def gpu_engaged(self) -> torch.Tensor:
        """Card tab depth below the PCIe slot mouth, shape (num_envs,), in metres."""
        rel = self._card_offset_in_case()
        return (self.cfg.gpu_slot_mouth_z - self.cfg.gpu_seat_pos[2]) - rel[:, 2]

    def gpu_seated(self) -> torch.Tensor:
        """Whether the card is seated in the PCIe slot, shape (num_envs,)."""
        c = self.cfg
        rel = self._card_offset_in_case()
        depth_ok = self.gpu_engaged() >= c.gpu_seat_depth
        xy_ok = rel[:, :2].norm(dim=-1) <= c.gpu_align_xy
        up_ok = self._part_axis_cos(self.card, 2) >= math.cos(math.radians(c.gpu_align_axis_deg))
        yaw_ok = self._part_axis_cos(self.card, 0) >= math.cos(math.radians(c.gpu_align_yaw_deg))
        return depth_ok & xy_ok & up_ok & yaw_ok

    def ram_engaged(self) -> torch.Tensor:
        """Blade depth below each DIMM slot mouth, shape (num_envs, num_slots), in metres."""
        c = self.cfg
        rel = self._ram_offsets_in_case()
        mouth = torch.tensor([c.ram_slot_mouth_z - p[2] for p in c.ram_seat_pos], device=rel.device)
        return mouth.unsqueeze(0) - rel[..., 2]

    def ram_seated(self) -> torch.Tensor:
        """Whether each stick is seated in its DIMM slot, shape (num_envs, num_slots)."""
        c = self.cfg
        rel = self._ram_offsets_in_case()
        depth_ok = self.ram_engaged() >= c.ram_seat_depth
        xy_ok = rel[..., 0:2].norm(dim=-1) <= c.ram_align_xy
        up_ok = torch.stack(
            [self._part_axis_cos(ram, 2) for ram in self.rams], dim=1
        ) >= math.cos(math.radians(c.ram_align_axis_deg))
        yaw_ok = torch.stack(
            [self._part_axis_cos(ram, 1) for ram in self.rams], dim=1
        ) >= math.cos(math.radians(c.ram_align_yaw_deg))
        return depth_ok & xy_ok & up_ok & yaw_ok

    def _card_offset_in_case(self) -> torch.Tensor:
        """The card origin's offset from its seated point, in the case's local frame, (n, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = quat_apply_inverse(
            self.case.data.root_quat_w, self.card.data.root_pos_w - self.case.data.root_pos_w
        )
        return rel - torch.tensor(self.cfg.gpu_seat_pos, device=rel.device)

    def _ram_offsets_in_case(self) -> torch.Tensor:
        """Each stick origin's offset from its seated point, in the case's local frame,
        (n, num_slots, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        out = []
        for k, ram in enumerate(self.rams):
            rel = quat_apply_inverse(
                self.case.data.root_quat_w, ram.data.root_pos_w - self.case.data.root_pos_w
            )
            out.append(rel - torch.tensor(self.cfg.ram_seat_pos[k], device=rel.device))
        return torch.stack(out, dim=1)

    def _part_axis_cos(self, part, axis: int) -> torch.Tensor:
        """cos of the angle between `part`'s and the case's local `axis`, shape (num_envs,).
        Seated orientation = the case's own axes."""
        from isaaclab.utils.math import quat_apply

        e = torch.zeros(3, device=self.env.device)
        e[axis] = 1.0
        e = e.expand(self.env.num_envs, 3)
        case_ax = quat_apply(self.case.data.root_quat_w, e)
        part_ax = quat_apply(part.data.root_quat_w, e)
        return (part_ax * case_ax).sum(dim=-1)

    # Grasp-weld contract: composed `GraspWeldContract` (robobench.core.grasp_weld),
    # created in `bind()`; this scene supplies the part-side `grasp_sites()`.
    # Grasp-weld contract constants — the scene's public knobs (solutions read these off the
    # scene, e.g. `scene.GRASP_PINCH_OFFSET`); the composed GraspWeldContract consumes them.
    GRASP_HAND_BODY: ClassVar[str] = "panda_hand"
    GRASP_FINGER_JOINTS: ClassVar[str] = "panda_finger_joint.*"
    GRASP_PINCH_OFFSET: ClassVar[float] = 0.1034  # hand origin -> finger-pad centre, along approach
    GRASP_POOL: ClassVar[int] = 8  # engages per (env, site) per run; exhausted -> warn, no weld
    GRASP_STALL: ClassVar[float] = 0.01  # max |finger vel| sum (m/s): fingers stopped ON the part
    GRASP_DEBOUNCE: ClassVar[int] = 8  # consecutive qualifying substeps before the weld engages
    GRASP_RELEASE_MARGIN: ClassVar[float] = 0.008  # release at window-top + this (m), hysteresis

