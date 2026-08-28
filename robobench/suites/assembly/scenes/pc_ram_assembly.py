"""PcRamAssemblyScene — press a dual-channel pair of RAM sticks into the PC's DIMM slots.

The same gaming-PC case as `pc_motherboard`/`pc_gpu` lies on its side on the table, opening up,
motherboard facing the ceiling — here the board's DIMM cluster is the work site. All four memory
slots sit empty (the model's factory sticks are removed from the case), and two loose TridentZ
sticks lie flat on the table beside the case. Goal (carried here, no task layer): stand each
stick upright over its target slot — the outermost and the second-from-socket, the alternating
pair a 2-stick dual-channel kit populates — line its PCB edge up with the slot, and press it
straight down until it seats.

The case is one kinematic body that never moves — the PC model stays visual-only; its physics is
an invisible fixture inside the case body: per empty slot a channel whose walls grip the stick's
1.6 mm PCB blade at 0.15 mm/side (flaring to a 1.2 mm/side funnel mouth — idealizing the real
slot's spring contacts), a floor whose top is the model's own seated blade height, end stops
(0.5 mm play, hidden inside the slot's latch blocks), and a flush board plate so a dropped stick
rests on the board face. Unlike `pc_gpu` there is no rear-panel constraint: RAM is a pure
vertical press in open air. Each stick's origin is its PCB-blade bottom CENTRE with axes equal
to the case's, so a seated pose is just `seat_pos[k]` + identity orientation in the case frame,
and insertion depth is a z difference.

The stick asset authors an inflated rotational inertia (1e-3 kg m^2 diagonal vs the real
~2.8e-5): the smoke's force-only PD "hand" reuses the pc_gpu rotation gains, which are unstable
at dt=1/120 on the true inertia; the inflation is invisible on camera and irrelevant to the
press regime.

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
class PcRamAssemblySceneCfg(BaseCfg):
    """Config for `PcRamAssemblyScene`. Nothing is locked — a variant is just a copy with a few
    fields changed."""

    # --- the curriculum / difficulty dials -----------------------------------------------
    # A stick is "seated" when — in the case's frame — its blade is >= `seat_depth` below the slot
    # mouth, its origin is within `align_xy` of its seated point, and its axes are within
    # `align_axis_deg` (tilt) / `align_yaw_deg` (heading along the slot) of the case's. The full
    # stroke from the slot mouth to the channel floor is 4.44 mm. The tilt gate is wider than
    # pc_gpu's: with the shallow 3.4 mm grip band a fully seated stick may legitimately rest
    # leaned against a channel wall at up to ~5 deg (its upright-restoring range is only ~2 deg),
    # so 3 deg would false-fail a good insertion.
    seat_depth: float = 0.0037  # min blade depth below the slot mouth (m) to count seated
    align_xy: float = 0.003  # max distance (m) of a stick origin from its seated point
    align_axis_deg: float = 6.0  # max tilt of a stick's up axis off the slot axis (deg)
    align_yaw_deg: float = 3.0  # max heading error of a stick's length axis (deg)
    reset_pos_jitter: float = 0.01  # uniform +/- xy jitter for the loose sticks at reset (m)
    # Uniform +/- XY jitter of the WHOLE CASE at reset (m; 0 = pinned at spawn, the historical
    # behavior). Translation only — solves and the grader read seat positions as case_pos +
    # seat_pos[k] with the case's IDENTITY orientation, so yaw must stay 0. Nonzero values give
    # image->press-location covariance (the grounding VLA distillation needs); the demos' solve
    # tracks the shifted case automatically (its waypoints derive from the live case pose).
    case_jitter_xy: float = 0.0
    # Optional asymmetric ranges ((x_lo, x_hi), (y_lo, y_hi)) in m overriding the symmetric
    # case_jitter_xy — the pc_ram solve's stick-0 pick is brittle when the case moves toward -x
    # (Stage-1 "A"; re-measured in the Stage-4 pilot: dx <= -9 mm fails, +x/±y passes), so
    # grounding data draws from the passing half-plane.
    case_jitter_range: tuple[tuple[float, float], tuple[float, float]] | None = None
    # Part friction (static = dynamic), set on every shape at bind. The moving stick runs
    # moderately slick against a grippier fixed case, so it slides down the channel but holds seat.
    ram_friction: float = 0.3
    case_friction: float = 0.75
    # Weld-on-closure grasping (the benchmark's auto-weld contract, PhysX
    # form — the grasp-weld machinery at the end of this scene class):
    # close the fingers flat across a stick's faces near its top edge and the stick welds to
    # the hand; open wide to release. Gripper envs only (no-op under robot="null").
    grasp_weld: bool = True
    grasp_weld_dist: float = 0.010  # pinch-point-to-grip-band engage radius (m)

    # --- structure, reset layout, masses, asset paths (fixed) -------------------------------
    # Seated stick origins (PCB-blade bottom centres) in the case's local frame, one per empty DIMM
    # slot; seated orientation = the case's own axes (identity). Baked into the committed USDs
    # (keep in sync if they change). Slot 0 is the outermost (farthest from the CPU socket).
    seat_pos: tuple[tuple[float, float, float], ...] = (
        (-0.1426893, -0.0678899, 0.0002058), (-0.1237320, -0.0678899, 0.0002058))

    slot_mouth_z: float = 0.0046456  # channel wall top in the case frame: depth datum
    board_top: float = 0.0  # board face height in the case frame (the asset's own origin)
    case_lift: float = 0.0289  # board face above the side panel the case lies on
    ram_mass: float = 0.25  # a real stick is ~45 g; 0.25 kg keeps the PD/solver in the
    # proven stability class (the asset also authors an inflated rotational inertia — see module
    # docstring)
    light_intensity: float = 2500.0
    # Loose stick start poses: lying flat (heat-spreader face down, RGB bar pointing away from the
    # case) on the table beside the case, end-to-end along y with a 34 mm tip gap.
    ram_init_xy: tuple[tuple[float, float], ...] = ((0.27, -0.085), (0.27, 0.085))
    ram_init_z: float = 0.0042  # origin height lying face-down (slab half 3.6 mm + pad)
    ram_init_quat: tuple[float, float, float, float] = (0.70711, 0.0, 0.70711, 0.0)  # flat
    ram_contact_offset: float = 0.0001  # well below the 0.15 mm/side channel grip
    case_contact_offset: float = 0.0001  # ditto for the slot fixtures' walls
    # Optional foam holders (per stick: a floor pad + two rails flanking the 7.3 mm body slab)
    # that present the sticks UPRIGHT for a parallel-jaw grasp. The lying default is ungraspable
    # by a Franka gripper: flat on its face a stick's only sub-80 mm dimension (its thickness)
    # points UP, so no top-down or side pinch can straddle it. Enable together with upright
    # `ram_init_quat` (identity = the seated orientation) and `ram_init_z` = the holders' floor
    # top; the rails cap a free stick's lean at ~5 deg and the pick pulls straight up out of them.
    ram_stand: bool = False
    ram_stand_gap: float = 0.0012  # rail clearance per side around the body slab (m)
    # Selectable work surface (same presets as the sibling scenes).
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
    ram_usd: str = ""

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets)
        self.case_usd = self.case_usd or str(Path(self.asset_dir) / "pc" / "pc_case_ram_assembly_mb.usd")
        self.ram_usd = self.ram_usd or str(Path(self.asset_dir) / "pc" / "ram_tridentz.usd")
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(assets / "props" / preset["usd"][0] / preset["usd"][1])

    @property
    def num_slots(self) -> int:
        return len(self.seat_pos)


@SCENES.register("pc_ram")
class PcRamAssemblyScene(BaseScene):
    cfg: PcRamAssemblySceneCfg

    # Stick-local x extent of the body collision slab (from ram_tridentz.usd
    # `/ram/collision/body`; it matches the visual shell). The holders' rails flank THESE
    # faces — the same pair a parallel-jaw grasp pinches.
    STICK_BODY_X: ClassVar[tuple[float, float]] = (-0.0037, 0.0036)

    #: L5 external view (see BaseScene.CAMERAS): over the case's south-west corner, high enough
    #: to see over the 195 mm walls into the DIMM cluster while the stick holders sit in the
    #: foreground — probed on the nominal_0 render (2026-08-24); the wrist view carries the
    #: fine insertion detail. Bands wiggle the eye a couple of cm per episode.
    CAMERAS: ClassVar[dict[str, dict]] = {
        # front: centred on the slot/stick midpoint from straight ahead, high enough (eye z 1.0)
        # that the ready-pose upper arm (horizontal at z 0.66) sits inside the frame instead of
        # crossing its top edge, and steep enough (~58 deg) to see INTO the case: the DIMM area
        # of the motherboard and both sticks in their holders are in view at once (candidate D
        # of the 2026-08-26 camera probes; the old (0.12,-0.62,0.62) view saw the slots at a
        # grazing angle over the near wall).
        "front": {"eye": (0.40, -0.78, 1.00), "target": (0.40, -0.18, 0.03), "focal": 16.0,
                  "bands": {
                      "eye_x": {"dist": "uniform", "lo": 0.38, "hi": 0.42},
                      "eye_y": {"dist": "uniform", "lo": -0.80, "hi": -0.76},
                      "eye_z": {"dist": "uniform", "lo": 0.98, "hi": 1.02},
                  }},
    }

    #: L4 per-env physics bands (data_engine sampler grammar; nominal = the cfg default, slot 0
    #: of every batch keeps it). The friction pair brackets the designed slick-stick / grippy-case
    #: ratio the second stick's gravity-seat rides on (stock 0.3 / 0.75 — the pc_ram IK campaign's
    #: robustness probe); mass ±20% around the 0.25 kg the PD/solver stability class was tuned at.
    PHYSICAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "ram_friction": {"dist": "uniform", "lo": 0.25, "hi": 0.40,
                         "reason": "around the 0.3 nominal; slick stick slides the channel"},
        "case_friction": {"dist": "uniform", "lo": 0.60, "hi": 0.90,
                          "reason": "around the 0.75 nominal; grippy case holds the seat"},
        "ram_mass": {"dist": "uniform", "lo": 0.20, "hi": 0.30,
                     "reason": "±20% of the 0.25 kg stability-class mass"},
    }
    #: L5 visual bands (replay/render, stage-wide per pass): the dome light is build-consumed.
    VISUAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "light_intensity": {"dist": "uniform", "lo": 2000.0, "hi": 3000.0,
                            "reason": "around the 2500 nominal"},
    }

    def __init__(self, cfg: PcRamAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or PcRamAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, dome light, table, the PC case lying on it (kinematic, with the invisible DIMM
        fixture), and two loose RAM sticks. The sticks load with the high solver-iteration count
        the snug channels need."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.case_usd, c.ram_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — the pc-ram assets ship with the repo under "
                    f"`suites/assembly/assets/`"
                )
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        table_z = c.surface_z - preset["top_offset"]
        ground_z = c.surface_z - preset["height"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd, scale=(preset["scale"],) * 3)
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

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
            # The case: kinematic; the invisible fixture inside it (two DIMM channels + board
            # plate) is what the sticks mate with. Fixture contact offsets are set here (not just
            # authored in the asset) so the 0.15 mm/side channel grip never fights speculative
            # contacts.
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
        }
        # Stick contact offset must stay well below the channel grip (0.15 mm/side) or
        # speculative contacts choke the fit.
        for k, (ix, iy) in enumerate(c.ram_init_xy):
            out[f"ram_{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Ram_{k}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.ram_usd,
                    activate_contact_sensors=True,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.ram_contact_offset, rest_offset=0.0
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=192,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.02,
                        linear_damping=2.0,
                        angular_damping=2.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ram_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + ix, wy + iy, c.surface_z + c.ram_init_z), rot=c.ram_init_quat
                ),
            )
        if c.ram_stand:
            # Foam holders for a gripper env: per stick, three STATIC boxes (no rigid body). The
            # floor pad's top is the stick spawn height (`ram_init_z` = the blade-bottom plane),
            # the two rails flank the body slab's faces at `ram_stand_gap` per side. Rail tops
            # stay several mm below the pick grip band, clear of descending fingertips.
            x0, x1 = self.STICK_BODY_X
            half_gap = 0.5 * (x1 - x0) + c.ram_stand_gap  # rail inner face off the slab mid-plane
            rail_h, rail_t, stand_l = 0.018, 0.008, 0.145
            foam = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.17, 0.17, 0.2), roughness=0.9)
            for k, (ix, iy) in enumerate(c.ram_init_xy):
                mid_x = wx + ix + 0.5 * (x0 + x1)  # body-slab mid-plane (the pinch/rail centre)
                for name, size, pos in (
                    (f"ram_stand_{k}_floor", (0.022, stand_l, c.ram_init_z),
                     (mid_x, wy + iy, c.surface_z + 0.5 * c.ram_init_z)),
                    (f"ram_stand_{k}_rail_a", (rail_t, stand_l, rail_h),
                     (mid_x - half_gap - 0.5 * rail_t, wy + iy, c.surface_z + c.ram_init_z + 0.5 * rail_h)),
                    (f"ram_stand_{k}_rail_b", (rail_t, stand_l, rail_h),
                     (mid_x + half_gap + 0.5 * rail_t, wy + iy, c.surface_z + c.ram_init_z + 0.5 * rail_h)),
                ):
                    out[name] = AssetBaseCfg(
                        prim_path="{ENV_REGEX_NS}/" + "".join(p_.capitalize() for p_ in name.split("_")),
                        spawn=sim_utils.CuboidCfg(
                            size=size,
                            collision_props=sim_utils.CollisionPropertiesCfg(
                                contact_offset=0.001, rest_offset=0.0
                            ),
                            visual_material=foam,
                        ),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
                    )
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
        """Grab the case + stick handles, cache env origins, and set the part frictions."""
        super().bind(env)
        self.case: RigidObject = env.iscene["case"]
        self.rams: list[RigidObject] = [env.iscene[f"ram_{k}"] for k in range(self.cfg.num_slots)]
        self.env_origins = env.iscene.env_origins
        self._set_friction(self.case, self.cfg.case_friction)
        for ram in self.rams:
            self._set_friction(ram, self.cfg.ram_friction)
        self.grasp_weld = GraspWeldContract(self)  # composed, publishes self.grasp_held
        self.grasp_weld.bind()

    def grasp_sites(self) -> list:
        """One grip band per stick: across the blade (faces at STICK_BODY_X, 7.3 mm wide),
        along the stick's length at its top edge (z 0.0401)."""
        x = 0.5 * (self.STICK_BODY_X[0] + self.STICK_BODY_X[1])
        return [
            (f"ram{k}", ram, (x, -0.055, 0.0401), (x, 0.055, 0.0401), (0.005, 0.010))
            for k, ram in enumerate(self.rams)
        ]

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Reconcile the weld-on-closure grasp contract every physics substep."""
        self.grasp_weld.step()

    def _set_friction(self, asset, value: float) -> None:
        """Overwrite the static + dynamic friction on every shape of `asset` (across all envs)."""
        mats = asset.root_physx_view.get_material_properties()
        mats[..., 0:2] = value  # [static, dynamic, restitution]
        asset.root_physx_view.set_material_properties(mats, torch.arange(self.env.num_envs, device="cpu"))

    def apply_physical_params(self, env: BaseEnv, values: dict[str, list]) -> None:
        """Write a PHYSICAL_PARAMS draw PER ENV through the PhysX views: `values[name]` is one
        value per env slot. Frictions go onto every shape of the part (static = dynamic, as at
        bind); the stick mass onto each stick's body."""
        ids = torch.arange(env.num_envs, device="cpu")
        for name, per_env in values.items():
            col = torch.tensor([float(v) for v in per_env], dtype=torch.float32)  # (n,)
            if name in ("ram_friction", "case_friction"):
                for asset in (self.rams if name == "ram_friction" else [self.case]):
                    mats = asset.root_physx_view.get_material_properties()  # (n, shapes, 3), cpu
                    mats[..., 0:2] = col.view(-1, 1, 1).expand(mats.shape[0], mats.shape[1], 2)
                    asset.root_physx_view.set_material_properties(mats, ids)
            elif name == "ram_mass":
                for ram in self.rams:
                    masses = ram.root_physx_view.get_masses()  # (n, bodies), cpu
                    masses[:] = col.view(-1, 1).expand_as(masses)
                    ram.root_physx_view.set_masses(masses, ids)
            else:
                raise KeyError(f"{type(self).__name__}.apply_physical_params: unknown knob {name!r}")

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start: the case pinned at spawn, both sticks lying flat on the table
        beside it, with xy jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]  # (m, 3)
        wx, wy = c.workbench_pos

        if c.case_jitter_xy > 0.0 or c.case_jitter_range is not None:
            # kinematic case: re-pin at spawn + a per-env XY draw (identity orientation — see cfg)
            pose = torch.zeros(m, 7, device=dev)
            pose[:, 0:3] = origin + torch.tensor(
                (wx, wy, c.surface_z + c.case_lift), device=dev)
            if c.case_jitter_range is not None:
                (xlo, xhi), (ylo, yhi) = c.case_jitter_range
                lo = torch.tensor((xlo, ylo), device=dev)
                hi = torch.tensor((xhi, yhi), device=dev)
                pose[:, 0:2] += lo + torch.rand(m, 2, device=dev) * (hi - lo)
            else:
                pose[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.case_jitter_xy
            pose[:, 3] = 1.0
            self.case.write_root_pose_to_sim(pose, env_ids)

        for ram, (ix, iy) in zip(self.rams, c.ram_init_xy):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.tensor((wx + ix, wy + iy, c.surface_z + c.ram_init_z), device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 3:7] = torch.tensor(c.ram_init_quat, device=dev)
            ram.write_root_state_to_sim(st, env_ids)
        self.grasp_weld.release_all(env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable scene state: world root states (13) of the case and both sticks."""
        out = {"case": self.case.data.root_state_w[env_ids].clone()}
        for k, ram in enumerate(self.rams):
            out[f"ram_{k}"] = ram.data.root_state_w[env_ids].clone()
        out.update(self.grasp_weld.state(env_ids))
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned. A stick's insertion depth is fully captured by its
        root state, so the channel holds it on restore."""
        self.case.write_root_pose_to_sim(state["case"][:, 0:7], env_ids)
        for k, ram in enumerate(self.rams):
            ram.write_root_state_to_sim(state[f"ram_{k}"], env_ids)
        self.grasp_weld.restore(state, env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        if self.cfg.ram_stand:
            sticks = (
                "Beside the case two loose RAM sticks stand upright in foam holders, already in "
                "their installation orientation.\nGoal: grip each stick by its top edge, lift it "
                "straight out of its holder, "
            )
        else:
            sticks = (
                "Beside the case lie two loose RAM sticks, flat on the table.\nGoal: stand each "
                "stick upright, "
            )
        return (
            "A gaming-PC case lying on its side on a sturdy table, opening up, its motherboard "
            f"facing the ceiling. All four memory slots are empty. {sticks}"
            "carry it over the case wall, line its gold edge connector up with its slot — the "
            "outermost and the second-from-socket, the alternating pair a dual-channel kit fills "
            "(the notch only fits one way — heat-spreader faces along the slot) — and press it "
            "straight down until it clicks fully home. A seated stick stays put on its own. The "
            "task is complete once both sticks are fully seated."
            + (
                " A stick holds in a firm pinch: close the fingers flat across its faces near "
                "the top edge and the grip locks; open wide to release."
                if self.cfg.grasp_weld
                else ""
            )
        )

    # ----- progress (public: seated()/engaged(); reads how far the assembly has got) -------------
    def engaged(self) -> torch.Tensor:
        """Blade depth below each slot mouth, shape (num_envs, num_slots), in metres (negative =
        still above the slot). A stick bottoms out on its channel floor at 0.00444 depth."""
        c = self.cfg
        rel = self._ram_offsets_in_case()  # (n, S, 3), zero at the seated poses
        mouth = torch.tensor([c.slot_mouth_z - p[2] for p in c.seat_pos], device=rel.device)
        return mouth.unsqueeze(0) - rel[..., 2]

    def seated(self) -> torch.Tensor:
        """Whether each stick is seated in its slot, shape (num_envs, num_slots): pressed down to
        `seat_depth` below the mouth, within `align_xy` of its seated point, and aligned in tilt
        AND heading."""
        c = self.cfg
        rel = self._ram_offsets_in_case()  # (n, S, 3)
        depth_ok = self.engaged() >= c.seat_depth
        xy_ok = rel[..., 0:2].norm(dim=-1) <= c.align_xy
        up_ok = self._axis_cos(2) >= math.cos(math.radians(c.align_axis_deg))
        yaw_ok = self._axis_cos(1) >= math.cos(math.radians(c.align_yaw_deg))
        return depth_ok & xy_ok & up_ok & yaw_ok

    def success(self) -> torch.Tensor:
        """(N,) bool: every stick seated in its slot — this scene's assembled state
        (scene-level success alias, matching the other suites' surface)."""
        return self.seated().all(dim=1)

    def _ram_offsets_in_case(self) -> torch.Tensor:
        """Each stick origin's offset from its seated point, in the case's local frame, shape
        (num_envs, num_slots, 3). Zero means that stick sits exactly at its seated pose."""
        from isaaclab.utils.math import quat_apply_inverse

        out = []
        for k, ram in enumerate(self.rams):
            rel = quat_apply_inverse(
                self.case.data.root_quat_w, ram.data.root_pos_w - self.case.data.root_pos_w
            )
            out.append(rel - torch.tensor(self.cfg.seat_pos[k], device=rel.device))
        return torch.stack(out, dim=1)

    def _axis_cos(self, axis: int) -> torch.Tensor:
        """cos of the angle between each stick's and the case's local `axis` (1=y: heading along
        the slot, 2=z: insertion axis), shape (num_envs, num_slots). Seated orientation = the
        case's own axes."""
        from isaaclab.utils.math import quat_apply

        e = torch.zeros(3, device=self.env.device)
        e[axis] = 1.0
        e = e.expand(self.env.num_envs, 3)
        case_ax = quat_apply(self.case.data.root_quat_w, e)
        cols = []
        for ram in self.rams:
            ram_ax = quat_apply(ram.data.root_quat_w, e)
            cols.append((ram_ax * case_ax).sum(dim=-1))
        return torch.stack(cols, dim=1)

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

