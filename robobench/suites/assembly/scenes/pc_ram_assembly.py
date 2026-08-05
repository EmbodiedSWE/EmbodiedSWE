"""PcRamAssemblyScene — press a dual-channel pair of RAM sticks into the PC's DIMM slots.

The same gaming-PC case as `pc_motherboard`/`pc_gpu` lies on its side on the table, opening up,
motherboard facing the ceiling — here the board's DIMM cluster is the work site. All four memory
slots sit empty (the model's factory sticks are removed from the case), and two loose TridentZ
sticks lie flat on the table beside the case. Goal (carried here, no task layer): stand each
stick upright over its target slot — the outermost and the second-from-socket, the alternating
pair a 2-stick dual-channel kit populates — line its PCB edge up with the slot, and press it
straight down until it seats.

The case is one kinematic body that never moves — the PC model's meshes stay visual-only; its
physics is an invisible fixture inside the case body (plus the base case's `shell_fixture`
walls — invisible colliders on the four standing sides, so nothing reaches through the shell): per empty slot a channel whose walls grip the stick's
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

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class PcRamAssemblySceneCfg(BaseCfg):
    """Config for `PcRamAssemblyScene`. Each field is a `tunable()` curriculum/difficulty dial or
    an `info()` structural constant (see `robobench.core.BaseCfg`)."""

    # --- tunable: the curriculum / difficulty dials -----------------------------------------------
    # A stick is "seated" when — in the case's frame — its blade is >= `seat_depth` below the slot
    # mouth, its origin is within `align_xy` of its seated point, and its axes are within
    # `align_axis_deg` (tilt) / `align_yaw_deg` (heading along the slot) of the case's. The full
    # stroke from the slot mouth to the channel floor is 4.44 mm. The tilt gate is wider than
    # pc_gpu's: with the shallow 3.4 mm grip band a fully seated stick may legitimately rest
    # leaned against a channel wall at up to ~5 deg (its upright-restoring range is only ~2 deg),
    # so 3 deg would false-fail a good insertion.
    seat_depth: float = tunable(0.0037)  # min blade depth below the slot mouth (m) to count seated
    align_xy: float = tunable(0.003)  # max distance (m) of a stick origin from its seated point
    align_axis_deg: float = tunable(6.0)  # max tilt of a stick's up axis off the slot axis (deg)
    align_yaw_deg: float = tunable(3.0)  # max heading error of a stick's length axis (deg)
    reset_pos_jitter: float = tunable(0.01)  # uniform +/- xy jitter for the loose sticks at reset (m)
    # Part friction (static = dynamic), set on every shape at bind. The moving stick runs
    # moderately slick against a grippier fixed case, so it slides down the channel but holds seat.
    ram_friction: float = tunable(0.3)
    case_friction: float = tunable(0.75)
    # Weld-on-closure grasping (the benchmark's auto-weld contract, PhysX
    # form — the grasp-weld machinery at the end of this scene class):
    # close the fingers flat across a stick's faces near its top edge and the stick welds to
    # the hand; open wide to release. Gripper envs only (no-op under robot="null").
    grasp_weld: bool = tunable(True)
    grasp_weld_dist: float = tunable(0.010)  # pinch-point-to-grip-band engage radius (m)

    # --- info: structure, reset layout, masses, asset paths (fixed) -------------------------------
    # Seated stick origins (PCB-blade bottom centres) in the case's local frame, one per empty DIMM
    # slot; seated orientation = the case's own axes (identity). Baked into the committed USDs
    # (keep in sync if they change). Slot 0 is the outermost (farthest from the CPU socket).
    seat_pos: tuple[tuple[float, float, float], ...] = info(
        ((-0.1426893, -0.0678899, 0.0002058), (-0.1237320, -0.0678899, 0.0002058))
    )
    slot_mouth_z: float = info(0.0046456)  # channel wall top in the case frame: depth datum
    board_top: float = info(0.0)  # board face height in the case frame (the asset's own origin)
    case_lift: float = info(0.0289)  # board face above the side panel the case lies on
    ram_mass: float = info(0.25)  # a real stick is ~45 g; 0.25 kg keeps the PD/solver in the
    # proven stability class (the asset also authors an inflated rotational inertia — see module
    # docstring)
    light_intensity: float = info(2500.0)
    # Loose stick start poses: lying flat (heat-spreader face down, RGB bar pointing away from the
    # case) on the table beside the case, end-to-end along y with a 34 mm tip gap.
    ram_init_xy: tuple[tuple[float, float], ...] = info(((0.27, -0.085), (0.27, 0.085)))
    ram_init_z: float = info(0.0042)  # origin height lying face-down (slab half 3.6 mm + pad)
    ram_init_quat: tuple[float, float, float, float] = info((0.70711, 0.0, 0.70711, 0.0))  # flat
    ram_contact_offset: float = info(0.0001)  # well below the 0.15 mm/side channel grip
    case_contact_offset: float = info(0.0001)  # ditto for the slot fixtures' walls
    # Optional foam holders (per stick: a floor pad + two rails flanking the 7.3 mm body slab)
    # that present the sticks UPRIGHT for a parallel-jaw grasp. The lying default is ungraspable
    # by a Franka gripper: flat on its face a stick's only sub-80 mm dimension (its thickness)
    # points UP, so no top-down or side pinch can straddle it. Enable together with upright
    # `ram_init_quat` (identity = the seated orientation) and `ram_init_z` = the holders' floor
    # top; the rails cap a free stick's lean at ~5 deg and the pick pulls straight up out of them.
    ram_stand: bool = info(False)
    ram_stand_gap: float = info(0.0012)  # rail clearance per side around the body slab (m)
    # Selectable work surface (same presets as the sibling scenes).
    table: str = info("lab_table")  # which work surface: "lab_table" | "packing"
    surface_z: float | None = info(None)  # table-top height (m); None -> the preset's
    workbench_pos: tuple[float, float] | None = info(None)  # xy the table sits at; None -> preset
    workbench_usd: str = info("")  # empty -> the preset's vendored USD
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.55, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }
    # Asset USDs; empty -> the prebuilt assets committed under `assets/`.
    asset_dir: str = info("")
    case_usd: str = info("")
    ram_usd: str = info("")

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
    def bind(self, env: BaseEnv) -> None:
        """Grab the case + stick handles, cache env origins, and set the part frictions."""
        super().bind(env)
        self.case: RigidObject = env.iscene["case"]
        self.rams: list[RigidObject] = [env.iscene[f"ram_{k}"] for k in range(self.cfg.num_slots)]
        self.env_origins = env.iscene.env_origins
        self._set_friction(self.case, self.cfg.case_friction)
        for ram in self.rams:
            self._set_friction(ram, self.cfg.ram_friction)
        self._grasp_weld_bind()

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
        self._grasp_weld_step()

    def _set_friction(self, asset, value: float) -> None:
        """Overwrite the static + dynamic friction on every shape of `asset` (across all envs)."""
        mats = asset.root_physx_view.get_material_properties()
        mats[..., 0:2] = value  # [static, dynamic, restitution]
        asset.root_physx_view.set_material_properties(mats, torch.arange(self.env.num_envs, device="cpu"))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start: the case pinned at spawn, both sticks lying flat on the table
        beside it, with xy jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]  # (m, 3)
        wx, wy = c.workbench_pos

        for ram, (ix, iy) in zip(self.rams, c.ram_init_xy):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.tensor((wx + ix, wy + iy, c.surface_z + c.ram_init_z), device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 3:7] = torch.tensor(c.ram_init_quat, device=dev)
            ram.write_root_state_to_sim(st, env_ids)
        self._grasp_weld_release_all(env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable scene state: world root states (13) of the case and both sticks."""
        out = {"case": self.case.data.root_state_w[env_ids].clone()}
        for k, ram in enumerate(self.rams):
            out[f"ram_{k}"] = ram.data.root_state_w[env_ids].clone()
        out.update(self._grasp_weld_state(env_ids))
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned. A stick's insertion depth is fully captured by its
        root state, so the channel holds it on restore."""
        self.case.write_root_pose_to_sim(state["case"][:, 0:7], env_ids)
        for k, ram in enumerate(self.rams):
            ram.write_root_state_to_sim(state[f"ram_{k}"], env_ids)
        self._grasp_weld_restore(state, env_ids)

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

    # ----- grasp-weld machinery (the weld-on-closure contract; private — not an agent action) ----
    # Pre-authored, normally-disabled FixedJoint pools, toggled and never created mid-sim — the
    # ikea/chair toggle pattern aimed hand<->part, with the pouring suite's closure criterion.
    # PhysX latches a joint's local frames on FIRST enable and ignores rewrites on a re-enable,
    # so every engage consumes a fresh pool joint: the live hand->part pose is written while the
    # joint is still disabled, it is enabled once, and on release it is retired for good.
    # Engage (reconciled every physics substep, debounced): pinch point within `grasp_weld_dist`
    # of a site's LIVE grip band + aperture inside the site's closure window (below = closed on
    # air, above = nothing snagged) + fingers STALLED (a closing sweep passes through the window;
    # a real pinch stops in it). Release: aperture past window-top + margin (hysteresis). One
    # hold per env (a parallel jaw pinches one part). Embodiment-agnostic: no hand on the stage
    # (e.g. robot="null") -> no joints, no-op contract. Holds ride get_state/set_state.
    GRASP_HAND_BODY: ClassVar[str] = "panda_hand"
    GRASP_FINGER_JOINTS: ClassVar[str] = "panda_finger_joint.*"
    GRASP_PINCH_OFFSET: ClassVar[float] = 0.1034  # hand origin -> finger-pad centre, along approach
    GRASP_POOL: ClassVar[int] = 8  # engages per (env, site) per run; exhausted -> warn, no weld
    GRASP_STALL: ClassVar[float] = 0.01  # max |finger vel| sum (m/s): fingers stopped ON the part
    GRASP_DEBOUNCE: ClassVar[int] = 8  # consecutive qualifying substeps before the weld engages
    GRASP_RELEASE_MARGIN: ClassVar[float] = 0.008  # release at window-top + this (m), hysteresis

    def _grasp_weld_bind(self) -> None:
        """Discover the hand, author the (disabled) joint pools, allocate the hold state. Called
        from `bind()` — authoring must happen BEFORE the sim starts playing, or PhysX only picks
        the joints up after a full `sim.reset()`."""
        env = self.env
        n = env.num_envs
        self._gw_on = bool(getattr(self.cfg, "grasp_weld", False))
        self._gw_art = None  # articulation handle, resolved lazily (the robot binds after us)
        self._gw_sites: list = []
        if not self._gw_on:
            return
        hand0 = self._gw_find_hand_prim()
        if hand0 is None:  # no gripper in this embodiment (e.g. robot="null") -> no-op contract
            self._gw_on = False
            print(f"[grasp-weld] no '{self.GRASP_HAND_BODY}' on the stage — contract disabled", flush=True)
            return
        self._gw_sites = list(self.grasp_sites())
        s = len(self._gw_sites)
        dev = env.device
        self.grasp_held = torch.zeros(n, s, dtype=torch.bool, device=dev)
        self._gw_rel_p = torch.zeros(n, s, 3, device=dev)
        self._gw_rel_q = torch.zeros(n, s, 4, device=dev)
        self._gw_count = torch.zeros(n, s, dtype=torch.int32, device=dev)
        self._gw_pool_i = [[0] * s for _ in range(n)]
        self._gw_pool_warned: set = set()
        self._gw_author_pools(hand0)

    def _gw_find_hand_prim(self) -> str | None:
        """The hand body's prim path under env_0 (clones are identical), or None if absent."""
        from pxr import Usd

        root = self.env.stage.GetPrimAtPath("/World/envs/env_0")
        if not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            if prim.GetName() == self.GRASP_HAND_BODY:
                return str(prim.GetPath())
        return None

    def _gw_part_path(self, obj, env_i: int) -> str:
        """The part's RIGID-BODY prim path in env `env_i`. The asset root from the cfg is not
        always the body (the allen-key USDs nest it one level down), so walk the subtree for the
        first `RigidBodyAPI` prim — the joint must bind the body, or PhysX ignores it."""
        from pxr import Usd, UsdPhysics

        p = obj.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        root = p.replace("env_.*", f"env_{env_i}")
        prim = self.env.stage.GetPrimAtPath(root)
        if not prim.IsValid():
            raise RuntimeError(f"[grasp-weld] part prim missing: {root}")
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(child.GetPath())
        raise RuntimeError(f"[grasp-weld] no RigidBodyAPI prim under {root}")

    def _gw_author_pools(self, hand0: str) -> None:
        """One pool of disabled FixedJoints per (env, site): body0 = the hand, body1 = the part,
        frames identity until an engage writes the live relative pose."""
        from pxr import Gf, UsdPhysics

        stage = self.env.stage
        self._gw_paths: list[list[list[str]]] = []  # [env][site][k]
        for i in range(self.env.num_envs):
            hand = hand0.replace("env_0", f"env_{i}")
            rows = []
            for name, obj, _p0, _p1, _win in self._gw_sites:
                part = self._gw_part_path(obj, i)
                row = []
                for k in range(self.GRASP_POOL):
                    jp = f"/World/envs/env_{i}/gweld_{name}_{k}"
                    j = UsdPhysics.FixedJoint.Define(stage, jp)
                    j.CreateBody0Rel().SetTargets([hand])
                    j.CreateBody1Rel().SetTargets([part])
                    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                    j.CreateJointEnabledAttr(False)
                    j.CreateExcludeFromArticulationAttr(True)  # maximal-coordinate, not an arm DOF
                    row.append(jp)
                rows.append(row)
            self._gw_paths.append(rows)

    def _gw_resolve_hand(self) -> bool:
        """Cache the articulation handle + indices on first use (the robot binds after the scene)."""
        if self._gw_art is not None:
            return True
        try:
            art = self.env.robot.articulation
            self._gw_hand_i = art.body_names.index(self.GRASP_HAND_BODY)
            self._gw_fingers = art.find_joints([self.GRASP_FINGER_JOINTS])[0]
            assert len(self._gw_fingers) == 2
        except Exception as e:  # articulated but not a gripper we understand -> disable, loudly
            self._gw_on = False
            print(f"[grasp-weld] DISABLED after error: {e!r}", flush=True)
            return False
        self._gw_art = art
        return True

    def _grasp_weld_step(self) -> None:
        """Reconcile engages + releases against the closure criterion. Called from `post_step()`."""
        if not self._gw_on or not self._gw_sites or not self._gw_resolve_hand():
            return
        from isaaclab.utils.math import quat_apply

        art = self._gw_art
        hp = art.data.body_pos_w[:, self._gw_hand_i]
        hq = art.data.body_quat_w[:, self._gw_hand_i]
        gap = art.data.joint_pos[:, self._gw_fingers].sum(dim=-1)
        stalled = art.data.joint_vel[:, self._gw_fingers].abs().sum(dim=-1) < self.GRASP_STALL
        approach = torch.zeros_like(hp)
        approach[:, 2] = self.GRASP_PINCH_OFFSET
        pinch = hp + quat_apply(hq, approach)

        # Releases first (a re-grasp in the same step then sees a free hand).
        for row, s in self.grasp_held.nonzero(as_tuple=False).tolist():
            if gap[row] > self._gw_sites[s][4][1] + self.GRASP_RELEASE_MARGIN:
                self._gw_release(row, s)

        free = ~self.grasp_held.any(dim=-1)  # (n,)
        dists = self._gw_site_dists(pinch)  # (n, s)
        c = getattr(self.cfg, "grasp_weld_dist", 0.010)
        ok = torch.stack(
            [
                (dists[:, s] < c) & (gap > win[0]) & (gap < win[1]) & stalled
                for s, (_n, _o, _p0, _p1, win) in enumerate(self._gw_sites)
            ],
            dim=-1,
        ) & free.unsqueeze(-1)
        self._gw_count = torch.where(ok, self._gw_count + 1, torch.zeros_like(self._gw_count))
        ready = (self._gw_count >= self.GRASP_DEBOUNCE).any(dim=-1) & free
        for row in ready.nonzero(as_tuple=False).flatten().tolist():
            masked = torch.where(
                self._gw_count[row] >= self.GRASP_DEBOUNCE, dists[row], torch.full_like(dists[row], torch.inf)
            )
            s = int(masked.argmin())
            self._gw_engage(row, s, hp[row], hq[row], gap[row])

    def _gw_site_dists(self, pinch: torch.Tensor) -> torch.Tensor:
        """Pinch-point distance to every site's live grip band, shape (num_envs, num_sites)."""
        from isaaclab.utils.math import quat_apply

        n = pinch.shape[0]
        out = []
        for _name, obj, p0, p1, _win in self._gw_sites:
            pp, pq = obj.data.root_pos_w, obj.data.root_quat_w
            a = pp + quat_apply(pq, torch.tensor(p0, device=pinch.device).expand(n, 3))
            b = pp + quat_apply(pq, torch.tensor(p1, device=pinch.device).expand(n, 3))
            ab = b - a
            t = ((pinch - a) * ab).sum(-1) / ab.pow(2).sum(-1).clamp_min(1e-12)
            closest = a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab
            out.append((pinch - closest).norm(dim=-1))
        return torch.stack(out, dim=-1)

    def _gw_engage(self, env_i: int, s: int, hp: torch.Tensor, hq: torch.Tensor, gap: torch.Tensor) -> None:
        """Weld (env_i, site s) to the hand at the live relative pose, on a fresh pool joint."""
        from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

        name, obj = self._gw_sites[s][0], self._gw_sites[s][1]
        rel_p = quat_apply_inverse(hq.unsqueeze(0), (obj.data.root_pos_w[env_i] - hp).unsqueeze(0))[0]
        rel_q = quat_mul(quat_conjugate(hq.unsqueeze(0)), obj.data.root_quat_w[env_i].unsqueeze(0))[0]
        if not self._gw_set_joint(env_i, s, rel_p, rel_q):
            return
        self._gw_rel_p[env_i, s] = rel_p
        self._gw_rel_q[env_i, s] = rel_q
        self.grasp_held[env_i, s] = True
        self._gw_count[env_i] = 0
        print(f"[grasp-weld] env {env_i}: GRIPPED {name} (aperture {float(gap) * 1000:.1f} mm)", flush=True)

    def _gw_set_joint(self, env_i: int, s: int, rel_p: torch.Tensor, rel_q: torch.Tensor) -> bool:
        """Write the hand-frame pose onto the next fresh pool joint and enable it. False = pool dry."""
        from pxr import Gf, UsdPhysics

        k = self._gw_pool_i[env_i][s]
        if k >= self.GRASP_POOL:
            if (env_i, s) not in self._gw_pool_warned:
                self._gw_pool_warned.add((env_i, s))
                print(f"[grasp-weld] env {env_i}: pool dry for {self._gw_sites[s][0]} — no weld", flush=True)
            return False
        j = UsdPhysics.FixedJoint.Get(self.env.stage, self._gw_paths[env_i][s][k])
        p, q = rel_p.tolist(), rel_q.tolist()
        j.GetLocalPos0Attr().Set(Gf.Vec3f(p[0], p[1], p[2]))
        j.GetLocalRot0Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
        j.GetJointEnabledAttr().Set(True)
        return True

    def _gw_release(self, env_i: int, s: int) -> None:
        """Cut (env_i, site s): disable the joint and retire it (frames latched — never reused)."""
        from pxr import UsdPhysics

        k = self._gw_pool_i[env_i][s]
        if k < self.GRASP_POOL:
            j = UsdPhysics.FixedJoint.Get(self.env.stage, self._gw_paths[env_i][s][k])
            j.GetJointEnabledAttr().Set(False)
        self._gw_pool_i[env_i][s] = k + 1
        self.grasp_held[env_i, s] = False
        print(f"[grasp-weld] env {env_i}: RELEASED {self._gw_sites[s][0]}", flush=True)

    def _grasp_weld_release_all(self, env_ids: torch.Tensor) -> None:
        """Cut every hold for `env_ids` (a fresh episode starts empty-handed). Called from `reset()`."""
        if not getattr(self, "_gw_on", False):
            return
        for row, s in self.grasp_held[env_ids].nonzero(as_tuple=False).tolist():
            self._gw_release(int(env_ids[row]), s)
        self._gw_count[env_ids] = 0

    def _grasp_weld_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """The contract's restorable state (empty when the contract is off)."""
        if not getattr(self, "_gw_on", False):
            return {}
        return {
            "grasp_held": self.grasp_held[env_ids].clone(),
            "grasp_rel_p": self._gw_rel_p[env_ids].clone(),
            "grasp_rel_q": self._gw_rel_q[env_ids].clone(),
        }

    def _grasp_weld_restore(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Re-arm the holds `get_state` recorded, at their RECORDED hand-frame poses (the bodies
        were just written, so live measurement is redundant), on fresh pool joints. Called from
        `set_state()` after the bodies are restored."""
        if not getattr(self, "_gw_on", False) or "grasp_held" not in state:
            return
        for row in range(len(env_ids)):
            i = int(env_ids[row])
            for s in range(len(self._gw_sites)):
                if self.grasp_held[i, s]:
                    self._gw_release(i, s)
                if bool(state["grasp_held"][row, s]) and self._gw_set_joint(
                    i, s, state["grasp_rel_p"][row, s], state["grasp_rel_q"][row, s]
                ):
                    self._gw_rel_p[i, s] = state["grasp_rel_p"][row, s]
                    self._gw_rel_q[i, s] = state["grasp_rel_q"][row, s]
                    self.grasp_held[i, s] = True
        self._gw_count[env_ids] = 0
