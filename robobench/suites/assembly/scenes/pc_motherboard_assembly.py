"""PcMotherboardAssemblyScene — screw a motherboard into a PC case, one allen bolt per hole.

A gaming-PC case lies on its side on the table, opening up, motherboard facing the ceiling. The
board carries 7 case-mount screw holes, each backed by an M8 thread insert hidden in the case.
Beside the case: 7 loose M8 socket-head cap screws and one long-series L-shaped allen key — its
210 mm working arm keeps the swinging handle above the case walls. Goal (carried here, no task
layer): stand each bolt in a hole and drive it down with the key (clockwise while pressing) until
every hole is fastened.

SEVEN fastenings share one fixed part, so the key must move hole-to-hole. The case is one rigid
body that never moves — the PC model is visual-only; its physics is 7 thread inserts + 7
invisible bored seat plates flush with the board face. The case origin is the CENTRE of the
board's top face, z=0 ON the face, so a bolt's tip depth below the board is just `case_z -
bolt_z` (the bolt origin IS its tip).

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
class PcMotherboardAssemblySceneCfg(BaseCfg):
    """Config for `PcMotherboardAssemblyScene`. Nothing is locked — a variant is just a copy with a few
    fields changed."""

    # --- the curriculum / difficulty dials -----------------------------------------------
    # A bolt is "seated" when — in the case's frame — its tip is >= `seat_depth` below the board
    # face, within `align_xy` of a hole's axis, and tilted <= `align_axis_deg` off it. The head
    # bottoms out at 12.4 mm tip depth, so 11 mm separates "seated" from "merely started".
    seat_depth: float = 0.011  # min tip depth below the board face (m) to count as seated
    align_xy: float = 0.003  # max lateral distance (m) of the bolt tip from the hole axis
    align_axis_deg: float = 5.0  # max tilt of the bolt axis off the hole axis (deg)
    reset_pos_jitter: float = 0.01  # uniform +/- xy jitter per loose part at reset (m)
    # Part friction (static = dynamic), set on every shape at bind. The MOVING threaded part runs
    # slick (0.01) against a grippier fixed part (0.75).
    bolt_friction: float = 0.01
    case_friction: float = 0.75
    # Weld-on-closure grasping (the benchmark's auto-weld contract, PhysX
    # form — the grasp-weld machinery at the end of this scene class):
    # close the fingers across the key handle's hex and the key welds to the hand; open wide
    # to release. Gripper envs only (no-op under robot="null").
    grasp_weld: bool = True
    grasp_weld_dist: float = 0.010  # pinch-point-to-grip-band engage radius (m)
    # Kinematic screw-joint threading: every bolt spawns STAGED hand-started in its hole, is
    # kinematic, and descends its 1 mm-pitch helix by following the key's hex-engaged
    # rotation through the lash, one-way, to a hard stop just above seating the head. The
    # thread inserts' collision is off (the joint IS the thread); the bolts' SOCKET walls
    # stay live, so insertion, press, cam-out, and slip are real contacts. False = dynamic
    # bolts lying beside the case and live inserts.
    screw_mechanic: bool = True
    stage_depth: float = 0.006  # staged bolts' tip depth below the board face (m)
    stage_yaw: float = 3.141592653589793  # staged bolts' yaw (a k*60 deg hex clocking)
    key_friction: float = 0.6

    # --- structure, reset layout, masses, asset paths (fixed) -------------------------------
    num_holes: int = 7  # motherboard case-mount screw holes (= number of bolts)
    # Hole axes in the case's local frame (xy on the z=0 board face), serpentine drive order.
    # Baked into the committed case USD (keep in sync if the asset changes). The board's
    # position keeps a hex key cranking in any hole clear of the IO-panel wall.
    hole_xy: tuple[tuple[float, float], ...] = (
        (+0.0376, -0.1448),  # top_left
        (-0.1650, -0.1444),  # top_right
        (-0.1651, +0.0109),  # mid_right
        (+0.0614, +0.0108),  # mid_left
        (+0.0619, +0.1345),  # bot_left
        (-0.0931, +0.1347),  # bot_mid
        (-0.1648, +0.1343),  # bot_right
    )
    board_top: float = 0.0  # board face height in the case frame (the asset's own origin)
    case_lift: float = 0.0289  # board face above the side panel the case lies on
    thread_len: float = 0.0124  # bolt thread length: tip depth at which the head bottoms out
    bolt_mass: float = 0.012  # M8 socket-head cap screw (kg)
    light_intensity: float = 2500.0
    # Loose parts' start pose: bolts lying in a row beside the case (+x side), key flat past the end.
    bolt_init_xy: tuple[tuple[float, float], ...] = ()  # per-bolt start xy (table-rel.)
    bolt_row_x: float = 0.24  # x of the bolt row (the case spans x < 0.13)
    bolt_row_y0: float = -0.27  # y of bolt0
    bolt_spacing: float = 0.09  # y gap between adjacent bolts
    bolt_init_z: float = 0.0065  # bolt-origin height when lying on its side (head rim + crest)
    bolt_init_quat: tuple[float, float, float, float] = (0.70711, 0.0, 0.70711, 0.0)  # lying
    key_init_xy: tuple[float, float] = (0.24, 0.38)  # key start xy (table-rel.)
    key_init_z: float = 0.004  # resting on a hex flat (apothem 3.1 mm) + margin
    key_init_quat: tuple[float, float, float, float] = (0.70711, 0.70711, 0.0, 0.0)  # flat
    key_mass: float = 0.10  # steel 6.25 mm long-series L-key, 210 mm arm (kg)
    key_disable_gravity: bool = False  # the force-driven key smoke sets this True (no hand to bear the handle's weight)
    # Optional upright stand (a four-wall pocket) that presents the key standing tip-down, its
    # handle 210 mm up as a ready top-down grip — a gripper env sets key_stand=True (and an
    # upright key_init_quat), because the flat-lying key demands a low pinch and a 90 deg
    # in-hand reorientation before it can screw anything.
    key_stand: bool = False
    key_stand_gap: float = 0.0022  # pocket clearance per side around the arm's 7.2 mm corners
    key_contact_offset: float = 0.00025  # well below the 0.375 mm/side socket clearance
    bolt_contact_offset: float = 0.00025  # ditto for the bolt's socket walls
    # Selectable work surface (same presets as the sibling scenes).
    table: str = "lab_table"  # which work surface: "lab_table" | "packing"
    surface_z: float | None = None  # table-top height (m); None -> the preset's
    workbench_pos: tuple[float, float] | None = None  # xy the table sits at; None -> preset
    workbench_usd: str = ""  # empty -> the preset's vendored USD
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.5, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }
    # Asset USDs; empty -> the prebuilt assets committed under `assets/`.
    asset_dir: str = ""
    case_usd: str = ""
    bolt_usd: str = ""
    key_usd: str = ""

    def __post_init__(self) -> None:
        if not self.bolt_init_xy:
            self.bolt_init_xy = tuple(
                (self.bolt_row_x, self.bolt_row_y0 + k * self.bolt_spacing) for k in range(self.num_holes)
            )
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets)
        self.case_usd = self.case_usd or str(Path(self.asset_dir) / "pc" / "pc_case_assembly_mb.usd")
        self.bolt_usd = self.bolt_usd or str(Path(self.asset_dir) / "allen_bolt" / "allen_bolt_m8.usd")
        self.key_usd = self.key_usd or str(Path(self.asset_dir) / "allen_key" / "allen_key_m8_long.usd")
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(assets / "props" / preset["usd"][0] / preset["usd"][1])


@SCENES.register("pc_motherboard")
class PcMotherboardAssemblyScene(BaseScene):
    cfg: PcMotherboardAssemblySceneCfg

    def __init__(self, cfg: PcMotherboardAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or PcMotherboardAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, dome light, table, the PC case lying on it (kinematic), `num_holes` loose bolts,
        and one allen key. Bolts load with the high solver-iteration count the threads need."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.case_usd, c.bolt_usd, c.key_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — the pc-motherboard assets ship with the repo under "
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
            # The case: kinematic; the 7 invisible thread inserts under the board's mount holes
            # are what the bolts screw into.
            "case": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Case",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.case_usd,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(wx, wy, c.surface_z + c.case_lift)),
            ),
        }
        for i in range(c.num_holes):
            bx, by = c.bolt_init_xy[i]
            out[f"bolt_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bolt_%d" % i,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.bolt_usd,
                    activate_contact_sensors=True,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.bolt_contact_offset, rest_offset=0.0
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=192,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.02,
                        linear_damping=2.0,
                        angular_damping=2.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bolt_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + bx, wy + by, c.surface_z + c.bolt_init_z), rot=c.bolt_init_quat
                ),
            )
        if c.key_stand:
            # Upright key stand for a gripper env: four STATIC walls (no rigid body) forming a
            # square pocket at the key spawn — the arm stands tip-down inside, its lean capped at
            # ~3 deg. Wall tops stay 160+ mm below the handle, far from any descending finger.
            kx, ky = c.key_init_xy
            inner = 0.0072 + 2 * c.key_stand_gap
            wall_h, wall_t = 0.045, 0.006
            foam = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.17, 0.17, 0.2), roughness=0.9)
            for name, size, (dx, dy) in (
                ("key_stand_n", (inner + 2 * wall_t, wall_t, wall_h), (0.0, +(inner + wall_t) / 2)),
                ("key_stand_s", (inner + 2 * wall_t, wall_t, wall_h), (0.0, -(inner + wall_t) / 2)),
                ("key_stand_e", (wall_t, inner, wall_h), (+(inner + wall_t) / 2, 0.0)),
                ("key_stand_w", (wall_t, inner, wall_h), (-(inner + wall_t) / 2, 0.0)),
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
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(wx + kx + dx, wy + ky + dy, c.surface_z + 0.5 * wall_h)
                    ),
                )
        # Key contact offset must stay well below the key<->socket clearance (0.375 mm/side) or
        # speculative contacts choke the fit.
        kx, ky = c.key_init_xy
        out["key"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Key",
            spawn=sim_utils.UsdFileCfg(
                usd_path=c.key_usd,
                activate_contact_sensors=True,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.key_contact_offset, rest_offset=0.0
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=c.key_disable_gravity,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=0.02,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.key_mass),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(wx + kx, wy + ky, c.surface_z + c.key_init_z), rot=c.key_init_quat
            ),
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
        """Grab the case + bolt + key handles, cache env origins, and set the part frictions."""
        super().bind(env)
        self.case: RigidObject = env.iscene["case"]
        self.bolts: list[RigidObject] = [env.iscene[f"bolt_{i}"] for i in range(self.cfg.num_holes)]
        self.key: RigidObject = env.iscene["key"]
        self.env_origins = env.iscene.env_origins
        self._set_friction(self.case, self.cfg.case_friction)
        self._set_friction(self.key, self.cfg.key_friction)
        for bolt in self.bolts:
            self._set_friction(bolt, self.cfg.bolt_friction)
        self.grasp_weld = GraspWeldContract(self)  # composed, publishes self.grasp_held
        self.grasp_weld.bind()
        self._screw_bind()

    def grasp_sites(self) -> list:
        """One grip band: the key's HANDLE (local +x off the elbow at the working arm's top,
        z 0.210), across its hex — 6.2 mm flats / 7.2 mm corners. The key USD's origin is the
        working arm's tip, arm up local +z."""
        return [("key", self.key, (0.005, 0.0, 0.210), (0.115, 0.0, 0.210), (0.004, 0.010))]

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Reconcile the weld-on-closure grasp contract and the screw joints every physics
        substep."""
        self.grasp_weld.step()
        self._screw_step()

    def _set_friction(self, asset, value: float) -> None:
        """Overwrite the static + dynamic friction on every shape of `asset` (across all envs)."""
        mats = asset.root_physx_view.get_material_properties()
        mats[..., 0:2] = value  # [static, dynamic, restitution]
        asset.root_physx_view.set_material_properties(mats, torch.arange(self.env.num_envs, device="cpu"))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start: the case pinned at spawn, bolts lying in a row beside it,
        the key flat past the row's end — all with xy jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]  # (m, 3)
        wx, wy = c.workbench_pos

        if self._screw_on:  # staged bolts: the mechanic owns them from spawn
            rows = [(self.key, c.key_init_xy, c.key_init_z, c.key_init_quat)]
        else:
            rows = [(bolt, xy, c.bolt_init_z, c.bolt_init_quat) for bolt, xy in zip(self.bolts, c.bolt_init_xy)]
            rows.append((self.key, c.key_init_xy, c.key_init_z, c.key_init_quat))
        for part, (x, y), init_z, init_quat in rows:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.tensor((wx + x, wy + y, c.surface_z + init_z), device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 3:7] = torch.tensor(init_quat, device=dev)
            part.write_root_state_to_sim(st, env_ids)
        self.grasp_weld.release_all(env_ids)
        self._screw_reset(env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable scene state: world root states (13) of the case, each bolt, and the key."""
        return {
            "case": self.case.data.root_state_w[env_ids].clone(),
            "bolts": torch.stack([b.data.root_state_w[env_ids].clone() for b in self.bolts], dim=1),
            "key": self.key.data.root_state_w[env_ids].clone(),
            **self.grasp_weld.state(env_ids),
            **self._screw_state(env_ids),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned. A bolt's threaded depth is fully captured by its
        root state, so thread friction holds it on restore."""
        self.case.write_root_pose_to_sim(state["case"][:, 0:7], env_ids)
        for i, bolt in enumerate(self.bolts):
            bolt.write_root_state_to_sim(state["bolts"][:, i], env_ids)
        self.key.write_root_state_to_sim(state["key"], env_ids)
        self.grasp_weld.restore(state, env_ids)
        self._screw_restore(state, env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        n = self.cfg.num_holes
        return (
            f"A gaming-PC case lying on its side on a sturdy table, opening up, its motherboard "
            f"facing the ceiling. The board carries {n} case-mount screw holes, each backed by an "
            f"M8 threaded insert in the case. Beside the case: {n} loose M8 allen (socket-head) "
            f"bolts lying in a row, and one long-series L-shaped allen key (its arm is long "
            f"enough that the swinging handle clears the case walls). Each bolt head carries a "
            f"7 mm hex socket.\n"
            + (
                f"Every hole already holds its bolt hand-started upright, a few threads in.\n"
                f"Goal: seat the key in a bolt's socket and drive the bolt down (turn clockwise "
                f"while pressing) until it seats — then move on to the next hole. "
                if self.cfg.screw_mechanic
                else f"Goal: stand a bolt tip-down in each hole, seat the key in its socket, and "
                f"drive it down (turn clockwise while pressing) until it seats — then move on to the "
                f"next hole. "
            )
            + f"A seated bolt locks in place. The task is complete once all {n} bolts "
            f"are seated."
            + (
                " The key holds in a firm pinch: close the fingers across its handle's hex and "
                "the grip locks; open wide to release."
                if self.cfg.grasp_weld
                else ""
            )
        )

    # ----- progress (public: seated()/engaged(); reads how far the assembly has got) -------------
    def engaged(self) -> torch.Tensor:
        """Tip depth below the board face at each bolt's nearest hole, shape (num_envs, num_holes),
        in metres (negative = still above the board). The head bottoms out at `thread_len` depth."""
        off = self._bolt_offsets_in_case()  # (n, B, H, 3)
        near = off[..., :2].norm(dim=-1).argmin(dim=-1)  # nearest hole per bolt
        z = torch.gather(off[..., 2], 2, near.unsqueeze(-1)).squeeze(-1)
        return self.cfg.board_top - z

    def seated(self) -> torch.Tensor:
        """Whether each bolt is seated in a hole, shape (num_envs, num_holes): threaded down to
        `seat_depth`, within `align_xy` of the hole axis, and within `align_axis_deg` of it.
        An env is assembled when every entry of its row is True (7 bolts, 7 holes)."""
        import math

        off = self._bolt_offsets_in_case()  # (n, B, H, 3)
        near_dist, near = off[..., :2].norm(dim=-1).min(dim=-1)
        z = torch.gather(off[..., 2], 2, near.unsqueeze(-1)).squeeze(-1)
        depth_ok = (self.cfg.board_top - z) >= self.cfg.seat_depth
        axis_ok = self._bolt_axis_cos() >= math.cos(math.radians(self.cfg.align_axis_deg))
        return depth_ok & (near_dist <= self.cfg.align_xy) & axis_ok

    def success(self) -> torch.Tensor:
        """(N,) bool: every bolt seated in a hole — this scene's assembled state
        (scene-level success alias, matching the other suites' surface)."""
        return self.seated().all(dim=1)

    def _bolt_offsets_in_case(self) -> torch.Tensor:
        """Each bolt's position relative to each hole, in the case's local frame, shape
        (num_envs, num_bolts, num_holes, 3): xy = lateral offset from that hole's axis, z = tip
        height above the board face (the bolt origin IS the tip, the case origin IS the face)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        cp = self.case.data.root_pos_w  # (n, 3)
        cq = self.case.data.root_quat_w  # (n, 4)
        holes = torch.tensor(c.hole_xy, device=cp.device)  # (H, 2)
        cols = []
        for bolt in self.bolts:
            rel = quat_apply_inverse(cq, bolt.data.root_pos_w - cp)  # (n, 3) in case frame
            off = rel[:, None, :].repeat(1, c.num_holes, 1)  # (n, H, 3)
            off[..., 0:2] -= holes
            cols.append(off)
        return torch.stack(cols, dim=1)  # (n, B, H, 3)

    def _bolt_axis_cos(self) -> torch.Tensor:
        """cos of the angle between each bolt's screw axis (its local +z) and the hole axis (the
        case's local +z), shape (num_envs, num_holes)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        case_up = quat_apply(self.case.data.root_quat_w, ez)  # (n, 3)
        bolt_up = torch.stack([quat_apply(b.data.root_quat_w, ez) for b in self.bolts], dim=1)  # (n, B, 3)
        return (bolt_up * case_up[:, None, :]).sum(dim=-1)

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


    # ----- screw-joint machinery (the thread mechanic; private — not an agent action) ---------
    # Each staged bolt is a kinematic screw DOF on its hole's axis: while the key's tip sits
    # hex-engaged in a bolt's socket, that bolt follows the key's measured rotation through the
    # hex lash — one-way, like a frictional thread — and descends its helix at SCREW_PITCH per
    # revolution to a hard stop just above seating the head. Parked bolts hold their pose; the
    # lash re-charges on every socket re-entry. Reconciled every physics substep from live
    # poses alone, embodiment-agnostic.
    SCREW_PITCH: ClassVar[float] = 0.001  # helix pitch (m per revolution)
    SCREW_SOCKET_MOUTH_Z: ClassVar[float] = 0.0214  # bolt-local: head top = the recess mouth
    SCREW_SEAT_MARGIN: ClassVar[float] = 0.0001  # hard stop: the head held this far off the board
    SCREW_LASH_HALF: ClassVar[float] = math.radians(8.0)  # key rotation before the flats engage
    SCREW_ENGAGE_AXIAL: ClassVar[float] = 0.0008  # engaged = tip below mouth by this margin
    SCREW_ENGAGE_LATERAL: ClassVar[float] = 0.004  # max tip-to-hole-axis distance while engaged

    def _screw_bind(self) -> None:
        """Flip every bolt kinematic, disable the thread inserts' collision (at bind = before
        play, so PhysX parses the edits with the scene), and allocate the joint state."""
        self._screw_on = bool(self.cfg.screw_mechanic)
        if not self._screw_on:
            return
        from pxr import UsdPhysics

        stage = self.env.stage
        c = self.cfg
        for e in range(self.env.num_envs):
            base = f"/World/envs/env_{e}"
            for i in range(c.num_holes):
                prim = stage.GetPrimAtPath(f"{base}/Bolt_{i}/allen_bolt")
                if not prim.IsValid():
                    raise RuntimeError(f"[screw] bolt body prim missing: bolt {i}, env {e}")
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                prim = stage.GetPrimAtPath(f"{base}/Case/case/hole_{i}/thread_insert")
                if not prim.IsValid():
                    raise RuntimeError(f"[screw] thread_insert prim missing: hole {i}, env {e}")
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        n = self.env.num_envs
        dev = self.env.device
        wx, wy = c.workbench_pos
        holes = torch.tensor(c.hole_xy, device=dev) + torch.tensor((wx, wy), device=dev)  # (B, 2)
        self._sj_holes = self.env_origins[:, None, 0:2] + holes[None]  # (n, B, 2), world
        self._sj_board_z = self.env_origins[:, 2] + c.surface_z + c.case_lift  # (n,), world
        self._sj_turn_max = (c.thread_len - self.SCREW_SEAT_MARGIN - c.stage_depth) * 2 * math.pi / self.SCREW_PITCH
        self.screw_turn = torch.zeros(n, c.num_holes, device=dev)  # per-hole screw-in rotation (rad)
        self._sj_coupled = torch.zeros(n, c.num_holes, device=dev)  # engaged key rotation per joint
        self._sj_eng = torch.zeros(n, c.num_holes, dtype=torch.bool, device=dev)
        self._sj_prev = torch.zeros(n, device=dev)  # key yaw last substep
        self._sj_fresh = torch.ones(n, dtype=torch.bool, device=dev)  # rows needing a yaw baseline

    @staticmethod
    def _sj_yaw(q: torch.Tensor) -> torch.Tensor:
        """Yaw about world +z of a wxyz quaternion batch, shape (n,)."""
        return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def _screw_step(self) -> None:
        """Advance engaged joints from the key's measured spin. Runs every physics substep."""
        if not getattr(self, "_screw_on", False):
            return
        yaw = self._sj_yaw(self.key.data.root_quat_w)
        dspin = -((yaw - self._sj_prev + math.pi) % (2 * math.pi) - math.pi)  # +ve = screw-in
        dspin = torch.where(self._sj_fresh, torch.zeros_like(dspin), dspin)  # fresh rows: baseline only
        self._sj_prev = yaw
        self._sj_fresh[:] = False

        kp = self.key.data.root_pos_w
        bolt_z = torch.stack([b.data.root_pos_w[:, 2] for b in self.bolts], dim=-1)  # (n, B)
        tip_ax = kp[:, 2, None] - bolt_z
        tip_lat = (kp[:, None, 0:2] - self._sj_holes).norm(dim=-1)
        engaged = (tip_ax < self.SCREW_SOCKET_MOUTH_Z - self.SCREW_ENGAGE_AXIAL) & (
            tip_lat < self.SCREW_ENGAGE_LATERAL
        )
        entered = engaged & ~self._sj_eng
        self._sj_coupled = torch.where(entered, self.screw_turn, self._sj_coupled)  # lash re-charges
        self._sj_coupled = torch.where(engaged, self._sj_coupled + dspin[:, None], self._sj_coupled)
        self._sj_eng = engaged
        follow = (self._sj_coupled - self.SCREW_LASH_HALF).clamp(max=self._sj_turn_max)
        self.screw_turn = torch.where(engaged, torch.maximum(self.screw_turn, follow), self.screw_turn)
        self._screw_write(engaged)

    def _screw_write(self, mask: torch.Tensor) -> None:
        """Write the kinematic pose of every (env, bolt) in `mask` from its screw state."""
        c = self.cfg
        for b, bolt in enumerate(self.bolts):
            rows = mask[:, b].nonzero(as_tuple=False).flatten()
            if not len(rows):
                continue
            turn = self.screw_turn[rows, b]
            yaw = c.stage_yaw - turn
            st = torch.zeros(len(rows), 7, device=turn.device)
            st[:, 0:2] = self._sj_holes[rows, b]
            st[:, 2] = self._sj_board_z[rows] - c.stage_depth - self.SCREW_PITCH * turn / (2 * math.pi)
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            bolt.write_root_pose_to_sim(st, rows)

    def _screw_reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: every bolt back to its staged hand-started pose, joints zeroed."""
        if not getattr(self, "_screw_on", False):
            return
        self.screw_turn[env_ids] = 0.0
        self._sj_coupled[env_ids] = 0.0
        self._sj_eng[env_ids] = False
        self._sj_fresh[env_ids] = True
        mask = torch.zeros_like(self._sj_eng)
        mask[env_ids] = True
        self._screw_write(mask)
        for b, bolt in enumerate(self.bolts):  # zero the (kinematic) velocities too
            bolt.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.env.device), env_ids)

    def _screw_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """The mechanic's restorable state (empty when it is off)."""
        if not getattr(self, "_screw_on", False):
            return {}
        return {
            "screw_turn": self.screw_turn[env_ids].clone(),
            "screw_coupled": self._sj_coupled[env_ids].clone(),
            "screw_engaged": self._sj_eng[env_ids].clone(),
        }

    def _screw_restore(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `_screw_state` recorded and re-write the bolts' kinematic poses from it
        (the recorded joint state is the truth; the bolts' body rows just mirror it)."""
        if not getattr(self, "_screw_on", False) or "screw_turn" not in state:
            return
        self.screw_turn[env_ids] = state["screw_turn"]
        self._sj_coupled[env_ids] = state["screw_coupled"]
        self._sj_eng[env_ids] = state["screw_engaged"]
        self._sj_fresh[env_ids] = True
        mask = torch.zeros_like(self._sj_eng)
        mask[env_ids] = True
        self._screw_write(mask)
