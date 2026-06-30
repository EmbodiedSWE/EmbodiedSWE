"""BulbAssemblyScene — a fixed lamp socket on a table and a loose bulb to screw into it.

A light bulb screws into a lamp socket. The socket (`bulb_socket.usd`) is a world-pinned fixed-base
articulation with an internal thread; the bulb (`bulb.usd`) is a free rigid body with a threaded cap
under a glass envelope. The bulb spawns cap-down and threads straight in, held by thread friction
(no weld). Mass + inertia are baked into bulb.usd — do NOT override the mass here (PhysX would recompute
inertia from the colliders and destabilise the screw). Goal (no task layer): screw each bulb down until
`seated()`. Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

    from robobench.core import BaseEnv


@dataclass
class BulbAssemblySceneCfg(BaseCfg):
    """Config for `BulbAssemblyScene`. Each field is a `tunable()` curriculum/difficulty dial or an
    `info()` structural constant (see `robobench.core.BaseCfg`); `cfg.tunables()` lists the dials.
    Nothing is locked — a curriculum/debug variant is just a `.copy()` with a few changed.
    """

    # --- tunable: the curriculum / difficulty dials -----------------------------------------------
    # A bulb is "seated" when, in a socket's frame, its origin is at/below `seat_z` above the socket origin,
    # within `align_xy` of the axis, and tilted <= `align_axis_deg` off it (any bulb may seat in any socket).
    # seat_z calibrated to the asset: seated origin ~22 mm up vs ~35 mm+ resting on the bore mouth.
    seat_z: float = tunable(0.027)  # max bulb-origin height above the socket origin (m) to count as seated
    align_xy: float = tunable(0.015)  # max lateral distance (m) from the nearest socket axis
    align_axis_deg: float = tunable(12.0)  # max tilt of the bulb's screw axis off the socket axis (deg)
    reset_pos_jitter: float = tunable(0.01)  # uniform +/- xy jitter per bulb at reset (m)
    # Part friction (static = dynamic): slick cap threads steadily, grippy socket holds.
    bulb_friction: float = tunable(0.01)
    socket_friction: float = tunable(0.75)

    # --- info: structure, reset layout, masses, asset paths (fixed) -------------------------------
    num_pairs: int = info(1)  # number of socket+bulb pairs
    # Socket xy slots, relative to the table centre. One socket per slot.
    socket_slots: tuple[tuple[float, float], ...] = info(((0.0, 0.0),))
    socket_opening_z: float = info(0.0385)  # bore-mouth height above the socket origin (m), from build_socket
    bulb_mass: float = info(0.05)  # informational; the real value (+ inertia) is baked into bulb.usd
    light_intensity: float = info(2500.0)  # the scene's dome light
    # Bulbs' start pose. Default: each bulb lying on its side (90° about x -> screw axis horizontal) in a
    # row on the +x side of the sockets, ready to be picked up. A curriculum/robot may set `bulb_init_xy`.
    bulb_init_xy: tuple[tuple[float, float], ...] = info(())  # per-bulb start xy (table-rel.); () -> the row below
    bulb_row_x0: float = info(0.13)  # x of bulb0 (the loose bulbs lie to the +x side of the sockets)
    bulb_row_y: float = info(0.0)  # y of the row
    bulb_spacing: float = info(0.12)  # x gap between adjacent bulbs (bulb k at x0 + k*spacing)
    bulb_init_z: float = info(0.024)  # [TUNE: to the asset] bulb-origin height above the surface when lying (~glass radius)
    bulb_init_quat: tuple[float, float, float, float] = info((2 ** -0.5, 2 ** -0.5, 0.0, 0.0))  # wxyz; 90° about x -> lying
    # Selectable work surface. `table` picks a preset in `TABLES`; the three fields below default to it
    # when left None/empty, or override it (e.g. raise `surface_z` so a standing robot can reach).
    table: str = info("lab_table")  # which work surface: "lab_table" | "packing"
    surface_z: float | None = info(None)  # table-top height (m); None -> the preset's
    workbench_pos: tuple[float, float] | None = info(None)  # xy the table (and sockets) sit at; None -> preset
    workbench_usd: str = info("")  # empty -> the preset's vendored USD
    # Work-surface presets (vendored under assets/props/) — same set as nut_thread.
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.55, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }
    # Bulb + socket USDs. Empty -> the packaged standalone assets under assets/bulb/.
    asset_dir: str = info("")
    bulb_usd: str = info("")
    socket_usd: str = info("")

    def __post_init__(self) -> None:
        if not self.bulb_init_xy:  # default: bulbs lying in a row to the +x side of the sockets
            self.bulb_init_xy = tuple((self.bulb_row_x0 + k * self.bulb_spacing, self.bulb_row_y) for k in range(self.num_pairs))
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "bulb")
        self.bulb_usd = self.bulb_usd or str(Path(self.asset_dir) / "bulb.usd")
        self.socket_usd = self.socket_usd or str(Path(self.asset_dir) / "bulb_socket.usd")
        # Fill the table placement from the chosen preset wherever the user left it unset.
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(assets / "props" / preset["usd"][0] / preset["usd"][1])


@SCENES.register("bulb")
class BulbAssemblyScene(BaseScene):
    cfg: BulbAssemblySceneCfg

    def __init__(self, cfg: BulbAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or BulbAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, dome light, table, and `num_pairs` fixed sockets + loose bulbs. Sockets load as fixed-base
        articulations; bulbs as free rigid bodies with the high solver iters the threaded contact needs
        and NO mass override (inertia is baked into bulb.usd)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        # Place the table so its top surface lands at `surface_z`, and sink the ground to its feet.
        table_z = c.surface_z - preset["top_offset"]
        ground_z = c.surface_z - preset["height"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd, scale=(preset["scale"],) * 3)
        if preset["kinematic"]:
            table_spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)

        out: dict[str, Any] = {
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
        }
        high_iters = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=192, solver_velocity_iteration_count=1, max_depenetration_velocity=5.0,
        )
        for i in range(c.num_pairs):
            sx, sy = c.socket_slots[i]
            # Socket: fixed-base articulation (its root_joint pins it to the world) carrying the internal
            # thread. Empty joint dicts so the default ".*" matcher skips its zero DOFs.
            out[f"socket_{i}"] = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Socket_%d" % i,
                spawn=sim_utils.UsdFileCfg(usd_path=c.socket_usd, activate_contact_sensors=True, rigid_props=high_iters),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(wx + sx, wy + sy, c.surface_z), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
                ),
                actuators={},
            )
            # Bulb: free rigid body (articulation root disabled). NO mass_props — bulb.usd bakes mass+COM+
            # inertia; overriding makes PhysX recompute it from the colliders and destabilise the screw.
            bx, by = c.bulb_init_xy[i]
            out[f"bulb_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bulb_%d" % i,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.bulb_usd,
                    activate_contact_sensors=True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                    rigid_props=high_iters,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(wx + bx, wy + by, c.surface_z + c.bulb_init_z)),
            )
        return out

    def sim_cfg(self) -> SimCfg:
        # Standard PhysX recipe + translucency (else the glass renders invisible). dt=1/240 not 1/120: the
        # fine thread tunnels at 1/120 under the gravity-driven plunge; the smaller step resolves the contact
        # so the bulb threads cleanly.
        return SimCfg(
            dt=1.0 / 240.0,
            render={"enable_translucency": True, "enable_reflections": True},
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
        """Grab handles, cache env origins, and set part friction. Called once after the build (physx ready)."""
        super().bind(env)
        self.sockets: list[Articulation] = [env.iscene[f"socket_{i}"] for i in range(self.cfg.num_pairs)]
        self.bulbs: list[RigidObject] = [env.iscene[f"bulb_{i}"] for i in range(self.cfg.num_pairs)]
        self.env_origins = env.iscene.env_origins
        # Friction (static = dynamic) on every shape, all envs: slick cap threads, grippy socket holds.
        ids = torch.arange(env.num_envs, device="cpu")
        for assets, mu in ((self.bulbs, self.cfg.bulb_friction), (self.sockets, self.cfg.socket_friction)):
            for a in assets:
                mats = a.root_physx_view.get_material_properties()
                mats[..., 0:2] = mu
                a.root_physx_view.set_material_properties(mats, ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Re-place the bulbs at their start pose (`bulb_init_xy/_z/_quat` + xy jitter); the fixed-base
        sockets stay put."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]  # (m, 3)
        wx, wy = c.workbench_pos

        quat = torch.tensor(c.bulb_init_quat, device=dev)
        for k, bulb in enumerate(self.bulbs):
            x, y = c.bulb_init_xy[k]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.tensor((wx + x, wy + y, c.surface_z + c.bulb_init_z), device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 3:7] = quat
            bulb.write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable state: world root states (13) of each socket + bulb."""
        return {
            "sockets": torch.stack([s.data.root_state_w[env_ids].clone() for s in self.sockets], dim=1),
            "bulbs": torch.stack([b.data.root_state_w[env_ids].clone() for b in self.bulbs], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore `get_state`: write each socket pose + each bulb's root state (thread friction holds it)."""
        for i, socket in enumerate(self.sockets):
            socket.write_root_pose_to_sim(state["sockets"][:, i, 0:7], env_ids)
        for i, bulb in enumerate(self.bulbs):
            bulb.write_root_state_to_sim(state["bulbs"][:, i], env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        n = c.num_pairs
        socket_word, bulb_word = ("socket", "bulb") if n == 1 else ("sockets", "bulbs")
        return (
            f"{n} lamp {socket_word} standing upright, fixed on a sturdy table, and "
            f"{n} loose light {bulb_word} lying on the table beside {'it' if n == 1 else 'them'}, "
            f"ready to be picked up and fitted. Each socket carries a real internal thread.\n"
            f"Goal: pick up {'the' if n == 1 else 'each'} bulb, set it on {'the' if n == 1 else 'a'} socket, "
            f"and screw it down (turn it clockwise while pressing down) until it seats. A seated bulb is held "
            f"by its thread. The task is complete once {'the bulb is' if n == 1 else f'all {n} bulbs are'} seated."
        )

    # ----- progress (public: seated(); reads how far the assembly has got) -------------
    def seated(self) -> torch.Tensor:
        """Whether each bulb is seated, (num_envs, num_pairs): threaded to seat depth, within align_xy of
        the nearest socket axis, and aligned within align_axis_deg. An env is assembled when all are True."""
        import math

        off = self._bulb_offsets_in_socket()  # (n, N_bulb, B_socket, 3): bulb pos in each socket's frame
        near_dist, near_socket = off[..., :2].norm(dim=-1).min(dim=-1)  # to nearest socket, and which one
        depth = torch.gather(off[..., 2], 2, near_socket.unsqueeze(-1)).squeeze(-1)  # z in the nearest socket frame
        depth_ok = depth <= self.cfg.seat_z
        axis_ok = self._bulb_axis_cos() >= math.cos(math.radians(self.cfg.align_axis_deg))
        return depth_ok & (near_dist <= self.cfg.align_xy) & axis_ok

    def _bulb_offsets_in_socket(self) -> torch.Tensor:
        """Each bulb's position in each socket's frame, (n, N_bulb, B_socket, 3): xy = offset from the
        socket axis, z = height above the socket origin (correct even if a socket is yawed)."""
        from isaaclab.utils.math import quat_apply_inverse

        sp = torch.stack([s.data.root_pos_w for s in self.sockets], dim=1)  # (n, B, 3)
        sq = torch.stack([s.data.root_quat_w for s in self.sockets], dim=1)  # (n, B, 4)
        cols = []
        for bulb in self.bulbs:  # for each bulb, its offset in every socket frame
            rel = bulb.data.root_pos_w[:, None, :] - sp  # (n, B, 3)
            cols.append(quat_apply_inverse(sq, rel))  # (n, B, 3)
        return torch.stack(cols, dim=1)  # (n, N, B, 3)

    def _bulb_axis_cos(self) -> torch.Tensor:
        """cos of the angle between each bulb's screw axis (its local +z) and the *nearest* socket's axis,
        shape (num_envs, num_pairs); 1.0 = perfectly aligned."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        sockets_up = torch.stack([quat_apply(s.data.root_quat_w, ez) for s in self.sockets], dim=1)  # (n, B, 3)
        bulbs_up = torch.stack([quat_apply(b.data.root_quat_w, ez) for b in self.bulbs], dim=1)  # (n, N, 3)
        off = self._bulb_offsets_in_socket()  # (n, N, B, 3)
        near_socket = off[..., :2].norm(dim=-1).argmin(dim=-1)  # (n, N)
        chosen_up = torch.gather(sockets_up, 1, near_socket.unsqueeze(-1).expand(-1, -1, 3))  # (n, N, 3)
        return (bulbs_up * chosen_up).sum(dim=-1)
