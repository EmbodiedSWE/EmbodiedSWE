"""NutThreadAssemblyScene — a fixed bolt on a table and a loose nut to thread onto it.

A single M16 bolt stands fixed and upright on a table; a single M16 nut rests loose beside it. Goal
(carried here, no task layer): pick the nut up, set it on the bolt, and screw it down until it seats.
Generalises to `num_pairs` bolt+nut pairs (default 1); a nut may seat on any bolt (order-independent).

The bolt is a fixed-base articulation (pinned to the world), so it never moves and carries the baked
SDF threads. The nut is a free rigid body: it threads down by real thread contact and is held by
thread friction — no weld needed. Progress is read live by `seated()`. Embodiment-agnostic: the nut
is a scene object the robot reaches through `env.scene`.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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
class NutThreadAssemblySceneCfg(BaseCfg):
    """Config for `NutThreadAssemblyScene`. Each field is a `tunable()` curriculum/difficulty dial
    or an `info()` structural constant (see `robobench.core.BaseCfg`); `cfg.tunables()` lists the
    dials. Nothing is locked — a curriculum/debug variant is just a `.copy()` with a few changed.
    """

    # --- tunable: the curriculum / difficulty dials -----------------------------------------------
    # A nut is "seated on a bolt" when — all measured in that bolt's own frame — it is threaded down to
    # at/below `seat_z` above the bolt origin, within `align_xy` of the bolt axis, and tilted
    # <= `align_axis_deg` off it (order-independent: any nut may seat on any bolt).
    seat_z: float = tunable(0.012)  # max nut-origin height above the bolt origin (m) to count as seated.
    # Calibrated to the asset (the nut USD's origin sits 10 mm below its bottom face): 12 mm puts the
    # nut's top face at/below the bolt's thread top — fully threaded on. Resting on the bolt top is
    # ~25 mm, true bottom-out ~1 mm; gripper pads graze the bolt head below ~4 mm.
    align_xy: float = tunable(0.015)  # max lateral distance (m) of the nut from the nearest bolt axis
    align_axis_deg: float = tunable(10.0)  # max tilt of the nut's screw axis off the bolt axis (deg)
    reset_pos_jitter: float = tunable(0.01)  # uniform +/- xy jitter per nut at reset (m); 0 = none
    # Part friction (static = dynamic), applied to every shape at bind.
    nut_friction: float = tunable(0.01)
    bolt_friction: float = tunable(0.75)

    # --- info: structure, reset layout, masses, asset paths (fixed) -------------------------------
    num_pairs: int = info(1)  # number of bolt+nut pairs
    # Bolt xy slots, relative to the table centre. One bolt per slot.
    bolt_slots: tuple[tuple[float, float], ...] = info(((0.0, 0.0),))
    bolt_height: float = info(0.025)  # informational; the real value is baked into the bolt USD
    nut_mass: float = info(0.03)  # M16 nut mass (kg)
    light_intensity: float = info(2500.0)
    # Nuts' start pose. Default: each nut resting flat (identity quat -> screw axis up) in a row on the
    # +x side of the bolts. A curriculum/robot may set `nut_init_xy` to override the row.
    nut_init_xy: tuple[tuple[float, float], ...] = info(())  # per-nut start xy (table-rel.); () -> the row below
    nut_row_x0: float = info(0.12)  # x of nut0 (the loose nuts sit to the +x side of the bolts)
    nut_row_y: float = info(0.0)  # y of the row
    nut_spacing: float = info(0.1)  # x gap between adjacent nuts (nut k at x0 + k*spacing)
    nut_init_z: float = info(0.02)  # [TUNE: to the asset] nut-origin height above the surface when resting flat
    nut_init_quat: tuple[float, float, float, float] = info((1.0, 0.0, 0.0, 0.0))  # wxyz; identity -> axis up
    # Selectable work surface. `table` picks a preset in `TABLES`; the three fields below default to it
    # when left None/empty, or override it (e.g. raise `surface_z` so a standing robot can reach).
    table: str = info("lab_table")  # which work surface: "lab_table" | "packing"
    surface_z: float | None = info(None)  # table-top height (m); None -> the preset's
    workbench_pos: tuple[float, float] | None = info(None)  # xy the table (and bolts) sit at; None -> preset
    workbench_usd: str = info("")  # empty -> the preset's vendored USD
    # Work-surface presets (vendored under assets/props/). Per table: usd (subdir, file), scale, orient
    # (wxyz), surface_z (top height) + pos (xy) defaults, top_offset (top above the USD origin), height
    # (top->feet, sinks the ground to the table's feet), kinematic (load as a fixed rigid body).
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.5, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }
    # Bolt + nut USDs. Empty -> the packaged M16 bolt (with head) and M16 nut under assets/factory/.
    asset_dir: str = info("")
    bolt_usd: str = info("")
    nut_usd: str = info("")

    def __post_init__(self) -> None:
        if not self.nut_init_xy:  # default: nuts resting in a row to the +x side of the bolts
            self.nut_init_xy = tuple((self.nut_row_x0 + k * self.nut_spacing, self.nut_row_y) for k in range(self.num_pairs))
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets / "factory")
        self.bolt_usd = self.bolt_usd or str(Path(self.asset_dir) / "factory_bolt_m16.usd")
        self.nut_usd = self.nut_usd or str(Path(self.asset_dir) / "factory_nut_m16.usd")
        # Fill the table placement from the chosen preset wherever the user left it unset.
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(assets / "props" / preset["usd"][0] / preset["usd"][1])


@SCENES.register("nut_thread")
class NutThreadAssemblyScene(BaseScene):
    cfg: NutThreadAssemblySceneCfg

    def __init__(self, cfg: NutThreadAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or NutThreadAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, a dome light, the table, `num_pairs` fixed bolts standing on it, and `num_pairs` loose
        nuts resting beside them. Each bolt loads as a fixed-base articulation; each nut as a free,
        contact-rich rigid body with the high solver-iteration count the SDF threads need."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        preset = c.TABLES[c.table]
        wx, wy = c.workbench_pos
        # Place the table so its top surface lands at `surface_z` (the preset's `top_offset` accounts for
        # tables whose origin is at the base), and sink the ground to its feet so it rests on the floor.
        table_z = c.surface_z - preset["top_offset"]
        ground_z = c.surface_z - preset["height"]
        table_spawn = sim_utils.UsdFileCfg(usd_path=c.workbench_usd, scale=(preset["scale"],) * 3)
        if preset["kinematic"]:  # load the table as a fixed (kinematic) rigid body
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
        }
        for i in range(c.num_pairs):
            bx, by = c.bolt_slots[i]
            # Bolt: fixed-base articulation (the USD's root_joint pins it to the world). Empty joint
            # dicts so the default ".*" matcher doesn't run on its zero DOFs. High solver-iteration
            # counts + the baked SDF threads are what the nut physically screws into.
            out[f"bolt_{i}"] = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Bolt_%d" % i,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.bolt_usd,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=192,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=5.0,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(wx + bx, wy + by, c.surface_z), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
                ),
                actuators={},
            )
            # Nut: free rigid body. Disabling its USD's articulation root loads it as a plain free body
            # the robot can grip/turn.
            nx, ny = c.nut_init_xy[i]
            out[f"nut_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Nut_%d" % i,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.nut_usd,
                    activate_contact_sensors=True,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=192,
                        solver_velocity_iteration_count=1,
                        max_depenetration_velocity=5.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.nut_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(wx + nx, wy + ny, c.surface_z + c.nut_init_z)),
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
                # 2**29: at 512 envs the threaded contact demanded ~269 MB, just past the
                # previous 2**28 stack — PhysX dropped contacts rather than failing loudly
                # (measured 2026-08-01, same overflow as allen_bolt at ~437 MB).
                "gpu_collision_stack_size": 2**29,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab the bolt + nut handles, cache env origins, and set the part frictions. Called once
        after the scene is built (the sim is already playing, so the physx views are ready)."""
        super().bind(env)
        self.bolts: list[Articulation] = [env.iscene[f"bolt_{i}"] for i in range(self.cfg.num_pairs)]
        self.nuts: list[RigidObject] = [env.iscene[f"nut_{i}"] for i in range(self.cfg.num_pairs)]
        self.env_origins = env.iscene.env_origins
        for nut in self.nuts:
            self._set_friction(nut, self.cfg.nut_friction)
        for bolt in self.bolts:
            self._set_friction(bolt, self.cfg.bolt_friction)

    def _set_friction(self, asset, value: float) -> None:
        """Overwrite the static + dynamic friction on every shape of `asset` (across all envs)."""
        mats = asset.root_physx_view.get_material_properties()
        mats[..., 0:2] = value  # [static, dynamic, restitution]
        asset.root_physx_view.set_material_properties(mats, torch.arange(self.env.num_envs, device="cpu"))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start: the bolts stand upright on the table and the nuts rest flat on it
        beside them. The bolts are fixed-base (pinned at spawn), so only the nuts are re-placed; their
        pose comes from `nut_init_xy/_z/_quat` (table-relative) plus a small xy jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]  # (m, 3)
        wx, wy = c.workbench_pos

        quat = torch.tensor(c.nut_init_quat, device=dev)
        for k, nut in enumerate(self.nuts):
            x, y = c.nut_init_xy[k]
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.tensor((wx + x, wy + y, c.surface_z + c.nut_init_z), device=dev)
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
            st[:, 3:7] = quat
            nut.write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable scene state for `env_ids`: world root states (pos+quat+lin/ang vel, 13) of each
        bolt and each nut. (Bolts are fixed, but capturing them keeps the snapshot self-describing.)"""
        return {
            "bolts": torch.stack([bolt.data.root_state_w[env_ids].clone() for bolt in self.bolts], dim=1),
            "nuts": torch.stack([nut.data.root_state_w[env_ids].clone() for nut in self.nuts], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned: write each bolt pose and each nut's full root state. The
        nut's threaded height is fully captured by its root state, so thread friction holds it on restore."""
        for i, bolt in enumerate(self.bolts):
            bolt.write_root_pose_to_sim(state["bolts"][:, i, 0:7], env_ids)
        for i, nut in enumerate(self.nuts):
            nut.write_root_state_to_sim(state["nuts"][:, i], env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        n = c.num_pairs
        bolt_word, nut_word = ("bolt", "nut") if n == 1 else ("bolts", "nuts")
        return (
            f"{n} M16 {bolt_word} standing upright, fixed on a sturdy table, and "
            f"{n} loose M16 {nut_word} resting flat on the table beside {'it' if n == 1 else 'them'}, "
            f"ready to be picked up and fitted. Each bolt carries real threads.\n"
            f"Goal: pick up {'the' if n == 1 else 'each'} nut, set it on {'the' if n == 1 else 'a'} bolt, "
            f"and screw it down (turn it clockwise while pressing down) until it seats. A seated nut locks "
            f"in place. The task is complete once {'the nut is' if n == 1 else f'all {n} nuts are'} seated."
        )

    # ----- progress (public: seated(); reads how far the assembly has got) -------------
    def seated(self) -> torch.Tensor:
        """Whether each nut is seated on a bolt, shape (num_envs, num_pairs). In a bolt's own frame the
        nut must be: threaded down to seat depth, within `align_xy` of *some* bolt axis, and its screw
        axis within `align_axis_deg` of the bolt axis. Order-independent — a nut may seat on ANY bolt.
        ('Seated' = a part has reached its final resting position.) An env is assembled when every entry
        of its row is True."""
        import math

        off = self._nut_offsets_in_bolt()  # (n, npairs_nut, npairs_bolt, 3): nut pos in each bolt's frame
        near_dist, near_bolt = off[..., :2].norm(dim=-1).min(dim=-1)  # to nearest bolt, and which one
        # depth of the nut in its nearest bolt's frame (gather the z of the chosen bolt)
        depth = torch.gather(off[..., 2], 2, near_bolt.unsqueeze(-1)).squeeze(-1)
        depth_ok = depth <= self.cfg.seat_z
        axis_ok = self._nut_axis_cos() >= math.cos(math.radians(self.cfg.align_axis_deg))
        return depth_ok & (near_dist <= self.cfg.align_xy) & axis_ok

    def success(self) -> torch.Tensor:
        """(N,) bool: every nut seated on a bolt — this scene's assembled state
        (scene-level success alias, matching the other suites' surface)."""
        return self.seated().all(dim=1)

    def _nut_offsets_in_bolt(self) -> torch.Tensor:
        """Each nut's position in each bolt's local frame, shape (num_envs, num_pairs_nut, num_pairs_bolt,
        3): xy = lateral offset from that bolt's axis, z = height above that bolt's origin. Measured in
        the bolt frame, so it is correct even if a bolt is placed with a yaw."""
        from isaaclab.utils.math import quat_apply_inverse

        bp = torch.stack([b.data.root_pos_w for b in self.bolts], dim=1)  # (n, B, 3)
        bq = torch.stack([b.data.root_quat_w for b in self.bolts], dim=1)  # (n, B, 4)
        cols = []
        for nut in self.nuts:  # for each nut, its offset in every bolt frame
            rel = nut.data.root_pos_w[:, None, :] - bp  # (n, B, 3)
            cols.append(quat_apply_inverse(bq, rel))  # (n, B, 3)
        return torch.stack(cols, dim=1)  # (n, N, B, 3)

    def _nut_axis_cos(self) -> torch.Tensor:
        """cos of the angle between each nut's screw axis (its local +z) and the *nearest* bolt's axis,
        shape (num_envs, num_pairs); 1.0 = perfectly aligned. Bolts stand on the bench, so their axes
        are world-up unless a bolt is placed with a tilt — we compare to the actual bolt axis to stay
        general."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        bolts_up = torch.stack([quat_apply(b.data.root_quat_w, ez) for b in self.bolts], dim=1)  # (n, B, 3)
        nuts_up = torch.stack([quat_apply(nut.data.root_quat_w, ez) for nut in self.nuts], dim=1)  # (n, N, 3)
        # nearest bolt per nut (by lateral xy), then cos against that bolt's axis
        off = self._nut_offsets_in_bolt()  # (n, N, B, 3)
        near_bolt = off[..., :2].norm(dim=-1).argmin(dim=-1)  # (n, N)
        chosen_up = torch.gather(bolts_up, 1, near_bolt.unsqueeze(-1).expand(-1, -1, 3))  # (n, N, 3)
        return (nuts_up * chosen_up).sum(dim=-1)
