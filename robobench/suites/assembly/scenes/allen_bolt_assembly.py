"""AllenBoltAssemblyScene — a fixed threaded steel platform, a loose allen bolt, and an allen key.

A small steel platform (a plate on two legs) stands fixed on a table, its plate carrying an M16
threaded through-hole; a loose M16 socket-head cap screw and an L-shaped allen key rest beside it.
Goal (carried here, no task layer): stand the bolt in the hole and drive it down with the key
(clockwise while pressing) until it seats — the platform holds the internal thread and the BOLT is
the part that turns, via the key in its 14 mm hex socket.

The platform stays fixed and never moves; the M16 thread lives inside the plate's hole and is the
only part of the platform that touches the bolt shank. The bolt is a free rigid body — origin at
the thread TIP, +z up through the head, so tip depth below the plate top is the direct progress
measure, read live by `seated()`/`engaged()`. The key is a free rigid body with exact convex
hex-prism colliders (tip at origin, arm up +z, handle +x).

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg, info, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class AllenBoltAssemblySceneCfg(BaseCfg):
    """Config for `AllenBoltAssemblyScene`. Each field is a `tunable()` curriculum/difficulty dial
    or an `info()` structural constant (see `robobench.core.BaseCfg`)."""

    # --- tunable: the curriculum / difficulty dials -----------------------------------------------
    # A bolt is "seated" when — in its platform's frame — the tip is >= `seat_depth` below the plate
    # top, within `align_xy` of the hole axis, and tilted <= `align_axis_deg` off it. The head
    # bottoms out at 24.8 mm tip depth, so 22 mm separates "seated" from "merely started".
    seat_depth: float = tunable(0.022)  # min tip depth below the plate top (m) to count as seated
    align_xy: float = tunable(0.004)  # max lateral distance (m) of the bolt tip from the hole axis
    align_axis_deg: float = tunable(5.0)  # max tilt of the bolt axis off the hole axis (deg)
    reset_pos_jitter: float = tunable(0.01)  # uniform +/- xy jitter per bolt at reset (m); 0 = none
    # Part friction (static = dynamic), set on every shape at bind. The MOVING threaded part runs
    # slick (0.01) against a grippier fixed part (0.75); the key gets the proven hex-cup friction.
    bolt_friction: float = tunable(0.01)
    platform_friction: float = tunable(0.75)
    key_friction: float = tunable(0.6)
    # Weld-on-closure grasping (the benchmark's auto-weld contract, PhysX
    # form — the grasp-weld machinery at the end of this scene class):
    # close the fingers across either arm's hex and the key welds to the hand; open wide to
    # release. Gripper envs only (no-op under robot="null").
    grasp_weld: bool = tunable(True)
    grasp_weld_dist: float = tunable(0.010)  # pinch-point-to-grip-band engage radius (m)

    # --- info: structure, reset layout, masses, asset paths (fixed) -------------------------------
    num_pairs: int = info(1)  # number of platform+bolt pairs
    platform_slots: tuple[tuple[float, float], ...] = info(((0.0, 0.0),))  # platform xy, table-rel.
    # Geometry baked into the committed USD assets (defaults; keep in sync if the assets change):
    plate_top: float = info(0.038)  # plate top above the platform origin (legs 25 mm + plate 13 mm)
    thread_len: float = info(0.0248)  # bolt thread length: tip depth at which the head bottoms out
    bolt_mass: float = info(0.05)  # M16 socket-head cap screw (kg)
    light_intensity: float = info(2500.0)
    # Bolts' start pose: lying on their sides in a row on the +x side of the platforms.
    bolt_init_xy: tuple[tuple[float, float], ...] = info(())  # per-bolt start xy (table-rel.)
    bolt_row_x0: float = info(0.14)  # x of bolt0
    bolt_row_y: float = info(0.0)  # y of the row
    bolt_spacing: float = info(0.1)  # x gap between adjacent bolts
    bolt_init_z: float = info(0.013)  # [TUNE: to the asset] bolt-origin height when lying on its side
    # (lying, the bolt rests on its 15 mm head rim + 7.8 mm thread crests, axis tilted ~13 deg)
    # Allen key start pose: lying flat on the table beyond the bolts (both arms in the table plane).
    key_init_xy: tuple[tuple[float, float], ...] = info(())  # per-key start xy (table-rel.)
    key_row_x0: float = info(0.26)  # x of key0
    key_row_y: float = info(0.0)
    key_spacing: float = info(0.1)
    key_init_z: float = info(0.0075)  # resting on a hex flat (apothem 6.25 mm) + margin
    key_init_quat: tuple[float, float, float, float] = info((0.70711, 0.70711, 0.0, 0.0))  # wxyz; flat
    key_mass: float = info(0.08)  # steel 12.5 mm L-key (kg)
    key_disable_gravity: bool = info(False)  # the force-driven key smoke sets this True (no hand to bear the handle's weight)
    # Contact offsets add PER PAIR: the key<->socket clearance is 0.75 mm/side, so the key's and the
    # bolt's authored offsets together must stay well under that, or speculative contacts seal the
    # socket mouth — a key teleported INTO the recess still works (the force smoke), but a carried
    # key can never ENTER from outside. The bolt previously had no authored offset at all, and the
    # engine default on its head SDF sealed the mouth outright.
    key_contact_offset: float = info(0.0002)
    bolt_contact_offset: float = info(0.0002)
    bolt_init_quat: tuple[float, float, float, float] = info((0.70711, 0.0, 0.70711, 0.0))  # wxyz; lying
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
    bolt_usd: str = info("")
    platform_usd: str = info("")
    key_usd: str = info("")

    def __post_init__(self) -> None:
        if not self.bolt_init_xy:
            self.bolt_init_xy = tuple((self.bolt_row_x0 + k * self.bolt_spacing, self.bolt_row_y) for k in range(self.num_pairs))
        if not self.key_init_xy:
            self.key_init_xy = tuple((self.key_row_x0 + k * self.key_spacing, self.key_row_y) for k in range(self.num_pairs))
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets)
        self.bolt_usd = self.bolt_usd or str(Path(self.asset_dir) / "allen_bolt" / "allen_bolt_m16.usd")
        self.platform_usd = self.platform_usd or str(Path(self.asset_dir) / "threaded_platform" / "threaded_platform_m16.usd")
        self.key_usd = self.key_usd or str(Path(self.asset_dir) / "allen_key" / "allen_key_m16.usd")
        preset = self.TABLES[self.table]
        if self.surface_z is None:
            self.surface_z = preset["surface_z"]
        if self.workbench_pos is None:
            self.workbench_pos = preset["pos"]
        self.workbench_usd = self.workbench_usd or str(assets / "props" / preset["usd"][0] / preset["usd"][1])


@SCENES.register("allen_bolt")
class AllenBoltAssemblyScene(BaseScene):
    cfg: AllenBoltAssemblySceneCfg

    def __init__(self, cfg: AllenBoltAssemblySceneCfg | None = None) -> None:
        super().__init__(cfg or AllenBoltAssemblySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Floor, dome light, table, `num_pairs` fixed threaded platforms, and `num_pairs` loose
        bolts + keys beside them. Platforms are kinematic; bolts/keys are free rigid bodies with
        the high solver-iteration count the SDF threads need."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.bolt_usd, c.platform_usd, c.key_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — the allen-bolt assets ship with the repo under "
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
        }
        for i in range(c.num_pairs):
            px, py = c.platform_slots[i]
            # Platform: kinematic; its invisible SDF thread insert is what the bolt screws into.
            out[f"platform_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Platform_%d" % i,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.platform_usd,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(wx + px, wy + py, c.surface_z)),
            )
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
                        max_depenetration_velocity=5.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bolt_mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(wx + bx, wy + by, c.surface_z + c.bolt_init_z), rot=c.bolt_init_quat
                ),
            )
            # Key contact offset must stay well below the key<->socket clearance (0.75 mm/side) or
            # speculative contacts choke the fit.
            kx, ky = c.key_init_xy[i]
            out[f"key_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Key_%d" % i,
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
                        max_depenetration_velocity=5.0,
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
    def bind(self, env: BaseEnv) -> None:
        """Grab the platform + bolt + key handles, cache env origins, and set the part frictions."""
        super().bind(env)
        self.platforms: list[RigidObject] = [env.iscene[f"platform_{i}"] for i in range(self.cfg.num_pairs)]
        self.bolts: list[RigidObject] = [env.iscene[f"bolt_{i}"] for i in range(self.cfg.num_pairs)]
        self.keys: list[RigidObject] = [env.iscene[f"key_{i}"] for i in range(self.cfg.num_pairs)]
        self.env_origins = env.iscene.env_origins
        for bolt in self.bolts:
            self._set_friction(bolt, self.cfg.bolt_friction)
        for platform in self.platforms:
            self._set_friction(platform, self.cfg.platform_friction)
        for key in self.keys:
            self._set_friction(key, self.cfg.key_friction)
        self._grasp_weld_bind()

    def grasp_sites(self) -> list:
        """Two grip bands per key, across its hex (12.6 mm flats / 14.4 mm corners): the
        120 mm HANDLE (local +x off the elbow at z 0.050) and the 50 mm SHORT ARM (local +z;
        the crank grip). The key USD's origin is the short arm's tip."""
        sites = []
        for i, key in enumerate(self.keys):
            sites.append((f"key{i}_handle", key, (0.005, 0.0, 0.050), (0.115, 0.0, 0.050), (0.009, 0.017)))
            sites.append((f"key{i}_arm", key, (0.0, 0.0, 0.008), (0.0, 0.0, 0.045), (0.009, 0.017)))
        return sites

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Reconcile the weld-on-closure grasp contract every physics substep."""
        self._grasp_weld_step()

    def _set_friction(self, asset, value: float) -> None:
        """Overwrite the static + dynamic friction on every shape of `asset` (across all envs)."""
        mats = asset.root_physx_view.get_material_properties()
        mats[..., 0:2] = value  # [static, dynamic, restitution]
        asset.root_physx_view.set_material_properties(mats, torch.arange(self.env.num_envs, device="cpu"))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh, unassembled start: platforms pinned at spawn, bolts + keys lying on their sides
        at `*_init_xy/_z/_quat` plus jitter."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]  # (m, 3)
        wx, wy = c.workbench_pos

        for parts, xy_rows, init_z, init_quat in (
            (self.bolts, c.bolt_init_xy, c.bolt_init_z, c.bolt_init_quat),
            (self.keys, c.key_init_xy, c.key_init_z, c.key_init_quat),
        ):
            quat = torch.tensor(init_quat, device=dev)
            for k, part in enumerate(parts):
                x, y = xy_rows[k]
                st = torch.zeros(m, 13, device=dev)
                st[:, 0:3] = origin + torch.tensor((wx + x, wy + y, c.surface_z + init_z), device=dev)
                st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
                st[:, 3:7] = quat
                part.write_root_state_to_sim(st, env_ids)
        self._grasp_weld_release_all(env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable scene state: world root states (13) of each platform, bolt, and key."""
        return {
            "platforms": torch.stack([p.data.root_state_w[env_ids].clone() for p in self.platforms], dim=1),
            "bolts": torch.stack([b.data.root_state_w[env_ids].clone() for b in self.bolts], dim=1),
            "keys": torch.stack([k.data.root_state_w[env_ids].clone() for k in self.keys], dim=1),
            **self._grasp_weld_state(env_ids),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore what `get_state` returned. The bolt's threaded depth is fully captured by its
        root state, so thread friction holds it on restore."""
        for i, platform in enumerate(self.platforms):
            platform.write_root_pose_to_sim(state["platforms"][:, i, 0:7], env_ids)
        for i, bolt in enumerate(self.bolts):
            bolt.write_root_state_to_sim(state["bolts"][:, i], env_ids)
        for i, key in enumerate(self.keys):
            key.write_root_state_to_sim(state["keys"][:, i], env_ids)
        self._grasp_weld_restore(state, env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        n = c.num_pairs
        p_word, b_word = ("platform", "bolt") if n == 1 else ("platforms", "bolts")
        return (
            f"{n} small steel {p_word} standing fixed on a sturdy table, each with an M16 threaded "
            f"hole through its plate, and beside {'it' if n == 1 else 'them'}: {n} loose M16 allen "
            f"(socket-head) {b_word} and {n} L-shaped allen {'key' if n == 1 else 'keys'} lying on "
            f"the table. The hole carries real threads; the bolt head carries a 14 mm hex socket.\n"
            f"Goal: stand {'the' if n == 1 else 'each'} bolt tip-down in {'the' if n == 1 else 'a'} "
            f"hole, seat the key in its socket, and drive it down (turn clockwise while pressing) "
            f"until it seats. A seated bolt locks in place. The task is complete once "
            f"{'the bolt is' if n == 1 else f'all {n} bolts are'} seated."
            + (
                " The key holds in a firm pinch: close the fingers across either arm's hex and "
                "the grip locks; open wide to release."
                if c.grasp_weld
                else ""
            )
        )

    # ----- progress (public: seated()/engaged(); reads how far the assembly has got) -------------
    def engaged(self) -> torch.Tensor:
        """Tip depth below the plate top of the nearest platform, shape (num_envs, num_pairs), in
        metres (negative = still above the plate). The head bottoms out at `thread_len` depth."""
        off = self._bolt_offsets_in_platform()  # (n, B, P, 3)
        near = off[..., :2].norm(dim=-1).argmin(dim=-1)  # nearest platform per bolt
        z = torch.gather(off[..., 2], 2, near.unsqueeze(-1)).squeeze(-1)
        return self.cfg.plate_top - z

    def seated(self) -> torch.Tensor:
        """Whether each bolt is seated in a platform, shape (num_envs, num_pairs): threaded down to
        `seat_depth`, within `align_xy` of the hole axis, and within `align_axis_deg` of upright."""
        import math

        off = self._bolt_offsets_in_platform()  # (n, B, P, 3)
        near_dist, near = off[..., :2].norm(dim=-1).min(dim=-1)
        z = torch.gather(off[..., 2], 2, near.unsqueeze(-1)).squeeze(-1)
        depth_ok = (self.cfg.plate_top - z) >= self.cfg.seat_depth
        axis_ok = self._bolt_axis_cos() >= math.cos(math.radians(self.cfg.align_axis_deg))
        return depth_ok & (near_dist <= self.cfg.align_xy) & axis_ok

    def _bolt_offsets_in_platform(self) -> torch.Tensor:
        """Each bolt's position in each platform's local frame, shape (num_envs, num_bolts,
        num_platforms, 3): xy = lateral offset from the hole axis, z = tip height above the
        platform origin (the bolt origin IS the tip)."""
        from isaaclab.utils.math import quat_apply_inverse

        pp = torch.stack([p.data.root_pos_w for p in self.platforms], dim=1)  # (n, P, 3)
        pq = torch.stack([p.data.root_quat_w for p in self.platforms], dim=1)  # (n, P, 4)
        cols = []
        for bolt in self.bolts:
            rel = bolt.data.root_pos_w[:, None, :] - pp  # (n, P, 3)
            cols.append(quat_apply_inverse(pq, rel))
        return torch.stack(cols, dim=1)  # (n, B, P, 3)

    def _bolt_axis_cos(self) -> torch.Tensor:
        """cos of the angle between each bolt's screw axis (local +z) and the nearest platform's
        hole axis (its local +z), shape (num_envs, num_pairs)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        plat_up = torch.stack([quat_apply(p.data.root_quat_w, ez) for p in self.platforms], dim=1)  # (n, P, 3)
        bolt_up = torch.stack([quat_apply(b.data.root_quat_w, ez) for b in self.bolts], dim=1)  # (n, B, 3)
        off = self._bolt_offsets_in_platform()
        near = off[..., :2].norm(dim=-1).argmin(dim=-1)  # (n, B)
        chosen_up = torch.gather(plat_up, 1, near.unsqueeze(-1).expand(-1, -1, 3))  # (n, B, 3)
        return (bolt_up * chosen_up).sum(dim=-1)

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
