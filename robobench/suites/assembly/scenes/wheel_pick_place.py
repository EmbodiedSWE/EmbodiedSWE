"""WheelPickPlaceScene — a steering wheel lying on a packing table, to be placed in the basket beside it.

The set piece is one asset: a heavy-duty packing table (top 0.694 m, crates and corrugated boxes on
its under-shelf) with a `container_h20` plastic basket standing open on the table top. The steering
wheel lies flat on the same table top, on the far side from the basket. Goal (carried here — no task
layer): pick the wheel up off the table and place it, settled, inside the basket.

**Nothing about the robot is in here** — the scene is embodiment-agnostic, as every `BaseScene` is.

GEOMETRY. The table set piece spawns at `workbench_pos` = (0, 0.55) with its own frame putting the
table top at z = 0.994, so `surface_z` = 0.694 places that top in the world and `table_z` derives the
asset origin from it. The wheel starts at (-0.35, 0.45) with `wheel_init_z` = 5.6 mm of drop
clearance above the top, and settles onto it. The target zone (`place_box_x/_y`, `place_max_h`) is
quoted in the TABLE's frame, so moving `workbench_pos` moves it along with the basket it is an inset
of. Asset provenance and the re-download recipe live in `assets/fetch_pick_place_assets.py`.

The ground plane stays at z = 0, which is where a floor-standing embodiment's feet belong; the
table's bottom 0.3 m is therefore below it. Raise `ground_z` to -0.3 to stand the table on its own
feet instead (the table top, and everything the task touches, does not move).

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class WheelPickPlaceSceneCfg(BaseCfg):
    """Config for `WheelPickPlaceScene`. Nothing is locked — a variant is just a copy with a few
    fields changed."""

    # --- the curriculum / difficulty dials -----------------------------------------------
    # The wheel is "placed" when its origin sits inside the basket box — `place_box_x/_y` in the
    # TABLE's frame, no higher than `place_max_h` above the table top — and it has settled to under
    # `settle_vel` on every axis. The box is an inset within the basket's 0.70 x 0.46 m interior, so
    # it is a genuine "in the bin", not "near it"; at the default table placement it works out to
    # world x(0.40, 0.85), y(0.35, 0.60), z < 1.10.
    place_box_x: tuple[float, float] = (0.40, 0.85)  # table-frame x band of the target zone (m)
    place_box_y: tuple[float, float] = (-0.20, 0.05)  # table-frame y band (m)
    place_max_h: float = 0.406  # max wheel-origin height above the table top (m): well clear of a
    # carried wheel, so "low enough to be in the basket, not held over it"
    settle_vel: float = 0.20  # max |v| per axis (m/s, and rad/s) to count as come-to-rest
    # The wheel has left the table for the floor once its origin falls this far below the table top.
    drop_below_surface: float = 0.194
    reset_pos_jitter: float = 0.0  # uniform +/- xy jitter for the wheel at reset (m). The default of
    # 0 keeps the layout deterministic and bit-reproducible; raise it for a randomized batch.
    # Part friction (static = dynamic), written across every wheel shape at bind. steering_wheel.usd
    # authors three physics materials (leather 0.7 / plastic 0.5 / metal 0.4); this single dial
    # flattens them to one value, and the nominal is the LEATHER rim, so the contact an end effector
    # meets on the outside of the wheel is unchanged from the asset and only the hub/spokes (which
    # just rest on the table) get grippier.
    wheel_friction: float = 0.7

    # --- structure, reset layout, asset paths (fixed) ---------------------------------------
    # The packing-table set piece: one kinematic asset carrying the table, the basket, and the
    # under-shelf crates. Its own frame puts the table top at TABLE_TOP_Z and its feet at 0.
    TABLE_TOP_Z: ClassVar[float] = 0.994  # table-top height in the asset's own frame (m)
    #: Basket (`container_h20`) geometry in the table asset's frame, for `describe()` and for anyone
    #: aiming at it: interior floor top, wall top, and the interior xy extent between the walls.
    BASKET_FLOOR_Z: ClassVar[float] = 1.0011
    BASKET_RIM_Z: ClassVar[float] = 1.0826
    BASKET_INNER_X: ClassVar[tuple[float, float]] = (0.2736, 0.9756)
    BASKET_INNER_Y: ClassVar[tuple[float, float]] = (-0.3254, 0.1366)

    surface_z: float = 0.694  # world height of the table top (asset origin then lands at z = -0.3)
    workbench_pos: tuple[float, float] = (0.0, 0.55)  # xy the table set piece sits at
    ground_z: float = 0.0  # world height of the ground plane (see the module docstring)
    wheel_scale: float = 0.75  # the asset is 0.381 m across at 1.0 -> a 0.286 m rim (0.143 m radius)
    wheel_init_xy: tuple[float, float] = (-0.35, 0.45)  # world xy the wheel lies at
    wheel_init_z: float = 0.0056  # wheel-origin drop height above the table top (m)
    wheel_init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz; flat on the table
    light_intensity: float = 3000.0  # the scene's dome light
    light_color: tuple[float, float, float] = (0.75, 0.75, 0.75)
    # Asset USDs; empty -> the vendored trees under `assets/` (see assets/fetch_pick_place_assets.py).
    asset_dir: str = ""
    wheel_usd: str = ""
    workbench_usd: str = ""

    def __post_init__(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "assets"
        self.asset_dir = self.asset_dir or str(assets)
        self.wheel_usd = self.wheel_usd or str(Path(self.asset_dir) / "steering_wheel" / "steering_wheel.usd")
        self.workbench_usd = self.workbench_usd or str(
            Path(self.asset_dir) / "props" / "packing_table_scene" / "packing_table.usd")

    @property
    def table_z(self) -> float:
        """World z the table asset's origin spawns at, so its top lands on `surface_z`."""
        return self.surface_z - self.TABLE_TOP_Z

    def place_box_world(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        """The target zone in ENV-LOCAL world coords: (x band, y band, max wheel height) — the
        table-frame bands above, shifted onto the table's placement."""
        wx, wy = self.workbench_pos
        return ((self.place_box_x[0] + wx, self.place_box_x[1] + wx),
                (self.place_box_y[0] + wy, self.place_box_y[1] + wy),
                self.surface_z + self.place_max_h)


@SCENES.register("wheel_pick_place")
class WheelPickPlaceScene(BaseScene):
    cfg: WheelPickPlaceSceneCfg

    #: Per-env world variation at generation (engine sampler). Only the wheel is a live rigid body
    #: with a PhysX view here — the table set piece is a static spawn — so friction is the one
    #: physical knob. Band widened both ways around the leather nominal.
    PHYSICAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "wheel_friction": {"dist": "uniform", "lo": 0.5, "hi": 0.9,
                           "reason": "around the asset's 0.7 leather rim — the gripped surface"},
    }

    #: Stage-wide look knobs for visual replay (data_engine render.py --visual_draw); the cfg
    #: defaults are the nominal look.
    VISUAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "light_intensity": {"dist": "uniform", "lo": 2500.0, "hi": 3500.0,
                            "reason": "around the 3000 nominal"},
    }

    #: L5 external views (see BaseScene.CAMERAS). `front` stands off to one side of the table at
    #: standing height, framing the whole work surface — wheel, table top and basket — in one shot.
    CAMERAS: ClassVar[dict[str, dict]] = {
        "front": {"eye": (1.60, -0.40, 1.35), "target": (-0.10, 0.50, 0.70), "focal": 20.0,
                  "bands": {
                      "eye_x": {"dist": "uniform", "lo": 1.50, "hi": 1.70},
                      "eye_y": {"dist": "uniform", "lo": -0.50, "hi": -0.30},
                      "eye_z": {"dist": "uniform", "lo": 1.30, "hi": 1.42},
                  }},
    }

    def __init__(self, cfg: WheelPickPlaceSceneCfg | None = None) -> None:
        super().__init__(cfg or WheelPickPlaceSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, dome light, the packing-table set piece (kinematic — table + basket + crates in
        one asset), and the loose steering wheel. The wheel keeps the asset's own convex-decomposition
        colliders and density-derived mass; only its friction is overridden (at bind)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.wheel_usd, c.workbench_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — re-vendor with "
                    f"`python robobench/suites/assembly/assets/fetch_pick_place_assets.py`")
        wx, wy = c.workbench_pos
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(__file__).resolve().parents[1] / "assets" / "props" / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, c.ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=c.light_color),
            ),
            # The set piece is spawned kinematic: the table's mesh collider and the
            # basket's five box colliders are then immovable furniture the wheel can rest against.
            "workbench": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(wx, wy, c.table_z), rot=(1.0, 0.0, 0.0, 0.0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.workbench_usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
            ),
            "wheel": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Wheel",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.wheel_usd,
                    scale=(c.wheel_scale,) * 3,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.wheel_init_xy[0], c.wheel_init_xy[1], c.surface_z + c.wheel_init_z),
                    rot=c.wheel_init_quat,
                ),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        # dt = 1/200 divides evenly into the humanoid control periods this scene is sized for (a
        # 0.02 s period is exactly 4 physics steps), so a 50 Hz control loop lands on whole steps. The
        # GPU buffers are widened for the wheel's convex-decomposition shell against the basket walls.
        return SimCfg(
            dt=1.0 / 200.0,
            render={"enable_reflections": True},
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def apply_visual_params(self, env: BaseEnv, values: dict[str, Any]) -> None:
        """The live subset of `VISUAL_PARAMS` (see BaseScene): the dome intensity is an attribute
        write (also reaching the light at build via the cfg, so this is redundant-but-harmless on a
        fresh build)."""
        unknown = set(values) - set(self.VISUAL_PARAMS)
        if unknown:
            raise ValueError(f"{type(self).__name__} cannot apply visuals: {sorted(unknown)}")
        if "light_intensity" in values:
            env.stage.GetPrimAtPath("/World/light").GetAttribute("inputs:intensity").Set(
                float(values["light_intensity"]))

    def apply_physical_params(self, env: BaseEnv, values: dict[str, list]) -> None:
        """Write the wheel's friction PER ENV (static = dynamic), `values[name]` one value per env.
        `bind()` routes the nominal through here with uniform values, so this is THE friction path —
        per-env sampling reuses it, never a copy."""
        unknown = set(values) - set(self.PHYSICAL_PARAMS)
        if unknown:
            raise ValueError(f"{type(self).__name__} cannot apply per-env: {sorted(unknown)}")
        if "wheel_friction" in values:
            mats = self.wheel.root_physx_view.get_material_properties()  # (n, n_shapes, 3)
            mats[..., 0:2] = torch.tensor(values["wheel_friction"], dtype=torch.float32).view(-1, 1, 1)
            self.wheel.root_physx_view.set_material_properties(mats, torch.arange(env.num_envs, device="cpu"))

    def bind(self, env: BaseEnv) -> None:
        """Grab the wheel handle, cache env origins, and set its friction. Once, after the build."""
        super().bind(env)
        self.wheel: RigidObject = env.iscene["wheel"]
        self.env_origins = env.iscene.env_origins
        self.apply_physical_params(env, {n: [getattr(self.cfg, n)] * env.num_envs for n in self.PHYSICAL_PARAMS})

    def reset(self, env_ids: torch.Tensor) -> None:
        """Re-place the wheel flat on the table at its start pose (+ optional xy jitter), at rest.
        The table set piece is a static spawn and never moves."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(
            (c.wheel_init_xy[0], c.wheel_init_xy[1], c.surface_z + c.wheel_init_z), device=dev)
        if c.reset_pos_jitter:
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 3:7] = torch.tensor(c.wheel_init_quat, device=dev)
        self.wheel.write_root_state_to_sim(st, env_ids)

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable state: the wheel's world root state (13). Everything else in the scene is
        static furniture."""
        return {"wheel": self.wheel.data.root_state_w[env_ids].clone()}

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore `get_state`: write the wheel's root state back."""
        self.wheel.write_root_state_to_sim(state["wheel"], env_ids)

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        d = 2 * 0.1905 * c.wheel_scale  # the asset is 0.381 m across at scale 1
        inner_x = c.BASKET_INNER_X[1] - c.BASKET_INNER_X[0]
        inner_y = c.BASKET_INNER_Y[1] - c.BASKET_INNER_Y[0]
        rim = c.surface_z + (c.BASKET_RIM_Z - c.TABLE_TOP_Z)
        return (
            f"A heavy-duty packing table, its top {c.surface_z:.2f} m off the floor, with a car "
            f"steering wheel about {d * 100:.0f} cm across lying flat on it, hub up, off to one side. "
            f"An open plastic basket stands on the same table top on the OTHER side, its rim "
            f"{rim:.2f} m up and its interior about {inner_x * 100:.0f} x {inner_y * 100:.0f} cm — "
            f"roomy enough to take the wheel lying flat.\n"
            f"Goal: pick the steering wheel up off the table and put it in the basket. The task is "
            f"complete once the wheel is resting inside the basket, released and come to rest — not "
            f"still held, not balanced on the rim."
        )

    # ----- progress (public: placed(); reads how far the task has got) --------------------------
    def placed(self) -> torch.Tensor:
        """Whether the wheel is placed, `(num_envs,)`: its origin inside the basket box (`place_box_x/_y`
        in the table frame, below `place_max_h` above the table top) and settled to under `settle_vel`
        on every axis. Embodiment-agnostic by construction: it reads the wheel and the basket only,
        never the robot."""
        (x0, x1), (y0, y1), max_z = self.cfg.place_box_world()
        p = self.wheel.data.root_pos_w - self.env_origins  # env-local, like every target here
        inside = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & (p[:, 2] < max_z)
        return inside & self.settled()

    def settled(self) -> torch.Tensor:
        """Whether the wheel has come to rest, `(num_envs,)`: every component of its linear AND
        angular velocity under `settle_vel`."""
        return self.wheel.data.root_vel_w.abs().amax(dim=-1) < self.cfg.settle_vel

    def dropped(self) -> torch.Tensor:
        """Whether the wheel has fallen off the table, `(num_envs,)` — a readable signal, not a
        forced reset: this scene never terminates an episode on its own."""
        z = self.wheel.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return z < self.cfg.surface_z - self.cfg.drop_below_surface

    def wheel_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The wheel's ENV-LOCAL position `(num_envs, 3)` and orientation `(num_envs, 4)` wxyz — the
        frame every target in this scene is quoted in."""
        return self.wheel.data.root_pos_w - self.env_origins, self.wheel.data.root_quat_w
