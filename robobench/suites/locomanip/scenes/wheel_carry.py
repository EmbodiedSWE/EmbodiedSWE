"""WheelCarryScene — a steering wheel picked off one packing table and carried to a basket on another.

Two work stations, three metres apart, and a wheel that has to travel between them. That distance is
the whole point: it is past any fixed-base humanoid's reach, so the task cannot be solved without the
robot moving its own base. Goal (carried here — no task layer): pick the wheel up off the PICK table,
carry it to the PLACE table, and put it in the basket standing there.

**Nothing about the robot is in here** — the scene is embodiment-agnostic, as every `BaseScene` is.
It never mentions legs, walking, or a base command; it only puts the two stations far enough apart
that an embodiment without those cannot finish.

LINEAGE. This is `assembly.wheel_pick_place` with the basket taken three metres away, and the
numbers are deliberately inherited rather than re-derived: same steering wheel at the same
table-frame spot, same 0.694 m surface, same basket target bands, same friction band and sim
substrate. Each station reproduces `wheel_pick_place`'s validated standing geometry exactly — a
robot at the station's `stand_xy` sees precisely the layout it saw there — so any failure here is
attributable to the transit, not to a new reach problem.

GEOMETRY. Both tables are spawned at yaw 0 and differ only in x, `carry_dx` apart, which is what
keeps the target bands a plain offset instead of a rotation. The PICK station is the bare heavy-duty
table (no basket); the PLACE station is the full packing-table set piece, whose `container_h20`
basket is the target — so there is exactly ONE basket in the world and no ambiguity about which.
Station frames put their tops at `surface_z`; `table_z`/`scene_table_z` derive each asset's origin
from that, because the two USDs are authored at different scales (the bare table is in centimetres).

The ground plane stays at z = 0, which is where a floor-standing embodiment's feet belong, so the
tables' bottom 0.3 m is below it. That is what Isaac Lab's own G1 loco-manipulation env ships
(`packing_table` at z = -0.3 over a ground plane at 0), and it is what makes a 0.694 m work surface
reachable by a humanoid whose pelvis stands at 0.75. Raise `ground_z` to -0.3 to stand the tables on
their own feet instead — the work surfaces do not move, but they become a 0.994 m chest-height pick
and the inherited reach calibration no longer applies.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from robobench.core.assets import asset_path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class WheelCarrySceneCfg(BaseCfg):
    """Config for `WheelCarryScene`. Nothing is locked — a variant is just a copy with a few fields
    changed. `carry_dx` is the difficulty dial: shrink it toward 0 and the task degenerates into
    `assembly.wheel_pick_place`; grow it and the transit dominates."""

    # --- the curriculum / difficulty dials -----------------------------------------------
    #: How far the PLACE station sits from the PICK station along +x (m). The default clears both
    #: 2.47 m tables with about half a metre between their ends, and is comfortably past any
    #: fixed-base reach — which is what makes this a locomotion task rather than a long stretch.
    carry_dx: float = 3.0
    # The wheel is "placed" when its origin sits inside the basket box — `place_box_x/_y` in the
    # PLACE TABLE's frame, no higher than `place_max_h` above the table top — and it has settled to
    # under `settle_vel` on every axis. The box is an inset within the basket's 0.70 x 0.46 m
    # interior, so it is a genuine "in the bin", not "near it". Inherited verbatim from
    # `assembly.wheel_pick_place`, where they were measured.
    place_box_x: tuple[float, float] = (0.40, 0.85)  # table-frame x band of the target zone (m)
    place_box_y: tuple[float, float] = (-0.20, 0.05)  # table-frame y band (m)
    place_max_h: float = 0.406  # max wheel-origin height above the table top (m)
    settle_vel: float = 0.20  # max |v| per axis (m/s, and rad/s) to count as come-to-rest
    # The wheel has left a table for the floor once its origin falls this far below the tops.
    drop_below_surface: float = 0.194
    #: Sustained-lift gate for the `lifted` latch: the wheel origin this far above the PICK table top
    #: (m). Above the drop threshold and well above resting noise, so brushing it does not latch.
    lift_h: float = 0.05
    reset_pos_jitter: float = 0.0  # uniform +/- xy jitter for the wheel at reset (m). The default of
    # 0 keeps the layout deterministic and bit-reproducible; raise it for a randomized batch.
    # Part friction (static = dynamic), written across every wheel shape at bind. steering_wheel.usd
    # authors three physics materials (leather 0.7 / plastic 0.5 / metal 0.4); this single dial
    # flattens them to one value, and the nominal is the LEATHER rim, so the contact an end effector
    # meets on the outside of the wheel is unchanged from the asset.
    wheel_friction: float = 0.7

    # --- structure, reset layout, asset paths (fixed) ---------------------------------------
    #: Table-top height in each asset's OWN frame (m). Both stations are the same
    #: `SM_HeavyDutyPackingTable_C02_01` mesh, but the set piece is authored in metres and the bare
    #: table in centimetres (hence `bare_scale`), so they share this number and not their origins.
    TABLE_TOP_Z: ClassVar[float] = 0.994
    #: Basket (`container_h20`) geometry in the PLACE table asset's frame, for `describe()` and for
    #: anyone aiming at it: interior floor top, wall top, and the interior xy extent between walls.
    BASKET_FLOOR_Z: ClassVar[float] = 1.0011
    BASKET_RIM_Z: ClassVar[float] = 1.0826
    BASKET_INNER_X: ClassVar[tuple[float, float]] = (0.2736, 0.9756)
    BASKET_INNER_Y: ClassVar[tuple[float, float]] = (-0.3254, 0.1366)
    #: The bare table USD is authored in centimetres; the set piece in metres.
    bare_scale: float = 0.01

    surface_z: float = 0.694  # world height of BOTH table tops (asset origins land at z = -0.3)
    pick_pos: tuple[float, float] = (0.0, 0.55)  # xy the PICK table sits at
    ground_z: float = 0.0  # world height of the ground plane (see the module docstring)
    wheel_scale: float = 0.75  # the asset is 0.381 m across at 1.0 -> a 0.286 m rim (0.143 m radius)
    #: Where the wheel lies, in the PICK TABLE's frame — the same spot it occupies in
    #: `assembly.wheel_pick_place`, so the grasp transfers.
    wheel_init_offset: tuple[float, float] = (-0.35, -0.10)
    wheel_init_z: float = 0.0056  # wheel-origin drop height above the table top (m)
    wheel_init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz; flat on the table
    light_intensity: float = 3000.0  # the scene's dome light
    light_color: tuple[float, float, float] = (0.75, 0.75, 0.75)
    # Asset USDs; empty -> the vendored trees under the assembly suite's `assets/` (cross-suite reuse
    # is the established convention here — see packing/scenes/tool_packing.py). Re-vendor with
    # `python robobench/suites/assembly/assets/fetch_pick_place_assets.py`.
    asset_dir: str = ""
    wheel_usd: str = ""
    pick_table_usd: str = ""
    place_table_usd: str = ""

    def __post_init__(self) -> None:
        assets = asset_path(Path(__file__).resolve().parents[2] / "assembly" / "assets")
        self.asset_dir = self.asset_dir or str(assets)
        d = Path(self.asset_dir)
        self.wheel_usd = self.wheel_usd or str(d / "steering_wheel" / "steering_wheel.usd")
        self.pick_table_usd = self.pick_table_usd or str(
            d / "props" / "packing_table" / "SM_HeavyDutyPackingTable_C02_01_physics.usd")
        self.place_table_usd = self.place_table_usd or str(
            d / "props" / "packing_table_scene" / "packing_table.usd")

    @property
    def place_pos(self) -> tuple[float, float]:
        """xy the PLACE table sits at — the pick station shifted by `carry_dx` along +x."""
        return (self.pick_pos[0] + self.carry_dx, self.pick_pos[1])

    @property
    def table_z(self) -> float:
        """World z the PLACE (metre-scale) table asset's origin spawns at, so its top lands on
        `surface_z`."""
        return self.surface_z - self.TABLE_TOP_Z

    @property
    def bare_table_z(self) -> float:
        """World z the PICK table asset's origin spawns at. The bare table is authored in centimetres,
        but after `bare_scale` its top sits 0.994 m above its origin — the same `TABLE_TOP_Z` as the
        metre-scale set piece (it is the same table mesh) — so the derivation is identical. Kept as
        its own property so the two stations can diverge if one asset is ever swapped."""
        return self.surface_z - self.TABLE_TOP_Z

    def wheel_init_world(self) -> tuple[float, float, float]:
        """The wheel's ENV-LOCAL start pose — its pick-table-frame offset put onto that table."""
        px, py = self.pick_pos
        ox, oy = self.wheel_init_offset
        return (px + ox, py + oy, self.surface_z + self.wheel_init_z)

    def place_box_world(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        """The target zone in ENV-LOCAL coords: (x band, y band, max wheel height) — the
        place-table-frame bands above, shifted onto that table's placement."""
        wx, wy = self.place_pos
        return ((self.place_box_x[0] + wx, self.place_box_x[1] + wx),
                (self.place_box_y[0] + wy, self.place_box_y[1] + wy),
                self.surface_z + self.place_max_h)

    def stand_xy(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Where a floor-standing embodiment works from at each station, ENV-LOCAL: (pick, place).
        These are not constraints — nothing checks them — but they are the two spots at which this
        scene reproduces `assembly.wheel_pick_place`'s validated geometry, which is what
        `describe()` hands the agent as a starting guess."""
        px, py = self.pick_pos
        return ((px, py - 0.55), (px + self.carry_dx, py - 0.55))


@SCENES.register("wheel_carry")
class WheelCarryScene(BaseScene):
    cfg: WheelCarrySceneCfg

    #: Per-env world variation at generation (engine sampler). Only the wheel is a live rigid body
    #: with a PhysX view here — both tables are static spawns — so friction is the one physical knob.
    PHYSICAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "wheel_friction": {"dist": "uniform", "lo": 0.5, "hi": 0.9,
                           "reason": "around the asset's 0.7 leather rim — the gripped surface"},
    }

    #: Stage-wide look knobs for visual replay (data_engine render.py --visual_draw).
    VISUAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "light_intensity": {"dist": "uniform", "lo": 2500.0, "hi": 3500.0,
                            "reason": "around the 3000 nominal"},
    }

    #: L5 external views (see BaseScene.CAMERAS). Unlike a tabletop scene, one close shot cannot hold
    #: the task: `wide` stands well back and frames BOTH stations and the floor between them (the
    #: transit is the story), while `pick` and `place` are the per-station close-ups.
    CAMERAS: ClassVar[dict[str, dict]] = {
        "wide": {"eye": (1.50, -4.60, 2.30), "target": (1.50, 0.50, 0.70), "focal": 18.0,
                 "bands": {
                     "eye_x": {"dist": "uniform", "lo": 1.20, "hi": 1.80},
                     "eye_y": {"dist": "uniform", "lo": -4.90, "hi": -4.30},
                     "eye_z": {"dist": "uniform", "lo": 2.10, "hi": 2.50},
                 }},
        "pick": {"eye": (1.60, -0.40, 1.35), "target": (-0.10, 0.50, 0.70), "focal": 20.0},
        "place": {"eye": (4.60, -0.40, 1.35), "target": (2.90, 0.50, 0.70), "focal": 20.0},
    }

    def __init__(self, cfg: WheelCarrySceneCfg | None = None) -> None:
        super().__init__(cfg or WheelCarrySceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, dome light, the two kinematic table stations, and the loose steering wheel. The
        wheel keeps the asset's own convex-decomposition colliders and density-derived mass; only its
        friction is overridden (at bind)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        for usd in (c.wheel_usd, c.pick_table_usd, c.place_table_usd):
            if not Path(usd).is_file():
                raise FileNotFoundError(
                    f"{usd} not found — re-vendor with "
                    f"`python robobench/suites/assembly/assets/fetch_pick_place_assets.py`")
        px, py = c.pick_pos
        qx, qy = c.place_pos
        wx, wy, wz = c.wheel_init_world()
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(usd_path=str(
                    Path(c.asset_dir) / "props" / "ground" / "default_ground.usd")),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, c.ground_z)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=c.light_color),
            ),
            # Both stations are immovable furniture the wheel — and the robot — can rest and lean
            # against, but they get there differently, because the two USDs are authored differently.
            # The BARE table carries a PhysicsCollisionAPI and no rigid body at all, so it is already
            # static: asking for `kinematic_enabled` here would find no rigid-body prim to set it on
            # and only log "Could not perform 'modify_rigid_body_properties' on any prims" on every
            # build. Left off deliberately — the collider is what the task needs.
            "pick_table": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/PickTable",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(px, py, c.bare_table_z), rot=(1.0, 0.0, 0.0, 0.0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.pick_table_usd,
                    scale=(c.bare_scale,) * 3,
                ),
            ),
            # The SET PIECE does contain a rigid body — the `container_h20` basket — so here
            # `kinematic_enabled` is load-bearing: without it the basket is a free body the wheel
            # would shove around the table instead of landing in.
            "place_table": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/PlaceTable",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(qx, qy, c.table_z), rot=(1.0, 0.0, 0.0, 0.0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=c.place_table_usd,
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
                init_state=RigidObjectCfg.InitialStateCfg(pos=(wx, wy, wz), rot=c.wheel_init_quat),
            ),
        }

    def sim_cfg(self) -> SimCfg:
        # dt = 1/200 divides evenly into the humanoid control periods this scene is sized for (a
        # 0.02 s period is exactly 4 physics steps), so a 50 Hz control loop lands on whole steps —
        # which is also the rate the vendored locomotion policy was trained at. The GPU buffers are
        # widened for the wheel's convex-decomposition shell against the basket walls.
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
        """Grab the wheel handle, cache env origins, allocate the journey latches, and set friction.
        Once, after the build."""
        super().bind(env)
        self.wheel: RigidObject = env.iscene["wheel"]
        self.env_origins = env.iscene.env_origins
        # Journey latches — see `post_step`. Long-horizon tasks here are scored on a chain of
        # milestones that must be REACHED, not merely be true at the end.
        z = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._lifted, self._transited = z.clone(), z.clone()
        self.apply_physical_params(env, {n: [getattr(self.cfg, n)] * env.num_envs for n in self.PHYSICAL_PARAMS})

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance the journey latches at sim rate. `lifted` = the wheel got clear of the pick table;
        `transited` = it then travelled most of the way to the place station. Both are one-way: they
        prove the path was walked, and releasing the wheel later does not un-prove it (the live
        `placed()` predicate is what proves the destination)."""
        p = self.wheel.data.root_pos_w - self.env_origins
        c = self.cfg
        self._lifted |= p[:, 2] > c.surface_z + c.lift_h
        self._transited |= self._lifted & (self.carried_fraction() > 0.9)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Re-place the wheel flat on the pick table at its start pose (+ optional xy jitter), at
        rest, and clear the journey latches. Both tables are static spawns and never move."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = self.env_origins[env_ids] + torch.tensor(c.wheel_init_world(), device=dev)
        if c.reset_pos_jitter:
            st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        st[:, 3:7] = torch.tensor(c.wheel_init_quat, device=dev)
        self.wheel.write_root_state_to_sim(st, env_ids)
        self._lifted[env_ids] = False
        self._transited[env_ids] = False

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Restorable state: the wheel's world root state (13) plus the journey latches. Everything
        else in the scene is static furniture. The latches travel with the state so a snapshot
        restored mid-carry does not silently lose the milestones already earned."""
        return {
            "wheel": self.wheel.data.root_state_w[env_ids].clone(),
            "lifted": self._lifted[env_ids].clone(),
            "transited": self._transited[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        """Restore `get_state`: the wheel's root state and the journey latches."""
        self.wheel.write_root_state_to_sim(state["wheel"], env_ids)
        self._lifted[env_ids] = state["lifted"]
        self._transited[env_ids] = state["transited"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        d = 2 * 0.1905 * c.wheel_scale  # the asset is 0.381 m across at scale 1
        inner_x = c.BASKET_INNER_X[1] - c.BASKET_INNER_X[0]
        inner_y = c.BASKET_INNER_Y[1] - c.BASKET_INNER_Y[0]
        rim = c.surface_z + (c.BASKET_RIM_Z - c.TABLE_TOP_Z)
        (sx, sy), (tx, ty) = c.stand_xy()
        return (
            f"Two heavy-duty packing tables stand {c.carry_dx:.1f} m apart, their tops "
            f"{c.surface_z:.2f} m off the floor. On the near table a car steering wheel about "
            f"{d * 100:.0f} cm across lies flat, hub up. The far table carries an open plastic "
            f"basket, its rim {rim:.2f} m up and its interior about {inner_x * 100:.0f} x "
            f"{inner_y * 100:.0f} cm — roomy enough to take the wheel lying flat. Nothing stands "
            f"between the two tables but open floor.\n"
            f"Goal: pick the steering wheel up off the near table, carry it to the far table, and "
            f"put it in the basket. The task is complete once the wheel is resting inside the "
            f"basket, released and come to rest — not still held, not balanced on the rim.\n"
            f"The two tables are further apart than any arm reaches, so the wheel can only get "
            f"there if the robot itself does. Working positions that face each table squarely are "
            f"about ({sx:.2f}, {sy:.2f}) for the pick and ({tx:.2f}, {ty:.2f}) for the place; they "
            f"are a starting guess, not a requirement."
        )

    # ----- progress (public: reads how far the task has got) ------------------------------------
    def success(self) -> torch.Tensor:
        """Whether the task is done, `(num_envs,)`. The latches prove the journey — the wheel really
        was lifted clear and carried across, not spawned or slid there — and `placed()` proves the
        destination is where it ended up."""
        return self._lifted & self._transited & self.placed()

    def score(self) -> torch.Tensor:
        """Progress in [0, 100], `(num_envs,)`, over the ordered chain lifted -> transited -> placed.
        A stage reached out of order earns nothing until its predecessors are in: the chain is the
        task. `carried_fraction` fills in the long middle so a run that walks half way and drops the
        wheel scores above one that never moved."""
        part = 40.0 * self.carried_fraction().clamp(0.0, 1.0)
        s = torch.zeros_like(part)
        s = torch.where(self._lifted, 20.0 + part, s)
        s = torch.where(self._lifted & self._transited, torch.full_like(s, 70.0), s)
        return torch.where(self.success(), torch.full_like(s, 100.0), s)

    def placed(self) -> torch.Tensor:
        """Whether the wheel is placed, `(num_envs,)`: its origin inside the basket box
        (`place_box_x/_y` in the place table's frame, below `place_max_h` above the top) and settled
        to under `settle_vel` on every axis. Embodiment-agnostic by construction: it reads the wheel
        and the basket only, never the robot."""
        (x0, x1), (y0, y1), max_z = self.cfg.place_box_world()
        p = self.wheel.data.root_pos_w - self.env_origins  # env-local, like every target here
        inside = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1) & (p[:, 2] < max_z)
        return inside & self.settled()

    def settled(self) -> torch.Tensor:
        """Whether the wheel has come to rest, `(num_envs,)`: every component of its linear AND
        angular velocity under `settle_vel`."""
        return self.wheel.data.root_vel_w.abs().amax(dim=-1) < self.cfg.settle_vel

    def dropped(self) -> torch.Tensor:
        """Whether the wheel has fallen to the floor, `(num_envs,)` — a readable signal, not a forced
        reset: this scene never terminates an episode on its own. Unlike the fixed-base version this
        is a live hazard for most of the run, since the wheel spends metres in the air."""
        z = self.wheel.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        return z < self.cfg.surface_z - self.cfg.drop_below_surface

    def carried_fraction(self) -> torch.Tensor:
        """How far along the pick -> place transit the wheel has got, `(num_envs,)` in [0, 1] — the
        fraction of `carry_dx` closed, measured on the WHEEL (the thing the task is about), not on
        the robot. 0 at the pick station, 1 at the place station."""
        x = self.wheel.data.root_pos_w[:, 0] - self.env_origins[:, 0]
        return ((x - self.cfg.pick_pos[0]) / self.cfg.carry_dx).clamp(0.0, 1.0)

    def wheel_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The wheel's ENV-LOCAL position `(num_envs, 3)` and orientation `(num_envs, 4)` wxyz — the
        frame every target in this scene is quoted in."""
        return self.wheel.data.root_pos_w - self.env_origins, self.wheel.data.root_quat_w
