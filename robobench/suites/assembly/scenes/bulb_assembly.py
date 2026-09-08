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

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

    from robobench.core import BaseEnv


@dataclass
class BulbAssemblySceneCfg(BaseCfg):
    """Config for `BulbAssemblyScene`. Nothing is locked — a variant is just a copy with a few
    fields changed.
    Nothing is locked — a curriculum/debug variant is just a `.copy()` with a few changed.
    """

    # --- the curriculum / difficulty dials -----------------------------------------------
    # A bulb is "seated" when, in a socket's frame, its origin is at/below `seat_z` above the socket origin,
    # within `align_xy` of the axis, and tilted <= `align_axis_deg` off it (any bulb may seat in any socket).
    # seat_z calibrated to the asset: seated origin ~22 mm up vs ~35 mm+ resting on the bore mouth.
    seat_z: float = 0.027  # max bulb-origin height above the socket origin (m) to count as seated
    align_xy: float = 0.015  # max lateral distance (m) from the nearest socket axis
    align_axis_deg: float = 12.0  # max tilt of the bulb's screw axis off the socket axis (deg)
    # The bulb LIGHTS UP as it screws home: the centre-contact spring compresses progressively, so
    # contact resistance falls and current rises with depth — the light ramps from dark at first
    # contact (light_start_z) to fully bright at the seated depth (light_full_z = seat_z), following
    # progress**light_gamma (higher gamma = fainter early turns, steeper finish), the filament colour
    # warming yellow -> white along the way (cool filament at low current). Driven per-step by the
    # glass OmniPBR emissive inputs (verified live-settable).
    light_start_z: float = 0.032  # depth (m, socket frame) where the glow begins (~free-rest height)
    light_full_z: float = 0.027  # depth at/below which it is fully bright (default = seat_z)
    light_gamma: float = 3.0  # brightness = lit_intensity * progress**gamma
    lit_intensity: float = 500000.0  # emissive_intensity fully lit (0 disables the mechanic)
    reset_pos_jitter: float = 0.01  # uniform +/- xy jitter per bulb at reset (m)
    # Part friction (static = dynamic), split per shape on the bulb: the metal cap/thread stays slick so
    # it threads steadily, while the GLASS is grippy — torque on the round glass is pad friction only, and
    # below ~0.3 no parallel-jaw gripper can self-lock on it (verified robot-unsolvable at glass mu=0.01).
    bulb_friction: float = 0.01  # the cap/thread shape
    bulb_glass_friction: float = 0.3  # the glass envelope shape
    socket_friction: float = 0.75

    # --- structure, reset layout, masses, asset paths (fixed) -------------------------------
    num_pairs: int = 1  # number of socket+bulb pairs
    # Socket xy slots, relative to the table centre. One socket per slot.
    socket_slots: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    socket_opening_z: float = 0.0385  # bore-mouth height above the socket origin (m), from build_socket
    bulb_mass: float = 0.05  # informational; the real value (+ inertia) is baked into bulb.usd
    light_intensity: float = 2500.0  # the scene's dome light
    # Bulbs' start pose. Default: each bulb lying on its side (90° about x -> screw axis horizontal) in a
    # row on the +x side of the sockets, ready to be picked up. A curriculum/robot may set `bulb_init_xy`.
    bulb_init_xy: tuple[tuple[float, float], ...] = ()  # per-bulb start xy (table-rel.); () -> the row below
    bulb_row_x0: float = 0.13  # x of bulb0 (the loose bulbs lie to the +x side of the sockets)
    bulb_row_y: float = 0.0  # y of the row
    bulb_spacing: float = 0.12  # x gap between adjacent bulbs (bulb k at x0 + k*spacing)
    bulb_init_z: float = 0.024  # bulb-origin DROP height above the surface; the lying bulb then settles
    # tilted ~25 deg (its Ø20 cap end droops to the table, origin ends ~5 mm up) — read the live pose, don't
    # assume a horizontal axis at this height.
    bulb_init_quat: tuple[float, float, float, float] = (2 ** -0.5, 2 ** -0.5, 0.0, 0.0)  # wxyz; 90° about x -> lying
    # Selectable work surface. `table` picks a preset in `TABLES`; the three fields below default to it
    # when left None/empty, or override it (e.g. raise `surface_z` so a standing robot can reach).
    table: str = "lab_table"  # which work surface: "lab_table" | "packing"
    surface_z: float | None = None  # table-top height (m); None -> the preset's
    workbench_pos: tuple[float, float] | None = None  # xy the table (and sockets) sit at; None -> preset
    workbench_usd: str = ""  # empty -> the preset's vendored USD
    # Work-surface presets (vendored under assets/props/) — same set as nut_thread.
    TABLES: ClassVar[dict[str, dict[str, Any]]] = {
        "lab_table": {"usd": ("lab_table", "table_instanceable.usd"), "scale": 1.0,
                      "orient": (0.70711, 0.0, 0.0, 0.70711), "surface_z": 0.0, "pos": (0.5, 0.0),
                      "top_offset": 0.0, "height": 1.05, "kinematic": False},
        "packing": {"usd": ("packing_table", "SM_HeavyDutyPackingTable_C02_01_physics.usd"), "scale": 0.01,
                    "orient": (1.0, 0.0, 0.0, 0.0), "surface_z": 0.994, "pos": (0.0, 0.0),
                    "top_offset": 0.994, "height": 0.994, "kinematic": True},
    }
    # Bulb + socket USDs. Empty -> the packaged standalone assets under assets/bulb/.
    asset_dir: str = ""
    bulb_usd: str = ""
    socket_usd: str = ""

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

    #: Per-env world variation at generation (engine sampler: slot 0 nominal, slot e >= 1 draws
    #: index env_draw + e - 1; --nominal skips). Same bands the bulb_data_gen campaign verified:
    #: glass widened UP only (the 0.3 nominal is the pad self-lock knee — below it the task is
    #: robot-unsolvable).
    PHYSICAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "bulb_friction": {"dist": "uniform", "lo": 0.005, "hi": 0.02},
        "bulb_glass_friction": {"dist": "uniform", "lo": 0.30, "hi": 0.45,
                                "reason": "self-lock knee at 0.3 — widen up only"},
        "socket_friction": {"dist": "uniform", "lo": 0.60, "hi": 0.90},
    }

    #: L5 visual dials for replay (data_engine render.py --visual_draw): stage-wide look knobs ->
    #: sampling bands, cfg default = the nominal look. Applied once per render pass, never per env
    #: (the dome light is one shared prim).
    VISUAL_PARAMS: ClassVar[dict[str, dict | None]] = {
        "light_intensity": {"dist": "uniform", "lo": 2000.0, "hi": 3000.0,
                            "reason": "around the 2500 nominal"},
        "lit_intensity": {"dist": "uniform", "lo": 4.0e5, "hi": 6.0e5,
                          "reason": "around the 5e5 nominal; post_step scales the ramp by it"},
    }

    #: L5 external views (see BaseScene.CAMERAS); bands wiggle the eye a few cm per episode.
    CAMERAS: ClassVar[dict[str, dict]] = {
        "front": {"eye": (0.78, -0.89, 0.54), "target": (0.30, -0.05, 0.10), "focal": 18.15,
                  "bands": {
                      "eye_x": {"dist": "uniform", "lo": 0.77, "hi": 0.79},
                      "eye_y": {"dist": "uniform", "lo": -0.90, "hi": -0.88},
                      "eye_z": {"dist": "uniform", "lo": 0.53, "hi": 0.55},
                  }},
    }

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
        # Standard PhysX recipe. dt=1/240 not 1/120: the fine thread tunnels at 1/120 under the
        # gravity-driven plunge; the smaller step resolves the contact so the bulb threads cleanly.
        # No translucency: the bulb glass is OPAQUE frosted OmniPBR (clear OmniGlass drew over the robot —
        # real-time translucency sorting — and can't glow; the frosted look supports the runtime
        # emissive_intensity dial for a lit bulb). Re-enable translucency if switching back to OmniGlass.
        return SimCfg(
            dt=1.0 / 240.0,
            render={"enable_reflections": True},
            physx={
                "solver_type": 1,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                # 2**28 overflowed at 512 envs (PhysX asked for 848 MB; 33k overflow errors in
                # one datagen wave). An overflow DROPS contacts nondeterministically, so every
                # recording made under it was physically wrong and none replayed.
                "gpu_collision_stack_size": 2**30,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ----------------------------------------------------------------------------
    def apply_visual_params(self, env: BaseEnv, values: dict[str, Any]) -> None:
        """The live subset of `VISUAL_PARAMS` (see BaseScene): both knobs are attribute writes, so
        both apply here — `light_intensity` also reaches the dome at build via the cfg, making this
        write redundant-but-harmless on a fresh build."""
        unknown = set(values) - set(self.VISUAL_PARAMS)
        if unknown:
            raise ValueError(f"{type(self).__name__} cannot apply visuals: {sorted(unknown)}")
        if "light_intensity" in values:
            env.stage.GetPrimAtPath("/World/light").GetAttribute("inputs:intensity").Set(
                float(values["light_intensity"]))
        if "lit_intensity" in values:
            self.cfg.lit_intensity = float(values["lit_intensity"])  # post_step reads it per frame

    def apply_physical_params(self, env: BaseEnv, values: dict[str, list]) -> None:
        """Write the scene's frictions PER ENV (static = dynamic), `values[name]` one value per env
        for names from `PHYSICAL_PARAMS`. `bind()` routes the nominal application through here with
        uniform values, so this is THE friction path — per-env sampling (data_engine `--env_draw`,
        eval replay of recorded draws) reuses it, never a copy. The bulb is split per shape — slick
        cap/thread vs grippy glass: bulb.usd binds a distinct physics material to each collider, and
        the glass shape is identified by that authored read-back (glass authored grippier), not by
        shape order."""
        unknown = set(values) - set(self.PHYSICAL_PARAMS)
        if unknown:
            raise ValueError(f"{type(self).__name__} cannot apply per-env: {sorted(unknown)}")
        ids = torch.arange(env.num_envs, device="cpu")
        if "socket_friction" in values:
            col = torch.tensor(values["socket_friction"], dtype=torch.float32).view(-1, 1, 1)
            for s in self.sockets:
                mats = s.root_physx_view.get_material_properties()
                mats[..., 0:2] = col
                s.root_physx_view.set_material_properties(mats, ids)
        if "bulb_friction" in values or "bulb_glass_friction" in values:
            for b in self.bulbs:
                mats = b.root_physx_view.get_material_properties()  # (n, n_shapes, 3)
                glass = int(mats[0, :, 0].argmax())  # detect BEFORE writing (glass stays grippiest)
                if "bulb_friction" in values:
                    mats[..., 0:2] = torch.tensor(values["bulb_friction"], dtype=torch.float32).view(-1, 1, 1)
                if "bulb_glass_friction" in values:
                    mats[:, glass, 0:2] = torch.tensor(values["bulb_glass_friction"], dtype=torch.float32).view(-1, 1)
                b.root_physx_view.set_material_properties(mats, ids)

    def bind(self, env: BaseEnv) -> None:
        """Grab handles, cache env origins, and set part friction. Called once after the build (physx ready)."""
        super().bind(env)
        self.sockets: list[Articulation] = [env.iscene[f"socket_{i}"] for i in range(self.cfg.num_pairs)]
        self.bulbs: list[RigidObject] = [env.iscene[f"bulb_{i}"] for i in range(self.cfg.num_pairs)]
        self.env_origins = env.iscene.env_origins
        c, E = self.cfg, env.num_envs
        self.apply_physical_params(env, {n: [getattr(c, n)] * E for n in self.PHYSICAL_PARAMS})
        # Lit-bulb mechanic: cache each bulb's glass emissive_intensity attr (per env — the cloner copies
        # the material under every env) and the last written on/off state, so post_step only writes on
        # transitions. Missing attrs (asset without the OmniPBR glass) disable the mechanic for that bulb.
        stage = env.stage

        def _emissive_attrs(n: int, i: int):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{n}/Bulb_{i}/visual/Materials/GLASS/OmniPBR")
            if not prim.IsValid():
                return None
            attrs = (prim.GetAttribute("inputs:emissive_intensity"), prim.GetAttribute("inputs:emissive_color"))
            return attrs if all(a.IsValid() for a in attrs) else None

        self._emissive = [[_emissive_attrs(n, i) for i in range(self.cfg.num_pairs)] for n in range(env.num_envs)]
        # last written brightness, quantized to 1/64 steps of full (so the per-step ramp only touches USD
        # when the visible level actually moves)
        self._lit_q = torch.zeros(env.num_envs, self.cfg.num_pairs, dtype=torch.int16, device=env.device)

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

    # filament colour endpoints: low current -> cool filament, warm yellow; full current -> warm white.
    _LIT_COLOR: ClassVar[tuple[float, float, float]] = (1.0, 0.85, 0.55)
    _DIM_COLOR: ClassVar[tuple[float, float, float]] = (1.0, 0.55, 0.18)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Lit-bulb mechanic, every step: a bulb aligned in a socket glows brighter as it screws home —
        intensity ramps as progress**light_gamma from dark at light_start_z to lit_intensity at
        light_full_z, the filament colour warming yellow -> white with it. Writes USD only when the
        quantized (1/64) brightness level moves."""
        import math

        c = self.cfg
        if c.lit_intensity <= 0:
            return
        off = self._bulb_offsets_in_socket()  # (n, N, B, 3)
        near_dist, near_socket = off[..., :2].norm(dim=-1).min(dim=-1)
        depth = torch.gather(off[..., 2], 2, near_socket.unsqueeze(-1)).squeeze(-1)
        aligned = (near_dist <= c.align_xy) & (self._bulb_axis_cos() >= math.cos(math.radians(c.align_axis_deg)))
        t = ((c.light_start_z - depth) / max(c.light_start_z - c.light_full_z, 1e-6)).clamp(0.0, 1.0)
        q = (t.pow(c.light_gamma) * 64).round().to(torch.int16) * aligned  # gamma ramp, 1/64 steps
        changed = q != self._lit_q
        if changed.any():
            for n, i in changed.nonzero().tolist():
                attrs = self._emissive[n][i]
                if attrs is not None:
                    s = float(q[n, i]) / 64.0  # 0..1 brightness fraction
                    attrs[0].Set(c.lit_intensity * s)
                    attrs[1].Set(tuple(d + (l - d) * s for d, l in zip(self._DIM_COLOR, self._LIT_COLOR)))
            self._lit_q = q

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
            f"by its thread, and GLOWS ever brighter as it screws home — from a faint orange at first "
            f"electrical contact to fully bright warm white exactly when seated. The task is complete once "
            f"{'the bulb is' if n == 1 else f'all {n} bulbs are'} seated."
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

    def success(self) -> torch.Tensor:
        """Task-level success per env: every bulb seated. The public name solutions written
        against earlier builds call (the rubric rewrite kept only `seated()`); identical to the
        grader's `check_success`."""
        return self.seated().all(dim=1)

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
