"""SliceFoodScene — transverse slicing: pre-split produce on a chopping board; a knife press
releases the welds. Env name: cutting.slice (default carrot; `food="banana"` etc.).

The object world: a food item baked offline into a welded rigid chain of transverse pieces,
lying on a chopping board on a kitchen island, with a chef knife lying edge-down on a
two-notch steel knife rest beside the board (held by gravity alone; the plate is vertical
and the edge level — a top-down pinch on the handle lifts it straight off). **Goal: pick up
the knife and press it through every scored plane until the food is in the target number
of pieces** — graded by `grader.SliceFoodGrader`. Embodiment-agnostic: the scene knows
nothing about who drives the knife.

Mechanic — weld release on press: piece welds are pre-authored ENABLED FixedJoints (one per
scored plane). Every step the scene evaluates the cut gate in the FOOD's live frame and,
when the knife's real edge is at a plane's flesh-centre point (within the plane, aligned,
pressed to the release depth, food settled, blade pressing), releases that plane's weld —
sticky and monotone. Separation afterwards is real physics. Cohesion is the ONLY scripted
part; no teleports, no mesh swaps.

Assets: `assets/<food>/` = per-piece USDs + texture + `manifest.json` (planes, centroids,
bounds), baked offline (skin texture + flesh caps, kerf-inset convex hulls).
Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, SimCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


@dataclass
class SliceFoodSceneCfg(BaseCfg):
    """Config for `SliceFoodScene`. A variant is just a `.copy()` with a few fields changed."""

    # --- the task ---------------------------------------------------------------------------
    food: str = "carrot"  # which assets/<food> dir to load
    target_pieces: int = 0  # 0 -> all planes cut (n_pieces)

    # --- the cut gate (pose + velocity, house style; no force sensing) -----------------------
    plane_tol: float = 0.006  # knife edge within this distance of a scored plane (m)
    blade_align_deg: float = 15.0  # blade plane normal within this angle of the plane axis
    # centre-point law: the edge must pass BELOW the midpoint of the two adjacent piece
    # centres by this depth (0 = release as the edge reaches the centre) — or bottom out
    # on the board (thin ends' centres sit closer than this to the board)
    depth_past_center: float = 0.010
    food_settle_speed: float = 0.10  # max piece speed at the moment of release (m/s)
    flesh_radius: float = 0.025  # edge point within this of the plane's flesh centre (m)
    press_vz_max: float = 0.02  # knife must be pressing (not rising) at release (m/s)

    # --- layout ------------------------------------------------------------------------------
    island_top: float = 0.858  # kitchen island height (the board rests on it)
    board_size: tuple[float, float, float] = (0.60, 0.26, 0.024)
    board_friction: float = 0.8
    piece_friction: float = 0.6
    piece_density: float = 900.0
    # knife rest: two notched steel blocks on the counter; the knife lies edge-down in the
    # notches, plate vertical, edge level (chop pose), held by gravity alone
    knife_present: tuple[float, float] = (0.10, -0.21)  # xy of the first notch (heel side)
    rest_notch_x: tuple[float, float] = (0.04, 0.16)  # knife-local x of the two notches
    rest_floor_h: float = 0.03  # notch floor height above the counter (m)
    rest_rail_h: float = 0.025  # rail height above the notch floor
    rest_gap: float = 0.002  # rail clearance per side around the 4.2 mm plate
    rest_block: tuple[float, float] = (0.03, 0.06)  # block footprint along / across the blade
    knife_mass: float = 0.18  # a 22 cm chef knife; heavier knives creep in a friction pinch
    # knife22r's cutting edge runs a 28 deg diagonal in its local frame (heel (0,-0.015) ->
    # tip (0.22,-0.134)): rest/hold it rotated by this about the blade normal so the EDGE
    # is level, else the tip grounds 6 cm before the mid-blade
    knife_edge_tilt_deg: float = 28.2
    # arm stand: a matching small cabinet (the island asset scaled) behind the island, top at
    # island height — an arm's reach wants its base ~0.55 m behind the cut, past the island's
    # y=-0.38 edge. None -> no stand.
    arm_stand: tuple[float, float] | None = (0.0, -0.58)  # xy of the stand centre
    arm_stand_size: tuple[float, float] = (0.40, 0.40)  # footprint (m)
    reset_pos_jitter: float = 0.0  # uniform +/- xy jitter of the food at reset
    # food rest orientation (w,x,y,z), applied to the whole welded assembly at reset
    food_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    light_intensity: float = 2800.0

    # Per-food conditions — each food cuts differently; a preset value applies only where
    # the field was left at its dataclass default, so explicit overrides still win.
    FOOD_PRESETS: ClassVar[dict[str, dict]] = {
        "carrot": {
            # firm, straight, 3.2 cm: release as the edge reaches the flesh centre — the
            # split forms mid-stroke and the knife need not drive on to the board
            "depth_past_center": 0.0,
        },
        "banana": {
            "depth_past_center": 0.0,
            # an arched fruit must lie on its SIDE (arch horizontal): stood as a bridge,
            # every interior cut frees an unsupported overhang that tips onto the buried
            # blade and clamps it (drawer jam — it rides the knife out of the kerf)
            "food_rot": (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),  # roll 90 deg about x
        },
    }

    def __post_init__(self) -> None:
        for k, v in self.FOOD_PRESETS.get(self.food, {}).items():
            if getattr(self, k) == getattr(type(self), k):
                setattr(self, k, v)
        # chop pose (blade down, length along +y) with the edge levelled: chop x tilt_z
        h = math.radians(self.knife_edge_tilt_deg) / 2
        a, b = (0.5, 0.5, 0.5, 0.5), (math.cos(h), 0.0, 0.0, math.sin(h))
        self.knife_rot = (
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])
        self.surface_z = self.island_top + self.board_size[2]  # board top;
        # record_video and external cameras anchor on this


@SCENES.register("slice")
class SliceFoodScene(BaseScene):
    """Transverse slicing (carrot / banana / ...). Env name: cutting.slice"""

    cfg: SliceFoodSceneCfg
    ASSETS: ClassVar[Path] = Path(__file__).resolve().parents[1] / "assets"
    KNIFE: ClassVar[str] = "knife22r"

    PHYSICAL_PARAMS: ClassVar[dict] = {
        "piece_friction": {"dist": "uniform", "lo": 0.4, "hi": 0.8},
        "board_friction": {"dist": "uniform", "lo": 0.6, "hi": 0.9},
    }

    CAMERAS: ClassVar[dict] = {
        "front": {"eye": (0.42, -0.50, 0.402), "target": (0.0, -0.02, 0.162), "focal": 22.0},  # surface-relative (the render/eval contract adds cfg.surface_z = board top 0.878): was declared absolute (z 1.28/1.04) -> camera saw only the floor (2026-09-09)
    }

    def __init__(self, cfg: SliceFoodSceneCfg | None = None) -> None:
        super().__init__(cfg or SliceFoodSceneCfg())
        self.asset_dir = self.ASSETS / self.cfg.food
        self.manifest = json.loads((self.asset_dir / "manifest.json").read_text())
        self.knife_manifest = json.loads((self.ASSETS / self.KNIFE / "manifest.json").read_text())
        # planes: ("x", value); plane k joins pieces k and k+1; cut mask is per plane per env
        self.planes = [("x", p) for p in self.manifest["planes_x"]]
        self._edge_samples: list[tuple[float, float, float]] | None = None

    # ----- knife geometry ---------------------------------------------------------------------
    def knife_edge(self) -> list[tuple[float, float, float]]:
        """THE REAL CUTTING EDGE in the knife frame, sampled from the hull (local y_min per
        local x, 1 cm steps). The bounds y_min is the TIP's height: the edge runs a 28 deg
        diagonal from the heel (0, -0.015) to the tip (0.22, -0.134); a single point anchored
        at (0, y_min) would sit 12 cm below the real heel and release cuts in the air."""
        if self._edge_samples is None:
            from pxr import Usd, UsdGeom

            stage = Usd.Stage.Open(str(self.ASSETS / self.KNIFE / "piece_0.usd"))
            pts = [tuple(p) for p in UsdGeom.Mesh(
                stage.GetPrimAtPath("/Piece/Collision")).GetPointsAttr().Get()]
            samples = []
            for i in range(21):
                x = 0.01 * i
                ys = [p[1] for p in pts if abs(p[0] - x) < 0.008]
                if ys:
                    samples.append((x, min(ys), 0.0))
            self._edge_samples = samples
        return self._edge_samples

    def _edge_point(self, x: float) -> tuple[float, float, float]:
        return min(self.knife_edge(), key=lambda p: abs(p[0] - x))

    def _rest_layout(self):
        """Knife root pose (env-relative) laying the level edge on the notch floors, the xy of
        the two notch blocks, and the notch floor height."""
        c = self.cfg
        R = self._rot3(c.knife_rot)
        floor_z = c.island_top + c.rest_floor_h
        offs = []
        for x in c.rest_notch_x:
            p = self._edge_point(x)
            offs.append([sum(R[i][d] * p[d] for d in range(3)) for i in range(3)])
        root = (c.knife_present[0] - offs[0][0], c.knife_present[1] - offs[0][1],
                floor_z + 0.001 - offs[0][2])
        notches = [(root[0] + o[0], root[1] + o[1]) for o in offs]
        return root, notches, floor_z

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        man = self.manifest
        board_top = c.surface_z
        b0 = man["bounds"][0]

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=c.light_intensity, color=(0.92, 0.92, 0.9)),
            ),
            "island": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Island",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.429)),
                spawn=sim_utils.UsdFileCfg(usd_path=str(self.ASSETS / "kitchen_island.usd")),
            ),
            "board_visual": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/BoardVisual",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.12, 0.13, c.island_top)),
                spawn=sim_utils.UsdFileCfg(usd_path=str(self.ASSETS / "chopping_board.usd"),
                                           scale=(0.01, 0.01, 0.01)),
            ),
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.island_top + c.board_size[2] / 2)),
                spawn=sim_utils.CuboidCfg(
                    size=c.board_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.board_friction, dynamic_friction=c.board_friction),
                    visible=False,
                ),
            ),
        }
        if c.arm_stand is not None:
            # kitchen_island.usd is 1.15 x 0.761 x 0.858 centred at its origin (top at
            # island_top when spawned at z 0.429); scaled to the stand footprint it reads as
            # a matching cabinet — same marble top and doors
            out["arm_stand"] = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/ArmStand",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(c.arm_stand[0], c.arm_stand[1], 0.429)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(self.ASSETS / "kitchen_island.usd"),
                    scale=(c.arm_stand_size[0] / 1.15, c.arm_stand_size[1] / 0.761, 1.0)),
            )
        food_z = board_top + 0.001 - b0[2]  # nominal; reset() places the rotated assembly
        for k, meta in enumerate(man["pieces"]):
            if meta is None:
                continue
            cx, cy, cz = meta["centroid"]
            out[f"piece_{k}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Piece_%d" % k,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(self.asset_dir / f"piece_{k}.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=64, solver_velocity_iteration_count=1,
                        max_depenetration_velocity=0.3, linear_damping=0.1, angular_damping=0.1,
                        # low damping: freed slices must VISIBLY part while the blade is
                        # still between them (0.8 made separation lag the cut by seconds)
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(density=c.piece_density),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cx, cy, cz + food_z)),
            )
        # the knife rest: two notch blocks (static floor pad + two rails flanking the plate)
        root, notches, floor_z = self._rest_layout()
        steel = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.73, 0.75), metallic=1.0,
                                            roughness=0.35)
        half_gap = 0.0021 + c.rest_gap
        bx, by = c.rest_block
        rail_w = (bx - 2 * half_gap) / 2
        for i, (nx, ny) in enumerate(notches):
            out[f"rest_floor_{i}"] = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/RestFloor{i}",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(nx, ny, c.island_top + c.rest_floor_h / 2)),
                spawn=sim_utils.CuboidCfg(size=(bx, by, c.rest_floor_h),
                                          collision_props=sim_utils.CollisionPropertiesCfg(),
                                          visual_material=steel),
            )
            for side, sgn in (("L", -1.0), ("R", 1.0)):
                out[f"rest_rail_{i}{side}"] = AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/RestRail{i}{side}",
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(nx + sgn * (half_gap + rail_w / 2), ny, floor_z + c.rest_rail_h / 2)),
                    spawn=sim_utils.CuboidCfg(size=(rail_w, by, c.rest_rail_h),
                                              collision_props=sim_utils.CollisionPropertiesCfg(),
                                              visual_material=steel),
                )
        # RIGID knife (no compliant contact — a strong pinch squashes through a compliant
        # plate; softness lives in the FOOD), lying edge-level on the rest
        out["knife"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Knife",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(self.ASSETS / self.KNIFE / "piece_0.usd"),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.2, angular_damping=0.5,
                    solver_position_iteration_count=32, solver_velocity_iteration_count=1,
                    max_depenetration_velocity=0.5,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.knife_mass),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=root, rot=tuple(c.knife_rot)),
        )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(dt=1.0 / 240.0, physx={"solver_type": 1, "gpu_max_num_partitions": 1})

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        man = self.manifest
        self.pieces: list[RigidObject] = []
        self._cents: list[tuple[float, float, float]] = []
        for k, meta in enumerate(man["pieces"]):
            if meta is None:
                continue
            self.pieces.append(env.iscene[f"piece_{k}"])
            self._cents.append(tuple(meta["centroid"]))
        self.knife: RigidObject = env.iscene["knife"]
        self.env_origins = env.iscene.env_origins
        self.cut = torch.zeros(env.num_envs, len(self.planes), dtype=torch.bool, device=env.device)
        self._board_top = self.cfg.surface_z
        self._edge_local = torch.tensor(self.knife_edge(), device=env.device)  # (N, 3) knife frame
        # per-plane FLESH aim point: midpoint of the two adjacent piece centroids (curved
        # foods bend away from the chord line — a plane-slab test alone passes in mid-air)
        aims = []
        for k, (_, p) in enumerate(self.planes):
            a = (torch.tensor(self._cents[k]) + torch.tensor(self._cents[k + 1])) / 2
            a[0] = p
            aims.append(a)
        self._plane_aims = torch.stack(aims).to(env.device)  # (n_planes, 3) food frame
        self._precreate_weld_joints()

    @staticmethod
    def _rot3(q):
        w, x, y, z = q
        return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]

    def food_height(self) -> float:
        """Standing height of the food at rest (the food_rot-rotated bounds)."""
        b0, b1 = self.manifest["bounds"]
        R = self._rot3(self.cfg.food_rot)
        zs = [R[2][0] * xx + R[2][1] * yy + R[2][2] * zz
              for xx in (b0[0], b1[0]) for yy in (b0[1], b1[1]) for zz in (b0[2], b1[2])]
        return max(zs) - min(zs)

    def reset(self, env_ids: torch.Tensor) -> None:
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        b0, b1 = self.manifest["bounds"]
        # the whole welded assembly at food_rot, resting on the board
        R = self._rot3(c.food_rot)
        zmin = min(R[2][0] * xx + R[2][1] * yy + R[2][2] * zz
                   for xx in (b0[0], b1[0]) for yy in (b0[1], b1[1]) for zz in (b0[2], b1[2]))
        food_z = self._board_top + 0.001 - zmin
        qf = torch.tensor(c.food_rot, device=dev)
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.reset_pos_jitter
        for k, pc in enumerate(self.pieces):
            cx, cy, cz = self._cents[k]
            px = R[0][0] * cx + R[0][1] * cy + R[0][2] * cz
            py = R[1][0] * cx + R[1][1] * cy + R[1][2] * cz
            pz = R[2][0] * cx + R[2][1] * cy + R[2][2] * cz
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = origin + torch.tensor((px, py, pz + food_z), device=dev)
            st[:, 0:2] += jit
            st[:, 3:7] = qf
            pc.write_root_state_to_sim(st, env_ids)
        # the knife back on its rest
        root, _, _ = self._rest_layout()
        ks = torch.zeros(m, 13, device=dev)
        ks[:, 0:3] = origin + torch.tensor(root, device=dev)
        ks[:, 3:7] = torch.tensor(c.knife_rot, device=dev)
        self.knife.write_root_state_to_sim(ks, env_ids)
        self._reconcile_cuts(env_ids, torch.zeros(m, len(self.planes), dtype=torch.bool, device=dev))

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else env_ids
        self._reconcile_cuts(ids, self._cut_targets()[ids])

    # ----- state ----------------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pieces": torch.stack([p.data.root_state_w[env_ids].clone() for p in self.pieces], dim=1),
            "knife": self.knife.data.root_state_w[env_ids].clone(),
            "cut": self.cut[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for k, pc in enumerate(self.pieces):
            pc.write_root_state_to_sim(state["pieces"][:, k], env_ids)
        self.knife.write_root_state_to_sim(state["knife"], env_ids)
        self._reconcile_cuts(env_ids, state["cut"])

    # ----- description / observables --------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A {c.food} lies on a chopping board on a kitchen island; a chef knife lies on a "
            f"knife rest beside the board. The {c.food} is scored at {len(self.planes)} "
            f"transverse planes. Goal: pick up the knife and press its edge down through each "
            f"scored plane (blade aligned with the plane) to cut the {c.food} into slices. "
            f"The {c.food} is cut once every plane has been pressed through and the pieces "
            f"separate, all of them staying on the board."
        )

    def pieces_count(self) -> torch.Tensor:
        """(N,) long: connected components of the live weld chain per env."""
        return 1 + self.cut.sum(dim=1)

    # ----- physical params --------------------------------------------------------------------
    def apply_physical_params(self, env: BaseEnv, values: dict[str, list]) -> None:
        unknown = set(values) - set(self.PHYSICAL_PARAMS)
        if unknown:
            raise ValueError(f"unknown physical params: {unknown}")

        def _set_friction(asset, vals):
            mats = asset.root_physx_view.get_material_properties()
            v = torch.as_tensor(vals, dtype=mats.dtype).view(-1, 1, 1)
            mats[:, :, 0:2] = v
            asset.root_physx_view.set_material_properties(mats, torch.arange(mats.shape[0]))

        if "piece_friction" in values:
            for pc in self.pieces:
                _set_friction(pc, values["piece_friction"])
        if "board_friction" in values:
            _set_friction(env.iscene["board"], values["board_friction"])

    # ----- the cut gate (private; the weld-release sim-hack, not an agent action) --------------
    def _cut_targets(self) -> torch.Tensor:
        """THE STANDARD — which planes SHOULD be released now (sticky, monotone). Everything
        is measured in the FOOD's live frame (anchored on the mid piece — the food settles,
        rocks and yaws after reset, so world-frame plane positions are meaningless). The
        knife's REAL edge, at the sample nearest the plane's flesh centre, must be: within
        `plane_tol` of the plane, within `flesh_radius` of the centre along the blade,
        blade normal aligned with the plane axis, pressed below the release depth, with the
        food settled and the knife pressing (not rising)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        dev = self.env.device
        n = self.env.num_envs
        ref = self.pieces[len(self.pieces) // 2]
        rp, rq = ref.data.root_pos_w, ref.data.root_quat_w
        ref_cent = torch.tensor(self._cents[len(self.pieces) // 2], device=dev)

        kp = self.knife.data.root_pos_w
        kq = self.knife.data.root_quat_w
        # the real edge, all samples into world: (n, N, 3)
        N = self._edge_local.shape[0]
        kq_rep = kq.unsqueeze(1).expand(n, N, 4).reshape(-1, 4)
        el_rep = self._edge_local.unsqueeze(0).expand(n, N, 3).reshape(-1, 3)
        edge_pts_w = kp.unsqueeze(1) + quat_apply(kq_rep, el_rep).reshape(n, N, 3)
        # blade-plane normal (knife local +z) into the food frame
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        bn_f = quat_apply_inverse(rq, quat_apply(kq, ez))

        settled = torch.ones(n, dtype=torch.bool, device=dev)
        for pc in self.pieces:
            settled &= pc.data.root_lin_vel_w.norm(dim=-1) < c.food_settle_speed
        # release only while the blade PRESSES, never while it rises out of the kerf —
        # otherwise a press that keeps the food above the settle speed releases at lift
        # start instead, and the split appears while the knife is already in the air
        pressing = self.knife.data.root_lin_vel_w[:, 2] < c.press_vz_max

        cos_tol = math.cos(math.radians(c.blade_align_deg))
        target = self.cut.clone()
        for idx, (_, p) in enumerate(self.planes):
            aim = self._plane_aims[idx]
            aim_w = rp + quat_apply(rq, (aim - ref_cent).expand(n, 3))
            # real edge sample nearest the flesh centre (horizontally)
            dh = (edge_pts_w[:, :, :2] - aim_w[:, None, :2]).norm(dim=-1)  # (n, N)
            ep_w = edge_pts_w[torch.arange(n, device=dev), dh.argmin(dim=1)]
            ep_f = quat_apply_inverse(rq, ep_w - rp) + ref_cent
            near = (ep_f[:, 0] - p).abs() < c.plane_tol
            at_flesh = (ep_f[:, 1] - aim[1]).abs() < c.flesh_radius
            aligned = bn_f[:, 0].abs() > cos_tol
            depth_thresh = torch.maximum(
                aim_w[:, 2] - c.depth_past_center,
                torch.full_like(aim_w[:, 2], self._board_top + 0.0015))
            pressed = ep_w[:, 2] < depth_thresh
            target[:, idx] |= near & at_flesh & aligned & pressed & settled & pressing
        return target

    def _reconcile_cuts(self, env_ids, target) -> None:
        have = self.cut[env_ids]
        to_cut = target & ~have
        to_heal = have & ~target
        for row, pl in to_cut.nonzero(as_tuple=False).tolist():
            self._set_plane(int(env_ids[row]), pl, enabled=False)
        for row, pl in to_heal.nonzero(as_tuple=False).tolist():
            self._set_plane(int(env_ids[row]), pl, enabled=True)

    def _precreate_weld_joints(self) -> None:
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        b = self.manifest["bounds"]
        prim_idx = [k for k, meta in enumerate(self.manifest["pieces"]) if meta is not None]
        self._joint_paths: list[list[str]] = []  # [env][plane] -> joint path
        for e in range(self.env.num_envs):
            base = f"/World/envs/env_{e}"
            per_plane: list[str] = []
            for k, (_, p) in enumerate(self.planes):  # plane k joins pieces k and k+1
                jp = f"{base}/cutweld_x_{k}"
                jt = UsdPhysics.FixedJoint.Define(stage, jp)
                jt.CreateBody0Rel().SetTargets([f"{base}/Piece_{prim_idx[k]}"])
                jt.CreateBody1Rel().SetTargets([f"{base}/Piece_{prim_idx[k + 1]}"])
                mid = [p, 0.0, (b[0][2] + b[1][2]) / 2]
                ca, cb = self._cents[k], self._cents[k + 1]
                jt.CreateLocalPos0Attr(Gf.Vec3f(*[mid[d] - ca[d] for d in range(3)]))
                jt.CreateLocalPos1Attr(Gf.Vec3f(*[mid[d] - cb[d] for d in range(3)]))
                jt.CreateJointEnabledAttr(True)
                per_plane.append(jp)
            self._joint_paths.append(per_plane)

    def _set_plane(self, env_i: int, plane_idx: int, *, enabled: bool) -> None:
        from pxr import UsdPhysics

        UsdPhysics.FixedJoint.Get(
            self.env.stage, self._joint_paths[env_i][plane_idx]).GetJointEnabledAttr().Set(enabled)
        self.cut[env_i, plane_idx] = not enabled
