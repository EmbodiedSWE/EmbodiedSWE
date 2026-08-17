"""LeaningChainScene — build a frozen domino cascade: topple three tiles into a lane
so each leans on the next and the last leans on the anvil (sim_gen task
`screw_nail_i309`).

Derived from rlbench/screw_nail, but STRATEGICALLY different: the seed is tool-mediated
rotary ASSEMBLY — grasp a screwdriver, mate its tip to a nail and drive the fastener IN
by continuous rotation about a vertical axis (one tool grasp, one long twisting motion,
judged by fastener depth). Here there is no tool, no rotation about an axis and no
fastener: the task is a MULTI-BODY LEANING CONSTRUCTION. A lane (two low guide rails
with a raised ANVIL block at its far end and a white post marking its open start end)
sits on the floor; three tiles — RED, GREEN, BLUE — stand upright beside it. The goal
is the end state of a fallen domino run, frozen: each tile leaning up-lane at an
intermediate angle, its foot on the floor and its head resting ON the next body in the
chain — RED's head on GREEN, GREEN's head on BLUE, BLUE's head on the anvil — ordered
red -> green -> blue from the open end toward the anvil. Every leaning rest is real
support: a settled tile at an intermediate tilt with its foot on the floor and its head
in the air is physically propped by whatever its head touches (nothing else in the
scene can hold it there — the rails are too low, asserted below). The build order is
mechanically forced BACK-TO-FRONT (blue first, red last): a tile toppled onto empty
lane just falls flat, because its support does not exist yet. A solver needs a
different PLAN from the seed (three dependent placements whose stability each depends
on the previously built support, in reverse chain order — not one continuous tool
rotation) and a different code structure (per-tile lane-frame lean/support predicates
chained by identity and order — not a fastener-depth readout).

success(): all three tiles simultaneously lean in the lane — tilt in
[tilt_min, tilt_max] from horizontal, head up-lane, foot near the floor, head high,
in-lane, ordered red < green < blue along the lane, each head reaching past the foot
of its support (blue's onto the anvil top) — with every tile settled.

score() is graded and latched ORDER-AWARE (credit never evaporates): 0.25 for blue
ever seated on the anvil + 0.25 for green ever seated while blue is seated + 0.25 for
red ever seated while green and blue are seated (streak-gated: the pose must hold
still for `streak_need` consecutive substeps before it latches — a fly-through pose
never counts), capped at 0.75; 1.0 iff success(). The null policy scores ~0 (tiles
spawn standing upright beside the lane: standing fails the tilt band, off-lane fails
containment).

Per-episode randomization (readback-verified in smoke): lane position + yaw, the
color -> spawn-slot permutation, per-tile spawn jitter + free yaw. Assets are fully
procedural compound-box spawners. Heavy imports (isaaclab, pxr) are deferred so
importing this module — and registering the scene — stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv

TILE_ORDER = ("red", "green", "blue")  # chain order, open end -> anvil


# ----- custom compound spawners ---------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _material(stage, path: str, static: float = 0.7, dynamic: float = 0.6):
    """One USD physics material (explicit friction + zero restitution — custom-spawner
    colliders otherwise land on engine defaults). Friction is deliberately high: the
    leaning chain is held by foot friction + head support."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float, material=None) -> None:
    """Author one box child prim (translate -> scale, authored once — the duplicate
    xformOp trap is avoided by never re-authoring an existing prim's ops)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(seg.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _rigid_root(stage, prim_path: str, translation, orientation, mass: float,
                kinematic: bool = False):
    """Author one rigid-body root Xform with the standard physics armor (zero
    sleep/stabilization thresholds: a sleeping body silently ignores applied wrenches,
    which the solve/smoke force probes depend on)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    return root, pxrb


def _spawn_lane(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The KINEMATIC lane fixture: two low guide rails, the raised ANVIL block at the
    +x (far) end, and a white start post beyond the -x (open) end. One rigid body;
    origin = lane center at ground level; +x runs from the open end toward the anvil."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, _pxrb = _rigid_root(stage, prim_path, translation, orientation, 5.0,
                              kinematic=True)
    mat = _material(stage, f"{prim_path}/phys_mat")
    c, co = cfg, cfg.contact_offset
    yellow, dark, white = (0.85, 0.75, 0.15), (0.28, 0.28, 0.32), (0.95, 0.95, 0.95)
    boxes = [
        ("rail_py", (c.rail_len, c.rail_w, c.rail_h),
         (0.0, +c.rail_y, c.rail_h / 2), yellow),
        ("rail_ny", (c.rail_len, c.rail_w, c.rail_h),
         (0.0, -c.rail_y, c.rail_h / 2), yellow),
        ("anvil", (c.anvil_len, c.anvil_wid, c.anvil_h),
         (c.anvil_face + c.anvil_len / 2, 0.0, c.anvil_h / 2), dark),
        ("post", (0.012, 0.012, 0.060), (-c.rail_len / 2 - 0.010, 0.0, 0.030), white),
    ]
    for name, size, center, col in boxes:
        _box(stage, f"{prim_path}/{name}", size, center, col, co, material=mat)
    del c
    return root


def _spawn_tile(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One tile: a single box, body frame x = width, y = thickness, z = length (so
    identity orientation = standing upright). Moderate damping so a toppled tile's
    bounce dies fast and the settle gates latch promptly."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root, pxrb = _rigid_root(stage, prim_path, translation, orientation, cfg.mass)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.30)
    mat = _material(stage, f"{prim_path}/phys_mat")
    _box(stage, f"{prim_path}/body", (cfg.width, cfg.thick, cfg.length),
         (0.0, 0.0, 0.0), cfg.color, cfg.contact_offset, material=mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg classes (lazily — app-free import)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "lane" not in _SPAWNER_CACHE:

        @configclass
        class LaneSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lane)
            rail_len: float = 0.40
            rail_w: float = 0.008
            rail_h: float = 0.010
            rail_y: float = 0.060
            anvil_face: float = 0.150
            anvil_len: float = 0.050
            anvil_wid: float = 0.112
            anvil_h: float = 0.055
            contact_offset: float = 0.001

        @configclass
        class TileSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tile)
            mass: float = 0.06
            width: float = 0.048
            thick: float = 0.016
            length: float = 0.120
            color: tuple = (0.8, 0.1, 0.1)
            contact_offset: float = 0.001

        _SPAWNER_CACHE.update(lane=LaneSpawnerCfg, tile=TileSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LeaningChainSceneCfg(BaseCfg):
    """Config for `LeaningChainScene`. The support-honesty is asserted in
    `__post_init__`: nothing in the scene except the anvil or another tile can hold a
    tile inside the accepted lean band (rails too low, a flat tile too thin for the
    head-height gate, standing rejected by the tilt band), so a settled in-band pose
    IS a propped rest on the chain."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    tilt_min_deg: float = tunable(15.0)  # lean band, from horizontal: flat / rail-lean
    tilt_max_deg: float = tunable(65.0)  # ... rejected below, standing rejected above
    head_z_min: float = tunable(0.030)  # head end point must be this high (m)
    foot_z_max: float = tunable(0.022)  # foot end point must be this low (m)
    axis_y_max: float = tunable(0.35)  # |lane-frame y| of the tile axis: head points up-lane
    lane_y_tol: float = tunable(0.052)  # |lane-frame y| of foot AND head (in-lane)
    overlap_min: float = tunable(0.005)  # head must reach past the support foot by this (m)
    settle_lin: float = tunable(0.08)  # max |lin vel| at judging (m/s, above phantom band)
    settle_ang: float = tunable(1.0)  # max |ang vel| at judging (rad/s)
    streak_need: int = tunable(25)  # substeps a seated pose must hold before latching

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    lane_jitter: float = tunable(0.03)  # +-xy jitter of the lane fixture (m)
    lane_yaw_deg: float = tunable(30.0)  # +- lane yaw (deg)
    tile_jitter: float = tunable(0.02)  # +-xy spawn jitter per tile (m)

    # --- info: structure (the geometry the spawners author) ----------------------------------
    lane_pos: tuple = info((0.45, 0.0))  # lane fixture origin (env frame, nominal)
    rail_len: float = info(0.40)
    rail_w: float = info(0.008)
    rail_h: float = info(0.010)
    rail_y: float = info(0.060)  # rail centerline |y|
    anvil_face: float = info(0.150)  # lane-frame x of the anvil's near (lane-side) face
    anvil_len: float = info(0.050)
    anvil_wid: float = info(0.112)
    anvil_h: float = info(0.055)  # anvil top height
    tile_len: float = info(0.120)
    tile_wid: float = info(0.048)
    tile_thick: float = info(0.016)
    tile_mass: float = info(0.06)
    tile_colors: tuple = info(((0.85, 0.10, 0.10), (0.10, 0.75, 0.15), (0.15, 0.30, 0.90)))
    # standing spawn slots, LANE frame (x along lane, |y| clear of the rails)
    spawn_slots: tuple = info(((-0.02, 0.17), (0.10, -0.17), (-0.14, 0.17)))
    lane_x_min: float = info(-0.19)  # feet must sit past the open end
    contact_offset: float = info(0.001)

    def __post_init__(self) -> None:
        c = self
        # -- the lean band is honest: only a propped rest can satisfy it --
        assert c.tile_thick < c.head_z_min, \
            "a FLAT tile's head must sit below the head-height gate (flat rejected)"
        assert c.rail_h + c.tile_thick < c.head_z_min, \
            "a tile lying across a rail must fail the head-height gate"
        assert math.degrees(math.asin(min(1.0, c.anvil_h / c.tile_len))) >= c.tilt_min_deg, \
            "any face-on-anvil-edge rest is inside the tilt band (blue seat honest)"
        assert c.tilt_max_deg < 90.0 - 5.0, \
            "a standing tile must fail the tilt band"
        assert c.foot_z_max < c.anvil_h / 2 + c.tile_thick / 2, \
            "a tile with its foot ON the anvil must fail the foot gate"
        # -- spawn slots clear of the lane corridor and of each other --
        half_diag = math.hypot(c.tile_wid, c.tile_thick) / 2
        for sx, sy in c.spawn_slots:
            assert abs(sy) - c.tile_jitter - half_diag > c.rail_y + c.rail_w / 2 + 0.005, \
                f"spawn slot ({sx},{sy}) can collide with a rail"
            assert sx + c.tile_jitter < c.anvil_face - 0.005, "slot behind the anvil face"
        for i, (ax, ay) in enumerate(c.spawn_slots):
            for bx, by in c.spawn_slots[i + 1:]:
                assert math.hypot(ax - bx, ay - by) > 2 * (half_diag + c.tile_jitter) + 0.01, \
                    "spawn slots can collide with each other"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("leaning_chain")
class LeaningChainScene(BaseScene):
    cfg: LeaningChainSceneCfg

    def __init__(self, cfg: LeaningChainSceneCfg | None = None) -> None:
        super().__init__(cfg or LeaningChainSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
        lx, ly = c.lane_pos

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.7, dynamic_friction=0.6, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "lane": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lane",
                spawn=sp["lane"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    rail_len=c.rail_len, rail_w=c.rail_w, rail_h=c.rail_h,
                    rail_y=c.rail_y, anvil_face=c.anvil_face, anvil_len=c.anvil_len,
                    anvil_wid=c.anvil_wid, anvil_h=c.anvil_h,
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(lx, ly, 0.0)),
            ),
        }
        for k, name in enumerate(TILE_ORDER):
            sx, sy = c.spawn_slots[k]
            out[f"tile_{name}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tile" + name.title(),
                spawn=sp["tile"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tile_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass=c.tile_mass, width=c.tile_wid, thick=c.tile_thick,
                    length=c.tile_len, color=c.tile_colors[k],
                    contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(lx + sx, ly + sy, c.tile_len / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n, dev = env.num_envs, env.device
        self.lane: RigidObject = env.iscene["lane"]
        self.tiles: dict[str, RigidObject] = {
            name: env.iscene[f"tile_{name}"] for name in TILE_ORDER}
        self.env_origins = env.iscene.env_origins
        # order-aware streak counters + latches (post_step), chain-tail order:
        # col 0 = red, 1 = green, 2 = blue (blue latches alone; green needs blue
        # currently seated; red needs green AND blue currently seated)
        self.seat_streak = torch.zeros(n, 3, device=dev)
        self.seat_latch = torch.zeros(n, 3, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: lane fixture at jittered position + yaw; tiles standing
        upright at a PERMUTED assignment of the lane-frame spawn slots, each with xy
        jitter + free yaw; streaks and latches zeroed."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(m, 2, device=dev)  # burn the first post-seed draw (degenerate)

        # lane fixture: base pos + jitter, uniform yaw
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.lane_yaw_deg)
        half = yaw / 2
        lane_q = torch.stack([torch.cos(half), torch.zeros(m, device=dev),
                              torch.zeros(m, device=dev), torch.sin(half)], dim=-1)
        lane_p = torch.zeros(m, 3, device=dev)
        lane_p[:, 0] = c.lane_pos[0]
        lane_p[:, 1] = c.lane_pos[1]
        lane_p[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.lane_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = lane_p + origin
        st[:, 3:7] = lane_q
        self.lane.write_root_state_to_sim(st, env_ids)

        # tiles: permuted slot assignment, standing upright, jitter + free yaw
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)  # slot index per tile
        slots = torch.tensor(c.spawn_slots, device=dev)  # (3, 2) lane frame
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        for k, name in enumerate(TILE_ORDER):
            loc = slots[perm[:, k]] + (torch.rand(m, 2, device=dev) * 2 - 1) * c.tile_jitter
            wx = lane_p[:, 0] + cy * loc[:, 0] - sy * loc[:, 1]
            wy = lane_p[:, 1] + sy * loc[:, 0] + cy * loc[:, 1]
            tyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            th = tyaw / 2
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = wx
            st[:, 1] = wy
            st[:, 2] = c.tile_len / 2 + 0.002
            st[:, 3] = torch.cos(th)
            st[:, 6] = torch.sin(th)
            st[:, 0:3] += origin
            self.tiles[name].write_root_state_to_sim(st, env_ids)

        self.seat_streak[env_ids] = 0.0
        self.seat_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "lane": self.lane.data.root_state_w[env_ids].clone(),
            "tiles": {nm: b.data.root_state_w[env_ids].clone()
                      for nm, b in self.tiles.items()},
            "seat_streak": self.seat_streak[env_ids].clone(),
            "seat_latch": self.seat_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.lane.write_root_state_to_sim(state["lane"], env_ids)
        for nm, b in self.tiles.items():
            b.write_root_state_to_sim(state["tiles"][nm], env_ids)
        self.seat_streak[env_ids] = state["seat_streak"]
        self.seat_latch[env_ids] = state["seat_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "On the floor lies a straight LANE marked by two low YELLOW guide rails. "
            "At its far end stands a raised DARK-GREY block — the ANVIL "
            f"({c.anvil_h * 1000:.0f} mm tall); a small WHITE post marks the opposite, "
            "open end. Standing upright beside the lane are three flat tiles of equal "
            f"size ({c.tile_len * 1000:.0f} x {c.tile_wid * 1000:.0f} x "
            f"{c.tile_thick * 1000:.0f} mm): one RED, one GREEN, one BLUE. The lane's "
            "position and direction and each tile's standing spot change every "
            "episode: read them by looking.\n"
            "Goal: build the frozen end state of a fallen domino run inside the lane. "
            "Each tile must end LEANING toward the anvil at an intermediate angle "
            f"(between {c.tilt_min_deg:.0f} and {c.tilt_max_deg:.0f} degrees from the "
            "floor), its lower end (foot) resting on the floor between the rails and "
            "its upper end (head) resting ON the next body in the chain: the BLUE "
            "tile's head on the ANVIL's top, the GREEN tile's head on the blue tile, "
            "and the RED tile's head on the green tile. Along the lane the order must "
            "be red, then green, then blue, with blue nearest the anvil. The chain "
            "can only be built from the anvil backwards — blue first, then green, "
            "then red — because a tile leaned onto empty lane just falls flat: each "
            "tile needs the next one already leaning to hold it up. A tile lying "
            "flat, standing vertical, leaning on a rail, leaning across the lane, or "
            "still moving counts for nothing; all three tiles must lean at once, at "
            "rest, for success."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "In the rail lane, lean the blue tile onto the anvil block, lean the "
            "green tile onto the blue one, then lean the red tile onto the green one, "
            "so all three rest tilted like fallen dominoes in red-green-blue order "
            "toward the anvil. All three must stay leaning at once; a tile flat, "
            "vertical, or out of the lane does not count."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _lane_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lane.data.root_pos_w, self.lane.data.root_quat_w

    def tile_geometry(self) -> dict[str, torch.Tensor]:
        """LANE-FRAME tile geometry, stacked in TILE_ORDER (red, green, blue):
        axis (N,3,3) = unit long axis with positive lane-z (head up), foot/head
        (N,3,3) = the two end points, still (N,3) bool."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        lp, lq = self._lane_frame()
        n = self.env.num_envs
        pos = torch.stack([b.data.root_pos_w for b in self.tiles.values()], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.tiles.values()], dim=1)
        lin = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.tiles.values()], dim=1)
        ang = torch.stack([b.data.root_ang_vel_w.norm(dim=-1)
                           for b in self.tiles.values()], dim=1)
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * 3, 3)
        axis_w = quat_apply(quat.reshape(n * 3, 4), ez)
        lq_e = lq[:, None, :].expand(n, 3, 4).reshape(n * 3, 4)
        axis = quat_apply_inverse(lq_e, axis_w).reshape(n, 3, 3)
        rel = quat_apply_inverse(lq_e, (pos - lp[:, None, :]).reshape(n * 3, 3))
        rel = rel.reshape(n, 3, 3)
        # orient the axis so its lane-z component is up (head = upper end)
        sign = torch.where(axis[:, :, 2] >= 0, 1.0, -1.0).unsqueeze(-1)
        axis = axis * sign
        foot = rel - axis * (c.tile_len / 2)
        head = rel + axis * (c.tile_len / 2)
        still = (lin < c.settle_lin) & (ang < c.settle_ang)
        return {"axis": axis, "foot": foot, "head": head, "still": still}

    def seated_now(self, geo: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """(N, 3) bool, TILE_ORDER, purely geometric + stillness: tile k leans in the
        lane — tilt in band, head up-lane, foot low, head high, in-lane, ordered, and
        its head reaching past its support's foot (tile k+1, or the anvil for blue) —
        and the tile is still. A settled pose satisfying this is a real propped rest:
        nothing else in the scene can hold it (asserted in cfg)."""
        c = self.cfg
        g = geo or self.tile_geometry()
        axis, foot, head = g["axis"], g["foot"], g["head"]
        sin_lo = math.sin(math.radians(c.tilt_min_deg))
        sin_hi = math.sin(math.radians(c.tilt_max_deg))
        tilt_ok = (axis[:, :, 2] >= sin_lo) & (axis[:, :, 2] <= sin_hi)
        up_lane = (axis[:, :, 0] > 0.25) & (axis[:, :, 1].abs() <= c.axis_y_max)
        low_foot = foot[:, :, 2] <= c.foot_z_max
        high_head = head[:, :, 2] >= c.head_z_min
        in_lane = (foot[:, :, 1].abs() <= c.lane_y_tol) \
            & (head[:, :, 1].abs() <= c.lane_y_tol) \
            & (foot[:, :, 0] >= c.lane_x_min) \
            & (foot[:, :, 0] <= c.anvil_face - 0.010)
        base = tilt_ok & up_lane & low_foot & high_head & in_lane & g["still"]
        # chain support: head reaches past the support's foot (identity + order)
        fx, hx = foot[:, :, 0], head[:, :, 0]
        sup = torch.zeros_like(base)
        sup[:, 2] = (hx[:, 2] >= c.anvil_face - 0.002) \
            & (hx[:, 2] <= c.anvil_face + c.anvil_len + 0.010) \
            & (head[:, 2, 2] >= c.anvil_h - 0.020)  # blue's head at anvil-top height
        for k in (0, 1):  # red on green, green on blue
            sup[:, k] = (fx[:, k] < fx[:, k + 1] - 0.010) \
                & (hx[:, k] >= fx[:, k + 1] + c.overlap_min) \
                & (hx[:, k] <= fx[:, k + 1] + c.tile_len)
        return base & sup

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Order-aware streak latches every physics substep: blue may latch alone;
        green only while blue is currently seated; red only while green AND blue are.
        A pose must hold `streak_need` consecutive substeps before it latches — a
        fly-through pose never counts."""
        seat = self.seated_now()
        gated = seat.clone()
        gated[:, 1] &= seat[:, 2]
        gated[:, 0] &= seat[:, 1] & seat[:, 2]
        self.seat_streak = torch.where(gated, self.seat_streak + 1.0,
                                       torch.zeros_like(self.seat_streak))
        self.seat_latch = torch.maximum(
            self.seat_latch, (self.seat_streak >= self.cfg.streak_need).float())

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: all three tiles simultaneously seated (leaning, chained,
        ordered, in-lane) and still — the frozen fallen-domino end state."""
        return self.seated_now().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.25 per chain stage ever held for `streak_need`
        substeps in order (blue on anvil; green on blue while blue holds; red on
        green while both hold), capped at 0.75; 1.0 iff success(). Latched — credit
        never evaporates; the null policy scores ~0 (tiles spawn standing off-lane)."""
        base = (0.25 * self.seat_latch.sum(dim=1)).clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="leaning_chain", robot="null", env_spacing=3.0))
