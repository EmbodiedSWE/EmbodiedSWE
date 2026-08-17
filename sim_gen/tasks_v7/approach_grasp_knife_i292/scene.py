"""KeyturnSpreaderScene — thread a paddle key through a roof slot and TWIST it a quarter
turn to cam two enclosed anchor blocks apart (seed: pick_place/approach_grasp_knife,
strategically inverted).

The seed task servos a gripper to a knife on a table, closes the jaw, and lifts; it is
judged on gripper-object distance (the manipulated object IS the judged object, and
picking it up IS the task). Here every element of that strategy is inverted:

- The judged objects (two coloured ANCHOR blocks) can never be grasped, or even touched:
  they sit inside a low roofed tunnel. The mouths are 56 mm letterboxes (a 20 mm-thick
  tool cannot pass above the 50 mm blocks through them), the only other opening is a
  30 x 98 mm roof slot whose underside sits 70 mm above the block tops (a poked finger
  reaches ~50 mm below the roof underside and dangles in free air).
- The knife-shaped object is a TOOL, not the goal: a flat steel paddle foot on a long
  square shank (a giant key). Lifting and carrying it — the seed's whole strategy —
  scores zero.
- The load-bearing interaction is a ROTATION UNDER LOAD, not a grasp: the slot admits
  the paddle only broadside (slot 30 mm along the tunnel vs paddle 88 mm), so the key
  must be lowered through the slot into the gap between the anchors and then twisted a
  quarter turn; the paddle's diagonal cams both anchors apart along the tunnel, through
  contact, to beyond `split_gap` (80 mm; the 88 mm paddle ends up broadside between
  them).
- The rubric latches an ARMED RUN: the spread only counts while both anchors remain
  continuously under the covered span of the roof, starting from the near-closed gap.
  Pulling an anchor out of a mouth breaks the run, and a run only re-arms with the
  anchors back together — so "drag the blocks out and rearrange them" can never latch
  the split. (Nothing can pull an anchor outward anyway: end pushes through the
  letterboxes only close the gap.)

Geometry honesty is asserted in `__post_init__`: the paddle fits the slot broadside with
5 mm side clearance and cannot pass turned (58 mm oversize); its swept diagonal clears
the tunnel walls; the letterbox opening minus block height (6 mm) is smaller than half
the paddle thickness (no over-the-top lever attack through a mouth); the fully cammed
anchors remain well inside the covered span; the anchors slide before they tip.

Rubric: score() = 0.10 (key paddle through the slot, latched) + 0.15 (paddle down in the
gap between the anchors, latched) + 0.45 * latched armed-run gap progress (close_gap ->
split_gap) + 0.15 (split latched at split_gap during an armed run); exactly 1.0 iff
success() = split latched AND both anchors covered now AND live gap >= live_gap AND both
anchors settled. Null policy scores 0 (the key spawns lying ~0.3 m from the slot; the
anchors rest below every threshold); latched credit never evaporates.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from robobench.core import SCENES, BaseCfg, BaseScene, EnvCfg, SimCfg, info, register_env, tunable

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from robobench.core import BaseEnv


# ----- custom compound spawner (the key: paddle foot + square shank, one rigid body) ------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_key(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the DYNAMIC key at `prim_path`. Origin = paddle-foot BOTTOM centre; local +z
    up the shank, local +y along the paddle's long dimension. Custom spawn funcs apply no
    cfg schemas, so mass (explicit CoM + diagonal inertia), damping, solver iterations and
    collision offsets are all authored here."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in cfg.diag_inertia]))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.20)
    px.CreateAngularDampingAttr(2.0)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(4)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)

    def box(tag: str, size, center) -> None:
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/{tag}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
        seg.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
        UsdPhysics.CollisionAPI.Apply(seg.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(seg.GetPrim())
        pc.CreateContactOffsetAttr(float(cfg.contact_offset))
        pc.CreateRestOffsetAttr(0.0)

    box("paddle", (cfg.bit_t, cfg.bit_l, cfg.bit_h), (0.0, 0.0, cfg.bit_h / 2))
    box("shank", (cfg.shank_w, cfg.shank_w, cfg.shank_l),
        (0.0, 0.0, cfg.bit_h + cfg.shank_l / 2))
    return root


def _key_spawner_cfg(*, bit_t: float, bit_l: float, bit_h: float, shank_w: float,
                     shank_l: float, mass: float, com_z: float, diag_inertia: tuple,
                     color: tuple, contact_offset: float) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "key" not in _SPAWNER_CACHE:

        @configclass
        class KeySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_key)
            bit_t: float = 0.020
            bit_l: float = 0.088
            bit_h: float = 0.048
            shank_w: float = 0.020
            shank_l: float = 0.210
            mass: float = 0.50
            com_z: float = 0.090
            diag_inertia: tuple = (3.0e-3, 3.0e-3, 2.2e-4)
            color: tuple = (0.62, 0.64, 0.68)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["key"] = KeySpawnerCfg

    return _SPAWNER_CACHE["key"](
        bit_t=bit_t, bit_l=bit_l, bit_h=bit_h, shank_w=shank_w, shank_l=shank_l,
        mass=mass, com_z=com_z, diag_inertia=diag_inertia, color=color,
        contact_offset=contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class KeyturnSpreaderSceneCfg(BaseCfg):
    """Config for `KeyturnSpreaderScene`. All geometry lives in the RIG-LOCAL frame: origin
    at the tunnel centre, x along the tunnel, z = 0 at the interior floor TOP. The rig is
    jittered and yawed per episode; the initial anchor gap and the key spawn pose are also
    randomized."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    close_gap: float = tunable(0.042)  # arming threshold: run arms when gap <= this, covered
    split_gap: float = tunable(0.080)  # split latch: gap >= this during an armed run (m)
    live_gap: float = tunable(0.075)  # success needs the LIVE gap >= this (hysteresis)
    x_cov: float = tunable(0.110)  # covered span: |anchor centre x| <= this (rig frame, m)
    settle_speed: float = tunable(0.10)  # max anchor |lin vel| when judging success (m/s)
    # (0.10 sits above the GPU phantom-velocity readback band, below any real motion)
    upright_max_deg: float = tunable(20.0)  # key latches require the key near-vertical

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    rig_pos: tuple = tunable((0.42, 0.0))  # nominal rig centre on the ground (m)
    rig_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the rig per episode
    rig_yaw_deg: float = tunable(20.0)  # uniform +/- yaw of the rig per episode
    gap_range: tuple = tunable((0.024, 0.034))  # initial anchor gap g0 ~ U(range) (m)
    key_pos: tuple = tunable((0.0, -0.30))  # key spawn origin, rig-local xy (m)
    key_jitter: float = tunable(0.03)  # uniform +/- xy jitter of the key spawn
    key_yaw_half_deg: float = tunable(130.0)  # shank yaw ~ 180 +/- this (points away from rig)

    # --- info: rig structure (rig-local constants; z = 0 at interior floor top) --------------
    interior_w: float = info(0.100)  # tunnel interior width (y)
    interior_h: float = info(0.120)  # floor top -> roof underside
    wall_t: float = info(0.012)
    roof_t: float = info(0.010)
    floor_t: float = info(0.012)
    rig_x_half: float = info(0.170)  # outer x half-extent
    lip_t: float = info(0.012)  # mouth lip thickness (x)
    lip_bottom: float = info(0.056)  # letterbox opening height at each mouth
    slot_x_half: float = info(0.015)  # roof slot: 30 mm along the tunnel ...
    slot_y_half: float = info(0.049)  # ... x 98 mm across it
    block_size: tuple = info((0.060, 0.090, 0.050))  # anchor block (x, y, z)
    block_mass: float = info(0.25)
    block_home: float = info(0.030)  # anchor centre |x| = this + g0/2 at reset
    bit_t: float = info(0.020)  # paddle thickness (the slot's admitting dimension)
    bit_l: float = info(0.088)  # paddle length (across the tunnel at insertion)
    bit_h: float = info(0.048)  # paddle height (2 mm under the block tops)
    shank_w: float = info(0.020)
    shank_l: float = info(0.210)  # key top = bit_h + shank_l = 258 mm (128 mm above roof)
    key_mass: float = info(0.50)
    key_com_z: float = info(0.090)  # authored CoM height above the paddle bottom
    key_diag_inertia: tuple = info((3.0e-3, 3.0e-3, 2.2e-4))  # summed-boxes estimate
    contact_offset: float = info(0.002)
    color_rig: tuple = info((0.30, 0.30, 0.34))
    color_roof: tuple = info((0.95, 0.62, 0.08))
    color_left: tuple = info((0.82, 0.12, 0.10))  # crimson anchor (-x)
    color_right: tuple = info((0.10, 0.30, 0.85))  # blue anchor (+x)
    color_key: tuple = info((0.62, 0.64, 0.68))

    # Derived (filled in __post_init__).
    pieces: tuple = field(default=None, init=False)  # ((name, size, local centre, rgb), ...)
    z_off: float = field(default=None, init=False)  # local z=0 above the ground plane
    lip_inner_x: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        iw, ih = self.interior_w, self.interior_h
        wt, rt, ft = self.wall_t, self.roof_t, self.floor_t
        xh = self.rig_x_half
        wy = iw / 2 + wt / 2  # wall centreline |y|
        oy = iw / 2 + wt  # outer |y|
        sx, sy = self.slot_x_half, self.slot_y_half
        roof_z = ih + rt / 2
        lip_h = ih - self.lip_bottom
        lip_cx = xh - self.lip_t / 2
        self.lip_inner_x = xh - self.lip_t
        rig, amber = self.color_rig, self.color_roof
        self.pieces = (
            ("floor", (2 * xh, 2 * oy, ft), (0.0, 0.0, -ft / 2), rig),
            ("wall_yn", (2 * xh, wt, ih), (0.0, -wy, ih / 2), rig),
            ("wall_yp", (2 * xh, wt, ih), (0.0, wy, ih / 2), rig),
            # roof: two big slabs beyond the slot + two thin strips flanking it (the slot
            # is the 2*sx x 2*sy hole between them)
            ("roof_xn", (xh - sx, 2 * oy, rt), (-(sx + (xh - sx) / 2), 0.0, roof_z), amber),
            ("roof_xp", (xh - sx, 2 * oy, rt), (sx + (xh - sx) / 2, 0.0, roof_z), amber),
            ("roof_yn", (2 * sx, oy - sy, rt), (0.0, -(sy + (oy - sy) / 2), roof_z), amber),
            ("roof_yp", (2 * sx, oy - sy, rt), (0.0, sy + (oy - sy) / 2, roof_z), amber),
            # mouth lips: the letterboxes (opening height = lip_bottom)
            ("lip_xn", (self.lip_t, iw, lip_h), (-lip_cx, 0.0, self.lip_bottom + lip_h / 2), amber),
            ("lip_xp", (self.lip_t, iw, lip_h), (lip_cx, 0.0, self.lip_bottom + lip_h / 2), amber),
        )
        self.z_off = ft  # rig base slab bottom rests on the ground plane

        # --- honesty asserts -----------------------------------------------------------------
        bx, by, bz = self.block_size
        # The slot admits the paddle broadside with real clearance, and refuses it turned.
        assert 2 * sx - self.bit_t >= 0.008, "slot must admit the paddle thickness"
        assert 2 * sy - self.bit_l >= 0.008, "slot must admit the paddle length"
        assert self.bit_l - 2 * sx >= 0.050, "slot must REFUSE the turned paddle"
        # The paddle's swept diagonal clears the walls, and the cam stroke covers the goal.
        diag = math.hypot(self.bit_t, self.bit_l)
        assert iw - diag >= 0.004, f"paddle sweep must clear the walls ({iw - diag})"
        assert self.bit_l >= self.split_gap + 0.006, "cammed gap must exceed split_gap"
        assert self.split_gap >= self.live_gap + 0.004, "live_gap hysteresis"
        # Insertion fits the smallest sampled gap; the paddle stays under the block tops.
        assert self.gap_range[0] - self.bit_t >= 0.004, "paddle must fit the closed gap"
        assert self.gap_range[1] <= self.close_gap - 0.006, "reset must spawn ARMED"
        assert bz - self.bit_h >= 0.002, "paddle must stay below the block tops"
        # Aperture honesty: fingers through the slot dangle far above the blocks; no tool
        # slab (>= bit_t thick) passes above a block through a letterbox.
        assert ih - bz >= 0.060, "roof underside must be >= 60 mm above the block tops"
        assert self.lip_bottom - bz <= self.bit_t / 2, "letterbox must bar over-the-top levers"
        # The fully cammed anchors stay well inside the covered span, and the covered span
        # ends well before the mouths (an anchor CAN be uncovered while still inside).
        assert self.bit_l / 2 + bx / 2 <= self.x_cov - 0.030, "cammed anchors stay covered"
        assert self.x_cov + bx / 2 <= self.lip_inner_x - 0.015, "uncovered band exists inside"
        # Anchors slide before they tip (push resolved at worst-case block-top height).
        assert 0.30 < (bx / 2) / bz, "anchors must slide, not tip"
        # Blocks ride the walls with clearance above the depenetration-nudge band.
        assert (iw - by) / 2 >= 0.004, "anchor-wall clearance"
        # The key spawn can never latch anything at reset.
        assert abs(self.key_pos[1]) - self.interior_w / 2 >= 0.15, "key spawns far from the slot"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("keyturn_spreader")
class KeyturnSpreaderScene(BaseScene):
    cfg: KeyturnSpreaderSceneCfg

    def __init__(self, cfg: KeyturnSpreaderSceneCfg | None = None) -> None:
        super().__init__(cfg or KeyturnSpreaderSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the 9 kinematic rig pieces at their canonical pose (reset()
        re-places everything), the two dynamic anchors, and the compound key."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        slick = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.06, dynamic_friction=0.05, restitution=0.0)
        grippy = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.35, dynamic_friction=0.30, restitution=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
        }
        for name, size, ctr, rgb in c.pieces:
            # walls slick (anchors slide along them); floor grippy (anchors hold station)
            mat = slick if name.startswith(("wall", "lip")) else grippy
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rig_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rig_pos[0] + ctr[0], c.rig_pos[1] + ctr[1], c.z_off + ctr[2])),
            )
        for name, sgn, rgb in (("anchor_l", -1.0, c.color_left), ("anchor_r", 1.0, c.color_right)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Anchor_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=c.block_size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4,
                        max_depenetration_velocity=0.5,
                        linear_damping=0.10,
                        angular_damping=0.30,
                        sleep_threshold=0.0,
                        stabilization_threshold=0.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.block_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=grippy,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rig_pos[0] + sgn * (c.block_home + 0.015), c.rig_pos[1],
                         c.z_off + c.block_size[2] / 2 + 0.001)),
            )
        out["key"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Key",
            spawn=_key_spawner_cfg(
                bit_t=c.bit_t, bit_l=c.bit_l, bit_h=c.bit_h, shank_w=c.shank_w,
                shank_l=c.shank_l, mass=c.key_mass, com_z=c.key_com_z,
                diag_inertia=c.key_diag_inertia, color=c.color_key,
                contact_offset=c.contact_offset),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.rig_pos[0] + c.key_pos[0], c.rig_pos[1] + c.key_pos[1], 0.012)),
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
        """Grab handles + allocate the per-env rig frame, the sampled gap, and the latches."""
        super().bind(env)
        c = self.cfg
        n = env.num_envs
        dev = env.device
        self.rig: dict[str, RigidObject] = {name: env.iscene[name]
                                            for name, _s, _c, _rgb in c.pieces}
        self.anchor_l: RigidObject = env.iscene["anchor_l"]
        self.anchor_r: RigidObject = env.iscene["anchor_r"]
        self.key: RigidObject = env.iscene["key"]
        self.env_origins = env.iscene.env_origins
        self.r_pos = torch.zeros(n, 2, device=dev)  # rig centre xy (env-local)
        self.r_yaw = torch.zeros(n, device=dev)
        self.g0 = torch.zeros(n, device=dev)  # sampled initial gap (readback reference)
        # latches / run state
        self.key_in = torch.zeros(n, dtype=torch.bool, device=dev)
        self.keyed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.split_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self.run_ok = torch.zeros(n, dtype=torch.bool, device=dev)
        self.run_maxgap = torch.full((n,), c.close_gap, device=dev)

    # ----- rig-frame transforms -----------------------------------------------------------------
    def local_to_world(self, p_local: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(m, 3) rig-local points -> world, applying yaw, rig origin, env origin."""
        cy, sy = torch.cos(self.r_yaw[env_ids]), torch.sin(self.r_yaw[env_ids])
        out = torch.empty_like(p_local)
        out[:, 0] = self.r_pos[env_ids, 0] + cy * p_local[:, 0] - sy * p_local[:, 1]
        out[:, 1] = self.r_pos[env_ids, 1] + sy * p_local[:, 0] + cy * p_local[:, 1]
        out[:, 2] = p_local[:, 2] + self.cfg.z_off
        return out + self.env_origins[env_ids]

    def world_to_local(self, p_world: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points -> rig-local, all envs."""
        p = p_world - self.env_origins
        cy, sy = torch.cos(self.r_yaw), torch.sin(self.r_yaw)
        dx = p[:, 0] - self.r_pos[:, 0]
        dy = p[:, 1] - self.r_pos[:, 1]
        out = torch.empty_like(p)
        out[:, 0] = cy * dx + sy * dy
        out[:, 1] = -sy * dx + cy * dy
        out[:, 2] = p[:, 2] - self.cfg.z_off
        return out

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the rig frame (xy jitter + yaw), re-pin the 9 kinematic
        pieces, place the anchors at the sampled near-closed gap, lay the key at its
        jittered spawn with a yaw that keeps the shank clear of the rig, clear the run."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        _ = torch.rand(m, device=dev)  # burn the degenerate first post-seed draw

        self.r_pos[env_ids, 0] = c.rig_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        self.r_pos[env_ids, 1] = c.rig_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.rig_jitter
        self.r_yaw[env_ids] = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rig_yaw_deg)
        self.g0[env_ids] = (c.gap_range[0]
                            + torch.rand(m, device=dev) * (c.gap_range[1] - c.gap_range[0]))

        half = self.r_yaw[env_ids] / 2
        qw, qz = torch.cos(half), torch.sin(half)
        for name, _size, ctr, _rgb in c.pieces:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(
                torch.tensor(ctr, device=dev).expand(m, 3).clone(), env_ids)
            st[:, 3] = qw
            st[:, 6] = qz
            self.rig[name].write_root_state_to_sim(st, env_ids)

        bz = c.block_size[2] / 2 + 0.001
        for body, sgn in ((self.anchor_l, -1.0), (self.anchor_r, 1.0)):
            px = sgn * (c.block_home + self.g0[env_ids] / 2)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(
                torch.stack([px, torch.zeros(m, device=dev),
                             torch.full((m,), bz, device=dev)], dim=1), env_ids)
            st[:, 3] = qw
            st[:, 6] = qz
            body.write_root_state_to_sim(st, env_ids)

        # key: lying on its side at the jittered spawn; shank yaw points away from the rig.
        kx = c.key_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.key_jitter
        ky = c.key_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.key_jitter
        psi = math.pi + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.key_yaw_half_deg)
        # lying flat: q = qz(rig_yaw + psi) * qy(pi/2); origin (paddle-bottom centre) rests
        # half a paddle thickness up.
        yaw_w = self.r_yaw[env_ids] + psi
        hw = yaw_w / 2
        c45 = math.cos(math.pi / 4)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = self.local_to_world(
            torch.stack([kx, ky, torch.full((m,), c.bit_t / 2 + 0.002, device=dev)],
                        dim=1), env_ids)
        st[:, 3] = torch.cos(hw) * c45
        st[:, 4] = -torch.sin(hw) * c45
        st[:, 5] = torch.cos(hw) * c45
        st[:, 6] = torch.sin(hw) * c45
        self.key.write_root_state_to_sim(st, env_ids)

        self.key_in[env_ids] = False
        self.keyed[env_ids] = False
        self.split_latch[env_ids] = False
        self.run_ok[env_ids] = False
        self.run_maxgap[env_ids] = c.close_gap

    # ----- geometry queries ---------------------------------------------------------------------
    def anchors_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N,3), (N,3)) anchor centres in the rig-local frame (left, right)."""
        return (self.world_to_local(self.anchor_l.data.root_pos_w),
                self.world_to_local(self.anchor_r.data.root_pos_w))

    def gap(self) -> torch.Tensor:
        """(N,) live face-to-face gap between the anchors along the tunnel (rig frame)."""
        pl, pr = self.anchors_local()
        return (pr[:, 0] - pl[:, 0]) - self.cfg.block_size[0]

    def covered_both(self) -> torch.Tensor:
        """(N,) bool: both anchor centres inside the covered span |x| <= x_cov."""
        pl, pr = self.anchors_local()
        return (pl[:, 0].abs() <= self.cfg.x_cov) & (pr[:, 0].abs() <= self.cfg.x_cov)

    def paddle_center_local(self) -> torch.Tensor:
        """(N,3) the key's paddle centre in the rig-local frame."""
        from isaaclab.utils.math import quat_apply

        off = torch.tensor([0.0, 0.0, self.cfg.bit_h / 2],
                           device=self.env.device).expand(self.env.num_envs, 3)
        p = self.key.data.root_pos_w + quat_apply(self.key.data.root_quat_w, off)
        return self.world_to_local(p)

    def key_upright(self) -> torch.Tensor:
        """(N,) bool: key local +z within `upright_max_deg` of world-up."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(self.key.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.upright_max_deg))

    def settled(self) -> torch.Tensor:
        """(N,) bool: both anchors below `settle_speed`."""
        return ((self.anchor_l.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed)
                & (self.anchor_r.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed))

    # ----- mechanics (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Key latches + the armed-run gap bookkeeping."""
        c = self.cfg
        b = self.paddle_center_local()
        in_col = (b[:, 0].abs() <= c.slot_x_half + 0.007) & (b[:, 1].abs() <= c.slot_y_half + 0.003)
        up = self.key_upright()
        self.key_in |= in_col & up & (b[:, 2] < c.interior_h - 0.005)
        self.keyed |= in_col & up & (b[:, 2] < c.block_size[2] - 0.005)
        g = self.gap()
        cov = self.covered_both()
        self.run_ok &= cov  # an uncovered anchor breaks the run
        self.run_ok |= cov & (g <= c.close_gap)  # (re-)arm only from the near-closed gap
        self.run_maxgap = torch.where(self.run_ok, torch.maximum(self.run_maxgap, g),
                                      self.run_maxgap)
        self.split_latch |= self.run_ok & (g >= c.split_gap)

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "anchor_l": self.anchor_l.data.root_state_w[env_ids].clone(),
            "anchor_r": self.anchor_r.data.root_state_w[env_ids].clone(),
            "key": self.key.data.root_state_w[env_ids].clone(),
            "rig": {"r_pos": self.r_pos[env_ids].clone(), "r_yaw": self.r_yaw[env_ids].clone(),
                    "g0": self.g0[env_ids].clone()},
            "latches": {"key_in": self.key_in[env_ids].clone(),
                        "keyed": self.keyed[env_ids].clone(),
                        "split": self.split_latch[env_ids].clone(),
                        "run_ok": self.run_ok[env_ids].clone(),
                        "run_maxgap": self.run_maxgap[env_ids].clone()},
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.r_pos[env_ids] = state["rig"]["r_pos"]
        self.r_yaw[env_ids] = state["rig"]["r_yaw"]
        self.g0[env_ids] = state["rig"]["g0"]
        dev = self.env.device
        m = len(env_ids)
        half = self.r_yaw[env_ids] / 2
        for name, _size, ctr, _rgb in self.cfg.pieces:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = self.local_to_world(
                torch.tensor(ctr, device=dev).expand(m, 3).clone(), env_ids)
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            self.rig[name].write_root_state_to_sim(st, env_ids)
        self.anchor_l.write_root_state_to_sim(state["anchor_l"], env_ids)
        self.anchor_r.write_root_state_to_sim(state["anchor_r"], env_ids)
        self.key.write_root_state_to_sim(state["key"], env_ids)
        self.key_in[env_ids] = state["latches"]["key_in"]
        self.keyed[env_ids] = state["latches"]["keyed"]
        self.split_latch[env_ids] = state["latches"]["split"]
        self.run_ok[env_ids] = state["latches"]["run_ok"]
        self.run_maxgap[env_ids] = state["latches"]["run_maxgap"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        bx, by, bz = (v * 1000 for v in c.block_size)
        return (
            f"A low roofed tunnel ({2 * c.rig_x_half * 100:.0f} cm long, interior "
            f"{c.interior_w * 1000:.0f} mm wide and {c.interior_h * 1000:.0f} mm tall, amber "
            f"roof on charcoal walls) sits on the ground; its position and heading vary per "
            f"episode. Inside, under the middle of the roof, two anchor blocks "
            f"({bx:.0f} x {by:.0f} x {bz:.0f} mm) stand nearly touching: a CRIMSON one and a "
            f"BLUE one, visible through the low letterbox mouths at the tunnel ends "
            f"({c.lip_bottom * 1000:.0f} mm tall openings under amber lintels). The roof's "
            f"centre has a rectangular slot ({2 * c.slot_x_half * 1000:.0f} mm along the "
            f"tunnel x {2 * c.slot_y_half * 1000:.0f} mm across it), directly above the gap "
            f"between the anchors. A steel KEY lies on the ground beside the tunnel: a flat "
            f"paddle foot ({c.bit_t * 1000:.0f} mm thick, {c.bit_l * 1000:.0f} mm long, "
            f"{c.bit_h * 1000:.0f} mm tall) at the base of a long square shank "
            f"({c.shank_w * 1000:.0f} mm square, key {(c.bit_h + c.shank_l) * 1000:.0f} mm "
            f"tall overall).\n"
            f"Goal: drive the crimson and blue anchors apart along the tunnel until the gap "
            f"between them exceeds {c.split_gap * 1000:.0f} mm, WITHOUT either anchor leaving "
            f"the covered middle of the tunnel (each anchor must stay within "
            f"{c.x_cov * 1000:.0f} mm of the tunnel centre, and the spreading itself only "
            f"counts while both are under cover — dragging an anchor out of a mouth and "
            f"re-arranging does not count, and any re-attempt must start again from a "
            f"near-closed gap). The anchors are out of reach: the mouths are too low to reach "
            f"over them and the roof slot is too deep for fingers. Use the key: hold it "
            f"vertical, align the paddle with the slot's long axis (the slot only admits the "
            f"paddle in that orientation), lower the paddle through the slot into the gap "
            f"between the anchors until it rests on the floor, then TWIST the shank a quarter "
            f"turn about the vertical. The paddle cams both anchors apart as it turns. Leave "
            f"the anchors spread and at rest; the key may stay in the slot."
        )

    def instruction(self) -> str:
        return (
            "Lower the steel key's paddle through the roof slot into the gap between the "
            "crimson and blue anchors inside the tunnel, then twist it a quarter turn so the "
            "paddle forces the two anchors more than 80 mm apart. Both anchors must stay "
            "under the tunnel roof the whole time; spreading them any other way, or letting "
            "one leave the covered span, fails."
        )

    # ----- progress / rubric ----------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: split latched during an armed covered run, AND both anchors covered
        now, AND the live gap still exceeds `live_gap`, AND both anchors settled."""
        return (self.split_latch & self.covered_both() & (self.gap() >= self.cfg.live_gap)
                & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.10 key-through-slot + 0.15 paddle-in-gap + 0.45 * armed-run
        gap progress (close_gap -> split_gap, latched max) + 0.15 split latch; exactly 1.0 iff
        success(). Latched credit never evaporates; the null policy scores 0."""
        c = self.cfg
        prog = ((self.run_maxgap - c.close_gap) / (c.split_gap - c.close_gap)).clamp(0.0, 1.0)
        s = (0.10 * self.key_in.float() + 0.15 * self.keyed.float()
             + 0.45 * prog + 0.15 * self.split_latch.float())
        return torch.where(self.success(), torch.ones_like(s), s)


register_env("simgen", lambda: EnvCfg(scene="keyturn_spreader", robot="null", env_spacing=3))
