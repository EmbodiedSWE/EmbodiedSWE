"""PackedToteScene — make room in a packed tote, then stand the blue can on its floor
(sim_gen task `living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket_i327`).

Derived from libero_90/living_room_scene2 "pick up the alphabet soup and put it in the
basket", but STRATEGICALLY different: the seed's container is EMPTY — its entire
strategy is one gravity drop (carry the can over the basket, release anywhere inside
the rim, bbox containment fires). Here the container starts FULL. A high-walled tote
holds two red blocks parked side by side across its middle; the side gaps they leave
to the walls are all NARROWER than the can (readback-asserted at reset), so the seed's
move — hover over the container and let go — lands the can STANDING ON TOP OF THE
BLOCKS, a stable perch 6 cm above the floor that the rubric rejects (the success
z-window reads the can's base on the tote FLOOR). To finish, the solver must first
RE-ARRANGE THE CONTAINER'S EXISTING CONTENTS — slide the block pair to one end of the
tote (or stack one block on the other), against sliding friction, without ejecting
either block — and only then insert the can upright into the floor space it created.
The judged end state is an occupancy readout: blue can upright ON THE TOTE FLOOR
(tote-frame z window that physically excludes block-top perches, ground level, and
lying cans), both red blocks still stowed inside the tote, everything settled.

Strategy vs the corpus (tasks read in full: i38 spring press-seat, pen_holder
exemplar; sibling cards read: i177 lever dump, i275 push-through drop chute, i178
capsize righting, i149 hook-hang; plus the one-line survey of every other tasks_v7
card): every same-seed sibling changes the CONTAINER'S MECHANISM (spring, lever,
flap, hook); none makes the obstacle the container's own cargo. The load-bearing
interaction here is a JOINT-FREE OCCUPANCY PUZZLE: the goal region is blocked by
free rigid bodies that must be displaced *within* the container (shoved along the
floor or stacked) while a keep-in constraint holds (blocks may not leave the tote).
No corpus task requires clearing resident objects out of the goal volume before the
target can occupy it; nearest neighbours are pile/bridge constructs, which ADD bodies
to a goal region rather than re-packing what is already there. The seed's own end
state (can released over the container's center) is constructed in smoke and REJECTED
here (it perches on the blocks).

success(): the BLUE can stands upright (axis within `upright_max_deg` of tote up) with
its center in the tote-frame slot windows — critically z - Z_F inside `slot_z_win`,
which brackets floor-standing (52.5 mm) and excludes block-top perch (112.5 mm),
lying-on-floor (33 mm) and ground-outside (40.5 mm) — AND both blocks inside the stow
volume AND can/blocks/tote settled AND the tote upright. score(): latched partial
credit 0.15 (the blue can has entered the tote volume) + 0.25 (a conservative
tote-frame readout of the blocks' floor footprints has shown a contiguous free span
>= `room_span_min` with both blocks stowed and settled — room genuinely made), capped
at 0.40; 1.0 iff success(). Latches update in `post_step`, reset in `reset`, and
travel with `get_state`/`set_state` so solver rollbacks stay consistent. The null
policy scores ~0 (the can spawns on an arc well outside the tote).

Assets are fully procedural:
  - tote (custom compound spawner, one heavy DYNAMIC body, 3 kg, zero
    sleep/stabilization thresholds): floor slab (top Z_F = 0.012) + four walls,
    interior 0.116 x 0.215 x 0.130 tall. The interior x span (116 mm) barely exceeds
    the block length (105 mm), so blocks can only re-arrange along y and their yaw is
    geometrically capped ~11 deg; walls (130 mm) overtop an upright can (105 mm).
  - two RED blocks (0.105 x 0.066 x 0.060, 0.40 kg): the resident cargo. The pair
    (132 mm wide) parks across the middle of the 215 mm floor; consolidated to one
    end they free a 79-83 mm slot >= the 66 mm can with finger margin.
  - BLUE can (r 33 mm x 105 mm, 350 g): the alphabet-soup stand-in; 66 mm diameter
    fits an ~80 mm parallel jaw.

Per-episode randomization (readback-verified in smoke): tote xy jitter + FREE yaw;
the block pair's parking offset along the tote y; tiny per-block x jitter; the can on
a jittered full-circle arc (angle + radius + free yaw) around the tote. Heavy imports
(isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- geometry constants (single source of truth: spawner + cfg asserts + rubric) -------------
R_CAN = 0.033  # can radius (66 mm dia < ~80 mm parallel jaw)
L_CAN = 0.105  # can length
BLK_L = 0.105  # block x length (along the tote's short interior axis)
BLK_W = 0.066  # block y width (the packing axis)
BLK_H = 0.060  # block height
IN_X = 0.116  # tote interior x span: BLK_L + 11 mm (yaw cap), 2*R_CAN + 50 mm fingers
IN_Y = 0.215  # tote interior y span: 2*BLK_W + 83 mm consolidated slot
WALL_T = 0.015  # wall thickness
WALL_H = 0.130  # wall height above the tote floor (overtops an upright can by 25 mm)
FLOOR_T = 0.012  # floor slab thickness
Z_F = FLOOR_T  # tote floor TOP above ground (tote frame z)
RIM_Z = Z_F + WALL_H  # rim top = 0.142


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (N,3) by quaternions q (N,4), wxyz."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def encode_force(mode: int, q_ref: torch.Tensor, q_now: torch.Tensor,
                 f_world: torch.Tensor) -> torch.Tensor:
    """Pre-encode a desired WORLD-frame force for `set_external_force_and_torque`.

    Some pods rotate an applied wrench by the body's rotation since its reference
    orientation (applied = R_now * R_ref^T * arg). mode 0 passes the world force
    through unchanged; mode 1 pre-encodes with R_ref * R_now^T so the applied force
    comes out as the desired world force. Callers PROBE which mode moves the body the
    right way and lock it in (`q_ref` = readback at the reference instant)."""
    if mode == 0:
        return f_world
    return _qapply(_qmul(q_ref, _qinv(q_now)), f_world)


# ----- custom compound spawner (the tote) ------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _friction_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, size, center, color, contact_offset: float,
         material=None) -> None:
    """Author one colliding box child prim (translate -> scale, authored once —
    idempotent per prim, the duplicate-xformOp trap)."""
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


def _spawn_tote(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tote as ONE heavy DYNAMIC compound body (never kinematic, zero
    sleep/stabilization thresholds — teleports at reset must take and the fixture
    must keep simulating). Origin = floor-slab center ON THE GROUND; +y = the long
    interior (packing) axis. Custom spawn funcs apply no cfg schemas — mass and
    materials are authored HERE."""
    import omni.usd
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    floor_mat = _friction_material(stage, f"{prim_path}/floor_mat",
                                   cfg.mu_floor_s, cfg.mu_floor_d)
    wall_mat = _friction_material(stage, f"{prim_path}/wall_mat",
                                  cfg.mu_wall_s, cfg.mu_wall_d)
    light = (0.72, 0.72, 0.75)
    gray = (0.45, 0.48, 0.50)
    ox = IN_X + 2 * WALL_T  # outer x = 0.146
    oy = IN_Y + 2 * WALL_T  # outer y = 0.245
    # floor slab (top = Z_F)
    _box(stage, f"{prim_path}/floor", (ox, oy, FLOOR_T), (0.0, 0.0, FLOOR_T / 2),
         light, 0.0015, material=floor_mat)
    # x-walls (normal to x; inner faces |x| = IN_X/2)
    for tag, sx in (("xp", 1.0), ("xm", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (WALL_T, oy, WALL_H),
             (sx * (IN_X / 2 + WALL_T / 2), 0.0, Z_F + WALL_H / 2), gray, 0.0015,
             material=wall_mat)
    # y-walls (normal to y; inner faces |y| = IN_Y/2 — the packing end walls)
    for tag, sy in (("yp", 1.0), ("ym", -1.0)):
        _box(stage, f"{prim_path}/wall_{tag}", (IN_X, WALL_T, WALL_H),
             (0.0, sy * (IN_Y / 2 + WALL_T / 2), Z_F + WALL_H / 2), gray, 0.0015,
             material=wall_mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Define (once) the compound-spawner cfg class (explicit @configclass subclass of
    RigidObjectSpawnerCfg, defined lazily so the module imports app-free)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tote" not in _SPAWNER_CACHE:

        @configclass
        class ToteSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tote)
            mu_floor_s: float = 0.45
            mu_floor_d: float = 0.40
            mu_wall_s: float = 0.25
            mu_wall_d: float = 0.20

        _SPAWNER_CACHE.update(tote=ToteSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PackedToteSceneCfg(BaseCfg):
    """Config for `PackedToteScene`. The honesty knobs are asserted in `__post_init__`:
    at reset every free floor gap is NARROWER than the can (nothing succeeds without
    moving cargo), the success z-window physically separates floor-standing from the
    block-top perch / lying / ground-outside states, the consolidated blocks free a
    slot that genuinely fits the can, and both the shove force and the grasp are
    inside a Franka parallel-jaw envelope."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    slot_x_max: float = tunable(0.030)  # |x| of the can center (tote frame)
    slot_y_max: float = tunable(0.085)  # |y| of the can center
    slot_z_win: tuple = tunable((0.044, 0.062))  # can center z - Z_F (floor-standing: 0.0525)
    upright_max_deg: float = tunable(15.0)  # can axis vs tote up
    room_span_min: float = tunable(0.072)  # L2: contiguous free floor span (>= can + margin)
    span_margin: float = tunable(0.002)  # conservative pad on each block's y footprint
    stow_x_max: float = tunable(0.052)  # block stow volume (tote frame)
    stow_y_max: float = tunable(0.100)
    stow_z_win: tuple = tunable((0.020, 0.120))  # flat: 0.042, on-end: 0.0645, stacked: 0.102
    in_x_max: float = tunable(0.062)  # loose in-tote volume (L1 credit)
    in_y_max: float = tunable(0.112)
    in_z_win: tuple = tunable((0.010, 0.155))
    tote_upright_max_deg: float = tunable(10.0)
    settle_lin: float = tunable(0.05)  # settle gates (m/s, rad/s)
    settle_ang: float = tunable(1.0)

    # --- tunable: randomization (the task-family knobs) ----------------------------------------
    tote_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the tote at reset (m)
    tote_yaw_max: float = tunable(180.0)  # uniform +/- tote yaw (deg; FREE heading)
    pair_off_max: float = tunable(0.012)  # uniform +/- parking offset of the block pair (m)
    blk_x_jitter: float = tunable(0.001)  # uniform +/- per-block x jitter (m)
    blk_gap: float = tunable(0.002)  # spawn gap between the two parked blocks (m)
    can_radius: float = tunable(0.40)  # can spawn arc radius around the tote (m)
    radius_jitter: float = tunable(0.03)

    # --- info: structure -----------------------------------------------------------------------
    can_r: float = info(R_CAN)
    can_l: float = info(L_CAN)
    can_mass: float = info(0.35)
    blk_mass: float = info(0.40)
    tote_mass: float = info(3.0)  # heavy dynamic fixture; ground friction pins it vs shoves
    mu_can_s: float = info(0.35)
    mu_can_d: float = info(0.30)
    mu_blk_s: float = info(0.45)
    mu_blk_d: float = info(0.40)
    mu_floor_s: float = info(0.45)
    mu_floor_d: float = info(0.40)
    mu_wall_s: float = info(0.25)
    mu_wall_d: float = info(0.20)
    mu_ground_s: float = info(0.60)
    mu_ground_d: float = info(0.50)

    # Derived (filled in __post_init__).
    slot_free: float = field(default=None, init=False)  # nominal consolidated slot width
    shove_force: float = field(default=None, init=False)  # force to slide both blocks

    def __post_init__(self) -> None:
        self.slot_free = IN_Y - 2 * BLK_W  # 0.083
        self.shove_force = self.mu_blk_s * 9.81 * 2 * self.blk_mass  # ~3.5 N
        # -- embodiment: can fits an ~80 mm parallel jaw; fingers fit in the interior x --
        assert 2 * R_CAN < 0.08, "can must fit an ~80 mm parallel jaw"
        assert IN_X - 2 * R_CAN >= 0.045, \
            "interior x must leave >=22 mm finger slots beside a gripped can"
        # -- blocks re-arrange along y only; yaw geometrically capped --
        assert 0.008 <= IN_X - BLK_L <= 0.014, \
            "interior x must barely exceed the block length (yaw cap ~11 deg)"
        # -- reset is genuinely blocked: every free floor gap is narrower than the can --
        side_gap_max = IN_Y / 2 - (BLK_W + self.blk_gap / 2) + self.pair_off_max
        assert side_gap_max <= 2 * R_CAN - 0.008, \
            f"widest reset side gap {side_gap_max * 1000:.1f} mm must undercut the can dia"
        side_gap_min = IN_Y / 2 - (BLK_W + self.blk_gap / 2) - self.pair_off_max
        assert side_gap_min >= 0.005, "blocks must not spawn flush against the end walls"
        assert IN_X / 2 - BLK_L / 2 - self.blk_x_jitter >= 0.004, \
            "blocks must not spawn flush against the x walls"
        # -- consolidated blocks genuinely free a can-sized slot --
        cons_span = self.slot_free - 2 * self.span_margin  # conservative readout value
        assert self.room_span_min >= 2 * R_CAN + 0.004, \
            "the L2 span gate must certify room for the can"
        assert cons_span >= self.room_span_min + 0.004, \
            f"consolidated conservative span {cons_span * 1000:.1f} mm must clear the gate"
        # -- the success z-window separates the four candidate states --
        stand_z = L_CAN / 2  # floor-standing can center - Z_F = 0.0525
        assert self.slot_z_win[0] + 0.006 <= stand_z <= self.slot_z_win[1] - 0.006, \
            "floor-standing can must sit inside the z window with margin"
        assert R_CAN <= self.slot_z_win[0] - 0.008, "lying can must fall below the window"
        assert L_CAN / 2 - Z_F <= self.slot_z_win[0] - 0.002, \
            "ground-standing can (outside the tote) must fall below the window"
        assert BLK_H + L_CAN / 2 >= self.slot_z_win[1] + 0.03, \
            "block-top perch must sit far above the window"
        # -- walls overtop the can (drop must enter from above; can is captive) --
        assert Z_F + L_CAN <= RIM_Z - 0.02, "an upright can must sit below the rim"
        assert RIM_Z - (Z_F + BLK_H) >= 0.05, "blocks cannot be shoved over the rim"
        # -- forces inside the arm envelope; the tote holds still under them --
        assert self.shove_force <= 10.0, "sliding both blocks must stay a light push"
        assert self.mu_ground_s * self.tote_mass * 9.81 >= 2 * self.shove_force + 5.0, \
            "ground friction must pin the tote against the shove reaction"
        # -- null policy scores 0: the can spawns well clear of the tote --
        circ = math.hypot(IN_X / 2 + WALL_T, IN_Y / 2 + WALL_T)
        assert self.can_radius - self.radius_jitter >= \
            circ + self.tote_jitter + R_CAN + 0.10, \
            "can spawn arc must start well outside the tote footprint"
        # -- stow volume consistent with block geometry --
        assert self.stow_z_win[0] < Z_F + BLK_H / 2 < self.stow_z_win[1]
        assert self.stow_z_win[0] < Z_F + BLK_H + BLK_H / 2 < self.stow_z_win[1], \
            "a stacked block must still count as stowed"
        assert self.stow_z_win[1] < RIM_Z - 0.015, \
            "a block perched on the rim must NOT count as stowed"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("packed_tote")
class PackedToteScene(BaseScene):
    cfg: PackedToteSceneCfg

    def __init__(self, cfg: PackedToteSceneCfg | None = None) -> None:
        super().__init__(cfg or PackedToteSceneCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_depenetration_velocity=0.5, sleep_threshold=0.0,
            stabilization_threshold=0.0, linear_damping=0.05, angular_damping=0.1)
        coll = sim_utils.CollisionPropertiesCfg(contact_offset=0.0015, rest_offset=0.0)
        blk_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_blk_s, dynamic_friction=c.mu_blk_d, restitution=0.0)
        can_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.mu_can_s, dynamic_friction=c.mu_can_d, restitution=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.mu_ground_s, dynamic_friction=c.mu_ground_d,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "tote": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tote",
                spawn=spawners["tote"](
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.tote_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mu_floor_s=c.mu_floor_s, mu_floor_d=c.mu_floor_d,
                    mu_wall_s=c.mu_wall_s, mu_wall_d=c.mu_wall_d),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
        }
        for name, y0 in (("blk_a", -0.034), ("blk_b", 0.034)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Blk_" + name.split("_")[1].upper(),
                spawn=sim_utils.CuboidCfg(
                    size=(BLK_L, BLK_W, BLK_H),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.12, 0.12)),
                    physics_material=blk_mat, rigid_props=rigid, collision_props=coll,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.blk_mass)),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, y0, Z_F + BLK_H / 2 + 0.001)),
            )
        out["can_blue"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Can_blue",
            spawn=sim_utils.CylinderCfg(
                radius=c.can_r, height=c.can_l, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.35, 0.85)),
                physics_material=can_mat, rigid_props=rigid, collision_props=coll,
                mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, c.can_l / 2 + 0.003)),
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
        self.tote: RigidObject = env.iscene["tote"]
        self.blk_a: RigidObject = env.iscene["blk_a"]
        self.blk_b: RigidObject = env.iscene["blk_b"]
        self.blue: RigidObject = env.iscene["can_blue"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.l1_in_tote = torch.zeros(n, dtype=torch.bool, device=dev)
        self.l2_room = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: jitter + free-yaw the tote (heavy dynamic teleport), park the
        block pair mid-tote at a randomized y offset (all remaining floor gaps
        narrower than the can), stand the can upright on a jittered arc outside, and
        clear the score latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(3, device=dev)  # burn: first post-seed draws can be degenerate

        def write(body, pos: torch.Tensor, quat: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = pos + origin
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        # --- tote: xy jitter + free yaw ---
        txy = (torch.rand(m, 2, device=dev) * 2 - 1) * c.tote_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.tote_yaw_max)
        q_tote = _qz(yaw)
        zeros = torch.zeros(m, device=dev)
        tote_pos = torch.cat([txy, zeros.unsqueeze(-1)], dim=-1)
        write(self.tote, tote_pos, q_tote)

        # --- blocks: parked pair across the middle, offset along tote y ---
        pair_off = (torch.rand(m, device=dev) * 2 - 1) * c.pair_off_max
        for body, s in ((self.blk_a, -1.0), (self.blk_b, 1.0)):
            bx = (torch.rand(m, device=dev) * 2 - 1) * c.blk_x_jitter
            by = pair_off + s * (BLK_W / 2 + c.blk_gap / 2)
            local = torch.stack(
                [bx, by, torch.full((m,), Z_F + BLK_H / 2 + 0.001, device=dev)], dim=-1)
            write(body, tote_pos + _qapply(q_tote, local), q_tote)

        # --- can: upright on a jittered full-circle arc around the tote ---
        ang = torch.rand(m, device=dev) * 2 * math.pi
        rad = c.can_radius + (torch.rand(m, device=dev) * 2 - 1) * c.radius_jitter
        pos = torch.stack([txy[:, 0] + rad * torch.cos(ang),
                           txy[:, 1] + rad * torch.sin(ang),
                           torch.full((m,), c.can_l / 2 + 0.003, device=dev)], dim=-1)
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        write(self.blue, pos, _qz(cyaw))

        # --- score latches ---
        self.l1_in_tote[env_ids] = False
        self.l2_room[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Update the score latches every substep (monotone credit)."""
        tote_ok = self.tote_upright() & self.settled(self.tote)
        self.l1_in_tote |= self.in_tote(self.blue) & tote_ok
        self.l2_room |= (self.cleared_span() >= self.cfg.room_span_min) \
            & self.blocks_stowed() & self.settled(self.blk_a) & self.settled(self.blk_b) \
            & tote_ok

    # ----- state (full, restorable — latches travel with it) ----------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {nm: getattr(self, nm).data.root_state_w[env_ids].clone()
               for nm in ("tote", "blk_a", "blk_b", "blue")}
        out["l1"] = self.l1_in_tote[env_ids].clone()
        out["l2"] = self.l2_room[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        for nm in ("tote", "blk_a", "blk_b", "blue"):
            getattr(self, nm).write_root_state_to_sim(state[nm], env_ids)
        self.l1_in_tote[env_ids] = state["l1"]
        self.l2_room[env_ids] = state["l2"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A high-walled gray tote sits on the floor: interior "
            f"{IN_X * 100:.1f} x {IN_Y * 100:.1f} cm, walls {WALL_H * 100:.0f} cm tall. "
            f"Inside, TWO RED BLOCKS ({BLK_L * 100:.1f} x {BLK_W * 100:.1f} x "
            f"{BLK_H * 100:.1f} cm, light enough to slide with a finger push) are "
            f"parked side by side across the middle of the floor. They nearly span the "
            f"tote's short axis, so they can only be re-arranged along the long axis — "
            f"and as parked, every free patch of floor beside them is NARROWER than the "
            f"blue can. A BLUE food can ({2 * R_CAN * 100:.1f} cm across, "
            f"{L_CAN * 100:.1f} cm tall, narrow enough for a parallel jaw) stands "
            f"upright on the floor outside. The tote's position and heading, the "
            f"blocks' parking spot, and the can's position change every episode — read "
            f"the scene by looking.\n"
            f"Goal: get the BLUE can standing upright ON THE TOTE FLOOR. Dropping it "
            f"in as-is only leaves it perched on top of the red blocks, which counts "
            f"for nothing. First MAKE ROOM: slide the two blocks along the tote to one "
            f"end (or stack one on the other) so a can-sized patch of floor opens up, "
            f"then lower the can upright into that space. Both red blocks must END "
            f"INSIDE the tote — clearing them out over the wall fails the task. Finish "
            f"with the can standing upright (within {c.upright_max_deg:.0f} deg) with "
            f"its base on the tote floor, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Make room in the packed tote: slide the two red blocks to one end (or "
            "stack them), keeping both inside, then stand the blue can upright on the "
            "cleared tote floor."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def tote_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> tote frame."""
        return _qapply(_qinv(self.tote.data.root_quat_w),
                       pos_w - self.tote.data.root_pos_w)

    def can_axis_local(self) -> torch.Tensor:
        """(N,3) the can's cylinder axis (its local +z) expressed in the tote frame."""
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        axis_w = _qapply(self.blue.data.root_quat_w, ez)
        return _qapply(_qinv(self.tote.data.root_quat_w), axis_w)

    def tote_upright(self) -> torch.Tensor:
        """(N,) bool: the tote's up axis within tote_upright_max_deg of world up."""
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        up = _qapply(self.tote.data.root_quat_w, ez)
        return up[:, 2] >= math.cos(math.radians(self.cfg.tote_upright_max_deg))

    def in_tote(self, body) -> torch.Tensor:
        """(N,) bool: body center inside the loose tote volume (L1 credit)."""
        c = self.cfg
        loc = self.tote_local(body.data.root_pos_w)
        return (loc[:, 0].abs() <= c.in_x_max) & (loc[:, 1].abs() <= c.in_y_max) \
            & (loc[:, 2] >= c.in_z_win[0]) & (loc[:, 2] <= c.in_z_win[1])

    def blk_y_halfwidth(self, body) -> torch.Tensor:
        """(N,) conservative tote-frame y half-extent of a block: the support function
        of the oriented box along tote y (sum of |projected| half-axes) + margin.
        Orientation-general — flat, yawed, tipped and stacked blocks all read
        honestly."""
        q_rel = _qmul(_qinv(self.tote.data.root_quat_w), body.data.root_quat_w)
        n, dev = q_rel.shape[0], q_rel.device
        h = torch.full((n,), self.cfg.span_margin, device=dev)
        for i, half in enumerate((BLK_L / 2, BLK_W / 2, BLK_H / 2)):
            e = torch.zeros(n, 3, device=dev)
            e[:, i] = 1.0
            h = h + _qapply(q_rel, e)[:, 1].abs() * half
        return h

    def cleared_span(self) -> torch.Tensor:
        """(N,) the largest contiguous tote-floor span (along y) free of both blocks'
        conservative footprints: max of (near-wall gap, inter-block gap, far-wall
        gap)."""
        w = IN_Y / 2
        spans = []
        los, his = [], []
        for body in (self.blk_a, self.blk_b):
            y = self.tote_local(body.data.root_pos_w)[:, 1]
            h = self.blk_y_halfwidth(body)
            los.append((y - h).clamp(-w, w))
            his.append((y + h).clamp(-w, w))
        left = torch.minimum(los[0], los[1]) + w
        right = w - torch.maximum(his[0], his[1])
        mid = (torch.maximum(los[0], los[1])
               - torch.minimum(his[0], his[1])).clamp(min=0.0)
        span = torch.maximum(torch.maximum(left, right), mid)
        return span.clamp(min=0.0)

    def blocks_stowed(self) -> torch.Tensor:
        """(N,) bool: both block centers inside the stow volume (still in the tote)."""
        c = self.cfg
        out = None
        for body in (self.blk_a, self.blk_b):
            loc = self.tote_local(body.data.root_pos_w)
            ok = (loc[:, 0].abs() <= c.stow_x_max) & (loc[:, 1].abs() <= c.stow_y_max) \
                & (loc[:, 2] >= c.stow_z_win[0]) & (loc[:, 2] <= c.stow_z_win[1])
            out = ok if out is None else out & ok
        return out

    def can_in_slot(self) -> torch.Tensor:
        """(N,) bool: the blue can stands upright with its base on the tote floor —
        the z window brackets floor-standing and excludes the block-top perch,
        lying-on-floor, and ground-outside states."""
        c = self.cfg
        loc = self.tote_local(self.blue.data.root_pos_w)
        ax = self.can_axis_local()
        return (loc[:, 0].abs() <= c.slot_x_max) & (loc[:, 1].abs() <= c.slot_y_max) \
            & (loc[:, 2] - Z_F >= c.slot_z_win[0]) & (loc[:, 2] - Z_F <= c.slot_z_win[1]) \
            & (ax[:, 2] >= math.cos(math.radians(c.upright_max_deg)))

    def settled(self, body) -> torch.Tensor:
        return (body.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_lin) \
            & (body.data.root_ang_vel_w.norm(dim=-1) < self.cfg.settle_ang)

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: blue can upright on the tote floor (slot windows), both blocks
        still stowed inside the tote, can + blocks + tote settled, tote upright."""
        return self.can_in_slot() & self.blocks_stowed() \
            & self.settled(self.blue) & self.settled(self.blk_a) \
            & self.settled(self.blk_b) & self.settled(self.tote) & self.tote_upright()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: latched 0.15 (the blue can has entered the tote) +
        latched 0.25 (a can-sized floor span was genuinely cleared with both blocks
        stowed and settled), capped at 0.40; 1.0 iff success(). Monotone along the
        intended solution (enter -> make room -> seat); the null policy scores ~0
        (the can spawns far outside the tote and the parked blocks never clear a
        span)."""
        base = (0.15 * self.l1_in_tote.float()
                + 0.25 * self.l2_room.float()).clamp(max=0.40)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="packed_tote", robot="null", env_spacing=3.0))
