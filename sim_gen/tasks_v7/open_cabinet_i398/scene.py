"""NookCabinetScene — drag the cabinet out of the wall nook, swing its door, retrieve
the keepsake cube (sim_gen task `open_cabinet_i398`).

Derived from mujoco_playground/open_cabinet ("open the cabinet": grasp the cabinet
handle and pull it along its articulation to a floating target position), but the
MANIPULATION MODEL is replaced wholesale. The seed's plan is ONE constrained-joint
pull on an articulation that is free to move from step zero, and the articulation's
own pose IS the goal. Here the door's hinge is equally free from step zero — and
completely useless at first: the cabinet spawns parked face-first in front of a
fixed WALL with only ~2 cm of daylight, so the door physically cannot swing (the
knob strikes the wall after a few degrees; smoke-verified with the same torque that
freely opens an unblocked door). Nothing is locked and no mechanism gates anything:
the blocker is the SPATIAL RELATION between the furniture and the room. The plan is

  (1) reposition the WHOLE CABINET — grab the roof carry-bar and drag/pivot the
      free-standing cabinet backwards across the floor until real swing clearance
      exists in front of its face;
  (2) swing the door open (vertical hinge, neutral under gravity — it parks
      wherever damping leaves it) past `open_min_deg`;
  (3) reach through the mouth, extract the keepsake cube, and set it on the
      delivery mat — while leaving the cabinet standing upright.

The articulation's motion is instrumental, never the goal; the scored object is a
free cube sealed behind it; and the load-bearing skill — moving the container
itself to un-block its own door — exists nowhere in the seed (whose cabinet is
`fix_base_link=True`: it CANNOT move).

Assets are fully procedural (compound-spawner pattern; custom spawners author
MassAPI + friction material explicitly — cfg schemas do not apply to them):
  - wall: STATIC slab (the room feature; never moves, world anchor of the nook);
  - cabinet: DYNAMIC compound — floor/roof/back/two sides around an open front
    mouth (interior 150 x 200 x 150 mm), plus a roof carry-bar on two posts
    (30 mm finger clearance); ~1.8 kg, slides on the floor under ~5-8 N;
  - door: DYNAMIC compound (panel + knob) on a spawn-authored RevoluteJoint to
    the sibling cabinet (vertical axis through the front-left edge, limits
    [-2 deg, +150 deg]); angular damping parks it where released;
  - cube: the 45 mm keepsake, sealed behind the door;
  - mat: KINEMATIC delivery pad, teleported per episode.

Per-episode randomization (readback-verifiable): cabinet slot along the wall
(y +/- 0.12 m), nose gap to the wall, small yaw; cube pose inside the chamber;
mat side (left/right of the workspace) AND xy.

Rubric (0..1; latched partial credit anchored in the demonstrated solve):
  0.10 * moved — cabinet ever displaced >= `moved_min` from its spawn slot
  0.20 * clear — swing clearance in front of the face ever >= `clear_min`
  0.20 * open  — door ever past `open_min_deg` (slow)
  0.25 * out   — cube ever outside the cabinet chamber (slow)
  1.0 iff success(): cube settled ON the mat + door >= `open_min_deg` + cabinet
  standing UPRIGHT at floor level + everything settled and finite. Non-success
  capped at 0.75; ~0 for the null policy.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- custom compound spawners ------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _root_xform(prim_path: str, translation, orientation):
    """Define an Xform root and author its (idempotent, single) translate/orient ops."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return stage, xform.GetPrim()


def _make_collide(contact_offset: float, mu: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)
        # Friction material, authored per prim (custom spawners apply no cfg schemas).
        stage = prim.GetStage()
        mat_path = str(prim.GetPath()) + "_mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(float(mu))
        pm.CreateDynamicFrictionAttr(float(mu) * 0.9)
        pm.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in [s / 1.0 for s in size]]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _apply_mass(root, mass: float) -> None:
    from pxr import UsdPhysics

    m = UsdPhysics.MassAPI.Apply(root)
    m.CreateMassAttr(float(mass))


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC shell: floor/roof/back/two sides around an open +x mouth, plus the
    roof carry-bar on two posts. Local frame: origin at the shell centre,
    mid-height; +x points out of the mouth."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateMaxDepenetrationVelocityAttr(0.5)
    rb.CreateLinearDampingAttr(0.2)
    rb.CreateAngularDampingAttr(0.5)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(4)
    _apply_mass(root, cfg.cab_mass)
    c = cfg
    collide = _make_collide(c.contact_offset, c.cab_mu)
    t = c.panel_t
    hd, hw, hh = c.inner_d / 2, c.inner_w / 2, c.inner_h / 2  # 0.075, 0.100, 0.075
    fp_x = 2 * hd + t          # footprint x extent (back wall included)
    fp_cx = -t / 2             # its centre offset
    wood, wood2 = c.cab_color, c.cab_color2
    _add_box(stage, f"{prim_path}/floor", center=(fp_cx, 0.0, -(hh + t / 2)),
             size=(fp_x, 2 * (hw + t), t), color=wood, collide=collide)
    _add_box(stage, f"{prim_path}/roof", center=(fp_cx, 0.0, +(hh + t / 2)),
             size=(fp_x, 2 * (hw + t), t), color=wood, collide=collide)
    _add_box(stage, f"{prim_path}/back", center=(-(hd + t / 2), 0.0, 0.0),
             size=(t, 2 * (hw + t), 2 * hh), color=wood2, collide=collide)
    for sy in (+1.0, -1.0):
        _add_box(stage, f"{prim_path}/side_{'l' if sy > 0 else 'r'}",
                 center=(fp_cx, sy * (hw + t / 2), 0.0),
                 size=(fp_x, t, 2 * hh), color=wood2, collide=collide)
    # roof carry-bar (grasp feature for dragging): two posts + bar, 30 mm clearance
    bar_z0 = hh + t  # roof top, local z
    for sy in (+1.0, -1.0):
        _add_box(stage, f"{prim_path}/post_{'l' if sy > 0 else 'r'}",
                 center=(fp_cx, sy * c.bar_span / 2, bar_z0 + c.bar_clear / 2),
                 size=(0.014, 0.014, c.bar_clear), color=c.bar_color, collide=collide)
    _add_box(stage, f"{prim_path}/bar",
             center=(fp_cx, 0.0, bar_z0 + c.bar_clear + 0.007),
             size=(0.014, c.bar_span + 0.014, 0.014), color=c.bar_color, collide=collide)
    return root


def _spawn_door(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC door on a spawn-authored RevoluteJoint to the sibling cabinet.
    Door local frame: ORIGIN ON THE HINGE AXIS (vertical, +z); the panel extends
    -y door-local; +theta swings the panel outward (through the mouth's front
    half-space). Closed (0 deg) the door frame coincides with the cabinet frame
    translated to the hinge point."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    rb.CreateMaxDepenetrationVelocityAttr(0.5)
    rb.CreateLinearDampingAttr(0.1)
    # Heavy angular damping = a STIFF hinge: the door parks instantly where
    # released, and a cabinet drag cannot ratchet it open by inertia (the
    # integrated fling angle scales ~ m*r*dv/(I*d) — at d=12 a full drag leaves
    # <~10 deg). Opening it takes a deliberate sustained ~0.03 N m push, which a
    # fingertip on the knob supplies trivially (~0.2 N at the knob radius).
    rb.CreateAngularDampingAttr(12.0)
    rb.CreateSolverPositionIterationCountAttr(16)
    rb.CreateSolverVelocityIterationCountAttr(4)
    _apply_mass(root, cfg.door_mass)
    c = cfg
    collide = _make_collide(c.contact_offset, 0.3)
    _add_box(stage, f"{prim_path}/panel",
             center=(0.0, -c.door_w / 2, 0.0),
             size=(c.door_t, c.door_w, c.door_h), color=c.door_color, collide=collide)
    _add_box(stage, f"{prim_path}/knob",
             center=(c.door_t / 2 + c.knob_len / 2, -(c.door_w - 0.030), 0.0),
             size=(c.knob_len, 0.016, 0.016), color=c.knob_color, collide=collide)
    base = prim_path.rsplit("/", 1)[0]
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("Z")
    j.CreateCollisionEnabledAttr(False)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x), float(c.hinge_y), 0.0))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(c.door_lo_deg))
    j.CreateUpperLimitAttr(float(c.door_hi_deg))
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class CabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            inner_d: float = 0.150
            inner_w: float = 0.200
            inner_h: float = 0.150
            panel_t: float = 0.012
            bar_span: float = 0.090
            bar_clear: float = 0.030
            cab_mass: float = 1.8
            cab_mu: float = 0.25
            cab_color: tuple = (0.45, 0.30, 0.16)
            cab_color2: tuple = (0.52, 0.36, 0.20)
            bar_color: tuple = (0.10, 0.10, 0.10)
            contact_offset: float = 0.002

        @configclass
        class DoorSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_door)
            door_w: float = 0.212
            # Panel height covers exactly the interior mouth (floor-top to
            # roof-bottom): 12 mm ground clearance so the door never scrapes the
            # floor while the cabinet is dragged (ground drag ratchets the free
            # hinge open).
            door_h: float = 0.150
            door_t: float = 0.012
            knob_len: float = 0.030
            hinge_x: float = 0.083
            hinge_y: float = 0.106
            door_lo_deg: float = -2.0
            door_hi_deg: float = 150.0
            door_mass: float = 0.30
            door_color: tuple = (0.20, 0.42, 0.62)
            knob_color: tuple = (0.05, 0.05, 0.05)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(cabinet=CabinetSpawnerCfg, door=DoorSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class NookCabinetSceneCfg(BaseCfg):
    """Config for `NookCabinetScene`. The nook geometry is honest by construction
    (asserted in __post_init__): the widest sampled nose gap blocks the door far
    below `open_min_deg`, and `clear_min` is enough daylight for the door to pass
    `open_min_deg` with margin."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    open_min_deg: float = tunable(45.0)   # door counts as open past this
    moved_min: float = tunable(0.05)      # cabinet displacement latch (m)
    clear_min: float = tunable(0.25)      # swing clearance latch: face-to-wall (m)
    mat_xy_tol: float = tunable(0.055)    # cube centre within this of the mat centre, per axis
    mat_z_tol: float = tunable(0.012)     # cube centre height window on the mat
    upright_max_deg: float = tunable(10.0)  # cabinet up-axis within this of world-up
    upright_z_tol: float = tunable(0.015)  # cabinet centre at floor rest height within this
    settle_speed: float = tunable(0.08)   # max |lin vel| when judging (m/s)
    door_settle_w: float = tunable(0.8)   # max door |ang vel| when judging (rad/s)
    latch_speed: float = tunable(0.15)    # calm gate for the out/open latches
    # actuation safety caps (post_step owns the wrench slots)
    cab_vmax: float = tunable(0.30)       # cabinet drag velocity cap (m/s)
    door_wmax: float = tunable(2.0)       # door swing angular velocity cap (rad/s)
    cube_vmax: float = tunable(0.25)      # cube push velocity cap (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    slot_y_amp: float = tunable(0.12)     # cabinet slot along the wall (+/- m)
    gap_lo: float = tunable(0.018)        # nose gap: knob face to wall, sampled window (m)
    gap_hi: float = tunable(0.028)
    yaw_amp_deg: float = tunable(2.0)     # cabinet spawn yaw (+/- deg)
    cube_x_lo: float = tunable(-0.045)    # cube spawn window, cabinet-local
    cube_x_hi: float = tunable(0.010)
    cube_y_amp: float = tunable(0.050)
    mat_x_lo: float = tunable(0.10)       # mat spawn window (world)
    mat_x_hi: float = tunable(0.25)
    mat_y_lo: float = tunable(0.33)       # |y| window; side sampled per episode
    mat_y_hi: float = tunable(0.40)

    # --- info: room layout -----------------------------------------------------------------------
    wall_x: float = info(0.70)            # wall INNER plane (world x)
    wall_t: float = info(0.06)
    wall_len: float = info(1.60)
    wall_h: float = info(0.40)
    wall_color: tuple = info((0.62, 0.60, 0.56))
    # --- info: cabinet geometry (keep in sync with the spawner defaults) -------------------------
    inner_d: float = info(0.150)          # interior depth (x, mouth axis)
    inner_w: float = info(0.200)          # interior width (y)
    inner_h: float = info(0.150)          # interior height (z)
    panel_t: float = info(0.012)
    bar_span: float = info(0.090)
    bar_clear: float = info(0.030)
    cab_mass: float = info(1.8)
    cab_mu: float = info(0.25)
    ground_mu: float = info(0.30)
    # --- info: door ------------------------------------------------------------------------------
    door_w: float = info(0.212)
    door_h: float = info(0.150)
    door_t: float = info(0.012)
    knob_len: float = info(0.030)
    hinge_x: float = info(0.083)          # hinge point, cabinet frame
    hinge_y: float = info(0.106)
    door_lo_deg: float = info(-2.0)
    door_hi_deg: float = info(150.0)
    door_mass: float = info(0.30)
    # --- info: cube / mat ------------------------------------------------------------------------
    cube_s: float = info(0.045)
    cube_mass: float = info(0.060)
    mat_s: float = info(0.150)
    mat_t: float = info(0.008)
    contact_offset: float = info(0.002)
    # rubric weights (0.10 + 0.20 + 0.20 + 0.25 = 0.75 = the non-success cap)
    w_moved: float = info(0.10)
    w_clear: float = info(0.20)
    w_open: float = info(0.20)
    w_out: float = info(0.25)

    # Derived (filled in __post_init__).
    half_h: float = info(0.0)             # shell half height (rest centre z)
    half_w: float = info(0.0)             # shell half width
    nose_x: float = info(0.0)             # cabinet-local x of the closed door's knob face

    def __post_init__(self) -> None:
        self.half_h = self.inner_h / 2 + self.panel_t
        self.half_w = self.inner_w / 2 + self.panel_t
        # closed door: panel centre local x = hinge_x, outer face + knob
        self.nose_x = self.hinge_x + self.door_t / 2 + self.knob_len
        # Honesty-by-construction asserts.
        # (1) At the widest sampled nose gap (+ yaw slack) the door is blocked far
        #     below open_min: the knob (innermost swing radius that can strike the
        #     wall) rides at r_knob from the hinge.
        r_knob = self.door_w - 0.030
        yaw_slack = self.half_w * math.sin(math.radians(self.yaw_amp_deg))
        theta_blocked = math.degrees(math.asin(
            min(1.0, (self.gap_hi + yaw_slack + 0.004) / r_knob)))
        assert theta_blocked < 0.4 * self.open_min_deg, \
            f"nook must block the door far below open_min ({theta_blocked:.1f} deg)"
        # (2) clear_min daylight lets the door pass open_min with margin (door
        #     leading corner radius = door_w + door_t).
        need = (self.door_w + self.door_t) * math.sin(math.radians(self.open_min_deg))
        assert self.clear_min > need + 0.03, \
            f"clear_min {self.clear_min} must exceed the open_min swing reach {need:.3f}"
        # (3) the cube passes the mouth with room, and cannot pass a blocked door
        assert self.cube_s < self.inner_w - 0.08 and self.cube_s < self.inner_h - 0.05
        # (4) mat spawns clear of the nook slot band
        assert self.mat_y_lo - self.mat_s / 2 > self.slot_y_amp + self.half_w + 0.02, \
            "mat band must not overlap the cabinet spawn band"


# ----- small quaternion helpers (wxyz, torch, batched) -------------------------------------------
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


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("nook_cabinet")
class NookCabinetScene(BaseScene):
    cfg: NookCabinetSceneCfg

    def __init__(self, cfg: NookCabinetSceneCfg | None = None) -> None:
        super().__init__(cfg or NookCabinetSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=c.ground_mu, dynamic_friction=c.ground_mu * 0.9,
                        restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "wall": AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Wall",
                spawn=sim_utils.CuboidCfg(
                    size=(c.wall_t, c.wall_len, c.wall_h),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.3, dynamic_friction=0.3, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.wall_color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(c.wall_x + c.wall_t / 2, 0.0, c.wall_h / 2)),
            ),
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=cls["cabinet"](),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, c.half_h)),
            ),
            "door": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Door",
                spawn=cls["door"](),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.50 + c.hinge_x, c.hinge_y, c.half_h)),
            ),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_s, c.cube_s, c.cube_s),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.1, angular_damping=0.2,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=4),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.4, dynamic_friction=0.35, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.15, 0.15)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.04)),
            ),
            "mat": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Mat",
                spawn=sim_utils.CuboidCfg(
                    size=(c.mat_s, c.mat_s, c.mat_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.7, dynamic_friction=0.65, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.55, 0.15)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.36, c.mat_t / 2)),
            ),
        }
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.door: RigidObject = env.iscene["door"]
        self.cube: RigidObject = env.iscene["cube"]
        self.mat: RigidObject = env.iscene["mat"]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self.spawn_xy = torch.zeros(n, 2, device=dev)     # cabinet spawn slot (env-local)
        self.mat_center = torch.zeros(n, 2, device=dev)   # mat centre (env-local)
        # rubric latches
        self._moved = torch.zeros(n, dtype=torch.bool, device=dev)
        self._clear = torch.zeros(n, dtype=torch.bool, device=dev)
        self._open = torch.zeros(n, dtype=torch.bool, device=dev)
        self._out = torch.zeros(n, dtype=torch.bool, device=dev)
        # wrench slots (solve/smoke write; post_step consumes with velocity caps —
        # never call set_external_force_and_torque directly)
        self.cab_f = torch.zeros(n, 3, device=dev)        # WORLD-frame force on the cabinet (N)
        self.door_tau = torch.zeros(n, device=dev)        # torque about the (vertical) hinge (N m)
        self.cube_f = torch.zeros(n, 3, device=dev)       # WORLD-frame force on the cube (N)

    def cab_door_states(self, cab_xy: torch.Tensor, cab_yaw: torch.Tensor,
                        door_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Consistent (cabinet, door) root states for a cabinet at env-local xy,
        upright at rest height, yawed `cab_yaw`, door at `door_deg` — teleport the
        LINKAGE as a whole (single-body writes get depenetrated by the joint)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = cab_xy.shape[0]
        dev = cab_xy.device
        st_c = torch.zeros(n, 13, device=dev)
        st_c[:, 0:2] = cab_xy
        st_c[:, 2] = c.half_h
        st_c[:, 3:7] = _qz(cab_yaw)
        st_d = torch.zeros(n, 13, device=dev)
        hinge = torch.tensor([c.hinge_x, c.hinge_y, 0.0], device=dev).expand(n, 3)
        st_d[:, 0:3] = st_c[:, 0:3] + quat_apply(st_c[:, 3:7], hinge)
        st_d[:, 3:7] = _qmul(st_c[:, 3:7], _qz(torch.deg2rad(door_deg)))
        return st_c, st_d

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: park the cabinet in the nook (slot along the wall + nose
        gap + small yaw), door shut, cube inside the chamber, mat on a sampled
        side; clear latches and wrench slots."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        _ = torch.rand(m, 4, device=dev)  # burn draws (first post-seed draw degenerate)

        # --- cabinet slot: nose gap to the wall + y slot + small yaw ---
        gap = c.gap_lo + torch.rand(m, device=dev) * (c.gap_hi - c.gap_lo)
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_amp_deg)
        cab_x = c.wall_x - gap - c.nose_x
        cab_y = (torch.rand(m, device=dev) * 2 - 1) * c.slot_y_amp
        cab_xy = torch.stack([cab_x, cab_y], dim=-1)
        st_c, st_d = self.cab_door_states(cab_xy, yaw, torch.zeros(m, device=dev))
        st_c[:, 0:3] += origin
        st_d[:, 0:3] += origin
        self.cabinet.write_root_state_to_sim(st_c, env_ids)
        self.door.write_root_state_to_sim(st_d, env_ids)
        self.spawn_xy[env_ids] = cab_xy

        # --- cube: inside the chamber, cabinet-local window, free yaw ---
        lx = c.cube_x_lo + torch.rand(m, device=dev) * (c.cube_x_hi - c.cube_x_lo)
        ly = (torch.rand(m, device=dev) * 2 - 1) * c.cube_y_amp
        lz = -c.inner_h / 2 + c.cube_s / 2 + 0.003
        loc = torch.stack([lx, ly, torch.full((m,), lz, device=dev)], dim=-1)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = st_c[:, 0:3] + quat_apply(st_c[:, 3:7], loc)
        st[:, 3:7] = _qmul(st_c[:, 3:7], _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi))
        self.cube.write_root_state_to_sim(st, env_ids)

        # --- mat: side sampled, xy window ---
        side = torch.where(torch.rand(m, device=dev) < 0.5,
                           torch.ones(m, device=dev), -torch.ones(m, device=dev))
        mx = c.mat_x_lo + torch.rand(m, device=dev) * (c.mat_x_hi - c.mat_x_lo)
        my = side * (c.mat_y_lo + torch.rand(m, device=dev) * (c.mat_y_hi - c.mat_y_lo))
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = mx
        st[:, 1] = my
        st[:, 2] = c.mat_t / 2
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.mat.write_root_state_to_sim(st, env_ids)
        self.mat_center[env_ids] = torch.stack([mx, my], dim=-1)

        # --- clear latches + wrench slots ---
        for latch in (self._moved, self._clear, self._open, self._out):
            latch[env_ids] = False
        self.cab_f[env_ids] = 0.0
        self.door_tau[env_ids] = 0.0
        self.cube_f[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "door": self.door.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "mat": self.mat.data.root_state_w[env_ids].clone(),
            "spawn_xy": self.spawn_xy[env_ids].clone(),
            "mat_center": self.mat_center[env_ids].clone(),
            "latches": torch.stack([self._moved[env_ids], self._clear[env_ids],
                                    self._open[env_ids], self._out[env_ids]], dim=-1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.door.write_root_state_to_sim(state["door"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.mat.write_root_state_to_sim(state["mat"], env_ids)
        self.spawn_xy[env_ids] = state["spawn_xy"]
        self.mat_center[env_ids] = state["mat_center"]
        lt = state["latches"]
        self._moved[env_ids], self._clear[env_ids] = lt[:, 0], lt[:, 1]
        self._open[env_ids], self._out[env_ids] = lt[:, 2], lt[:, 3]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A small free-standing wooden CABINET (about {2 * c.half_w * 100:.0f} cm wide, "
            f"{(c.inner_h + 2 * c.panel_t) * 100:.0f} cm tall, {c.inner_d * 100:.0f} cm deep, "
            f"with a black CARRY-BAR bridging its roof) stands parked face-first in front of "
            f"a fixed gray WALL, only about 2 cm away from it. Its single blue DOOR (vertical "
            f"hinge on one front edge, small black knob near the other) faces the wall, so "
            f"the door physically CANNOT swing open where the cabinet stands — the knob hits "
            f"the wall after a few degrees. Sealed inside the cabinet is a red KEEPSAKE CUBE "
            f"({c.cube_s * 1000:.0f} mm). A green DELIVERY MAT ({c.mat_s * 100:.0f} cm square) "
            f"lies on the floor off to one side; its side and position vary per episode, as do "
            f"the cabinet's slot along the wall and the cube's place inside.\n"
            f"Goal: get the keepsake cube onto the delivery mat. You must first MOVE THE WHOLE "
            f"CABINET — drag it by the roof carry-bar (or push it) backwards out of the nook "
            f"until there is real swing room in front of its face — then swing the door open "
            f"past {c.open_min_deg:.0f} degrees (the hinge is free and stays where you leave "
            f"it), reach through the mouth, take the cube out, and set it down centred on the "
            f"mat. Finish with the door still open at least {c.open_min_deg:.0f} degrees and "
            f"the cabinet standing UPRIGHT on the floor (not tipped over); the cube must rest "
            f"on the mat. A cube outside the cabinet but off the mat, a door left less than "
            f"{c.open_min_deg:.0f} degrees open, or a toppled cabinet all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drag the wooden cabinet backwards out of the wall nook by its roof carry-bar, "
            "swing its blue door wide open, and set the red cube from inside onto the green "
            "delivery mat. Leave the door at least 45 degrees open and the cabinet standing "
            "upright."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _cab_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points (N,3) -> cabinet body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.cabinet.data.root_quat_w,
                                  pos_w - self.cabinet.data.root_pos_w)

    def door_angle(self) -> torch.Tensor:
        """(N,) door hinge angle in rad (0 = shut, + = open), from the relative
        yaw of the door frame w.r.t. the cabinet frame."""
        qc = self.cabinet.data.root_quat_w
        qd = self.door.data.root_quat_w
        qc_inv = torch.cat([qc[:, :1], -qc[:, 1:]], dim=-1)
        qr = _qmul(qc_inv, qd)
        return 2.0 * torch.atan2(qr[:, 3], qr[:, 0])

    def front_clearance(self) -> torch.Tensor:
        """(N,) daylight between the wall inner plane and the forward-most front
        corner of the SHELL (env-local x). Negative would mean interpenetration."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        best = torch.full((n,), -torch.inf, device=dev)
        for sy in (+1.0, -1.0):
            corner = torch.tensor([c.inner_d / 2, sy * c.half_w, 0.0],
                                  device=dev).expand(n, 3)
            p = self.cabinet.data.root_pos_w + quat_apply(
                self.cabinet.data.root_quat_w, corner)
            best = torch.maximum(best, p[:, 0] - self.env_origins[:, 0])
        return c.wall_x - best

    def cube_inside(self) -> torch.Tensor:
        """(N,) bool: cube centre inside the (slightly padded) chamber volume, in
        the cabinet body frame."""
        c = self.cfg
        p = self._cab_local(self.cube.data.root_pos_w)
        return (p[:, 0] > -c.inner_d / 2 - 0.02) & (p[:, 0] < c.inner_d / 2 + 0.05) \
            & (p[:, 1].abs() < c.half_w + 0.03) & (p[:, 2].abs() < c.inner_h / 2 + 0.06)

    def cab_upright(self) -> torch.Tensor:
        """(N,) bool: cabinet up-axis within `upright_max_deg` of world-up AND its
        centre at floor rest height (not perched, not toppled)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.cabinet.data.root_quat_w, ez)
        ok_tilt = up[:, 2] >= math.cos(math.radians(c.upright_max_deg))
        ok_z = (self.cabinet.data.root_pos_w[:, 2] - self.env_origins[:, 2]
                - c.half_h).abs() < c.upright_z_tol
        return ok_tilt & ok_z

    def cube_on_mat(self) -> torch.Tensor:
        """(N,) bool: cube centred on the mat (per-axis window), resting at mat-top
        height, slow."""
        c = self.cfg
        p = self.cube.data.root_pos_w - self.env_origins
        exy = (p[:, :2] - self.mat_center).abs()
        z_ref = c.mat_t + c.cube_s / 2
        return (exy[:, 0] < c.mat_xy_tol) & (exy[:, 1] < c.mat_xy_tol) \
            & ((p[:, 2] - z_ref).abs() < c.mat_z_tol) \
            & (self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.cabinet.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.door.data.root_ang_vel_w.norm(dim=-1) < c.door_settle_w)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Apply the wrench slots with velocity caps (bang-bang: force gated off
        above the cap so probes cannot slingshot), then latch rubric progress."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        n, dev = self.env.num_envs, self.env.device
        zero3 = torch.zeros(n, 1, 3, device=dev)
        # cabinet: WORLD-frame drag force, re-encoded into the body frame per step
        gate = (self.cabinet.data.root_lin_vel_w.norm(dim=-1) < c.cab_vmax).float()
        fb = quat_apply_inverse(self.cabinet.data.root_quat_w,
                                self.cab_f * gate.unsqueeze(-1))
        self.cabinet.set_external_force_and_torque(fb.unsqueeze(1), zero3.clone())
        # door: torque about its (vertical) body z axis
        wgate = (self.door.data.root_ang_vel_w.norm(dim=-1) < c.door_wmax).float()
        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 2] = self.door_tau * wgate
        self.door.set_external_force_and_torque(zero3.clone(), tq)
        # cube: WORLD-frame push force, re-encoded per step
        vgate = (self.cube.data.root_lin_vel_w.norm(dim=-1) < c.cube_vmax).float()
        fb = quat_apply_inverse(self.cube.data.root_quat_w,
                                self.cube_f * vgate.unsqueeze(-1))
        self.cube.set_external_force_and_torque(fb.unsqueeze(1), zero3.clone())

        # --- latches ---
        cab_xy = (self.cabinet.data.root_pos_w - self.env_origins)[:, :2]
        self._moved |= (cab_xy - self.spawn_xy).norm(dim=-1) > c.moved_min
        cab_slow = self.cabinet.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._clear |= (self.front_clearance() > c.clear_min) & cab_slow
        door_slow = self.door.data.root_ang_vel_w.norm(dim=-1) < c.door_wmax
        self._open |= (self.door_angle() > math.radians(c.open_min_deg)) & door_slow
        cube_slow = self.cube.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self._out |= (~self.cube_inside()) & cube_slow \
            & torch.isfinite(self.cube.data.root_pos_w).all(dim=-1)

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: cube settled ON the mat + door open past `open_min_deg` +
        cabinet standing upright at floor level + everything settled and finite.
        All clauses are live physical outcomes."""
        finite = torch.isfinite(self.cube.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.cabinet.data.root_pos_w).all(dim=-1) \
            & torch.isfinite(self.door.data.root_pos_w).all(dim=-1)
        door_open = self.door_angle() > math.radians(self.cfg.open_min_deg)
        return self.cube_on_mat() & door_open & self.cab_upright() \
            & self.settled() & finite

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.10*moved + 0.20*clear + 0.20*open + 0.25*out
        (all latched; ~0 for the null policy), capped at 0.75 — and exactly 1.0
        iff success() holds live."""
        c = self.cfg
        base = (c.w_moved * self._moved.float() + c.w_clear * self._clear.float()
                + c.w_open * self._open.float()
                + c.w_out * self._out.float()).clamp(max=0.75)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="nook_cabinet", robot="null"))
