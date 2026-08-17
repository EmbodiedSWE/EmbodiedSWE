"""DrawerRefitScene — load and RE-INSTALL a removed drawer into the tagged bay of a
two-bay sideboard (sim_gen task `libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_i102`).

Derived from libero_90/libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet,
but STRATEGICALLY different: the seed's whole skill is one guided push on a drawer
that already rides its prismatic joint — the fixture's articulation carries the
alignment and the goal is a joint reading. Here THERE IS NO JOINT AND THE DRAWER IS
NOT IN THE CABINET: it has been pulled out completely and stands on the ground in
front of a two-bay sideboard, next to a green payload block. The plan is a
different shape entirely:

  1. LOAD — put the green block into the drawer's open-top basin while the drawer
     is still accessible (once the drawer is home, the bay roof leaves a gap
     smaller than the block: loading after closure is physically impossible).
  2. RE-RAIL — carry the loaded drawer up to the sideboard, identify the TARGET
     BAY (the one whose face frame carries the BLUE marker chip just above its
     opening — which bay is the target is resampled every episode), align the
     drawer with the face opening, and slide it home through the opening until
     its blue face plate sits FLUSH against the face frame.

The judged outcome is a 6-DOF re-installation produced by real sliding contact
(box floor on bay floor, jambs as lateral guides, the face plate on the jamb
fronts as the flush stop), with a containment clause on the payload — versus the
seed's single 1-DOF push. Pushing the drawer around on the ground (the seed's
skill executed here) scores ~0: the bays are elevated, and insertion credit is
gated on being inside the target bay.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): sideboard (KINEMATIC: plinth, jamb side walls, separator, roof,
rear wall — two identical open bays), tag chip (KINEMATIC, blue, re-posed at
reset onto the target bay's face frame), drawer (DYNAMIC compound: open-top box,
oversized blue face plate — the flush stop, protruding T-handle), payload block
(DYNAMIC green cube).

Per-episode randomization (readback-verified by smoke): sideboard yaw FREE
(+/-180 deg) + xy jitter, WHICH bay is the target (tag chip re-posed), drawer and
block ground poses (sector + jitter + free yaw).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.15  cube_l   — block ever settled inside the drawer basin (latched)
  0.10  eng_l    — drawer nose ever >= 2 cm past the target bay's face plane,
                   upright, in-lane (latched)
  0.55  ins_f    — latched max insertion fraction into the TARGET bay (gated on
                   upright + in-lane + target-bay height band)
capped at 0.80; exactly 1.0 iff success(): drawer flush in the TARGET bay
(face-plate gap < closed_tol, in-lane, at bay-floor height, upright, facing out)
with the block settled inside its basin, everything settled and finite.
Null policy ~0 (nothing is loaded, nothing is installed). The seed's strategy —
push the drawer — slides it along the ground into the plinth and scores ~0.

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


# ----- custom compound spawners -----------------------------------------------------------------
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


def _make_collide(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _span(stage, path: str, *, x, y, z, color, collide: Callable):
    """Box child from axis spans (x0, x1), (y0, y1), (z0, z1)."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _spawn_sideboard(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the sideboard at `prim_path`: KINEMATIC compound. Local frame: origin
    at the FACE plane (x=0) on the GROUND (z=0); +x runs OUT toward the apron where
    the drawer lies; the two identical bay cavities open at the face and extend to
    -x. Bay k's floor top is `bay_floors[k]`."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    W2 = c.open_w / 2                     # bay interior half-width
    Y = W2 + c.jamb_t                     # outer half-width
    D = c.bay_d + c.back_t                # outer depth
    z1, z2 = c.bay_floors                 # bay floor tops
    ztop = z2 + c.open_h + c.roof_t

    # side jamb walls (full height; their FRONT faces at x=0 are the flush stop)
    _span(stage, f"{prim_path}/jamb_p", x=(-D, 0.0), y=(W2, Y), z=(0.0, ztop),
          color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/jamb_n", x=(-D, 0.0), y=(-Y, -W2), z=(0.0, ztop),
          color=c.body_color, collide=collide)
    # plinth (solid up to the bottom bay's floor top)
    _span(stage, f"{prim_path}/plinth", x=(-D, 0.0), y=(-W2, W2), z=(0.0, z1),
          color=c.plinth_color, collide=collide)
    # separator between the bays
    _span(stage, f"{prim_path}/sep", x=(-D, 0.0), y=(-W2, W2),
          z=(z1 + c.open_h, z2), color=c.body_color, collide=collide)
    # roof
    _span(stage, f"{prim_path}/roof", x=(-D, 0.0), y=(-W2, W2),
          z=(z2 + c.open_h, ztop), color=c.body_color, collide=collide)
    # rear wall sealing both cavities
    _span(stage, f"{prim_path}/rear", x=(-D, -c.bay_d), y=(-W2, W2),
          z=(z1, z2 + c.open_h), color=c.body_color, collide=collide)

    mat = _mk_material(prim_path, "wood", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at `prim_path`: DYNAMIC compound. Local frame: origin at
    the BOX's xy centre with z=0 at the BOTTOM face; +x is the front (plate/handle)
    direction. Open-top box, oversized blue face plate (the flush stop), protruding
    yellow T-handle."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    L2, W2 = c.box_l / 2, c.box_w / 2
    _span(stage, f"{prim_path}/floor", x=(-L2, L2), y=(-W2, W2),
          z=(0.0, c.floor_t), color=c.box_color, collide=collide)
    zw = (c.floor_t, c.box_h)
    _span(stage, f"{prim_path}/w_front", x=(L2 - c.wall_t, L2), y=(-W2, W2),
          z=zw, color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(-L2, -L2 + c.wall_t), y=(-W2, W2),
          z=zw, color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(-L2, L2), y=(W2 - c.wall_t, W2),
          z=zw, color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(-L2, L2), y=(-W2, -W2 + c.wall_t),
          z=zw, color=c.box_color, collide=collide)
    # face plate: wider than the bay opening -> it CANNOT enter; it is the flush stop
    _span(stage, f"{prim_path}/plate", x=(L2, L2 + c.plate_t),
          y=(-c.plate_w / 2, c.plate_w / 2), z=(0.0, c.plate_h),
          color=c.plate_color, collide=collide)
    # T-handle: stem + crossbar, high on the plate for a clean top-down grasp
    hz = (c.handle_z, c.handle_z + c.handle_sec)
    x0 = L2 + c.plate_t
    _span(stage, f"{prim_path}/stem", x=(x0, x0 + c.stem_len),
          y=(-c.handle_sec / 2, c.handle_sec / 2), z=hz,
          color=c.handle_color, collide=collide)
    _span(stage, f"{prim_path}/bar", x=(x0 + c.stem_len, x0 + c.stem_len + c.handle_sec),
          y=(-c.bar_w / 2, c.bar_w / 2), z=hz, color=c.handle_color, collide=collide)

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.drawer_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.20)
    prb.CreateAngularDampingAttr(2.0)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    mat = _mk_material(prim_path, "wood", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "sideboard" not in _SPAWNER_CACHE:

        @configclass
        class SideboardSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_sideboard)
            open_w: float = 0.164
            open_h: float = 0.100
            bay_d: float = 0.172
            back_t: float = 0.012
            jamb_t: float = 0.040
            roof_t: float = 0.016
            bay_floors: tuple = (0.160, 0.284)
            body_color: tuple = (0.42, 0.44, 0.50)
            plinth_color: tuple = (0.28, 0.28, 0.32)
            contact_offset: float = 0.0015
            mu_s: float = 0.45
            mu_d: float = 0.40

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            box_l: float = 0.160
            box_w: float = 0.140
            box_h: float = 0.070
            floor_t: float = 0.008
            wall_t: float = 0.008
            plate_t: float = 0.012
            plate_w: float = 0.190
            plate_h: float = 0.100
            handle_z: float = 0.072
            handle_sec: float = 0.016
            stem_len: float = 0.045
            bar_w: float = 0.064
            drawer_mass: float = 0.30
            box_color: tuple = (0.62, 0.56, 0.46)
            plate_color: tuple = (0.16, 0.34, 0.80)
            handle_color: tuple = (0.85, 0.75, 0.15)
            contact_offset: float = 0.0015
            mu_s: float = 0.45
            mu_d: float = 0.40

        _SPAWNER_CACHE["sideboard"] = SideboardSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DrawerRefitSceneCfg(BaseCfg):
    """Config for `DrawerRefitScene`. The refit contract is asserted in
    `__post_init__`: the box passes the opening with honest clearance, the plate
    CANNOT pass (it is the flush stop, bearing on the jamb fronts), the plate
    stops before the box rear can hit the rear wall, the payload block fits the
    basin but NOT the roof gap over an installed drawer (loading after closure is
    physically impossible), and the tag chip sits clear of the plate's sweep."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_lin: float = tunable(0.05)      # max |lin vel| when judging (m/s)
    closed_tol: float = tunable(0.008)     # plate inner face within this of the face plane (m)
    lane_tol: float = tunable(0.016)       # drawer centred in the bay when judged (m)
    z_tol: float = tunable(0.012)          # drawer bottom within this band of the bay floor (m)
    tilt_max_deg: float = tunable(10.0)    # drawer up-axis within this of the sideboard's up
    align_max_deg: float = tunable(12.0)   # drawer +x within this of the sideboard's +x
    eng_depth: float = tunable(0.020)      # nose past the face plane by this = "engaged"

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # sideboard yaw uniform +/- (FREE heading)
    xy_jitter: float = tunable(0.05)       # sideboard xy jitter (+/- m)
    randomize_bay: bool = tunable(True)    # sample WHICH bay is the target per episode
    drawer_x: tuple = tunable((0.30, 0.40))   # drawer ground sector (sideboard-local)
    drawer_y: tuple = tunable((-0.16, -0.02))
    cube_x: tuple = tunable((0.30, 0.45))     # block ground sector (sideboard-local)
    cube_y: tuple = tunable((0.02, 0.16))

    # --- info: sideboard (local frame: origin at the face plane on the ground) -------------------
    open_w: float = info(0.164)            # bay opening / interior width
    open_h: float = info(0.100)            # bay opening / interior height
    bay_d: float = info(0.172)             # bay interior depth (rear gap: plate stops first)
    back_t: float = info(0.012)
    jamb_t: float = info(0.040)            # side wall thickness (plate bears on jamb fronts)
    roof_t: float = info(0.016)
    bay_floors: tuple = info((0.160, 0.284))  # bay floor top heights (bottom, top)
    # --- info: tag chip (marks the TARGET bay, re-posed at reset) --------------------------------
    tag_t: float = info(0.014)             # protrusion from the face
    tag_w: float = info(0.060)
    tag_h: float = info(0.018)
    tag_dz: float = info(0.016)            # chip centre above the bay opening top
    # --- info: drawer ----------------------------------------------------------------------------
    box_l: float = info(0.160)
    box_w: float = info(0.140)
    box_h: float = info(0.070)
    floor_t: float = info(0.008)
    wall_t: float = info(0.008)
    plate_t: float = info(0.012)
    plate_w: float = info(0.190)           # > open_w: the plate CANNOT enter the bay
    plate_h: float = info(0.100)
    handle_z: float = info(0.072)          # stem height above the drawer bottom
    handle_sec: float = info(0.016)
    stem_len: float = info(0.045)
    bar_w: float = info(0.064)
    drawer_mass: float = info(0.30)
    # --- info: payload block ---------------------------------------------------------------------
    cube_s: float = info(0.050)
    cube_mass: float = info(0.06)
    # --- info: materials / colors ----------------------------------------------------------------
    mu_s: float = info(0.45)
    mu_d: float = info(0.40)
    contact_offset: float = info(0.0015)
    tag_color: tuple = info((0.16, 0.34, 0.80))
    cube_color: tuple = info((0.10, 0.65, 0.15))
    # --- info: rubric weights (0.15 + 0.10 + 0.55 = 0.80 = the non-success cap) ------------------
    w_cube: float = info(0.15)
    w_eng: float = info(0.10)
    w_ins: float = info(0.55)

    def __post_init__(self) -> None:
        # the box passes the opening with honest clearance on every side
        assert (self.open_w - self.box_w) / 2 >= 0.010, "lateral clearance >= 10 mm per side"
        assert self.open_h - self.box_h >= 0.025, "vertical clearance >= 25 mm"
        # the plate is the flush stop: cannot enter, bears on the jamb fronts
        assert self.plate_w > self.open_w + 0.020, "plate must NOT pass the opening"
        assert self.plate_w / 2 < self.open_w / 2 + self.jamb_t, "plate must bear on the jambs"
        assert self.plate_h <= self.open_h + 1e-9, "plate no taller than the opening"
        # the plate stops the travel before the box rear can touch the rear wall
        assert self.bay_d > self.box_l + self.closed_tol + 0.002, "plate is the stop, not the rear wall"
        assert self.bay_d - self.box_l <= 0.030, "rear gap stays small (drawer fills its bay)"
        # payload: fits the basin, rides fully inside the box envelope...
        assert self.cube_s < self.box_w - 2 * self.wall_t - 0.020, "block fits the basin"
        assert self.floor_t + self.cube_s < self.box_h - 0.003, "block rides below the box top"
        # ...but CANNOT be loaded once the drawer is home (roof gap is sub-block,
        # and the flush plate seals the whole opening)
        assert self.cube_s > self.open_h - self.box_h + 0.005, \
            "roof gap over an installed drawer must be smaller than the block"
        assert self.plate_h >= self.open_h - 1e-9, "flush plate seals the opening"
        # a block left loose in the bay physically jams the drawer's travel
        assert self.box_l + self.cube_s > self.bay_d + self.closed_tol + 0.010, \
            "a loose block in the bay must arrest the drawer before flush"
        # tag chip sits above the opening, clear of the plate's sweep
        assert self.tag_dz - self.tag_h / 2 > self.plate_h - self.open_h + 0.001, \
            "tag chip must sit clear of the plate at flush"
        # bays identical & separated far beyond the z tolerance
        assert self.bay_floors[1] - self.bay_floors[0] > self.open_h + 0.010
        assert self.bay_floors[1] - self.bay_floors[0] > 4 * self.z_tol
        # ground sector clear of the sideboard footprint at any drawer yaw
        assert self.drawer_x[0] >= 0.28 and self.cube_x[0] >= 0.28
        # handle stays graspable at flush
        assert self.stem_len >= 0.035
        assert abs(self.w_cube + self.w_eng + self.w_ins - 0.80) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("drawer_refit")
class DrawerRefitScene(BaseScene):
    cfg: DrawerRefitSceneCfg

    def __init__(self, cfg: DrawerRefitSceneCfg | None = None) -> None:
        super().__init__(cfg or DrawerRefitSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        board_spawn = cls["sideboard"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            open_w=c.open_w, open_h=c.open_h, bay_d=c.bay_d, back_t=c.back_t,
            jamb_t=c.jamb_t, roof_t=c.roof_t, bay_floors=c.bay_floors,
            contact_offset=c.contact_offset, mu_s=c.mu_s, mu_d=c.mu_d)
        drawer_spawn = cls["drawer"](
            box_l=c.box_l, box_w=c.box_w, box_h=c.box_h, floor_t=c.floor_t,
            wall_t=c.wall_t, plate_t=c.plate_t, plate_w=c.plate_w, plate_h=c.plate_h,
            handle_z=c.handle_z, handle_sec=c.handle_sec, stem_len=c.stem_len,
            bar_w=c.bar_w, drawer_mass=c.drawer_mass,
            contact_offset=c.contact_offset, mu_s=c.mu_s, mu_d=c.mu_d)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "board": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Board", spawn=board_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "tag": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tag",
                spawn=sim_utils.CuboidCfg(
                    size=(c.tag_t, c.tag_w, c.tag_h),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.tag_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.2, 0.8, 0.10))),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer", spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, 0.0, 0.05))),
            "cube": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_s, c.cube_s, c.cube_s),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.cube_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.4, -0.6, 0.05))),
        }

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
        self.board: RigidObject = env.iscene["board"]
        self.tag: RigidObject = env.iscene["tag"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.cube: RigidObject = env.iscene["cube"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.target_bay = torch.zeros(n, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._cube_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._eng_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ins_f = torch.zeros(n, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the sideboard (free yaw + xy jitter), sample WHICH bay
        is the target and stick the blue tag chip above its opening, stand the drawer
        and the block on the ground in front (sector + jitter + free yaw), clear the
        latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_board = _qz(yaw)
        bp = torch.zeros(m, 3, device=dev)
        bp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        bp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin
        st[:, 3:7] = q_board
        self.board.write_root_state_to_sim(st, env_ids)

        # which bay is the target; tag chip re-posed onto its face frame
        if c.randomize_bay:
            k = torch.randint(0, 2, (m,), device=dev)
        else:
            k = torch.zeros(m, dtype=torch.long, device=dev)
        self.target_bay[env_ids] = k
        floors = torch.tensor(c.bay_floors, device=dev)
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.tag_t / 2
        loc[:, 2] = floors[k] + c.open_h + c.tag_dz
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin + quat_apply(q_board, loc)
        st[:, 3:7] = q_board
        self.tag.write_root_state_to_sim(st, env_ids)

        # drawer: standing on the ground in front of the face, free yaw
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.drawer_x[0] + torch.rand(m, device=dev) * (c.drawer_x[1] - c.drawer_x[0])
        loc[:, 1] = c.drawer_y[0] + torch.rand(m, device=dev) * (c.drawer_y[1] - c.drawer_y[0])
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin + quat_apply(q_board, loc)
        st[:, 2] = 0.002
        st[:, 2] += origin[:, 2]
        dyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st[:, 3:7] = _qz(dyaw)
        self.drawer.write_root_state_to_sim(st, env_ids)

        # block: on the ground in its own sector, free yaw
        loc = torch.zeros(m, 3, device=dev)
        loc[:, 0] = c.cube_x[0] + torch.rand(m, device=dev) * (c.cube_x[1] - c.cube_x[0])
        loc[:, 1] = c.cube_y[0] + torch.rand(m, device=dev) * (c.cube_y[1] - c.cube_y[0])
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = bp + origin + quat_apply(q_board, loc)
        st[:, 2] = c.cube_s / 2 + 0.003
        st[:, 2] += origin[:, 2]
        cyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st[:, 3:7] = _qz(cyaw)
        self.cube.write_root_state_to_sim(st, env_ids)

        self._cube_l[env_ids] = False
        self._eng_l[env_ids] = False
        self._ins_f[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "board": self.board.data.root_state_w[env_ids].clone(),
            "tag": self.tag.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "cube": self.cube.data.root_state_w[env_ids].clone(),
            "target_bay": self.target_bay[env_ids].clone(),
            "cube_l": self._cube_l[env_ids].clone(),
            "eng_l": self._eng_l[env_ids].clone(),
            "ins_f": self._ins_f[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.board.write_root_state_to_sim(state["board"], env_ids)
        self.tag.write_root_state_to_sim(state["tag"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.cube.write_root_state_to_sim(state["cube"], env_ids)
        self.target_bay[env_ids] = state["target_bay"]
        self._cube_l[env_ids] = state["cube_l"]
        self._eng_l[env_ids] = state["eng_l"]
        self._ins_f[env_ids] = state["ins_f"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey two-bay SIDEBOARD stands on the ground: two identical open "
            f"drawer bays, one above the other (openings "
            f"{c.open_w * 1000:.0f} x {c.open_h * 1000:.0f} mm, lower sill at "
            f"{c.bay_floors[0] * 1000:.0f} mm, upper sill at "
            f"{c.bay_floors[1] * 1000:.0f} mm). Both bays are EMPTY. A small BLUE "
            f"CHIP is stuck to the face frame just above ONE bay's opening — that "
            f"bay is the TARGET, and which bay carries the chip changes from "
            f"episode to episode.\n"
            f"On the ground in front of the sideboard stands its removed DRAWER: an "
            f"open-top wooden box ({c.box_l * 1000:.0f} x {c.box_w * 1000:.0f} x "
            f"{c.box_h * 1000:.0f} mm) with a BLUE face plate (wider than the bay "
            f"opening — it cannot go inside; it is the flush stop) and a yellow "
            f"T-handle protruding from the plate. Nearby lies a GREEN BLOCK "
            f"({c.cube_s * 1000:.0f} mm cube).\n"
            f"Goal: (1) put the green block into the drawer's open-top basin, and "
            f"(2) re-install the drawer into the TAGGED bay: lift it to the "
            f"opening, right side up with the blue plate facing you, slide the box "
            f"in through the opening along the bay floor, and push until the face "
            f"plate sits FLUSH against the face frame (gap under "
            f"{c.closed_tol * 1000:.0f} mm). Load the block FIRST: once the drawer "
            f"is home, the remaining roof gap is smaller than the block and the "
            f"closed plate seals the front, so it can no longer be loaded. A "
            f"drawer seated in the untagged bay, seated upside down, left "
            f"half-inserted, or closed without the block inside does not count. "
            f"Placing the block loose inside the bay instead of in the drawer also "
            f"does not count (and physically jams the drawer's travel). The "
            f"episode ends settled: drawer flush in the tagged bay, block resting "
            f"inside its basin, everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Put the green block into the removed drawer's basin, then slide the "
            "drawer back into the bay marked by the blue chip until its blue face "
            "plate sits flush against the cabinet face. The block must end up "
            "inside the closed drawer; the untagged bay does not count."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _board_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.board.data.root_quat_w,
                                  pos_w - self.board.data.root_pos_w)

    def plate_gap(self) -> torch.Tensor:
        """(N,) the drawer's plate inner face x in the board frame: ~0 = flush at
        the face plane; ~box_l = nose at the face plane; large = out on the apron."""
        return self._board_local(self.drawer.data.root_pos_w)[:, 0] + self.cfg.box_l / 2

    def _axes_dots(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(up_dot, x_dot): drawer up-axis vs board up; drawer +x vs board +x."""
        from isaaclab.utils.math import quat_apply

        n = self.env.num_envs
        dev = self.env.device
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        dq = self.drawer.data.root_quat_w
        bq = self.board.data.root_quat_w
        up = (quat_apply(dq, ez) * quat_apply(bq, ez)).sum(-1)
        fx = (quat_apply(dq, ex) * quat_apply(bq, ex)).sum(-1)
        return up, fx

    def _gates(self) -> torch.Tensor:
        """(N,) bool: in-lane, at the TARGET bay's floor height, upright, facing
        out — the frame in which insertion progress counts."""
        c = self.cfg
        loc = self._board_local(self.drawer.data.root_pos_w)
        floors = torch.tensor(c.bay_floors, device=self.env.device)
        zb = floors[self.target_bay]
        up, fx = self._axes_dots()
        return (loc[:, 1].abs() < c.lane_tol) \
            & ((loc[:, 2] - zb - 0.002).abs() < c.z_tol) \
            & (up >= math.cos(math.radians(c.tilt_max_deg))) \
            & (fx >= math.cos(math.radians(c.align_max_deg)))

    def drawer_seated(self) -> torch.Tensor:
        """(N,) bool: drawer flush in the TARGET bay — plate inner face within
        `closed_tol` of the face plane, centred, at the bay floor, upright,
        facing out."""
        return self._gates() & (self.plate_gap() < self.cfg.closed_tol)

    def cube_in_drawer(self) -> torch.Tensor:
        """(N,) bool: block centre inside the drawer's basin (drawer body frame)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        loc = quat_apply_inverse(self.drawer.data.root_quat_w,
                                 self.cube.data.root_pos_w - self.drawer.data.root_pos_w)
        return (loc[:, 0].abs() < c.box_l / 2 - c.wall_t - 0.004) \
            & (loc[:, 1].abs() < c.box_w / 2 - c.wall_t - 0.004) \
            & (loc[:, 2] > c.floor_t - 0.006) & (loc[:, 2] < c.box_h + 0.010)

    def settled(self) -> torch.Tensor:
        """(N,) bool: drawer and block at rest."""
        c = self.cfg
        return (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.drawer.data.root_ang_vel_w.norm(dim=-1) < 0.5)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.drawer.data.root_pos_w, self.cube.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        cslow = self.cube.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        self._cube_l |= self.cube_in_drawer() & cslow & fin
        gates = self._gates()
        gap = self.plate_gap().clamp(min=0.0)
        self._eng_l |= gates & (gap < c.box_l - c.eng_depth) & fin
        frac = ((c.box_l - gap) / c.box_l).clamp(0.0, 1.0) * gates.float()
        self._ins_f = torch.where(fin, torch.maximum(self._ins_f, frac), self._ins_f)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: drawer flush in the TARGET bay with the block settled inside
        its basin — live physical outcomes, settled and finite. The flush pose can
        only be produced by a real guided slide through the opening (the plate is
        physically unable to enter, the box physically unable to be flush any other
        way), and the block can only be aboard if it was loaded before closure
        (the roof gap over an installed drawer is smaller than the block)."""
        self._update_latches()
        return self.drawer_seated() & self.cube_in_drawer() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 block-in-basin (latched) + 0.10 engaged
        (latched) + 0.55 * latched max insertion fraction into the TARGET bay,
        capped at 0.80; exactly 1.0 iff success() holds live. Doing nothing scores
        ~0; the seed's strategy (push the drawer along the ground) scores ~0 —
        the bays are elevated and insertion credit is gated on being inside the
        target bay."""
        c = self.cfg
        self._update_latches()
        base = (c.w_cube * self._cube_l.float() + c.w_eng * self._eng_l.float()
                + c.w_ins * self._ins_f).clamp(max=0.80)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="drawer_refit", robot="null"))
