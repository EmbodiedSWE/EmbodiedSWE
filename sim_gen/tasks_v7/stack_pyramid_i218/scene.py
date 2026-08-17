"""PyramidFreightScene — assemble the seed's cube pyramid ON a delivery cart, then push
the loaded cart along guide rails INTO a low-roofed depot.

Derived from maniskill/stack_pyramid ("pick up the red cube, place it next to the green
cube, stack the blue cube on top of both"), but the PLAN is inverted from build-in-place
to BUILD-THEN-DELIVER. The seed's entire strategy is repeated free-space pick-and-place
at the goal location: every placement happens exactly where it will be judged, each
intermediate is independently stable, and nothing that has been built ever has to move
again. Here the goal LOCATION is inside a depot whose roof leaves only ~17 mm of
headroom above the finished pyramid — no gripper (and no falling cube) can get a block
onto the stack in there, which smoke certifies by dropping a cube onto the pocket from
above (it lands on the roof). The pyramid must therefore be assembled OUTSIDE, in the
shallow tray of a wheeled cart parked at the open end of the guide rails, and then the
whole FRAGILE ASSEMBLY must be transported: pushed down the rails, through the depot
door, until the cart rests against the back wall. The transport is the signature
interaction and it is honest dynamics — the bridging cube sits free on the two base
cubes (only they are confined by the tray lips), so a hard shove slides it off the back
(smoke's shove certificate), while a gentle push keeps the stack together. The seed
demands none of this: no forced ordering (its placements commute), no transport of
anything already built, no stability-limited motion.

Bodies (all procedural, no external assets):
  - cube_red / cube_green / cube_blue: the seed's 40 mm cubes, 50 g, high friction.
  - sled: YELLOW cart, dynamic compound — low-friction skid plate + high-friction deck,
    a shallow tray (2 cube cells wide, 10 mm lips) at the front, and a BLACK push post
    at the rear. The tray confines the two BASE cubes; the bridging cube rides free.
  - garage: the depot + rails, ONE kinematic compound — steel-blue side walls and back
    stop, an ORANGE roof (underside 115 mm; deck-top + pyramid = 98 mm), and low guide
    rails running from the depot door to the loading area. The pocket is closed on
    every side except the door.

Per-episode randomization (readback-verified in smoke): depot world xy + FULL yaw (the
push heading varies over 360 deg), cart start slot along the rails, cube scatter xy +
free yaw, and a random permutation of which cube occupies which scatter slot.

Rubric (0..1; latched, debounced partial credit anchored in the demonstrated solve):
  0.20 * seated    — red AND green settled in the cart tray (cart anywhere)
  0.25 * built     — full pyramid settled on the cart: blue bridging the seam,
                     centred over BOTH base cubes (judged in the CART BODY FRAME,
                     so the moving cart judges identically)
  0.40 * delivered — pyramid intact AND cart settled inside the depot pocket
  1.0 iff success(): pyramid on the cart + cart in the pocket, everything still
                     and finite. Non-success capped at 0.85.

Cube-on-cart geometry is judged in the cart body frame; cart-in-depot in the depot
frame — both from pose readback, so no clause depends on world heading.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


# ----- USD authoring helpers (custom compound spawners) -----------------------------------------
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _bind_phys_material(stage, mat_path: str, prims, *, static: float, dynamic: float,
                        restitution: float, combine: str) -> None:
    """Author a UsdShade physics material and bind it to `prims`. Custom-spawner
    colliders otherwise fall back to a ~0.5-friction default; the skid NEEDS a low
    combine-min material (slides under a modest push) and the deck a high one (the
    cubes must ride the cart, not skate on it)."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static))
    api.CreateDynamicFrictionAttr(float(dynamic))
    api.CreateRestitutionAttr(float(restitution))
    pxm = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    pxm.CreateFrictionCombineModeAttr(combine)
    pxm.CreateRestitutionCombineModeAttr("min")
    for prim in prims:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics")


def _spawn_sled(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The cart: DYNAMIC compound. Local frame: origin at the skid bottom centre,
    +x toward the tray (the FRONT, which leads into the depot). Everything physical
    (rigid body, explicit mass + low CoM, damping, solver iters, split materials) is
    authored here — the clone-wrapped custom func gets no schema help."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateLinearDampingAttr(0.05)
    px.CreateAngularDampingAttr(0.1)
    px.CreateSolverPositionIterationCountAttr(16)
    px.CreateSolverVelocityIterationCountAttr(1)
    px.CreateMaxDepenetrationVelocityAttr(0.5)
    px.CreateSleepThresholdAttr(0.0)
    px.CreateStabilizationThresholdAttr(0.0)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, float(cfg.com_z)))

    collide = _make_collide(cfg.contact_offset)
    c = cfg
    skid = _add_box(stage, f"{prim_path}/skid",
                    center=(0.0, 0.0, c.skid_t / 2),
                    size=(c.sled_l, c.sled_w, c.skid_t),
                    color=(0.25, 0.25, 0.28), collide=collide)
    grip_prims = [
        _add_box(stage, f"{prim_path}/deck",
                 center=(0.0, 0.0, c.skid_t + c.deck_t / 2),
                 size=(c.sled_l, c.sled_w, c.deck_t),
                 color=c.deck_color, collide=collide),
    ]
    z_lip = c.skid_t + c.deck_t + c.lip_h / 2
    tray_c, tray_hl = c.tray_cx, c.tray_l / 2
    grip_prims += [
        _add_box(stage, f"{prim_path}/lip_front",
                 center=(tray_c + tray_hl + c.lip_t / 2, 0.0, z_lip),
                 size=(c.lip_t, c.tray_w + 2 * c.lip_t, c.lip_h),
                 color=c.lip_color, collide=collide),
        _add_box(stage, f"{prim_path}/lip_rear",
                 center=(tray_c - tray_hl - c.lip_t / 2, 0.0, z_lip),
                 size=(c.lip_t, c.tray_w + 2 * c.lip_t, c.lip_h),
                 color=c.lip_color, collide=collide),
        _add_box(stage, f"{prim_path}/lip_left",
                 center=(tray_c, c.tray_w / 2 + c.lip_t / 2, z_lip),
                 size=(c.tray_l + 2 * c.lip_t, c.lip_t, c.lip_h),
                 color=c.lip_color, collide=collide),
        _add_box(stage, f"{prim_path}/lip_right",
                 center=(tray_c, -c.tray_w / 2 - c.lip_t / 2, z_lip),
                 size=(c.tray_l + 2 * c.lip_t, c.lip_t, c.lip_h),
                 color=c.lip_color, collide=collide),
        _add_box(stage, f"{prim_path}/post",
                 center=(c.post_x, 0.0, c.skid_t + c.deck_t + c.post_h / 2),
                 size=(c.post_s, c.post_s, c.post_h),
                 color=(0.08, 0.08, 0.08), collide=collide),
    ]
    _bind_phys_material(stage, f"{prim_path}/skid_mat", (skid,),
                        static=c.skid_mu, dynamic=c.skid_mu * 0.9,
                        restitution=0.0, combine="min")
    _bind_phys_material(stage, f"{prim_path}/deck_mat", grip_prims,
                        static=c.deck_mu, dynamic=c.deck_mu * 0.9,
                        restitution=0.0, combine="average")
    return root


def _spawn_garage(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The depot + rails: ONE KINEMATIC compound. Local frame: origin at the back
    stop's inner face, on the ground, +x toward the door / rails / loading area.
    Low-friction combine-min material everywhere so the cart never binds on a wall."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    px.CreateSolverPositionIterationCountAttr(4)
    px.CreateSolverVelocityIterationCountAttr(1)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(20.0)

    collide = _make_collide(cfg.contact_offset)
    c = cfg
    wall_c = (0.85, 0.45, 0.10)  # orange roof
    steel = (0.30, 0.38, 0.52)   # steel-blue walls
    rail_c = (0.62, 0.64, 0.70)
    prims = [
        _add_box(stage, f"{prim_path}/backstop",
                 center=(-c.stop_t / 2, 0.0, (c.wall_h + c.roof_t) / 2),
                 size=(c.stop_t, c.gap_half * 2 + 2 * c.wall_t, c.wall_h + c.roof_t),
                 color=steel, collide=collide),
        _add_box(stage, f"{prim_path}/wall_left",
                 center=(c.wall_len / 2, c.gap_half + c.wall_t / 2, c.wall_h / 2),
                 size=(c.wall_len, c.wall_t, c.wall_h),
                 color=steel, collide=collide),
        _add_box(stage, f"{prim_path}/wall_right",
                 center=(c.wall_len / 2, -c.gap_half - c.wall_t / 2, c.wall_h / 2),
                 size=(c.wall_len, c.wall_t, c.wall_h),
                 color=steel, collide=collide),
        _add_box(stage, f"{prim_path}/roof",
                 center=(c.roof_len / 2, 0.0, c.wall_h + c.roof_t / 2),
                 size=(c.roof_len, c.gap_half * 2 + 2 * c.wall_t, c.roof_t),
                 color=wall_c, collide=collide),
        _add_box(stage, f"{prim_path}/rail_left",
                 center=((c.rail_x0 + c.rail_x1) / 2, c.gap_half + c.rail_t / 2,
                         c.rail_h / 2),
                 size=(c.rail_x1 - c.rail_x0, c.rail_t, c.rail_h),
                 color=rail_c, collide=collide),
        _add_box(stage, f"{prim_path}/rail_right",
                 center=((c.rail_x0 + c.rail_x1) / 2, -c.gap_half - c.rail_t / 2,
                         c.rail_h / 2),
                 size=(c.rail_x1 - c.rail_x0, c.rail_t, c.rail_h),
                 color=rail_c, collide=collide),
    ]
    _bind_phys_material(stage, f"{prim_path}/mat", prims,
                        static=c.mu, dynamic=c.mu * 0.9,
                        restitution=0.0, combine="min")
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "sled" not in _SPAWNER_CACHE:

        @configclass
        class SledSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_sled)
            sled_l: float = 0.16
            sled_w: float = 0.06
            skid_t: float = 0.006
            deck_t: float = 0.012
            lip_h: float = 0.010
            lip_t: float = 0.006
            tray_cx: float = 0.030
            tray_l: float = 0.084
            tray_w: float = 0.044
            post_x: float = -0.050
            post_s: float = 0.022
            post_h: float = 0.100
            mass: float = 0.25
            com_z: float = 0.012
            skid_mu: float = 0.12
            deck_mu: float = 0.90
            deck_color: tuple = (0.95, 0.80, 0.10)
            lip_color: tuple = (0.80, 0.65, 0.05)
            contact_offset: float = 0.002

        @configclass
        class GarageSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_garage)
            gap_half: float = 0.034
            wall_t: float = 0.020
            wall_h: float = 0.115
            wall_len: float = 0.135
            roof_len: float = 0.115
            roof_t: float = 0.012
            stop_t: float = 0.020
            rail_x0: float = 0.10
            rail_x1: float = 0.50
            rail_t: float = 0.015
            rail_h: float = 0.022
            mu: float = 0.15
            contact_offset: float = 0.002

        _SPAWNER_CACHE["sled"] = SledSpawnerCfg
        _SPAWNER_CACHE["garage"] = GarageSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class PyramidFreightSceneCfg(BaseCfg):
    """Config for `PyramidFreightScene`. Honesty bounds: the roof underside sits at
    `wall_h` (115 mm) while deck-top + finished pyramid = 18 + 80 = 98 mm, so the
    delivered pyramid fits with ~17 mm to spare, but no cube can be brought DOWN onto
    the stack inside (a dropped cube lands on the roof — smoke's roof certificate),
    forcing build-then-deliver. The tray confines only the BASE pair (lips 10 mm,
    cube 40 mm); the bridging cube rides free, so transport gentleness is real
    physics (smoke's shove certificate). `span_tol` (10 mm) < the 21 mm offset of a
    cube parked on a single base cube, so 'bridging BOTH' is enforced."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    z_tol: float = tunable(0.012)        # cube-centre height window on the cart (m)
    tray_x_tol: float = tunable(0.033)   # base-cube x window around the tray centre (m)
    tray_y_tol: float = tunable(0.014)   # base-cube |y| window in the tray (m)
    span_tol: float = tunable(0.010)     # blue x-offset from the base-pair midpoint (m)
    span_y_tol: float = tunable(0.012)   # blue y-offset from the base-pair midpoint (m)
    deliver_x: float = tunable(0.10)     # cart centre depot-x below this = inside the pocket
    deliver_y: float = tunable(0.020)    # cart centre |depot-y| window
    align_min_dot: float = tunable(0.90)  # cart front axis vs door direction (into the depot)
    settle_speed: float = tunable(0.06)  # max |lin vel| of every dynamic body when judging
    latch_steps: int = tunable(5)        # consecutive still steps before a latch fires

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    fix_jitter: float = tunable(0.06)    # depot world xy jitter (+/- m)
    fix_yaw_deg: float = tunable(180.0)  # depot yaw (+/- deg): FULL heading randomization
    sled_x_jitter: float = tunable(0.03)  # cart start slot along the rails (+/- m)
    cube_jitter: float = tunable(0.03)   # cube scatter xy jitter (+/- m)
    slot_permute: bool = tunable(True)   # random cube-to-slot permutation per episode

    # --- info: layout (depot frame: origin at the back stop, +x toward the door) ----------------
    fix_pos: tuple = info((-0.10, 0.0))  # nominal depot origin, world xy
    sled_x0: float = info(0.36)          # nominal cart start (centre), depot x
    cube_slots: tuple = info(((0.52, 0.16), (0.52, -0.16), (0.64, 0.0)))
    # --- info: cubes (the seed's trio) ----------------------------------------------------------
    cube_s: float = info(0.04)
    cube_mass: float = info(0.05)
    # --- info: cart geometry (must match SledSpawnerCfg) ----------------------------------------
    sled_l: float = info(0.16)
    sled_w: float = info(0.06)
    deck_top: float = info(0.018)        # skid_t + deck_t: deck top, cart local z
    tray_cx: float = info(0.030)         # tray centre, cart local x
    tray_l: float = info(0.084)
    tray_w: float = info(0.044)
    cell_dx: float = info(0.021)         # cell centres at tray_cx +/- cell_dx
    post_x: float = info(-0.050)
    sled_mass: float = info(0.25)
    # --- info: depot geometry (must match GarageSpawnerCfg) -------------------------------------
    gap_half: float = info(0.034)
    wall_h: float = info(0.115)          # roof underside height
    roof_len: float = info(0.115)
    wall_len: float = info(0.135)
    rail_x1: float = info(0.50)
    contact_offset: float = info(0.002)
    # rubric weights (0.20 + 0.25 + 0.40 = 0.85 = the non-success cap)
    w_seated: float = info(0.20)
    w_built: float = info(0.25)
    w_deliv: float = info(0.40)


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("pyramid_freight")
class PyramidFreightScene(BaseScene):
    cfg: PyramidFreightSceneCfg

    CUBE_NAMES = ("cube_red", "cube_green", "cube_blue")
    CUBE_COLORS = ((0.90, 0.08, 0.08), (0.08, 0.75, 0.12), (0.10, 0.25, 0.92))

    def __init__(self, cfg: PyramidFreightSceneCfg | None = None) -> None:
        super().__init__(cfg or PyramidFreightSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        sp = _spawner_classes()
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
            "garage": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Garage",
                spawn=sp["garage"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.fix_pos[0], c.fix_pos[1], 0.0)),
            ),
            "sled": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Sled",
                spawn=sp["sled"](contact_offset=c.contact_offset),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + c.sled_x0, c.fix_pos[1], 0.001)),
            ),
        }
        for name, color, slot in zip(self.CUBE_NAMES, self.CUBE_COLORS, c.cube_slots):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(c.cube_s, c.cube_s, c.cube_s),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.05,
                        sleep_threshold=0.0, stabilization_threshold=0.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.cube_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.9, dynamic_friction=0.85, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.fix_pos[0] + slot[0], c.fix_pos[1] + slot[1],
                         c.cube_s / 2 + 0.001)),
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
        super().bind(env)
        self.garage: RigidObject = env.iscene["garage"]
        self.sled: RigidObject = env.iscene["sled"]
        self.cubes: dict[str, RigidObject] = {n: env.iscene[n] for n in self.CUBE_NAMES}
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.fix_xy = torch.zeros(n, 2, device=dev)   # depot origin, world (incl env origin)
        self.fix_yaw = torch.zeros(n, device=dev)
        self.slot_perm = torch.zeros(n, 3, dtype=torch.long, device=dev)
        # latches (partial credit survives transients; success is judged live) + debounce
        self._seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self._built = torch.zeros(n, dtype=torch.bool, device=dev)
        self._deliv = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seated_ct = torch.zeros(n, dtype=torch.long, device=dev)
        self._built_ct = torch.zeros(n, dtype=torch.long, device=dev)
        self._deliv_ct = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the depot (world xy jitter + full yaw), the cart at
        its start slot on the rails (front toward the door), and the cubes at a
        random permutation of the scatter slots with jitter + free yaw. Clear
        latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- depot: kinematic compound, world jitter + full yaw ---
        psi = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg)
        fxy = torch.zeros(m, 2, device=dev)
        fxy[:, 0] = c.fix_pos[0]
        fxy[:, 1] = c.fix_pos[1]
        fxy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        fxy += origin[:, 0:2]
        self.fix_xy[env_ids] = fxy
        self.fix_yaw[env_ids] = psi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = fxy
        st[:, 2] = origin[:, 2]
        st[:, 3:7] = _qz(psi)
        self.garage.write_root_state_to_sim(st, env_ids)

        cosp, sinp = torch.cos(psi), torch.sin(psi)

        def fix_to_world(x_loc: torch.Tensor, y_loc: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(m, 2, device=dev)
            out[:, 0] = fxy[:, 0] + cosp * x_loc - sinp * y_loc
            out[:, 1] = fxy[:, 1] + sinp * x_loc + cosp * y_loc
            return out

        # --- cart: on the rails at the start slot, FRONT (tray) toward the door ---
        sx = c.sled_x0 + (torch.rand(m, device=dev) * 2 - 1) * c.sled_x_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = fix_to_world(sx, torch.zeros(m, device=dev))
        st[:, 2] = origin[:, 2] + 0.001
        st[:, 3:7] = _qz(psi + math.pi)  # cart +x (front) points toward depot -x direction
        self.sled.write_root_state_to_sim(st, env_ids)

        # --- cubes: random slot permutation + jitter + free yaw ---
        # (torch.rand-based permutation: the first torch.randint after a manual seed
        # is near-degenerate across seeds)
        perm = torch.rand(m, 3, device=dev).argsort(dim=1)
        self.slot_perm[env_ids] = perm
        slots = torch.tensor(c.cube_slots, device=dev)  # (3, 2) depot frame
        for i, name in enumerate(self.CUBE_NAMES):
            sl = slots[perm[:, i]]
            jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.cube_jitter
            yaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = fix_to_world(sl[:, 0] + jit[:, 0], sl[:, 1] + jit[:, 1])
            st[:, 2] = origin[:, 2] + c.cube_s / 2 + 0.001
            st[:, 3:7] = _qz(yaw)
            self.cubes[name].write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        for t in (self._seated, self._built, self._deliv):
            t[env_ids] = False
        for t in (self._seated_ct, self._built_ct, self._deliv_ct):
            t[env_ids] = 0

    # ----- state (full, restorable) -------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "garage": self.garage.data.root_state_w[env_ids].clone(),
            "sled": self.sled.data.root_state_w[env_ids].clone(),
            "cubes": {n: b.data.root_state_w[env_ids].clone()
                      for n, b in self.cubes.items()},
            "fix_xy": self.fix_xy[env_ids].clone(),
            "fix_yaw": self.fix_yaw[env_ids].clone(),
            "slot_perm": self.slot_perm[env_ids].clone(),
            "seated": self._seated[env_ids].clone(),
            "built": self._built[env_ids].clone(),
            "deliv": self._deliv[env_ids].clone(),
            "seated_ct": self._seated_ct[env_ids].clone(),
            "built_ct": self._built_ct[env_ids].clone(),
            "deliv_ct": self._deliv_ct[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.garage.write_root_state_to_sim(state["garage"], env_ids)
        self.sled.write_root_state_to_sim(state["sled"], env_ids)
        for n, b in self.cubes.items():
            b.write_root_state_to_sim(state["cubes"][n], env_ids)
        self.fix_xy[env_ids] = state["fix_xy"]
        self.fix_yaw[env_ids] = state["fix_yaw"]
        self.slot_perm[env_ids] = state["slot_perm"]
        self._seated[env_ids] = state["seated"]
        self._built[env_ids] = state["built"]
        self._deliv[env_ids] = state["deliv"]
        self._seated_ct[env_ids] = state["seated_ct"]
        self._built_ct[env_ids] = state["built_ct"]
        self._deliv_ct[env_ids] = state["deliv_ct"]

    # ----- description --------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A small freight DEPOT stands on the ground: steel-blue side walls and "
            f"back wall under a low ORANGE roof (opening height "
            f"{c.wall_h * 1000:.0f} mm), open only through its front door, with two "
            f"low gray GUIDE RAILS running {c.rail_x1 * 100:.0f} cm out from the "
            f"door. A YELLOW CART sits between the rails near their open end: a flat "
            f"cart with a shallow tray at its front (inner {c.tray_l * 1000:.0f} x "
            f"{c.tray_w * 1000:.0f} mm, {10:.0f} mm lips — exactly two cube cells, "
            f"one behind the other) and a BLACK vertical push post at its rear. "
            f"Scattered on the open ground beyond the rails lie three "
            f"{c.cube_s * 1000:.0f} mm cubes: RED, GREEN and BLUE. The depot's "
            f"position and heading, the cart's start slot, and which cube lies where "
            f"all vary per episode.\n"
            f"Goal: deliver the seed pyramid into the depot ON the cart. First build "
            f"it in the cart's tray: set the RED and GREEN cubes side by side into "
            f"the two tray cells (either order, square to the cart so they seat "
            f"between the lips), then balance the BLUE cube on top, bridging the "
            f"joint so it rests centred on BOTH lower cubes. Then push the loaded "
            f"cart by its black post along the rails, through the door, until it "
            f"rests against the depot's back wall. The roof is far too low to stack "
            f"cubes inside — assemble on the cart FIRST, and push gently: only the "
            f"two lower cubes are held by the tray lips, the blue cube rides free "
            f"and slides off if the cart is jerked or crashed. Success is judged "
            f"with everything at rest: red and green seated in the tray, blue "
            f"bridging both, cart fully inside the depot. A pyramid built anywhere "
            f"off the cart, a cart delivered empty or half-loaded, the wrong cube "
            f"on top, or a blue cube resting on just one lower cube all fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place the red and green cubes side by side in the yellow cart's tray "
            "and balance the blue cube on top bridging both. Then push the cart by "
            "its black post along the rails into the depot until it rests against "
            "the back wall, gently enough that the pyramid stays intact — the blue "
            "cube is unrestrained and falls off if the cart is jerked."
        )

    # ----- frames -------------------------------------------------------------------------------
    def _cube_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(pos_w (N,3,3), |lin vel| (N,3)) for red/green/blue, in CUBE_NAMES order."""
        pos = torch.stack([b.data.root_pos_w for b in self.cubes.values()], dim=1)
        vel = torch.stack([b.data.root_lin_vel_w.norm(dim=-1)
                           for b in self.cubes.values()], dim=1)
        return pos, vel

    def cubes_sled_local(self) -> torch.Tensor:
        """(N,3,3): cube centres in the CART BODY FRAME (origin at skid bottom
        centre, +x = front/tray). The on-cart rubric lives here so a moving or
        rotated cart judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        pos, _v = self._cube_tensors()
        n = pos.shape[0]
        sq = self.sled.data.root_quat_w[:, None, :].expand(n, 3, 4).reshape(n * 3, 4)
        sp = self.sled.data.root_pos_w[:, None, :]
        return quat_apply_inverse(sq, (pos - sp).reshape(n * 3, 3)).reshape(n, 3, 3)

    def sled_fix_local(self) -> torch.Tensor:
        """(N,3): cart centre in the DEPOT FRAME (origin at the back stop, +x toward
        the door)."""
        p = self.sled.data.root_pos_w
        d = p[:, 0:2] - self.fix_xy
        cosp, sinp = torch.cos(self.fix_yaw), torch.sin(self.fix_yaw)
        out = torch.zeros(p.shape[0], 3, device=p.device)
        out[:, 0] = cosp * d[:, 0] + sinp * d[:, 1]
        out[:, 1] = -sinp * d[:, 0] + cosp * d[:, 1]
        out[:, 2] = p[:, 2] - self.env_origins[:, 2]
        return out

    def point_fix_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,2) world xy -> depot-frame xy."""
        d = p_w - self.fix_xy
        cosp, sinp = torch.cos(self.fix_yaw), torch.sin(self.fix_yaw)
        return torch.stack([cosp * d[:, 0] + sinp * d[:, 1],
                            -sinp * d[:, 0] + cosp * d[:, 1]], dim=-1)

    # ----- live predicates ----------------------------------------------------------------------
    def base_seated(self) -> torch.Tensor:
        """(N,2) bool: red / green cube seated in the tray (cart frame): centre at
        cube-rest height over the deck, inside the tray windows."""
        c = self.cfg
        loc = self.cubes_sled_local()[:, 0:2, :]  # red, green
        z_ok = (loc[:, :, 2] - (c.deck_top + c.cube_s / 2)).abs() < c.z_tol
        x_ok = (loc[:, :, 0] - c.tray_cx).abs() < c.tray_x_tol
        y_ok = loc[:, :, 1].abs() < c.tray_y_tol
        return z_ok & x_ok & y_ok

    def pyramid_ok(self) -> torch.Tensor:
        """(N,) bool: the seed pyramid stands in the cart tray — red AND green
        seated, BLUE at stack height, centred over the base pair's midpoint (so it
        genuinely rests on BOTH: `span_tol` is less than half the offset of a cube
        parked on a single base cube)."""
        c = self.cfg
        loc = self.cubes_sled_local()
        seated = self.base_seated().all(dim=1)
        blue = loc[:, 2, :]
        mid = (loc[:, 0, :] + loc[:, 1, :]) / 2
        z_ok = (blue[:, 2] - (c.deck_top + 1.5 * c.cube_s)).abs() < c.z_tol
        x_ok = (blue[:, 0] - mid[:, 0]).abs() < c.span_tol
        y_ok = (blue[:, 1] - mid[:, 1]).abs() < c.span_y_tol
        return seated & z_ok & x_ok & y_ok

    def delivered_ok(self) -> torch.Tensor:
        """(N,) bool: the cart rests inside the depot pocket — centre past
        `deliver_x`, centred in the channel, on the ground, front axis aligned into
        the depot."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = self.sled_fix_local()
        n = loc.shape[0]
        ex = torch.tensor([1.0, 0.0, 0.0], device=loc.device).expand(n, 3)
        front_w = quat_apply(self.sled.data.root_quat_w, ex)
        door_w = torch.stack([-torch.cos(self.fix_yaw), -torch.sin(self.fix_yaw),
                              torch.zeros_like(self.fix_yaw)], dim=-1)
        aligned = (front_w * door_w).sum(dim=-1) > c.align_min_dot
        return (loc[:, 0] < c.deliver_x) & (loc[:, 0] > 0.02) \
            & (loc[:, 1].abs() < c.deliver_y) & (loc[:, 2] < 0.02) & aligned

    def still(self) -> torch.Tensor:
        """(N,) bool: every dynamic body below the settle speed."""
        _p, vel = self._cube_tensors()
        sv = self.sled.data.root_lin_vel_w.norm(dim=-1)
        return (vel < self.cfg.settle_speed).all(dim=1) & (sv < self.cfg.settle_speed)

    def _finite(self) -> torch.Tensor:
        pos, _v = self._cube_tensors()
        return torch.isfinite(pos).all(dim=-1).all(dim=-1) \
            & torch.isfinite(self.sled.data.root_pos_w).all(dim=-1)

    # ----- latches ------------------------------------------------------------------------------
    def _update_latches(self) -> None:
        """Debounced (counter) latching: a stage must hold `latch_steps` consecutive
        post-steps while STILL before it counts — a cube sweeping through a window
        mid-flight, or a cart shoved fast through the pocket, carries velocity and
        never latches."""
        c = self.cfg
        still = self.still()
        seated_now = self.base_seated().all(dim=1) & still
        built_now = self.pyramid_ok() & still
        deliv_now = self.pyramid_ok() & self.delivered_ok() & still
        for now, ct, latch in ((seated_now, self._seated_ct, self._seated),
                               (built_now, self._built_ct, self._built),
                               (deliv_now, self._deliv_ct, self._deliv)):
            ct[:] = torch.where(now, ct + 1, torch.zeros_like(ct))
            latch |= ct >= c.latch_steps

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric -------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: pyramid standing in the cart tray AND cart resting inside the
        depot pocket, everything still and finite. All clauses are live physical
        outcomes — a toppled bridge cube or a cart short of the pocket fails on the
        spot, so success must persist on its own."""
        self._update_latches()
        return self.pyramid_ok() & self.delivered_ok() & self.still() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.20*seated + 0.25*built + 0.40*delivered (latched,
        ~0 for the null policy), capped at 0.85 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        base = (c.w_seated * self._seated.float() + c.w_built * self._built.float()
                + c.w_deliv * self._deliv.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="pyramid_freight", robot="null"))
