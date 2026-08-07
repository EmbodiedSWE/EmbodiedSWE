"""LedgeCatchScene — stage the open basket on the ground under the shelf's side edge,
then push the ORANGE juice carton off that edge so gravity drops it into the basket.

Derived from libero/libero_pick_orange_juice, with the seed's whole transport model
inverted. The seed's plan is the canonical grasp-and-carry: identify the orange juice
among distractors, pick it up, carry it through free space, drop it into a basket
that never moves. Here the target carton is UNGRASPABLE — its 100 mm footprint
exceeds the 80 mm Franka jaw span on every horizontal axis — and it stands on an
elevated pedestal shelf, so no grasp-and-carry plan exists at all. The receptacle is
the mobile body instead: the basket spawns on the ground far from the shelf and must
be staged UNDER the shelf's overhanging side edge FIRST; then the carton is pushed
off that edge and gravity delivers it ballistically into the waiting basket. The
execution order is forced: a carton pushed off with no basket below lands on the
ground, where (ungraspable, the basket's walls block any ground-level push-in, and
the fell-in gate rejects lowering the basket over it) it can never legitimately
enter the basket — the episode is unrecoverable. A WHITE milk
carton stands at the shelf's opposite edge as a distractor (the seed's distractor
set, reduced to the one that matters) and must be left standing on the shelf.

Assets are fully procedural, authored by custom compound spawners (the stopper_vault
pattern — child colliders of one body never self-collide):
  - shelf: KINEMATIC compound "mushroom pedestal" — one central column plus a top
    slab (360 x 440 x 12 mm, top at 260 mm) overhanging on all sides, so the ground
    under each side edge is open: a basket (108 mm tall) slides under the 248 mm
    underside with headroom. Local frame: origin on the ground below the slab
    centre; the side edges are y = +/-0.22.
  - carton (target): ORANGE box 100 x 100 x 115 mm, 120 g, defined friction 0.35
    (the defined-friction rule) — slides under ~0.45 N, tips only above ~1.0 N.
  - decoy: WHITE carton, identical box, at the opposite side edge.
  - basket: DYNAMIC open-top box, outer 280 x 280 mm, walls 100 mm above an 8 mm
    floor, 8 mm wall thickness (a rim any parallel jaw can grasp), 500 g.

Per-episode randomization (readback-verifiable): shelf yaw + xy jitter, WHICH side
edge holds the target (the decoy takes the other side), the carton slot x along the
edge + jitter + free yaw for both cartons, basket ground slot (side + band + jitter)
with free yaw.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * approach — latched max of the basket's progress toward the catch point
                    under the target's shelf edge, normalized by its own spawn
                    distance (exactly 0 for the null policy)
  0.25 * staged   — basket ever upright on the ground with its centre within
                    `stage_r` of the catch point WHILE the target carton is still
                    on the shelf (latched; staging after the carton is already off
                    earns nothing — the order gate)
  0.35 * caught   — target carton ever observed FALLING INTO the staged basket:
                    fully inside the basket interior at the catch station while
                    DESCENDING (vz < fall_vz) with the staging latch already set
                    (latched). This is the anti-capture gate: a carton placed into
                    a basket anywhere else (the seed's own end state — item into
                    the never-moved basket at its spawn), or a basket lowered
                    OVER a carton already on the ground ("capping"), never
                    satisfies it — the carton must fall in.
  1.0 iff success() — target settled fully inside the upright, grounded basket,
                    the basket still at the catch station under the drop edge,
                    the caught (fell-in) latch set, decoy still standing on the
                    shelf, everything settled. Non-success is capped at 0.85.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


# ----- custom compound spawners ---------------------------------------------------------------
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


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _spawn_shelf(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the pedestal shelf at `prim_path`: KINEMATIC rigid body (repositionable
    at reset via write_root_state, immovable to contacts). Local frame: origin on the
    GROUND below the slab centre. One central column + the overhanging top slab —
    the ground under all four slab edges stays open for the basket."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    _add_box(stage, f"{prim_path}/column",
             center=(0.0, 0.0, (c.slab_top - c.slab_t) / 2),
             size=(c.col_w, c.col_w, c.slab_top - c.slab_t),
             color=c.col_color, collide=collide)
    _add_box(stage, f"{prim_path}/slab",
             center=(0.0, 0.0, c.slab_top - c.slab_t / 2),
             size=(2 * c.hx, 2 * c.hy, c.slab_t),
             color=c.color, collide=collide)
    return root


def _spawn_basket(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the basket: DYNAMIC compound — 8 mm floor + four 8 mm walls, open top.
    Local origin at the OUTER BOTTOM centre (rest pose root z ~= 0). Sleep /
    stabilization thresholds zeroed (a sleeping body silently ignores the carton
    impact and external probe forces)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.20)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg)
    c = cfg
    ho, t, wh = c.half_out, c.wall_t, c.wall_h
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, t / 2),
             size=(2 * ho, 2 * ho, t), color=c.color, collide=collide)
    wall_zc = t + wh / 2
    for nm, ctr, sz in (
        ("wall_px", (ho - t / 2, 0.0, wall_zc), (t, 2 * ho, wh)),
        ("wall_nx", (-ho + t / 2, 0.0, wall_zc), (t, 2 * ho, wh)),
        ("wall_py", (0.0, ho - t / 2, wall_zc), (2 * ho - 2 * t, t, wh)),
        ("wall_ny", (0.0, -ho + t / 2, wall_zc), (2 * ho - 2 * t, t, wh)),
    ):
        _add_box(stage, f"{prim_path}/{nm}", center=ctr, size=sz,
                 color=c.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shelf" not in _SPAWNER_CACHE:

        @configclass
        class ShelfSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shelf)
            hx: float = 0.18
            hy: float = 0.22
            slab_top: float = 0.26
            slab_t: float = 0.012
            col_w: float = 0.06
            color: tuple = (0.62, 0.64, 0.68)
            col_color: tuple = (0.35, 0.36, 0.40)
            contact_offset: float = 0.002

        @configclass
        class BasketSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_basket)
            half_out: float = 0.14
            wall_h: float = 0.10
            wall_t: float = 0.008
            color: tuple = (0.55, 0.38, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(shelf=ShelfSpawnerCfg, basket=BasketSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LedgeCatchSceneCfg(BaseCfg):
    """Config for `LedgeCatchScene`. The slab's side edges are y = +/-hy = +/-0.22 in
    the shelf frame; the catch point for the target's side is (slot_x,
    side*(hy + catch_off)) on the ground. The basket interior is 264 x 264 mm
    (half 0.132); the 100 mm carton tumbling off the 260 mm slab from a slow push
    lands 30-120 mm outboard of the edge — inside a basket centred at
    catch_off = 60 mm outboard. The 80 mm jaw-span bound is what makes the carton
    ungraspable and forces the catch (the embodiment argument in TASK.md)."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    inside_xy_max: float = tunable(0.105)  # |carton centre, basket frame xy| below this (m)
    # (interior half-span 0.132; a carton leaning bottom-in against a wall sits ~0.085)
    inside_z_min: float = tunable(0.010)  # carton centre above the basket floor (m)
    inside_z_max: float = tunable(0.105)  # carton centre below the wall top (wall_t + wall_h
    # = 0.108; a carton straddling the rim sits >= 0.145)
    basket_up_min: float = tunable(0.95)  # basket local +z . world up >= this (~18 deg)
    basket_z_max: float = tunable(0.020)  # basket root (outer bottom centre) height <= this (m)
    stage_r: float = tunable(0.09)  # basket centre within this of the catch point = staged (m)
    catch_hold_r: float = tunable(0.13)  # basket within this of the catch point when judging
    # containment (looser than stage_r: absorbs impact drift; still under the drop edge)
    fall_vz: float = tunable(-0.25)  # carton vz below this while inside = "fell in" (m/s;
    # a 14+ cm free drop enters the interior at ~-1.6; a capped carton at rest never passes)
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    shelf_yaw_deg: float = tunable(10.0)  # shelf yaw jitter about nominal (+/- deg)
    shelf_jitter: float = tunable(0.02)  # shelf xy jitter (+/- m)
    swap_sides: bool = tunable(True)  # 50%: target takes the +y / -y slab edge
    slot_x_range: tuple = tunable((-0.06, 0.04))  # carton slot x along the edge (shelf frame, m)
    slot_jitter: float = tunable(0.015)  # carton xy jitter within the slot (+/- m)
    carton_yaw_deg: float = tunable(180.0)  # carton free yaw (+/- deg)
    basket_x_range: tuple = tunable((0.08, 0.22))  # basket ground band, world x (m)
    basket_y_band: tuple = tunable((0.34, 0.46))  # |basket world y| band (sign random) (m)
    basket_yaw_deg: float = tunable(180.0)  # basket free yaw (+/- deg)

    # --- info: structure (shelf local frame: origin on the ground below the slab centre) ---------
    shelf_pos: tuple = info((0.55, 0.0))  # shelf origin on the ground
    hx: float = info(0.18)  # slab half-depth (x)
    hy: float = info(0.22)  # slab half-width (y) — the side edges the cartons stand at
    slab_top: float = info(0.26)  # slab TOP surface height
    slab_t: float = info(0.012)  # slab thickness (underside at 0.248)
    col_w: float = info(0.06)  # central column width (basket never reaches it)
    carton_s: float = info(0.100)  # carton footprint (> the 80 mm Franka jaw span: ungraspable)
    carton_h: float = info(0.115)  # carton height
    carton_mass: float = info(0.12)
    carton_mu: float = info(0.35)  # defined friction: slides ~0.45 N, tips ~1.0 N
    edge_gap: float = info(0.085)  # carton slot centre inboard of the side edge (m)
    catch_off: float = info(0.06)  # catch point outboard of the side edge (m)
    basket_half_out: float = info(0.14)  # basket outer half-span
    basket_wall_h: float = info(0.10)  # wall height above the floor slab
    basket_wall_t: float = info(0.008)  # wall/floor thickness (a graspable rim)
    basket_mass: float = info(0.50)
    shelf_color: tuple = info((0.62, 0.64, 0.68))
    col_color: tuple = info((0.35, 0.36, 0.40))
    target_color: tuple = info((0.95, 0.45, 0.05))  # ORANGE juice carton
    decoy_color: tuple = info((0.92, 0.92, 0.90))  # WHITE milk carton
    basket_color: tuple = info((0.55, 0.38, 0.20))
    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.25 + 0.35 = 0.75 <= the 0.85 non-success cap)
    w_approach: float = info(0.15)
    w_stage: float = info(0.25)
    w_catch: float = info(0.35)

    # Derived (filled in __post_init__).
    basket_inner_half: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.basket_inner_half = round(self.basket_half_out - self.basket_wall_t, 4)  # 0.132


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("ledge_catch")
class LedgeCatchScene(BaseScene):
    cfg: LedgeCatchSceneCfg

    def __init__(self, cfg: LedgeCatchSceneCfg | None = None) -> None:
        super().__init__(cfg or LedgeCatchSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        shelf_spawn = cls["shelf"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            hx=c.hx, hy=c.hy, slab_top=c.slab_top, slab_t=c.slab_t, col_w=c.col_w,
            color=c.shelf_color, col_color=c.col_color, contact_offset=c.contact_offset)
        basket_spawn = cls["basket"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.basket_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            half_out=c.basket_half_out, wall_h=c.basket_wall_h, wall_t=c.basket_wall_t,
            color=c.basket_color, contact_offset=c.contact_offset)

        def carton_spawn(color: tuple) -> Any:
            return sim_utils.CuboidCfg(
                size=(c.carton_s, c.carton_s, c.carton_h),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5, linear_damping=0.20,
                    angular_damping=0.20, sleep_threshold=0.0,
                    stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.carton_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=c.carton_mu, dynamic_friction=c.carton_mu - 0.05,
                    restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        z_slab = c.slab_top + c.carton_h / 2 + 0.003
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "shelf": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shelf",
                spawn=shelf_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.shelf_pos[0], c.shelf_pos[1], 0.0)),
            ),
            "target": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/OrangeCarton",
                spawn=carton_spawn(c.target_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.shelf_pos[0], c.shelf_pos[1] + (c.hy - c.edge_gap), z_slab)),
            ),
            "decoy": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/WhiteCarton",
                spawn=carton_spawn(c.decoy_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.shelf_pos[0], c.shelf_pos[1] - (c.hy - c.edge_gap), z_slab)),
            ),
            "basket": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Basket",
                spawn=basket_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, -0.40, 0.002)),
            ),
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
        self.shelf: RigidObject = env.iscene["shelf"]
        self.target: RigidObject = env.iscene["target"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.basket: RigidObject = env.iscene["basket"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._approach_max = torch.zeros(n, device=env.device)
        self._staged = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._caught = torch.zeros(n, dtype=torch.bool, device=env.device)
        self._side = torch.ones(n, device=env.device)  # +1: target at +y edge
        self._catch_local = torch.zeros(n, 2, device=env.device)  # catch point, shelf frame
        self._d0_basket = torch.full((n,), 0.5, device=env.device)  # spawn dist to catch

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the shelf (yaw + xy jitter), stand the target carton
        at one (sampled) side edge and the decoy at the other, drop the basket in its
        (side-sampled) ground band with free yaw; clear latches; store the catch
        point and the basket's own spawn distance to it (the normalizer that makes
        null-policy credit exactly zero)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- shelf: kinematic, yaw + xy jitter ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.shelf_yaw_deg)
        sx = c.shelf_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.shelf_jitter
        sy = c.shelf_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.shelf_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = sx, sy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.shelf.write_root_state_to_sim(st, env_ids)

        # --- which side edge holds the target ---
        side = torch.where(
            (torch.rand(m, device=dev) < 0.5) if c.swap_sides
            else torch.zeros(m, dtype=torch.bool, device=dev),
            torch.ones(m, device=dev), -torch.ones(m, device=dev))
        self._side[env_ids] = side

        def band(lo_hi: tuple, mm: int) -> torch.Tensor:
            return lo_hi[0] + torch.rand(mm, device=dev) * (lo_hi[1] - lo_hi[0])

        # --- cartons: standing on the slab at their side's edge slot, free yaw ---
        z_slab = c.slab_top + c.carton_h / 2 + 0.003
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        slot_x = band(c.slot_x_range, m)
        for obj, sgn in ((self.target, side), (self.decoy, -side)):
            lx = band(c.slot_x_range, m) if obj is self.decoy else slot_x
            lx = lx + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            ly = sgn * (c.hy - c.edge_gap) \
                + (torch.rand(m, device=dev) * 2 - 1) * c.slot_jitter
            cyaw = yaw + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.carton_yaw_deg)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = sx + cos_y * lx - sin_y * ly
            st[:, 1] = sy + sin_y * lx + cos_y * ly
            st[:, 2] = z_slab
            st[:, 3], st[:, 6] = torch.cos(cyaw / 2), torch.sin(cyaw / 2)
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)

        # --- basket: on the ground in its band, random side, free yaw ---
        bx = band(c.basket_x_range, m)
        bside = torch.where(torch.rand(m, device=dev) < 0.5,
                            torch.ones(m, device=dev), -torch.ones(m, device=dev))
        by = bside * band(c.basket_y_band, m)
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.basket_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1], st[:, 2] = bx, by, 0.002
        st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.basket.write_root_state_to_sim(st, env_ids)

        # --- clear latches; store the catch point + spawn-distance normalizer ---
        self._approach_max[env_ids] = 0.0
        self._staged[env_ids] = False
        self._caught[env_ids] = False
        catch = torch.stack([slot_x, side * (c.hy + c.catch_off)], dim=1)
        self._catch_local[env_ids] = catch
        # basket spawn position in the shelf frame
        dxw, dyw = bx - sx, by - sy
        bxl = cos_y * dxw + sin_y * dyw
        byl = -sin_y * dxw + cos_y * dyw
        self._d0_basket[env_ids] = torch.sqrt(
            (bxl - catch[:, 0]) ** 2 + (byl - catch[:, 1]) ** 2).clamp(min=0.05)

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shelf": self.shelf.data.root_state_w[env_ids].clone(),
            "target": self.target.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "basket": self.basket.data.root_state_w[env_ids].clone(),
            "approach_max": self._approach_max[env_ids].clone(),
            "staged": self._staged[env_ids].clone(),
            "caught": self._caught[env_ids].clone(),
            "side": self._side[env_ids].clone(),
            "catch_local": self._catch_local[env_ids].clone(),
            "d0_basket": self._d0_basket[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shelf.write_root_state_to_sim(state["shelf"], env_ids)
        self.target.write_root_state_to_sim(state["target"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.basket.write_root_state_to_sim(state["basket"], env_ids)
        self._approach_max[env_ids] = state["approach_max"]
        self._staged[env_ids] = state["staged"]
        self._caught[env_ids] = state["caught"]
        self._side[env_ids] = state["side"]
        self._catch_local[env_ids] = state["catch_local"]
        self._d0_basket[env_ids] = state["d0_basket"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A grey PEDESTAL SHELF stands on the ground: a single central column "
            f"carrying a flat top slab ({2 * c.hx * 100:.0f} x {2 * c.hy * 100:.0f} cm, "
            f"top surface {c.slab_top * 100:.0f} cm up) that overhangs the column on all "
            f"sides, so the ground under every slab edge is open. Two identical juice "
            f"cartons ({c.carton_s * 100:.0f} x {c.carton_s * 100:.0f} cm footprint, "
            f"{c.carton_h * 100:.1f} cm tall) stand on the slab, one near each LONG side "
            f"edge: the ORANGE juice carton (the target — which side it is on varies per "
            f"episode) and a WHITE milk carton (a distractor that must be left standing "
            f"on the shelf). An open-top brown BASKET (outer "
            f"{2 * c.basket_half_out * 100:.0f} cm square, {c.basket_wall_h * 100:.0f} cm "
            f"walls) sits on the ground well away from the shelf.\n"
            f"Both cartons are {c.carton_s * 100:.0f} cm wide on every horizontal axis — "
            f"wider than a parallel-jaw gripper can span — so the orange carton cannot be "
            f"grasped and carried; the only way to get it into the basket is to let it "
            f"FALL in. Goal, in this order: FIRST move the basket (its "
            f"{c.basket_wall_t * 1000:.0f} mm rim is graspable, or slide it along the "
            f"ground) so it sits upright on the ground directly under the slab edge "
            f"nearest the ORANGE carton — its open top under the drop point, roughly "
            f"{c.catch_off * 100:.0f} cm outboard of the edge; THEN push the ORANGE "
            f"carton horizontally along the slab, over that edge, so it topples off and "
            f"lands inside the basket. Push gently: the carton slides when pushed low "
            f"and slowly, and a hard shove can overshoot the basket. The order is "
            f"forced: only a carton that FALLS from the shelf into the already-placed "
            f"basket counts. A carton that reaches the ground outside the basket is "
            f"lost — it is too wide to grasp, the basket walls block pushing it in, and "
            f"lowering the basket down over a grounded carton does NOT count. Success: the ORANGE carton at rest fully "
            f"inside the upright basket on the ground, with the basket still sitting at "
            f"that catch spot under the drop edge, the WHITE carton still standing on "
            f"the shelf, everything settled. The orange carton on the ground, balanced "
            f"on the basket rim, the white carton pushed off, a tipped-over basket, or "
            f"the basket (even with the carton in it) moved away from the drop edge all "
            f"fail."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Place the open basket on the ground under the shelf edge next to the "
            "ORANGE juice carton, then push the orange carton off that edge so it falls "
            "into the basket. It is too wide to grasp, so stage the basket first; "
            "leave the basket under that edge and leave the WHITE milk carton standing "
            "on the shelf."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _shelf_local(self, obj) -> torch.Tensor:
        """Object centre in the SHELF'S body frame, (N, 3) — slot geometry and the
        catch point live in this frame so a yawed/jittered shelf judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.shelf.data.root_pos_w
        return quat_apply_inverse(self.shelf.data.root_quat_w, rel)

    def _basket_local(self, obj) -> torch.Tensor:
        """Object centre in the BASKET'S body frame, (N, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.basket.data.root_pos_w
        return quat_apply_inverse(self.basket.data.root_quat_w, rel)

    def _on_shelf(self, obj) -> torch.Tensor:
        """(N,) bool: carton resting ON the slab top (any orientation)."""
        c = self.cfg
        loc = self._shelf_local(obj)
        return (loc[:, 2] > c.slab_top + 0.02) & (loc[:, 2] < c.slab_top + 0.16) \
            & (loc[:, 0].abs() < c.hx + 0.01) & (loc[:, 1].abs() < c.hy + 0.01)

    def _basket_upright(self) -> torch.Tensor:
        """(N,) bool: basket upright ON the ground (its root is the outer bottom
        centre — a toppled or lifted basket fails)."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.basket.data.root_quat_w, ez)
        z = (self.basket.data.root_pos_w - self.env_origins)[:, 2]
        return (up[:, 2] >= c.basket_up_min) & (z < c.basket_z_max) & (z > -0.01)

    def _basket_at_catch(self, r: float) -> torch.Tensor:
        """(N,) bool: basket centre within `r` of the catch point (shelf frame)."""
        bl = self._shelf_local(self.basket)
        return (bl[:, :2] - self._catch_local).norm(dim=-1) < r

    def _target_inside_now(self) -> torch.Tensor:
        """(N,) bool: target carton fully inside the basket interior — centre near
        the basket axis, above the floor, BELOW the wall top (a carton straddling
        the rim sits ~4 cm higher) — with the basket itself upright on the ground
        AND still at the catch station under the target's slab edge (containment
        anywhere else, e.g. the seed's item-into-the-spawned-basket end state, does
        not count: the only physical route in is the fall at this edge)."""
        c = self.cfg
        loc = self._basket_local(self.target)
        return (loc[:, 0].abs() < c.inside_xy_max) & (loc[:, 1].abs() < c.inside_xy_max) \
            & (loc[:, 2] > c.inside_z_min) & (loc[:, 2] < c.inside_z_max) \
            & self._basket_upright() & self._basket_at_catch(c.catch_hold_r)

    def _staged_now(self) -> torch.Tensor:
        """(N,) bool: basket upright on the ground, centred within `stage_r` of the
        catch point, WHILE the target is still on the shelf (the order gate)."""
        c = self.cfg
        bl = self._shelf_local(self.basket)
        d = (bl[:, :2] - self._catch_local).norm(dim=-1)
        return (d < c.stage_r) & self._basket_upright() & self._on_shelf(self.target)

    def _update_latches(self) -> None:
        """Refresh the latches: `approach` is the running max of the basket's
        progress toward the catch point (normalized by its own spawn distance —
        exactly 0 for the null policy); `staged` once the basket is at the catch
        point with the carton still up (order-gated); `caught` once the target is
        observed FALLING INTO the already-staged basket (inside the interior at
        the catch station while descending faster than `fall_vz`) — the
        anti-capture gate: lowering the basket over a grounded carton, or any
        teleport-constructed containment at rest, never sets it."""
        bl = self._shelf_local(self.basket)
        d = (bl[:, :2] - self._catch_local).norm(dim=-1)
        appr = (1.0 - d / self._d0_basket).clamp(0.0, 1.0)
        self._approach_max = torch.maximum(self._approach_max, appr)
        self._staged |= self._staged_now()
        vz = self.target.data.root_lin_vel_w[:, 2]
        self._caught |= (self._target_inside_now() & self._staged
                         & (vz < self.cfg.fall_vz))

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the ORANGE carton settled fully inside the upright, grounded
        basket still at the catch station, having demonstrably FALLEN in (the
        caught latch — capping a grounded carton never sets it), the WHITE decoy
        still standing on the shelf, everything settled."""
        c = self.cfg
        self._update_latches()
        still = (self.target.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.basket.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
        return self._target_inside_now() & self._caught & self._on_shelf(self.decoy) \
            & still

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*approach + 0.25*staged + 0.35*caught (all
        latched; staging order-gated on the carton still being shelf-borne; ~0 for
        doing nothing) — capped at 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_approach * self._approach_max + c.w_stage * self._staged.float()
                + c.w_catch * self._caught.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="ledge_catch", robot="null"))
