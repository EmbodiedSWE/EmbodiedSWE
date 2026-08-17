"""LipHangRackScene — hang the wine bottle by its LIP in the GREEN-tagged slot of an
overhead hanging rack. Derived from libero_90 "put the wine bottle on the wine rack",
but the support relation is INVERTED: the seed rests the bottle ON TOP of a rack
surface (support from below, judged by a bbox over the rack top); here the bottle must
end SUSPENDED — hanging from a pair of overhead rails by its neck lip, body dangling
in free air — and the only way in is a horizontal THREADING of the neck into the open
front end of the correct slot followed by an 11 cm slide back to the tower stop.

Scene: a floor-standing RACK TOWER carries two cantilevered HANGING SLOTS side by
side; each slot is a pair of parallel horizontal rails with a 34 mm gap between them,
open at the FRONT end, closed at the back by the tower itself. A colored TAG above
each slot marks it: the GREEN tag marks the target slot, the RED tag the wrong one —
and the tags swap sides between episodes, so the slot must be identified by color,
not remembered by side. One green glass WINE BOTTLE (55 mm body, 20 mm neck, 50 mm
lip disk at the top of the neck) stands on the floor in front of the rack.

Goal: the bottle hangs by its lip in the GREEN slot, pushed back to the tower. The
lip is wider than the rail gap, the neck narrower: with the neck lowered between the
rails the lip catches on the rail tops and the bottle hangs vertically, body in free
air. Metric interlocks kill every non-threading route:
  - resting the bottle anywhere ON the rack (across the rail tops, on the tower top)
    is the seed's relation and scores nothing — the rubric demands the hanging pose
    (lip at rail-top height, bottle upright, body below the rails);
  - the space BETWEEN the two slots' inner rails is 76 mm — wider than the 50 mm lip —
    so the "false middle slot" cannot hold a bottle: it falls straight through;
  - the slot is closed at the back and tagged: hanging in the RED slot, or short of
    the tower stop, fails.

A solver therefore needs a different PLAN from the seed (grasp the body, lift to rail
height, thread the neck between the rails from the open front end with the lip above
the rail tops, slide backward ~11 cm to the stop, release into a stable hang —
versus lower-onto-a-surface), and a different code structure (suspension rubric in
the rack frame + slide-to-stop progress, versus a static top-surface bbox test).

Assets are fully procedural (compound spawners, raw pxr authoring; child colliders of
one body never self-collide):
  - rack: KINEMATIC compound — tower slab + 4 rail boxes (2 slots). Origin at the
    tower base centre on the floor; local +x points FRONT (toward the robot), rails
    run along +x from the tower face (x=0.03) to their open ends (x=0.17), rail top
    plane at z=0.320.
  - tags: two small KINEMATIC cubes (green/red), NO colliders (pure markers),
    re-posed to their episode sides at reset.
  - bottle: DYNAMIC compound — body cylinder + neck cylinder + lip disk, origin at
    the assembly's mid-length (CoM there by MassAPI), so a hanging bottle's origin
    sits 112.5 mm below the lip underside. Damping so the pendulum hang settles.

Per-episode randomization (readback-verifiable): rack xy jitter + yaw, GREEN side
Bernoulli (tags physically swap), bottle spawn side Bernoulli + xy jitter + yaw.

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.15 * lift      — bottle origin ever above 0.17 m (any carry; a null policy never)
  0.15 * mouth     — neck ever threaded into the GREEN slot (rack-frame lateral +
                     height + upright bands, latched)
  0.30 * depth     — running max of slide-to-stop progress while threaded (latched)
  1.0 iff success()— bottle hanging by its lip in the GREEN slot at the tower stop:
                     depth within the stop band, lip at rail-top height, upright,
                     laterally in the slot, at rest. Non-success cap 0.60.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering
the scene — stays app-free.
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


def _add_box(stage, path: str, *, center, size, color, collide: Callable) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _add_cyl(stage, path: str, *, center, radius, height, color, collide: Callable) -> None:
    """One z-axis cylinder collider (PhysX convex-approximates the gprim; the flat
    end faces stay flat — the lip's underside slides on flat rail tops)."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    r, h = float(radius), float(height) / 2
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h), Gf.Vec3f(r, r, h)])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())


def _phys_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(static))
    pm.CreateDynamicFrictionAttr(float(dynamic))
    pm.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _spawn_rack(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the rack: KINEMATIC compound. Origin at the tower base centre on the
    floor; local +x = front. Tower slab + two rail pairs (slots at v = +/- slot_v,
    inner gap `gap`, rail tops at rail_top_z, open front ends at x = face_x +
    rail_len). Rails and tower face get a slick material so the lip slides."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    slick = _phys_material(stage, f"{prim_path}/slick", c.mu_static, c.mu_dynamic)
    _add_box(stage, f"{prim_path}/tower",
             center=(0.0, 0.0, c.tower_h / 2),
             size=(c.tower_t, c.tower_w, c.tower_h), color=c.rack_color,
             collide=collide)
    _bind_material(stage.GetPrimAtPath(f"{prim_path}/tower"), slick)
    x0 = c.tower_t / 2
    for s_sgn, s_nm in ((1.0, "a"), (-1.0, "b")):
        for r_sgn, r_nm in ((1.0, "i"), (-1.0, "o")):
            # rails at v = slot_v -/+ (gap/2 + rail_w/2): "i" toward the centre
            v = s_sgn * (c.slot_v - r_sgn * (c.gap / 2 + c.rail_w / 2))
            nm = f"rail_{s_nm}{r_nm}"
            _add_box(stage, f"{prim_path}/{nm}",
                     center=(x0 + c.rail_len / 2, v, c.rail_top_z - c.rail_t / 2),
                     size=(c.rail_len, c.rail_w, c.rail_t), color=c.rail_color,
                     collide=collide)
            _bind_material(stage.GetPrimAtPath(f"{prim_path}/{nm}"), slick)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bottle: DYNAMIC compound (body + neck + lip disk), origin at the
    assembly mid-length (MassAPI puts the CoM there — 112.5 mm below the lip
    underside, so the hang is a stable pendulum). Damping so the hang settles."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.50)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(1)
    collide = _make_collide(cfg.contact_offset)
    slick = _phys_material(stage, f"{prim_path}/slick", cfg.mu_static, cfg.mu_dynamic)
    c = cfg
    half_l = (c.h_body + c.h_neck + c.h_lip) / 2
    _add_cyl(stage, f"{prim_path}/body",
             center=(0.0, 0.0, -half_l + c.h_body / 2),
             radius=c.r_body, height=c.h_body, color=c.body_color, collide=collide)
    _add_cyl(stage, f"{prim_path}/neck",
             center=(0.0, 0.0, -half_l + c.h_body + c.h_neck / 2),
             radius=c.r_neck, height=c.h_neck, color=c.body_color, collide=collide)
    _add_cyl(stage, f"{prim_path}/lip",
             center=(0.0, 0.0, half_l - c.h_lip / 2),
             radius=c.r_lip, height=c.h_lip, color=c.lip_color, collide=collide)
    for child in ("neck", "lip"):
        _bind_material(stage.GetPrimAtPath(f"{prim_path}/{child}"), slick)
    return root


def _spawner_classes() -> dict[str, Any]:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "rack" not in _SPAWNER_CACHE:

        @configclass
        class RackSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rack)
            tower_t: float = 0.06
            tower_w: float = 0.30
            tower_h: float = 0.38
            rail_len: float = 0.14
            rail_w: float = 0.020
            rail_t: float = 0.012
            rail_top_z: float = 0.320
            gap: float = 0.034
            slot_v: float = 0.075
            rack_color: tuple = (0.45, 0.30, 0.16)
            rail_color: tuple = (0.62, 0.46, 0.26)
            mu_static: float = 0.25
            mu_dynamic: float = 0.20
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            r_body: float = 0.0275
            h_body: float = 0.160
            r_neck: float = 0.010
            h_neck: float = 0.075
            r_lip: float = 0.025
            h_lip: float = 0.010
            mass: float = 0.45
            body_color: tuple = (0.10, 0.32, 0.12)
            lip_color: tuple = (0.16, 0.14, 0.10)
            mu_static: float = 0.25
            mu_dynamic: float = 0.20
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(rack=RackSpawnerCfg, bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LipHangRackSceneCfg(BaseCfg):
    """Config for `LipHangRackScene`. The interlocks are metric: the lip (50 mm) is
    wider than the rail gap (34 mm) so a threaded neck hangs, the neck (20 mm) is
    narrower so it threads with +/-7 mm of play, the inter-slot span (76 mm) is wider
    than the lip so the false middle slot cannot hold anything, the rails are open
    ONLY at the front, and the hanging bottle's bottom floats 75 mm above the floor
    (asserted in __post_init__)."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.60)  # max |ang vel| when judging (rad/s)
    hang_z_lo: float = tunable(0.009)  # hang band below nominal origin height (m)
    hang_z_hi: float = tunable(0.012)  # hang band above nominal origin height (m)
    upright_max_deg: float = tunable(15.0)  # bottle axis within this of world-up
    lat_tol: float = tunable(0.012)  # |origin v - green slot centre| bound (m)
    d_min: float = tunable(0.008)  # success depth window: lip against the tower ...
    d_stop: float = tunable(0.045)  # ... to at most this far from the tower face
    p_full: float = tunable(0.040)  # depth at which slide progress saturates to 1
    mouth_lat: float = tunable(0.016)  # threading latch: looser lateral band
    mouth_z: float = tunable(0.022)  # threading latch: looser height band
    mouth_max_deg: float = tunable(25.0)  # threading latch: looser upright band
    lift_z: float = tunable(0.170)  # lift latch: origin height (stand is 0.1225)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    rack_jitter: float = tunable(0.03)  # rack xy jitter (+/- m)
    rack_yaw_deg: float = tunable(10.0)  # rack yaw about front-facing (+/- deg)
    bottle_jitter: float = tunable(0.03)  # bottle spawn xy jitter (+/- m)
    swap_sides: bool = tunable(True)  # Bernoulli green-slot side + bottle side

    # --- info: layout (single Franka base at the origin; radii 0.25-0.65 m) ---------------------
    rack_x: float = info(0.62)  # tower centre distance from the base
    bottle_slot: tuple = info((0.30, 0.20))  # bottle spawn (x, +/-y Bernoulli)

    # --- info: rack structure --------------------------------------------------------------------
    tower_t: float = info(0.06)  # tower thickness (x) — the back stop
    tower_w: float = info(0.30)
    tower_h: float = info(0.38)
    rail_len: float = info(0.14)  # rail length (x), open front ends
    rail_w: float = info(0.020)
    rail_t: float = info(0.012)
    rail_top_z: float = info(0.320)  # rail TOP plane (the lip rides here)
    gap: float = info(0.034)  # inner gap between a slot's two rails
    slot_v: float = info(0.075)  # slot centres at v = +/- slot_v
    tag_size: float = info(0.030)
    tag_x: float = info(0.046)  # tag centre, rack-local x (proud of the tower face)
    tag_z: float = info(0.355)  # tag centre height (above the rails)
    rack_color: tuple = info((0.45, 0.30, 0.16))
    rail_color: tuple = info((0.62, 0.46, 0.26))
    green: tuple = info((0.05, 0.75, 0.15))
    red: tuple = info((0.85, 0.08, 0.08))

    # --- info: bottle ----------------------------------------------------------------------------
    r_body: float = info(0.0275)  # 55 mm body — fits the 80 mm Franka jaw
    h_body: float = info(0.160)
    r_neck: float = info(0.010)  # 20 mm neck — threads the 34 mm gap
    h_neck: float = info(0.075)
    r_lip: float = info(0.025)  # 50 mm lip — cannot pass the 34 mm gap
    h_lip: float = info(0.010)
    bottle_mass: float = info(0.45)
    body_color: tuple = info((0.10, 0.32, 0.12))
    lip_color: tuple = info((0.16, 0.14, 0.10))
    mu_static: float = info(0.25)  # slick rails/neck/lip: the slide must not jam
    mu_dynamic: float = info(0.20)

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.15 + 0.30 = 0.60 = the non-success cap)
    w_lift: float = info(0.15)
    w_mouth: float = info(0.15)
    w_depth: float = info(0.30)

    # Derived (filled in __post_init__).
    half_l: float = field(default=0.0, init=False)  # origin to either end
    lip_under: float = field(default=0.0, init=False)  # origin to lip underside
    hang_z0: float = field(default=0.0, init=False)  # nominal hanging origin height
    stand_z: float = field(default=0.0, init=False)  # standing origin height
    face_x: float = field(default=0.0, init=False)  # tower front face, rack-local x
    d_mouth: float = field(default=0.0, init=False)  # staging depth (progress ramp 0)

    def __post_init__(self) -> None:
        total = self.h_body + self.h_neck + self.h_lip
        self.half_l = total / 2
        self.lip_under = self.half_l - self.h_lip
        self.hang_z0 = self.rail_top_z - self.lip_under
        self.stand_z = self.half_l
        self.face_x = self.tower_t / 2
        self.d_mouth = self.rail_len - 0.030
        # --- honesty-by-construction assertions -------------------------------------------------
        assert self.gap >= 2 * self.r_neck + 0.012, "neck threading play too tight"
        assert self.r_lip >= self.gap / 2 + 0.006, "lip could fall through the slot"
        mid_span = 2 * (self.slot_v - self.gap / 2 - self.rail_w)
        assert mid_span >= 2 * self.r_lip + 0.020, "false middle slot could hold the lip"
        assert self.rail_top_z - total > 0.05, "hanging bottle would touch the floor"
        assert total < self.rail_top_z - self.rail_t - 0.05, \
            "a bottle standing under the slot would reach the rails"
        assert 2 * self.r_body <= 0.078, "body too wide for the Franka jaw"
        assert self.d_stop > self.r_lip + 2 * self.contact_offset + 0.004, \
            "success depth window unreachable (lip stops on the tower face)"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("lip_hang_rack")
class LipHangRackScene(BaseScene):
    cfg: LipHangRackSceneCfg

    def __init__(self, cfg: LipHangRackSceneCfg | None = None) -> None:
        super().__init__(cfg or LipHangRackSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        rack_spawn = spawners["rack"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            tower_t=c.tower_t, tower_w=c.tower_w, tower_h=c.tower_h,
            rail_len=c.rail_len, rail_w=c.rail_w, rail_t=c.rail_t,
            rail_top_z=c.rail_top_z, gap=c.gap, slot_v=c.slot_v,
            rack_color=c.rack_color, rail_color=c.rail_color,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset,
        )
        bottle_spawn = spawners["bottle"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            r_body=c.r_body, h_body=c.h_body, r_neck=c.r_neck, h_neck=c.h_neck,
            r_lip=c.r_lip, h_lip=c.h_lip, mass=c.bottle_mass,
            body_color=c.body_color, lip_color=c.lip_color,
            mu_static=c.mu_static, mu_dynamic=c.mu_dynamic,
            contact_offset=c.contact_offset,
        )

        def tag_spawn(color):
            # pure marker: kinematic, NO collider — it cannot be knocked or leaned on
            return sim_utils.CuboidCfg(
                size=(c.tag_size, c.tag_size, c.tag_size),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            )

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
            "rack": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rack",
                spawn=rack_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_x, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),  # yaw pi
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=bottle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.bottle_slot[0], c.bottle_slot[1], c.stand_z + 0.002)),
            ),
            "green_tag": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/GreenTag",
                spawn=tag_spawn(c.green),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_x - c.tag_x, c.slot_v, c.tag_z)),
            ),
            "red_tag": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedTag",
                spawn=tag_spawn(c.red),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.rack_x - c.tag_x, -c.slot_v, c.tag_z)),
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
        self.rack: RigidObject = env.iscene["rack"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.green_tag: RigidObject = env.iscene["green_tag"]
        self.red_tag: RigidObject = env.iscene["red_tag"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # green slot centre, rack-local v (+/- slot_v; the tags physically mark it)
        self._green_v = torch.full((n,), self.cfg.slot_v, device=dev)
        # latches: partial progress survives transient achievements
        self._lift_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._mouth_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._depth_max = torch.zeros(n, device=dev)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: rack re-posed (xy jitter + yaw about front-facing), GREEN
        side sampled and the tag bodies physically swapped to it, bottle stood on a
        Bernoulli spawn side with jitter + free yaw, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- rack (kinematic): front faces the robot (yaw pi) + jitter/yaw ---
        yaw = math.pi + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.rack_yaw_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.rack_x
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.rack_jitter
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.rack.write_root_state_to_sim(st, env_ids)
        r_pos, r_quat = st[:, 0:3].clone(), st[:, 3:7].clone()

        # --- green side + tags (kinematic markers re-posed in the rack frame) ---
        from isaaclab.utils.math import quat_apply

        if c.swap_sides:
            g_sgn = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            g_sgn = torch.ones(m, device=dev)
        self._green_v[env_ids] = g_sgn * c.slot_v
        for body, sgn in ((self.green_tag, g_sgn), (self.red_tag, -g_sgn)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0], loc[:, 1], loc[:, 2] = c.tag_x, sgn * c.slot_v, c.tag_z
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = r_pos + quat_apply(r_quat, loc)
            st[:, 3:7] = r_quat
            body.write_root_state_to_sim(st, env_ids)

        # --- bottle: standing on a Bernoulli spawn side + jitter + free yaw ---
        if c.swap_sides:
            b_sgn = torch.where(torch.rand(m, device=dev) < 0.5, 1.0, -1.0)
        else:
            b_sgn = torch.ones(m, device=dev)
        byaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.bottle_slot[0]
        st[:, 1] = b_sgn * c.bottle_slot[1]
        st[:, 0:2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.bottle_jitter
        st[:, 2] = c.stand_z + 0.002
        st[:, 3], st[:, 6] = torch.cos(byaw / 2), torch.sin(byaw / 2)
        st[:, 0:3] += origin
        self.bottle.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._lift_ever[env_ids] = False
        self._mouth_ever[env_ids] = False
        self._depth_max[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "rack": self.rack.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "green_tag": self.green_tag.data.root_state_w[env_ids].clone(),
            "red_tag": self.red_tag.data.root_state_w[env_ids].clone(),
            "green_v": self._green_v[env_ids].clone(),
            "lift_ever": self._lift_ever[env_ids].clone(),
            "mouth_ever": self._mouth_ever[env_ids].clone(),
            "depth_max": self._depth_max[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.rack.write_root_state_to_sim(state["rack"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.green_tag.write_root_state_to_sim(state["green_tag"], env_ids)
        self.red_tag.write_root_state_to_sim(state["red_tag"], env_ids)
        self._green_v[env_ids] = state["green_v"]
        self._lift_ever[env_ids] = state["lift_ever"]
        self._mouth_ever[env_ids] = state["mouth_ever"]
        self._depth_max[env_ids] = state["depth_max"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wooden RACK TOWER stands on the floor in front of the robot: a "
            f"{c.tower_h * 100:.0f} cm tall slab carrying TWO HANGING SLOTS side by "
            f"side at {c.rail_top_z * 100:.0f} cm height. Each slot is a pair of "
            f"parallel horizontal rails ({c.rail_len * 100:.0f} cm long) with a "
            f"{c.gap * 1000:.0f} mm gap between them, OPEN at the front end (the end "
            f"facing the robot) and closed at the back by the tower. Above each slot, "
            f"a colored cube TAG on the tower face marks it: one GREEN, one RED — the "
            f"colors swap sides between episodes, so identify the slot by its tag, "
            f"not by side. On the floor in front of the rack stands a green glass "
            f"WINE BOTTLE ({2 * c.r_body * 1000:.0f} mm body, "
            f"{2 * c.r_neck * 1000:.0f} mm neck, and a {2 * c.r_lip * 1000:.0f} mm "
            f"LIP disk at the top of the neck).\n"
            f"Goal: the bottle must end up HANGING by its lip in the GREEN-tagged "
            f"slot, pushed all the way back to the tower. The lip is wider than the "
            f"rail gap and the neck narrower, so: grasp the bottle body, lift it "
            f"upright until the lip is just above the rail tops, thread the neck "
            f"DOWN-BETWEEN the two rails of the green slot from the OPEN FRONT END, "
            f"slide the bottle backward along the slot (lip riding on the rail tops) "
            f"until it stops against the tower, and release it so it hangs "
            f"vertically by the lip with the body dangling in free air. A bottle "
            f"resting ON TOP of the rack (across the rails or on the tower top), "
            f"standing on the floor, left short of the tower stop, dropped between "
            f"the two slots (the middle span is wider than the lip — it falls "
            f"through), or hung in the RED-tagged slot — all failure."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pick up the wine bottle and hang it in the rack slot marked by the "
            "GREEN tag: thread its neck between that slot's two rails from the open "
            "front end, slide it back until it stops at the tower, and let it hang "
            "freely by its lip. Hanging in the red-tagged slot fails."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def _rack_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(N, 3) world points in the rack frame (+x front, +y across, z up)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.rack.data.root_quat_w,
                                  pos_w - self.rack.data.root_pos_w)

    def _bottle_loc(self) -> torch.Tensor:
        """(N, 3) bottle ORIGIN in the rack frame."""
        return self._rack_local(self.bottle.data.root_pos_w)

    def _bottle_up(self) -> torch.Tensor:
        """(N,) world-z component of the bottle's +z axis (1 = perfectly upright)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(self.bottle.data.root_quat_w, ez)[:, 2].clamp(-1.0, 1.0)

    def _threaded(self, lat_tol: float, z_tol: float, max_deg: float) -> torch.Tensor:
        """(N,) bool: neck threaded into the GREEN slot — origin laterally within
        lat_tol of the green slot centre, at hanging height within z_tol, within the
        rail span, upright within max_deg. Judged in the rack frame."""
        c = self.cfg
        loc = self._bottle_loc()
        lat_ok = (loc[:, 1] - self._green_v).abs() <= lat_tol
        z_ok = (loc[:, 2] - c.hang_z0).abs() <= z_tol
        x_ok = (loc[:, 0] >= c.face_x - 0.005) & (loc[:, 0] <= c.face_x + c.rail_len + 0.015)
        up_ok = self._bottle_up() >= math.cos(math.radians(max_deg))
        return lat_ok & z_ok & x_ok & up_ok

    def _depth(self) -> torch.Tensor:
        """(N,) distance from the bottle origin to the tower face along the slot."""
        return self._bottle_loc()[:, 0] - self.cfg.face_x

    def _settled(self) -> torch.Tensor:
        c = self.cfg
        return ((self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed)
                & (self.bottle.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega))

    def _update_latches(self) -> None:
        c = self.cfg
        loc = self._bottle_loc()
        self._lift_ever |= loc[:, 2] >= c.lift_z
        threaded = self._threaded(c.mouth_lat, c.mouth_z, c.mouth_max_deg)
        self._mouth_ever |= threaded
        # slide-to-stop progress, counted only while threaded in the green slot
        p = ((c.d_mouth - self._depth()) / (c.d_mouth - c.p_full)).clamp(0.0, 1.0)
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0) * threaded.float()
        self._depth_max = torch.maximum(self._depth_max, p)

    # ----- step-coupled bookkeeping (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """No plant (the rack is kinematic and jointless) — just latch rubric
        progress every step so transient achievements keep credit."""
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the bottle HANGS by its lip in the GREEN slot at the tower
        stop — depth within [d_min, d_stop] of the tower face, lip at rail-top
        height (hang band), upright, laterally in the slot, at rest. Physical,
        settled outcomes only."""
        c = self.cfg
        self._update_latches()
        d = self._depth()
        loc = self._bottle_loc()
        hang_z = ((loc[:, 2] - c.hang_z0 >= -c.hang_z_lo)
                  & (loc[:, 2] - c.hang_z0 <= c.hang_z_hi))
        lat_ok = (loc[:, 1] - self._green_v).abs() <= c.lat_tol
        up_ok = self._bottle_up() >= math.cos(math.radians(c.upright_max_deg))
        depth_ok = (d >= c.d_min) & (d <= c.d_stop)
        return hang_z & lat_ok & up_ok & depth_ok & self._settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*lift + 0.15*mouth (threading, green-gated) +
        0.30*depth progress — all latched, ~0 for doing nothing, capped 0.60 — and
        exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_lift * self._lift_ever.float()
                + c.w_mouth * self._mouth_ever.float()
                + c.w_depth * self._depth_max).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through applied wrenches.
register_env("simgen", lambda: EnvCfg(scene="lip_hang_rack", robot="null"))
