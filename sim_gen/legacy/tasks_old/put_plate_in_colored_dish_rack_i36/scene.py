"""PlateCarouselScene — indexed loading of a shrouded plate carousel (sim_gen task
`put_plate_in_colored_dish_rack_i36`, derived from rlbench/put_plate_in_colored_dish_rack).

The seed is a single free placement: pick the plate off its stand and slide it into the
colored slot of a static, fully exposed dish rack — one grasp, one transport, one
insertion, every goal slot reachable at all times. This task keeps the seed's object
vocabulary (flat colored plates + a color-slotted dish fixture) but makes the fixture a
MACHINE whose goal slots are accessible only one at a time:

  - The dish rack becomes a rotating CAROUSEL: a free-spinning turntable deck riding on
    a low-friction drum (a physical bearing — the deck's under-skirt is captured
    laterally around the drum, so it spins freely but cannot be dragged off), carrying
    three color-coded plate pockets (red / green / blue walled corrals) at 120 deg
    spacing.
  - A fixed SHROUD (roof + outer wall) covers the carousel except for one 120 deg
    loading WINDOW at a per-episode azimuth. A plate can only be lowered into a pocket
    while that pocket is rotated under the window — everywhere else the roof and wall
    physically block insertion.
  - Three colored dinner plates start flat on the floor on the far side, on shuffled
    stands. Each must end seated flat inside the pocket of its own color.

So the seed's plan (carry each plate straight to its visible colored slot) is
IMPOSSIBLE as stated: at most one pocket is ever exposed. The solver must interleave
MECHANISM ACTUATION with transport — spin the deck until the right pocket indexes under
the window, load that plate, spin again (already-loaded plates ride the carousel under
the roof), load the next, and again: rotate -> insert, three times, ~6 dependent stages.

Judged on PHYSICAL outcomes plus through-the-window entry latches:
  - `seated_geo[i]`: plate i's centre within `pocket_xy_tol` of ITS pocket's axis IN THE
    DECK BODY FRAME (the pocket moves; the rubric rides with it), at seat height on the
    deck top, lying flat (axis within `flat_max_deg` of the deck axis). Honest by
    construction: the corral walls put the max physical seated offset at 8 mm < the
    15 mm tolerance, and a plate resting on a corral wall or on the shroud roof is
    rejected by the z band.
  - `loaded[i]` (latch, post_step): set only at an ENTRY EVENT — seated_geo becomes
    true from not-seated on the previous substep, with a per-substep displacement under
    5 cm (anti-teleport), while pocket i sits within `window_half_deg` of the window
    azimuth. A plate teleported into a covered pocket is seated_geo but never loaded.
  - `engaged` (latch): cumulative smooth deck rotation exceeds ~29 deg (per-substep yaw
    steps above 3.4 deg do not accumulate — a yaw teleport earns nothing).
  - success(): all three plates seated_geo AND settled AND loaded.
  - score(): 0.10 * engaged + 0.25 * (# plates currently seated + settled + loaded);
    exactly 1.0 iff success (max non-success partial 0.60 + 0.10).

Per-episode randomization (readback-verified in the smoke): initial deck yaw, shroud
window azimuth, and the plates' stand permutation + xy jitter — how far and in which
order the deck must be indexed differs every episode.

Assets are fully procedural: kinematic drum (low-friction top, combine-mode "min"),
one dynamic compound deck (disc + capture skirt + 3 colored corrals), one kinematic
compound shroud (roof + wall ring, 240 deg), three dynamic cylinder plates. No external
files. Heavy imports (isaaclab, pxr) are deferred so importing this module — and
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


# ----- custom compound spawners ----------------------------------------------------------------
# One rigid body each, child colliders authored with raw pxr APIs; `isaaclab.sim.utils.clone`
# supplies the regex-resolve + per-env replication (the pen_holder pattern — each cloned prim is
# authored fresh, so the duplicate-xformOp trap does not arise).

_SPAWNER_CACHE: dict[str, Any] = {}


def _spawn_deck(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the turntable deck at `prim_path`: root Xform with RigidBodyAPI + explicit
    MassAPI, a disc collider (body origin = disc centre), an octagonal under-skirt that
    captures the deck laterally around the kinematic drum (the physical bearing), and three
    hexagonal pocket corrals on the top face, wall boxes colored per pocket."""
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
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.20)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    # main disc, centred on the body origin
    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateRadiusAttr(cfg.deck_r)
    disc.CreateHeightAttr(cfg.deck_t)
    disc.CreateExtentAttr([Gf.Vec3f(-cfg.deck_r, -cfg.deck_r, -cfg.deck_t / 2),
                           Gf.Vec3f(cfg.deck_r, cfg.deck_r, cfg.deck_t / 2)])
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.deck_color)])
    collide(disc.GetPrim())

    # under-skirt: 8 boxes, octagon of inradius `skirt_in` hanging below the disc — wraps
    # around the drum, capturing the deck laterally while leaving rotation free
    n = 8
    r_mid = cfg.skirt_in + cfg.skirt_t / 2
    seg_len = 2 * (cfg.skirt_in + cfg.skirt_t) * math.tan(math.pi / n) + 0.002
    zc = -cfg.deck_t / 2 - cfg.skirt_h / 2
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/skirt_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), zc))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.skirt_t, seg_len, cfg.skirt_h))
        seg.CreateDisplayColorAttr([Gf.Vec3f(*cfg.deck_color)])
        collide(seg.GetPrim())

    # 3 pocket corrals: hexagons of inradius `pocket_in` at radius `pocket_r`, walls
    # colored with the pocket's identity color
    m = 6
    wr_mid = cfg.pocket_in + cfg.pocket_wall_t / 2
    wseg_len = 2 * (cfg.pocket_in + cfg.pocket_wall_t) * math.tan(math.pi / m) + 0.002
    wz = cfg.deck_t / 2 + cfg.pocket_wall_h / 2
    for p, (alpha_deg, color) in enumerate(zip(cfg.pocket_alphas_deg, cfg.pocket_colors)):
        a = math.radians(alpha_deg)
        cx, cy = cfg.pocket_r * math.cos(a), cfg.pocket_r * math.sin(a)
        col = Gf.Vec3f(*color)
        for k in range(m):
            ang = 2 * math.pi * k / m
            seg = UsdGeom.Cube.Define(stage, f"{prim_path}/pocket{p}_wall{k}")
            seg.CreateSizeAttr(1.0)
            sxf = UsdGeom.Xformable(seg.GetPrim())
            sxf.AddTranslateOp().Set(Gf.Vec3d(cx + wr_mid * math.cos(ang),
                                              cy + wr_mid * math.sin(ang), wz))
            sxf.AddRotateZOp().Set(math.degrees(ang))
            sxf.AddScaleOp().Set(Gf.Vec3f(cfg.pocket_wall_t, wseg_len, cfg.pocket_wall_h))
            seg.CreateDisplayColorAttr([col])
            collide(seg.GetPrim())
    return root


def _spawn_shroud(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the fixed shroud at `prim_path` (KINEMATIC, re-posed per reset): a 240 deg
    roof annulus + a 240 deg outer wall ring, both built from 6 chord boxes; the open
    120 deg sector (shroud-local azimuth 0 +/- 60 deg) is the loading window."""
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
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    color = Gf.Vec3f(*cfg.color)
    # 6 boxes at 40 deg spacing covering shroud-local azimuth 60..300 deg
    azimuths = [80.0 + 40.0 * k for k in range(6)]
    for k, az in enumerate(azimuths):
        a = math.radians(az)
        roof = UsdGeom.Cube.Define(stage, f"{prim_path}/roof_{k}")
        roof.CreateSizeAttr(1.0)
        rxf = UsdGeom.Xformable(roof.GetPrim())
        rxf.AddTranslateOp().Set(Gf.Vec3d(cfg.roof_rmid * math.cos(a),
                                          cfg.roof_rmid * math.sin(a), cfg.roof_zc))
        rxf.AddRotateZOp().Set(az)
        rxf.AddScaleOp().Set(Gf.Vec3f(cfg.roof_radial, cfg.roof_tangential, cfg.roof_t))
        roof.CreateDisplayColorAttr([color])
        collide(roof.GetPrim())

        wall = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        wall.CreateSizeAttr(1.0)
        wxf = UsdGeom.Xformable(wall.GetPrim())
        wxf.AddTranslateOp().Set(Gf.Vec3d(cfg.wall_rmid * math.cos(a),
                                          cfg.wall_rmid * math.sin(a), cfg.wall_zc))
        wxf.AddRotateZOp().Set(az)
        wxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, cfg.wall_tangential, cfg.wall_h))
        wall.CreateDisplayColorAttr([color])
        collide(wall.GetPrim())
    return root


def _deck_spawner_cfg(c: "PlateCarouselSceneCfg") -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "deck" not in _SPAWNER_CACHE:

        @configclass
        class DeckSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_deck)
            deck_r: float = 0.165
            deck_t: float = 0.012
            skirt_in: float = 0.118
            skirt_t: float = 0.008
            skirt_h: float = 0.020
            pocket_r: float = 0.090
            pocket_in: float = 0.058
            pocket_wall_t: float = 0.006
            pocket_wall_h: float = 0.012
            pocket_alphas_deg: tuple = (90.0, 210.0, 330.0)
            pocket_colors: tuple = ()
            deck_color: tuple = (0.55, 0.55, 0.58)
            mass: float = 0.45
            contact_offset: float = 0.002

        _SPAWNER_CACHE["deck"] = DeckSpawnerCfg

    return _SPAWNER_CACHE["deck"](
        deck_r=c.deck_r, deck_t=c.deck_t, skirt_in=c.skirt_in, skirt_t=c.skirt_t,
        skirt_h=c.skirt_h, pocket_r=c.pocket_r, pocket_in=c.pocket_in,
        pocket_wall_t=c.pocket_wall_t, pocket_wall_h=c.pocket_wall_h,
        pocket_alphas_deg=c.pocket_alphas_deg, pocket_colors=c.plate_colors,
        deck_color=c.deck_color, mass=c.deck_mass, contact_offset=c.contact_offset,
    )


def _shroud_spawner_cfg(c: "PlateCarouselSceneCfg") -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shroud" not in _SPAWNER_CACHE:

        @configclass
        class ShroudSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shroud)
            roof_rmid: float = 0.120
            roof_radial: float = 0.180
            roof_tangential: float = 0.155
            roof_t: float = 0.012
            roof_zc: float = 0.203
            wall_rmid: float = 0.183
            wall_t: float = 0.010
            wall_tangential: float = 0.140
            wall_h: float = 0.085
            wall_zc: float = 0.1625
            color: tuple = (0.45, 0.40, 0.35)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["shroud"] = ShroudSpawnerCfg

    return _SPAWNER_CACHE["shroud"](
        roof_rmid=c.roof_rmid, roof_radial=c.roof_radial, roof_tangential=c.roof_tangential,
        roof_t=c.roof_t, roof_zc=c.roof_zc, wall_rmid=c.wall_rmid, wall_t=c.wall_t,
        wall_tangential=c.wall_tangential, wall_h=c.wall_h, wall_zc=c.wall_zc,
        color=c.shroud_color, contact_offset=c.contact_offset,
    )


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class PlateCarouselSceneCfg(BaseCfg):
    """Config for `PlateCarouselScene`. Env-local frame: carousel hub at `hub` on the
    ground; plates start flat on stands on the far side. Pocket i (red/green/blue) is the
    home of plate i; pockets sit at fixed deck-frame azimuths `pocket_alphas_deg`."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    pocket_xy_tol: float = tunable(0.015)  # plate centre within this of its pocket axis in
    # the DECK frame. Honest by construction: corral inradius - plate radius = 8 mm, so any
    # plate physically seated counts; the next pocket centre is 156 mm away.
    seat_z_tol: float = tunable(0.006)     # |deck-local z - seat height| under this. A plate
    # resting on a corral wall sits ~12 mm high; on the shroud roof ~45 mm high — both out.
    flat_max_deg: float = tunable(15.0)    # plate axis within this of the deck axis
    settle_lin: float = tunable(0.05)      # plate AND deck |lin vel| gate (m/s)
    settle_ang: float = tunable(0.80)      # plate |ang vel| gate (rad/s)
    deck_settle_ang: float = tunable(0.30)  # deck |ang vel| gate (rad/s) when judging

    # --- tunable: latch parameters -----------------------------------------------------------
    window_half_deg: float = tunable(50.0)  # pocket azimuth within this of the window centre
    # at the entry event (geometric opening is +/- 60 deg — the latch is slightly stricter)
    entry_dx_max: float = tunable(0.05)     # anti-teleport: per-substep plate displacement
    engage_min_rad: float = tunable(0.50)   # cumulative smooth rotation to latch `engaged`
    yaw_step_max: float = tunable(0.06)     # per-substep yaw above this does not accumulate

    # --- tunable: score weights --------------------------------------------------------------
    w_engage: float = tunable(0.10)
    w_plate: float = tunable(0.25)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    deck_yaw_range_deg: float = tunable(180.0)    # uniform +/- initial deck yaw
    window_yaw_range_deg: float = tunable(180.0)  # uniform +/- shroud (window) azimuth
    stand_jitter: float = tunable(0.03)           # uniform +/- xy jitter of each plate stand
    permute_stands: bool = tunable(True)          # shuffle which plate starts on which stand

    # --- info: structure ---------------------------------------------------------------------
    hub: tuple = info((0.30, 0.0))     # carousel axis (env-local xy)
    drum_r: float = info(0.110)
    drum_top: float = info(0.150)      # drum top face height (deck rests here)
    drum_friction: float = info(0.03)  # bearing surface: low friction, combine-mode "min"
    deck_r: float = info(0.165)
    deck_t: float = info(0.012)
    skirt_in: float = info(0.118)      # skirt octagon inradius: 8 mm play around the drum
    skirt_t: float = info(0.008)
    skirt_h: float = info(0.020)
    deck_mass: float = info(0.45)
    pocket_r: float = info(0.090)      # pocket centres' radius on the deck
    pocket_in: float = info(0.058)     # corral hexagon inradius (plate r 50 -> 8 mm funnel)
    pocket_wall_t: float = info(0.006)
    pocket_wall_h: float = info(0.012)
    pocket_alphas_deg: tuple = info((90.0, 210.0, 330.0))
    plate_r: float = info(0.050)
    plate_h: float = info(0.012)
    plate_mass: float = info(0.10)
    plate_colors: tuple = info(((0.85, 0.15, 0.15), (0.15, 0.62, 0.20), (0.20, 0.35, 0.85)))
    deck_color: tuple = info((0.55, 0.55, 0.58))
    shroud_color: tuple = info((0.45, 0.40, 0.35))
    # shroud: roof underside at deck_top + 35 mm (seated plates + corral walls pass under
    # with >= 23 mm clearance; a plate cannot be lowered through it), roof 12 mm thick
    # (a plate dropped onto it at ~7 mm/substep cannot tunnel — the pen_holder floor
    # lesson), outer wall inradius 178 mm (deck spins inside with 13 mm clearance),
    # window = 120 deg open sector
    roof_rmid: float = info(0.120)
    roof_radial: float = info(0.180)
    roof_tangential: float = info(0.155)
    roof_t: float = info(0.012)
    roof_zc: float = info(0.203)
    wall_rmid: float = info(0.183)
    wall_t: float = info(0.010)
    wall_tangential: float = info(0.140)
    wall_h: float = info(0.085)
    wall_zc: float = info(0.1625)
    stand_nominal: tuple = info(((-0.16, -0.20), (-0.24, 0.0), (-0.16, 0.20)))
    plate_friction: float = info(0.60)
    contact_offset: float = info(0.002)  # small everywhere: the tightest running clearance
    # (skirt-drum 8 mm, deck-wall 13 mm) must not be eaten by speculative contact

    # Derived (filled in __post_init__).
    n_plates: int = field(default=3, init=False)
    deck_rest_z: float = field(default=None, init=False)   # deck body origin height at rest
    deck_top_z: float = field(default=None, init=False)
    seat_z_local: float = field(default=None, init=False)  # plate centre in DECK frame

    def __post_init__(self) -> None:
        self.deck_rest_z = round(self.drum_top + self.deck_t / 2, 4)
        self.deck_top_z = round(self.drum_top + self.deck_t, 4)
        self.seat_z_local = round(self.deck_t / 2 + self.plate_h / 2, 4)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("plate_carousel")
class PlateCarouselScene(BaseScene):
    cfg: PlateCarouselSceneCfg

    def __init__(self, cfg: PlateCarouselSceneCfg | None = None) -> None:
        super().__init__(cfg or PlateCarouselSceneCfg())

    # ----- assets ---------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, the kinematic drum (low-friction bearing top), the dynamic deck
        resting on it, the kinematic shroud, three dynamic plates on their stands. reset()
        re-poses everything."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        bearing_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.drum_friction, dynamic_friction=c.drum_friction,
            restitution=0.0, friction_combine_mode="min")
        plate_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=c.plate_friction, dynamic_friction=c.plate_friction,
            restitution=0.0)

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
            "drum": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drum",
                spawn=sim_utils.CylinderCfg(
                    radius=c.drum_r, height=c.drum_top,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=coll, physics_material=bearing_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.28, 0.28, 0.32)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hub[0], c.hub[1], c.drum_top / 2)),
            ),
            "deck": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Deck",
                spawn=_deck_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hub[0], c.hub[1], c.deck_rest_z + 0.001)),
            ),
            "shroud": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shroud",
                spawn=_shroud_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hub[0], c.hub[1], 0.0)),
            ),
        }
        for i in range(c.n_plates):
            out[f"plate_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Plate_" + str(i),
                spawn=sim_utils.CylinderCfg(
                    radius=c.plate_r, height=c.plate_h,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.5,
                        linear_damping=0.05, angular_damping=0.20),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                    collision_props=coll, physics_material=plate_mat,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.plate_colors[i]),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.stand_nominal[i][0], c.stand_nominal[i][1],
                         c.plate_h / 2 + 0.002)),
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
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the per-episode buffers (latches, previous-substep
        trackers for the entry-event and smooth-rotation logic)."""
        super().bind(env)
        n, dev = env.num_envs, env.device
        c = self.cfg
        self.deck: RigidObject = env.iscene["deck"]
        self.shroud: RigidObject = env.iscene["shroud"]
        self.plates: list[RigidObject] = [env.iscene[f"plate_{i}"] for i in range(c.n_plates)]
        self.env_origins = env.iscene.env_origins
        self.alphas = torch.tensor(
            [math.radians(a) for a in c.pocket_alphas_deg], device=dev)
        # pocket centres in the deck body frame (constant)
        self.pocket_local = torch.stack(
            [c.pocket_r * torch.cos(self.alphas), c.pocket_r * torch.sin(self.alphas),
             torch.full((c.n_plates,), c.seat_z_local, device=dev)], dim=-1)  # (3, 3)
        self.window_az = torch.zeros(n, device=dev)     # shroud yaw = window azimuth
        self.stand_xy = torch.zeros(n, c.n_plates, 2, device=dev)
        self.loaded = torch.zeros(n, c.n_plates, dtype=torch.bool, device=dev)
        self.engaged = torch.zeros(n, dtype=torch.bool, device=dev)
        self.cum_rot = torch.zeros(n, device=dev)
        self._prev_yaw = torch.zeros(n, device=dev)
        self._prev_ppos = torch.zeros(n, c.n_plates, 3, device=dev)
        self._prev_seated = torch.zeros(n, c.n_plates, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample deck yaw + window azimuth, re-pose deck (full state write,
        rest height, zero velocity) and shroud (pose-only, kinematic), scatter the plates on
        permuted + jittered stands, clear every latch and tracker."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- deck: rest pose on the drum at a sampled yaw ---
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.deck_yaw_range_deg)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.hub[0]
        st[:, 1] = c.hub[1]
        st[:, 2] = c.deck_rest_z + 0.001
        st[:, 3] = torch.cos(yaw / 2)
        st[:, 6] = torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.deck.write_root_state_to_sim(st, env_ids)

        # --- shroud: kinematic pose-only write at the sampled window azimuth ---
        waz = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.window_yaw_range_deg)
        self.window_az[env_ids] = waz
        ps = torch.zeros(m, 7, device=dev)
        ps[:, 0] = c.hub[0]
        ps[:, 1] = c.hub[1]
        ps[:, 3] = torch.cos(waz / 2)
        ps[:, 6] = torch.sin(waz / 2)
        ps[:, 0:3] += origin
        self.shroud.write_root_pose_to_sim(ps, env_ids)

        # --- plates: flat on permuted + jittered stands ---
        nominal = torch.tensor(c.stand_nominal, device=dev)  # (3, 2)
        if c.permute_stands:
            perm = torch.rand(m, c.n_plates, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(c.n_plates, device=dev).expand(m, -1)
        for i in range(c.n_plates):
            slot = nominal[perm[:, i]]
            slot = slot + (torch.rand(m, 2, device=dev) * 2 - 1) * c.stand_jitter
            self.stand_xy[env_ids, i] = slot
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = slot
            st[:, 2] = c.plate_h / 2 + 0.002
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            self.plates[i].write_root_state_to_sim(st, env_ids)

        self.loaded[env_ids] = False
        self.engaged[env_ids] = False
        self.cum_rot[env_ids] = 0.0
        self._prev_yaw[env_ids] = yaw
        self._prev_seated[env_ids] = False
        # fresh world positions for the dx guard, from the states just WRITTEN (data
        # buffers may be stale right after a write — avoid a phantom entry on substep 1)
        for i in range(c.n_plates):
            slot = self.stand_xy[env_ids, i]
            pp = torch.zeros(m, 3, device=dev)
            pp[:, 0:2] = slot
            pp[:, 2] = c.plate_h / 2 + 0.002
            self._prev_ppos[env_ids, i] = pp + origin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch bookkeeping every physics substep (buffers fresh):
        - `cum_rot`/`engaged`: accumulate |d yaw| only while the per-substep step is small
          (smooth rotation — a yaw teleport does not accumulate);
        - `loaded[i]`: ENTRY EVENT — seated_geo flips false -> true with a small per-substep
          displacement while pocket i is under the window. Sticky for the episode."""
        c = self.cfg
        yaw = self._deck_yaw()
        dyaw = self._wrap(yaw - self._prev_yaw)
        smooth = dyaw.abs() < c.yaw_step_max
        self.cum_rot += torch.where(smooth, dyaw.abs(), torch.zeros_like(dyaw))
        self.engaged |= self.cum_rot >= c.engage_min_rad
        self._prev_yaw = yaw

        seated = self.seated_geo()                       # (n, 3)
        ppos = self._plate_pos_w()                       # (n, 3, 3)
        dx = (ppos - self._prev_ppos).norm(dim=-1)       # (n, 3)
        entry = seated & ~self._prev_seated & (dx < c.entry_dx_max) & self.window_ok()
        self.loaded |= entry
        self._prev_seated = seated
        self._prev_ppos = ppos

    # ----- state (full, restorable) ----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "deck": self.deck.data.root_state_w[env_ids].clone(),
            "shroud": self.shroud.data.root_state_w[env_ids].clone(),
            "plates": [b.data.root_state_w[env_ids].clone() for b in self.plates],
            "window_az": self.window_az[env_ids].clone(),
            "stand_xy": self.stand_xy[env_ids].clone(),
            "loaded": self.loaded[env_ids].clone(),
            "engaged": self.engaged[env_ids].clone(),
            "cum_rot": self.cum_rot[env_ids].clone(),
            "prev_yaw": self._prev_yaw[env_ids].clone(),
            "prev_ppos": self._prev_ppos[env_ids].clone(),
            "prev_seated": self._prev_seated[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.deck.write_root_state_to_sim(state["deck"], env_ids)
        self.shroud.write_root_pose_to_sim(state["shroud"][:, 0:7], env_ids)
        for b, st in zip(self.plates, state["plates"]):
            b.write_root_state_to_sim(st, env_ids)
        self.window_az[env_ids] = state["window_az"]
        self.stand_xy[env_ids] = state["stand_xy"]
        self.loaded[env_ids] = state["loaded"]
        self.engaged[env_ids] = state["engaged"]
        self.cum_rot[env_ids] = state["cum_rot"]
        self._prev_yaw[env_ids] = state["prev_yaw"]
        self._prev_ppos[env_ids] = state["prev_ppos"]
        self._prev_seated[env_ids] = state["prev_seated"]

    # ----- description -----------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A plate carousel stands on a drum pedestal: a free-spinning turntable "
            f"({2 * c.deck_r * 100:.0f} cm across) carrying three shallow walled pockets, "
            f"color-coded red, green and blue, one sized for a single "
            f"{2 * c.plate_r * 100:.0f} cm dinner plate. A fixed shroud (roof and outer "
            f"wall) encloses the carousel except for one open loading window about a third "
            f"of the circumference wide; only the pocket rotated under the window can "
            f"receive a plate — everywhere else the roof blocks it. Three matching colored "
            f"plates lie flat on stands on the floor nearby.\n"
            f"Goal: every plate seated flat inside the pocket of its own color, at rest. "
            f"The deck spins freely on its bearing (and carries loaded plates with it under "
            f"the roof), so the plan is a loop: spin the deck until a target color pocket "
            f"is at the window, lower that plate in through the window, and repeat for the "
            f"other two colors. A plate balanced on a pocket wall, left on the shroud roof, "
            f"or slipped into a pocket of the wrong color does not count, and a plate that "
            f"never entered through the window does not count either."
        )

    # ----- kinematic helpers -----------------------------------------------------------------
    @staticmethod
    def _wrap(a: torch.Tensor) -> torch.Tensor:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _deck_yaw(self) -> torch.Tensor:
        """(N,) deck yaw from its root quaternion (deck stays level on the drum)."""
        q = self.deck.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _plate_pos_w(self) -> torch.Tensor:
        """(N, 3plates, 3) plate centres, world frame."""
        return torch.stack([b.data.root_pos_w for b in self.plates], dim=1)

    def _plate_pos_deck(self) -> torch.Tensor:
        """(N, 3plates, 3) plate centres in the DECK body frame."""
        from isaaclab.utils.math import quat_apply_inverse

        pos = self._plate_pos_w()
        n, p = pos.shape[0], pos.shape[1]
        dq = self.deck.data.root_quat_w[:, None, :].expand(n, p, 4).reshape(n * p, 4)
        dp = self.deck.data.root_pos_w[:, None, :]
        return quat_apply_inverse(dq, (pos - dp).reshape(n * p, 3)).reshape(n, p, 3)

    def pocket_world_xy(self) -> torch.Tensor:
        """(N, 3pockets, 2) pocket centres, env-local xy (for oracle staging)."""
        from isaaclab.utils.math import quat_apply

        n, p = self.env.num_envs, self.cfg.n_plates
        dq = self.deck.data.root_quat_w[:, None, :].expand(n, p, 4).reshape(n * p, 4)
        loc = self.pocket_local[None].expand(n, p, 3).reshape(n * p, 3)
        w = quat_apply(dq, loc).reshape(n, p, 3)
        return (self.deck.data.root_pos_w[:, None, :2] - self.env_origins[:, None, :2]
                + w[:, :, :2])

    def pocket_azimuth(self) -> torch.Tensor:
        """(N, 3pockets) world azimuth of each pocket about the hub."""
        return self._wrap(self._deck_yaw().unsqueeze(1) + self.alphas.unsqueeze(0))

    def window_ok(self) -> torch.Tensor:
        """(N, 3pockets) bool: pocket within `window_half_deg` of the window centre."""
        d = self._wrap(self.pocket_azimuth() - self.window_az.unsqueeze(1))
        return d.abs() < math.radians(self.cfg.window_half_deg)

    # ----- progress / rubric ------------------------------------------------------------------
    def seated_geo(self) -> torch.Tensor:
        """(N, 3plates) bool, geometric, judged in the DECK frame (rides with the pocket):
        plate centre within `pocket_xy_tol` of ITS pocket axis, at seat height, flat."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        loc = self._plate_pos_deck()                              # (n, 3, 3)
        near = (loc[:, :, :2] - self.pocket_local[None, :, :2]).norm(dim=-1) <= c.pocket_xy_tol
        z_ok = (loc[:, :, 2] - c.seat_z_local).abs() <= c.seat_z_tol
        # flatness: plate axis vs deck axis (both in world)
        n, p = loc.shape[0], loc.shape[1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=loc.device)
        dup = quat_apply(self.deck.data.root_quat_w, ez.expand(n, 3))       # (n, 3)
        cos_max = math.cos(math.radians(c.flat_max_deg))
        flats = []
        for b in self.plates:
            pax = quat_apply(b.data.root_quat_w, ez.expand(n, 3))
            flats.append((pax * dup).sum(dim=-1).abs() >= cos_max)
        return near & z_ok & torch.stack(flats, dim=1)

    def settled(self) -> torch.Tensor:
        """(N, 3plates) bool: plate lin/ang gates AND the deck itself at rest."""
        c = self.cfg
        deck_still = ((self.deck.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                      & (self.deck.data.root_ang_vel_w.norm(dim=-1) < c.deck_settle_ang))
        cols = []
        for b in self.plates:
            lin = b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
            ang = b.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang
            cols.append(lin & ang)
        return torch.stack(cols, dim=1) & deck_still.unsqueeze(-1)

    def counted(self) -> torch.Tensor:
        """(N, 3plates) bool: seated in the right pocket, settled, AND loaded through the
        window (the latch)."""
        return self.seated_geo() & self.settled() & self.loaded

    def success(self) -> torch.Tensor:
        """(N,) bool: all three plates counted — a current-state physical predicate gated
        by the through-the-window entry latches."""
        return self.counted().all(dim=1)

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: w_engage for the latched mechanism engagement +
        w_plate per plate currently counted; exactly 1.0 iff success. Ladder:
        0 -> 0.10 (first indexing rotation) -> 0.35 -> 0.60 -> 1.0."""
        c = self.cfg
        k = self.counted().sum(dim=1).float()
        base = c.w_engage * self.engaged.float() + c.w_plate * k
        return torch.where(self.success(), torch.ones_like(base), base)


# ----- env registration ------------------------------------------------------------------------
register_env("simgen", lambda: EnvCfg(scene="plate_carousel", robot="null", env_spacing=4.0))
