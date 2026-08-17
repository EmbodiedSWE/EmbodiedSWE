"""DrawerBeadPourScene — pull the drawer open, pour the bowl's beads in, park the bowl.

Derived from libero_90/kitchen_scene4 "put the black bowl in the bottom drawer of the
cabinet" (pick the bowl off the table, place it inside the drawer, judged by a
bowl-in-drawer bbox test) — but the BOWL'S ROLE IS INVERTED: it is no longer the cargo,
it is the PITCHER. The black bowl starts on the floor holding 2-4 loose yellow beads;
the goal is that every bead ends up loose ON THE BOTTOM DRAWER'S FLOOR, out of the
bowl, with the emptied bowl set back down upright on the open floor. Putting the bowl
itself into the drawer — the seed's entire terminal relation — is a FAILURE state here
(the bowl must end OUTSIDE the drawer, and a bead still sitting in the bowl counts as
not delivered even if the bowl is in the drawer). A solver therefore needs a
different plan: (1) pull the bottom drawer out by its blue handle bar (it starts
CLOSED; the 8 mm slot under the cabinet top is far narrower than a 24 mm bead, so no
content transfer is possible until the drawer is out), (2) transfer the beads — pour
them over the exposed drawer mouth, or place them one by one — until all of them rest
loose in the drawer, (3) park the empty bowl upright on the floor away from the
cabinet. The cabinet also carries an always-open TOP SHELF compartment: the tempting
zero-effort receptacle, and the wrong one — beads left there score nothing.

Assets are fully procedural (compound spawners; child colliders of one body never
self-collide):
  - shell: KINEMATIC compound — plinth, bay floor, bay side/back walls, cabinet top
    panel (front edge overhangs the drawer face; the closed drawer's rim clears it by
    8 mm), and four shelf walls forming the open-top decoy compartment above.
  - drawer: DYNAMIC compound — floor, four walls (interior 244 x 184 x 92 mm) and a
    protruding blue HANDLE BAR on the front face. Origin at the front-bottom-centre.
  - drawer slide: bind-time UsdPhysics.PrismaticJoint shell->drawer along +X, limits
    [-travel, 0] (0 = closed, -travel = fully out), joint-pair collision disabled.
    No spring: the drawer stays where it is left (heavy linear damping = slide
    friction), so "open" is a persistent state the solver must create.
  - bowl: DYNAMIC compound — octagonal cup (8 wall boxes + floor disc, inner inradius
    50 mm, wall 50 mm tall), matte black. The seed's black bowl, now a tool.
  - beads: 4 dynamic spheres (r 12 mm, yellow); a per-episode random subset of 2-4 is
    present (absent beads park in a ground depot far outside the workspace).

Per-episode randomization (readback-verifiable): present bead count 2-4, Bernoulli
LEFT/RIGHT bowl slot swap + xy jitter + random bead ring phase inside the bowl.

Rubric (0..1; progress latched so correct behavior never loses credit):
  0.15 * opened     — drawer ever pulled past open_min (latched bool)
  0.15 * carry      — bowl carried toward the open drawer's mouth WHILE holding at
                      least one bead, gated on opened (latched running max)
  0.45 * deposited  — fraction of present beads ever resting in the drawer AND out of
                      the bowl (latched running max of the count)
  0.10 * parked     — bowl upright on the floor outside the drawer while ALL present
                      beads are currently deposited (latched bool)
  1.0 iff success() — all present beads loose in the drawer (not in the bowl), beads
                      and bowl at rest, bowl parked upright on the floor outside the
                      drawer. Non-success cap 0.85.

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


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             rot_z_deg: float = 0.0) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if rot_z_deg:
        xf.AddRotateZOp().Set(float(rot_z_deg))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


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


def _spawn_shell(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet: KINEMATIC compound. Origin at the bay's front-bottom-centre
    (the closed drawer's front face plane); the drawer pulls out toward -x. Parts:
    plinth, bay floor, bay side/back walls, top panel (overhanging the front by
    `panel_overhang` so the closed drawer's mouth is capped), and the open-top decoy
    SHELF compartment on top."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    d, w = c.bay_depth, c.bay_w  # bay footprint: x in [0, d], |y| <= w/2 (interior)
    xc = d / 2
    t = c.t
    # plinth + bay floor (drawer rides the joint just above the floor top)
    _add_box(stage, f"{prim_path}/plinth", center=(xc, 0.0, c.plinth_h / 2),
             size=(d, w + 2 * t, c.plinth_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/bay_floor", center=(xc, 0.0, c.plinth_h + t / 2),
             size=(d, w + 2 * t, t), color=c.color, collide=collide)
    z0 = c.plinth_h + t  # bay floor top
    bay_h = c.bay_h
    for sgn, nm in ((1.0, "bay_wall_l"), (-1.0, "bay_wall_r")):
        _add_box(stage, f"{prim_path}/{nm}", center=(xc, sgn * (w / 2 + t / 2), z0 + bay_h / 2),
                 size=(d, t, bay_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/bay_back", center=(d - t / 2, 0.0, z0 + bay_h / 2),
             size=(t, w, bay_h), color=c.color, collide=collide)
    # top panel: overhangs the front so the shut drawer's mouth is capped
    px0, px1 = -c.panel_overhang, d
    _add_box(stage, f"{prim_path}/top_panel",
             center=((px0 + px1) / 2, 0.0, z0 + bay_h + t / 2),
             size=(px1 - px0, w + 2 * t + 0.004, t), color=c.top_color, collide=collide)
    # decoy shelf: open-top compartment on the panel (4 walls, no roof)
    sz0 = z0 + bay_h + t  # shelf floor = panel top
    sh = c.shelf_h
    _add_box(stage, f"{prim_path}/shelf_front", center=(px0 + t / 2, 0.0, sz0 + sh / 2),
             size=(t, w + 2 * t + 0.004, sh), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/shelf_back", center=(d - t / 2, 0.0, sz0 + sh / 2),
             size=(t, w + 2 * t + 0.004, sh), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "shelf_wall_l"), (-1.0, "shelf_wall_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=((px0 + px1) / 2, sgn * (w / 2 + t / 2), sz0 + sh / 2),
                 size=(px1 - px0, t, sh), color=c.color, collide=collide)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer: DYNAMIC open-top box + blue handle bar on the front (-x)
    face. Origin at the front-bottom-centre. Sleep/stabilization zeroed (force-driven
    on its slide)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(float(cfg.slide_damping))
    pxrb.CreateAngularDampingAttr(1.0)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg)
    c = cfg
    t, D, W, H = c.t, c.depth, c.width, c.height  # outer depth/width, wall top height
    _add_box(stage, f"{prim_path}/floor", center=(D / 2, 0.0, t / 2),
             size=(D, W, t), color=c.color, collide=collide)
    wall_h = H - t
    _add_box(stage, f"{prim_path}/wall_front", center=(t / 2, 0.0, t + wall_h / 2),
             size=(t, W, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_back", center=(D - t / 2, 0.0, t + wall_h / 2),
             size=(t, W, wall_h), color=c.color, collide=collide)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        _add_box(stage, f"{prim_path}/{nm}", center=(D / 2, sgn * (W / 2 - t / 2), t + wall_h / 2),
                 size=(D, t, wall_h), color=c.color, collide=collide)
    _add_box(stage, f"{prim_path}/handle",
             center=(-c.handle_out / 2, 0.0, c.handle_z),
             size=(c.handle_out, c.handle_len, c.handle_t),
             color=c.handle_color, collide=collide)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the black bowl: DYNAMIC octagonal cup — 8 wall boxes around a floor
    disc. Origin at the geometric centre (rest centre height = total_h / 2)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.1)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg)
    c = cfg
    color = Gf.Vec3f(*c.color)
    half = c.total_h / 2
    # floor disc
    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/floor")
    r_f = c.inner_r + c.t
    disc.CreateRadiusAttr(r_f)
    disc.CreateHeightAttr(c.floor_t)
    disc.CreateExtentAttr([Gf.Vec3f(-r_f, -r_f, -c.floor_t / 2),
                           Gf.Vec3f(r_f, r_f, c.floor_t / 2)])
    UsdGeom.Xformable(disc.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -half + c.floor_t / 2))
    disc.CreateDisplayColorAttr([color])
    collide(disc.GetPrim())
    # 8 wall boxes, thickness radial, inner face at inner_r
    wall_h = c.total_h - c.floor_t
    wall_w = 2 * (c.inner_r + c.t) * math.tan(math.pi / 8) + 0.004  # overlap corners
    for k in range(8):
        phi = k * 45.0
        rad = math.radians(phi)
        rc = c.inner_r + c.t / 2
        _add_box(stage, f"{prim_path}/wall_{k}",
                 center=(rc * math.cos(rad), rc * math.sin(rad), -half + c.floor_t + wall_h / 2),
                 size=(c.t, wall_w, wall_h), color=c.color, collide=collide,
                 rot_z_deg=phi)
    return root


def _spawn_bead(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one bead: DYNAMIC sphere, heavy angular damping (it must settle on the
    drawer floor instead of orbiting it)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.3)
    pxrb.CreateAngularDampingAttr(1.5)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg)
    sph = UsdGeom.Sphere.Define(stage, f"{prim_path}/ball")
    sph.CreateRadiusAttr(float(cfg.radius))
    r = float(cfg.radius)
    sph.CreateExtentAttr([Gf.Vec3f(-r, -r, -r), Gf.Vec3f(r, r, r)])
    sph.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    collide(sph.GetPrim())
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "shell" not in _SPAWNER_CACHE:

        @configclass
        class ShellSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_shell)
            bay_depth: float = 0.28
            bay_w: float = 0.224
            bay_h: float = 0.11
            plinth_h: float = 0.038
            panel_overhang: float = 0.01
            shelf_h: float = 0.07
            t: float = 0.01
            color: tuple = (0.45, 0.42, 0.40)
            top_color: tuple = (0.55, 0.52, 0.50)
            contact_offset: float = 0.002

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            depth: float = 0.26
            width: float = 0.20
            height: float = 0.10
            t: float = 0.008
            handle_out: float = 0.030
            handle_len: float = 0.090
            handle_t: float = 0.014
            handle_z: float = 0.085
            handle_color: tuple = (0.10, 0.25, 0.80)
            color: tuple = (0.80, 0.70, 0.50)
            slide_damping: float = 6.0
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.050
            t: float = 0.008
            floor_t: float = 0.006
            total_h: float = 0.056
            color: tuple = (0.05, 0.05, 0.05)
            contact_offset: float = 0.002

        @configclass
        class BeadSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bead)
            radius: float = 0.012
            color: tuple = (0.95, 0.80, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(shell=ShellSpawnerCfg, drawer=DrawerSpawnerCfg,
                              bowl=BowlSpawnerCfg, bead=BeadSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DrawerBeadPourSceneCfg(BaseCfg):
    """Config for `DrawerBeadPourScene`. The interlock is metric: the shut drawer's rim
    clears the cabinet top panel by 8 mm, far below the 24 mm bead diameter, and the
    front face is solid — no bead can enter the drawer until it is pulled out. The
    slide has no spring (heavy damping = friction), so the open state persists."""

    # --- tunable: rubric thresholds -------------------------------------------------------------
    open_min: float = tunable(0.10)  # drawer travel (m) past which "opened" latches
    settle_speed: float = tunable(0.05)  # max |lin vel| of beads / bowl when judging (m/s)
    park_tilt_max_deg: float = tunable(20.0)  # bowl "upright" cone for parking
    park_z_tol: float = tunable(0.012)  # bowl centre height tolerance about rest (m)
    carry_d0: float = tunable(0.40)  # carry ramp: p = 1 - d/carry_d0

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    slot_jitter: float = tunable(0.03)  # bowl spawn xy jitter (+/- m)
    swap_slots: bool = tunable(True)  # Bernoulli bowl slot side swap (demo sets False)
    min_beads: int = tunable(2)  # present bead count sampled in [min_beads, 4]

    # --- info: layout (single Franka base at the world origin; radii 0.34-0.63 m) ---------------
    front_x: float = info(0.50)  # closed drawer front face plane (cabinet front)
    travel: float = info(0.16)  # slide travel (m); open = drawer at front_x - travel
    slot_a: tuple = info((0.28, 0.30))  # bowl spawn slot A (left)
    slot_b: tuple = info((0.28, -0.30))  # bowl spawn slot B (right)
    depot: tuple = info((0.95, 0.95))  # ground depot for absent beads

    # --- info: cabinet structure ----------------------------------------------------------------
    bay_depth: float = info(0.28)
    bay_w: float = info(0.224)
    bay_h: float = info(0.11)  # bay wall height above the bay floor
    plinth_h: float = info(0.038)
    shell_t: float = info(0.01)
    panel_overhang: float = info(0.01)
    shelf_h: float = info(0.07)  # decoy shelf compartment wall height

    # --- info: drawer structure -----------------------------------------------------------------
    drawer_depth: float = info(0.26)  # outer (x)
    drawer_w: float = info(0.20)  # outer (y)
    drawer_h: float = info(0.10)  # wall top above drawer origin
    drawer_t: float = info(0.008)
    drawer_mass: float = info(1.2)
    drawer_z: float = info(0.050)  # drawer origin height (rides the joint here)
    slide_damping: float = info(6.0)
    handle_out: float = info(0.030)
    handle_len: float = info(0.090)
    handle_t: float = info(0.014)
    handle_z: float = info(0.085)  # bar height above the drawer origin

    # --- info: bowl / beads ---------------------------------------------------------------------
    bowl_inner_r: float = info(0.050)
    bowl_t: float = info(0.008)
    bowl_floor_t: float = info(0.006)
    bowl_h: float = info(0.056)  # total height; rest centre z = bowl_h/2
    bowl_mass: float = info(0.15)
    bead_r: float = info(0.012)
    bead_mass: float = info(0.02)
    n_beads: int = info(4)  # bodies in the scene; a present subset is judged

    contact_offset: float = info(0.002)
    # rubric weights (0.15 + 0.15 + 0.45 + 0.10 = 0.85 = the non-success cap)
    w_open: float = info(0.15)
    w_carry: float = info(0.15)
    w_dep: float = info(0.45)
    w_park: float = info(0.10)

    # Derived (filled in __post_init__).
    drawer_box_lo: tuple = field(default=None, init=False)  # drawer-local deposit box
    drawer_box_hi: tuple = field(default=None, init=False)
    mouth_anchor_local: tuple = field(default=None, init=False)  # drawer-local carry anchor

    def __post_init__(self) -> None:
        t, D, W, H = self.drawer_t, self.drawer_depth, self.drawer_w, self.drawer_h
        m = 0.004
        self.drawer_box_lo = (t - m, -(W / 2 - t) - m, t - m)
        self.drawer_box_hi = (D - t + m, (W / 2 - t) + m, H + m)
        # exposed-mouth centre when fully open: drawer-local x mid of [t, travel - overhang]
        self.mouth_anchor_local = ((t + self.travel - self.panel_overhang) / 2, 0.0, H + 0.05)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("drawer_bead_pour")
class DrawerBeadPourScene(BaseScene):
    cfg: DrawerBeadPourSceneCfg

    def __init__(self, cfg: DrawerBeadPourSceneCfg | None = None) -> None:
        super().__init__(cfg or DrawerBeadPourSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        spawners = _spawner_classes()
        shell_spawn = spawners["shell"](
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            bay_depth=c.bay_depth, bay_w=c.bay_w, bay_h=c.bay_h, plinth_h=c.plinth_h,
            panel_overhang=c.panel_overhang, shelf_h=c.shelf_h, t=c.shell_t,
            contact_offset=c.contact_offset,
        )
        drawer_spawn = spawners["drawer"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.drawer_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            depth=c.drawer_depth, width=c.drawer_w, height=c.drawer_h, t=c.drawer_t,
            handle_out=c.handle_out, handle_len=c.handle_len, handle_t=c.handle_t,
            handle_z=c.handle_z, slide_damping=c.slide_damping,
            contact_offset=c.contact_offset,
        )
        bowl_spawn = spawners["bowl"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            inner_r=c.bowl_inner_r, t=c.bowl_t, floor_t=c.bowl_floor_t, total_h=c.bowl_h,
            contact_offset=c.contact_offset,
        )
        bead_spawn = spawners["bead"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bead_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            radius=c.bead_r, contact_offset=c.contact_offset,
        )
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
            "shell": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Shell",
                spawn=shell_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.front_x, 0.0, 0.0)),
            ),
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer",
                spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.front_x, 0.0, c.drawer_z)),
            ),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl",
                spawn=bowl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.slot_a[0], c.slot_a[1], c.bowl_h / 2 + 0.002)),
            ),
        }
        for i in range(c.n_beads):
            out[f"bead_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bead_" + str(i),
                spawn=bead_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot[0] + 0.08 * (i % 2), c.depot[1] + 0.08 * (i // 2),
                         c.bead_r + 0.002)),
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

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.shell: RigidObject = env.iscene["shell"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.beads: list[RigidObject] = [env.iscene[f"bead_{i}"] for i in range(c.n_beads)]
        self.env_origins = env.iscene.env_origins
        self._author_slide()
        n = env.num_envs
        dev = env.device
        # present[e, i]: bead i participates in episode e (sampled at reset)
        self.present = torch.ones(n, c.n_beads, dtype=torch.bool, device=dev)
        # latches: partial progress survives transient achievements
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._carry_max = torch.zeros(n, device=dev)
        self._dep_max = torch.zeros(n, device=dev)  # max fraction of present beads deposited
        self._park = torch.zeros(n, dtype=torch.bool, device=dev)
        # External drive input (solve.py writes; post_step consumes and OWNS the
        # drawer's external-wrench slot).
        self.drawer_drive = torch.zeros(n, device=dev)  # force along +x (N); pull = negative

    def _author_slide(self) -> None:
        """Per env: a +X prismatic joint shell->drawer, limits [-travel, 0]
        (0 = closed), joint-pair collision disabled (the joint owns the drawer-shell
        relation; the shell still blocks the beads and the bowl everywhere)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Shell"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.drawer_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.travel))
            j.CreateUpperLimitAttr(0.0)

    # ----- reset ---------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: drawer shut, bowl at a Bernoulli-swapped slot with xy jitter,
        a sampled subset of 2-4 beads ringed inside the bowl (random phase), absent
        beads parked in the ground depot, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- shell (kinematic, fixed) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = c.front_x
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.shell.write_root_state_to_sim(st, env_ids)

        # --- drawer: shut (joint coordinate 0) ---
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 2] = c.front_x, c.drawer_z
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.drawer.write_root_state_to_sim(st, env_ids)

        # --- bowl: Bernoulli slot swap + xy jitter, upright ---
        # (torch.rand-based draws: the first randint after manual_seed is degenerate)
        if c.swap_slots:
            swap = torch.rand(m, device=dev) < 0.5
        else:
            swap = torch.zeros(m, dtype=torch.bool, device=dev)
        slot_a = torch.tensor(c.slot_a, device=dev).expand(m, 2)
        slot_b = torch.tensor(c.slot_b, device=dev).expand(m, 2)
        bowl_xy = torch.where(swap.unsqueeze(1), slot_b, slot_a) \
            + (torch.rand(m, 2, device=dev) * 2 - 1) * c.slot_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:2] = bowl_xy
        st[:, 2] = c.bowl_h / 2 + 0.002
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.bowl.write_root_state_to_sim(st, env_ids)

        # --- beads: sample present count k in [min_beads, n], ring the present ones
        # inside the bowl (random phase), park the rest in the depot ---
        span = c.n_beads - c.min_beads + 1
        k = c.min_beads + (torch.rand(m, device=dev) * span).long().clamp(max=span - 1)
        rank = torch.rand(m, c.n_beads, device=dev).argsort(dim=1).argsort(dim=1)
        pres = rank < k.unsqueeze(1)
        self.present[env_ids] = pres
        phase = torch.rand(m, device=dev) * 2 * math.pi
        ring_r = 0.020
        floor_top = c.bowl_h / 2 + 0.002 - c.bowl_h / 2 + c.bowl_floor_t  # ~bowl floor top z
        for i, bead in enumerate(self.beads):
            ang = phase + i * (2 * math.pi / c.n_beads)
            in_bowl = torch.stack([
                bowl_xy[:, 0] + ring_r * torch.cos(ang),
                bowl_xy[:, 1] + ring_r * torch.sin(ang),
                torch.full((m,), floor_top + c.bead_r + 0.003 + 0.001 * i, device=dev),
            ], dim=1)
            depot = torch.tensor(
                [c.depot[0] + 0.08 * (i % 2), c.depot[1] + 0.08 * (i // 2),
                 c.bead_r + 0.002], device=dev).expand(m, 3)
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:3] = torch.where(pres[:, i].unsqueeze(1), in_bowl, depot) + origin
            st[:, 3] = 1.0
            bead.write_root_state_to_sim(st, env_ids)

        # --- clear latches + drive ---
        self._opened[env_ids] = False
        self._carry_max[env_ids] = 0.0
        self._dep_max[env_ids] = 0.0
        self._park[env_ids] = False
        self.drawer_drive[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "shell": self.shell.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "beads": [b.data.root_state_w[env_ids].clone() for b in self.beads],
            "present": self.present[env_ids].clone(),
            "opened": self._opened[env_ids].clone(),
            "carry_max": self._carry_max[env_ids].clone(),
            "dep_max": self._dep_max[env_ids].clone(),
            "park": self._park[env_ids].clone(),
            "drawer_drive": self.drawer_drive[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.shell.write_root_state_to_sim(state["shell"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        for b, s in zip(self.beads, state["beads"]):
            b.write_root_state_to_sim(s, env_ids)
        self.present[env_ids] = state["present"]
        self._opened[env_ids] = state["opened"]
        self._carry_max[env_ids] = state["carry_max"]
        self._dep_max[env_ids] = state["dep_max"]
        self._park[env_ids] = state["park"]
        self.drawer_drive[env_ids] = state["drawer_drive"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        gap = (c.plinth_h + c.shell_t + c.bay_h) - (c.drawer_z + c.drawer_h)
        return (
            f"A gray floor cabinet stands on the floor. Its BOTTOM level is a sliding "
            f"DRAWER (tan box, interior ~{(c.drawer_depth - 2 * c.drawer_t) * 100:.0f} x "
            f"{(c.drawer_w - 2 * c.drawer_t) * 100:.0f} x {(c.drawer_h - c.drawer_t) * 100:.0f} cm) "
            f"with a BLUE HANDLE BAR on its front face; it starts fully SHUT and slides "
            f"straight out toward you (travel {c.travel * 100:.0f} cm), staying wherever "
            f"it is left. While shut, the cabinet top caps the drawer with only a "
            f"{gap * 1000:.0f} mm slit — nothing can be put inside a shut drawer. Above "
            f"the drawer, the cabinet TOP carries an always-open SHELF compartment; it "
            f"is NOT the goal — anything left there counts for nothing. On the floor to "
            f"one side of the cabinet (the side varies between episodes) stands a BLACK "
            f"BOWL (~{2 * (c.bowl_inner_r + c.bowl_t) * 100:.0f} cm across, "
            f"{c.bowl_h * 100:.1f} cm tall) holding several loose YELLOW BEADS "
            f"({2 * c.bead_r * 1000:.0f} mm balls; between {c.min_beads} and {c.n_beads} "
            f"are present — count what you see).\n"
            f"Goal: EVERY yellow bead must end up resting loose on the BOTTOM DRAWER'S "
            f"floor — out of the bowl — and the emptied black bowl must be set back "
            f"down UPRIGHT on the open floor, clear of the drawer. Required order: the "
            f"drawer must be pulled open first (beads cannot enter it shut); then "
            f"transfer the beads — carry the bowl over the open drawer and pour them "
            f"in, or move them however you can; then park the bowl. Failure states: "
            f"the bowl (or any bead) left in the shelf compartment, beads spilled on "
            f"the floor, any bead still in the bowl, or the bowl itself ending up "
            f"inside the drawer — the bowl is the container you pour FROM, it does not "
            f"go in."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Pull the cabinet's bottom drawer open by its blue handle, pour all the "
            "yellow beads from the black bowl onto the drawer floor, then set the "
            "empty bowl down upright on the floor beside the cabinet. Do not put the "
            "bowl in the drawer, and do not leave beads in the bowl, on the floor, or "
            "in the top shelf."
        )

    # ----- readings / rubric -----------------------------------------------------------------------
    def drawer_open(self) -> torch.Tensor:
        """(N,) drawer travel in m (0 = shut, travel = fully out)."""
        x = (self.drawer.data.root_pos_w - self.env_origins)[:, 0]
        return (self.cfg.front_x - x).clamp(min=0.0)

    def _bead_pos(self) -> torch.Tensor:
        """(N, B, 3) world bead positions."""
        return torch.stack([b.data.root_pos_w for b in self.beads], dim=1)

    def _bead_speed(self) -> torch.Tensor:
        """(N, B) bead |lin vel|."""
        return torch.stack([b.data.root_lin_vel_w.norm(dim=-1) for b in self.beads], dim=1)

    def beads_in_bowl(self) -> torch.Tensor:
        """(N, B) bool: bead centre inside the bowl's interior cylinder (bowl body
        frame — a carried, tilted bowl still holds its beads)."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        pos = self._bead_pos()
        n, bnum = pos.shape[0], pos.shape[1]
        bq = self.bowl.data.root_quat_w[:, None, :].expand(n, bnum, 4).reshape(-1, 4)
        bp = self.bowl.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(bq, (pos - bp).reshape(-1, 3)).reshape(n, bnum, 3)
        r_ok = loc[:, :, :2].norm(dim=-1) < c.bowl_inner_r
        z_ok = (loc[:, :, 2] > -c.bowl_h / 2 + 0.001) & (loc[:, :, 2] < c.bowl_h / 2 + 0.035)
        return r_ok & z_ok

    def _in_drawer_box(self, pos_w: torch.Tensor) -> torch.Tensor:
        """(..., ) bool: world points inside the drawer's interior box (drawer frame;
        the drawer never rotates, so this is a translation)."""
        c = self.cfg
        loc = pos_w - self.drawer.data.root_pos_w.reshape(
            (-1,) + (1,) * (pos_w.dim() - 2) + (3,))
        lo = torch.tensor(c.drawer_box_lo, device=pos_w.device)
        hi = torch.tensor(c.drawer_box_hi, device=pos_w.device)
        return ((loc >= lo) & (loc <= hi)).all(dim=-1)

    def beads_deposited(self) -> torch.Tensor:
        """(N, B) bool: bead resting in the drawer's interior AND out of the bowl —
        delivered cargo. A bead sitting in the bowl inside the drawer does not count."""
        return self._in_drawer_box(self._bead_pos()) & ~self.beads_in_bowl()

    def bowl_in_drawer(self) -> torch.Tensor:
        """(N,) bool: bowl centre inside the drawer's (slightly expanded) interior."""
        c = self.cfg
        loc = self.bowl.data.root_pos_w - self.drawer.data.root_pos_w
        lo = torch.tensor((c.drawer_box_lo[0] - 0.03, c.drawer_box_lo[1] - 0.03, -0.02),
                          device=loc.device)
        # z-hi 0.15: catches the bowl resting inside (centre ~0.036) or perched on the
        # drawer rim (~0.128), but not a bowl hovering well above the open mouth.
        hi = torch.tensor((c.drawer_box_hi[0] + 0.03, c.drawer_box_hi[1] + 0.03, 0.15),
                          device=loc.device)
        return ((loc >= lo) & (loc <= hi)).all(dim=-1)

    def bowl_parked(self) -> torch.Tensor:
        """(N,) bool: bowl upright (within park_tilt cone), resting at floor height,
        OUTSIDE the drawer, and still. The 'put the tool back' clause."""
        from isaaclab.utils.math import quat_apply

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        up = quat_apply(self.bowl.data.root_quat_w, ez)
        upright = up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(c.park_tilt_max_deg))
        z = (self.bowl.data.root_pos_w - self.env_origins)[:, 2]
        on_floor = (z - (c.bowl_h / 2 + 0.002)).abs() < c.park_z_tol
        still = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return upright & on_floor & still & ~self.bowl_in_drawer()

    def all_delivered(self) -> torch.Tensor:
        """(N,) bool: every PRESENT bead is deposited (live)."""
        return (self.beads_deposited() | ~self.present).all(dim=1)

    def _update_latches(self) -> None:
        c = self.cfg
        self._opened |= self.drawer_open() >= c.open_min
        holding = (self.beads_in_bowl() & self.present).any(dim=1)
        anchor = self.drawer.data.root_pos_w + torch.tensor(
            c.mouth_anchor_local, device=self.env.device)
        d = (self.bowl.data.root_pos_w - anchor).norm(dim=-1)
        carry = (1.0 - d / c.carry_d0).clamp(0.0, 1.0) * self._opened.float() * holding.float()
        carry = torch.nan_to_num(carry, nan=0.0, posinf=0.0, neginf=0.0)
        self._carry_max = torch.maximum(self._carry_max, carry)
        k = self.present.sum(dim=1).clamp(min=1).float()
        frac = (self.beads_deposited() & self.present).sum(dim=1).float() / k
        frac = torch.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
        self._dep_max = torch.maximum(self._dep_max, frac)
        self._park |= self.bowl_parked() & self.all_delivered()

    # ----- step-coupled mechanics (every substep) --------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Slide plant: the external drive force along the drawer's axis (the
        applied-wrench emulation of pulling/pushing the blue handle); then latch rubric
        progress. Owns the drawer's external-wrench slot."""
        n = self.env.num_envs
        force = torch.zeros(n, 1, 3, device=self.env.device)
        force[:, 0, 0] = self.drawer_drive
        self.drawer.set_external_force_and_torque(
            force, torch.zeros(n, 1, 3, device=self.env.device))
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: all present beads resting loose in the drawer (out of the bowl),
        beads and bowl at rest, bowl parked upright on the floor outside the drawer.
        Physical outcomes only."""
        c = self.cfg
        self._update_latches()
        beads_still = ((self._bead_speed() < c.settle_speed) | ~self.present).all(dim=1)
        return self.all_delivered() & beads_still & self.bowl_parked()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15*opened + 0.15*carry (gated on opened + holding) +
        0.45*deposited-fraction + 0.10*parked-while-delivered — all latched, ~0 for
        doing nothing, capped 0.85 — and exactly 1.0 iff success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_open * self._opened.float() + c.w_carry * self._carry_max
                + c.w_dep * self._dep_max + c.w_park * self._park.float()).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="drawer_bead_pour", robot="null"))
