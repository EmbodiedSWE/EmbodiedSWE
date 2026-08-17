"""StowflatScene — the drawer can only close after its own CARGO is LAID FLAT
(sim_gen task `libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet_i418`).

Derived from libero_90/libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet,
but STRATEGICALLY different: in the seed the drawer is free to close and the whole
skill is one guided push on the judged part. Here the seed's skill is executed
against a drawer that CANNOT close as it stands: a tall blue CARTON stands upright
in the drawer's protruding front band, and the carton (130 mm) is taller than the
cabinet MOUTH (89 mm between the drawer floor and the top slab). Pushing the
drawer shut rams the carton's rear face into the slab's front edge; a few mm
later the drawer's own tall front panel (top only 4 mm below the slab) pinches
the carton from the other side, and the carton wedges diagonally between panel
and slab — a HARD geometric jam (tilting only INCREASES the carton's horizontal
diagonal, so the wedge locks; the pinch heights on both faces are within 4 mm,
so it can neither pitch into the mouth, 130 > 89, nor out over the panel).
The intended skill is IN-PLACE CARGO REORIENTATION: tip the carton over
SIDEWAYS so it lies flat on the drawer floor (44 mm ≪ 89 mm), keeping it INSIDE
the drawer, then push the drawer seated. Removing the carton instead fails —
success requires BOTH original contents (carton + a small white bar deeper
inside) to finish inside the seated drawer. The closing translation itself is
still the robot's push, but it is only REACHABLE after the reorientation: the
order is forced by geometry, not by a declared rule.

Assets are fully procedural (compound-spawner pattern): cabinet station
(KINEMATIC, fixed pose — bind-time joint anchors on a kinematic body are
world-fixed), drawer (DYNAMIC compound: floor, tall front panel + handle bar,
rear and side walls; on a bind-time prismatic joint whose limits are the hard
stops), the tall carton and the small companion bar (free cuboids riding the
drawer floor).

Per-episode randomization (readback-verified by smoke): the drawer's initial
opening q0, the carton's floor pose (x, side ±y, small yaw), the companion
bar's pose (x, y, free yaw).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.25  carton ever LYING FLAT inside the drawer (latched)
  0.45  x latched max drawer closing fraction (q0-q)/q0, GATED on the cargo
        being stowed (carton flat in the drawer AND companion in the drawer)
capped at 0.70; exactly 1.0 iff success(): drawer seated (q <= q_goal, in its
channel), carton lying flat inside, companion inside, all settled and finite —
live. Null policy ~0. The seed's strategy (push the drawer shut as-is) jams at
q >= ~0.05 and scores ~0 (the closing credit is gated on the stowed cargo).
Ejecting the carton lets the drawer close but can never succeed and earns ~0.

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


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped)."""
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the cabinet station at `prim_path`: KINEMATIC compound. Local
    frame: x=0 is the cabinet FRONT face plane (+x = out of the cabinet, toward
    the robot), z=0 is the ground. A plinth carries the drawer cavity: two side
    panels and a TOP SLAB whose underside is the cavity MOUTH — the height
    gate the cargo must fit under."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/plinth", x=(c.plinth_x0, 0.0), y=(-c.wall_y1, c.wall_y1),
          z=(0.0, c.plinth_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_p", x=(c.back_x0, 0.0), y=(c.wall_y0, c.wall_y1),
          z=(c.plinth_z1, c.slab_z0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_n", x=(c.back_x0, 0.0), y=(-c.wall_y1, -c.wall_y0),
          z=(c.plinth_z1, c.slab_z0), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/back", x=(c.plinth_x0, c.back_x0), y=(-c.wall_y1, c.wall_y1),
          z=(c.plinth_z1, c.slab_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/slab", x=(c.back_x0, 0.0), y=(-c.wall_y1, c.wall_y1),
          z=(c.slab_z0, c.slab_z1), color=c.trim_color, collide=collide)
    wood = _mk_material(prim_path, "wood", c.wood_mu_s, c.wood_mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the drawer at its CLOSED pose, root origin = station origin (so
    the prismatic joint anchors coincide and the joint coordinate IS the
    opening q). Open box: floor, TALL front panel (its top reaches to within a
    few mm of the slab underside — the anti-pitch-out geometry) + handle bar,
    rear wall, two side walls."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/panel", x=(-c.panel_t, 0.0), y=(-c.body_hy, c.body_hy),
          z=(c.floor_z0, c.panel_z1), color=c.panel_color, collide=collide)
    _span(stage, f"{prim_path}/handle", x=(0.0, 0.014), y=(-0.05, 0.05),
          z=(0.345, 0.357), color=c.handle_color, collide=collide)
    _span(stage, f"{prim_path}/floor", x=(c.floor_x0, -c.panel_t), y=(-c.body_hy, c.body_hy),
          z=(c.floor_z0, c.floor_z1), color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_rear", x=(c.floor_x0, c.floor_x0 + 0.008),
          y=(-c.body_hy, c.body_hy), z=(c.floor_z1, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yp", x=(c.floor_x0 + 0.008, -c.panel_t),
          y=(c.w_in, c.body_hy), z=(c.floor_z1, c.wall_z1),
          color=c.box_color, collide=collide)
    _span(stage, f"{prim_path}/w_yn", x=(c.floor_x0 + 0.008, -c.panel_t),
          y=(-c.body_hy, -c.w_in), z=(c.floor_z1, c.wall_z1),
          color=c.box_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp)
    wood = _mk_material(prim_path, "wood", c.wood_mu_s, c.wood_mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "station" not in _SPAWNER_CACHE:

        @configclass
        class StationSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_station)
            plinth_x0: float = -0.219
            plinth_z1: float = 0.300
            back_x0: float = -0.207
            slab_z0: float = 0.398
            slab_z1: float = 0.428
            wall_y0: float = 0.157
            wall_y1: float = 0.169
            body_color: tuple = (0.52, 0.38, 0.24)
            trim_color: tuple = (0.40, 0.28, 0.17)
            contact_offset: float = 0.0015
            wood_mu_s: float = 0.55
            wood_mu_d: float = 0.50

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            panel_t: float = 0.018
            panel_z1: float = 0.394
            body_hy: float = 0.142
            w_in: float = 0.134
            floor_x0: float = -0.201
            floor_z0: float = 0.301
            floor_z1: float = 0.309
            wall_z1: float = 0.389
            mass: float = 0.35
            lin_damp: float = 6.0
            ang_damp: float = 4.0
            panel_color: tuple = (0.46, 0.48, 0.54)
            handle_color: tuple = (0.20, 0.20, 0.22)
            box_color: tuple = (0.68, 0.56, 0.40)
            contact_offset: float = 0.0015
            wood_mu_s: float = 0.55
            wood_mu_d: float = 0.50

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class StowflatSceneCfg(BaseCfg):
    """Config for `StowflatScene`. The jam/pass geometry is asserted in
    `__post_init__`: the upright carton really is taller than the mouth (jam),
    the flat carton really fits under it (pass), the front panel really pinches
    within a few mm of the slab underside (no pitch-out slack), the hard-pinch
    opening really sits far above the seated tolerance, the sideways tip really
    lands inside the drawer interior clear of the far wall and the companion,
    and every spawn band is clear of walls, mouth and the other body."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    q_goal: float = tunable(0.010)        # drawer opening for "seated" (m)
    settle_lin: float = tunable(0.05)     # max |lin vel| of movers when judging (m/s)
    settle_ang: float = tunable(0.60)     # max |ang vel| of the carton when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    q0_lo: float = tunable(0.125)         # drawer initial opening, low (m)
    q0_hi: float = tunable(0.160)         # drawer initial opening, high (m)
    cart_x: tuple = tunable((-0.085, -0.060))  # carton spawn x band (drawer-local)
    cart_y: tuple = tunable((0.035, 0.050))    # carton spawn |y| band (side swaps randomly)
    cart_yaw_deg: float = tunable(8.0)    # carton spawn |yaw| bound (deg)
    comp_x: tuple = tunable((-0.160, -0.150))  # companion spawn x band (drawer-local)
    comp_y: float = tunable(0.080)        # companion spawn |y| bound
    carton_mass: float = tunable(0.12)    # carton mass (kg)
    comp_mass: float = tunable(0.05)      # companion bar mass (kg)

    # --- info: station (local frame: x=0 cabinet face, +x = out, z=0 ground) ---------------------
    plinth_x0: float = info(-0.219)
    plinth_z1: float = info(0.300)
    back_x0: float = info(-0.207)         # cavity rear plane
    slab_z0: float = info(0.398)          # slab underside: the MOUTH top
    slab_z1: float = info(0.428)
    wall_y0: float = info(0.157)          # cavity opening half-width
    wall_y1: float = info(0.169)
    # --- info: drawer (authored closed; prismatic axis X, limits = hard stops) -------------------
    panel_t: float = info(0.018)          # front panel thickness
    panel_z1: float = info(0.394)         # front panel top (4 mm below the slab underside)
    body_hy: float = info(0.142)
    w_in: float = info(0.134)             # drawer interior half-width
    floor_x0: float = info(-0.201)
    floor_z0: float = info(0.301)
    floor_z1: float = info(0.309)         # drawer floor top: the cargo surface
    wall_z1: float = info(0.389)
    stroke: float = info(0.175)           # prismatic upper limit (m)
    drawer_mass: float = info(0.35)
    # --- info: cargo ------------------------------------------------------------------------------
    carton_w: float = info(0.044)         # carton footprint edge (square)
    carton_hz: float = info(0.130)        # carton height (upright) — TALLER than the mouth
    comp_lx: float = info(0.045)          # companion bar extents
    comp_ly: float = info(0.030)
    comp_lz: float = info(0.020)
    # --- info: "inside the drawer" box (drawer-local root-position test) -------------------------
    box_x: tuple = info((-0.189, -0.022))
    box_hy: float = info(0.130)
    box_z: tuple = info((0.295, 0.392))
    flat_z_max: float = info(0.352)       # flat carton centre must sit ON the floor, not perched
    flat_axis_max: float = info(0.35)     # |long-axis z-component| bound for "lying flat"
    # --- info: materials --------------------------------------------------------------------------
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.25 + 0.45 = 0.70 = the non-success cap) -------------------------
    w_flat: float = info(0.25)
    w_close: float = info(0.45)

    def __post_init__(self) -> None:
        mouth = self.slab_z0 - self.floor_z1            # 0.089
        panel_h = self.panel_z1 - self.floor_z1         # 0.085
        g = math.radians(self.cart_yaw_deg)
        half = self.carton_w / 2
        reach_up = half * (math.cos(g) + math.sin(g))   # upright footprint x/y half-reach at yaw
        flat_rx = (self.carton_hz / 2) * math.sin(g) + half * math.cos(g)  # flat x half-reach
        # --- the height gate: upright jams, flat passes ------------------------------------------
        assert self.carton_hz - mouth >= 0.035, "upright carton must be well taller than the mouth"
        assert mouth - self.carton_w >= 0.030, "flat carton must pass well under the mouth"
        # --- the anti-pitch-out pinch: panel top within a few mm of the slab underside -----------
        assert 0.0 < mouth - panel_h <= 0.006, "panel must pinch at nearly the mouth height"
        # panel/slab vertical gap must exceed the summed contact offsets (no phantom contact)
        assert self.slab_z0 - self.panel_z1 >= 2 * self.contact_offset + 0.0005
        # --- the hard-pinch opening sits FAR above the seated tolerance --------------------------
        # (pinned carton rear at the mouth plane, panel interior face against its front:
        #  q_pinch = carton_w + panel_t; allow 30 mm of tilt/compliance slack)
        assert self.carton_w + self.panel_t - 0.030 >= self.q_goal + 0.010, \
            "the jam must arrest the drawer far from seated"
        # --- spawn bands ---------------------------------------------------------------------------
        assert 0.0 < self.q0_lo < self.q0_hi <= self.stroke - 0.012
        # upright carton fully OUTSIDE the mouth at every sampled q0 (open sky above)
        assert self.q0_lo + self.cart_x[0] - reach_up >= 0.010, \
            "carton must spawn clear of the mouth plane"
        # carton top pokes above the whole cabinet (top-grasp / side-push access)
        assert self.floor_z1 + self.carton_hz >= self.slab_z1 + 0.008
        # upright carton clear of the near side wall
        assert self.cart_y[1] + reach_up <= self.w_in - 0.015
        assert 0.0 < self.cart_y[0] < self.cart_y[1]
        # --- the sideways tip: lands flat INSIDE, clear of far wall and companion ----------------
        # far edge of the flat carton (tipped inward across the centreline):
        far_y = self.cart_y[0] - (half + self.carton_hz) * math.cos(g) - half * math.sin(g)
        assert far_y >= -(self.w_in - 0.006), "flat carton must land inside the far wall"
        # swing arc (radius = carton diagonal from the pivot edge) never reaches the far wall
        diag = math.hypot(self.carton_hz, self.carton_w)
        assert self.w_in + (self.cart_y[0] - half) >= diag + 0.004, \
            "the tip swing must clear the far wall"
        # flat carton stays inside the interior x-wise and clear of the front panel
        assert self.cart_x[1] + flat_rx <= -self.panel_t - 0.005
        assert self.cart_x[0] - flat_rx >= self.floor_x0 + 0.016
        # --- companion band: clear of rear wall, side walls, and the carton (upright AND flat) ---
        comp_r = math.hypot(self.comp_lx / 2, self.comp_ly / 2)
        assert self.comp_x[0] - comp_r >= self.floor_x0 + 0.008 + 0.004
        assert self.comp_x[1] + comp_r <= self.cart_x[0] - flat_rx - 0.004, \
            "companion band must stay clear of the flat carton's footprint"
        assert self.comp_y + comp_r <= self.w_in - 0.008
        # companion fits under the mouth with margin (it never gates anything by height)
        assert mouth - self.comp_lz >= 0.050
        # --- drawer-in-cabinet clearances --------------------------------------------------------
        assert self.floor_x0 - self.back_x0 >= 0.004          # rear gap at closed
        assert self.body_hy <= self.wall_y0 - 0.008           # panel inside the opening
        assert self.floor_z0 - self.plinth_z1 >= 0.0005       # floor hangs above the plinth
        assert self.wall_z1 <= self.slab_z0 - 0.006           # side walls under the slab
        # --- judged boxes consistent with the authored interior ----------------------------------
        assert self.box_x[0] >= self.floor_x0 + 0.008 and self.box_x[1] <= -self.panel_t + 0.002
        assert self.box_hy <= self.w_in - 0.002
        assert self.box_z[0] <= self.floor_z0 and self.box_z[1] >= self.floor_z1 + self.carton_w
        assert self.flat_z_max >= self.floor_z1 + self.carton_w / 2 + 0.010
        assert self.flat_z_max <= self.floor_z1 + panel_h - 0.030, \
            "flat clause must exclude wall-perched cargo"
        # --- rubric weights ----------------------------------------------------------------------
        assert abs(self.w_flat + self.w_close - 0.70) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("stowflat_cabinet")
class StowflatScene(BaseScene):
    cfg: StowflatSceneCfg

    def __init__(self, cfg: StowflatSceneCfg | None = None) -> None:
        super().__init__(cfg or StowflatSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            plinth_x0=c.plinth_x0, plinth_z1=c.plinth_z1, back_x0=c.back_x0,
            slab_z0=c.slab_z0, slab_z1=c.slab_z1, wall_y0=c.wall_y0, wall_y1=c.wall_y1,
            contact_offset=c.contact_offset)
        drawer_spawn = cls["drawer"](
            panel_t=c.panel_t, panel_z1=c.panel_z1, body_hy=c.body_hy, w_in=c.w_in,
            floor_x0=c.floor_x0, floor_z0=c.floor_z0, floor_z1=c.floor_z1,
            wall_z1=c.wall_z1, mass=c.drawer_mass, contact_offset=c.contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.15, angular_damping=0.15,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)

        def box(size, color, mass):
            return sim_utils.CuboidCfg(
                size=size,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=rigid, collision_props=coll,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.55, dynamic_friction=0.50, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station", spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # Drawer authored IN PLACE at its closed pose: the bind-time prismatic
            # joint anchors at this authored position.
            "drawer": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Drawer", spawn=drawer_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "carton": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Carton",
                spawn=box((c.carton_w, c.carton_w, c.carton_hz), (0.16, 0.32, 0.75),
                          c.carton_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.6, 0.10))),
            "comp": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Comp",
                spawn=box((c.comp_lx, c.comp_ly, c.comp_lz), (0.92, 0.92, 0.90),
                          c.comp_mass),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, -0.6, 0.05))),
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
        self.station: RigidObject = env.iscene["station"]
        self.drawer: RigidObject = env.iscene["drawer"]
        self.carton: RigidObject = env.iscene["carton"]
        self.comp: RigidObject = env.iscene["comp"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.q0 = torch.zeros(n, device=dev)          # initial drawer opening
        # latches (partial credit survives transients; success is judged live)
        self._fflat = torch.zeros(n, dtype=torch.bool, device=dev)
        self._fclose = torch.zeros(n, device=dev)     # max gated closing fraction
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env prismatic joint, authored ONCE at bind time against the
        AUTHORED poses (drawer root coincides with the station root). Joint
        LIMITS are the hard stops (0 = seated .. stroke = fully out)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Station"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.stroke))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: station re-asserted at its fixed pose (kinematic joint
        anchors are world-fixed — the station must never move), drawer at a
        SAMPLED opening q0, carton standing upright in the drawer's front band
        (sampled x, random side, small yaw), companion bar deeper inside
        (sampled x/y, free yaw), latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def place(body, dx, dy, dz, yaw=None) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = origin[:, 0] + dx
            s[:, 1] = origin[:, 1] + dy
            s[:, 2] = origin[:, 2] + dz
            if yaw is None:
                s[:, 3] = 1.0
            else:
                s[:, 3] = torch.cos(yaw / 2)
                s[:, 6] = torch.sin(yaw / 2)
            body.write_root_state_to_sim(s, env_ids)

        zeros = torch.zeros(m, device=dev)
        place(self.station, zeros, zeros, zeros)

        q0 = c.q0_lo + torch.rand(m, device=dev) * (c.q0_hi - c.q0_lo)
        self.q0[env_ids] = q0
        place(self.drawer, q0, zeros, zeros)

        # cargo rides the drawer: spawn poses are drawer-local + q0 (torch.rand-
        # based — the first randint after a manual seed is degenerate across seeds)
        u = torch.rand(m, 7, device=dev)
        side = torch.where(u[:, 6] < 0.5, torch.ones(m, device=dev), -torch.ones(m, device=dev))
        cx = c.cart_x[0] + u[:, 0] * (c.cart_x[1] - c.cart_x[0])
        cy = side * (c.cart_y[0] + u[:, 1] * (c.cart_y[1] - c.cart_y[0]))
        cyaw = (u[:, 2] * 2 - 1) * math.radians(c.cart_yaw_deg)
        place(self.carton, q0 + cx, cy, zeros + c.floor_z1 + c.carton_hz / 2 + 0.002,
              yaw=cyaw)
        bx = c.comp_x[0] + u[:, 3] * (c.comp_x[1] - c.comp_x[0])
        by = (u[:, 4] * 2 - 1) * c.comp_y
        place(self.comp, q0 + bx, by, zeros + c.floor_z1 + c.comp_lz / 2 + 0.002,
              yaw=(u[:, 5] * 2 - 1) * math.pi)

        self._fflat[env_ids] = False
        self._fclose[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "drawer": self.drawer.data.root_state_w[env_ids].clone(),
            "carton": self.carton.data.root_state_w[env_ids].clone(),
            "comp": self.comp.data.root_state_w[env_ids].clone(),
            "q0": self.q0[env_ids].clone(),
            "fflat": self._fflat[env_ids].clone(),
            "fclose": self._fclose[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.drawer.write_root_state_to_sim(state["drawer"], env_ids)
        self.carton.write_root_state_to_sim(state["carton"], env_ids)
        self.comp.write_root_state_to_sim(state["comp"], env_ids)
        self.q0[env_ids] = state["q0"]
        self._fflat[env_ids] = state["fflat"]
        self._fclose[env_ids] = state["fclose"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        mouth = c.slab_z0 - c.floor_z1
        return (
            f"A wooden KITCHEN CABINET stands on the ground; its top drawer starts "
            f"pulled OUT (the exact opening varies by episode). The drawer's front "
            f"band protrudes into open air, and standing upright on the drawer "
            f"floor there is a tall BLUE CARTON "
            f"({c.carton_w * 100:.1f} x {c.carton_w * 100:.1f} x "
            f"{c.carton_hz * 100:.0f} cm) — TALLER than the cabinet's mouth (the "
            f"gap under the top slab is only {mouth * 100:.1f} cm). Deeper inside "
            f"the drawer lies a small WHITE BAR. If you simply push the drawer "
            f"shut (grey front panel with a dark handle bar), the carton rams the "
            f"top slab and wedges between the slab and the drawer's tall front "
            f"panel: the drawer hard-jams several centimetres from closed, and "
            f"no amount of force closes it.\n"
            f"Instead, first TIP THE CARTON OVER SIDEWAYS (toward the drawer's "
            f"centreline) so it lies FLAT on the drawer floor — lying down it is "
            f"only {c.carton_w * 100:.1f} cm tall and passes freely under the "
            f"slab. It must stay INSIDE the drawer: lifting it out (or tossing "
            f"the white bar out) and closing the empty drawer does NOT count — "
            f"the goal is the drawer seated WITH all its original contents "
            f"stowed inside. Then push the drawer shut by its handle until it "
            f"seats.\n"
            f"Goal: carton lying flat inside the drawer, white bar inside, drawer "
            f"seated within {c.q_goal * 1000:.0f} mm — all at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "The tall blue carton standing in the open drawer is taller than the "
            "cabinet opening, so the drawer will not close as it stands. Tip the "
            "carton over sideways so it lies flat inside the drawer, keep the "
            "white bar inside too, then push the drawer fully shut."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the station frame (station fixed at the
        env origin with identity heading)."""
        return body.data.root_pos_w - self.env_origins

    def drawer_open(self) -> torch.Tensor:
        """(N,) drawer opening (its prismatic coordinate; 0 = seated/closed)."""
        return self._local(self.drawer)[:, 0]

    def drawer_in_channel(self) -> torch.Tensor:
        """(N,) bool: drawer root on its slide, un-rotated (guards constructed
        stolen-drawer fakes; the joint makes violations physically impossible)."""
        p = self._local(self.drawer)
        return (p[:, 1].abs() < 0.03) & (p[:, 2].abs() < 0.03) \
            & (p[:, 0] > -0.02) & (p[:, 0] < self.cfg.stroke + 0.02) \
            & (self.drawer.data.root_quat_w[:, 0].abs() > 0.999)

    def _drawer_frame(self, body) -> torch.Tensor:
        """(N,3) body root position in the DRAWER frame (rides the slide)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(
            self.drawer.data.root_quat_w,
            body.data.root_pos_w - self.drawer.data.root_pos_w)

    def _in_box(self, body) -> torch.Tensor:
        c = self.cfg
        p = self._drawer_frame(body)
        return (p[:, 0] > c.box_x[0]) & (p[:, 0] < c.box_x[1]) \
            & (p[:, 1].abs() < c.box_hy) \
            & (p[:, 2] > c.box_z[0]) & (p[:, 2] < c.box_z[1])

    def carton_in(self) -> torch.Tensor:
        """(N,) bool: carton root inside the drawer's interior box."""
        return self._in_box(self.carton) & self.drawer_in_channel()

    def comp_in(self) -> torch.Tensor:
        """(N,) bool: companion bar root inside the drawer's interior box."""
        return self._in_box(self.comp) & self.drawer_in_channel()

    def carton_axis_z(self) -> torch.Tensor:
        """(N,) |z-component| of the carton's long (body-z) axis in the world."""
        from isaaclab.utils.math import quat_apply

        ez = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        ez[:, 2] = 1.0
        return quat_apply(self.carton.data.root_quat_w, ez)[:, 2].abs()

    def carton_flat(self) -> torch.Tensor:
        """(N,) bool: carton LYING FLAT on the drawer floor inside the drawer
        (long axis horizontal, centre at floor height — excludes upright,
        leaning, and wall-perched poses)."""
        c = self.cfg
        rel_z = self._drawer_frame(self.carton)[:, 2]
        return (self.carton_axis_z() < c.flat_axis_max) \
            & (rel_z < c.flat_z_max) & self.carton_in()

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.drawer.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.carton.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.carton.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.comp.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.drawer.data.root_pos_w, self.carton.data.root_pos_w,
                         self.comp.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        self._fflat |= self.carton_flat() & fin
        # closing credit only accrues while the cargo is genuinely stowed:
        # carton flat in the drawer AND companion in the drawer (kills both the
        # seed strategy and every eject-the-cargo route)
        gate = self.carton_flat() & self.comp_in() & fin
        fc = ((self.q0 - self.drawer_open()) / self.q0.clamp(min=1e-4)).clamp(0.0, 1.0)
        fc = torch.where(gate, fc, torch.zeros_like(fc))
        self._fclose = torch.maximum(self._fclose, fc)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, judged LIVE on the settled physical state: drawer seated
        (<= q_goal, in its channel), carton lying FLAT inside the drawer,
        companion bar inside the drawer — all settled and finite. Geometry
        makes the clauses jointly reachable only through the intended order:
        an upright (or leaning, or wall-perched) carton is taller than the
        mouth and jams the drawer, so `seated & carton_in` physically implies
        the carton was laid flat BEFORE the final push."""
        c = self.cfg
        self._update_latches()
        return (self.drawer_open() <= c.q_goal) & self.drawer_in_channel() \
            & self.carton_flat() & self.comp_in() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 carton-flat-in-drawer (latched) + 0.45 x
        gated closing fraction (latched max), capped at 0.70; exactly 1.0 iff
        success() holds live. Null ~0. The seed's strategy (push the drawer
        shut as it stands) jams and earns ~0: the closing credit is gated on
        the stowed cargo."""
        c = self.cfg
        self._update_latches()
        base = (c.w_flat * self._fflat.float() + c.w_close * self._fclose).clamp(max=0.70)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="stowflat_cabinet", robot="null"))
