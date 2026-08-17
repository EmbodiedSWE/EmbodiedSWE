"""GondolaWheelScene — ride the black bowl over the crank wheel into the roofed rooftop
gallery (libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet_i362).

Derived from libero_90 kitchen_scene1 "put the black bowl on top of the cabinet", where
the whole task is one pick-and-place of the black bowl onto the cabinet's open top
surface. Here the destination is a ROOFED ROOFTOP GALLERY on the cabinet: a parapet-
walled deck under a solid roof slab, reachable only through a narrow window in the
front parapet. The seed's plan (lift the bowl, set it down on top from above) is dead
by construction — the roof seals the top.

The only sanctioned route is a RIDE on the hand-cranked gondola wheel that stands
beside the cabinet:

  1. LOAD  — set the bowl into the pendulum gondola tray hanging from the wheel's rim
     pin while the wheel rests on its bottom loading stop;
  2. CRANK — drive the wheel ~150 degrees by its rim pegs; the free-hanging gondola
     tray self-levels through the whole arc (a pendulum on its own hinge), carrying the
     bowl up and over the top of the wheel;
  3. PARK  — the wheel's upper travel stop is 6 degrees PAST top-dead-center, so at the
     top the hanging load's own weight presses the wheel onto the stop (an over-center
     gravity latch, no holding torque needed) with the tray parked flush beside the
     gallery window;
  4. SERVE — slide the bowl off the tray's open front edge, across the 15 mm gap,
     through the window; it drops 9 mm onto the gallery deck and must rest upright
     inside.

EXECUTION-ORDER RULE (declared, latch-enforced): the bowl must reach the gallery BY
RIDING THE GONDOLA — aboard the tray at the bottom stop, aboard through mid-arc, aboard
at the parked top. An end state with the bowl on the deck but without that ride
provenance (e.g. hand-carried to the window and pushed in) is out-of-order and scores
~0 with success() false. The roof independently kills the seed's drop-from-above plan.

Mechanism notes (proven robobench cribs):
  - The fixture (pylon + axle stub + staging table + cabinet + parapet + roof) is ONE
    kinematic compound body, never teleported. The wheel disc and the gondola are each
    one compound rigid body on per-env authored USD revolute joints (bind-time
    authoring): fixture<->disc axis Y with limits, disc<->gondola axis Y free (the
    pendulum). Jointed pairs have pair collision disabled.
  - The wheel travels world 30..186 deg. To keep the revolute far from the PhysX
    +-180 wrap boundary, the disc is AUTHORED at mid-travel (108 deg) so the joint
    limits are +-78 deg (well under the 175-deg slingshot threshold).
  - The gondola's CoM is authored BELOW its hinge pin (explicit MassAPI center of
    mass), so the tray is a stable pendulum and self-levels through the ride.
  - Reset re-poses the WHOLE linkage (disc + gondola together, consistent poses about
    the unchanged hinges) with a 2-substep grace re-pin.

Everything is procedural. Heavy imports (isaaclab, pxr) are deferred so importing this
module stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _xform_root(stage, prim_path: str, translation, orientation):
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    return xform.GetPrim()


def _part_box(stage, path, size, center, color, contact_offset):
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(cube.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    bxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _part_cyl_y(stage, path, radius, height, center, color, contact_offset):
    """Cylinder with axis along local Y."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Y")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    r, h = float(radius), float(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-r, -h / 2, -r), Gf.Vec3f(r, h / 2, r)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _spawn_fixture(prim_path: str, cfg: Any, translation=None, orientation=None):
    """ONE kinematic compound body: wheel pylon + axle stub, staging table, cabinet
    plinth, parapet walls with the front window, and the roof slab. Env-local coords
    (the fixture spawns at the env origin, identity)."""
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    co = cfg.contact_offset

    def box(name, size, center, color):
        _part_box(stage, f"{prim_path}/{name}", size, center, color, co)

    # wheel support: pylon column on the -y side of the disc + axle stub toward the
    # hub. NOTE: 10 mm clear of the disc face (y -0.01) — jointed-pair clearance must
    # be REAL (authored gaps, not the joint collision filter), and beyond the summed
    # contact offsets.
    box("pylon", (0.06, 0.04, cfg.hub_z), (0.0, -0.095, cfg.hub_z / 2), cfg.steel_color)
    _part_cyl_y(stage, f"{prim_path}/axle", 0.02, 0.055, (0.0, -0.0475, cfg.hub_z),
                cfg.steel_color, co)
    # staging table (bowl + plate spawn here)
    box("table", (0.32, 0.36, cfg.table_top), (-0.46, 0.08, cfg.table_top / 2),
        cfg.table_color)
    # cabinet plinth, deck top at z = deck_z
    box("plinth", (0.28, 0.24, cfg.deck_z), (0.02, 0.29, cfg.deck_z / 2),
        cfg.plinth_color)
    # parapet walls z deck_z .. 0.65, t = 0.02; front wall carries the window
    box("wall_left", (0.02, 0.24, 0.13), (-0.11, 0.29, 0.585), cfg.wall_color)
    box("wall_right", (0.02, 0.24, 0.13), (0.15, 0.29, 0.585), cfg.wall_color)
    box("wall_back", (0.28, 0.02, 0.13), (0.02, 0.40, 0.585), cfg.wall_color)
    # front wall: window x [-0.055, 0.095], z [deck_z, 0.62]; jambs + header
    box("jamb_left", (0.065, 0.02, 0.13), (-0.0875, 0.18, 0.585), cfg.wall_color)
    box("jamb_right", (0.065, 0.02, 0.13), (0.1275, 0.18, 0.585), cfg.wall_color)
    box("header", (0.15, 0.02, 0.03), (0.02, 0.18, 0.635), cfg.wall_color)
    # roof slab — seals the gallery from above (kills the seed's drop-in plan)
    box("roof", (0.30, 0.252, 0.02), (0.02, 0.294, 0.66), cfg.roof_color)
    return root


def _spawn_disc(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the crank wheel. Body frame: disc plane = local x-z, axis =
    local y, hub at the origin. Rim pin (gondola hinge) on the +y face at local
    (0, ~0.05, -arm_r); four crank pegs on the -y face at radius peg_rad."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass_props.mass))
    from pxr import Gf

    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.010, 0.016, 0.010))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(cfg.ang_damping))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    co = cfg.contact_offset

    _part_cyl_y(stage, f"{prim_path}/disc", cfg.disc_r, 0.02, (0.0, 0.0, 0.0),
                cfg.disc_color, co)
    # gondola hinge pin, +y face, radially at -z (bottom of the wheel at joint zero)
    _part_cyl_y(stage, f"{prim_path}/pin", cfg.pin_r, 0.025,
                (0.0, 0.0475, -cfg.arm_r), cfg.pin_color, co)
    # crank pegs, -y face (the graspable handles that drive the wheel)
    for k in range(4):
        a = math.pi / 2 * k
        _part_cyl_y(stage, f"{prim_path}/peg_{k}", 0.011, 0.05,
                    (cfg.peg_rad * math.cos(a), -0.035, cfg.peg_rad * math.sin(a)),
                    cfg.peg_color, co)
    return root


def _spawn_gondola(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the pendulum gondola. Body frame: origin at the tray-floor
    center; hinge pin bore up at local (0, -0.04, +hang_z); the tray has curbs on
    +-x and the back (-y) — the FRONT (+y, toward the gallery) is OPEN. CoM authored
    below the pin so the tray self-levels."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(cfg.mass_props.mass))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, -0.005, 0.020))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0015, 0.0015, 0.0015))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(float(cfg.ang_damping))
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    co = cfg.contact_offset

    def box(name, size, center):
        _part_box(stage, f"{prim_path}/{name}", size, center, cfg.color, co)

    # NOTE: the arms top out at local z 0.083, 13 mm BELOW the hinge pin's collision
    # bottom (pin center z = hang_z 0.110, r 0.014), and there is NO crossbar over the
    # basin — jointed-pair clearance must be REAL (authored gaps, not the joint
    # collision filter), and the bowl's teleport-release corridor above the basin must
    # be free geometry.
    box("floor", (0.15, 0.13, 0.01), (0.0, 0.0, 0.0))
    box("curb_xp", (0.01, 0.13, 0.035), (0.08, 0.0, 0.0225))
    box("curb_xn", (0.01, 0.13, 0.035), (-0.08, 0.0, 0.0225))
    box("curb_back", (0.17, 0.01, 0.035), (0.0, -0.07, 0.0225))
    box("arm_p", (0.012, 0.012, 0.078), (0.08, -0.04, 0.044))
    box("arm_n", (0.012, 0.012, 0.078), (-0.08, -0.04, 0.044))
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: the black bowl — an open octagonal cup (bottom disc + 8 wall
    segments). Body frame: axis = +z (up when upright), origin at mid-height."""
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = _xform_root(stage, prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.05)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    color = Gf.Vec3f(*cfg.color)

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    outer_r = cfg.inner_r + cfg.wall_t
    bot = UsdGeom.Cylinder.Define(stage, f"{prim_path}/bottom")
    bot.CreateRadiusAttr(outer_r)
    bot.CreateHeightAttr(cfg.bot_t)
    bot.CreateExtentAttr([Gf.Vec3f(-outer_r, -outer_r, -cfg.bot_t / 2),
                          Gf.Vec3f(outer_r, outer_r, cfg.bot_t / 2)])
    UsdGeom.Xformable(bot.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, -cfg.height / 2 + cfg.bot_t / 2))
    bot.CreateDisplayColorAttr([color])
    collide(bot.GetPrim())

    n = 8
    r_mid = cfg.inner_r + cfg.wall_t / 2
    seg_len = 2 * (cfg.inner_r + cfg.wall_t) * math.tan(math.pi / n) + 0.002
    for k in range(n):
        ang = 2 * math.pi * k / n
        seg = UsdGeom.Cube.Define(stage, f"{prim_path}/wall_{k}")
        seg.CreateSizeAttr(1.0)
        sxf = UsdGeom.Xformable(seg.GetPrim())
        sxf.AddTranslateOp().Set(Gf.Vec3d(r_mid * math.cos(ang), r_mid * math.sin(ang), 0.0))
        sxf.AddRotateZOp().Set(math.degrees(ang))
        sxf.AddScaleOp().Set(Gf.Vec3f(cfg.wall_t, seg_len, cfg.height))
        seg.CreateDisplayColorAttr([color])
        collide(seg.GetPrim())
    return root


def _fixture_spawner_cfg(c: GondolaWheelSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "fixture" not in _SPAWNER_CACHE:

        @configclass
        class FixtureSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_fixture)
            hub_z: float = 0.41
            table_top: float = 0.15
            deck_z: float = 0.52
            steel_color: tuple = (0.45, 0.47, 0.52)
            table_color: tuple = (0.52, 0.37, 0.22)
            plinth_color: tuple = (0.36, 0.25, 0.15)
            wall_color: tuple = (0.30, 0.21, 0.13)
            roof_color: tuple = (0.24, 0.17, 0.11)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["fixture"] = FixtureSpawnerCfg

    return _SPAWNER_CACHE["fixture"](
        mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        hub_z=c.hub_z, table_top=c.table_top, deck_z=c.deck_z,
        contact_offset=c.contact_offset,
    )


def _disc_spawner_cfg(c: GondolaWheelSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "disc" not in _SPAWNER_CACHE:

        @configclass
        class DiscSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_disc)
            disc_r: float = 0.22
            arm_r: float = 0.20
            pin_r: float = 0.014
            peg_rad: float = 0.14
            ang_damping: float = 0.15
            disc_color: tuple = (0.55, 0.12, 0.12)
            pin_color: tuple = (0.80, 0.68, 0.20)
            peg_color: tuple = (0.75, 0.75, 0.78)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["disc"] = DiscSpawnerCfg

    return _SPAWNER_CACHE["disc"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.disc_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        disc_r=c.disc_r, arm_r=c.arm_r, pin_r=0.014, peg_rad=c.peg_rad,
        ang_damping=c.disc_ang_damping, contact_offset=c.contact_offset,
    )


def _gondola_spawner_cfg(c: GondolaWheelSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "gondola" not in _SPAWNER_CACHE:

        @configclass
        class GondolaSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gondola)
            hang_z: float = 0.110
            ang_damping: float = 0.3
            color: tuple = (0.72, 0.54, 0.30)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["gondola"] = GondolaSpawnerCfg

    return _SPAWNER_CACHE["gondola"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.gondola_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        hang_z=c.hang_z, ang_damping=c.gondola_ang_damping,
        contact_offset=c.contact_offset,
    )


def _bowl_spawner_cfg(c: GondolaWheelSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bowl" not in _SPAWNER_CACHE:

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            inner_r: float = 0.040
            wall_t: float = 0.011
            height: float = 0.055
            bot_t: float = 0.010
            color: tuple = (0.08, 0.08, 0.10)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg

    return _SPAWNER_CACHE["bowl"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.bowl_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        inner_r=c.bowl_inner_r, wall_t=c.bowl_wall_t, height=c.bowl_h,
        bot_t=c.bowl_bot_t, color=c.bowl_color, contact_offset=0.002,
    )


# ----- scene cfg -----------------------------------------------------------------------------------
@dataclass
class GondolaWheelSceneCfg(BaseCfg):
    """Config for `GondolaWheelScene`. World frame (per env origin): the crank wheel
    turns in the x-z plane about a y axis hub at (0, 0, hub_z); the staging table is at
    -x; the roofed gallery cabinet at +y; the gallery window faces -y toward the
    wheel's over-center park position."""

    # --- tunable: rubric thresholds ----------------------------------------------------------------
    settle_speed: float = tunable(0.05)  # max |lin vel| (bowl, plate, gondola) when judging (m/s)
    wheel_settle_w: float = tunable(0.15)  # max |ang vel| of the wheel when judging (rad/s)
    bowl_up_max_deg: float = tunable(15.0)  # bowl axis within this of world-up
    # bowl root in the GONDOLA frame to count "aboard the tray"
    tray_x_abs: float = tunable(0.055)
    tray_y_abs: float = tunable(0.055)
    tray_z_lo: float = tunable(0.015)
    tray_z_hi: float = tunable(0.050)  # rest is ~0.0325; release above this latches nothing
    # bowl root in the ENV frame to count "on the gallery deck" (inside the parapet,
    # past the window, resting on the deck top; z band rejects hovering/stacking)
    deck_x_lo: float = tunable(-0.08)
    deck_x_hi: float = tunable(0.12)
    deck_y_lo: float = tunable(0.23)
    deck_y_hi: float = tunable(0.37)
    deck_z_lo: float = tunable(0.53)
    deck_z_hi: float = tunable(0.59)  # rest is ~0.5475
    # ride-provenance latch angles (wheel angle, deg)
    load_max_deg: float = tunable(60.0)  # "loaded": bowl aboard with the wheel this low
    ride_lo_deg: float = tunable(95.0)  # "rode": bowl aboard through mid-arc
    ride_hi_deg: float = tunable(125.0)
    park_deg: float = tunable(183.0)  # "hoisted": bowl aboard with the wheel parked

    # --- tunable: mechanism ------------------------------------------------------------------------
    travel_lo_deg: float = tunable(30.0)  # bottom loading stop
    travel_hi_deg: float = tunable(186.0)  # over-center park stop (6 deg past TDC)
    reset_lo_deg: float = tunable(32.0)  # sampled initial wheel angle (falls to the stop)
    reset_hi_deg: float = tunable(46.0)

    # --- tunable: randomization --------------------------------------------------------------------
    spawn_jitter: float = tunable(0.03)  # uniform +/- xy jitter of bowl and plate spawns
    slot_a: tuple = tunable((-0.50, 0.14))  # staging slots; bowl/plate assignment sampled
    slot_b: tuple = tunable((-0.42, 0.02))

    # --- info: fixture (kinematic, FIXED — never teleported) ---------------------------------------
    hub_z: float = info(0.435)  # wheel hub height
    table_top: float = info(0.15)  # staging table top
    deck_z: float = info(0.52)  # gallery deck top (plinth top)
    authored_deg: float = info(108.0)  # disc authored at mid-travel (joint limits +-78)
    contact_offset: float = info(0.003)

    # --- info: wheel / gondola ---------------------------------------------------------------------
    disc_r: float = info(0.22)
    arm_r: float = info(0.20)  # hub -> gondola pin radial distance
    peg_rad: float = info(0.14)  # crank peg radius
    pin_y: float = info(0.05)  # gondola hinge anchor, world y
    hang_y: float = info(0.04)  # gondola origin y = pin_y + hang_y
    hang_z: float = info(0.110)  # gondola origin sits this far below the pin
    disc_mass: float = info(0.8)
    disc_ang_damping: float = info(0.15)
    gondola_mass: float = info(0.25)
    gondola_ang_damping: float = info(0.3)

    # --- info: movable objects ---------------------------------------------------------------------
    bowl_inner_r: float = info(0.040)
    bowl_wall_t: float = info(0.011)
    bowl_h: float = info(0.055)
    bowl_bot_t: float = info(0.010)
    bowl_mass: float = info(0.35)
    bowl_color: tuple = info((0.08, 0.08, 0.10))  # BLACK bowl (the goal object)
    plate_r: float = info(0.08)
    plate_h: float = info(0.015)
    plate_mass: float = info(0.15)
    plate_color: tuple = info((0.92, 0.92, 0.90))  # white plate (distractor)

    # Derived (filled in __post_init__).
    bowl_outer_r: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.bowl_outer_r = round(self.bowl_inner_r + self.bowl_wall_t, 4)


# ----- scene ----------------------------------------------------------------------------------------
@SCENES.register("gondola_wheel")
class GondolaWheelScene(BaseScene):
    cfg: GondolaWheelSceneCfg

    def __init__(self, cfg: GondolaWheelSceneCfg | None = None) -> None:
        super().__init__(cfg or GondolaWheelSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        tight = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset,
                                                 rest_offset=0.0)
        mid = math.radians(c.authored_deg)
        qw, qy = math.cos(mid / 2), math.sin(mid / 2)
        pin = self._pin_at_py(mid)
        gpos = (pin[0], pin[1] + c.hang_y, pin[2] - c.hang_z)

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
        out["fixture"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Fixture",
            spawn=_fixture_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        # disc authored at mid-travel (joint zero); reset swings the linkage down
        out["disc"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Disc",
            spawn=_disc_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, c.hub_z), rot=(qw, 0.0, qy, 0.0)),
        )
        out["gondola"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Gondola",
            spawn=_gondola_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(pos=gpos),
        )
        out["bowl"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            spawn=_bowl_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.slot_a[0], c.slot_a[1], c.table_top + c.bowl_h / 2 + 0.002)),
        )
        out["plate"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            spawn=sim_utils.CylinderCfg(
                radius=c.plate_r, height=c.plate_h, axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.plate_mass),
                collision_props=tight,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.plate_color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.slot_b[0], c.slot_b[1], c.table_top + c.plate_h / 2 + 0.002)),
        )
        return out

    def sim_cfg(self) -> SimCfg:
        return SimCfg(
            dt=1.0 / 120.0,
            physx={
                "solver_type": 1,
                "enable_external_forces_every_iteration": True,
                "bounce_threshold_velocity": 0.2,
                "friction_offset_threshold": 0.01,
                "friction_correlation_distance": 0.00625,
                "gpu_max_rigid_contact_count": 2**23,
                "gpu_max_rigid_patch_count": 2**23,
                "gpu_collision_stack_size": 2**28,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle ------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.fixture: RigidObject = env.iscene["fixture"]
        self.disc: RigidObject = env.iscene["disc"]
        self.gondola: RigidObject = env.iscene["gondola"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.plate: RigidObject = env.iscene["plate"]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        # latched ride provenance (post_step)
        self._loaded_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._rode_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._hoisted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._delivered_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self._theta0 = torch.full((n,), math.radians(self.cfg.reset_lo_deg), device=dev)
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)

    def _author_joints(self) -> None:
        """Per env, two revolute joints (axis Y), pair collision disabled:
        1. wheel_hinge: fixture -> disc at the hub. The disc is AUTHORED at
           mid-travel (authored_deg), so localRot0 = qy(authored) and the limits are
           +-(travel/2) — far from the PhysX +-180 wrap boundary. World travel is
           [travel_lo_deg, travel_hi_deg].
        2. gondola_hinge: disc -> gondola at the rim pin, free (the pendulum).
           localRot0 = qy(-authored) so the joint zero is the gondola hanging level."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        mid = math.radians(c.authored_deg)
        half = (c.travel_hi_deg - c.travel_lo_deg) / 2.0
        qw, qy = math.cos(mid / 2), math.sin(mid / 2)
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/wheel_hinge")
            j.CreateBody0Rel().SetTargets([f"{base}/Fixture"])
            j.CreateBody1Rel().SetTargets([f"{base}/Disc"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, c.hub_z))
            j.CreateLocalRot0Attr(Gf.Quatf(qw, Gf.Vec3f(0.0, qy, 0.0)))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-half)
            j.CreateUpperLimitAttr(half)

            g = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/gondola_hinge")
            g.CreateBody0Rel().SetTargets([f"{base}/Disc"])
            g.CreateBody1Rel().SetTargets([f"{base}/Gondola"])
            g.CreateCollisionEnabledAttr(False)
            g.CreateAxisAttr("Y")
            g.CreateLocalPos0Attr(Gf.Vec3f(0.0, c.pin_y, -c.arm_r))
            g.CreateLocalRot0Attr(Gf.Quatf(qw, Gf.Vec3f(0.0, -qy, 0.0)))
            g.CreateLocalPos1Attr(Gf.Vec3f(0.0, -c.hang_y, c.hang_z))
            g.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # ----- wheel geometry -------------------------------------------------------------------------
    def _pin_at_py(self, theta: float) -> tuple:
        """Python-float pin world position (env-local) at wheel angle theta (rad)."""
        c = self.cfg
        return (-c.arm_r * math.sin(theta), c.pin_y, c.hub_z - c.arm_r * math.cos(theta))

    def wheel_angle(self) -> torch.Tensor:
        """(N,) wheel angle (rad), unwrapped around the authored mid-travel so the
        whole travel band [travel_lo, travel_hi] (which crosses 180 deg) is
        single-valued. The hinge admits only y-rotation, so the root quat is qy(th)."""
        q = self.disc.data.root_quat_w
        th = 2.0 * torch.atan2(q[:, 2], q[:, 0])
        mid = math.radians(self.cfg.authored_deg)
        d = torch.remainder(th - mid + math.pi, 2 * math.pi) - math.pi
        return mid + d

    def wheel_pose_at(self, theta: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(M,13) disc root state at wheel angle `theta` (zero velocity), world frame."""
        c = self.cfg
        st = torch.zeros(theta.shape[0], 13, device=theta.device)
        st[:, 2] = c.hub_z
        st[:, 3] = torch.cos(theta / 2)
        st[:, 5] = torch.sin(theta / 2)
        st[:, 0:3] += self.env_origins[env_ids]
        return st

    def gondola_pose_at(self, theta: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """(M,13) gondola root state (hanging level) at wheel angle `theta`."""
        c = self.cfg
        st = torch.zeros(theta.shape[0], 13, device=theta.device)
        st[:, 0] = -c.arm_r * torch.sin(theta)
        st[:, 1] = c.pin_y + c.hang_y
        st[:, 2] = c.hub_z - c.arm_r * torch.cos(theta) - c.hang_z
        st[:, 3] = 1.0
        st[:, 0:3] += self.env_origins[env_ids]
        return st

    def wheel_parked(self) -> torch.Tensor:
        return self.wheel_angle() >= math.radians(self.cfg.park_deg)

    def wheel_still(self) -> torch.Tensor:
        return self.disc.data.root_ang_vel_w.norm(dim=-1) < self.cfg.wheel_settle_w

    def _to_gondola_frame(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.gondola.data.root_quat_w,
                                  pos_w - self.gondola.data.root_pos_w)

    # ----- predicates -----------------------------------------------------------------------------
    def bowl_in_tray(self) -> torch.Tensor:
        """(N,) bool, geometric: bowl root inside the tray basin, gondola frame."""
        c = self.cfg
        loc = self._to_gondola_frame(self.bowl.data.root_pos_w)
        return ((loc[:, 0].abs() < c.tray_x_abs) & (loc[:, 1].abs() < c.tray_y_abs)
                & (loc[:, 2] > c.tray_z_lo) & (loc[:, 2] < c.tray_z_hi))

    def _upright(self, body) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        up = quat_apply(body.data.root_quat_w, ez)
        return up[:, 2].clamp(-1.0, 1.0) >= math.cos(math.radians(self.cfg.bowl_up_max_deg))

    def bowl_upright(self) -> torch.Tensor:
        return self._upright(self.bowl)

    def bowl_on_deck(self) -> torch.Tensor:
        """(N,) bool: bowl root inside the gallery, resting on the deck, upright
        (env frame; the deck is part of the kinematic fixture)."""
        c = self.cfg
        loc = self.bowl.data.root_pos_w - self.env_origins
        return ((loc[:, 0] > c.deck_x_lo) & (loc[:, 0] < c.deck_x_hi)
                & (loc[:, 1] > c.deck_y_lo) & (loc[:, 1] < c.deck_y_hi)
                & (loc[:, 2] > c.deck_z_lo) & (loc[:, 2] < c.deck_z_hi)
                & self.bowl_upright())

    def settled(self) -> torch.Tensor:
        """(N,) bool: bowl, plate, gondola still and the wheel not spinning."""
        c = self.cfg
        sb = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        sp = self.plate.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        sg = self.gondola.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return sb & sp & sg & self.wheel_still()

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        # reset grace: re-pin the whole freshly posed linkage while write timing settles
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            self.disc.write_root_state_to_sim(self.wheel_pose_at(self._theta0[gids], gids), gids)
            self.gondola.write_root_state_to_sim(
                self.gondola_pose_at(self._theta0[gids], gids), gids)
            self._grace[gids] -= 1

        # latch ride provenance, order-chained (all physical)
        c = self.cfg
        th = self.wheel_angle()
        aboard = self.bowl_in_tray()
        self._loaded_ever |= aboard & (th <= math.radians(c.load_max_deg))
        self._rode_ever |= (self._loaded_ever & aboard
                            & (th >= math.radians(c.ride_lo_deg))
                            & (th <= math.radians(c.ride_hi_deg)))
        self._hoisted_ever |= (self._rode_ever & aboard
                               & (th >= math.radians(c.park_deg)))
        self._delivered_ever |= self._hoisted_ever & self.bowl_on_deck()

    # ----- reset ----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        """Fixture stays put (jointed pair, never teleported). Sample: wheel start
        angle (falls to the bottom stop), bowl/plate staging-slot assignment, xy
        jitter, free yaw. The whole wheel+gondola linkage is re-posed consistently."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(m, device=dev)  # burn the degenerate first post-seed draw
        th0 = torch.deg2rad(c.reset_lo_deg + (c.reset_hi_deg - c.reset_lo_deg)
                            * torch.rand(m, device=dev))
        self._theta0[env_ids] = th0
        self.disc.write_root_state_to_sim(self.wheel_pose_at(th0, env_ids), env_ids)
        self.gondola.write_root_state_to_sim(self.gondola_pose_at(th0, env_ids), env_ids)

        # bowl and plate: sampled slot assignment, xy jitter, free yaw
        swap = torch.rand(m, device=dev) < 0.5
        sa = torch.tensor(c.slot_a, device=dev)
        sb = torch.tensor(c.slot_b, device=dev)
        bowl_xy = torch.where(swap.unsqueeze(1), sa.expand(m, 2), sb.expand(m, 2))
        plate_xy = torch.where(swap.unsqueeze(1), sb.expand(m, 2), sa.expand(m, 2))
        for body, xy, rest_z in (
                (self.bowl, bowl_xy, c.table_top + c.bowl_h / 2 + 0.002),
                (self.plate, plate_xy, c.table_top + c.plate_h / 2 + 0.002)):
            st = torch.zeros(m, 13, device=dev)
            st[:, :2] = xy + (torch.rand(m, 2, device=dev) * 2 - 1) * c.spawn_jitter
            st[:, 2] = rest_z
            half = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            st[:, 3] = torch.cos(half)
            st[:, 6] = torch.sin(half)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        self._loaded_ever[env_ids] = False
        self._rode_ever[env_ids] = False
        self._hoisted_ever[env_ids] = False
        self._delivered_ever[env_ids] = False
        self._grace[env_ids] = 2

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"disc": self.disc, "gondola": self.gondola,
                  "bowl": self.bowl, "plate": self.plate}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "theta0": self._theta0[env_ids].clone(),
            "latches": torch.stack([self._loaded_ever[env_ids], self._rode_ever[env_ids],
                                    self._hoisted_ever[env_ids],
                                    self._delivered_ever[env_ids]], dim=1).clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"disc": self.disc, "gondola": self.gondola,
                  "bowl": self.bowl, "plate": self.plate}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self._theta0[env_ids] = state["theta0"]
        lat = state["latches"]
        self._loaded_ever[env_ids] = lat[:, 0]
        self._rode_ever[env_ids] = lat[:, 1]
        self._hoisted_ever[env_ids] = lat[:, 2]
        self._delivered_ever[env_ids] = lat[:, 3]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            "A hand-cranked GONDOLA WHEEL (a crimson disc, 44 cm across, on a steel "
            "pylon) stands beside a brown cabinet. A wooden pendulum GONDOLA TRAY "
            "hangs from a brass pin on the wheel's rim; it swings freely and stays "
            "level as the wheel turns. Four steel pegs on the wheel's other face are "
            "the crank handles. The wheel rests on its bottom travel stop with the "
            "tray hanging low, and its upper stop is just past top-dead-center, so a "
            "loaded tray parks itself there by its own weight.\n"
            "The cabinet top is a ROOFED ROOFTOP GALLERY: a walled deck sealed by a "
            "solid roof slab — nothing can be lowered in from above. The only opening "
            f"is a narrow window ({0.15 * 100:.0f} cm wide) in the front wall, exactly "
            "where the gondola tray parks at the top of the ride. The tray's front "
            "edge is open toward the window.\n"
            f"On the staging table lie a BLACK BOWL (open cup, {2 * c.bowl_outer_r * 100:.0f} cm "
            f"wide, 11 mm rim) and a WHITE PLATE ({2 * c.plate_r * 100:.0f} cm, a distractor).\n"
            "Goal: the black bowl resting upright on the gallery deck, everything at "
            "rest. RULE (execution order, enforced by the score): the bowl must reach "
            "the gallery by RIDING THE GONDOLA — set aboard the tray at the bottom "
            "stop, aboard through the whole cranked arc, aboard when the wheel parks "
            "at the top — and only then slide through the window onto the deck. A bowl "
            "that arrives any other way (e.g. carried straight to the window) does not "
            "count. The plate is not the goal object."
        )

    def instruction(self) -> str:
        return (
            "Put the black bowl into the gondola tray while the wheel rests on its "
            "bottom stop, crank the wheel by its rim pegs up and over the top until it "
            "parks on its over-center stop beside the gallery window, then slide the "
            "bowl off the tray's open front edge through the window so it rests "
            "upright on the roofed gallery deck. The bowl must ride the gondola the "
            "whole way — the roof blocks the top, and a bowl hand-delivered through "
            "the window does not count. Leave the white plate alone."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1], latched order-chained stages of the demonstrated ride:
        0.15 bowl ever aboard at the bottom stop (loaded) + 0.15 aboard through
        mid-arc after loading (rode) + 0.25 aboard at the over-center park after
        riding (hoisted) + 0.40 on the gallery deck after the full ride (delivered);
        exactly 1.0 iff success(). Null policy ~0; hand-delivering the bowl to the
        deck without the ride earns nothing (order-chained latches)."""
        s = (0.15 * self._loaded_ever.float()
             + 0.15 * self._rode_ever.float()
             + 0.25 * self._hoisted_ever.float()
             + 0.40 * self._delivered_ever.float())
        return torch.where(self.success(),
                           torch.ones(self.env.num_envs, device=self.env.device),
                           s.clamp(0.0, 0.95))

    def success(self) -> torch.Tensor:
        """(N,) bool: bowl upright at rest on the gallery deck, everything settled,
        AND the ride provenance held (bowl was aboard at the park after the full
        ride) — the declared execution order."""
        return self.bowl_on_deck() & self.settled() & self._hoisted_ever


register_env("simgen", lambda: EnvCfg(scene="gondola_wheel", robot="null"))
