"""BeamBalanceScene — read the cargo's mass off its painted bands, pick the unique
counterweight subset, and load it into the opposite pocket until the beam levels
(sim_gen task `scene_c_i171`).

Derived from calvin/scene_C but STRATEGICALLY different: the CALVIN table is a
menu of independent binary single-DOF primitives (button, switch, slider,
drawer — each "done" at an end-stop) plus free blocks to pick or push on an
open tabletop. Here the only articulated DOF is a BALANCE BEAM on a knife-edge
pivot, and no end-stop is ever the goal: the beam starts slammed against a hard
stop by a CARGO slab riding one hanging pocket, and the goal is a CONTINUOUS
equilibrium — the beam standing level within a tolerance — reached only by
placing the correct subset of three candidate weight cubes into the OPPOSITE
pocket. The cargo's mass is banded on its paint job (each band = one mass unit)
and the three candidates are 1, 2 and 4 units, so exactly one subset of the
color-coded cubes balances it: the task is visual mass estimation + subset-sum
reasoning + placement, graded by a live physical readout (the beam angle), not
by any recorded trajectory.

No stored energy beyond gravity on the pendulum beam: the beam is a plain
damped revolute rotor with hard stops and an authored below-pivot CoM (its
restoring spring), the cubes and slabs free bodies — every outcome persists
hands-off.

Assets are fully procedural: tower (KINEMATIC: base + two pivot plates), beam
(DYNAMIC compound: arm, ballast keel, two hanging open-top pockets; explicit
MassAPI mass/CoM/inertia), five cargo slabs (one per band count, only one used
per episode, the rest parked in a far ground depot), three candidate cubes
(ivory 1u / orange 2u / charcoal 4u). The beam revolute (axis y, limits = hard
stops) is authored per-env at bind time; joint collision filtering applies to
the tower<->beam pair only, so every cube/slab contact — the load path — stays
live.

Per-episode randomization (readback-verified by smoke): cargo band count
k in 1..5, cargo side s in {+x, -x}, and the candidate cubes' ground scatter
(slot permutation + jitter + yaw).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.20  loaded  — cargo seated AND >=1 candidate cube in the counter pocket
  0.25  lifted  — loaded, all candidates legal, and the beam swung up off its
                  stop (|tilt| < lift_theta with the beam slow, sustained)
capped at 0.45; exactly 1.0 iff success(): beam level within theta_tol, cargo
seated on its start side, >=1 candidate in the counter pocket, every candidate
either in the counter pocket or on the ground, spare cargo slabs untouched in
the depot, everything settled and finite. Null policy ~0 (the beam rests on
its stop, >= stop angle, forever). Wrong-by-one-unit subsets settle at >= the
stop-side equilibrium, far outside theta_tol. Physics closes the prop exploit:
even a 3-cube ground stack under a pocket cannot hold the beam inside
lift_theta (asserted in cfg).

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

_G = 9.81


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


def _dyn_body(root, mass: float, lin_damp: float, ang_damp: float,
              com=None, inertia=None) -> None:
    """Standard dynamic compound body physics (32/4 iters, no sleep, damped).
    Custom spawners must author MassAPI explicitly (density-mass trap) — and
    mass alone leaves the CoM at the body origin, so CoM / diagonal inertia
    are authored here too when given."""
    from pxr import Gf, PhysxSchema, UsdPhysics

    UsdPhysics.RigidBodyAPI.Apply(root)
    mapi = UsdPhysics.MassAPI.Apply(root)
    mapi.CreateMassAttr(float(mass))
    if com is not None:
        mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    if inertia is not None:
        mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(float(lin_damp))
    prb.CreateAngularDampingAttr(float(ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)


def _spawn_tower(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the pivot tower: KINEMATIC compound. Local frame: pivot axis along
    y at (0, 0, pivot_z), z=0 ground. A base slab and two upright plates that
    flank the beam arm."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/base", x=(-c.base_half, c.base_half),
          y=(-c.base_half, c.base_half), z=(0.0, c.base_h), color=c.body_color, collide=collide)
    for tag, sgn in (("yp", 1.0), ("yn", -1.0)):
        _span(stage, f"{prim_path}/plate_{tag}", x=(-c.plate_hx, c.plate_hx),
              y=(min(sgn * c.plate_y0, sgn * c.plate_y1), max(sgn * c.plate_y0, sgn * c.plate_y1)),
              z=(c.base_h, c.plate_top), color=c.body_color, collide=collide)
    wood = _mk_material(prim_path, "wood", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the balance beam with root at the PIVOT (the revolute anchor):
    a long arm along x, a ballast keel hanging below the pivot (the visible
    counterpart of the authored below-pivot CoM = the restoring spring), and
    two hanging open-top pockets at x = +/-pock_x whose floors sit below the
    arm. Mass/CoM/inertia are authored explicitly."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/arm", x=(-c.arm_half, c.arm_half), y=(-c.arm_hy, c.arm_hy),
          z=(-c.arm_ht, c.arm_ht), color=c.arm_color, collide=collide)
    _span(stage, f"{prim_path}/keel", x=(-c.keel_hx, c.keel_hx), y=(-c.keel_hy, c.keel_hy),
          z=(c.keel_z0, c.keel_z1), color=c.keel_color, collide=collide)
    o = c.pock_a + c.pock_t
    for cx in (c.pock_x, -c.pock_x):
        p = f"{prim_path}/pock_{'p' if cx > 0 else 'n'}"
        _span(stage, f"{p}_floor", x=(cx - o, cx + o), y=(-o, o),
              z=(c.floor_bot, c.floor_top), color=c.pock_color, collide=collide)
        _span(stage, f"{p}_xp", x=(cx + c.pock_a, cx + o), y=(-o, o),
              z=(c.floor_top, c.wall_top), color=c.pock_color, collide=collide)
        _span(stage, f"{p}_xn", x=(cx - o, cx - c.pock_a), y=(-o, o),
              z=(c.floor_top, c.wall_top), color=c.pock_color, collide=collide)
        _span(stage, f"{p}_yp", x=(cx - c.pock_a, cx + c.pock_a), y=(c.pock_a, o),
              z=(c.floor_top, c.wall_top), color=c.pock_color, collide=collide)
        _span(stage, f"{p}_yn", x=(cx - c.pock_a, cx + c.pock_a), y=(-o, -c.pock_a),
              z=(c.floor_top, c.wall_top), color=c.pock_color, collide=collide)
    _dyn_body(root, c.mass, c.lin_damp, c.ang_damp,
              com=(0.0, 0.0, c.com_z), inertia=c.inertia)
    grippy = _mk_material(prim_path, "grippy", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, grippy)
    return root


def _spawn_cargo(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author one cargo slab with root at its geometric CENTRE: `bands` stacked
    box children with alternating paint (each band = one mass unit — the mass
    is readable off the paint job). Mass = bands * unit_mass, authored
    explicitly with the CoM at the origin and a cuboid diagonal inertia."""
    from isaaclab.sim.utils import bind_physics_material

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    h = c.bands * c.band_h
    half = c.slab_s / 2
    for b in range(c.bands):
        z0 = -h / 2 + b * c.band_h
        color = c.band_color_a if b % 2 == 0 else c.band_color_b
        _span(stage, f"{prim_path}/band_{b}", x=(-half, half), y=(-half, half),
              z=(z0, z0 + c.band_h), color=color, collide=collide)
    m = c.bands * c.unit_mass
    ixy = m * (c.slab_s**2 + h**2) / 12.0
    izz = m * (2 * c.slab_s**2) / 12.0
    _dyn_body(root, m, c.lin_damp, c.ang_damp, com=(0.0, 0.0, 0.0),
              inertia=(ixy, ixy, izz))
    mat = _mk_material(prim_path, "slab", c.mu_s, c.mu_d, "average")
    bind_physics_material(prim_path, mat)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "tower" not in _SPAWNER_CACHE:

        @configclass
        class TowerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tower)
            base_half: float = 0.12
            base_h: float = 0.02
            plate_hx: float = 0.035
            plate_y0: float = 0.020
            plate_y1: float = 0.038
            plate_top: float = 0.36
            body_color: tuple = (0.48, 0.36, 0.24)
            contact_offset: float = 0.0015
            mu_s: float = 0.60
            mu_d: float = 0.55

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            arm_half: float = 0.445
            arm_hy: float = 0.015
            arm_ht: float = 0.012
            keel_hx: float = 0.020
            keel_hy: float = 0.014
            keel_z0: float = -0.11
            keel_z1: float = -0.03
            pock_x: float = 0.40
            pock_a: float = 0.036
            pock_t: float = 0.008
            floor_top: float = -0.060
            floor_bot: float = -0.068
            wall_top: float = 0.015
            mass: float = 1.3
            com_z: float = -0.050
            inertia: tuple = (0.010, 0.060, 0.065)
            lin_damp: float = 0.0
            ang_damp: float = 8.0
            arm_color: tuple = (0.35, 0.45, 0.62)
            keel_color: tuple = (0.55, 0.15, 0.12)
            pock_color: tuple = (0.24, 0.26, 0.30)
            contact_offset: float = 0.0015
            mu_s: float = 0.60
            mu_d: float = 0.55

        @configclass
        class CargoSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cargo)
            bands: int = 1
            band_h: float = 0.017
            slab_s: float = 0.05
            unit_mass: float = 0.1
            band_color_a: tuple = (0.60, 0.16, 0.13)
            band_color_b: tuple = (0.86, 0.76, 0.55)
            lin_damp: float = 0.10
            ang_damp: float = 0.10
            contact_offset: float = 0.0015
            mu_s: float = 0.60
            mu_d: float = 0.55

        _SPAWNER_CACHE["tower"] = TowerSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
        _SPAWNER_CACHE["cargo"] = CargoSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class BeamBalanceSceneCfg(BaseCfg):
    """Config for `BeamBalanceScene`. The margin contract is asserted in
    `__post_init__`: the correct subset settles well inside theta_tol even at
    worst-case in-pocket placement play, every wrong-by-one subset settles far
    outside it, the null / prop / stack exploits land outside lift_theta, and
    ground cubes can never touch the beam at judge-relevant angles."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    theta_tol: float = tunable(0.262)     # |beam tilt| for success (rad) (~15 deg)
    lift_theta: float = tunable(0.28)     # |tilt| below this + slow -> "lifted" latch (rad)
    lift_w: float = tunable(0.35)         # beam |ang vel| gate for the lifted streak (rad/s)
    lift_streak: int = tunable(30)        # consecutive steps to latch "lifted"
    settle_w: float = tunable(0.10)       # beam |ang vel| when judging success (rad/s)
    settle_lin: float = tunable(0.05)     # max mover |lin vel| when judging (m/s)
    ground_z: float = tunable(0.12)       # cube centre below this = "on the ground" (m)
    pocket_dxy: float = tunable(0.022)    # in-pocket box half-extent about the pocket centre (m)
    pocket_dz_lo: float = tunable(-0.010)  # in-pocket z band, relative to the pocket floor (m)
    pocket_dz_hi: float = tunable(0.130)
    depot_half: float = tunable(0.50)     # spare-cargo depot half-extent about its centre (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    k_min: int = tunable(1)               # cargo band count band (inclusive)
    k_max: int = tunable(5)
    slot_jitter: float = tunable(0.03)    # candidate ground slot xy jitter (m)

    # --- info: tower / pivot (local frame: pivot axis || y at (0,0,pivot_z)) ---------------------
    pivot_z: float = info(0.34)
    plate_top: float = info(0.36)
    base_half: float = info(0.12)
    base_h: float = info(0.02)
    stop_deg: float = info(28.65)         # revolute hard stops (+/-, deg; 0.50 rad)
    tilt0: float = info(0.47)             # reset write angle (rad, just inside the stop)
    # --- info: beam --------------------------------------------------------------------------------
    arm_half: float = info(0.445)
    arm_hy: float = info(0.015)
    arm_ht: float = info(0.012)
    pock_x: float = info(0.40)            # pocket centre lever arm
    pock_a: float = info(0.036)           # pocket inner half-width
    pock_t: float = info(0.008)           # pocket wall thickness
    floor_top: float = info(-0.060)       # pocket floor top, rel pivot
    floor_bot: float = info(-0.068)
    wall_top: float = info(0.015)         # pocket wall top, rel pivot
    beam_mass: float = info(1.3)
    com_z: float = info(-0.050)           # authored CoM below the pivot (the spring)
    # --- info: cargo / candidates ------------------------------------------------------------------
    unit_mass: float = info(0.1)          # one band = one unit = 100 g
    band_h: float = info(0.017)
    slab_s: float = info(0.05)            # cargo slab footprint (square)
    cand_s: float = info(0.05)            # candidate cube edge
    cand_units: tuple = info((1, 2, 4))   # ivory / orange / charcoal
    cand_slots: tuple = info(((-0.18, -0.38), (0.0, -0.44), (0.18, -0.38)))
    depot_xy: tuple = info((2.1, 2.1))    # spare-cargo depot centre (far beyond arm reach)
    depot_slots: tuple = info(((1.9, 1.9), (2.1, 1.9), (2.3, 1.9), (1.9, 2.2), (2.1, 2.2)))
    contact_offset: float = info(0.0015)
    # --- info: rubric weights (0.20 + 0.25 = 0.45 = the non-success cap) -------------------------
    w_loaded: float = info(0.20)
    w_lifted: float = info(0.25)

    def __post_init__(self) -> None:
        stop = math.radians(self.stop_deg)
        k_beam = self.beam_mass * _G * (-self.com_z)          # restoring stiffness (N m / rad)
        tau_unit = self.unit_mass * _G * self.pock_x          # torque of one mass unit
        # -- angle ladder: tol < lift < 3-stack prop < wrong-by-one < stop
        assert self.theta_tol < self.lift_theta < stop, "tol < lift < stop"
        # -- correct subset, worst-case placement play on BOTH sides, max total load (k=5):
        #    play = pock_a - cand_s/2 per body; loads below the pivot only ADD stiffness,
        #    so k_beam alone is the conservative denominator.
        play = self.pock_a - self.cand_s / 2
        assert play >= 0.008, "pocket must pass the cube with real clearance"
        tau_err = 2 * (self.k_max * self.unit_mass) * _G * play
        assert tau_err / k_beam <= 0.80 * self.theta_tol, \
            "worst-case placement play must keep the correct subset well inside tol"
        # -- wrong-by-one-unit: even with the max extra load stiffness (all loads as low
        #    as they can ride in the pockets), the equilibrium sits far outside tol.
        z_load_min = abs(self.floor_top + self.cand_s / 2)    # lowest-riding load CoM |z|
        k_load_max = 2 * (self.k_max * self.unit_mass) * _G * z_load_min
        theta_wrong = min(tau_unit / (k_beam + k_load_max), stop)
        assert theta_wrong >= self.theta_tol + 0.10, \
            "wrong-by-one must settle far outside theta_tol"
        # -- null policy: the un-countered cargo (even k=1) pins the beam at/past the stop
        assert tau_unit * self.k_min / k_beam >= stop, "null must rest on the hard stop"
        assert stop >= self.lift_theta + 0.15
        # -- prop exploit: a 3-candidate ground stack under a pocket holds the beam at
        #    an angle STILL outside lift_theta (and a fortiori outside theta_tol)
        drop_3stack = self.pivot_z + self.floor_bot - 3 * self.cand_s
        theta_prop = math.asin(drop_3stack / self.pock_x)
        assert theta_prop >= self.lift_theta + 0.02, "3-stack prop must stay outside lift_theta"
        # -- ground cubes never touch the beam at judge-relevant angles: the beam's
        #    lowest point (pocket floor bottom) stays above a single cube even at the stop
        low_at_stop = self.pivot_z + self.floor_bot - self.pock_x * math.sin(stop)
        assert low_at_stop >= self.cand_s + 0.020, "beam must clear a ground cube at the stop"
        # -- in-pocket cubes never read as "on the ground" while judging (|tilt| <= lift):
        z_pock_cube = self.pivot_z + self.floor_top + self.cand_s / 2 \
            - self.pock_x * math.sin(self.lift_theta)
        assert z_pock_cube >= self.ground_z + 0.04, "pocket cubes must sit above ground_z"
        # -- pocket geometry: floors/walls sane, pocket sweep clears the tower plates
        assert self.floor_bot < self.floor_top < self.wall_top
        assert self.wall_top - self.floor_top >= self.cand_s + 0.020, \
            "walls overtop a seated cube"
        assert self.pock_x - (self.pock_a + self.pock_t) > 0.12, "pockets clear of the tower"
        assert self.pock_x + self.pock_a + self.pock_t <= self.arm_half
        # -- the tallest cargo (k_max bands) still loads open-top and stays wall-braced:
        #    the wall contacts it above its CoM, so it cannot topple out at the stop
        h_max = self.k_max * self.band_h
        assert self.floor_top + h_max / 2 < self.wall_top, "walls brace the cargo above CoM"
        # -- a two-cube stack in a pocket stays statically topple-free at theta_tol
        assert math.tan(self.theta_tol) * (1.5 * self.cand_s) <= self.cand_s / 2 - 0.004
        # -- in-pocket box test is honest: any physically-seated cube passes it
        assert self.pocket_dxy >= play + 0.005
        assert self.pocket_dz_hi >= 1.5 * self.cand_s + 0.02   # stacked top cube
        assert self.pocket_dz_lo <= 0.0
        # -- reset write angle sits inside the stop (no limit-violation write)
        assert self.tilt0 < stop
        # -- depot far beyond a fixed-base arm's reach; slots clear of the play area
        for sx, sy in self.depot_slots:
            assert math.hypot(sx, sy) >= 1.5
            assert abs(sx - self.depot_xy[0]) < self.depot_half - 0.05
            assert abs(sy - self.depot_xy[1]) < self.depot_half - 0.05
        # -- candidate slots: on the near side, clear of the beam's vertical sweep plane
        for sx, sy in self.cand_slots:
            assert sy <= -0.30 and abs(sx) <= 0.30
            assert abs(sy) - self.slot_jitter > self.pock_a + self.pock_t + 0.05
        # -- Franka jaw feasibility (~80 mm parallel jaw)
        assert self.cand_s <= 0.075 and self.slab_s <= 0.075
        # -- the unique-subset arithmetic really covers the k band
        units = sorted(self.cand_units)
        sums = {0}
        for u in units:
            sums |= {s + u for s in sums}
        for k in range(self.k_min, self.k_max + 1):
            assert k in sums, "every cargo count must have a candidate subset"
        assert len(sums) == 2 ** len(units), "subset sums must be unique (powers of two)"
        # -- rubric weights
        assert abs(self.w_loaded + self.w_lifted - 0.45) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("beam_balance")
class BeamBalanceScene(BaseScene):
    cfg: BeamBalanceSceneCfg

    def __init__(self, cfg: BeamBalanceSceneCfg | None = None) -> None:
        super().__init__(cfg or BeamBalanceSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        tower_spawn = cls["tower"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            plate_top=c.plate_top, base_half=c.base_half, base_h=c.base_h,
            contact_offset=c.contact_offset)
        beam_spawn = cls["beam"](
            arm_half=c.arm_half, arm_hy=c.arm_hy, arm_ht=c.arm_ht, pock_x=c.pock_x,
            pock_a=c.pock_a, pock_t=c.pock_t, floor_top=c.floor_top, floor_bot=c.floor_bot,
            wall_top=c.wall_top, mass=c.beam_mass, com_z=c.com_z,
            contact_offset=c.contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.10, angular_damping=0.10,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.contact_offset, rest_offset=0.0)
        mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.60, dynamic_friction=0.55, restitution=0.0)

        cand_colors = ((0.92, 0.90, 0.80), (0.90, 0.50, 0.10), (0.15, 0.15, 0.17))

        def cand(units: int, color) -> Any:
            return sim_utils.CuboidCfg(
                size=(c.cand_s, c.cand_s, c.cand_s),
                mass_props=sim_utils.MassPropertiesCfg(mass=units * c.unit_mass),
                rigid_props=rigid, collision_props=coll, physics_material=mat,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "tower": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tower", spawn=tower_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # the beam is authored at the joint's zero pose (level): the bind-time
            # revolute anchors at this authored position.
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam", spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.pivot_z))),
        }
        for i, (u, col) in enumerate(zip(c.cand_units, cand_colors)):
            sx, sy = c.cand_slots[i]
            out[f"cand_{u}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cand{u}", spawn=cand(u, col),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(sx, sy, c.cand_s / 2 + 0.002)))
        for k in range(1, c.k_max + 1):
            spawn = cls["cargo"](bands=k, band_h=c.band_h, slab_s=c.slab_s,
                                 unit_mass=c.unit_mass, contact_offset=c.contact_offset)
            dx, dy = c.depot_slots[k - 1]
            out[f"cargo_{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Cargo{k}", spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(dx, dy, k * c.band_h / 2 + 0.002)))
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
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        self._quat_apply = quat_apply
        self._quat_apply_inv = quat_apply_inverse
        c = self.cfg
        self.tower: RigidObject = env.iscene["tower"]
        self.beam: RigidObject = env.iscene["beam"]
        self.cands: list[RigidObject] = [env.iscene[f"cand_{u}"] for u in c.cand_units]
        self.cargos: list[RigidObject] = [env.iscene[f"cargo_{k}"]
                                          for k in range(1, c.k_max + 1)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke): [k, side, cand xy x3]
        self.layout = torch.zeros(n, 8, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._loaded_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lifted_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lift_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env revolute, authored ONCE at bind time against the AUTHORED
        (level) pose: axis y through (0, 0, pivot_z), hard stops at
        +/-stop_deg. Collision filtering disables only the tower<->beam pair,
        so every cube/slab contact — the load path — still collides."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/beam_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Tower"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.pivot_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.stop_deg))
            j.CreateUpperLimitAttr(float(c.stop_deg))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: tower re-asserted at its fixed pose; a SAMPLED cargo
        (k bands) seated in a SAMPLED side's pocket with the beam written
        tilted onto that side just inside its stop (the whole linkage written
        together); the four spare cargos parked in the far depot; the three
        candidate cubes scattered on SAMPLED front-ground slots; latches
        cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        u = torch.rand(m, 4, device=dev)
        nk = c.k_max - c.k_min + 1
        k = (c.k_min + (u[:, 0] * nk).long().clamp(max=nk - 1)).float()   # bands, 1..5
        side = torch.where(u[:, 1] < 0.5, -1.0, torch.ones(m, device=dev))
        # beam tilt: cargo side goes DOWN. rot +a about y drops the +x end when a > 0
        # in our tilt convention (see beam_tilt); write angle a = side * tilt0.
        a = side * c.tilt0
        perm = torch.argsort(torch.rand(m, 3, device=dev), dim=1)          # slot permutation
        jit = (torch.rand(m, 3, 2, device=dev) * 2 - 1) * c.slot_jitter
        yaw = torch.rand(m, 3, device=dev) * (2 * math.pi)
        slots = torch.tensor(c.cand_slots, device=dev)                     # (3, 2)
        cand_xy = slots[perm] + jit                                        # (m, 3, 2)

        self.layout[env_ids, 0] = k
        self.layout[env_ids, 1] = side
        self.layout[env_ids, 2:8] = cand_xy.reshape(m, 6)

        def write(body, dx, dy, dz, quat=None, yaw_ang=None) -> None:
            s = torch.zeros(m, 13, device=dev)
            s[:, 0] = origin[:, 0] + dx
            s[:, 1] = origin[:, 1] + dy
            s[:, 2] = origin[:, 2] + dz
            if quat is not None:
                s[:, 3:7] = quat
            elif yaw_ang is not None:
                s[:, 3] = torch.cos(yaw_ang / 2)
                s[:, 6] = torch.sin(yaw_ang / 2)
            else:
                s[:, 3] = 1.0
            body.write_root_state_to_sim(s, env_ids)

        zeros = torch.zeros(m, device=dev)
        write(self.tower, zeros, zeros, zeros)
        # beam: rotation a about +y at the pivot
        bq = torch.zeros(m, 4, device=dev)
        bq[:, 0] = torch.cos(a / 2)
        bq[:, 2] = torch.sin(a / 2)
        write(self.beam, zeros, zeros, zeros + c.pivot_z, quat=bq)
        # cargos: the sampled one seated in the side pocket (rotated with the beam:
        # local (side*pock_x, 0, floor_top + h/2 + gap) about y by a), spares in depot
        ca, sa = torch.cos(a), torch.sin(a)
        for j, cargo in enumerate(self.cargos, start=1):
            h = j * c.band_h
            lx = side * c.pock_x
            lz = c.floor_top + h / 2 + 0.0005
            px = lx * ca + lz * sa
            pz = -lx * sa + lz * ca + c.pivot_z
            dx0, dy0 = c.depot_slots[j - 1]
            sel = k == float(j)
            write(cargo,
                  torch.where(sel, px, zeros + dx0),
                  torch.where(sel, zeros, zeros + dy0),
                  torch.where(sel, pz, zeros + h / 2 + 0.002),
                  quat=torch.where(sel.unsqueeze(-1), bq,
                                   torch.tensor([1.0, 0, 0, 0], device=dev).expand(m, 4)))
        for i, cand in enumerate(self.cands):
            write(cand, cand_xy[:, i, 0], cand_xy[:, i, 1],
                  zeros + c.cand_s / 2 + 0.002, yaw_ang=yaw[:, i])

        self._loaded_l[env_ids] = False
        self._lifted_l[env_ids] = False
        self._lift_streak[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "tower": self.tower.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "layout": self.layout[env_ids].clone(),
            "loaded_l": self._loaded_l[env_ids].clone(),
            "lifted_l": self._lifted_l[env_ids].clone(),
            "lift_streak": self._lift_streak[env_ids].clone(),
        }
        for u, cand in zip(self.cfg.cand_units, self.cands):
            out[f"cand_{u}"] = cand.data.root_state_w[env_ids].clone()
        for j, cargo in enumerate(self.cargos, start=1):
            out[f"cargo_{j}"] = cargo.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.tower.write_root_state_to_sim(state["tower"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for u, cand in zip(self.cfg.cand_units, self.cands):
            cand.write_root_state_to_sim(state[f"cand_{u}"], env_ids)
        for j, cargo in enumerate(self.cargos, start=1):
            cargo.write_root_state_to_sim(state[f"cargo_{j}"], env_ids)
        self.layout[env_ids] = state["layout"]
        self._loaded_l[env_ids] = state["loaded_l"]
        self._lifted_l[env_ids] = state["lifted_l"]
        self._lift_streak[env_ids] = state["lift_streak"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        stop = math.radians(self.cfg.stop_deg)
        return (
            f"A BALANCE SCALE stands on the ground: a wooden pivot tower "
            f"({c.plate_top * 100:.0f} cm tall) carries a steel-blue BEAM "
            f"({2 * c.arm_half * 100:.0f} cm long) on a free hinge with hard stops at "
            f"+/-{c.stop_deg:.0f} degrees. A dark red ballast keel under the hinge makes the "
            f"beam self-level when its two hanging POCKETS (open-top boxes at "
            f"{c.pock_x * 100:.0f} cm on each side) carry equal weight.\n"
            f"One pocket starts holding a painted CARGO slab, so the beam starts slammed "
            f"down on that side against its stop. The cargo's mass is written on its paint "
            f"job: it is striped in horizontal bands (alternating dark red / tan), and each "
            f"band weighs exactly {c.unit_mass * 1000:.0f} g — count the bands to know the "
            f"mass (1 to {c.k_max} bands; which slab, and which side it rides, varies by "
            f"episode).\n"
            f"Three CANDIDATE weight cubes ({c.cand_s * 100:.0f} cm) lie scattered on the "
            f"ground in front of the scale (positions vary by episode): the IVORY cube "
            f"weighs {c.cand_units[0] * c.unit_mass * 1000:.0f} g, the ORANGE cube "
            f"{c.cand_units[1] * c.unit_mass * 1000:.0f} g, and the CHARCOAL cube "
            f"{c.cand_units[2] * c.unit_mass * 1000:.0f} g. Exactly one combination of them "
            f"matches the cargo's mass.\n"
            f"Goal: place that combination into the EMPTY pocket (opposite the cargo) so "
            f"the beam swings up and settles LEVEL (within about "
            f"{math.degrees(c.theta_tol):.0f} degrees), with the cargo still seated in its "
            f"pocket, and hold nothing. Cubes you do not use must stay on the ground. "
            f"Being off by even one band ({c.unit_mass * 1000:.0f} g) leaves the beam "
            f"visibly tilted ({math.degrees(stop):.0f}-degree stops) — the beam angle is "
            f"the only judge. Do not pile cubes under the beam or drape them on the arm: "
            f"only weight riding IN the empty pocket (or left on the ground) counts as "
            f"legal. Far behind the scale, spare cargo slabs are parked in a depot — "
            f"leave them there."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Count the paint bands on the cargo slab riding one pocket of the balance "
            "beam — each band is 100 g. Pick the combination of the ivory (100 g), "
            "orange (200 g) and charcoal (400 g) cubes that matches, place those cubes "
            "in the empty pocket on the other side, and leave the rest on the ground, "
            "so the beam settles level."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the scene frame (tower fixed at the origin)."""
        return body.data.root_pos_w - self.env_origins

    def beam_tilt(self) -> torch.Tensor:
        """(N,) beam tilt about y (rad; 0 = level). Positive tilt = +x end DOWN."""
        ex = self._quat_apply(self.beam.data.root_quat_w,
                              torch.tensor([1.0, 0.0, 0.0], device=self.env.device
                                           ).expand(self.env.num_envs, 3))
        return torch.atan2(-ex[:, 2], ex[:, 0])

    def _in_pocket(self, body, side: torch.Tensor) -> torch.Tensor:
        """(N,) bool: body centre inside the pocket on `side` (+1 = +x pocket),
        in the BEAM frame (a generous box any physically seated cube passes)."""
        c = self.cfg
        rel = body.data.root_pos_w - self.beam.data.root_pos_w
        lp = self._quat_apply_inv(self.beam.data.root_quat_w, rel)
        dz = lp[:, 2] - c.floor_top
        return ((lp[:, 0] - side * c.pock_x).abs() < c.pocket_dxy) \
            & (lp[:, 1].abs() < c.pocket_dxy) \
            & (dz > c.pocket_dz_lo) & (dz < c.pocket_dz_hi)

    def _on_ground(self, body) -> torch.Tensor:
        return self._local(body)[:, 2] < self.cfg.ground_z

    def cargo_side(self) -> torch.Tensor:
        """(N,) the sampled cargo side (+1 / -1) from the layout."""
        return self.layout[:, 1]

    def cargo_seated(self) -> torch.Tensor:
        """(N,) bool: the episode's sampled cargo slab rides its start-side pocket."""
        k = self.layout[:, 0]
        side = self.cargo_side()
        ok = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for j, cargo in enumerate(self.cargos, start=1):
            ok |= (k == float(j)) & self._in_pocket(cargo, side)
        return ok

    def counter_loaded(self) -> torch.Tensor:
        """(N,) bool: at least one candidate cube rides the counter pocket."""
        side = -self.cargo_side()
        out = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for cand in self.cands:
            out |= self._in_pocket(cand, side)
        return out

    def cands_legal(self) -> torch.Tensor:
        """(N,) bool: every candidate is either in the COUNTER pocket or on the
        ground — no draping on the arm, no perching on the tower, no riding the
        cargo pocket."""
        side = -self.cargo_side()
        out = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for cand in self.cands:
            out &= self._in_pocket(cand, side) | self._on_ground(cand)
        return out

    def spares_in_depot(self) -> torch.Tensor:
        """(N,) bool: every NON-sampled cargo slab still sits in the far depot."""
        c = self.cfg
        k = self.layout[:, 0]
        dx = torch.tensor(c.depot_xy, device=self.env.device)
        out = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for j, cargo in enumerate(self.cargos, start=1):
            p = self._local(cargo)
            in_dep = ((p[:, :2] - dx).abs() < c.depot_half).all(dim=-1) \
                & (p[:, 2] < c.ground_z)
            out &= (k == float(j)) | in_dep
        return out

    def level(self) -> torch.Tensor:
        return self.beam_tilt().abs() < self.cfg.theta_tol

    def settled(self) -> torch.Tensor:
        """(N,) bool: beam rotation slow, every cube/slab slow."""
        c = self.cfg
        ok = self.beam.data.root_ang_vel_w.norm(dim=-1) < c.settle_w
        for body in (*self.cands, *self.cargos):
            ok &= body.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        return ok

    def _finite(self) -> torch.Tensor:
        ps = [self.beam.data.root_pos_w] + [b.data.root_pos_w for b in self.cands] \
            + [b.data.root_pos_w for b in self.cargos]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        loaded = self.cargo_seated() & self.counter_loaded() & fin
        self._loaded_l |= loaded
        lift_now = loaded & self.cands_legal() \
            & (self.beam_tilt().abs() < c.lift_theta) \
            & (self.beam.data.root_ang_vel_w.norm(dim=-1) < c.lift_w)
        self._lift_streak = torch.where(lift_now, self._lift_streak + 1,
                                        torch.zeros_like(self._lift_streak))
        self._lifted_l |= self._lift_streak >= c.lift_streak

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the beam stands LEVEL (within theta_tol) with the cargo
        seated on its start side, at least one candidate riding the counter
        pocket, every candidate legally placed (counter pocket or ground),
        the spare slabs untouched in the depot, everything settled and finite
        — a LIVE physical outcome. The only mass allowed on the beam is the
        cargo plus counter-pocket candidates, so a level beam means the
        candidate subset really balances the cargo."""
        self._update_latches()
        return self.cargo_seated() & self.counter_loaded() & self.cands_legal() \
            & self.spares_in_depot() & self.level() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.20 loaded + 0.25 lifted, latched, capped at
        0.45; exactly 1.0 iff success() holds live. Doing nothing scores ~0
        (the beam never leaves its stop); the seed's pick-a-block reflex tops
        out at 0.20 (a cube dropped in the pocket) unless the subset really
        balances."""
        c = self.cfg
        self._update_latches()
        base = (c.w_loaded * self._loaded_l.float()
                + c.w_lifted * self._lifted_l.float()).clamp(max=0.45)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="beam_balance", robot="null"))
