"""WeighStationScene — deliver a parcel onto the cradle of a counterweighed balance
beam and leave the scale reading LEVEL (sim_gen task `handover_i124`).

Derived from mujoco_playground/handover, but STRATEGICALLY different: the seed's whole
skill is a direct free-space transfer — one gripper hands a cube to the other, which
carries it to a floating target point; success is a pose match reached by carrying.
Here carrying the parcel to its destination is the TRIVIAL part and by itself earns
almost nothing: the destination is the CRADLE of a pivoting balance beam, and the
episode only succeeds if the beam ends LEVEL. An unballasted scale keels over under
the parcel (the seed's plan, executed verbatim, ends at the ±20° hard stop — smoke
proves it with a settled construct). The real skill is WEIGHING: the parcel's size
announces its mass (all parcels share one density — 46/58/66 mm cube = 1/2/3 mass
units), and the solver must seat exactly that many 0.15 kg unit counterweights in the
rack on the OPPOSITE arm of the beam. One unit too few or too many tilts the beam
~14–20°, far outside the 6° level tolerance. Extra counterweights are distractors:
there are always four on the apron, never exactly the number needed, and any weight
riding the beam outside the rack (or a spare parcel smuggled aboard as ballast) fails
the cleanliness clause. No execution order is required — ballast-first and
parcel-first both reach equilibrium; only the settled end state is judged.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - station (heavy DYNAMIC compound — a kinematic root would orphan the beam joint
    anchor when reset teleports it): ground slab (counter), centre pillar with a
    two-prong gimbal fork, a green shipment pad;
  - beam (DYNAMIC compound on an authored Y-axis revolute joint with ±20° limits —
    the hard stops): crossbar, RED walled cradle on the +x arm, BLUE 3-pocket
    counterweight rack on the -x arm, and a hanging two-plate yoke bob. Mass, CoM
    (0, 0, -0.095) and diagonal inertia are AUTHORED so the pendulum restoring
    stiffness K = M g |z_com| = 1.49 N·m/rad is auditable: a one-unit imbalance
    (0.368 N·m at the 0.25 m arm) tilts atan(0.368/1.49) ≈ 14° ≥ 2x the 6° level
    tolerance, while worst-case placement slop inside the pockets stays < 3.5°.
    The hinge is a plain PhysX revolute (frictionless outside articulations);
    settling comes from the beam body's angular damping.
  - four identical 40 mm / 0.15 kg steel-blue counterweight cubes on the apron;
  - three GREEN parcels (46/58/66 mm, 0.15/0.30/0.45 kg — one shared density, so
    size IS mass); exactly ONE (the shipment) stands on the green pad per episode,
    the other two park in an off-station ground depot.

Per-episode randomization (readback-verified by smoke): station yaw FREE (±180°) +
xy jitter, WHICH parcel ships (k ∈ {1,2,3} mass units), shipment jitter + yaw,
counterweight jitter + yaw. All discrete draws derive from torch.rand (the first
torch.randint after manual_seed is degenerate on this stack).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.30  ballast — latched max fraction of the REQUIRED ballast ever seated in the
                  rack (min(seated, k)/k — over-ballasting earns no extra)
  0.30  parcel  — the shipment ever seated in the cradle (latched)
capped at 0.60; exactly 1.0 iff success(): shipment seated in the cradle, beam
within 6° of level, exactly k weights in the rack, nothing else riding the beam,
everything settled and finite. Null policy ~0 (nothing aboard, beam level but empty).
The seed's strategy (carry the parcel to the destination) latches 0.30 and keels the
beam onto its stop — never 1.0.

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


def _author_mass(root, mass: float, com, inertia) -> None:
    """Explicit MassAPI mass + CoM + diagonal inertia. On this stack, MassAPI mass on
    a compound root leaves the CoM at the body ORIGIN and the shape-derived inertia
    is unknown — author all three so the pendulum stiffness K = M g |z_com| and the
    swing dynamics are auditable."""
    from pxr import Gf, UsdPhysics

    api = UsdPhysics.MassAPI.Apply(root)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode="average"))
    return mat_path


def _spawn_station(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the weigh station base at `prim_path`: heavy DYNAMIC compound. Local
    frame: origin at the slab TOP centre (z=0); the pillar rises to the gimbal fork;
    +x is the cradle (delivery) side, -x the rack (ballast) side; the green shipment
    pad sits on the +x apron."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    # --- slab (the counter everything stands on) --------------------------------------
    _span(stage, f"{prim_path}/slab", x=(-c.slab_hx, c.slab_hx), y=(-c.slab_hy, c.slab_hy),
          z=(-c.slab_t, 0.0), color=c.slab_color, collide=collide)
    # --- centre pillar (top BELOW the bar's swing envelope) ---------------------------
    _span(stage, f"{prim_path}/pillar", x=(-c.pillar_h2, c.pillar_h2),
          y=(-c.pillar_h2, c.pillar_h2), z=(0.0, c.pillar_z1),
          color=c.pillar_color, collide=collide)
    # --- gimbal fork: two prong plates flanking the bar (never touching it) -----------
    for tag, sgn in (("p", 1.0), ("n", -1.0)):
        y0, y1 = sorted((sgn * c.prong_y0, sgn * c.prong_y1))
        _span(stage, f"{prim_path}/prong_{tag}", x=(-c.prong_hx, c.prong_hx),
              y=(y0, y1), z=(c.prong_z0, c.prong_z1),
              color=c.pillar_color, collide=collide)
    # --- green shipment pad (where the parcel to weigh stands) ------------------------
    px, py = c.pad_center
    _span(stage, f"{prim_path}/pad", x=(px - c.pad_half, px + c.pad_half),
          y=(py - c.pad_half, py + c.pad_half), z=(0.0, c.pad_t),
          color=c.pad_color, collide=collide)

    # --- dynamic root (NOT kinematic: the beam joint anchor must follow teleports) ----
    UsdPhysics.RigidBodyAPI.Apply(root)
    _author_mass(root, c.station_mass, (0.0, 0.0, -0.02), c.station_inertia)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.5)
    prb.CreateAngularDampingAttr(2.0)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    bind_physics_material(prim_path, _mk_material(prim_path, "base", c.base_mu_s, c.base_mu_d))
    return root


def _spawn_beam(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the balance beam at `prim_path`: DYNAMIC compound, origin AT the hinge
    axis (station-local (0, 0, hinge_z)). Crossbar along local x; RED walled cradle
    centred at +arm_r; BLUE rack (three open-top pockets in a tangential row, ALL at
    -arm_r so pocket choice never changes the moment arm) at -x; two-plate yoke bob
    hanging below the pivot (the visual carrier of the authored low CoM)."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    r = c.arm_r
    f0, f1, w1 = c.floor_z0, c.floor_z1, c.wall_z1

    # --- crossbar ---------------------------------------------------------------------
    _span(stage, f"{prim_path}/bar", x=(-c.bar_hx, c.bar_hx), y=(-c.bar_hy, c.bar_hy),
          z=(-c.bar_hz, c.bar_hz), color=c.bar_color, collide=collide)

    # --- cradle (+x arm): floor + 4 walls, interior cradle_in square ------------------
    ch = c.cradle_in / 2 + c.cradle_t  # outer half-extent
    _span(stage, f"{prim_path}/cradle_floor", x=(r - ch, r + ch), y=(-ch, ch),
          z=(f0, f1), color=c.cradle_color, collide=collide)
    _span(stage, f"{prim_path}/cradle_xn", x=(r - ch, r - ch + c.cradle_t), y=(-ch, ch),
          z=(f1, w1), color=c.cradle_color, collide=collide)
    _span(stage, f"{prim_path}/cradle_xp", x=(r + ch - c.cradle_t, r + ch), y=(-ch, ch),
          z=(f1, w1), color=c.cradle_color, collide=collide)
    _span(stage, f"{prim_path}/cradle_yn", x=(r - ch + c.cradle_t, r + ch - c.cradle_t),
          y=(-ch, -ch + c.cradle_t), z=(f1, w1), color=c.cradle_color, collide=collide)
    _span(stage, f"{prim_path}/cradle_yp", x=(r - ch + c.cradle_t, r + ch - c.cradle_t),
          y=(ch - c.cradle_t, ch), z=(f1, w1), color=c.cradle_color, collide=collide)

    # --- rack (-x arm): floor, thick x walls (pocket x-interior = rack_in), outer y
    #     walls and two dividers -> three pockets in a row along y, all at x = -arm_r -
    rx0, rx1 = -r - c.rack_hx, -r + c.rack_hx
    ry = c.rack_hy
    _span(stage, f"{prim_path}/rack_floor", x=(rx0, rx1), y=(-ry, ry),
          z=(f0, f1), color=c.rack_color, collide=collide)
    xw = (2 * c.rack_hx - c.rack_in) / 2  # thick x-wall
    _span(stage, f"{prim_path}/rack_xn", x=(rx0, rx0 + xw), y=(-ry, ry),
          z=(f1, w1), color=c.rack_color, collide=collide)
    _span(stage, f"{prim_path}/rack_xp", x=(rx1 - xw, rx1), y=(-ry, ry),
          z=(f1, w1), color=c.rack_color, collide=collide)
    _span(stage, f"{prim_path}/rack_yn", x=(rx0 + xw, rx1 - xw), y=(-ry, -ry + c.rack_t),
          z=(f1, w1), color=c.rack_color, collide=collide)
    _span(stage, f"{prim_path}/rack_yp", x=(rx0 + xw, rx1 - xw), y=(ry - c.rack_t, ry),
          z=(f1, w1), color=c.rack_color, collide=collide)
    for tag, yc in (("a", -c.div_y), ("b", c.div_y)):
        _span(stage, f"{prim_path}/rack_div_{tag}", x=(rx0 + xw, rx1 - xw),
              y=(yc - c.div_t / 2, yc + c.div_t / 2), z=(f1, w1),
              color=c.rack_color, collide=collide)

    # --- yoke bob (two plates either side of the pillar; the low-CoM carrier) ---------
    for tag, sgn in (("p", 1.0), ("n", -1.0)):
        y0, y1 = sorted((sgn * c.bob_y0, sgn * c.bob_y1))
        _span(stage, f"{prim_path}/bob_{tag}", x=(-c.bob_hx, c.bob_hx), y=(y0, y1),
              z=(c.bob_z0, c.bob_z1), color=c.bob_color, collide=collide)

    UsdPhysics.RigidBodyAPI.Apply(root)
    _author_mass(root, c.beam_mass, (0.0, 0.0, c.beam_com_z), c.beam_inertia)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateAngularDampingAttr(cfg.beam_ang_damping)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    bind_physics_material(prim_path, _mk_material(prim_path, "grip", c.beam_mu_s, c.beam_mu_d))
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
            slab_hx: float = 0.35
            slab_hy: float = 0.24
            slab_t: float = 0.03
            pillar_h2: float = 0.025
            pillar_z1: float = 0.230
            prong_hx: float = 0.020
            prong_y0: float = 0.025
            prong_y1: float = 0.039
            prong_z0: float = 0.22
            prong_z1: float = 0.29
            pad_center: tuple = (0.18, -0.16)
            pad_half: float = 0.05
            pad_t: float = 0.002
            station_mass: float = 30.0
            station_inertia: tuple = (1.0, 1.0, 1.8)
            slab_color: tuple = (0.42, 0.40, 0.36)
            pillar_color: tuple = (0.30, 0.32, 0.38)
            pad_color: tuple = (0.15, 0.60, 0.20)
            contact_offset: float = 0.0015
            base_mu_s: float = 0.50
            base_mu_d: float = 0.45

        @configclass
        class BeamSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_beam)
            arm_r: float = 0.25
            bar_hx: float = 0.30
            bar_hy: float = 0.015
            bar_hz: float = 0.010
            floor_z0: float = 0.010
            floor_z1: float = 0.016
            wall_z1: float = 0.048
            cradle_in: float = 0.076
            cradle_t: float = 0.008
            rack_hx: float = 0.046
            rack_hy: float = 0.107
            rack_in: float = 0.056
            rack_t: float = 0.008
            div_y: float = 0.031
            div_t: float = 0.006
            bob_hx: float = 0.020
            bob_y0: float = 0.045
            bob_y1: float = 0.085
            bob_z0: float = -0.16
            bob_z1: float = -0.06
            beam_mass: float = 1.6
            beam_com_z: float = -0.095
            beam_inertia: tuple = (0.020, 0.055, 0.045)
            beam_ang_damping: float = 3.0
            bar_color: tuple = (0.55, 0.56, 0.60)
            cradle_color: tuple = (0.80, 0.15, 0.12)
            rack_color: tuple = (0.20, 0.30, 0.55)
            bob_color: tuple = (0.20, 0.20, 0.22)
            contact_offset: float = 0.0015
            beam_mu_s: float = 0.60
            beam_mu_d: float = 0.55

        _SPAWNER_CACHE["station"] = StationSpawnerCfg
        _SPAWNER_CACHE["beam"] = BeamSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class WeighStationSceneCfg(BaseCfg):
    """Config for `WeighStationScene`. The weighing contract is asserted in
    `__post_init__`: a one-unit imbalance tilts the beam at least twice the level
    tolerance; worst-case placement slop inside the pockets stays well inside the
    tolerance; the swinging beam clears the pillar, the fork, and every apron
    object at the ±20° stops; pockets accept the weights, the cradle accepts every
    parcel; and no stack of apron objects can prop the beam ends up."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    level_tol_deg: float = tunable(6.0)    # |beam tilt| at judging (deg)
    settle_lin: float = tunable(0.05)      # max object |lin vel| when judging (m/s)
    settle_rate: float = tunable(0.10)     # max |FD beam tilt rate| when judging (rad/s)
    cradle_xy_tol: float = tunable(0.030)  # shipment centre box in the beam frame (m)
    cradle_z_tol: float = tunable(0.016)   # shipment resting-height tolerance (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    yaw_deg: float = tunable(180.0)        # station yaw uniform ± (FREE heading)
    xy_jitter: float = tunable(0.05)       # station xy jitter (± m)
    obj_jitter: float = tunable(0.012)     # apron object xy jitter (± m)
    randomize_k: bool = tunable(True)      # sample WHICH parcel ships (k in 1..3)

    # --- info: station (local origin at the slab top centre) -------------------------------------
    base_z: float = info(0.031)            # root height: slab bottom 1 mm above ground
    slab_hx: float = info(0.35)
    slab_hy: float = info(0.24)
    slab_t: float = info(0.03)
    pillar_h2: float = info(0.025)
    pillar_z1: float = info(0.230)         # >= 4 mm below the bar's lowest swing point
    prong_hx: float = info(0.020)
    prong_y0: float = info(0.025)          # fork plates flank the 30 mm bar with 10 mm gap
    prong_y1: float = info(0.039)
    prong_z0: float = info(0.22)
    prong_z1: float = info(0.29)
    pad_center: tuple = info((0.18, -0.16))
    pad_half: float = info(0.05)
    pad_t: float = info(0.002)
    hinge_z: float = info(0.28)            # hinge height above the slab top
    station_mass: float = info(30.0)       # heavy DYNAMIC fixture: the joint anchor must
    station_inertia: tuple = info((1.0, 1.0, 1.8))  # follow reset teleports
    # --- info: beam ------------------------------------------------------------------------------
    arm_r: float = info(0.25)              # cradle centre AND rack pockets, both at this radius
    bar_hx: float = info(0.30)
    bar_hy: float = info(0.015)
    bar_hz: float = info(0.010)
    floor_z0: float = info(0.010)          # pocket/cradle floor spans (beam frame)
    floor_z1: float = info(0.016)
    wall_z1: float = info(0.048)
    cradle_in: float = info(0.076)         # cradle interior square (largest parcel 66 mm)
    cradle_t: float = info(0.008)
    rack_hx: float = info(0.046)           # rack outer x half-extent
    rack_hy: float = info(0.107)
    rack_in: float = info(0.056)           # pocket x-interior (weight 40 mm -> ±8 mm slop)
    rack_t: float = info(0.008)
    div_y: float = info(0.031)
    div_t: float = info(0.006)
    pocket_y: tuple = info((-0.0665, 0.0, 0.0665))  # pocket centres (all at x = -arm_r)
    bob_hx: float = info(0.020)
    bob_y0: float = info(0.045)
    bob_y1: float = info(0.085)
    bob_z0: float = info(-0.16)
    bob_z1: float = info(-0.06)
    beam_mass: float = info(1.6)
    beam_com_z: float = info(-0.095)       # K = M g |com_z| = 1.49 N*m/rad restoring
    beam_inertia: tuple = info((0.020, 0.055, 0.045))
    beam_ang_damping: float = info(3.0)    # settles the swing in ~1-2 s (zeta ~ 0.2)
    stop_deg: float = info(20.0)           # revolute joint hard stops (±)
    # --- info: cargo -----------------------------------------------------------------------------
    weight_size: float = info(0.040)
    weight_mass: float = info(0.15)        # THE mass unit
    n_weights: int = info(4)               # always more than needed (distractors)
    parcel_sizes: tuple = info((0.046, 0.058, 0.066))  # one density: size**3 ~ mass
    parcel_masses: tuple = info((0.15, 0.30, 0.45))    # k = 1, 2, 3 units
    weight_slots: tuple = info(((-0.26, 0.16), (-0.14, 0.16), (-0.14, -0.16), (-0.26, -0.16)))
    depot_xy: tuple = info(((1.5, 0.9), (1.5, 0.6), (1.5, 0.3)))  # off-station parcel park
    weight_color: tuple = info((0.25, 0.45, 0.85))
    parcel_color: tuple = info((0.10, 0.65, 0.15))
    # --- info: contact/materials -----------------------------------------------------------------
    contact_offset: float = info(0.0015)
    cargo_contact_offset: float = info(0.004)  # speculative margin catches the 3 cm drops
    # --- info: rubric weights (0.30 + 0.30 = 0.60 = the non-success cap) -------------------------
    w_ballast: float = info(0.30)
    w_parcel: float = info(0.30)
    score_cap: float = info(0.60)
    rate_clamp: float = info(20.0)         # FD tilt-rate clamp (teleport transients)

    def __post_init__(self) -> None:
        c = self
        g = 9.81
        k_bob = c.beam_mass * g * abs(c.beam_com_z)
        tau_u = c.weight_mass * g * c.arm_r
        # one-unit imbalance must tilt at least twice the level tolerance (worst case:
        # the destabilizing load-height term only INCREASES the tilt, so k_bob alone
        # is the conservative stiffness)
        assert math.degrees(math.atan(tau_u / k_bob)) >= 2.0 * c.level_tol_deg, \
            "a one-unit imbalance must read far outside the level tolerance"
        # loads must never overwhelm the pendulum stiffness (level stays stable)
        h_max = c.floor_z1 + max(c.parcel_sizes) / 2
        m_loads = max(c.parcel_masses) + 3 * c.weight_mass
        assert k_bob > 2.0 * m_loads * g * h_max, "level equilibrium must stay stiffly stable"
        # worst-case placement slop must stay well inside the tolerance
        slop_w = (c.rack_in - c.weight_size) / 2
        slop_p = (c.cradle_in - min(c.parcel_sizes)) / 2
        noise = 3 * c.weight_mass * g * slop_w + max(
            c.parcel_masses[0] * g * slop_p,
            c.parcel_masses[2] * g * (c.cradle_in - c.parcel_sizes[2]) / 2)
        k_net = k_bob - m_loads * g * h_max
        assert math.degrees(math.atan(noise / k_net)) < 0.65 * c.level_tol_deg, \
            "placement slop must not eat the level tolerance"
        # fits: pockets accept a weight, the cradle accepts every parcel
        assert c.rack_in >= c.weight_size + 0.012, "pocket x-interior must accept a weight"
        assert min(0.099 - 0.034, 0.028 * 2) >= c.weight_size + 0.012, \
            "pocket y-interior must accept a weight"
        assert c.cradle_in >= max(c.parcel_sizes) + 0.008, "cradle must accept every parcel"
        assert c.n_weights > 3, "there must always be spare (distractor) weights"
        # swing clearances at the ±stop: bar over the pillar, arm tips over the apron
        th = math.radians(c.stop_deg)
        bar_low = c.hinge_z - (c.pillar_h2 + c.prong_hx) * math.sin(th) - c.bar_hz * math.cos(th)
        assert bar_low - c.pillar_z1 >= 0.004, "bar must clear the pillar top at the stops"
        tip_low = c.hinge_z - c.bar_hx * math.sin(th) - c.wall_z1 * math.cos(th) + 0.0
        top_of_apron = c.weight_size + 0.002
        assert tip_low - max(c.parcel_sizes) > 0.02 and tip_low > top_of_apron + 0.02, \
            "arm tips must clear apron objects at the stops"
        # no stack of spare apron objects can prop the beam LEVEL from below
        # (anti-prop; worst case k=1: three spare weights + both bigger parcels)
        stack = 3 * c.weight_size + c.parcel_sizes[1] + c.parcel_sizes[2]
        assert stack + 0.02 < c.hinge_z - c.bar_hz, \
            "apron stacks must not reach the level beam underside"
        # apron slots: on the slab, outside the beam footprint and the bob sweep
        for sx, sy in c.weight_slots + (c.pad_center,):
            assert abs(sx) + c.obj_jitter + 0.05 < c.slab_hx, "slot on the slab (x)"
            assert abs(sy) + c.obj_jitter + 0.05 < c.slab_hy, "slot on the slab (y)"
            assert abs(sy) - c.obj_jitter - 0.035 > c.rack_hy, \
                "slots must sit outside the beam footprint in y"
        # rubric weights
        assert abs(c.w_ballast + c.w_parcel - c.score_cap) < 1e-9


# ----- small quaternion helpers (wxyz, torch, batched) ------------------------------------------
def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _qconj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("weigh_station")
class WeighStationScene(BaseScene):
    cfg: WeighStationSceneCfg

    def __init__(self, cfg: WeighStationSceneCfg | None = None) -> None:
        super().__init__(cfg or WeighStationSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        station_spawn = cls["station"](
            slab_hx=c.slab_hx, slab_hy=c.slab_hy, slab_t=c.slab_t,
            pillar_h2=c.pillar_h2, pillar_z1=c.pillar_z1,
            prong_hx=c.prong_hx, prong_y0=c.prong_y0, prong_y1=c.prong_y1,
            prong_z0=c.prong_z0, prong_z1=c.prong_z1,
            pad_center=c.pad_center, pad_half=c.pad_half, pad_t=c.pad_t,
            station_mass=c.station_mass, station_inertia=c.station_inertia,
            contact_offset=c.contact_offset)
        beam_spawn = cls["beam"](
            arm_r=c.arm_r, bar_hx=c.bar_hx, bar_hy=c.bar_hy, bar_hz=c.bar_hz,
            floor_z0=c.floor_z0, floor_z1=c.floor_z1, wall_z1=c.wall_z1,
            cradle_in=c.cradle_in, cradle_t=c.cradle_t,
            rack_hx=c.rack_hx, rack_hy=c.rack_hy, rack_in=c.rack_in, rack_t=c.rack_t,
            div_y=c.div_y, div_t=c.div_t,
            bob_hx=c.bob_hx, bob_y0=c.bob_y0, bob_y1=c.bob_y1,
            bob_z0=c.bob_z0, bob_z1=c.bob_z1,
            beam_mass=c.beam_mass, beam_com_z=c.beam_com_z, beam_inertia=c.beam_inertia,
            beam_ang_damping=c.beam_ang_damping, contact_offset=c.contact_offset)

        rigid = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=0.5, linear_damping=0.20, angular_damping=0.20,
            sleep_threshold=0.0, stabilization_threshold=0.0,
            solver_position_iteration_count=32, solver_velocity_iteration_count=1)
        coll = sim_utils.CollisionPropertiesCfg(
            contact_offset=c.cargo_contact_offset, rest_offset=0.0)

        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "station": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Station", spawn=station_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, c.base_z))),
            "beam": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Beam", spawn=beam_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, c.base_z + c.hinge_z))),
        }
        for k in range(c.n_weights):
            out[f"weight_{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Weight_{k}",
                spawn=sim_utils.CuboidCfg(
                    size=(c.weight_size, c.weight_size, c.weight_size),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.weight_mass),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.weight_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0 + 0.1 * k, -0.8, 0.05)))
        for k in range(3):
            s = c.parcel_sizes[k]
            out[f"parcel_{k}"] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Parcel_{k}",
                spawn=sim_utils.CuboidCfg(
                    size=(s, s, s),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.parcel_masses[k]),
                    rigid_props=rigid, collision_props=coll,
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.50, dynamic_friction=0.45, restitution=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=c.parcel_color)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.4 + 0.2 * k, -0.8, 0.05)))
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
        self.station: RigidObject = env.iscene["station"]
        self.beam: RigidObject = env.iscene["beam"]
        self.weights: list[RigidObject] = [env.iscene[f"weight_{k}"]
                                           for k in range(c.n_weights)]
        self.parcels: list[RigidObject] = [env.iscene[f"parcel_{k}"] for k in range(3)]
        self.env_origins = env.iscene.env_origins
        self._author_joints()
        n = env.num_envs
        dev = env.device
        # episode readbacks (verified by smoke)
        self.k_units = torch.ones(n, dtype=torch.long, device=dev)  # required ballast
        # FD tilt rate (do not trust root_ang_vel through joint/teleport transients)
        self.rate_fd = torch.zeros(n, device=dev)
        self._tilt_prev = torch.zeros(n, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._ballast = torch.zeros(n, device=dev)   # max min(seated, k)/k
        self._parcel = torch.zeros(n, dtype=torch.bool, device=dev)

    def _author_joints(self) -> None:
        """Per env: a Y-axis revolute joint station->beam at the fork, with ±stop_deg
        hard limits (the keel stops). Body0 is the heavy DYNAMIC station root so the
        anchor follows reset teleports (a kinematic body0 anchor stays world-fixed at
        the spawn pose on this stack). The joint pair is collision-filtered by PhysX;
        the beam additionally clears the pillar/fork geometrically at every angle
        inside the stops, so filtering is never load-bearing. PhysX ignores joint
        friction outside articulations — the hinge is honestly frictionless, and all
        settling comes from the beam body's angular damping."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/beam_pivot")
            j.CreateBody0Rel().SetTargets([f"{base}/Station"])
            j.CreateBody1Rel().SetTargets([f"{base}/Beam"])
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(c.hinge_z)))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-float(c.stop_deg))
            j.CreateUpperLimitAttr(float(c.stop_deg))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: pose the station (free yaw + xy jitter) and the beam
        together at the hinge, LEVEL (write the whole linkage — teleporting one body
        of a joint pair gets depenetrated back by the other); sample k (WHICH parcel
        ships); stand the shipment on the green pad, park the other two parcels in
        the ground depot; scatter the four counterweights on their apron slots. All
        discrete draws derive from torch.rand (first-randint degeneracy)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        from isaaclab.utils.math import quat_apply

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.yaw_deg)
        q_h = _qz(yaw)
        dp = torch.zeros(m, 3, device=dev)
        dp[:, 0] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 1] = (torch.rand(m, device=dev) * 2 - 1) * c.xy_jitter
        dp[:, 2] = c.base_z
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin
        st[:, 3:7] = q_h
        self.station.write_root_state_to_sim(st, env_ids)

        # beam: level at the hinge (same yaw as the station)
        hinge = torch.tensor([0.0, 0.0, c.hinge_z], device=dev).expand(m, 3)
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = dp + origin + quat_apply(q_h, hinge)
        st[:, 3:7] = q_h
        self.beam.write_root_state_to_sim(st, env_ids)

        # k: how many mass units ship this episode (rand-derived, readback-verified)
        k = (torch.rand(m, device=dev) * 3).long().clamp(max=2) + 1
        if not c.randomize_k:
            k = torch.full((m,), 2, dtype=torch.long, device=dev)
        self.k_units[env_ids] = k

        # parcels: the shipment (index k-1) stands on the green pad; others park
        px, py = c.pad_center
        for j in range(3):
            ship = (k - 1) == j
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = px + (torch.rand(m, device=dev) * 2 - 1) * c.obj_jitter
            loc[:, 1] = py + (torch.rand(m, device=dev) * 2 - 1) * c.obj_jitter
            loc[:, 2] = c.pad_t + c.parcel_sizes[j] / 2 + 0.003
            yaw_p = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = dp + origin + quat_apply(q_h, loc)
            s[:, 3:7] = _qmul(q_h, _qz(yaw_p))
            depot = torch.tensor([c.depot_xy[j][0], c.depot_xy[j][1],
                                  c.parcel_sizes[j] / 2 + 0.002], device=dev)
            s[:, 0:3] = torch.where(ship.unsqueeze(-1), s[:, 0:3], origin + depot)
            s[:, 3] = torch.where(ship, s[:, 3], torch.ones(m, device=dev))
            s[:, 4:7] = torch.where(ship.unsqueeze(-1), s[:, 4:7],
                                    torch.zeros(m, 3, device=dev))
            self.parcels[j].write_root_state_to_sim(s, env_ids)

        # counterweights: apron slots + jitter + free yaw
        for j, (sx, sy) in enumerate(c.weight_slots):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = sx + (torch.rand(m, device=dev) * 2 - 1) * c.obj_jitter
            loc[:, 1] = sy + (torch.rand(m, device=dev) * 2 - 1) * c.obj_jitter
            loc[:, 2] = c.weight_size / 2 + 0.003
            yaw_w = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            s = torch.zeros(m, 13, device=dev)
            s[:, 0:3] = dp + origin + quat_apply(q_h, loc)
            s[:, 3:7] = _qmul(q_h, _qz(yaw_w))
            self.weights[j].write_root_state_to_sim(s, env_ids)

        self.rate_fd[env_ids] = 0.0
        self._tilt_prev[env_ids] = 0.0
        self._ballast[env_ids] = 0.0
        self._parcel[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "station": self.station.data.root_state_w[env_ids].clone(),
            "beam": self.beam.data.root_state_w[env_ids].clone(),
            "k_units": self.k_units[env_ids].clone(),
            "rate_fd": self.rate_fd[env_ids].clone(),
            "tilt_prev": self._tilt_prev[env_ids].clone(),
            "ballast": self._ballast[env_ids].clone(),
            "parcel": self._parcel[env_ids].clone(),
        }
        for j, b in enumerate(self.weights):
            out[f"weight_{j}"] = b.data.root_state_w[env_ids].clone()
        for j, b in enumerate(self.parcels):
            out[f"parcel_{j}"] = b.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.station.write_root_state_to_sim(state["station"], env_ids)
        self.beam.write_root_state_to_sim(state["beam"], env_ids)
        for j, b in enumerate(self.weights):
            b.write_root_state_to_sim(state[f"weight_{j}"], env_ids)
        for j, b in enumerate(self.parcels):
            b.write_root_state_to_sim(state[f"parcel_{j}"], env_ids)
        self.k_units[env_ids] = state["k_units"]
        self.rate_fd[env_ids] = state["rate_fd"]
        self._tilt_prev[env_ids] = state["tilt_prev"]
        self._ballast[env_ids] = state["ballast"]
        self._parcel[env_ids] = state["parcel"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A WEIGH STATION stands on a low counter slab: a centre pillar carries a "
            f"pivoting BALANCE BEAM on a gimbal fork. One arm of the beam ends in a RED "
            f"open-top CRADLE, the other in a BLUE COUNTERWEIGHT RACK with three open-top "
            f"pockets in a row; a dark yoke bob hangs under the pivot, so the empty beam "
            f"rests level and tilts under any load imbalance until it leans on its ±"
            f"{c.stop_deg:.0f}° stops. Cradle centre and rack pockets sit at the SAME "
            f"{1000 * c.arm_r:.0f} mm radius from the pivot, so balance is pure mass "
            f"arithmetic.\n"
            f"On the slab apron lie FOUR identical steel-blue counterweight cubes "
            f"({1000 * c.weight_size:.0f} mm, {c.weight_mass:.2f} kg each — the mass "
            f"unit), and ONE green parcel stands on the green shipment pad. All parcels "
            f"are cast from one resin, so SIZE announces MASS: a "
            f"{1000 * c.parcel_sizes[0]:.0f} mm cube weighs 1 unit, "
            f"{1000 * c.parcel_sizes[1]:.0f} mm weighs 2 units, "
            f"{1000 * c.parcel_sizes[2]:.0f} mm weighs 3 units "
            f"(0.15/0.30/0.45 kg). Which parcel ships — and every pose — varies by "
            f"episode.\n"
            f"Goal: hand the parcel over to the scale IN BALANCE. Seat EXACTLY as many "
            f"counterweight cubes in the blue rack pockets as the parcel weighs in units "
            f"(judge its size), set the parcel into the red cradle, and leave the beam "
            f"LEVEL — within {c.level_tol_deg:.0f}° — and at rest. One cube too few or "
            f"too many tilts the beam ~14° or onto a stop and fails. Spare cubes must "
            f"stay OFF the scale: any weight riding the beam outside the rack pockets, "
            f"or a depot parcel smuggled aboard, fails the episode. Loading order is "
            f"free (ballast-first keeps the swing smallest); only the settled end state "
            f"is judged."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Weigh the green parcel by its size (46/58/66 mm = 1/2/3 units), seat "
            "exactly that many blue counterweight cubes in the rack pockets on the "
            "balance beam's blue arm, place the parcel in the red cradle, and leave "
            "the beam level and at rest with every spare cube off the scale."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def tilt(self) -> torch.Tensor:
        """(N,) beam tilt about the hinge axis (rad, wrapped): pitch of the relative
        quaternion station->beam. 0 = level; +ve = cradle arm down."""
        q = _qmul(_qconj(self.station.data.root_quat_w), self.beam.data.root_quat_w)
        return _wrap(2.0 * torch.atan2(q[:, 2], q[:, 0]))

    def _beam_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.beam.data.root_quat_w,
                                  pos_w - self.beam.data.root_pos_w)

    def in_cradle(self) -> torch.Tensor:
        """(N,) bool: the SHIPMENT parcel seated in the cradle (beam frame): centred
        on the cradle, resting on the cradle floor."""
        c = self.cfg
        sizes = torch.tensor(c.parcel_sizes, device=self.env.device)
        size = sizes[self.k_units - 1]
        pos = torch.stack([p.data.root_pos_w for p in self.parcels], dim=1)
        idx = (self.k_units - 1).view(-1, 1, 1).expand(-1, 1, 3)
        ship_pos = torch.gather(pos, 1, idx).squeeze(1)
        loc = self._beam_local(ship_pos)
        rest_z = c.floor_z1 + size / 2
        return ((loc[:, 0] - c.arm_r).abs() < c.cradle_xy_tol) \
            & (loc[:, 1].abs() < c.cradle_xy_tol) \
            & ((loc[:, 2] - rest_z).abs() < c.cradle_z_tol)

    def _weight_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        """((N,W) in-rack, (N,W) on-beam) for every counterweight, beam frame. The
        rack box covers seated AND stacked weights; the on-beam box covers the whole
        beam envelope (bar, cradle, rack, bob tops)."""
        c = self.cfg
        pos = torch.stack([w.data.root_pos_w for w in self.weights], dim=1)
        n, wn = pos.shape[0], pos.shape[1]
        loc = self._beam_local(pos.reshape(n * wn, 3)).reshape(n, wn, 3)
        in_rack = ((loc[:, :, 0] + c.arm_r).abs() < 0.045) \
            & (loc[:, :, 1].abs() < 0.115) \
            & (loc[:, :, 2] > 0.020) & (loc[:, :, 2] < 0.110)
        on_beam = (loc[:, :, 0].abs() < 0.34) & (loc[:, :, 1].abs() < 0.14) \
            & (loc[:, :, 2] > -0.03) & (loc[:, :, 2] < 0.16)
        return in_rack, on_beam

    def _parcels_on_beam(self) -> torch.Tensor:
        """(N,3) bool: parcel j anywhere on the beam envelope (the anti-smuggling
        clause: only the shipment belongs aboard, and only in the cradle)."""
        pos = torch.stack([p.data.root_pos_w for p in self.parcels], dim=1)
        n = pos.shape[0]
        loc = self._beam_local(pos.reshape(n * 3, 3)).reshape(n, 3, 3)
        return (loc[:, :, 0].abs() < 0.34) & (loc[:, :, 1].abs() < 0.14) \
            & (loc[:, :, 2] > -0.03) & (loc[:, :, 2] < 0.16)

    def level(self) -> torch.Tensor:
        """(N,) bool: |tilt| within the level tolerance."""
        return self.tilt().abs() < math.radians(self.cfg.level_tol_deg)

    def settled(self) -> torch.Tensor:
        """(N,) bool: station/weights/parcels slow, beam FD tilt rate slow."""
        c = self.cfg
        ok = self.station.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        for b in self.weights + self.parcels:
            ok = ok & (b.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
        return ok & (self.rate_fd.abs() < c.settle_rate)

    def _finite(self) -> torch.Tensor:
        ps = [self.station.data.root_pos_w, self.beam.data.root_pos_w] \
            + [b.data.root_pos_w for b in self.weights] \
            + [b.data.root_pos_w for b in self.parcels]
        return torch.isfinite(torch.stack(ps, dim=1)).all(dim=-1).all(dim=-1)

    # ----- step-coupled mechanics (every substep) ------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """FD tilt rate (teleport-transient clamped) + rubric latches. The beam is a
        PASSIVE pendulum plant — no external wrenches anywhere in this scene."""
        c = self.cfg
        dt = self.env.dt
        th = self.tilt()
        fin = torch.isfinite(th)
        raw = _wrap(th - self._tilt_prev) / dt
        self.rate_fd = torch.where(
            fin, raw.clamp(-c.rate_clamp, c.rate_clamp), torch.zeros_like(raw))
        self._tilt_prev = torch.where(fin, th, self._tilt_prev)
        self._update_latches()

    def _update_latches(self) -> None:
        fin = self._finite()
        in_rack, _ = self._weight_flags()
        frac = torch.minimum(in_rack.sum(dim=1), self.k_units).float() \
            / self.k_units.float()
        self._ballast = torch.where(fin, torch.maximum(self._ballast, frac), self._ballast)
        self._parcel |= self.in_cradle() & fin

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the shipment seated in the cradle, the beam LEVEL, exactly k
        counterweights in the rack, no weight riding the beam outside the rack, no
        depot parcel aboard, everything settled and finite — all live physical
        outcomes. Level with the parcel aboard is only physically reachable with the
        matching ballast (a one-unit error tilts ≥ 2x the tolerance)."""
        self._update_latches()
        in_rack, on_beam = self._weight_flags()
        clean_w = (~on_beam | in_rack).all(dim=1)
        exact = in_rack.sum(dim=1) == self.k_units
        pob = self._parcels_on_beam()
        ship = torch.nn.functional.one_hot(self.k_units - 1, 3).bool()
        clean_p = ~(pob & ~ship).any(dim=1)
        return self.in_cradle() & self.level() & exact & clean_w & clean_p \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30 * latched required-ballast fraction + 0.30 *
        latched parcel-ever-cradled, capped at 0.60; exactly 1.0 iff success() holds
        live. Doing nothing scores ~0; the seed's carry-to-destination strategy
        latches 0.30 and keels the beam onto its stop — never 1.0."""
        c = self.cfg
        self._update_latches()
        base = (c.w_ballast * self._ballast + c.w_parcel * self._parcel.float()) \
            .clamp(max=c.score_cap)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="weigh_station", robot="null"))
