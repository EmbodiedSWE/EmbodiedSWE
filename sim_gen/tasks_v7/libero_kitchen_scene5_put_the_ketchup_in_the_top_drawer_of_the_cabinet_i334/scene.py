"""LetterboxCradleScene — stand the sauce bottle UPRIGHT inside a sealed vault whose
only opening is a low LETTER SLOT, using the through-wall crank of an over-centre
TILT CRADLE (sim_gen task
`libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet_i334`).

Derived from libero_90/libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the
_cabinet, but STRATEGICALLY different: the seed's plan is *pull the drawer open, pick
the upright ketchup up, drop it in* — the receptacle is opened by translation and the
bottle keeps its upright pose the whole way. Here NOTHING opens and the bottle's pose
is the puzzle: the vault's only aperture is a wide, LOW letter slot (the standing
bottle is almost twice its height), so the bottle must first be LAID ON ITS SIDE in
the apron's guide lane, then SLID BASE-FIRST through the slot onto a hinged tilt
cradle inside, and finally RE-ERECTED by the machine: an external crank (through-wall
axle) raises the cradle; past its over-centre angle (~68 deg) the loaded cradle tips
onto its 90-degree stop by itself and lands the bottle standing on the cradle's end
plate — upright, inside a chamber no hand can enter. The seed's whole skill
(pick-and-place the upright bottle into an opened receptacle) is physically refused
by the aperture, and the goal pose is produced by the mechanism, not by the hand.

Execution order is REQUIRED and physics-enforced: (1) upright entry is blocked by the
slot header (smoke-proven with a real shove); (2) erecting the EMPTY cradle first
strands the bottle lying on the chamber floor — the vertical bed leaves nothing to
ride, and sector... the erect credit is guarded by "bottle riding the cradle", so the
wrong order also earns nothing for the cranking; (3) releasing the crank below the
over-centre angle drops the loaded cradle back onto its rest stop (bottle lying
again) — the crank must be carried past the tip-over before letting go.

No stored energy against the goal: the cradle is gravity-BISTABLE about its hinge
(authored CoM offset). Unloaded and below ~63 deg it rests on its 0-degree stop;
past over-centre it rests on its 90-degree stop; the erected, loaded terminal state
presses the 90-degree stop with ~0.14 N.m of gravity margin and persists hands-off.

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide): vault (KINEMATIC: plinth, chamber walls, roof, slotted front wall,
apron with a guide lane, and a small square axle port in the +y side wall), tilt
cradle (DYNAMIC: bed + side rails + end plate + through-wall axle + external crank
arm and knob, on a revolute joint whose limits [-90, 0] deg ARE the hard stops;
vault<->cradle contact is joint-filtered — the stops are the limits, and the axle
rides its port without touching), plus the free bottle (native cylinder). The
authored MassAPI CoM (0.06, 0, 0.03) is the whole bistability mechanism.

Per-episode randomization (readback-verified by smoke): the bottle's standing pose
on the apron (x, y over a 140 x 200 mm band + free yaw).

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.15  laid    — bottle ever settled-lying (axis within ~20 deg of horizontal)
  0.35  inside  — bottle root ever fully through the slot (chamber box)
  0.40  x latched max erect fraction phi/90 WHILE the bottle rides the cradle
capped at 0.90; exactly 1.0 iff success(): bottle UPRIGHT (axis within
`upright_max_deg` of world-up), root inside the chamber box, settled and finite —
judged LIVE on the bottle's physical pose only. Null policy ~0 (the bottle starts
standing outside). The seed's strategy (carry the upright bottle to the receptacle)
earns ~0 — it cannot even reach the inside.

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


def _cyl(stage, path: str, *, axis: str, radius: float, half: float, center, color,
         collide: Callable):
    """Cylinder child (native PhysX cylinder collision), axis-aligned."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(2 * half))
    r, h = float(radius), float(half)
    ext = {"X": [(-h, -r, -r), (h, r, r)], "Y": [(-r, -h, -r), (r, h, r)],
           "Z": [(-r, -r, -h), (r, r, h)]}[axis]
    cyl.CreateExtentAttr([Gf.Vec3f(*ext[0]), Gf.Vec3f(*ext[1])])
    xf = UsdGeom.Xformable(cyl.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(cyl.GetPrim())
    return cyl.GetPrim()


def _mk_material(prim_path: str, name: str, mu_s: float, mu_d: float, combine: str) -> str:
    import isaaclab.sim as sim_utils

    mat_path = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat_path, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0,
        friction_combine_mode=combine))
    return mat_path


def _spawn_vault(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the vault at `prim_path`: KINEMATIC compound. Local frame: x=0 is the
    front wall's OUTER face plane (+x = toward the robot), z=0 the ground. Chamber
    interior x [-0.30, -0.02], |y| < 0.14, z [0.45, 0.70]; the only aperture is the
    letter slot (|y| <= slot_hw, apron top .. slot_top) plus a small square axle
    port in the +y side wall that the bottle cannot pass."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    # plinth: solid pedestal whose top IS the chamber floor
    _span(stage, f"{prim_path}/plinth", x=(c.back_x1, 0.0), y=(-c.wall_y1, c.wall_y1),
          z=(0.0, c.floor_z1), color=c.body_color, collide=collide)
    # back wall
    _span(stage, f"{prim_path}/back", x=(c.back_x1, c.back_x0),
          y=(-c.wall_y1, c.wall_y1), z=(c.floor_z1, c.roof_z1), color=c.body_color,
          collide=collide)
    # -y side wall (solid)
    _span(stage, f"{prim_path}/side_n", x=(c.back_x0, 0.0), y=(-c.wall_y1, -c.wall_y0),
          z=(c.floor_z1, c.roof_z1), color=c.body_color, collide=collide)
    # +y side wall with the square axle port (x hinge_x +/- port_half, floor..z port top)
    px0, px1 = c.hinge_x - c.port_half, c.hinge_x + c.port_half
    _span(stage, f"{prim_path}/side_p_front", x=(px1, 0.0), y=(c.wall_y0, c.wall_y1),
          z=(c.floor_z1, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_p_back", x=(c.back_x0, px0), y=(c.wall_y0, c.wall_y1),
          z=(c.floor_z1, c.roof_z1), color=c.body_color, collide=collide)
    _span(stage, f"{prim_path}/side_p_above", x=(px0, px1), y=(c.wall_y0, c.wall_y1),
          z=(c.hinge_z + c.port_half, c.roof_z1), color=c.body_color, collide=collide)
    # roof
    _span(stage, f"{prim_path}/roof", x=(c.back_x1, 0.0), y=(-c.wall_y1, c.wall_y1),
          z=(c.roof_z0, c.roof_z1), color=c.roof_color, collide=collide)
    # front wall with the letter slot (|y| <= slot_hw, z floor..slot_top open)
    _span(stage, f"{prim_path}/front_yn", x=(c.front_x0, 0.0), y=(-c.wall_y1, -c.slot_hw),
          z=(c.floor_z1, c.roof_z1), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/front_yp", x=(c.front_x0, 0.0), y=(c.slot_hw, c.wall_y1),
          z=(c.floor_z1, c.roof_z1), color=c.front_color, collide=collide)
    _span(stage, f"{prim_path}/front_header", x=(c.front_x0, 0.0),
          y=(-c.slot_hw, c.slot_hw), z=(c.slot_top, c.roof_z1),
          color=c.front_color, collide=collide)
    # apron shelf (its top is the sliding plane / effective slot sill)
    _span(stage, f"{prim_path}/apron", x=(0.0, c.apron_x1), y=(-c.wall_y1, c.wall_y1),
          z=(0.0, c.apron_z1), color=c.apron_color, collide=collide)
    # guide lane fences on the apron, leading to the slot
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _span(stage, f"{prim_path}/fence_{tag}", x=(0.0, c.fence_x1),
              y=(sgn * c.fence_y0, sgn * c.fence_y1) if sgn > 0
              else (sgn * c.fence_y1, sgn * c.fence_y0),
              z=(c.apron_z1, c.apron_z1 + c.fence_h), color=c.fence_color, collide=collide)
    wood = _mk_material(prim_path, "struct", c.struct_mu_s, c.struct_mu_d, "average")
    bind_physics_material(prim_path, wood)
    return root


def _spawn_cradle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the tilt cradle with root origin AT the hinge, at its REST pose
    (bed horizontal toward the slot). Bed slab + two side rails + end plate (the
    erected floor) + through-wall axle (cylinder, axis Y) + external crank arm and
    knob riding outside the +y wall. MassAPI CoM authored at (com_x, 0, com_z):
    gravity-bistable about the hinge — rest stop below the over-centre angle, the
    90-degree stop above it. No springs, no stored energy against the goal."""
    from isaaclab.sim.utils import bind_physics_material
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    _span(stage, f"{prim_path}/bed", x=(c.foot_x1, c.bed_x1), y=(-c.bed_hy, c.bed_hy),
          z=(-0.016, -0.004), color=c.bed_color, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        y0, y1 = sgn * c.rail_y0, sgn * c.rail_y1
        _span(stage, f"{prim_path}/rail_{tag}", x=(c.foot_x1, c.bed_x1),
              y=(min(y0, y1), max(y0, y1)), z=(-0.004, c.rail_top),
              color=c.bed_color, collide=collide)
    _span(stage, f"{prim_path}/foot", x=(0.0, c.foot_x1), y=(-c.foot_hy, c.foot_hy),
          z=(-0.004, c.foot_top), color=c.foot_color, collide=collide)
    _cyl(stage, f"{prim_path}/axle", axis="Y", radius=c.axle_r,
         half=(c.arm_y1 + c.bed_hy) / 2, center=(0.0, (c.arm_y1 - c.bed_hy) / 2, 0.0),
         color=c.crank_color, collide=collide)
    _span(stage, f"{prim_path}/arm", x=(0.0, c.arm_len), y=(c.arm_y0, c.arm_y1),
          z=(-0.0125, 0.0125), color=c.crank_color, collide=collide)
    _span(stage, f"{prim_path}/knob", x=(c.arm_len, c.arm_len + 0.04),
          y=(c.arm_y0 - 0.003, c.arm_y1 + 0.003), z=(-0.02, 0.02),
          color=c.knob_color, collide=collide)
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(float(c.mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(float(c.com_x), 0.0, float(c.com_z)))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    prb.CreateLinearDampingAttr(0.2)
    prb.CreateAngularDampingAttr(float(c.ang_damp))
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    struct = _mk_material(prim_path, "struct", c.struct_mu_s, c.struct_mu_d, "average")
    bind_physics_material(prim_path, struct)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "vault" not in _SPAWNER_CACHE:

        @configclass
        class VaultSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_vault)
            back_x1: float = -0.32
            back_x0: float = -0.30
            wall_y0: float = 0.14
            wall_y1: float = 0.16
            floor_z1: float = 0.45
            roof_z0: float = 0.70
            roof_z1: float = 0.72
            front_x0: float = -0.02
            slot_hw: float = 0.075
            slot_top: float = 0.562
            apron_x1: float = 0.44
            apron_z1: float = 0.472
            fence_x1: float = 0.20
            fence_y0: float = 0.036
            fence_y1: float = 0.052
            fence_h: float = 0.017
            hinge_x: float = -0.195
            hinge_z: float = 0.474
            port_half: float = 0.022
            body_color: tuple = (0.42, 0.45, 0.52)
            roof_color: tuple = (0.32, 0.35, 0.42)
            front_color: tuple = (0.50, 0.53, 0.60)
            apron_color: tuple = (0.62, 0.50, 0.34)
            fence_color: tuple = (0.20, 0.22, 0.25)
            contact_offset: float = 0.0015
            struct_mu_s: float = 0.45
            struct_mu_d: float = 0.40

        @configclass
        class CradleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cradle)
            bed_x1: float = 0.170
            bed_hy: float = 0.055
            rail_y0: float = 0.034
            rail_y1: float = 0.048
            rail_top: float = 0.026
            foot_x1: float = 0.012
            foot_hy: float = 0.048
            foot_top: float = 0.079
            # NOTE: axle_r must stay BELOW foot_x1 (the end-plate face plane): an axle
            # tangent to that plane catches the tilted bottle's base rim near the top
            # of the swing and wedges the cradle ~7 deg short of its 90-deg stop.
            axle_r: float = 0.010
            arm_y0: float = 0.185
            arm_y1: float = 0.210
            arm_len: float = 0.20
            mass: float = 0.35
            com_x: float = 0.06
            com_z: float = 0.03
            ang_damp: float = 1.0
            bed_color: tuple = (0.78, 0.66, 0.42)
            foot_color: tuple = (0.85, 0.72, 0.30)
            crank_color: tuple = (0.90, 0.45, 0.10)
            knob_color: tuple = (0.95, 0.55, 0.15)
            contact_offset: float = 0.0015
            struct_mu_s: float = 0.45
            struct_mu_d: float = 0.40

        _SPAWNER_CACHE["vault"] = VaultSpawnerCfg
        _SPAWNER_CACHE["cradle"] = CradleSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class LetterboxCradleSceneCfg(BaseCfg):
    """Config for `LetterboxCradleScene`. The design contract is asserted in
    `__post_init__`: the slot really blocks the upright bottle and admits the lying
    one; the axle port really blocks the bottle; the cradle's swing really clears
    floor, roof, walls and the front wall; the erected bottle really fits (base
    fully on the end plate, clearance to the back wall); the over-centre angles
    (empty and loaded) really lie inside the crank's travel with margin; the loaded
    erected state really presses the 90-degree stop; and the insertion push can
    never torque the resting cradle off its stop."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    upright_max_deg: float = tunable(15.0)  # bottle axis within this of world-up
    laid_max_upz: float = tunable(0.35)     # |axis dot z| below this counts as lying
    settle_lin: float = tunable(0.08)       # max bottle |lin vel| when judging (m/s)
    settle_ang: float = tunable(0.50)       # max cradle |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    stage_x: tuple = tunable((0.26, 0.38))  # bottle standing spawn x band on the apron
    stage_y: tuple = tunable((-0.10, 0.10))  # bottle standing spawn y band

    # --- info: bottle ----------------------------------------------------------------------------
    bottle_r: float = info(0.030)
    bottle_l: float = info(0.160)
    bottle_mass: float = info(0.15)
    # --- info: vault (local frame: x=0 front outer face, +x toward robot, z=0 ground) ------------
    back_x1: float = info(-0.32)
    back_x0: float = info(-0.30)          # back wall interior face
    wall_y0: float = info(0.14)           # chamber interior half-width
    wall_y1: float = info(0.16)
    floor_z1: float = info(0.45)          # chamber floor top
    roof_z0: float = info(0.70)           # roof interior face
    roof_z1: float = info(0.72)
    front_x0: float = info(-0.02)         # front wall interior face
    slot_hw: float = info(0.075)          # letter slot half-width
    slot_top: float = info(0.562)         # letter slot header (bottom face)
    apron_x1: float = info(0.44)
    apron_z1: float = info(0.472)         # apron top = the sliding plane
    fence_x1: float = info(0.20)
    fence_y0: float = info(0.036)         # guide lane inner half-gap
    fence_y1: float = info(0.052)
    fence_h: float = info(0.017)
    port_half: float = info(0.022)        # axle port half-size in the +y wall
    # --- info: cradle (root/hinge at (hinge_x, 0, hinge_z); revolute axis Y) ---------------------
    hinge_x: float = info(-0.195)
    hinge_z: float = info(0.474)
    bed_x1: float = info(0.170)           # bed reach from the hinge (cradle-local +x)
    bed_hy: float = info(0.055)
    rail_y0: float = info(0.034)          # rail inner faces: the lying bottle's lane
    rail_y1: float = info(0.048)
    rail_top: float = info(0.026)
    foot_x1: float = info(0.012)          # end plate thickness (its face seats the base)
    foot_hy: float = info(0.048)
    foot_top: float = info(0.079)         # end plate reach (the erected floor half-size)
    axle_r: float = info(0.010)           # < foot_x1: sunk below the end-plate face plane
    arm_len: float = info(0.20)           # crank arm length (knob at arm_len..+0.04)
    arm_y0: float = info(0.185)
    arm_y1: float = info(0.210)
    cradle_mass: float = info(0.35)
    com_x: float = info(0.06)             # authored CoM: the whole bistability mechanism
    com_z: float = info(0.03)
    phi_max_deg: float = info(90.0)       # revolute travel (joint limits are the stops)
    # --- info: rubric boxes / weights ------------------------------------------------------------
    inside_x1: float = info(-0.075)       # bottle root past this = fully through the slot
    ride_x: tuple = info((0.02, 0.18))    # bottle root, cradle-local: riding the bed
    ride_hy: float = info(0.05)
    ride_z: tuple = info((-0.01, 0.09))
    w_laid: float = info(0.15)
    w_inside: float = info(0.35)
    w_erect: float = info(0.40)
    # --- info: interaction bounds (asserted, used by solve/smoke servos) -------------------------
    push_f_max: float = info(4.5)         # insertion push force cap (N)
    crank_tau_max: float = info(1.2)      # crank torque cap (N.m) — fingertip effort at 0.2 m
    contact_offset: float = info(0.0015)
    struct_mu_s: float = info(0.45)
    struct_mu_d: float = info(0.40)

    def __post_init__(self) -> None:
        d, length = 2 * self.bottle_r, self.bottle_l
        slot_h = self.slot_top - self.apron_z1
        # aperture contract: lying passes with margin, upright can never pass
        assert slot_h >= d + 0.025, f"slot must admit the lying bottle ({slot_h:.3f})"
        assert 2 * self.slot_hw >= d + 0.06, "slot must admit the lying bottle + fingers"
        assert length >= slot_h + 0.06, "the standing bottle must be far too tall for the slot"
        # axle port never passes the bottle
        assert 2 * self.port_half <= d - 0.012, "axle port must block the bottle"
        assert self.axle_r + 0.008 <= self.port_half, "axle must clear its port"
        # the axle surface is coaxial with the hinge (it never moves): it must sit
        # below the end-plate face plane or the erecting bottle's base rim catches it
        assert self.axle_r <= self.foot_x1 - 0.0015, "axle must sit below the plate face"
        # sliding plane: bed top just BELOW the apron top (no proud lip to catch on)
        bed_top = self.hinge_z - 0.004
        assert 0.0005 <= self.apron_z1 - bed_top <= 0.004, "bed top ~2mm below the apron top"
        # bed hangs above the chamber floor (rests on the JOINT stop, not on geometry)
        assert self.hinge_z - 0.016 >= self.floor_z1 + 0.006, "bed must clear the floor"
        # bed tip clears the front wall; the lying bottle fits fully inside the chamber
        assert self.hinge_x + self.bed_x1 <= self.front_x0 - 0.004, "bed tip clears the wall"
        base_x = self.hinge_x + self.foot_x1  # lying bottle base plane (world)
        assert base_x + length <= self.front_x0 - 0.0025, "the seated bottle clears the slot"
        assert length - (self.bed_x1 - self.foot_x1) <= 0.01, "bed carries all but ~mm of it"
        # rails lane the bottle
        assert 2 * self.rail_y0 >= d + 0.006 and 2 * self.rail_y0 <= d + 0.014
        assert self.fence_y0 * 2 >= d + 0.008 and self.fence_y0 + 0.004 < self.slot_hw
        # swing clearance: bottle tip arc + rail corner arc under the roof, inside walls
        tip_r = math.hypot(self.foot_x1 + length, 0.026 + self.bottle_r * 2)
        assert self.hinge_z + tip_r <= self.roof_z0 - 0.02, f"swing apex {tip_r:.3f} hits roof"
        assert self.bed_hy + 0.004 <= self.wall_y0 - 0.06, "swing clear of the side walls"
        # erected bottle: centred on the end plate, clear of the back wall, under the roof
        ex = self.hinge_x - 0.026            # erected bottle centre x (bed face - r)
        assert ex - self.bottle_r >= self.back_x0 + 0.02, "erected bottle clears the back wall"
        # lying bottle spans cradle-local z [-0.004, -0.004 + d]; the end plate must
        # reach past its far flank with margin so the erected base is fully supported
        assert 2 * self.foot_hy >= d + 0.03 and self.foot_top >= -0.004 + d + 0.02, \
            "end plate must fully support the erected base"
        top = self.hinge_z + self.foot_x1 + length
        assert top <= self.roof_z0 - 0.04, "erected bottle clears the roof"
        # erected pose lands inside the rubric boxes
        assert self.back_x0 < ex < self.inside_x1 - 0.05
        assert self.ride_x[0] < self.foot_x1 + length / 2 < self.ride_x[1]
        # over-centre: empty and loaded tip-over angles inside the travel, with margin
        cc = self.cradle_mass * self.com_x
        cs = self.cradle_mass * self.com_z
        bx, bz = self.foot_x1 + length / 2, -0.004 + self.bottle_r
        lc = cc + self.bottle_mass * bx
        ls = cs + self.bottle_mass * bz
        phi_empty = math.degrees(math.atan2(cc, cs))
        phi_loaded = math.degrees(math.atan2(lc, ls))
        assert 55.0 <= phi_empty <= 72.0, f"empty over-centre {phi_empty:.1f} out of band"
        assert 60.0 <= phi_loaded <= 78.0, f"loaded over-centre {phi_loaded:.1f} out of band"
        assert self.phi_max_deg - phi_loaded >= 10.0, "release band above over-centre"
        # erected stop-press margin (both loaded and empty press the 90-deg stop)
        tau_press = _G * ls  # at phi=90 the gravity torque is g*(s-coef), pressing the stop
        assert tau_press >= 0.10, f"erected stop-press margin {tau_press:.3f} too small"
        assert _G * cs >= 0.05, "empty cradle must also hold its erected stop"
        # the insertion push can never lift the resting cradle off its rest stop:
        # worst-case moment arm = bottle centre height above the hinge
        tau_push = self.push_f_max * (bz + 0.008)
        tau_hold = _G * lc  # holding torque at rest (phi=0)
        assert tau_push <= 0.5 * tau_hold, "push force could unseat the resting cradle"
        # crank authority: full torque >= 2x the worst gravity lift torque on the travel
        assert self.crank_tau_max >= 2.0 * _G * math.hypot(lc, ls), "crank torque margin too small"
        # staging band: on the apron, clear of the fences and the vault front
        assert self.stage_x[0] >= self.fence_x1 + 0.03
        assert self.stage_x[1] + self.bottle_r <= self.apron_x1 - 0.02
        assert self.stage_y[1] + self.bottle_r <= self.wall_y1 - 0.02
        assert abs(self.w_laid + self.w_inside + self.w_erect - 0.90) < 1e-9


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("letterbox_cradle_vault")
class LetterboxCradleScene(BaseScene):
    cfg: LetterboxCradleSceneCfg

    def __init__(self, cfg: LetterboxCradleSceneCfg | None = None) -> None:
        super().__init__(cfg or LetterboxCradleSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        vault_spawn = cls["vault"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            back_x1=c.back_x1, back_x0=c.back_x0, wall_y0=c.wall_y0, wall_y1=c.wall_y1,
            floor_z1=c.floor_z1, roof_z0=c.roof_z0, roof_z1=c.roof_z1, front_x0=c.front_x0,
            slot_hw=c.slot_hw, slot_top=c.slot_top, apron_x1=c.apron_x1, apron_z1=c.apron_z1,
            fence_x1=c.fence_x1, fence_y0=c.fence_y0, fence_y1=c.fence_y1, fence_h=c.fence_h,
            hinge_x=c.hinge_x, hinge_z=c.hinge_z, port_half=c.port_half,
            contact_offset=c.contact_offset, struct_mu_s=c.struct_mu_s,
            struct_mu_d=c.struct_mu_d)
        cradle_spawn = cls["cradle"](
            bed_x1=c.bed_x1, bed_hy=c.bed_hy, rail_y0=c.rail_y0, rail_y1=c.rail_y1,
            rail_top=c.rail_top, foot_x1=c.foot_x1, foot_hy=c.foot_hy, foot_top=c.foot_top,
            axle_r=c.axle_r, arm_y0=c.arm_y0, arm_y1=c.arm_y1, arm_len=c.arm_len,
            mass=c.cradle_mass, com_x=c.com_x, com_z=c.com_z,
            contact_offset=c.contact_offset, struct_mu_s=c.struct_mu_s,
            struct_mu_d=c.struct_mu_d)

        bottle_spawn = sim_utils.CylinderCfg(
            radius=c.bottle_r, height=c.bottle_l, axis="Z",
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5, linear_damping=0.05, angular_damping=0.10,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32, solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.40, dynamic_friction=0.35, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.10, 0.08)))

        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "vault": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Vault", spawn=vault_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            # The cradle is authored IN PLACE at its rest pose: the bind-time joint
            # anchors at this authored position (root origin = the hinge).
            "cradle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cradle", spawn=cradle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(c.hinge_x, 0.0, c.hinge_z))),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle", spawn=bottle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.33, 0.0, c.apron_z1 + c.bottle_l / 2 + 0.002))),
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
        self.vault: RigidObject = env.iscene["vault"]
        self.cradle: RigidObject = env.iscene["cradle"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # readbacks (verified by smoke)
        self.stage_xy = torch.zeros(n, 2, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._flaid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._finside = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ferect = torch.zeros(n, device=dev)  # max phi/90 fraction while riding
        self._author_joints()

    def _author_joints(self) -> None:
        """Per-env revolute joint, authored ONCE at bind time against the AUTHORED
        poses (the cradle root IS its hinge). Joint LIMITS are the hard stops:
        [-phi_max, 0] deg about +Y (0 = bed-horizontal rest; -90 = erected).
        Collision between vault and cradle is joint-filtered — the stops are the
        limits and the axle rides its wall port without touching."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        c = self.cfg
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            r = UsdPhysics.RevoluteJoint.Define(stage, f"{base}/cradle_hinge")
            r.CreateBody0Rel().SetTargets([f"{base}/Vault"])
            r.CreateBody1Rel().SetTargets([f"{base}/Cradle"])
            r.CreateCollisionEnabledAttr(False)
            r.CreateAxisAttr("Y")
            r.CreateLocalPos0Attr(Gf.Vec3f(float(c.hinge_x), 0.0, float(c.hinge_z)))
            r.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            r.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            r.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            # theta about +Y in [-90, 0]; gravity presses theta positive -> rests at 0.
            r.CreateLowerLimitAttr(-float(c.phi_max_deg))
            r.CreateUpperLimitAttr(0.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: vault re-asserted at its fixed pose, cradle at rest on its
        0-degree stop, bottle STANDING UPRIGHT at a sampled apron pose (x, y, yaw),
        latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        _ = torch.rand(4, device=dev)  # burn: the first post-seed draw is degenerate

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
        place(self.vault, zeros, zeros, zeros)
        place(self.cradle, zeros + c.hinge_x, zeros, zeros + c.hinge_z)

        u = torch.rand(m, 3, device=dev)
        bx = c.stage_x[0] + u[:, 0] * (c.stage_x[1] - c.stage_x[0])
        by = c.stage_y[0] + u[:, 1] * (c.stage_y[1] - c.stage_y[0])
        self.stage_xy[env_ids, 0] = bx
        self.stage_xy[env_ids, 1] = by
        place(self.bottle, bx, by, zeros + c.apron_z1 + c.bottle_l / 2 + 0.002,
              yaw=(u[:, 2] * 2 - 1) * math.pi)

        self._flaid[env_ids] = False
        self._finside[env_ids] = False
        self._ferect[env_ids] = 0.0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "vault": self.vault.data.root_state_w[env_ids].clone(),
            "cradle": self.cradle.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "stage_xy": self.stage_xy[env_ids].clone(),
            "flaid": self._flaid[env_ids].clone(),
            "finside": self._finside[env_ids].clone(),
            "ferect": self._ferect[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.vault.write_root_state_to_sim(state["vault"], env_ids)
        self.cradle.write_root_state_to_sim(state["cradle"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.stage_xy[env_ids] = state["stage_xy"]
        self._flaid[env_ids] = state["flaid"]
        self._finside[env_ids] = state["finside"]
        self._ferect[env_ids] = state["ferect"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        slot_h = (c.slot_top - c.apron_z1) * 100
        return (
            f"A sealed grey steel VAULT stands on a plinth at counter height. Its only "
            f"opening is a wide, low LETTER SLOT in the front wall "
            f"({2 * c.slot_hw * 100:.0f} cm wide, {slot_h:.0f} cm tall), level with a "
            f"wooden APRON shelf in front of it; two dark GUIDE FENCES on the apron form "
            f"a straight lane into the slot. Visible through the slot, a tan TILT CRADLE "
            f"hangs on a horizontal axle inside: a flat bed with low side rails, "
            f"pointing at the slot, ending at an amber END PLATE by the axle. The axle "
            f"passes through a small port in the RIGHT side wall to an orange CRANK ARM "
            f"with a knob, lying horizontal and pointing forward.\n"
            f"A red SAUCE BOTTLE ({2 * c.bottle_r * 100:.0f} cm across, "
            f"{c.bottle_l * 100:.0f} cm tall) stands upright on the apron; its position "
            f"varies by episode.\n"
            f"Goal: the bottle standing UPRIGHT (within {c.upright_max_deg:.0f} deg of "
            f"vertical) INSIDE the vault, at rest. It is far too tall to pass the slot "
            f"standing, and no hand fits inside — so: (1) LAY the bottle on its side in "
            f"the guide lane, base toward the slot; (2) SLIDE it base-first through the "
            f"slot onto the cradle bed until its base seats against the end plate; "
            f"(3) RAISE the orange crank arm through its quarter-turn arc. Past roughly "
            f"two-thirds of the arc the loaded cradle tips over-centre and falls onto "
            f"its 90-degree stop by itself, standing the bottle on the end plate — "
            f"upright inside the vault. Release the crank BELOW that tipping angle and "
            f"the cradle drops back, laying the bottle down again: carry the crank past "
            f"the tip-over before letting go. Order is forced: cranking the empty "
            f"cradle first leaves the bottle stranded lying on the vault floor with the "
            f"bed vertical — nothing left to erect it."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Lay the red bottle on its side in the guide lane, slide it base-first "
            "through the letter slot onto the tilt cradle, then raise the orange crank "
            "past the tip-over angle so the cradle erects and leaves the bottle "
            "standing upright inside the vault. The bottle cannot enter upright."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, body) -> torch.Tensor:
        """(N,3) body root position in the vault frame (vault fixed at the env
        origin with identity heading)."""
        return body.data.root_pos_w - self.env_origins

    def bottle_up(self) -> torch.Tensor:
        """(N,3) the bottle's body +z axis in world coordinates."""
        from isaaclab.utils.math import quat_apply

        ez = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        ez[:, 2] = 1.0
        return quat_apply(self.bottle.data.root_quat_w, ez)

    def bottle_upright(self) -> torch.Tensor:
        """(N,) bool: bottle axis within `upright_max_deg` of world-up."""
        return self.bottle_up()[:, 2].clamp(-1.0, 1.0) \
            >= math.cos(math.radians(self.cfg.upright_max_deg))

    def bottle_lying(self) -> torch.Tensor:
        """(N,) bool: bottle axis within ~20 deg of horizontal."""
        return self.bottle_up()[:, 2].abs() <= self.cfg.laid_max_upz

    def bottle_inside(self) -> torch.Tensor:
        """(N,) bool: bottle root fully through the slot, in the chamber box."""
        c = self.cfg
        p = self._local(self.bottle)
        return (p[:, 0] > c.back_x0) & (p[:, 0] < c.inside_x1) \
            & (p[:, 1].abs() < c.wall_y0 - 0.01) \
            & (p[:, 2] > c.floor_z1 - 0.01) & (p[:, 2] < c.roof_z0)

    def phi(self) -> torch.Tensor:
        """(N,) cradle erection angle in rad (0 = rest stop, +pi/2 = erected)."""
        q = self.cradle.data.root_quat_w
        th = 2.0 * torch.atan2(q[:, 2], q[:, 0])
        th = (th + math.pi) % (2.0 * math.pi) - math.pi
        return -th

    def cradle_on_hinge(self) -> torch.Tensor:
        """(N,) bool: cradle root still at its hinge (guards teleported fakes)."""
        c = self.cfg
        tgt = self.env_origins.clone()
        tgt[:, 0] += c.hinge_x
        tgt[:, 2] += c.hinge_z
        return (self.cradle.data.root_pos_w - tgt).norm(dim=-1) < 0.02

    def bottle_riding(self) -> torch.Tensor:
        """(N,) bool: bottle root inside the cradle-local riding box (on the bed,
        between the rails) — the guard on erect credit."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        p = quat_apply_inverse(
            self.cradle.data.root_quat_w,
            self.bottle.data.root_pos_w - self.cradle.data.root_pos_w)
        return (p[:, 0] > c.ride_x[0]) & (p[:, 0] < c.ride_x[1]) \
            & (p[:, 1].abs() < c.ride_hy) \
            & (p[:, 2] > c.ride_z[0]) & (p[:, 2] < c.ride_z[1]) \
            & self.cradle_on_hinge()

    def settled(self) -> torch.Tensor:
        """(N,) bool: bottle slow (lin) and cradle slow (ang — its CoM rides near
        the hinge, so linear velocity is small even mid-swing)."""
        c = self.cfg
        return (self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bottle.data.root_ang_vel_w.norm(dim=-1) < 1.0) \
            & (self.cradle.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.bottle.data.root_pos_w, self.cradle.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        lin_ok = self.bottle.data.root_lin_vel_w.norm(dim=-1) < 0.15
        self._flaid |= self.bottle_lying() & lin_ok & fin
        self._finside |= self.bottle_inside() & fin
        fr = (self.phi() / math.radians(c.phi_max_deg)).clamp(0.0, 1.0)
        fr = torch.where(self.bottle_riding() & self.bottle_inside() & fin,
                         fr, torch.zeros_like(fr))
        self._ferect = torch.maximum(self._ferect, fr)

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool, judged LIVE on the bottle's settled physical pose: upright
        within tolerance, root inside the chamber box (fully through the slot),
        settled and finite. The aperture makes this state reachable only through
        the lay-slide-erect plan; the state itself is what is judged."""
        self._update_latches()
        return self.bottle_upright() & self.bottle_inside() \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.15 laid (latched) + 0.35 inside (latched) +
        0.40 x latched max erect fraction while the bottle rides the cradle,
        capped at 0.90; exactly 1.0 iff success() holds live. Null ~0. The
        seed's strategy (carry the upright bottle to the receptacle) earns ~0."""
        c = self.cfg
        self._update_latches()
        base = (c.w_laid * self._flaid.float() + c.w_inside * self._finside.float()
                + c.w_erect * self._ferect).clamp(max=0.90)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="letterbox_cradle_vault", robot="null"))
