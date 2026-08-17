"""RoofShuttleScene — slide the cabinet's only roof panel off the bowl bay, then put
the black bowl on top of that panel (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet_i371`).

Derived from libero_90/kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet, but
STRATEGICALLY different: the seed is one pick-and-place of a freely accessible bowl
onto a cabinet's FIXED flat top. Here the cabinet has exactly ONE top surface — a
captive SLIDING ROOF PANEL riding in rails along the cabinet's long axis — and at
reset that panel is parked over the FRONT compartment, sealing the black bowl
inside it (the bowl starts CAPTIVE and untouchable; the seed's bowl is free on the
table from t=0). The rear half of the cabinet is an open, topless shaft with a
deep-red floor: anything dropped "on top of the cabinet" there just falls 30 cm
into the shaft. A single prismatic slide of the panel does BOTH halves of the work
at once — it uncovers the bowl bay (releasing the source object) and carries the
cabinet's only top surface clear of it (providing the destination). The plan is
therefore inverted relative to the seed: actuate the roof itself, extract the
now-reachable bowl, and stand it upright on the very panel that was just moved.
Ordering is forced by geometry, not by a declared rule: the bowl physically cannot
be on the panel's top face without first having left the bay, and it cannot leave
the bay while the panel roofs it (smoke proves the roofed extraction fails under
real force).

Assets are fully procedural (compound-spawner pattern; children of one body never
self-collide):
  - cabinet: KINEMATIC compound (outer walls, mid partition, side-rail curbs, bay
    and shaft floor plates). Fixed pose, axis-aligned; the shaft floor is deep red
    so the topless half reads visually.
  - roof panel ("slab"): DYNAMIC plate with a raised yellow handle bar, suspended
    2 mm above the wall tops by a Y-axis PRISMATIC joint to the cabinet (limits
    [0, travel]; joint collision-filtered, so the suspension gap never chatters).
    The joint is the rail; the curbs are visual insurance.
  - bowl: DYNAMIC compound (cylinder base disc + 8 octagonal rim wall boxes),
    matte black, 104 mm across, 50 mm tall — the rim fits a parallel jaw.
  - pedestal: KINEMATIC riser block inside the bay (re-posed per reset) so the
    bowl presents at a comfortable grasp height once the roof is open.

Per-episode randomization (readback-verified by smoke): pedestal xy inside the
bay, bowl xy on the pedestal + free yaw, panel start offset along its rails.

Rubric (0..1; latched credit anchored in the demonstrated solve trajectory):
  0.25  slide  — latched max normalized panel travel from the front stop
  0.15  open   — latched: bay uncovered wide enough to pass the bowl (latched)
  0.25  out    — latched: bowl ever fully out of the bay volume
  0.20  seated — latched: bowl upright, slow, resting on the panel's top face
capped at 0.85; exactly 1.0 iff success(): bowl upright and settled ON the roof
panel's top face (any rail position), panel slow, everything finite. Null policy
~0 (panel jitter is a few mm of its 290 mm travel); the seed's strategy — carry
the bowl to the cabinet top — cannot even begin (the bowl is sealed in), and its
nearest analog (drop something over the open rear half) scores no seat credit.

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


# ----- USD authoring helpers ---------------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _define_root(prim_path: str, translation, orientation):
    """Define an Xform root and author its (single, idempotent) translate/orient ops."""
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


def _collider(contact_offset: float) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def apply(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return apply


def _box(stage, path: str, *, x, y, z, color, collide: Callable):
    """Axis-aligned box child from (lo, hi) spans per axis."""
    from pxr import Gf, UsdGeom

    b = UsdGeom.Cube.Define(stage, path)
    b.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(b.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(b.GetPrim())
    return b.GetPrim()


def _yaw_box(stage, path: str, *, center, size, yaw_deg: float, color, collide: Callable):
    """Box child rotated about z (for the bowl's octagonal rim)."""
    from pxr import Gf, UsdGeom

    b = UsdGeom.Cube.Define(stage, path)
    b.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(b.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    half = math.radians(yaw_deg) / 2
    xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    b.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(b.GetPrim())
    return b.GetPrim()


def _bind_material(prim_path: str, name: str, mu_s: float, mu_d: float) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material

    mat = f"{prim_path}/{name}"
    sim_utils.spawn_rigid_body_material(mat, sim_utils.RigidBodyMaterialCfg(
        static_friction=float(mu_s), dynamic_friction=float(mu_d), restitution=0.0))
    bind_physics_material(prim_path, mat)


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """KINEMATIC cabinet carcass. Local frame: origin at the footprint centre on the
    ground; +y runs front (-y) to back (+y). Front half = the bowl bay (dark floor
    plate); back half = the topless shaft (deep-red floor plate); side walls carry
    thin rail curbs above the rim plane."""
    from pxr import UsdPhysics

    stage, root = _define_root(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _collider(cfg.contact_offset)
    c = cfg
    X, Y, H, t = c.cab_hx, c.cab_hy, c.cab_h, c.wall_t
    body, dark = c.body_color, c.frame_color

    _box(stage, f"{prim_path}/wall_xp", x=(X - t, X), y=(-Y, Y), z=(0.0, H), color=body, collide=collide)
    _box(stage, f"{prim_path}/wall_xn", x=(-X, -X + t), y=(-Y, Y), z=(0.0, H), color=body, collide=collide)
    _box(stage, f"{prim_path}/wall_yn", x=(-X, X), y=(-Y, -Y + t), z=(0.0, H), color=body, collide=collide)
    _box(stage, f"{prim_path}/wall_yp", x=(-X, X), y=(Y - t, Y), z=(0.0, H), color=body, collide=collide)
    _box(stage, f"{prim_path}/partition", x=(-X + t, X - t), y=(-c.part_hw, c.part_hw),
         z=(0.0, H), color=dark, collide=collide)
    # floor plates: bay (dark) and shaft (deep red — the visible "no top here" cue)
    _box(stage, f"{prim_path}/bay_floor", x=(-X + t, X - t), y=(-Y + t, -c.part_hw),
         z=(0.0, c.floor_t), color=(0.15, 0.15, 0.18), collide=collide)
    _box(stage, f"{prim_path}/shaft_floor", x=(-X + t, X - t), y=(c.part_hw, Y - t),
         z=(0.0, c.floor_t), color=c.shaft_color, collide=collide)
    # rail curbs above the rim plane (visual insurance; the joint is the real rail)
    _box(stage, f"{prim_path}/curb_xp", x=(c.curb_x0, X), y=(-Y, Y),
         z=(H, H + c.curb_h), color=dark, collide=collide)
    _box(stage, f"{prim_path}/curb_xn", x=(-X, -c.curb_x0), y=(-Y, Y),
         z=(H, H + c.curb_h), color=dark, collide=collide)
    _bind_material(prim_path, "mat", 0.60, 0.55)
    return root


def _spawn_slab(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC roof panel. Local frame: origin at the plate centre; the raised
    yellow handle bar (two posts + crossbar, finger gap underneath) sits near the
    local -y edge (the cabinet-front edge at the parked pose)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _define_root(prim_path, translation, orientation)
    collide = _collider(cfg.contact_offset)
    c = cfg
    hx, hy, ht = c.slab_hx, c.slab_hy, c.slab_ht
    _box(stage, f"{prim_path}/plate", x=(-hx, hx), y=(-hy, hy), z=(-ht, ht),
         color=c.slab_color, collide=collide)
    y0, y1 = c.bar_y - c.bar_hw, c.bar_y + c.bar_hw
    _box(stage, f"{prim_path}/post_p", x=(0.05, 0.07), y=(y0, y1),
         z=(ht, c.bar_z0), color=c.bar_color, collide=collide)
    _box(stage, f"{prim_path}/post_n", x=(-0.07, -0.05), y=(y0, y1),
         z=(ht, c.bar_z0), color=c.bar_color, collide=collide)
    _box(stage, f"{prim_path}/bar", x=(-0.08, 0.08), y=(y0, y1),
         z=(c.bar_z0, c.bar_z1), color=c.bar_color, collide=collide)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.slab_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(float(c.slab_damping))
    prb.CreateAngularDampingAttr(2.0)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    _bind_material(prim_path, "mat", 0.55, 0.50)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """DYNAMIC black bowl: cylinder base disc + 8 octagonal rim wall boxes. Local
    frame: origin 15 mm above the base bottom (the authored CoM sits there)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _define_root(prim_path, translation, orientation)
    collide = _collider(cfg.contact_offset)
    c = cfg
    zb = -c.bowl_org  # base bottom in the body frame
    base = UsdGeom.Cylinder.Define(stage, f"{prim_path}/base")
    base.CreateRadiusAttr(float(c.bowl_base_r))
    base.CreateHeightAttr(float(c.bowl_base_t))
    base.CreateAxisAttr("Z")
    base.CreateExtentAttr([Gf.Vec3f(-c.bowl_base_r, -c.bowl_base_r, -c.bowl_base_t / 2),
                           Gf.Vec3f(c.bowl_base_r, c.bowl_base_r, c.bowl_base_t / 2)])
    UsdGeom.Xformable(base.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, zb + c.bowl_base_t / 2))
    base.CreateDisplayColorAttr([Gf.Vec3f(*c.bowl_color)])
    collide(base.GetPrim())
    rm = c.bowl_r - c.bowl_wall_t / 2  # rim wall mid radius
    seg_l = 2.0 * rm * math.tan(math.pi / 8) + 0.004  # tangential, slight overlap
    z0 = zb + c.bowl_base_t
    for k in range(8):
        ang = k * 45.0
        a = math.radians(ang)
        _yaw_box(stage, f"{prim_path}/rim_{k}",
                 center=(rm * math.cos(a), rm * math.sin(a), (z0 + (zb + c.bowl_h)) / 2),
                 size=(c.bowl_wall_t, seg_l, c.bowl_h - c.bowl_base_t),
                 yaw_deg=ang, color=c.bowl_color, collide=collide)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(c.bowl_mass))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(1)
    prb.CreateLinearDampingAttr(0.10)
    prb.CreateAngularDampingAttr(0.10)
    prb.CreateSleepThresholdAttr(0.0)
    prb.CreateStabilizationThresholdAttr(0.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.5)
    _bind_material(prim_path, "mat", 0.60, 0.55)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "cabinet" not in _SPAWNER_CACHE:

        @configclass
        class CabinetSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_cabinet)
            cab_hx: float = 0.20
            cab_hy: float = 0.30
            cab_h: float = 0.30
            wall_t: float = 0.02
            part_hw: float = 0.01
            floor_t: float = 0.008
            curb_x0: float = 0.193
            curb_h: float = 0.04
            body_color: tuple = (0.42, 0.44, 0.50)
            frame_color: tuple = (0.30, 0.32, 0.38)
            shaft_color: tuple = (0.50, 0.10, 0.10)
            contact_offset: float = 0.0015

        @configclass
        class SlabSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_slab)
            slab_hx: float = 0.19
            slab_hy: float = 0.155
            slab_ht: float = 0.008
            bar_y: float = -0.125
            bar_hw: float = 0.012
            bar_z0: float = 0.043
            bar_z1: float = 0.063
            slab_mass: float = 1.2
            slab_damping: float = 3.0
            slab_color: tuple = (0.72, 0.58, 0.35)
            bar_color: tuple = (0.85, 0.72, 0.12)
            contact_offset: float = 0.0015

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            bowl_r: float = 0.055
            bowl_base_r: float = 0.052
            bowl_base_t: float = 0.012
            bowl_wall_t: float = 0.009
            bowl_h: float = 0.050
            bowl_org: float = 0.015
            bowl_mass: float = 0.30
            bowl_color: tuple = (0.05, 0.05, 0.06)
            contact_offset: float = 0.0015

        _SPAWNER_CACHE["cabinet"] = CabinetSpawnerCfg
        _SPAWNER_CACHE["slab"] = SlabSpawnerCfg
        _SPAWNER_CACHE["bowl"] = BowlSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg ---------------------------------------------------------------------------------
@dataclass
class RoofShuttleSceneCfg(BaseCfg):
    """Config for `RoofShuttleScene`. The captivity/coverage/trap contract is
    asserted in `__post_init__` so a bad parameter change fails at import, not on
    the forge."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    settle_lin: float = tunable(0.06)      # max |lin vel| (bowl AND panel) when judging (m/s)
    settle_ang: float = tunable(0.50)      # max |ang vel| (bowl) when judging (rad/s)
    upright_max_deg: float = tunable(10.0)  # bowl axis within this of world-up
    seat_z_lo: float = tunable(-0.005)     # bowl base bottom minus panel top: rest band (m)
    seat_z_hi: float = tunable(0.010)
    seat_margin: float = tunable(0.012)    # bowl centre this far inside the panel footprint (m)
    open_pass: float = tunable(0.13)       # uncovered bay width that counts as "open" (m)

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    ped_jitter: float = tunable(0.02)      # pedestal xy jitter inside the bay (+/- m)
    bowl_jitter: float = tunable(0.02)     # bowl xy jitter on the pedestal (+/- m)
    slab_jitter: float = tunable(0.012)    # panel start offset along the rails (0..this, m)

    # --- info: cabinet (local frame: origin at the footprint centre on the ground) ---------------
    cab_hx: float = info(0.20)             # outer half-extent, x (rail-normal)
    cab_hy: float = info(0.30)             # outer half-extent, y (rail axis; -y = front)
    cab_h: float = info(0.30)              # wall-top rim plane
    wall_t: float = info(0.02)
    part_hw: float = info(0.01)            # mid-partition half-width
    floor_t: float = info(0.008)
    curb_x0: float = info(0.193)           # curb inner face (3 mm lateral panel clearance)
    curb_h: float = info(0.04)
    # --- info: roof panel + rail (prismatic joint) -----------------------------------------------
    slab_hx: float = info(0.19)
    slab_hy: float = info(0.155)
    slab_ht: float = info(0.008)           # plate half-thickness
    slab_gap: float = info(0.002)          # suspension gap above the wall tops
    slab_mass: float = info(1.2)
    slab_damping: float = info(3.0)        # kills coasting after the push lets go
    travel: float = info(0.29)             # prismatic limits [0, travel]; 0 = parked front
    bar_y: float = info(-0.125)            # handle bar centre, panel-local y
    bar_z1: float = info(0.063)            # bar top above the plate centre
    # --- info: bowl ------------------------------------------------------------------------------
    bowl_r: float = info(0.055)            # rim outer radius
    bowl_base_r: float = info(0.052)
    bowl_base_t: float = info(0.012)
    bowl_wall_t: float = info(0.009)
    bowl_h: float = info(0.050)
    bowl_org: float = info(0.015)          # body origin above the base bottom
    bowl_mass: float = info(0.30)
    # --- info: pedestal --------------------------------------------------------------------------
    ped_half: float = info(0.08)           # square half-side
    ped_top: float = info(0.16)            # riser top above the ground
    # --- info: rubric weights (sum = 0.85 = the non-success cap) ---------------------------------
    w_slide: float = info(0.25)
    w_open: float = info(0.15)
    w_out: float = info(0.25)
    w_seat: float = info(0.20)
    contact_offset: float = info(0.0015)

    def __post_init__(self) -> None:
        c = self
        bay_y0, bay_y1 = -c.cab_hy + c.wall_t, -c.part_hw   # bay opening span
        slab_front_c = -c.cab_hy + c.slab_hy                # parked panel centre
        # coverage: the parked panel roofs the whole bay opening with margin > start jitter
        front_margin = bay_y0 - (slab_front_c - c.slab_hy)
        back_margin = (slab_front_c + c.slab_hy) - bay_y1
        assert front_margin >= 0.015 and back_margin >= 0.015, "parked panel must cover the bay"
        assert c.slab_jitter < min(front_margin, back_margin) - 0.002, \
            "start jitter must not uncover the bay"
        # captivity: bowl on its pedestal fits UNDER the suspended panel
        slab_z0 = c.cab_h + c.slab_gap
        assert slab_z0 > c.ped_top + c.bowl_h + 0.02, "roofed bowl must clear the panel underside"
        # grasp room once open: >= 40 mm above the bowl inside the bay
        assert slab_z0 - c.ped_top - c.bowl_h >= 0.04
        # passage: full travel uncovers the bay far wider than the bowl
        uncovered_max = (slab_front_c + c.travel - c.slab_hy) - bay_y0
        assert uncovered_max > 2 * c.bowl_r + 0.06, "open bay must pass the bowl with margin"
        assert c.open_pass > 2 * c.bowl_r + 0.015, "open latch must certify a real passage"
        # trap: the shaft opening swallows the bowl; its rim stands far above a fallen bowl
        assert (c.cab_hy - c.wall_t) - c.part_hw > 2 * c.bowl_r + 0.05
        assert 2 * (c.cab_hx - c.wall_t) > 2 * c.bowl_r + 0.05
        assert c.cab_h > c.floor_t + c.bowl_h + 0.10, "shaft must be deep: a fallen bowl is not on top"
        # rails: panel clears the curbs laterally; pedestal + jitter stays inside the bay
        assert c.slab_hx < c.curb_x0 - 0.002
        assert c.ped_half + c.ped_jitter < c.cab_hx - c.wall_t - 0.01
        assert c.ped_half + c.ped_jitter < (bay_y1 - bay_y0) / 2 - 0.01
        assert c.bowl_base_r + c.bowl_jitter < c.ped_half, "jittered bowl stays on the pedestal"
        # handle: finger gap under the crossbar
        assert 0.043 - c.slab_ht >= 0.030, "under-bar finger gap"
        assert abs(c.w_slide + c.w_open + c.w_out + c.w_seat - 0.85) < 1e-9


# ----- scene -------------------------------------------------------------------------------------
@SCENES.register("roof_shuttle")
class RoofShuttleScene(BaseScene):
    cfg: RoofShuttleSceneCfg

    def __init__(self, cfg: RoofShuttleSceneCfg | None = None) -> None:
        super().__init__(cfg or RoofShuttleSceneCfg())

    # ----- assets --------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        cab_spawn = cls["cabinet"](
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            cab_hx=c.cab_hx, cab_hy=c.cab_hy, cab_h=c.cab_h, wall_t=c.wall_t,
            part_hw=c.part_hw, floor_t=c.floor_t, curb_x0=c.curb_x0, curb_h=c.curb_h,
            contact_offset=c.contact_offset)
        slab_spawn = cls["slab"](
            slab_hx=c.slab_hx, slab_hy=c.slab_hy, slab_ht=c.slab_ht, bar_y=c.bar_y,
            slab_mass=c.slab_mass, slab_damping=c.slab_damping,
            contact_offset=c.contact_offset)
        bowl_spawn = cls["bowl"](
            bowl_r=c.bowl_r, bowl_base_r=c.bowl_base_r, bowl_base_t=c.bowl_base_t,
            bowl_wall_t=c.bowl_wall_t, bowl_h=c.bowl_h, bowl_org=c.bowl_org,
            bowl_mass=c.bowl_mass, contact_offset=c.contact_offset)

        slab_c0 = self.slab_front_center()
        return {
            "ground": AssetBaseCfg(
                prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))),
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet", spawn=cab_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0))),
            "slab": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Slab", spawn=slab_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, slab_c0, self.slab_zc()))),
            "pedestal": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * c.ped_half, 2 * c.ped_half, c.ped_top - c.floor_t),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.75, 0.75, 0.78))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, self.bay_center_y(), (c.floor_t + c.ped_top) / 2))),
            "bowl": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bowl", spawn=bowl_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, self.bay_center_y(), c.ped_top + c.bowl_org + 0.002))),
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

    # ----- geometry helpers (cabinet is axis-aligned at the env origin) --------------------------
    def slab_front_center(self) -> float:
        return -self.cfg.cab_hy + self.cfg.slab_hy

    def slab_zc(self) -> float:
        return self.cfg.cab_h + self.cfg.slab_gap + self.cfg.slab_ht

    def bay_center_y(self) -> float:
        c = self.cfg
        return ((-c.cab_hy + c.wall_t) + (-c.part_hw)) / 2

    def slab_top_z(self) -> float:
        return self.slab_zc() + self.cfg.slab_ht

    # ----- lifecycle -----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.slab: RigidObject = env.iscene["slab"]
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # randomization readbacks (verified by smoke)
        self.slab_off0 = torch.zeros(n, device=dev)
        # latches (partial credit survives transients; success is judged live)
        self._slide_f = torch.zeros(n, device=dev)
        self._open_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._out_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._seat_l = torch.zeros(n, dtype=torch.bool, device=dev)
        self._author_rail()

    def _author_rail(self) -> None:
        """Per env: Y-axis prismatic joint cabinet -> panel, limits [0, travel]
        (0 = parked at the front, roofing the bay). The joint suspends the panel
        `slab_gap` above the wall tops, so the slide never scrapes; the joint pair
        is collision-filtered (PhysX filters jointed bodies by default here)."""
        import omni.usd
        from pxr import Gf, UsdPhysics

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/roof_rail")
            j.CreateBody0Rel().SetTargets([f"{base}/Cabinet"])
            j.CreateBody1Rel().SetTargets([f"{base}/Slab"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("Y")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, self.slab_front_center(), self.slab_zc()))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(0.0)
            j.CreateUpperLimitAttr(float(c.travel))

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: panel parked at the front stop (+ small start offset),
        pedestal re-posed inside the bay, bowl standing on it with xy jitter and
        free yaw, latches cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        torch.rand(1, device=dev)  # burn the degenerate first post-seed draw
        u = torch.rand(m, 6, device=dev)

        off = c.slab_jitter * u[:, 0]
        self.slab_off0[env_ids] = off
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = 0.0
        st[:, 1] = self.slab_front_center() + off
        st[:, 2] = self.slab_zc()
        st[:, 3] = 1.0
        st[:, 0:3] += origin
        self.slab.write_root_state_to_sim(st, env_ids)

        ped = torch.zeros(m, 13, device=dev)
        ped[:, 0] = (u[:, 1] * 2 - 1) * c.ped_jitter
        ped[:, 1] = self.bay_center_y() + (u[:, 2] * 2 - 1) * c.ped_jitter
        ped[:, 2] = (c.floor_t + c.ped_top) / 2
        ped[:, 3] = 1.0
        ped[:, 0:3] += origin
        self.pedestal.write_root_state_to_sim(ped, env_ids)

        bw = torch.zeros(m, 13, device=dev)
        bw[:, 0] = ped[:, 0] - origin[:, 0] + (u[:, 3] * 2 - 1) * c.bowl_jitter
        bw[:, 1] = ped[:, 1] - origin[:, 1] + (u[:, 4] * 2 - 1) * c.bowl_jitter
        bw[:, 2] = c.ped_top + c.bowl_org + 0.002
        half = (u[:, 5] * 2 - 1) * math.pi
        bw[:, 3] = torch.cos(half / 2)
        bw[:, 6] = torch.sin(half / 2)
        bw[:, 0:3] += origin
        self.bowl.write_root_state_to_sim(bw, env_ids)

        self._slide_f[env_ids] = 0.0
        self._open_l[env_ids] = False
        self._out_l[env_ids] = False
        self._seat_l[env_ids] = False

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "slab": self.slab.data.root_state_w[env_ids].clone(),
            "pedestal": self.pedestal.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "slab_off0": self.slab_off0[env_ids].clone(),
            "slide_f": self._slide_f[env_ids].clone(),
            "open_l": self._open_l[env_ids].clone(),
            "out_l": self._out_l[env_ids].clone(),
            "seat_l": self._seat_l[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.slab.write_root_state_to_sim(state["slab"], env_ids)
        self.pedestal.write_root_state_to_sim(state["pedestal"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.slab_off0[env_ids] = state["slab_off0"]
        self._slide_f[env_ids] = state["slide_f"]
        self._open_l[env_ids] = state["open_l"]
        self._out_l[env_ids] = state["out_l"]
        self._seat_l[env_ids] = state["seat_l"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A slate-grey open-topped cabinet ({2 * c.cab_hx * 100:.0f} x "
            f"{2 * c.cab_hy * 100:.0f} cm footprint, {c.cab_h * 100:.0f} cm tall) stands "
            f"on the ground, its long axis running front to back. Its only top surface "
            f"is a single tan SLIDING ROOF PANEL that rides in rails along that axis; "
            f"the panel is captive in its rails (it can only translate, "
            f"~{c.travel * 100:.0f} cm of travel) and carries a raised YELLOW HANDLE BAR "
            f"near its front edge, with finger room under the bar. At the start the "
            f"panel is parked at the FRONT end, roofing the front compartment; sealed "
            f"inside that compartment, standing on a light-grey pedestal, is the BLACK "
            f"BOWL ({2 * c.bowl_r * 100:.0f} cm across, {c.bowl_h * 100:.0f} cm tall, "
            f"matte black). The rear half of the cabinet is an open shaft with NO top — "
            f"you can see its deep-red floor {c.cab_h * 100:.0f} cm down; anything "
            f"released over that half just falls into the shaft and is not on top of "
            f"anything.\n"
            f"Goal: put the black bowl on top of the cabinet — concretely, stand it "
            f"UPRIGHT on the TOP FACE of the sliding roof panel, fully on the panel "
            f"(any rail position of the panel is fine). The only way there: (1) slide "
            f"the roof panel back along its rails by the yellow handle (push or pull) "
            f"until the front compartment is uncovered; (2) reach in and lift the black "
            f"bowl out; (3) set it down upright on the panel's top face, clear of the "
            f"handle bar. The bowl does not count while it is inside either "
            f"compartment, on the ground, on the pedestal, tipped over or upside down, "
            f"or fallen into the open rear shaft. The panel cannot be lifted off its "
            f"rails, and forcing the bowl upward through the closed roof is impossible "
            f"— open first."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Slide the cabinet's roof panel along its rails by the yellow handle to "
            "uncover the front compartment, lift out the black bowl, and stand it "
            "upright on the panel's top face. Do not drop the bowl into the open rear "
            "shaft or leave it inside the cabinet."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Cabinet-local position (the cabinet is axis-aligned at the env origin)."""
        return pos_w - self.env_origins

    def slab_off(self) -> torch.Tensor:
        """(N,) panel displacement along its rail from the front stop (m)."""
        return self._local(self.slab.data.root_pos_w)[:, 1] - self.slab_front_center()

    def uncovered(self) -> torch.Tensor:
        """(N,) width of the bay opening no longer under the panel (m)."""
        c = self.cfg
        slab_front_edge = self._local(self.slab.data.root_pos_w)[:, 1] - c.slab_hy
        bay_y0 = -c.cab_hy + c.wall_t
        return (slab_front_edge - bay_y0).clamp(min=0.0)

    def bowl_in_bay(self) -> torch.Tensor:
        """(N,) bool: bowl centre inside the front-compartment volume (below the
        rim plane, within the bay footprint)."""
        c = self.cfg
        loc = self._local(self.bowl.data.root_pos_w)
        return (loc[:, 0].abs() < c.cab_hx) \
            & (loc[:, 1] > -c.cab_hy) & (loc[:, 1] < -c.part_hw + 0.01) \
            & (loc[:, 2] < c.cab_h + 0.01)

    def _bowl_up(self) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.bowl.data.root_quat_w, ez)

    def bowl_on_slab(self) -> torch.Tensor:
        """(N,) bool, geometric: bowl base resting on the panel's TOP face — centre
        inside the panel footprint with margin (panel frame), base bottom within the
        rest band of the panel top, axis upright."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        loc = quat_apply_inverse(self.slab.data.root_quat_w,
                                 self.bowl.data.root_pos_w - self.slab.data.root_pos_w)
        in_xy = (loc[:, 0].abs() < c.slab_hx - c.seat_margin) \
            & (loc[:, 1].abs() < c.slab_hy - c.seat_margin)
        gap = (loc[:, 2] - c.bowl_org) - c.slab_ht   # base bottom above the plate top
        on_z = (gap > c.seat_z_lo) & (gap < c.seat_z_hi)
        upright = self._bowl_up()[:, 2] >= math.cos(math.radians(c.upright_max_deg))
        return in_xy & on_z & upright

    def settled(self) -> torch.Tensor:
        c = self.cfg
        return (self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin) \
            & (self.bowl.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang) \
            & (self.slab.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([self.slab.data.root_pos_w, self.bowl.data.root_pos_w], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        frac = (self.slab_off() / c.travel).clamp(0.0, 1.0)
        self._slide_f = torch.where(fin, torch.maximum(self._slide_f, frac), self._slide_f)
        self._open_l |= (self.uncovered() >= c.open_pass) & fin
        self._out_l |= (~self.bowl_in_bay()) & fin
        slow = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin
        self._seat_l |= self.bowl_on_slab() & slow & fin

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: the black bowl standing upright ON the roof panel's top face
        (live physical outcome, any rail position), everything settled and finite.
        Reaching it physically requires the panel slide (the bowl starts sealed
        under the panel) and a real extraction — the rubric judges only the settled
        contact outcome."""
        self._update_latches()
        return self.bowl_on_slab() & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.25 * latched slide fraction + 0.15 bay-opened +
        0.25 bowl-out-of-bay + 0.20 seated-on-panel (all latched), capped at 0.85;
        exactly 1.0 iff success() holds live. Doing nothing scores ~0; dropping a
        bowl over the topless rear half earns no seat credit."""
        c = self.cfg
        self._update_latches()
        base = (c.w_slide * self._slide_f + c.w_open * self._open_l.float()
                + c.w_out * self._out_l.float() + c.w_seat * self._seat_l.float()
                ).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="roof_shuttle", robot="null"))
