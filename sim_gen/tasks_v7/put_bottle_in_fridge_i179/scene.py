"""HangChillerScene — hang the green bottle by its cap flange from the slotted chill
rail, then slide it back into the chiller's cold zone.

Derived from rlbench/put_bottle_in_fridge, but the STORAGE PRINCIPLE is different, not
just the fixture: the seed's fridge is an articulated cabinet — open the revolute
door, then STAND the bottle upright on the shelf floor, one grasp-carry-release once
the door is out of the way. Here the chiller has NO door (open front) and NO usable
floor: the interior floor is a sparse drip GRATE whose gaps are wider than the bottle
— a bottle stood inside falls straight through into the drain pan below, and a bottle
laid across the rungs is just lost produce, not storage. The only storage the chiller
offers is SUSPENSION: an overhead slotted rail (two parallel bars with a neck-wide gap
between them) runs from a short loading tongue that protrudes out the front, back into
the cold zone under a low canopy. The bottle has a wide flanged cap: with the neck in
the slot, the cap flange rests on the two bars and the bottle hangs.

The slot is END-LOADABLE ONLY, by topology, not by tuning: hanging means the body is
BELOW the rail plane while the cap is above it, and the body (60 mm) can never pass
down through the 32 mm slot — so the bottle must be threaded onto the rail at the
exposed tongue (neck slid into the slot's open front end) and then SLID rearward,
hanging, under the canopy into the cold zone. The solver's plan is therefore: keep the
bottle VERTICAL (never lay it down), lift it high, thread the neck into the slot at
the tongue, seat the cap flange on the bars, and drag the suspended bottle backward at
least `chill_depth` past the front opening. Success is settled suspension in the cold
zone — not containment on any surface.

Assets are fully procedural, authored by custom compound spawners (child colliders of
one body never self-collide):
  - cabinet: KINEMATIC compound — drain pan floor, 3 sparse grate rungs (gaps ~70 mm
    > bottle dia 60 mm), side walls, back wall, roof (underside 24 mm above the rail
    plane: nothing can be dropped into the interior; the front is open). Local frame:
    origin at the FASCIA (front-opening plane) centre on the ground, +x = OUT.
  - rail: KINEMATIC compound, repositioned per episode (the slot line is randomized
    laterally) — two orange bars (gap 32 mm) running from x=+0.06 (tongue, proud of
    the fascia) to x=-0.22 (back wall), two splayed entry-guide flares at the tongue
    tip, and a below-rail-top neck stop near the back. Low-friction physics material
    on the bar tops (the cap slides on them).
  - bottle (target): DYNAMIC compound — green body cylinder r30 x 140 mm + neck
    r11 x 50 mm + light-grey flanged cap r28 x 14 mm (204 mm standing).
  - can (distractor): plain red cylinder r28 x 115 mm; no neck, no flange — it can
    never hang (56 mm > the 32 mm slot).

Per-episode randomization (readback-verifiable): cabinet yaw + xy jitter, the rail
slot's lateral offset y_off, bottle and can ground positions in the front field with
guaranteed separation (deterministic fallback slots).

Rubric (0..1; partial progress latched so transient achievements keep credit):
  0.30 * hung       — bottle ever suspended from the rail (cap seated in the slot
                      band, bottle vertical), anywhere along the rail (latched)
  0.50 * depth      — latched max of chill-depth progress (fascia -> chill_depth),
                      counted ONLY while the bottle is currently hanging
  1.0 iff success() — bottle hanging from the rail, cap at least `chill_depth` behind
                      the fascia, everything settled. Non-success capped at 0.80.

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


def _slick_material(stage, path: str, mu_s: float, mu_d: float):
    """Author a physics material (the cap must SLIDE on the rail bars: authored
    friction, not the ~0.5 default a custom spawner would otherwise leave)."""
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu_s))
    api.CreateDynamicFrictionAttr(float(mu_d))
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind_material(prim, mat) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, materialPurpose=UsdShade.Tokens.allPurpose)


def _add_box(stage, path: str, *, center, size, color, collide: Callable,
             yaw_deg: float = 0.0, material=None) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if yaw_deg:
        half = math.radians(yaw_deg) / 2
        xf.AddOrientOp().Set(Gf.Quatf(math.cos(half), Gf.Vec3f(0.0, 0.0, math.sin(half))))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    if material is not None:
        _bind_material(box.GetPrim(), material)


def _spawn_cabinet(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the chiller cabinet at `prim_path`: KINEMATIC rigid body (repositionable
    at reset, immovable to contacts). Local frame: origin at the FASCIA centre on the
    ground, +x = OUT toward the open front.

    Children: drain pan floor, 3 sparse grate rungs (gaps wider than the bottle — no
    floor storage), back wall, 2 side walls, roof (underside just above cap-slide
    height — no top access to the interior). The front face is fully open."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    def box(name, center, size, color=None):
        _add_box(stage, f"{prim_path}/{name}", center=center, size=size,
                 color=color or c.color, collide=collide)

    xb = -c.depth                       # back wall inner face
    top = c.roof_lo + c.roof_t          # walls reach the roof top
    # drain pan: solid slab under the grate, interior footprint
    box("pan", (xb / 2, 0.0, c.pan_t / 2), (-xb, 2 * c.in_hw, c.pan_t))
    # sparse grate rungs (along y): gaps ~70 mm > bottle dia — nothing stands on this
    for i, gx in enumerate(c.grate_xs):
        box(f"rung_{i}", (gx, 0.0, c.grate_top - c.rung_s / 2),
            (c.rung_s, 2 * c.in_hw, c.rung_s), color=c.grate_color)
    # back wall
    box("wall_back", (xb - c.wall_t / 2, 0.0, top / 2), (c.wall_t, 2 * c.in_hw, top))
    # side walls (interior run)
    for sgn, nm in ((1.0, "wall_l"), (-1.0, "wall_r")):
        box(nm, (xb / 2, sgn * (c.in_hw + c.wall_t / 2), top / 2),
            (-xb + c.wall_t, c.wall_t, top))
    # roof: underside at roof_lo — 24 mm above the rail plane, no top access
    box("roof", (xb / 2, 0.0, c.roof_lo + c.roof_t / 2),
        (-xb + c.wall_t, 2 * (c.in_hw + c.wall_t), c.roof_t), color=c.roof_color)
    return root


def _spawn_rail(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the slotted chill rail at `prim_path`: KINEMATIC compound, repositioned
    per episode (the slot line is randomized laterally). Local frame: origin on the
    slot centreline at the FASCIA plane, z = the rail TOP plane.

    Children: two bars (tongue -> back), two splayed entry-guide flares at the tongue
    tip, and a neck stop below the rail-top plane near the back (the cap passes over
    it; the neck cannot). Bar tops carry a low-friction material — the cap slides."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    slick = _slick_material(stage, f"{prim_path}/slickMat", c.mu_s, c.mu_d)

    bar_cx = (c.x_in + c.x_out) / 2
    bar_len = c.x_out - c.x_in
    for sgn, nm in ((1.0, "bar_l"), (-1.0, "bar_r")):
        _add_box(stage, f"{prim_path}/{nm}",
                 center=(bar_cx, sgn * (c.slot_hw + c.bar_w / 2), -c.bar_t / 2),
                 size=(bar_len, c.bar_w, c.bar_t), color=c.color, collide=collide,
                 material=slick)
    # entry-guide flares: splayed outward at the tongue tip (funnel the neck in)
    for sgn in (1.0, -1.0):
        _add_box(stage, f"{prim_path}/flare_{'l' if sgn > 0 else 'r'}",
                 center=(c.x_out + 0.002, sgn * (c.slot_hw + 0.014), -c.bar_t / 2 - 0.001),
                 size=(0.032, 0.010, c.bar_t), color=c.color, collide=collide,
                 yaw_deg=sgn * 20.0, material=slick)
    # neck stop: below the rail-top plane — the sliding cap clears it, the neck hits it
    _add_box(stage, f"{prim_path}/stop",
             center=(c.stop_x, 0.0, -0.002 - c.stop_h / 2),
             size=(0.010, 2 * c.slot_hw - 0.002, c.stop_h), color=c.color,
             collide=collide)
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the bottle: DYNAMIC compound — body + neck + wide flanged cap stacked
    along local +z, origin at the BODY cylinder's centre. The cap flange is what hangs
    on the rail: it carries the same low-friction material as the bar tops. Damping is
    tuned so the hanging pendulum settles promptly; solver velocity iterations are
    raised (cylinder-on-box resting contacts creep at the default)."""
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.08)
    pxrb.CreateAngularDampingAttr(0.40)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    pxrb.CreateSolverPositionIterationCountAttr(16)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    collide = _make_collide(cfg.contact_offset)
    c = cfg
    slick = _slick_material(stage, f"{prim_path}/slickMat", c.mu_s, c.mu_d)
    z_neck = c.body_h / 2 + c.neck_h / 2
    z_cap = c.body_h / 2 + c.neck_h + c.cap_t / 2
    for nm, r, h, z0, col, mat in (
            ("body", c.body_r, c.body_h, 0.0, c.color, None),
            ("neck", c.neck_r, c.neck_h, z_neck, c.color, None),
            ("cap", c.cap_r, c.cap_t, z_cap, c.cap_color, slick)):
        cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{nm}")
        cyl.CreateRadiusAttr(r)
        cyl.CreateHeightAttr(h)
        cyl.CreateExtentAttr([Gf.Vec3f(-r, -r, -h / 2), Gf.Vec3f(r, r, h / 2)])
        UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z0))
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*col)])
        collide(cyl.GetPrim())
        if mat is not None:
            _bind_material(cyl.GetPrim(), mat)
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
            depth: float = 0.22
            in_hw: float = 0.15
            wall_t: float = 0.012
            pan_t: float = 0.012
            grate_top: float = 0.105
            rung_s: float = 0.010
            grate_xs: tuple = (-0.05, -0.13, -0.21)
            roof_lo: float = 0.369
            roof_t: float = 0.010
            color: tuple = (0.72, 0.78, 0.84)
            roof_color: tuple = (0.45, 0.52, 0.62)
            grate_color: tuple = (0.20, 0.22, 0.26)
            contact_offset: float = 0.002

        @configclass
        class RailSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_rail)
            slot_hw: float = 0.016
            bar_w: float = 0.030
            bar_t: float = 0.012
            x_in: float = -0.22
            x_out: float = 0.06
            stop_x: float = -0.20
            stop_h: float = 0.018
            mu_s: float = 0.20
            mu_d: float = 0.15
            color: tuple = (0.88, 0.45, 0.08)
            contact_offset: float = 0.002

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.030
            body_h: float = 0.140
            neck_r: float = 0.011
            neck_h: float = 0.050
            cap_r: float = 0.028
            cap_t: float = 0.014
            mass: float = 0.30
            mu_s: float = 0.20
            mu_d: float = 0.15
            color: tuple = (0.10, 0.45, 0.16)
            cap_color: tuple = (0.75, 0.75, 0.78)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(cabinet=CabinetSpawnerCfg, rail=RailSpawnerCfg,
                              bottle=BottleSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class HangChillerSceneCfg(BaseCfg):
    """Config for `HangChillerScene`. Load-bearing geometry facts: the bottle body
    (60 mm) can never pass the 32 mm slot, so the rail is end-loadable only (thread at
    the tongue, slide in hanging); the grate gaps (~70 mm) exceed the bottle diameter,
    so nothing can be stood on the interior floor; the roof underside sits 24 mm above
    the rail plane, so the interior has no top access and the cap (14 mm) slides under
    it with 10 mm clearance."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    chill_depth: float = tunable(0.10)  # cap this far behind the fascia = in the cold zone (m)
    slot_y_tol: float = tunable(0.012)  # |cap y| in the rail frame while hanging (m)
    seat_z_band: tuple = tunable((0.002, 0.012))  # cap CENTRE z in the rail frame when seated
    hang_x_range: tuple = tunable((-0.21, 0.085))  # cap x range that counts as "on the rail"
    upright_min: float = tunable(0.9781)  # bottle axis . up >= this (within ~12 deg) to hang
    settle_speed: float = tunable(0.05)  # max |lin vel| when judging (m/s)
    settle_omega: float = tunable(0.8)  # max |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) -----------------------------------------
    cab_yaw_deg: float = tunable(8.0)  # cabinet yaw jitter about nominal (+/- deg)
    cab_jitter: float = tunable(0.02)  # cabinet xy jitter (+/- m)
    rail_y_off: float = tunable(0.05)  # slot lateral offset range (+/- m)
    obj_x_range: tuple = tunable((0.14, 0.32))  # bottle + can: world x field (m)
    obj_y_range: tuple = tunable((-0.32, 0.32))  # bottle + can: world y field (m)
    min_sep: float = tunable(0.26)  # bottle-can minimum spawn separation (m)

    # --- info: structure (cabinet local frame: origin fascia centre on ground, +x = OUT) --------
    cab_pos: tuple = info((0.50, 0.0))  # fascia centre on the ground
    cab_yaw_nominal: float = info(180.0)  # deg; local +x (front) -> world -x
    depth: float = info(0.22)  # interior depth (back wall inner face at -depth)
    in_hw: float = info(0.15)  # interior half width
    wall_t: float = info(0.012)
    pan_t: float = info(0.012)  # drain pan slab (top = pan_t)
    grate_top: float = info(0.105)  # grate rung top plane
    rung_s: float = info(0.010)  # rung cross-section (square)
    grate_xs: tuple = info((-0.05, -0.13, -0.21))  # rung centres; gaps ~70 mm > bottle 60 mm
    roof_lo: float = info(0.369)  # roof underside (rail_top + 24 mm)
    roof_t: float = info(0.010)
    rail_top: float = info(0.345)  # rail TOP plane height (the hang plane)
    slot_hw: float = info(0.016)  # slot half-gap (32 mm between bar inner faces)
    bar_w: float = info(0.030)  # each rail bar width
    bar_t: float = info(0.012)
    rail_x_in: float = info(-0.22)  # bars reach the back wall
    rail_x_out: float = info(0.06)  # loading tongue, proud of the fascia
    stop_x: float = info(-0.20)  # neck stop centre (below rail top; cap clears it)
    body_r: float = info(0.030)  # bottle body (60 mm dia — wider than the slot)
    body_h: float = info(0.140)
    neck_r: float = info(0.011)  # neck (22 mm dia — 5 mm slack per side in the slot)
    neck_h: float = info(0.050)
    cap_r: float = info(0.028)  # cap flange (56 mm dia — overlaps each bar by 12 mm)
    cap_t: float = info(0.014)
    bottle_mass: float = info(0.30)
    can_r: float = info(0.028)  # distractor (56 mm dia > slot: can never hang)
    can_h: float = info(0.115)
    can_mass: float = info(0.25)
    mu_s: float = info(0.20)  # authored cap/bar friction (the cap must slide)
    mu_d: float = info(0.15)
    cab_color: tuple = info((0.72, 0.78, 0.84))
    roof_color: tuple = info((0.45, 0.52, 0.62))
    grate_color: tuple = info((0.20, 0.22, 0.26))
    rail_color: tuple = info((0.88, 0.45, 0.08))
    bottle_color: tuple = info((0.10, 0.45, 0.16))
    cap_color: tuple = info((0.75, 0.75, 0.78))
    can_color: tuple = info((0.80, 0.10, 0.08))
    contact_offset: float = info(0.002)
    fallback_bottle: tuple = info((0.20, -0.24))  # deterministic separation fallback
    fallback_can: tuple = info((0.20, 0.24))
    # rubric weights (0.30 + 0.50 = 0.80 = the non-success cap)
    w_hung: float = info(0.30)
    w_depth: float = info(0.50)

    # Derived (filled in __post_init__).
    cap_off: float = field(default=None, init=False)  # bottle origin -> cap centre (local z)
    cap_bot_off: float = field(default=None, init=False)  # bottle origin -> cap bottom

    def __post_init__(self) -> None:
        self.cap_bot_off = round(self.body_h / 2 + self.neck_h, 4)          # 0.120
        self.cap_off = round(self.cap_bot_off + self.cap_t / 2, 4)          # 0.127


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("hang_chiller")
class HangChillerScene(BaseScene):
    cfg: HangChillerSceneCfg

    def __init__(self, cfg: HangChillerSceneCfg | None = None) -> None:
        super().__init__(cfg or HangChillerSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        cabinet_spawn = cls["cabinet"](
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            depth=c.depth, in_hw=c.in_hw, wall_t=c.wall_t, pan_t=c.pan_t,
            grate_top=c.grate_top, rung_s=c.rung_s, grate_xs=c.grate_xs,
            roof_lo=c.roof_lo, roof_t=c.roof_t, color=c.cab_color,
            roof_color=c.roof_color, grate_color=c.grate_color,
            contact_offset=c.contact_offset)
        rail_spawn = cls["rail"](
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            slot_hw=c.slot_hw, bar_w=c.bar_w, bar_t=c.bar_t, x_in=c.rail_x_in,
            x_out=c.rail_x_out, stop_x=c.stop_x, mu_s=c.mu_s, mu_d=c.mu_d,
            color=c.rail_color, contact_offset=c.contact_offset)
        bottle_spawn = cls["bottle"](
            mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            body_r=c.body_r, body_h=c.body_h, neck_r=c.neck_r, neck_h=c.neck_h,
            cap_r=c.cap_r, cap_t=c.cap_t, mass=c.bottle_mass, mu_s=c.mu_s,
            mu_d=c.mu_d, color=c.bottle_color, cap_color=c.cap_color,
            contact_offset=c.contact_offset)
        can_spawn = sim_utils.CylinderCfg(
            radius=c.can_r, height=c.can_h,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5, linear_damping=0.05,
                angular_damping=0.05, sleep_threshold=0.0, stabilization_threshold=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=c.can_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=c.contact_offset, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.can_color),
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
            "cabinet": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=cabinet_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cab_pos[0], c.cab_pos[1], 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
            ),
            "rail": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Rail",
                spawn=rail_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.cab_pos[0], c.cab_pos[1], c.rail_top), rot=(0.0, 0.0, 0.0, 1.0)),
            ),
            "bottle": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=bottle_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.20, -0.24, c.body_h / 2 + 0.002)),
            ),
            "can": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedCan",
                spawn=can_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.20, 0.24, c.can_h / 2 + 0.002)),
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
        self.cabinet: RigidObject = env.iscene["cabinet"]
        self.rail: RigidObject = env.iscene["rail"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.can: RigidObject = env.iscene["can"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        # latches: partial progress survives transient achievements (rubric requirement)
        self._hung = torch.zeros(n, dtype=torch.bool, device=env.device)  # ever suspended
        self._depth_max = torch.zeros(n, device=env.device)  # chill progress, gated
        self._y_off = torch.zeros(n, device=env.device)  # rail slot lateral offset

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the cabinet (yaw + xy jitter), hang the rail at a
        randomized lateral slot offset (cabinet-frame pose), stand the bottle and the
        can upright in the front field with guaranteed separation; clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- cabinet: kinematic, yaw + xy jitter ---
        yaw = math.radians(c.cab_yaw_nominal) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.cab_yaw_deg)
        cx = c.cab_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.cab_jitter
        cy = c.cab_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.cab_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0], st[:, 1] = cx, cy
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.cabinet.write_root_state_to_sim(st, env_ids)

        # --- rail: same yaw, slot line offset laterally by y_off (cabinet frame) ---
        y_off = (torch.rand(m, device=dev) * 2 - 1) * c.rail_y_off
        st = torch.zeros(m, 13, device=dev)
        st[:, 0] = cx - torch.sin(yaw) * y_off
        st[:, 1] = cy + torch.cos(yaw) * y_off
        st[:, 2] = c.rail_top
        st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        st[:, 0:3] += origin
        self.rail.write_root_state_to_sim(st, env_ids)
        self._y_off[env_ids] = y_off

        # --- bottle + can: upright in the front field, min separation (det. fallback) ---
        bx = c.obj_x_range[0] + torch.rand(m, device=dev) * (c.obj_x_range[1] - c.obj_x_range[0])
        by = c.obj_y_range[0] + torch.rand(m, device=dev) * (c.obj_y_range[1] - c.obj_y_range[0])
        kx = c.obj_x_range[0] + torch.rand(m, device=dev) * (c.obj_x_range[1] - c.obj_x_range[0])
        ky = c.obj_y_range[0] + torch.rand(m, device=dev) * (c.obj_y_range[1] - c.obj_y_range[0])
        for _try in range(12):
            bad = ((bx - kx) ** 2 + (by - ky) ** 2).sqrt() < c.min_sep
            if not bad.any():
                break
            nb = int(bad.sum())
            kx[bad] = c.obj_x_range[0] + torch.rand(nb, device=dev) \
                * (c.obj_x_range[1] - c.obj_x_range[0])
            ky[bad] = c.obj_y_range[0] + torch.rand(nb, device=dev) \
                * (c.obj_y_range[1] - c.obj_y_range[0])
        # deterministic by-construction fallback for any still-bad env
        bad = ((bx - kx) ** 2 + (by - ky) ** 2).sqrt() < c.min_sep
        if bad.any():
            bx[bad], by[bad] = c.fallback_bottle
            kx[bad], ky[bad] = c.fallback_can
        for obj, x, y, z in ((self.bottle, bx, by, c.body_h / 2 + 0.002),
                             (self.can, kx, ky, c.can_h / 2 + 0.002)):
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1], st[:, 2] = x, y, z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._hung[env_ids] = False
        self._depth_max[env_ids] = 0.0

    # ----- state (full, restorable) ----------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "cabinet": self.cabinet.data.root_state_w[env_ids].clone(),
            "rail": self.rail.data.root_state_w[env_ids].clone(),
            "bottle": self.bottle.data.root_state_w[env_ids].clone(),
            "can": self.can.data.root_state_w[env_ids].clone(),
            "hung": self._hung[env_ids].clone(),
            "depth_max": self._depth_max[env_ids].clone(),
            "y_off": self._y_off[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.cabinet.write_root_state_to_sim(state["cabinet"], env_ids)
        self.rail.write_root_state_to_sim(state["rail"], env_ids)
        self.bottle.write_root_state_to_sim(state["bottle"], env_ids)
        self.can.write_root_state_to_sim(state["can"], env_ids)
        self._hung[env_ids] = state["hung"]
        self._depth_max[env_ids] = state["depth_max"]
        self._y_off[env_ids] = state["y_off"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A pale-blue chiller cabinet stands on the ground: an open-fronted box "
            f"(interior {2 * c.in_hw * 100:.0f} cm wide, {c.depth * 100:.0f} cm deep, roof at "
            f"{(c.roof_lo + c.roof_t) * 100:.0f} cm) with NO door. Its interior floor is a sparse "
            f"dark drip GRATE over a drain pan — the gaps between the rungs are wider than the "
            f"bottle, so nothing can be stood or stored on the floor: whatever is put down "
            f"there falls through into the pan. The only storage is the ORANGE chill rail "
            f"under the roof: two horizontal bars with a {2 * c.slot_hw * 100:.1f} cm slot "
            f"between them, running from a short loading TONGUE that sticks "
            f"{c.rail_x_out * 100:.0f} cm out of the front opening (with splayed entry guides) "
            f"back to the rear wall. On the ground in front stand two upright objects: a GREEN "
            f"glass bottle ({(c.body_h + c.neck_h + c.cap_t) * 100:.1f} cm tall, "
            f"{2 * c.body_r * 100:.0f} cm body, with a narrow neck and a wide light-grey cap "
            f"whose flange overhangs the neck) and a RED soda can, a distractor with no neck — "
            f"it cannot hang.\n"
            f"Goal: store the GREEN bottle hanging from the chill rail, inside the cold zone. "
            f"Keep the bottle VERTICAL: lift it up to the exposed tongue in front of the "
            f"cabinet, thread its neck into the open front end of the slot, and set the cap "
            f"flange down on the two bars so the bottle hangs. Then slide the hanging bottle "
            f"backward along the slot, under the roof, until its cap is at least "
            f"{c.chill_depth * 100:.0f} cm behind the front opening plane (about halfway to the "
            f"back wall; a stop near the back end limits travel). The bottle's body is wider "
            f"than the slot, so it can never be dropped in from above — the rail only loads "
            f"from its front end — and the roof means nothing can be lowered into the interior. "
            f"Success: the bottle at rest, suspended by its cap flange from the rail, at least "
            f"{c.chill_depth * 100:.0f} cm deep. A bottle left standing anywhere (it falls "
            f"through the grate), lying in the pan or across the grate, hung but still out on "
            f"the tongue, or the red can anywhere, counts for nothing."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Hang the green bottle from the orange chill rail: keep it vertical, thread its "
            "neck into the slot at the rail's front tongue so the grey cap flange rests on "
            "the two bars, then slide it backward into the chiller until the cap is at least "
            "10 cm behind the front opening. It must end suspended from the rail — not on "
            "any floor. Leave the red can alone."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _cap_center_w(self) -> torch.Tensor:
        """(N, 3) cap-flange centre in world coordinates."""
        from isaaclab.utils.math import quat_apply

        off = torch.tensor([0.0, 0.0, self.cfg.cap_off], device=self.env.device)
        return self.bottle.data.root_pos_w + quat_apply(
            self.bottle.data.root_quat_w, off.expand(self.env.num_envs, 3))

    def _in_frame(self, pos_w: torch.Tensor, obj) -> torch.Tensor:
        """World points -> `obj`'s body frame, (N, 3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(obj.data.root_quat_w, pos_w - obj.data.root_pos_w)

    def _bottle_up(self) -> torch.Tensor:
        """(N,) world-z component of the bottle's +z axis."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(self.bottle.data.root_quat_w, ez)[:, 2]

    def _hung_now(self) -> torch.Tensor:
        """(N,) bool: bottle currently SUSPENDED from the rail — cap centre in the
        slot band of the RAIL frame (|y| within slot_y_tol, z within the seated band
        over the bar tops, x on the rail run) with the bottle vertical. Geometry makes
        this suspension: a seated cap with the bottle vertical puts the body 5 cm
        clear above the grate, hanging in free air."""
        c = self.cfg
        loc = self._in_frame(self._cap_center_w(), self.rail)
        seated = (loc[:, 2] >= c.seat_z_band[0]) & (loc[:, 2] <= c.seat_z_band[1])
        in_slot = loc[:, 1].abs() < c.slot_y_tol
        on_run = (loc[:, 0] > c.hang_x_range[0]) & (loc[:, 0] < c.hang_x_range[1])
        return seated & in_slot & on_run & (self._bottle_up() >= c.upright_min)

    def _cap_depth(self) -> torch.Tensor:
        """(N,) cap depth behind the fascia, CABINET frame (positive = inside)."""
        return -self._in_frame(self._cap_center_w(), self.cabinet)[:, 0]

    def _update_latches(self) -> None:
        """Refresh latches: `hung` once the bottle first hangs anywhere on the rail;
        `depth` is the running max of chill progress, counted ONLY while currently
        hanging (a can pushed inside, or a bottle in the pan, earns nothing)."""
        hung_now = self._hung_now()
        self._hung |= hung_now
        prog = (self._cap_depth() / self.cfg.chill_depth).clamp(0.0, 1.0)
        self._depth_max = torch.maximum(self._depth_max, prog * hung_now.float())

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def settled(self) -> torch.Tensor:
        """(N,) bool: bottle at rest (linear AND angular — a swinging pendulum is not
        a stored bottle)."""
        c = self.cfg
        return (self.bottle.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.bottle.data.root_ang_vel_w.norm(dim=-1) < c.settle_omega)

    def success(self) -> torch.Tensor:
        """(N,) bool: the bottle hangs from the rail (live suspension test in the rail
        frame), its cap at least `chill_depth` behind the fascia (cabinet frame), at
        rest. Judged on the physical settled state only."""
        self._update_latches()
        return self._hung_now() & (self._cap_depth() >= self.cfg.chill_depth) & self.settled()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.30*hung + 0.50*depth (both latched; depth gated on
        actually hanging; ~0 for doing nothing) — capped at 0.80 — and exactly 1.0 iff
        success() holds live."""
        c = self.cfg
        self._update_latches()
        base = (c.w_hung * self._hung.float() + c.w_depth * self._depth_max).clamp(max=0.80)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="hang_chiller", robot="null"))
