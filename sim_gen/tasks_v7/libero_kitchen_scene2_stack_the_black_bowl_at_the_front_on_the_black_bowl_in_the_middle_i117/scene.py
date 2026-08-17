"""WaitersPullScene — seat the black dessert bowl on the black pedestal bowl by DRAWING OUT
the serving slat from under it (the waiter's tablecloth pull), then stow the slat in the tray.

Derived from libero_90 kitchen_scene2 "stack the black bowl at the front on the black bowl in
the middle". The seed is ONE unconstrained grasp-carry-place: pick bowl_1, set it on bowl_2,
success = an xy/z proximity relation between the two bowl origins. Here the stack is still the
goal image — one black bowl resting centred on the other — but every element of the seed's
plan is inverted:

  * NEITHER bowl is ever grasped, carried, or even pushed. The pedestal bowl is 140 mm across
    and the dessert bowl's flange is 124 mm across — both wider than the 80 mm Franka jaw
    span, and neither ever needs to move horizontally;
  * at spawn the two bowls are ALREADY vertically aligned, separated by an interposed serving
    slat: the slat lies across the pedestal's rim and the dessert bowl rides ON the slat. The
    stack is produced by REMOVAL, not placement: draw the slat out horizontally from under
    the loaded bowl so the bowl descends onto the pedestal rim by gravity;
  * the extraction is a real friction/shear problem (the waiter's tablecloth pull). The slat
    is slick (a bound low-friction, min-combine physics material) but not frictionless: while
    the slat slides, kinetic friction drags the bowl toward the pull direction. Draw the slat
    out too slowly and the bowl walks off the pedestal's 24 mm radial capture window and
    lands on the rim wall or the counter — unrecoverable, since the bowl cannot be grasped.
    The only control surface is the slat's red grip tab (24 mm — the one pinchable feature in
    the scene);
  * the goal has a second, ordered clause: the extracted slat must then be laid flat INSIDE
    the open stowage tray. The seed-strategy end state — the bowls stacked, however achieved,
    with the slat left lying about — is an explicitly tested non-success;
  * physics orders the phases: the slat cannot be stowed while it still carries the bowl
    (drawing it out is forced first), and the bowl cannot seat while the slat spans the rim.

Success (simultaneous, settled):
  * the dessert bowl is SEATED on the pedestal: flange resting on the rim top with the foot
    hanging into the recess — bowl-to-pedestal horizontal offset within `seat_xy_tol`, bowl
    origin in the seated z band (`seat_dz` above the pedestal origin), bowl upright. A bowl
    perched with its FOOT on the rim wall sits 12 mm too high and fails the z band; a bowl
    on the counter is far below the band;
  * the slat rests flat INSIDE the tray (tray-frame footprint, floor z band, long axis
    aligned with the tray, tilt-gated);
  * the pedestal is upright at counter height (it must not have been capsized or dragged off
    the counter);
  * every dynamic body settled.

Rubric (graded 0..1, latched in post_step, additive; 1.0 iff success()):
  0.00  nothing happened (the spawn sandwich scores 0: the bowl rides 20 mm above the
        seated z band)
  +0.20 clean draw in progress: the slat has been displaced >= 6 cm from its spawn while
        IN MOTION, with the bowl still hovering over the pedestal (within capture, at
        riding height, nearly still) — a fly-through, a capsize, or a slat teleported away
        with zero velocity latches nothing
  +0.25 the bowl has rested SEATED on the pedestal with the slat fully clear, at least once
  +0.25 the slat has rested stowed in the tray, at least once
  1.00  success() (overrides the 0.70 partial sum)

Assets are fully procedural (no external files), one rigid body each:
  * pedestal bowl: black disc base + octagonal raised rim wall enclosing a 108 mm recess
    (origin at the bottom centre); heavy (0.8 kg) so shear from the sliding slat cannot
    drag it;
  * dessert bowl: black 60 mm foot disc under a 124 mm flange disc (origin at the foot
    bottom) — the foot self-centres into the recess, the flange lands on the rim;
  * slat: 300 x 90 x 8 mm pale board with a raised RED grip tab near its leading end; a
    low-friction (mu 0.08, combine-mode "min") physics material is bound to its colliders;
  * tray: kinematic open stowage tray (floor + two side rails + one end rail) fixed to the
    counter, entry open toward the robot side.

Per-episode randomization (verified by readback in the smoke): pedestal xy, pull-axis yaw
(+/- 25 deg about "toward the robot"), bowl-on-slat jitter, tray side (left/right), tray xy
jitter and yaw, bowl spawn yaw.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and registering the
scene — stays app-free.
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


# ----- custom compound spawners -------------------------------------------------------------------
# One rigid body per object: root Xform with RigidBodyAPI (+ explicit MassAPI on dynamics),
# child collider shapes. Authored through `clone()` so per-env replication is idempotent
# (no duplicate xformOps).

_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(prim_path: str, translation, orientation, *, kinematic: bool, mass: float | None):
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
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(mass))
        # Cap the contact-solver pop on landings (the bowl drops onto the rim in one 120 Hz
        # step) and add light damping so landed bodies cross the settle gate promptly.
        px_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        px_rb.CreateMaxDepenetrationVelocityAttr(0.5)
        px_rb.CreateLinearDampingAttr(0.05)
        px_rb.CreateAngularDampingAttr(0.10)
    return root


def _bind_friction(prim_path: str, collider_prim, mu: float) -> None:
    """Bind a low-friction physics material (combine mode 'min') to a collider so the PAIR
    friction is `mu` regardless of the partner's default ~0.5."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    mat_path = f"{prim_path}/slickmat"
    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(mu))
    api.CreateDynamicFrictionAttr(float(mu))
    api.CreateRestitutionAttr(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr().Set("min")
    px.CreateRestitutionCombineModeAttr().Set("min")
    UsdShade.MaterialBindingAPI.Apply(collider_prim).Bind(
        mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")


def _child_cyl(prim_path: str, name: str, *, radius: float, height: float, z: float,
               color: tuple, contact_offset: float):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    cyl = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}")
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateExtentAttr([Gf.Vec3f(-radius, -radius, -height / 2),
                          Gf.Vec3f(radius, radius, height / 2)])
    UsdGeom.Xformable(cyl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(cyl.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return cyl


def _child_box(prim_path: str, name: str, *, tx: float, ty: float, tz: float,
               sx: float, sy: float, sz: float, color: tuple, contact_offset: float,
               yaw_deg: float = 0.0):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    box = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
    box.CreateSizeAttr(1.0)
    bxf = UsdGeom.Xformable(box.GetPrim())
    bxf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    if yaw_deg:
        bxf.AddRotateZOp().Set(float(yaw_deg))
    bxf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    px = PhysxSchema.PhysxCollisionAPI.Apply(box.GetPrim())
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)
    return box


def _spawn_pedestal(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The pedestal bowl: black base disc + octagonal raised rim wall enclosing the recess.
    Origin at the bottom centre (z = counter top when resting)."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass)
    _child_cyl(prim_path, "base", radius=cfg.disc_r, height=cfg.disc_h, z=cfg.disc_h / 2,
               color=cfg.color, contact_offset=cfg.contact_offset)
    wall_mid_r = cfg.recess_r + cfg.wall_t / 2
    seg_len = 2.0 * math.tan(math.pi / 8) * wall_mid_r
    for k in range(8):
        a = k * 45.0
        ar = math.radians(a)
        _child_box(prim_path, f"wall{k}",
                   tx=wall_mid_r * math.cos(ar), ty=wall_mid_r * math.sin(ar),
                   tz=cfg.disc_h + cfg.wall_h / 2,
                   sx=cfg.wall_t, sy=seg_len, sz=cfg.wall_h,
                   color=cfg.color, contact_offset=cfg.contact_offset, yaw_deg=a)
    return root


def _spawn_bowl(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The dessert bowl: narrow foot disc under a wide flange disc. Origin at the foot
    bottom. The foot drops into the pedestal recess; the flange lands on the rim top."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass)
    _child_cyl(prim_path, "foot", radius=cfg.foot_r, height=cfg.foot_h, z=cfg.foot_h / 2,
               color=cfg.color, contact_offset=cfg.contact_offset)
    _child_cyl(prim_path, "flange", radius=cfg.flange_r, height=cfg.flange_h,
               z=cfg.foot_h + cfg.flange_h / 2,
               color=cfg.color, contact_offset=cfg.contact_offset)
    return root


def _spawn_slat(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The serving slat: slick pale board (origin at board centre) + raised RED grip tab
    near the local +x (leading) end. A low-friction min-combine material is bound to both
    colliders."""
    root = _apply_root(prim_path, translation, orientation, kinematic=False,
                       mass=cfg.mass_props.mass)
    board = _child_box(prim_path, "board", tx=0.0, ty=0.0, tz=0.0,
                       sx=cfg.length, sy=cfg.width, sz=cfg.thick,
                       color=cfg.color, contact_offset=cfg.contact_offset)
    tab = _child_box(prim_path, "tab", tx=cfg.tab_x, ty=0.0,
                     tz=cfg.thick / 2 + cfg.tab_h / 2,
                     sx=cfg.tab_l, sy=cfg.tab_w, sz=cfg.tab_h,
                     color=cfg.tab_color, contact_offset=cfg.contact_offset)
    _bind_friction(prim_path, board.GetPrim(), cfg.mu)
    _bind_friction(prim_path, tab.GetPrim(), cfg.mu)
    return root


def _spawn_tray(prim_path: str, cfg: Any, translation=None, orientation=None):
    """The stowage tray, kinematic: floor + two side rails + one end rail (local -x open)."""
    root = _apply_root(prim_path, translation, orientation, kinematic=True, mass=None)
    _child_box(prim_path, "floor", tx=0.0, ty=0.0, tz=cfg.floor_t / 2,
               sx=cfg.tray_l, sy=cfg.tray_w, sz=cfg.floor_t,
               color=cfg.color, contact_offset=cfg.contact_offset)
    rail_y = cfg.tray_w / 2 + cfg.rail_t / 2
    for s, nm in ((1.0, "rail_l"), (-1.0, "rail_r")):
        _child_box(prim_path, nm, tx=0.0, ty=s * rail_y, tz=cfg.rail_h / 2,
                   sx=cfg.tray_l, sy=cfg.rail_t, sz=cfg.rail_h,
                   color=cfg.rail_color, contact_offset=cfg.contact_offset)
    _child_box(prim_path, "rail_end", tx=cfg.tray_l / 2 + cfg.rail_t / 2, ty=0.0,
               tz=cfg.rail_h / 2, sx=cfg.rail_t, sy=cfg.tray_w + 2 * cfg.rail_t,
               sz=cfg.rail_h, color=cfg.rail_color, contact_offset=cfg.contact_offset)
    return root


def _spawner_cfgs() -> dict[str, Any]:
    """Lazily-built @configclass spawner cfg types (heavy imports deferred)."""
    if "pedestal" not in _SPAWNER_CACHE:
        from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
        from isaaclab.sim.utils import clone
        from isaaclab.utils import configclass

        @configclass
        class PedestalSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pedestal)
            disc_r: float = 0.070
            disc_h: float = 0.014
            recess_r: float = 0.054
            wall_t: float = 0.012
            wall_h: float = 0.032
            color: tuple = (0.05, 0.05, 0.06)
            contact_offset: float = 0.002

        @configclass
        class BowlSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bowl)
            foot_r: float = 0.030
            foot_h: float = 0.012
            flange_r: float = 0.062
            flange_h: float = 0.014
            color: tuple = (0.05, 0.05, 0.06)
            contact_offset: float = 0.002

        @configclass
        class SlatSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_slat)
            length: float = 0.30
            width: float = 0.09
            thick: float = 0.008
            tab_x: float = 0.115
            tab_l: float = 0.030
            tab_w: float = 0.024
            tab_h: float = 0.030
            mu: float = 0.08
            color: tuple = (0.85, 0.82, 0.72)
            tab_color: tuple = (0.78, 0.12, 0.12)
            contact_offset: float = 0.002

        @configclass
        class TraySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_tray)
            tray_l: float = 0.36
            tray_w: float = 0.13
            floor_t: float = 0.010
            rail_t: float = 0.012
            rail_h: float = 0.034
            color: tuple = (0.25, 0.30, 0.40)
            rail_color: tuple = (0.35, 0.42, 0.55)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(pedestal=PedestalSpawnerCfg, bowl=BowlSpawnerCfg,
                              slat=SlatSpawnerCfg, tray=TraySpawnerCfg)
    return _SPAWNER_CACHE


def _compound_spawner_cfg(kind: str, defaults: dict, *, mass: float, kinematic: bool) -> Any:
    import isaaclab.sim as sim_utils

    return _spawner_cfgs()[kind](
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
        **defaults,
    )


# ----- scene cfg ------------------------------------------------------------------------------------
@dataclass
class WaitersPullSceneCfg(BaseCfg):
    """Config for `WaitersPullScene`. Gate honesty margins are derived in the module
    docstring."""

    # --- tunable: rubric thresholds ------------------------------------------------------------
    seat_xy_tol: float = tunable(0.020)  # bowl centre to pedestal centre, horizontal (m)
    seat_z_tol: float = tunable(0.007)  # |bowl z - pedestal z - seat_dz| below this (m)
    upright_max_deg: float = tunable(10.0)  # bowl / pedestal local +z within this of world up
    stow_x_tol: float = tunable(0.022)  # slat centre inside the tray, long axis (m)
    stow_y_tol: float = tunable(0.020)  # slat centre inside the tray, short axis (m)
    stow_z_tol: float = tunable(0.006)  # |slat centre z - tray floor rest z| below this (m)
    stow_yaw_max_deg: float = tunable(25.0)  # slat long axis vs tray long axis (mod 180)
    stow_flat_max_deg: float = tunable(8.0)  # slat tilt gate (a slat propped on a rail fails)
    pedestal_z_tol: float = tunable(0.008)  # pedestal still resting on the counter
    settle_speed: float = tunable(0.05)  # max |v| of every dynamic body when judging (m/s)
    latch_speed: float = tunable(0.10)  # rest milestones only latch while this slow
    draw_disp: float = tunable(0.06)  # slat displacement that certifies "draw in progress"
    draw_capture: float = tunable(0.025)  # bowl-over-pedestal radius during the draw latch
    draw_bowl_speed: float = tunable(0.30)  # bowl must be near-still during the draw latch
    draw_slat_speed: float = tunable(0.15)  # slat must be IN MOTION during the draw latch
    clear_dist: float = tunable(0.16)  # slat-to-pedestal horizontal distance = "clear"

    # --- tunable: randomization (the task-family knobs) ------------------------------------------
    pedestal_jitter: float = tunable(0.030)  # uniform +/- xy jitter of the pedestal
    pull_yaw_deg: float = tunable(25.0)  # uniform +/- yaw of the pull axis about "toward robot"
    bowl_jitter: float = tunable(0.008)  # uniform +/- xy jitter of the bowl on the slat
    tray_jitter: float = tunable(0.020)  # uniform +/- x jitter of the tray
    tray_yaw_deg: float = tunable(12.0)  # uniform +/- yaw of the tray about its park heading
    mirror_tray: bool = tunable(True)  # per-episode: tray parks on +y or -y side

    # --- tunable: placement (counter frame; intended arm base at (-0.42, 0, surface_z)) ---------
    surface_z: float = tunable(0.20)  # counter height; the arm base is mounted on the counter
    pedestal_c: tuple = tunable((0.06, 0.0))  # nominal pedestal centre
    slat_lead: float = tunable(0.07)  # slat centre offset from the pedestal along the pull axis
    tray_park: tuple = tunable((0.14, 0.34))  # tray centre (y mirrored per episode)

    # --- info: structure -------------------------------------------------------------------------
    bench_size: tuple = info((1.5, 1.3))  # kinematic counter slab top (x, y)
    disc_r: float = info(0.070)  # pedestal base disc: 140 mm across > 80 mm jaw -> ungraspable
    disc_h: float = info(0.014)
    recess_r: float = info(0.054)  # recess radius: 24 mm radial capture for the 30 mm foot
    wall_t: float = info(0.012)
    wall_h: float = info(0.032)  # rim top = disc_h + wall_h = 46 mm above the counter
    pedestal_mass: float = info(0.80)  # heavy: slat shear cannot drag it
    foot_r: float = info(0.030)
    foot_h: float = info(0.012)
    flange_r: float = info(0.062)  # 124 mm flange > 80 mm jaw; overhangs the 108 mm recess
    flange_h: float = info(0.014)
    bowl_mass: float = info(0.15)
    bowl_color: tuple = info((0.05, 0.05, 0.06))  # matte black, like the seed's akita bowls
    slat_l: float = info(0.30)
    slat_w: float = info(0.09)
    slat_t: float = info(0.008)
    tab_x: float = info(0.115)  # tab centre along the slat local +x (leading end)
    tab_l: float = info(0.030)
    tab_w: float = info(0.024)  # pinchable across the 24 mm tab
    tab_h: float = info(0.030)
    slat_mass: float = info(0.06)
    slat_mu: float = info(0.08)  # bound material, combine-mode "min": pair friction = 0.08
    tray_l: float = info(0.36)
    tray_w: float = info(0.13)
    tray_floor_t: float = info(0.010)
    tray_rail_t: float = info(0.012)
    tray_rail_h: float = info(0.034)
    contact_offset: float = info(0.002)
    rim_top: float = info(0.046)  # disc_h + wall_h (derived, kept explicit for the gates)
    seat_dz: float = info(0.034)  # seated bowl origin above pedestal origin: rim_top - foot_h

    def __post_init__(self) -> None:
        assert abs(self.rim_top - (self.disc_h + self.wall_h)) < 1e-9
        assert abs(self.seat_dz - (self.rim_top - self.foot_h)) < 1e-9


# ----- scene -----------------------------------------------------------------------------------------
@SCENES.register("waiters_pull")
class WaitersPullScene(BaseScene):
    cfg: WaitersPullSceneCfg

    def __init__(self, cfg: WaitersPullSceneCfg | None = None) -> None:
        super().__init__(cfg or WaitersPullSceneCfg())

    # ----- assets ---------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        """Ground, light, counter slab, pedestal bowl, dessert bowl, slat, tray, at nominal
        poses (reset() re-places everything and samples the layout)."""
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        z0 = c.surface_z

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
            "bench": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bench",
                spawn=sim_utils.CuboidCfg(
                    size=(c.bench_size[0], c.bench_size[1], z0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.08, 0.0, z0 / 2)),
            ),
        }

        out["pedestal"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pedestal",
            spawn=_compound_spawner_cfg(
                "pedestal",
                dict(disc_r=c.disc_r, disc_h=c.disc_h, recess_r=c.recess_r, wall_t=c.wall_t,
                     wall_h=c.wall_h, color=c.bowl_color, contact_offset=c.contact_offset),
                mass=c.pedestal_mass, kinematic=False),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.pedestal_c[0], c.pedestal_c[1], z0 + 0.001)),
        )
        out["bowl"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            spawn=_compound_spawner_cfg(
                "bowl",
                dict(foot_r=c.foot_r, foot_h=c.foot_h, flange_r=c.flange_r,
                     flange_h=c.flange_h, color=c.bowl_color, contact_offset=c.contact_offset),
                mass=c.bowl_mass, kinematic=False),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.pedestal_c[0], c.pedestal_c[1], z0 + c.rim_top + c.slat_t + 0.001)),
        )
        out["slat"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Slat",
            spawn=_compound_spawner_cfg(
                "slat",
                dict(length=c.slat_l, width=c.slat_w, thick=c.slat_t, tab_x=c.tab_x,
                     tab_l=c.tab_l, tab_w=c.tab_w, tab_h=c.tab_h, mu=c.slat_mu,
                     contact_offset=c.contact_offset),
                mass=c.slat_mass, kinematic=False),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.pedestal_c[0] - c.slat_lead, c.pedestal_c[1],
                     z0 + c.rim_top + c.slat_t / 2 + 0.001)),
        )
        out["tray"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            spawn=_compound_spawner_cfg(
                "tray",
                dict(tray_l=c.tray_l, tray_w=c.tray_w, floor_t=c.tray_floor_t,
                     rail_t=c.tray_rail_t, rail_h=c.tray_rail_h,
                     contact_offset=c.contact_offset),
                mass=1.0, kinematic=True),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(c.tray_park[0], c.tray_park[1], z0 + 0.0005)),
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

    # ----- lifecycle --------------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        """Grab handles + allocate the layout readback tensors and the progress latches."""
        super().bind(env)
        self.pedestal: RigidObject = env.iscene["pedestal"]
        self.bowl: RigidObject = env.iscene["bowl"]
        self.slat: RigidObject = env.iscene["slat"]
        self.tray: RigidObject = env.iscene["tray"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        self.pedestal_xy = torch.zeros(n, 2, device=dev)  # local-frame layout readback
        self.pull_yaw = torch.zeros(n, device=dev)  # world pull direction (slat local +x)
        self.slat_spawn_xy = torch.zeros(n, 2, device=dev)
        self.tray_xy = torch.zeros(n, 2, device=dev)
        self.tray_yaw = torch.zeros(n, device=dev)
        self.tray_side = torch.ones(n, device=dev)
        # progress latches (post_step; cleared per reset)
        self.ever_drawn = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_seated = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ever_stowed = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: sample the pedestal spot, the pull axis, the tray park; author the
        loaded sandwich (pedestal, slat across its rim, bowl riding the slat); clear latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]
        z0 = c.surface_z

        def u(amp: float, shape=(1,)) -> torch.Tensor:
            return (torch.rand(m, *shape, device=dev) * 2 - 1) * amp

        def sign(enabled: bool) -> torch.Tensor:
            if not enabled:
                return torch.ones(m, device=dev)
            return (torch.rand(m, device=dev) < 0.5).float() * 2 - 1

        def write(body: RigidObject, xy: torch.Tensor, z: float, yaw: torch.Tensor) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0:2] = xy
            st[:, 2] = z
            st[:, 3] = torch.cos(yaw / 2)
            st[:, 6] = torch.sin(yaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        # --- pedestal: near the counter centre ---
        pxy = torch.tensor(c.pedestal_c, device=dev).expand(m, 2) + u(c.pedestal_jitter, (2,))
        self.pedestal_xy[env_ids] = pxy
        write(self.pedestal, pxy, z0 + 0.001, torch.zeros(m, device=dev))

        # --- pull axis: toward the robot side (-x), +/- pull_yaw_deg ---
        theta = math.pi + u(math.radians(c.pull_yaw_deg)).squeeze(-1)
        self.pull_yaw[env_ids] = theta
        pull = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # (m,2)

        # --- slat: across the pedestal rim, leading (tab) end toward the robot ---
        sxy = pxy + c.slat_lead * pull
        self.slat_spawn_xy[env_ids] = sxy
        write(self.slat, sxy, z0 + c.rim_top + c.slat_t / 2 + 0.001, theta)

        # --- bowl: riding the slat, over the pedestal (small jitter), random yaw ---
        bxy = pxy + u(c.bowl_jitter, (2,))
        write(self.bowl, bxy, z0 + c.rim_top + c.slat_t + 0.001,
              u(math.pi).squeeze(-1))

        # --- tray: parked to one side, long axis roughly along x, open end toward -x ---
        ts = sign(c.mirror_tray)
        self.tray_side[env_ids] = ts
        txy = torch.stack([
            torch.full((m,), c.tray_park[0], device=dev) + u(c.tray_jitter).squeeze(-1),
            c.tray_park[1] * ts], dim=1)
        tyaw = u(math.radians(c.tray_yaw_deg)).squeeze(-1)
        self.tray_xy[env_ids] = txy
        self.tray_yaw[env_ids] = tyaw
        write(self.tray, txy, z0 + 0.0005, tyaw)

        for latch in (self.ever_drawn, self.ever_seated, self.ever_stowed):
            latch[env_ids] = False

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch progress milestones at sim rate. The draw latch demands the bowl still
        hovering over the pedestal at riding height while the slat has visibly moved — a
        capsize or a fly-through latches nothing. Rest latches are velocity-gated. Latched
        credit survives later mishaps, so along a correct run the score never decreases."""
        c = self.cfg
        ped = self.pedestal.data.root_pos_w
        bowl = self.bowl.data.root_pos_w
        d_xy = (bowl[:, :2] - ped[:, :2]).norm(dim=-1)
        bowl_speed = self.bowl.data.root_lin_vel_w.norm(dim=-1)
        slat_disp = (self.slat.data.root_pos_w[:, :2] - self.env_origins[:, :2]
                     - self.slat_spawn_xy).norm(dim=-1)
        riding = (bowl[:, 2] - ped[:, 2]) > (c.seat_dz + c.seat_z_tol + 0.004)
        slat_speed = self.slat.data.root_lin_vel_w.norm(dim=-1)
        # The draw credit demands the slat IN MOTION sliding out (draw_slat_speed) while the
        # bowl still hovers over the pedestal: a teleported slat (velocity written to zero)
        # latches nothing (smoke-tested).
        self.ever_drawn |= ((slat_disp > c.draw_disp) & (d_xy < c.draw_capture) & riding
                            & (bowl_speed < c.draw_bowl_speed)
                            & (slat_speed > c.draw_slat_speed))
        bowl_slow = bowl_speed < c.latch_speed
        self.ever_seated |= self.seated() & self.slat_clear() & bowl_slow
        slat_slow = self.slat.data.root_lin_vel_w.norm(dim=-1) < c.latch_speed
        self.ever_stowed |= self.stowed() & slat_slow

    # ----- state (full, restorable) ------------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "pedestal": self.pedestal.data.root_state_w[env_ids].clone(),
            "bowl": self.bowl.data.root_state_w[env_ids].clone(),
            "slat": self.slat.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "layout": torch.cat([
                self.pedestal_xy[env_ids], self.pull_yaw[env_ids, None],
                self.slat_spawn_xy[env_ids], self.tray_xy[env_ids],
                self.tray_yaw[env_ids, None], self.tray_side[env_ids, None]], dim=1),
            "latches": torch.stack([self.ever_drawn[env_ids], self.ever_seated[env_ids],
                                    self.ever_stowed[env_ids]], dim=1),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.pedestal.write_root_state_to_sim(state["pedestal"], env_ids)
        self.bowl.write_root_state_to_sim(state["bowl"], env_ids)
        self.slat.write_root_state_to_sim(state["slat"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        lay = state["layout"]
        self.pedestal_xy[env_ids] = lay[:, 0:2]
        self.pull_yaw[env_ids] = lay[:, 2]
        self.slat_spawn_xy[env_ids] = lay[:, 3:5]
        self.tray_xy[env_ids] = lay[:, 5:7]
        self.tray_yaw[env_ids] = lay[:, 7]
        self.tray_side[env_ids] = lay[:, 8]
        lat = state["latches"]
        self.ever_drawn[env_ids] = lat[:, 0]
        self.ever_seated[env_ids] = lat[:, 1]
        self.ever_stowed[env_ids] = lat[:, 2]

    # ----- description -------------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A wide gray kitchen counter. Near its centre stands a matte-BLACK PEDESTAL bowl "
            f"({2 * c.disc_r * 1000:.0f} mm across, {c.rim_top * 1000:.0f} mm tall): a heavy "
            f"black dish whose raised rim encloses an open {2 * c.recess_r * 1000:.0f} mm "
            f"round recess. A slick pale serving SLAT ({c.slat_l * 1000:.0f} x "
            f"{c.slat_w * 1000:.0f} x {c.slat_t * 1000:.0f} mm) lies flat ACROSS the "
            f"pedestal's rim, its overhanging end — which carries a raised RED grip tab "
            f"({c.tab_l * 1000:.0f} x {c.tab_w * 1000:.0f} x {c.tab_h * 1000:.0f} mm) — "
            f"pointing roughly toward the robot (the exact direction changes every episode). "
            f"A matte-black DESSERT bowl ({2 * c.flange_r * 1000:.0f} mm wide flange over a "
            f"{2 * c.foot_r * 1000:.0f} mm foot) rides ON the slat, directly above the "
            f"pedestal's recess. To one side (left or right, changing every episode) an open "
            f"blue-gray stowage TRAY (inner {c.tray_l * 1000:.0f} x {c.tray_w * 1000:.0f} mm, "
            f"low rails on three sides, open end toward the robot) is fixed to the counter. "
            f"The pedestal position, pull direction, and tray pose change every episode.\n"
            f"Goal: seat the dessert bowl on the pedestal, then stow the slat. Both bowls are "
            f"wider than a gripper can open and must never be grasped or shoved: the ONLY way "
            f"to seat the bowl is to DRAW THE SLAT OUT from under it by its red tab — the "
            f"waiter's tablecloth pull. Draw briskly and straight: while the slat slides, "
            f"friction drags the bowl with it, and the bowl's foot must still be over the "
            f"recess when support vanishes so it drops in and the flange lands centred on the "
            f"rim (a slow draw walks the bowl off the {c.recess_r * 1000 - c.foot_r * 1000:.0f} "
            f"mm capture margin and strands it on the rim wall or the counter — it cannot be "
            f"picked back up). Then lay the slat flat INSIDE the tray, aligned with it, fully "
            f"within the rails. The task is complete only when the bowl sits seated on the "
            f"pedestal, the slat rests in the tray, the pedestal stands upright in place, and "
            f"everything is at rest."
        )

    def instruction(self) -> str:
        return (
            "Pull the slat out from under the black bowl by its red tab — quickly, so the "
            "bowl drops onto the black pedestal bowl below and seats centred on its rim. "
            "Never grasp or push the bowls. Then lay the slat flat inside the stowage tray."
        )

    # ----- geometric predicates ------------------------------------------------------------------------
    def _up_z(self, quat: torch.Tensor) -> torch.Tensor:
        """z-component of a body's local +z in world (…,) for tilt gates."""
        from isaaclab.utils.math import quat_apply

        shape = quat.shape[:-1]
        ez = torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(*shape, 3)
        return quat_apply(quat.reshape(-1, 4), ez.reshape(-1, 3)).reshape(*shape, 3)[..., 2]

    def seated(self) -> torch.Tensor:
        """(N,) bool: dessert bowl seated on the pedestal — centred within `seat_xy_tol`,
        origin in the seated z band (`seat_dz` above the pedestal origin: flange on the rim,
        foot in the recess), upright. A bowl whose FOOT stands on the rim wall is 12 mm too
        high; a bowl on the counter is ~34 mm too low; both fail the z band."""
        c = self.cfg
        ped = self.pedestal.data.root_pos_w
        bowl = self.bowl.data.root_pos_w
        near = (bowl[:, :2] - ped[:, :2]).norm(dim=-1) < c.seat_xy_tol
        in_z = ((bowl[:, 2] - ped[:, 2]) - c.seat_dz).abs() < c.seat_z_tol
        up = self._up_z(self.bowl.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(c.upright_max_deg))
        return near & in_z & up

    def slat_clear(self) -> torch.Tensor:
        """(N,) bool: the slat is horizontally clear of the pedestal (no part of it can
        still be interposed)."""
        d = (self.slat.data.root_pos_w[:, :2]
             - self.pedestal.data.root_pos_w[:, :2]).norm(dim=-1)
        return d > self.cfg.clear_dist

    def pedestal_home(self) -> torch.Tensor:
        """(N,) bool: pedestal upright, resting on the counter (not capsized / knocked off)."""
        c = self.cfg
        z = self.pedestal.data.root_pos_w[:, 2] - self.env_origins[:, 2]
        in_z = (z - c.surface_z).abs() < c.pedestal_z_tol
        up = self._up_z(self.pedestal.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(c.upright_max_deg))
        return in_z & up

    def stowed(self) -> torch.Tensor:
        """(N,) bool: slat resting flat INSIDE the tray — tray-frame centre within the
        footprint tolerances, centre z in the floor-rest band, long axes aligned (mod 180),
        near-flat. A slat on the bare counter fails x/y; one propped on a rail fails the
        flat gate; one dumped crosswise fails the yaw gate."""
        from isaaclab.utils.math import quat_apply_inverse

        c = self.cfg
        rel = self.slat.data.root_pos_w - self.tray.data.root_pos_w
        loc = quat_apply_inverse(self.tray.data.root_quat_w, rel)  # (N,3)
        in_x = loc[:, 0].abs() < c.stow_x_tol
        in_y = loc[:, 1].abs() < c.stow_y_tol
        rest_z = c.tray_floor_t + c.slat_t / 2  # tray origin at floor bottom
        in_z = (loc[:, 2] - rest_z).abs() < c.stow_z_tol
        # long-axis alignment, mod pi
        from isaaclab.utils.math import quat_apply
        ex = torch.tensor([1.0, 0.0, 0.0], device=rel.device).expand(rel.shape[0], 3)
        sx = quat_apply(self.slat.data.root_quat_w, ex)
        tx = quat_apply(self.tray.data.root_quat_w, ex)
        cosang = (sx[:, :2] * tx[:, :2]).sum(dim=-1) / (
            sx[:, :2].norm(dim=-1) * tx[:, :2].norm(dim=-1) + 1e-9)
        aligned = cosang.abs().clamp(max=1.0) >= math.cos(math.radians(c.stow_yaw_max_deg))
        flat = self._up_z(self.slat.data.root_quat_w).clamp(-1, 1) >= \
            math.cos(math.radians(c.stow_flat_max_deg))
        return in_x & in_y & in_z & aligned & flat

    def settled(self) -> torch.Tensor:
        """(N,) bool: every dynamic body |lin vel| below `settle_speed`."""
        c = self.cfg
        still = self.bowl.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        still &= self.pedestal.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        still &= self.slat.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed
        return still

    def success(self) -> torch.Tensor:
        """(N,) bool: bowl seated on the upright in-place pedestal + slat stowed in the
        tray + everything settled."""
        return (self.seated() & self.stowed() & self.pedestal_home() & self.slat_clear()
                & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float 0..1 — additive latched milestones (monotone along a correct run):
        +0.20 clean draw in progress, +0.25 bowl seated with the slat clear, +0.25 slat
        stowed; 1.0 iff success()."""
        s = torch.zeros(self.env.num_envs, device=self.env.device)
        s = s + 0.20 * self.ever_drawn.float()
        s = s + 0.25 * self.ever_seated.float()
        s = s + 0.25 * self.ever_stowed.float()
        return torch.where(self.success(), torch.ones_like(s), s)


# Scene-level task (robot="null"): solve.py is the teleport certificate; the intended
# embodiment (single Franka + parallel jaw) is argued in TASK.md.
register_env("simgen", lambda: EnvCfg(scene="waiters_pull", robot="null"))
