"""DrawerStashScene — stash the wine bottle inside a SELF-CLOSING bottom drawer (i22).

Derived from libero_90 scene4 "put the wine bottle in the bottom drawer of the cabinet",
but the drawer FIGHTS BACK: it rides a prismatic slide with a return spring that pulls it
shut the moment nothing holds it (a shop-cabinet self-closing runner). The seed's plan —
open the drawer, then pick and place the bottle into the standing-open drawer — fails
here by construction: the drawer has sprung shut long before the bottle arrives.

The intended plan (order REQUIRED: open ≺ prop ≺ load ≺ shut):
  1. pull the drawer open by its handle, against the spring;
  2. prop it open — a yellow stop bar fits the floor gap between the drawer's face panel
     and the cabinet plinth (the classic doorstop move; the bar is the only way a single
     manipulator can free its grip);
  3. lay the bottle FLAT into the open basin (the cavity headroom is far below the
     bottle's standing height — an upright bottle can never ride in);
  4. remove the prop and let the spring glide the drawer shut with the bottle inside.

Success is judged on the PHYSICAL terminal state: the bottle fully below the basin rim
(both endpoints inside the basin interior, in the drawer's body frame), the drawer
CLOSED (slide displacement under `closed_tol`), everything settled — plus the latched
`loaded` flag, which only ever sets while the bottle is physically inside the OPEN
drawer without a same-substep teleport (so writing the bottle straight into the closed
drawer never counts). The seed's terminal state (bottle sitting in a standing-open
drawer) scores partial credit only.

Everything is procedural: cabinet = 6 kinematic slabs (plinth, side walls, back, top,
front header lintel); drawer = one compound rigid body (basin + oversized face panel +
handle bar) on a per-env authored USD prismatic joint; spring/damper applied in
`post_step` along the drawer's body-x axis (forces are body-frame, so the mechanism
survives the per-episode yaw of the whole cabinet). Bottle = body + neck cylinder
colliders with a visual foil cap. Randomization: whole cabinet+drawer pair translated
and yawed rigidly (the joint sees an unchanged relative pose — the pressure_plate
precedent), bottle and prop bar jittered + free yaw on opposite flanks, clear of the
drawer's sweep.

Heavy imports (isaaclab, pxr) are deferred so importing this module stays app-free.
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


# ----- custom compound spawners (the pen_holder pattern) ----------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_root(stage_path: str, cfg: Any, translation, orientation):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, stage_path)
    root = xform.GetPrim()
    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.mass_props.mass))
    # Gentle overlap resolution (the pen_holder lesson): cap the contact-solver pop so a
    # dropped bottle or a slamming drawer never gets ejected ballistically.
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    return stage, root


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset=None):
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(cube.GetPrim(), contact_offset)
    return cube


def _spawn_drawer(prim_path: str, cfg: Any, translation=None, orientation=None):
    """One rigid body: basin floor + 4 walls, an oversized face panel on the -x end, and a
    handle bar (collider) standing off the panel front on two visual posts. Drawer body
    frame: origin at the basin-floor TOP center; -x = opening (pull) direction."""
    stage, root = _apply_root(prim_path, cfg, translation, orientation)
    from pxr import PhysxSchema

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.0)
    # Never sleep: the post_step spring is applied through the (deprecated) external-force
    # API, which does not wake a sleeping body — a drawer parked on the prop for a few
    # seconds would otherwise ignore the spring after the prop is pulled.
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)

    co = cfg.contact_offset
    bx, by, wt, wh, ft = cfg.basin_x, cfg.basin_y, cfg.wall_t, cfg.wall_h, cfg.floor_t
    body_col = cfg.color
    _box(stage, f"{prim_path}/floor", (bx + 2 * wt, by + 2 * wt, ft),
         (0.0, 0.0, -ft / 2), body_col, co)
    _box(stage, f"{prim_path}/wall_l", (bx + 2 * wt, wt, wh),
         (0.0, -(by + wt) / 2, wh / 2), body_col, co)
    _box(stage, f"{prim_path}/wall_r", (bx + 2 * wt, wt, wh),
         (0.0, (by + wt) / 2, wh / 2), body_col, co)
    _box(stage, f"{prim_path}/wall_f", (wt, by, wh),
         (-(bx + wt) / 2, 0.0, wh / 2), body_col, co)
    _box(stage, f"{prim_path}/wall_b", (wt, by, wh),
         ((bx + wt) / 2, 0.0, wh / 2), body_col, co)
    # face panel: wider and taller than the cabinet opening — the closed stop AND the
    # surface the prop bar presses against
    px_c = -(bx / 2 + wt + cfg.panel_t / 2)
    _box(stage, f"{prim_path}/panel", (cfg.panel_t, cfg.panel_w, cfg.panel_h),
         (px_c, 0.0, cfg.panel_zc), cfg.panel_color, co)
    # handle: horizontal bar collider on two visual posts
    hb_x = px_c - cfg.panel_t / 2 - cfg.handle_standoff - 0.010
    _box(stage, f"{prim_path}/handle", (0.020, 0.14, 0.022),
         (hb_x, 0.0, cfg.panel_zc), (0.75, 0.75, 0.78), co)
    for k, py in enumerate((-0.055, 0.055)):
        _box(stage, f"{prim_path}/post_{k}", (cfg.handle_standoff, 0.016, 0.016),
             (px_c - cfg.panel_t / 2 - cfg.handle_standoff / 2, py, cfg.panel_zc),
             (0.75, 0.75, 0.78))
    return root


def _spawn_bottle(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Wine bottle: body cylinder + neck cylinder (both colliders) along local +z, plus a
    visual-only foil cap. Root at the mid-point of the full collider length."""
    stage, root = _apply_root(prim_path, cfg, translation, orientation)
    from pxr import Gf, PhysxSchema, UsdGeom

    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateLinearDampingAttr(0.05)
    pxrb.CreateAngularDampingAttr(0.15)  # a rolling bottle must cross the settle gate

    L = cfg.body_l + cfg.neck_l
    body_zc = -L / 2 + cfg.body_l / 2
    neck_zc = L / 2 - cfg.neck_l / 2

    body = UsdGeom.Cylinder.Define(stage, f"{prim_path}/body")
    body.CreateRadiusAttr(cfg.body_r)
    body.CreateHeightAttr(cfg.body_l)
    body.CreateExtentAttr([Gf.Vec3f(-cfg.body_r, -cfg.body_r, -cfg.body_l / 2),
                           Gf.Vec3f(cfg.body_r, cfg.body_r, cfg.body_l / 2)])
    UsdGeom.Xformable(body.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, body_zc))
    body.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _collide(body.GetPrim(), cfg.contact_offset)

    neck = UsdGeom.Cylinder.Define(stage, f"{prim_path}/neck")
    neck.CreateRadiusAttr(cfg.neck_r)
    neck.CreateHeightAttr(cfg.neck_l)
    neck.CreateExtentAttr([Gf.Vec3f(-cfg.neck_r, -cfg.neck_r, -cfg.neck_l / 2),
                           Gf.Vec3f(cfg.neck_r, cfg.neck_r, cfg.neck_l / 2)])
    UsdGeom.Xformable(neck.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, neck_zc))
    neck.CreateDisplayColorAttr([Gf.Vec3f(*cfg.color)])
    _collide(neck.GetPrim(), cfg.contact_offset)

    cap = UsdGeom.Cylinder.Define(stage, f"{prim_path}/cap")  # visual only — no collider
    cr = cfg.neck_r + 0.002
    cap.CreateRadiusAttr(cr)
    cap.CreateHeightAttr(0.012)
    cap.CreateExtentAttr([Gf.Vec3f(-cr, -cr, -0.006), Gf.Vec3f(cr, cr, 0.006)])
    UsdGeom.Xformable(cap.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, L / 2 - 0.006))
    cap.CreateDisplayColorAttr([Gf.Vec3f(*cfg.cap_color)])
    return root


def _drawer_spawner_cfg(c: DrawerStashSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "drawer" not in _SPAWNER_CACHE:

        @configclass
        class DrawerSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_drawer)
            basin_x: float = 0.20
            basin_y: float = 0.24
            wall_t: float = 0.010
            wall_h: float = 0.075
            floor_t: float = 0.012
            panel_t: float = 0.020
            panel_w: float = 0.34
            panel_h: float = 0.14
            panel_zc: float = 0.0
            handle_standoff: float = 0.028
            color: tuple = (0.92, 0.90, 0.86)
            panel_color: tuple = (0.96, 0.95, 0.92)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["drawer"] = DrawerSpawnerCfg

    return _SPAWNER_CACHE["drawer"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.drawer_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        basin_x=c.basin_x, basin_y=c.basin_y, wall_t=c.wall_t, wall_h=c.wall_h,
        floor_t=c.floor_t, panel_t=c.panel_t, panel_w=c.panel_w, panel_h=c.panel_h,
        panel_zc=c.panel_zc, handle_standoff=c.handle_standoff,
        contact_offset=c.contact_offset,
    )


def _bottle_spawner_cfg(c: DrawerStashSceneCfg) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "bottle" not in _SPAWNER_CACHE:

        @configclass
        class BottleSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bottle)
            body_r: float = 0.030
            body_l: float = 0.130
            neck_r: float = 0.012
            neck_l: float = 0.055
            color: tuple = (0.10, 0.26, 0.12)
            cap_color: tuple = (0.55, 0.08, 0.10)
            contact_offset: float = 0.003

        _SPAWNER_CACHE["bottle"] = BottleSpawnerCfg

    return _SPAWNER_CACHE["bottle"](
        mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        body_r=c.body_r, body_l=c.body_l, neck_r=c.neck_r, neck_l=c.neck_l,
        contact_offset=c.contact_offset,
    )


# ----- scene cfg --------------------------------------------------------------------------------
@dataclass
class DrawerStashSceneCfg(BaseCfg):
    """Config for `DrawerStashScene`. Cabinet-local frame: front (opening) faces -x,
    ground at z=0; the whole assembly is translated + yawed per episode."""

    # --- tunable: rubric thresholds -----------------------------------------------------------
    closed_tol: float = tunable(0.020)  # slide open-distance below this counts as CLOSED (m)
    open_min: float = tunable(0.10)  # open distance that latches the `opened` achievement
    load_open_min: float = tunable(0.06)  # bottle can only latch as loaded while open >= this
    teleport_step: float = tunable(0.05)  # per-substep bottle jump above this = teleport (no latch)
    settle_speed: float = tunable(0.05)  # max |v| (bottle AND drawer) when judging (m/s)
    open_ref: float = tunable(0.121)  # nominal propped-open distance (score shaping only)

    # --- tunable: spring (the self-closing runner) --------------------------------------------
    spring_k: float = tunable(25.0)  # N/m toward the closed home
    spring_c: float = tunable(6.0)  # N*s/m along the slide axis
    spring_fmax: float = tunable(8.0)  # force cap (N)

    # --- tunable: randomization ---------------------------------------------------------------
    # NOTE the jointed cabinet+drawer pair is NEVER teleported per episode: PhysX does
    # not reliably re-anchor the authored joint's body0 frame after kinematic teleports
    # (probes 2/3: yawed or even translated resets intermittently left the drawer
    # squeeze-wedged against the cavity walls). Randomization lives in the OBJECTS:
    # bottle/prop jitter + free yaw + a per-episode flank swap (bottle left or right of
    # the cabinet, prop bar on the opposite flank).
    base_jitter: float = tunable(0.0)  # keep 0 (see note above)
    base_yaw_deg: float = tunable(0.0)  # keep 0 (see note above)
    item_jitter: float = tunable(0.06)  # +/- xy jitter of bottle and prop bar (their flanks)
    flank_swap: bool = tunable(True)  # per-episode: which side the bottle spawns on

    # --- tunable: placement (cabinet frame) -----------------------------------------------------
    base_pos: tuple = tunable((0.18, 0.0))  # cabinet frame origin on the ground
    bottle_slot: tuple = tunable((-0.30, 0.27))  # bottle spawn (|y| flank; sign sampled)
    prop_slot: tuple = tunable((-0.30, 0.27))  # prop bar spawn (opposite flank)

    # --- info: drawer / cabinet structure -------------------------------------------------------
    basin_x: float = info(0.20)  # basin interior depth (slide direction)
    basin_y: float = info(0.24)  # basin interior width — the bottle lies along this
    wall_t: float = info(0.010)
    wall_h: float = info(0.075)  # basin rim height above the floor top
    floor_t: float = info(0.012)  # thick floor: no tunneling under a dropped bottle
    slide_z: float = info(0.055)  # drawer underside height (5 mm above the plinth top)
    panel_t: float = info(0.020)
    panel_w: float = info(0.34)  # wider than the cabinet: the panel can never enter it
    panel_h: float = info(0.14)
    panel_bot: float = info(0.010)  # panel lower edge height (reaches down to the prop gap)
    handle_standoff: float = info(0.028)
    front_x: float = info(-0.13)  # cabinet front plane (frame face), cabinet-local
    slab_top: float = info(0.05)  # plinth top = bottom of the opening
    header_bot: float = info(0.165)  # lintel bottom = top of the opening (jams a standing bottle)
    top_bot: float = info(0.175)
    cab_h: float = info(0.195)
    cab_t: float = info(0.020)
    side_clear: float = info(0.005)
    back_clear: float = info(0.013)
    travel: float = info(0.16)  # full pull; joint limits are +/-(travel + 0.015)
    drawer_mass: float = info(0.5)
    bar_size: tuple = info((0.12, 0.05, 0.042))  # prop bar (x = along slide); < slide_z tall
    bar_mass: float = info(0.15)
    body_r: float = info(0.030)
    body_l: float = info(0.130)
    neck_r: float = info(0.012)
    neck_l: float = info(0.055)
    bottle_mass: float = info(0.45)
    contact_offset: float = info(0.003)  # tight offsets: 5 mm slide clearances stay real

    # Derived (filled in __post_init__).
    bottle_len: float = field(default=None, init=False)
    root_z: float = field(default=None, init=False)  # drawer origin (basin-floor top) height
    home_x: float = field(default=None, init=False)  # drawer origin x at CLOSED, cabinet-local
    panel_zc: float = field(default=None, init=False)  # panel center z, drawer-local
    cab_half_w: float = field(default=None, init=False)
    back_outer: float = field(default=None, init=False)
    inner_back: float = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.bottle_len = round(self.body_l + self.neck_l, 4)
        self.root_z = round(self.slide_z + self.floor_t, 4)
        # closed pose: panel BACK face (drawer-local -(basin_x/2 + wall_t)) 1 mm proud of
        # the front plane — the spring's home. NOTE no panel_t term: adding it placed the
        # panel 21 mm INSIDE the cabinet walls and the depenetration ejection of that
        # overlap was probe run 1's mystery 21 mm open-attractor.
        self.home_x = round(self.front_x + 0.001 + self.basin_x / 2 + self.wall_t, 4)
        self.panel_zc = round(self.panel_bot + self.panel_h / 2 - self.root_z, 4)
        self.cab_half_w = round(self.basin_y / 2 + self.wall_t + self.side_clear
                                + self.cab_t, 4)
        self.inner_back = round(self.home_x + self.basin_x / 2 + self.wall_t
                                + self.back_clear, 4)
        self.back_outer = round(self.inner_back + self.cab_t, 4)

    # cabinet parts: name -> (size, center), cabinet-local (shared by assets() and reset())
    def cab_parts(self) -> dict[str, tuple[tuple, tuple]]:
        fx, ib, bo = self.front_x, self.inner_back, self.back_outer
        w2 = self.cab_half_w
        st, hb, tb, H, t = self.slab_top, self.header_bot, self.top_bot, self.cab_h, self.cab_t
        return {
            "slab": ((bo - fx, 2 * w2, st), ((fx + bo) / 2, 0.0, st / 2)),
            "wall_l": ((ib - fx, t, tb - st), ((fx + ib) / 2, -(w2 - t / 2), (st + tb) / 2)),
            "wall_r": ((ib - fx, t, tb - st), ((fx + ib) / 2, (w2 - t / 2), (st + tb) / 2)),
            "back_wall": ((t, 2 * w2, H - st), ((ib + bo) / 2, 0.0, (st + H) / 2)),
            "top_plate": ((bo - fx, 2 * w2, H - tb), ((fx + bo) / 2, 0.0, (tb + H) / 2)),
            "header": ((t, 2 * w2, H - hb), (fx + t / 2, 0.0, (hb + H) / 2)),
        }


# ----- scene ------------------------------------------------------------------------------------
@SCENES.register("drawer_stash")
class DrawerStashScene(BaseScene):
    cfg: DrawerStashSceneCfg

    CAB_PARTS = ("slab", "wall_l", "wall_r", "back_wall", "top_plate", "header")

    def __init__(self, cfg: DrawerStashSceneCfg | None = None) -> None:
        super().__init__(cfg or DrawerStashSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        bx, by = c.base_pos
        tight = sim_utils.CollisionPropertiesCfg(contact_offset=c.contact_offset,
                                                 rest_offset=0.0)
        cab_col = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.80, 0.75))

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
        for name, (size, ctr) in c.cab_parts().items():
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cab_" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=tight,
                    visual_material=cab_col,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(bx + ctr[0], by + ctr[1], ctr[2])),
            )
        out["drawer"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Drawer",
            spawn=_drawer_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(bx + c.home_x, by, c.root_z)),
        )
        out["bottle"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bottle",
            spawn=_bottle_spawner_cfg(c),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(bx + c.bottle_slot[0], by + c.bottle_slot[1],
                     c.bottle_len / 2 + 0.002)),
        )
        out["prop"] = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/PropBar",
            spawn=sim_utils.CuboidCfg(
                size=c.bar_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.bar_mass),
                collision_props=tight,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.76, 0.10)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(bx + c.prop_slot[0], by + c.prop_slot[1],
                     c.bar_size[2] / 2 + 0.002)),
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

    # ----- lifecycle ----------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        n = env.num_envs
        dev = env.device
        self.drawer: RigidObject = env.iscene["drawer"]
        self.bottle: RigidObject = env.iscene["bottle"]
        self.prop: RigidObject = env.iscene["prop"]
        self.cab: dict[str, RigidObject] = {k: env.iscene[k] for k in self.CAB_PARTS}
        self.env_origins = env.iscene.env_origins
        self._author_joint()
        # per-episode base transform of the cabinet frame (world)
        self._base_pos = torch.zeros(n, 3, device=dev)
        self._base_quat = torch.zeros(n, 4, device=dev)
        self._base_quat[:, 0] = 1.0
        self._home_w = torch.zeros(n, 3, device=dev)  # drawer origin at CLOSED, world
        # latched achievements + anti-teleport tracker
        self._opened = torch.zeros(n, dtype=torch.bool, device=dev)
        self._loaded = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_bottle = torch.zeros(n, 3, device=dev)
        # reset grace: for the first two substeps after a reset the drawer is re-pinned
        # at home — absorbs any write-timing transient around the kinematic cabinet.
        self._grace = torch.zeros(n, dtype=torch.long, device=dev)
        # which flank the bottle spawned on (+1 = +y, -1 = -y); prop is on -side.
        self._bottle_side = torch.ones(n, device=dev)

    def _author_joint(self) -> None:
        """Per env: one prismatic slide (axis X) between the kinematic plinth and the
        drawer. Local frames both identity; symmetric limits (sign-convention hedge —
        the microwave-button lesson); pair collision disabled (the 5 mm underside gap
        is real clearance, not a contact)."""
        from pxr import Gf, PhysxSchema, UsdPhysics

        import omni.usd

        c = self.cfg
        stage = omni.usd.get_context().get_stage()
        slab_ctr = c.cab_parts()["slab"][1]
        lim = c.travel + 0.015
        for i in range(self.env.num_envs):
            base = f"/World/envs/env_{i}"
            j = UsdPhysics.PrismaticJoint.Define(stage, f"{base}/drawer_slide")
            j.CreateBody0Rel().SetTargets([f"{base}/Cab_slab"])
            j.CreateBody1Rel().SetTargets([f"{base}/Drawer"])
            j.CreateCollisionEnabledAttr(False)
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(c.home_x - slab_ctr[0], 0.0,
                                           c.root_z - slab_ctr[2]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-lim)
            j.CreateUpperLimitAttr(lim)
            plim = PhysxSchema.PhysxLimitAPI.Apply(j.GetPrim(), "linear")
            if hasattr(plim, "CreateContactDistanceAttr"):
                plim.CreateContactDistanceAttr(0.001)

    # ----- reset --------------------------------------------------------------------------------
    def to_world(self, p_local, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Cabinet-local point -> world (N,3), through the per-episode base transform."""
        from isaaclab.utils.math import quat_apply

        ids = env_ids if env_ids is not None else torch.arange(
            self.env.num_envs, device=self.env.device)
        p = torch.as_tensor(p_local, dtype=torch.float, device=self.env.device)
        if p.dim() == 1:
            p = p.unsqueeze(0).expand(len(ids), 3)
        return self._base_pos[ids] + quat_apply(self._base_quat[ids], p)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Rigid pair teleport: every cabinet part AND the closed drawer get the same
        sampled planar transform (the joint sees an unchanged relative pose). Bottle
        upright and prop bar flat on opposite flanks, jitter + free yaw."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.base_yaw_deg)
        half = yaw / 2
        q = torch.zeros(m, 4, device=dev)
        q[:, 0] = torch.cos(half)
        q[:, 3] = torch.sin(half)
        base = torch.zeros(m, 3, device=dev)
        base[:, 0] = c.base_pos[0]
        base[:, 1] = c.base_pos[1]
        base[:, :2] += (torch.rand(m, 2, device=dev) * 2 - 1) * c.base_jitter
        base += origin
        self._base_pos[env_ids] = base
        self._base_quat[env_ids] = q

        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def rot_xy(px: torch.Tensor, py: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return cy * px - sy * py, sy * px + cy * py

        def write(body, px, py, pz, quat) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = base[:, 0] + px
            st[:, 1] = base[:, 1] + py
            st[:, 2] = pz
            st[:, 3:7] = quat
            body.write_root_state_to_sim(st, env_ids)

        for name, (_size, ctr) in c.cab_parts().items():
            px, py = rot_xy(torch.full((m,), ctr[0], device=dev),
                            torch.full((m,), ctr[1], device=dev))
            write(self.cab[name], px, py, torch.full((m,), ctr[2], device=dev), q)

        hx, hy = rot_xy(torch.full((m,), c.home_x, device=dev),
                        torch.zeros(m, device=dev))
        write(self.drawer, hx, hy, torch.full((m,), c.root_z, device=dev), q)
        self._home_w[env_ids] = torch.stack(
            [base[:, 0] + hx, base[:, 1] + hy,
             torch.full((m,), c.root_z, device=dev)], dim=1)

        # flank swap: bottle on one side of the cabinet, prop bar on the other
        if c.flank_swap:
            side = torch.where(torch.rand(m, device=dev) < 0.5,
                               torch.ones(m, device=dev), -torch.ones(m, device=dev))
        else:
            side = torch.ones(m, device=dev)
        self._bottle_side[env_ids] = side

        # bottle: standing upright on its flank, free yaw about its own axis
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.item_jitter
        bx0, by0 = rot_xy(torch.full((m,), c.bottle_slot[0], device=dev) + jit[:, 0],
                          side * abs(c.bottle_slot[1]) + jit[:, 1])
        bhalf = torch.rand(m, device=dev) * math.pi
        bq = torch.zeros(m, 4, device=dev)
        bq[:, 0] = torch.cos(bhalf)
        bq[:, 3] = torch.sin(bhalf)
        write(self.bottle, bx0, by0,
              torch.full((m,), c.bottle_len / 2 + 0.002, device=dev), bq)

        # prop bar: flat on the opposite flank, free yaw
        jit = (torch.rand(m, 2, device=dev) * 2 - 1) * c.item_jitter
        px0, py0 = rot_xy(torch.full((m,), c.prop_slot[0], device=dev) + jit[:, 0],
                          -side * abs(c.prop_slot[1]) + jit[:, 1])
        phalf = torch.rand(m, device=dev) * math.pi
        pq = torch.zeros(m, 4, device=dev)
        pq[:, 0] = torch.cos(phalf)
        pq[:, 3] = torch.sin(phalf)
        write(self.prop, px0, py0,
              torch.full((m,), c.bar_size[2] / 2 + 0.002, device=dev), pq)

        self._opened[env_ids] = False
        self._loaded[env_ids] = False
        self._prev_bottle[env_ids] = self.bottle.data.root_pos_w[env_ids]
        self._grace[env_ids] = 2

    # ----- geometry queries -----------------------------------------------------------------------
    def axis_w(self) -> torch.Tensor:
        """(N,3) the slide axis (cabinet local +x) in world. Opening = -axis direction."""
        from isaaclab.utils.math import quat_apply

        ex = torch.tensor([1.0, 0.0, 0.0], device=self.env.device).expand(
            self.env.num_envs, 3)
        return quat_apply(self._base_quat, ex)

    def slide_disp(self) -> torch.Tensor:
        """(N,) signed slide displacement from the closed home (negative = open)."""
        d = self.drawer.data.root_pos_w - self._home_w
        return (d * self.axis_w()).sum(dim=-1)

    def open_dist(self) -> torch.Tensor:
        return (-self.slide_disp()).clamp(min=0.0)

    def closed(self) -> torch.Tensor:
        return self.open_dist() < self.cfg.closed_tol

    def _bottle_ends_drawer(self) -> torch.Tensor:
        """Both bottle endpoints in the DRAWER body frame, shape (N, 2, 3)."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        c = self.cfg
        n = self.env.num_envs
        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(n, 3)
        axis = quat_apply(self.bottle.data.root_quat_w, ez)
        pos = self.bottle.data.root_pos_w
        ends = torch.stack([pos - axis * c.bottle_len / 2,
                            pos + axis * c.bottle_len / 2], dim=1)  # (N,2,3)
        dq = self.drawer.data.root_quat_w[:, None, :].expand(n, 2, 4).reshape(n * 2, 4)
        dp = self.drawer.data.root_pos_w[:, None, :]
        return quat_apply_inverse(dq, (ends - dp).reshape(n * 2, 3)).reshape(n, 2, 3)

    def inside(self) -> torch.Tensor:
        """(N,) bool: bottle FULLY inside the basin — both endpoints within the basin
        walls (small contact slop; the walls bound xy physically) and BELOW THE RIM, in
        the drawer's body frame. The z term is the teeth: a bottle standing upright or
        leaning with an end past the rim never counts."""
        c = self.cfg
        e = self._bottle_ends_drawer()
        ok_x = e[:, :, 0].abs() < c.basin_x / 2 + 0.003
        ok_y = e[:, :, 1].abs() < c.basin_y / 2 + 0.003
        ok_z = (e[:, :, 2] > -0.005) & (e[:, :, 2] < c.wall_h)
        return (ok_x & ok_y & ok_z).all(dim=1)

    def settled(self) -> torch.Tensor:
        v_b = self.bottle.data.root_lin_vel_w.norm(dim=-1)
        v_d = self.drawer.data.root_lin_vel_w.norm(dim=-1)
        return (v_b < self.cfg.settle_speed) & (v_d < self.cfg.settle_speed)

    # ----- mechanism (every substep) --------------------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device

        # reset grace: re-pin the drawer at home while the kinematic cabinet catches up
        gids = (self._grace > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(gids):
            st = torch.zeros(len(gids), 13, device=dev)
            st[:, 0:3] = self._home_w[gids]
            st[:, 3:7] = self._base_quat[gids]
            self.drawer.write_root_state_to_sim(st, gids)
            self._grace[gids] -= 1

        open_d = self.open_dist()
        self._opened |= open_d >= c.open_min
        b_pos = self.bottle.data.root_pos_w
        d_step = (b_pos - self._prev_bottle).norm(dim=-1)
        self._loaded |= (self.inside() & (open_d >= c.load_open_min)
                         & (d_step < c.teleport_step))
        self._prev_bottle = b_pos.clone()

        # return spring toward the closed home + damping, along the drawer's own x axis
        # (external wrenches are BODY-frame, so this follows the cabinet yaw for free)
        disp = self.slide_disp()
        v_ax = (self.drawer.data.root_lin_vel_w * self.axis_w()).sum(dim=-1)
        f = (-c.spring_k * disp - c.spring_c * v_ax).clamp(-c.spring_fmax, c.spring_fmax)
        forces = torch.zeros(n, 1, 3, device=dev)
        forces[:, 0, 0] = f
        self.drawer.set_external_force_and_torque(forces, torch.zeros(n, 1, 3, device=dev))

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        bodies = {"drawer": self.drawer, "bottle": self.bottle, "prop": self.prop,
                  **self.cab}
        return {
            "bodies": {k: b.data.root_state_w[env_ids].clone() for k, b in bodies.items()},
            "base_pos": self._base_pos[env_ids].clone(),
            "base_quat": self._base_quat[env_ids].clone(),
            "home_w": self._home_w[env_ids].clone(),
            "opened": self._opened[env_ids].clone(),
            "loaded": self._loaded[env_ids].clone(),
            "prev_bottle": self._prev_bottle[env_ids].clone(),
            "bottle_side": self._bottle_side[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        bodies = {"drawer": self.drawer, "bottle": self.bottle, "prop": self.prop,
                  **self.cab}
        for k, b in bodies.items():
            b.write_root_state_to_sim(state["bodies"][k], env_ids)
        self._base_pos[env_ids] = state["base_pos"]
        self._base_quat[env_ids] = state["base_quat"]
        self._home_w[env_ids] = state["home_w"]
        self._opened[env_ids] = state["opened"]
        self._loaded[env_ids] = state["loaded"]
        self._prev_bottle[env_ids] = state["prev_bottle"]
        self._bottle_side[env_ids] = state["bottle_side"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A low white cabinet stands on the floor with one bottom drawer behind an "
            f"oversized face panel with a steel handle. The drawer rides a SELF-CLOSING "
            f"runner: a return spring pulls it shut the moment nothing holds it open. "
            f"A green wine bottle ({c.bottle_len * 100:.0f} cm tall, body "
            f"{2 * c.body_r * 100:.0f} cm wide) stands on one flank of the cabinet; a "
            f"yellow stop bar ({c.bar_size[0] * 100:.0f} x {c.bar_size[1] * 100:.0f} x "
            f"{c.bar_size[2] * 100:.0f} cm) lies on the other. The drawer basin is "
            f"shallow and the opening lintel is low: the bottle only fits LYING FLAT, "
            f"and an upright bottle jams the drawer on the lintel.\n"
            f"Goal: get the bottle stowed inside the CLOSED drawer. Pull the drawer "
            f"open by its handle; the stop bar fits the floor gap between the face "
            f"panel and the cabinet plinth to prop the drawer open while you fetch the "
            f"bottle; lay the bottle flat inside the basin; then remove the bar and let "
            f"the drawer glide shut. Success requires the bottle fully below the basin "
            f"rim inside the closed, settled drawer — a bottle in a still-open drawer, "
            f"or one balanced upright, does not finish the job."
        )

    # ----- rubric ---------------------------------------------------------------------------------
    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.15 latched once the drawer was ever pulled properly
        open; 0.45 latched once the bottle was ever physically inside the open drawer;
        0.45..0.80 live as the loaded drawer approaches closed; 1.0 iff success()."""
        c = self.cfg
        n = self.env.num_envs
        dev = self.env.device
        s = torch.zeros(n, device=dev)
        s = torch.where(self._opened, torch.full_like(s, 0.15), s)
        s = torch.where(self._loaded, torch.maximum(s, torch.full_like(s, 0.45)), s)
        live = 0.45 + 0.35 * (1.0 - self.open_dist() / c.open_ref).clamp(0.0, 1.0)
        s = torch.where(self._loaded & self.inside(), torch.maximum(s, live), s)
        s = torch.where(self.success(), torch.ones_like(s), s)
        return s

    def success(self) -> torch.Tensor:
        """(N,) bool: the bottle was loaded through the open drawer (latched,
        teleport-proof), is NOW fully inside the basin, the drawer is CLOSED, and
        bottle + drawer are settled — the physical terminal state."""
        return self._loaded & self.inside() & self.closed() & self.settled()


register_env("simgen", lambda: EnvCfg(scene="drawer_stash", robot="null"))
