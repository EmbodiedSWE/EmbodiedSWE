"""FlatPackCrateScene — assemble the crate around the salad-dressing bottle, then seal it.

Derived from libero/libero_pick_salad_dressing ("pick the salad dressing and place it in
the basket": grasp the amber bottle among grocery distractors, carry it through free air,
drop it into an open basket that exists, fully formed, from the first frame — one
pick-and-place judged by a containment bbox). Here the receptacle strategy is inverted:
the container DOES NOT EXIST at reset. The episode starts with a FLAT-PACK — a heavy
BASE FIXTURE (floor pad + two fixed walls on its +x/+y sides + two open vertical SLOT
CHANNELS with funnel flares on its -x/-y sides), two loose WALL PANELS lying flat on the
ground, a LID PLATE (with a square registration lip on its underside) lying off to the
side, the amber DRESSING bottle standing on the open floor, and a red DECOY bottle. The
goal is to BUILD the crate: slide each panel down into a slot channel until it seats on
the pad, stand the dressing bottle on the pad interior, and finally seat the lid on the
wall tops so its lip registers inside the walls — sealing the bottle in. The decoy must
stay OUT. The lid physically forces the ordering: once seated it covers both slot mouths
and the open top, so neither a panel nor the bottle can enter afterwards (entering while
the lid is seated latches a permanent BREACH fail — anti-teleport).

Assets are fully procedural (compound-spawner pattern — child colliders of one rigid
compound never self-collide):
  - base: heavy DYNAMIC compound (30 kg; dynamic, not kinematic — kinematic anchors
    stay world-fixed after teleports on this stack). Local frame: origin at the pad
    centre on the ground. Children: 300x300x20 mm floor pad; two fixed walls, 12 mm
    thick, 170 mm tall, inner faces at +/-65 mm (+x "north", +y "east" sides); two
    slot channels on the -x ("south") and -y ("west") sides, each an inner+outer rail
    pair (gap 22 mm centred at -71 mm, rails 50 mm tall on the pad) with funnel
    flares on top (gap 34 mm, 15 mm tall) and end posts (gap +/-54 mm along the
    channel). Channel hardware stops 2 mm short of the OTHER channel's panel plane,
    so the two seated panels never collide (they meet only near the shared corner,
    leaving a ~20 mm corner gap far smaller than the bottle).
  - panels (x2, interchangeable): plain cuboids 16 x 100 x 170 mm, 0.12 kg, spawned
    lying FLAT on the ground. Seated in a channel: bottom on the pad top (20 mm),
    top edge flush with the fixed walls at 190 mm. Channel slack: +/-3 mm across the
    thickness, +/-4 mm along the channel; the flare mouth captures +/-9 mm.
  - lid: DYNAMIC compound (0.30 kg), origin at the plate centre: 200x200x10 mm
    plate + 112x112x12 mm registration lip underneath. Seated: plate underside on
    the wall/panel tops (z 190 mm, root z 195 mm), lip inside the walls (6-7 mm
    xy clearance per side). Seated, the plate covers both slot mouths and the whole
    interior; the lip bottom (178 mm) clears the standing bottle top (160 mm).
  - dressing / decoy: plain cylinders r 26 mm, h 140 mm, 0.15 kg — amber target,
    red decoy — standing upright on the open ground.

Per-episode randomization (readback-verifiable): base yaw +/-12 deg + xy jitter,
scatter-station jitter + free yaw for panels and lid, dressing/decoy station swap.

Rubric (0..1; latched partial credit, anchored in the demonstrated solve):
  0.12 * filled_s + 0.12 * filled_w — a panel seated in the south / west channel
  0.16 * placed — the dressing bottle upright on the pad interior
  0.20 * lidded — the lid seated (z band, xy registration, level)
  fail latch (permanent, freezes credit and blocks success): breach — the bottle
  enters the interior, or a slot becomes filled, WHILE the lid is seated (the lid
  blocks those paths physically; only a teleport can do it).
  1.0 iff success() — both slots filled, dressing upright inside, decoy NOT inside,
  lid seated, no breach, everything settled (consecutive-still counter) and finite.
  Non-success is capped at 0.60.

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


def _qinv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] = -out[:, 1:]
    return out


def _qapply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat([torch.zeros_like(q[:, :1]), v], dim=-1)
    return _qmul(_qmul(q, qv), _qinv(q))[:, 1:]


def _qz(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 3] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


def _qy(ang: torch.Tensor) -> torch.Tensor:
    q = torch.zeros(ang.shape[0], 4, device=ang.device)
    q[:, 0], q[:, 2] = torch.cos(ang / 2), torch.sin(ang / 2)
    return q


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


def _add_box(stage, path: str, *, center, size, color, collide: Callable):
    """One box child: translate + scale, displayColor, collider."""
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())
    return box.GetPrim()


def _spawn_base(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the base fixture: heavy DYNAMIC compound. Local frame: origin at the
    floor-pad centre on the ground. Children: floor pad, two fixed walls (+x, +y),
    and two slot channels (-x "south", -y "west"): inner/outer rails 50 mm tall
    (gap 22 mm centred at -71 mm), funnel flares on top (gap 34 mm, 15 mm tall),
    and end posts (channel span +/-54 mm). All channel hardware is kept >= 2 mm
    clear of the OTHER channel's seated-panel volume (x or y in [-79, -63] mm)."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.base_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.5)
    pxrb.CreateAngularDampingAttr(0.5)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)
    c = cfg

    # floor pad (top at z = 0.020)
    _add_box(stage, f"{prim_path}/pad", center=(0.0, 0.0, 0.010),
             size=(0.300, 0.300, 0.020), color=c.pad_color, collide=collide)
    # fixed walls: inner faces at +0.065; tops at z = 0.190
    _add_box(stage, f"{prim_path}/wall_xp", center=(0.071, 0.0, 0.105),
             size=(0.012, 0.154, 0.170), color=c.wall_color, collide=collide)
    _add_box(stage, f"{prim_path}/wall_yp", center=(0.0, 0.071, 0.105),
             size=(0.154, 0.012, 0.170), color=c.wall_color, collide=collide)
    # south channel (-x): panel plane perpendicular to x, centred at x = -0.071
    _add_box(stage, f"{prim_path}/rail_s_in", center=(-0.056, 0.0, 0.045),
             size=(0.008, 0.108, 0.050), color=c.rail_color, collide=collide)
    _add_box(stage, f"{prim_path}/rail_s_out", center=(-0.086, 0.0, 0.045),
             size=(0.008, 0.108, 0.050), color=c.rail_color, collide=collide)
    _add_box(stage, f"{prim_path}/flare_s_in", center=(-0.050, 0.0, 0.0775),
             size=(0.008, 0.108, 0.015), color=c.rail_color, collide=collide)
    _add_box(stage, f"{prim_path}/flare_s_out", center=(-0.092, 0.0, 0.0775),
             size=(0.008, 0.108, 0.015), color=c.rail_color, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/post_s_{tag}", center=(-0.071, sgn * 0.0575, 0.045),
                 size=(0.038, 0.007, 0.050), color=c.rail_color, collide=collide)
    # west channel (-y): mirror of the south channel with x <-> y
    _add_box(stage, f"{prim_path}/rail_w_in", center=(0.0, -0.056, 0.045),
             size=(0.108, 0.008, 0.050), color=c.rail_color, collide=collide)
    _add_box(stage, f"{prim_path}/rail_w_out", center=(0.0, -0.086, 0.045),
             size=(0.108, 0.008, 0.050), color=c.rail_color, collide=collide)
    _add_box(stage, f"{prim_path}/flare_w_in", center=(0.0, -0.050, 0.0775),
             size=(0.108, 0.008, 0.015), color=c.rail_color, collide=collide)
    _add_box(stage, f"{prim_path}/flare_w_out", center=(0.0, -0.092, 0.0775),
             size=(0.108, 0.008, 0.015), color=c.rail_color, collide=collide)
    for sgn, tag in ((1.0, "p"), (-1.0, "n")):
        _add_box(stage, f"{prim_path}/post_w_{tag}", center=(sgn * 0.0575, -0.071, 0.045),
                 size=(0.007, 0.038, 0.050), color=c.rail_color, collide=collide)
    return root


def _spawn_lid(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the lid: DYNAMIC compound, origin at the plate centre. Children: the
    200x200x10 plate and the 112x112x12 registration lip underneath (lip bottom at
    local z = -0.017). Mass is authored explicitly on the root — custom spawn funcs
    ignore the standard mass_props/rigid_props cfg schemas."""
    from pxr import PhysxSchema, UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(cfg.lid_mass))
    pxrb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    pxrb.CreateMaxDepenetrationVelocityAttr(0.5)
    pxrb.CreateLinearDampingAttr(0.10)
    pxrb.CreateAngularDampingAttr(0.20)
    pxrb.CreateSolverPositionIterationCountAttr(32)
    pxrb.CreateSolverVelocityIterationCountAttr(4)
    pxrb.CreateSleepThresholdAttr(0.0)
    pxrb.CreateStabilizationThresholdAttr(0.0)
    collide = _make_collide(cfg.contact_offset)

    _add_box(stage, f"{prim_path}/plate", center=(0.0, 0.0, 0.0),
             size=(0.200, 0.200, 0.010), color=cfg.lid_color, collide=collide)
    _add_box(stage, f"{prim_path}/lip", center=(0.0, 0.0, -0.011),
             size=(0.112, 0.112, 0.012), color=cfg.lip_color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "base" not in _SPAWNER_CACHE:

        @configclass
        class BaseSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_base)
            base_mass: float = 30.0
            pad_color: tuple = (0.30, 0.32, 0.36)
            wall_color: tuple = (0.55, 0.44, 0.28)
            rail_color: tuple = (0.20, 0.22, 0.26)
            contact_offset: float = 0.002

        @configclass
        class LidSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_lid)
            lid_mass: float = 0.30
            lid_color: tuple = (0.62, 0.50, 0.30)
            lip_color: tuple = (0.45, 0.35, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["base"] = BaseSpawnerCfg
        _SPAWNER_CACHE["lid"] = LidSpawnerCfg
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class FlatPackCrateSceneCfg(BaseCfg):
    """Config for `FlatPackCrateScene`. The breach gate is honest by construction:
    with the lid seated, its plate covers the whole interior and both slot mouths,
    so no physical path lets the bottle enter the interior or a panel enter a slot;
    such a transition can only be a teleport."""

    # --- tunable: rubric thresholds --------------------------------------------------------------
    seat_tol_x: float = tunable(0.008)     # panel centre offset across the channel (m)
    seat_tol_y: float = tunable(0.030)     # panel centre offset along the channel (m)
    panel_tilt_deg: float = tunable(10.0)  # max panel tilt from vertical when seated
    bottle_tilt_deg: float = tunable(20.0)  # max dressing tilt from vertical
    lid_xy_tol: float = tunable(0.010)     # lid centre offset from the crate centre (m)
    lid_tilt_deg: float = tunable(8.0)     # max lid tilt from level
    settle_speed: float = tunable(0.05)    # max |lin vel| of loose pieces when judging (m/s)
    still_steps: int = tunable(60)         # consecutive still substeps required (0.5 s)

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    base_yaw_deg: float = tunable(12.0)    # base yaw about its nominal heading (+/- deg)
    base_jitter: float = tunable(0.03)     # base xy jitter (+/- m)
    scatter_jitter: float = tunable(0.03)  # per-piece xy jitter at its scatter station (+/- m)
    bottle_swap: bool = tunable(True)      # shuffle dressing/decoy start stations

    # --- info: layout (base-local scatter stations: (x, y) on the open ground) --------------------
    base_pos: tuple = info((0.42, 0.0))    # base origin on the ground (nominal)
    base_yaw_nom_deg: float = info(180.0)  # nominal heading (slots face world +x-ish)
    st_panel_a: tuple = info((-0.3545, -0.0625))   # 190 deg, r 0.36
    st_panel_b: tuple = info((-0.1231, -0.3383))   # 250 deg, r 0.36
    st_lid: tuple = info((0.2314, -0.2758))        # 310 deg, r 0.36
    st_bottle_1: tuple = info((0.2121, 0.2121))    # 45 deg, r 0.30
    st_bottle_2: tuple = info((-0.2121, 0.2121))   # 135 deg, r 0.30
    # --- info: base fixture geometry (base-local; pad top z = 0.020, wall top z = 0.190) ----------
    pad_top: float = info(0.020)
    interior_half: float = info(0.065)     # fixed-wall inner faces at +x/+y
    wall_h: float = info(0.170)
    wall_top: float = info(0.190)
    slot_center: float = info(0.071)       # channel centreline at -0.071 (x for S, y for W)
    channel_gap: float = info(0.022)       # rail gap across the panel thickness
    funnel_gap: float = info(0.034)        # flare-mouth gap (capture +/- 9 mm)
    rail_h: float = info(0.050)            # rails z 0.020..0.070; flares 0.070..0.085
    channel_span: float = info(0.054)      # end-post gap: +/-0.054 along the channel
    base_mass: float = info(30.0)
    # --- info: loose pieces ------------------------------------------------------------------------
    panel_size: tuple = info((0.016, 0.100, 0.170))
    panel_mass: float = info(0.12)
    lid_plate: tuple = info((0.200, 0.200, 0.010))
    lid_lip: tuple = info((0.112, 0.112, 0.012))
    lid_mass: float = info(0.30)
    lid_seat_z: float = info(0.195)        # lid root height when seated (base-local)
    bottle_r: float = info(0.026)
    bottle_h: float = info(0.140)
    bottle_mass: float = info(0.15)
    dressing_color: tuple = info((0.90, 0.62, 0.12))
    decoy_color: tuple = info((0.75, 0.10, 0.10))
    panel_color: tuple = info((0.25, 0.40, 0.70))
    contact_offset: float = info(0.002)
    # --- info: rubric geometry bands (base-local) --------------------------------------------------
    panel_seat_z: tuple = info((0.095, 0.116))   # seated panel centre height band
    bottle_xy: float = info(0.045)               # dressing centre inside the interior
    bottle_z: tuple = info((0.075, 0.115))       # dressing centre height band (on the pad)
    lid_z: tuple = info((0.188, 0.208))          # seated lid root height band
    decoy_xy: float = info(0.075)                # decoy-exclusion half-extent
    decoy_z: tuple = info((0.030, 0.200))        # decoy-exclusion height band
    # rubric weights (0.12 + 0.12 + 0.16 + 0.20 = 0.60 = the non-success cap)
    w_slot: float = info(0.12)
    w_bottle: float = info(0.16)
    w_lid: float = info(0.20)


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("flat_pack_crate")
class FlatPackCrateScene(BaseScene):
    cfg: FlatPackCrateSceneCfg

    def __init__(self, cfg: FlatPackCrateSceneCfg | None = None) -> None:
        super().__init__(cfg or FlatPackCrateSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        base_spawn = cls["base"](base_mass=c.base_mass, contact_offset=c.contact_offset)
        lid_spawn = cls["lid"](lid_mass=c.lid_mass, contact_offset=c.contact_offset)

        piece_props = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.5,
                linear_damping=0.05, angular_damping=0.10,
                sleep_threshold=0.0, stabilization_threshold=0.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.40, dynamic_friction=0.32, restitution=0.0),
        )

        px, py = c.base_pos
        yaw0 = math.radians(c.base_yaw_nom_deg)
        qz0 = (math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2))

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
            "base": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Base",
                spawn=base_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px, py, 0.0), rot=qz0),
            ),
            "lid": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Lid",
                spawn=lid_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(px + 0.31, py - 0.28, 0.017)),
            ),
        }
        for name, x0 in (("panel_a", 0.9), ("panel_b", 1.2)):
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Panel_" + name[-1].upper(),
                spawn=sim_utils.CuboidCfg(
                    size=c.panel_size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.panel_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.panel_color),
                    **piece_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(x0, 1.0, 0.10)),
            )
        for name in ("dressing", "decoy"):
            color = c.dressing_color if name == "dressing" else c.decoy_color
            out[name] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle_" + name,
                spawn=sim_utils.CylinderCfg(
                    radius=c.bottle_r,
                    height=c.bottle_h,
                    axis="Z",
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.bottle_mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    **piece_props,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.9 if name == "dressing" else 1.2, 1.4, 0.07)),
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
        self.base: RigidObject = env.iscene["base"]
        self.panel_a: RigidObject = env.iscene["panel_a"]
        self.panel_b: RigidObject = env.iscene["panel_b"]
        self.lid: RigidObject = env.iscene["lid"]
        self.dressing: RigidObject = env.iscene["dressing"]
        self.decoy: RigidObject = env.iscene["decoy"]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # dressing_station[e] = 1 / 2: which scatter station holds the DRESSING bottle
        self.dressing_station = torch.ones(n, dtype=torch.long, device=dev)
        # credit latches (survive transients; success is judged live)
        self._filled_s = torch.zeros(n, dtype=torch.bool, device=dev)
        self._filled_w = torch.zeros(n, dtype=torch.bool, device=dev)
        self._placed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lidded = torch.zeros(n, dtype=torch.bool, device=dev)
        # fail latch (permanent; freezes credit, blocks success)
        self._breach = torch.zeros(n, dtype=torch.bool, device=dev)
        # transition tracking + settle bookkeeping
        self._prev_fs = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_fw = torch.zeros(n, dtype=torch.bool, device=dev)
        self._prev_in = torch.zeros(n, dtype=torch.bool, device=dev)
        self._still = torch.zeros(n, dtype=torch.long, device=dev)
        self._steps = torch.zeros(n, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the base (yaw + xy jitter), lay both panels FLAT at
        their scatter stations (jitter + free yaw), rest the lid lip-down at its
        station, stand the two bottles at their (possibly swapped) stations, clear
        all latches. Stations are fixed base-local anchors spaced so pieces cannot
        overlap by construction (no rejection sampling needed)."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- base: nominal heading + yaw + xy jitter ---
        yaw = math.radians(c.base_yaw_nom_deg) \
            + (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.base_yaw_deg)
        q_b = _qz(yaw)
        pp = torch.zeros(m, 3, device=dev)
        pp[:, 0] = c.base_pos[0] + (torch.rand(m, device=dev) * 2 - 1) * c.base_jitter
        pp[:, 1] = c.base_pos[1] + (torch.rand(m, device=dev) * 2 - 1) * c.base_jitter
        st = torch.zeros(m, 13, device=dev)
        st[:, 0:3] = pp + origin
        st[:, 3:7] = q_b
        self.base.write_root_state_to_sim(st, env_ids)

        def scatter(station: tuple, z: float, q_extra: torch.Tensor | None) -> torch.Tensor:
            """(m,13) state at a base-local station with xy jitter + free yaw."""
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0] = station[0] + (torch.rand(m, device=dev) * 2 - 1) * c.scatter_jitter
            loc[:, 1] = station[1] + (torch.rand(m, device=dev) * 2 - 1) * c.scatter_jitter
            qy = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            q = _qmul(qy, q_extra) if q_extra is not None else qy
            out = torch.zeros(m, 13, device=dev)
            out[:, 0:3] = pp + _qapply(q_b, loc) + origin
            out[:, 2] = z + origin[:, 2]
            out[:, 3:7] = q
            return out

        # --- panels: lying FLAT (rotated 90 deg about body y: thickness axis vertical) ---
        q_flat = _qy(torch.full((m,), math.pi / 2, device=dev))
        self.panel_a.write_root_state_to_sim(
            scatter(c.st_panel_a, c.panel_size[0] / 2 + 0.002, q_flat), env_ids)
        self.panel_b.write_root_state_to_sim(
            scatter(c.st_panel_b, c.panel_size[0] / 2 + 0.002, q_flat), env_ids)

        # --- lid: lip-down on the ground (root z = plate half + lip height) ---
        self.lid.write_root_state_to_sim(
            scatter(c.st_lid, c.lid_plate[2] / 2 + c.lid_lip[2] + 0.002, None), env_ids)

        # --- bottles: upright at stations 1/2, swapped per episode ---
        if c.bottle_swap:
            stn = torch.where(torch.rand(m, device=dev) < 0.5,
                              torch.ones(m, dtype=torch.long, device=dev),
                              torch.full((m,), 2, dtype=torch.long, device=dev))
        else:
            stn = torch.ones(m, dtype=torch.long, device=dev)
        self.dressing_station[env_ids] = stn
        s1 = torch.tensor(c.st_bottle_1, device=dev)
        s2 = torch.tensor(c.st_bottle_2, device=dev)
        for body, own in ((self.dressing, stn == 1), (self.decoy, stn == 2)):
            loc = torch.zeros(m, 3, device=dev)
            loc[:, 0:2] = torch.where(own.unsqueeze(-1), s1, s2)
            loc[:, 0] += (torch.rand(m, device=dev) * 2 - 1) * c.scatter_jitter
            loc[:, 1] += (torch.rand(m, device=dev) * 2 - 1) * c.scatter_jitter
            stt = torch.zeros(m, 13, device=dev)
            stt[:, 0:3] = pp + _qapply(q_b, loc) + origin
            stt[:, 2] = c.bottle_h / 2 + 0.002 + origin[:, 2]
            stt[:, 3:7] = _qz((torch.rand(m, device=dev) * 2 - 1) * math.pi)
            body.write_root_state_to_sim(stt, env_ids)

        # --- clear latches / bookkeeping ---
        self._filled_s[env_ids] = False
        self._filled_w[env_ids] = False
        self._placed[env_ids] = False
        self._lidded[env_ids] = False
        self._breach[env_ids] = False
        self._prev_fs[env_ids] = False
        self._prev_fw[env_ids] = False
        self._prev_in[env_ids] = False
        self._still[env_ids] = 0
        self._steps[env_ids] = 0

    # ----- state (full, restorable) --------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "base": self.base.data.root_state_w[env_ids].clone(),
            "panel_a": self.panel_a.data.root_state_w[env_ids].clone(),
            "panel_b": self.panel_b.data.root_state_w[env_ids].clone(),
            "lid": self.lid.data.root_state_w[env_ids].clone(),
            "dressing": self.dressing.data.root_state_w[env_ids].clone(),
            "decoy": self.decoy.data.root_state_w[env_ids].clone(),
            "dressing_station": self.dressing_station[env_ids].clone(),
            "filled_s": self._filled_s[env_ids].clone(),
            "filled_w": self._filled_w[env_ids].clone(),
            "placed": self._placed[env_ids].clone(),
            "lidded": self._lidded[env_ids].clone(),
            "breach": self._breach[env_ids].clone(),
            "prev_fs": self._prev_fs[env_ids].clone(),
            "prev_fw": self._prev_fw[env_ids].clone(),
            "prev_in": self._prev_in[env_ids].clone(),
            "still": self._still[env_ids].clone(),
            "steps": self._steps[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.base.write_root_state_to_sim(state["base"], env_ids)
        self.panel_a.write_root_state_to_sim(state["panel_a"], env_ids)
        self.panel_b.write_root_state_to_sim(state["panel_b"], env_ids)
        self.lid.write_root_state_to_sim(state["lid"], env_ids)
        self.dressing.write_root_state_to_sim(state["dressing"], env_ids)
        self.decoy.write_root_state_to_sim(state["decoy"], env_ids)
        self.dressing_station[env_ids] = state["dressing_station"]
        self._filled_s[env_ids] = state["filled_s"]
        self._filled_w[env_ids] = state["filled_w"]
        self._placed[env_ids] = state["placed"]
        self._lidded[env_ids] = state["lidded"]
        self._breach[env_ids] = state["breach"]
        self._prev_fs[env_ids] = state["prev_fs"]
        self._prev_fw[env_ids] = state["prev_fw"]
        self._prev_in[env_ids] = state["prev_in"]
        self._still[env_ids] = state["still"]
        self._steps[env_ids] = state["steps"]

    # ----- description ---------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A heavy BASE FIXTURE sits on the ground: a "
            f"{300:.0f} mm square floor pad with TWO fixed wooden walls "
            f"({c.wall_h * 1000:.0f} mm tall) on two adjacent sides, and on the other "
            f"two sides only low SLOT CHANNELS — vertical grooves "
            f"{c.channel_gap * 1000:.0f} mm wide with flared funnel mouths, "
            f"{c.rail_h * 1000:.0f} mm tall. The crate is FLAT-PACKED: two loose BLUE "
            f"WALL PANELS ({c.panel_size[0] * 1000:.0f} x {c.panel_size[1] * 1000:.0f} x "
            f"{c.panel_size[2] * 1000:.0f} mm) lie flat on the ground nearby, and a "
            f"square LID ({c.lid_plate[0] * 1000:.0f} mm, with a raised registration "
            f"lip on its underside) rests off to the side. An AMBER salad-dressing "
            f"bottle and a RED decoy bottle (each r {c.bottle_r * 1000:.0f} mm, "
            f"h {c.bottle_h * 1000:.0f} mm) stand upright on the open ground; which "
            f"bottle stands where is shuffled per episode — the color is the only cue.\n"
            f"Goal: BUILD the crate around the amber bottle and seal it. Stand each "
            f"blue panel up and slide it DOWN into a slot channel until it seats on "
            f"the floor pad (its top edge flush with the fixed walls); stand the "
            f"AMBER bottle upright on the pad, inside the walls; then seat the lid "
            f"on the wall tops so its lip registers just inside the walls, sealing "
            f"the bottle in. The RED decoy must stay OUTSIDE. The lid must go on "
            f"LAST: once seated it covers the open top and both slot mouths, so "
            f"neither a panel nor a bottle can enter afterwards — anything that "
            f"appears inside a sealed crate is a permanent fail. Finish with both "
            f"panels seated, the amber bottle upright inside, the decoy outside, "
            f"the lid seated level, and everything at rest."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Assemble the flat-packed crate: slide each blue wall panel down into a "
            "slot channel of the base until it seats on the floor pad, stand the "
            "amber salad-dressing bottle upright on the pad between the walls, then "
            "seat the lid on the wall tops so its lip registers inside the walls, "
            "sealing the bottle in. Leave the red decoy bottle outside."
        )

    # ----- frames / live predicates --------------------------------------------------------------
    def _base_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World points -> the base-fixture frame, (N,3) -> (N,3)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.base.data.root_quat_w,
                                  pos_w - self.base.data.root_pos_w)

    def _axis_w(self, body: RigidObject, axis: tuple) -> torch.Tensor:
        """(N,3) a body-frame unit axis expressed in world coordinates."""
        n = body.data.root_quat_w.shape[0]
        v = torch.tensor([list(axis)], device=self.env.device).expand(n, 3)
        return _qapply(body.data.root_quat_w, v)

    def _axis_base(self, body: RigidObject, axis: tuple) -> torch.Tensor:
        """(N,3) a body-frame unit axis expressed in the base-fixture frame."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.base.data.root_quat_w, self._axis_w(body, axis))

    def _panel_seated(self, panel: RigidObject, slot: str) -> torch.Tensor:
        """(N,) bool: `panel` seated in slot 's' (channel at base-local x=-0.071,
        thickness axis along base x) or 'w' (y=-0.071, thickness axis along y):
        centre in the channel xy window, centre height in the seat band, panel
        near-vertical, thickness axis aligned with the channel normal."""
        c = self.cfg
        loc = self._base_local(panel.data.root_pos_w)
        ax = self._axis_base(panel, (1.0, 0.0, 0.0))    # thickness axis
        up = self._axis_w(panel, (0.0, 0.0, 1.0))       # height axis
        if slot == "s":
            across, along = loc[:, 0] + c.slot_center, loc[:, 1]
            norm = ax[:, 0].abs()
        else:
            across, along = loc[:, 1] + c.slot_center, loc[:, 0]
            norm = ax[:, 1].abs()
        z = loc[:, 2]
        return (across.abs() < c.seat_tol_x) & (along.abs() < c.seat_tol_y) \
            & (z > c.panel_seat_z[0]) & (z < c.panel_seat_z[1]) \
            & (up[:, 2] > math.cos(math.radians(c.panel_tilt_deg))) \
            & (norm > 0.9)

    def slot_filled(self, slot: str) -> torch.Tensor:
        """(N,) bool (live): SOME panel is seated in the given slot ('s'/'w').
        Panels are interchangeable; one panel cannot occupy both slots."""
        return self._panel_seated(self.panel_a, slot) | self._panel_seated(self.panel_b, slot)

    def bottle_inside(self, body: RigidObject) -> torch.Tensor:
        """(N,) bool (live): bottle upright on the pad interior (base frame)."""
        c = self.cfg
        loc = self._base_local(body.data.root_pos_w)
        up = self._axis_w(body, (0.0, 0.0, 1.0))
        return (loc[:, 0].abs() < c.bottle_xy) & (loc[:, 1].abs() < c.bottle_xy) \
            & (loc[:, 2] > c.bottle_z[0]) & (loc[:, 2] < c.bottle_z[1]) \
            & (up[:, 2] > math.cos(math.radians(c.bottle_tilt_deg)))

    def decoy_excluded(self) -> torch.Tensor:
        """(N,) bool (live): decoy centre NOT within the (loose) interior volume."""
        c = self.cfg
        loc = self._base_local(self.decoy.data.root_pos_w)
        inside = (loc[:, 0].abs() < c.decoy_xy) & (loc[:, 1].abs() < c.decoy_xy) \
            & (loc[:, 2] > c.decoy_z[0]) & (loc[:, 2] < c.decoy_z[1])
        return ~inside

    def lid_seated(self) -> torch.Tensor:
        """(N,) bool (live): lid root at the seat height, centred over the crate,
        level, and yaw-registered (square lip: any 90-deg multiple registers)."""
        c = self.cfg
        loc = self._base_local(self.lid.data.root_pos_w)
        up = self._axis_w(self.lid, (0.0, 0.0, 1.0))
        ax = self._axis_base(self.lid, (1.0, 0.0, 0.0))
        yaw_ok = torch.maximum(ax[:, 0].abs(), ax[:, 1].abs()) \
            > math.cos(math.radians(15.0))
        return (loc[:, 0].abs() < c.lid_xy_tol) & (loc[:, 1].abs() < c.lid_xy_tol) \
            & (loc[:, 2] > c.lid_z[0]) & (loc[:, 2] < c.lid_z[1]) \
            & (up[:, 2].abs() > math.cos(math.radians(c.lid_tilt_deg))) \
            & (up[:, 2] > 0) & yaw_ok

    def settled(self) -> torch.Tensor:
        """(N,) bool: consecutive-still counter satisfied (not an instantaneous
        velocity gate — teleports zero velocities; the counter runs in post_step)."""
        return (self._still >= self.cfg.still_steps) & (self._steps >= self.cfg.still_steps)

    def _finite(self) -> torch.Tensor:
        p = torch.stack([b.data.root_pos_w for b in
                         (self.base, self.panel_a, self.panel_b, self.lid,
                          self.dressing, self.decoy)], dim=1)
        return torch.isfinite(p).all(dim=-1).all(dim=-1)

    def _update_latches(self) -> None:
        c = self.cfg
        fin = self._finite()
        fs = self.slot_filled("s")
        fw = self.slot_filled("w")
        din = self.bottle_inside(self.dressing)
        lid = self.lid_seated()
        # --- breach: interior/slot entry while the lid is seated = teleport ---
        entered = (fs & ~self._prev_fs) | (fw & ~self._prev_fw) | (din & ~self._prev_in)
        self._breach |= entered & lid & fin
        ok = fin & ~self._breach
        # --- credit latches (frozen once breached), gated on the piece being slow ---
        slow_a = self.panel_a.data.root_lin_vel_w.norm(dim=-1) < 0.08
        slow_b = self.panel_b.data.root_lin_vel_w.norm(dim=-1) < 0.08
        slow_p = slow_a & slow_b
        self._filled_s |= fs & slow_p & ok
        self._filled_w |= fw & slow_p & ok
        self._placed |= din & (self.dressing.data.root_lin_vel_w.norm(dim=-1) < 0.08) & ok
        self._lidded |= lid & (self.lid.data.root_lin_vel_w.norm(dim=-1) < 0.08) & ok
        self._prev_fs, self._prev_fw, self._prev_in = fs, fw, din
        # --- consecutive-still counter ---
        still = (self.dressing.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.decoy.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.panel_a.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.panel_b.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.lid.data.root_lin_vel_w.norm(dim=-1) < c.settle_speed) \
            & (self.base.data.root_ang_vel_w.norm(dim=-1) < 0.20)
        self._still = torch.where(still, self._still + 1, torch.zeros_like(self._still))
        self._steps = self._steps + 1

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    # ----- rubric --------------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: both slots filled by seated panels, the amber bottle upright
        on the pad interior, the decoy NOT inside, the lid seated level, no breach
        latch, everything settled and finite. Containment/seating clauses are live
        physical outcomes; the breach latch enforces that nothing entered a sealed
        crate by teleport."""
        return self.slot_filled("s") & self.slot_filled("w") \
            & self.bottle_inside(self.dressing) & self.decoy_excluded() \
            & self.lid_seated() & ~self._breach \
            & self.settled() & self._finite()

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.12*filled_s + 0.12*filled_w + 0.16*placed +
        0.20*lidded (latched; ~0 for doing nothing — every term needs a piece
        moved from its scatter station into its assembled pose), capped at 0.60 —
        and exactly 1.0 iff success() holds live."""
        c = self.cfg
        base = (c.w_slot * self._filled_s.float() + c.w_slot * self._filled_w.float()
                + c.w_bottle * self._placed.float()
                + c.w_lid * self._lidded.float()).clamp(max=0.60)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="flat_pack_crate", robot="null"))
