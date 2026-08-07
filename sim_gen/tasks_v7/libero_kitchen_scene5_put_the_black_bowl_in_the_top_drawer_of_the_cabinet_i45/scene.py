"""DominoRelayScene — build a standing domino relay from the red trigger tile to the
gallery window, then topple ONLY the trigger: the cascade's last tile falls through the
letterbox window and sweeps the untouchable ball off its shelf into the chamber pit
(sim_gen task `libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i45`).

Derived from libero_90 kitchen_scene5 "put the black bowl in the top drawer of the
cabinet", but STRATEGICALLY different: the seed's plan is articulated pick-and-place —
grasp the bowl, pull the drawer open, carry the bowl across free space, set it down
inside a bbox; the hand holds the payload through every judged interaction. Here the
payload (an orange ball) still must END UP inside a receptacle (the gallery's sunken
pit chamber), but it is physically UNTOUCHABLE: it sits on a shelf inside a roofed,
walled gallery whose only opening is a letterbox window shadowed by an eave, and the
judged delivery must be performed by a KINETIC TRANSMISSION LINE the agent constructs
out of free bodies — eight loose tiles stood upright in a spaced relay running from a
red TRIGGER tile to the window, then fired with a single nudge on the trigger. The
cascade does the transport: each falling tile knocks the next, the last tile pitches
through the window and bats the ball off its shelf. A solver therefore needs a
different PLAN (survey the corridor -> erect a spaced standing relay -> verify spacing
-> one trigger nudge -> hands-off cascade) and a different code STRUCTURE (chain
layout geometry + a causality/timing rubric), not a grasp-carry-place waypoint
follower. It is also different from every constructed task seen during design: the
corpus builds STATIC structures (bridges, cribs) or launches single projectiles;
nothing constructs a multi-body dynamic cascade whose propagation IS the manipulation.

CAUSALITY RUBRIC (the anti-manual teeth, latched every physics substep):
  - t_trig  = first substep the trigger tile tips past `trig_deg`;
  - t_ball  = first substep the ball leaves its shelf seat by > `dislodge_dist`;
  - causal  = latched TRUE iff the ball dislodged AFTER the trigger fell and within
    `tau_s` seconds of it — a genuine cascade covers the relay in well under that; an
    agent toppling tiles one by one (or ferrying anything by hand between the two
    ends) cannot beat the window;
  - built   = latched max count of relay tiles SIMULTANEOUSLY standing inside the
    trigger->window corridor (partial credit for construction);
  - pit     = latched ball-on-chamber-floor, gated on `causal` (teleporting or poking
    the ball in with no cascade earns nothing).
success() iff: causal AND >= `k_relay` relay tiles lie FALLEN inside the corridor AND
the ball rests on the chamber floor (inside the gallery, below the shelf) AND
everything is settled. score() = 0.30*built/K + 0.30*causal + 0.15*pit (cap 0.75),
exactly 1.0 iff success(). Doing nothing scores ~0; all credit is latched so it never
evaporates while the cascade runs or bodies settle.

Assets are fully procedural (no external files):
  - gallery: ONE kinematic fixture — a 260 mm-wide, 190 mm-tall roofed box; the front
    wall carries a 55 mm-wide letterbox WINDOW from z 45..155 mm (sill at 45 mm), an
    EAVE plate shadowing the window mouth, an interior SHELF at sill height right
    behind the window with a low front/side lip around the ball seat (the pit-side
    edge is OPEN), and a sunken PIT (the chamber floor) behind the shelf, sealed by
    roof, side and back walls;
  - tiles: eight slate-blue 14 x 36 x 105 mm dominoes (50 g) scattered lying flat,
    plus ONE crimson TRIGGER tile standing on a crimson floor disc (visual marker;
    its collider is buried below the floor so the disc adds no step);
  - ball: orange 32 mm sphere (10 g) seated on the shelf, visible through the window.
Contact offsets are explicit and small (2 mm): relay gaps are ~30-55 mm.

Per-episode randomization (verified by readback in smoke): gallery xy + yaw, trigger
pad distance (300..400 mm) AND bearing (+/-20 deg off the window normal), trigger yaw
jitter, tile scatter with keep-out resampling (pad, other tiles, AND the pad->window
corridor, so no scatter tile ever pre-loads the fallen count). Heavy imports (isaaclab, pxr) are
deferred so importing this module — and registering the scene — stays app-free.
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


# ----- custom compound spawners ----------------------------------------------------------------
_SPAWNER_CACHE: dict[str, Any] = {}


def _apply_xform(xform, translation, orientation) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(xform)
    if translation is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translation]))
    if orientation is not None:
        w, x, y, z = (float(v) for v in orientation)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _collide(prim, contact_offset: float) -> None:
    from pxr import PhysxSchema, UsdPhysics

    UsdPhysics.CollisionAPI.Apply(prim)
    px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    px.CreateContactOffsetAttr(float(contact_offset))
    px.CreateRestOffsetAttr(0.0)


def _box(stage, path: str, size, center, color, contact_offset: float | None) -> None:
    """Author one axis-aligned box. `contact_offset=None` -> VISUAL ONLY (no collider)."""
    from pxr import Gf, UsdGeom

    seg = UsdGeom.Cube.Define(stage, path)
    seg.CreateSizeAttr(1.0)
    sxf = UsdGeom.Xformable(seg.GetPrim())
    sxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    sxf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    seg.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if contact_offset is not None:
        _collide(seg.GetPrim(), contact_offset)


def _spawn_gallery(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC gallery at `prim_path`. Local origin = centre of the OUTER
    FRONT FACE at ground level; +y runs INTO the chamber. Front wall spans y in
    [0, wall_t]; the letterbox window pierces it from `sill_z` to `win_z1` over a width
    `win_w`; the shelf sits right behind the window at sill height with a low retaining
    ring around the ball seat; the rest of the interior floor is the sunken pit."""
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(20.0)

    co = cfg.contact_offset
    col, dark, shelf_col = cfg.color, cfg.dark_color, cfg.shelf_color
    hw, wt, wh = cfg.half_w, cfg.wall_t, cfg.wall_h
    whw, sz, wz1 = cfg.win_w / 2, cfg.sill_z, cfg.win_z1
    yb = wt / 2  # front-wall y centre

    # --- front wall: sill band, two side bands, top band (the window is the hole) ---
    _box(stage, f"{prim_path}/fw_sill", (2 * hw, wt, sz), (0.0, yb, sz / 2), col, co)
    _box(stage, f"{prim_path}/fw_top", (2 * hw, wt, wh - wz1),
         (0.0, yb, (wz1 + wh) / 2), col, co)
    for tag, x0, x1 in (("l", -hw, -whw), ("r", whw, hw)):
        _box(stage, f"{prim_path}/fw_{tag}", (x1 - x0, wt, wz1 - sz),
             ((x0 + x1) / 2, yb, (sz + wz1) / 2), col, co)

    # --- eave shadowing the window mouth ---
    _box(stage, f"{prim_path}/eave", (2 * hw, cfg.eave_d, cfg.eave_t),
         (0.0, -cfg.eave_d / 2, wz1 + cfg.eave_t / 2), dark, co)

    # --- roof, side walls, back wall (a sealed chamber) ---
    depth = cfg.chamber_y1 + cfg.back_t
    _box(stage, f"{prim_path}/roof", (2 * hw, depth, cfg.roof_t),
         (0.0, depth / 2, wh + cfg.roof_t / 2), dark, co)
    for tag, s in (("side_l", -1.0), ("side_r", 1.0)):
        _box(stage, f"{prim_path}/{tag}", (cfg.side_t, depth, wh),
             (s * (hw - cfg.side_t / 2), depth / 2, wh / 2), col, co)
    _box(stage, f"{prim_path}/back", (2 * hw, cfg.back_t, wh),
         (0.0, cfg.chamber_y1 + cfg.back_t / 2, wh / 2), col, co)

    # --- shelf at sill height right behind the window + the ball's retaining lip.
    # Front + side lip segments only: the pit-side edge is OPEN — even a 1.5 mm lip
    # stops a slowly-swept ball, and the exit toward the pit must stay unobstructed.
    _box(stage, f"{prim_path}/shelf", (2 * cfg.shelf_hw, cfg.shelf_d, sz),
         (0.0, wt + cfg.shelf_d / 2, sz / 2), shelf_col, co)
    rr, rt, rw = cfg.ring_r, cfg.ring_t, cfg.ring_w
    sy = cfg.seat_y
    for tag, cx, cy_, sx_, sy_ in (
            ("f", 0.0, sy - rr, 2 * rr + rw, rw),
            ("l", -rr, sy, rw, 2 * rr - rw), ("r", rr, sy, rw, 2 * rr - rw)):
        _box(stage, f"{prim_path}/ring_{tag}", (sx_, sy_, rt),
             (cx, cy_, sz + rt / 2), shelf_col, co)
    return root


def _gallery_spawner_cfg(c) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "gallery" not in _SPAWNER_CACHE:

        @configclass
        class RelayGallerySpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_gallery)
            half_w: float = 0.13
            wall_t: float = 0.010
            wall_h: float = 0.19
            roof_t: float = 0.008
            win_w: float = 0.055
            sill_z: float = 0.045
            win_z1: float = 0.155
            eave_d: float = 0.045
            eave_t: float = 0.008
            side_t: float = 0.012
            back_t: float = 0.010
            chamber_y1: float = 0.20
            shelf_hw: float = 0.048
            shelf_d: float = 0.040
            seat_y: float = 0.030
            ring_r: float = 0.019
            ring_t: float = 0.0015
            ring_w: float = 0.004
            color: tuple = (0.52, 0.54, 0.58)
            dark_color: tuple = (0.34, 0.36, 0.40)
            shelf_color: tuple = (0.72, 0.70, 0.62)
            contact_offset: float = 0.002

        _SPAWNER_CACHE["gallery"] = RelayGallerySpawnerCfg

    return _SPAWNER_CACHE["gallery"](
        mass_props=None, rigid_props=None,
        half_w=c.half_w, wall_t=c.wall_t, wall_h=c.wall_h, roof_t=c.roof_t,
        win_w=c.win_w, sill_z=c.sill_z, win_z1=c.win_z1, eave_d=c.eave_d,
        eave_t=c.eave_t, side_t=c.side_t, back_t=c.back_t, chamber_y1=c.chamber_y1,
        shelf_hw=c.shelf_hw, shelf_d=c.shelf_d, seat_y=c.seat_y, ring_r=c.ring_r,
        ring_t=c.ring_t, ring_w=c.ring_w, color=c.gallery_color,
        dark_color=c.gallery_dark_color, shelf_color=c.shelf_color,
        contact_offset=c.contact_offset,
    )


def _spawn_pad(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the KINEMATIC trigger-pad marker: a thin crimson disc (VISUAL ONLY — no
    collider, so it adds no step under a standing tile) plus a small collider box
    BURIED below the floor so the rigid body legitimately owns a collision shape."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, prim_path)
    root = xform.GetPrim()
    _apply_xform(xform, translation, orientation)
    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(1.0)

    disc = UsdGeom.Cylinder.Define(stage, f"{prim_path}/disc")
    disc.CreateRadiusAttr(float(cfg.pad_r))
    disc.CreateHeightAttr(0.0015)
    disc.CreateAxisAttr("Z")
    UsdGeom.Xformable(disc.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.00075))
    disc.CreateDisplayColorAttr([Gf.Vec3f(*cfg.pad_color)])
    _box(stage, f"{prim_path}/buried", (0.02, 0.02, 0.006), (0.0, 0.0, -0.006),
         cfg.pad_color, 0.001)
    return root


def _pad_spawner_cfg(c) -> Any:
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "pad" not in _SPAWNER_CACHE:

        @configclass
        class TriggerPadSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_pad)
            pad_r: float = 0.05
            pad_color: tuple = (0.75, 0.10, 0.10)

        _SPAWNER_CACHE["pad"] = TriggerPadSpawnerCfg

    return _SPAWNER_CACHE["pad"](mass_props=None, rigid_props=None,
                                 pad_r=c.pad_r, pad_color=c.pad_color)


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DominoRelayCfg(BaseCfg):
    """Config for `DominoRelayScene`. Honesty knobs asserted in `__post_init__`: a
    falling tile fits through the window and reaches the ball, the ball's only way off
    the shelf ends on the sunken chamber floor, the relay tiles can bridge the longest
    pad distance within reliable topple spacing, and the standing/fallen bands are
    mutually exclusive."""

    # --- tunable: rubric thresholds ----------------------------------------------------------
    trig_deg: float = tunable(50.0)  # trigger counts as FELLED once tipped past this
    # (50 deg: far beyond the ~7.6 deg balance point, yet reliably crossed even when
    # the trigger comes to rest LEANING on the fallen relay pile at tight gaps)
    fallen_deg: float = tunable(50.0)  # a relay tile counts fallen beyond this tilt ...
    # (50 deg: the LAST tile legitimately comes to rest leaning ~54-60 deg through the
    # window on the sill corner — that is a delivered tile, not a standing one)
    fallen_z: float = tunable(0.045)  # ... AND centre below this (m)
    standing_deg: float = tunable(15.0)  # a relay tile counts standing within this of vertical
    dislodge_dist: float = tunable(0.010)  # ball leaving its seat by more than this (m)
    tau_s: float = tunable(2.75)  # max trigger-fall -> ball-dislodge delay (cascade window)
    k_relay: int = tunable(4)  # min relay tiles fallen in the corridor at success
    corr_halfw: float = tunable(0.09)  # corridor half-width around the trigger->window line
    pit_z: float = tunable(0.035)  # ball centre below this = on the chamber floor (m)
    settle_lin: float = tunable(0.05)  # max |lin vel| (tiles AND ball) when judging (m/s)
    settle_ang: float = tunable(0.60)  # max tile |ang vel| when judging (rad/s)

    # --- tunable: randomization (the task-family knobs) --------------------------------------
    fix_jitter: float = tunable(0.05)  # uniform +/- xy jitter of the gallery (m)
    fix_yaw_deg: float = tunable(20.0)  # uniform +/- gallery yaw (read it from the scene)
    pad_dist: tuple = tunable((0.30, 0.40))  # trigger pad distance from the window (m)
    pad_bear_deg: float = tunable(20.0)  # +/- pad bearing off the window outward normal
    trig_yaw_jit_deg: float = tunable(8.0)  # trigger tile yaw jitter about the chain line
    scatter_x: tuple = tunable((-0.34, 0.34))  # tile scatter zone, gallery-local x (m)
    scatter_y: tuple = tunable((-0.60, -0.14))  # tile scatter zone, gallery-local y (m)
    keepout_pad: float = tunable(0.11)  # scatter keep-out radius around the trigger pad
    keepout_tile: float = tunable(0.12)  # scatter keep-out between lying tiles

    # --- tunable: placement ------------------------------------------------------------------
    fix_pos: tuple = tunable((0.0, 0.15))  # gallery front-face centre, WORLD xy nominal

    # --- info: gallery structure (gallery-local; +y into the chamber) ------------------------
    half_w: float = info(0.13)  # gallery half-width
    wall_t: float = info(0.010)  # front wall thickness
    wall_h: float = info(0.19)  # wall height (roofed: nothing goes over)
    roof_t: float = info(0.008)
    win_w: float = info(0.055)  # letterbox window width (a tile fits through)
    sill_z: float = info(0.045)  # window sill = shelf height
    win_z1: float = info(0.155)  # window top
    eave_d: float = info(0.045)  # eave plate shadowing the mouth
    eave_t: float = info(0.008)
    side_t: float = info(0.012)
    back_t: float = info(0.010)
    chamber_y1: float = info(0.20)  # inner face of the back wall
    shelf_hw: float = info(0.048)  # shelf half-width (wider than the window)
    shelf_d: float = info(0.040)  # shelf depth from the inner wall face (short: a
    #   swept ball needs only ~20 mm of travel to roll off the OPEN back edge)
    seat_y: float = info(0.030)  # ball seat centre, local y
    ring_r: float = info(0.019)  # retaining lip half-extent around the seat
    ring_t: float = info(0.0015)  # lip height (anti-jitter; front + sides ONLY —
    #   the pit-side edge is open so the swept ball exits unobstructed)
    ring_w: float = info(0.004)
    pad_r: float = info(0.05)  # crimson trigger-pad disc radius (visual marker)
    # --- info: bodies ------------------------------------------------------------------------
    n_tiles: int = info(8)  # loose relay tiles
    tile_t: float = info(0.014)  # tile thickness (local x)
    tile_w: float = info(0.036)  # tile width (local y)
    tile_h: float = info(0.105)  # tile height (local z)
    tile_mass: float = info(0.050)
    ball_r: float = info(0.016)
    ball_mass: float = info(0.010)
    gallery_color: tuple = info((0.52, 0.54, 0.58))
    gallery_dark_color: tuple = info((0.34, 0.36, 0.40))
    shelf_color: tuple = info((0.72, 0.70, 0.62))
    tile_color: tuple = info((0.22, 0.34, 0.72))
    trigger_color: tuple = info((0.85, 0.10, 0.10))
    ball_color: tuple = info((0.95, 0.55, 0.10))
    pad_color: tuple = info((0.75, 0.10, 0.10))
    contact_offset: float = info(0.002)
    max_gap: float = info(0.055)  # reliable topple spacing bound (~0.52 * tile_h)

    # Derived (filled in __post_init__).
    tile_z0: float = field(default=None, init=False)  # standing tile centre height
    seat_z: float = field(default=None, init=False)  # ball centre height on the shelf seat

    def __post_init__(self) -> None:
        self.tile_z0 = self.tile_h / 2
        self.seat_z = self.sill_z + self.ball_r

        # a falling tile fits through the window ...
        assert self.win_w - self.tile_w >= 0.015, "tile must pass the window width"
        # ... clears the sill even when placed as far back as 75 mm from the wall ...
        assert math.sqrt(self.tile_h**2 - 0.075**2) > self.sill_z + 0.005, (
            "tile tip must still clear the sill from a 75 mm stand-off")
        # ... and can reach past the ball seat from a 55 mm stand-off
        assert self.tile_h - 0.055 - self.wall_t > self.seat_y, (
            "tile tip must reach past the ball seat")
        # ball: seated behind the wall, visible in the window band, only exit = the pit
        assert self.seat_y - self.ball_r > self.wall_t / 2, "ball seat behind the wall plane"
        assert self.seat_z + self.ball_r < self.win_z1, "ball visible through the window"
        assert self.ball_r + 0.005 < self.pit_z < self.sill_z - 0.005, (
            "pit height band must separate floor-rest from shelf-rest")
        assert self.shelf_d + self.wall_t < self.chamber_y1 - 0.06, (
            "pit must be deeper than the shelf")
        assert 2 * self.shelf_hw > self.win_w + 0.02, "shelf must back the whole window"
        # relay feasibility: 8 tiles bridge the longest pad distance; k_relay is
        # achievable at the SHORTEST distance
        assert self.n_tiles * self.max_gap >= self.pad_dist[1], (
            "tiles must bridge the longest pad distance")
        assert (self.pad_dist[0] - 0.08) / self.max_gap >= self.k_relay - 1e-6, (
            "k_relay must be forced even at the shortest pad distance")
        assert self.corr_halfw > self.tile_w / 2 + 0.03, "corridor tolerates placement slack"
        # bands separate; eave clears a standing tile
        assert self.fallen_deg >= self.standing_deg + 30.0, "standing/fallen bands separate"
        assert self.trig_deg >= self.standing_deg + 30.0, (
            "trigger latch far beyond any standing wobble")
        assert self.fallen_z < self.tile_z0 - 0.005, "fallen height excludes standing"
        assert self.win_z1 - self.tile_h >= 0.04, "eave/window top clears a standing tile"
        assert self.tile_w / 2 < self.pad_r < self.keepout_pad, (
            "pad disc covers the trigger footprint but stays inside the scatter keep-out")


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("domino_relay")
class DominoRelayScene(BaseScene):
    cfg: DominoRelayCfg

    def __init__(self, cfg: DominoRelayCfg | None = None) -> None:
        super().__init__(cfg or DominoRelayCfg())

    # ----- assets -----------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg

        def tile_cfg(color: tuple) -> Any:
            return sim_utils.CuboidCfg(
                size=(c.tile_t, c.tile_w, c.tile_h),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.02, angular_damping=0.02,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    sleep_threshold=0.0, stabilization_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=c.tile_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.55, dynamic_friction=0.50, restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        fx, fy = c.fix_pos
        out: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.55, dynamic_friction=0.50, restitution=0.0)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "gallery": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Gallery",
                spawn=_gallery_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(fx, fy, 0.0)),
            ),
            "pad": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TriggerPad",
                spawn=_pad_spawner_cfg(c),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(fx, fy - 0.35, 0.0)),
            ),
            "trigger": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/TriggerTile",
                spawn=tile_cfg(c.trigger_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx, fy - 0.35, c.tile_z0 + 0.002)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Ball",
                spawn=sim_utils.SphereCfg(
                    radius=c.ball_r,
                    # damping = rolling-resistance proxy: PhysX spheres have zero
                    # rolling friction, and a ball that never stops rolling inside the
                    # sealed chamber could never satisfy "rests on the pit floor"
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        linear_damping=0.15, angular_damping=0.80,
                        max_depenetration_velocity=1.0,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        sleep_threshold=0.0, stabilization_threshold=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=c.ball_mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=c.contact_offset, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=0.40, dynamic_friction=0.35, restitution=0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c.ball_color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx, fy + c.seat_y, c.seat_z + 0.001)),
            ),
        }
        for i in range(c.n_tiles):
            out[f"tile_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tile_" + str(i),
                spawn=tile_cfg(c.tile_color),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(fx - 0.5 + 0.12 * i, fy - 0.55, c.tile_t / 2 + 0.002)),
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
                "gpu_max_rigid_contact_count": 2**22,
                "gpu_max_rigid_patch_count": 2**22,
                "gpu_collision_stack_size": 2**26,
                "gpu_max_num_partitions": 1,
            },
        )

    # ----- lifecycle --------------------------------------------------------------------------
    def bind(self, env: BaseEnv) -> None:
        super().bind(env)
        c = self.cfg
        self.gallery: RigidObject = env.iscene["gallery"]
        self.pad: RigidObject = env.iscene["pad"]
        self.trigger: RigidObject = env.iscene["trigger"]
        self.ball: RigidObject = env.iscene["ball"]
        self.tiles: list[RigidObject] = [env.iscene[f"tile_{i}"] for i in range(c.n_tiles)]
        self.env_origins = env.iscene.env_origins
        n, dev = env.num_envs, env.device
        self._steps = 0  # global substep counter (post_step ticks)
        self.t0 = torch.zeros(n, device=dev)  # substep count at last reset, per env
        self.pad_local = torch.zeros(n, 2, device=dev)  # trigger pad centre, gallery-local
        self.u_local = torch.zeros(n, 2, device=dev)  # unit pad -> window, gallery-local
        self.span = torch.full((n,), 0.35, device=dev)  # |pad -> window|
        self.seat_w = torch.zeros(n, 3, device=dev)  # ball seat, world (dislodge baseline)
        inf = float("inf")
        self.t_trig = torch.full((n,), inf, device=dev)  # substep the trigger fell
        self.t_ball = torch.full((n,), inf, device=dev)  # substep the ball dislodged
        self.causal = torch.zeros(n, device=dev)  # latched cascade causality
        self.built = torch.zeros(n, device=dev)  # latched max standing-in-corridor count
        self.pit_latch = torch.zeros(n, device=dev)  # latched causal-gated ball-in-pit

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: gallery with xy jitter + yaw, trigger pad at a randomized
        distance AND bearing in front of the window, the crimson trigger tile standing
        on the pad facing the window (yaw jitter), the ball seated on the shelf, the
        eight relay tiles scattered lying flat with keep-out resampling; all latches
        and baselines cleared."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        # --- gallery: xy jitter + yaw ---
        fix_xy = torch.tensor(c.fix_pos, device=dev).expand(m, 2).clone()
        fix_xy += (torch.rand(m, 2, device=dev) * 2 - 1) * c.fix_jitter
        yaw = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.fix_yaw_deg)
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def write(body, local_xy: torch.Tensor, z: float | torch.Tensor,
                  local_yaw: torch.Tensor | None = None) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = fix_xy[:, 0] + local_xy[:, 0] * cy - local_xy[:, 1] * sy
            st[:, 1] = fix_xy[:, 1] + local_xy[:, 0] * sy + local_xy[:, 1] * cy
            st[:, 2] = z
            wyaw = yaw if local_yaw is None else yaw + local_yaw
            st[:, 3], st[:, 6] = torch.cos(wyaw / 2), torch.sin(wyaw / 2)
            st[:, 0:3] += origin
            body.write_root_state_to_sim(st, env_ids)

        zeros2 = torch.zeros(m, 2, device=dev)
        write(self.gallery, zeros2, 0.0)

        # --- trigger pad: distance + bearing off the window outward normal (-y local) ---
        d = c.pad_dist[0] + torch.rand(m, device=dev) * (c.pad_dist[1] - c.pad_dist[0])
        bear = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.pad_bear_deg)
        pad_local = torch.stack([d * torch.sin(bear), -d * torch.cos(bear)], dim=-1)
        write(self.pad, pad_local, 0.0)

        # --- trigger tile: standing on the pad, thickness axis toward the window ---
        u = -pad_local / d.unsqueeze(-1)  # unit pad -> window (local)
        chain_yaw = torch.atan2(u[:, 1], u[:, 0])
        tjit = (torch.rand(m, device=dev) * 2 - 1) * math.radians(c.trig_yaw_jit_deg)
        write(self.trigger, pad_local, c.tile_z0 + 0.002, local_yaw=chain_yaw + tjit)

        # --- ball: seated on the shelf behind the window ---
        seat_local = torch.tensor([0.0, c.seat_y], device=dev).expand(m, 2)
        write(self.ball, seat_local, c.seat_z + 0.001)

        # --- relay tiles: scattered lying flat (largest face down), keep-out resampled.
        # Keep-outs: the trigger pad, other lying tiles, AND the pad->window CORRIDOR —
        # a lying scatter tile inside the corridor would inflate fallen_in_corridor()
        # without any cascade, so the corridor spawns clear.
        x0, x1 = c.scatter_x
        y0, y1 = c.scatter_y
        c45 = math.cos(math.pi / 4)

        def in_corridor_band(xy: torch.Tensor) -> torch.Tensor:
            rel = xy - pad_local
            d_along = (rel * u).sum(-1)
            d_lat = rel[:, 0] * u[:, 1] - rel[:, 1] * u[:, 0]
            return ((d_along > -0.02) & (d_along < d + 0.08)
                    & (d_lat.abs() < c.corr_halfw + 0.06))

        placed: list[torch.Tensor] = [pad_local]
        radii: list[float] = [c.keepout_pad]
        for i in range(c.n_tiles):
            xy = torch.stack([x0 + torch.rand(m, device=dev) * (x1 - x0),
                              y0 + torch.rand(m, device=dev) * (y1 - y0)], dim=-1)
            for _try in range(24):
                bad = in_corridor_band(xy)
                for q, r in zip(placed, radii):
                    bad |= (xy - q).norm(dim=-1) < r
                if not bad.any():
                    break
                k = int(bad.sum())
                xy[bad] = torch.stack([x0 + torch.rand(k, device=dev) * (x1 - x0),
                                       y0 + torch.rand(k, device=dev) * (y1 - y0)], dim=-1)
            placed.append(xy)
            radii.append(c.keepout_tile)
            tyaw = (torch.rand(m, device=dev) * 2 - 1) * math.pi
            # lying flat: q = qz(gallery_yaw + tyaw) * qy(90 deg) -> thickness axis vertical
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = fix_xy[:, 0] + xy[:, 0] * cy - xy[:, 1] * sy
            st[:, 1] = fix_xy[:, 1] + xy[:, 0] * sy + xy[:, 1] * cy
            st[:, 2] = c.tile_t / 2 + 0.002
            half = (yaw + tyaw) / 2
            st[:, 3] = torch.cos(half) * c45
            st[:, 4] = -torch.sin(half) * c45
            st[:, 5] = torch.cos(half) * c45
            st[:, 6] = torch.sin(half) * c45
            st[:, 0:3] += origin
            self.tiles[i].write_root_state_to_sim(st, env_ids)

        # --- baselines + latches ---
        self.pad_local[env_ids] = pad_local
        self.u_local[env_ids] = u
        self.span[env_ids] = d
        self.seat_w[env_ids] = self.ball.data.root_pos_w[env_ids].clone()
        self.t0[env_ids] = float(self._steps)
        self.t_trig[env_ids] = float("inf")
        self.t_ball[env_ids] = float("inf")
        self.causal[env_ids] = 0.0
        self.built[env_ids] = 0.0
        self.pit_latch[env_ids] = 0.0

    # ----- state (full, restorable) -----------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        return {
            "gallery": self.gallery.data.root_state_w[env_ids].clone(),
            "pad": self.pad.data.root_state_w[env_ids].clone(),
            "trigger": self.trigger.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "tiles": [t.data.root_state_w[env_ids].clone() for t in self.tiles],
            "pad_local": self.pad_local[env_ids].clone(),
            "u_local": self.u_local[env_ids].clone(),
            "span": self.span[env_ids].clone(),
            "seat_w": self.seat_w[env_ids].clone(),
            "t0": self.t0[env_ids].clone(),
            "t_trig": self.t_trig[env_ids].clone(),
            "t_ball": self.t_ball[env_ids].clone(),
            "causal": self.causal[env_ids].clone(),
            "built": self.built[env_ids].clone(),
            "pit_latch": self.pit_latch[env_ids].clone(),
        }

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.gallery.write_root_state_to_sim(state["gallery"], env_ids)
        self.pad.write_root_state_to_sim(state["pad"], env_ids)
        self.trigger.write_root_state_to_sim(state["trigger"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        for t, st in zip(self.tiles, state["tiles"]):
            t.write_root_state_to_sim(st, env_ids)
        self.pad_local[env_ids] = state["pad_local"]
        self.u_local[env_ids] = state["u_local"]
        self.span[env_ids] = state["span"]
        self.seat_w[env_ids] = state["seat_w"]
        self.t0[env_ids] = state["t0"]
        self.t_trig[env_ids] = state["t_trig"]
        self.t_ball[env_ids] = state["t_ball"]
        self.causal[env_ids] = state["causal"]
        self.built[env_ids] = state["built"]
        self.pit_latch[env_ids] = state["pit_latch"]

    # ----- description ------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A sealed gray GALLERY box ({2 * c.half_w * 1000:.0f} mm wide, "
            f"{c.wall_h * 1000:.0f} mm tall, roofed over, with side and back walls) "
            f"stands on the floor. Its front wall has one LETTERBOX WINDOW "
            f"({c.win_w * 1000:.0f} mm wide, from {c.sill_z * 1000:.0f} to "
            f"{c.win_z1 * 1000:.0f} mm height) shadowed by an overhanging EAVE. Just "
            f"inside the window, an ORANGE BALL ({2 * c.ball_r * 1000:.0f} mm) sits on "
            f"a sand-colored SHELF at sill height; behind and below the shelf the "
            f"chamber floor forms a sunken PIT. The ball cannot be grasped: hands and "
            f"tools do not fit through the window under the eave, and the roof seals "
            f"the top.\n"
            f"On the open floor, {c.pad_dist[0] * 1000:.0f}-{c.pad_dist[1] * 1000:.0f} mm "
            f"in front of the window (bearing up to {c.pad_bear_deg:.0f} degrees off its "
            f"axis — look for it), lies a CRIMSON DISC pad with a matching CRIMSON "
            f"TRIGGER TILE standing on it, already facing the window. Eight loose "
            f"SLATE-BLUE TILES ({c.tile_t * 1000:.0f} x {c.tile_w * 1000:.0f} x "
            f"{c.tile_h * 1000:.0f} mm, {c.tile_mass * 1000:.0f} g) lie flat, scattered "
            f"around the floor.\n"
            f"Goal: get the orange ball down onto the sunken pit floor of the gallery — "
            f"delivered by a DOMINO CASCADE, the only permitted impulse. Stand the blue "
            f"tiles upright in a spaced relay line from the crimson trigger tile to the "
            f"window (faces square to the line, gaps clearly smaller than a tile height; "
            f"about {c.max_gap * 1000:.0f} mm works; put the last tile roughly "
            f"{50:.0f} mm from the wall, under the eave edge), then tip ONLY the crimson "
            f"trigger tile toward the window. The falling relay must run unbroken: the "
            f"last tile pitches through the window and bats the ball off its shelf into "
            f"the pit. Judged when settled, ALL of: (1) the ball rests on the pit floor "
            f"inside the gallery; (2) the ball left its seat within {c.tau_s:.1f} s "
            f"AFTER the trigger tile fell — toppling tiles one by one, or knocking the "
            f"ball loose before/without the trigger, fails permanently; (3) at least "
            f"{c.k_relay} blue tiles lie fallen inside the corridor between pad and "
            f"window. Partial credit is latched for standing relay tiles in the "
            f"corridor and for the cascade itself."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Stand the eight blue tiles in a spaced domino line from the crimson "
            "trigger tile on the red disc to the gallery window, gaps under a tile "
            "height, then tip only the trigger tile toward the window so the cascade's "
            "last tile falls through the window and knocks the orange ball off its "
            "shelf into the pit. Do not touch the ball or topple tiles by hand — the "
            "ball must fall to the cascade, within seconds of the trigger falling."
        )

    # ----- geometry helpers -------------------------------------------------------------------
    def _fix_local(self, p_w: torch.Tensor) -> torch.Tensor:
        """(N,3) world points -> gallery body frame (origin = outer front face centre
        at ground level, +y into the chamber)."""
        from isaaclab.utils.math import quat_apply_inverse

        return quat_apply_inverse(self.gallery.data.root_quat_w,
                                  p_w - self.gallery.data.root_pos_w)

    def _up_z(self, body) -> torch.Tensor:
        """(N,) world-z component of the body's local +z (1 = upright, 0 = lying)."""
        from isaaclab.utils.math import quat_apply

        ez = torch.tensor([0.0, 0.0, 1.0], device=self.env.device).expand(self.env.num_envs, 3)
        return quat_apply(body.data.root_quat_w, ez)[:, 2]

    def _z_rel(self, body) -> torch.Tensor:
        """(N,) body origin height above the env-origin ground plane."""
        return (body.data.root_pos_w - self.env_origins)[:, 2]

    def _tiles_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(local pos (N,T,3), up_z (N,T)) for the relay tiles."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        pos = torch.stack([t.data.root_pos_w for t in self.tiles], dim=1)
        quat = torch.stack([t.data.root_quat_w for t in self.tiles], dim=1)
        n, tn = pos.shape[0], pos.shape[1]
        gq = self.gallery.data.root_quat_w[:, None, :].expand(n, tn, 4).reshape(-1, 4)
        gp = self.gallery.data.root_pos_w[:, None, :]
        loc = quat_apply_inverse(gq, (pos - gp).reshape(-1, 3)).reshape(n, tn, 3)
        ez = torch.tensor([0.0, 0.0, 1.0], device=pos.device).expand(n * tn, 3)
        upz = quat_apply(quat.reshape(-1, 4), ez).reshape(n, tn, 3)[:, :, 2]
        return loc, upz

    def _in_corridor(self, loc_xy: torch.Tensor, lo: float, hi_pad: float) -> torch.Tensor:
        """(N,T) bool: gallery-local xy inside the pad->window corridor band."""
        rel = loc_xy - self.pad_local[:, None, :]
        d_along = (rel * self.u_local[:, None, :]).sum(-1)
        d_lat = (rel[:, :, 0] * self.u_local[:, None, 1]
                 - rel[:, :, 1] * self.u_local[:, None, 0]).abs()
        return ((d_along > lo) & (d_along < self.span[:, None] + hi_pad)
                & (d_lat < self.cfg.corr_halfw))

    # ----- predicates -------------------------------------------------------------------------
    def standing_in_corridor(self) -> torch.Tensor:
        """(N,) count of relay tiles simultaneously STANDING inside the corridor."""
        c = self.cfg
        loc, upz = self._tiles_tensors()
        standing = ((upz > math.cos(math.radians(c.standing_deg)))
                    & ((loc[:, :, 2] - c.tile_z0).abs() < 0.02))
        return (standing & self._in_corridor(loc[:, :, :2], 0.02, 0.02)).sum(dim=1).float()

    def fallen_in_corridor(self) -> torch.Tensor:
        """(N,) count of relay tiles lying FALLEN inside the corridor (the last one may
        lean into the window, hence the forward slack)."""
        c = self.cfg
        loc, upz = self._tiles_tensors()
        fallen = ((upz < math.cos(math.radians(c.fallen_deg)))
                  & (loc[:, :, 2] < c.fallen_z))
        return (fallen & self._in_corridor(loc[:, :, :2], 0.0, 0.06)).sum(dim=1).float()

    def ball_in_pit(self) -> torch.Tensor:
        """(N,) bool: ball resting on the sunken chamber floor INSIDE the gallery —
        below shelf height, behind the front wall, inside the side/back walls."""
        c = self.cfg
        bl = self._fix_local(self.ball.data.root_pos_w)
        return ((bl[:, 2] < c.pit_z)
                & (bl[:, 1] > c.wall_t + 0.002) & (bl[:, 1] < c.chamber_y1 - 0.002)
                & (bl[:, 0].abs() < c.half_w - c.side_t - 0.002))

    def ball_dislodged(self) -> torch.Tensor:
        """(N,) bool: ball moved off its shelf seat by more than `dislodge_dist`."""
        return ((self.ball.data.root_pos_w - self.seat_w).norm(dim=-1)
                > self.cfg.dislodge_dist)

    def trigger_fallen(self) -> torch.Tensor:
        """(N,) bool: trigger tile tipped past `trig_deg` from vertical."""
        return self._up_z(self.trigger) < math.cos(math.radians(self.cfg.trig_deg))

    def settled(self) -> torch.Tensor:
        """(N,) bool: every tile + the trigger + the ball below the settle thresholds."""
        c = self.cfg
        ok = ((self.trigger.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
              & (self.trigger.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang)
              & (self.ball.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin))
        for t in self.tiles:
            ok = ok & ((t.data.root_lin_vel_w.norm(dim=-1) < c.settle_lin)
                       & (t.data.root_ang_vel_w.norm(dim=-1) < c.settle_ang))
        return ok

    # ----- progress latches (step-coupled) ----------------------------------------------------
    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        """Latch, every physics substep: the trigger-fall time, the ball-dislodge time,
        cascade causality (dislodge AFTER trigger and within `tau_s`), the best
        standing-relay count, and causal-gated ball-in-pit."""
        c = self.cfg
        self._steps += 1
        now = float(self._steps)

        trig = self.trigger_fallen()
        new_t = trig & torch.isinf(self.t_trig)
        self.t_trig = torch.where(new_t, torch.full_like(self.t_trig, now), self.t_trig)

        dis = self.ball_dislodged()
        new_b = dis & torch.isinf(self.t_ball)
        if bool(new_b.any()):
            tau_steps = c.tau_s / self.env.dt
            ok = (torch.isfinite(self.t_trig) & ((now - self.t_trig) <= tau_steps)
                  & ((now - self.t_trig) > 0))
            self.causal = torch.where(new_b, torch.maximum(self.causal, ok.float()),
                                      self.causal)
            self.t_ball = torch.where(new_b, torch.full_like(self.t_ball, now), self.t_ball)

        self.built = torch.maximum(
            self.built, self.standing_in_corridor().clamp(max=float(c.k_relay)))
        self.pit_latch = torch.maximum(
            self.pit_latch, (self.ball_in_pit().float() * self.causal))

    # ----- rubric -----------------------------------------------------------------------------
    def success(self) -> torch.Tensor:
        """(N,) bool: cascade-causal delivery — trigger fell, ball dislodged within the
        cascade window after it, >= k_relay relay tiles lie fallen in the corridor, the
        ball rests on the pit floor, everything settled."""
        c = self.cfg
        return ((self.causal > 0.5)
                & (self.fallen_in_corridor() >= float(c.k_relay))
                & self.ball_in_pit() & self.settled())

    def score(self) -> torch.Tensor:
        """(N,) float in [0,1]: 0.30 * built/K (latched standing relay in the corridor)
        + 0.30 * causal (latched trigger->ball cascade) + 0.15 * causal-gated ball in
        pit, capped at 0.75; exactly 1.0 iff success(). Null policy ~0; teleporting or
        hand-poking the ball into the pit without a cascade earns nothing; the latched
        credit never evaporates while the relay falls or bodies settle."""
        c = self.cfg
        base = (0.30 * self.built / float(c.k_relay) + 0.30 * self.causal
                + 0.15 * self.pit_latch).clamp(0.0, 0.75)
        return torch.where(self.success(), base.new_tensor(1.0), base)


register_env("simgen", lambda: EnvCfg(scene="domino_relay", robot="null"))
