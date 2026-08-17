"""DrainPlugScene — seat the red ball over the drain gap of a tilted hopper floor
FIRST (the ball is a VALVE), then pour the white marbles in; marbles poured before
the drain is plugged roll down the tilted floor, drop through the gap into the
sealed vault underneath, and are lost forever.

Derived from maniskill/place_sphere, with the sphere's ROLE inverted. The seed's
plan is one precise placement: grasp the red sphere, carry it, balance it ON TOP of
a shallow static bin — the sphere is the payload and placement precision is the
whole task. Here the very same 60 mm red sphere is not the payload but the TOOL:
it is the only body that can seal the 55 mm square drain gap at the low corner of
a hopper's tilted false floor (the 25 mm marbles fall straight through, and the
sphere itself cannot: 60 > 55). Placement precision is deliberately ZERO — the
ball or a marble dropped ANYWHERE on the tilted floor is routed by gravity down
the fall line into the drain corner — and the strategic content moves entirely
into ORDERING: the drain must be plugged before any marble is committed, because
the vault below the false floor is walled shut on every side; a marble that goes
down the drain is unrecoverable and success (which requires EVERY present marble
contained on the floor) becomes impossible. K itself varies per episode (2-4
marbles), and the hopper's yaw is free, so the drain corner's bearing must be read
from the scene, not memorized.

Assets are fully procedural, authored by custom compound spawners (the stopper_vault
pattern — child colliders of one body never self-collide):
  - hopper: KINEMATIC compound — four walls (ground -> rim at 280 mm, so the volume
    under the false floor is a sealed vault) around a 300 x 300 mm inner square,
    plus a false floor of two coplanar plates tilted 8 deg about the (1,-1) diagonal
    axis (top surface from 160 mm at the low corner to ~220 mm at the high corner),
    leaving one 55 x 55 mm square DRAIN GAP at the low corner. Local frame: origin
    on the ground at the centre of the inner square; the drain corner is at
    (-h, -h).
  - ball: DYNAMIC red sphere, r = 30 mm (the seed's own sphere), 100 g — the plug.
  - marbles 0..3: DYNAMIC white spheres, r = 12.5 mm, 20 g. K in {2,3,4} spawn in
    the tray; the rest are parked in the walled depot far outside the workspace.
  - tray: KINEMATIC shallow open bin where the ball + marbles start (it is also the
    seed's own success geometry: a sphere resting in/on a shallow bin scores 0 here).
  - depot: KINEMATIC walled parking bin for absent marbles.

Per-episode randomization (readback-verifiable): hopper yaw FREE (+/-180 deg, the
drain corner bearing varies) + xy jitter, tray ground slot (side + band + jitter +
free yaw), K = number of live marbles in {2..4}, per-object spawn jitter.

Rubric (0..1; latched, monotone; ~0 for the null policy):
  0.35 * plug   — the ball ever SEATED in the drain gap (fixture-frame distance to
                  the drain corner < seat_r, centre inside the sunk z-band that a
                  ball resting ON the floor plates cannot reach, still, for a
                  settle streak) (latched)
  0.50 * beads  — per-marble, gated on the plug latch (the physical order): marble
                  inside the hopper, resting ON or above the false floor (z above
                  the local floor plane; a drained marble sits ~150 mm BELOW it on
                  the vault ground), still, for a settle streak (latched, each
                  worth 0.50/K)
  1.0 iff success() — plug seated live + every present marble contained live +
                  everything settled + all latches earned. Non-success capped 0.85.

Heavy imports (isaaclab, pxr) are deferred so importing this module — and
registering the scene — stays app-free.
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


def _make_collide(cfg: Any) -> Callable:
    from pxr import PhysxSchema, UsdPhysics

    def collide(prim) -> None:
        UsdPhysics.CollisionAPI.Apply(prim)
        px = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        px.CreateContactOffsetAttr(float(cfg.contact_offset))
        px.CreateRestOffsetAttr(0.0)

    return collide


def _add_box(stage, path: str, *, center, size, color, collide: Callable, orient=None) -> None:
    from pxr import Gf, UsdGeom

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
    if orient is not None:
        w, x, y, z = (float(v) for v in orient)
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    collide(box.GetPrim())


def _spawn_hopper(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author the hopper at `prim_path`: KINEMATIC rigid body. Four walls run from
    the GROUND to the rim, so the volume under the tilted false floor is a sealed
    vault whose only opening is the square drain gap at the low corner (-h, -h).
    The false floor is two coplanar plates sharing one tilt quaternion (8 deg about
    the (1,-1) diagonal), their top faces in the plane z = z_mid + s*(x + y)."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    h, t, hw = c.h, c.wall_t, c.wall_h
    span = 2 * h + 2 * t
    for nm, ctr, sz in (
        ("wall_px", (h + t / 2, 0.0, hw / 2), (t, span, hw)),
        ("wall_nx", (-h - t / 2, 0.0, hw / 2), (t, span, hw)),
        ("wall_py", (0.0, h + t / 2, hw / 2), (2 * h, t, hw)),
        ("wall_ny", (0.0, -h - t / 2, hw / 2), (2 * h, t, hw)),
    ):
        _add_box(stage, f"{prim_path}/{nm}", center=ctr, size=sz,
                 color=c.wall_color, collide=collide)
    # tilted false floor: two coplanar plates, one shared quaternion
    th = math.atan(c.slope_s * math.sqrt(2.0))  # tilt angle about the diagonal
    half, ax = th / 2.0, 1.0 / math.sqrt(2.0)
    q = (math.cos(half), math.sin(half) * ax, -math.sin(half) * ax, 0.0)
    proj = (1.0 + math.cos(th)) / 2.0  # plan-length shortening of the tilted axes
    nrm = math.sqrt(1.0 + 2.0 * c.slope_s ** 2)
    e, g, pt = c.embed, c.gap, c.plate_t

    def plate(name: str, x0: float, x1: float, y0: float, y1: float) -> None:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        cz = c.z_mid + c.slope_s * (cx + cy) - (pt / 2.0) * nrm
        _add_box(stage, f"{prim_path}/{name}", center=(cx, cy, cz),
                 size=((x1 - x0) / proj, (y1 - y0) / proj, pt),
                 color=c.plate_color, collide=collide, orient=q)

    plate("plate_a", -h + g, h + e, -h - e, h + e)   # everything x >= gap edge
    plate("plate_b", -h - e, -h + g, -h + g, h + e)  # the strip beside the gap
    return root


def _spawn_bin(prim_path: str, cfg: Any, translation=None, orientation=None):
    """Author a KINEMATIC open-top bin (tray / depot): floor slab + four walls.
    Local origin at the OUTER BOTTOM centre."""
    from pxr import UsdPhysics

    stage, root = _root_xform(prim_path, translation, orientation)
    UsdPhysics.RigidBodyAPI.Apply(root).CreateKinematicEnabledAttr(True)
    collide = _make_collide(cfg)
    c = cfg
    hx, hy, t, wh, ft = c.half_x, c.half_y, c.wall_t, c.wall_h, c.floor_t
    _add_box(stage, f"{prim_path}/floor", center=(0.0, 0.0, ft / 2),
             size=(2 * hx, 2 * hy, ft), color=c.color, collide=collide)
    wall_zc = ft + wh / 2
    for nm, ctr, sz in (
        ("wall_px", (hx - t / 2, 0.0, wall_zc), (t, 2 * hy, wh)),
        ("wall_nx", (-hx + t / 2, 0.0, wall_zc), (t, 2 * hy, wh)),
        ("wall_py", (0.0, hy - t / 2, wall_zc), (2 * hx - 2 * t, t, wh)),
        ("wall_ny", (0.0, -hy + t / 2, wall_zc), (2 * hx - 2 * t, t, wh)),
    ):
        _add_box(stage, f"{prim_path}/{nm}", center=ctr, size=sz,
                 color=c.color, collide=collide)
    return root


def _spawner_classes() -> dict[str, Any]:
    """Declare (once) the compound spawner configclasses (heavy imports deferred)."""
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    if "hopper" not in _SPAWNER_CACHE:

        @configclass
        class HopperSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_hopper)
            h: float = 0.15
            wall_t: float = 0.012
            wall_h: float = 0.28
            gap: float = 0.055
            embed: float = 0.010
            plate_t: float = 0.012
            slope_s: float = 0.0994
            z_mid: float = 0.1898
            wall_color: tuple = (0.36, 0.40, 0.48)
            plate_color: tuple = (0.72, 0.72, 0.66)
            contact_offset: float = 0.002

        @configclass
        class BinSpawnerCfg(RigidObjectSpawnerCfg):
            func: Callable = clone(_spawn_bin)
            half_x: float = 0.13
            half_y: float = 0.09
            wall_h: float = 0.025
            wall_t: float = 0.008
            floor_t: float = 0.008
            color: tuple = (0.55, 0.38, 0.20)
            contact_offset: float = 0.002

        _SPAWNER_CACHE.update(hopper=HopperSpawnerCfg, bin=BinSpawnerCfg)
    return _SPAWNER_CACHE


# ----- scene cfg -------------------------------------------------------------------------------
@dataclass
class DrainPlugSceneCfg(BaseCfg):
    """Config for `DrainPlugScene`. The hopper's inner square is x,y in [-h, +h]
    (h = 0.15) in the fixture frame, walls ground -> wall_h; the false floor's top
    surface is the plane z = z_mid + slope_s*(x + y) (low corner (-h,-h) at z_low
    = 0.16, high corner at ~0.22) and the only under-floor opening is the gap x,y
    in [-h, -h+gap]. `__post_init__` derives the plane and PROVES the geometry:
    the marble passes the open gap, the ball cannot; the seated ball's centre
    lands inside the seat window while a ball resting ON the plates anywhere in
    that window does not; the marble cannot sneak past the seated ball at the
    wall corner."""

    # --- tunable: rubric thresholds ---------------------------------------------------------------
    seat_r: float = tunable(0.075)  # ball centre within this (plan) of the drain corner (m)
    seat_z_lo: float = tunable(0.150)  # seat window floor (m; vault is at ~0.03 — unreachable
    # for the ball anyway: it cannot pass the gap)
    seat_z_hi: float = tunable(0.192)  # seat window ceiling (m): a SEATED ball sinks to ~0.185;
    # a ball resting ON the plates in the same plan window sits >= 0.196 — rejected
    bead_margin: float = tunable(0.004)  # marble containment: |x|,|y| < h - margin (m)
    bead_z_tol: float = tunable(0.012)  # marble containment: z > plane(x,y) + r - tol (m); a
    # drained marble rests on the vault ground ~0.15 BELOW the plane — rejected
    settle_speed: float = tunable(0.05)  # max |lin vel| for LATCHING (strict settle proof) (m/s)
    judge_speed: float = tunable(0.15)  # max |lin vel| for the LIVE success gate (m/s). The
    # seated ball rests on two sharp plate edges: a PhysX edge-contact limit cycle pumps a
    # bounded phantom oscillation (measured peak ~0.09 m/s) into an otherwise motionless
    # pile, then self-resets. Latches still demand a 0.05 m/s x 24-step streak (earned
    # settling); the live gate only rejects GENUINE motion (anything actually rolling in
    # this funnel moves >= 0.3 m/s), so the artifact cannot flip a settled success.
    settle_streak: int = tunable(24)  # consecutive still+placed steps to latch (0.2 s @ 120 Hz;
    # long enough that the slow turnaround of a residual pocket oscillation cannot latch)

    # --- tunable: randomization (the task-family knobs) -------------------------------------------
    hopper_yaw_deg: float = tunable(180.0)  # hopper free yaw (+/- deg): drain bearing varies
    hopper_jitter: float = tunable(0.025)  # hopper xy jitter (+/- m)
    tray_x_range: tuple = tunable((0.02, 0.12))  # tray ground band, world x (m)
    tray_y_band: tuple = tunable((0.34, 0.44))  # |tray world y| band (sign random) (m)
    tray_yaw_deg: float = tunable(180.0)  # tray free yaw (+/- deg)
    k_range: tuple = tunable((2, 4))  # live marble count K sampled uniformly in this range
    slot_jitter: float = tunable(0.008)  # per-object spawn jitter in the tray (+/- m)

    # --- info: structure (hopper local frame: origin on the ground, centre of the inner square) ---
    hopper_pos: tuple = info((0.58, 0.0))  # hopper origin on the ground
    h: float = info(0.15)  # inner square half-span; drain corner at (-h, -h)
    wall_t: float = info(0.012)  # wall thickness
    wall_h: float = info(0.28)  # wall/rim height (walls run ground -> rim: sealed vault)
    alpha_deg: float = info(8.0)  # floor tilt about the (1,-1) diagonal (deg)
    z_low: float = info(0.16)  # floor TOP surface at the drain corner
    gap: float = info(0.055)  # square drain gap side (marble 25 mm passes; ball 60 mm cannot)
    plate_t: float = info(0.012)  # false-floor plate thickness
    embed: float = info(0.010)  # plate embedment into the walls (sealed perimeter)
    ball_r: float = info(0.030)  # the seed's own red sphere (60 mm dia < 80 mm jaw: graspable)
    ball_mass: float = info(0.10)
    ball_mu: float = info(0.45)  # defined friction (rolls truly on the plates)
    marble_r: float = info(0.0125)  # 25 mm dia — passes the 55 mm gap, fits the jaw
    marble_mass: float = info(0.02)
    marble_mu: float = info(0.35)
    n_marbles: int = info(4)  # marble bodies authored; K <= 4 live per episode
    tray_half_x: float = info(0.13)  # tray outer half-spans (shallow open bin)
    tray_half_y: float = info(0.09)
    tray_wall_h: float = info(0.025)
    tray_wall_t: float = info(0.008)
    tray_floor_t: float = info(0.008)
    depot_pos: tuple = info((-0.50, -0.45))  # parking bin, far outside the workspace
    depot_half: float = info(0.10)
    depot_wall_h: float = info(0.05)
    wall_color: tuple = info((0.36, 0.40, 0.48))  # blue-grey hopper shell
    plate_color: tuple = info((0.72, 0.72, 0.66))  # bone false floor
    ball_color: tuple = info((0.85, 0.08, 0.08))  # RED (the seed's sphere)
    marble_color: tuple = info((0.93, 0.93, 0.95))  # white marbles
    tray_color: tuple = info((0.55, 0.38, 0.20))  # brown tray
    depot_color: tuple = info((0.25, 0.25, 0.28))
    contact_offset: float = info(0.002)
    # rubric weights (0.35 + 0.50 = 0.85 == the non-success cap)
    w_plug: float = info(0.35)
    w_beads: float = info(0.50)

    # Derived (filled in __post_init__).
    slope_s: float = field(default=None, init=False)  # per-axis slope: z = z_mid + s*(x+y)
    z_mid: float = field(default=None, init=False)  # plane height at the centre
    z_high: float = field(default=None, init=False)  # plane height at the high corner
    z_seat: float = field(default=None, init=False)  # expected seated-ball centre height

    def __post_init__(self) -> None:
        h, g, r, rm = self.h, self.gap, self.ball_r, self.marble_r
        self.slope_s = math.tan(math.radians(self.alpha_deg)) / math.sqrt(2.0)
        self.z_mid = round(self.z_low + 2.0 * h * self.slope_s, 6)
        self.z_high = round(self.z_mid + 2.0 * h * self.slope_s, 6)

        def plane(x: float, y: float) -> float:
            return self.z_mid + self.slope_s * (x + y)

        # --- the gap sorts the spheres: marble passes (with slop), ball cannot ---
        assert 2.0 * rm < g - 0.006, "marble must fall through the open drain gap"
        assert 2.0 * r > g + 0.004, "ball must be unable to pass the drain gap"
        # --- seated-ball geometry: centre pushed to (-h+r, -h+r) by the two walls, resting on
        # the two plate edge lines x=-h+g and y=-h+g (horizontal reach g-r each) ---
        assert 0.0 < g - r < r, "ball must rest on the plate EDGES (sunk), not on plate faces"
        drop = math.sqrt(r * r - (g - r) * (g - r))
        self.z_seat = round(plane(-h + g, -h + r) + drop, 6)
        assert self.seat_z_lo + 0.004 < self.z_seat < self.seat_z_hi - 0.004, \
            f"seat window must accept the seated ball (z_seat={self.z_seat})"
        # a ball resting ON the plates: centre >= plane + r everywhere it has support; the
        # lowest supported plan point inside the seat window is the gap edge corner region
        z_on_floor_min = plane(-h + r, -h + r) + r  # (support ends at the gap edges; the true
        # minimum supported rest is even higher — this is the conservative bound)
        assert z_on_floor_min > self.seat_z_hi + 0.003, \
            "an unsunk ball resting on the floor must be rejected by the seat ceiling"
        # seat plan distance: sqrt(2)*r from the corner — well inside seat_r
        assert math.sqrt(2.0) * r < self.seat_r - 0.008, "seat_r must accept the seated ball"
        # --- the seated ball SEALS: max sphere passing the wall-corner leak ---
        leak = r * (math.sqrt(2.0) - 1.0) / (math.sqrt(2.0) + 1.0)  # ~5.1 mm radius
        assert leak < rm - 0.004, "marble must not fit between the seated ball and the corner"
        # plate-corner side: plan clearance between ball surface and the plate corner (g,g)
        assert math.sqrt(2.0) * (g - r) - r < 0.008, "plate-corner leak must be sub-marble"
        # --- containment plumbing ---
        assert self.z_high + 2.0 * rm + 0.02 < self.wall_h, "parapet must retain settled marbles"
        assert self.z_seat + r + 2.0 * rm < self.wall_h, "a marble atop the plug stays below rim"
        assert h - self.bead_margin > h - rm, "wall-hugging marble must still count as inside"
        # marble in the vault: rests on the ground, ~0.15 below the plane — clearly rejected
        assert rm + 0.01 < self.z_low - 0.05, "vault rest must sit far below the floor plane"
        # --- weights ---
        assert abs(self.w_plug + self.w_beads - 0.85) < 1e-9, "weights must sum to the cap"


# ----- scene -----------------------------------------------------------------------------------
@SCENES.register("drain_plug")
class DrainPlugScene(BaseScene):
    cfg: DrainPlugSceneCfg

    def __init__(self, cfg: DrainPlugSceneCfg | None = None) -> None:
        super().__init__(cfg or DrainPlugSceneCfg())

    # ----- assets -------------------------------------------------------------------------------
    def assets(self) -> dict[str, Any]:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        c = self.cfg
        cls = _spawner_classes()
        hopper_spawn = cls["hopper"](
            mass_props=sim_utils.MassPropertiesCfg(mass=25.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            h=c.h, wall_t=c.wall_t, wall_h=c.wall_h, gap=c.gap, embed=c.embed,
            plate_t=c.plate_t, slope_s=c.slope_s, z_mid=c.z_mid,
            wall_color=c.wall_color, plate_color=c.plate_color,
            contact_offset=c.contact_offset)
        tray_spawn = cls["bin"](
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            half_x=c.tray_half_x, half_y=c.tray_half_y, wall_h=c.tray_wall_h,
            wall_t=c.tray_wall_t, floor_t=c.tray_floor_t, color=c.tray_color,
            contact_offset=c.contact_offset)
        depot_spawn = cls["bin"](
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            half_x=c.depot_half, half_y=c.depot_half, wall_h=c.depot_wall_h,
            wall_t=0.010, floor_t=0.008, color=c.depot_color,
            contact_offset=c.contact_offset)

        def sphere_spawn(radius: float, mass: float, mu: float, color: tuple,
                         lin_damp: float, ang_damp: float) -> Any:
            # Damping is the rolling-resistance stand-in: PhysX models NO rolling
            # friction, so an undamped sphere oscillates in the corner pocket for
            # tens of seconds under the 0.05 m/s stillness gate (turning-point
            # latches + persistence flicker). Slope accel (~1 m/s^2 at 8 deg)
            # dwarfs damping*v at rolling speeds, so the funnel behaviour is
            # unchanged.
            return sim_utils.SphereCfg(
                radius=radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5, linear_damping=lin_damp,
                    angular_damping=ang_damp, sleep_threshold=0.0,
                    stabilization_threshold=0.0,
                    solver_velocity_iteration_count=4),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=c.contact_offset, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=mu, dynamic_friction=mu - 0.05,
                    restitution=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )

        assets: dict[str, Any] = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            ),
            "light": AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
            ),
            "hopper": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Hopper",
                spawn=hopper_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.hopper_pos[0], c.hopper_pos[1], 0.0)),
            ),
            "tray": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Tray",
                spawn=tray_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.07, 0.38, 0.0)),
            ),
            "depot": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Depot",
                spawn=depot_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(c.depot_pos[0], c.depot_pos[1], 0.0)),
            ),
            "ball": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RedBall",
                spawn=sphere_spawn(c.ball_r, c.ball_mass, c.ball_mu, c.ball_color,
                                   lin_damp=0.15, ang_damp=0.45),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.00, 0.38, c.ball_r + 0.01)),
            ),
        }
        for i in range(c.n_marbles):
            assets[f"marble{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Marble" + str(i),
                spawn=sphere_spawn(c.marble_r, c.marble_mass, c.marble_mu, c.marble_color,
                                   lin_damp=0.40, ang_damp=0.90),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.06 + 0.05 * i, 0.38, c.marble_r + 0.01)),
            )
        return assets

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
        self.hopper: RigidObject = env.iscene["hopper"]
        self.tray: RigidObject = env.iscene["tray"]
        self.depot: RigidObject = env.iscene["depot"]
        self.ball: RigidObject = env.iscene["ball"]
        self.marbles: list[RigidObject] = [env.iscene[f"marble{i}"] for i in range(c.n_marbles)]
        self.env_origins = env.iscene.env_origins
        n = env.num_envs
        dev = env.device
        # latches: partial progress survives transient achievements (rubric requirement)
        self._plug_latch = torch.zeros(n, dtype=torch.bool, device=dev)
        self._plug_streak = torch.zeros(n, dtype=torch.long, device=dev)
        self._bead_latch = torch.zeros(n, c.n_marbles, dtype=torch.bool, device=dev)
        self._bead_streak = torch.zeros(n, c.n_marbles, dtype=torch.long, device=dev)
        self._present = torch.zeros(n, c.n_marbles, dtype=torch.bool, device=dev)
        self._k = torch.full((n,), 2, dtype=torch.long, device=dev)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Fresh episode: place the hopper (FREE yaw + xy jitter — the drain corner
        bearing must be read per episode), the tray in its (side-sampled) ground
        band, sample K in k_range, seat the ball + the K live marbles in the tray
        (jittered slots) and park the rest in the depot; clear all latches."""
        c = self.cfg
        dev = self.env.device
        m = len(env_ids)
        origin = self.env_origins[env_ids]

        def uni(lo: float, hi: float) -> torch.Tensor:
            return lo + torch.rand(m, device=dev) * (hi - lo)

        def kin_place(obj, x, y, yaw) -> None:
            st = torch.zeros(m, 13, device=dev)
            st[:, 0], st[:, 1] = x, y
            st[:, 3], st[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
            st[:, 0:3] += origin
            obj.write_root_state_to_sim(st, env_ids)

        # --- hopper: free yaw, xy jitter ---
        hyaw = uni(-1.0, 1.0) * math.radians(c.hopper_yaw_deg)
        hx = c.hopper_pos[0] + uni(-1.0, 1.0) * c.hopper_jitter
        hy = c.hopper_pos[1] + uni(-1.0, 1.0) * c.hopper_jitter
        kin_place(self.hopper, hx, hy, hyaw)

        # --- tray: ground band, random side, free yaw ---
        tx = uni(*c.tray_x_range)
        tside = torch.where(torch.rand(m, device=dev) < 0.5,
                            torch.ones(m, device=dev), -torch.ones(m, device=dev))
        ty = tside * uni(*c.tray_y_band)
        tyaw = uni(-1.0, 1.0) * math.radians(c.tray_yaw_deg)
        kin_place(self.tray, tx, ty, tyaw)

        # --- depot: fixed far parking ---
        kin_place(self.depot, torch.full((m,), c.depot_pos[0], device=dev),
                  torch.full((m,), c.depot_pos[1], device=dev),
                  torch.zeros(m, device=dev))

        # --- K live marbles (torch.rand, not randint: the first-randint trap) ---
        klo, khi = c.k_range
        k = klo + (torch.rand(m, device=dev) * (khi - klo + 1)).long().clamp(max=khi - klo)
        self._k[env_ids] = k
        idx = torch.arange(c.n_marbles, device=dev).unsqueeze(0).expand(m, -1)
        present = idx < k.unsqueeze(1)
        self._present[env_ids] = present

        # --- ball + live marbles in the tray (tray-local slots, jittered) ---
        cos_t, sin_t = torch.cos(tyaw), torch.sin(tyaw)

        def tray_place(obj, lx0: float, ly0: float, z: float) -> torch.Tensor:
            lx = lx0 + uni(-1.0, 1.0) * c.slot_jitter
            ly = ly0 + uni(-1.0, 1.0) * c.slot_jitter
            st = torch.zeros(m, 13, device=dev)
            st[:, 0] = tx + cos_t * lx - sin_t * ly
            st[:, 1] = ty + sin_t * lx + cos_t * ly
            st[:, 2] = z
            st[:, 3] = 1.0
            st[:, 0:3] += origin
            return st

        zb = c.tray_floor_t + c.ball_r + 0.003
        self.ball.write_root_state_to_sim(tray_place(self.ball, -0.075, 0.0, zb), env_ids)
        zm = c.tray_floor_t + c.marble_r + 0.003
        slots = ((0.005, -0.045), (0.005, 0.045), (0.065, -0.045), (0.065, 0.045))
        for i, mb in enumerate(self.marbles):
            st_tray = tray_place(mb, slots[i][0], slots[i][1], zm)
            # depot slot for absent marbles
            dslots = ((-0.045, -0.045), (-0.045, 0.045), (0.045, -0.045), (0.045, 0.045))
            st_depot = torch.zeros(m, 13, device=dev)
            st_depot[:, 0] = c.depot_pos[0] + dslots[i][0]
            st_depot[:, 1] = c.depot_pos[1] + dslots[i][1]
            st_depot[:, 2] = 0.008 + c.marble_r + 0.003
            st_depot[:, 3] = 1.0
            st_depot[:, 0:3] += origin
            st = torch.where(present[:, i:i + 1], st_tray, st_depot)
            mb.write_root_state_to_sim(st, env_ids)

        # --- clear latches ---
        self._plug_latch[env_ids] = False
        self._plug_streak[env_ids] = 0
        self._bead_latch[env_ids] = False
        self._bead_streak[env_ids] = 0

    # ----- state (full, restorable) ---------------------------------------------------------------
    def get_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        out = {
            "hopper": self.hopper.data.root_state_w[env_ids].clone(),
            "tray": self.tray.data.root_state_w[env_ids].clone(),
            "depot": self.depot.data.root_state_w[env_ids].clone(),
            "ball": self.ball.data.root_state_w[env_ids].clone(),
            "plug_latch": self._plug_latch[env_ids].clone(),
            "plug_streak": self._plug_streak[env_ids].clone(),
            "bead_latch": self._bead_latch[env_ids].clone(),
            "bead_streak": self._bead_streak[env_ids].clone(),
            "present": self._present[env_ids].clone(),
            "k": self._k[env_ids].clone(),
        }
        for i, mb in enumerate(self.marbles):
            out[f"marble{i}"] = mb.data.root_state_w[env_ids].clone()
        return out

    def set_state(self, state: dict[str, Any], env_ids: torch.Tensor) -> None:
        self.hopper.write_root_state_to_sim(state["hopper"], env_ids)
        self.tray.write_root_state_to_sim(state["tray"], env_ids)
        self.depot.write_root_state_to_sim(state["depot"], env_ids)
        self.ball.write_root_state_to_sim(state["ball"], env_ids)
        for i, mb in enumerate(self.marbles):
            mb.write_root_state_to_sim(state[f"marble{i}"], env_ids)
        self._plug_latch[env_ids] = state["plug_latch"]
        self._plug_streak[env_ids] = state["plug_streak"]
        self._bead_latch[env_ids] = state["bead_latch"]
        self._bead_streak[env_ids] = state["bead_streak"]
        self._present[env_ids] = state["present"]
        self._k[env_ids] = state["k"]

    # ----- description ----------------------------------------------------------------------------
    def describe(self) -> str:
        c = self.cfg
        return (
            f"A blue-grey square HOPPER stands on the ground: four walls "
            f"({c.wall_h * 100:.0f} cm tall, running all the way to the ground) around a "
            f"{2 * c.h * 100:.0f} x {2 * c.h * 100:.0f} cm opening, holding a bone-white "
            f"FALSE FLOOR tilted {c.alpha_deg:.0f} degrees toward one corner (surface from "
            f"{c.z_low * 100:.0f} cm up at the low corner to {c.z_high * 100:.0f} cm at the "
            f"high corner). At the LOW corner the floor stops short of the walls, leaving a "
            f"single square DRAIN GAP of {c.gap * 1000:.0f} mm. The space under the false "
            f"floor is a sealed VAULT: its only opening is that gap, so anything that falls "
            f"through is lost for good. Which corner is the low one varies per episode "
            f"(the hopper's yaw is random) — read the floor's tilt to find it. A brown "
            f"TRAY on the ground holds one RED BALL ({2 * c.ball_r * 1000:.0f} mm) and "
            f"between {c.k_range[0]} and {c.k_range[1]} white MARBLES "
            f"({2 * c.marble_r * 1000:.0f} mm; the exact number varies — count them). A "
            f"dark parking bin far outside the work area is not part of the task.\n"
            f"Anything dropped onto the tilted floor rolls down to the low corner by "
            f"itself, so placement needs no precision — but the ORDER is everything. A "
            f"marble ({2 * c.marble_r * 1000:.0f} mm) fits through the "
            f"{c.gap * 1000:.0f} mm gap and drains into the vault; the red ball "
            f"({2 * c.ball_r * 1000:.0f} mm) does not fit and instead SEATS in the gap, "
            f"sinking part-way in and sealing it like a valve. Goal, in this order: FIRST "
            f"take the red ball from the tray and drop it anywhere onto the false floor so "
            f"it rolls down and plugs the drain; THEN move every white marble from the "
            f"tray into the hopper — each rolls down and collects against the seated "
            f"ball. Any marble sent in before the ball is seated goes down the drain and "
            f"can never be recovered, which makes success impossible. Success: the red "
            f"ball seated in the drain gap (sunk in, at the low corner) and ALL marbles "
            f"that started in the tray resting inside the hopper on the false floor, "
            f"everything settled. A marble in the vault, a marble left in the tray or "
            f"on the ground, the ball resting in the tray or anywhere but the drain "
            f"seat, or anything still moving all fail. The ball merely resting on the "
            f"floor without sinking into the gap does not count as seated."
        )

    def instruction(self) -> str:
        """SHORT imperative form of the goal for VLA training."""
        return (
            "Drop the red ball into the hopper first so it rolls down the tilted floor "
            "and plugs the square drain gap at the low corner. Then move every white "
            "marble from the tray into the hopper; they roll down and settle against the "
            "seated ball. Marbles dropped in before the ball is seated fall through the "
            "drain into the sealed vault and are lost, so seat the ball first."
        )

    # ----- progress / rubric ------------------------------------------------------------------------
    def _fixture_local(self, obj) -> torch.Tensor:
        """Object centre in the HOPPER's body frame, (N, 3) — the drain corner, the
        floor plane and the containment windows live in this frame, so a yawed /
        jittered hopper judges identically."""
        from isaaclab.utils.math import quat_apply_inverse

        rel = obj.data.root_pos_w - self.hopper.data.root_pos_w
        return quat_apply_inverse(self.hopper.data.root_quat_w, rel)

    def _plane_z(self, lx: torch.Tensor, ly: torch.Tensor) -> torch.Tensor:
        """False-floor TOP surface height at fixture-frame (lx, ly)."""
        return self.cfg.z_mid + self.cfg.slope_s * (lx + ly)

    def _plug_now(self) -> torch.Tensor:
        """(N,) bool: ball centre inside the SEAT window — within seat_r (plan) of
        the drain corner AND inside the sunk z-band [seat_z_lo, seat_z_hi]. A ball
        resting ON the plates sits >= ~0.196 (above the ceiling); only a ball sunk
        into the gap reaches ~0.185 (proved in __post_init__)."""
        c = self.cfg
        loc = self._fixture_local(self.ball)
        corner = torch.tensor([-c.h, -c.h], device=loc.device)
        d = (loc[:, :2] - corner).norm(dim=-1)
        return (d < c.seat_r) & (loc[:, 2] > c.seat_z_lo) & (loc[:, 2] < c.seat_z_hi)

    def _bead_in(self, i: int) -> torch.Tensor:
        """(N,) bool: marble i inside the hopper, resting ON (or above) the false
        floor: |x|,|y| inside the inner square, z above the local floor plane + r
        - tol (a drained marble rests on the vault ground ~0.15 BELOW the plane)
        and below the rim."""
        c = self.cfg
        loc = self._fixture_local(self.marbles[i])
        lim = c.h - c.bead_margin
        z_floor = self._plane_z(loc[:, 0], loc[:, 1]) + c.marble_r - c.bead_z_tol
        return (loc[:, 0].abs() < lim) & (loc[:, 1].abs() < lim) \
            & (loc[:, 2] > z_floor) & (loc[:, 2] < c.wall_h - 0.005)

    def _still(self, obj) -> torch.Tensor:
        """Strict stillness — the LATCHING gate (streak-verified settle proof)."""
        return obj.data.root_lin_vel_w.norm(dim=-1) < self.cfg.settle_speed

    def _slow(self, obj) -> torch.Tensor:
        """Loose stillness — the LIVE success gate: rejects genuine motion but not
        the bounded sharp-edge phantom oscillation of the seated pile."""
        return obj.data.root_lin_vel_w.norm(dim=-1) < self.cfg.judge_speed

    def _update_latches(self) -> None:
        """Refresh the latches. `plug` latches after the ball holds the seat window
        still for a settle streak (a rolling pass through the window never latches).
        Each `bead` latches after marble i holds containment still for a streak
        WITH the plug latch already set — the order gate that mirrors the physics
        (without the plug there is nothing for a marble to rest on: it drains)."""
        c = self.cfg
        cond_p = self._plug_now() & self._still(self.ball)
        self._plug_streak = torch.where(cond_p, self._plug_streak + 1,
                                        torch.zeros_like(self._plug_streak))
        self._plug_latch |= self._plug_streak >= c.settle_streak
        for i, mb in enumerate(self.marbles):
            cond_b = (self._bead_in(i) & self._still(mb) & self._plug_latch
                      & self._present[:, i])
            self._bead_streak[:, i] = torch.where(
                cond_b, self._bead_streak[:, i] + 1,
                torch.zeros_like(self._bead_streak[:, i]))
            self._bead_latch[:, i] |= self._bead_streak[:, i] >= c.settle_streak

    def post_step(self, env_ids: torch.Tensor | None = None) -> None:
        self._update_latches()

    def success(self) -> torch.Tensor:
        """(N,) bool: the red ball seated in the drain gap LIVE, every present
        marble contained on the false floor LIVE, everything settled, and the
        streak-verified latches all earned (a conjunction of instantaneous
        near-misses at velocity turning points never qualifies). Settling is
        PROVED by the strict streak latches; the live gate uses judge_speed so
        the bounded edge-contact phantom oscillation cannot flip a settled
        outcome (see cfg.judge_speed)."""
        self._update_latches()
        ok = self._plug_now() & self._slow(self.ball) & self._plug_latch
        for i, mb in enumerate(self.marbles):
            bead_ok = (self._bead_in(i) & self._slow(mb) & self._bead_latch[:, i]) \
                | ~self._present[:, i]
            ok = ok & bead_ok
        return ok

    def score(self) -> torch.Tensor:
        """(N,) float in [0, 1]: 0.35*plug + 0.50*(latched beads / K), all latched
        and monotone (order-gated: beads only earn after the plug latch, exactly as
        the physics forces), capped at 0.85 — and exactly 1.0 iff success()."""
        c = self.cfg
        self._update_latches()
        beads = (self._bead_latch & self._present).sum(dim=1).float() / self._k.float()
        base = (c.w_plug * self._plug_latch.float() + c.w_beads * beads).clamp(max=0.85)
        return torch.where(self.success(), torch.ones_like(base), base)


# Scene-level task: no robot in the slot; bodies are driven through scene handles.
register_env("simgen", lambda: EnvCfg(scene="drain_plug", robot="null"))
